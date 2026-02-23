"""
OTW (Online Time Warping) score follower — binding to the ConcertCue algorithm.

Replaces the placeholder fastdtw-based implementation with the actual
streaming OTW algorithm from models/otw/ (Simon Dixon 2005, adapted by
Caren & Egozy for the web-score-following / ConcertCue project, WAC 2024).

Default algorithm parameters are the tuned values from the paper
(found via grid search on 18 concert recordings):

    sample_rate    = 44100 Hz
    n_fft          = 8192 samples
    ref_hop_len    = 4096 samples
    live_hop_len   = 3686 samples  (~90 % of ref hop)
    OTW c          = 300           (search window width)
    max_run_count  = 3
    diag_weight    = 0.4           (lower → less prone to getting stuck)

Reported benchmark (simulated Python, Table 1 in paper):
    median error  0.077 s  |  95th pct  0.537 s  |  mean error  0.130 s
"""

import sys
import os
import time
import types
from typing import Dict, Any, Optional

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Pyodide / browser-API guard
#
# models/otw/file_utils.py has a top-level  "from js import Blob, document, window"
# (Pyodide browser API).  The package __init__.py re-exports everything from that
# module, so importing *any* symbol from models.otw would fail on desktop Python.
# We inject a lightweight mock into sys.modules before the first import so that
# Python resolves the name without error.  save_from_browser() — the only
# function that actually uses those objects — is never called in this binding.
# ---------------------------------------------------------------------------
if "js" not in sys.modules:
    _js_mock = types.ModuleType("js")
    _js_mock.Blob = None
    _js_mock.document = None
    _js_mock.window = None
    sys.modules["js"] = _js_mock

# Add the project root to sys.path so that "models.*" package imports resolve.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.base_model import BaseScoreFollower
from models.otw.otw import OTW
from models.otw.features import ChromaMaker, audio_to_np_cens


class OTWModel(BaseScoreFollower):
    """
    Score follower using Online Time Warping (OTW).

    Algorithm (streaming, O(n) time and space):
      1.  Load reference audio → extract CENS features (12-dim per frame).
      2.  For each live audio chunk:
          a.  Resample to self.sr if necessary.
          b.  Accumulate in a rolling buffer.
          c.  For every full live-hop window in the buffer, extract one CENS
              vector via ChromaMaker and call OTW.insert().
          d.  Convert the returned reference frame index to seconds and store
              it as current_position.
    """

    # Paper-tuned defaults (kept as class-level constants for readability)
    _SR            = 44100
    _N_FFT         = 8192
    _REF_HOP_LEN   = 4096
    _LIVE_HOP_LEN  = 3686   # ~90 % of ref hop — important per paper
    _C             = 300
    _MAX_RUN_COUNT = 3
    _DIAG_WEIGHT   = 0.4

    def __init__(
        self,
        sr: int            = _SR,
        n_fft: int         = _N_FFT,
        ref_hop_len: int   = _REF_HOP_LEN,
        live_hop_len: int  = _LIVE_HOP_LEN,
        c: int             = _C,
        max_run_count: int = _MAX_RUN_COUNT,
        diag_weight: float = _DIAG_WEIGHT,
    ):
        super().__init__(name="OTW-ConcertCue")

        self.sr           = sr
        self.n_fft        = n_fft
        self.ref_hop_len  = ref_hop_len
        self.live_hop_len = live_hop_len
        self._otw_params  = {
            "c":             c,
            "max_run_count": max_run_count,
            "diag_weight":   diag_weight,
        }

        # Populated by load_reference()
        self._ref_cens:    Optional[np.ndarray] = None   # shape [12, N_ref]
        self._n_ref_frames: int   = 0
        self._ref_duration: float = 0.0

        # Runtime state — (re-)created by _init_runtime_state()
        self._otw:          Optional[OTW]         = None
        self._chroma_maker: Optional[ChromaMaker] = None
        self._buf:          np.ndarray            = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # BaseScoreFollower interface
    # ------------------------------------------------------------------

    def load_reference(self, reference_path: str) -> None:
        """
        Load a reference audio (WAV / MP3 / FLAC) or MIDI file.

        Extracts CENS features at self.sr using the same ChromaMaker that
        will be used for live audio, then initialises the OTW cost matrix.
        """
        print(f"[OTWModel] Loading reference: {reference_path}")

        audio = self._load_audio(reference_path)

        self._ref_cens      = audio_to_np_cens(audio, self.sr, self.n_fft, self.ref_hop_len)
        self._n_ref_frames  = self._ref_cens.shape[1]
        self._ref_duration  = self._n_ref_frames * self.ref_hop_len / self.sr
        self.reference_score = reference_path

        self._init_runtime_state()

        print(
            f"[OTWModel] Reference loaded: {self._n_ref_frames} frames, "
            f"duration: {self._ref_duration:.2f}s"
        )

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Accept one chunk of live audio and advance the OTW alignment.

        The chunk may arrive at any sample rate; it is resampled to self.sr
        internally.  Each call may execute 0 or more OTW steps depending on
        how many full live-hop windows have accumulated in the buffer.

        Returns a dict with keys: position (s), confidence [0,1], tempo (BPM),
        latency (ms).
        """
        t0 = time.time()

        if self._otw is None or self._ref_cens is None:
            raise RuntimeError("Reference not loaded. Call load_reference() first.")

        # Resample to OTW internal rate if necessary
        chunk = audio_frame.astype(np.float32)
        if sample_rate != self.sr:
            chunk = librosa.resample(chunk, orig_sr=sample_rate, target_sr=self.sr)

        # Accumulate into the rolling buffer
        self._buf = np.concatenate([self._buf, chunk])

        # Process every full live-hop window available in the buffer.
        # Each window is n_fft samples wide; consecutive windows overlap by
        # (n_fft - live_hop_len) samples — matching the reference extraction.
        while len(self._buf) >= self.n_fft:
            window  = self._buf[: self.n_fft]
            cens    = self._chroma_maker.insert(window)
            ref_idx = self._otw.insert(cens)

            # Convert reference frame index → seconds
            self.current_position = ref_idx * self.ref_hop_len / self.sr

            # Slide buffer forward by one live hop
            self._buf = self._buf[self.live_hop_len :]

        latency = (time.time() - t0) * 1000  # ms

        return {
            "position":   float(self.current_position),
            "confidence": self._confidence(),
            "tempo":      0.0,
            "latency":    latency,
        }

    def reset(self) -> None:
        """Reset alignment state for a new piece.  Reference CENS is kept."""
        self.current_position = 0.0
        if self._ref_cens is not None:
            self._init_runtime_state()
        print("[OTWModel] Reset.")

    def requires_training(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_runtime_state(self) -> None:
        """Create a fresh OTW engine and ChromaMaker for a new alignment run."""
        self._otw          = OTW(self._ref_cens, self._otw_params)
        self._chroma_maker = ChromaMaker(self.sr, self.n_fft)
        self._buf          = np.empty(0, dtype=np.float32)

    def _confidence(self) -> float:
        """
        Cosine similarity between the current reference and live CENS vectors.

        Both vectors are L2-normalised (guaranteed by ChromaMaker), so the dot
        product gives a value in [0, 1] that reflects how well the live audio
        matches the reference at the estimated position.
        Returns 0.0 before any frame has been processed.
        """
        if self._otw is None or self._otw.t < 0:
            return 0.0

        j        = self._otw.last_j
        t        = self._otw.t
        ref_vec  = self._ref_cens[:, min(j, self._n_ref_frames - 1)]
        live_vec = self._otw.live[:, t]
        return float(np.clip(np.dot(ref_vec, live_vec), 0.0, 1.0))

    def _load_audio(self, path: str) -> np.ndarray:
        """
        Load audio from a WAV/MP3/FLAC file or synthesise from MIDI.
        Always returns a float32 mono array resampled to self.sr.
        """
        ext = os.path.splitext(path)[1].lower()

        if ext in (".wav", ".mp3", ".flac", ".ogg", ".aac"):
            audio, sr = librosa.load(path, sr=None, mono=True)
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            return audio.astype(np.float32)

        if ext in (".mid", ".midi"):
            return self._synth_midi(path)

        raise ValueError(f"Unsupported reference format: '{ext}'")

    def _synth_midi(self, midi_path: str) -> np.ndarray:
        """
        Synthesise MIDI to audio via pretty_midi / FluidSynth.
        Falls back to silence if FluidSynth is not available (common on Windows).
        """
        try:
            import pretty_midi
            pm    = pretty_midi.PrettyMIDI(midi_path)
            audio = pm.fluidsynth(fs=self.sr)
            return audio.astype(np.float32)
        except Exception as exc:
            duration_est = 60
            print(
                f"[OTWModel] MIDI synthesis failed ({exc}). "
                f"Falling back to {duration_est}s of silence as reference."
            )
            return np.zeros(int(duration_est * self.sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
#
# Existing experiment scripts that import DTWModel or OnlineTimeWarping
# continue to work unchanged.  Both are now thin wrappers around OTWModel.
# ---------------------------------------------------------------------------

class DTWModel(OTWModel):
    """Backwards-compatible alias for OTWModel (replaces the old fastdtw placeholder)."""

    def __init__(self, window_size: int = 100, hop_length: int = 512,
                 feature_type: str = "chroma", **kwargs):
        # The old parameters (window_size, hop_length, feature_type) are no
        # longer used; the real OTW algorithm manages its own windowing.
        super().__init__(**kwargs)
        self.name = "OTW-ConcertCue"


class OnlineTimeWarping(OTWModel):
    """Backwards-compatible alias for OTWModel (replaces the old search-margin variant)."""

    def __init__(self, window_size: int = 100, search_margin: int = 50, **kwargs):
        # search_margin is superseded by the OTW 'c' parameter.
        super().__init__(**kwargs)
        self.name = "OTW-ConcertCue"
