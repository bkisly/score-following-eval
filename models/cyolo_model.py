"""
CYOLO-SB+A binding — score following via score sheet object detection.

Wraps the pretrained CPJKU CYOLO model (models/cyolo/) in the BaseScoreFollower
interface.  Follows the real-time inference loop from models/cyolo/
cyolo_score_following/test.py exactly.

Reference loading supports:
  * MSMD .npz file  — score images + annotation data for precise time conversion
  * PNG / JPG image — score image only; position is returned as normalised [0,1]
"""

import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Add CYOLO source directory to sys.path (must be done before any
# cyolo_score_following imports, which happen lazily inside _load_network).
# ---------------------------------------------------------------------------
_CYOLO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyolo")
if _CYOLO_SRC not in sys.path:
    sys.path.insert(0, _CYOLO_SRC)

# Sentinel used in _load_network to distinguish "attribute absent" from None
_MISSING = object()

# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.base_model import BaseScoreFollower


class CYOLOModel(BaseScoreFollower):
    """
    Real-time score following binding for the pretrained CYOLO-SB+A model.

    The model takes a 416×416 score sheet image and a streaming audio signal and
    returns the bounding box of the currently played note / bar / system.  Here
    we expose only the note-level (class 0) prediction and convert its centre
    x-coordinate to a time position in seconds using MSMD onset annotations.

    Inference follows test.py from the CPJKU repository:
      • network.compute_spec()                   — log-frequency spectrogram (78 bins)
      • conditioning_network.get_conditioning()  — streaming LSTM hidden state update
      • network.predict()                        — YOLO forward pass with FiLM conditioning

    Parameters
    ----------
    param_path : str, optional
        Path to a CYOLO .pt checkpoint.  Defaults to the bundled
        cyolo_sb_a/best_model.pt pretrained model.
    device : str, optional
        PyTorch device string ('cpu', 'cuda', 'cuda:0', …).
        Defaults to CUDA when available, otherwise CPU.
    """

    DEFAULT_CHECKPOINT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cyolo", "trained_models", "cyolo_sb_a", "best_model.pt",
    )
    SCALE_WIDTH: int = 416  # YOLO input image size (pixels, square)

    # Audio processing constants from cyolo_score_following/utils/data_utils.py
    _CYOLO_SR: int = 22050
    _HOP_SIZE: int = 1102
    _CYOLO_FPS: float = 22050 / 1102  # ≈ 20.0 frames/second

    def __init__(
        self,
        param_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name="CYOLO-SB+A")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.param_path = param_path or self.DEFAULT_CHECKPOINT

        # Score data (populated by load_reference)
        self.score_tensor: Optional[torch.Tensor] = None  # [n_pages, 1, 416, 416]
        self.scale_factor: float = 1.0   # original_height / SCALE_WIDTH
        self.pad: int = 0                # horizontal pixels added to make image square

        # Annotation-based time conversion (MSMD .npz only)
        self.interpol_fnc = None         # frame_idx → (y, x, sys, bar, page)
        self.interpol_c2o: Optional[Dict] = None  # page → interp(x_unrolled → onset_frame)
        self.staff_coords: Optional[Dict[int, List[float]]] = None
        self.add_per_staff: Optional[Dict[int, np.ndarray]] = None

        # Runtime state (reset between pieces)
        self.hidden = None               # LSTM (h, c) tuple
        self.elapsed_samples: int = 0   # total samples seen at _CYOLO_SR
        self.current_page: int = 0

        # Load pretrained network
        self._load_network()

    # ------------------------------------------------------------------
    # Network loading
    # ------------------------------------------------------------------

    def _load_network(self) -> None:
        """
        Import and load the pretrained CYOLO network.

        Two compatibility patches are applied for the duration of the import:

        1. numpy — madmom 0.16.1 (used by custom_modules.py for the log-frequency
           filterbank) references type aliases removed in numpy 1.24 (np.int,
           np.float, …).  We restore them as aliases to the built-in types.

        2. torchaudio — custom_modules.py calls torchaudio.set_audio_backend("sox_io")
           at module level.  On Windows (or torchaudio >= 2.0 where the API was
           removed) this raises RuntimeError / AttributeError.  We replace the
           attribute with a no-op for the duration of the import.
        """
        import numpy as np

        # Patch 1: numpy type-alias compatibility for madmom 0.16.1
        _numpy_aliases = {
            "int": int, "float": float, "complex": complex,
            "bool": bool, "object": object, "str": str,
        }
        _numpy_added: list = []
        for _alias, _builtin in _numpy_aliases.items():
            if not hasattr(np, _alias):
                setattr(np, _alias, _builtin)
                _numpy_added.append(_alias)

        import torchaudio  # type: ignore[import]

        # Patch 2: torchaudio sox_io backend
        _orig_backend = getattr(torchaudio, "set_audio_backend", _MISSING)
        torchaudio.set_audio_backend = lambda *_: None  # type: ignore[attr-defined]

        try:
            from cyolo_score_following.models.yolo import load_pretrained_model  # type: ignore[import]
        finally:
            if _orig_backend is not _MISSING:
                torchaudio.set_audio_backend = _orig_backend  # type: ignore[attr-defined]
            # Remove the numpy aliases we added (leave pre-existing ones untouched)
            for _alias in _numpy_added:
                try:
                    delattr(np, _alias)
                except AttributeError:
                    pass

        self.network, _ = load_pretrained_model(self.param_path)
        self.network.to(self.device)
        self.network.eval()
        self.is_trained = True

    # ------------------------------------------------------------------
    # BaseScoreFollower interface
    # ------------------------------------------------------------------

    def load_reference(self, reference_path: str) -> None:
        """
        Load the score reference.

        Parameters
        ----------
        reference_path : str
            Path to an MSMD .npz file **or** a single score image (.png / .jpg).

            .npz  — provides score images + onset annotations; enables accurate
                    time conversion and automatic page tracking.
            image — score image only; position is returned as a normalised
                    fraction [0, 1] of the score width.
        """
        import cv2  # type: ignore[import]

        self.reference_score = reference_path

        if reference_path.lower().endswith(".npz"):
            self._load_npz(reference_path, cv2)
        elif reference_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            self._load_image(reference_path, cv2)
        else:
            raise ValueError(
                f"Unsupported reference format for CYOLO: {reference_path!r}. "
                "Expected an MSMD .npz file or a score image (.png / .jpg)."
            )

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Process one audio frame and return the estimated score position.

        Follows the per-frame inference loop from test.py:
          1. Resample to 22 050 Hz if necessary.
          2. Compute log-frequency spectrogram (78 bands).
          3. Advance page counter via annotation lookup (with .npz only).
          4. Update streaming LSTM conditioning via get_conditioning().
          5. Run YOLO predict() on the current score page.
          6. Extract highest-confidence note detection (class 0).
          7. Convert predicted x-coordinate to time in seconds.

        Parameters
        ----------
        audio_frame : np.ndarray
            Raw audio samples (any length; resampled to 22 050 Hz internally).
        sample_rate : int
            Sample rate of *audio_frame*.

        Returns
        -------
        dict with keys 'position' (s), 'confidence' [0,1], 'tempo' (BPM),
        'latency' (ms).
        """
        t0 = time.time()

        if self.score_tensor is None:
            raise RuntimeError("load_reference() must be called before process_frame().")

        # 1. Resample to CYOLO sample rate when necessary
        audio = audio_frame.astype(np.float32)
        if sample_rate != self._CYOLO_SR:
            import librosa  # type: ignore[import]
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self._CYOLO_SR)

        sig = torch.from_numpy(audio).to(self.device)

        with torch.no_grad():
            # 2. Log-frequency spectrogram ([T, 78])
            spec_frame = self.network.compute_spec([sig], tempo_aug=False)[0]

            # 3. Page tracking via annotation interpolation (.npz only)
            if self.interpol_fnc is not None:
                frame_idx = int(self.elapsed_samples / self._HOP_SIZE)
                true_pos = np.asarray(self.interpol_fnc(frame_idx), dtype=np.float32)
                new_page = int(true_pos[-1])
                if new_page != self.current_page:
                    # Reset LSTM hidden state on page change (mirrors test.py behaviour)
                    self.current_page = new_page
                    self.hidden = None

            # 4. Streaming LSTM conditioning update
            z, self.hidden = self.network.conditioning_network.get_conditioning(
                spec_frame, hidden=self.hidden
            )

            # 5. YOLO forward pass on current page
            inference_out, _ = self.network.predict(
                self.score_tensor[self.current_page : self.current_page + 1], z
            )

        # 6. Best note detection (class 0 = Note in CYOLO-SB+A)
        note_mask = inference_out[0, :, -1] == 0
        note_dets = inference_out[0, note_mask]  # [n, 6]: xywh + conf + class_idx

        if len(note_dets) > 0:
            _, top_idx = torch.sort(note_dets[:, 4], descending=True)
            best = note_dets[top_idx[0]]
            x_c = best[0].item()   # centre x in 416-space
            y_c = best[1].item()   # centre y in 416-space
            confidence = best[4].item()

            # 7. Convert to time in seconds
            pos = self._bbox_center_to_time(x_c, y_c, self.current_page)
            if pos is None:
                # Fallback when annotations are unavailable: normalised [0, 1]
                pos = x_c / self.SCALE_WIDTH
        else:
            confidence = 0.0
            pos = self.current_position

        self.current_position = float(pos)
        self.elapsed_samples += len(audio)

        return {
            "position": self.current_position,
            "confidence": float(confidence),
            "tempo": 0.0,
            "latency": (time.time() - t0) * 1000,
        }

    def reset(self) -> None:
        """Reset all runtime state before processing a new piece."""
        self.hidden = None
        self.elapsed_samples = 0
        self.current_page = 0
        self.current_position = 0.0
        if self.network is not None:
            # Clear the LSTM streaming deque and step counter in get_conditioning()
            self.network.conditioning_network.inference_x.clear()
            self.network.conditioning_network.step_cnt = 0

    def requires_training(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_npz(self, npz_path: str, cv2: Any) -> None:
        """
        Load an MSMD .npz file.

        Replicates the score-loading and annotation-processing logic from
        data_utils.load_piece() and data_utils.load_sequences() without
        loading the paired WAV file (audio is provided via process_frame).
        """
        from scipy import interpolate  # type: ignore[import]

        npzfile = np.load(npz_path, allow_pickle=True)
        scores_raw = npzfile["sheets"]         # [n_pages, H, W]  uint8
        coords_raw = list(npzfile["coords"])   # list of onset dicts
        systems    = list(npzfile["systems"])  # list of system geometry dicts

        _, h, w = scores_raw.shape
        dim_diff = abs(h - w)
        pad1 = dim_diff // 2
        pad2 = dim_diff - pad1
        self.pad = pad1

        # Pad width to square (MSMD score images are portrait; padding is horizontal)
        padded = np.pad(
            scores_raw, ((0, 0), (0, 0), (pad1, pad2)),
            mode="constant", constant_values=255,
        )

        # Adjust x coordinates to account for the added padding
        coords: List[Dict] = []
        for c in coords_raw:
            c = dict(c)
            if c["note_x"] > 0:
                c["note_x"] += pad1
            coords.append(c)
        for s in systems:
            s["x"] += pad1

        # Normalise to float [0,1] with CYOLO convention: 1 = ink, 0 = background
        scores_float = 1.0 - padded.astype(np.float32) / 255.0
        self.scale_factor = float(scores_float.shape[1]) / self.SCALE_WIDTH  # H / 416

        scaled = [
            cv2.resize(s, (self.SCALE_WIDTH, self.SCALE_WIDTH), interpolation=cv2.INTER_AREA)
            for s in scores_float
        ]
        self.score_tensor = (
            torch.from_numpy(np.stack(scaled)).unsqueeze(1).to(self.device)
        )  # [n_pages, 1, 416, 416]

        # ------------------------------------------------------------------
        # Build onset-indexed annotation arrays  (mirrors load_piece logic)
        # ------------------------------------------------------------------

        # Convert onset times (seconds) → frame indices at CYOLO FPS
        for c in coords:
            c["onset"] = int(c["onset"] * self._CYOLO_FPS)

        onsets_all = np.unique([c["onset"] for c in coords]).astype(np.int64)

        # Deduplicate: one representative entry per unique onset frame
        coords_new_list: List[List[float]] = []
        for onset in onsets_all:
            grp = [c for c in coords if c["onset"] == onset]
            merged: Dict[str, list] = {}
            for entry in grp:
                for key, val in entry.items():
                    merged.setdefault(key, []).append(val)

            system_idx = int(Counter(merged["system_idx"]).most_common(1)[0][0])
            note_x = float(np.mean([
                merged["note_x"][i]
                for i in range(len(merged["system_idx"]))
                if merged["system_idx"][i] == system_idx
            ]))
            page_nr  = int(Counter(merged["page_nr"]).most_common(1)[0][0])
            bar_idx  = int(Counter(merged["bar_idx"]).most_common(1)[0][0])

            note_y = float(systems[system_idx]["y"]) if note_x > 0 else -1.0
            coords_new_list.append([note_y, note_x, system_idx, bar_idx, page_nr])

        coords_new = np.asarray(coords_new_list, dtype=np.float32)
        # shape [n_onsets, 5]: [note_y, note_x, sys_idx, bar_idx, page_nr]

        # interpol_fnc: frame_idx → (y, x, sys, bar, page)  — page tracking in process_frame
        self.interpol_fnc = interpolate.interp1d(
            onsets_all, coords_new.T,
            kind="previous", bounds_error=False,
            fill_value=(coords_new[0, :], coords_new[-1, :]),  # type: ignore[arg-type]
        )

        # ------------------------------------------------------------------
        # Per-page coord-to-onset interpolators  (mirrors load_sequences logic)
        # Maps unrolled-x pixel coordinate → onset frame number
        # ------------------------------------------------------------------
        self.interpol_c2o = {}
        self.staff_coords  = {}
        self.add_per_staff = {}

        for page_nr in np.unique(coords_new[:, -1]).astype(int):
            mask         = coords_new[:, -1] == page_nr
            page_coords  = coords_new[mask]   # [m, 5]
            page_onsets  = onsets_all[mask]   # [m]

            staff_ys: List[float] = sorted(np.unique(page_coords[:, 0]).tolist())
            self.staff_coords[page_nr] = staff_ys

            # Accumulate staff widths to form the unrolled x axis
            max_xes = [0.0]
            coords_per_staff = []
            for y in staff_ys:
                cs = page_coords[page_coords[:, 0] == y, :-1]  # drop page_nr column
                coords_per_staff.append(cs)
                max_xes.append(float(np.max(cs[:, 1])))

            add_ps = np.cumsum(max_xes)[:-1]
            self.add_per_staff[page_nr] = add_ps

            unrolled_x = np.concatenate([
                cs[:, 1] + add_ps[i]
                for i, cs in enumerate(coords_per_staff)
            ])

            # page_onsets temporal order matches staff spatial order for standard notation
            self.interpol_c2o[page_nr] = interpolate.interp1d(
                unrolled_x, page_onsets,
                kind="nearest", bounds_error=False,
                fill_value=(float(page_onsets[0]), float(page_onsets[-1])),  # type: ignore[arg-type]
            )

    def _load_image(self, image_path: str, cv2: Any) -> None:
        """
        Load a single score image file.

        No annotation data is available in this mode; position is returned as a
        normalised fraction [0, 1] of the score width.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot open score image: {image_path!r}")

        h, w = img.shape
        dim_diff = abs(h - w)
        pad1 = dim_diff // 2
        pad2 = dim_diff - pad1
        self.pad = pad1

        if w < h:
            img = np.pad(img, ((0, 0), (pad1, pad2)), constant_values=255)
        elif h < w:
            img = np.pad(img, ((pad1, pad2), (0, 0)), constant_values=255)

        img_float = 1.0 - img.astype(np.float32) / 255.0
        self.scale_factor = float(img_float.shape[0]) / self.SCALE_WIDTH

        img_scaled = cv2.resize(
            img_float, (self.SCALE_WIDTH, self.SCALE_WIDTH), interpolation=cv2.INTER_AREA
        )
        self.score_tensor = (
            torch.from_numpy(img_scaled[None, None]).to(self.device)
        )  # [1, 1, 416, 416]

        self.interpol_fnc  = None
        self.interpol_c2o  = None
        self.staff_coords  = None
        self.add_per_staff = None

    def _bbox_center_to_time(
        self, x_scaled: float, y_scaled: float, page_nr: int
    ) -> Optional[float]:
        """
        Convert a YOLO bounding-box centre (in 416-space) to time in seconds.

        1. Scale from 416-space back to original image pixel coordinates.
        2. Find the nearest staff row by y-coordinate.
        3. Compute the unrolled x offset (staves concatenated horizontally).
        4. Interpolate to find the nearest annotated onset frame.
        5. onset_frame × HOP_SIZE / SAMPLE_RATE → seconds.

        Returns None when annotation data is unavailable.
        """
        if (
            self.interpol_c2o  is None
            or self.staff_coords  is None
            or self.add_per_staff is None
            or page_nr not in self.interpol_c2o
        ):
            return None

        # Convert 416-space coordinates back to original image pixel coordinates
        x_orig = x_scaled * self.scale_factor
        y_orig = y_scaled * self.scale_factor

        staff_ys = self.staff_coords[page_nr]
        staff_id = int(np.argmin([abs(y_orig - sy) for sy in staff_ys]))

        x_unrolled  = x_orig + self.add_per_staff[page_nr][staff_id]
        onset_frame = float(self.interpol_c2o[page_nr](x_unrolled))

        return onset_frame * self._HOP_SIZE / self._CYOLO_SR


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("CYOLO-SB+A binding — smoke test")
    model = CYOLOModel()
    print(f"  name             : {model.name}")
    print(f"  requires_training: {model.requires_training()}")
    print(f"  is_trained       : {model.is_trained}")
    print(f"  device           : {model.device}")
    print("Model loaded successfully.")
