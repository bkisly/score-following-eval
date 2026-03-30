"""
HeurMiT — Neural Score Follower
================================
Based on: "A Neural Score Follower for Computer Accompaniment of Polyphonic
Musical Instruments", Ashwin Pillay, Carnegie Mellon University, 2024.

Architecture: MiniTyke
  - Ec : Conv1d(128 → 64, k=3, pad=1) + ReLU   [context encoder]
  - Ew : Conv1d(128 → 64, k=3, pad=1) + ReLU   [window encoder]
  - Cross-correlation in latent space → probability vector P'
  - Heuristic Decision Maker: ring buffer + linear regression

Training: MAESTRO v3 dataset (MIDI train split), MSF-S sampling strategy.
  - Binary piano rolls (velocity > 0 → 1)
  - MIDIOgre-style MIDI augmentations applied to the window
  - Cross-entropy loss with AdamW + cosine-annealing LR scheduler

Inference:
  - Reference MIDI → binary piano roll (fps=100)
  - Incoming audio chunks accumulated into a rolling audio buffer
  - CQT-based piano roll extraction from the audio buffer
  - MiniTyke forward + heuristics → position in seconds
"""

import os
import sys
import json
import csv
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pretty_midi
from scipy.signal import find_peaks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.score_follower import ScoreFollower
from utils.midi_processing import MIDIProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Neural architecture
# ─────────────────────────────────────────────────────────────────────────────

class MiniTyke(nn.Module):
    """
    Compact 1-D CNN encoder for piano rolls.

    Exactly as described in Table 5.1 of the thesis:
      Conv1d(128, 64, kernel_size=3, stride=1, padding=1) + ReLU
      ~24 640 parameters per encoder.

    Works on inputs of any time-axis length (temporally equivariant).
    """

    def __init__(self, input_channels: int = 128, latent_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_channels, latent_dim,
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 128, T]
        Returns : [B, 64, T]  (time dimension preserved)
        """
        return self.relu(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# MIDIOgre-style piano-roll augmentations  (Table 5.2 configurations)
# ─────────────────────────────────────────────────────────────────────────────

def _pitch_shift_piano_roll(roll: np.ndarray, max_shift: int = 5, p: float = 0.1) -> np.ndarray:
    """Randomly shift all note rows up or down by ±max_shift semitones."""
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        roll[shift:, :] = roll[:-shift, :]
        roll[:shift, :] = 0
    elif shift < 0:
        roll[:shift, :] = roll[-shift:, :]
        roll[shift:, :] = 0
    return roll


def _onset_time_shift_piano_roll(roll: np.ndarray, max_shift: int = 5, p: float = 0.1) -> np.ndarray:
    """Randomly shift the entire piano roll along the time axis by ±max_shift frames."""
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        roll[:, shift:] = roll[:, :-shift]
        roll[:, :shift] = 0
    elif shift < 0:
        roll[:, :shift] = roll[:, -shift:]
        roll[:, shift:] = 0
    return roll


def _duration_shift_piano_roll(roll: np.ndarray, max_frac: float = 0.25, p: float = 0.1) -> np.ndarray:
    """Randomly stretch or compress note durations by ±max_frac fraction."""
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    frac = np.random.uniform(-max_frac, max_frac)
    if abs(frac) < 1e-3:
        return roll
    T = roll.shape[1]
    new_T = max(1, int(round(T * (1.0 + frac))))
    resized = np.zeros_like(roll)
    for n in range(roll.shape[0]):
        if roll[n].any():
            stretched = np.interp(
                np.linspace(0, T - 1, new_T),
                np.arange(T), roll[n].astype(float)
            )
            copy_len = min(T, new_T)
            resized[n, :copy_len] = (stretched[:copy_len] > 0.5).astype(roll.dtype)
    return resized


def _note_delete_piano_roll(roll: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Randomly zero out entire pitch rows with probability p per pitch."""
    roll = roll.copy()
    mask = np.random.random(roll.shape[0]) < p
    roll[mask, :] = 0
    return roll


def _note_add_piano_roll(roll: np.ndarray,
                          note_range: Tuple[int, int] = (20, 120),
                          dur_range:  Tuple[int, int] = (2, 10),
                          p: float = 0.1) -> np.ndarray:
    """Randomly insert spurious note activations."""
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    T = roll.shape[1]
    n_add = max(1, int(T * 0.01))
    for _ in range(n_add):
        pitch = np.random.randint(*note_range)
        start = np.random.randint(0, max(1, T - dur_range[1]))
        dur   = np.random.randint(*dur_range)
        end   = min(T, start + dur)
        roll[pitch, start:end] = 1
    return roll


def apply_midi_augmentations(roll: np.ndarray) -> np.ndarray:
    """Apply all five MIDIOgre augmentations sequentially (Table 5.2 settings)."""
    roll = _pitch_shift_piano_roll(roll, max_shift=5, p=0.1)
    roll = _onset_time_shift_piano_roll(roll, max_shift=5, p=0.1)
    roll = _duration_shift_piano_roll(roll, max_frac=0.25, p=0.1)
    roll = _note_delete_piano_roll(roll, p=0.1)
    roll = _note_add_piano_roll(roll, note_range=(20, 120), dur_range=(2, 10), p=0.1)
    return roll


# ─────────────────────────────────────────────────────────────────────────────
# MAESTRO dataset reader — returns train-split MIDI paths
# ─────────────────────────────────────────────────────────────────────────────

def _read_maestro_train_midi_paths(dataset_path: str) -> List[str]:
    """Return absolute paths of MIDI files in the MAESTRO *train* split."""
    dataset_path = Path(dataset_path)
    json_meta = dataset_path / "maestro-v3.0.0.json"
    csv_meta  = dataset_path / "maestro-v3.0.0.csv"

    if json_meta.exists():
        with open(json_meta, "r") as f:
            meta = json.load(f)
        paths = [
            str(dataset_path / meta["midi_filename"][k])
            for k, split in meta["split"].items()
            if split == "train"
        ]
    elif csv_meta.exists():
        with open(csv_meta, newline="") as f:
            reader = csv.DictReader(f)
            paths = [
                str(dataset_path / row["midi_filename"])
                for row in reader
                if row["split"] == "train"
            ]
    else:
        raise FileNotFoundError(
            f"MAESTRO metadata not found in '{dataset_path}'. "
            "Expected 'maestro-v3.0.0.json' or 'maestro-v3.0.0.csv'."
        )
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────────────────────────────────────

class HeurMiTModel(ScoreFollower):
    """
    HeurMiT score follower — O(1) CNN-based real-time position tracker.

    Parameters
    ----------
    latent_dim : int
        Latent channel dimension e (default 64, as in MiniTyke).
    c : int
        Context length in piano-roll frames (default 512 ≈ 5.1 s at 100 fps).
    w : int
        Window length in piano-roll frames (default 256 ≈ 2.56 s at 100 fps).
    fps : int
        Piano-roll frames per second (default 100, matches MIDIProcessor).
    device : str | None
        PyTorch device; auto-detects CUDA if None.
    ring_buffer_size : int
        Number of recent predictions stored by the heuristic ring buffer.
    stabilization_steps : int
        Steps before multi-peak heuristics engage (default 5).
    max_consecutive_buffer : int
        Maximum consecutive fallback-to-buffer steps before forcing the model.
    """

    # Heuristic validation constants (optimal values from thesis Section 5.2)
    _VALID_BEHIND = -48    # frames the prediction may lag behind buffer estimate
    _VALID_AHEAD  =  96    # frames the prediction may lead the buffer estimate
    _RATE_LOW     = 0.5    # minimum rate-of-change relative to expected
    _RATE_HIGH    = 1.5    # maximum rate-of-change relative to expected

    def __init__(self,
                 latent_dim:              int           = 64,
                 c:                       int           = 512,
                 w:                       int           = 256,
                 fps:                     int           = 100,
                 device:                  Optional[str] = None,
                 ring_buffer_size:        int           = 20,
                 stabilization_steps:     int           = 5,
                 max_consecutive_buffer:  int           = 5):

        super().__init__(name="CNN-HeurMiT")

        self.latent_dim       = latent_dim
        self.c                = c
        self.w                = w
        self.fps              = fps
        self.ring_buffer_size = ring_buffer_size
        self.stab_steps       = stabilization_steps
        self.max_consec_buf   = max_consecutive_buffer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ── Two separate encoders, same architecture, different weights ──
        self.Ec = MiniTyke(input_channels=128, latent_dim=latent_dim).to(device)
        self.Ew = MiniTyke(input_channels=128, latent_dim=latent_dim).to(device)

        # ── Reference (loaded per piece) ──
        self.reference_piano_roll: Optional[np.ndarray] = None
        self.midi_duration: float = 0.0

        # ── Rolling audio buffer (accumulates incoming chunks) ──
        self._audio_buffer:    List[np.ndarray] = []
        self._audio_buffer_sr: int              = 22050

        # ── Heuristic state ──
        self._pred_buffer:      deque           = deque(maxlen=ring_buffer_size)
        self._consec_buf:       int             = 0
        self._step:             int             = 0
        self._prev_pred_frame:  Optional[float] = None
        self._context_start:    int             = 0

        print(f"[HeurMiT] Initialized  device={device}  c={c}  w={w}  fps={fps}  "
              f"params={sum(p.numel() for p in list(self.Ec.parameters())+list(self.Ew.parameters())):,}")

    # =========================================================================
    # ScoreFollower interface
    # =========================================================================

    def load_reference(self, reference_path: str) -> None:
        """Load a reference MIDI file and build its binary piano roll."""
        midi = pretty_midi.PrettyMIDI(reference_path)
        raw  = midi.get_piano_roll(fs=self.fps)          # [128, T], velocity values
        # Binarise (Algorithm 1: velocity > 0 → 1)
        self.reference_piano_roll = (raw > 0).astype(np.float32)
        self.midi_duration = midi.get_end_time()
        print(f"[HeurMiT] Reference loaded: shape={self.reference_piano_roll.shape} "
              f"duration={self.midi_duration:.2f}s")

    def reset(self) -> None:
        """Reset all mutable state before starting a new piece."""
        self.current_position = 0.0
        self._context_start   = 0
        self._audio_buffer    = []
        self._pred_buffer     = deque(maxlen=self.ring_buffer_size)
        self._consec_buf      = 0
        self._step            = 0
        self._prev_pred_frame = None

    def requires_training(self) -> bool:
        return True

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Process one audio chunk and return the estimated score position.

        Steps:
          1. Accumulate audio into a rolling buffer.
          2. Convert the last w-worth of samples to a binary piano roll (CQT).
          3. Retrieve the context window C from the reference piano roll.
          4. Encode C and W through Ec / Ew.
          5. Compute cross-correlation P' in latent space.
          6. Apply heuristic decision-making to obtain the final position.
          7. Slide the context window for the next call.
        """
        t0 = time.time()

        if self.reference_piano_roll is None:
            raise RuntimeError("Call load_reference() before process_frame().")

        # ── Fallback for untrained model (linear extrapolation) ──
        if not self.is_trained:
            chunk_dur = len(audio_frame) / sample_rate
            self.current_position = min(
                self.current_position + chunk_dur,
                self.midi_duration
            )
            return {
                "position":   self.current_position,
                "confidence": 0.0,
                "tempo":      0.0,
                "latency":    (time.time() - t0) * 1000,
            }

        # ── 1. Accumulate audio ──
        self._audio_buffer.append(audio_frame)
        self._audio_buffer_sr = sample_rate

        audio_cat     = np.concatenate(self._audio_buffer)
        window_samples = int(self.w * sample_rate / self.fps)

        # Trim buffer to 2× window to keep memory bounded
        max_keep = 2 * window_samples
        if len(audio_cat) > max_keep:
            audio_cat         = audio_cat[-max_keep:]
            self._audio_buffer = [audio_cat]

        # ── 2. Build window piano roll from audio ──
        audio_window = audio_cat[-window_samples:] if len(audio_cat) >= window_samples else audio_cat
        W_np = self._audio_to_piano_roll(audio_window, sample_rate)   # [128, w]

        # ── 3. Retrieve context from reference piano roll ──
        ref   = self.reference_piano_roll
        T_ref = ref.shape[1]
        ctx_end   = min(self._context_start + self.c, T_ref)
        ctx_start = max(0, ctx_end - self.c)
        C_np  = ref[:, ctx_start:ctx_end].copy()         # [128, c_actual]
        c_actual = C_np.shape[1]
        if c_actual < self.c:                            # pad at end-of-piece
            C_np = np.pad(C_np, ((0, 0), (0, self.c - c_actual)))

        # ── 4. Encode and cross-correlate ──
        with torch.no_grad():
            C_t     = torch.from_numpy(C_np).unsqueeze(0).to(self.device)    # [1,128,c]
            W_t     = torch.from_numpy(W_np).unsqueeze(0).to(self.device)    # [1,128,w]
            C_prime = self.Ec(C_t).squeeze(0)                                # [64, c]
            W_prime = self.Ew(W_t).squeeze(0)                                # [64, w]
            P_prime = self._cross_correlate(C_prime, W_prime)                # [c+w-1]
            P_np_arr = P_prime.cpu().numpy()

        # ── 5. Heuristic decision ──
        frame_in_ctx  = self._heuristic_decision(P_np_arr)    # offset within context
        abs_frame     = float(np.clip(ctx_start + max(0, frame_in_ctx), 0, T_ref - 1))

        # ── 6. Slide context window (centred on predicted position) ──
        new_ctx_start = int(abs_frame) - self.c // 2
        self._context_start = int(np.clip(new_ctx_start, 0, max(0, T_ref - self.c)))

        # ── 7. Convert to seconds ──
        predicted_time = float(abs_frame / self.fps)
        self.current_position = predicted_time

        confidence = float(torch.softmax(P_prime, dim=0).max().item())
        tempo      = self._estimate_tempo()

        return {
            "position":   predicted_time,
            "confidence": confidence,
            "tempo":      tempo,
            "latency":    (time.time() - t0) * 1000,
        }

    # =========================================================================
    # Training
    # =========================================================================

    def train(self, train_data: Any = None, **kwargs) -> None:
        """
        Train MiniTyke on MAESTRO (train split).

        Parameters
        ----------
        train_data : str | dict
            • str  — path to the MAESTRO root directory.
            • dict — config dict with keys:
                'dataset_path'       : str   (required)
                'epochs'             : int   (default 50)
                'batch_size'         : int   (default 64)
                'samples_per_epoch'  : int   (default 500)
                'val_samples'        : int   (default 50)
                'lr'                 : float (default 5e-4)
                'weight_decay'       : float (default 1e-2)
                'save_path'          : str   (checkpoint path, optional)
        """
        # ── Parse configuration ──
        if isinstance(train_data, str):
            cfg = {"dataset_path": train_data}
        elif isinstance(train_data, dict):
            cfg = dict(train_data)
        else:
            raise ValueError(
                "train_data must be a path string or a config dict "
                "with at least {'dataset_path': '/path/to/maestro'}."
            )

        dataset_path      = cfg["dataset_path"]
        epochs            = int(cfg.get("epochs",            50))
        batch_size        = int(cfg.get("batch_size",        64))
        samples_per_epoch = int(cfg.get("samples_per_epoch", 500))
        val_samples       = int(cfg.get("val_samples",       50))
        lr                = float(cfg.get("lr",              5e-4))
        weight_decay      = float(cfg.get("weight_decay",    1e-2))
        save_path         = cfg.get("save_path", None)

        # ── Load MIDI paths ──
        print(f"[HeurMiT] Reading MAESTRO train split from '{dataset_path}' …")
        midi_paths = _read_maestro_train_midi_paths(dataset_path)
        if not midi_paths:
            raise RuntimeError("No train-split MIDI files found.")
        print(f"[HeurMiT] Found {len(midi_paths)} train-split MIDI files.")

        # Pre-load up to 100 piano rolls into memory
        cache_size = min(len(midi_paths), 100)
        print(f"[HeurMiT] Pre-loading {cache_size} piano rolls …")
        rolls_cache = self._build_piano_roll_cache(midi_paths[:cache_size])
        if not rolls_cache:
            raise RuntimeError("Could not load any piano rolls. Check MAESTRO paths.")
        print(f"[HeurMiT] Cached {len(rolls_cache)} piano rolls.")

        # ── Optimiser (AdamW) + cosine-annealing scheduler ──
        params    = list(self.Ec.parameters()) + list(self.Ew.parameters())
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                      betas=(0.9, 0.999))
        T_max     = max(10, epochs // 4)          # quarter-cycle length
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=T_max, eta_min=1e-6
        )

        # ── Training loop ──
        self.Ec.train()
        self.Ew.train()
        best_val_acc = 0.0
        best_state   = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(
                rolls_cache, samples_per_epoch, batch_size, optimiser, augment=True
            )
            self.Ec.eval()
            self.Ew.eval()
            val_loss, val_acc, val_bacc = self._run_validation(rolls_cache, val_samples)
            self.Ec.train()
            self.Ew.train()
            scheduler.step()

            ratio = val_acc / max(train_acc, 1e-6)
            print(f"[HeurMiT] Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
                  f"val_bacc={val_bacc:.1f}%  ratio={ratio:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {
                    "Ec": {k: v.cpu().clone() for k, v in self.Ec.state_dict().items()},
                    "Ew": {k: v.cpu().clone() for k, v in self.Ew.state_dict().items()},
                }

        # Restore best weights
        if best_state is not None:
            self.Ec.load_state_dict(best_state["Ec"])
            self.Ew.load_state_dict(best_state["Ew"])
            self.Ec = self.Ec.to(self.device)
            self.Ew = self.Ew.to(self.device)
            print(f"[HeurMiT] Restored best checkpoint (val_acc={best_val_acc:.1f}%)")

        self.Ec.eval()
        self.Ew.eval()
        self.is_trained = True

        if save_path:
            self.save_checkpoint(save_path)
            print(f"[HeurMiT] Saved checkpoint → {save_path}")

    # =========================================================================
    # Checkpoint I/O
    # =========================================================================

    def save_checkpoint(self, path: str) -> None:
        """Persist model weights and configuration to a .pth file."""
        torch.save({
            "Ec": self.Ec.state_dict(),
            "Ew": self.Ew.state_dict(),
            "latent_dim": self.latent_dim,
            "c":   self.c,
            "w":   self.w,
            "fps": self.fps,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore model weights from a .pth checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.Ec.load_state_dict(ckpt["Ec"])
        self.Ew.load_state_dict(ckpt["Ew"])
        self.Ec.eval()
        self.Ew.eval()
        self.is_trained = True
        print(f"[HeurMiT] Loaded checkpoint '{path}'.")

    # =========================================================================
    # Private — neural
    # =========================================================================

    def _cross_correlate(self,
                         C_prime: torch.Tensor,
                         W_prime: torch.Tensor) -> torch.Tensor:
        """
        Cross-correlation P' = C' ⋆ W'  (Equation 3.3 of the thesis).

        C_prime : [e, c]  — latent context
        W_prime : [e, w]  — latent window
        Returns  : [c + w - 1]  — raw correlation scores

        PyTorch's F.conv1d computes cross-correlation (not flipped convolution),
        which matches the paper's definition directly.

        With (w−1) zero-padding on both sides of C',the output at index k equals:
          P'[k] = Σ_e Σ_t  C'[e, k+t] · W'[e, t]   for k = 0 … c+w−2
        so that the ground-truth label (right-edge position in context) equals k.
        """
        e, c = C_prime.shape
        _, w = W_prime.shape

        # Pad C' with (w-1) zeros on each side → [1, e, c + 2(w-1)]
        C_padded = F.pad(C_prime.unsqueeze(0), (w - 1, w - 1))

        # W' as convolution kernel → [out=1, in=e, w]
        kernel  = W_prime.unsqueeze(0)

        # F.conv1d output: [1, 1, (c+2(w-1)) - w + 1] = [1, 1, c+w-1]
        P_prime = F.conv1d(C_padded, kernel)
        return P_prime.squeeze()                      # [c+w-1]

    def _batch_cross_correlate(self,
                                C_prime: torch.Tensor,
                                W_prime: torch.Tensor) -> torch.Tensor:
        """
        Batched version of _cross_correlate.

        C_prime : [B, e, c]
        W_prime : [B, e, w]
        Returns  : [B, c + w - 1]
        """
        return torch.stack(
            [self._cross_correlate(C_prime[i], W_prime[i])
             for i in range(C_prime.shape[0])],
            dim=0
        )

    # =========================================================================
    # Private — heuristic decision maker  (Section 3.2.3)
    # =========================================================================

    def _heuristic_decision(self, P_np: np.ndarray) -> int:
        """
        Apply the heuristic rules from Section 3.2.3 of the thesis to select
        the final window position within the current context.

        Returns the predicted *frame offset within context* (may be negative if
        the window leads the context start), derived from the cross-correlation
        index k via:  frame_offset = k - (w - 1)
        """
        self._step += 1
        len_P = len(P_np)

        # ── Rule 1: Smooth P' and find significant peaks ──
        smoothed = np.convolve(P_np, np.ones(5) / 5, mode="same")
        peaks, _  = find_peaks(smoothed, prominence=3)

        if len(peaks) == 0:
            model_k = int(np.argmax(smoothed))
        else:
            # Use the peak with the highest smoothed value
            model_k = int(peaks[int(np.argmax(smoothed[peaks]))])

        # ── Rule 2: Stabilisation phase ──
        if self._step <= self.stab_steps:
            self._pred_buffer.append(model_k)
            self._prev_pred_frame = float(model_k)
            return model_k - (self.w - 1)

        # ── Rule 3: Linear regression on ring buffer ──
        buf_arr = np.array(list(self._pred_buffer), dtype=float)
        if len(buf_arr) >= 2:
            xs     = np.arange(len(buf_arr), dtype=float)
            coeffs = np.polyfit(xs, buf_arr, 1)          # [slope, intercept]
            slope  = coeffs[0]
            buf_k  = int(np.clip(
                round(coeffs[1] + slope * len(buf_arr)),
                0, len_P - 1
            ))
            expected_delta = max(1.0, abs(slope))
        else:
            buf_k          = model_k
            slope          = 1.0
            expected_delta = 1.0

        # ── Rule 4: Validate model prediction ──
        prev = self._prev_pred_frame if self._prev_pred_frame is not None else float(model_k)
        delta = model_k - prev

        mono_ok  = model_k >= prev + self._VALID_BEHIND
        range_ok = (buf_k + self._VALID_BEHIND) <= model_k <= (buf_k + self._VALID_AHEAD)
        rate_ok  = (self._RATE_LOW * expected_delta
                    <= abs(delta)
                    <= self._RATE_HIGH * expected_delta)

        valid = mono_ok and range_ok and rate_ok

        # ── Rule 5: Choose final k ──
        if valid:
            final_k      = model_k
            self._consec_buf = 0
        elif self._consec_buf >= self.max_consec_buf:
            # Too many consecutive buffer fallbacks → force model
            final_k          = model_k
            self._consec_buf = 0
        else:
            mean_k = int(round((model_k + buf_k) / 2.0))
            if abs(mean_k - model_k) < abs(mean_k - buf_k):
                final_k = mean_k
            else:
                final_k = buf_k
            self._consec_buf += 1

        final_k = int(np.clip(final_k, 0, len_P - 1))
        self._pred_buffer.append(final_k)
        self._prev_pred_frame = float(final_k)
        return final_k - (self.w - 1)

    def _estimate_tempo(self) -> float:
        """
        Estimate BPM from the linear slope of the ring buffer.
        Returns 0.0 if insufficient history.
        """
        buf = list(self._pred_buffer)
        if len(buf) < 4:
            return 0.0
        xs     = np.arange(len(buf), dtype=float)
        coeffs = np.polyfit(xs, buf, 1)
        slope  = coeffs[0]                # frames per inference step
        if slope <= 0:
            return 0.0
        # slope [frames/step] × fps [frames/s]⁻¹ → steps/s; ×60 → BPM
        bpm = slope * 60.0
        return float(np.clip(bpm, 20.0, 400.0))

    # =========================================================================
    # Private — audio processing
    # =========================================================================

    def _audio_to_piano_roll(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Convert a raw audio segment to a binary [128, w] piano roll.

        Uses librosa CQT over the standard piano MIDI range (MIDI 21–108,
        88 bins) with a hop size matching the target fps, thresholded to
        produce binary activations, then zero-padded into the full 128-bin
        MIDI space and resampled to exactly w columns.
        """
        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) < 64:
            return np.zeros((128, self.w), dtype=np.float32)

        hop_length = max(1, int(round(sample_rate / self.fps)))   # ~220 at 22 050 Hz
        n_bins     = 88                                            # MIDI 21–108
        fmin       = librosa.midi_to_hz(21)                        # A0

        try:
            C_cqt = np.abs(librosa.cqt(
                audio,
                sr=sample_rate,
                hop_length=hop_length,
                n_bins=n_bins,
                bins_per_octave=12,
                fmin=fmin,
            ))                                                     # [88, T_cqt]
        except Exception:
            return np.zeros((128, self.w), dtype=np.float32)

        if C_cqt.shape[1] == 0:
            return np.zeros((128, self.w), dtype=np.float32)

        # Per-frame normalise then binarise at the median active value
        col_max = C_cqt.max(axis=0, keepdims=True)
        col_max = np.where(col_max < 1e-8, 1.0, col_max)
        C_norm  = C_cqt / col_max                                  # [88, T_cqt]

        active_vals = C_norm[C_norm > 0]
        threshold   = float(np.median(active_vals)) if len(active_vals) > 0 else 0.5
        C_bin       = (C_norm > threshold).astype(np.float32)      # [88, T_cqt]

        # Embed into full 128-bin MIDI array (MIDI offsets 21–108)
        T_cqt          = C_bin.shape[1]
        full_roll      = np.zeros((128, T_cqt), dtype=np.float32)
        full_roll[21:109, :] = C_bin                               # 88 bins → 21..108

        # Resample to exactly w frames (nearest-neighbour)
        if T_cqt == self.w:
            return full_roll
        src_idx = np.round(np.linspace(0, T_cqt - 1, self.w)).astype(int)
        src_idx = np.clip(src_idx, 0, T_cqt - 1)
        return full_roll[:, src_idx]

    # =========================================================================
    # Private — training helpers
    # =========================================================================

    @staticmethod
    def _build_piano_roll_cache(midi_paths: List[str]) -> List[np.ndarray]:
        """Pre-load and binarise piano rolls from a list of MIDI file paths."""
        rolls = []
        for path in midi_paths:
            try:
                midi = pretty_midi.PrettyMIDI(path)
                raw  = midi.get_piano_roll(fs=100)
                roll = (raw > 0).astype(np.float32)         # [128, T], binary
                if roll.shape[1] > 0:
                    rolls.append(roll)
            except Exception as exc:
                warnings.warn(f"[HeurMiT] Skipping '{path}': {exc}")
        return rolls

    def _sample_triple(self,
                       rolls_cache: List[np.ndarray],
                       augment: bool
                       ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Sample one (C, W, label_k) training triple.

        label_k is the ground-truth argmax index in P' (length c+w-1):
            label_k = (win_start − ctx_start) + (w − 1)

        Returns None if the selected roll is too short.
        """
        roll = rolls_cache[np.random.randint(len(rolls_cache))]
        T    = roll.shape[1]
        if T < self.c + self.w:
            return None

        # Random context window
        ctx_start = np.random.randint(0, T - self.c + 1)
        ctx_end   = ctx_start + self.c
        C         = roll[:, ctx_start:ctx_end].copy()        # [128, c]

        # Random window entirely within [0, T-w], at least partially within context
        min_ws = max(0, ctx_start - self.w + 1)
        max_ws = min(T - self.w, ctx_end - 1)
        if max_ws < min_ws:
            return None
        win_start = np.random.randint(min_ws, max_ws + 1)
        win_end   = win_start + self.w
        if win_end > T:
            return None
        W = roll[:, win_start:win_end].copy()                # [128, w]

        if augment:
            W = apply_midi_augmentations(W)

        # Ground-truth cross-correlation index (right-edge position)
        label_k = (win_start - ctx_start) + (self.w - 1)
        label_k = int(np.clip(label_k, 0, self.c + self.w - 2))

        return C, W, label_k

    def _run_epoch(self,
                   rolls_cache:  List[np.ndarray],
                   n_samples:    int,
                   batch_size:   int,
                   optimiser:    torch.optim.Optimizer,
                   augment:      bool) -> Tuple[float, float]:
        """Run one training epoch; returns (mean_loss, accuracy%)."""
        total_loss = 0.0
        n_correct  = 0
        n_total    = 0
        tol        = max(1, int(round(0.005 * self.fps)))   # 5 ms tolerance

        C_batch, W_batch, Y_batch = [], [], []

        def _flush():
            nonlocal total_loss, n_correct, n_total
            if not C_batch:
                return
            C_t = torch.from_numpy(np.stack(C_batch)).to(self.device)
            W_t = torch.from_numpy(np.stack(W_batch)).to(self.device)
            Y_t = torch.tensor(Y_batch, dtype=torch.long).to(self.device)

            optimiser.zero_grad()
            C_prime  = self.Ec(C_t)
            W_prime  = self.Ew(W_t)
            P_prime  = self._batch_cross_correlate(C_prime, W_prime)  # [B, c+w-1]
            loss     = F.cross_entropy(P_prime, Y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.Ec.parameters()) + list(self.Ew.parameters()), 1.0
            )
            optimiser.step()

            total_loss += loss.item() * len(Y_batch)
            n_correct  += int(((P_prime.argmax(1) - Y_t).abs() <= tol).sum().item())
            n_total    += len(Y_batch)
            C_batch.clear(); W_batch.clear(); Y_batch.clear()

        sampled = 0
        while sampled < n_samples:
            triple = self._sample_triple(rolls_cache, augment=augment)
            if triple is None:
                continue
            C, W, lk = triple
            C_batch.append(C); W_batch.append(W); Y_batch.append(lk)
            sampled += 1
            if len(C_batch) >= batch_size:
                _flush()
        _flush()

        if n_total == 0:
            return 0.0, 0.0
        return total_loss / n_total, 100.0 * n_correct / n_total

    def _run_validation(self,
                        rolls_cache: List[np.ndarray],
                        n_samples:   int) -> Tuple[float, float, float]:
        """
        Validate on n_samples triples (no augmentation).

        Returns (val_loss, val_acc%, baseline_acc%).
        Baseline: raw (unencoded) piano-roll cross-correlation.
        """
        total_loss_m = total_loss_b = 0.0
        n_corr_m = n_corr_b = n_total = 0
        tol = max(1, int(round(0.005 * self.fps)))

        for _ in range(n_samples):
            triple = self._sample_triple(rolls_cache, augment=False)
            if triple is None:
                continue
            C, W, lk = triple
            C_t = torch.from_numpy(C).unsqueeze(0).to(self.device)
            W_t = torch.from_numpy(W).unsqueeze(0).to(self.device)
            Y_t = torch.tensor([lk], dtype=torch.long).to(self.device)

            with torch.no_grad():
                # Model
                P_model = self._batch_cross_correlate(self.Ec(C_t), self.Ew(W_t))
                lm      = F.cross_entropy(P_model, Y_t)
                n_corr_m     += int(((P_model.argmax(1) - Y_t).abs() <= tol).sum())
                total_loss_m += lm.item()
                # Baseline (raw piano roll, no encoding)
                P_base  = self._batch_cross_correlate(C_t.float(), W_t.float())
                lb      = F.cross_entropy(P_base, Y_t)
                n_corr_b     += int(((P_base.argmax(1) - Y_t).abs() <= tol).sum())
                total_loss_b += lb.item()

            n_total += 1

        if n_total == 0:
            return 0.0, 0.0, 0.0
        return (
            total_loss_m / n_total,
            100.0 * n_corr_m / n_total,
            100.0 * n_corr_b / n_total,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HeurMiT — architecture smoke test")
    print("=" * 60)

    model = HeurMiTModel(latent_dim=64, c=512, w=256, fps=100)
    total_params = (
        sum(p.numel() for p in model.Ec.parameters()) +
        sum(p.numel() for p in model.Ew.parameters())
    )
    print(f"  Name        : {model.name}")
    print(f"  Device      : {model.device}")
    print(f"  Parameters  : {total_params:,}  (expect 49 280)")

    # Cross-correlation shape test
    e, c, w = 64, 512, 256
    Cp = torch.randn(e, c)
    Wp = torch.randn(e, w)
    P  = model._cross_correlate(Cp, Wp)
    assert P.shape == (c + w - 1,), f"Unexpected shape: {P.shape}"
    print(f"  P' shape    : {tuple(P.shape)}  ✓  (expect ({c+w-1},))")

    # Audio-to-piano-roll shape test
    fake_audio = np.random.randn(22050 * 3).astype(np.float32)  # 3 s
    pr = model._audio_to_piano_roll(fake_audio, 22050)
    assert pr.shape == (128, 256), f"Unexpected shape: {pr.shape}"
    print(f"  Piano roll  : {pr.shape}  ✓  (expect (128, 256))")

    print("=" * 60)
    print("All checks passed.")
    print()
    print("To train:")
    print("  model.train({'dataset_path': '/path/to/maestro-v3.0.0'})")
    print("  model.train({'dataset_path': '...', 'epochs': 50, "
          "'save_path': 'heurmit.pth'})")
    print()
    print("To load a saved checkpoint:")
    print("  model.load_checkpoint('heurmit.pth')")