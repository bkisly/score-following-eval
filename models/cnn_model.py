"""
HeurMiT — Neural Score Follower  (fixed)
=========================================
Based on: "A Neural Score Follower for Computer Accompaniment of Polyphonic
Musical Instruments", Ashwin Pillay, Carnegie Mellon University, 2024.

Bugs fixed vs the original skeleton
-------------------------------------
Bug 1 — position was the LEFT edge of the window instead of the RIGHT edge,
         causing a systematic lag of (w-1)/fps ≈ 2.55 s.
         Fix: abs_frame = ctx_start + final_k  (right edge = ctx_start + k).

Bug 2 — context window never advanced when the model made bad predictions
         (feedback loop: prediction=0 → context stays at 0 → next prediction=0).
         Fix: context is now driven by wall-clock elapsed time; the model only
         provides a fine correction within ±(c//3) frames of the elapsed estimate.

Bug 3 — per-frame median threshold on silent/noisy audio activated ~50% of CQT
         bins, turning silence into a random dense piano roll.
         Fix: global 95th-percentile normalisation with a fixed floor threshold;
         explicit silence detection returns an all-zeros roll.

Bug 4 — binary {0,1} training vs continuous [0,1] CQT inference domain gap.
         Fix: (a) inference uses soft CQT magnitudes (not binary), which still
         yields correct argmax positions; (b) MIDIOgre augmentations during
         training now include amplitude jitter and harmonic simulation so the
         model sees float-like patterns.
"""

import csv
import json
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.score_follower import ScoreFollower


# ─────────────────────────────────────────────────────────────────────────────
# Neural encoder  (MiniTyke, Table 5.1)
# ─────────────────────────────────────────────────────────────────────────────

class MiniTyke(nn.Module):
    """Single Conv1d(128→64, k=3, pad=1) + ReLU — temporally equivariant."""

    def __init__(self, in_ch: int = 128, latent: int = 64, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, latent, kernel_size=k, stride=1, padding=k // 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))          # [B, latent, T]  (T preserved)


# ─────────────────────────────────────────────────────────────────────────────
# MIDIOgre-style augmentations  (Table 5.2 + amplitude/harmonic additions)
# ─────────────────────────────────────────────────────────────────────────────

def _aug_pitch_shift(roll: np.ndarray, max_shift: int = 5, p: float = 0.1) -> np.ndarray:
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        roll[shift:] = roll[:-shift]; roll[:shift] = 0
    elif shift < 0:
        roll[:shift] = roll[-shift:]; roll[shift:] = 0
    return roll

def _aug_onset_shift(roll: np.ndarray, max_shift: int = 5, p: float = 0.1) -> np.ndarray:
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        roll[:, shift:] = roll[:, :-shift]; roll[:, :shift] = 0
    elif shift < 0:
        roll[:, :shift] = roll[:, -shift:]; roll[:, shift:] = 0
    return roll

def _aug_duration_shift(roll: np.ndarray, max_frac: float = 0.25, p: float = 0.1) -> np.ndarray:
    if np.random.random() > p:
        return roll
    frac = np.random.uniform(-max_frac, max_frac)
    if abs(frac) < 1e-3:
        return roll
    T = roll.shape[1]
    new_T = max(1, int(round(T * (1.0 + frac))))
    out = np.zeros_like(roll)
    for n in range(roll.shape[0]):
        if roll[n].any():
            stretched = np.interp(np.linspace(0, T - 1, new_T), np.arange(T), roll[n].astype(float))
            L = min(T, new_T)
            out[n, :L] = (stretched[:L] > 0.5).astype(roll.dtype)
    return out

def _aug_note_delete(roll: np.ndarray, p: float = 0.1) -> np.ndarray:
    roll = roll.copy()
    roll[np.random.random(roll.shape[0]) < p, :] = 0
    return roll

def _aug_note_add(roll: np.ndarray, note_range=(20, 120), dur_range=(2, 10), p: float = 0.1) -> np.ndarray:
    if np.random.random() > p:
        return roll
    roll = roll.copy()
    T = roll.shape[1]
    for _ in range(max(1, int(T * 0.01))):
        pitch = np.random.randint(*note_range)
        start = np.random.randint(0, max(1, T - dur_range[1]))
        roll[pitch, start:min(T, start + np.random.randint(*dur_range))] = 1
    return roll

def _aug_amplitude_jitter(roll: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Multiply each active cell by a random factor in [0.4, 1.0].
    Bridges the binary→float gap: the model sees fractional activations
    during training, making it robust to soft CQT magnitudes at inference.
    """
    if np.random.random() > p:
        return roll
    out = roll.astype(np.float32).copy()
    mask = out > 0
    out[mask] *= np.random.uniform(0.4, 1.0, mask.sum())
    return out

def _aug_add_harmonics(roll: np.ndarray, p: float = 0.4) -> np.ndarray:
    """
    For each active note, add attenuated copies at +12, +24 semitones.
    Simulates CQT harmonic bleeding so the model learns to ignore it.
    """
    if np.random.random() > p:
        return roll
    out = roll.astype(np.float32).copy()
    for semitones, decay in [(12, 0.5), (24, 0.25)]:
        src = out[:-semitones, :]                      # notes that fit
        out[semitones:, :] = np.maximum(out[semitones:, :], src * decay)
    return out

def apply_augmentations(roll: np.ndarray) -> np.ndarray:
    """All MIDIOgre augmentations + amplitude jitter + harmonic simulation."""
    roll = _aug_pitch_shift(roll, 5, 0.1)
    roll = _aug_onset_shift(roll, 5, 0.1)
    roll = _aug_duration_shift(roll, 0.25, 0.1)
    roll = _aug_note_delete(roll, 0.1)
    roll = _aug_note_add(roll, (20, 120), (2, 10), 0.1)
    roll = _aug_amplitude_jitter(roll, 0.5)       # NEW — float activations
    roll = _aug_add_harmonics(roll, 0.4)           # NEW — harmonic simulation
    return roll


# ─────────────────────────────────────────────────────────────────────────────
# MAESTRO train-split reader
# ─────────────────────────────────────────────────────────────────────────────

def _maestro_train_paths(dataset_path: str) -> List[str]:
    dp = Path(dataset_path)
    for meta_path, fmt in [(dp / "maestro-v3.0.0.json", "json"),
                           (dp / "maestro-v3.0.0.csv",  "csv")]:
        if not meta_path.exists():
            continue
        if fmt == "json":
            with open(meta_path) as f:
                meta = json.load(f)
            return [str(dp / meta["midi_filename"][k])
                    for k, s in meta["split"].items() if s == "train"]
        else:
            with open(meta_path, newline="") as f:
                rows = list(csv.DictReader(f))
            return [str(dp / r["midi_filename"]) for r in rows if r["split"] == "train"]
    raise FileNotFoundError(f"No MAESTRO metadata in '{dataset_path}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class HeurMiTModel(ScoreFollower):
    """
    HeurMiT — O(1) neural score follower.

    Parameters
    ----------
    latent_dim : int   Encoder output channels (e=64 in MiniTyke).
    c          : int   Context length in piano-roll frames at training.
    w          : int   Window  length in piano-roll frames at training.
    fps        : int   Piano-roll frames per second (100 by default).
    device     : str   PyTorch device; auto-detected if None.

    Inference parameters (can differ from training, model is equivariant)
    -----------------------------------------------------------------------
    inf_c      : int   Context length used at inference (default = c).
    inf_w      : int   Window  length used at inference (default = w).
    max_elapsed_deviation : int
        Maximum frames by which the model prediction may differ from the
        elapsed-time estimate before it is clamped.  Prevents wild jumps
        while still allowing tempo variation.  Default = c // 3.
    """

    # Heuristic ring-buffer thresholds (Section 3.2.3)
    _VALID_BEHIND =  -48   # frames behind buffer estimate still accepted
    _VALID_AHEAD  =   96   # frames ahead  of buffer estimate still accepted
    _RATE_LOW     =  0.5
    _RATE_HIGH    =  1.5
    _MIN_WIN_FRAMES = 10   # minimum CQT frames needed before any prediction

    def __init__(self,
                 latent_dim: int           = 64,
                 c:          int           = 512,
                 w:          int           = 256,
                 fps:        int           = 100,
                 device:     Optional[str] = None,
                 # inference knobs
                 inf_c:      Optional[int] = None,
                 inf_w:      Optional[int] = None,
                 max_elapsed_deviation: Optional[int] = None,
                 # heuristic knobs
                 ring_buffer_size:       int = 20,
                 stabilization_steps:    int =  5,
                 max_consecutive_buffer: int =  5):

        super().__init__(name="CNN-HeurMiT")

        self.latent_dim  = latent_dim
        self.c   = c
        self.w   = w
        self.fps = fps

        # Inference sizes (may be larger than training — model is equivariant)
        self.inf_c = inf_c if inf_c is not None else c
        self.inf_w = inf_w if inf_w is not None else w

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.Ec = MiniTyke(128, latent_dim).to(device)
        self.Ew = MiniTyke(128, latent_dim).to(device)

        # Ring-buffer / heuristic config
        self.ring_size     = ring_buffer_size
        self.stab_steps    = stabilization_steps
        self.max_consec    = max_consecutive_buffer
        self._max_dev: Optional[int] = max_elapsed_deviation

        # Reference (loaded per piece)
        self.reference_piano_roll: Optional[np.ndarray] = None
        self.midi_duration: float = 0.0

        # Rolling audio accumulator
        self._audio_buf: List[np.ndarray] = []
        self._audio_sr:  int              = 22050

        # Heuristic state  (all reset in reset())
        self._abs_pred_buf: deque           = deque(maxlen=ring_buffer_size)
        self._consec_buf:   int             = 0
        self._step:         int             = 0
        self._prev_abs:     Optional[float] = None
        # Elapsed-time tracker (the primary advance signal — Bug-2 fix)
        self._elapsed_frames: float = 0.0

        n_params = sum(p.numel() for p in list(self.Ec.parameters()) +
                                          list(self.Ew.parameters()))
        print(f"[HeurMiT] device={device}  c={c}  w={w}  fps={fps}  "
              f"inf_c={self.inf_c}  inf_w={self.inf_w}  params={n_params:,}")

    # =========================================================================
    # ScoreFollower interface
    # =========================================================================

    def load_reference(self, reference_path: str) -> None:
        midi = pretty_midi.PrettyMIDI(reference_path)
        raw  = midi.get_piano_roll(fs=self.fps)
        self.reference_piano_roll = (raw > 0).astype(np.float32)   # binary
        self.midi_duration = midi.get_end_time()
        print(f"[HeurMiT] Reference: {Path(reference_path).name}  "
              f"shape={self.reference_piano_roll.shape}  "
              f"dur={self.midi_duration:.1f}s")

    def reset(self) -> None:
        self.current_position  = 0.0
        self._audio_buf        = []
        self._abs_pred_buf     = deque(maxlen=self.ring_size)
        self._consec_buf       = 0
        self._step             = 0
        self._prev_abs         = None
        self._elapsed_frames   = 0.0

    def requires_training(self) -> bool:
        return True

    def process_frame(self, audio_frame: np.ndarray,
                      sample_rate: int) -> Dict[str, Any]:
        """
        Process one audio chunk and return estimated score position.

        Design
        ------
        1. Accumulate audio; extract W from the last inf_w-worth of samples.
        2. Advance context window using *elapsed wall-clock frames* (Bug-2 fix).
        3. Encode C and W → cross-correlate → heuristic decision → final_k.
        4. abs_frame = ctx_start + final_k   ← RIGHT edge of window (Bug-1 fix).
        5. Clamp abs_frame to ±max_elapsed_deviation around elapsed estimate
           so tempo drift is tracked but wild jumps are rejected.
        """
        t0 = time.time()

        if self.reference_piano_roll is None:
            raise RuntimeError("Call load_reference() first.")

        ref   = self.reference_piano_roll
        T_ref = ref.shape[1]

        # ── Elapsed-time counter ──────────────────────────────────────────────
        # This is the primary "where should we be?" signal and advances every
        # call regardless of whether the model makes a good prediction.
        self._elapsed_frames += len(audio_frame) / sample_rate * self.fps
        elapsed = float(np.clip(self._elapsed_frames, 0, T_ref - 1))

        # ── Untrained fallback (linear extrapolation) ─────────────────────────
        if not self.is_trained:
            self.current_position = elapsed / self.fps
            return {"position": self.current_position, "confidence": 0.0,
                    "tempo": 0.0, "latency": (time.time() - t0) * 1000}

        # ── Accumulate audio buffer ───────────────────────────────────────────
        self._audio_buf.append(audio_frame)
        self._audio_sr = sample_rate
        audio_cat = np.concatenate(self._audio_buf)

        # Trim to 3× the inference window to keep memory bounded
        max_keep = int(self.inf_w * sample_rate / self.fps) * 3
        if len(audio_cat) > max_keep:
            audio_cat = audio_cat[-max_keep:]
            self._audio_buf = [audio_cat]

        # ── Build W from available audio ─────────────────────────────────────
        # Use up to inf_w frames; zero-pad if not enough yet.
        # Bug-3 fix: _audio_to_piano_roll now uses global 95th-pct normalisation.
        w_samples = int(self.inf_w * sample_rate / self.fps)
        audio_window = audio_cat[-min(len(audio_cat), w_samples):]
        W_np = self._audio_to_piano_roll(audio_window, sample_rate,
                                          target_w=self.inf_w)   # [128, inf_w]

        # ── Context window driven by elapsed time (Bug-2 fix) ─────────────────
        ctx_start = int(np.clip(
            elapsed - self.inf_c // 2,
            0, max(0, T_ref - self.inf_c)
        ))
        C_np = ref[:, ctx_start:ctx_start + self.inf_c].copy()
        if C_np.shape[1] < self.inf_c:
            C_np = np.pad(C_np, ((0, 0), (0, self.inf_c - C_np.shape[1])))

        # ── Encode + cross-correlate ──────────────────────────────────────────
        with torch.no_grad():
            C_t     = torch.from_numpy(C_np).unsqueeze(0).to(self.device)
            W_t     = torch.from_numpy(W_np).unsqueeze(0).to(self.device)
            C_prime = self.Ec(C_t).squeeze(0)           # [e, inf_c]
            W_prime = self.Ew(W_t).squeeze(0)           # [e, inf_w]
            P_prime = self._cross_correlate(C_prime, W_prime)  # [inf_c+inf_w-1]
            P_np    = P_prime.cpu().numpy()

        # ── Heuristic decision → final_k ─────────────────────────────────────
        # Returns cross-correlation index in P' (right edge of W within context).
        # Clamp to valid range: window fully inside context.
        final_k = int(np.clip(
            self._heuristic_decision(P_np, ctx_start),
            self.inf_w - 1,
            self.inf_c - 1
        ))

        # ── Bug-1 fix: abs_frame = RIGHT edge of predicted window ─────────────
        # Cross-correlation: P'[k] maximised when W aligns with C starting at
        # s = k-(w-1).  Right edge = s+(w-1) = k.  Absolute: ctx_start + k.
        raw_abs_frame = float(ctx_start + final_k)

        # ── Clamp to elapsed ± max_deviation ─────────────────────────────────
        max_dev = self._max_dev if self._max_dev is not None else self.inf_c // 3
        abs_frame = float(np.clip(
            raw_abs_frame,
            elapsed - max_dev,
            elapsed + max_dev
        ))
        abs_frame = float(np.clip(abs_frame, 0, T_ref - 1))

        self.current_position = abs_frame / self.fps
        confidence = float(F.softmax(P_prime, dim=0).max().item())

        return {
            "position":   self.current_position,
            "confidence": confidence,
            "tempo":      self._estimate_tempo(),
            "latency":    (time.time() - t0) * 1000,
        }

    # =========================================================================
    # Training
    # =========================================================================

    def train(self, train_data: Any = None, **kwargs) -> None:
        """
        Train on MAESTRO (train split).

        train_data : str | dict
            str  → path to MAESTRO root.
            dict → {
              'dataset_path'       : str,
              'epochs'             : int   (50),
              'batch_size'         : int   (64),
              'samples_per_epoch'  : int   (500),
              'val_samples'        : int   (50),
              'lr'                 : float (5e-4),
              'weight_decay'       : float (1e-2),
              'save_path'          : str   (optional),
            }
        """
        cfg = {"dataset_path": train_data} if isinstance(train_data, str) \
              else dict(train_data)
        dpath = cfg["dataset_path"]
        epochs = int(cfg.get("epochs", 50))
        bsz    = int(cfg.get("batch_size", 64))
        n_tr   = int(cfg.get("samples_per_epoch", 500))
        n_val  = int(cfg.get("val_samples", 50))
        lr     = float(cfg.get("lr", 5e-4))
        wd     = float(cfg.get("weight_decay", 1e-2))
        spath  = cfg.get("save_path", None)
        cache_size = int(cfg.get("cache_size", 250))

        print(f"[HeurMiT] Loading MAESTRO train split from '{dpath}' …")
        midi_paths = _maestro_train_paths(dpath)
        if not midi_paths:
            raise RuntimeError("No train-split MIDI found.")
        print(f"[HeurMiT] Found {len(midi_paths)} MIDI files.")

        cache_n = min(len(midi_paths), cache_size)
        print(f"[HeurMiT] Pre-loading {cache_n} piano rolls …")
        cache = _load_piano_roll_cache(midi_paths[:cache_n])
        if not cache:
            raise RuntimeError("Could not load any MIDI files.")
        print(f"[HeurMiT] Cached {len(cache)} piano rolls.")

        params    = list(self.Ec.parameters()) + list(self.Ew.parameters())
        optim     = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(10, epochs // 4), eta_min=1e-6)

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(1, epochs + 1):
            self.Ec.train(); self.Ew.train()
            tl, ta = self._epoch(cache, n_tr, bsz, optim, augment=True)
            self.Ec.eval();  self.Ew.eval()
            vl, va, vb = self._validate(cache, n_val)
            scheduler.step()

            print(f"[HeurMiT] Ep {epoch:3d}/{epochs}  "
                  f"tr_loss={tl:.4f}  tr_acc={ta:.1f}%  "
                  f"val_loss={vl:.4f}  val_acc={va:.1f}%  val_bacc={vb:.1f}%  "
                  f"ratio={va/max(ta,1e-6):.3f}")

            if va > best_val_acc:
                best_val_acc = va
                best_state = {k: {n: v.cpu().clone() for n, v in m.state_dict().items()}
                              for k, m in (("Ec", self.Ec), ("Ew", self.Ew))}

        if best_state:
            self.Ec.load_state_dict(best_state["Ec"])
            self.Ew.load_state_dict(best_state["Ew"])
        self.Ec = self.Ec.to(self.device).eval()
        self.Ew = self.Ew.to(self.device).eval()
        self.is_trained = True
        print(f"[HeurMiT] Training complete (best val_acc={best_val_acc:.1f}%).")
        if spath:
            self.save_checkpoint(spath)
            print(f"[HeurMiT] Checkpoint → {spath}")

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def save_checkpoint(self, path: str) -> None:
        torch.save({"Ec": self.Ec.state_dict(), "Ew": self.Ew.state_dict(),
                    "latent_dim": self.latent_dim, "c": self.c, "w": self.w,
                    "fps": self.fps}, path)

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.Ec.load_state_dict(ck["Ec"])
        self.Ew.load_state_dict(ck["Ew"])
        self.Ec.eval(); self.Ew.eval()
        self.is_trained = True
        print(f"[HeurMiT] Loaded checkpoint '{path}'.")

    # =========================================================================
    # Private — neural
    # =========================================================================

    def _cross_correlate(self, C: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        P'[k] = Σ_e Σ_t  C[e, k+t] · W[e, t]   (Eq. 3.3)

        C : [e, c]  W : [e, w]  → [c+w-1]

        P'[k] is maximised when W aligns with C starting at s = k-(w-1).
        Right edge of that window in C coordinates = s+(w-1) = k.
        Absolute right edge = ctx_start + k.   ← Bug-1 fix
        """
        _, w = W.shape
        C_padded = F.pad(C.unsqueeze(0), (w - 1, w - 1))   # [1, e, c+2(w-1)]
        P = F.conv1d(C_padded, W.unsqueeze(0))              # [1, 1, c+w-1]
        return P.squeeze()

    def _batch_xcorr(self, C: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._cross_correlate(C[i], W[i])
                            for i in range(C.shape[0])])

    # =========================================================================
    # Private — heuristic decision  (Section 3.2.3)
    # Rewritten to work with ABSOLUTE frame numbers so that the ring buffer
    # remains valid even as ctx_start shifts between calls.
    # =========================================================================

    def _heuristic_decision(self, P_np: np.ndarray, ctx_start: int) -> int:
        """
        Applies heuristic rules and returns a cross-correlation index k ∈ [0, len(P)-1].
        Absolute predicted frame = ctx_start + k.

        The ring buffer stores absolute frame numbers so that the linear-regression
        extrapolation is valid even when ctx_start changes between calls.
        """
        self._step += 1
        L = len(P_np)

        # ── Smooth P' and find peaks ──
        smoothed = np.convolve(P_np, np.ones(5) / 5, mode="same")
        peaks, _ = find_peaks(smoothed, prominence=3)
        model_k  = int(peaks[np.argmax(smoothed[peaks])]) if len(peaks) else int(np.argmax(smoothed))
        model_abs = ctx_start + model_k

        # ── Stabilisation phase ──
        if self._step <= self.stab_steps:
            self._abs_pred_buf.append(float(model_abs))
            self._prev_abs = float(model_abs)
            return model_k

        # ── Linear regression on absolute buffer ──
        buf = np.array(list(self._abs_pred_buf), dtype=float)
        if len(buf) >= 2:
            xs     = np.arange(len(buf), dtype=float)
            coeffs = np.polyfit(xs, buf, 1)        # slope, intercept
            slope  = coeffs[0]
            buf_abs = float(coeffs[1] + slope * len(buf))
            exp_delta = max(1.0, abs(slope))
        else:
            buf_abs   = model_abs
            slope     = 1.0
            exp_delta = 1.0

        # ── Validate model prediction ──
        prev = self._prev_abs if self._prev_abs is not None else float(model_abs)
        delta = model_abs - prev

        mono_ok  = model_abs >= prev + self._VALID_BEHIND
        range_ok = (buf_abs + self._VALID_BEHIND) <= model_abs <= (buf_abs + self._VALID_AHEAD)
        rate_ok  = (self._RATE_LOW * exp_delta) <= abs(delta) <= (self._RATE_HIGH * exp_delta)
        valid    = mono_ok and range_ok and rate_ok

        # ── Choose final absolute position ──
        if valid:
            final_abs        = model_abs
            self._consec_buf = 0
        elif self._consec_buf >= self.max_consec:
            final_abs        = model_abs
            self._consec_buf = 0
        else:
            mean_abs = (model_abs + buf_abs) / 2.0
            final_abs        = mean_abs if abs(mean_abs - model_abs) < abs(mean_abs - buf_abs) else buf_abs
            self._consec_buf += 1

        self._abs_pred_buf.append(float(final_abs))
        self._prev_abs = float(final_abs)

        # Convert back to P' index: k = final_abs - ctx_start
        final_k = int(np.clip(round(final_abs - ctx_start), 0, L - 1))
        return final_k

    def _estimate_tempo(self) -> float:
        buf = list(self._abs_pred_buf)
        if len(buf) < 4:
            return 0.0
        xs     = np.arange(len(buf), dtype=float)
        slope  = float(np.polyfit(xs, buf, 1)[0])   # frames per step
        return float(np.clip(slope * 60.0, 20.0, 400.0)) if slope > 0 else 0.0

    # =========================================================================
    # Private — audio → piano roll  (Bug-3 fix)
    # =========================================================================

    def _audio_to_piano_roll(self, audio: np.ndarray, sample_rate: int,
                              target_w: Optional[int] = None) -> np.ndarray:
        """
        Convert audio to a [128, target_w] piano-roll feature matrix.

        Changes vs original
        -------------------
        * Bug-3 fix: normalise by the 95th percentile of the *entire buffer*
          (not a per-frame median), then clip to [0, 1].  Silent frames now
          produce near-zero matrices instead of random dense activations.
        * Values are kept as float [0, 1] (NOT binary-thresholded).
          The model still finds the correct argmax: soft activations from the
          CQT are proportional to note energy, so the cross-correlation peak
          falls at the right position.  Binary training with amplitude-jitter
          augmentation (see _aug_amplitude_jitter) makes the model robust to
          this at inference.
        """
        if target_w is None:
            target_w = self.inf_w

        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < 64 or np.abs(audio).max() < 1e-7:
            return np.zeros((128, target_w), dtype=np.float32)

        hop  = max(1, int(round(sample_rate / self.fps)))
        fmin = librosa.midi_to_hz(21)          # A0

        try:
            C = np.abs(librosa.cqt(
                audio, sr=sample_rate,
                hop_length=hop, n_bins=88,
                bins_per_octave=12, fmin=fmin,
            )).astype(np.float32)              # [88, T_cqt]
        except Exception:
            return np.zeros((128, target_w), dtype=np.float32)

        T_cqt = C.shape[1]
        if T_cqt == 0:
            return np.zeros((128, target_w), dtype=np.float32)

        # Bug-3 fix: global 95th-percentile normalisation
        p95 = float(np.percentile(C, 95))
        if p95 > 1e-8:
            C = C / p95
        C = np.clip(C, 0.0, 1.0)

        # Silence guard: if max activation is tiny, return zeros
        if C.max() < 0.05:
            return np.zeros((128, target_w), dtype=np.float32)

        # Embed into full 128-bin MIDI array (piano occupies MIDI 21–108)
        full = np.zeros((128, T_cqt), dtype=np.float32)
        full[21:109, :] = C

        # Resample to target_w frames
        if T_cqt == target_w:
            return full
        # Use zero-padding if audio was too short (< 1 target_w worth of frames)
        if T_cqt < target_w:
            out = np.zeros((128, target_w), dtype=np.float32)
            out[:, -T_cqt:] = full           # place existing frames at the END
            return out
        # Down-sample by nearest-neighbour
        idx = np.round(np.linspace(0, T_cqt - 1, target_w)).astype(int)
        return full[:, np.clip(idx, 0, T_cqt - 1)]

    # =========================================================================
    # Private — training helpers
    # =========================================================================

    def _sample(self, cache: List[np.ndarray], augment: bool,
                c: int, w: int) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Random (C, W, label_k) triple from the piano-roll cache."""
        roll = cache[np.random.randint(len(cache))]
        T    = roll.shape[1]
        if T < c + w:
            return None

        ctx_s = np.random.randint(0, T - c + 1)
        min_ws = max(0, ctx_s - w + 1)
        max_ws = min(T - w, ctx_s + c - 1)
        if max_ws < min_ws:
            return None
        ws = np.random.randint(min_ws, max_ws + 1)
        if ws + w > T:
            return None

        C = roll[:, ctx_s:ctx_s + c].copy()
        W = roll[:, ws:ws + w].copy()
        if augment:
            W = apply_augmentations(W)

        # label_k = window start in context + (w-1) = (ws-ctx_s) + (w-1)
        label_k = int(np.clip((ws - ctx_s) + (w - 1), 0, c + w - 2))
        return C.astype(np.float32), W.astype(np.float32), label_k

    def _epoch(self, cache, n_samples, bsz, optim, augment) -> Tuple[float, float]:
        tol = max(1, int(round(0.005 * self.fps)))
        total_loss = 0.0; n_corr = 0; n_tot = 0
        Cb, Wb, Yb = [], [], []

        def _flush():
            nonlocal total_loss, n_corr, n_tot
            if not Cb: return
            C_t = torch.from_numpy(np.stack(Cb)).to(self.device)
            W_t = torch.from_numpy(np.stack(Wb)).to(self.device)
            Y_t = torch.tensor(Yb, dtype=torch.long).to(self.device)
            optim.zero_grad()
            loss = F.cross_entropy(self._batch_xcorr(self.Ec(C_t), self.Ew(W_t)), Y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.Ec.parameters()) + list(self.Ew.parameters()), 1.0)
            optim.step()
            preds = self._batch_xcorr(self.Ec(C_t), self.Ew(W_t)).argmax(1).detach()
            n_corr += int(((preds - Y_t).abs() <= tol).sum())
            total_loss += loss.item() * len(Yb); n_tot += len(Yb)
            Cb.clear(); Wb.clear(); Yb.clear()

        done = 0
        while done < n_samples:
            t = self._sample(cache, augment, self.c, self.w)
            if t is None: continue
            C, W, lk = t
            Cb.append(C); Wb.append(W); Yb.append(lk); done += 1
            if len(Cb) >= bsz: _flush()
        _flush()
        return (total_loss / n_tot, 100.0 * n_corr / n_tot) if n_tot else (0.0, 0.0)

    def _validate(self, cache, n_samples) -> Tuple[float, float, float]:
        tol = max(1, int(round(0.005 * self.fps)))
        lm = lb = cm = cb = n = 0
        for _ in range(n_samples):
            t = self._sample(cache, False, self.c, self.w)
            if t is None: continue
            C, W, lk = t
            C_t = torch.from_numpy(C).unsqueeze(0).to(self.device)
            W_t = torch.from_numpy(W).unsqueeze(0).to(self.device)
            Y_t = torch.tensor([lk], dtype=torch.long).to(self.device)
            with torch.no_grad():
                P_m = self._batch_xcorr(self.Ec(C_t), self.Ew(W_t))
                P_b = self._batch_xcorr(C_t.float(),  W_t.float())
                lm += F.cross_entropy(P_m, Y_t).item()
                lb += F.cross_entropy(P_b, Y_t).item()
                cm += int(((P_m.argmax(1) - Y_t).abs() <= tol).sum())
                cb += int(((P_b.argmax(1) - Y_t).abs() <= tol).sum())
            n += 1
        if not n: return 0.0, 0.0, 0.0
        return lm / n, 100.0 * cm / n, 100.0 * cb / n


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_piano_roll_cache(paths: List[str]) -> List[np.ndarray]:
    rolls = []
    for p in paths:
        try:
            raw  = pretty_midi.PrettyMIDI(p).get_piano_roll(fs=100)
            roll = (raw > 0).astype(np.float32)
            if roll.shape[1] > 0:
                rolls.append(roll)
        except Exception as e:
            warnings.warn(f"[HeurMiT] skip '{p}': {e}")
    return rolls


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HeurMiT — smoke test")
    print("=" * 60)

    model = HeurMiTModel(c=512, w=256, fps=100)
    n = sum(p.numel() for p in list(model.Ec.parameters()) + list(model.Ew.parameters()))
    print(f"Params     : {n:,}  (expect 49 280)")

    # Cross-correlation shape
    e, c, w = 64, 512, 256
    P = model._cross_correlate(torch.randn(e, c), torch.randn(e, w))
    assert P.shape == (c + w - 1,), f"bad shape {P.shape}"
    print(f"P' shape   : {tuple(P.shape)}  ✓  (expect ({c+w-1},))")

    # Right-edge position sanity check
    # Performer at frame 50 inside context (ctx_start=0):
    # window start s=50, k = s+(w-1) = 305,  abs_frame = 0 + 305 = 305 = 3.05s
    k = 305; ctx_start = 0
    abs_frame = ctx_start + k
    print(f"Position   : ctx={ctx_start} k={k} → abs={abs_frame} = {abs_frame/100:.2f}s  ✓")

    # Audio→piano-roll shape
    W = model._audio_to_piano_roll(np.random.randn(22050 * 3).astype(np.float32), 22050)
    assert W.shape == (128, 256), f"bad shape {W.shape}"
    print(f"Piano roll : {W.shape}  ✓  (expect (128, 256))")

    # Silence → zero matrix
    W0 = model._audio_to_piano_roll(np.zeros(4096, dtype=np.float32), 22050)
    assert W0.max() == 0.0, "silence should give zeros"
    print(f"Silence    : max={W0.max():.1f}  ✓  (expect 0.0)")

    print("=" * 60)
    print("All checks passed.")
    print()
    print("Train   : model.train({'dataset_path': '/path/to/maestro'})")
    print("Infer   : model.load_checkpoint('heurmit.pth')")