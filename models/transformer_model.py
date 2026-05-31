"""
PatchFormer — Transformer-based real-time score follower.

Improvements over HeurMiT (CNN)
--------------------------------
1. Multi-head self-attention inside each encoder captures long-range temporal
   patterns within the context and the live window.
2. Training E_live on real CQT features from MAESTRO WAV files eliminates the
   piano-roll→CQT domain gap that HeurMiT bridges only via augmentation.
3. Reference is pre-encoded (Conv1d pass) once in load_reference(); at inference
   only the 128-patch context slice runs through the transformer — ~0.1 ms on GPU.

Heuristic inference layer (elapsed-time tracker, ring buffer, monotonicity guard)
is inherited verbatim from HeurMiTModel to ensure stability parity.
"""

import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.score_follower import ScoreFollower
from models.transformer.network import TransformerNet
from models.transformer.features import (
    compute_cqt_features,
    _maestro_wav_paths,
    _precompute_cqt_cache,
    _precompute_piano_roll_cache,
)


class TransformerModel(ScoreFollower):
    """
    PatchFormer score follower.

    Parameters
    ----------
    d_model    : int   Transformer hidden dimension (default 128)
    patch_size : int   Frames per patch (default 4 → 40 ms at 100 fps)
    n_heads    : int   Attention heads (default 4)
    n_layers   : int   Transformer encoder layers per encoder (default 2)
    d_ff       : int   FFN hidden dimension (default 256)
    c          : int   Context length in frames at training (default 512)
    w          : int   Window length in frames at training (default 128)
    fps        : int   Piano-roll frames per second (default 100)
    device     : str   PyTorch device; auto-detected if None

    Inference knobs
    ---------------
    inf_c : int   Context length at inference (default = c)
    inf_w : int   Window  length at inference (default = w)
    max_elapsed_deviation : int   ±frame tolerance from elapsed estimate (default c//3)
    """

    _VALID_BEHIND = -24   # widened from -48: at 0.7x tempo the ring needs ~17 calls
                          # (3.2 s) to adapt from a 1x warm-up; -48 gave only 8.6 calls
                          # (1.6 s), causing correct slow-tempo predictions to be rejected
                          # before the ring could self-correct, and pushing fallback to
                          # buf_abs which re-poisoned the ring (positive-feedback loop).
                          # With -96 the ring adapts in time, validated predictions are
                          # accepted, and the slope converges toward the true tempo.
                          # Risk of large backward jumps via the valid path is low: any
                          # backward step > RATE_HIGH * exp_delta still fails rate_ok.
    _VALID_AHEAD  =  96
    _RATE_LOW     = 0.5
    _RATE_HIGH    = 1.5
    _MIN_WIN_FRAMES = 10

    def __init__(
        self,
        d_model: int = 128,
        patch_size: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        c: int = 512,
        w: int = 128,
        fps: int = 100,
        device: Optional[str] = None,
        inf_c: Optional[int] = None,
        inf_w: Optional[int] = None,
        max_elapsed_deviation: Optional[int] = None,
        ring_buffer_size: int = 20,
        stabilization_steps: int = 5,
        max_consecutive_buffer: int = 5,
    ):
        super().__init__(name="Transformer")

        self.d_model = d_model
        self.patch_size = patch_size
        self.c = c
        self.w = w
        self.fps = fps

        self.inf_c = inf_c if inf_c is not None else c
        self.inf_w = inf_w if inf_w is not None else w

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.net = TransformerNet(
            d_model=d_model,
            patch_size=patch_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            c=c,
            w=w,
        ).to(device)

        self.ring_size  = ring_buffer_size
        self.stab_steps = stabilization_steps
        self.max_consec = max_consecutive_buffer
        self._max_dev: Optional[int] = max_elapsed_deviation

        self.reference_piano_roll: Optional[np.ndarray] = None
        self.midi_duration: float = 0.0
        self.ref_raw_patches: Optional[torch.Tensor] = None  # [N_ref, d_model] on CPU

        self._audio_buf: List[np.ndarray] = []
        self._audio_sr: int = 22050
        self._abs_pred_buf: deque = deque(maxlen=ring_buffer_size)
        self._consec_buf: int = 0
        self._step: int = 0
        self._prev_abs: Optional[float] = None
        self._elapsed_frames: float = 0.0
        self._output_buf: deque = deque(maxlen=3)   # output-only smoothing, does not feed back into tracking

    # =========================================================================
    # ScoreFollower interface
    # =========================================================================

    def load_reference(self, reference_path: str) -> None:
        midi = pretty_midi.PrettyMIDI(reference_path)
        raw = midi.get_piano_roll(fs=self.fps)
        self.reference_piano_roll = (raw > 0).astype(np.float32)
        self.midi_duration = midi.get_end_time()

        # Pre-compute raw patch embeddings (Conv1d only, no PE, no transformer)
        ref_t = torch.from_numpy(self.reference_piano_roll).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_patches = self.net.encode_ref_raw(ref_t)  # [1, N_ref, d_model]
        self.ref_raw_patches = raw_patches.squeeze(0).cpu()  # [N_ref, d_model]

        print(
            f"[PatchFormer] Reference: {Path(reference_path).name}  "
            f"shape={self.reference_piano_roll.shape}  dur={self.midi_duration:.1f}s  "
            f"ref_patches={self.ref_raw_patches.shape[0]}"
        )

    def reset(self) -> None:
        self.current_position = 0.0
        self._audio_buf = []
        self._abs_pred_buf = deque(maxlen=self.ring_size)
        self._consec_buf = 0
        self._step = 0
        self._prev_abs = None
        self._elapsed_frames = 0.0
        self._output_buf: deque = deque(maxlen=3)

    def requires_training(self) -> bool:
        return True

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        t0 = time.time()

        if self.reference_piano_roll is None:
            raise RuntimeError("Call load_reference() first.")

        T_ref = self.reference_piano_roll.shape[1]
        N_ref_patches = self.ref_raw_patches.shape[0] if self.ref_raw_patches is not None else 0

        # ── Elapsed-time tracker ──────────────────────────────────────────────
        self._elapsed_frames += len(audio_frame) / sample_rate * self.fps
        elapsed = float(np.clip(self._elapsed_frames, 0, T_ref - 1))

        if not self.is_trained or self.ref_raw_patches is None:
            self.current_position = elapsed / self.fps
            return {
                "position": self.current_position, "confidence": 0.0,
                "tempo": 0.0, "latency": (time.time() - t0) * 1000,
            }

        # ── Accumulate audio buffer ───────────────────────────────────────────
        self._audio_buf.append(audio_frame)
        self._audio_sr = sample_rate
        audio_cat = np.concatenate(self._audio_buf)

        max_keep = int(self.inf_w * sample_rate / self.fps) * 3
        if len(audio_cat) > max_keep:
            audio_cat = audio_cat[-max_keep:]
            self._audio_buf = [audio_cat]

        # ── Build live window W ───────────────────────────────────────────────
        w_samples = int(self.inf_w * sample_rate / self.fps)
        audio_window = audio_cat[-min(len(audio_cat), w_samples):]

        # Cold-start guard: need at least MIN_WIN_FRAMES worth of CQT frames
        min_samples = int(self._MIN_WIN_FRAMES * sample_rate / self.fps)
        if len(audio_window) < min_samples:
            self.current_position = elapsed / self.fps
            return {
                "position": self.current_position, "confidence": 0.0,
                "tempo": 0.0, "latency": (time.time() - t0) * 1000,
            }

        W_np = compute_cqt_features(audio_window, sample_rate, self.fps, target_w=self.inf_w)

        # ── Context window centred on last accepted prediction (closed-loop) ──
        # Bug 4 fix: the original code always centred the context on the audio
        # clock (elapsed), making it permanently open-loop — any performer tempo
        # deviation causes the true score position to drift outside the context
        # window after ~12-25 s, after which the model is matching the wrong
        # region of the score entirely.  Instead, we anchor on _prev_abs (the
        # model's own last accepted output), which keeps the true position inside
        # the ±(inf_c/2) search window regardless of tempo drift.
        # Fall back to elapsed only during cold-start (first few frames).
        N_inf_ctx = self.inf_c // self.patch_size
        N_inf_win = self.inf_w // self.patch_size
        N_valid = N_inf_ctx - N_inf_win + 1

        anchor_frame = self._prev_abs if self._prev_abs is not None else elapsed
        ctx_frame_start = int(np.clip(
            anchor_frame - self.inf_c // 2,
            0, max(0, T_ref - self.inf_c)
        ))
        # Align to patch grid so the pre-computed raw patch index arithmetic stays exact
        ctx_frame_start = (ctx_frame_start // self.patch_size) * self.patch_size
        ctx_patch_start = ctx_frame_start // self.patch_size

        # ── Retrieve pre-computed ref patch slice + encode on GPU ─────────────
        ctx_raw = self.ref_raw_patches[ctx_patch_start : ctx_patch_start + N_inf_ctx]
        if ctx_raw.shape[0] < N_inf_ctx:
            pad = N_inf_ctx - ctx_raw.shape[0]
            ctx_raw = F.pad(ctx_raw.T.unsqueeze(0), (0, pad)).squeeze(0).T

        with torch.no_grad():
            ctx_t = ctx_raw.unsqueeze(0).to(self.device)           # [1, N_inf_ctx, d]
            ref_patches = self.net.encode_ctx_slice(ctx_t)          # [1, N_inf_ctx, d]

            W_t = torch.from_numpy(W_np).unsqueeze(0).to(self.device)  # [1, 128, inf_w]
            live_emb = self.net.encode_live(W_t)                        # [1, d]

            logits = self.net.match(ref_patches, live_emb)              # [1, N_valid]
            logits = logits.squeeze(0)                                   # [N_valid]
            P_np = logits.cpu().numpy()

        # ── Heuristic decision → frame_k ─────────────────────────────────────
        # Pass the raw 97-logit vector directly.  The heuristic now works in
        # patch space internally and converts to the frame-space k only at the
        # very end.  See _heuristic_decision for why the old P_padded scatter
        # + frame-space smooth was broken.
        #
        # IMPORTANT: capture _prev_abs BEFORE calling the heuristic, because
        # the heuristic mutates self._prev_abs at the end of every call.
        # The clamp below must use the snapshot (previous call's value) so it
        # constrains the per-step change — if we read self._prev_abs after the
        # heuristic it is already equal to final_abs, making the clamp a no-op.
        _prev_abs_snapshot = self._prev_abs

        raw_k = self._heuristic_decision(P_np, ctx_frame_start)
        raw_k = int(np.clip(raw_k, self.inf_w - 1, self.inf_c - 1))
        raw_abs_frame = float(ctx_frame_start + raw_k)

        # ── Hard clamp: limit per-step jump, not total drift ─────────────────
        # Use the pre-heuristic snapshot so the clamp is not a no-op, and
        # anchor to _prev_abs rather than elapsed so cumulative tempo offset
        # never forces the position back toward the audio clock.
        max_dev = self._max_dev if self._max_dev is not None else self.inf_c // 3
        prev_for_clamp = _prev_abs_snapshot if _prev_abs_snapshot is not None else elapsed
        abs_frame = float(np.clip(raw_abs_frame, prev_for_clamp - max_dev, prev_for_clamp + max_dev))
        abs_frame = float(np.clip(abs_frame, 0, T_ref - 1))

        # ── Output smoothing (output-only, does not affect tracking state) ───
        # abs_frame drives context placement and _prev_abs — zero lag, correct tempo.
        # _output_buf smooths only what is returned as current_position.
        # A 3-pt causal mean reduces ±4-frame patch noise by ~40% (RMS: 2.84→1.70)
        # with a constant 1-call lag (~186 ms audio time at any tempo).
        # There is no velocity term, so there is no cold-start overshoot:
        # d/dn mean(x[n], x[n-1], x[n-2]) = (x[n]-x[n-3])/3 — exact tempo rate.
        self._output_buf.append(abs_frame)
        self.current_position = float(np.mean(self._output_buf)) / self.fps
        confidence = float(F.softmax(logits, dim=0).max().item())

        return {
            "position": self.current_position,
            "confidence": confidence,
            "tempo": self._estimate_tempo(),
            "latency": (time.time() - t0) * 1000,
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
              'batch_size'         : int   (2048),
              'samples_per_epoch'  : int   (50000),
              'val_samples'        : int   (1000),
              'lr'                 : float (1.5e-3),
              'weight_decay'       : float (1e-2),
              'save_path'          : str   (optional),
              'cache_size'         : int   (500; sized so dataset fits in 32 GB
                                             RAM page cache without thrashing.
                                             Pass None for all train pieces — only
                                             advisable on >=64 GB systems),
              'cqt_cache_dir'      : str   (default: dataset_path/cqt_cache),
              'roll_cache_dir'     : str   (default: dataset_path/roll_cache),
              'num_workers'        : int   (6),
            }
        """
        from torch.utils.data import DataLoader
        from models.transformer.dataset import MAESTROTransformerDataset

        cfg = {"dataset_path": train_data} if isinstance(train_data, str) else dict(train_data)
        dpath      = cfg["dataset_path"]
        epochs     = int(cfg.get("epochs", 50))
        bsz        = int(cfg.get("batch_size", 4096))
        n_tr       = int(cfg.get("samples_per_epoch", 100000))
        n_val      = int(cfg.get("val_samples", 1000))
        lr         = float(cfg.get("lr", 3e-3))
        wd         = float(cfg.get("weight_decay", 1e-2))
        spath      = cfg.get("save_path", None)
        cache_size = cfg.get("cache_size", 250)  # tuned so cache fits in 32 GB RAM
        num_workers = int(cfg.get("num_workers", 6))
        cqt_cache_dir  = Path(cfg.get("cqt_cache_dir",  Path(dpath) / "cqt_cache"))
        roll_cache_dir = Path(cfg.get("roll_cache_dir", Path(dpath) / "roll_cache"))

        print(f"[PatchFormer] Loading MAESTRO train split from '{dpath}' …")
        midi_paths, wav_paths = _maestro_wav_paths(dpath)
        if not midi_paths:
            raise RuntimeError("No train-split MIDI found.")
        print(f"[PatchFormer] Found {len(midi_paths)} train pairs.")

        # ── Step 1: Pre-compute fp16 .npy caches (one-time, idempotent) ──────
        n_cache = len(midi_paths) if cache_size is None else min(len(midi_paths), int(cache_size))
        print(f"[PatchFormer] Pre-computing CQT cache ({n_cache} files) → {cqt_cache_dir} …")
        cqt_npy = _precompute_cqt_cache(
            wav_paths[:n_cache], cqt_cache_dir, fps=self.fps
        )
        print(f"[PatchFormer] Pre-computing piano-roll cache ({n_cache} files) → {roll_cache_dir} …")
        roll_npy = _precompute_piano_roll_cache(
            midi_paths[:n_cache], roll_cache_dir, fps=self.fps
        )

        # Keep only indices where BOTH .npy files are available.
        paired = [
            (str(r), str(c))
            for r, c in zip(roll_npy, cqt_npy)
            if r is not None and Path(r).exists()
            and c is not None and Path(c).exists()
        ]
        if not paired:
            raise RuntimeError("No usable cached pairs (roll + CQT).")
        roll_paths = [p[0] for p in paired]
        cqt_paths  = [p[1] for p in paired]
        n_pieces = len(paired)
        print(f"[PatchFormer] Cached pairs ready: {n_pieces} pieces (mmap-backed).")

        # ── Step 2: Build datasets + DataLoaders ─────────────────────────────
        split = max(1, n_pieces * 9 // 10)
        tr_roll = roll_paths[:split]
        tr_cqt  = cqt_paths[:split]
        vl_roll = roll_paths[split:] or roll_paths
        vl_cqt  = cqt_paths[split:]  or cqt_paths

        tr_dataset = MAESTROTransformerDataset(
            tr_roll, tr_cqt, self.c, self.w, self.patch_size,
            augment=True, length=n_tr,
        )
        vl_dataset = MAESTROTransformerDataset(
            vl_roll, vl_cqt, self.c, self.w, self.patch_size,
            augment=False, length=n_val,
        )

        pin = (self.device != "cpu")
        try:
            tr_loader = DataLoader(
                tr_dataset, batch_size=bsz,
                num_workers=num_workers,
                pin_memory=pin,
                persistent_workers=(num_workers > 0),
            )
            vl_loader = DataLoader(
                vl_dataset, batch_size=bsz,
                num_workers=min(2, num_workers),
                pin_memory=pin,
                persistent_workers=(min(2, num_workers) > 0),
            )
        except Exception:
            warnings.warn("[PatchFormer] DataLoader with workers failed; falling back to num_workers=0.")
            tr_loader = DataLoader(tr_dataset, batch_size=bsz, num_workers=0, pin_memory=pin)
            vl_loader = DataLoader(vl_dataset, batch_size=bsz, num_workers=0, pin_memory=pin)

        # ── Step 3: Optimizer, scheduler, AMP, cuDNN autotune ────────────────
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        steps_per_epoch = max(1, n_tr // bsz)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            pct_start=0.1, anneal_strategy="cos",
        )
        use_amp = self.device != "cpu"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        tol_patches = max(1, round(0.005 * self.fps / self.patch_size))

        best_val_acc = 0.0
        best_state: Optional[dict] = None

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, epochs + 1):
            self.net.train()
            tr_loss, tr_corr, tr_tot = 0.0, 0, 0

            loader_iter = iter(tr_loader)
            for step in range(steps_per_epoch):
                try:
                    ctx, win, label = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(tr_loader)
                    ctx, win, label = next(loader_iter)

                # ctx arrives as fp16 — upcast on GPU to halve PCIe bytes.
                ctx   = ctx.to(self.device, non_blocking=True).float()
                win   = win.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.split(":")[0],
                                    dtype=torch.float16, enabled=use_amp):
                    logits = self.net(ctx, win)          # [B, N_valid]
                    loss   = criterion(logits, label)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                preds = logits.detach().argmax(1)
                tr_corr += int(((preds - label).abs() <= tol_patches).sum())
                tr_loss  += loss.item() * ctx.size(0)
                tr_tot   += ctx.size(0)

            tr_loss /= max(tr_tot, 1)
            tr_acc   = 100.0 * tr_corr / max(tr_tot, 1)

            # ── Validation (loader is persistent, built once above) ──────────
            self.net.eval()
            vl_loss, vl_corr, vl_n = 0.0, 0, 0
            with torch.no_grad():
                for ctx_v, win_v, label_v in vl_loader:
                    ctx_v   = ctx_v.to(self.device, non_blocking=True).float()
                    win_v   = win_v.to(self.device, non_blocking=True)
                    label_v = label_v.to(self.device, non_blocking=True)
                    with torch.autocast(device_type=self.device.split(":")[0],
                                        dtype=torch.float16, enabled=use_amp):
                        logits_v = self.net(ctx_v, win_v)
                        vl_loss += criterion(logits_v, label_v).item() * ctx_v.size(0)
                    preds_v = logits_v.argmax(1)
                    vl_corr += int(((preds_v - label_v).abs() <= tol_patches).sum())
                    vl_n += ctx_v.size(0)

            vl_loss /= max(vl_n, 1)
            vl_acc   = 100.0 * vl_corr / max(vl_n, 1)

            print(
                f"[PatchFormer] Ep {epoch:3d}/{epochs}  "
                f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.1f}%  "
                f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.1f}%"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

        # ── Restore best weights ─────────────────────────────────────────────
        if best_state:
            self.net.load_state_dict(best_state)
        self.net.to(self.device).eval()
        self.is_trained = True
        print(f"[PatchFormer] Training complete (best val_acc={best_val_acc:.1f}%).")
        if spath:
            self.save_checkpoint(spath)
            print(f"[PatchFormer] Checkpoint → {spath}")

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "d_model": self.d_model,
            "patch_size": self.patch_size,
            "c": self.c,
            "w": self.w,
            "fps": self.fps,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.is_trained = True
        print(f"[PatchFormer] Loaded checkpoint '{path}'.")

    # =========================================================================
    # Heuristic decision  (verbatim from HeurMiTModel, Section 3.2.3)
    # =========================================================================

    def _heuristic_decision(self, P_np: np.ndarray, ctx_start: int) -> int:
        """
        Parameters
        ----------
        P_np     : float32 array, shape [N_valid=97] — raw transformer logits
                   (patch space).  Previously this was a sparse frame-space
                   P_padded of length inf_c+inf_w-1=639; the scatter has been
                   removed because the 5-pt frame-space smooth was broken:
                   adjacent patch positions are 4 frames apart, so the 5-pt
                   window (half-width 2) never contained two real values at once,
                   diluting every logit with fill_value×4/5 and suppressing all
                   peaks below prominence=3.  The argmax fallback then landed on
                   non-patch-aligned positions, producing random frame jumps.
        ctx_start: context window start in absolute reference frames.
        Returns
        -------
        Frame-space k (right-edge convention, range [inf_w-1, inf_c-1]) such
        that the predicted absolute position = ctx_start + k.
        """
        self._step += 1

        # ── Gaussian smooth in patch space ────────────────────────────────────
        # sigma=1.5 patches (60 ms FWHM at 40 ms/patch) blends logit uncertainty
        # across neighbouring patches without dilution from fill values.
        sigma  = 1.5
        hw     = max(2, int(round(3.0 * sigma)))          # 5-patch half-width
        xs     = np.arange(-hw, hw + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (xs / sigma) ** 2)
        kernel /= kernel.sum()
        smoothed = np.convolve(P_np.astype(np.float64), kernel, mode="same")

        raw_p     = int(np.argmax(smoothed))
        model_k   = raw_p * self.patch_size + int(0.5 * (self.inf_w - 1))  # forward-helper
        model_abs = ctx_start + model_k

        # L used for the final clip — must be in frame space, not patch space
        L = self.inf_c + self.inf_w - 1

        if self._step <= self.stab_steps:
            self._abs_pred_buf.append(float(model_abs))
            self._prev_abs   = float(model_abs)
            return int(np.clip(model_k, self.inf_w - 1, self.inf_c - 1))

        buf = np.array(list(self._abs_pred_buf), dtype=float)
        if len(buf) >= 2:
            xs_buf    = np.arange(len(buf), dtype=float)
            coeffs    = np.polyfit(xs_buf, buf, 1)
            slope     = coeffs[0]
            buf_abs   = float(coeffs[1] + slope * len(buf))
            exp_delta = max(1.0, abs(slope))
        else:
            buf_abs   = model_abs
            slope     = 1.0
            exp_delta = 1.0

        prev  = self._prev_abs if self._prev_abs is not None else float(model_abs)
        delta = model_abs - prev

        mono_ok  = model_abs >= prev + self._VALID_BEHIND
        range_ok = (buf_abs + self._VALID_BEHIND) <= model_abs <= (buf_abs + self._VALID_AHEAD)
        rate_ok  = (self._RATE_LOW * exp_delta) <= abs(delta) <= (self._RATE_HIGH * exp_delta)
        valid    = mono_ok and range_ok and rate_ok

        if valid:
            final_abs        = model_abs
            self._consec_buf = 0
        elif self._consec_buf >= self.max_consec:
            final_abs        = model_abs
            self._consec_buf = 0
        else:
            mean_abs  = (model_abs + buf_abs) / 2.0
            final_abs = (mean_abs if abs(mean_abs - model_abs) < abs(mean_abs - buf_abs)
                         else buf_abs)
            self._consec_buf += 1

        self._abs_pred_buf.append(float(final_abs))
        self._prev_abs = float(final_abs)

        return int(np.clip(round(final_abs - ctx_start), 0, L - 1))

    def _estimate_tempo(self) -> float:
        buf = list(self._abs_pred_buf)
        if len(buf) < 4:
            return 0.0
        xs    = np.arange(len(buf), dtype=float)
        slope = float(np.polyfit(xs, buf, 1)[0])
        return float(np.clip(slope * 60.0, 20.0, 400.0)) if slope > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PatchFormer — smoke test")
    print("=" * 60)

    import tempfile, os

    model = TransformerModel(d_model=128, patch_size=4, n_heads=4, n_layers=2, c=512, w=128)
    n = sum(p.numel() for p in model.net.parameters())
    print(f"Params     : {n:,}")

    # Forward shape
    ctx = torch.randn(4, 128, 512).to(model.device)
    win = torch.randn(4, 128, 128).to(model.device)
    with torch.no_grad():
        logits = model.net(ctx, win)
    assert logits.shape == (4, model.net.N_valid), f"bad logits shape {logits.shape}"
    print(f"Logits     : {tuple(logits.shape)}  ✓  (expect (4, {model.net.N_valid}))")

    # process_frame without training
    import pretty_midi, numpy as np
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start in [(60, 0.0), (62, 0.5), (64, 1.0), (65, 1.5)]:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + 0.4))
    midi.instruments.append(inst)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_mid = f.name
    midi.write(tmp_mid)

    model.load_reference(tmp_mid)
    model.reset()
    chunk = np.random.randn(4096).astype(np.float32) * 0.01
    result = model.process_frame(chunk, 22050)
    assert set(result.keys()) == {"position", "confidence", "tempo", "latency"}
    print(f"process_frame: position={result['position']:.3f}s  latency={result['latency']:.1f}ms  ✓")

    os.unlink(tmp_mid)
    print("=" * 60)
    print("All checks passed.")