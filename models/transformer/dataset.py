"""
MAESTRO training dataset for PatchFormer.

Each sample is a (context C, window W, label_patch) triplet:
  C: [128, c]   MIDI piano roll slice  (E_ref domain)  — fp16
  W: [128, w]   CQT feature slice      (E_live domain) — fp32 (post-augment)
  label_patch: int — window start position in C, in patch units
               label_patch = (ws - ctx_s) // patch_size  ∈ [0, N_ctx - N_win]

Caches are memory-mapped on-disk .npy files (fp16). With persistent_workers=True
and num_workers>0, each worker holds only its own per-piece mmap handles; the
underlying file pages are shared across workers via the OS page cache. This
keeps RAM use flat regardless of num_workers and dataset size.
"""

import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from models.transformer.features import apply_augmentations


class MAESTROTransformerDataset(Dataset):
    """
    Indexed dataset that yields random (C, W, label_patch) triplets.

    Parameters
    ----------
    roll_paths : list of paths to fp16 piano-roll .npy files
    cqt_paths  : list of paths to fp16 CQT .npy files (aligned with roll_paths)
    c          : context length in frames
    w          : window length in frames
    patch_size : frames per patch (determines label granularity)
    augment    : whether to apply augmentations to W
    length     : virtual epoch length (number of __getitem__ calls per epoch)
    """

    def __init__(
        self,
        roll_paths: List[str],
        cqt_paths: List[str],
        c: int = 512,
        w: int = 128,
        patch_size: int = 4,
        augment: bool = True,
        length: int = 10000,
    ):
        assert len(roll_paths) == len(cqt_paths), (
            f"roll_paths ({len(roll_paths)}) and cqt_paths ({len(cqt_paths)}) "
            "must be aligned (same number of pieces)."
        )
        self.roll_paths = [str(p) for p in roll_paths]
        self.cqt_paths = [str(p) for p in cqt_paths]
        self.c = c
        self.w = w
        self.patch_size = patch_size
        self.augment = augment
        self.length = length
        self.N_ctx = c // patch_size
        self.N_win = w // patch_size
        self._max_label = self.N_ctx - self.N_win  # inclusive upper bound

        # Lazy per-worker mmap handles. Populated on first access in
        # __getitem__ so each DataLoader worker opens its own descriptors;
        # OS page cache shares the underlying bytes across workers.
        self._roll_mmap: Dict[int, np.ndarray] = {}
        self._cqt_mmap: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.length

    def _roll(self, i: int) -> np.ndarray:
        m = self._roll_mmap.get(i)
        if m is None:
            m = np.load(self.roll_paths[i], mmap_mode="r")
            self._roll_mmap[i] = m
        return m

    def _cqt(self, i: int) -> np.ndarray:
        m = self._cqt_mmap.get(i)
        if m is None:
            m = np.load(self.cqt_paths[i], mmap_mode="r")
            self._cqt_mmap[i] = m
        return m

    def __getitem__(self, idx: int):
        while True:
            i = random.randrange(len(self.roll_paths))
            roll = self._roll(i)                          # [128, T] fp16, mmap
            cqt  = self._cqt(i)                           # [128, T] fp16, mmap

            T = min(roll.shape[1], cqt.shape[1])
            if T < self.c + self.w:
                continue

            # Window must start INSIDE the context so the label is never clipped.
            # Previous bounds allowed ws < ctx_s (window partially before context),
            # which caused two compounding bugs:
            #   1. Training on misaligned pairs: W contains audio from BEFORE the
            #      context region, yet the label says "window aligns with start of C".
            #   2. 33x label pile-up at label 0 and label 96 (all out-of-bounds ws
            #      values collapse to the nearest extreme label after clipping).
            # Fix: ws ∈ [ctx_s,  ctx_s + c - w]  →  label ∈ [0, N_ctx - N_win]
            # uniformly, with each label covered by exactly patch_size ws values.
            if T < self.c + self.w:
                continue
            ctx_s = random.randint(0, T - self.c)
            min_ws = ctx_s
            max_ws = ctx_s + self.c - self.w   # = ctx_s + (max_label * patch_size)
            ws = random.randint(min_ws, max_ws)
            if ws + self.w > T:
                continue

            # Slice first, materialize a small fp16 chunk (page-cache hit
            # after warmup). The full piece is never loaded into RAM.
            C_fp16 = np.array(roll[:, ctx_s : ctx_s + self.c])     # [128, c] fp16
            W_fp16 = np.array(cqt[:,  ws    : ws    + self.w])     # [128, w] fp16

            # Augmentations are written for fp32 numpy arrays; cast the
            # tiny window slice (not the full piece, as before).
            W = W_fp16.astype(np.float32, copy=False)
            if self.augment:
                W = apply_augmentations(W)

            label_patch = int(
                np.clip((ws - ctx_s) // self.patch_size, 0, self._max_label)
            )

            return (
                torch.from_numpy(C_fp16),                            # fp16 [128, c]
                torch.from_numpy(np.ascontiguousarray(W)),           # fp32 [128, w]
                torch.tensor(label_patch, dtype=torch.long),
            )