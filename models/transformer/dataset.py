"""
MAESTRO training dataset for PatchFormer.

Each sample is a (context C, window W, label_patch) triplet:
  C: [128, c]   MIDI piano roll slice  (E_ref domain)
  W: [128, w]   CQT feature slice      (E_live domain)
  label_patch: int — window start position in C, in patch units
               label_patch = (ws - ctx_s) // patch_size  ∈ [0, N_ctx - N_win]
"""

import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from models.transformer.features import apply_augmentations


class MAESTROTransformerDataset(Dataset):
    """
    Indexed dataset that yields random (C, W, label_patch) triplets.

    Parameters
    ----------
    roll_cache : list of [128, T] float32 piano roll arrays (one per piece)
    cqt_cache  : list of [128, T] float16 CQT arrays (aligned with roll_cache)
    c          : context length in frames
    w          : window length in frames
    patch_size : frames per patch (determines label granularity)
    augment    : whether to apply MIDIOgre augmentations to W
    length     : virtual epoch length (number of __getitem__ calls per epoch)
    """

    def __init__(
        self,
        roll_cache: List[np.ndarray],
        cqt_cache: List[np.ndarray],
        c: int = 512,
        w: int = 128,
        patch_size: int = 4,
        augment: bool = True,
        length: int = 10000,
    ):
        assert len(roll_cache) == len(cqt_cache), (
            f"roll_cache ({len(roll_cache)}) and cqt_cache ({len(cqt_cache)}) "
            "must be aligned (same number of pieces)."
        )
        self.rolls = roll_cache
        self.cqts = cqt_cache
        self.c = c
        self.w = w
        self.patch_size = patch_size
        self.augment = augment
        self.length = length
        self.N_ctx = c // patch_size
        self.N_win = w // patch_size
        self._max_label = self.N_ctx - self.N_win  # inclusive upper bound

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        while True:
            i = random.randrange(len(self.rolls))
            roll = self.rolls[i]                          # [128, T] float32
            cqt = self.cqts[i].astype(np.float32)        # float16 → float32

            T = min(roll.shape[1], cqt.shape[1])
            if T < self.c + self.w:
                continue

            ctx_s = random.randint(0, T - self.c)
            min_ws = ctx_s                              # window must start within context
            max_ws = min(T - self.w, ctx_s + self.c - self.w)  # window must end within context
            if max_ws < min_ws:
                continue
            ws = random.randint(min_ws, max_ws)
            if ws + self.w > T:
                continue

            C = roll[:, ctx_s : ctx_s + self.c].copy()   # piano roll context
            W = cqt[: , ws     : ws  + self.w].copy()    # CQT live window

            if self.augment:
                W = apply_augmentations(W)

            label_patch = int(
                np.clip((ws - ctx_s) // self.patch_size, 0, self._max_label)
            )

            return (
                torch.from_numpy(C),                                # [128, c]
                torch.from_numpy(W),                                # [128, w]
                torch.tensor(label_patch, dtype=torch.long),        # scalar
            )
