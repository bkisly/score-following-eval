from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.transformer.config import TransformerConfig


def select_window(
    ref_emb: torch.Tensor,
    prev_idx: int,
    config: TransformerConfig,
    ref_hop_length: int,
    ref_sample_rate: int,
) -> Tuple[torch.Tensor, np.ndarray, int, int]:
    radius_frames = max(
        1, int(round(config.window_seconds * ref_sample_rate / float(ref_hop_length)))
    )
    t_ref = ref_emb.shape[0]
    start = max(0, prev_idx - radius_frames)
    end = min(t_ref, prev_idx + radius_frames + 1)
    window = ref_emb[start:end]
    indices = np.arange(start, end, dtype=np.int64)
    return window, indices, start, end


def compute_alignment_logits(live_query: torch.Tensor, win_emb: torch.Tensor) -> torch.Tensor:
    # live_query: [1, D], win_emb: [W, D] -> [1, W]
    scores = torch.matmul(win_emb, live_query.squeeze(0)) / np.sqrt(win_emb.shape[-1])
    return scores.unsqueeze(0)


def confidence_from_probs(probs: torch.Tensor, floor: float = 1e-6) -> float:
    max_prob = float(torch.max(probs).item())
    return max(floor, min(1.0, max_prob))


def alignment_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)

