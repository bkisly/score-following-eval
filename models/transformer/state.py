from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class ReferenceBundle:
    ref_cqt: np.ndarray
    ref_emb: torch.Tensor
    ref_times_sec: np.ndarray
    hop_length: int
    sample_rate: int


@dataclass
class TransformerRuntimeState:
    reference: Optional[ReferenceBundle] = None
    current_ref_idx: int = 0
    prev_ref_idx: int = 0
    elapsed_seconds: float = 0.0
    stall_chunks: int = 0
    audio_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    initialized: bool = False

