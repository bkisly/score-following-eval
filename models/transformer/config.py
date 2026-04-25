from dataclasses import dataclass


@dataclass
class TransformerConfig:
    sample_rate: int = 22050
    chunk_size: int = 4096

    # CQT
    cqt_hop_length: int = 512
    cqt_n_bins: int = 84
    cqt_bins_per_octave: int = 12
    cqt_fmin_midi: int = 24

    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Tracking
    window_seconds: float = 6.0
    low_confidence_threshold: float = 0.20
    max_backstep: int = 2
    max_forward_step: int = 64

    # Live features
    live_context_seconds: float = 4.0
    min_confidence_floor: float = 1e-6

