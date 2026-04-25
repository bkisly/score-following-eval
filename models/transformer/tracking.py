from models.transformer.config import TransformerConfig


def update_position(
    prev_idx: int,
    candidate_idx: int,
    confidence: float,
    config: TransformerConfig,
    max_idx: int,
) -> int:
    if confidence < config.low_confidence_threshold:
        lower = max(0, prev_idx - config.max_backstep)
    else:
        lower = prev_idx
    upper = min(max_idx, prev_idx + config.max_forward_step)
    return int(max(lower, min(candidate_idx, upper)))


def estimate_tempo(
    prev_idx: int,
    new_idx: int,
    chunk_size: int,
    sample_rate: int,
    ref_hop_length: int,
    ref_sample_rate: int,
) -> float:
    delta_ref_sec = (new_idx - prev_idx) * (ref_hop_length / float(ref_sample_rate))
    delta_live_sec = chunk_size / float(sample_rate)
    if delta_live_sec <= 1e-8:
        return 0.0
    # Pseudo-BPM proxy derived from local speed ratio.
    ratio = max(0.0, delta_ref_sec / delta_live_sec)
    return float(max(0.0, min(400.0, ratio * 60.0)))

