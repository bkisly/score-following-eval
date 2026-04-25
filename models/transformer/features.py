from typing import Tuple

import librosa
import numpy as np

from models.transformer.config import TransformerConfig
from models.transformer.state import TransformerRuntimeState


def _compute_cqt(audio: np.ndarray, config: TransformerConfig, sample_rate: int) -> np.ndarray:
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-8:
        return np.zeros((1, config.cqt_n_bins), dtype=np.float32)

    try:
        cqt = np.abs(
            librosa.cqt(
                audio,
                sr=sample_rate,
                hop_length=config.cqt_hop_length,
                n_bins=config.cqt_n_bins,
                bins_per_octave=config.cqt_bins_per_octave,
                fmin=librosa.midi_to_hz(config.cqt_fmin_midi),
            )
        ).astype(np.float32)
    except Exception:
        return np.zeros((1, config.cqt_n_bins), dtype=np.float32)

    cqt = np.log1p(cqt)
    p95 = float(np.percentile(cqt, 95))
    if p95 > 1e-8:
        cqt = cqt / p95
    cqt = np.clip(cqt, 0.0, 1.0)
    return cqt.T


def extract_live_cqt(
    audio_chunk: np.ndarray,
    state: TransformerRuntimeState,
    config: TransformerConfig,
    sample_rate: int,
) -> Tuple[np.ndarray, TransformerRuntimeState]:
    chunk = np.asarray(audio_chunk, dtype=np.float32)
    if chunk.ndim > 1:
        chunk = np.mean(chunk, axis=-1)

    if len(state.audio_buffer) == 0:
        merged = chunk
    else:
        merged = np.concatenate([state.audio_buffer, chunk], axis=0)

    max_keep = int(config.live_context_seconds * sample_rate) + len(chunk)
    if len(merged) > max_keep:
        merged = merged[-max_keep:]

    cqt_all = _compute_cqt(merged, config, sample_rate)
    target_frames = max(1, int(np.ceil(len(chunk) / config.cqt_hop_length)))
    live_cqt = cqt_all[-target_frames:]

    state.audio_buffer = merged
    return live_cqt, state


def chunk_to_cqt(audio_chunk: np.ndarray, config: TransformerConfig, sample_rate: int) -> np.ndarray:
    return _compute_cqt(np.asarray(audio_chunk, dtype=np.float32), config, sample_rate)

