from dataclasses import dataclass

import librosa
import numpy as np
import pretty_midi
import torch

from models.transformer.config import TransformerConfig
from models.transformer.features import chunk_to_cqt
from models.transformer.state import ReferenceBundle


def _synthesize_midi_audio(midi: pretty_midi.PrettyMIDI, sample_rate: int) -> np.ndarray:
    try:
        audio = midi.fluidsynth(fs=sample_rate)
        if audio is not None and len(audio) > 0:
            return np.asarray(audio, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(0, dtype=np.float32)


def _symbolic_fallback_cqt(midi: pretty_midi.PrettyMIDI, config: TransformerConfig) -> np.ndarray:
    fps = config.sample_rate / float(config.cqt_hop_length)
    roll = midi.get_piano_roll(fs=fps).astype(np.float32)  # [128, T]
    if roll.shape[1] == 0:
        return np.zeros((1, config.cqt_n_bins), dtype=np.float32)

    start = config.cqt_fmin_midi
    end = min(128, start + config.cqt_n_bins)
    out = np.zeros((config.cqt_n_bins, roll.shape[1]), dtype=np.float32)
    out[: end - start] = roll[start:end]
    out = np.log1p(out)
    p95 = float(np.percentile(out, 95))
    if p95 > 1e-8:
        out = out / p95
    out = np.clip(out, 0.0, 1.0)
    return out.T


def build_reference_from_midi(
    reference_path: str,
    config: TransformerConfig,
    device: torch.device,
    input_proj: torch.nn.Module,
) -> ReferenceBundle:
    midi = pretty_midi.PrettyMIDI(reference_path)
    synth_audio = _synthesize_midi_audio(midi, config.sample_rate)
    if len(synth_audio) > 0:
        ref_cqt = chunk_to_cqt(synth_audio, config, config.sample_rate)
    else:
        ref_cqt = _symbolic_fallback_cqt(midi, config)

    if ref_cqt.shape[0] == 0:
        ref_cqt = np.zeros((1, config.cqt_n_bins), dtype=np.float32)

    with torch.no_grad():
        ref_tensor = torch.from_numpy(ref_cqt).to(device)
        ref_emb = input_proj(ref_tensor).detach().cpu()

    t_ref = ref_cqt.shape[0]
    ref_times_sec = (
        np.arange(t_ref, dtype=np.float32) * config.cqt_hop_length / float(config.sample_rate)
    )
    return ReferenceBundle(
        ref_cqt=ref_cqt,
        ref_emb=ref_emb,
        ref_times_sec=ref_times_sec,
        hop_length=config.cqt_hop_length,
        sample_rate=config.sample_rate,
    )

