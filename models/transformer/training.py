import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from models.transformer.alignment import compute_alignment_logits, select_window
from models.transformer.config import TransformerConfig
from models.transformer.features import chunk_to_cqt
from models.transformer.reference import build_reference_from_midi


def load_maestro_entries(dataset_path: str, split: str = "train") -> List[Tuple[str, str]]:
    root = Path(dataset_path)
    json_path = root / "maestro-v3.0.0.json"
    csv_path = root / "maestro-v3.0.0.csv"

    entries: List[Tuple[str, str]] = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key, s in meta["split"].items():
            if s != split:
                continue
            midi_rel = meta["midi_filename"][key]
            audio_rel = meta["audio_filename"][key]
            entries.append((str(root / midi_rel), str(root / audio_rel)))
        return entries

    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["split"] != split:
                continue
            entries.append((str(root / row["midi_filename"]), str(root / row["audio_filename"])))
        return entries

    raise FileNotFoundError(f"No MAESTRO metadata found in '{dataset_path}'")


def _cache_file_name(midi_path: str) -> str:
    key = hashlib.sha1(midi_path.encode("utf-8")).hexdigest()
    stem = Path(midi_path).stem
    return f"{stem}_{key[:12]}.npz"


def prepare_reference_cqt_cache(
    train_entries: List[Tuple[str, str]],
    config: TransformerConfig,
    cache_dir: str,
    model,
    overwrite: bool = False,
) -> Dict[str, str]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    index: Dict[str, str] = {}
    for midi_path, _ in train_entries:
        out_path = cache_path / _cache_file_name(midi_path)
        index[midi_path] = str(out_path)

        if out_path.exists() and not overwrite:
            continue

        try:
            ref = build_reference_from_midi(
                midi_path,
                config=config,
                device=model.device,
                input_proj=model.input_proj,
            )
            np.savez_compressed(
                out_path,
                ref_cqt=ref.ref_cqt.astype(np.float32),
                ref_times_sec=ref.ref_times_sec.astype(np.float32),
                hop_length=np.array([ref.hop_length], dtype=np.int32),
                sample_rate=np.array([ref.sample_rate], dtype=np.int32),
            )
        except Exception:
            # Skip broken references; they will be ignored in epoch sampling.
            continue
    return index


def _load_cached_reference_npz(path: str, model) -> Tuple[torch.Tensor, np.ndarray, int, int]:
    data = np.load(path)
    ref_cqt = data["ref_cqt"].astype(np.float32)
    ref_times_sec = data["ref_times_sec"].astype(np.float32)
    hop_length = int(data["hop_length"][0])
    sample_rate = int(data["sample_rate"][0])

    with torch.no_grad():
        ref_t = torch.from_numpy(ref_cqt).to(model.device)
        ref_emb = model.input_proj(ref_t)
    return ref_emb, ref_times_sec, hop_length, sample_rate


def train_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    train_entries: List[Tuple[str, str]],
    ref_cache_index: Dict[str, str],
    config: TransformerConfig,
    samples_per_epoch: int,
    window_seconds: float,
) -> Dict[str, float]:
    model.encoder.train()
    model.input_proj.train()

    total_loss = 0.0
    n_steps = 0

    for _ in range(samples_per_epoch):
        midi_path, audio_path = train_entries[np.random.randint(len(train_entries))]
        cache_file = ref_cache_index.get(midi_path)
        if cache_file is None or not Path(cache_file).exists():
            continue

        try:
            ref_emb, ref_times_sec, ref_hop, ref_sr = _load_cached_reference_npz(cache_file, model)
            audio, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
        except Exception:
            continue

        if len(audio) < config.chunk_size or len(ref_times_sec) < 4:
            continue

        max_start = len(audio) - config.chunk_size
        start = np.random.randint(0, max_start + 1)
        chunk = audio[start : start + config.chunk_size]
        chunk_time_sec = start / float(sr)
        gt_idx = int(np.searchsorted(ref_times_sec, chunk_time_sec, side="left"))
        gt_idx = int(np.clip(gt_idx, 0, len(ref_times_sec) - 1))

        live_cqt = chunk_to_cqt(chunk, config, sr)
        live_t = torch.from_numpy(live_cqt).unsqueeze(0).to(model.device)
        _, live_query = model.encoder(live_t)

        prev_jitter = int(
            np.random.randint(
                -int(window_seconds * sr / config.cqt_hop_length),
                int(window_seconds * sr / config.cqt_hop_length) + 1,
            )
        )
        prev_idx = int(np.clip(gt_idx + prev_jitter, 0, len(ref_times_sec) - 1))
        win_emb, win_idx, _, _ = select_window(
            ref_emb,
            prev_idx=prev_idx,
            config=config,
            ref_hop_length=ref_hop,
            ref_sample_rate=ref_sr,
        )
        if len(win_idx) < 2:
            continue

        logits = compute_alignment_logits(live_query, win_emb)
        local_target = int(np.where(win_idx == gt_idx)[0][0]) if gt_idx in win_idx else None
        if local_target is None:
            local_target = int(np.argmin(np.abs(win_idx - gt_idx)))

        probs = F.softmax(logits, dim=-1)
        pred_local = int(torch.argmax(probs, dim=-1).item())
        pred_global = int(win_idx[pred_local])

        loss_pos = F.cross_entropy(
            logits, torch.tensor([local_target], dtype=torch.long, device=model.device)
        )
        mono_penalty = max(0.0, float(prev_idx - pred_global))
        loss_mono = torch.tensor(mono_penalty, dtype=torch.float32, device=model.device)

        target_correct = 1.0 if abs(ref_times_sec[pred_global] - chunk_time_sec) <= 0.5 else 0.0
        conf_pred = torch.max(probs)
        loss_conf = F.binary_cross_entropy(
            conf_pred, torch.tensor(target_correct, dtype=torch.float32, device=model.device)
        )

        loss = loss_pos + 0.05 * loss_mono + 0.2 * loss_conf
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters_for_training(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_steps += 1

    return {"loss": (total_loss / n_steps) if n_steps > 0 else float("inf"), "steps": n_steps}

