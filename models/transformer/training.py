import csv
import hashlib
import json
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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


def _cache_config_dict(config: TransformerConfig) -> Dict[str, int]:
    return {
        "sample_rate": int(config.sample_rate),
        "cqt_hop_length": int(config.cqt_hop_length),
        "cqt_n_bins": int(config.cqt_n_bins),
        "cqt_bins_per_octave": int(config.cqt_bins_per_octave),
        "cqt_fmin_midi": int(config.cqt_fmin_midi),
    }


def _read_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        return {"version": 1, "config": {}, "entries": {}}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "config": {}, "entries": {}}


def _write_manifest(manifest_path: Path, manifest: Dict) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def prepare_reference_cqt_cache(
    train_entries: List[Tuple[str, str]],
    config: TransformerConfig,
    cache_dir: str,
    model,
    cache_mode: str = "reuse",
    show_progress: bool = True,
) -> Dict[str, object]:
    """
    cache_mode:
        - reuse: use existing cache and build missing files
        - rebuild: recompute all entries
        - readonly: use existing cache only, never build missing files
    """
    if cache_mode not in {"reuse", "rebuild", "readonly"}:
        raise ValueError("cache_mode must be one of: reuse, rebuild, readonly")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / "index.json"
    manifest = _read_manifest(manifest_path)
    current_cfg = _cache_config_dict(config)
    cfg_mismatch = manifest.get("config", {}) not in ({}, current_cfg)
    if cache_mode == "rebuild" or cfg_mismatch:
        manifest = {"version": 1, "config": current_cfg, "entries": {}}
    else:
        manifest["config"] = current_cfg

    index: Dict[str, str] = {}
    reused = 0
    rebuilt = 0
    readonly_skipped = 0
    failed = 0

    iterator = tqdm(
        train_entries,
        total=len(train_entries),
        desc="Building reference CQT cache",
        disable=not show_progress,
    )
    for midi_path, _ in iterator:
        out_path = cache_path / _cache_file_name(midi_path)
        midi_key = str(Path(midi_path).resolve())
        index[midi_key] = str(out_path)

        file_exists = out_path.exists()
        if file_exists and cache_mode != "rebuild":
            reused += 1
            manifest["entries"][midi_key] = str(out_path.name)
            if show_progress:
                iterator.set_postfix(
                    reused=reused, rebuilt=rebuilt, skipped=readonly_skipped, failed=failed
                )
            continue
        if cache_mode == "readonly":
            readonly_skipped += 1
            if show_progress:
                iterator.set_postfix(
                    reused=reused, rebuilt=rebuilt, skipped=readonly_skipped, failed=failed
                )
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
            rebuilt += 1
            manifest["entries"][midi_key] = str(out_path.name)
        except Exception:
            failed += 1
            index.pop(midi_key, None)
            continue
        finally:
            if show_progress:
                iterator.set_postfix(
                    reused=reused, rebuilt=rebuilt, skipped=readonly_skipped, failed=failed
                )

    _write_manifest(manifest_path, manifest)
    usable = {
        k: v for k, v in index.items() if Path(v).exists()
    }
    stats = {
        "total_entries": len(train_entries),
        "usable_entries": len(usable),
        "reused": reused,
        "rebuilt": rebuilt,
        "readonly_skipped": readonly_skipped,
        "failed": failed,
        "cache_dir": str(cache_path),
        "manifest_path": str(manifest_path),
        "config_mismatch": cfg_mismatch,
        "cache_mode": cache_mode,
    }
    return {"index": usable, "stats": stats}


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


def _get_cached_audio(
    audio_path: str,
    target_sr: int,
    audio_cache: "OrderedDict[str, np.ndarray]",
    cache_max_files: int,
) -> Optional[Tuple[np.ndarray, int]]:
    if audio_path in audio_cache:
        audio = audio_cache.pop(audio_path)
        audio_cache[audio_path] = audio
        return audio, target_sr
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception:
        return None
    audio_cache[audio_path] = audio
    while len(audio_cache) > cache_max_files:
        audio_cache.popitem(last=False)
    return audio, sr


def train_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    train_entries: List[Tuple[str, str]],
    ref_cache_index: Dict[str, str],
    config: TransformerConfig,
    samples_per_epoch: int,
    window_seconds: float,
    batch_size: int = 16,
    use_amp: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    audio_cache_max_files: int = 16,
    epoch_idx: int = 1,
    total_epochs: int = 1,
) -> Dict[str, float]:
    model.encoder.train()
    model.input_proj.train()

    total_loss = 0.0
    n_steps = 0
    audio_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
    amp_enabled = bool(use_amp and model.device.type == "cuda")
    progress = min(1.0, max(0.0, epoch_idx / float(max(1, total_epochs))))
    jitter_scale = 0.25 + 0.75 * progress

    sample_buffer = []
    max_attempts = max(samples_per_epoch * 8, samples_per_epoch + 1)
    attempts = 0
    while len(sample_buffer) < samples_per_epoch and attempts < max_attempts:
        attempts += 1
        midi_path, audio_path = train_entries[np.random.randint(len(train_entries))]
        cache_file = ref_cache_index.get(str(Path(midi_path).resolve()))
        if cache_file is None or not Path(cache_file).exists():
            continue
        try:
            ref_emb, ref_times_sec, ref_hop, ref_sr = _load_cached_reference_npz(cache_file, model)
        except Exception:
            continue
        loaded = _get_cached_audio(
            audio_path=audio_path,
            target_sr=config.sample_rate,
            audio_cache=audio_cache,
            cache_max_files=audio_cache_max_files,
        )
        if loaded is None:
            continue
        audio, sr = loaded
        if len(audio) < config.chunk_size or len(ref_times_sec) < 4:
            continue
        max_start = len(audio) - config.chunk_size
        start = np.random.randint(0, max_start + 1)
        chunk = audio[start : start + config.chunk_size]
        live_cqt = chunk_to_cqt(chunk, config, sr)
        chunk_time_sec = start / float(sr)
        gt_idx = int(np.searchsorted(ref_times_sec, chunk_time_sec, side="left"))
        gt_idx = int(np.clip(gt_idx, 0, len(ref_times_sec) - 1))

        base_jitter = int(window_seconds * sr / config.cqt_hop_length)
        jitter_radius = max(1, int(base_jitter * jitter_scale))
        prev_jitter = int(np.random.randint(-jitter_radius, jitter_radius + 1))
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
        local_target = int(np.where(win_idx == gt_idx)[0][0]) if gt_idx in win_idx else None
        if local_target is None:
            local_target = int(np.argmin(np.abs(win_idx - gt_idx)))
        sample_buffer.append(
            {
                "live_cqt": live_cqt,
                "win_emb": win_emb,
                "win_idx": win_idx,
                "local_target": local_target,
                "prev_idx": prev_idx,
                "chunk_time_sec": chunk_time_sec,
                "ref_times_sec": ref_times_sec,
            }
        )

    for start in range(0, len(sample_buffer), max(1, batch_size)):
        batch = sample_buffer[start : start + max(1, batch_size)]
        if not batch:
            continue
        live_batch = np.stack([item["live_cqt"] for item in batch], axis=0).astype(np.float32)
        live_t = torch.from_numpy(live_batch).to(model.device)

        optimizer.zero_grad()
        autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else nullcontext()
        with autocast_ctx:
            _, live_queries = model.encoder(live_t)
            sample_losses = []
            for i, item in enumerate(batch):
                live_query = live_queries[i : i + 1]
                logits = compute_alignment_logits(live_query, item["win_emb"])
                probs = F.softmax(logits, dim=-1)
                pred_local = int(torch.argmax(probs, dim=-1).item())
                pred_global = int(item["win_idx"][pred_local])
                loss_pos = F.cross_entropy(
                    logits,
                    torch.tensor([item["local_target"]], dtype=torch.long, device=model.device),
                )
                mono_penalty = max(0.0, float(item["prev_idx"] - pred_global))
                loss_mono = torch.tensor(mono_penalty, dtype=torch.float32, device=model.device)
                target_correct = (
                    1.0 if abs(item["ref_times_sec"][pred_global] - item["chunk_time_sec"]) <= 0.5 else 0.0
                )
                conf_pred = torch.max(probs)
                loss_conf = F.binary_cross_entropy(
                    conf_pred,
                    torch.tensor(target_correct, dtype=torch.float32, device=model.device),
                )
                sample_losses.append(loss_pos + 0.05 * loss_mono + 0.2 * loss_conf)
            loss = torch.stack(sample_losses).mean()

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters_for_training(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters_for_training(), 1.0)
            optimizer.step()

        total_loss += float(loss.item()) * len(batch)
        n_steps += len(batch)

    return {"loss": (total_loss / n_steps) if n_steps > 0 else float("inf"), "steps": n_steps}

