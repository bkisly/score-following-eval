"""
Feature extraction and data utilities for PatchFormer.

Re-exports from cnn_model:
  apply_augmentations, _load_piano_roll_cache, _maestro_train_paths

Adds:
  compute_cqt_features   — shared CQT pipeline (pre-computation + inference)
  _maestro_wav_paths     — extract paired WAV paths from MAESTRO metadata
  _precompute_cqt_cache  — one-time WAV→CQT→.npy conversion
  _load_cqt_cache        — load pre-computed .npy arrays as float16
"""

import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.cnn_model import apply_augmentations, _load_piano_roll_cache, _maestro_train_paths  # noqa: E402

__all__ = [
    "apply_augmentations",
    "_load_piano_roll_cache",
    "_maestro_train_paths",
    "compute_cqt_features",
    "_maestro_wav_paths",
    "_precompute_cqt_cache",
    "_load_cqt_cache",
]


# ─────────────────────────────────────────────────────────────────────────────
# CQT feature pipeline  (same normalisation as HeurMiT._audio_to_piano_roll)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cqt_features(
    audio: np.ndarray,
    sample_rate: int,
    fps: int = 100,
    target_w: Optional[int] = None,
) -> np.ndarray:
    """
    Convert audio to [128, target_w] CQT feature matrix.

    Identical normalisation to HeurMiT._audio_to_piano_roll:
      - 88-bin CQT (A0–C8), hop derived from fps
      - 95th-percentile normalisation, clipped to [0, 1]
      - Silence guard: returns zeros if max < 0.05
      - Embedded into full 128-bin MIDI array (piano range = bins 21–108)
      - Resampled to target_w frames (nearest-neighbour)

    Returns float32 array [128, target_w].
    If target_w is None the native CQT frame count is returned.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 64 or np.abs(audio).max() < 1e-7:
        if target_w is not None:
            return np.zeros((128, target_w), dtype=np.float32)
        return np.zeros((128, 1), dtype=np.float32)

    hop = max(1, int(round(sample_rate / fps)))
    fmin = librosa.midi_to_hz(21)  # A0

    try:
        C = np.abs(
            librosa.cqt(audio, sr=sample_rate, hop_length=hop, n_bins=88,
                        bins_per_octave=12, fmin=fmin)
        ).astype(np.float32)  # [88, T_cqt]
    except Exception:
        if target_w is not None:
            return np.zeros((128, target_w), dtype=np.float32)
        return np.zeros((128, 1), dtype=np.float32)

    T_cqt = C.shape[1]
    if T_cqt == 0:
        if target_w is not None:
            return np.zeros((128, target_w), dtype=np.float32)
        return np.zeros((128, 1), dtype=np.float32)

    p95 = float(np.percentile(C, 95))
    if p95 > 1e-8:
        C = C / p95
    C = np.clip(C, 0.0, 1.0)

    if C.max() < 0.05:
        if target_w is not None:
            return np.zeros((128, target_w), dtype=np.float32)
        return np.zeros((128, T_cqt), dtype=np.float32)

    # Embed into full 128-bin MIDI array
    full = np.zeros((128, T_cqt), dtype=np.float32)
    full[21:109, :] = C

    if target_w is None:
        return full

    if T_cqt == target_w:
        return full

    if T_cqt < target_w:
        out = np.zeros((128, target_w), dtype=np.float32)
        out[:, -T_cqt:] = full  # place existing frames at the END (matches HeurMiT)
        return out

    # Down-sample by nearest-neighbour
    idx = np.round(np.linspace(0, T_cqt - 1, target_w)).astype(int)
    return full[:, np.clip(idx, 0, T_cqt - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# MAESTRO WAV path reader
# ─────────────────────────────────────────────────────────────────────────────

def _maestro_wav_paths(dataset_path: str) -> Tuple[List[str], List[str]]:
    """
    Returns (midi_paths, wav_paths) for the train split from MAESTRO metadata.
    Both lists are aligned: midi_paths[i] ↔ wav_paths[i].
    """
    dp = Path(dataset_path)
    for meta_path, fmt in [
        (dp / "maestro-v3.0.0.json", "json"),
        (dp / "maestro-v3.0.0.csv", "csv"),
    ]:
        if not meta_path.exists():
            continue
        if fmt == "json":
            with open(meta_path) as f:
                meta = json.load(f)
            midi_paths = [
                str(dp / meta["midi_filename"][k])
                for k, s in meta["split"].items()
                if s == "train"
            ]
            wav_paths = [
                str(dp / meta["audio_filename"][k])
                for k, s in meta["split"].items()
                if s == "train"
            ]
        else:
            with open(meta_path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
            midi_paths = [str(dp / r["midi_filename"]) for r in rows]
            wav_paths = [str(dp / r["audio_filename"]) for r in rows]
        return midi_paths, wav_paths
    raise FileNotFoundError(f"No MAESTRO metadata in '{dataset_path}'.")


# ─────────────────────────────────────────────────────────────────────────────
# One-time CQT pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_cqt_cache(
    wav_paths: List[str],
    cache_dir: Path,
    fps: int = 100,
    sample_rate: int = 22050,
) -> List[Path]:
    """
    For each WAV file, compute CQT features and save as float16 .npy.
    Skips files whose .npy already exists. Returns list of .npy paths.

    Storage: ~7.5 MB / file (float16).  250 files ≈ 1.9 GB.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_paths: List[Path] = []

    for wav_path in wav_paths:
        npy_path = cache_dir / (Path(wav_path).stem + "_cqt.npy")
        npy_paths.append(npy_path)
        if npy_path.exists():
            continue
        try:
            audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
            cqt = compute_cqt_features(audio, sr, fps=fps)  # [128, T] float32
            np.save(npy_path, cqt.astype(np.float16))
            print(
                f"  [CQT cache] {Path(wav_path).stem}: "
                f"shape={cqt.shape}  → {npy_path.name}"
            )
        except Exception as e:
            warnings.warn(f"[CQT cache] skip '{wav_path}': {e}")
            npy_paths[-1] = None  # mark as unavailable

    return npy_paths


# ─────────────────────────────────────────────────────────────────────────────
# CQT cache loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_cqt_cache(npy_paths: List[Optional[Path]]) -> List[np.ndarray]:
    """
    Load pre-computed CQT .npy files as float16 arrays.
    Returns a list aligned with npy_paths; missing files are skipped.
    """
    cache = []
    for npy_path in npy_paths:
        if npy_path is None or not Path(npy_path).exists():
            continue
        try:
            arr = np.load(str(npy_path))  # float16
            if arr.ndim == 2 and arr.shape[0] == 128 and arr.shape[1] > 0:
                cache.append(arr)
        except Exception as e:
            warnings.warn(f"[CQT cache] could not load '{npy_path}': {e}")
    return cache
