import json
import csv
import os
from pathlib import Path

from evaluation.data import Piece, PieceMetadata


def get_maestro_test_pairs(
    dataset_path: str,
    max_pieces: int | None = None,
    max_duration: float | None = None,
) -> list[Piece]:
    """
    Returns (midi_file, audio_file) path pairs for the test split of the MAESTRO dataset.

    Args:
        dataset_path:  Path to the root directory of the downloaded MAESTRO dataset.
        max_pieces:    If set, returns only the first N pairs, ordered deterministically
                       by (midi_filename, audio_filename). If None, all test pairs are returned.
        max_duration:  If set, only pairs whose duration (in seconds) is <= this value
                       are included. Filtering is applied before the max_pieces slice
                       so that the piece count is always respected.

    Returns:
        List of (midi_path, audio_path) tuples with absolute paths.
    """
    dataset_path = Path(dataset_path)

    json_meta = dataset_path / "maestro-v3.0.0.json"
    csv_meta  = dataset_path / "maestro-v3.0.0.csv"

    if json_meta.exists():
        with open(json_meta, "r") as f:
            meta = json.load(f)

        pairs = [
            (
                meta["midi_filename"][key],
                meta["audio_filename"][key],
                float(meta["duration"][key]),
            )
            for key, split in meta["split"].items()
            if split == "test"
        ]

    elif csv_meta.exists():
        with open(csv_meta, newline="") as f:
            reader = csv.DictReader(f)
            pairs = [
                (
                    row["midi_filename"],
                    row["audio_filename"],
                    float(row["duration"]),
                )
                for row in reader
                if row["split"] == "test"
            ]

    else:
        raise FileNotFoundError(
            f"No MAESTRO metadata file found in '{dataset_path}'. "
            "Expected 'maestro-v3.0.0.json' or 'maestro-v3.0.0.csv'."
        )

    if max_duration is not None:
        pairs = [p for p in pairs if p[2] <= max_duration]

    # Sort before slicing to guarantee a stable, deterministic order
    pairs.sort(key=lambda p: (p[0], p[1]))

    if max_pieces is not None:
        pairs = pairs[:max_pieces]

    # Resolve to absolute paths only after filtering/slicing
    return [
        Piece(midi_path=str(dataset_path / midi), audio_path=str(dataset_path / audio),
              metadata=get_piece_metadata(str(dataset_path), audio, midi))
        for midi, audio, _ in pairs
    ]

def get_piece_metadata(dataset_path: str, wav_path: str, midi_path: str) -> PieceMetadata:
    """
    Retrieve metadata for a MAESTRO piece given its wav and midi paths.

    Args:
        dataset_path: Root path to the MAESTRO dataset directory.
        wav_path:     Relative path to the .wav file (as stored in the metadata).
        midi_path:    Relative path to the .midi/.mid file (as stored in the metadata).

    Returns:
        PieceMetadata with 'title' and 'composer' fields populated.

    Raises:
        FileNotFoundError: If no metadata file is found in dataset_path.
        ValueError:        If no record matches both wav_path and midi_path.
    """
    # MAESTRO ships metadata as both JSON and CSV — prefer JSON
    json_meta = os.path.join(dataset_path, "maestro-v3.0.0.json")
    csv_meta  = os.path.join(dataset_path, "maestro-v3.0.0.csv")

    if os.path.exists(json_meta):
        return _lookup_json(json_meta, wav_path, midi_path)
    elif os.path.exists(csv_meta):
        return _lookup_csv(csv_meta, wav_path, midi_path)
    else:
        raise FileNotFoundError(
            f"No MAESTRO metadata file found in '{dataset_path}'. "
            "Expected 'maestro-v3.0.0.json' or 'maestro-v3.0.0.csv'."
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalise(path: str) -> str:
    """Strip leading slashes / './' so paths compare cleanly."""
    return path.lstrip("./").lstrip("/")


def _match(row_wav: str, row_midi: str, wav_path: str, midi_path: str) -> bool:
    return (
        _normalise(row_wav)  == _normalise(wav_path) and
        _normalise(row_midi) == _normalise(midi_path)
    )


def _lookup_json(meta_path: str, wav_path: str, midi_path: str) -> PieceMetadata:
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The JSON structure uses parallel arrays keyed by field name
    audio_filenames = data["audio_filename"]
    midi_filenames  = data["midi_filename"]
    titles          = data["canonical_title"]
    composers       = data["canonical_composer"]

    for idx in range(len(audio_filenames)):
        idx_str = str(idx)
        if _match(audio_filenames[idx_str], midi_filenames[idx_str], wav_path, midi_path):
            return PieceMetadata(
                title=titles[idx_str],
                composer=composers[idx_str],
            )

    raise ValueError(
        f"No record found for wav='{wav_path}', midi='{midi_path}' in {meta_path}"
    )


def _lookup_csv(meta_path: str, wav_path: str, midi_path: str) -> PieceMetadata:
    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _match(row["audio_filename"], row["midi_filename"], wav_path, midi_path):
                return PieceMetadata(
                    title=row["canonical_title"],
                    composer=row["canonical_composer"],
                )

    raise ValueError(
        f"No record found for wav='{wav_path}', midi='{midi_path}' in {meta_path}"
    )