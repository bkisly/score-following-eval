import json
import csv
from pathlib import Path
from typing import List

from evaluation.data import Piece


def get_maestro_test_pairs(
    dataset_path: str,
    max_pieces: int | None = 25,
) -> List[Piece]:
    """
    Returns (midi_file, audio_file) path pairs for the test split of the MAESTRO dataset.

    Args:
        dataset_path: Path to the root directory of the downloaded MAESTRO dataset.
        max_pieces:   If set, returns only the first N pairs, ordered deterministically
                      by (midi_filename, audio_filename). If None, all test pairs are returned.

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
            )
            for key, split in meta["split"].items()
            if split == "test"
        ]

    elif csv_meta.exists():
        with open(csv_meta, newline="") as f:
            reader = csv.DictReader(f)
            pairs = [
                (row["midi_filename"], row["audio_filename"])
                for row in reader
                if row["split"] == "test"
            ]

    else:
        raise FileNotFoundError(
            f"No MAESTRO metadata file found in '{dataset_path}'. "
            "Expected 'maestro-v3.0.0.json' or 'maestro-v3.0.0.csv'."
        )

    # Sort before slicing to guarantee a stable, deterministic order
    pairs.sort(key=lambda p: (p[0], p[1]))

    if max_pieces is not None:
        pairs = pairs[:max_pieces]

    # Resolve to absolute paths only after filtering/slicing
    return [
        Piece(midi_path=str(dataset_path / midi), audio_path=str(dataset_path / audio))
        for midi, audio in pairs
    ]