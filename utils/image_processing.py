"""
midi_to_matrices.py
-------------------
Full pipeline: MIDI file → multi-page sheet music images → list of 416x416 numpy matrices,
with optional per-page start timestamps for real-time page tracking.

Pipeline overview
-----------------
Stage 1a  MIDI → PNG pages          (MuseScore subprocess)
Stage 1b  MIDI → MusicXML           (MuseScore subprocess, only when timestamps requested)
Stage 2   PNG pages → numpy arrays  (PIL resize + normalise)
Stage 3   MusicXML + MIDI → timestamps
          • Parse MusicXML for <print new-page="yes"> → measure numbers at page breaks
          • pretty_midi.get_downbeats() → precise measure-start times in seconds
          • Map page-break measure numbers → seconds

Requirements:
    pip install numpy pillow pretty_midi

External tools (one of):
    - MuseScore 4:  https://musescore.org/en/download  (recommended)
    - MuseScore 3:  also works, adjust MUSESCORE_PATH accordingly

Windows Notes:
    - Make sure MuseScore is installed and MUSESCORE_PATH below points to its .exe
    - Default paths for common MuseScore versions are pre-filled; uncomment the right one.
"""

import re
import subprocess
import glob
import os
import sys
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# CONFIG — adjust these to match your setup
# ---------------------------------------------------------------------------

# Common Windows install paths — uncomment the one that matches your version:
MUSESCORE_PATH = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
# MUSESCORE_PATH = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"
# MUSESCORE_PATH = r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe"

TARGET_SIZE = (416, 416)   # (width, height) for output matrices
GRAYSCALE = False           # True  → shape (416, 416),   dtype float32, range [0,1]
                            # False → shape (416, 416, 3), dtype float32, range [0,1]


# ---------------------------------------------------------------------------
# STAGE 1: MIDI → PNG images (one per page)
# ---------------------------------------------------------------------------

def find_musescore() -> str:
    """
    Return the MuseScore executable path.
    Tries MUSESCORE_PATH first, then PATH, then common install locations.
    Raises FileNotFoundError if nothing is found.
    """
    # 1. Explicit config at top of file
    if os.path.isfile(MUSESCORE_PATH):
        return MUSESCORE_PATH

    # 2. On PATH (e.g. if user added it manually)
    for name in ("MuseScore4", "MuseScore3", "mscore"):
        found = shutil.which(name)
        if found:
            return found

    # 3. Other common Windows locations
    candidates = [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "MuseScore executable not found.\n"
        "Please install MuseScore from https://musescore.org/en/download\n"
        "or set MUSESCORE_PATH at the top of this script."
    )


def midi_to_sheet_images(midi_path: str, output_dir: str) -> list[str]:
    """
    Convert a MIDI file to a list of PNG sheet music image paths (one per page).

    Parameters
    ----------
    midi_path   : Path to the input .mid / .midi file.
    output_dir  : Directory where page images will be saved.

    Returns
    -------
    Sorted list of PNG file paths, e.g. ['out/sheet-1.png', 'out/sheet-2.png', ...]
    """
    midi_path = str(Path(midi_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    mscore = find_musescore()
    output_prefix = os.path.join(output_dir, "sheet")

    cmd = [
        mscore,
        midi_path,
        "-o", f"{output_prefix}.png"
    ]

    print(f"[1/3] Running MuseScore: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(
            f"MuseScore failed with return code {result.returncode}.\n"
            f"Stderr: {result.stderr.strip()}"
        )

    # MuseScore names pages: sheet-1.png, sheet-2.png ...
    # (MuseScore 4 may use sheet-1.png or sheet-page-1.png depending on version)
    image_paths = sorted(
        glob.glob(os.path.join(output_dir, "sheet*.png")),
        key=lambda p: [int(c) if c.isdigit() else c for c in Path(p).stem.split("-")]
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG files were generated in '{output_dir}'.\n"
            "Check that the MIDI file is valid and MuseScore can open it."
        )

    print(f"[1/3] Generated {len(image_paths)} page image(s).")
    return image_paths


def midi_to_musicxml(midi_path: str, output_dir: str) -> str:
    """
    Export a MIDI file to uncompressed MusicXML using MuseScore.

    MuseScore performs a full score layout during export, so the resulting
    MusicXML contains <print new-page="yes"> markers at the exact measures
    that begin each new page — matching the PNG page layout produced by
    midi_to_sheet_images() when called on the same MIDI file.

    Parameters
    ----------
    midi_path  : Path to the input .mid / .midi file.
    output_dir : Directory where the .musicxml file will be saved.

    Returns
    -------
    Absolute path to the generated .musicxml file.
    """
    midi_path  = str(Path(midi_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    mscore   = find_musescore()
    mxl_path = os.path.join(output_dir, "sheet.musicxml")

    cmd = [mscore, midi_path, "-o", mxl_path]
    print(f"[1b] Exporting MusicXML: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(
            f"MuseScore MusicXML export failed (code {result.returncode}).\n"
            f"Stderr: {result.stderr.strip()}"
        )

    if not os.path.isfile(mxl_path):
        raise FileNotFoundError(
            f"MuseScore did not produce '{mxl_path}'.\n"
            "Check that the MIDI file is valid and MuseScore can open it."
        )

    print(f"[1b] MusicXML written to: {mxl_path}")
    return mxl_path


# ---------------------------------------------------------------------------
# STAGE 1c (internal): MusicXML parsing helpers
# ---------------------------------------------------------------------------

def _parse_page_break_measures(mxl_path: str) -> list[int]:
    """
    Parse a MusicXML file and return the 1-based measure numbers at which
    each new page starts.

    MuseScore inserts ``<print new-page="yes">`` inside the first <measure>
    of every page after the first.  This function reads those markers and
    prepends measure 1 for page 1, so the returned list always has one entry
    per page.

    Parameters
    ----------
    mxl_path : Path to an uncompressed .musicxml file.

    Returns
    -------
    Sorted list of 1-based measure numbers, e.g. [1, 14, 27] for 3 pages.
    """
    tree = ET.parse(mxl_path)
    root = tree.getroot()

    # Detect optional XML namespace (e.g. xmlns="http://www.musicxml.org/ns/2.0")
    ns_match = re.match(r'\{[^}]+\}', root.tag)
    ns = ns_match.group(0) if ns_match else ""

    page_break_measures: list[int] = [1]  # page 1 always starts at measure 1

    for measure in root.iter(f"{ns}measure"):
        for print_el in measure.findall(f"{ns}print"):
            if print_el.get("new-page") == "yes":
                raw = measure.get("number")
                try:
                    page_break_measures.append(int(raw))
                except (TypeError, ValueError):
                    pass  # malformed measure number — skip

    return sorted(set(page_break_measures))


def _build_measure_time_map(midi_path: str) -> dict[int, float]:
    """
    Build a mapping from 1-based measure number to start time in seconds.

    Uses ``pretty_midi.PrettyMIDI.get_downbeats()`` which accounts for all
    tempo changes and time signature changes in the MIDI file.  The first
    downbeat corresponds to measure 1.

    Parameters
    ----------
    midi_path : Path to the .mid / .midi file.

    Returns
    -------
    Dict ``{measure_number: start_time_seconds}``, e.g. {1: 0.0, 2: 2.1, …}.
    """
    try:
        import pretty_midi  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is required for timestamp extraction.\n"
            "Install it with:  pip install pretty_midi"
        ) from exc

    pm        = pretty_midi.PrettyMIDI(midi_path)
    downbeats = pm.get_downbeats()  # shape (n_measures,), seconds

    return {i + 1: float(t) for i, t in enumerate(downbeats)}


# ---------------------------------------------------------------------------
# Public helper: MusicXML + MIDI → per-page timestamps
# ---------------------------------------------------------------------------

def extract_page_timestamps(
    mxl_path: str,
    midi_path: str,
    n_pages: int,
) -> list[float]:
    """
    Compute the start time in seconds of each sheet music page.

    Combines MusicXML page-break markers with pretty_midi's measure timing.

    Parameters
    ----------
    mxl_path  : Path to the MusicXML file exported by MuseScore for this MIDI.
    midi_path : Path to the source .mid / .midi file.
    n_pages   : Expected number of pages (must equal the number of PNG pages
                produced by midi_to_sheet_images for the same MIDI).

    Returns
    -------
    List of ``n_pages`` floats.  ``timestamps[0]`` is always 0.0 (page 1
    starts at the beginning).  ``timestamps[i]`` is the time in seconds at
    which the score turns to page ``i+1``.

    Notes
    -----
    If the number of page breaks found in the MusicXML does not match
    ``n_pages``, a warning is printed and the list is trimmed or padded with
    the final known timestamp so that ``len(result) == n_pages`` always holds.
    """
    page_break_measures = _parse_page_break_measures(mxl_path)
    measure_time_map    = _build_measure_time_map(midi_path)
    max_measure         = max(measure_time_map.keys())

    timestamps: list[float] = []
    for measure_num in page_break_measures:
        # Clamp to the last known measure if MusicXML has more measures
        # than pretty_midi counted (e.g. trailing empty measures).
        clamped = min(measure_num, max_measure)
        timestamps.append(measure_time_map.get(clamped, measure_time_map[max_measure]))

    # Reconcile with the actual number of PNG pages
    if len(timestamps) != n_pages:
        print(
            f"[WARNING] extract_page_timestamps: found {len(timestamps)} page-break "
            f"marker(s) in MusicXML but {n_pages} PNG page(s) were generated.\n"
            f"          The two MuseScore export calls may have produced different "
            f"layouts.  Timestamps will be trimmed / padded to match {n_pages} page(s)."
        )
        if len(timestamps) > n_pages:
            timestamps = timestamps[:n_pages]
        else:
            last = timestamps[-1] if timestamps else 0.0
            timestamps += [last] * (n_pages - len(timestamps))

    # Guarantee page 1 starts at 0.0 (MuseScore sometimes puts measure 1
    # at a small non-zero offset when there is a pickup bar).
    timestamps[0] = 0.0

    print(f"[3/3] Page timestamps (seconds): {[round(t, 3) for t in timestamps]}")
    return timestamps


# ---------------------------------------------------------------------------
# STAGE 2: PNG images → numpy matrices
# ---------------------------------------------------------------------------

def images_to_numpy(
    image_paths: list[str],
    target_size: tuple[int, int] = TARGET_SIZE,
    grayscale: bool = GRAYSCALE,
) -> list[np.ndarray]:
    """
    Load and resize PNG images into normalised numpy arrays.

    Parameters
    ----------
    image_paths : List of paths returned by midi_to_sheet_images().
    target_size : (width, height) to resize each page to.
    grayscale   : If True, output shape is (H, W); otherwise (H, W, 3).

    Returns
    -------
    List of float32 numpy arrays with pixel values in [0.0, 1.0].
    """
    print(f"[2/3] Converting {len(image_paths)} image(s) to numpy "
          f"({'grayscale' if grayscale else 'RGB'}, {target_size})...")

    matrices = []
    for i, path in enumerate(image_paths):
        img = Image.open(path)

        # Sheet music exported by MuseScore may be RGBA (transparent background).
        # Composite onto white before converting so transparency → white pixels.
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # use alpha channel as mask
            img = background
        else:
            img = img.convert("RGB")

        if grayscale:
            img = img.convert("L")

        img = img.resize(target_size, resample=Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        matrices.append(arr)
        print(f"    Page {i + 1}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")

    print(f"[2/3] Done. {len(matrices)} matrix/matrices ready.")
    return matrices


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def midi_to_matrices(
    midi_path: str,
    output_dir: str = "sheet_tmp",
    target_size: tuple[int, int] = TARGET_SIZE,
    grayscale: bool = GRAYSCALE,
    keep_images: bool = True,
    return_timestamps: bool = False,
) -> "list[np.ndarray] | tuple[list[np.ndarray], list[float]]":
    """
    End-to-end pipeline: MIDI → sheet music PNGs → numpy matrices.

    Parameters
    ----------
    midi_path         : Path to the .mid / .midi file.
    output_dir        : Folder to store intermediate PNG images (and MusicXML
                        when ``return_timestamps=True``).
    target_size       : Output matrix size (width, height). Default (416, 416).
    grayscale         : Produce single-channel matrices. Default False (RGB).
    keep_images       : If False, delete the intermediate folder when done.
    return_timestamps : If True, also compute and return per-page start times.
                        Requires an extra MuseScore call to export MusicXML and
                        ``pretty_midi`` to be installed.

    Returns
    -------
    ``return_timestamps=False`` (default)
        List of float32 numpy arrays, one per sheet music page.
        Shape: (416, 416) if grayscale, else (416, 416, 3).

    ``return_timestamps=True``
        ``(matrices, timestamps)`` where ``matrices`` is as above and
        ``timestamps`` is a ``list[float]`` of length ``n_pages``:
        ``timestamps[i]`` is the time in seconds at which page ``i+1`` begins.
        ``timestamps[0]`` is always ``0.0``.
    """
    print(f"\n{'='*55}")
    print(f"  MIDI → Matrices Pipeline")
    print(f"  Input : {midi_path}")
    print(f"  Output: {target_size[0]}x{target_size[1]} {'grayscale' if grayscale else 'RGB'}")
    if return_timestamps:
        print(f"  Mode  : matrices + page timestamps")
    print(f"{'='*55}\n")

    image_paths = midi_to_sheet_images(midi_path, output_dir)
    matrices    = images_to_numpy(image_paths, target_size, grayscale)

    timestamps: list[float] | None = None
    if return_timestamps:
        mxl_path   = midi_to_musicxml(midi_path, output_dir)
        timestamps = extract_page_timestamps(mxl_path, midi_path, n_pages=len(matrices))

    if not keep_images:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"[3/3] Cleaned up temporary folder '{output_dir}'.")
    else:
        print(f"[3/3] Intermediate files kept in '{output_dir}'.")

    print(f"\n✓ Pipeline complete: {len(matrices)} page(s) converted.\n")

    if return_timestamps:
        return matrices, timestamps
    return matrices


# ---------------------------------------------------------------------------
# OPTIONAL: save matrices to .npy files for later use
# ---------------------------------------------------------------------------

def save_matrices(matrices: list[np.ndarray], save_dir: str) -> list[str]:
    """Save each matrix as a .npy file. Returns the list of saved paths."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, mat in enumerate(matrices):
        path = os.path.join(save_dir, f"page_{i + 1:04d}.npy")
        np.save(path, mat)
        paths.append(path)
        print(f"  Saved {path}  (shape={mat.shape})")
    return paths


def load_matrices(save_dir: str) -> list[np.ndarray]:
    """Load previously saved .npy matrix files from a directory."""
    paths = sorted(glob.glob(os.path.join(save_dir, "page_*.npy")))
    return [np.load(p) for p in paths]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Basic usage — edit the path below and run:                         #
    #      python midi_to_matrices.py                                     #
    #  Or pass the MIDI path as a command-line argument:                  #
    #      python midi_to_matrices.py my_song.mid                         #
    # ------------------------------------------------------------------ #

    midi_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\bkisl\Desktop\maestro-v3.0.0\maestro-v3.0.0\2018\MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi"

    if not os.path.isfile(midi_file):
        print(f"ERROR: File not found: '{midi_file}'")
        print("Usage: python midi_to_matrices.py <path_to_midi>")
        sys.exit(1)

    # Run the pipeline
    matrices = midi_to_matrices(
        midi_path=midi_file,
        output_dir="sheet_tmp",
        target_size=(416, 416),
        grayscale=False,    # set True for single-channel output
        keep_images=True,   # set False to auto-delete PNGs after conversion
    )

    # Print summary
    print("Summary:")
    for i, m in enumerate(matrices):
        print(f"  matrices[{i}]  shape={m.shape}  dtype={m.dtype}  "
              f"min={m.min():.3f}  max={m.max():.3f}")

    # Optional: stack into a single 4-D batch tensor (N, H, W, C) or (N, H, W)
    batch = np.stack(matrices, axis=0)
    print(f"\nBatch tensor shape: {batch.shape}")

    # Optional: save to disk for reuse
    # save_matrices(matrices, save_dir="matrices_out")