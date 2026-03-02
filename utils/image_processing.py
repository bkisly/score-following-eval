"""
midi_to_matrices.py
-------------------
Full pipeline: MIDI file → multi-page sheet music images → list of 416x416 numpy matrices.

Requirements:
    pip install numpy pillow music21

External tools (one of):
    - MuseScore 4:  https://musescore.org/en/download  (recommended)
    - MuseScore 3:  also works, adjust MUSESCORE_PATH accordingly

Windows Notes:
    - Make sure MuseScore is installed and MUSESCORE_PATH below points to its .exe
    - Default paths for common MuseScore versions are pre-filled; uncomment the right one.
"""

import subprocess
import glob
import os
import sys
import shutil
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
) -> list[np.ndarray]:
    """
    End-to-end pipeline: MIDI → sheet music PNGs → numpy matrices.

    Parameters
    ----------
    midi_path   : Path to the .mid / .midi file.
    output_dir  : Folder to store intermediate PNG images.
    target_size : Output matrix size (width, height). Default (416, 416).
    grayscale   : Produce single-channel matrices. Default False (RGB).
    keep_images : If False, delete the intermediate PNG folder when done.

    Returns
    -------
    List of float32 numpy arrays, one per sheet music page.
    Shape: (416, 416) if grayscale, else (416, 416, 3).
    """
    print(f"\n{'='*55}")
    print(f"  MIDI → Matrices Pipeline")
    print(f"  Input : {midi_path}")
    print(f"  Output: {len(TARGET_SIZE)}x{TARGET_SIZE[0]} {'grayscale' if grayscale else 'RGB'}")
    print(f"{'='*55}\n")

    image_paths = midi_to_sheet_images(midi_path, output_dir)
    matrices = images_to_numpy(image_paths, target_size, grayscale)

    if not keep_images:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"[3/3] Cleaned up temporary folder '{output_dir}'.")
    else:
        print(f"[3/3] Intermediate images kept in '{output_dir}'.")

    print(f"\n✓ Pipeline complete: {len(matrices)} page(s) converted.\n")
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