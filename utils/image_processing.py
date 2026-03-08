"""
image_processing.py
-------------------
Full pipeline: MIDI file → multi-page sheet music images → list of 416×416
numpy matrices, with optional per-page start timestamps for real-time page
tracking.

Pipeline overview
-----------------
Stage 1a  MIDI → LilyPond source (.ly)   (midi2ly, ships with LilyPond)
Stage 1b  LilyPond source → PNG pages    (lilypond subprocess)
Stage 2   PNG pages → numpy arrays       (PIL pad-to-square + resize + normalise)
Stage 3   MIDI → timestamps
          • pretty_midi.get_downbeats() → precise measure-start times in seconds
          • Measures distributed evenly across pages → page-break measure numbers
          • Per-page x-fraction → time interpolators (measure-level resolution)

Why LilyPond instead of MuseScore
----------------------------------
CYOLO was trained on MSMD, which uses LilyPond for all score rendering.
LilyPond and MuseScore produce visually distinct output (different notehead
fonts, staff-line weights, beam angles, spacing).  Using the same renderer
as the training data eliminates the most significant source of domain gap
and is expected to bring accuracy from ~4% into the 40-60% range on
clean MIDI.

Requirements
------------
    pip install numpy pillow pretty_midi scipy

External tool (required):
    LilyPond >= 2.24  --  https://lilypond.org/download.html
    Includes both `lilypond` and `midi2ly` executables.

Windows install paths
---------------------
Set LILYPOND_DIR below to the folder that contains lilypond.exe and midi2ly.
"""

import subprocess
import glob
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Set this to the folder containing lilypond.exe / midi2ly on your system:
LILYPOND_DIR = r"C:\Program Files\LilyPond\bin"
# LILYPOND_DIR = r"C:\Program Files (x86)\LilyPond\usr\bin"
# LILYPOND_DIR = r"C:\lilypond\usr\bin"

TARGET_SIZE = (416, 416)
GRAYSCALE   = True          # CYOLO expects single-channel input


# ---------------------------------------------------------------------------
# STAGE 1a: Locate executables
# ---------------------------------------------------------------------------

def find_lilypond() -> tuple[str, str]:
    """
    Return (lilypond_exe, midi2ly_script) paths.

    midi2ly is always a plain Python script (no .exe).  Run it via
    _find_lilypond_python() rather than executing it directly.

    Search order:
      1. LILYPOND_DIR config at top of this file.
      2. System PATH (Linux / macOS package installs).
      3. Common Windows install locations.
    """
    def _pair(directory: str):
        ext = ".exe" if os.name == "nt" else ""
        ly  = os.path.join(directory, f"lilypond{ext}")
        m2l = os.path.join(directory, "midi2ly.py")  # always a script, never .exe
        if os.path.isfile(ly) and os.path.isfile(m2l):
            return ly, m2l
        return None

    pair = _pair(LILYPOND_DIR)
    if pair:
        return pair

    ly_path  = shutil.which("lilypond")
    m2l_path = shutil.which("midi2ly")
    if ly_path and m2l_path:
        return ly_path, m2l_path

    for d in [
        r"C:\Program Files\LilyPond\usr\bin",
        r"C:\Program Files (x86)\LilyPond\usr\bin",
        r"C:\lilypond\usr\bin",
        r"C:\Program Files\LilyPond 2.25\usr\bin",
        r"C:\Program Files\LilyPond 2.24\usr\bin",
    ]:
        pair = _pair(d)
        if pair:
            return pair

    raise FileNotFoundError(
        "LilyPond executables (lilypond + midi2ly) not found.\n"
        "Install LilyPond from https://lilypond.org/download.html\n"
        "then set LILYPOND_DIR at the top of image_processing.py to the\n"
        "folder containing lilypond.exe and midi2ly, e.g.:\n"
        r"  C:\Program Files\LilyPond\usr\bin"
    )


def _find_lilypond_python() -> str:
    """
    Return the Python interpreter to use for running midi2ly.

    Prefers LilyPond's own bundled Python (lives alongside lilypond.exe) so
    that midi2ly's imports resolve correctly without touching the project venv.
    Falls back to sys.executable if not found.
    """
    import sys
    lilypond_exe, _ = find_lilypond()
    bin_dir = os.path.dirname(lilypond_exe)

    for name in ("python3.exe", "python.exe", "python3", "python"):
        candidate = os.path.join(bin_dir, name)
        if os.path.isfile(candidate):
            return candidate

    return sys.executable  # project venv Python as last resort


# ---------------------------------------------------------------------------
# STAGE 1b: MIDI → LilyPond source
# ---------------------------------------------------------------------------

def midi_to_ly(midi_path: str, output_dir: str) -> str:
    """
    Convert a MIDI file to a LilyPond source file (.ly) using midi2ly.

    midi2ly is a Python script bundled with LilyPond.  It is invoked as
    ``python midi2ly ...`` rather than as a standalone executable.

    Returns
    -------
    Absolute path to the generated .ly file.
    """
    midi_path  = str(Path(midi_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    _, midi2ly_script = find_lilypond()
    python_exe        = _find_lilypond_python()
    ly_path           = os.path.join(output_dir, "sheet.ly")

    # Run as: python midi2ly -o sheet.ly input.mid
    cmd    = [python_exe, midi2ly_script, "-o", ly_path, midi_path]
    print(f"[1a] midi2ly: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"midi2ly failed (code {result.returncode}).\n"
            f"Stderr: {result.stderr.strip()}"
        )
    if not os.path.isfile(ly_path):
        raise FileNotFoundError(
            f"midi2ly did not produce '{ly_path}'.\n"
            f"Stderr: {result.stderr.strip()}"
        )

    print(f"[1a] LilyPond source: {ly_path}")
    return ly_path


# ---------------------------------------------------------------------------
# STAGE 1c: LilyPond source → PNG pages
# ---------------------------------------------------------------------------

def ly_to_sheet_images(ly_path: str, output_dir: str, resolution: int = 150) -> list[str]:
    """
    Render a LilyPond .ly file to one PNG image per page.

    LilyPond uses the same rendering engine as MSMD, so the visual style
    matches the training distribution of the CYOLO model.

    Parameters
    ----------
    ly_path    : Path to the .ly file produced by midi_to_ly().
    output_dir : Directory where page images will be saved.
    resolution : PNG resolution in DPI. Default 150.

    Returns
    -------
    Sorted list of PNG file paths, one per page.
    LilyPond naming: sheet.png, sheet-2.png, sheet-3.png, ...
    """
    ly_path    = str(Path(ly_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    lilypond, _ = find_lilypond()
    out_stem    = os.path.join(output_dir, "sheet")

    cmd = [
        lilypond,
        "--png",
        f"-dresolution={resolution}",
        "-dno-point-and-click",
        "-o", out_stem,
        ly_path,
    ]

    print(f"[1b] LilyPond: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(
            f"LilyPond failed (code {result.returncode}).\n"
            f"Stderr: {result.stderr.strip()[-500:]}"
        )

    # LilyPond page naming: sheet.png (page 1), sheet-2.png, sheet-3.png, ...
    image_paths = sorted(
        glob.glob(os.path.join(output_dir, "sheet*.png")),
        key=lambda p: int(Path(p).stem.split("-")[1]) if "-" in Path(p).stem else 1,
    )

    if not image_paths:
        raise FileNotFoundError(
            f"LilyPond produced no PNG files in '{output_dir}'.\n"
            f"Stderr: {result.stderr.strip()[-300:]}"
        )

    print(f"[1b] Generated {len(image_paths)} page image(s).")
    return image_paths


def midi_to_sheet_images(midi_path: str, output_dir: str, resolution: int = 150) -> list[str]:
    """MIDI → PNG pages via LilyPond. Calls midi_to_ly then ly_to_sheet_images."""
    ly_path = midi_to_ly(midi_path, output_dir)
    return ly_to_sheet_images(ly_path, output_dir, resolution=resolution)


# ---------------------------------------------------------------------------
# STAGE 2: PNG images → numpy matrices
# ---------------------------------------------------------------------------

def images_to_numpy(
    image_paths: list[str],
    target_size: tuple[int, int] = TARGET_SIZE,
    grayscale:   bool            = GRAYSCALE,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Load and resize PNG images into normalised numpy arrays.

    Processing: RGBA-flatten → grayscale → pad-to-square → resize.
    Matches the preprocessing in data_utils.load_sequences() exactly.

    Returns
    -------
    matrices : list[np.ndarray]
        Float32 in [0, 1], shape (H, W) or (H, W, 3).
    content_width_fractions : list[float]
        original_width / max(original_width, original_height) per page.
        Used to correct YOLO x predictions for padding offset.
    """
    print(f"[2/3] Converting {len(image_paths)} image(s) to numpy "
          f"({'grayscale' if grayscale else 'RGB'}, {target_size})...")

    matrices: list[np.ndarray] = []
    cwfs:     list[float]      = []

    for i, path in enumerate(image_paths):
        img = Image.open(path)

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        img = img.convert("L")

        w, h = img.size
        cwf  = w / max(w, h)
        cwfs.append(cwf)

        if w != h:
            side   = max(w, h)
            canvas = Image.new("L", (side, side), 255)
            canvas.paste(img, (0, 0))
            img    = canvas

        if not grayscale:
            img = img.convert("RGB")

        img = img.resize(target_size, resample=Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        matrices.append(arr)
        print(f"    Page {i+1}: shape={arr.shape}, cwf={cwf:.3f}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}")

    print(f"[2/3] Done.")
    return matrices, cwfs


# ---------------------------------------------------------------------------
# STAGE 3: timing helpers
# ---------------------------------------------------------------------------

def _build_measure_time_map(midi_path: str) -> dict[int, float]:
    """1-based measure number → start time in seconds (via pretty_midi downbeats)."""
    try:
        import pretty_midi
    except ImportError as e:
        raise ImportError("pip install pretty_midi") from e
    pm = pretty_midi.PrettyMIDI(midi_path)
    return {i + 1: float(t) for i, t in enumerate(pm.get_downbeats())}


def _infer_page_break_measures(n_pages: int, measure_time_map: dict[int, float]) -> list[int]:
    """
    Distribute measures evenly across pages to estimate page-break positions.

    LilyPond determines breaks at render time and does not encode them in the
    .ly source, so we approximate by splitting the measure range evenly.
    """
    total = max(measure_time_map.keys())
    breaks = []
    for p in range(n_pages):
        m = round(1 + p * (total - 1) / max(n_pages - 1, 1))
        breaks.append(max(1, min(m, total)))
    return breaks


def extract_page_timestamps(
    midi_path: str,
    n_pages:   int,
    measure_time_map: "dict[int, float] | None" = None,
) -> list[float]:
    """
    Start time in seconds of each page. timestamps[0] is always 0.0.

    Parameters
    ----------
    measure_time_map : Pass a pre-computed dict to avoid loading pretty_midi twice.
    """
    if measure_time_map is None:
        measure_time_map = _build_measure_time_map(midi_path)

    breaks      = _infer_page_break_measures(n_pages, measure_time_map)
    max_measure = max(measure_time_map.keys())

    timestamps = [
        measure_time_map.get(min(m, max_measure), measure_time_map[max_measure])
        for m in breaks
    ]
    timestamps[0] = 0.0
    print(f"[3/3] Page timestamps (s): {[round(t, 3) for t in timestamps]}")
    return timestamps


def build_page_measure_interpolators(
    page_break_measures: list[int],
    measure_time_map:    dict[int, float],
    n_pages:             int,
) -> list:
    """
    Per-page scipy.interp1d objects: x-fraction [0,1] → seconds.

    Distributes measure timestamps evenly across page width — more accurate
    than a single linear interpolation for music with tempo changes.
    Returns None for pages with fewer than 2 measures.
    """
    from scipy import interpolate

    max_measure = max(measure_time_map.keys())
    interps = []

    for p in range(n_pages):
        first_m = page_break_measures[p]
        last_m  = (
            page_break_measures[p + 1] - 1
            if p + 1 < len(page_break_measures)
            else max_measure
        )
        measures = [m for m in range(first_m, last_m + 1) if m in measure_time_map]

        if len(measures) < 2:
            interps.append(None)
            continue

        x_frac = np.linspace(0.0, 1.0, len(measures))
        times  = np.array([measure_time_map[m] for m in measures], dtype=np.float64)
        interps.append(
            interpolate.interp1d(
                x_frac, times,
                kind="linear", bounds_error=False,
                fill_value=(times[0], times[-1]),
            )
        )

    return interps


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def midi_to_matrices(
    midi_path:         str,
    output_dir:        str             = "sheet_tmp",
    target_size:       tuple[int, int] = TARGET_SIZE,
    grayscale:         bool            = GRAYSCALE,
    keep_images:       bool            = True,
    return_timestamps: bool            = False,
    resolution:        int             = 150,
) -> "list[np.ndarray] | tuple[list[np.ndarray], list[float], list[float], list]":
    """
    End-to-end: MIDI → LilyPond → PNG → numpy matrices.

    Returns
    -------
    return_timestamps=False
        list[np.ndarray]

    return_timestamps=True
        (matrices, page_timestamps, content_width_fractions, page_measure_interpolators)
    """
    print(f"\n{'='*55}")
    print(f"  MIDI → Matrices Pipeline  (LilyPond)")
    print(f"  Input : {midi_path}")
    print(f"  Output: {target_size[0]}x{target_size[1]} "
          f"{'grayscale' if grayscale else 'RGB'}, {resolution} DPI")
    print(f"{'='*55}\n")

    image_paths    = midi_to_sheet_images(midi_path, output_dir, resolution)
    matrices, cwfs = images_to_numpy(image_paths, target_size, grayscale)

    page_timestamps = page_measure_interpolators = None

    if return_timestamps:
        measure_time_map    = _build_measure_time_map(midi_path)
        page_break_measures = _infer_page_break_measures(len(matrices), measure_time_map)
        page_timestamps     = extract_page_timestamps(
            midi_path, n_pages=len(matrices), measure_time_map=measure_time_map
        )
        page_measure_interpolators = build_page_measure_interpolators(
            page_break_measures, measure_time_map, n_pages=len(matrices)
        )

    if not keep_images:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Cleaned up '{output_dir}'.")
    else:
        print(f"Intermediate files kept in '{output_dir}'.")

    print(f"\n✓ Pipeline complete: {len(matrices)} page(s).\n")

    if return_timestamps:
        return matrices, page_timestamps, cwfs, page_measure_interpolators
    return matrices


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_matrices(matrices: list[np.ndarray], save_dir: str) -> list[str]:
    """Save each matrix as a .npy file."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, mat in enumerate(matrices):
        path = os.path.join(save_dir, f"page_{i+1:04d}.npy")
        np.save(path, mat)
        paths.append(path)
    return paths


def load_matrices(save_dir: str) -> list[np.ndarray]:
    """Load previously saved .npy matrix files."""
    return [np.load(p) for p in sorted(glob.glob(os.path.join(save_dir, "page_*.npy")))]


if __name__ == "__main__":
    import sys
    midi_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not midi_file or not os.path.isfile(midi_file):
        print("Usage: python image_processing.py <path_to_midi>")
        sys.exit(1)

    matrices, timestamps, cwfs, interps = midi_to_matrices(
        midi_path=midi_file, output_dir="sheet_tmp",
        return_timestamps=True, resolution=150,
    )
    for i, m in enumerate(matrices):
        print(f"  page {i+1}: shape={m.shape}  cwf={cwfs[i]:.3f}  "
              f"t={timestamps[i]:.2f}s  interp={'yes' if interps[i] else 'no'}")