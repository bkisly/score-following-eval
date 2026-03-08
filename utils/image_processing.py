"""
image_processing.py
-------------------
Full pipeline: MIDI file → multi-page sheet music images → list of 416×416
numpy matrices, with optional per-page start timestamps for real-time page
tracking.

Pipeline overview
-----------------
Stage 1a  MIDI → MusicXML               (MuseScore subprocess)
Stage 1b  MusicXML → LilyPond source    (musicxml2ly, ships with LilyPond)
Stage 1c  LilyPond source → PNG pages   (lilypond subprocess)
Stage 2   PNG pages → numpy arrays      (PIL pad-to-square + resize + normalise)
Stage 3   MIDI → timestamps
          • pretty_midi.get_downbeats() → precise measure-start times in seconds
          • Measures distributed evenly across pages → page-break measure numbers
          • Per-page x-fraction → time interpolators (measure-level resolution)

Why this pipeline
-----------------
CYOLO was trained on MSMD, which was produced via MusicXML → musicxml2ly →
LilyPond → PNG.  Replicating this exact toolchain eliminates the domain gap
between training data and inference input:

  • MuseScore handles MIDI import well: correct 2-staff piano layout,
    triplets, ties, and intelligent quantization.  Neither midi2ly nor
    MuseScore's direct .ly export produce acceptable results for piano MIDI.

  • musicxml2ly (bundled with LilyPond) converts the clean MusicXML to
    LilyPond source with proper voice separation — the same step used in MSMD.

  • LilyPond renders with the same notehead font, staff-line weights,
    beam angles, and spacing as the training data.

Requirements
------------
    pip install numpy pillow pretty_midi scipy

External tools (both required):
    MuseScore 3 or 4  —  https://musescore.org/en/download
    LilyPond >= 2.24  —  https://lilypond.org/download.html
    (musicxml2ly ships inside the LilyPond installation)
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

# Path to the LilyPond bin folder (contains lilypond.exe):
LILYPOND_DIR = r"C:\Program Files\LilyPond\bin"
# LILYPOND_DIR = r"C:\Program Files (x86)\LilyPond\usr\bin"
# LILYPOND_DIR = r"C:\lilypond\usr\bin"

# Path to the MuseScore executable (used only for MIDI → .ly conversion):
MUSESCORE_PATH = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
# MUSESCORE_PATH = r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"

TARGET_SIZE = (416, 416)
GRAYSCALE   = True          # CYOLO expects single-channel input


# ---------------------------------------------------------------------------
# STAGE 1a: Locate executables
# ---------------------------------------------------------------------------

def find_lilypond() -> str:
    """
    Return the path to the lilypond executable.

    Search order:
      1. LILYPOND_DIR config at top of this file.
      2. System PATH (Linux / macOS package installs).
      3. Common Windows install locations.
    """
    ext = ".exe" if os.name == "nt" else ""

    candidate = os.path.join(LILYPOND_DIR, f"lilypond{ext}")
    if os.path.isfile(candidate):
        return candidate

    on_path = shutil.which("lilypond")
    if on_path:
        return on_path

    for d in [
        r"C:\Program Files\LilyPond\usr\bin",
        r"C:\Program Files (x86)\LilyPond\usr\bin",
        r"C:\lilypond\usr\bin",
        r"C:\Program Files\LilyPond 2.25\usr\bin",
        r"C:\Program Files\LilyPond 2.24\usr\bin",
    ]:
        c = os.path.join(d, f"lilypond{ext}")
        if os.path.isfile(c):
            return c

    raise FileNotFoundError(
        "lilypond executable not found.\n"
        "Install LilyPond from https://lilypond.org/download.html\n"
        "then set LILYPOND_DIR at the top of image_processing.py, e.g.:\n"
        r"  C:\Program Files\LilyPond\usr\bin"
    )


def find_musescore() -> str:
    """
    Return the path to the MuseScore executable.
    Used only for MIDI → LilyPond (.ly) export.
    """
    if os.path.isfile(MUSESCORE_PATH):
        return MUSESCORE_PATH

    for name in ("MuseScore4", "MuseScore3", "mscore"):
        found = shutil.which(name)
        if found:
            return found

    for path in [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
    ]:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "MuseScore executable not found.\n"
        "Install MuseScore from https://musescore.org/en/download\n"
        "or set MUSESCORE_PATH at the top of image_processing.py."
    )


# ---------------------------------------------------------------------------
# STAGE 1b: MIDI → MusicXML  (via MuseScore)
# ---------------------------------------------------------------------------

def midi_to_musicxml(midi_path: str, output_dir: str) -> str:
    """
    Export a MIDI file to uncompressed MusicXML using MuseScore.

    MuseScore has an excellent MIDI importer: it correctly separates tracks
    into treble + bass staves, handles triplets, ties, and complex rhythms.
    MusicXML is then passed to musicxml2ly for LilyPond-style rendering.

    Returns
    -------
    Absolute path to the generated .musicxml file.
    """
    midi_path  = str(Path(midi_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    mscore   = find_musescore()
    mxl_path = os.path.join(output_dir, "sheet.musicxml")

    cmd    = [mscore, midi_path, "-o", mxl_path]
    print(f"[1a] MuseScore → MusicXML: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")

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
            f"Stderr: {result.stderr.strip()}"
        )

    print(f"[1a] MusicXML written to: {mxl_path}")
    return mxl_path


# ---------------------------------------------------------------------------
# STAGE 1c: MusicXML → LilyPond source  (via musicxml2ly)
# ---------------------------------------------------------------------------

def _find_musicxml2ly() -> tuple[str, str]:
    """
    Return (python_exe, musicxml2ly_script) for running musicxml2ly.

    musicxml2ly ships with LilyPond (same bin folder as lilypond.exe) and,
    like midi2ly, is a plain Python script that must be run via an interpreter.
    Prefers LilyPond's bundled Python so its imports resolve without touching
    the project venv.
    """
    import sys
    lilypond_exe = find_lilypond()
    bin_dir      = os.path.dirname(lilypond_exe)

    m2l = os.path.join(bin_dir, "musicxml2ly")
    if not os.path.isfile(m2l):
        m2l = os.path.join(bin_dir, "musicxml2ly.py")   # Windows LilyPond installer uses .py
    if not os.path.isfile(m2l):
        raise FileNotFoundError(
            f"musicxml2ly not found in '{bin_dir}'.\n"
            "It ships with LilyPond — check your LilyPond installation."
        )

    for name in ("python3.exe", "python.exe", "python3", "python"):
        candidate = os.path.join(bin_dir, name)
        if os.path.isfile(candidate):
            return candidate, m2l

    return sys.executable, m2l   # fall back to project venv Python


def musicxml_to_ly(mxl_path: str, output_dir: str) -> str:
    """
    Convert a MusicXML file to LilyPond source (.ly) using musicxml2ly.

    musicxml2ly (bundled with LilyPond) handles multi-staff piano scores,
    voice separation, triplets, and ties correctly — unlike midi2ly.
    This is also how the MSMD dataset was originally produced.

    Returns
    -------
    Absolute path to the generated .ly file.
    """
    mxl_path   = str(Path(mxl_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    python_exe, musicxml2ly_script = _find_musicxml2ly()
    ly_path = os.path.join(output_dir, "sheet.ly")

    cmd    = [python_exe, musicxml2ly_script, "-o", ly_path, mxl_path]
    print(f"[1b] musicxml2ly: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")

    if result.returncode != 0:
        raise RuntimeError(
            f"musicxml2ly failed (code {result.returncode}).\n"
            f"Stderr: {result.stderr.strip()}"
        )
    if not os.path.isfile(ly_path):
        raise FileNotFoundError(
            f"musicxml2ly did not produce '{ly_path}'.\n"
            f"Stderr: {result.stderr.strip()}"
        )

    print(f"[1b] LilyPond source written to: {ly_path}")
    return ly_path


def midi_to_ly(midi_path: str, output_dir: str) -> str:
    """
    Full MIDI → LilyPond source pipeline.

    MIDI → MuseScore → MusicXML → musicxml2ly → .ly

    MuseScore handles the MIDI import (proper 2-staff piano layout, good
    quantization). musicxml2ly handles the MusicXML → LilyPond conversion
    (correct voice separation, matching MSMD rendering style).
    """
    mxl_path = midi_to_musicxml(midi_path, output_dir)
    return musicxml_to_ly(mxl_path, output_dir)


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

    lilypond  = find_lilypond()
    out_stem  = os.path.join(output_dir, "sheet")

    cmd = [
        lilypond,
        "--png",
        f"-dresolution={resolution}",
        "-dno-point-and-click",
        "-o", out_stem,
        ly_path,
    ]

    print(f"[1c] LilyPond: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=output_dir,
        encoding="utf-8", errors="replace",
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(
            f"LilyPond failed (code {result.returncode}).\n"
            f"Stderr: {result.stderr.strip()[-500:]}"
        )

    # LilyPond page naming: sheet.png (page 1), sheet-2.png, sheet-3.png, ...
    image_paths = sorted(
        glob.glob(os.path.join(output_dir, "sheet-page*.png")),
        key=lambda p: int(Path(p).stem.split("-page")[1]) if "-" in Path(p).stem else 1,
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