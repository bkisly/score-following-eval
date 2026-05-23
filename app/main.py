"""
Score Following Visualizer — entry point.

Usage
-----
    python -m app.main --wav track.wav --midi reference.mid --model 1
    python -m app.main --wav track.wav --midi reference.mid --model 3 --checkpoint heurmit.pth
    python -m app.main --wav track.wav --midi reference.mid --model 4 --checkpoint patchformer.pth

Model IDs
---------
    1  OTW — ConcertCue (no training required)
    2  CYOLO-SB+A       (pretrained checkpoint loaded from cyolo/trained_models/)
    3  HeurMiT (CNN)    (requires --checkpoint)
    4  PatchFormer      (requires --checkpoint)
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Ensure the project root is on sys.path regardless of how the script is
#    invoked (python -m app.main, or python app/main.py, or from a subdir). ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="score-follower",
        description="Real-time score following visualizer with piano roll display.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--wav",        required=True,          help="Input audio track (.wav)")
    p.add_argument("--midi",       required=True,          help="Reference MIDI file (.mid / .midi)")
    p.add_argument("--model",      required=True, type=int,
                   choices=[1, 2, 3, 4],
                   help="Model ID: 1=OTW, 2=CYOLO, 3=HeurMiT, 4=PatchFormer")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pth checkpoint for models 3 (HeurMiT) or 4 (PatchFormer)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Validate input files ──────────────────────────────────────────────────
    for flag, path in [("--wav", args.wav), ("--midi", args.midi)]:
        if not os.path.isfile(path):
            print(f"[ERROR] {flag}: file not found — {path!r}", file=sys.stderr)
            sys.exit(1)

    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        print(f"[ERROR] --checkpoint: file not found — {args.checkpoint!r}", file=sys.stderr)
        sys.exit(1)

    # ── Build the model ───────────────────────────────────────────────────────
    from app.model_factory import get_model, MODEL_NAMES

    model_name = MODEL_NAMES[args.model]
    print(f"[ScoreFollower] Initialising model {args.model}: {model_name}")

    try:
        model = get_model(args.model, args.checkpoint)
    except Exception as exc:
        print(f"[ERROR] Could not create model: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Load reference (may take a few seconds for OTW / CYOLO) ──────────────
    print(f"[ScoreFollower] Loading reference: {args.midi}")
    try:
        model.load_reference(args.midi)
    except Exception as exc:
        print(f"[ERROR] load_reference() failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[ScoreFollower] Reference loaded.  Starting GUI…")

    # ── Launch the Qt application ─────────────────────────────────────────────
    # Import Qt only after all CLI validation has passed so that --help and
    # error messages are printed instantly (Qt import can be slow).
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from app.app import MainWindow

    # High-DPI scaling is enabled by default in Qt6; explicit attribute
    # setting is only needed for Qt5 compatibility shims.
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Score Follower")

    window = MainWindow(
        wav_path=args.wav,
        midi_path=args.midi,
        model=model,
        model_name=model_name,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
