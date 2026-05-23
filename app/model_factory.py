"""
Model factory — maps a model ID (1-4) to a configured ScoreFollower instance.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path so models/* imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.score_follower import ScoreFollower

MODEL_NAMES: dict[int, str] = {
    1: "OTW — ConcertCue",
    2: "CYOLO-SB+A",
    3: "HeurMiT (CNN)",
    4: "PatchFormer (Transformer)",
}

_REQUIRES_CHECKPOINT = {3, 4}


def get_model(model_id: int, checkpoint_path: str | None = None) -> ScoreFollower:
    """
    Instantiate and optionally restore a score-following model.

    Parameters
    ----------
    model_id : int
        1 = OTWModel, 2 = CYOLOModel, 3 = HeurMiTModel, 4 = TransformerModel
    checkpoint_path : str | None
        Path to a .pth checkpoint file.  Required for models 3 and 4 to produce
        meaningful output; ignored for models 1 and 2.

    Returns
    -------
    ScoreFollower
        Configured model instance (reference NOT yet loaded — caller must call
        model.load_reference(midi_path) before starting playback).
    """
    if model_id not in MODEL_NAMES:
        raise ValueError(
            f"Invalid model ID {model_id!r}. Choose from: "
            + ", ".join(f"{k}={v}" for k, v in MODEL_NAMES.items())
        )

    if model_id == 1:
        from models.otw_model import OTWModel
        return OTWModel()

    if model_id == 2:
        from models.cyolo_model import CYOLOModel
        return CYOLOModel()

    if model_id == 3:
        from models.cnn_model import HeurMiTModel
        model = HeurMiTModel()
        if checkpoint_path:
            print(f"[ModelFactory] Loading HeurMiT checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)
        else:
            print(
                "[ModelFactory] WARNING: No --checkpoint provided for HeurMiT. "
                "The model will run in degraded (elapsed-time fallback) mode."
            )
        return model

    if model_id == 4:
        from models.transformer_model import TransformerModel
        model = TransformerModel()
        if checkpoint_path:
            print(f"[ModelFactory] Loading PatchFormer checkpoint: {checkpoint_path}")
            model.load_checkpoint(checkpoint_path)
        else:
            print(
                "[ModelFactory] WARNING: No --checkpoint provided for PatchFormer. "
                "The model will run in degraded (elapsed-time fallback) mode."
            )
        return model

    raise RuntimeError("Unreachable")
