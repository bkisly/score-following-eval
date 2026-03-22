import dataclasses
from enum import Enum

from evaluation.metrics import EvaluationMetrics


@dataclasses.dataclass
class Piece:
    midi_path: str
    audio_path: str

@dataclasses.dataclass
class ExperimentVariation:
    factor: float
    result: EvaluationMetrics
