import dataclasses
from enum import Enum

from utils.metrics import EvaluationMetrics


@dataclasses.dataclass
class Piece:
    midi_path: str
    audio_path: str

@dataclasses.dataclass
class ExperimentVariation:
    factor: float
    result: EvaluationMetrics

class MetricKeys(str, Enum):
    ACCURACY = "accuracy"
    MEAN_ERROR = "mean_error"
    STD_ERROR = "std_error"
    LATENCY = "latency"
    TEMPO_ROBUSTNESS = "tempo_robustness"