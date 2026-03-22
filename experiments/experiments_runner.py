from typing import List

from evaluation.evaluator import Evaluator
from evaluation.data import ExperimentVariation, Piece, MetricKeys
from models.base_model import BaseScoreFollower
from utils.metrics import EvaluationMetrics


class ExperimentsRunner:
    def __init__(self, evaluator: Evaluator, models: List[BaseScoreFollower]):
        self.evaluator = evaluator
        self.models = models

    def test_single_metric(self, metric_key: MetricKeys, pieces: List[Piece]) -> dict[BaseScoreFollower, float]:
        pass

    def test_tempo_robustness(self, pieces: List[Piece]) -> dict[BaseScoreFollower, List[ExperimentVariation]]:
        pass

    def test_noise_robustness(self, pieces: List[Piece]) -> dict[BaseScoreFollower, List[ExperimentVariation]]:
        pass

    def test_pitch_robustness(self, pieces: List[Piece]) -> dict[BaseScoreFollower, List[ExperimentVariation]]:
        pass

    def test_technical_difficulty_robustness(self, pieces: List[Piece]) -> dict[BaseScoreFollower, EvaluationMetrics]:
        pass

    def test_artistic_figures_robustness(self, pieces: List[Piece]) -> dict[BaseScoreFollower, EvaluationMetrics]:
        pass

    def test_recovery_time(self, pieces: List[Piece]) -> dict[BaseScoreFollower, dict[int, float]]:
        pass