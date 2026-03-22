from typing import List

from evaluation.evaluator import Evaluator
from evaluation.data import ExperimentVariation, Piece
from models.score_follower import ScoreFollower
from evaluation.metrics import EvaluationMetrics, MetricKeys


class ExperimentsRunner:
    def __init__(self, evaluator: Evaluator, models: List[ScoreFollower]):
        self.evaluator = evaluator
        self.models = models

    def test_single_metric(self, metric_key: MetricKeys, pieces: List[Piece]) -> dict[ScoreFollower, float]:
        pass

    def test_tempo_robustness(self, pieces: List[Piece]) -> dict[ScoreFollower, List[ExperimentVariation]]:
        pass

    def test_noise_robustness(self, pieces: List[Piece]) -> dict[ScoreFollower, List[ExperimentVariation]]:
        pass

    def test_pitch_robustness(self, pieces: List[Piece]) -> dict[ScoreFollower, List[ExperimentVariation]]:
        pass

    def test_technical_difficulty_robustness(self, pieces: List[Piece]) -> dict[ScoreFollower, EvaluationMetrics]:
        pass

    def test_artistic_figures_robustness(self, pieces: List[Piece]) -> dict[ScoreFollower, EvaluationMetrics]:
        pass

    def test_recovery_time(self, pieces: List[Piece]) -> dict[ScoreFollower, dict[int, float]]:
        pass