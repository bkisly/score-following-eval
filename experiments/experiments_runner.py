from typing import List, Dict

from tqdm import tqdm

from evaluation.evaluator import Evaluator
from evaluation.data import ExperimentVariation, Piece
from models.score_follower import ScoreFollower
from evaluation.metrics import EvaluationMetrics, MetricKeys


class ExperimentsRunner:
    def __init__(self, evaluator: Evaluator, models: List[ScoreFollower]):
        self.evaluator = evaluator
        self.models = models

    def test_single_metric(self, metric_key: MetricKeys, pieces: List[Piece], verbose: bool = False) -> Dict[str, float]:
        metrics = self._test_average_metrics(pieces, verbose)
        results_for_metric: Dict[str, float] = {}

        for model_name in metrics:
            results_for_metric[model_name] = metrics[model_name].to_dict()[metric_key]

        return results_for_metric

    def test_tempo_robustness(self, pieces: List[Piece]) -> Dict[str, List[ExperimentVariation]]:
        pass

    def test_noise_robustness(self, pieces: List[Piece]) -> Dict[str, List[ExperimentVariation]]:
        pass

    def test_pitch_robustness(self, pieces: List[Piece]) -> Dict[str, List[ExperimentVariation]]:
        pass

    def test_technical_difficulty_robustness(self, pieces: List[Piece]) -> Dict[str, EvaluationMetrics]:
        pass

    def test_artistic_figures_robustness(self, pieces: List[Piece]) -> Dict[str, EvaluationMetrics]:
        pass

    def test_recovery_time(self, pieces: List[Piece]) -> Dict[str, Dict[int, float]]:
        pass

    def _test_average_metrics(self, pieces: List[Piece], verbose: bool = False) -> Dict[str, EvaluationMetrics]:
        results = {}

        for model in self.models:
            results[model.name] = []

        iterator = tqdm(enumerate(pieces), total=len(pieces), disable=not verbose)
        for i, piece in iterator:
            iterator.set_description(f"Evaluating piece no. {i}... "
                                     f"(Path to MIDI: {piece.midi_path}, "
                                     f"path to audio: {piece.audio_path}, )")

            evaluation_results = self.evaluator.compare_all_models(
                self.models, piece.audio_path, piece.midi_path, save_results=False)

            for key in evaluation_results:
                results[key].append(evaluation_results[key])

        calculated_results = {}
        for key in results:
            calculated_results[key] = EvaluationMetrics.avg(results[key])

        return calculated_results