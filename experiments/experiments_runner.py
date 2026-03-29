from typing import List, Dict, Callable

import numpy as np
from tqdm import tqdm

from evaluation.evaluator import Evaluator
from evaluation.data import ExperimentVariation, Piece
from models.score_follower import ScoreFollower
from evaluation.metrics import EvaluationMetrics, MetricKeys
from utils.audio_processing import AudioProcessor


class ExperimentsRunner:
    def __init__(self, evaluator: Evaluator, models: List[ScoreFollower]):
        self.evaluator = evaluator
        self.models = models

    def test_tempo_robustness(self, pieces: List[Piece], tempo_shifts: List[float] = None, verbose: bool = False) -> Dict[str, List[ExperimentVariation]]:
        results = self._create_dict_for_models()

        if tempo_shifts is None:
            tempo_shifts = [i / 10 for i in range(-5, 6)]

        for tempo_shift in tempo_shifts:
            if verbose:
                print(f"Beginning test for tempo shift {tempo_shift}...")

            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = lambda a, ap: ap.time_stretch(a, tempo_shift)
            variation_results = self.test_average_metrics(pieces, audio_transformator=audio_transformator, verbose=verbose)

            for key in variation_results:
                results[key].append(ExperimentVariation(factor=tempo_shift, result=variation_results[key]))

        return results

    def test_noise_robustness(self, pieces: List[Piece], noise_factors: List[float] = None, verbose: bool = False) -> Dict[str, List[ExperimentVariation]]:
        results = self._create_dict_for_models()

        if noise_factors is None:
            noise_factors = [i / 10 for i in range(-5, 6)]

        for noise_factor in noise_factors:
            if verbose:
                print(f"Beginning test for noise factor {noise_factor}...")

            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = lambda a, ap: ap.add_noise(a, noise_factor)
            variation_results = self.test_average_metrics(pieces, audio_transformator=audio_transformator)

            for key in variation_results:
                results[key].append(ExperimentVariation(factor=noise_factor, result=variation_results[key]))

        return results

    def test_pitch_robustness(self, pieces: List[Piece], semitone_shifts: List[int] = None, verbose: bool = False) -> Dict[str, List[ExperimentVariation]]:
        results = self._create_dict_for_models()

        if semitone_shifts is None:
            semitone_shifts = [i for i in range(-5, 6)]

        for semitone_shift in semitone_shifts:
            if verbose:
                print(f"Beginning test for pitch shift {semitone_shift}...")

            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = lambda a, ap: ap.pitch_shift(a, semitone_shift)
            variation_results = self.test_average_metrics(pieces, audio_transformator=audio_transformator, verbose=verbose)

            for key in variation_results:
                results[key].append(ExperimentVariation(factor=semitone_shift, result=variation_results[key]))

        return results

    def test_recovery_time(self, pieces: List[Piece]) -> Dict[str, Dict[int, float]]:
        pass

    def test_average_metrics(
            self,
            pieces: List[Piece],
            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = None,
            verbose: bool = False
    ) -> Dict[str, EvaluationMetrics]:
        results = self._create_dict_for_models()

        iterator = tqdm(enumerate(pieces), total=len(pieces), disable=not verbose)
        for i, piece in iterator:
            iterator.set_description(f"Evaluating piece no. {i+1}... "
                                     f"(Path to MIDI: {piece.midi_path}, "
                                     f"path to audio: {piece.audio_path}, )")

            evaluation_results = self.evaluator.compare_all_models(
                self.models, piece.audio_path, piece.midi_path, audio_transformator=audio_transformator, save_results=False)

            for key in evaluation_results:
                results[key].append(evaluation_results[key])

        calculated_results = {}
        for key in results:
            calculated_results[key] = EvaluationMetrics.avg(results[key])

        return calculated_results

    def _create_dict_for_models(self) -> Dict[str, List]:
        results = {}

        for model in self.models:
            results[model.name] = []

        return results