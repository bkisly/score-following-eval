import csv
from pathlib import Path
from typing import List, Dict, Callable

import numpy as np
from numba.core.types import none
from tqdm import tqdm

from evaluation.data import ExperimentVariation, Piece, EvaluationResult
from evaluation.evaluator import Evaluator
from evaluation.metrics import EvaluationMetrics
from models.score_follower import ScoreFollower
from utils.audio_processing import AudioProcessor


class ExperimentsRunner:
    def __init__(self, evaluator: Evaluator, models: List[ScoreFollower]):
        self.evaluator = evaluator
        self.models = models

    def test_tempo_robustness(
            self,
            pieces: List[Piece],
            tempo_shifts: List[float] = None,
            results_path: str = None,
            verbose: bool = False) -> Dict[str, List[ExperimentVariation]]:
        results = self._create_dict_for_models()

        if tempo_shifts is None:
            tempo_shifts = [i / 10 for i in range(-5, 6)]

        for tempo_shift in tempo_shifts:
            if verbose:
                print(f"Beginning test for tempo shift {tempo_shift}...")

            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = lambda a, ap: ap.time_stretch(a, tempo_shift)
            variation_results = self.test_average_metrics(
                pieces,
                audio_transformator=audio_transformator,
                results_path=self._add_to_path_stem(results_path, f"_tempo{tempo_shift}"),
                verbose=verbose)

            for key in variation_results:
                results[key].append(ExperimentVariation(factor=tempo_shift, result=variation_results[key]))

        return results

    def test_noise_robustness(
            self,
            pieces: List[Piece],
            noise_factors: List[float] = None,
            results_path: str = None,
            verbose: bool = False) -> Dict[str, List[ExperimentVariation]]:
        results = self._create_dict_for_models()

        if noise_factors is None:
            noise_factors = [i / 10 for i in range(-5, 6)]

        for noise_factor in noise_factors:
            if verbose:
                print(f"Beginning test for noise factor {noise_factor}...")

            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = lambda a, ap: ap.add_noise(a, noise_factor)
            variation_results = self.test_average_metrics(
                pieces,
                audio_transformator=audio_transformator,
                results_path=self._add_to_path_stem(results_path, f"_noise{noise_factor}"),
                verbose=verbose)

            for key in variation_results:
                results[key].append(ExperimentVariation(factor=noise_factor, result=variation_results[key]))

        return results

    def test_pitch_robustness(
            self,
            pieces: List[Piece],
            semitone_shifts: List[int] = None,
            results_path: str = None,
            verbose: bool = False) -> Dict[str, List[ExperimentVariation]]:
        results = self._create_dict_for_models()

        if semitone_shifts is None:
            semitone_shifts = [i for i in range(-5, 6)]

        for semitone_shift in semitone_shifts:
            if verbose:
                print(f"Beginning test for pitch shift {semitone_shift}...")

            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = lambda a, ap: ap.pitch_shift(a, semitone_shift)
            variation_results = self.test_average_metrics(
                pieces,
                audio_transformator=audio_transformator,
                results_path=self._add_to_path_stem(results_path, f"_pitch{semitone_shift}"),
                verbose=verbose)

            for key in variation_results:
                results[key].append(ExperimentVariation(factor=semitone_shift, result=variation_results[key]))

        return results

    def test_recovery_time(
            self,
            pieces: List[Piece],
            noise_start: int = 10,
            noise_duration: int = 3,
            verbose: bool = False) -> Dict[str, Dict[int, float]]:
        results = {model.name: {} for model in self.models}
        errors = self._create_dict_for_models()
        models_dict = {model.name: model for model in self.models}
        audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = \
            lambda a, ap: ap.replace_fragment_with_noise(a, noise_start, noise_duration)

        iterator = tqdm(enumerate(pieces), total=len(pieces), disable=not verbose)
        for i, piece in iterator:
            iterator.set_description(f"Starting evaluation of piece no. {i+1}... "
                                     f"(Path to MIDI: {piece.midi_path}, "
                                     f"path to audio: {piece.audio_path})")

            for model_name in results:
                predictions, _, ground_truth_values = self.evaluator.evaluate_model_per_chunk(
                    models_dict[model_name],
                    piece.audio_path,
                    piece.midi_path,
                    audio_transformator=audio_transformator,
                    verbose=False
                )

                errors[model_name].append(
                    [abs(prediction - ground_truth) for prediction, ground_truth in zip(predictions, ground_truth_values)]
                )

        for model_name in errors:
            results[model_name] = {i: sum(chunk_errors) / len(chunk_errors) for i, chunk_errors in enumerate(zip(*errors[model_name]))}

        return results

    def test_average_metrics(
            self,
            pieces: List[Piece],
            audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = None,
            results_path: str = None,
            verbose: bool = False
    ) -> Dict[str, EvaluationMetrics]:
        results: Dict[str, List[EvaluationResult]] = self._create_dict_for_models()

        iterator = tqdm(enumerate(pieces), total=len(pieces), disable=not verbose)
        for i, piece in iterator:
            iterator.set_description(f"Evaluating piece no. {i+1}... "
                                     f"(Path to MIDI: {piece.midi_path}, "
                                     f"path to audio: {piece.audio_path})")

            evaluation_results = self.evaluator.compare_all_models(
                self.models, piece.audio_path, piece.midi_path, audio_transformator=audio_transformator, save_results=False)

            for key in evaluation_results:
                results[key].append(EvaluationResult(piece_metadata=piece.metadata, result=evaluation_results[key]))

        calculated_results = {}
        for key in results:
            calculated_results[key] = EvaluationMetrics.avg(list(map(lambda r: r.result, results[key])))

        if results_path is not None:
            self._dump_results(results_path, results)

        return calculated_results

    def _create_dict_for_models(self) -> Dict[str, List]:
        return {model.name: [] for model in self.models}

    @staticmethod
    def _dump_results(path: str, results: Dict[str, List[EvaluationResult]]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["composer", "title", "model_name", *EvaluationMetrics.empty().to_dict().keys()])

            for model_name in results:
                writer.writerows(map(
                    lambda r: [r.piece_metadata.composer, r.piece_metadata.title,
                               model_name, *r.result.to_dict().values()],
                    results[model_name]
                ))

    @staticmethod
    def _add_to_path_stem(path: str, text: str) -> str | None:
        if path is None:
            return None

        path_obj = Path(path)
        return str(path_obj.with_stem(f"{path_obj.stem}{text}"))