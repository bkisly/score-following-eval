"""
Główny ewaluator - porównuje wszystkie modele na wspólnych danych.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple
from pathlib import Path
import json

from numba.core.types import none
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.score_follower import ScoreFollower
from evaluation.metrics import MetricsCalculator, EvaluationMetrics, compare_models
from utils.audio_processing import AudioProcessor, simulate_real_time_input
from utils.midi_processing import MIDIProcessor


CHUNK_SIZE = 2048
SAMPLE_RATE = 22050


class Evaluator:
    """
    Ewaluator do porównywania modeli śledzenia partytury.
    """
    
    def __init__(self, 
                 tolerance_seconds: float = 0.5,
                 results_dir: str = "results"):
        """
        Args:
            tolerance_seconds: Tolerancja dla uznania predykcji za poprawną
            results_dir: Katalog do zapisywania wyników
        """
        self.tolerance = tolerance_seconds
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator(tolerance_seconds)
        self.audio_processor = AudioProcessor()
        self.midi_processor = MIDIProcessor()
        
        print(f"Evaluator initialized (tolerance: {tolerance_seconds}s)")

    def evaluate_model_per_chunk(self, model: ScoreFollower,
                                 audio_path: str,
                                 reference_path: str,
                                 audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = None,
                                 ground_truth_alignment: np.ndarray = None,
                                 verbose: bool = True):
        # Wczytaj audio
        midi = self.midi_processor.load_midi(reference_path)
        audio, sr = self.audio_processor.load_audio(audio_path) if ".mid" not in audio_path \
            else (self.midi_processor.synthesize_audio(midi), SAMPLE_RATE)

        if audio_transformator is not None:
            audio = audio_transformator(audio, self.audio_processor)

        # Wczytaj referencję
        model.load_reference(reference_path)
        model.reset()

        # Symuluj real-time processing
        chunks = simulate_real_time_input(
            audio,
            chunk_size=CHUNK_SIZE,
            sr=sr
        )

        number_of_frames = len(chunks)

        if ground_truth_alignment is None:
            ground_truth_alignment = self._generate_ground_truth(midi, number_of_frames)

        # Przetwarzaj chunk po chunku
        predictions = []
        latencies = []

        iterator = tqdm(enumerate(chunks), total=number_of_frames, disable=not verbose)

        for i, chunk in iterator:
            # Predykcja
            result = model.process_frame(chunk, sr)

            predictions.append(result['position'])
            latencies.append(result['latency'])

            if verbose and i % 50 == 0:
                iterator.set_description(
                    f"Position: {result['position']:.2f}s, "
                    f"Latency: {result['latency']:.1f}ms"
                )

        # Dopasuj długości (ground truth może być dłuższe)
        min_len = min(len(predictions), len(ground_truth_alignment))
        predictions = np.array(predictions[:min_len])

        return predictions, latencies, ground_truth_alignment[:min_len]
    
    def evaluate_single_model(self,
                              model: ScoreFollower,
                              audio_path: str,
                              reference_path: str,
                              audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = None,
                              ground_truth_alignment: np.ndarray = None,
                              verbose: bool = True) -> EvaluationMetrics:
        """
        Ewaluuje pojedynczy model na jednym utworze.
        
        Args:
            model: Model do ewaluacji
            audio_path: Ścieżka do pliku audio
            reference_path: Ścieżka do referencji (MIDI)
            ground_truth_alignment: Opcjonalnie - prawdziwe wyrównanie
            verbose: Czy wyświetlać progress bar
            
        Returns:
            EvaluationMetrics
        """
        if verbose:
            print(f"\nEvaluating {model.name} on {Path(audio_path).name}")

        predictions, latencies, ground_truth_alignment = self.evaluate_model_per_chunk(
            model, audio_path, reference_path, audio_transformator, ground_truth_alignment, verbose=verbose
        )
        
        # Oblicz metryki
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions=predictions,
            ground_truth=ground_truth_alignment,
            latencies=latencies
        )
        
        if verbose:
            print(metrics)
        
        return metrics

    @staticmethod
    def _generate_ground_truth(midi, num_frames, chunk_size=CHUNK_SIZE, sample_rate=SAMPLE_RATE):
        """
        Generates simplified ground truth alignment by linear mapping of MIDI file length in seconds
        to the amount of frames in the given audio file.
        """
        midi_duration = midi.get_end_time()
        frame_times = np.arange(num_frames) * chunk_size / sample_rate
        frame_times = np.clip(frame_times, 0.0, midi_duration)

        return frame_times

    def evaluate_tempo_robustness(self,
                                  model: ScoreFollower,
                                  audio_path: str,
                                  reference_path: str,
                                  tempo_ratios: List[float] = [0.9, 1.0, 1.1, 1.2],
                                  verbose: bool = True) -> Dict[float, EvaluationMetrics]:
        """
        Testuje model przy różnych tempach.
        
        Args:
            model: Model do testowania
            audio_path: Ścieżka do audio
            reference_path: Ścieżka do MIDI
            tempo_ratios: Lista współczynników tempa (1.0 = oryginał)
            verbose: Czy wyświetlać informacje
            
        Returns:
            Dict {tempo_ratio: EvaluationMetrics}
        """
        results = {}
        
        # Wczytaj oryginalne audio
        audio_orig, sr = self.audio_processor.load_audio(audio_path)
        
        for ratio in tempo_ratios:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Testing tempo ratio: {ratio}x ({ratio*100:.0f}%)")
                print(f"{'='*60}")
            
            # Time-stretch audio
            if ratio != 1.0:
                audio = self.audio_processor.time_stretch(audio_orig, rate=ratio)
            else:
                audio = audio_orig
            
            # Zapisz tymczasowo
            import soundfile as sf
            temp_audio_path = f"/tmp/temp_audio_{ratio}.wav"
            sf.write(temp_audio_path, audio, sr)
            
            # Ewaluuj
            metrics = self.evaluate_single_model(
                model, temp_audio_path, reference_path, verbose=verbose
            )
            
            results[ratio] = metrics
            
            # Usuń tymczasowy plik
            Path(temp_audio_path).unlink()
        
        return results
    
    def compare_all_models(self,
                           models: List[ScoreFollower],
                           audio_path: str,
                           reference_path: str,
                           audio_transformator: Callable[[np.ndarray, AudioProcessor], np.ndarray] = None,
                           save_results: bool = True,
                           verbose: bool = False) -> Dict[str, EvaluationMetrics]:
        """
        Porównuje wszystkie modele na tym samym utworze.
        
        Args:
            models: Lista modeli do porównania
            audio_path: Ścieżka do audio
            reference_path: Ścieżka do MIDI
            save_results: Czy zapisać wyniki do pliku
            verbose: Wyświetl postęp
            
        Returns:
            Dict {model_name: EvaluationMetrics}
        """
        results = {}

        if verbose:
            print(f"\n{'='*70}")
            print(f"COMPARING {len(models)} MODELS")
            print(f"Audio: {Path(audio_path).name}")
            print(f"{'='*70}")
        
        for model in models:
            # Sprawdź czy model wymaga treningu
            if model.requires_training() and not model.is_trained:
                print(f"\nSkipping {model.name} - not trained yet")
                continue
            
            # Ewaluuj
            metrics = self.evaluate_single_model(
                model, audio_path, reference_path, audio_transformator=audio_transformator, verbose=verbose
            )
            
            results[model.name] = metrics
        
        # Wyświetl porównanie
        if verbose:
            print(f"\n{'='*70}")
            print("COMPARISON RESULTS")
            print(f"{'='*70}")
            print(compare_models(results))
        
        # Zapisz wyniki
        if save_results:
            self._save_results(results, audio_path)
        
        return results
    
    def _save_results(self, 
                     results: Dict[str, EvaluationMetrics],
                     audio_path: str) -> None:
        """
        Zapisuje wyniki do JSON.
        
        Args:
            results: Wyniki ewaluacji
            audio_path: Ścieżka do pliku audio (do nazwy)
        """
        # Konwertuj do dict
        results_dict = {
            model_name: metrics.to_dict()
            for model_name, metrics in results.items()
        }
        
        # Dodaj metadane
        results_dict['metadata'] = {
            'audio_file': str(Path(audio_path).name),
            'tolerance': self.tolerance,
            'timestamp': str(Path().absolute())
        }
        
        # Zapisz
        audio_name = Path(audio_path).stem
        output_path = self.results_dir / f"results_{audio_name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def create_test_case(self,
                        audio_path: str,
                        reference_path: str,
                        name: str = "test_case") -> Dict[str, Any]:
        """
        Tworzy test case - parę audio + MIDI z ground truth.
        
        Args:
            audio_path: Ścieżka do audio
            reference_path: Ścieżka do MIDI
            name: Nazwa test case
            
        Returns:
            Dict z test case
        """
        # Wczytaj
        audio, sr = self.audio_processor.load_audio(audio_path)
        midi = self.midi_processor.load_midi(reference_path)
        
        # Stwórz ground truth
        midi_duration = self.midi_processor.get_duration(midi)
        audio_duration = len(audio) / sr
        n_frames = len(audio) // self.audio_processor.hop_length
        ground_truth = np.linspace(0, midi_duration, n_frames)
        
        test_case = {
            'name': name,
            'audio_path': audio_path,
            'reference_path': reference_path,
            'audio_duration': audio_duration,
            'midi_duration': midi_duration,
            'ground_truth': ground_truth,
            'n_frames': n_frames
        }
        
        return test_case


# Przykład użycia
if __name__ == "__main__":
    print("Evaluator ready!")
    print("\nExample usage:")
    print("""
from evaluation.evaluator import Evaluator
from models.dtw_model import DTWModel, OnlineTimeWarping

# Create evaluator
evaluator = Evaluator(tolerance_seconds=0.5)

# Create models
dtw = DTWModel()
otw = OnlineTimeWarping()

# Compare
results = evaluator.compare_all_models(
    models=[dtw, otw],
    audio_path="path/to/audio.wav",
    reference_path="path/to/reference.mid"
)
    """)
