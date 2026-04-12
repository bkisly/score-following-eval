"""
Moduł definiujący metryki ewaluacji dla systemów śledzenia partytury.
Bazuje na metrykach z literatury naukowej.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Self

import numpy as np


@dataclass
class EvaluationMetrics:
    """
    Klasa przechowująca wszystkie metryki ewaluacji.
    """
    # Podstawowe metryki
    frame_accuracy: float       # Procent poprawnie zidentyfikowanych ramek
    mean_error: float          # Średni błąd w sekundach
    median_error: float        # Mediana błędu
    std_error: float           # Odchylenie standardowe błędu
    
    # Metryki wydajności
    mean_latency: float        # Średnia latencja w ms
    max_latency: float         # Maksymalna latencja
    
    # Metryki robustness
    tempo_robustness: float    # Skuteczność przy zmianach tempa (0-1)
    error_recovery_time: float # Średni czas odzyskania po błędzie (s)
    
    # Dodatkowe statystyki
    total_frames: int
    correct_frames: int
    
    def to_dict(self) -> Dict:
        """Konwersja do słownika."""
        return {
            MetricKeys.ACCURACY: self.frame_accuracy,
            MetricKeys.MEAN_ERROR: self.mean_error,
            'median_error': self.median_error,
            MetricKeys.STD_ERROR: self.std_error,
            MetricKeys.MEAN_LATENCY: self.mean_latency,
            'max_latency': self.max_latency,
            MetricKeys.TEMPO_ROBUSTNESS: self.tempo_robustness,
            'error_recovery_time': self.error_recovery_time,
            'total_frames': self.total_frames,
            'correct_frames': self.correct_frames
        }
    
    def __str__(self) -> str:
        """Czytelne wyświetlanie metryk."""
        return f"""
Evaluation Metrics:
==================
Frame Accuracy:        {self.frame_accuracy:.2%}
Mean Error:           {self.mean_error:.3f}s
Median Error:         {self.median_error:.3f}s
Std Error:            {self.std_error:.3f}s
Mean Latency:         {self.mean_latency:.2f}ms
Max Latency:          {self.max_latency:.2f}ms
Tempo Robustness:     {self.tempo_robustness:.2%}
Error Recovery:       {self.error_recovery_time:.3f}s
Correct/Total:        {self.correct_frames}/{self.total_frames}
"""

    def __add__(self, other) -> Self:
        return EvaluationMetrics(
            frame_accuracy=self.frame_accuracy + other.frame_accuracy,
            mean_error=self.mean_error + other.mean_error,
            median_error=self.median_error + other.median_error,
            std_error=self.std_error + other.std_error,
            mean_latency=self.mean_latency + other.mean_latency,
            max_latency=self.max_latency + other.max_latency,
            tempo_robustness=self.tempo_robustness + other.tempo_robustness,
            error_recovery_time=self.error_recovery_time + other.error_recovery_time,
            total_frames=self.total_frames + other.total_frames,
            correct_frames=self.correct_frames + other.correct_frames
        )

    @classmethod
    def avg(cls, metrics: List[Self]) -> Self:
        aggregated_metrics: Self = sum(metrics, start=EvaluationMetrics.empty())
        metrics_len = len(metrics)
        return EvaluationMetrics(
            frame_accuracy=aggregated_metrics.frame_accuracy / metrics_len,
            mean_error=aggregated_metrics.mean_error / metrics_len,
            median_error=aggregated_metrics.median_error / metrics_len,
            std_error=aggregated_metrics.std_error / metrics_len,
            mean_latency=aggregated_metrics.mean_latency / metrics_len,
            max_latency=aggregated_metrics.max_latency / metrics_len,
            tempo_robustness=aggregated_metrics.tempo_robustness / metrics_len,
            error_recovery_time=aggregated_metrics.error_recovery_time / metrics_len,
            total_frames=round(aggregated_metrics.total_frames / metrics_len),
            correct_frames=round(aggregated_metrics.correct_frames / metrics_len)
        )

    @classmethod
    def empty(cls) -> Self:
        return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

class MetricKeys(str, Enum):
    ACCURACY = "frame_accuracy"
    MEAN_ERROR = "mean_error"
    STD_ERROR = "std_error"
    MEAN_LATENCY = "mean_latency"
    TEMPO_ROBUSTNESS = "tempo_robustness"


class MetricsCalculator:
    """
    Kalkulator metryk ewaluacji.
    """
    
    def __init__(self, tolerance_seconds: float = 0.5):
        """
        Args:
            tolerance_seconds: Tolerancja dla uznania ramki za poprawną (domyślnie 0.5s)
        """
        self.tolerance = tolerance_seconds
        
    def calculate_frame_accuracy(self, 
                                 predictions: np.ndarray,
                                 ground_truth: np.ndarray) -> float:
        """
        Oblicza Frame Accuracy - procent ramek zidentyfikowanych poprawnie.
        
        Args:
            predictions: Przewidziane pozycje [N]
            ground_truth: Prawdziwe pozycje [N]
            
        Returns:
            Accuracy w przedziale [0, 1]
        """
        errors = np.abs(predictions - ground_truth)
        correct = errors <= self.tolerance
        return np.mean(correct)
    
    def calculate_error_statistics(self,
                                   predictions: np.ndarray,
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Oblicza statystyki błędów.
        
        Returns:
            Dict ze średnią, medianą i odchyleniem standardowym
        """
        errors = np.abs(predictions - ground_truth)
        return {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors)
        }
    
    def calculate_tempo_robustness(self,
                                   predictions_dict: Dict[float, np.ndarray],
                                   ground_truth_dict: Dict[float, np.ndarray]) -> float:
        """
        Oblicza Tempo Robustness - jak dobrze model radzi sobie przy różnych tempach.
        
        Args:
            predictions_dict: {tempo_ratio: predictions} np. {1.0: [...], 1.1: [...]}
            ground_truth_dict: {tempo_ratio: ground_truth}
            
        Returns:
            Średnia accuracy dla wszystkich temp (0-1)
        """
        accuracies = []
        
        for tempo_ratio, preds in predictions_dict.items():
            if tempo_ratio in ground_truth_dict:
                gt = ground_truth_dict[tempo_ratio]
                acc = self.calculate_frame_accuracy(preds, gt)
                accuracies.append(acc)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def calculate_error_recovery_time(self,
                                     predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     fps: float = 43.0) -> float:
        """
        Oblicza średni czas potrzebny na odzyskanie po błędzie.
        
        Błąd = ramka gdzie error > tolerance
        Odzyskanie = następna ramka gdzie error <= tolerance
        
        Args:
            predictions: Przewidziane pozycje
            ground_truth: Prawdziwe pozycje
            fps: Frames per second (dla konwersji ramek na sekundy)
            
        Returns:
            Średni czas odzyskania w sekundach
        """
        errors = np.abs(predictions - ground_truth)
        is_correct = errors <= self.tolerance
        
        recovery_times = []
        in_error = False
        error_start = 0
        
        for i, correct in enumerate(is_correct):
            if not correct and not in_error:
                # Rozpoczęcie błędu
                in_error = True
                error_start = i
            elif correct and in_error:
                # Odzyskanie po błędzie
                recovery_time = (i - error_start) / fps
                recovery_times.append(recovery_time)
                in_error = False
        
        return np.mean(recovery_times) if recovery_times else 0.0
    
    def calculate_all_metrics(self,
                             predictions: np.ndarray,
                             ground_truth: np.ndarray,
                             latencies: List[float],
                             tempo_predictions: Dict[float, np.ndarray] = None,
                             tempo_ground_truth: Dict[float, np.ndarray] = None,
                             fps: float = 43.0) -> EvaluationMetrics:
        """
        Oblicza wszystkie metryki naraz.
        
        Args:
            predictions: Przewidziane pozycje [N]
            ground_truth: Prawdziwe pozycje [N]
            latencies: Lista latencji w ms [N]
            tempo_predictions: Opcjonalnie - predykcje dla różnych temp
            tempo_ground_truth: Opcjonalnie - ground truth dla różnych temp
            fps: Frames per second
            
        Returns:
            EvaluationMetrics z wszystkimi metrykami
        """
        # Frame accuracy
        frame_acc = self.calculate_frame_accuracy(predictions, ground_truth)
        
        # Error statistics
        error_stats = self.calculate_error_statistics(predictions, ground_truth)
        
        # Latency statistics
        latencies_array = np.array(latencies)
        mean_latency = np.mean(latencies_array)
        max_latency = np.max(latencies_array)
        
        # Tempo robustness
        if tempo_predictions and tempo_ground_truth:
            tempo_rob = self.calculate_tempo_robustness(
                tempo_predictions, tempo_ground_truth
            )
        else:
            tempo_rob = frame_acc  # Fallback
        
        # Error recovery
        recovery_time = self.calculate_error_recovery_time(
            predictions, ground_truth, fps
        )
        
        # Correct frames count
        errors = np.abs(predictions - ground_truth)
        correct_frames = np.sum(errors <= self.tolerance)
        
        return EvaluationMetrics(
            frame_accuracy=frame_acc,
            mean_error=error_stats['mean'],
            median_error=error_stats['median'],
            std_error=error_stats['std'],
            mean_latency=mean_latency,
            max_latency=max_latency,
            tempo_robustness=tempo_rob,
            error_recovery_time=recovery_time,
            total_frames=len(predictions),
            correct_frames=int(correct_frames)
        )


def compare_models(results_dict: Dict[str, EvaluationMetrics]) -> str:
    """
    Porównuje wyniki różnych modeli i tworzy czytelną tabelę.
    
    Args:
        results_dict: {'model_name': EvaluationMetrics}
        
    Returns:
        Sformatowana tabela porównawcza
    """
    # Przygotuj nagłówek
    header = f"{'Model':<20} {'Accuracy':<12} {'Mean Error':<12} {'Latency':<12} {'Tempo Rob.':<12}"
    separator = "=" * 68
    
    lines = [separator, header, separator]
    
    # Dodaj wiersze dla każdego modelu
    for model_name, metrics in results_dict.items():
        line = (f"{model_name:<20} "
                f"{metrics.frame_accuracy:>10.2%}  "
                f"{metrics.mean_error:>10.3f}s  "
                f"{metrics.mean_latency:>10.2f}ms  "
                f"{metrics.tempo_robustness:>10.2%}")
        lines.append(line)
    
    lines.append(separator)
    
    return "\n".join(lines)
