"""
Abstrakcyjna klasa bazowa dla wszystkich modeli śledzenia partytury.
Implementuje wzorzec adaptera - każdy model musi zaimplementować te same metody.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import time


class ScoreFollower(ABC):
    """
    Klasa bazowa dla wszystkich systemów śledzenia partytury.
    
    Każdy model (DTW, CYOLO, CNN) musi dziedziczyć po tej klasie
    i zaimplementować wymagane metody.
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Nazwa modelu (np. "DTW", "CYOLO", "CNN-HeurMiT")
        """
        self.name = name
        self.is_trained = False
        self.reference_score = None  # Partytura referencyjna
        self.current_position = 0     # Aktualna pozycja w utworze

    def __eq__(self, other):
        return self.name == other.name
        
    @abstractmethod
    def load_reference(self, reference_path: str) -> None:
        """
        Wczytuje partyturę referencyjną (MIDI, spektrogram, itp.)
        
        Args:
            reference_path: Ścieżka do pliku referencyjnego
        """
        pass
    
    @abstractmethod
    def process_frame(self, audio_frame: np.ndarray, 
                      sample_rate: int) -> Dict[str, Any]:
        """
        Przetwarza pojedynczą ramkę audio i zwraca predykcję pozycji.
        To jest główna metoda wywoływana w czasie rzeczywistym.
        
        Args:
            audio_frame: Fragment audio (numpy array)
            sample_rate: Częstotliwość próbkowania
            
        Returns:
            Dict zawierający:
                - 'position': pozycja w utworze (w sekundach lub ramkach MIDI)
                - 'confidence': pewność predykcji (0-1)
                - 'tempo': estymowane tempo (BPM)
                - 'latency': czas przetwarzania (ms)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resetuje stan modelu (np. przed rozpoczęciem nowego utworu).
        """
        pass
    
    def train(self, train_data: Any) -> None:
        """
        Trenuje model (tylko dla modeli ML).
        Dla DTW/OTW ta metoda nie robi nic.
        
        Args:
            train_data: Dane treningowe
        """
        # Domyślnie nic nie robi (DTW nie wymaga treningu)
        self.is_trained = True
        
    def requires_training(self) -> bool:
        """
        Sprawdza czy model wymaga treningu.
        
        Returns:
            True jeśli model wymaga treningu (CNN, CYOLO)
            False jeśli nie wymaga (DTW)
        """
        return False
    
    def evaluate_frame(self, audio_frame: np.ndarray, 
                       ground_truth_position: float,
                       sample_rate: int) -> Dict[str, float]:
        """
        Ewaluuje pojedynczą ramkę i porównuje z ground truth.
        
        Args:
            audio_frame: Fragment audio
            ground_truth_position: Prawdziwa pozycja w utworze
            sample_rate: Częstotliwość próbkowania
            
        Returns:
            Dict z metrykami dla tej ramki:
                - 'error': błąd w sekundach
                - 'latency': opóźnienie w ms
                - 'correct': czy predykcja była poprawna (w tolerancji)
        """
        start_time = time.time()
        
        # Predykcja modelu
        prediction = self.process_frame(audio_frame, sample_rate)
        
        # Oblicz metryki
        latency = (time.time() - start_time) * 1000  # ms
        predicted_position = prediction['position']
        error = abs(predicted_position - ground_truth_position)
        
        # Tolerancja: ±0.5 sekundy
        TOLERANCE = 0.5
        correct = error <= TOLERANCE
        
        return {
            'error': error,
            'latency': latency,
            'correct': correct,
            'predicted_position': predicted_position,
            'confidence': prediction.get('confidence', 0.0)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Zwraca informacje o modelu.
        
        Returns:
            Dict z metadanymi modelu
        """
        return {
            'name': self.name,
            'requires_training': self.requires_training(),
            'is_trained': self.is_trained
        }


class ScoreFollowerAdapter:
    """
    Adapter Pattern - umożliwia jednolite używanie różnych modeli.
    
    Przykład użycia:
        adapter = ScoreFollowerAdapter(DTWModel())
        results = adapter.follow_audio(audio, reference, ground_truth)
    """
    
    def __init__(self, model: ScoreFollower):
        """
        Args:
            model: Instancja modelu dziedziczącego po BaseScoreFollower
        """
        self.model = model
        
    def follow_audio(self, audio_path: str, 
                     reference_path: str,
                     ground_truth: Optional[np.ndarray] = None,
                     frame_size: int = 2048,
                     hop_length: int = 512) -> Dict[str, Any]:
        """
        Śledzi cały utwór audio krok po kroku.
        
        Args:
            audio_path: Ścieżka do pliku audio
            reference_path: Ścieżka do referencji (MIDI)
            ground_truth: Opcjonalnie - prawdziwe pozycje dla ewaluacji
            frame_size: Rozmiar ramki audio
            hop_length: Przesunięcie między ramkami
            
        Returns:
            Dict z wynikami śledzenia
        """
        import librosa
        
        # Wczytaj audio
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Wczytaj referencję
        self.model.load_reference(reference_path)
        self.model.reset()
        
        # Przetwarzaj ramka po ramce
        predictions = []
        errors = []
        latencies = []
        
        for i in range(0, len(audio) - frame_size, hop_length):
            frame = audio[i:i+frame_size]
            
            if ground_truth is not None and i < len(ground_truth):
                # Ewaluacja z ground truth
                result = self.model.evaluate_frame(
                    frame, ground_truth[i], sr
                )
                errors.append(result['error'])
            else:
                # Tylko predykcja
                result = self.model.process_frame(frame, sr)
                
            predictions.append(result)
            latencies.append(result.get('latency', 0))
        
        # Agreguj wyniki
        return {
            'predictions': predictions,
            'mean_error': np.mean(errors) if errors else None,
            'mean_latency': np.mean(latencies),
            'accuracy': np.mean([p.get('correct', False) for p in predictions])
        }
