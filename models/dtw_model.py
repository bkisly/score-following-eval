"""
Implementacja modelu DTW/OTW (Online Time Warping) jako baseline.
Ten model nie wymaga treningu i może działać od razu.
"""

import numpy as np
from typing import Dict, Any
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import librosa

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BaseScoreFollower
from utils.audio_processing import AudioProcessor
from utils.midi_processing import MIDIProcessor


class DTWModel(BaseScoreFollower):
    """
    Model bazujący na Dynamic Time Warping.
    
    Algorytm:
    1. Ekstrahuj chromagram z referencyjnego MIDI
    2. Dla każdej ramki audio:
       - Ekstrahuj chromagram z okna audio
       - Dopasuj do referencji używając DTW
       - Zwróć najlepsze dopasowanie jako pozycję
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 hop_length: int = 512,
                 feature_type: str = 'chroma'):
        """
        Args:
            window_size: Rozmiar okna w ramkach (dla OTW)
            hop_length: Przesunięcie między ramkami
            feature_type: 'chroma' lub 'mfcc' lub 'mel'
        """
        super().__init__(name="DTW-OTW")
        
        self.window_size = window_size
        self.hop_length = hop_length
        self.feature_type = feature_type
        
        # Procesory
        self.audio_processor = AudioProcessor(hop_length=hop_length)
        self.midi_processor = MIDIProcessor()
        
        # Stan
        self.reference_features = None
        self.reference_times = None
        self.audio_buffer = []  # Bufor dla okna czasowego
        
    def load_reference(self, reference_path: str) -> None:
        """
        Wczytuje MIDI referencyjny i ekstrahuje features.
        
        Args:
            reference_path: Ścieżka do pliku MIDI
        """
        print(f"Loading reference MIDI: {reference_path}")
        
        # Wczytaj MIDI
        midi = self.midi_processor.load_midi(reference_path)
        
        # Syntetyzuj audio z MIDI (żeby mieć wspólną reprezentację)
        reference_audio = self.midi_processor.synthesize_audio(midi, fs=22050)
        
        # Ekstrahuj features
        if self.feature_type == 'chroma':
            self.reference_features = self.audio_processor.compute_chroma(reference_audio)
        elif self.feature_type == 'mfcc':
            self.reference_features = self.audio_processor.compute_mfcc(reference_audio)
        elif self.feature_type == 'mel':
            self.reference_features = self.audio_processor.compute_spectrogram(reference_audio)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        # Zapisz timing
        n_frames = self.reference_features.shape[1]
        self.reference_times = np.arange(n_frames) * self.hop_length / 22050
        
        print(f"Reference loaded: {n_frames} frames, duration: {self.reference_times[-1]:.2f}s")
        
        self.reference_score = reference_audio  # Zapisz dla zgodności z API
    
    def _extract_features_from_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Ekstrahuje features z pojedynczej ramki audio.
        
        Args:
            audio_frame: Fragment audio
            
        Returns:
            Feature vector
        """
        # Dodaj do bufora
        self.audio_buffer.append(audio_frame)
        
        # Utrzymuj rozmiar bufora
        if len(self.audio_buffer) > self.window_size:
            self.audio_buffer.pop(0)
        
        # Połącz bufor w jeden sygnał
        window_audio = np.concatenate(self.audio_buffer)
        
        # Ekstrahuj features
        if self.feature_type == 'chroma':
            features = self.audio_processor.compute_chroma(window_audio)
        elif self.feature_type == 'mfcc':
            features = self.audio_processor.compute_mfcc(window_audio)
        elif self.feature_type == 'mel':
            features = self.audio_processor.compute_spectrogram(window_audio)
        
        return features
    
    def process_frame(self, 
                     audio_frame: np.ndarray,
                     sample_rate: int) -> Dict[str, Any]:
        """
        Przetwarza pojedynczą ramkę używając Online Time Warping.
        
        Args:
            audio_frame: Fragment audio
            sample_rate: Sample rate
            
        Returns:
            Dict z predykcją pozycji
        """
        import time
        start_time = time.time()
        
        if self.reference_features is None:
            raise ValueError("Reference not loaded! Call load_reference() first.")
        
        # Ekstrahuj features z okna
        query_features = self._extract_features_from_frame(audio_frame)
        
        # DTW alignment
        # Używamy fastdtw dla szybkości
        distance, path = fastdtw(
            query_features.T,  # [time, features]
            self.reference_features.T,
            dist=euclidean
        )
        
        # Znajdź najlepsze dopasowanie
        # path to lista krotek (query_idx, ref_idx)
        if len(path) > 0:
            # Weź ostatni punkt dopasowania jako aktualną pozycję
            _, ref_idx = path[-1]
            predicted_time = self.reference_times[min(ref_idx, len(self.reference_times)-1)]
            
            # Aktualizuj pozycję
            self.current_position = predicted_time
            
            # Oblicz confidence (im mniejsza odległość, tym lepiej)
            # Znormalizuj do [0, 1]
            max_distance = np.sqrt(query_features.shape[0] * query_features.shape[1])
            confidence = max(0, 1 - distance / max_distance)
        else:
            predicted_time = self.current_position
            confidence = 0.0
        
        # Oblicz latencję
        latency = (time.time() - start_time) * 1000  # ms
        
        # Estymuj tempo (uproszczone - zwróć stałe tempo)
        tempo = 120.0  # BPM - można to ulepszyć
        
        return {
            'position': predicted_time,
            'confidence': confidence,
            'tempo': tempo,
            'latency': latency
        }
    
    def reset(self) -> None:
        """
        Resetuje stan modelu.
        """
        self.current_position = 0
        self.audio_buffer = []
        print("DTW model reset.")
    
    def requires_training(self) -> bool:
        """
        DTW nie wymaga treningu.
        """
        return False


class OnlineTimeWarping(DTWModel):
    """
    Wariant Online Time Warping z dodatkowymi optymalizacjami.
    
    Różnice od zwykłego DTW:
    - Analizuje tylko okno czasowe zamiast całego utworu
    - Używa kontekstu z poprzednich ramek
    - Szybsze dla real-time processing
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 search_margin: int = 50,
                 **kwargs):
        """
        Args:
            window_size: Rozmiar okna audio
            search_margin: Margines wyszukiwania wokół ostatniej pozycji
        """
        super().__init__(window_size=window_size, **kwargs)
        self.name = "OTW (Online Time Warping)"
        self.search_margin = search_margin
        
    def process_frame(self,
                     audio_frame: np.ndarray,
                     sample_rate: int) -> Dict[str, Any]:
        """
        OTW - szuka tylko w okolicy ostatniej znanej pozycji.
        """
        import time
        start_time = time.time()
        
        if self.reference_features is None:
            raise ValueError("Reference not loaded!")
        
        # Ekstrahuj features
        query_features = self._extract_features_from_frame(audio_frame)
        
        # Określ zakres wyszukiwania wokół ostatniej pozycji
        current_frame = int(self.current_position / (self.hop_length / 22050))
        search_start = max(0, current_frame - self.search_margin)
        search_end = min(
            self.reference_features.shape[1],
            current_frame + self.search_margin
        )
        
        # DTW tylko w tym zakresie
        ref_slice = self.reference_features[:, search_start:search_end]
        
        if ref_slice.shape[1] > 0:
            distance, path = fastdtw(
                query_features.T,
                ref_slice.T,
                dist=euclidean
            )
            
            if len(path) > 0:
                _, ref_idx = path[-1]
                # Dodaj offset od początku referencji
                absolute_idx = search_start + ref_idx
                predicted_time = self.reference_times[min(absolute_idx, len(self.reference_times)-1)]
                
                self.current_position = predicted_time
                
                max_distance = np.sqrt(query_features.shape[0] * query_features.shape[1])
                confidence = max(0, 1 - distance / max_distance)
            else:
                predicted_time = self.current_position
                confidence = 0.0
        else:
            predicted_time = self.current_position
            confidence = 0.0
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'position': predicted_time,
            'confidence': confidence,
            'tempo': 120.0,
            'latency': latency
        }


# Przykład użycia
if __name__ == "__main__":
    # Test DTW model
    print("Testing DTW Model...")
    
    model = DTWModel(window_size=100)
    print(f"Model: {model.name}")
    print(f"Requires training: {model.requires_training()}")
    
    # OTW model
    otw = OnlineTimeWarping(window_size=100, search_margin=30)
    print(f"\nModel: {otw.name}")
    print(f"Search margin: {otw.search_margin} frames")
