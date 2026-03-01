"""
Narzędzia do przetwarzania audio dla systemów śledzenia partytury.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
#import pretty_midi


class AudioProcessor:
    """
    Klasa do przetwarzania sygnałów audio.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Args:
            sample_rate: Częstotliwość próbkowania
            n_fft: Rozmiar FFT
            hop_length: Przesunięcie między ramkami
            n_mels: Liczba pasm mel
        """
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Wczytuje plik audio.
        
        Args:
            audio_path: Ścieżka do pliku audio
            
        Returns:
            (audio_signal, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sr)
        trimmed_audio, _ = librosa.effects.trim(audio)
        return trimmed_audio, sr
    
    def compute_spectrogram(self, 
                           audio: np.ndarray,
                           use_mel: bool = True) -> np.ndarray:
        """
        Oblicza spektrogram.
        
        Args:
            audio: Sygnał audio
            use_mel: Czy użyć mel-spektrogramu (True) czy zwykłego (False)
            
        Returns:
            Spektrogram [freq_bins, time_frames]
        """
        if use_mel:
            # Mel-spektrogram (lepszy dla muzyki)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            # Konwersja do dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        else:
            # Zwykły spektrogram STFT
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            return spec_db
    
    def compute_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Oblicza chromagram - przydatny dla śledzenia wysokości dźwięków.
        
        Args:
            audio: Sygnał audio
            
        Returns:
            Chromagram [12, time_frames] - 12 półtonów
        """
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )
        return chroma
    
    def compute_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Oblicza MFCC (Mel-Frequency Cepstral Coefficients).
        
        Args:
            audio: Sygnał audio
            n_mfcc: Liczba współczynników MFCC
            
        Returns:
            MFCC features [n_mfcc, time_frames]
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_features(self, 
                        audio: np.ndarray,
                        features: list = ['mel', 'chroma']) -> dict:
        """
        Ekstraktuje wiele feature'ów naraz.
        
        Args:
            audio: Sygnał audio
            features: Lista feature'ów do ekstrahowania 
                     ['mel', 'chroma', 'mfcc', 'spec']
                     
        Returns:
            Dict z feature'ami: {'mel': array, 'chroma': array, ...}
        """
        result = {}
        
        if 'mel' in features:
            result['mel'] = self.compute_spectrogram(audio, use_mel=True)
        if 'spec' in features:
            result['spec'] = self.compute_spectrogram(audio, use_mel=False)
        if 'chroma' in features:
            result['chroma'] = self.compute_chroma(audio)
        if 'mfcc' in features:
            result['mfcc'] = self.compute_mfcc(audio)
            
        return result
    
    def time_stretch(self, 
                     audio: np.ndarray, 
                     rate: float) -> np.ndarray:
        """
        Zmienia tempo audio (do testowania tempo robustness).
        
        Args:
            audio: Sygnał audio
            rate: Współczynnik zmiany (1.0 = bez zmiany, 1.2 = 20% szybciej)
            
        Returns:
            Audio z zmienionym tempem
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_noise(self, 
                  audio: np.ndarray, 
                  noise_factor: float = 0.005) -> np.ndarray:
        """
        Dodaje szum do audio (do testowania robustness).
        
        Args:
            audio: Sygnał audio
            noise_factor: Siła szumu (0.005 = delikatny szum)
            
        Returns:
            Audio z szumem
        """
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented
    
    def frames_to_time(self, frame_idx: int) -> float:
        """
        Konwertuje indeks ramki na czas w sekundach.
        
        Args:
            frame_idx: Indeks ramki
            
        Returns:
            Czas w sekundach
        """
        return librosa.frames_to_time(
            frame_idx, 
            sr=self.sr, 
            hop_length=self.hop_length
        )
    
    def time_to_frames(self, time_seconds: float) -> int:
        """
        Konwertuje czas na indeks ramki.
        
        Args:
            time_seconds: Czas w sekundach
            
        Returns:
            Indeks ramki
        """
        return librosa.time_to_frames(
            time_seconds,
            sr=self.sr,
            hop_length=self.hop_length
        )


def create_sliding_window(audio: np.ndarray,
                          window_size: int = 2048,
                          hop_length: int = 512) -> np.ndarray:
    """
    Tworzy sliding window z audio (dla online processing).
    
    Args:
        audio: Sygnał audio [N_samples]
        window_size: Rozmiar okna
        hop_length: Przesunięcie między oknami
        
    Returns:
        Windows [N_windows, window_size]
    """
    n_windows = (len(audio) - window_size) // hop_length + 1
    windows = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        start = i * hop_length
        windows[i] = audio[start:start + window_size]
    
    return windows


def simulate_real_time_input(audio: np.ndarray,
                             chunk_size: int = 2048,
                             sr: int = 22050) -> list:
    """
    Symuluje strumień audio w czasie rzeczywistym.
    Przydatne do testowania latencji.
    
    Args:
        audio: Pełny sygnał audio
        chunk_size: Rozmiar chunka (ramki)
        sr: Sample rate
        
    Returns:
        Lista chunków audio
    """
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) == chunk_size:  # Pełne chunki
            chunks.append(chunk)
    return chunks


# Przykładowe użycie
if __name__ == "__main__":
    # Przykład użycia
    processor = AudioProcessor(sample_rate=22050)
    
    # Wczytaj audio (zakomentowane - wymaga pliku)
    # audio, sr = processor.load_audio("example.wav")
    
    # Ekstrahuj features
    # features = processor.extract_features(audio, ['mel', 'chroma'])
    # mel_spec = features['mel']
    # chroma = features['chroma']
    
    print("AudioProcessor ready to use!")
    print(f"Sample rate: {processor.sr}")
    print(f"Hop length: {processor.hop_length}")
