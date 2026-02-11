"""
Narzędzia do przetwarzania plików MIDI.
"""

import numpy as np
import pretty_midi
from typing import Tuple, List, Optional


class MIDIProcessor:
    """
    Klasa do przetwarzania plików MIDI.
    """
    
    def __init__(self, fps: int = 100):
        """
        Args:
            fps: Frames per second dla konwersji MIDI (domyślnie 100 fps)
        """
        self.fps = fps
        
    def load_midi(self, midi_path: str) -> pretty_midi.PrettyMIDI:
        """
        Wczytuje plik MIDI.
        
        Args:
            midi_path: Ścieżka do pliku MIDI
            
        Returns:
            Obiekt PrettyMIDI
        """
        return pretty_midi.PrettyMIDI(midi_path)
    
    def midi_to_piano_roll(self, 
                          midi: pretty_midi.PrettyMIDI,
                          fps: Optional[int] = None) -> np.ndarray:
        """
        Konwertuje MIDI do formatu piano roll.
        Piano roll = macierz [128 nut, time_steps] gdzie 1 = nuta zagrana.
        
        Args:
            midi: Obiekt PrettyMIDI
            fps: Frames per second (None = użyj self.fps)
            
        Returns:
            Piano roll [128, time_steps]
        """
        if fps is None:
            fps = self.fps
            
        # Pobierz piano roll z biblioteki
        piano_roll = midi.get_piano_roll(fs=fps)
        return piano_roll
    
    def get_note_events(self, 
                       midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, int, int]]:
        """
        Ekstrahuje listę wydarzeń nutowych.
        
        Args:
            midi: Obiekt PrettyMIDI
            
        Returns:
            Lista krotek: (time, note_number, velocity)
        """
        events = []
        
        for instrument in midi.instruments:
            for note in instrument.notes:
                events.append((note.start, note.pitch, note.velocity))
        
        # Sortuj po czasie
        events.sort(key=lambda x: x[0])
        return events
    
    def get_tempo_changes(self, 
                         midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, float]]:
        """
        Ekstrahuje zmiany tempa z MIDI.
        
        Args:
            midi: Obiekt PrettyMIDI
            
        Returns:
            Lista krotek: (time, tempo_bpm)
        """
        tempo_changes = midi.get_tempo_changes()
        times = tempo_changes[0]
        tempos = tempo_changes[1]
        
        return list(zip(times, tempos))
    
    def create_ground_truth_alignment(self,
                                     midi: pretty_midi.PrettyMIDI,
                                     audio_duration: float,
                                     fps: Optional[int] = None) -> np.ndarray:
        """
        Tworzy ground truth alignment - mapowanie ramek audio na pozycje MIDI.
        Zakłada liniową synchronizację (bez zmian tempa).
        
        Args:
            midi: Obiekt PrettyMIDI
            audio_duration: Długość audio w sekundach
            fps: Frames per second
            
        Returns:
            Array [n_frames] z pozycjami MIDI dla każdej ramki audio
        """
        if fps is None:
            fps = self.fps
            
        midi_duration = midi.get_end_time()
        n_frames = int(audio_duration * fps)
        
        # Liniowe mapowanie
        ground_truth = np.linspace(0, midi_duration, n_frames)
        return ground_truth
    
    def synthesize_audio(self, 
                        midi: pretty_midi.PrettyMIDI,
                        fs: int = 22050) -> np.ndarray:
        """
        Syntetyzuje audio z MIDI (przydatne do generowania danych testowych).
        
        Args:
            midi: Obiekt PrettyMIDI
            fs: Sample rate
            
        Returns:
            Sygnał audio [n_samples]
        """
        audio = midi.fluidsynth(fs=fs)
        return audio
    
    def extract_chroma_from_midi(self, 
                                midi: pretty_midi.PrettyMIDI,
                                fps: Optional[int] = None) -> np.ndarray:
        """
        Ekstrahuje chromagram z MIDI (12 półtonów).
        Przydatne do porównań z chromagramem z audio.
        
        Args:
            midi: Obiekt PrettyMIDI
            fps: Frames per second
            
        Returns:
            Chromagram [12, time_steps]
        """
        if fps is None:
            fps = self.fps
            
        piano_roll = self.midi_to_piano_roll(midi, fps)
        
        # Zredukuj 128 nut do 12 klas chromatycznych (modulo 12)
        chroma = np.zeros((12, piano_roll.shape[1]))
        for pitch in range(128):
            chroma_class = pitch % 12
            chroma[chroma_class] += piano_roll[pitch]
        
        # Normalizacja
        chroma = chroma / (np.max(chroma) + 1e-8)
        return chroma
    
    def get_duration(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Pobiera długość utworu MIDI w sekundach.
        
        Args:
            midi: Obiekt PrettyMIDI
            
        Returns:
            Długość w sekundach
        """
        return midi.get_end_time()
    
    def slice_midi(self, 
                   midi: pretty_midi.PrettyMIDI,
                   start_time: float,
                   end_time: float) -> pretty_midi.PrettyMIDI:
        """
        Wycina fragment MIDI.
        
        Args:
            midi: Obiekt PrettyMIDI
            start_time: Początek fragmentu (sekundy)
            end_time: Koniec fragmentu (sekundy)
            
        Returns:
            Nowy obiekt PrettyMIDI z fragmentem
        """
        # Stwórz nowy MIDI
        sliced_midi = pretty_midi.PrettyMIDI()
        
        for instrument in midi.instruments:
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )
            
            # Skopiuj nuty w zakresie czasowym
            for note in instrument.notes:
                if start_time <= note.start <= end_time:
                    # Przesuń czas na początek
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start - start_time,
                        end=note.end - start_time
                    )
                    new_instrument.notes.append(new_note)
            
            sliced_midi.instruments.append(new_instrument)
        
        return sliced_midi


def align_audio_midi(audio_duration: float,
                    midi_duration: float,
                    n_frames: int) -> np.ndarray:
    """
    Tworzy podstawowe wyrównanie audio-MIDI.
    
    Args:
        audio_duration: Długość audio (s)
        midi_duration: Długość MIDI (s)
        n_frames: Liczba ramek audio
        
    Returns:
        Mapowanie [n_frames] -> pozycja w MIDI
    """
    # Liniowe mapowanie
    return np.linspace(0, midi_duration, n_frames)


# Przykładowe użycie
if __name__ == "__main__":
    processor = MIDIProcessor(fps=100)
    
    # Przykład (zakomentowane - wymaga pliku MIDI)
    # midi = processor.load_midi("example.mid")
    # piano_roll = processor.midi_to_piano_roll(midi)
    # chroma = processor.extract_chroma_from_midi(midi)
    # duration = processor.get_duration(midi)
    
    print("MIDIProcessor ready to use!")
    print(f"Default FPS: {processor.fps}")
