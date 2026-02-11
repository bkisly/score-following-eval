# Code Snippets - Przydatne Fragmenty Kodu

Ten plik zawiera gotowe snippety do kopiowania w Twoich eksperymentach.

## 1. Generowanie Syntetycznych Danych

```python
# Wygeneruj audio z MIDI (przydatne do testów)
from utils.midi_processing import MIDIProcessor
import soundfile as sf

processor = MIDIProcessor()
midi = processor.load_midi("reference.mid")
audio = processor.synthesize_audio(midi, fs=22050)

# Zapisz jako WAV
sf.write("synthetic_audio.wav", audio, 22050)
print(f"Generated audio: {len(audio)/22050:.2f} seconds")
```

## 2. Batch Processing - Ewaluacja Wielu Plików

```python
from pathlib import Path
from evaluation.evaluator import Evaluator
from models.dtw_model import DTWModel
import pandas as pd

# Przygotuj listę plików
data_dir = Path("data/maestro/")
test_files = [
    (data_dir / "audio1.wav", data_dir / "midi1.mid"),
    (data_dir / "audio2.wav", data_dir / "midi2.mid"),
    # ... więcej
]

# Stwórz model i ewaluator
model = DTWModel()
evaluator = Evaluator()

# Ewaluuj wszystkie
all_results = []
for audio_path, midi_path in test_files:
    print(f"\nProcessing: {audio_path.name}")
    
    metrics = evaluator.evaluate_single_model(
        model=model,
        audio_path=str(audio_path),
        reference_path=str(midi_path),
        verbose=False
    )
    
    all_results.append({
        'file': audio_path.stem,
        'accuracy': metrics.frame_accuracy,
        'error': metrics.mean_error,
        'latency': metrics.mean_latency
    })

# Stwórz DataFrame z wynikami
df = pd.DataFrame(all_results)
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(df.describe())

# Zapisz do CSV
df.to_csv('batch_results.csv', index=False)
```

## 3. Wizualizacja Predykcji vs Ground Truth

```python
import matplotlib.pyplot as plt
import numpy as np
from evaluation.evaluator import Evaluator
from models.dtw_model import DTWModel
from utils.audio_processing import AudioProcessor, simulate_real_time_input
from utils.midi_processing import MIDIProcessor

# Przygotuj dane
audio_processor = AudioProcessor()
midi_processor = MIDIProcessor()

audio, sr = audio_processor.load_audio("audio.wav")
midi = midi_processor.load_midi("reference.mid")

# Ground truth
midi_duration = midi_processor.get_duration(midi)
n_frames = len(audio) // audio_processor.hop_length
ground_truth = np.linspace(0, midi_duration, n_frames)

# Predykcje
model = DTWModel()
model.load_reference("reference.mid")
model.reset()

chunks = simulate_real_time_input(audio, chunk_size=2048)
predictions = []

for chunk in chunks:
    result = model.process_frame(chunk, sr)
    predictions.append(result['position'])

# Dopasuj długości
min_len = min(len(predictions), len(ground_truth))
predictions = predictions[:min_len]
ground_truth_cut = ground_truth[:min_len]

# Wykres
time_axis = np.arange(min_len) / (sr / audio_processor.hop_length)

plt.figure(figsize=(14, 6))

# Predykcje vs Ground Truth
plt.subplot(2, 1, 1)
plt.plot(time_axis, ground_truth_cut, 'g-', label='Ground Truth', linewidth=2)
plt.plot(time_axis, predictions, 'r--', label='Predictions', linewidth=1.5, alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Position in Score (s)')
plt.title('Predictions vs Ground Truth')
plt.legend()
plt.grid(True, alpha=0.3)

# Błąd w czasie
plt.subplot(2, 1, 2)
errors = np.abs(np.array(predictions) - ground_truth_cut)
plt.plot(time_axis, errors, 'b-', linewidth=1)
plt.axhline(y=0.5, color='r', linestyle='--', label='Tolerance (0.5s)')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Error (s)')
plt.title('Prediction Error Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=300)
plt.show()

print(f"Mean error: {np.mean(errors):.3f}s")
print(f"Max error: {np.max(errors):.3f}s")
print(f"Frames within tolerance: {np.sum(errors <= 0.5)/len(errors):.2%}")
```

## 4. Porównanie Features (Chroma vs MFCC vs Mel)

```python
from models.dtw_model import DTWModel
from evaluation.evaluator import Evaluator

# Testuj różne features
features = ['chroma', 'mfcc', 'mel']
results = {}

evaluator = Evaluator()

for feature_type in features:
    print(f"\nTesting with {feature_type} features...")
    
    model = DTWModel(
        window_size=100,
        feature_type=feature_type
    )
    
    metrics = evaluator.evaluate_single_model(
        model=model,
        audio_path="audio.wav",
        reference_path="reference.mid",
        verbose=False
    )
    
    results[feature_type] = metrics

# Porównaj
print("\n" + "="*60)
print("FEATURE COMPARISON")
print("="*60)

for feature, metrics in results.items():
    print(f"\n{feature.upper()}:")
    print(f"  Accuracy: {metrics.frame_accuracy:.2%}")
    print(f"  Error: {metrics.mean_error:.3f}s")
    print(f"  Latency: {metrics.mean_latency:.2f}ms")

# Znajdź najlepszy
best = max(results.items(), key=lambda x: x[1].frame_accuracy)
print(f"\nBest feature type: {best[0]}")
```

## 5. Custom Test Case - Własne Ground Truth

```python
# Jeśli masz ręcznie zaznaczone pozycje (np. co sekundę)
import numpy as np

# Przykład: pozycje zaznaczone ręcznie
manual_annotations = [
    (0.0, 0.0),    # (czas_audio, pozycja_w_midi)
    (1.0, 0.95),
    (2.0, 2.1),
    (3.0, 3.0),
    (5.0, 4.8),
    # ... więcej
]

# Interpoluj do pełnego ground truth
audio_times = [a[0] for a in manual_annotations]
midi_positions = [a[1] for a in manual_annotations]

# Stwórz gęsty ground truth (co ramkę)
from scipy.interpolate import interp1d

# Interpolacja liniowa
interpolator = interp1d(
    audio_times, 
    midi_positions, 
    kind='linear',
    fill_value='extrapolate'
)

# Generuj dla każdej ramki
fps = 43  # Frames per second
total_duration = max(audio_times)
n_frames = int(total_duration * fps)

frame_times = np.linspace(0, total_duration, n_frames)
ground_truth = interpolator(frame_times)

# Teraz możesz użyć tego ground truth w ewaluacji
evaluator = Evaluator()
metrics = evaluator.evaluate_single_model(
    model=model,
    audio_path="audio.wav",
    reference_path="reference.mid",
    ground_truth_alignment=ground_truth,
    verbose=True
)
```

## 6. Eksport Wyników do LaTeX (do pracy)

```python
import pandas as pd

# Zbierz wyniki
results_data = {
    'Model': ['DTW', 'OTW', 'CNN', 'CYOLO'],
    'Accuracy (\\%)': [85.3, 87.1, 92.4, 94.2],
    'Error (s)': [0.42, 0.38, 0.21, 0.18],
    'Latency (ms)': [95, 82, 105, 120]
}

df = pd.DataFrame(results_data)

# Eksport do LaTeX
latex_table = df.to_latex(
    index=False,
    float_format="%.2f",
    caption="Comparison of score following methods",
    label="tab:results"
)

print(latex_table)

# Zapisz do pliku
with open('results_table.tex', 'w') as f:
    f.write(latex_table)
```

## 7. Logowanie Eksperymentów

```python
import json
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_dir="experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = {
            'timestamp': datetime.now().isoformat(),
            'experiments': []
        }
    
    def log_experiment(self, model_name, params, results):
        """Log pojedynczy eksperyment"""
        self.current_log['experiments'].append({
            'model': model_name,
            'params': params,
            'results': results,
            'time': datetime.now().isoformat()
        })
    
    def save(self, name=None):
        """Zapisz log"""
        if name is None:
            name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        path = self.log_dir / name
        with open(path, 'w') as f:
            json.dump(self.current_log, f, indent=2)
        
        print(f"Log saved to: {path}")

# Użycie
logger = ExperimentLogger()

# Eksperyment 1
logger.log_experiment(
    model_name='DTW',
    params={'window_size': 100, 'feature': 'chroma'},
    results={'accuracy': 0.85, 'error': 0.42}
)

# Eksperyment 2
logger.log_experiment(
    model_name='OTW',
    params={'window_size': 100, 'search_margin': 30},
    results={'accuracy': 0.87, 'error': 0.38}
)

# Zapisz
logger.save()
```

## 8. Debugging - Wizualizacja Features

```python
from utils.audio_processing import AudioProcessor
from utils.midi_processing import MIDIProcessor
import matplotlib.pyplot as plt

# Wczytaj audio i MIDI
audio_proc = AudioProcessor()
midi_proc = MIDIProcessor()

audio, sr = audio_proc.load_audio("audio.wav")
midi = midi_proc.load_midi("reference.mid")

# Ekstrahuj features
audio_chroma = audio_proc.compute_chroma(audio)
midi_chroma = midi_proc.extract_chroma_from_midi(midi)

# Wizualizuj obok siebie
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Audio chromagram
im1 = axes[0].imshow(
    audio_chroma, 
    aspect='auto', 
    origin='lower',
    cmap='viridis',
    interpolation='nearest'
)
axes[0].set_ylabel('Pitch Class')
axes[0].set_xlabel('Time (frames)')
axes[0].set_title('Audio Chromagram')
axes[0].set_yticks(range(12))
axes[0].set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 
                          'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.colorbar(im1, ax=axes[0])

# MIDI chromagram
im2 = axes[1].imshow(
    midi_chroma,
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='nearest'
)
axes[1].set_ylabel('Pitch Class')
axes[1].set_xlabel('Time (frames)')
axes[1].set_title('MIDI Chromagram')
axes[1].set_yticks(range(12))
axes[1].set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('feature_comparison.png', dpi=300)
plt.show()

print("✓ Features look similar = good alignment")
print("✗ Features look different = check synchronization")
```

## 9. Profilowanie Wydajności

```python
import time
import cProfile
import pstats
from io import StringIO

def profile_model(model, audio_path, reference_path):
    """Profiluj wydajność modelu"""
    
    # Przygotuj
    from utils.audio_processing import simulate_real_time_input, AudioProcessor
    
    processor = AudioProcessor()
    audio, sr = processor.load_audio(audio_path)
    chunks = simulate_real_time_input(audio, chunk_size=2048)
    
    model.load_reference(reference_path)
    model.reset()
    
    # Profiluj
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    for chunk in chunks[:100]:  # Pierwszych 100 chunków
        model.process_frame(chunk, sr)
    elapsed = time.time() - start_time
    
    pr.disable()
    
    # Wyniki
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20
    
    print(f"\nProcessed 100 chunks in {elapsed:.2f}s")
    print(f"Average per chunk: {elapsed/100*1000:.2f}ms")
    print("\nTop time consumers:")
    print(s.getvalue())

# Użycie
from models.dtw_model import DTWModel

model = DTWModel()
profile_model(model, "audio.wav", "reference.mid")
```

## 10. Grid Search dla Hiperparametrów

```python
from itertools import product
from evaluation.evaluator import Evaluator
from models.dtw_model import DTWModel

# Parametry do testowania
window_sizes = [50, 100, 150, 200]
feature_types = ['chroma', 'mfcc', 'mel']

evaluator = Evaluator()
results = []

# Grid search
for window, feature in product(window_sizes, feature_types):
    print(f"\nTesting: window={window}, feature={feature}")
    
    model = DTWModel(
        window_size=window,
        feature_type=feature
    )
    
    metrics = evaluator.evaluate_single_model(
        model=model,
        audio_path="audio.wav",
        reference_path="reference.mid",
        verbose=False
    )
    
    results.append({
        'window_size': window,
        'feature_type': feature,
        'accuracy': metrics.frame_accuracy,
        'error': metrics.mean_error,
        'latency': metrics.mean_latency
    })

# Znajdź najlepszą konfigurację
import pandas as pd
df = pd.DataFrame(results)
df_sorted = df.sort_values('accuracy', ascending=False)

print("\n" + "="*60)
print("BEST CONFIGURATIONS")
print("="*60)
print(df_sorted.head(5))

# Zapisz
df.to_csv('hyperparameter_search.csv', index=False)
```

---

## Dodatkowe Noty

- Wszystkie snippety zakładają że jesteś w głównym katalogu projektu
- Pamiętaj o zmianie ścieżek do plików na swoje
- Dla większych eksperymentów rozważ użycie narzędzi jak MLflow lub Weights & Biases
