# Przewodnik Krok Po Kroku: Platforma Badawcza

Ten dokument przeprowadzi Cię przez cały proces od instalacji do pierwszych eksperymentów.

## 📋 Spis Treści

1. [Instalacja i Setup](#instalacja)
2. [Zrozumienie Struktury Projektu](#struktura)
3. [Pierwsze Uruchomienie](#pierwsze-uruchomienie)
4. [Ewaluacja Modeli](#ewaluacja)
5. [Dalsze Kroki](#dalsze-kroki)

---

## 1. Instalacja i Setup {#instalacja}

### Krok 1.1: Wymagania

- Python 3.8 lub nowszy
- GPU (opcjonalnie, ale zalecane dla modeli ML)
- 8GB RAM minimum
- Około 5GB wolnego miejsca na dysku

### Krok 1.2: Instalacja Zależności

```bash
# W katalogu projektu
pip install -r requirements.txt
```

**Czas trwania**: ~5-10 minut (zależy od internetu)

**Co się dzieje**: Instalowane są biblioteki do:
- Przetwarzania audio (librosa, soundfile)
- Deep learning (PyTorch)
- DTW (fastdtw)
- MIDI (pretty_midi)
- Wizualizacji (matplotlib, seaborn)

### Krok 1.3: Sprawdź Instalację

```python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import librosa; print('Librosa:', librosa.__version__)"
```

Jeśli nie ma błędów - wszystko działa! ✓

### Krok 1.4: Pobierz Dane Testowe

Możesz użyć:

**Opcja A: MAESTRO Dataset (zalecane)**
```bash
# Pobierz z https://magenta.tensorflow.org/datasets/maestro
# Wybierz małą część (1-2 utwory) na początek
```

**Opcja B: Własne pliki**
- Nagraj prostą melodię na fortepianie
- Stwórz odpowiadający plik MIDI
- Lub użyj syntetycznego audio z MIDI (pokazane w kodzie)

**Opcja C: Generuj syntetyczne dane**
```python
from utils.midi_processing import MIDIProcessor

processor = MIDIProcessor()
midi = processor.load_midi("twoj_plik.mid")
audio = processor.synthesize_audio(midi)

import soundfile as sf
sf.write("syntetyczny.wav", audio, 22050)
```

---

## 2. Zrozumienie Struktury Projektu {#struktura}

```
score_following_platform/
│
├── models/               ← Tu są modele
│   ├── base_model.py    ← Abstrakcyjna klasa (wzorzec adaptera)
│   ├── dtw_model.py     ← DTW i OTW (działa od razu!)
│   ├── cnn_model.py     ← CNN HeurMiT (wymaga treningu)
│   └── cyolo_model.py   ← CYOLO (wymaga treningu)
│
├── utils/               ← Narzędzia pomocnicze
│   ├── audio_processing.py  ← Spektrogramy, MFCC, itp.
│   ├── midi_processing.py   ← Obsługa MIDI
│   └── metrics.py           ← Metryki ewaluacji
│
├── evaluation/          ← System ewaluacji
│   └── evaluator.py     ← Główny ewaluator
│
├── experiments/         ← Skrypty do uruchamiania
│   └── run_comparison.py
│
└── notebooks/           ← Jupyter notebooks
    └── tutorial.ipynb
```

### Kluczowe Koncepty

**Wzorzec Adaptera**: Wszystkie modele dziedziczą po `BaseScoreFollower` i implementują te same metody:
- `load_reference()` - wczytuje MIDI/partyturę
- `process_frame()` - przetwarza ramkę audio
- `reset()` - resetuje stan

Dzięki temu możesz łatwo wymieniać modele bez zmiany kodu ewaluacji.

---

## 3. Pierwsze Uruchomienie {#pierwsze-uruchomienie}

### Test 1: Sprawdź czy DTW działa

Otwórz Python:

```python
from models.dtw_model import DTWModel

# Stwórz model
model = DTWModel(window_size=100)
print(f"Model: {model.name}")
print(f"Wymaga treningu: {model.requires_training()}")

# Powinno wyświetlić:
# Model: DTW-OTW
# Wymaga treningu: False
```

### Test 2: Użyj Notebooka

```bash
jupyter notebook notebooks/tutorial.ipynb
```

Przejdź przez komórki krok po kroku. Notebook zawiera:
- Wizualizacje audio i MIDI
- Test pojedynczego modelu
- Porównanie modeli
- Wykresy wyników

**Czas na przejście całego notebooka**: ~30 minut

### Test 3: Uruchom Skrypt CLI

```bash
# Podstawowe porównanie
python experiments/run_comparison.py audio.wav reference.mid

# Test tempo robustness
python experiments/run_comparison.py audio.wav reference.mid --tempo-test
```

---

## 4. Ewaluacja Modeli {#ewaluacja}

### Podstawowy Workflow

```python
from models.dtw_model import DTWModel, OnlineTimeWarping
from evaluation.evaluator import Evaluator

# 1. Stwórz ewaluator
evaluator = Evaluator(tolerance_seconds=0.5)

# 2. Stwórz modele
dtw = DTWModel()
otw = OnlineTimeWarping()

# 3. Porównaj
results = evaluator.compare_all_models(
    models=[dtw, otw],
    audio_path="audio.wav",
    reference_path="reference.mid"
)

# 4. Wyniki są automatycznie zapisane w results/
```

### Metryki które otrzymujesz:

- **Frame Accuracy**: % poprawnie zidentyfikowanych ramek (tolerancja ±0.5s)
- **Mean Error**: Średni błąd w sekundach
- **Median Error**: Mediana błędu (mniej wrażliwa na outliers)
- **Mean Latency**: Średnie opóźnienie w ms
- **Tempo Robustness**: Jak dobrze radzi sobie przy zmianach tempa
- **Error Recovery Time**: Jak szybko odzyskuje się po błędzie

### Interpretacja Wyników

**Dobre wyniki** (na podstawie literatury):
- Frame Accuracy > 90%
- Mean Error < 0.3s
- Mean Latency < 100ms

**Słabe wyniki**:
- Frame Accuracy < 70%
- Mean Error > 1.0s
- Mean Latency > 200ms

---

## 5. Dalsze Kroki {#dalsze-kroki}

### Etap 1: Solidny Baseline (Tydzień 1-2)

✅ **Cel**: Działający DTW/OTW na kilku utworach

1. Pobierz 5-10 utworów z MAESTRO
2. Uruchom ewaluację DTW i OTW
3. Zapisz wyniki
4. Zidentyfikuj problemy (gdzie modele się mylą?)

**Zadanie praktyczne**:
```python
# Stwórz listę test cases
test_cases = [
    ("audio1.wav", "midi1.mid"),
    ("audio2.wav", "midi2.mid"),
    # ... więcej
]

# Ewaluuj wszystkie
for audio, midi in test_cases:
    results = evaluator.evaluate_single_model(
        model=dtw,
        audio_path=audio,
        reference_path=midi
    )
    # Zapisz wyniki
```

### Etap 2: Analiza Problemów (Tydzień 3)

✅ **Cel**: Zrozumienie gdzie modele zawodzą

1. **Tempo Robustness Test**:
   ```python
   tempo_results = evaluator.evaluate_tempo_robustness(
       model=dtw,
       audio_path=audio,
       reference_path=midi,
       tempo_ratios=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
   )
   ```

2. **Test z szumem**:
   ```python
   from utils.audio_processing import AudioProcessor
   
   processor = AudioProcessor()
   audio, sr = processor.load_audio("audio.wav")
   
   # Dodaj szum
   noisy_audio = processor.add_noise(audio, noise_factor=0.01)
   ```

3. **Przypadki brzegowe**:
   - Ozdobniki (tryle, arpeggia)
   - Fermaty
   - Skoki tempa
   - Powtórzenia

### Etap 3: Implementacja CNN (Tydzień 4-6)

**UWAGA**: To jest najtrudniejsza część!

1. **Przygotuj dataset treningowy**:
   ```python
   # Potrzebujesz par (audio, MIDI) z wyrównaniem
   # MAESTRO ma to gotowe!
   
   # Format:
   train_data = [
       {
           'audio': audio_array,
           'midi': midi_piano_roll,
           'alignment': ground_truth_positions
       },
       # ... więcej
   ]
   ```

2. **Zaimplementuj training loop**:
   
   Otwórz `models/cnn_model.py` i wypełnij metodę `train()`:
   
   ```python
   def train(self, train_data):
       optimizer = torch.optim.Adam(
           list(self.query_encoder.parameters()) + 
           list(self.context_encoder.parameters()),
           lr=0.001
       )
       
       for epoch in range(num_epochs):
           for batch in train_loader:
               # ... implementuj forward pass
               # ... oblicz loss
               # ... backward pass
   ```

3. **Loss Function**:
   
   Możliwe opcje:
   - **MSE Loss**: `torch.nn.MSELoss()` (najprostsze)
   - **Contrastive Loss**: dla embeddings
   - **Triplet Loss**: dla ranking
   
   Zacznij od MSE!

4. **Trening**:
   ```python
   # Uruchom na GPU (Google Colab)
   model = HeurMiTModel()
   model.train(train_data)
   
   # Zapisz wagi
   torch.save(model.state_dict(), 'cnn_weights.pth')
   ```

### Etap 4: Ostateczne Porównanie (Tydzień 7)

1. Ewaluuj wszystkie modele na tym samym zbiorze testowym
2. Stwórz wykresy porównawcze
3. Przeprowadź testy statystyczne (t-test)
4. Napisz sekcję wyników w pracy

### Etap 5: Praca Pisemna (Tydzień 8+)

**Struktura rozdziału eksperymentalnego**:

1. **Metodologia**
   - Dataset (MAESTRO)
   - Metryki
   - Setup eksperymentu

2. **Wyniki**
   - Tabele porównawcze
   - Wykresy
   - Przykłady

3. **Dyskusja**
   - Co zadziałało?
   - Co nie zadziałało?
   - Dlaczego?

4. **Wnioski**
   - Najlepszy model
   - Ograniczenia
   - Future work

---

## 📚 Dodatkowe Zasoby

### Debugging

**Problem**: Model DTW zwraca bardzo duże błędy
- Sprawdź czy audio i MIDI są zsynchronizowane
- Wizualizuj chromagram audio i MIDI - powinny być podobne
- Zmniejsz window_size

**Problem**: Latencja > 200ms
- Zmniejsz rozmiar okna
- Użyj OTW zamiast DTW
- Ogranicz zakres wyszukiwania

**Problem**: PyTorch nie wykrywa GPU
```python
import torch
print(torch.cuda.is_available())  # Powinno być True

# Jeśli False, zainstaluj CUDA-enabled PyTorch
```

### Pomocne Komendy

```bash
# Sprawdź rozmiar danych
du -sh data/

# Monitoruj GPU
nvidia-smi -l 1

# Uruchom w tle
nohup python train.py &

# Jupyter na serwerze zdalnym
jupyter notebook --no-browser --port=8888
# Lokalnie: ssh -L 8888:localhost:8888 user@server
```

### Literatura Pomocnicza

1. **DTW**: "Dynamic Time Warping" - Tavenard (link w bibliografii)
2. **MAESTRO**: https://magenta.tensorflow.org/datasets/maestro
3. **PyTorch Tutorial**: https://pytorch.org/tutorials/
4. **Librosa**: https://librosa.org/doc/latest/tutorial.html

---

## ❓ FAQ

**Q: Czy muszę implementować wszystkie modele?**
A: Nie! DTW + OTW są wystarczające jako baseline. CNN możesz zostawić jako "future work" jeśli brakuje czasu.

**Q: Ile danych potrzebuję?**
A: Do ewaluacji: 10-20 utworów. Do treningu CNN: przynajmniej 50-100 utworów.

**Q: Jak długo trwa trening CNN?**
A: Na GPU: 2-6 godzin. Na CPU: nie próbuj (kilka dni).

**Q: Co jeśli nie mam GPU?**
A: Użyj Google Colab (darmowe GPU!) lub skup się na DTW/OTW (nie wymaga GPU).

**Q: Skąd wziąć więcej utworów?**
A: MAESTRO dataset (200h muzyki fortepianowej, darmowy).

---

## 🎯 Podsumowanie

**Ścieżka minimalna** (jeśli masz mało czasu):
1. Zaimplementuj DTW i OTW ✓ (gotowe!)
2. Ewaluuj na 10 utworach
3. Testy tempo robustness
4. Napisz wyniki

**Ścieżka średnia** (zalecana):
1. DTW i OTW
2. Szeroka ewaluacja (50+ utworów)
3. Analiza przypadków brzegowych
4. Propozycja ulepszeń (bez implementacji)

**Ścieżka pełna** (jeśli masz czas i chcesz mocną pracę):
1. DTW i OTW
2. Zaimplementuj i wytrenuj CNN
3. Pełna ewaluacja wszystkich modeli
4. Nowa propozycja (Transformer/LSTM)

Powodzenia! 🎵
