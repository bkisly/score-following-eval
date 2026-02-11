# Platforma do Ewaluacji Systemów Śledzenia Partytury

Platforma badawcza do porównania różnych metod śledzenia partytury w czasie rzeczywistym.

## Struktura projektu

```
score_following_platform/
├── data/                      # Dane i datasety
│   ├── maestro/              # MAESTRO dataset
│   └── custom/               # Własne przypadki testowe
├── models/                    # Implementacje modeli
│   ├── base_model.py         # Abstrakcyjna klasa bazowa
│   ├── dtw_model.py          # DTW/OTW
│   ├── cyolo_model.py        # CYOLO-SB+A (szkielet)
│   └── cnn_model.py          # CNN HeurMiT-like (szkielet)
├── utils/                     # Narzędzia pomocnicze
│   ├── audio_processing.py   # Przetwarzanie audio
│   ├── midi_processing.py    # Obsługa MIDI
│   └── metrics.py            # Metryki ewaluacji
├── evaluation/               # Ewaluacja i testy
│   ├── evaluator.py         # Główny ewaluator
│   └── visualizer.py        # Wizualizacja wyników
├── experiments/              # Skrypty eksperymentów
│   └── run_comparison.py    # Porównanie modeli
├── notebooks/               # Jupyter notebooks
│   └── tutorial.ipynb      # Tutorial krok po kroku
└── requirements.txt         # Zależności

```

## Instalacja

```bash
pip install -r requirements.txt
```

## Jak używać

### 1. Przygotowanie danych
```python
from utils.audio_processing import prepare_maestro_dataset
prepare_maestro_dataset('path/to/maestro')
```

### 2. Trening modelu (tylko dla ML)
```python
from models.cnn_model import CNNModel
model = CNNModel()
model.train(train_data)
```

### 3. Ewaluacja
```python
from evaluation.evaluator import Evaluator
evaluator = Evaluator()
results = evaluator.evaluate_all_models(test_data)
```

## Metryki

- **Frame Accuracy** - procent poprawnie zidentyfikowanych pozycji
- **Latency** - średnie opóźnienie w ms
- **Tempo Robustness** - skuteczność przy zmianach tempa
- **Error Recovery** - zdolność do odzyskania po błędzie

## Modele

1. **DTW/OTW** - Baseline, bez treningu
2. **CYOLO-SB+A** - Detektor wizyjny (wymaga treningu)
3. **CNN HeurMiT** - Sieć konwolucyjna (wymaga treningu)

## Workflow badawczy

1. Zaimplementuj DTW jako baseline
2. Przetestuj na małym zbiorze danych
3. Zaimplementuj kolejne modele
4. Przeprowadź pełną ewaluację
5. Zbierz wyniki i stwórz wykresy
