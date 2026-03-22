"""
Prosty skrypt do uruchomienia porównania modeli.
Użyj tego jako punktu wyjścia do swoich eksperymentów.
"""

import sys
from pathlib import Path

# Dodaj projekt do path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.otw_model import DTWModel, OnlineTimeWarping
from models.cnn_model import HeurMiTModel
from models.cyolo_model import CYOLOModel
from evaluation.evaluator import Evaluator


def run_basic_comparison(audio_path: str, midi_path: str):
    """
    Uruchamia podstawowe porównanie dostępnych modeli.
    
    Args:
        audio_path: Ścieżka do pliku audio (.wav, .mp3)
        midi_path: Ścieżka do pliku MIDI (.mid)
    """
    print("="*70)
    print("SCORE FOLLOWING PLATFORM - MODEL COMPARISON")
    print("="*70)
    
    # Sprawdź czy pliki istnieją
    if not Path(audio_path).exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    
    if not Path(midi_path).exists():
        print(f"ERROR: MIDI file not found: {midi_path}")
        return
    
    # Stwórz ewaluator
    evaluator = Evaluator(
        tolerance_seconds=0.5,
        results_dir="results"
    )
    
    # Stwórz modele
    print("\nInitializing models...")
    models = []
    
    # 1. DTW - zawsze działa (nie wymaga treningu)
    try:
        dtw = DTWModel(window_size=100)
        models.append(dtw)
        print(f"✓ {dtw.name} ready")
    except Exception as e:
        print(f"✗ DTW failed to initialize: {e}")
    
    # 2. OTW - zawsze działa
    try:
        otw = OnlineTimeWarping(window_size=100, search_margin=30)
        models.append(otw)
        print(f"✓ {otw.name} ready")
    except Exception as e:
        print(f"✗ OTW failed to initialize: {e}")
    
    # 3. CNN - tylko jeśli wytrenowany
    # (Zakomentowane - wymaga treningu)
    # try:
    #     cnn = HeurMiTModel()
    #     if cnn.is_trained:
    #         models.append(cnn)
    #         print(f"✓ {cnn.name} ready")
    #     else:
    #         print(f"⊗ {cnn.name} - not trained yet, skipping")
    # except Exception as e:
    #     print(f"✗ CNN failed to initialize: {e}")
    
    # 4. CYOLO - tylko jeśli wytrenowany
    # (Zakomentowane - wymaga treningu)
    # try:
    #     cyolo = CYOLOModel()
    #     if cyolo.is_trained:
    #         models.append(cyolo)
    #         print(f"✓ {cyolo.name} ready")
    #     else:
    #         print(f"⊗ {cyolo.name} - not trained yet, skipping")
    # except Exception as e:
    #     print(f"✗ CYOLO failed to initialize: {e}")
    
    if not models:
        print("\nERROR: No models available for comparison!")
        return
    
    print(f"\nReady to compare {len(models)} model(s)")
    
    # Uruchom porównanie
    results = evaluator.compare_all_models(
        models=models,
        audio_path=audio_path,
        reference_path=midi_path,
        save_results=True
    )
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    
    # Znajdź najlepszy model
    best_model = max(results.items(), key=lambda x: x[1].frame_accuracy)
    print(f"\nBest model: {best_model[0]}")
    print(f"Frame Accuracy: {best_model[1].frame_accuracy:.2%}")
    print(f"Mean Error: {best_model[1].mean_error:.3f}s")
    print(f"Mean Latency: {best_model[1].mean_latency:.2f}ms")


def run_tempo_test(audio_path: str, midi_path: str, model_name: str = "DTW"):
    """
    Testuje robustness na zmiany tempa.
    
    Args:
        audio_path: Ścieżka do audio
        midi_path: Ścieżka do MIDI
        model_name: Nazwa modelu do testowania
    """
    print("="*70)
    print("TEMPO ROBUSTNESS TEST")
    print("="*70)
    
    # Stwórz model
    if model_name == "DTW":
        model = DTWModel(window_size=100)
    elif model_name == "OTW":
        model = OnlineTimeWarping(window_size=100, search_margin=30)
    else:
        print(f"Unknown model: {model_name}")
        return
    
    # Stwórz ewaluator
    evaluator = Evaluator(tolerance_seconds=0.5)
    
    # Testuj różne tempa
    tempo_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]  # 80% do 120% oryginalnego tempa
    
    results = evaluator.evaluate_tempo_robustness(
        model=model,
        audio_path=audio_path,
        reference_path=midi_path,
        tempo_ratios=tempo_ratios,
        verbose=True
    )
    
    # Podsumowanie
    print("\n" + "="*70)
    print("TEMPO ROBUSTNESS SUMMARY")
    print("="*70)
    
    for ratio, metrics in results.items():
        print(f"\nTempo: {ratio*100:.0f}%")
        print(f"  Accuracy: {metrics.frame_accuracy:.2%}")
        print(f"  Mean Error: {metrics.mean_error:.3f}s")
        print(f"  Latency: {metrics.mean_latency:.2f}ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Score Following Platform - Porównanie modeli"
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Ścieżka do pliku audio"
    )
    parser.add_argument(
        "midi",
        type=str,
        help="Ścieżka do pliku MIDI"
    )
    parser.add_argument(
        "--tempo-test",
        action="store_true",
        help="Uruchom test tempo robustness zamiast porównania"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DTW",
        choices=["DTW", "OTW"],
        help="Model do testu tempo (domyślnie DTW)"
    )
    
    args = parser.parse_args()
    
    if args.tempo_test:
        run_tempo_test(args.audio, args.midi, args.model)
    else:
        run_basic_comparison(args.audio, args.midi)


# Przykład użycia:
"""
# Podstawowe porównanie:
python experiments/run_comparison.py path/to/audio.wav path/to/reference.mid

# Test tempo robustness:
python experiments/run_comparison.py path/to/audio.wav path/to/reference.mid --tempo-test --model OTW
"""
