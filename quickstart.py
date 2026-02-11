"""
QUICKSTART - Najprostszy możliwy przykład użycia platformy

Ten skrypt pokazuje jak w 10 linijkach kodu uruchomić ewaluację modeli.
"""

from models.dtw_model import DTWModel, OnlineTimeWarping
from evaluation.evaluator import Evaluator


def main():
    print("=" * 70)
    print("QUICKSTART - Platforma Ewaluacji Śledzenia Partytury")
    print("=" * 70)
    
    # KROK 1: Ustaw ścieżki do swoich plików
    # ZMIEŃ TE ŚCIEŻKI NA SWOJE!
    AUDIO_FILE = "path/to/your/audio.wav"
    MIDI_FILE = "path/to/your/reference.mid"
    
    # KROK 2: Stwórz modele
    print("\n[1/3] Creating models...")
    dtw = DTWModel(window_size=100)
    otw = OnlineTimeWarping(window_size=100, search_margin=30)
    print(f"  ✓ Created {dtw.name}")
    print(f"  ✓ Created {otw.name}")
    
    # KROK 3: Stwórz ewaluator
    print("\n[2/3] Creating evaluator...")
    evaluator = Evaluator(tolerance_seconds=0.5)
    print("  ✓ Evaluator ready")
    
    # KROK 4: Uruchom porównanie
    print("\n[3/3] Running comparison...")
    results = evaluator.compare_all_models(
        models=[dtw, otw],
        audio_path=AUDIO_FILE,
        reference_path=MIDI_FILE,
        save_results=True
    )
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nResults saved to: results/")
    print("\nBest model:")
    best = max(results.items(), key=lambda x: x[1].frame_accuracy)
    print(f"  → {best[0]}")
    print(f"  → Accuracy: {best[1].frame_accuracy:.2%}")
    print(f"  → Latency: {best[1].mean_latency:.2f}ms")


if __name__ == "__main__":
    # Sprawdź czy ścieżki zostały ustawione
    import sys
    
    try:
        main()
    except FileNotFoundError as e:
        print("\n" + "!" * 70)
        print("ERROR: File not found!")
        print("!" * 70)
        print("\nPlease edit this file and set correct paths:")
        print("  - AUDIO_FILE = 'path/to/your/audio.wav'")
        print("  - MIDI_FILE = 'path/to/your/reference.mid'")
        print("\nYou can:")
        print("  1. Use files from MAESTRO dataset")
        print("  2. Create synthetic audio from MIDI (see GUIDE.md)")
        print("  3. Use your own recordings")
        print()
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFor help, see:")
        print("  - README.md")
        print("  - GUIDE.md")
        print("  - notebooks/tutorial.ipynb")
        sys.exit(1)
