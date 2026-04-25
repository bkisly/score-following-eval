"""
QUICKSTART - Najprostszy możliwy przykład użycia platformy

Ten skrypt pokazuje jak w 10 linijkach kodu uruchomić ewaluację modeli.
"""
import os
import traceback

from models.cnn_model import HeurMiTModel
from models.otw_model import OTWModel
from models.cyolo_model import CYOLOModel
from evaluation.evaluator import Evaluator
from models.transformer_model import TransformerModel

# KROK 1: Ustaw ścieżki do swoich plików
# ZMIEŃ TE ŚCIEŻKI NA SWOJE!

BASE_PATH = r"C:\Users\bkisl\Desktop\maestro-v3.0.0\maestro-v3.0.0"

AUDIO_FILE = fr"{BASE_PATH}\2018\MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav"
MIDI_FILE = fr"{BASE_PATH}\2018\MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi"

# AUDIO_FILE = r"C:\Users\bkisl\Desktop\chopin-etude-op10-no4.mp3"
# MIDI_FILE = r"C:\Users\bkisl\Desktop\chopin-etude-op10-no4.mid"

def main():
    print("=" * 70)
    print("QUICKSTART - Platforma Ewaluacji Śledzenia Partytury")
    print("=" * 70)
    
    # KROK 2: Stwórz modele
    # print("\n[1/3] Creating models...")
    # otw = OTWModel()
    # print(f"  ✓ Created {otw.name}")
    # cyolo = CYOLOModel()
    # print(f"  ✓ Created {cyolo.name}")
    #
    # HEURMIT_CHECKPOINT_PATH = 'heurmit.pth'
    #
    # heurMiT = HeurMiTModel()
    # if not os.path.exists(HEURMIT_CHECKPOINT_PATH):
    #     heurMiT.train({'dataset_path': BASE_PATH, 'save_path': HEURMIT_CHECKPOINT_PATH})
    #
    # heurMiT.load_checkpoint(HEURMIT_CHECKPOINT_PATH)
    # print(f"  ✓ Created {heurMiT.name}")

    transformer = TransformerModel()
    TRANSFORMER_CHECKPOINT_PATH = 'transformer.pth'

    if os.path.exists(TRANSFORMER_CHECKPOINT_PATH):
        transformer.load_checkpoint(TRANSFORMER_CHECKPOINT_PATH)
    else:
        transformer.train({'save_path': TRANSFORMER_CHECKPOINT_PATH, 'dataset_path': BASE_PATH})

    # KROK 3: Stwórz ewaluator
    print("\n[2/3] Creating evaluator...")
    evaluator = Evaluator(tolerance_seconds=0.5)
    print("  ✓ Evaluator ready")
    
    # KROK 4: Uruchom porównanie
    print("\n[3/3] Running comparison...")
    results = evaluator.compare_all_models(
        models=[transformer],
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
        print(f"  - AUDIO_FILE = '{AUDIO_FILE}'")
        print(f"  - MIDI_FILE = '{MIDI_FILE}'")
        print("\nYou can:")
        print("  1. Use files from MAESTRO dataset")
        print("  2. Create synthetic audio from MIDI (see GUIDE.md)")
        print("  3. Use your own recordings")
        print()
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nFor help, see:")
        print("  - README.md")
        print("  - GUIDE.md")
        print("  - notebooks/tutorial.ipynb")
        traceback.print_exc()
        sys.exit(1)
