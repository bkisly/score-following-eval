# CLAUDE.md — Score Following Evaluation Platform

## Project context

This is a **research platform for a Master's thesis** titled *"Real-time Score Following Using DTW-type Algorithms and Machine Learning"*. The goal is to compare existing score following systems across several metrics: latency, alignment accuracy (frame accuracy), robustness to tempo changes (tempo robustness), and error recovery time.

**Score following** is the task of continuously estimating the current position in a reference recording based on a live audio stream. Two main paradigms:
- **Symbolic/OTW**: chroma-based (CENS) feature matching between audio streams — no training required.
- **Deep learning / CYOLO**: matching audio to a score sheet image framed as object detection — requires training.

---

## Project structure

```
score_following_platform/
├── CLAUDE.md                     ← this file
├── models/
│   ├── base_model.py             ← interface (adapter pattern) — DO NOT MODIFY
│   ├── dtw_model.py              ← DTW/OTW skeleton using fastdtw (placeholder)
│   ├── cyolo_model.py            ← CYOLO-SB+A skeleton (placeholder)
│   ├── cnn_model.py              ← CNN HeurMiT-like skeleton (placeholder)
│   ├── otw/                      ← source code from github.com/matthewcaren/web-score-following
│   │   └── (files from ConcertCue repo)
│   └── cyolo/                    ← source code from github.com/CPJKU/cyolo_score_following
│       └── (files from CYOLO repo)
├── evaluation/
│   └── evaluator.py              ← main evaluator — ready to use
├── experiments/
│   └── run_comparison.py         ← experiment runner — ready to use
├── utils/
│   ├── audio_processing.py       ← AudioProcessor, simulate_real_time_input
│   ├── midi_processing.py        ← MIDIProcessor (pretty_midi, no FluidSynth on Windows)
│   └── metrics.py                ← EvaluationMetrics, MetricsCalculator
├── notebooks/
│   └── tutorial.ipynb
├── requirements.txt
└── results/                      ← auto-generated JSON results from evaluations
```

---

## Core contract: `BaseScoreFollower` (models/base_model.py)

**Every model must inherit from `BaseScoreFollower` and implement exactly these methods:**

```python
class BaseScoreFollower(ABC):
    def __init__(self, name: str)

    @abstractmethod
    def load_reference(self, reference_path: str) -> None:
        """Load the reference (MIDI or score image)."""

    @abstractmethod
    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Process a single audio frame. MAIN METHOD CALLED IN THE REAL-TIME LOOP.

        Must return a dict with:
            'position'   : float  — estimated position in seconds within the reference
            'confidence' : float  — prediction confidence [0.0, 1.0]
            'tempo'      : float  — estimated BPM (can be None or 0.0 if unavailable)
            'latency'    : float  — processing time in ms
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset model state before a new piece."""

    def requires_training(self) -> bool:
        """Return True for ML models, False for DTW-based."""
        return False

    def train(self, train_data: Any) -> None:
        """Optional — override in ML models."""
```

`ScoreFollowerAdapter` (in the same file) wraps a model and exposes `follow_audio(audio_path, reference_path, ground_truth, ...)` — already implemented, do not modify.

---

## Models to implement (bindings)

### 1. OTW — ConcertCue (`models/otw/`)

**Source**: https://github.com/matthewcaren/web-score-following  
**Paper**: *"Real-time In-browser Time Warping for Live Score Following"* (WAC 2024, Caren & Egozy, MIT)

**Algorithm overview:**
- Feature: **CENS** (chroma energy normalized statistics) — 12-dimensional vector per frame
- Alignment: **Online Time Warping (OTW)** — Dixon's (2005) online variant of DTW
- Cost function: cosine distance between live and reference CENS vectors
- O(n) time and space complexity (vs. O(n²) for offline DTW)
- OTW searches only within a window around the current position (parameter `c`)
- Predictions are **clamped** to be monotonically non-decreasing

**Tuned algorithm parameters (from paper, found via grid search):**
```
sample_rate           = 44100 Hz
reference_hop_size    = 4096 samples
live_hop_size         = 3686 samples   ← ~90% of reference hop — important!
CENS FFT window size  = 8192 samples
OTW "c"               = 300            ← search window width
OTW "Max Run Count"   = 3
OTW "Diagonal Weight" = 0.4            ← lower = better avoidance of "getting stuck"
```

**Benchmark results from paper:**
| Metric       | Browser | Simulated Python |
|--------------|---------|-----------------|
| Median error | 0.089s  | 0.077s          |
| 95th pctile  | 0.639s  | 0.537s          |
| Mean error   | 0.174s  | 0.130s          |

**What needs to be done — OTW binding:**
- Create `models/otw_binding.py` inheriting from `BaseScoreFollower`
- `load_reference()` → load reference WAV/MIDI, extract CENS features for the full file, store in a reference buffer
- `process_frame()` → append audio frame to live buffer, extract CENS, run one OTW step (update cost matrix + determine current position), return dict
- `reset()` → clear OTW state (cost matrix, position pointer)
- Note: the source code targets Pyodide (WebAssembly/browser); when binding to desktop Python, skip or mock browser-specific parts (JS interop, AnalyzerNode). The core algorithm logic is pure Python/NumPy.

**Potential issue**: OTW assumes a fixed hop size. In desktop simulation, hop is deterministic (unlike browser where it had ~±2ms jitter). This actually simplifies the desktop binding.

**Sample rate mismatch**: The platform's `AudioProcessor` defaults to 22050 Hz, while OTW uses 44100 Hz internally. The binding must either resample the input or reconfigure the processor for this model.

---

### 2. CYOLO-SB+A — JKU (`models/cyolo/`)

**Source**: https://github.com/CPJKU/cyolo_score_following  
**Paper**: *"Real-Time Music Following in Score Sheet Images via Multi-Resolution Prediction"* (Frontiers in Computer Science, 2021, Henkel & Widmer)

**Algorithm overview:**
- Input: **score sheet image** (416×416 px) + **audio spectrogram** (78 log-frequency bins, ~20fps)
- Architecture: **Conditional YOLO** — U-Net-like network with **FiLM** (feature-wise linear modulation) layers fusing audio with the image
- Output: **bounding box** around the currently played note/bar/system (3 granularity levels)
- Audio encoder: CNN → LSTM (64 hidden units) → conditioning vector z (updated every 40 frames ≈ 2s context)
- CYOLO-SB+A: predicts notes, bars (S+B) simultaneously, trained on synthetic MSMD + additional real data (A)
- Works on **full pages** — no score unrolling required

**Benchmark results from paper (best model CYOLO-SB+A):**
| Dataset | ≤0.05s | ≤0.5s | ≤5.0s |
|---------|--------|-------|-------|
| MSMD (synth audio+image) | 0.846 | 0.908 | 0.984 |
| MSMD-Rec (real audio) | 0.682 | 0.865 | 0.981 |
| RealScores-Rec (hardest) | 0.456 | 0.670 | 0.929 |

Inference time: ~6ms/frame on GPU (GTX 1080).

**Audio spectrogram parameters (from paper):**
```
sample_rate    = 22050 Hz
STFT window    = 2048 samples (Hann)
hop_size       = 1102 samples  → ~20 frames/second
frequency bins = 78 log-frequency bins (60 Hz – 6 kHz)
LSTM context   = 40 latest frames (~2s) + hidden state
```

**What needs to be done — CYOLO binding:**
- Create `models/cyolo_binding.py` inheriting from `BaseScoreFollower`
- `load_reference()` → load score image (PNG), resize to 416×416, load position→time ground truth map (if available) for converting predicted bbox pixel coordinates to seconds
- `process_frame()` → build log-frequency spectrogram (78 bins), update LSTM audio encoder hidden state, run network forward pass, decode bounding box → time position
- `reset()` → zero out LSTM hidden state
- Load the **pre-trained checkpoint** (`.pth`) from the CYOLO repo
- `requires_training()` → `return True`; set `is_trained = True` after loading the checkpoint

**Key conversion**: CYOLO returns pixel coordinates of a bounding box on the score image. Converting to seconds requires a `(x, y) → time` map specific to each score page — this must be provided alongside the score image (from MSMD annotations or manually annotated).

**Known failure cases (from paper):** very slow pieces (sparse onsets), trills, arpeggios, highly repetitive passages that differ only by 1-2 semitones.

---

## Metrics system (`utils/metrics.py`)

`EvaluationMetrics` (dataclass) stores:
- `frame_accuracy` — % of frames with error ≤ tolerance (default 0.5s)
- `mean_error`, `median_error`, `std_error` — in seconds
- `mean_latency`, `max_latency` — in milliseconds
- `tempo_robustness` — mean accuracy across different tempo ratios [0, 1]
- `error_recovery_time` — mean time from error onset to recovery (seconds)

`MetricsCalculator(tolerance_seconds=0.5)`:
- `calculate_all_metrics(predictions, ground_truth, latencies, ...)` → `EvaluationMetrics`
- `calculate_tempo_robustness(predictions_dict, ground_truth_dict)` — takes `{tempo_ratio: preds}` dict
- `calculate_error_recovery_time(predictions, ground_truth, fps=43.0)` — measures frames from first error until return within tolerance

`compare_models(results_dict)` → formats a comparison table as a string.

Results are auto-saved to `results/results_{audio_stem}.json`.

---

## Evaluator (`evaluation/evaluator.py`)

`Evaluator(tolerance_seconds=0.5, results_dir="results")`:

- **`evaluate_single_model(model, audio_path, reference_path, ground_truth_alignment, verbose)`**  
  Main evaluation method. Loads audio, simulates real-time via `simulate_real_time_input()` (chunk_size=2048), collects predictions and latencies, computes metrics.

- **`evaluate_tempo_robustness(model, audio_path, reference_path, tempo_ratios=[0.9,1.0,1.1,1.2])`**  
  Time-stretches audio via `librosa.effects.time_stretch()` and evaluates the model at each tempo.

- **`compare_all_models(models, audio_path, reference_path, save_results=True)`**  
  Iterates over a list of models, skips untrained ones, aggregates results.

**Important**: `ground_truth_alignment` is a `np.ndarray` of shape `[n_frames]` — position in seconds in the reference recording for each frame. If not provided, the evaluator creates a linear mapping (assumes no tempo variation) — sufficient for synthetic test data.

---

## Audio processing (`utils/audio_processing.py`)

`AudioProcessor(sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128)`:
- `load_audio(path)` → `(np.ndarray, int)` via librosa
- `compute_chroma(audio)` → `[12, T]` chroma_cqt
- `compute_spectrogram(audio, use_mel=True)` → `[n_mels, T]` or `[n_fft//2+1, T]`
- `compute_mfcc(audio, n_mfcc=13)` → `[n_mfcc, T]`
- `time_stretch(audio, rate)` → librosa time stretch
- `add_noise(audio, noise_factor=0.005)` → noise augmentation

`simulate_real_time_input(audio, chunk_size=2048, sr=22050)` → list of audio chunks (complete chunks only, last partial chunk dropped).

---

## MIDI processing (`utils/midi_processing.py`)

`MIDIProcessor(fps=100)`:
- `load_midi(path)` → `pretty_midi.PrettyMIDI`
- `midi_to_piano_roll(midi)` → `[128, T]`
- `extract_chroma_from_midi(midi)` → `[12, T]`
- `get_note_events(midi)` → list of `(time, pitch, velocity)`
- `get_tempo_changes(midi)` → list of `(time, bpm)`
- `synthesize_audio(midi, fs=22050)` → numpy audio array; **requires FluidSynth** — on Windows may not work, returns silence with a warning

---

## Test data and datasets

**MSMD** (Multi-modal Sheet Music Dataset) — primary dataset for CYOLO; synthetic score images + audio rendered from MIDI via Fluidsynth. Available on Zenodo.

**MAESTRO** — piano recordings with aligned MIDI. Used by ConcertCue and as training data for audio-based models.

**ConcertCue evaluation set** — 18 recordings (6 pieces × 3 performances), hand-annotated downbeats. Includes: string quartet, symphony, piano concerto, chamber music. Errors measured per-downbeat (not per-frame).

For synthetic testing, audio can be generated from MIDI via `MIDIProcessor.synthesize_audio()` (if FluidSynth is available), or paired WAV+MIDI files can be downloaded from MAESTRO.

---

## What is done vs. what is not

### ✅ Done
- `BaseScoreFollower` interface (adapter pattern) — **do not modify**
- `ScoreFollowerAdapter.follow_audio()` — ready
- `Evaluator` with evaluation and comparison methods — ready
- `MetricsCalculator` and `EvaluationMetrics` — ready
- `AudioProcessor` and `MIDIProcessor` — ready
- `run_comparison.py` — ready experiment runner
- Model skeletons (`dtw_model.py`, `cyolo_model.py`, `cnn_model.py`) — **to be replaced with proper bindings**

### ❌ To be implemented (main goal)
1. **`models/otw_binding.py`** — binding to OTW code in `models/otw/` (ConcertCue)
2. **`models/cyolo_binding.py`** — binding to CYOLO code in `models/cyolo/` (CPJKU)
3. Optionally: `evaluation/visualizer.py` (matplotlib plots) — mentioned in README but does not exist yet
4. Loading the pre-trained CYOLO checkpoint

### ⚠️ Skeletons to replace / fix
- `dtw_model.py` uses `fastdtw` as an offline DTW — this is **not** the real OTW from ConcertCue. Serves as a rough baseline until the proper OTW binding is implemented.
- `cyolo_model.py` — randomly initialized weights, produces nonsense output; placeholder showing architecture only.

---

## How to write a new binding — template

```python
# models/otw_binding.py
import sys
sys.path.append('models/otw')  # add path to repo source code

from models.base_model import BaseScoreFollower
# import required modules from models/otw/

class OTWBinding(BaseScoreFollower):
    def __init__(self, **kwargs):
        super().__init__(name="OTW-ConcertCue")
        # initialize OTW state

    def load_reference(self, reference_path: str) -> None:
        # 1. load reference audio file (WAV) or MIDI
        # 2. extract CENS features for the full file
        # 3. store as reference_buffer
        pass

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        import time
        t0 = time.time()
        # 1. extract CENS from audio_frame
        # 2. run one OTW step (update cost matrix + determine position)
        # 3. return required dict
        latency = (time.time() - t0) * 1000
        return {
            'position': self.current_position,
            'confidence': 0.0,  # or compute from cost matrix
            'tempo': 0.0,
            'latency': latency
        }

    def reset(self) -> None:
        self.current_position = 0
        # reset OTW internal state

    def requires_training(self) -> bool:
        return False
```

---

## Running experiments

```bash
# Basic model comparison
python experiments/run_comparison.py path/to/audio.wav path/to/reference.mid

# Tempo robustness test
python experiments/run_comparison.py audio.wav ref.mid --tempo-test --model OTW
```

Or via Python API:
```python
from evaluation.evaluator import Evaluator
from models.otw_binding import OTWBinding       # after implementation
from models.cyolo_binding import CYOLOBinding   # after implementation

evaluator = Evaluator(tolerance_seconds=0.5)

otw = OTWBinding()
cyolo = CYOLOBinding()
cyolo.load_checkpoint('models/cyolo/checkpoints/best.pth')
cyolo.is_trained = True

results = evaluator.compare_all_models(
    models=[otw, cyolo],
    audio_path='data/beethoven_op18.wav',
    reference_path='data/beethoven_op18.mid',
    save_results=True
)
```

---

## Conventions and design decisions

- **Position always in seconds** — `process_frame()` must return time in seconds, not frames
- **Latency in milliseconds** — measured inside `process_frame()` using `time.time()`
- **Tolerance = 0.5s** — default value for `frame_accuracy`; configurable via `Evaluator(tolerance_seconds=...)`
- **Default sample rate = 22050 Hz** — platform default (AudioProcessor); OTW internally uses 44100 Hz — the binding must handle this (resample input or reconfigure the processor for that model)
- **Model names**: `model.name` is a short identifier used in result tables and JSON output
- **Results JSON**: auto-saved to `results/results_{stem}.json` by `Evaluator`
- **No global state** — each model is stateful (has `current_position` etc.), reset via `reset()`
- **Untrained ML models are skipped** by `compare_all_models()` — check `model.requires_training() and not model.is_trained`

---

## References

- **ConcertCue / OTW**: Caren & Egozy, "Real-time In-browser Time Warping for Live Score Following", WAC 2024. GitHub: `matthewcaren/web-score-following`
- **CYOLO**: Henkel & Widmer, "Real-Time Music Following in Score Sheet Images via Multi-Resolution Prediction", Frontiers in Computer Science, 2021. GitHub: `CPJKU/cyolo_score_following`
- **Dixon OTW**: Dixon, "Live Tracking of Musical Performances using On-Line Time Warping", DAFx 2005
- **CENS features**: Müller, "Fundamentals of Music Processing", Springer, 2021
