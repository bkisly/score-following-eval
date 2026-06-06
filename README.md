# Score Following Evaluation Platform

This repository contains evaluation platform for comparing real-time audio-MIDI alignment algorithm and an application,
demonstrating online use.

## Models included

1. Online Time Warping (OTW) - taken from ConcertCue project
2. CYOLO-SB+A
3. Convolutional neural model, implemented based on *HeurMiT* system by A. Pillay
4. self-designed transformer model, inspired by *HeurMiT* 

## Requirements

Apart from Python packages defined in `requirements.txt`, this platform requires external tools such as:
- MuseScore / LilyPond - required for CYOLO model
- fluidsynth - required for MIDI-to-audio synthesis for OTW model

## Installation

Use the command:

```bash
pip install -r requirements.txt
```

## Data

Experiments use MAESTRO data set, which requires correct configuration of paths pointing to it.

## Running benchmarks

Use `notebooks/experiments.ipynb`, which is a Jupyter notebook containing complete benchmarks of included models.

## Running demo app

Use the command:
```bash
python -m app.main --wav [path to audio file] --midi [path to midi file] --model [model id, same as in 'Models included' section] --checkpoint [.pth file]
```

## Checkpoints

Transformer model requires `transformer.pth` checkpoint file. CYOLO model contains original checkpoints in `models/cyolo/trained_models` directory.
*HeurMiT* model requires `heurmit.pth` checkpoint file.

## Limitations

Demo application uses audio files streamed to the models, not microphone input.
Results regard mostly audio-to-MIDI alignment for piano pieces.