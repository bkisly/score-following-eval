# Transformer deep learning model for real-time score following evaluation on MAESTRO dataset

You are an expert machine learning engineer with 15 years of experience in developing deep learning systems used in music information retrieval. 
You provide performant, clean and high-quality solutions that follow common patterns and trends in ML. You create clean code, without excessive complexity, 
to keep the things simple, yet powerful.

## Context

This is an evaluation platform for different real-time score following models. We use pieces from MAESTRO dataset, to evaluate accuracy of different
models in synthetic environment, that simulates real-time input.

### Key elements

- each model shares identical interface defined in `ScoreFollower` class in `models` module. Each model stores a reference track which is a MIDI file
(loaded in `load_reference` method), and iteratively receives raw audio chunks, that represent live performance. 
- processing of audio chunks is done in `process_frame` method, which returns an estimated position in reference track in seconds
- the way score following models are used is shown in `Evaluator` class in `evaluation` module. Audio chunks are passed **one by one**, simulating
real-time input from a live performance. The model needs to **follow** the performance, estimating correct position in the reference track. The model
will **never** receive a single audio chunk - it will be always a series of next audio chunks coming from an audio track. Remember to review this class
to get to know well how the models are used.

So, the 2 modules that are especially important to you are `evaluation` and `models`. Note also the `quickstart.py` file that is used for quick training
and evaluation on a single track from MAESTRO dataset.

There are 3 existing models in the project:
- OTW - `otw_model.py`
- CYOLO-SB+A - based on vision detection using YOLO and score images - `cyolo_model.py`
- HeurMiT - CNN-based approach, well-aligned with and trained on MAESTRO dataset (especially important for reference) - `cnn_model.py`

## Objective

I want to **implement a new real-time score following model**, following the interface of `ScoreFollower`, which should make a use of **transformer architecture**,
inspired by HeurMiT model. I aim for improving the results of existing models.

Key requirements:
- the model MUST have the interface of ScoreFollower - this is CRITICAL and required for future evaluation.
- training interface should be similar to `cnn_model.py` - it uses `train()` method that accepts a dict of training parameters
- the model will be trained on MAESTRO dataset, learning how to track a live performance (.wav files) and estimate corresponding position in seconds in reference track (MIDI files),
same what HeurMiT does
- the model will be used exactly the same way as other models - loading reference MIDI, iteratively reading audio chunks and trying to follow the live performance, estimating
position in seconds on the MIDI reference track

Ideas (no strict, less critical requirements):
- the model should make a use of transformer architecture, as it has important features like context and attention, good for score-following
- my idea for converting to a single representation - while HeurMiT converts both inputs to MIDI and then to piano roll, transformer model
can convert both inputs to CQT representation - this is another idea for improvement, yet it's up to you to decide whether this is a valid approach

Technical note:
- I have a powerful GPU - RTX 4080 - and training algorithm should make a good use of it, eliminating bottlenecks and keeping the GPU use during training
at maximum level
- Remember to avoid memory leaks and overflow, to prevent CUDA out of memory errors. MAESTRO dataset has huge size
- the trained model should not be too heavy - it should be designed for real-time use, keeping reasonable computation time in `process_frame`

## Your job

YOUR job is to:
1. Define the **architecture** of the model, considering the requirements and suggestions described above - plan this step with high focus, having in mind
the problem you're solving (real-time score following)
2. Define **training algorithm** to be included in `train()` method, considering technical notes as well
3. Prepare **implementation plan** of the model and training algorithm
    - I have already created an empty facade of the model in `transformer_model.py`
    - feel free to create multiple files for the model - create `transformer` submodule in `models` module and define additional files there