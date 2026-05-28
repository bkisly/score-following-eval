# PatchFormer — Architecture Description

**Model:** `TransformerModel` (`models/transformer_model.py`)  
**Network:** `TransformerNet` (`models/transformer/network.py`)  
**Dataset:** `MAESTROTransformerDataset` (`models/transformer/dataset.py`)

---

## Overview

PatchFormer is a dual-encoder transformer model for real-time score following. It addresses the core challenge of continuously estimating playback position in a MIDI reference score given a stream of incoming live audio.

The model operates on two heterogeneous signal representations:

- **Reference** — a binary piano roll derived from a MIDI file (`[128, T]` at 100 fps), capturing the symbolic score.
- **Live audio** — a Constant-Q Transform (CQT) spectrogram derived from real WAV audio (`[128, T]` at 100 fps), capturing the acoustic performance.

The task is formulated as **local alignment**: given a short live audio window and a longer reference context slice (centred around the current estimated position), rank every candidate starting position in the context and return the best match.

This avoids the global alignment problem (O(n²) DTW) by reducing inference to a fast dot-product search over ~128 patch embeddings, enabling sub-millisecond computation per frame on GPU.

---

## High-Level Architecture

```
MIDI reference                          Live audio (streaming)
      │                                        │
[pretty_midi piano roll]              [librosa CQT]
[128, T] binary, 100 fps              [128, w=128] float32, 100 fps
      │                                        │
      ▼                                        ▼
 E_ref (PatchEncoder)              E_live (PatchEncoder)
 ─────────────────────              ───────────────────
 Conv1d  [128 → d, k=4]            Conv1d  [128 → d, k=4]
 + Sinusoidal PE                   + Sinusoidal PE
 + 2-layer TransformerEncoder      + 2-layer TransformerEncoder
      │                                        │
[B, N_ctx, d]                      [B, N_win, d]
      │                               mean-pool ▼
      │                            live_emb [B, d]
      │                                        │
      └────────── dot-product matching ────────┘
                          │
                  logits [B, N_valid]
                          │
               argmax → predicted window-start patch
                          │
              + heuristic stability filter
                          │
              current_position (seconds)
```

Both encoders share identical structure but have **separate, independently learned weights**. `E_ref` learns to represent symbolic note patterns; `E_live` learns to represent acoustic audio patterns. The shared embedding space is learned end-to-end via cross-entropy on the alignment task.

---

## Technical Details

### 1. Feature Representations

#### Reference (piano roll)

```
pretty_midi.PrettyMIDI.get_piano_roll(fs=100)
→ binarised: (roll > 0).astype(float32)
→ shape: [128, T_ref]   (128 MIDI pitch bins, 0.01 s/frame)
```

The full piano roll is loaded once in `load_reference()`. It is never windowed to RAM; only a context slice of length `c = 512` frames (5.12 s) is encoded per inference step.

#### Live audio (CQT)

```
librosa.cqt(audio, sr=22050, hop_length=220, n_bins=88, fmin=A0)
→ |CQT| normalised by 95th percentile, clipped to [0, 1]
→ embedded in MIDI pitch space: bins 21–108 of a 128-bin array
→ shape: [128, w=128]   (1.28 s at 100 fps)
```

The CQT hop length is `round(sample_rate / fps) = 220` samples (≈ 10 ms), giving a time resolution that matches the piano roll frame rate exactly.

Both representations occupy the same 128-bin MIDI pitch space, enabling the two encoders to operate on identically shaped inputs.

---

### 2. PatchEncoder

`PatchEncoder` (`models/transformer/network.py`) is the shared building block used for both `E_ref` and `E_live`.

```
Input: [B, 128, T]
```

**Step 1 — Patch embedding (Conv1d)**

```python
nn.Conv1d(in_channels=128, out_channels=d_model,
          kernel_size=patch_size, stride=patch_size)
→ [B, d_model, T // patch_size]
→ transpose → [B, N, d_model]   where N = T // patch_size
```

With `patch_size = 4` and `fps = 100`, each patch covers 4 frames = 40 ms. For the reference context (`T = c = 512`): `N_ctx = 128` patches. For the live window (`T = w = 128`): `N_win = 32` patches.

The Conv1d layer acts as a learned linear projection over non-overlapping 40 ms segments; it replaces a naive frame-by-frame linear projection with one that captures short local temporal patterns within the projection.

**Step 2 — Sinusoidal positional encoding**

```python
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Stored as a fixed (non-learned) buffer of shape `[max_seq_len, d_model]`. Added to the patch embeddings before the transformer. Provides patch-order information; necessary because attention is permutation-invariant by design.

**Step 3 — TransformerEncoder (pre-LN)**

```
2 × TransformerEncoderLayer(
    d_model  = 128,
    nhead    = 4,          → head_dim = 32
    dim_feedforward = 256,
    dropout  = 0.1,
    norm_first = True      → pre-LN for training stability
)
```

Pre-LayerNorm places the normalisation before the sub-layer (as in the GPT-2 / PaLM formulation), which avoids gradient explosion at initialisation and typically converges faster than post-LN.

Each layer applies:
1. `LN → multi-head self-attention → residual`
2. `LN → position-wise FFN (d → 4d → d) → residual`

Self-attention allows every patch to attend to every other patch in the same sequence, capturing long-range harmonic and rhythmic dependencies within the context.

```
Output: [B, N, d_model]
```

---

### 3. Matching Head

After encoding:

- `ref_patches`: `[B, N_ctx, d_model]` — encoded reference context
- `live_emb`: `[B, d_model]` — mean-pooled encoded live window

```python
scores = torch.bmm(ref_patches, live_emb.unsqueeze(-1)).squeeze(-1)  # [B, N_ctx]
logits = scores[:, :N_valid]                                          # [B, N_valid]
```

where `N_valid = N_ctx − N_win + 1 = 97`.

Each of the `N_valid` logit values scores one candidate position: "how well does the live window match the reference starting at patch `p`?"

This is a single batched matrix-vector multiply — O(N_ctx · d_model) per inference step, negligible compared to the transformer forward pass.

---

### 4. Training

**Task formulation**

Given a reference context `C` (piano roll) of length `c = 512` frames and a live window `W` (CQT) of length `w = 128` frames, predict the patch index `p ∈ [0, N_valid − 1]` such that `W` aligns with `C` starting at frame `p · patch_size`.

```
label_patch = (ws − ctx_s) // patch_size
```

where `ws` is the window-start frame and `ctx_s` is the context-start frame, both drawn randomly from the same aligned CQT/piano-roll pair.

**Loss**

```python
CrossEntropyLoss(label_smoothing=0.1)
```

Label smoothing reduces overconfidence and improves calibration.

**Accuracy tolerance**

A prediction is counted correct if `|pred − label| ≤ tol_patches`, where `tol_patches = max(1, round(0.005 · fps / patch_size)) = 1 patch = 40 ms`. This mirrors the 5 ms tolerance used in HeurMiT but scaled to patch granularity.

**Optimiser and schedule**

```
AdamW(lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999))
OneCycleLR(max_lr=1e-3, pct_start=0.1, anneal_strategy="cos")
```

The one-cycle schedule warms up for 10% of training then cosine-anneals to zero, which is effective for transformer models and avoids the need for manual LR tuning.

**Mixed-precision training**

`torch.cuda.amp.GradScaler` with `torch.autocast(dtype=float16)` is used throughout, roughly halving memory usage and doubling throughput on CUDA GPUs with Tensor Cores.

**Data augmentation**

`apply_augmentations(W)` is applied to the CQT live window during training. Augmentations simulate acoustic variability (dynamic range changes, mild pitch perturbations, time jitter) to improve robustness to real performance conditions.

**Dataset pipeline**

Training data is drawn from the MAESTRO dataset (train split). Piano rolls and CQT features are pre-computed once and cached as float16 `.npy` files (≈ 7.5 MB/file). The `MAESTROTransformerDataset` draws random `(C, W, label)` triplets on-the-fly, with shuffled piece selection and random window placement, providing effectively unlimited training variety.

---

### 5. Inference Pipeline

```
process_frame(audio_frame, sample_rate)
```

**Step 1 — Elapsed-time tracker**

```python
_elapsed_frames += len(audio_frame) / sample_rate * fps
```

Maintains a running count of processed audio frames, used to centre the reference context and as a fallback position estimate.

**Step 2 — Audio buffer and live window**

Incoming audio chunks are appended to a ring buffer. The last `w_samples = inf_w · sample_rate / fps = 28 224` samples are extracted as the live window, converted to CQT features (`[128, inf_w]`), and encoded by `E_live`.

**Step 3 — Reference context retrieval**

The context is centred on `elapsed_patch ± N_ctx / 2`:

```
ctx_patch_start = clamp(elapsed_patch − 64, 0, N_ref_patches − N_ctx)
ctx_frame_start = ctx_patch_start × patch_size
```

Raw patch embeddings (Conv1d-only, no PE or transformer) are pre-computed for the entire reference in `load_reference()` and stored on CPU. Per frame, only a slice of `N_ctx = 128` embeddings is fetched; positional encoding and the transformer are then applied on the GPU to that slice only.

**Step 4 — Dot-product matching**

```
logits = match(ref_patches, live_emb)   [N_valid = 97]
```

**Step 5 — Heuristic stability filter**

Directly using `argmax(logits)` can be noisy when the network is uncertain or the context is ambiguous. A lightweight heuristic (adapted from HeurMiT) applies:

1. **Smoothing** — 5-point moving average over the logit vector.
2. **Peak detection** — `scipy.find_peaks` to locate prominent local maxima.
3. **Ring-buffer extrapolation** — linear regression over the last 20 predictions to project an expected absolute position (`buf_abs`).
4. **Validity checks** — the network's prediction is accepted only if it satisfies three criteria simultaneously:
   - **Monotonicity**: does not retreat more than 48 frames (0.48 s) below the previous position.
   - **Range**: within `[buf_abs − 48, buf_abs + 96]` of the extrapolated position.
   - **Rate**: step size is within `[0.5×, 1.5×]` of the expected rate.
5. If the prediction fails all checks for more than `max_consecutive_buffer = 5` frames, the network's output is accepted anyway (error-recovery override).

**Step 6 — Position output**

```python
raw_abs_frame = ctx_frame_start + raw_k + inf_w
current_position = raw_abs_frame / fps   [seconds]
```

The `+ inf_w` offset converts the predicted **window start** frame (what the model was trained to predict) to the **window end** frame (the current playback position). The window end equals the current elapsed time in the reference, which is what the evaluator's ground truth measures.

The result is then clamped to `[elapsed ± max_deviation]` and `[0, T_ref − 1]` before being returned.

---

### 6. Model Parameters (Default Configuration)

| Hyperparameter | Value | Meaning |
|---|---|---|
| `d_model` | 128 | Transformer hidden dimension |
| `patch_size` | 4 | Frames per patch (40 ms at 100 fps) |
| `n_heads` | 4 | Attention heads (head\_dim = 32) |
| `n_layers` | 2 | TransformerEncoder layers per encoder |
| `d_ff` | 256 | FFN hidden dimension (2× d\_model) |
| `c` | 512 | Reference context length (5.12 s) |
| `w` | 128 | Live window length (1.28 s) |
| `fps` | 100 | Frames per second |
| `N_ctx` | 128 | Context patches |
| `N_win` | 32 | Window patches |
| `N_valid` | 97 | Output positions |

Total parameter count: approximately **320 000** (two `PatchEncoder` instances with separate weights).

---

### 7. Complexity and Real-Time Suitability

| Operation | Cost per frame |
|---|---|
| CQT feature extraction | ~1–2 ms (CPU) |
| `E_live` forward pass | O(N_win² · d\_model) attention |
| Context slice retrieval | O(N_ctx · d\_model) copy |
| `encode_ctx_slice` (PE + transformer) | O(N_ctx² · d\_model) attention |
| Dot-product matching | O(N_ctx · d\_model) |
| Heuristic decision | O(N_valid) |

The heaviest transformer computation (over `N_ctx = 128` patches) involves only 128² = 16 384 attention pairs per layer — trivial on GPU. In practice, the entire `process_frame` call executes well under 10 ms on a modern GPU, comfortably within real-time constraints at the 100 fps frame rate.

Reference pre-encoding is performed once in `load_reference()` (Conv1d pass over the entire MIDI duration), after which only a 128-patch slice is transformer-encoded per inference step.
