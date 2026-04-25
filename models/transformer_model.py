import time
from tempfile import mkdtemp
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from models.score_follower import ScoreFollower
from models.transformer import (
    LiveEncoder,
    TransformerConfig,
    TransformerRuntimeState,
    alignment_probs,
    build_reference_from_midi,
    compute_alignment_logits,
    confidence_from_probs,
    estimate_tempo,
    extract_live_cqt,
    select_window,
    update_position,
)
from models.transformer.training import load_maestro_entries, train_epoch
from models.transformer.training import prepare_reference_cqt_cache


class TransformerModel(ScoreFollower):
    """
    Streaming score follower (V1): CQT-to-CQT alignment with a lightweight transformer.
    """

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        device: Optional[str] = None,
    ):
        super().__init__(name="Transformer-V1-CQT")
        self.config = config or TransformerConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.input_proj = torch.nn.Linear(self.config.cqt_n_bins, self.config.d_model).to(self.device)
        self.encoder = LiveEncoder(self.config).to(self.device)

        self.state = TransformerRuntimeState()
        self.is_trained = False

        n_params = sum(
            p.numel() for p in list(self.input_proj.parameters()) + list(self.encoder.parameters())
        )
        print(f"[TransformerModel] device={self.device} params={n_params:,}")

    def parameters_for_training(self):
        return list(self.input_proj.parameters()) + list(self.encoder.parameters())

    def requires_training(self) -> bool:
        return True

    def load_reference(self, reference_path: str) -> None:
        reference = build_reference_from_midi(
            reference_path=reference_path,
            config=self.config,
            device=self.device,
            input_proj=self.input_proj,
        )
        self.state.reference = reference
        self.state.current_ref_idx = 0
        self.state.prev_ref_idx = 0
        self.state.elapsed_seconds = 0.0
        self.state.audio_buffer = np.zeros(0, dtype=np.float32)
        self.state.initialized = True
        self.current_position = 0.0
        print(
            f"[TransformerModel] Reference loaded: {Path(reference_path).name}, "
            f"frames={len(reference.ref_times_sec)}"
        )

    def process_frame(self, audio_frame: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        t0 = time.time()
        if self.state.reference is None:
            raise RuntimeError("Call load_reference() before process_frame().")

        chunk = np.asarray(audio_frame, dtype=np.float32)
        self.state.elapsed_seconds += len(chunk) / float(sample_rate)
        elapsed_pos = float(
            np.clip(self.state.elapsed_seconds, 0.0, self.state.reference.ref_times_sec[-1])
        )

        # Untrained fallback: preserve API behavior and allow evaluator runs.
        if not self.is_trained:
            self.current_position = elapsed_pos
            return {
                "position": self.current_position,
                "confidence": 0.0,
                "tempo": 0.0,
                "latency": (time.time() - t0) * 1000.0,
            }

        live_cqt, self.state = extract_live_cqt(
            audio_chunk=chunk,
            state=self.state,
            config=self.config,
            sample_rate=sample_rate,
        )
        live_tensor = torch.from_numpy(live_cqt).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, live_query = self.encoder(live_tensor)
            ref = self.state.reference
            win_emb, win_idx, _, _ = select_window(
                ref_emb=ref.ref_emb.to(self.device),
                prev_idx=self.state.current_ref_idx,
                config=self.config,
                ref_hop_length=ref.hop_length,
                ref_sample_rate=ref.sample_rate,
            )
            logits = compute_alignment_logits(live_query, win_emb)
            probs = alignment_probs(logits)
            local_idx = int(torch.argmax(probs, dim=-1).item())
            candidate_idx = int(win_idx[local_idx])
            confidence = confidence_from_probs(probs, self.config.min_confidence_floor)

            new_idx = update_position(
                prev_idx=self.state.current_ref_idx,
                candidate_idx=candidate_idx,
                confidence=confidence,
                config=self.config,
                max_idx=len(ref.ref_times_sec) - 1,
            )
            tempo = estimate_tempo(
                prev_idx=self.state.current_ref_idx,
                new_idx=new_idx,
                chunk_size=len(chunk),
                sample_rate=sample_rate,
                ref_hop_length=ref.hop_length,
                ref_sample_rate=ref.sample_rate,
            )

        self.state.prev_ref_idx = self.state.current_ref_idx
        self.state.current_ref_idx = new_idx
        self.current_position = float(self.state.reference.ref_times_sec[new_idx])

        return {
            "position": self.current_position,
            "confidence": float(confidence),
            "tempo": float(tempo),
            "latency": (time.time() - t0) * 1000.0,
        }

    def reset(self) -> None:
        self.current_position = 0.0
        self.state.current_ref_idx = 0
        self.state.prev_ref_idx = 0
        self.state.elapsed_seconds = 0.0
        self.state.audio_buffer = np.zeros(0, dtype=np.float32)

    def train(self, train_data: Any = None, **kwargs) -> None:
        """
        Train the transformer model on MAESTRO.

        train_data:
            - str: dataset path
            - dict: config dictionary
        """
        cfg = {"dataset_path": train_data} if isinstance(train_data, str) else dict(train_data or {})
        cfg.update(kwargs)
        if "dataset_path" not in cfg:
            raise ValueError("train_data must provide 'dataset_path'.")

        dataset_path = cfg["dataset_path"]
        epochs = int(cfg.get("epochs", 30))
        lr = float(cfg.get("lr", 3e-4))
        weight_decay = float(cfg.get("weight_decay", 1e-2))
        samples_per_epoch = int(cfg.get("samples_per_epoch", 400))
        save_path = cfg.get("save_path")
        window_seconds = float(cfg.get("window_seconds", self.config.window_seconds))
        cache_overwrite = bool(cfg.get("cache_overwrite", False))
        reference_cache_dir = cfg.get("reference_cache_dir")
        if reference_cache_dir is None:
            reference_cache_dir = mkdtemp(prefix="transformer_ref_cqt_")

        train_entries = load_maestro_entries(dataset_path, split="train")
        if not train_entries:
            raise RuntimeError("No MAESTRO training entries found.")

        print(f"[TransformerModel] Preparing reference CQT cache in: {reference_cache_dir}")
        ref_cache_index = prepare_reference_cqt_cache(
            train_entries=train_entries,
            config=self.config,
            cache_dir=reference_cache_dir,
            model=self,
            overwrite=cache_overwrite,
        )
        if not ref_cache_index:
            raise RuntimeError("Reference CQT cache is empty. Cannot start training.")

        optimizer = torch.optim.AdamW(
            self.parameters_for_training(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_loss = float("inf")
        best_state = None
        print(
            f"[TransformerModel] Training start: epochs={epochs}, samples_per_epoch={samples_per_epoch}, "
            f"entries={len(train_entries)}"
        )
        for epoch in range(1, epochs + 1):
            stats = train_epoch(
                model=self,
                optimizer=optimizer,
                train_entries=train_entries,
                ref_cache_index=ref_cache_index,
                config=self.config,
                samples_per_epoch=samples_per_epoch,
                window_seconds=window_seconds,
            )
            loss = float(stats["loss"])
            steps = int(stats["steps"])
            print(f"[TransformerModel] Epoch {epoch}/{epochs} loss={loss:.4f} steps={steps}")
            if loss < best_loss:
                best_loss = loss
                best_state = {
                    "input_proj": {k: v.detach().cpu().clone() for k, v in self.input_proj.state_dict().items()},
                    "encoder": {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()},
                }

        if best_state is not None:
            self.input_proj.load_state_dict(best_state["input_proj"])
            self.encoder.load_state_dict(best_state["encoder"])

        self.input_proj.eval()
        self.encoder.eval()
        self.is_trained = True
        print(f"[TransformerModel] Training complete. best_loss={best_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path)
            print(f"[TransformerModel] Saved checkpoint to {save_path}")

    def save_checkpoint(self, path: str) -> None:
        ckpt = {
            "input_proj": self.input_proj.state_dict(),
            "encoder": self.encoder.state_dict(),
            "config": vars(self.config),
            "version": "transformer_v1_cqt",
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.input_proj.load_state_dict(ckpt["input_proj"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.input_proj.eval()
        self.encoder.eval()
        self.is_trained = True
        print(f"[TransformerModel] Loaded checkpoint '{path}'.")