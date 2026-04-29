"""
PatchFormer — transformer encoder pair for real-time score following.

Architecture
------------
Two independent encoders (E_ref, E_live) with identical structure but
separate weights:
  Conv1d patch embedding (128 freq bins → d_model, stride=patch_size)
  + sinusoidal positional encoding
  + 2-layer pre-LN TransformerEncoder

Matching is a dot product between mean-pooled live embedding and each
reference context patch embedding, producing N_valid_positions logits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe  # [max_len, d_model]


class PatchEncoder(nn.Module):
    """
    Patch embedding + sinusoidal PE + TransformerEncoder.

    Input : [B, in_channels, T]
    Output: [B, T//patch_size, d_model]
    """

    def __init__(
        self,
        in_channels: int = 128,
        d_model: int = 128,
        patch_size: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embed = nn.Conv1d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        pe = _make_sinusoidal_pe(max_seq_len, d_model)
        self.register_buffer("pe", pe)  # [max_seq_len, d_model]

    def patch_embed_only(self, x: torch.Tensor) -> torch.Tensor:
        """Conv1d patch embedding without PE or transformer. [B,C,T] → [B,N,d]"""
        return self.patch_embed(x).transpose(1, 2)  # [B, N, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode: patch embed + PE + transformer. [B,C,T] → [B,N,d]"""
        patches = self.patch_embed(x).transpose(1, 2)  # [B, N, d_model]
        N = patches.size(1)
        patches = patches + self.pe[:N]
        return self.transformer(patches)


class TransformerNet(nn.Module):
    """
    PatchFormer: two separate PatchEncoders (E_ref, E_live) with dot-product matching.

    Training
    --------
    forward(ctx, win) → logits [B, N_valid]
        ctx: [B, 128, C]  — piano roll reference context
        win: [B, 128, W]  — CQT live window
        N_valid = N_ctx - N_win + 1  (positions where window fits in context)

    Inference helpers
    -----------------
    encode_ref_raw(ctx)     → raw patch embeddings [B, N, d] (no PE, no transformer)
    encode_ctx_slice(raw)   → [B, N, d]  (PE + transformer on a pre-computed raw slice)
    encode_live(win)        → live_emb [B, d]  (full E_live pipeline + mean pool)
    match(ref_patches, live_emb) → logits [B, N_valid]
    """

    def __init__(
        self,
        d_model: int = 128,
        patch_size: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        c: int = 512,
        w: int = 128,
        in_channels: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.c = c
        self.w = w
        self.N_ctx = c // patch_size
        self.N_win = w // patch_size
        self.N_valid = self.N_ctx - self.N_win + 1

        shared_kw = dict(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max(self.N_ctx, self.N_win) + 16,
        )
        self.E_ref = PatchEncoder(**shared_kw)
        self.E_live = PatchEncoder(**shared_kw)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"[PatchFormer] d_model={d_model}  patch_size={patch_size}  "
            f"n_layers={n_layers}  c={c}  w={w}  "
            f"N_ctx={self.N_ctx}  N_win={self.N_win}  N_valid={self.N_valid}  "
            f"params={n_params:,}"
        )

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(self, ctx: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
        """
        ctx: [B, 128, C]  win: [B, 128, W]  →  logits [B, N_valid]
        """
        ref_patches = self.E_ref(ctx)            # [B, N_ctx, d_model]
        live_patches = self.E_live(win)           # [B, N_win, d_model]
        live_emb = live_patches.mean(dim=1)       # [B, d_model]
        return self._score(ref_patches, live_emb) # [B, N_valid]

    # ── Inference helpers ─────────────────────────────────────────────────────

    def encode_ref_raw(self, ctx: torch.Tensor) -> torch.Tensor:
        """Patch embedding only (no PE, no transformer). [B,128,T] → [B,N,d]"""
        return self.E_ref.patch_embed_only(ctx)

    def encode_ctx_slice(self, raw_patches: torch.Tensor) -> torch.Tensor:
        """
        Apply PE + transformer to a pre-computed raw patch slice.
        raw_patches: [B, N_ctx, d_model] (output of encode_ref_raw)
        Returns: [B, N_ctx, d_model]
        """
        N = raw_patches.size(1)
        x = raw_patches + self.E_ref.pe[:N]
        return self.E_ref.transformer(x)

    def encode_live(self, win: torch.Tensor) -> torch.Tensor:
        """Full E_live forward + mean pool. [B,128,W] → [B, d_model]"""
        live_patches = self.E_live(win)           # [B, N_win, d_model]
        return live_patches.mean(dim=1)           # [B, d_model]

    def match(
        self, ref_patches: torch.Tensor, live_emb: torch.Tensor
    ) -> torch.Tensor:
        """Dot-product scores. ref_patches: [B,N,d]  live_emb: [B,d] → [B,N_valid]"""
        return self._score(ref_patches, live_emb)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score(
        self, ref_patches: torch.Tensor, live_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        ref_patches: [B, N_ctx, d_model]
        live_emb:    [B, d_model]
        Returns:     [B, N_valid]  — only positions where window fits inside context
        """
        scores = torch.bmm(ref_patches, live_emb.unsqueeze(-1)).squeeze(-1)  # [B, N_ctx]
        return scores[:, : self.N_valid]
