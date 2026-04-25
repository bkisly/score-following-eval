import torch
import torch.nn as nn

from models.transformer.config import TransformerConfig


class LiveEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.cqt_n_bins, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, live_cqt: torch.Tensor):
        """
        live_cqt: [B, T, F]
        returns:
            live_emb_seq: [B, T, D]
            live_query: [B, D]
        """
        x = self.input_proj(live_cqt)
        x = self.encoder(x)
        x = self.norm(x)
        return x, x[:, -1, :]

