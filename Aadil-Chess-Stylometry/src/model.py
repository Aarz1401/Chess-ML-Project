"""Compact CNN+Transformer classifier for chess style sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from board_dataset import INPUT_CHANNELS


class MoveEncoder(nn.Module):
    """Encode one board-shaped move tensor into a dense vector.

    The same CNN is applied to every ply in the sequence. It sees ordinary
    board channels, role/context planes, and Stockfish planes as one unified
    board-like representation.
    """

    def __init__(self, in_channels: int = INPUT_CHANNELS, channels: int = 32, out_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed position encoding so move order is visible to the Transformer."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ChessStyleTransformer(nn.Module):
    """Small CNN+Transformer multiclass classifier.

    Input shape is (batch, sequence_length, channels, 8, 8). The output is one
    set of logits per game, not per move.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = INPUT_CHANNELS,
        conv_channels: int = 32,
        move_dim: int = 128,
        transformer_dim: int = 192,
        n_layers: int = 2,
        n_heads: int = 4,
        feedforward_dim: int = 384,
        dropout: float = 0.1,
        max_timesteps: int = 512,
    ) -> None:
        super().__init__()
        self.move_encoder = MoveEncoder(in_channels=in_channels, channels=conv_channels, out_dim=move_dim)
        self.input_projection = nn.Linear(move_dim, transformer_dim)
        self.position = SinusoidalPositionalEncoding(transformer_dim, max_len=max_timesteps)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes),
        )
        self.move_dim = move_dim

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = x.shape

        # Flatten time into the batch dimension so the CNN treats every ply as
        # the same kind of board-like image.
        x = x.float().view(batch_size * timesteps, channels, height, width)
        x = self.move_encoder(x)
        x = x.view(batch_size, timesteps, self.move_dim)

        x = self.input_projection(x)
        x = self.position(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Average only real plies. Padded zero tensors should never become a
        # style cue, so the same mask used by the Transformer is reused here.
        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
