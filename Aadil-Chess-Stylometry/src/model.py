"""CNN+Transformer encoder for chess style sequences.

The per-move encoder is an AlphaZero/Maia-style residual tower instead of a
plain conv stack, and games are pooled with a learned attention query rather
than a masked mean, so informative plies can dominate the game embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from board_dataset import INPUT_CHANNELS


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(x + out)


class MoveEncoder(nn.Module):
    """Encode one board-shaped move tensor into a dense vector.

    The same tower is applied to every ply in the sequence. A 1x1 reduction
    keeps the full 8x8 spatial layout in the final projection (square identity
    matters for style) without a giant flatten layer.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        channels: int = 64,
        n_blocks: int = 6,
        out_dim: int = 256,
        reduced_channels: int = 16,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduced_channels * 8 * 8, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.reduce(x)
        return self.projection(x)


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


class AttentionPooling(nn.Module):
    """Pool a masked sequence with one learned attention query."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = d_model**-0.5

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        scores = (x @ self.query) * self.scale
        scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class ChessStyleTransformer(nn.Module):
    """CNN+Transformer game encoder with classification and metric heads.

    Input shape is (batch, sequence_length, channels, 8, 8). The output is one
    set of logits per game, not per move.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = INPUT_CHANNELS,
        conv_channels: int = 64,
        n_blocks: int = 6,
        move_dim: int = 256,
        transformer_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        feedforward_dim: int = 1024,
        metric_dim: int = 128,
        dropout: float = 0.1,
        max_timesteps: int = 512,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError(f"Unknown pooling {pooling!r}, expected 'attention' or 'mean'")

        self.move_encoder = MoveEncoder(
            in_channels=in_channels,
            channels=conv_channels,
            n_blocks=n_blocks,
            out_dim=move_dim,
        )
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
        self.pooling = pooling
        self.attention_pool = AttentionPooling(transformer_dim) if pooling == "attention" else None
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, num_classes),
        )
        self.metric_projection = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, metric_dim),
        )
        self.move_dim = move_dim
        self.embedding_dim = transformer_dim
        self.metric_dim = metric_dim

    def config(self) -> dict:
        """Return the constructor settings needed to rebuild this model."""

        first_layer = self.transformer.layers[0]
        return {
            "in_channels": self.move_encoder.stem[0].in_channels,
            "conv_channels": self.move_encoder.stem[0].out_channels,
            "n_blocks": len(self.move_encoder.blocks),
            "move_dim": self.move_dim,
            "transformer_dim": self.embedding_dim,
            "n_layers": len(self.transformer.layers),
            "n_heads": first_layer.self_attn.num_heads,
            "feedforward_dim": first_layer.linear1.out_features,
            "metric_dim": self.metric_dim,
            "pooling": self.pooling,
        }

    def encode_game(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return one learned embedding vector per game.

        This is the raw pooled Transformer output just before the classifier
        and metric projection heads.
        """

        batch_size, timesteps, channels, height, width = x.shape

        # Flatten time into the batch dimension so the CNN treats every ply as
        # the same kind of board-like image.
        x = x.float().view(batch_size * timesteps, channels, height, width)
        x = self.move_encoder(x)
        x = x.view(batch_size, timesteps, self.move_dim)

        x = self.input_projection(x)
        x = self.position(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        if self.attention_pool is not None:
            return self.attention_pool(x, padding_mask)

        # Average only real plies. Padded zero tensors should never become a
        # style cue, so the same mask used by the Transformer is reused here.
        valid = (~padding_mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_game(x, padding_mask)
        return self.classify_embedding(embedding)

    def classify_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify already-computed game embeddings."""

        return self.classifier(embedding)

    def project_metric(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project pooled game embeddings into normalized similarity space."""

        return F.normalize(self.metric_projection(embedding), dim=1)

    def encode_metric(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Return embeddings intended for cosine retrieval and visualization."""

        return self.project_metric(self.encode_game(x, padding_mask))


def build_model_from_checkpoint(checkpoint: dict, num_classes: int) -> ChessStyleTransformer:
    """Rebuild a ChessStyleTransformer from a saved checkpoint payload."""

    model_config = dict(checkpoint.get("model_config", {}))
    model_config.setdefault("in_channels", int(checkpoint.get("input_channels", INPUT_CHANNELS)))
    model_config.setdefault("metric_dim", int(checkpoint.get("metric_dim", 128)))
    return ChessStyleTransformer(num_classes=num_classes, **model_config)
