"""Neural architectures used for Model 1.

Move tensors are encoded with CNN blocks, move sequences are aggregated with a
Transformer, and the hybrid model fuses that board embedding with the
33-dimensional style_trajectory feature vector.
"""

import torch
import torch.nn as nn

MOVE_CHANNELS = 34


class BoardEncoder(nn.Module):
    """Small baseline CNN that turns one move tensor into a 256-d vector."""

    def __init__(self, in_channels: int = MOVE_CHANNELS) -> None:
        super().__init__()
        # Keep spatial size at 8x8 with padding=1 while increasing feature
        # channels. The final global average pool removes board dimensions.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:  (batch, channels, 8, 8)
        # Output: (batch, 256)
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class ModelA(nn.Module):
    """Simplest neural baseline: encode moves independently and average them."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.encoder = BoardEncoder()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = x.shape

        # Run the CNN on every board position as if it were an independent
        # image, then restore the original sequence shape.
        x = x.view(batch_size * timesteps, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, timesteps, 256)

        # Mean pooling is a simple "average style over this window" baseline.
        x = x.mean(dim=1)
        return self.classifier(x)


class SqueezeExcitation(nn.Module):
    """Channel attention block used by the stylometry/Maia-style encoder."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The gate learns which channels are useful for this position. This is
        # a lightweight attention mechanism over convolutional feature maps.
        return x * self.gate(self.pool(x))


class ResidualSEBlock(nn.Module):
    """Residual conv block with squeeze-and-excitation channel gating."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            SqueezeExcitation(channels, reduction=reduction),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection preserves the input representation and lets the
        # block learn a correction. This makes deeper CNN stacks easier to train.
        return self.relu(x + self.block(x))


class StylometryMoveEncoder(nn.Module):
    """Compact version of the paper's residual-SE move feature extractor."""

    def __init__(
        self,
        in_channels: int = MOVE_CHANNELS,
        channels: int = 64,
        n_blocks: int = 6,
        out_dim: int = 320,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualSEBlock(channels) for _ in range(n_blocks)))
        self.head = nn.Sequential(
            nn.Conv2d(channels, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each input is one 34x8x8 move tensor. The CNN extracts local board
        # patterns, then global average pooling compresses the board to one
        # move vector.
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        return x.flatten(1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encoding for variable-length windows."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        # Fixed Transformer position encoding: move 1 and move 10 should not be
        # interchangeable, even if their board tensors look similar.
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ModelC(nn.Module):
    """Paper-inspired neural stylometry architecture.

    This follows the structure of behavioral stylometry in chess: encode each
    move, aggregate the sequence, then classify the player/source.
    """

    def __init__(
        self,
        num_classes: int,
        n_layers: int = 12,
        n_heads: int = 8,
        max_timesteps: int = 500,
        conv_channels: int = 64,
        residual_blocks: int = 6,
        move_dim: int = 320,
        transformer_dim: int = 1024,
        feedforward_dim: int = 2048,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = StylometryMoveEncoder(
            in_channels=MOVE_CHANNELS,
            channels=conv_channels,
            n_blocks=residual_blocks,
            out_dim=move_dim,
        )
        self.input_projection = nn.Linear(move_dim, transformer_dim)
        self.position = SinusoidalPositionalEncoding(d_model=transformer_dim, max_len=max_timesteps)

        # The Transformer models dependencies across moves in the window. This
        # is the part that lets the model look for sequential style patterns
        # instead of only isolated move snapshots.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embedding = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.move_dim = move_dim

    def embed(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return one style embedding for each move/game sequence.

        padding_mask is optional. When supplied, it has shape (batch, timesteps)
        with True marking padded positions. This lets the same ModelC support
        padded whole-game input and the short-window ablation.
        """

        batch_size, timesteps, channels, height, width = x.shape

        # Flatten time into the batch dimension so the same CNN processes every
        # move tensor independently.
        x = x.view(batch_size * timesteps, channels, height, width)
        x = self.encoder(x)

        # Restore sequence shape before the Transformer.
        x = x.view(batch_size, timesteps, self.move_dim)

        # The stylometry paper reports better results from averaging all
        # transformer outputs than from using a CLS token.
        x = self.input_projection(x)
        x = self.position(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        if padding_mask is None:
            x = x.mean(dim=1)
        else:
            # Average only real moves. Without this masked mean, padded zero
            # moves could become a fake style signal.
            valid = (~padding_mask).unsqueeze(-1).float()
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


class HybridStockfishModel(nn.Module):
    """Fuse board-sequence embeddings with engine-assisted style vectors.

    One branch keeps the CNN+Transformer board encoder. The second branch
    learns from compact game-level trajectory features.
    """

    def __init__(
        self,
        num_classes: int,
        stockfish_feature_dim: int = 33,
        stockfish_hidden_dim: int = 64,
        board_embedding_dim: int = 512,
        fusion_hidden_dim: int = 256,
        board_encoder_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        board_encoder_kwargs = board_encoder_kwargs or {}

        # Reuse ModelC as a feature extractor. Its classifier head is ignored;
        # HybridStockfishModel calls board_encoder.embed(...) directly.
        self.board_encoder = ModelC(
            num_classes=num_classes,
            embedding_dim=board_embedding_dim,
            **board_encoder_kwargs,
        )
        self.stockfish_encoder = nn.Sequential(
            # LayerNorm makes handcrafted features easier for the MLP to use
            # because they arrive on different natural scales.
            nn.LayerNorm(stockfish_feature_dim),
            nn.Linear(stockfish_feature_dim, stockfish_hidden_dim),
            nn.GELU(),
            nn.Linear(stockfish_hidden_dim, stockfish_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(stockfish_hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(board_embedding_dim + stockfish_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(
        self,
        board_windows: torch.Tensor,
        stockfish_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # board_windows:       (batch, timesteps, 34, 8, 8)
        # stockfish_features:  (batch, stockfish_feature_dim)
        # padding_mask:        optional (batch, timesteps), True for padding
        board_embedding = self.board_encoder.embed(board_windows, padding_mask=padding_mask)
        stockfish_embedding = self.stockfish_encoder(stockfish_features)

        # Fusion step: concatenate learned board style with engineered
        # trajectory style, then classify.
        fused = torch.cat([board_embedding, stockfish_embedding], dim=1)
        return self.classifier(fused)
