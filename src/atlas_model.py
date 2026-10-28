"""Small temporal metric model over frozen, target-relative chess decisions."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class AtlasEncoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        width=128,
        embedding_dim=128,
        target_dim=288,
        layers=2,
        dropout=0.1,
        num_players=0,
    ):
        super().__init__()
        self.config = dict(
            input_dim=input_dim,
            width=width,
            embedding_dim=embedding_dim,
            target_dim=target_dim,
            layers=layers,
            dropout=dropout,
            num_players=num_players,
        )
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width,
                4,
                width * 2,
                dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            layers,
        )
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, embedding_dim)
        )
        self.concepts = nn.Linear(embedding_dim, target_dim)
        self.player_head = (
            nn.Linear(embedding_dim, num_players) if num_players else None
        )
        self.register_buffer(
            "frequencies",
            torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width)),
        )

    def forward(self, sequence, mask, plies):
        if bool(mask.all(1).any()):
            raise ValueError("A game has no valid decision tokens")
        x = self.input(sequence.float())
        phase = plies.float().unsqueeze(-1) / 2 * self.frequencies
        position = torch.zeros_like(x)
        position[:, :, 0::2] = phase.sin()
        position[:, :, 1::2] = phase.cos()
        x = self.transformer(x + position, src_key_padding_mask=mask)
        scores = (x @ self.query) / math.sqrt(x.shape[-1])
        scores = scores.masked_fill(mask, float("-inf"))
        pooled = (x * scores.softmax(1).unsqueeze(-1)).sum(1)
        embedding = F.normalize(self.projection(pooled), dim=1)
        return embedding, self.concepts(embedding)


def paired_contrastive(a, b, game_groups, temperature=0.1):
    """Same-view pair positives; alternate perspectives of a game are not negatives."""
    n = len(a)
    z = torch.cat([a, b]).float()
    logits = z @ z.T / temperature
    same = game_groups[:, None].eq(game_groups[None, :]).repeat(2, 2)
    positives = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    allowed = ~same
    allowed[torch.arange(2 * n, device=z.device), positives] = True
    logits = logits.masked_fill(~allowed, float("-inf"))
    return F.cross_entropy(logits, positives)


def player_contrastive(a, b, labels, temperature=0.1):
    z = torch.cat([a, b]).float()
    labels = labels.repeat(2)
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    logits = logits.masked_fill(diagonal, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(
        log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1)
    ).mean()
