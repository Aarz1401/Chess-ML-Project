"""Shared PGN identity, exact player matching, and deterministic grouped splits."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chess
import chess.pgn


def matches_player(name: str, label: str) -> bool:
    key = re.sub(r"[^a-z0-9]", "", label.lower())
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return bool(key) and (
        key in tokens or key == re.sub(r"[^a-z0-9]", "", name.lower())
    )


def target_color(headers: dict, label: str) -> bool | None:
    sides = [
        color
        for field, color in (("White", chess.WHITE), ("Black", chess.BLACK))
        if matches_player(headers.get(field, ""), label)
    ]
    return sides[0] if len(sides) == 1 else None


def canonical_game_id(game: chess.pgn.Game) -> str:
    identity = (
        game.board().fen()
        + " "
        + " ".join(move.uci() for move in game.mainline_moves())
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def split_for_game(
    game_id: str, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1
) -> str:
    if not game_id:
        raise ValueError("A canonical game ID is required for a leakage-safe split")
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Invalid split fractions")
    value = (
        int(hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()[:16], 16) / 2**64
    )
    return (
        "train"
        if value < train_frac
        else "val" if value < train_frac + val_frac else "test"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
