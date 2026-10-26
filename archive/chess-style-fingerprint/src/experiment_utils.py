"""Small helpers shared by experiment entry points."""

from __future__ import annotations

import random

import chess.pgn
import torch

from src.dataset import GameRecord, _open_pgn, game_to_windows, records_to_windows

MODEL_PRESETS = {
    # These presets let scripts trade speed for capacity without rewriting the
    # ModelC constructor arguments each time.
    "tiny": {
        "n_layers": 2,
        "n_heads": 4,
        "conv_channels": 32,
        "residual_blocks": 2,
        "move_dim": 128,
        "transformer_dim": 256,
        "feedforward_dim": 512,
        "embedding_dim": 128,
    },
    "small": {
        "n_layers": 4,
        "n_heads": 4,
        "conv_channels": 48,
        "residual_blocks": 3,
        "move_dim": 192,
        "transformer_dim": 512,
        "feedforward_dim": 1024,
        "embedding_dim": 256,
    },
    "paper": {
        "n_layers": 12,
        "n_heads": 8,
        "conv_channels": 64,
        "residual_blocks": 6,
        "move_dim": 320,
        "transformer_dim": 1024,
        "feedforward_dim": 2048,
        "embedding_dim": 512,
    },
}


def active_labels(all_labels: dict[str, int], exclude_labels: set[str]) -> dict[str, int]:
    """Build contiguous label ids after excluding classes from a run."""

    names = [name for name in all_labels if name not in exclude_labels]
    return {name: idx for idx, name in enumerate(names)}


def remap_record(record: GameRecord, labels: dict[str, int]) -> GameRecord:
    """Return a copy of a record with the active contiguous label id."""

    return GameRecord(
        pgn_path=record.pgn_path,
        game_index=record.game_index,
        label=labels[record.label_name],
        label_name=record.label_name,
        ply_count=record.ply_count,
        target_color=record.target_color,
    )


def record_to_windows(
    record: GameRecord,
    window_size: int,
    stride: int,
    skip_moves: int,
    max_windows_per_game: int | None = None,
) -> torch.Tensor:
    """Materialize windows for one game record."""

    windows = records_to_windows(
        [record],
        window_size=window_size,
        stride=stride,
        skip_moves=skip_moves,
        max_windows_per_game=max_windows_per_game,
    )
    if not windows:
        raise ValueError(f"No windows extracted for {record.pgn_path} game {record.game_index}")
    return torch.stack([window for window, _ in windows], dim=0)


def precompute_game_windows(
    records: list[GameRecord],
    window_size: int,
    stride: int,
    skip_moves: int,
    max_windows_per_game: int | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """Materialize windows once per selected game.

    Records are grouped by PGN path so a multi-game PGN file is scanned once,
    instead of reopening the same archive separately for every selected game.
    """

    examples: list[tuple[torch.Tensor, int]] = []

    # Group by PGN path so a file with many games is parsed once.
    records_by_path: dict[str, dict[int, GameRecord]] = {}
    for record in records:
        records_by_path.setdefault(record.pgn_path, {})[record.game_index] = record

    completed = 0
    total = len(records)
    for pgn_path, selected_games in records_by_path.items():
        with _open_pgn(pgn_path) as pgn_file:
            game_index = 0
            while selected_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                record = selected_games.pop(game_index, None)
                if record is not None:
                    windows = game_to_windows(
                        game,
                        label=record.label,
                        window_size=window_size,
                        stride=stride,
                        skip_moves=skip_moves,
                        target_color=record.target_color,
                        max_windows=max_windows_per_game,
                    )
                    if not windows:
                        raise ValueError(f"No windows extracted for {record.pgn_path} game {record.game_index}")
                    examples.append((torch.stack([window for window, _ in windows], dim=0), record.label))
                    completed += 1
                    if completed % 50 == 0 or completed == total:
                        print(f"  precomputed {completed}/{total} games", flush=True)

                game_index += 1

        if selected_games:
            missing = ", ".join(str(index) for index in sorted(selected_games))
            raise ValueError(f"Could not find selected game indices in {pgn_path}: {missing}")

    random.shuffle(examples)
    return examples
