"""Board-tensor dataset for the compact multiclass stylometry classifier.

This module turns raw PGN games into one neural example per game. The important
design choice is that we keep both the target player's moves and the opponent's
moves in chronological order, then add role planes so the model can learn the
target player's decisions in context.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import chess.pgn
import torch
from torch.utils.data import Dataset

LABEL_NAMES = [
    "Anand",
    "Carlsen",
    "Caruana",
    "Duda",
    "Erigaisi",
    "Firouzja",
    "Praggnanandhaa",
    "Tal",
]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}

PIECE_TO_RELATIVE_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

BASE_CHANNELS = 34
CONTEXT_CHANNELS = 4
STOCKFISH_CHANNELS = 7
INPUT_CHANNELS = BASE_CHANNELS + CONTEXT_CHANNELS + STOCKFISH_CHANNELS


@dataclass(frozen=True)
class GameRecord:
    """Metadata that identifies one usable game inside one PGN file."""

    pgn_path: Path
    game_index: int
    label_name: str
    label_id: int
    target_color: bool
    ply_count: int


@dataclass(frozen=True)
class TensorConfig:
    """Settings that control how one game becomes one tensor sequence."""

    sequence_length: int = 80
    stockfish_depth: int = 2
    multipv: int = 3
    board_flip: bool = True


class ChessStyleDataset(Dataset):
    """Minimal PyTorch dataset for precomputed game tensors."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]


def normalize_player_name(name: str) -> str:
    """Make PGN header names and file labels comparable."""

    return re.sub(r"[^a-z0-9]", "", name.lower())


def target_color_for_game(game: chess.pgn.Game, label_name: str) -> bool | None:
    """Return the side played by the labeled player in this game."""

    target = normalize_player_name(label_name)
    white = normalize_player_name(game.headers.get("White", ""))
    black = normalize_player_name(game.headers.get("Black", ""))

    if target and target in white:
        return chess.WHITE
    if target and target in black:
        return chess.BLACK
    return None


def iter_pgn_files(data_dir: Path) -> Iterable[Path]:
    """Yield only the PGN files whose stems match the configured labels."""

    for path in sorted(data_dir.glob("*.pgn")):
        if path.stem in LABEL_TO_ID:
            yield path


def count_mainline_plies(game: chess.pgn.Game) -> int:
    """Count all half-moves in the main line."""

    return sum(1 for _ in game.mainline_moves())


def collect_game_records(data_dir: Path, min_plies: int = 1) -> list[GameRecord]:
    """Scan PGN_data and collect games whose target player is identifiable."""

    records: list[GameRecord] = []

    for pgn_path in iter_pgn_files(data_dir):
        label_name = pgn_path.stem
        label_id = LABEL_TO_ID[label_name]

        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
            game_index = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                target_color = target_color_for_game(game, label_name)
                ply_count = count_mainline_plies(game)
                if target_color is not None and ply_count >= min_plies:
                    records.append(
                        GameRecord(
                            pgn_path=pgn_path,
                            game_index=game_index,
                            label_name=label_name,
                            label_id=label_id,
                            target_color=target_color,
                            ply_count=ply_count,
                        )
                    )
                game_index += 1

    return records


def balance_records(records: list[GameRecord], max_games_per_player: int, seed: int) -> list[GameRecord]:
    """Downsample every class to the same number of games."""

    rng = random.Random(seed)
    by_label: dict[str, list[GameRecord]] = {}
    for record in records:
        by_label.setdefault(record.label_name, []).append(record)

    if set(by_label) != set(LABEL_NAMES):
        missing = sorted(set(LABEL_NAMES) - set(by_label))
        raise ValueError(f"Missing usable games for labels: {missing}")

    target_count = min(min(len(items) for items in by_label.values()), max_games_per_player)
    balanced: list[GameRecord] = []
    for label_name in LABEL_NAMES:
        items = list(by_label[label_name])
        rng.shuffle(items)
        balanced.extend(items[:target_count])

    rng.shuffle(balanced)
    return balanced


def split_records(
    records: list[GameRecord],
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, list[GameRecord]]:
    """Make a stratified game-level split to prevent leakage across splits."""

    rng = random.Random(seed)
    by_label: dict[str, list[GameRecord]] = {}
    for record in records:
        by_label.setdefault(record.label_name, []).append(record)

    splits = {"train": [], "val": [], "test": []}
    for label_name in LABEL_NAMES:
        items = list(by_label[label_name])
        rng.shuffle(items)
        train_end = int(len(items) * train_frac)
        val_end = train_end + int(len(items) * val_frac)
        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])

    for items in splits.values():
        rng.shuffle(items)
    return splits


def class_counts(records: list[GameRecord]) -> dict[str, int]:
    """Return readable class counts for logging."""

    counts = Counter(record.label_name for record in records)
    return {name: counts.get(name, 0) for name in LABEL_NAMES}


def square_to_coords(square: chess.Square, flip: bool) -> tuple[int, int]:
    """Map a python-chess square id to tensor row/column coordinates."""

    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)

    if flip:
        file_idx = 7 - file_idx
        rank_idx = 7 - rank_idx

    return 7 - rank_idx, file_idx


def piece_planes(board: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode pieces as target-relative planes.

    The first six planes are the target player's pieces and the next six are
    the opponent's pieces. If the target is Black, the board is rotated so the
    target side is still shown from the bottom.
    """

    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK

    for square, piece in board.piece_map().items():
        base_channel = PIECE_TO_RELATIVE_CHANNEL[piece.piece_type]
        channel = base_channel if piece.color == target_color else base_channel + 6
        row, col = square_to_coords(square, flip)
        planes[channel, row, col] = 1.0

    return planes


def metadata_planes(board: chess.Board, target_color: bool) -> torch.Tensor:
    """Create global board-state planes that match the old 34-channel setup."""

    planes = torch.zeros((10, 8, 8), dtype=torch.float32)
    opponent = not target_color

    # Repeating scalar metadata across 8x8 lets the CNN consume these facts
    # without a separate non-board feature branch.
    planes[0].fill_(float(board.is_repetition(2)))
    planes[1].fill_(float(board.is_repetition(3)))
    planes[2].fill_(float(board.has_kingside_castling_rights(target_color)))
    planes[3].fill_(float(board.has_queenside_castling_rights(target_color)))
    planes[4].fill_(float(board.has_kingside_castling_rights(opponent)))
    planes[5].fill_(float(board.has_queenside_castling_rights(opponent)))
    planes[6].fill_(float(target_color == chess.WHITE))
    planes[7].fill_(min(board.halfmove_clock / 100.0, 1.0))
    planes[8].fill_(min(board.fullmove_number / 200.0, 1.0))
    planes[9, 0, :] = 1.0
    planes[9, 7, :] = 1.0
    planes[9, :, 0] = 1.0
    planes[9, :, 7] = 1.0
    return planes


def base_move_tensor(before: chess.Board, after: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Build the 34 paper-style channels for one before/after move pair."""

    return torch.cat(
        [
            piece_planes(before, target_color=target_color, board_flip=board_flip),
            piece_planes(after, target_color=target_color, board_flip=board_flip),
            metadata_planes(before, target_color=target_color),
        ],
        dim=0,
    )


def score_to_cp(score: chess.engine.PovScore, perspective: bool) -> float:
    """Convert a Stockfish score into centipawns from one side's perspective."""

    cp = score.pov(perspective).score(mate_score=100000)
    return float(cp if cp is not None else 0.0)


def normalize_eval_cp(value: float, scale: float = 1000.0) -> float:
    """Squash centipawn values to a stable range for neural input planes."""

    return max(min(value / scale, 10.0), -10.0)


def eval_cache_key(board: chess.Board, target_color: bool, multipv: int) -> tuple[str, bool, int]:
    """Key Stockfish results by position, perspective, and requested multipv."""

    return board.fen(), target_color, multipv


def analyse_position(
    board: chess.Board,
    target_color: bool,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    multipv: int,
    cache: dict[tuple[str, bool, int], tuple[float, list[chess.Move]]],
) -> tuple[float, list[chess.Move]]:
    """Return eval and top engine moves for one position, using a small cache."""

    key = eval_cache_key(board, target_color, multipv)
    if key in cache:
        return cache[key]

    try:
        info = engine.analyse(board, limit, multipv=multipv)
        if isinstance(info, dict):
            info = [info]

        eval_cp = score_to_cp(info[0]["score"], target_color) if info else 0.0
        top_moves = []
        for item in info[:multipv]:
            pv = item.get("pv") or []
            if pv:
                top_moves.append(pv[0])
    except Exception:
        eval_cp = 0.0
        top_moves = []

    cache[key] = (eval_cp, top_moves)
    return cache[key]


def context_planes(mover: bool, target_color: bool) -> torch.Tensor:
    """Encode whose move this ply is as board-shaped scalar planes."""

    planes = torch.zeros((CONTEXT_CHANNELS, 8, 8), dtype=torch.float32)
    is_target_move = mover == target_color
    planes[0].fill_(float(is_target_move))
    planes[1].fill_(float(not is_target_move))
    planes[2].fill_(float(mover == chess.WHITE))
    planes[3].fill_(float(target_color == chess.WHITE))
    return planes


def stockfish_planes(
    before: chess.Board,
    after: chess.Board,
    move: chess.Move,
    target_color: bool,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    multipv: int,
    board_flip: bool,
    cache: dict[tuple[str, bool, int], tuple[float, list[chess.Move]]],
) -> torch.Tensor:
    """Map Stockfish evaluation into extra 8x8 channels.

    Three scalar planes carry before/after/delta evaluation. One heatmap marks
    the played move's destination with eval delta, and three heatmaps mark
    Stockfish's top-1, top-2, and top-3 destination squares.
    """

    planes = torch.zeros((STOCKFISH_CHANNELS, 8, 8), dtype=torch.float32)
    before_eval, top_moves = analyse_position(before, target_color, engine, limit, multipv, cache)
    after_eval, _ = analyse_position(after, target_color, engine, limit, multipv, cache)
    eval_delta = after_eval - before_eval

    planes[0].fill_(normalize_eval_cp(before_eval))
    planes[1].fill_(normalize_eval_cp(after_eval))
    planes[2].fill_(normalize_eval_cp(eval_delta, scale=300.0))

    flip = board_flip and target_color == chess.BLACK
    row, col = square_to_coords(move.to_square, flip)
    planes[3, row, col] = normalize_eval_cp(eval_delta, scale=300.0)

    for rank, top_move in enumerate(top_moves[:3]):
        row, col = square_to_coords(top_move.to_square, flip)
        planes[4 + rank, row, col] = 1.0

    return planes


def game_to_tensor_sample(
    game: chess.pgn.Game,
    record: GameRecord,
    engine: chess.engine.SimpleEngine,
    config: TensorConfig,
    cache: dict[tuple[str, bool, int], tuple[float, list[chess.Move]]],
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Convert one PGN game to a padded full-context tensor sequence."""

    move_tensors: list[torch.Tensor] = []
    board = game.board()
    limit = chess.engine.Limit(depth=config.stockfish_depth)

    for move in game.mainline_moves():
        mover = board.turn
        before = board.copy(stack=False)
        board.push(move)
        after = board.copy(stack=False)

        base = base_move_tensor(before, after, record.target_color, config.board_flip)
        context = context_planes(mover, record.target_color)
        stockfish = stockfish_planes(
            before,
            after,
            move,
            record.target_color,
            engine,
            limit,
            config.multipv,
            config.board_flip,
            cache,
        )
        move_tensors.append(torch.cat([base, context, stockfish], dim=0))

        if len(move_tensors) == config.sequence_length:
            break

    if not move_tensors:
        return None

    valid_moves = len(move_tensors)
    sequence = torch.zeros((config.sequence_length, INPUT_CHANNELS, 8, 8), dtype=torch.float16)
    sequence[:valid_moves] = torch.stack(move_tensors, dim=0).to(dtype=torch.float16)

    # PyTorch Transformer convention: True marks padded timesteps to ignore.
    padding_mask = torch.ones((config.sequence_length,), dtype=torch.bool)
    padding_mask[:valid_moves] = False
    return sequence, padding_mask, record.label_id


def materialize_samples(
    records: list[GameRecord],
    engine: chess.engine.SimpleEngine,
    config: TensorConfig,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Read selected games and convert them into tensors.

    Records are grouped by PGN file so each large multi-game file is scanned
    once per split instead of once per selected game.
    """

    by_path: dict[Path, dict[int, GameRecord]] = {}
    for record in records:
        by_path.setdefault(record.pgn_path, {})[record.game_index] = record

    samples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    cache: dict[tuple[str, bool, int], tuple[float, list[chess.Move]]] = {}
    completed = 0
    total = len(records)

    for pgn_path, selected_games in by_path.items():
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
            game_index = 0
            while selected_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                record = selected_games.pop(game_index, None)
                if record is not None:
                    sample = game_to_tensor_sample(game, record, engine, config, cache)
                    if sample is not None:
                        samples.append(sample)

                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"  materialized {completed}/{total} games", flush=True)

                game_index += 1

        if selected_games:
            missing = ", ".join(str(index) for index in sorted(selected_games))
            raise ValueError(f"Could not find selected games in {pgn_path}: {missing}")

    random.shuffle(samples)
    return samples
