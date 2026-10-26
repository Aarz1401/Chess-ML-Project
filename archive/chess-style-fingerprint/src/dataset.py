"""PGN parsing and tensor construction for the neural stylometry models.

This module is the bridge between chess games and Model 1. It reads PGN games,
keeps the moves made by the labeled player, converts each move into a
34-channel before/after tensor, and builds fixed-length windows for training.
"""

import gzip
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, TextIO, Tuple

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

LABELS = {
    # Stable original label ids. Experiment scripts may remap these to a
    # smaller contiguous set after --include-label/--exclude-label filtering.
    "tal": 0,
    "petrosian": 1,
    "karpov": 2,
    "fischer": 3,
    "kasparov": 4,
    "carlsen": 5,
    "anand": 6,
    "botvinnik": 7,
    "capablanca": 8,
    "morphy": 9,
    "stockfish": 10,
}

PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

PIECE_TO_RELATIVE_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


@dataclass(frozen=True)
class GameRecord:
    """Metadata for one usable game inside a possibly multi-game PGN file."""

    pgn_path: str
    game_index: int
    label: int
    label_name: str
    ply_count: int
    target_color: bool | None


def _open_pgn(path: str) -> TextIO:
    """Open plain PGN files and compressed .pgn.gz files through one helper."""

    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def _map_square_to_tensor_coords(square: chess.Square, flip: bool) -> Tuple[int, int]:
    """Map python-chess square ids to tensor row/column coordinates."""

    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)

    # Rotate 180 degrees so side-to-move is always shown at the bottom.
    if flip:
        file_idx = 7 - file_idx
        rank_idx = 7 - rank_idx

    row = 7 - rank_idx
    col = file_idx
    return row, col


def board_to_tensor(board: chess.Board, perspective_color: bool | None = None) -> torch.Tensor:
    """Encode one chess board as 14 planes from one side's perspective."""

    tensor = torch.zeros((14, 8, 8), dtype=torch.float32)
    if perspective_color is None:
        perspective_color = board.turn
    flip = perspective_color == chess.BLACK

    # First fill absolute white/black piece planes.
    for square, piece in board.piece_map().items():
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        row, col = _map_square_to_tensor_coords(square, flip)
        tensor[channel, row, col] = 1.0

    # Attack planes expose tactical pressure that a shallow CNN may not infer
    # easily from piece placement alone.
    for square in chess.SQUARES:
        row, col = _map_square_to_tensor_coords(square, flip)
        if board.is_attacked_by(chess.WHITE, square):
            tensor[12, row, col] = 1.0
        if board.is_attacked_by(chess.BLACK, square):
            tensor[13, row, col] = 1.0

    if flip:
        # Swap white/black channels after rotation to keep channels relative to side-to-move.
        tensor[:6], tensor[6:12] = tensor[6:12].clone(), tensor[:6].clone()
        tensor[12], tensor[13] = tensor[13].clone(), tensor[12].clone()

    return tensor


def _piece_planes(board: chess.Board, perspective_color: bool, board_flip: bool = True) -> torch.Tensor:
    """Encode pieces as 12 side-relative planes from one player's perspective."""

    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    flip = board_flip and perspective_color == chess.BLACK

    for square, piece in board.piece_map().items():
        base_channel = PIECE_TO_RELATIVE_CHANNEL[piece.piece_type]
        channel = base_channel if piece.color == perspective_color else base_channel + 6
        row, col = _map_square_to_tensor_coords(square, flip)
        tensor[channel, row, col] = 1.0

    return tensor


def _metadata_planes(board: chess.Board, perspective_color: bool) -> torch.Tensor:
    """Create 10 simple global state planes to reach the paper's 34 channels."""

    tensor = torch.zeros((10, 8, 8), dtype=torch.float32)
    opponent = not perspective_color

    # These global facts are repeated across the 8x8 plane so they can be
    # concatenated with spatial piece planes without changing tensor shape.
    tensor[0].fill_(float(board.is_repetition(2)))
    tensor[1].fill_(float(board.is_repetition(3)))
    tensor[2].fill_(float(board.has_kingside_castling_rights(perspective_color)))
    tensor[3].fill_(float(board.has_queenside_castling_rights(perspective_color)))
    tensor[4].fill_(float(board.has_kingside_castling_rights(opponent)))
    tensor[5].fill_(float(board.has_queenside_castling_rights(opponent)))
    tensor[6].fill_(float(perspective_color == chess.WHITE))
    tensor[7].fill_(min(board.halfmove_clock / 100.0, 1.0))
    tensor[8].fill_(min(board.fullmove_number / 200.0, 1.0))
    tensor[9, 0, :] = 1.0
    tensor[9, 7, :] = 1.0
    tensor[9, :, 0] = 1.0
    tensor[9, :, 7] = 1.0

    return tensor


def move_to_tensor(
    before: chess.Board,
    after: chess.Board,
    perspective_color: bool,
    board_flip: bool = True,
) -> torch.Tensor:
    """Encode a move as 34 channels: before pieces, after pieces, metadata."""

    return torch.cat(
        [
            _piece_planes(before, perspective_color, board_flip=board_flip),
            _piece_planes(after, perspective_color, board_flip=board_flip),
            _metadata_planes(before, perspective_color),
        ],
        dim=0,
    )


def pgn_to_windows(
    pgn_path: str,
    label: int,
    window_size: int = 15,
    stride: int = 5,
    skip_moves: int = 10,
) -> List[Tuple[torch.Tensor, int]]:
    """Convert every game in one PGN file into labeled tensor windows."""

    samples: List[Tuple[torch.Tensor, int]] = []

    with _open_pgn(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            samples.extend(
                game_to_windows(
                    game,
                    label=label,
                    window_size=window_size,
                    stride=stride,
                    skip_moves=skip_moves,
                )
            )

    return samples


def game_to_windows(
    game: chess.pgn.Game,
    label: int,
    window_size: int = 15,
    stride: int = 5,
    skip_moves: int = 10,
    target_color: bool | None = None,
    max_windows: int | None = None,
    board_flip: bool = True,
) -> List[Tuple[torch.Tensor, int]]:
    """Convert one parsed game into overlapping windows of move tensors."""

    board = game.board()
    states: List[torch.Tensor] = []

    for move in game.mainline_moves():
        mover = board.turn
        before = board.copy(stack=False)
        board.push(move)
        # For human labels, keep only positions reached by the target player's
        # own moves. For Stockfish, target_color is None and all moves are kept.
        # Encode each move as before-board + after-board from the mover's
        # perspective, matching the stylometry paper's decision-focused setup.
        if target_color is None or mover == target_color:
            states.append(move_to_tensor(before, board, perspective_color=mover, board_flip=board_flip))

    # Skip early opening positions so the model sees more style-revealing play.
    states = states[skip_moves:]
    if len(states) < window_size:
        return []

    samples: List[Tuple[torch.Tensor, int]] = []
    for start in range(0, len(states) - window_size + 1, stride):
        # Each sample has shape (window_size, 34, 8, 8). The model later sees
        # this as a sequence of move decisions.
        window = torch.stack(states[start : start + window_size], dim=0)
        samples.append((window, label))

    if max_windows is not None and len(samples) > max_windows:
        # Keep windows spread across the whole game instead of only taking the
        # opening/middlegame prefix. This makes game-level experiments faster
        # without changing the tensor format.
        keep = torch.linspace(0, len(samples) - 1, steps=max_windows).long().tolist()
        samples = [samples[i] for i in keep]

    return samples


def _find_pgn_files(folder_path: str) -> Iterable[str]:
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            lower = file_name.lower()
            if lower.endswith(".pgn") or lower.endswith(".pgn.gz"):
                yield os.path.join(root, file_name)


def _count_mainline_plies(game: chess.pgn.Game) -> int:
    return sum(1 for _ in game.mainline_moves())


def _normalize_player_name(name: str) -> str:
    """Make PGN header names easier to compare with folder labels."""

    return re.sub(r"[^a-z0-9]", "", name.lower())


def _target_color_for_game(game: chess.pgn.Game, label_name: str) -> bool | None:
    """Return the side played by the labeled human, or None for Stockfish."""

    if label_name == "stockfish":
        return None

    target = _normalize_player_name(label_name)
    white = _normalize_player_name(game.headers.get("White", ""))
    black = _normalize_player_name(game.headers.get("Black", ""))

    if target and target in white:
        return chess.WHITE
    if target and target in black:
        return chess.BLACK
    return None


def _count_target_moves(game: chess.pgn.Game, target_color: bool | None) -> int:
    if target_color is None:
        return _count_mainline_plies(game)

    board = game.board()
    count = 0
    for move in game.mainline_moves():
        mover = board.turn
        board.push(move)
        if mover == target_color:
            count += 1
    return count


def collect_game_records(
    data_dir: str = "data/raw",
    window_size: int = 15,
    skip_moves: int = 10,
    exclude_labels: set[str] | None = None,
) -> List[GameRecord]:
    """Scan the data directory and collect usable game-level records."""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    exclude_labels = exclude_labels or set()
    records: List[GameRecord] = []
    min_plies = skip_moves + window_size

    for folder_name in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label_name = folder_name.lower()
        if label_name not in LABELS:
            continue
        if label_name in exclude_labels:
            continue

        label_id = LABELS[label_name]
        for pgn_path in _find_pgn_files(folder_path):
            with _open_pgn(pgn_path) as pgn_file:
                game_index = 0
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    target_color = _target_color_for_game(game, label_name)
                    if label_name != "stockfish" and target_color is None:
                        # Some PGNs can include games where the folder player is
                        # not clearly listed in the headers; skip those.
                        game_index += 1
                        continue

                    ply_count = _count_target_moves(game, target_color)
                    if ply_count >= min_plies:
                        # Store metadata only. Actual tensor/materialization is
                        # delayed until after balancing and splitting.
                        records.append(
                            GameRecord(
                                pgn_path=pgn_path,
                                game_index=game_index,
                                label=label_id,
                                label_name=label_name,
                                ply_count=ply_count,
                                target_color=target_color,
                            )
                        )
                    game_index += 1

    return records


def _print_game_counts(records: List[GameRecord], title: str) -> None:
    counts = Counter(record.label_name for record in records)
    print(f"\n{title} game counts:")
    if not counts:
        print("  (no games)")
        return

    for name in sorted(counts):
        print(f"  {name:12s}: {counts[name]}")


def _balance_records_by_label(
    records: List[GameRecord],
    max_games_per_label: int | None = None,
) -> List[GameRecord]:
    """Downsample labels to the same number of usable games."""

    by_label: dict[str, List[GameRecord]] = {}
    for record in records:
        by_label.setdefault(record.label_name, []).append(record)

    if not by_label:
        return []

    missing_labels = sorted(set(LABELS) - set(by_label))
    if missing_labels:
        print("\nLabels with no usable games:")
        for name in missing_labels:
            print(f"  {name}")

    target_count = min(len(items) for items in by_label.values())
    if max_games_per_label is not None:
        target_count = min(target_count, max_games_per_label)

    balanced: List[GameRecord] = []
    for label_name, items in sorted(by_label.items()):
        random.shuffle(items)
        balanced.extend(items[:target_count])
        print(f"Using {target_count} games for {label_name}")

    random.shuffle(balanced)
    return balanced


def _split_records_by_label(
    records: List[GameRecord],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, List[GameRecord]]:
    """Stratified game-level split so every label appears in each split."""

    split_records = {"train": [], "val": [], "test": []}
    by_label: dict[str, List[GameRecord]] = {}
    for record in records:
        by_label.setdefault(record.label_name, []).append(record)

    for items in by_label.values():
        random.shuffle(items)
        total = len(items)
        train_end = int(train_frac * total)
        val_end = train_end + int(val_frac * total)

        # Split at game level, not window level, so windows from the same game
        # cannot appear in both train and test.
        split_records["train"].extend(items[:train_end])
        split_records["val"].extend(items[train_end:val_end])
        split_records["test"].extend(items[val_end:])

    for items in split_records.values():
        random.shuffle(items)

    return split_records


def records_to_windows(
    records: List[GameRecord],
    window_size: int = 15,
    stride: int = 5,
    skip_moves: int = 10,
    max_windows_per_game: int | None = None,
    board_flip: bool = True,
) -> List[Tuple[torch.Tensor, int]]:
    """Re-open selected PGN games and materialize their tensor windows."""

    # Group by path to avoid repeatedly reopening large PGN files.
    samples: List[Tuple[torch.Tensor, int]] = []
    records_by_path: dict[str, dict[int, tuple[int, bool | None]]] = {}

    for record in records:
        records_by_path.setdefault(record.pgn_path, {})[record.game_index] = (
            record.label,
            record.target_color,
        )

    for pgn_path, selected_games in records_by_path.items():
        with _open_pgn(pgn_path) as pgn_file:
            game_index = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                selected = selected_games.get(game_index)
                if selected is not None:
                    label, target_color = selected
                    samples.extend(
                        game_to_windows(
                            game,
                            label=label,
                            window_size=window_size,
                            stride=stride,
                            skip_moves=skip_moves,
                            target_color=target_color,
                            max_windows=max_windows_per_game,
                            board_flip=board_flip,
                        )
                    )
                game_index += 1

    random.shuffle(samples)
    return samples


def _print_split_counts(split_name: str, samples: List[Tuple[torch.Tensor, int]]) -> None:
    inv_labels = {v: k for k, v in LABELS.items()}
    counts = Counter(label for _, label in samples)

    print(f"\\n{split_name} split class counts:")
    if not counts:
        print("  (no samples)")
        return

    for label_id in sorted(inv_labels.keys()):
        name = inv_labels[label_id]
        print(f"  {name:12s}: {counts.get(label_id, 0)}")


def _balance_samples_by_label(samples: List[Tuple[torch.Tensor, int]]) -> List[Tuple[torch.Tensor, int]]:
    """Downsample windows so long games do not dominate a split."""

    by_label: dict[int, List[Tuple[torch.Tensor, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    if not by_label:
        return samples

    target_count = min(len(items) for items in by_label.values())
    balanced: List[Tuple[torch.Tensor, int]] = []
    for items in by_label.values():
        random.shuffle(items)
        balanced.extend(items[:target_count])

    random.shuffle(balanced)
    return balanced


def build_dataset(
    data_dir: str = "data/raw",
    window_size: int = 15,
    stride: int = 5,
    skip_moves: int = 10,
    balance_games: bool = True,
    balance_windows: bool = True,
    max_games_per_label: int | None = None,
    exclude_labels: set[str] | None = None,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]], List[Tuple[torch.Tensor, int]]]:
    """Build balanced train/validation/test window datasets from raw PGNs."""

    # Step 1: collect game-level records. No tensors are built yet.
    records = collect_game_records(
        data_dir=data_dir,
        window_size=window_size,
        skip_moves=skip_moves,
        exclude_labels=exclude_labels,
    )
    _print_game_counts(records, "Usable")

    if balance_games:
        # Step 2: balance games before splitting so each class contributes the
        # same number of original games.
        records = _balance_records_by_label(records, max_games_per_label=max_games_per_label)
        _print_game_counts(records, "Balanced usable")

    split_games = _split_records_by_label(records)
    for split_name, split_records in split_games.items():
        _print_game_counts(split_records, split_name.capitalize())

    # Step 3: convert each split into neural tensor windows.
    train_samples = records_to_windows(
        split_games["train"],
        window_size=window_size,
        stride=stride,
        skip_moves=skip_moves,
    )
    val_samples = records_to_windows(
        split_games["val"],
        window_size=window_size,
        stride=stride,
        skip_moves=skip_moves,
    )
    test_samples = records_to_windows(
        split_games["test"],
        window_size=window_size,
        stride=stride,
        skip_moves=skip_moves,
    )

    if balance_windows:
        # Step 4: optionally balance the number of windows too, because longer
        # games otherwise create more training samples.
        train_samples = _balance_samples_by_label(train_samples)
        val_samples = _balance_samples_by_label(val_samples)
        test_samples = _balance_samples_by_label(test_samples)

    _print_split_counts("Train", train_samples)
    _print_split_counts("Val", val_samples)
    _print_split_counts("Test", test_samples)

    return train_samples, val_samples, test_samples


class ChessDataset(Dataset):
    """Minimal PyTorch Dataset wrapper around precomputed window tensors."""

    def __init__(self, samples: List[Tuple[torch.Tensor, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window_tensor, label = self.samples[idx]
        return window_tensor, label
