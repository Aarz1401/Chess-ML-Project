"""Board-plane scaffolding for the chess stylometry dataset.

This file is intentionally thin while we decide the representation. Keep the
plane builders explicit and small; build the sequence/dataset pipeline only
after the per-move channels make chess sense.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from game_identity import canonical_game_id, target_color, split_for_game

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

PIECE_PLANE_NAMES = [
    "target_P",
    "target_N",
    "target_B",
    "target_R",
    "target_Q",
    "target_K",
    "opponent_P",
    "opponent_N",
    "opponent_B",
    "opponent_R",
    "opponent_Q",
    "opponent_K",
]
MOVE_PLANE_NAMES = ["move_from", "move_to"]
LEGAL_PLANE_NAMES = [
    "before_legal_dest_P",
    "before_legal_dest_N",
    "before_legal_dest_B",
    "before_legal_dest_R",
    "before_legal_dest_Q",
    "before_legal_dest_K",
]
ATTACK_PLANE_NAMES = [
    "target_attack_P",
    "target_attack_N",
    "target_attack_B",
    "target_attack_R",
    "target_attack_Q",
    "target_attack_K",
    "opponent_attack_P",
    "opponent_attack_N",
    "opponent_attack_B",
    "opponent_attack_R",
    "opponent_attack_Q",
    "opponent_attack_K",
]
AFTER_ATTACK_PLANE_NAMES = [
    "after_target_attacks",
    "after_opponent_attacks",
    "new_target_attacks",
    "new_opponent_attacks",
]
CAPTURE_THREAT_PLANE_NAMES = [
    "target_capture_targets",
    "opponent_capture_targets",
    "target_threatens_pieces",
    "opponent_threatens_pieces",
    "new_target_threats",
    "new_opponent_threats",
]
KING_ZONE_PLANE_NAMES = [
    "target_king_zone",
    "opponent_king_zone",
    "target_pressure_opponent_king",
    "opponent_pressure_target_king",
]
EFFECT_PLANE_NAMES = [
    "new_piece_attacks",
    "removed_piece_attacks",
    "new_piece_defenses",
    "removed_piece_defenses",
]
ROLE_PLANE_NAMES = ["target_move", "opponent_move"]
PLANE_NAMES = (
    PIECE_PLANE_NAMES
    + MOVE_PLANE_NAMES
    + LEGAL_PLANE_NAMES
    + ATTACK_PLANE_NAMES
    + AFTER_ATTACK_PLANE_NAMES
    + CAPTURE_THREAT_PLANE_NAMES
    + KING_ZONE_PLANE_NAMES
    + EFFECT_PLANE_NAMES
    + ROLE_PLANE_NAMES
)
DERIVED_GAME_PLANE_NAMES = [
    "game_target_from_P",
    "game_target_from_N",
    "game_target_from_B",
    "game_target_from_R",
    "game_target_from_Q",
    "game_target_from_K",
    "game_target_to_P",
    "game_target_to_N",
    "game_target_to_B",
    "game_target_to_R",
    "game_target_to_Q",
    "game_target_to_K",
    "game_opponent_from",
    "game_opponent_to",
    "game_target_legal_dest",
    "game_opponent_legal_dest",
    "game_target_after_attacks",
    "game_target_new_attacks",
    "game_opponent_new_attacks",
    "game_target_capture_targets",
    "game_target_threatens_pieces",
    "game_target_new_threats",
    "game_target_king_pressure",
    "game_opponent_king_pressure",
]
DIRECT_GAME_PLANE_NAMES = [
    "game_direct_target_occupancy_P",
    "game_direct_target_occupancy_N",
    "game_direct_target_occupancy_B",
    "game_direct_target_occupancy_R",
    "game_direct_target_occupancy_Q",
    "game_direct_target_occupancy_K",
    "game_direct_target_capture_to",
    "game_direct_target_check_to",
]
GAME_PLANE_NAMES = DERIVED_GAME_PLANE_NAMES + DIRECT_GAME_PLANE_NAMES
FULL_INPUT_PLANE_NAMES = PLANE_NAMES + GAME_PLANE_NAMES

PIECE_CHANNELS = 12
MOVE_CHANNELS = 2
LEGAL_CHANNELS = 6
ATTACK_CHANNELS = 12
AFTER_ATTACK_CHANNELS = 4
CAPTURE_THREAT_CHANNELS = 6
KING_ZONE_CHANNELS = 4
EFFECT_CHANNELS = 4
ROLE_CHANNELS = 2
BASE_CHANNELS = (
    PIECE_CHANNELS
    + MOVE_CHANNELS
    + LEGAL_CHANNELS
    + ATTACK_CHANNELS
    + AFTER_ATTACK_CHANNELS
    + CAPTURE_THREAT_CHANNELS
    + KING_ZONE_CHANNELS
    + EFFECT_CHANNELS
    + ROLE_CHANNELS
)
CONTEXT_CHANNELS = 0
STOCKFISH_CHANNELS = 0
MOVE_INPUT_CHANNELS = BASE_CHANNELS + CONTEXT_CHANNELS + STOCKFISH_CHANNELS
DERIVED_GAME_CHANNELS = len(DERIVED_GAME_PLANE_NAMES)
DIRECT_GAME_CHANNELS = len(DIRECT_GAME_PLANE_NAMES)
GAME_CHANNELS = len(GAME_PLANE_NAMES)
GAME_CHANNEL_START = MOVE_INPUT_CHANNELS
INPUT_CHANNELS = MOVE_INPUT_CHANNELS + GAME_CHANNELS

TARGET_WEIGHT = 1.0
OPPONENT_WEIGHT = 0.5
GAME_TARGET_WEIGHT = 1.0
GAME_OPPONENT_WEIGHT = 0.5

TARGET_PIECE_OFFSET = 0
OPPONENT_PIECE_OFFSET = 6
MOVE_FROM_CHANNEL = PIECE_CHANNELS
MOVE_TO_CHANNEL = PIECE_CHANNELS + 1
LEGAL_CHANNEL_START = PIECE_CHANNELS + MOVE_CHANNELS
ATTACK_CHANNEL_START = LEGAL_CHANNEL_START + LEGAL_CHANNELS
AFTER_ATTACK_CHANNEL_START = ATTACK_CHANNEL_START + ATTACK_CHANNELS
AFTER_TARGET_ATTACKS_CHANNEL = AFTER_ATTACK_CHANNEL_START
AFTER_OPPONENT_ATTACKS_CHANNEL = AFTER_ATTACK_CHANNEL_START + 1
NEW_TARGET_ATTACKS_CHANNEL = AFTER_ATTACK_CHANNEL_START + 2
NEW_OPPONENT_ATTACKS_CHANNEL = AFTER_ATTACK_CHANNEL_START + 3
CAPTURE_THREAT_CHANNEL_START = AFTER_ATTACK_CHANNEL_START + AFTER_ATTACK_CHANNELS
TARGET_CAPTURE_TARGETS_CHANNEL = CAPTURE_THREAT_CHANNEL_START
OPPONENT_CAPTURE_TARGETS_CHANNEL = CAPTURE_THREAT_CHANNEL_START + 1
TARGET_THREATENS_PIECES_CHANNEL = CAPTURE_THREAT_CHANNEL_START + 2
OPPONENT_THREATENS_PIECES_CHANNEL = CAPTURE_THREAT_CHANNEL_START + 3
NEW_TARGET_THREATS_CHANNEL = CAPTURE_THREAT_CHANNEL_START + 4
NEW_OPPONENT_THREATS_CHANNEL = CAPTURE_THREAT_CHANNEL_START + 5
KING_ZONE_CHANNEL_START = CAPTURE_THREAT_CHANNEL_START + CAPTURE_THREAT_CHANNELS
TARGET_KING_ZONE_CHANNEL = KING_ZONE_CHANNEL_START
OPPONENT_KING_ZONE_CHANNEL = KING_ZONE_CHANNEL_START + 1
TARGET_PRESSURE_OPPONENT_KING_CHANNEL = KING_ZONE_CHANNEL_START + 2
OPPONENT_PRESSURE_TARGET_KING_CHANNEL = KING_ZONE_CHANNEL_START + 3
EFFECT_CHANNEL_START = KING_ZONE_CHANNEL_START + KING_ZONE_CHANNELS
NEW_ATTACKS_CHANNEL = EFFECT_CHANNEL_START
REMOVED_ATTACKS_CHANNEL = EFFECT_CHANNEL_START + 1
NEW_DEFENSES_CHANNEL = EFFECT_CHANNEL_START + 2
REMOVED_DEFENSES_CHANNEL = EFFECT_CHANNEL_START + 3
TARGET_MOVE_CHANNEL = EFFECT_CHANNEL_START + EFFECT_CHANNELS
OPPONENT_MOVE_CHANNEL = TARGET_MOVE_CHANNEL + 1
GAME_TARGET_FROM_START = 0
GAME_TARGET_TO_START = GAME_TARGET_FROM_START + 6
GAME_OPPONENT_FROM_CHANNEL = GAME_TARGET_TO_START + 6
GAME_OPPONENT_TO_CHANNEL = GAME_OPPONENT_FROM_CHANNEL + 1
GAME_TARGET_LEGAL_DEST_CHANNEL = GAME_OPPONENT_TO_CHANNEL + 1
GAME_OPPONENT_LEGAL_DEST_CHANNEL = GAME_TARGET_LEGAL_DEST_CHANNEL + 1
GAME_TARGET_AFTER_ATTACKS_CHANNEL = GAME_OPPONENT_LEGAL_DEST_CHANNEL + 1
GAME_TARGET_NEW_ATTACKS_CHANNEL = GAME_TARGET_AFTER_ATTACKS_CHANNEL + 1
GAME_OPPONENT_NEW_ATTACKS_CHANNEL = GAME_TARGET_NEW_ATTACKS_CHANNEL + 1
GAME_TARGET_CAPTURE_TARGETS_CHANNEL = GAME_OPPONENT_NEW_ATTACKS_CHANNEL + 1
GAME_TARGET_THREATENS_PIECES_CHANNEL = GAME_TARGET_CAPTURE_TARGETS_CHANNEL + 1
GAME_TARGET_NEW_THREATS_CHANNEL = GAME_TARGET_THREATENS_PIECES_CHANNEL + 1
GAME_TARGET_KING_PRESSURE_CHANNEL = GAME_TARGET_NEW_THREATS_CHANNEL + 1
GAME_OPPONENT_KING_PRESSURE_CHANNEL = GAME_TARGET_KING_PRESSURE_CHANNEL + 1
DIRECT_GAME_CHANNEL_START = DERIVED_GAME_CHANNELS
GAME_DIRECT_TARGET_OCCUPANCY_START = DIRECT_GAME_CHANNEL_START
GAME_DIRECT_TARGET_CAPTURE_TO_CHANNEL = GAME_DIRECT_TARGET_OCCUPANCY_START + 6
GAME_DIRECT_TARGET_CHECK_TO_CHANNEL = GAME_DIRECT_TARGET_CAPTURE_TO_CHANNEL + 1


@dataclass(frozen=True)
class GameRecord:
    """Metadata that identifies one usable game inside one PGN file."""

    pgn_path: Path
    game_index: int
    label_name: str
    label_id: int
    target_color: bool
    ply_count: int
    game_id: str = ""


@dataclass(frozen=True)
class TensorConfig:
    """Settings that will control game-to-tensor conversion later."""

    sequence_length: int = 80
    skip_plies: int = 0
    target_moves_only: bool = False
    stockfish_depth: int = 0
    multipv: int = 0
    board_flip: bool = True


class ChessStyleDataset(Dataset):
    """Minimal PyTorch dataset wrapper for already-built tensor samples."""

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

    return target_color(game.headers, label_name)


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
    seen: set[tuple[str, bool]] = set()

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
                game_id = canonical_game_id(game)
                if target_color is not None and ply_count >= min_plies and not game.errors and (game_id, target_color) not in seen:
                    seen.add((game_id, target_color))
                    records.append(
                        GameRecord(
                            pgn_path=pgn_path,
                            game_index=game_index,
                            label_name=label_name,
                            label_id=label_id,
                            target_color=target_color,
                            ply_count=ply_count,
                            game_id=game_id,
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
    """Assign every perspective of an identical game to the same split."""
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        split = split_for_game(record.game_id, seed, train_frac, val_frac)
        splits[split].append(record)
    for items in splits.values():
        random.Random(seed).shuffle(items)
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


def piece_channel(piece: chess.Piece, target_color: bool) -> int:
    """Return the target-relative channel index for one piece."""

    base_channel = PIECE_TO_RELATIVE_CHANNEL[piece.piece_type]
    offset = TARGET_PIECE_OFFSET if piece.color == target_color else OPPONENT_PIECE_OFFSET
    return offset + base_channel


def side_weight(color: bool, target_color: bool) -> float:
    """Return the configured weight for target-owned vs opponent-owned data."""

    return TARGET_WEIGHT if color == target_color else OPPONENT_WEIGHT


def piece_planes(board: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode piece locations in target-relative channels.

    Channels are defined by PIECE_PLANE_NAMES:
    target P/N/B/R/Q/K, then opponent P/N/B/R/Q/K. Target pieces are marked
    with 1.0 and opponent pieces are marked with 0.5.
    """

    planes = torch.zeros((PIECE_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK

    for square, piece in board.piece_map().items():
        channel = piece_channel(piece, target_color)
        value = side_weight(piece.color, target_color)
        row, col = square_to_coords(square, flip)
        planes[channel, row, col] = value

    return planes


def legal_destination_planes(board: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode legal destination squares for the side to move by piece type."""

    planes = torch.zeros((LEGAL_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK
    value = side_weight(board.turn, target_color)

    for legal_move in board.legal_moves:
        piece = board.piece_at(legal_move.from_square)
        if piece is None:
            continue

        channel = PIECE_TO_RELATIVE_CHANNEL[piece.piece_type]
        row, col = square_to_coords(legal_move.to_square, flip)
        planes[channel, row, col] = max(float(planes[channel, row, col]), value)

    return planes


def attack_control_planes(board: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode attacked squares by target/opponent and piece type."""

    planes = torch.zeros((ATTACK_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK

    for square, piece in board.piece_map().items():
        channel = piece_channel(piece, target_color)
        value = side_weight(piece.color, target_color)
        for attacked_square in board.attacks(square):
            row, col = square_to_coords(attacked_square, flip)
            planes[channel, row, col] += value

    return planes


def side_attack_counts(board: chess.Board, color: bool, board_flip: bool) -> torch.Tensor:
    """Return one aggregate attack-count map for a single side."""

    plane = torch.zeros((8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        for attacked_square in board.attacks(square):
            row, col = square_to_coords(attacked_square, board_flip)
            plane[row, col] += 1.0

    return plane


def weighted_side_attack_counts(board: chess.Board, color: bool, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Return attack counts scaled by target/opponent ownership."""

    return side_attack_counts(board, color, board_flip) * side_weight(color, target_color)


def after_attack_planes(before: chess.Board, after: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode aggregate after-position attacks and newly created pressure.

    The delta channels keep only non-negative increases so these remain
    readable heatmaps instead of signed tensors.
    """

    planes = torch.zeros((AFTER_ATTACK_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK
    opponent = not target_color

    before_target = side_attack_counts(before, target_color, flip)
    after_target = side_attack_counts(after, target_color, flip)
    before_opponent = side_attack_counts(before, opponent, flip)
    after_opponent = side_attack_counts(after, opponent, flip)

    planes[0] = after_target * TARGET_WEIGHT
    planes[1] = after_opponent * OPPONENT_WEIGHT
    planes[2] = torch.clamp(after_target - before_target, min=0.0) * TARGET_WEIGHT
    planes[3] = torch.clamp(after_opponent - before_opponent, min=0.0) * OPPONENT_WEIGHT

    return planes


def side_defense_counts(board: chess.Board, color: bool, board_flip: bool) -> torch.Tensor:
    """Return counts of defended own-piece squares for a single side."""

    plane = torch.zeros((8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        row, col = square_to_coords(square, board_flip)
        plane[row, col] = float(len(board.attackers(color, square)))

    return plane


def capture_target_plane(board: chess.Board, color: bool, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Return legal capture destination squares for one side in this position."""

    plane = torch.zeros((8, 8), dtype=torch.float32)
    value = side_weight(color, target_color)
    side_board = board.copy(stack=False)
    side_board.turn = color

    for legal_move in side_board.legal_moves:
        if not side_board.is_capture(legal_move):
            continue

        row, col = square_to_coords(legal_move.to_square, board_flip)
        plane[row, col] = max(float(plane[row, col]), value)

    return plane


def threatened_piece_plane(board: chess.Board, attacker_color: bool, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Return enemy occupied squares attacked by one side."""

    plane = torch.zeros((8, 8), dtype=torch.float32)
    value = side_weight(attacker_color, target_color)

    for square, piece in board.piece_map().items():
        if piece.color == attacker_color:
            continue
        if not board.is_attacked_by(attacker_color, square):
            continue

        row, col = square_to_coords(square, board_flip)
        plane[row, col] = value

    return plane


def capture_threat_planes(before: chess.Board, after: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode legal captures, current threats, and newly created threats."""

    planes = torch.zeros((CAPTURE_THREAT_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK
    opponent = not target_color

    before_target_threats = threatened_piece_plane(before, target_color, target_color, flip)
    before_opponent_threats = threatened_piece_plane(before, opponent, target_color, flip)
    after_target_threats = threatened_piece_plane(after, target_color, target_color, flip)
    after_opponent_threats = threatened_piece_plane(after, opponent, target_color, flip)

    planes[0] = capture_target_plane(before, target_color, target_color, flip)
    planes[1] = capture_target_plane(before, opponent, target_color, flip)
    planes[2] = before_target_threats
    planes[3] = before_opponent_threats
    planes[4] = torch.clamp(after_target_threats - before_target_threats, min=0.0)
    planes[5] = torch.clamp(after_opponent_threats - before_opponent_threats, min=0.0)

    return planes


def king_zone_mask(board: chess.Board, color: bool, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Return the king square plus adjacent squares for one side."""

    plane = torch.zeros((8, 8), dtype=torch.float32)
    king_square = board.king(color)
    if king_square is None:
        return plane

    value = side_weight(color, target_color)
    zone_squares = set(board.attacks(king_square))
    zone_squares.add(king_square)
    for square in zone_squares:
        row, col = square_to_coords(square, board_flip)
        plane[row, col] = value

    return plane


def king_zone_pressure_plane(
    board: chess.Board,
    attacker_color: bool,
    king_color: bool,
    target_color: bool,
    board_flip: bool,
) -> torch.Tensor:
    """Return attacking pressure inside a king zone."""

    king_square = board.king(king_color)
    if king_square is None:
        return torch.zeros((8, 8), dtype=torch.float32)

    zone = torch.zeros((8, 8), dtype=torch.float32)
    for square in set(board.attacks(king_square)) | {king_square}:
        row, col = square_to_coords(square, board_flip)
        zone[row, col] = 1.0

    attacks = weighted_side_attack_counts(board, attacker_color, target_color, board_flip)
    return attacks * zone


def king_zone_planes(before: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode king zones and pressure near each king in the before position."""

    planes = torch.zeros((KING_ZONE_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK
    opponent = not target_color

    planes[0] = king_zone_mask(before, target_color, target_color, flip)
    planes[1] = king_zone_mask(before, opponent, target_color, flip)
    planes[2] = king_zone_pressure_plane(before, target_color, opponent, target_color, flip)
    planes[3] = king_zone_pressure_plane(before, opponent, target_color, target_color, flip)

    return planes


def piece_presence_mask(board: chess.Board, color: bool, board_flip: bool) -> torch.Tensor:
    """Return a mask of occupied squares for one side."""

    plane = torch.zeros((8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        row, col = square_to_coords(square, board_flip)
        plane[row, col] = 1.0

    return plane


def move_effect_planes(before: chess.Board, after: chess.Board, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Encode how the move changed attacks/defenses on occupied pieces.

    Attack effects are restricted to opponent pieces. Defense effects are
    restricted to own pieces. Removed effects only count pieces still present
    after the move, so moving away from a square does not mark that now-empty
    square as a lost defense.
    """

    planes = torch.zeros((EFFECT_CHANNELS, 8, 8), dtype=torch.float32)
    mover = before.turn
    flip = board_flip and target_color == chess.BLACK
    value = side_weight(mover, target_color)

    opponent = not mover
    before_opponent = piece_presence_mask(before, opponent, flip)
    after_opponent = piece_presence_mask(after, opponent, flip)
    stable_opponent = before_opponent * after_opponent

    before_attacks = side_attack_counts(before, mover, flip) * stable_opponent
    after_attacks = side_attack_counts(after, mover, flip) * after_opponent
    attack_delta = after_attacks - before_attacks
    planes[0] = torch.clamp(attack_delta, min=0.0) * value
    planes[1] = torch.clamp(-attack_delta, min=0.0) * value

    before_own = piece_presence_mask(before, mover, flip)
    after_own = piece_presence_mask(after, mover, flip)
    stable_own = before_own * after_own

    before_defenses = side_defense_counts(before, mover, flip) * stable_own
    after_defenses = side_defense_counts(after, mover, flip) * after_own
    defense_delta = after_defenses - before_defenses
    planes[2] = torch.clamp(defense_delta, min=0.0) * value
    planes[3] = torch.clamp(-defense_delta, min=0.0) * value

    return planes


def move_planes(before: chess.Board, after: chess.Board, move: chess.Move, target_color: bool, board_flip: bool) -> torch.Tensor:
    """Build the per-ply board planes.

    The tensor contains before-position pieces, the exact move, legal
    destination heatmaps, attack/control heatmaps, move-effect heatmaps, and
    target/opponent role planes.
    """

    planes = torch.zeros((MOVE_INPUT_CHANNELS, 8, 8), dtype=torch.float32)
    planes[:PIECE_CHANNELS] = piece_planes(before, target_color=target_color, board_flip=board_flip)
    planes[LEGAL_CHANNEL_START:ATTACK_CHANNEL_START] = legal_destination_planes(
        before, target_color=target_color, board_flip=board_flip
    )
    planes[ATTACK_CHANNEL_START:AFTER_ATTACK_CHANNEL_START] = attack_control_planes(
        before, target_color=target_color, board_flip=board_flip
    )
    planes[AFTER_ATTACK_CHANNEL_START:CAPTURE_THREAT_CHANNEL_START] = after_attack_planes(
        before, after, target_color=target_color, board_flip=board_flip
    )
    planes[CAPTURE_THREAT_CHANNEL_START:KING_ZONE_CHANNEL_START] = capture_threat_planes(
        before, after, target_color=target_color, board_flip=board_flip
    )
    planes[KING_ZONE_CHANNEL_START:EFFECT_CHANNEL_START] = king_zone_planes(
        before, target_color=target_color, board_flip=board_flip
    )
    planes[EFFECT_CHANNEL_START:TARGET_MOVE_CHANNEL] = move_effect_planes(
        before, after, target_color=target_color, board_flip=board_flip
    )

    is_target_move = before.turn == target_color
    move_value = side_weight(before.turn, target_color)
    flip = board_flip and target_color == chess.BLACK

    from_row, from_col = square_to_coords(move.from_square, flip)
    to_row, to_col = square_to_coords(move.to_square, flip)
    planes[MOVE_FROM_CHANNEL, from_row, from_col] = move_value
    planes[MOVE_TO_CHANNEL, to_row, to_col] = move_value

    if is_target_move:
        planes[TARGET_MOVE_CHANNEL].fill_(1.0)
    else:
        planes[OPPONENT_MOVE_CHANNEL].fill_(1.0)

    return planes


def is_target_move_tensor(move_tensor: torch.Tensor) -> bool:
    """Return whether a per-ply tensor represents a target-player move."""

    return float(move_tensor[TARGET_MOVE_CHANNEL].mean()) > 0.5


def target_moving_piece_type(move_tensor: torch.Tensor) -> int | None:
    """Infer the moving target piece type from piece and from-square planes."""

    from_square = move_tensor[MOVE_FROM_CHANNEL]
    piece_scores = (move_tensor[:6] * from_square).flatten(1).sum(dim=1)
    if float(piece_scores.max()) <= 0.0:
        return None
    return int(piece_scores.argmax().item())


def average_game_plane(plane: torch.Tensor, count: int) -> torch.Tensor:
    """Normalize an accumulated game plane without making empty games fail."""

    if count <= 0:
        return plane
    return plane / float(count)


def direct_game_planes_from_ply_history(
    ply_history: list[tuple[chess.Board, chess.Move, chess.Board]],
    target_color: bool,
    board_flip: bool,
) -> torch.Tensor:
    """Build direct PGN-derived game maps that do not depend on move planes.

    The six occupancy planes average where target pieces stood across the
    initial position and every after-move position. Capture/check destination
    planes average over target-player plies only.
    """

    planes = torch.zeros((DIRECT_GAME_CHANNELS, 8, 8), dtype=torch.float32)
    flip = board_flip and target_color == chess.BLACK

    position_boards: list[chess.Board] = []
    if ply_history:
        position_boards.append(ply_history[0][0])
        position_boards.extend(after for _, _, after in ply_history)

    for board in position_boards:
        for square, piece in board.piece_map().items():
            if piece.color != target_color:
                continue

            channel = PIECE_TO_RELATIVE_CHANNEL[piece.piece_type]
            row, col = square_to_coords(square, flip)
            planes[channel, row, col] += GAME_TARGET_WEIGHT

    position_count = len(position_boards)
    for channel in range(6):
        planes[channel] = average_game_plane(planes[channel], position_count)

    target_move_count = 0
    for before, move, after in ply_history:
        if before.turn != target_color:
            continue

        target_move_count += 1
        row, col = square_to_coords(move.to_square, flip)
        if before.is_capture(move):
            planes[GAME_DIRECT_TARGET_CAPTURE_TO_CHANNEL - DIRECT_GAME_CHANNEL_START, row, col] += GAME_TARGET_WEIGHT
        if after.is_check():
            planes[GAME_DIRECT_TARGET_CHECK_TO_CHANNEL - DIRECT_GAME_CHANNEL_START, row, col] += GAME_TARGET_WEIGHT

    for channel in [
        GAME_DIRECT_TARGET_CAPTURE_TO_CHANNEL - DIRECT_GAME_CHANNEL_START,
        GAME_DIRECT_TARGET_CHECK_TO_CHANNEL - DIRECT_GAME_CHANNEL_START,
    ]:
        planes[channel] = average_game_plane(planes[channel], target_move_count)

    return planes


def game_planes_from_move_sequence(move_sequence: torch.Tensor) -> torch.Tensor:
    """Aggregate one game's per-ply planes into target-emphasized game maps.

    The aggregate maps are linear combinations of existing per-ply heatmaps.
    Target-player maps are kept at full scale. Opponent-context maps receive an
    additional 0.5 multiplier on top of the per-ply opponent weighting, so the
    final game tensor is intentionally target-heavy.
    """

    if move_sequence.ndim != 4 or move_sequence.shape[1:] != (MOVE_INPUT_CHANNELS, 8, 8):
        raise ValueError(
            f"Expected move sequence shape (T, {MOVE_INPUT_CHANNELS}, 8, 8), got {tuple(move_sequence.shape)}"
        )

    planes = torch.zeros((GAME_CHANNELS, 8, 8), dtype=torch.float32)
    target_move_count = 0
    opponent_move_count = 0

    for move_tensor in move_sequence.float():
        target_role = float(move_tensor[TARGET_MOVE_CHANNEL].mean())
        opponent_role = float(move_tensor[OPPONENT_MOVE_CHANNEL].mean())
        if target_role <= 0.5 and opponent_role <= 0.5:
            continue

        legal_destinations = move_tensor[LEGAL_CHANNEL_START:ATTACK_CHANNEL_START].sum(dim=0)

        if target_role > 0.5:
            target_move_count += 1
            piece_idx = target_moving_piece_type(move_tensor)
            if piece_idx is not None:
                planes[GAME_TARGET_FROM_START + piece_idx] += move_tensor[MOVE_FROM_CHANNEL] * GAME_TARGET_WEIGHT
                planes[GAME_TARGET_TO_START + piece_idx] += move_tensor[MOVE_TO_CHANNEL] * GAME_TARGET_WEIGHT

            planes[GAME_TARGET_LEGAL_DEST_CHANNEL] += legal_destinations * GAME_TARGET_WEIGHT
            planes[GAME_TARGET_AFTER_ATTACKS_CHANNEL] += move_tensor[AFTER_TARGET_ATTACKS_CHANNEL] * GAME_TARGET_WEIGHT
            planes[GAME_TARGET_NEW_ATTACKS_CHANNEL] += move_tensor[NEW_TARGET_ATTACKS_CHANNEL] * GAME_TARGET_WEIGHT
            planes[GAME_TARGET_CAPTURE_TARGETS_CHANNEL] += move_tensor[TARGET_CAPTURE_TARGETS_CHANNEL] * GAME_TARGET_WEIGHT
            planes[GAME_TARGET_THREATENS_PIECES_CHANNEL] += move_tensor[TARGET_THREATENS_PIECES_CHANNEL] * GAME_TARGET_WEIGHT
            planes[GAME_TARGET_NEW_THREATS_CHANNEL] += move_tensor[NEW_TARGET_THREATS_CHANNEL] * GAME_TARGET_WEIGHT
            planes[GAME_TARGET_KING_PRESSURE_CHANNEL] += move_tensor[TARGET_PRESSURE_OPPONENT_KING_CHANNEL] * GAME_TARGET_WEIGHT
        else:
            opponent_move_count += 1
            planes[GAME_OPPONENT_FROM_CHANNEL] += move_tensor[MOVE_FROM_CHANNEL] * GAME_OPPONENT_WEIGHT
            planes[GAME_OPPONENT_TO_CHANNEL] += move_tensor[MOVE_TO_CHANNEL] * GAME_OPPONENT_WEIGHT
            planes[GAME_OPPONENT_LEGAL_DEST_CHANNEL] += legal_destinations * GAME_OPPONENT_WEIGHT
            planes[GAME_OPPONENT_NEW_ATTACKS_CHANNEL] += move_tensor[NEW_OPPONENT_ATTACKS_CHANNEL] * GAME_OPPONENT_WEIGHT
            planes[GAME_OPPONENT_KING_PRESSURE_CHANNEL] += (
                move_tensor[OPPONENT_PRESSURE_TARGET_KING_CHANNEL] * GAME_OPPONENT_WEIGHT
            )

    for channel in range(GAME_TARGET_FROM_START, GAME_OPPONENT_FROM_CHANNEL):
        planes[channel] = average_game_plane(planes[channel], target_move_count)
    for channel in [
        GAME_TARGET_LEGAL_DEST_CHANNEL,
        GAME_TARGET_AFTER_ATTACKS_CHANNEL,
        GAME_TARGET_NEW_ATTACKS_CHANNEL,
        GAME_TARGET_CAPTURE_TARGETS_CHANNEL,
        GAME_TARGET_THREATENS_PIECES_CHANNEL,
        GAME_TARGET_NEW_THREATS_CHANNEL,
        GAME_TARGET_KING_PRESSURE_CHANNEL,
    ]:
        planes[channel] = average_game_plane(planes[channel], target_move_count)
    for channel in [
        GAME_OPPONENT_FROM_CHANNEL,
        GAME_OPPONENT_TO_CHANNEL,
        GAME_OPPONENT_LEGAL_DEST_CHANNEL,
        GAME_OPPONENT_NEW_ATTACKS_CHANNEL,
        GAME_OPPONENT_KING_PRESSURE_CHANNEL,
    ]:
        planes[channel] = average_game_plane(planes[channel], opponent_move_count)

    return planes


def game_planes_from_move_sequence_and_direct(
    move_sequence: torch.Tensor,
    direct_game_planes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine derived per-move aggregates with direct PGN-derived maps."""

    game_planes = game_planes_from_move_sequence(move_sequence)
    if direct_game_planes is None:
        return game_planes
    if direct_game_planes.shape != (DIRECT_GAME_CHANNELS, 8, 8):
        raise ValueError(
            f"Expected direct game planes shape {(DIRECT_GAME_CHANNELS, 8, 8)}, got {tuple(direct_game_planes.shape)}"
        )

    game_planes[DIRECT_GAME_CHANNEL_START:] = direct_game_planes
    return game_planes


def concatenate_game_planes(move_sequence: torch.Tensor, game_planes: torch.Tensor) -> torch.Tensor:
    """Attach one game's aggregate maps to every ply tensor in the sequence."""

    if move_sequence.ndim != 4 or move_sequence.shape[1:] != (MOVE_INPUT_CHANNELS, 8, 8):
        raise ValueError(
            f"Expected move sequence shape (T, {MOVE_INPUT_CHANNELS}, 8, 8), got {tuple(move_sequence.shape)}"
        )
    if game_planes.shape != (GAME_CHANNELS, 8, 8):
        raise ValueError(f"Expected game planes shape {(GAME_CHANNELS, 8, 8)}, got {tuple(game_planes.shape)}")

    repeated_game_planes = game_planes.unsqueeze(0).expand(move_sequence.shape[0], -1, -1, -1)
    return torch.cat([move_sequence, repeated_game_planes], dim=1)


def build_game_tensor(
    move_sequence: torch.Tensor,
    direct_game_planes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return concatenated per-ply tensors plus the standalone game planes."""

    game_planes = game_planes_from_move_sequence_and_direct(move_sequence, direct_game_planes=direct_game_planes)
    return concatenate_game_planes(move_sequence, game_planes), game_planes


def game_to_model_tensors(
    game: chess.pgn.Game,
    target_color: bool,
    config: TensorConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Convert one PGN game into one padded model input tensor.

    The returned sequence has shape ``(sequence_length, INPUT_CHANNELS, 8, 8)``.
    The padding mask has shape ``(sequence_length,)`` where True means padded.
    """

    board = game.board()
    move_tensors: list[torch.Tensor] = []
    ply_history: list[tuple[chess.Board, chess.Move, chess.Board]] = []

    for ply_index, move in enumerate(game.mainline_moves()):
        before = board.copy(stack=False)
        board.push(move)
        after = board.copy(stack=False)

        if ply_index < config.skip_plies:
            continue
        if config.target_moves_only and before.turn != target_color:
            continue

        move_tensors.append(
            move_planes(before, after, move, target_color=target_color, board_flip=config.board_flip)
        )
        ply_history.append((before, move, after))

        if len(move_tensors) == config.sequence_length:
            break

    if not move_tensors:
        return None

    move_sequence = torch.stack(move_tensors, dim=0)
    direct_game_planes = direct_game_planes_from_ply_history(
        ply_history,
        target_color=target_color,
        board_flip=config.board_flip,
    )
    full_sequence, _ = build_game_tensor(move_sequence, direct_game_planes=direct_game_planes)

    valid_moves = full_sequence.shape[0]
    padded_sequence = torch.zeros((config.sequence_length, INPUT_CHANNELS, 8, 8), dtype=torch.float16)
    padded_sequence[:valid_moves] = full_sequence.to(dtype=torch.float16)

    padding_mask = torch.ones((config.sequence_length,), dtype=torch.bool)
    padding_mask[:valid_moves] = False
    return padded_sequence, padding_mask


def materialize_samples(
    records: list[GameRecord],
    *args: object,
    **kwargs: object,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Read selected games and convert them into padded model tensors."""

    config = next((arg for arg in args if isinstance(arg, TensorConfig)), None)
    if config is None:
        config = kwargs.get("config")
    if not isinstance(config, TensorConfig):
        raise TypeError("materialize_samples requires a TensorConfig argument.")

    by_path: dict[Path, dict[int, GameRecord]] = {}
    for record in records:
        by_path.setdefault(record.pgn_path, {})[record.game_index] = record

    samples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    kept_records: list[GameRecord] = []
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
                    tensors = game_to_model_tensors(game, record.target_color, config)
                    if tensors is not None:
                        sequence, padding_mask = tensors
                        samples.append((sequence, padding_mask, record.label_id))
                        kept_records.append(record)

                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"  materialized {completed}/{total} games", flush=True)

                game_index += 1

        if selected_games:
            missing = ", ".join(str(index) for index in sorted(selected_games))
            raise ValueError(f"Could not find selected games in {pgn_path}: {missing}")

    combined = list(zip(samples, kept_records))
    random.shuffle(combined)
    metadata_records = kwargs.get("metadata_records")
    if isinstance(metadata_records, list):
        metadata_records.extend(record for _, record in combined)
    return [sample for sample, _ in combined]
