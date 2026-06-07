"""Visualize the per-move board planes.

This is a debug tool only. It builds a few known positions, writes a PNG grid
for each move, and prints compact numeric summaries for every plane.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import chess
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from board_dataset import (
    AFTER_OPPONENT_ATTACKS_CHANNEL,
    AFTER_TARGET_ATTACKS_CHANNEL,
    LEGAL_CHANNELS,
    LEGAL_CHANNEL_START,
    MOVE_FROM_CHANNEL,
    MOVE_INPUT_CHANNELS,
    MOVE_TO_CHANNEL,
    NEW_OPPONENT_ATTACKS_CHANNEL,
    NEW_OPPONENT_THREATS_CHANNEL,
    NEW_TARGET_ATTACKS_CHANNEL,
    NEW_TARGET_THREATS_CHANNEL,
    OPPONENT_CAPTURE_TARGETS_CHANNEL,
    OPPONENT_KING_ZONE_CHANNEL,
    OPPONENT_MOVE_CHANNEL,
    OPPONENT_PRESSURE_TARGET_KING_CHANNEL,
    OPPONENT_THREATENS_PIECES_CHANNEL,
    PIECE_TO_RELATIVE_CHANNEL,
    PLANE_NAMES,
    REMOVED_DEFENSES_CHANNEL,
    TARGET_CAPTURE_TARGETS_CHANNEL,
    TARGET_KING_ZONE_CHANNEL,
    TARGET_MOVE_CHANNEL,
    TARGET_PRESSURE_OPPONENT_KING_CHANNEL,
    TARGET_THREATENS_PIECES_CHANNEL,
    move_planes,
    side_weight,
    side_attack_counts,
    square_to_coords,
)

HEATMAP_CMAP = LinearSegmentedColormap.from_list("plane_heat", ["#ffffff", "#fff3a3", "#ffd000"])
PIECE_LABELS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


@dataclass(frozen=True)
class DebugCase:
    """One visualized move and its generated planes."""

    name: str
    before: chess.Board
    after: chess.Board
    move: chess.Move
    target_color: bool
    planes: torch.Tensor


def tensor_coord_to_square(row: int, col: int, target_color: bool, board_flip: bool = True) -> str:
    """Map tensor coordinates back to absolute chess square names."""

    flip = board_flip and target_color == chess.BLACK
    if flip:
        file_idx = 7 - col
        rank_idx = row
    else:
        file_idx = col
        rank_idx = 7 - row

    return chess.square_name(chess.square(file_idx, rank_idx))


def axis_labels(target_color: bool, board_flip: bool = True) -> tuple[list[str], list[str]]:
    """Return absolute file/rank labels for the displayed tensor orientation."""

    files = [tensor_coord_to_square(7, col, target_color, board_flip)[0] for col in range(8)]
    ranks = [tensor_coord_to_square(row, 0, target_color, board_flip)[1] for row in range(8)]
    return files, ranks


def nonzero_coords(plane: torch.Tensor, target_color: bool) -> str:
    """Return square-name, value pairs for nonzero cells."""

    coords: list[tuple[str, float]] = []
    for row, col in plane.nonzero(as_tuple=False).tolist():
        square_name = tensor_coord_to_square(row, col, target_color)
        coords.append((square_name, float(plane[row, col])))
    if len(coords) == 64:
        return "all 64 squares"
    return str(coords)


def nonzero_square_set(plane: torch.Tensor, target_color: bool) -> set[str]:
    """Return nonzero heatmap cells as absolute chess square names."""

    squares: set[str] = set()
    for row, col in plane.nonzero(as_tuple=False).tolist():
        squares.add(tensor_coord_to_square(row, col, target_color))
    return squares


def legal_moves_by_piece(board: chess.Board) -> dict[int, list[chess.Move]]:
    """Group legal moves by moving piece type."""

    grouped = {piece_type: [] for piece_type in PIECE_LABELS}
    for legal_move in board.legal_moves:
        piece = board.piece_at(legal_move.from_square)
        if piece is None:
            raise AssertionError(f"Legal move without piece: {legal_move.uci()}")
        grouped[piece.piece_type].append(legal_move)
    return grouped


def legal_capture_destinations(board: chess.Board, color: bool) -> set[str]:
    """Return legal capture destination squares for a hypothetical side to move."""

    side_board = board.copy(stack=False)
    side_board.turn = color
    destinations: set[str] = set()
    for legal_move in side_board.legal_moves:
        if side_board.is_capture(legal_move):
            destinations.add(chess.square_name(legal_move.to_square))
    return destinations


def threatened_piece_squares(board: chess.Board, attacker_color: bool) -> set[str]:
    """Return enemy occupied squares attacked by one side."""

    threatened: set[str] = set()
    for square, piece in board.piece_map().items():
        if piece.color == attacker_color:
            continue
        if board.is_attacked_by(attacker_color, square):
            threatened.add(chess.square_name(square))
    return threatened


def king_zone_squares(board: chess.Board, color: bool) -> set[str]:
    """Return the king square plus adjacent king-zone squares."""

    king_square = board.king(color)
    if king_square is None:
        return set()
    return {chess.square_name(square) for square in set(board.attacks(king_square)) | {king_square}}


def weighted_square_set(plane: torch.Tensor, target_color: bool) -> set[str]:
    """Return nonzero squares from a weighted binary-ish plane."""

    return nonzero_square_set(plane, target_color)


def summarize_planes(case: DebugCase) -> None:
    """Print compact numeric plane summaries."""

    print(f"\n{case.name}")
    print(f"move={case.move.uci()} target={'white' if case.target_color == chess.WHITE else 'black'}")
    print(f"shape={tuple(case.planes.shape)}")
    for name, plane in zip(PLANE_NAMES, case.planes):
        coords = nonzero_coords(plane, case.target_color)
        print(
            f"  {name:30s} min={float(plane.min()):.1f} max={float(plane.max()):.1f} "
            f"sum={float(plane.sum()):.1f} nonzero={coords}"
        )


def validate_case(case: DebugCase) -> None:
    """Validate key planes against python-chess ground truth."""

    if case.planes.shape != (MOVE_INPUT_CHANNELS, 8, 8):
        raise AssertionError(f"{case.name}: expected {(MOVE_INPUT_CHANNELS, 8, 8)}, got {tuple(case.planes.shape)}")
    if case.move not in case.before.legal_moves:
        raise AssertionError(f"{case.name}: illegal move {case.move.uci()} for {case.before.fen()}")

    move_value = side_weight(case.before.turn, case.target_color)
    flip = case.target_color == chess.BLACK
    from_row, from_col = square_to_coords(case.move.from_square, flip)
    to_row, to_col = square_to_coords(case.move.to_square, flip)

    if float(case.planes[MOVE_FROM_CHANNEL, from_row, from_col]) != move_value:
        raise AssertionError(f"{case.name}: move_from does not mark {chess.square_name(case.move.from_square)}")
    if float(case.planes[MOVE_TO_CHANNEL, to_row, to_col]) != move_value:
        raise AssertionError(f"{case.name}: move_to does not mark {chess.square_name(case.move.to_square)}")

    target_role_sum = float(case.planes[TARGET_MOVE_CHANNEL].sum())
    opponent_role_sum = float(case.planes[OPPONENT_MOVE_CHANNEL].sum())
    if case.before.turn == case.target_color and (target_role_sum, opponent_role_sum) != (64.0, 0.0):
        raise AssertionError(f"{case.name}: target role planes are wrong")
    if case.before.turn != case.target_color and (target_role_sum, opponent_role_sum) != (0.0, 64.0):
        raise AssertionError(f"{case.name}: opponent role planes are wrong")

    grouped_legal_moves = legal_moves_by_piece(case.before)
    for piece_type, label in PIECE_LABELS.items():
        legal_destinations = {chess.square_name(move.to_square) for move in grouped_legal_moves[piece_type]}
        channel = LEGAL_CHANNEL_START + PIECE_TO_RELATIVE_CHANNEL[piece_type]
        heatmap_destinations = nonzero_square_set(case.planes[channel], case.target_color)
        if legal_destinations != heatmap_destinations:
            missing = sorted(legal_destinations - heatmap_destinations)
            extra = sorted(heatmap_destinations - legal_destinations)
            raise AssertionError(f"{case.name}: before_legal_dest_{label} mismatch missing={missing} extra={extra}")

        for legal_move in grouped_legal_moves[piece_type]:
            row, col = square_to_coords(legal_move.to_square, flip)
            if float(case.planes[channel, row, col]) < move_value:
                raise AssertionError(f"{case.name}: legal map missing value for {legal_move.uci()}")

    if case.after.piece_at(case.move.from_square) is None:
        if float(case.planes[REMOVED_DEFENSES_CHANNEL, from_row, from_col]) != 0.0:
            raise AssertionError(f"{case.name}: removed_defenses marks now-empty from-square")

    opponent = not case.target_color
    expected_after_target = side_attack_counts(case.after, case.target_color, flip)
    expected_after_opponent = side_attack_counts(case.after, opponent, flip) * 0.5
    expected_new_target = torch.clamp(
        side_attack_counts(case.after, case.target_color, flip) - side_attack_counts(case.before, case.target_color, flip),
        min=0.0,
    )
    expected_new_opponent = torch.clamp(
        side_attack_counts(case.after, opponent, flip) - side_attack_counts(case.before, opponent, flip),
        min=0.0,
    ) * 0.5

    if not torch.equal(case.planes[AFTER_TARGET_ATTACKS_CHANNEL], expected_after_target):
        raise AssertionError(f"{case.name}: after_target_attacks mismatch")
    if not torch.equal(case.planes[AFTER_OPPONENT_ATTACKS_CHANNEL], expected_after_opponent):
        raise AssertionError(f"{case.name}: after_opponent_attacks mismatch")
    if not torch.equal(case.planes[NEW_TARGET_ATTACKS_CHANNEL], expected_new_target):
        raise AssertionError(f"{case.name}: new_target_attacks mismatch")
    if not torch.equal(case.planes[NEW_OPPONENT_ATTACKS_CHANNEL], expected_new_opponent):
        raise AssertionError(f"{case.name}: new_opponent_attacks mismatch")

    capture_expectations = [
        (TARGET_CAPTURE_TARGETS_CHANNEL, legal_capture_destinations(case.before, case.target_color), "target_capture_targets"),
        (OPPONENT_CAPTURE_TARGETS_CHANNEL, legal_capture_destinations(case.before, opponent), "opponent_capture_targets"),
        (TARGET_THREATENS_PIECES_CHANNEL, threatened_piece_squares(case.before, case.target_color), "target_threatens_pieces"),
        (OPPONENT_THREATENS_PIECES_CHANNEL, threatened_piece_squares(case.before, opponent), "opponent_threatens_pieces"),
        (NEW_TARGET_THREATS_CHANNEL, threatened_piece_squares(case.after, case.target_color) - threatened_piece_squares(case.before, case.target_color), "new_target_threats"),
        (NEW_OPPONENT_THREATS_CHANNEL, threatened_piece_squares(case.after, opponent) - threatened_piece_squares(case.before, opponent), "new_opponent_threats"),
        (TARGET_KING_ZONE_CHANNEL, king_zone_squares(case.before, case.target_color), "target_king_zone"),
        (OPPONENT_KING_ZONE_CHANNEL, king_zone_squares(case.before, opponent), "opponent_king_zone"),
    ]
    for channel, expected_squares, name in capture_expectations:
        actual_squares = weighted_square_set(case.planes[channel], case.target_color)
        if actual_squares != expected_squares:
            missing = sorted(expected_squares - actual_squares)
            extra = sorted(actual_squares - expected_squares)
            raise AssertionError(f"{case.name}: {name} mismatch missing={missing} extra={extra}")

    target_pressure = weighted_square_set(case.planes[TARGET_PRESSURE_OPPONENT_KING_CHANNEL], case.target_color)
    if not target_pressure <= king_zone_squares(case.before, opponent):
        raise AssertionError(f"{case.name}: target king-zone pressure outside opponent king zone")
    opponent_pressure = weighted_square_set(case.planes[OPPONENT_PRESSURE_TARGET_KING_CHANNEL], case.target_color)
    if not opponent_pressure <= king_zone_squares(case.before, case.target_color):
        raise AssertionError(f"{case.name}: opponent king-zone pressure outside target king zone")

    print(f"validated {case.name}")


def draw_board_panel(ax: plt.Axes, board: chess.Board, case: DebugCase, title: str, highlight_move: bool) -> None:
    """Draw one board, oriented exactly like the heatmap planes."""

    board_colors = torch.zeros((8, 8), dtype=torch.float32)
    for row in range(8):
        for col in range(8):
            square_name = tensor_coord_to_square(row, col, case.target_color)
            square = chess.parse_square(square_name)
            board_colors[row, col] = float((chess.square_file(square) + chess.square_rank(square)) % 2)

    ax.imshow(board_colors.numpy(), cmap=ListedColormap(["#f0d9b5", "#b58863"]), vmin=0, vmax=1)

    flip = case.target_color == chess.BLACK
    if highlight_move:
        from_row, from_col = square_to_coords(case.move.from_square, flip)
        to_row, to_col = square_to_coords(case.move.to_square, flip)
        ax.add_patch(Rectangle((from_col - 0.5, from_row - 0.5), 1, 1, fill=False, edgecolor="#2f80ed", linewidth=3))
        ax.add_patch(Rectangle((to_col - 0.5, to_row - 0.5), 1, 1, fill=False, edgecolor="#eb5757", linewidth=3))

    for square, piece in board.piece_map().items():
        row, col = square_to_coords(square, flip)
        symbol = piece.symbol()
        color = "white" if piece.color == chess.WHITE else "black"
        ax.text(
            col,
            row,
            symbol,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": "white" if color == "black" else "black", "alpha": 0.35, "edgecolor": "none"},
        )

    files, ranks = axis_labels(case.target_color)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(8))
    ax.set_xticklabels(files)
    ax.set_yticks(range(8))
    ax.set_yticklabels(ranks)
    ax.set_xticks([idx - 0.5 for idx in range(9)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(9)], minor=True)
    ax.grid(which="minor", color="black", linewidth=0.6, alpha=0.45)
    ax.tick_params(labelsize=6, length=0)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_aspect("equal")


def draw_heatmap_panel(ax: plt.Axes, name: str, plane: torch.Tensor, files: list[str], ranks: list[str]) -> None:
    """Draw one heatmap with square-aligned grid lines."""

    vmax = max(float(plane.max()), 1.0)
    ax.imshow(plane.numpy(), cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{name}\n0=white high=yellow", fontsize=8)
    ax.set_xticks(range(8))
    ax.set_xticklabels(files)
    ax.set_yticks(range(8))
    ax.set_yticklabels(ranks)
    ax.set_xticks([idx - 0.5 for idx in range(9)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(9)], minor=True)
    ax.grid(which="minor", color="#444444", linewidth=0.45, alpha=0.65)
    ax.tick_params(labelsize=6, length=0)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_aspect("equal")


def save_plane_grid(case: DebugCase, output_dir: Path) -> None:
    """Save one grid containing the position and all move planes."""

    cols = 6
    panel_count = len(PLANE_NAMES) + 2
    rows = ceil(panel_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), constrained_layout=True)
    flat_axes = axes.reshape(-1)
    files, ranks = axis_labels(case.target_color)

    draw_board_panel(flat_axes[0], case.before, case, f"BEFORE {case.move.uci()}\nblue=from red=to", highlight_move=True)
    draw_board_panel(flat_axes[1], case.after, case, f"AFTER {case.move.uci()}", highlight_move=False)

    for idx, (name, plane) in enumerate(zip(PLANE_NAMES, case.planes), start=2):
        ax = flat_axes[idx]
        draw_heatmap_panel(ax, name, plane, files, ranks)

    for ax in flat_axes[panel_count:]:
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.name}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved {output_path}")


def write_case_report(case: DebugCase, output_dir: Path) -> None:
    """Write exact legal move and heatmap square listings for manual checks."""

    grouped_legal_moves = legal_moves_by_piece(case.before)
    lines = [
        f"case: {case.name}",
        f"target: {'white' if case.target_color == chess.WHITE else 'black'}",
        f"move: {case.move.uci()}",
        f"before_fen: {case.before.fen()}",
        "",
        "Before-legal destination heatmaps are destination-square unions by moving piece type in the BEFORE board.",
        "For example, before_legal_dest_Q contains the to-squares of all legal queen moves before the shown move is played.",
        "These planes intentionally do not show moves that become legal only on the AFTER board.",
        "",
        "New attack/threat planes are non-negative after-minus-before increases.",
        "Capture targets are legal capture destination squares from the BEFORE board, evaluated once for each side.",
        "Threat planes mark enemy occupied squares attacked by the named side.",
        "King-zone pressure is attack/control inside the opponent or target king zone in the BEFORE board.",
        "",
        "legal moves by piece:",
    ]

    for piece_type, label in PIECE_LABELS.items():
        moves = sorted(move.uci() for move in grouped_legal_moves[piece_type])
        channel = LEGAL_CHANNEL_START + PIECE_TO_RELATIVE_CHANNEL[piece_type]
        heatmap_squares = sorted(nonzero_square_set(case.planes[channel], case.target_color))
        lines.append(f"  {label}: moves={moves}")
        lines.append(f"     heatmap_squares={heatmap_squares}")

    lines.extend(
        [
            "",
            "capture/threat/king-zone heatmaps:",
            f"  target_capture_targets={sorted(nonzero_square_set(case.planes[TARGET_CAPTURE_TARGETS_CHANNEL], case.target_color))}",
            f"  opponent_capture_targets={sorted(nonzero_square_set(case.planes[OPPONENT_CAPTURE_TARGETS_CHANNEL], case.target_color))}",
            f"  target_threatens_pieces={sorted(nonzero_square_set(case.planes[TARGET_THREATENS_PIECES_CHANNEL], case.target_color))}",
            f"  opponent_threatens_pieces={sorted(nonzero_square_set(case.planes[OPPONENT_THREATENS_PIECES_CHANNEL], case.target_color))}",
            f"  new_target_threats={sorted(nonzero_square_set(case.planes[NEW_TARGET_THREATS_CHANNEL], case.target_color))}",
            f"  new_opponent_threats={sorted(nonzero_square_set(case.planes[NEW_OPPONENT_THREATS_CHANNEL], case.target_color))}",
            f"  target_king_zone={sorted(nonzero_square_set(case.planes[TARGET_KING_ZONE_CHANNEL], case.target_color))}",
            f"  opponent_king_zone={sorted(nonzero_square_set(case.planes[OPPONENT_KING_ZONE_CHANNEL], case.target_color))}",
            f"  target_pressure_opponent_king={sorted(nonzero_square_set(case.planes[TARGET_PRESSURE_OPPONENT_KING_CHANNEL], case.target_color))}",
            f"  opponent_pressure_target_king={sorted(nonzero_square_set(case.planes[OPPONENT_PRESSURE_TARGET_KING_CHANNEL], case.target_color))}",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{case.name}.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {report_path}")


def build_case(case_name: str, board: chess.Board, move_uci: str, target_color: bool) -> DebugCase:
    """Create planes for one known board and move."""

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal visualized move {move_uci} in position {board.fen()}")

    before = board.copy(stack=False)
    after = board.copy(stack=False)
    after.push(move)
    planes = move_planes(before, after, move, target_color=target_color, board_flip=True)
    return DebugCase(case_name, before, after, move, target_color, planes)


def board_after_moves(*move_uci_list: str) -> chess.Board:
    """Return a board after applying a short legal move sequence."""

    board = chess.Board()
    for move_uci in move_uci_list:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal setup move {move_uci} in position {board.fen()}")
        board.push(move)
    return board


def main() -> None:
    """Generate debug summaries and PNGs for known positions."""

    if MOVE_INPUT_CHANNELS != len(PLANE_NAMES):
        raise RuntimeError(f"Expected {len(PLANE_NAMES)} move channels, got {MOVE_INPUT_CHANNELS}")
    if LEGAL_CHANNELS != len(PIECE_LABELS):
        raise RuntimeError(f"Expected one legal channel per piece type, got {LEGAL_CHANNELS}")

    output_dir = ROOT / "results" / "plane_debug"

    after_e2e4 = board_after_moves("e2e4")
    after_e4_d5 = board_after_moves("e2e4", "d7d5")
    after_e4_e5 = board_after_moves("e2e4", "e7e5")
    after_e4_e5_nf3_nc6 = board_after_moves("e2e4", "e7e5", "g1f3", "b8c6")
    rook_file_board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")

    cases = [
        build_case("target_white_e2e4", chess.Board(), "e2e4", chess.WHITE),
        build_case("target_black_e7e5_after_e2e4", after_e2e4, "e7e5", chess.BLACK),
        build_case("opponent_black_e7e5_after_e2e4", after_e2e4, "e7e5", chess.WHITE),
        build_case("target_white_exd5_capture", after_e4_d5, "e4d5", chess.WHITE),
        build_case("target_white_ng1f3_knight", after_e4_e5, "g1f3", chess.WHITE),
        build_case("target_white_bf1c4_bishop", after_e4_e5_nf3_nc6, "f1c4", chess.WHITE),
        build_case("target_white_re2e7_rook_file", rook_file_board, "e2e7", chess.WHITE),
    ]

    for case in cases:
        validate_case(case)
        summarize_planes(case)
        save_plane_grid(case, output_dir)
        write_case_report(case, output_dir)

    print(f"\nWrote {len(cases)} heatmap PNGs to {output_dir}")


if __name__ == "__main__":
    main()
