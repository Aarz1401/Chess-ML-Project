"""Visualize target-emphasized per-game aggregate planes."""

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
import torch
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from board_dataset import (
    DIRECT_GAME_CHANNEL_START,
    DIRECT_GAME_CHANNELS,
    GAME_CHANNELS,
    GAME_DIRECT_TARGET_CAPTURE_TO_CHANNEL,
    GAME_DIRECT_TARGET_CHECK_TO_CHANNEL,
    GAME_DIRECT_TARGET_OCCUPANCY_START,
    GAME_OPPONENT_FROM_CHANNEL,
    GAME_OPPONENT_TO_CHANNEL,
    GAME_PLANE_NAMES,
    GAME_TARGET_TO_START,
    INPUT_CHANNELS,
    MOVE_INPUT_CHANNELS,
    build_game_tensor,
    direct_game_planes_from_ply_history,
    move_planes,
    square_to_coords,
)

HEATMAP_CMAP = LinearSegmentedColormap.from_list("game_plane_heat", ["#ffffff", "#fff3a3", "#ffd000"])


@dataclass(frozen=True)
class GameDebugCase:
    """One short game and its aggregate planes."""

    name: str
    move_uci_list: list[str]
    target_color: bool
    move_sequence: torch.Tensor
    direct_game_planes: torch.Tensor
    game_planes: torch.Tensor
    full_sequence: torch.Tensor


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
    """Return absolute file/rank labels for displayed tensor orientation."""

    files = [tensor_coord_to_square(7, col, target_color, board_flip)[0] for col in range(8)]
    ranks = [tensor_coord_to_square(row, 0, target_color, board_flip)[1] for row in range(8)]
    return files, ranks


def nonzero_coords(plane: torch.Tensor, target_color: bool) -> str:
    """Return square-name, value pairs for nonzero cells."""

    coords: list[tuple[str, float]] = []
    for row, col in plane.nonzero(as_tuple=False).tolist():
        coords.append((tensor_coord_to_square(row, col, target_color), float(plane[row, col])))
    return str(coords)


def build_game_case(name: str, move_uci_list: list[str], target_color: bool) -> GameDebugCase:
    """Build all per-move and per-game planes for a known short game."""

    board = chess.Board()
    move_tensors: list[torch.Tensor] = []
    ply_history: list[tuple[chess.Board, chess.Move, chess.Board]] = []
    for move_uci in move_uci_list:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {move_uci} in {board.fen()}")

        before = board.copy(stack=False)
        board.push(move)
        after = board.copy(stack=False)
        ply_history.append((before, move, after))
        move_tensors.append(move_planes(before, after, move, target_color=target_color, board_flip=True))

    move_sequence = torch.stack(move_tensors, dim=0)
    direct_game_planes = direct_game_planes_from_ply_history(ply_history, target_color=target_color, board_flip=True)
    full_sequence, game_planes = build_game_tensor(move_sequence, direct_game_planes=direct_game_planes)
    return GameDebugCase(name, move_uci_list, target_color, move_sequence, direct_game_planes, game_planes, full_sequence)


def validate_case(case: GameDebugCase) -> None:
    """Check shape and target-emphasis invariants."""

    expected_move_shape = (len(case.move_uci_list), MOVE_INPUT_CHANNELS, 8, 8)
    expected_full_shape = (len(case.move_uci_list), INPUT_CHANNELS, 8, 8)
    if tuple(case.move_sequence.shape) != expected_move_shape:
        raise AssertionError(f"{case.name}: expected {expected_move_shape}, got {tuple(case.move_sequence.shape)}")
    if tuple(case.game_planes.shape) != (GAME_CHANNELS, 8, 8):
        raise AssertionError(f"{case.name}: bad game plane shape {tuple(case.game_planes.shape)}")
    if tuple(case.direct_game_planes.shape) != (DIRECT_GAME_CHANNELS, 8, 8):
        raise AssertionError(f"{case.name}: bad direct game plane shape {tuple(case.direct_game_planes.shape)}")
    if tuple(case.full_sequence.shape) != expected_full_shape:
        raise AssertionError(f"{case.name}: expected {expected_full_shape}, got {tuple(case.full_sequence.shape)}")

    target_to_sum = float(case.game_planes[GAME_TARGET_TO_START : GAME_TARGET_TO_START + 6].sum())
    opponent_to_sum = float(case.game_planes[GAME_OPPONENT_TO_CHANNEL].sum())
    opponent_from_sum = float(case.game_planes[GAME_OPPONENT_FROM_CHANNEL].sum())
    if target_to_sum <= opponent_to_sum:
        raise AssertionError(f"{case.name}: target move-to aggregate is not emphasized")
    if opponent_from_sum != opponent_to_sum:
        raise AssertionError(f"{case.name}: opponent from/to aggregate sums differ")
    if not torch.equal(case.game_planes[DIRECT_GAME_CHANNEL_START:], case.direct_game_planes):
        raise AssertionError(f"{case.name}: direct game planes were not concatenated into the game block")

    occupancy_sum = float(case.game_planes[GAME_DIRECT_TARGET_OCCUPANCY_START : GAME_DIRECT_TARGET_OCCUPANCY_START + 6].sum())
    capture_sum = float(case.game_planes[GAME_DIRECT_TARGET_CAPTURE_TO_CHANNEL].sum())
    check_sum = float(case.game_planes[GAME_DIRECT_TARGET_CHECK_TO_CHANNEL].sum())
    if occupancy_sum <= 0.0:
        raise AssertionError(f"{case.name}: direct occupancy planes are empty")

    print(f"validated {case.name}")
    print(f"  move_sequence={tuple(case.move_sequence.shape)}")
    print(f"  direct_game_planes={tuple(case.direct_game_planes.shape)}")
    print(f"  game_planes={tuple(case.game_planes.shape)}")
    print(f"  full_sequence={tuple(case.full_sequence.shape)}")
    print(f"  target_to_sum={target_to_sum:.3f} opponent_to_sum={opponent_to_sum:.3f}")
    print(f"  direct_occupancy_sum={occupancy_sum:.3f} capture_sum={capture_sum:.3f} check_sum={check_sum:.3f}")


def draw_heatmap_panel(ax: plt.Axes, name: str, plane: torch.Tensor, files: list[str], ranks: list[str]) -> None:
    """Draw one game aggregate heatmap."""

    vmax = max(float(plane.max()), 1.0)
    ax.imshow(plane.numpy(), cmap=HEATMAP_CMAP, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(name, fontsize=8)
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


def save_game_grid(case: GameDebugCase, output_dir: Path) -> None:
    """Save one grid of all game aggregate planes."""

    cols = 6
    rows = ceil(len(GAME_PLANE_NAMES) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), constrained_layout=True)
    flat_axes = axes.reshape(-1)
    files, ranks = axis_labels(case.target_color)

    for idx, (name, plane) in enumerate(zip(GAME_PLANE_NAMES, case.game_planes)):
        draw_heatmap_panel(flat_axes[idx], name, plane, files, ranks)

    for ax in flat_axes[len(GAME_PLANE_NAMES) :]:
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.name}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved {output_path}")


def write_game_report(case: GameDebugCase, output_dir: Path) -> None:
    """Write exact numeric summaries for every aggregate plane."""

    lines = [
        f"case: {case.name}",
        f"target: {'white' if case.target_color == chess.WHITE else 'black'}",
        f"moves: {case.move_uci_list}",
        f"move_sequence_shape: {tuple(case.move_sequence.shape)}",
        f"direct_game_planes_shape: {tuple(case.direct_game_planes.shape)}",
        f"game_planes_shape: {tuple(case.game_planes.shape)}",
        f"full_sequence_shape: {tuple(case.full_sequence.shape)}",
        "",
        "Game planes are averaged over target or opponent plies.",
        "Target maps use full scale. Opponent context maps are additionally scaled by 0.5.",
        "Direct game planes come from PGN board history, not from per-move heatmap channels.",
        "",
    ]

    for name, plane in zip(GAME_PLANE_NAMES, case.game_planes):
        lines.append(
            f"{name}: min={float(plane.min()):.3f} max={float(plane.max()):.3f} "
            f"sum={float(plane.sum()):.3f} nonzero={nonzero_coords(plane, case.target_color)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{case.name}.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {report_path}")


def main() -> None:
    """Generate aggregate game heatmap debug outputs."""

    if GAME_CHANNELS != len(GAME_PLANE_NAMES):
        raise RuntimeError(f"Expected {len(GAME_PLANE_NAMES)} game channels, got {GAME_CHANNELS}")

    line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]
    capture_check_line = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    output_dir = ROOT / "results" / "game_plane_debug"
    cases = [
        build_game_case("short_italian_target_white", line, chess.WHITE),
        build_game_case("short_italian_target_black", line, chess.BLACK),
        build_game_case("target_white_capture_check", capture_check_line, chess.WHITE),
    ]

    for case in cases:
        validate_case(case)
        save_game_grid(case, output_dir)
        write_game_report(case, output_dir)

    print(f"\nWrote {len(cases)} game heatmap PNGs to {output_dir}")


if __name__ == "__main__":
    main()
