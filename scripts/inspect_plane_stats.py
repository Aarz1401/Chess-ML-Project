"""Inspect channel-scale statistics for materialized chess style tensors."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    print("Importing torch and dataset utilities...", flush=True)

import torch

from board_dataset import (
    FULL_INPUT_PLANE_NAMES,
    INPUT_CHANNELS,
    TensorConfig,
    balance_records,
    class_counts,
    collect_game_records,
    materialize_samples,
    split_records,
)


def compute_plane_stats(samples: list[tuple[torch.Tensor, torch.Tensor, int]]) -> list[dict[str, float | int | str]]:
    """Compute per-channel stats across all non-padded ply tensors."""

    sums = torch.zeros(INPUT_CHANNELS, dtype=torch.float64)
    sums_sq = torch.zeros(INPUT_CHANNELS, dtype=torch.float64)
    mins = torch.full((INPUT_CHANNELS,), float("inf"), dtype=torch.float64)
    maxs = torch.full((INPUT_CHANNELS,), float("-inf"), dtype=torch.float64)
    nonzero = torch.zeros(INPUT_CHANNELS, dtype=torch.float64)
    total_cells = 0

    for sequence, padding_mask, _ in samples:
        valid = sequence[~padding_mask].float()
        if valid.numel() == 0:
            continue

        total_cells += valid.shape[0] * valid.shape[2] * valid.shape[3]
        sums += valid.sum(dim=(0, 2, 3)).double()
        sums_sq += valid.square().sum(dim=(0, 2, 3)).double()
        mins = torch.minimum(mins, valid.amin(dim=(0, 2, 3)).double())
        maxs = torch.maximum(maxs, valid.amax(dim=(0, 2, 3)).double())
        nonzero += valid.ne(0).sum(dim=(0, 2, 3)).double()

    if total_cells == 0:
        raise RuntimeError("No non-padded ply tensors were available for plane stats.")

    rows: list[dict[str, float | int | str]] = []
    for channel, name in enumerate(FULL_INPUT_PLANE_NAMES):
        mean = sums[channel] / total_cells
        variance = (sums_sq[channel] / total_cells) - mean.square()
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        rows.append(
            {
                "channel": channel,
                "name": name,
                "min": float(mins[channel]),
                "max": float(maxs[channel]),
                "mean": float(mean),
                "std": float(std),
                "nonzero_fraction": float(nonzero[channel] / total_cells),
            }
        )

    return rows


def save_stats(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    """Save plane stats as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["channel", "name", "min", "max", "mean", "std", "nonzero_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect scale statistics for every input plane.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "pgn"))
    parser.add_argument("--output", default=str(ROOT / "results" / "plane_stats" / "plane_stats.csv"))
    parser.add_argument("--max-games-per-player", type=int, default=100)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="train")
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--skip-plies", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-board-flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Scanning PGNs under {args.data_dir}...", flush=True)
    records = collect_game_records(Path(args.data_dir), min_plies=1)
    print("Usable game counts:", class_counts(records), flush=True)
    records = balance_records(records, max_games_per_player=args.max_games_per_player, seed=args.seed)
    print("Balanced game counts:", class_counts(records), flush=True)

    if args.split == "all":
        selected_records = records
    else:
        selected_records = split_records(records, seed=args.seed)[args.split]
    print(f"Selected {args.split} counts:", class_counts(selected_records), flush=True)

    config = TensorConfig(
        sequence_length=args.sequence_length,
        skip_plies=args.skip_plies,
        board_flip=not args.no_board_flip,
    )
    print("Materializing selected games...", flush=True)
    samples = materialize_samples(selected_records, config)
    print("Computing channel statistics...", flush=True)
    rows = compute_plane_stats(samples)
    output_path = Path(args.output)
    save_stats(rows, output_path)

    print(f"Saved {output_path}")
    print("Largest channel max values:")
    for row in sorted(rows, key=lambda item: float(item["max"]), reverse=True)[:10]:
        print(f"  {int(row['channel']):02d} {row['name']}: max={float(row['max']):.4f} std={float(row['std']):.4f}")

    print("Largest channel std values:")
    for row in sorted(rows, key=lambda item: float(item["std"]), reverse=True)[:10]:
        print(f"  {int(row['channel']):02d} {row['name']}: std={float(row['std']):.4f} max={float(row['max']):.4f}")


if __name__ == "__main__":
    main()
