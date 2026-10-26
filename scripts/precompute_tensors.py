"""Precompute game tensors once and cache them to disk.

Tensor construction is pure-Python chess logic and dominates every training
run, so this script does it a single time with a multiprocessing pool and
writes memory-mapped arrays that training jobs can stream.

Example (full dataset, all eight players):

    python3 scripts/precompute_tensors.py \
      --cache-dir results/tensor_cache_full \
      --max-games-per-player 4000 \
      --sequence-length 60 \
      --skip-plies 12 \
      --workers 8

Example smoke test:

    python3 scripts/precompute_tensors.py \
      --cache-dir results/tensor_cache_smoke \
      --max-games-per-player 8 --workers 2
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import multiprocessing as mp

import chess
import chess.pgn
import numpy as np

from board_dataset import (
    INPUT_CHANNELS,
    LABEL_NAMES,
    TensorConfig,
    game_to_model_tensors,
    normalize_player_name,
)
from tensor_cache import SPLIT_NAMES, save_manifest, split_file


@dataclass(frozen=True)
class OffsetRecord:
    """One playable game located by byte offset inside its PGN file."""

    pgn_path: str
    offset: int
    label_name: str
    label_id: int
    target_color: bool
    white: str
    black: str
    date: str
    event: str
    result: str


def target_color_from_headers(headers: chess.pgn.Headers, label_name: str) -> bool | None:
    """Return which side the labeled player took, or None if absent."""

    target = normalize_player_name(label_name)
    if not target:
        return None
    if target in normalize_player_name(headers.get("White", "")):
        return chess.WHITE
    if target in normalize_player_name(headers.get("Black", "")):
        return chess.BLACK
    return None


def scan_label_file(pgn_path: Path, label_name: str, label_id: int) -> list[OffsetRecord]:
    """Collect byte offsets of usable games without parsing move text."""

    records: list[OffsetRecord] = []
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
        while True:
            offset = pgn_file.tell()
            headers = chess.pgn.read_headers(pgn_file)
            if headers is None:
                break

            target_color = target_color_from_headers(headers, label_name)
            if target_color is None:
                continue

            records.append(
                OffsetRecord(
                    pgn_path=str(pgn_path),
                    offset=offset,
                    label_name=label_name,
                    label_id=label_id,
                    target_color=target_color,
                    white=headers.get("White", ""),
                    black=headers.get("Black", ""),
                    date=headers.get("Date", ""),
                    event=headers.get("Event", ""),
                    result=headers.get("Result", ""),
                )
            )
    return records


def balance_and_split(
    records_by_label: dict[str, list[OffsetRecord]],
    max_games_per_player: int,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[OffsetRecord]]:
    """Downsample every label to a common count, then split stratified."""

    rng = random.Random(seed)
    target_count = min(min(len(items) for items in records_by_label.values()), max_games_per_player)
    print(f"Balancing every label to {target_count} games.")

    splits: dict[str, list[OffsetRecord]] = {name: [] for name in SPLIT_NAMES}
    for label_name in sorted(records_by_label):
        items = list(records_by_label[label_name])
        rng.shuffle(items)
        items = items[:target_count]
        train_end = int(len(items) * train_frac)
        val_end = train_end + int(len(items) * val_frac)
        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])

    for items in splits.values():
        rng.shuffle(items)
    return splits


_WORKER_CONFIG: TensorConfig | None = None
_WORKER_MIN_VALID = 1


def _init_worker(config: TensorConfig, min_valid_plies: int) -> None:
    global _WORKER_CONFIG, _WORKER_MIN_VALID
    _WORKER_CONFIG = config
    _WORKER_MIN_VALID = min_valid_plies


def _build_one(task: tuple[int, str, int, int, bool]) -> tuple[int, np.ndarray, np.ndarray, int] | None:
    """Build one game tensor inside a worker process."""

    record_idx, pgn_path, offset, label_id, target_color = task
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
        pgn_file.seek(offset)
        game = chess.pgn.read_game(pgn_file)
    if game is None:
        return None

    tensors = game_to_model_tensors(game, bool(target_color), _WORKER_CONFIG)
    if tensors is None:
        return None

    sequence, padding_mask = tensors
    valid_plies = int((~padding_mask).sum())
    if valid_plies < _WORKER_MIN_VALID:
        return None

    return record_idx, sequence.numpy(), padding_mask.numpy(), label_id


def materialize_split(
    split: str,
    records: list[OffsetRecord],
    cache_dir: Path,
    config: TensorConfig,
    min_valid_plies: int,
    workers: int,
) -> int:
    """Convert one split's games to tensors and write memory-mapped arrays."""

    n_records = len(records)
    sequences_path = split_file(cache_dir, split, "sequences")
    staging_path = sequences_path.with_suffix(".building.npy")
    shape = (n_records, config.sequence_length, INPUT_CHANNELS, 8, 8)
    sequences = np.lib.format.open_memmap(staging_path, mode="w+", dtype=np.float16, shape=shape)
    masks = np.ones((n_records, config.sequence_length), dtype=bool)
    labels = np.zeros((n_records,), dtype=np.int64)

    tasks = [
        (idx, record.pgn_path, record.offset, record.label_id, record.target_color)
        for idx, record in enumerate(records)
    ]

    metadata_rows: list[list[object]] = []
    write_idx = 0
    started = time.time()

    with mp.get_context("spawn").Pool(
        processes=workers, initializer=_init_worker, initargs=(config, min_valid_plies)
    ) as pool:
        for result in pool.imap_unordered(_build_one, tasks, chunksize=4):
            if result is None:
                continue

            record_idx, sequence, padding_mask, label_id = result
            record = records[record_idx]
            sequences[write_idx] = sequence
            masks[write_idx] = padding_mask
            labels[write_idx] = label_id
            metadata_rows.append(
                [
                    write_idx,
                    label_id,
                    record.label_name,
                    record.pgn_path,
                    record.offset,
                    int((~padding_mask).sum()),
                    record.white,
                    record.black,
                    record.date,
                    record.event,
                    record.result,
                ]
            )
            write_idx += 1

            if write_idx % 250 == 0:
                rate = write_idx / max(time.time() - started, 1e-6)
                remaining = (n_records - write_idx) / max(rate, 1e-6)
                print(
                    f"  [{split}] {write_idx}/{n_records} games "
                    f"({rate:.1f} games/s, ~{remaining / 60:.0f} min left)",
                    flush=True,
                )

    sequences.flush()
    del sequences

    if write_idx == n_records:
        staging_path.replace(sequences_path)
    else:
        # Some games were skipped (too short / unreadable); write a trimmed copy.
        print(f"  [{split}] trimming {n_records} -> {write_idx} games")
        built = np.load(staging_path, mmap_mode="r")
        trimmed = np.lib.format.open_memmap(
            sequences_path, mode="w+", dtype=np.float16, shape=(write_idx, *shape[1:])
        )
        chunk = 256
        for start in range(0, write_idx, chunk):
            end = min(start + chunk, write_idx)
            trimmed[start:end] = built[start:end]
        trimmed.flush()
        del built, trimmed
        staging_path.unlink()

    np.save(split_file(cache_dir, split, "masks"), masks[:write_idx])
    np.save(split_file(cache_dir, split, "labels"), labels[:write_idx])

    with open(cache_dir / f"{split}_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_index",
                "label_id",
                "label_name",
                "pgn_path",
                "byte_offset",
                "valid_plies",
                "white",
                "black",
                "date",
                "event",
                "result",
            ]
        )
        writer.writerows(metadata_rows)

    print(f"  [{split}] cached {write_idx} games ({n_records - write_idx} skipped)")
    return write_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and cache game tensors.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "pgn"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--labels", nargs="+", default=list(LABEL_NAMES), help="Player labels to include.")
    parser.add_argument("--max-games-per-player", type=int, default=4000)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--skip-plies", type=int, default=12)
    parser.add_argument("--target-moves-only", action="store_true")
    parser.add_argument("--no-board-flip", action="store_true")
    parser.add_argument(
        "--min-valid-plies",
        type=int,
        default=20,
        help="Skip games with fewer recorded plies than this after skip-plies.",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_labels = list(dict.fromkeys(args.labels))
    unknown = sorted(set(selected_labels) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}. Available: {LABEL_NAMES}")

    print(f"Scanning PGN headers for {len(selected_labels)} labels...")
    records_by_label: dict[str, list[OffsetRecord]] = {}
    for label_id, label_name in enumerate(selected_labels):
        pgn_path = data_dir / f"{label_name}.pgn"
        if not pgn_path.exists():
            raise FileNotFoundError(f"Missing PGN file: {pgn_path}")
        found = scan_label_file(pgn_path, label_name, label_id)
        records_by_label[label_name] = found
        print(f"  {label_name}: {len(found)} usable games")

    splits = balance_and_split(
        records_by_label,
        max_games_per_player=args.max_games_per_player,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    for split in SPLIT_NAMES:
        print(f"  {split}: {len(splits[split])} games")

    config = TensorConfig(
        sequence_length=args.sequence_length,
        skip_plies=args.skip_plies,
        target_moves_only=args.target_moves_only,
        board_flip=not args.no_board_flip,
    )

    counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        print(f"Materializing {split} tensors with {args.workers} workers...")
        counts[split] = materialize_split(
            split,
            splits[split],
            cache_dir,
            config,
            min_valid_plies=args.min_valid_plies,
            workers=args.workers,
        )

    save_manifest(
        cache_dir,
        {
            "labels": selected_labels,
            "input_channels": INPUT_CHANNELS,
            "sequence_length": args.sequence_length,
            "skip_plies": args.skip_plies,
            "target_moves_only": args.target_moves_only,
            "board_flip": not args.no_board_flip,
            "min_valid_plies": args.min_valid_plies,
            "max_games_per_player": args.max_games_per_player,
            "seed": args.seed,
            "counts": counts,
        },
    )
    print(f"Cache complete at {cache_dir}")


if __name__ == "__main__":
    main()
