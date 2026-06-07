"""Export learned game embeddings from a trained multiclass checkpoint."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import torch
from torch.utils.data import DataLoader

from board_dataset import (
    ID_TO_LABEL,
    INPUT_CHANNELS,
    LABEL_NAMES,
    LABEL_TO_ID,
    ChessStyleDataset,
    TensorConfig,
    balance_records,
    class_counts,
    collect_game_records,
    materialize_samples,
    split_records,
)
from model import ChessStyleTransformer


def get_device() -> torch.device:
    """Use a GPU when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_labels(checkpoint: dict) -> dict[int, str]:
    """Read label names from a checkpoint with a stable fallback."""

    labels = checkpoint.get("labels", ID_TO_LABEL)
    return {int(key): value for key, value in labels.items()}


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ChessStyleTransformer, dict[int, str], dict[str, object]]:
    """Load a trained model and its label mapping."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    labels = checkpoint_labels(checkpoint)
    input_channels = int(checkpoint.get("input_channels", INPUT_CHANNELS))
    metric_dim = int(checkpoint.get("metric_dim", 128))
    model = ChessStyleTransformer(num_classes=len(labels), in_channels=input_channels, metric_dim=metric_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels, checkpoint.get("args", {})


def encode_samples(
    model: ChessStyleTransformer,
    samples: list[tuple[torch.Tensor, torch.Tensor, int]],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model.encode_metric over materialized samples."""

    loader = DataLoader(ChessStyleDataset(samples), batch_size=batch_size)
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for board, mask, y in loader:
            board = board.to(device)
            mask = mask.to(device)
            embedding = model.encode_metric(board, mask)
            embeddings.append(embedding.cpu())
            labels.append(y.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def save_embedding_split(
    split_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    label_names: dict[int, str],
    output_dir: Path,
) -> None:
    """Save one split as a tensor file plus traceable CSV metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": split_name,
            "embeddings": embeddings,
            "labels": labels,
            "label_names": label_names,
            "input_channels": INPUT_CHANNELS,
            "embedding_dim": embeddings.shape[1],
            "embedding_kind": "metric",
        },
        output_dir / f"embeddings_{split_name}.pt",
    )

    with open(output_dir / f"embeddings_{split_name}_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "split", "label_id", "label_name"])
        for row_idx, label_id in enumerate(labels.tolist()):
            writer.writerow([row_idx, split_name, label_id, label_names[int(label_id)]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned game embeddings from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--data-dir", default=str(ROOT / "PGN_data"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "embeddings"))
    parser.add_argument("--max-games-per-player", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--skip-plies", type=int, default=0, help="Skip this many opening plies before recording tensors.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-board-flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)
    model, label_names, checkpoint_args = load_model(Path(args.checkpoint), device)

    records = collect_game_records(Path(args.data_dir), min_plies=1)
    selected_labels = [label_names[idx] for idx in sorted(label_names)]
    label_to_id = {name: idx for idx, name in label_names.items()}
    records = [
        replace(record, label_id=label_to_id[record.label_name])
        for record in records
        if record.label_name in label_to_id
    ]
    LABEL_NAMES[:] = selected_labels
    LABEL_TO_ID.clear()
    LABEL_TO_ID.update(label_to_id)
    ID_TO_LABEL.clear()
    ID_TO_LABEL.update(label_names)

    print("Usable game counts:", class_counts(records))
    records = balance_records(records, max_games_per_player=args.max_games_per_player, seed=args.seed)
    print("Balanced game counts:", class_counts(records))
    splits = split_records(records, seed=args.seed)

    config = TensorConfig(
        sequence_length=args.sequence_length,
        skip_plies=args.skip_plies,
        target_moves_only=bool(checkpoint_args.get("target_moves_only", False)),
        board_flip=not args.no_board_flip,
    )

    for split_name in args.splits:
        split_records_for_name = splits[split_name]
        print(f"Materializing {split_name} tensors...")
        samples = materialize_samples(split_records_for_name, config)
        print(f"Encoding {split_name} embeddings...")
        embeddings, labels = encode_samples(model, samples, args.batch_size, device)
        save_embedding_split(split_name, embeddings, labels, label_names, output_dir)
        print(f"Saved {split_name}: embeddings={tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
