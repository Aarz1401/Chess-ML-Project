"""Train the compact multiclass chess style classifier.

Example local smoke test:

    python3 scripts/train_multiclass.py --max-games-per-player 5 --epochs 1 --stockfish-depth 1

Example proof run:

    python3 scripts/train_multiclass.py --max-games-per-player 500 --epochs 10 --stockfish-depth 2 --multipv 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Matplotlib can fail on clusters or sandboxes if the default user cache is not
# writable. Put its cache inside this project and use a non-GUI backend.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import chess.engine
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from board_dataset import (
    ID_TO_LABEL,
    INPUT_CHANNELS,
    ChessStyleDataset,
    TensorConfig,
    balance_records,
    class_counts,
    collect_game_records,
    materialize_samples,
    split_records,
)
from model import ChessStyleTransformer


def set_seed(seed: int) -> None:
    """Make data shuffling and model initialization repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use a GPU on HPC when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    """Return loss, accuracy, true labels, and predicted labels."""

    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for board, mask, y in loader:
            board = board.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(board, mask)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    return total_loss / max(total, 1), accuracy, y_true, y_pred


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_dir: Path) -> None:
    """Save confusion matrix as both CSV and a readable PNG figure."""

    labels = list(ID_TO_LABEL)
    label_names = [ID_TO_LABEL[idx] for idx in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = output_dir / "confusion_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", *label_names])
        for idx, row in zip(labels, matrix):
            writer.writerow([ID_TO_LABEL[idx], *row.tolist()])

    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title("Multiclass Chess Style Confusion Matrix")
        ax.set_xlabel("Predicted player")
        ax.set_ylabel("True player")
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: saved confusion_matrix.csv but could not save PNG: {exc}")


def save_classification_report(y_true: list[int], y_pred: list[int], output_dir: Path) -> None:
    """Save sklearn's per-class report in machine-readable JSON and CSV."""

    labels = list(ID_TO_LABEL)
    label_names = [ID_TO_LABEL[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(output_dir / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1_score", "support"])
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                writer.writerow(
                    [
                        label,
                        metrics.get("precision", ""),
                        metrics.get("recall", ""),
                        metrics.get("f1-score", ""),
                        metrics.get("support", ""),
                    ]
                )


def train(args: argparse.Namespace) -> None:
    """Run one full experiment and save all outputs under args.output_dir."""

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    records = collect_game_records(data_dir=data_dir, min_plies=1)
    print("Usable game counts:", class_counts(records))

    records = balance_records(records, max_games_per_player=args.max_games_per_player, seed=args.seed)
    print("Balanced game counts:", class_counts(records))

    splits = split_records(records, seed=args.seed)
    print("Split counts:")
    for split_name, split_records_for_name in splits.items():
        print(f"  {split_name}: {class_counts(split_records_for_name)}")

    config = TensorConfig(
        sequence_length=args.sequence_length,
        stockfish_depth=args.stockfish_depth,
        multipv=args.multipv,
        board_flip=not args.no_board_flip,
    )

    engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    try:
        print("Materializing train tensors...")
        train_samples = materialize_samples(splits["train"], engine, config)
        print("Materializing validation tensors...")
        val_samples = materialize_samples(splits["val"], engine, config)
        print("Materializing test tensors...")
        test_samples = materialize_samples(splits["test"], engine, config)
    finally:
        engine.quit()

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("At least one split is empty after tensor materialization.")

    train_loader = DataLoader(ChessStyleDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ChessStyleDataset(val_samples), batch_size=args.batch_size)
    test_loader = DataLoader(ChessStyleDataset(test_samples), batch_size=args.batch_size)

    device = get_device()
    print(f"Using device: {device}")
    model = ChessStyleTransformer(num_classes=len(ID_TO_LABEL), in_channels=INPUT_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_true: list[int] = []
        train_pred: list[int] = []

        for board, mask, y in train_loader:
            board = board.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(board, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.size(0)
            train_total += y.size(0)
            train_true.extend(y.cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = accuracy_score(train_true, train_pred)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, test_true, test_pred = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": f1_score(test_true, test_pred, average="macro", zero_division=0),
            "test_weighted_f1": f1_score(test_true, test_pred, average="weighted", zero_division=0),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_acc={train_acc * 100:.1f}% val_acc={val_acc * 100:.1f}% "
            f"test_acc={test_acc * 100:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": ID_TO_LABEL,
                    "input_channels": INPUT_CHANNELS,
                    "args": vars(args),
                },
                checkpoint_path,
            )

        with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    print(f"Best-checkpoint test_loss={test_loss:.4f} test_acc={test_acc * 100:.1f}%")

    save_classification_report(test_true, test_pred, output_dir)
    save_confusion_matrix(test_true, test_pred, output_dir)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "input_channels": INPUT_CHANNELS,
                "labels": ID_TO_LABEL,
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an 8-class chess style classifier")
    parser.add_argument("--data-dir", default=str(ROOT / "PGN_data"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "multiclass_proof"))
    parser.add_argument("--engine-path", default="stockfish")
    parser.add_argument("--max-games-per-player", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--stockfish-depth", type=int, default=2)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-board-flip", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
