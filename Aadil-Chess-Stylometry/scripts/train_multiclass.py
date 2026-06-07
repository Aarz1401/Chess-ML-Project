"""Train the compact multiclass chess style classifier.

Example local smoke test:

    python3 scripts/train_multiclass.py --max-games-per-player 5 --epochs 1

Example proof run:

    python3 scripts/train_multiclass.py --max-games-per-player 500 --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Matplotlib can fail on clusters or sandboxes if the default user cache is not
# writable. Put its cache inside this project and use a non-GUI backend.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Sampler

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


def set_seed(seed: int) -> None:
    """Make data shuffling and model initialization repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use a GPU on HPC when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield batches with multiple games from several players.

    Supervised contrastive loss needs at least two samples from the same class
    in a batch. This sampler makes that an explicit training condition instead
    of hoping random shuffling produces useful positives.
    """

    def __init__(
        self,
        samples: list[tuple[torch.Tensor, torch.Tensor, int]],
        labels_per_batch: int,
        samples_per_label: int,
        seed: int,
    ) -> None:
        self.labels_per_batch = max(1, labels_per_batch)
        self.samples_per_label = max(1, samples_per_label)
        self.seed = seed
        self.epoch = 0
        self.label_to_indices: dict[int, list[int]] = {}
        for idx, (_, _, label_id) in enumerate(samples):
            self.label_to_indices.setdefault(int(label_id), []).append(idx)

        self.labels = sorted(self.label_to_indices)
        if not self.labels:
            raise ValueError("BalancedBatchSampler received no samples.")
        self.labels_per_batch = min(self.labels_per_batch, len(self.labels))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {label_id: indices[:] for label_id, indices in self.label_to_indices.items()}
        for indices in pools.values():
            rng.shuffle(indices)

        while True:
            eligible_labels = [
                label_id for label_id, indices in pools.items() if len(indices) >= self.samples_per_label
            ]
            if len(eligible_labels) < self.labels_per_batch:
                break

            chosen_labels = rng.sample(eligible_labels, self.labels_per_batch)
            batch: list[int] = []
            for label_id in chosen_labels:
                for _ in range(self.samples_per_label):
                    batch.append(pools[label_id].pop())

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        label_chunks = sum(len(indices) // self.samples_per_label for indices in self.label_to_indices.values())
        return label_chunks // self.labels_per_batch


def make_train_loader(
    samples: list[tuple[torch.Tensor, torch.Tensor, int]],
    args: argparse.Namespace,
) -> DataLoader:
    """Build either a balanced contrastive loader or the ordinary shuffled one."""

    dataset = ChessStyleDataset(samples)
    if args.no_balanced_batches:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    label_counts = Counter(int(label_id) for _, _, label_id in samples)
    labels_per_batch = min(args.labels_per_batch, len(label_counts))
    requested_samples_per_label = args.samples_per_label or max(2, args.batch_size // max(labels_per_batch, 1))
    samples_per_label = min(requested_samples_per_label, min(label_counts.values()))
    if samples_per_label < 1:
        raise RuntimeError("Cannot build balanced batches from empty training labels.")

    print(
        "Using balanced training batches: "
        f"labels_per_batch={labels_per_batch} samples_per_label={samples_per_label} "
        f"effective_batch_size={labels_per_batch * samples_per_label}"
    )
    sampler = BalancedBatchSampler(
        samples,
        labels_per_batch=labels_per_batch,
        samples_per_label=samples_per_label,
        seed=args.seed,
    )
    return DataLoader(dataset, batch_sampler=sampler)


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


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Pull same-label game embeddings together and push other labels apart."""

    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    batch_size = labels.size(0)

    self_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid_anchors = positive_mask.any(dim=1)
    if not bool(valid_anchors.any()):
        return embeddings.sum() * 0.0

    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -positive_log_prob[valid_anchors].mean()


def collect_embeddings(
    model: ChessStyleTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every game into normalized metric-space embeddings."""

    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for board, mask, y in loader:
            board = board.to(device)
            mask = mask.to(device)
            embeddings.append(model.encode_metric(board, mask).cpu())
            labels.append(y.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    top_k: int,
    exclude_self: bool = False,
) -> tuple[dict[str, object], list[int]]:
    """Compute nearest-neighbor retrieval quality in cosine space."""

    query_embeddings = F.normalize(query_embeddings.float(), dim=1)
    reference_embeddings = F.normalize(reference_embeddings.float(), dim=1)
    similarities = query_embeddings @ reference_embeddings.T

    if exclude_self:
        if similarities.shape[0] != similarities.shape[1]:
            raise ValueError("exclude_self=True requires query and reference sets with the same length.")
        similarities.fill_diagonal_(float("-inf"))

    sorted_indices = similarities.argsort(dim=1, descending=True)
    sorted_labels = reference_labels[sorted_indices]
    relevant = sorted_labels.eq(query_labels[:, None])

    k = min(top_k, similarities.shape[1])
    topk_correct = relevant[:, :k]
    pred_labels = sorted_labels[:, 0].tolist()

    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    for query_idx in range(relevant.shape[0]):
        relevant_positions = torch.nonzero(relevant[query_idx], as_tuple=False).flatten()
        if relevant_positions.numel() == 0:
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            continue

        ranks = relevant_positions.float() + 1.0
        reciprocal_ranks.append(float(1.0 / ranks[0]))
        hits_at_rank = torch.arange(1, relevant_positions.numel() + 1, dtype=torch.float32)
        average_precisions.append(float((hits_at_rank / ranks).mean()))

    label_top1: dict[str, float] = {}
    top1_correct = relevant[:, 0]
    for label_id, label_name in ID_TO_LABEL.items():
        label_mask = query_labels.eq(label_id)
        if bool(label_mask.any()):
            label_top1[label_name] = float(top1_correct[label_mask].float().mean())

    same_label = query_labels[:, None].eq(reference_labels[None, :])
    if exclude_self:
        same_label.fill_diagonal_(False)
    finite = torch.isfinite(similarities)
    same_values = similarities[same_label & finite]
    different_values = similarities[(~same_label) & finite]
    same_mean = float(same_values.mean()) if same_values.numel() else 0.0
    different_mean = float(different_values.mean()) if different_values.numel() else 0.0

    metrics: dict[str, object] = {
        "top1": float(top1_correct.float().mean()) if top1_correct.numel() else 0.0,
        f"top{k}": float(topk_correct.any(dim=1).float().mean()) if topk_correct.numel() else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "map": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "same_label_cosine_mean": same_mean,
        "different_label_cosine_mean": different_mean,
        "cosine_margin": same_mean - different_mean,
        "per_label_top1": label_top1,
    }
    return metrics, [int(label_id) for label_id in pred_labels]


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


def save_retrieval_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_dir: Path,
    name: str,
) -> None:
    """Save nearest-neighbor label confusion for a retrieval split."""

    labels = list(ID_TO_LABEL)
    label_names = [ID_TO_LABEL[idx] for idx in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    with open(output_dir / f"retrieval_confusion_{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", *label_names])
        for idx, row in zip(labels, matrix):
            writer.writerow([ID_TO_LABEL[idx], *row.tolist()])


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

    if args.labels:
        selected_labels = list(dict.fromkeys(args.labels))
        unknown_labels = sorted(set(selected_labels) - set(LABEL_NAMES))
        if unknown_labels:
            raise ValueError(f"Unknown labels: {unknown_labels}. Available labels: {LABEL_NAMES}")

        label_to_id = {name: idx for idx, name in enumerate(selected_labels)}
        records = [
            replace(record, label_id=label_to_id[record.label_name])
            for record in records
            if record.label_name in label_to_id
        ]
        LABEL_NAMES[:] = selected_labels
        LABEL_TO_ID.clear()
        LABEL_TO_ID.update(label_to_id)
        ID_TO_LABEL.clear()
        ID_TO_LABEL.update({idx: name for name, idx in label_to_id.items()})

    print("Usable game counts:", class_counts(records))

    records = balance_records(records, max_games_per_player=args.max_games_per_player, seed=args.seed)
    print("Balanced game counts:", class_counts(records))

    splits = split_records(records, seed=args.seed)
    print("Split counts:")
    for split_name, split_records_for_name in splits.items():
        print(f"  {split_name}: {class_counts(split_records_for_name)}")

    config = TensorConfig(
        sequence_length=args.sequence_length,
        skip_plies=args.skip_plies,
        target_moves_only=args.target_moves_only,
        stockfish_depth=args.stockfish_depth,
        multipv=args.multipv,
        board_flip=not args.no_board_flip,
    )

    print("Materializing train tensors...")
    train_samples = materialize_samples(splits["train"], config)
    print("Materializing validation tensors...")
    val_samples = materialize_samples(splits["val"], config)
    print("Materializing test tensors...")
    test_samples = materialize_samples(splits["test"], config)

    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("At least one split is empty after tensor materialization.")

    train_loader = make_train_loader(train_samples, args)
    train_eval_loader = DataLoader(ChessStyleDataset(train_samples), batch_size=args.batch_size)
    val_loader = DataLoader(ChessStyleDataset(val_samples), batch_size=args.batch_size)
    test_loader = DataLoader(ChessStyleDataset(test_samples), batch_size=args.batch_size)

    device = get_device()
    print(f"Using device: {device}")
    model = ChessStyleTransformer(
        num_classes=len(ID_TO_LABEL),
        in_channels=INPUT_CHANNELS,
        metric_dim=args.metric_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_retrieval_map = -1.0
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "best_model.pt"
    embedding_checkpoint_path = output_dir / "best_embedding_model.pt"

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
            embeddings = model.encode_game(board, mask)
            logits = model.classify_embedding(embeddings)
            metric_embeddings = model.project_metric(embeddings)
            classification_loss = criterion(logits, y)
            contrastive_loss = supervised_contrastive_loss(metric_embeddings, y, args.contrastive_temperature)
            loss = args.classification_weight * classification_loss + args.contrastive_weight * contrastive_loss
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
        train_embeddings, train_embedding_labels = collect_embeddings(model, train_eval_loader, device)
        val_embeddings, val_embedding_labels = collect_embeddings(model, val_loader, device)
        val_retrieval, _ = retrieval_metrics(
            val_embeddings,
            val_embedding_labels,
            train_embeddings,
            train_embedding_labels,
            top_k=args.retrieval_top_k,
        )

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
            "val_retrieval_top1": float(val_retrieval["top1"]),
            "val_retrieval_map": float(val_retrieval["map"]),
            "val_retrieval_cosine_margin": float(val_retrieval["cosine_margin"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_acc={train_acc * 100:.1f}% val_acc={val_acc * 100:.1f}% "
            f"test_acc={test_acc * 100:.1f}% val_retrieval_top1={float(val_retrieval['top1']) * 100:.1f}%"
        )

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "labels": ID_TO_LABEL,
            "input_channels": INPUT_CHANNELS,
            "embedding_dim": model.embedding_dim,
            "metric_dim": model.metric_dim,
            "args": vars(args),
        }
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint_payload, checkpoint_path)
        if float(val_retrieval["map"]) > best_val_retrieval_map:
            best_val_retrieval_map = float(val_retrieval["map"])
            torch.save(checkpoint_payload, embedding_checkpoint_path)

        with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    print(f"Best-checkpoint test_loss={test_loss:.4f} test_acc={test_acc * 100:.1f}%")

    embedding_checkpoint = torch.load(embedding_checkpoint_path, map_location=device)
    model.load_state_dict(embedding_checkpoint["model_state_dict"])
    train_embeddings, train_embedding_labels = collect_embeddings(model, train_eval_loader, device)
    val_embeddings, val_embedding_labels = collect_embeddings(model, val_loader, device)
    test_embeddings, test_embedding_labels = collect_embeddings(model, test_loader, device)
    train_retrieval, train_retrieval_pred = retrieval_metrics(
        train_embeddings,
        train_embedding_labels,
        train_embeddings,
        train_embedding_labels,
        top_k=args.retrieval_top_k,
        exclude_self=True,
    )
    val_retrieval, val_retrieval_pred = retrieval_metrics(
        val_embeddings,
        val_embedding_labels,
        train_embeddings,
        train_embedding_labels,
        top_k=args.retrieval_top_k,
    )
    test_retrieval, test_retrieval_pred = retrieval_metrics(
        test_embeddings,
        test_embedding_labels,
        train_embeddings,
        train_embedding_labels,
        top_k=args.retrieval_top_k,
    )
    print(
        "Retrieval accuracy | "
        f"train_top1={train_retrieval['top1'] * 100:.1f}% "
        f"val_top1={val_retrieval['top1'] * 100:.1f}% "
        f"test_top1={test_retrieval['top1'] * 100:.1f}%"
    )

    save_classification_report(test_true, test_pred, output_dir)
    save_confusion_matrix(test_true, test_pred, output_dir)
    save_retrieval_confusion_matrix(
        train_embedding_labels.tolist(),
        train_retrieval_pred,
        output_dir,
        name="train_vs_train_excluding_self",
    )
    save_retrieval_confusion_matrix(val_embedding_labels.tolist(), val_retrieval_pred, output_dir, name="val_vs_train")
    save_retrieval_confusion_matrix(test_embedding_labels.tolist(), test_retrieval_pred, output_dir, name="test_vs_train")

    retrieval_metrics_payload = {
        "top_k": args.retrieval_top_k,
        "train_vs_train_excluding_self": train_retrieval,
        "val_vs_train": val_retrieval,
        "test_vs_train": test_retrieval,
    }
    with open(output_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_metrics_payload, f, indent=2)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "input_channels": INPUT_CHANNELS,
                "embedding_dim": model.embedding_dim,
                "metric_dim": model.metric_dim,
                "labels": ID_TO_LABEL,
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
                "best_val_retrieval_map": best_val_retrieval_map,
                "retrieval_metrics": retrieval_metrics_payload,
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a selected-player chess style classifier")
    parser.add_argument("--data-dir", default=str(ROOT / "PGN_data"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "multiclass_proof"))
    parser.add_argument("--labels", nargs="+", help="Train on a selected subset of player labels.")
    parser.add_argument("--max-games-per-player", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--skip-plies", type=int, default=0, help="Skip this many opening plies before recording tensors.")
    parser.add_argument("--target-moves-only", action="store_true", help="Encode only decisions made by the target player.")
    parser.add_argument("--stockfish-depth", type=int, default=0)
    parser.add_argument("--multipv", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--metric-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--labels-per-batch", type=int, default=len(ID_TO_LABEL))
    parser.add_argument("--samples-per-label", type=int, default=0)
    parser.add_argument("--no-balanced-batches", action="store_true")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-board-flip", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
