"""Train the chess style encoder with contrastive + classification objectives.

Preferred workflow (fast, HPC-friendly): precompute tensors once, then train
from the cache:

    python3 scripts/precompute_tensors.py --cache-dir results/tensor_cache_full
    python3 scripts/train_multiclass.py --cache-dir results/tensor_cache_full

Without --cache-dir the script falls back to building tensors in-process the
way the original proof of concept did.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

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
from evaluation import (
    aggregated_player_accuracy,
    collect_embeddings,
    player_centroids,
    retrieval_metrics,
)
from model import ChessStyleTransformer
from tensor_cache import CachedChessDataset, load_manifest, validate_cache


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
        labels: list[int],
        labels_per_batch: int,
        samples_per_label: int,
        seed: int,
    ) -> None:
        self.labels_per_batch = max(1, labels_per_batch)
        self.samples_per_label = max(1, samples_per_label)
        self.seed = seed
        self.epoch = 0
        self.label_to_indices: dict[int, list[int]] = {}
        for idx, label_id in enumerate(labels):
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


def dataset_labels(dataset: Dataset) -> list[int]:
    """Return per-sample integer labels without loading tensors."""

    if isinstance(dataset, CachedChessDataset):
        return [int(label_id) for label_id in dataset.labels]
    return [int(label_id) for _, _, label_id in dataset]


def make_train_loader(dataset: Dataset, labels: list[int], args: argparse.Namespace) -> DataLoader:
    """Build either a balanced contrastive loader or the ordinary shuffled one."""

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if args.no_balanced_batches:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    label_counts = Counter(labels)
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
        labels,
        labels_per_batch=labels_per_batch,
        samples_per_label=samples_per_label,
        seed=args.seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)


def make_eval_loader(dataset: Dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def capped_reference_dataset(dataset: Dataset, labels: list[int], cap_per_label: int, seed: int) -> Dataset:
    """Subsample the train set so per-epoch retrieval eval stays cheap."""

    if cap_per_label <= 0:
        return dataset

    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for idx, label_id in enumerate(labels):
        by_label.setdefault(label_id, []).append(idx)

    chosen: list[int] = []
    for indices in by_label.values():
        indices = indices[:]
        rng.shuffle(indices)
        chosen.extend(indices[:cap_per_label])

    if len(chosen) >= len(labels):
        return dataset
    return Subset(dataset, sorted(chosen))


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    """Return loss, accuracy, true labels, and predicted labels."""

    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for board, mask, y in loader:
            board = board.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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

    embeddings = F.normalize(embeddings.float(), dim=1)
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


def cosine_warmup_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_frac: float):
    """Linear warmup followed by cosine decay to min_frac of the peak LR."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def set_label_globals(selected_labels: list[str]) -> None:
    """Point the module-level label tables at the labels used by this run."""

    label_to_id = {name: idx for idx, name in enumerate(selected_labels)}
    LABEL_NAMES[:] = selected_labels
    LABEL_TO_ID.clear()
    LABEL_TO_ID.update(label_to_id)
    ID_TO_LABEL.clear()
    ID_TO_LABEL.update({idx: name for name, idx in label_to_id.items()})


def load_cached_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    """Load precomputed splits and align label globals with the manifest."""

    cache_dir = Path(args.cache_dir)
    manifest = validate_cache(cache_dir, Path(args.data_dir))
    for key in ("sequence_length", "skip_plies", "target_moves_only"):
        setattr(args, key, manifest[key])
    args.no_board_flip = not manifest["board_flip"]
    set_label_globals(list(manifest["labels"]))
    print(f"Loaded tensor cache {cache_dir} with labels {manifest['labels']}")
    print(f"Cache counts: {manifest['counts']}")

    return (
        CachedChessDataset(cache_dir, "train"),
        CachedChessDataset(cache_dir, "val"),
        CachedChessDataset(cache_dir, "test"),
    )


def build_datasets_in_process(args: argparse.Namespace) -> tuple[Dataset, Dataset, Dataset]:
    """Original slow path: parse PGNs and build tensors inside this run."""

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
        set_label_globals(selected_labels)

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

    return ChessStyleDataset(train_samples), ChessStyleDataset(val_samples), ChessStyleDataset(test_samples)


def train(args: argparse.Namespace) -> None:
    """Run one full experiment and save all outputs under args.output_dir."""

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        train_dataset, val_dataset, test_dataset = load_cached_datasets(args)
    else:
        args.num_workers = 0
        train_dataset, val_dataset, test_dataset = build_datasets_in_process(args)

    train_labels = dataset_labels(train_dataset)
    train_loader = make_train_loader(train_dataset, train_labels, args)
    train_eval_loader = make_eval_loader(train_dataset, args)
    val_loader = make_eval_loader(val_dataset, args)
    test_loader = make_eval_loader(test_dataset, args)
    reference_dataset = capped_reference_dataset(
        train_dataset, train_labels, args.eval_reference_cap, args.seed
    )
    reference_loader = make_eval_loader(reference_dataset, args)

    device = get_device()
    print(f"Using device: {device}")

    model = ChessStyleTransformer(
        num_classes=len(ID_TO_LABEL),
        in_channels=INPUT_CHANNELS,
        conv_channels=args.conv_channels,
        n_blocks=args.n_blocks,
        move_dim=args.move_dim,
        transformer_dim=args.transformer_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feedforward_dim=args.feedforward_dim,
        metric_dim=args.metric_dim,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = cosine_warmup_schedule(
        optimizer,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        total_steps=args.epochs * steps_per_epoch,
        min_frac=args.min_lr_frac,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    if amp_enabled:
        print(f"Mixed precision enabled ({amp_dtype})")

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
            board = board.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                embeddings = model.encode_game(board, mask)
                logits = model.classify_embedding(embeddings)
                metric_embeddings = model.project_metric(embeddings)
                classification_loss = criterion(logits, y)
                contrastive_loss = supervised_contrastive_loss(
                    metric_embeddings, y, args.contrastive_temperature
                )
                loss = (
                    args.classification_weight * classification_loss
                    + args.contrastive_weight * contrastive_loss
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * y.size(0)
            train_total += y.size(0)
            train_true.extend(y.cpu().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = accuracy_score(train_true, train_pred)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        reference_embeddings, reference_labels = collect_embeddings(model, reference_loader, device)
        val_embeddings, val_embedding_labels = collect_embeddings(model, val_loader, device)
        val_retrieval, _ = retrieval_metrics(
            val_embeddings,
            val_embedding_labels,
            reference_embeddings,
            reference_labels,
            top_k=args.retrieval_top_k,
            id_to_label=ID_TO_LABEL,
        )

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_retrieval_top1": float(val_retrieval["top1"]),
            "val_retrieval_map": float(val_retrieval["map"]),
            "val_retrieval_cosine_margin": float(val_retrieval["cosine_margin"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={row['lr']:.2e} train_acc={train_acc * 100:.1f}% val_acc={val_acc * 100:.1f}% "
            f"val_retrieval_top1={float(val_retrieval['top1']) * 100:.1f}% "
            f"val_map={float(val_retrieval['map']) * 100:.1f}%"
        )

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "labels": dict(ID_TO_LABEL),
            "input_channels": INPUT_CHANNELS,
            "embedding_dim": model.embedding_dim,
            "metric_dim": model.metric_dim,
            "model_config": model.config(),
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

    # Final evaluation. Classification numbers come from the best-accuracy
    # checkpoint; retrieval and aggregation numbers from the best-MAP one.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    test_macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    print(f"Best-checkpoint test_loss={test_loss:.4f} test_acc={test_acc * 100:.1f}% macro_f1={test_macro_f1:.3f}")

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
        id_to_label=ID_TO_LABEL,
        exclude_self=True,
    )
    val_retrieval, val_retrieval_pred = retrieval_metrics(
        val_embeddings,
        val_embedding_labels,
        train_embeddings,
        train_embedding_labels,
        top_k=args.retrieval_top_k,
        id_to_label=ID_TO_LABEL,
    )
    test_retrieval, test_retrieval_pred = retrieval_metrics(
        test_embeddings,
        test_embedding_labels,
        train_embeddings,
        train_embedding_labels,
        top_k=args.retrieval_top_k,
        id_to_label=ID_TO_LABEL,
    )
    print(
        "Retrieval accuracy | "
        f"train_top1={train_retrieval['top1'] * 100:.1f}% "
        f"val_top1={val_retrieval['top1'] * 100:.1f}% "
        f"test_top1={test_retrieval['top1'] * 100:.1f}%"
    )

    # Player-level identification from aggregated games: the headline metric.
    centroids = player_centroids(train_embeddings, train_embedding_labels, num_classes=len(ID_TO_LABEL))
    aggregated = aggregated_player_accuracy(
        test_embeddings,
        test_embedding_labels,
        centroids,
        ks=args.agg_ks,
        trials=args.agg_trials,
        seed=args.seed,
        id_to_label=ID_TO_LABEL,
    )
    for k in args.agg_ks:
        overall = aggregated["per_k"][str(k)]["overall"]
        print(f"Aggregated player ID accuracy (k={k} games): {overall * 100:.1f}%")
    with open(output_dir / "aggregated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

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
                "model_config": model.config(),
                "param_count": param_count,
                "labels": dict(ID_TO_LABEL),
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
                "final_test_macro_f1": test_macro_f1,
                "best_val_retrieval_map": best_val_retrieval_map,
                "retrieval_metrics": retrieval_metrics_payload,
                "aggregated_metrics": aggregated,
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a selected-player chess style classifier")
    parser.add_argument("--cache-dir", help="Tensor cache from scripts/precompute_tensors.py (preferred).")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "pgn"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "multiclass_proof"))
    parser.add_argument("--labels", nargs="+", help="Train on a selected subset of player labels (no-cache path).")
    parser.add_argument("--max-games-per-player", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--skip-plies", type=int, default=12, help="Skip this many opening plies before recording tensors.")
    parser.add_argument("--target-moves-only", action="store_true", help="Encode only decisions made by the target player.")
    parser.add_argument("--no-board-flip", action="store_true")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size (and train batch without balancing).")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.02, help="Final LR as a fraction of peak.")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on GPU.")
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--move-dim", type=int, default=256)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--metric-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["attention", "mean"], default="attention")

    parser.add_argument("--classification-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--labels-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-label", type=int, default=32)
    parser.add_argument("--no-balanced-batches", action="store_true")

    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument(
        "--eval-reference-cap",
        type=int,
        default=250,
        help="Per-label cap on train reference embeddings for per-epoch retrieval eval (0 = full train set).",
    )
    parser.add_argument("--agg-ks", type=int, nargs="+", default=[1, 3, 5, 10, 25])
    parser.add_argument("--agg-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
