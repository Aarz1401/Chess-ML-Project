"""Original board-only neural training pipeline.

This file trains ModelA/ModelC on PGN-derived tensor windows. It is useful for
explaining Model 1's first attempt: direct neural player/style classification
from board-state sequences, before adding the style_trajectory feature branch.
"""

import os
import random
import time
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ChessDataset, LABELS, build_dataset, pgn_to_windows
from src.model import ModelA, ModelC

DATA_DIR = "data/raw"

# Defaults chosen for the original neural experiment. The final notebook uses
# saved histories from these runs rather than rerunning them every time.
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
WINDOW_SIZE = 15
STRIDE = 5
SKIP_MOVES = 10
SEED = 42
MODEL_CHOICE = "c"
SAVE_PATH = "best_model.pt"


def set_seed(seed: int = SEED) -> None:
    """Make Python, NumPy, and PyTorch randomness repeatable."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use CUDA if available; otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compute average loss and accuracy for one dataloader."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def make_model(model_choice: str, num_classes: int) -> nn.Module:
    """Factory for switching between the baseline and final model."""

    choice = model_choice.lower()
    if choice == "a":
        return ModelA(num_classes=num_classes)
    if choice == "c":
        return ModelC(num_classes=num_classes)
    raise ValueError("MODEL_CHOICE must be 'a' or 'c'")


def _active_labels(exclude_labels: set[str] | None = None) -> dict[str, int]:
    """Build a contiguous label mapping after optional label exclusion."""

    exclude_labels = exclude_labels or set()
    names = [name for name in LABELS if name not in exclude_labels]
    return {name: idx for idx, name in enumerate(names)}


def _remap_samples(
    samples: List[Tuple[torch.Tensor, int]],
    active_labels: dict[str, int],
) -> List[Tuple[torch.Tensor, int]]:
    """Convert original dataset label ids to active contiguous label ids."""

    original_to_active = {LABELS[name]: idx for name, idx in active_labels.items()}
    return [(x, original_to_active[y]) for x, y in samples]


def train_model(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    skip_moves: int = SKIP_MOVES,
    model_choice: str = MODEL_CHOICE,
    save_path: str = SAVE_PATH,
    balance_games: bool = True,
    balance_windows: bool = True,
    max_games_per_label: int | None = None,
    label_smoothing: float = LABEL_SMOOTHING,
    early_stopping_patience: int | None = 5,
    exclude_labels: set[str] | None = None,
) -> tuple[nn.Module, DataLoader, dict, torch.device]:
    """Build data, train the selected model, and save the best checkpoint."""

    set_seed(SEED)

    # This call does the expensive PGN parsing, game balancing, and window
    # extraction before wrapping tensors in PyTorch DataLoaders.
    train_samples, val_samples, test_samples = build_dataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        skip_moves=skip_moves,
        balance_games=balance_games,
        balance_windows=balance_windows,
        max_games_per_label=max_games_per_label,
        exclude_labels=exclude_labels,
    )

    active_labels = _active_labels(exclude_labels)
    train_samples = _remap_samples(train_samples, active_labels)
    val_samples = _remap_samples(val_samples, active_labels)
    test_samples = _remap_samples(test_samples, active_labels)

    print(f"\nWindow counts: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    train_loader = DataLoader(ChessDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ChessDataset(val_samples), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ChessDataset(test_samples), batch_size=batch_size, shuffle=False)

    model = make_model(model_choice, len(active_labels))
    model.style_labels = active_labels
    device = get_device()
    model = model.to(device)

    # AdamW is a standard choice for Transformer/CNN models; label smoothing
    # makes the classifier less overconfident on this small noisy dataset.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Keep a lightweight history for plotting/export in the notebook.
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    epochs_without_improvement = 0

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Standard supervised classification step.
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_count += x.size(0)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            # Store the model choice and window settings with the weights so
            # inference can rebuild the same architecture/preprocessing.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_choice": model_choice.lower(),
                    "window_size": window_size,
                    "stride": stride,
                    "skip_moves": skip_moves,
                    "labels": active_labels,
                },
                save_path,
            )
            print(f"Saved new best model to {save_path} (val_acc={val_acc * 100:.2f}%)")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch} epochs without validation improvement.")
                break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f}s. Best val accuracy: {best_val_acc * 100:.2f}%")
    if os.path.exists(save_path):
        print(f"Best checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best checkpoint weights for evaluation.")

    return model, test_loader, history, device


def test_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run the trained model on test data and print common confusions."""

    model.eval()
    labels = getattr(model, "style_labels", LABELS)
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            # Highest logit is the predicted player/source class.
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    if not y_true:
        print("No test samples available.")
        return [], []

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    print(f"Test accuracy: {acc * 100:.2f}%")

    inv = {v: k for k, v in labels.items()}
    print("Per-class accuracy:")
    for label_id in sorted(inv):
        total = sum(1 for y in y_true if y == label_id)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == label_id and p == label_id)
        class_acc = correct / total if total else 0.0
        print(f"  {inv[label_id]:12s}: {class_acc * 100:5.1f}% ({correct}/{total})")

    confusions = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            confusions[(t, p)] += 1

    print("Top confusions:")
    for (t, p), c in confusions.most_common(10):
        print(f"  true={inv[t]:12s} pred={inv[p]:12s} count={c}")

    return y_true, y_pred


def export_history_csv(history: dict, out_path: str = "training_history.csv") -> None:
    """Save training curves in a simple CSV format for later plotting."""

    rows = ["epoch,train_loss,val_loss,val_acc"]
    for i in range(len(history["train_loss"])):
        rows.append(
            f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},{history['val_acc'][i]:.6f}"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"Saved: {out_path}")


def infer_style_distribution(pgn_path: str, save_path: str = SAVE_PATH, data_dir: str = DATA_DIR):
    """Load a checkpoint and return the averaged style distribution for a PGN."""

    checkpoint = torch.load(save_path, map_location="cpu")
    choice = checkpoint.get("model_choice", MODEL_CHOICE).lower()
    window_size = checkpoint.get("window_size", WINDOW_SIZE)
    stride = checkpoint.get("stride", STRIDE)
    skip_moves = checkpoint.get("skip_moves", SKIP_MOVES)

    labels = checkpoint.get("labels", LABELS)
    model = make_model(choice, len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.style_labels = labels

    windows = pgn_to_windows(
        pgn_path=pgn_path,
        label=-1,
        window_size=window_size,
        stride=stride,
        skip_moves=skip_moves,
    )
    if not windows:
        raise ValueError("No windows extracted from PGN. Use a longer game.")

    # Predict every window independently, then average probabilities to get one
    # distribution for the whole PGN file.
    x = torch.stack([w for w, _ in windows], dim=0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        mean_probs = probs.mean(dim=0)

    idx_to_name = {v: k for k, v in labels.items()}
    ranking = sorted(
        [(idx_to_name[i], float(mean_probs[i])) for i in range(len(idx_to_name))],
        key=lambda item: item[1],
        reverse=True,
    )

    print("Style fingerprint:")
    for name, p in ranking:
        print(f"{name.capitalize():12s} {p * 100:5.1f}%")

    return ranking


def run_full_pipeline(inference_pgn: str = ""):
    """Convenience wrapper for notebook experiments."""

    model, test_loader, history, device = train_model()
    test_predictions(model, test_loader, device)
    export_history_csv(history)

    if inference_pgn:
        infer_style_distribution(inference_pgn)

    return model, history
