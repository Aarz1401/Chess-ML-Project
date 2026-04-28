"""Train and evaluate Model 1's hybrid neural architecture.

This script builds one example per game by default:

1. A padded whole-game tensor for the CNN/Transformer branch.
2. A handcrafted feature vector for the style-feature branch.

The fixed short-window setup is still available with --sequence-mode window for
ablation runs.
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from retrieval.run_engine_quality_baseline import (
    feature_dimension,
    features_for_game,
    record_with_stockfish_side,
)
from src.dataset import (
    LABELS,
    GameRecord,
    _balance_records_by_label,
    _open_pgn,
    _split_records_by_label,
    collect_game_records,
    game_to_windows,
    move_to_tensor,
)
from src.experiment_utils import active_labels, remap_record
from src.model import HybridStockfishModel
from src.pipeline import set_seed


class HybridDataset(Dataset):
    """PyTorch wrapper for precomputed board-window + feature-vector samples."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]


def padded_window_for_game(
    game: chess.pgn.Game,
    label: int,
    target_color: bool | None,
    window_length: int,
    skip_moves: int,
    board_flip: bool,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Build one padded whole-game sequence.

    The output is always (window_length, 34, 8, 8). If the game has fewer real
    moves, the rest is zero padding and the mask marks those positions True.
    """

    board = game.board()
    move_tensors: list[torch.Tensor] = []
    for move in game.mainline_moves():
        mover = board.turn
        before = board.copy(stack=False)
        board.push(move)
        if target_color is None or mover == target_color:
            move_tensors.append(move_to_tensor(before, board, perspective_color=mover, board_flip=board_flip))

    move_tensors = move_tensors[skip_moves : skip_moves + window_length]
    if not move_tensors:
        return None

    valid_moves = len(move_tensors)
    board_window = torch.zeros((window_length, 34, 8, 8), dtype=torch.float32)
    board_window[:valid_moves] = torch.stack(move_tensors, dim=0)

    # PyTorch Transformer convention: True marks tokens to ignore.
    padding_mask = torch.ones((window_length,), dtype=torch.bool)
    padding_mask[:valid_moves] = False
    return board_window, padding_mask, label


def materialize_hybrid_samples(
    records: list[GameRecord],
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    window_size: int,
    window_length: int,
    skip_moves: int,
    max_analyzed_moves: int,
    multipv: int,
    stockfish_side: str,
    feature_set: str,
    board_flip: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Build one aligned board-window + style-feature sample per game."""

    # Group records by file so each PGN is scanned once. This matters because a
    # single PGN file can contain many games.
    by_path: dict[str, dict[int, GameRecord]] = {}
    for record in records:
        by_path.setdefault(record.pgn_path, {})[record.game_index] = record

    cache = {}
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    done = 0
    total = len(records)

    for pgn_path, selected in by_path.items():
        with _open_pgn(pgn_path) as pgn:
            game_index = 0
            while selected:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                record = selected.pop(game_index, None)
                if record is not None:
                    record = record_with_stockfish_side(record, stockfish_side)

                    # Default board branch input: one padded whole-game
                    # trajectory.
                    if window_length > 0:
                        padded = padded_window_for_game(
                            game,
                            label=record.label,
                            target_color=record.target_color,
                            window_length=window_length,
                            skip_moves=skip_moves,
                            board_flip=board_flip,
                        )
                        windows = [padded] if padded is not None else []
                    else:
                        # Ablation path: one short window per game.
                        short_windows = game_to_windows(
                            game,
                            label=record.label,
                            window_size=window_size,
                            stride=window_size,
                            skip_moves=skip_moves,
                            target_color=record.target_color,
                            max_windows=1,
                            board_flip=board_flip,
                        )
                        windows = [
                            (
                                board_window,
                                torch.zeros((board_window.shape[0],), dtype=torch.bool),
                                label,
                            )
                            for board_window, label in short_windows
                        ]
                    if windows:
                        board_window, padding_mask, label = windows[0]

                        # Feature branch input from the same game.
                        style_features = features_for_game(
                            game,
                            record,
                            engine,
                            limit,
                            max_moves=max_analyzed_moves,
                            skip_moves=skip_moves,
                            multipv=multipv,
                            cache=cache,
                            feature_set=feature_set,
                        )
                        samples.append(
                            (
                                board_window.float(),
                                padding_mask,
                                torch.from_numpy(style_features).float(),
                                int(label),
                            )
                        )
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"  materialized {done}/{total} games", flush=True)
                game_index += 1

    random.shuffle(samples)
    return samples


def evaluate(model, loader, criterion, device):
    """Evaluate loss and accuracy without updating model weights."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for board, mask, sf, y in loader:
            board = board.to(device)
            mask = mask.to(device)
            sf = sf.to(device)
            y = y.to(device)
            logits = model(board, sf, padding_mask=mask)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def make_hybrid_model(num_classes: int, scale: str, feature_dim: int) -> HybridStockfishModel:
    """Choose an architecture size that fits the available compute.

    tiny/small/paper are scaled versions of the same ModelC idea:
    CNN move encoder -> Transformer sequence encoder -> fused classifier.
    """

    if scale == "tiny":
        return HybridStockfishModel(
            num_classes=num_classes,
            stockfish_feature_dim=feature_dim,
            stockfish_hidden_dim=32,
            board_embedding_dim=128,
            fusion_hidden_dim=64,
            board_encoder_kwargs={
                "n_layers": 1,
                "n_heads": 2,
                "conv_channels": 16,
                "residual_blocks": 1,
                "move_dim": 64,
                "transformer_dim": 128,
                "feedforward_dim": 256,
            },
        )
    if scale == "small":
        return HybridStockfishModel(
            num_classes=num_classes,
            stockfish_feature_dim=feature_dim,
            stockfish_hidden_dim=64,
            board_embedding_dim=256,
            fusion_hidden_dim=128,
            board_encoder_kwargs={
                "n_layers": 2,
                "n_heads": 4,
                "conv_channels": 32,
                "residual_blocks": 2,
                "move_dim": 128,
                "transformer_dim": 256,
                "feedforward_dim": 512,
            },
        )
    # "paper" uses ModelC's default dimensions, which are much heavier and are
    # mainly included to show the intended full architecture.
    return HybridStockfishModel(num_classes=num_classes, stockfish_feature_dim=feature_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test the hybrid board + style-feature neural model")
    parser.add_argument("--data-dir", default="data/controlled_tal_engine_raw")
    parser.add_argument("--max-games-per-label", type=int, default=40)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--stockfish-side", choices=["both", "white", "black", "alternate"], default="alternate")
    parser.add_argument(
        "--sequence-mode",
        choices=["padded", "window"],
        default="padded",
        help="padded uses one fixed-length game sequence; window uses the short-window ablation.",
    )
    parser.add_argument("--window-length", type=int, default=80, help="Length for padded whole-game sequences.")
    parser.add_argument("--window-size", type=int, default=15, help="Short window size used only when --sequence-mode window.")
    parser.add_argument("--skip-moves", type=int, default=0)
    parser.add_argument("--max-analyzed-moves", type=int, default=8)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument(
        "--feature-set",
        choices=["engine_agreement", "style_trajectory", "combined"],
        default="style_trajectory",
        help="Use style_trajectory to avoid direct top-move identity/rank features.",
    )
    parser.add_argument("--engine-path", default="stockfish")
    parser.add_argument("--engine-depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-scale", choices=["tiny", "small", "paper"], default="tiny")
    parser.add_argument(
        "--no-board-flip",
        action="store_true",
        help="Keep absolute board coordinates instead of rotating Black moves to the mover's perspective.",
    )
    parser.add_argument("--results-path", default="outputs/hybrid_smoke_history.csv")
    args = parser.parse_args()

    set_seed()
    random.seed(0)
    np.random.seed(0)

    if args.include_label:
        excluded = set(active_labels(LABELS, set())) - set(args.include_label)
        excluded.update(args.exclude_label)
    else:
        excluded = set(args.exclude_label)
    labels = active_labels(LABELS, excluded)

    records = collect_game_records(
        data_dir=args.data_dir,
        window_size=1 if args.sequence_mode == "padded" else args.window_size,
        skip_moves=args.skip_moves,
        exclude_labels=excluded,
    )
    records = [remap_record(record, labels) for record in records]

    # Keep class counts equal before splitting, so the reported accuracy is not
    # just a majority-class effect.
    records = _balance_records_by_label(records, max_games_per_label=args.max_games_per_label)
    splits = _split_records_by_label(records)

    print("Game counts:")
    for split_name, split_records in splits.items():
        print(f"  {split_name}: {dict(sorted(Counter(r.label_name for r in split_records).items()))}")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    limit = chess.engine.Limit(depth=args.engine_depth)
    active_window_length = args.window_length if args.sequence_mode == "padded" else 0
    try:
        print("Materializing train samples...")
        train_samples = materialize_hybrid_samples(
            splits["train"],
            engine,
            limit,
            args.window_size,
            active_window_length,
            args.skip_moves,
            args.max_analyzed_moves,
            args.multipv,
            args.stockfish_side,
            args.feature_set,
            board_flip=not args.no_board_flip,
        )
        print("Materializing validation samples...")
        val_samples = materialize_hybrid_samples(
            splits["val"],
            engine,
            limit,
            args.window_size,
            active_window_length,
            args.skip_moves,
            args.max_analyzed_moves,
            args.multipv,
            args.stockfish_side,
            args.feature_set,
            board_flip=not args.no_board_flip,
        )
        print("Materializing test samples...")
        test_samples = materialize_hybrid_samples(
            splits["test"],
            engine,
            limit,
            args.window_size,
            active_window_length,
            args.skip_moves,
            args.max_analyzed_moves,
            args.multipv,
            args.stockfish_side,
            args.feature_set,
            board_flip=not args.no_board_flip,
        )
    finally:
        engine.quit()

    print(f"Sample counts: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Not enough samples to train/evaluate the hybrid model.")

    train_loader = DataLoader(HybridDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(HybridDataset(val_samples), batch_size=args.batch_size)
    test_loader = DataLoader(HybridDataset(test_samples), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_hybrid_model(
        num_classes=len(labels),
        scale=args.model_scale,
        feature_dim=feature_dimension(args.feature_set),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for board, mask, sf, y in train_loader:
            board = board.to(device)
            mask = mask.to(device)
            sf = sf.to(device)
            y = y.to(device)
            # Standard PyTorch training step for a fused two-input model.
            optimizer.zero_grad()
            logits = model(board, sf, padding_mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "sequence_mode": args.sequence_mode,
            "window_length": args.window_length if args.sequence_mode == "padded" else "",
            "window_size": args.window_size if args.sequence_mode == "window" else "",
            "feature_set": args.feature_set,
            "feature_dim": feature_dimension(args.feature_set),
            "board_orientation": "absolute_no_flip" if args.no_board_flip else "mover_relative_flipped",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% | "
            f"val_acc={val_acc*100:.1f}% | test_acc={test_acc*100:.1f}%"
        )

        if args.results_path:
            # Save after every epoch so long experimental runs still leave a
            # usable CSV if they are interrupted before the final epoch.
            Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.results_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

    if args.results_path:
        print(f"Saved {args.results_path}")


if __name__ == "__main__":
    main()
