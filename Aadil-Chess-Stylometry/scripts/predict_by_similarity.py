"""Predict a game's style label by cosine similarity to saved embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from board_dataset import ID_TO_LABEL, INPUT_CHANNELS, TensorConfig, game_to_model_tensors
from model import ChessStyleTransformer


def parse_color(value: str) -> bool:
    """Parse a CLI color value into python-chess's bool color."""

    lowered = value.lower()
    if lowered == "white":
        return chess.WHITE
    if lowered == "black":
        return chess.BLACK
    raise argparse.ArgumentTypeError("target color must be 'white' or 'black'")


def read_game_at(path: Path, game_index: int) -> chess.pgn.Game:
    """Read one game by zero-based index from a PGN file."""

    with open(path, "r", encoding="utf-8", errors="ignore") as pgn_file:
        for idx in range(game_index + 1):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                raise ValueError(f"Could not find game index {game_index} in {path}")
        return game


def checkpoint_labels(checkpoint: dict) -> dict[int, str]:
    """Read label names from a checkpoint with a stable fallback."""

    labels = checkpoint.get("labels", ID_TO_LABEL)
    return {int(key): value for key, value in labels.items()}


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ChessStyleTransformer, dict[int, str]]:
    """Load the trained model used to create the reference embeddings."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_names = checkpoint_labels(checkpoint)
    input_channels = int(checkpoint.get("input_channels", INPUT_CHANNELS))
    metric_dim = int(checkpoint.get("metric_dim", 128))
    model = ChessStyleTransformer(num_classes=len(label_names), in_channels=input_channels, metric_dim=metric_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_names


def load_reference_embeddings(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    """Load exported reference embeddings."""

    payload = torch.load(path, map_location="cpu")
    label_names = {int(key): value for key, value in payload["label_names"].items()}
    return payload["embeddings"].float(), payload["labels"].long(), label_names


def encode_query_game(
    model: ChessStyleTransformer,
    pgn_path: Path,
    game_index: int,
    target_color: bool,
    sequence_length: int,
    skip_plies: int,
    board_flip: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build tensors for a query PGN game and return its embedding."""

    game = read_game_at(pgn_path, game_index)
    config = TensorConfig(sequence_length=sequence_length, skip_plies=skip_plies, board_flip=board_flip)
    tensors = game_to_model_tensors(game, target_color=target_color, config=config)
    if tensors is None:
        raise ValueError(f"Game {game_index} in {pgn_path} has no mainline moves.")

    sequence, padding_mask = tensors
    with torch.no_grad():
        embedding = model.encode_metric(sequence.unsqueeze(0).to(device), padding_mask.unsqueeze(0).to(device))
    return embedding.squeeze(0).cpu()


def label_scores_from_top_k(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    label_names: dict[int, str],
    top_k_per_label: int,
) -> list[tuple[str, float, int]]:
    """Score labels by mean of their top-k cosine similarities."""

    scores: list[tuple[str, float, int]] = []
    for label_id, label_name in sorted(label_names.items()):
        label_sims = similarities[labels == label_id]
        if label_sims.numel() == 0:
            continue

        k = min(top_k_per_label, label_sims.numel())
        score = float(torch.topk(label_sims, k=k).values.mean())
        scores.append((label_name, score, int(label_id)))

    return sorted(scores, key=lambda item: item[1], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a PGN game's label by embedding cosine similarity.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--reference-embeddings", required=True, help="Path to embeddings_train.pt or similar")
    parser.add_argument("--pgn", required=True, help="Query PGN file")
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--target-color", type=parse_color, required=True, help="white or black")
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument("--skip-plies", type=int, default=0, help="Skip this many opening plies before recording tensors.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest games to print")
    parser.add_argument("--top-k-per-label", type=int, default=5, help="Mean top-k similarities per label")
    parser.add_argument("--no-board-flip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_label_names = load_model(Path(args.checkpoint), device)
    reference_embeddings, reference_labels, reference_label_names = load_reference_embeddings(Path(args.reference_embeddings))
    label_names = reference_label_names or checkpoint_label_names

    query_embedding = encode_query_game(
        model,
        Path(args.pgn),
        args.game_index,
        args.target_color,
        args.sequence_length,
        args.skip_plies,
        not args.no_board_flip,
        device,
    )

    reference_norm = F.normalize(reference_embeddings, dim=1)
    query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
    similarities = (reference_norm @ query_norm.T).squeeze(1)

    label_scores = label_scores_from_top_k(similarities, reference_labels, label_names, args.top_k_per_label)
    predicted_label, predicted_score, _ = label_scores[0]
    print(f"Predicted label: {predicted_label} cosine={predicted_score:.4f}")

    print("\nLabel scores:")
    for label_name, score, _ in label_scores:
        print(f"  {label_name:18s} {score:.4f}")

    print("\nTop nearest games:")
    top_k = min(args.top_k, similarities.numel())
    for rank, idx in enumerate(torch.topk(similarities, k=top_k).indices.tolist(), start=1):
        label_id = int(reference_labels[idx])
        print(f"  {rank:02d}. {label_names[label_id]:18s} cosine={float(similarities[idx]):.4f} row={idx}")


if __name__ == "__main__":
    main()
