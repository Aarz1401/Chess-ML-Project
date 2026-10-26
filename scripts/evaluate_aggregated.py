"""Player-level evaluation and figures for a trained style encoder.

Loads a checkpoint plus a tensor cache, embeds the reference and query splits,
and produces the headline stylometry results:

  aggregation_curve.png           accuracy vs number of aggregated games
  embedding_map.png               2D projection with player centroids
  player_similarity_heatmap.png   centroid-to-centroid cosine similarity
  aggregated_confusion.png        confusion at the largest aggregation k
  aggregated_metrics.json         all numbers behind the figures
  retrieval_metrics.json          single-game retrieval metrics

Example:

    python3 scripts/evaluate_aggregated.py \
      --checkpoint results/full8/best_embedding_model.pt \
      --cache-dir results/tensor_cache_full \
      --output-dir results/full8/aggregated_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

from evaluation import (
    aggregated_player_accuracy,
    collect_embeddings,
    player_centroids,
    retrieval_metrics,
)
from model import build_model_from_checkpoint
from tensor_cache import CachedChessDataset, load_manifest

# Categorical palette (validated, fixed slot order — one slot per player id).
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Sequential blue ramp, light -> dark, for magnitude encodings.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)


def apply_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
        }
    )


def series_color(label_id: int) -> str:
    return SERIES[label_id % len(SERIES)]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_aggregation_curve(aggregated: dict, label_names: list[str], baseline: float, path: Path) -> None:
    """Accuracy as a function of how many games are aggregated per player."""

    ks = [int(k) for k in aggregated["ks"]]
    fig, ax = plt.subplots(figsize=(8, 5))

    for label_id, name in enumerate(label_names):
        accs = [aggregated["per_k"][str(k)]["per_label"][name] * 100 for k in ks]
        ax.plot(ks, accs, color=series_color(label_id), linewidth=1.3, alpha=0.8,
                marker="o", markersize=4, label=name)

    overall = [aggregated["per_k"][str(k)]["overall"] * 100 for k in ks]
    ax.plot(ks, overall, color=INK, linewidth=2.4, marker="o", markersize=6, label="All players")
    for k, acc in zip(ks, overall):
        ax.annotate(f"{acc:.0f}%", (k, acc), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=INK)

    ax.axhline(baseline * 100, color=MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
    ax.annotate("random baseline", (ks[-1], baseline * 100), textcoords="offset points",
                xytext=(0, 6), ha="right", fontsize=8, color=MUTED)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(k) for k in ks]))
    ax.set_xlabel("Games aggregated per player (k)")
    ax.set_ylabel("Player identification accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Player identification vs games aggregated")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_map(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    path: Path,
    max_points: int,
    seed: int,
) -> str:
    """2D projection of single-game embeddings with projected centroids."""

    rng = np.random.default_rng(seed)
    points = embeddings.numpy()
    label_array = labels.numpy()
    if points.shape[0] > max_points:
        chosen = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[chosen]
        label_array = label_array[chosen]

    # Neighborhood projections are not meaningful for tiny smoke-test splits.
    if points.shape[0] < 4:
        from sklearn.decomposition import PCA

        method = "PCA"
        if points.shape[0] < 2:
            projected = np.zeros((points.shape[0], 2))
        else:
            projected = PCA(n_components=2).fit_transform(points)
    else:
        method = "UMAP"
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
            projected = reducer.fit_transform(points)
        except ImportError:
            from sklearn.manifold import TSNE

            method = "t-SNE"
            perplexity = min(30.0, max(1.0, (points.shape[0] - 1) / 3))
            projected = TSNE(
                n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed
            ).fit_transform(points)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.grid(False)
    for label_id, name in enumerate(label_names):
        mask = label_array == label_id
        if not mask.any():
            continue
        ax.scatter(projected[mask, 0], projected[mask, 1], s=14, alpha=0.55,
                   color=series_color(label_id), linewidths=0, label=name)

    # Centroids of the projected clouds, ringed with the surface color so they
    # stay readable on top of overlapping points.
    for label_id, name in enumerate(label_names):
        mask = label_array == label_id
        if not mask.any():
            continue
        cx, cy = projected[mask, 0].mean(), projected[mask, 1].mean()
        ax.scatter([cx], [cy], s=170, marker="X", color=series_color(label_id),
                   edgecolors=SURFACE, linewidths=2, zorder=5)

    ax.set_title(f"Game embeddings ({method}), one point per game")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, markerscale=1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return method


def plot_similarity_heatmap(centroids: torch.Tensor, label_names: list[str], path: Path) -> None:
    """Cosine similarity between player centroids."""

    matrix = (centroids @ centroids.T).numpy()
    n = len(label_names)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.grid(False)
    off_diagonal = matrix[~np.eye(n, dtype=bool)]
    vmin, vmax = float(off_diagonal.min()), 1.0
    im = ax.imshow(matrix, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax)

    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            frac = (value - vmin) / max(vmax - vmin, 1e-9)
            color = SURFACE if frac > 0.6 else INK
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_title("Player centroid cosine similarity")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(BASELINE)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregated_confusion(aggregated: dict, path: Path) -> None:
    """Row-normalized confusion matrix at the largest aggregation k."""

    matrix = np.array(aggregated["confusion_at_max_k"], dtype=np.float64)
    label_names = aggregated["confusion_labels"]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.clip(row_sums, 1, None)
    n = len(label_names)
    max_k = max(int(k) for k in aggregated["ks"])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.grid(False)
    im = ax.imshow(normalized, cmap=SEQ_CMAP, vmin=0.0, vmax=1.0)

    for row in range(n):
        for col in range(n):
            value = normalized[row, col]
            if value < 0.005:
                continue
            color = SURFACE if value > 0.6 else INK
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted player")
    ax.set_ylabel("True player")
    ax.set_title(f"Aggregated identification confusion (k={max_k} games)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(BASELINE)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Player-level evaluation of a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_embedding_model.pt")
    parser.add_argument("--cache-dir", required=True, help="Tensor cache from scripts/precompute_tensors.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 25])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=1500, help="Point cap for the 2D projection plot.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_chart_style()
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    manifest = load_manifest(cache_dir)
    label_names = list(manifest["labels"])
    id_to_label = {idx: name for idx, name in enumerate(label_names)}

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model = build_model_from_checkpoint(checkpoint, num_classes=len(label_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint {args.checkpoint} on {device}")

    loaders = {}
    for split in {args.reference_split, args.query_split}:
        dataset = CachedChessDataset(cache_dir, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(f"{split}: {len(dataset)} games")

    print("Encoding embeddings...")
    reference_embeddings, reference_labels = collect_embeddings(model, loaders[args.reference_split], device)
    if args.query_split == args.reference_split:
        query_embeddings, query_labels = reference_embeddings, reference_labels
    else:
        query_embeddings, query_labels = collect_embeddings(model, loaders[args.query_split], device)

    retrieval, _ = retrieval_metrics(
        query_embeddings,
        query_labels,
        reference_embeddings,
        reference_labels,
        top_k=args.retrieval_top_k,
        id_to_label=id_to_label,
        exclude_self=args.query_split == args.reference_split,
    )
    print(
        f"Single-game retrieval: top1={retrieval['top1'] * 100:.1f}% "
        f"map={retrieval['map'] * 100:.1f}% margin={retrieval['cosine_margin']:.3f}"
    )
    with open(output_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2)

    centroids = player_centroids(reference_embeddings, reference_labels, num_classes=len(label_names))
    aggregated = aggregated_player_accuracy(
        query_embeddings,
        query_labels,
        centroids,
        ks=sorted(args.ks),
        trials=args.trials,
        seed=args.seed,
        id_to_label=id_to_label,
    )
    for k in sorted(args.ks):
        print(f"Aggregated player ID accuracy (k={k}): {aggregated['per_k'][str(k)]['overall'] * 100:.1f}%")
    with open(output_dir / "aggregated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    baseline = 1.0 / len(label_names)
    plot_aggregation_curve(aggregated, label_names, baseline, output_dir / "aggregation_curve.png")
    method = plot_embedding_map(
        query_embeddings, query_labels, label_names, output_dir / "embedding_map.png",
        max_points=args.max_points, seed=args.seed,
    )
    plot_similarity_heatmap(centroids, label_names, output_dir / "player_similarity_heatmap.png")
    plot_aggregated_confusion(aggregated, output_dir / "aggregated_confusion.png")

    print(f"Saved figures to {output_dir} (projection method: {method})")


if __name__ == "__main__":
    main()
