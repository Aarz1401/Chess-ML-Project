"""Visualize saved game embeddings with 2D projections."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_embedding_files(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    """Load and concatenate one or more exported embedding files."""

    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    label_names: dict[int, str] | None = None

    for path in paths:
        payload = torch.load(path, map_location="cpu")
        embeddings.append(payload["embeddings"].float())
        labels.append(payload["labels"].long())
        if label_names is None:
            label_names = {int(key): value for key, value in payload["label_names"].items()}

    if label_names is None:
        raise ValueError("No embedding files were provided.")

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0), label_names


def save_projection_plot(
    points: torch.Tensor,
    labels: torch.Tensor,
    label_names: dict[int, str],
    title: str,
    output_path: Path,
) -> None:
    """Save a 2D embedding scatter plot."""

    fig, ax = plt.subplots(figsize=(9, 7))
    for label_id, label_name in sorted(label_names.items()):
        mask = labels == label_id
        if not bool(mask.any()):
            continue

        xy = points[mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.75, label=label_name)

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def try_umap(embeddings: torch.Tensor) -> torch.Tensor | None:
    """Run UMAP if the optional dependency is installed."""

    try:
        import umap  # type: ignore
    except ImportError:
        return None

    reducer = umap.UMAP(n_components=2, random_state=42)
    return torch.tensor(reducer.fit_transform(embeddings.numpy()), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize exported game embeddings.")
    parser.add_argument("embedding_files", nargs="+", help="One or more embeddings_*.pt files.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "embedding_plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    embeddings, labels, label_names = load_embedding_files([Path(path) for path in args.embedding_files])

    pca_points = torch.tensor(PCA(n_components=2, random_state=42).fit_transform(embeddings.numpy()), dtype=torch.float32)
    save_projection_plot(pca_points, labels, label_names, "Game Embeddings PCA", output_dir / "embedding_pca.png")
    print(f"saved {output_dir / 'embedding_pca.png'}")

    perplexity = min(30.0, max(5.0, (len(embeddings) - 1) / 3))
    tsne_points = torch.tensor(
        TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(
            embeddings.numpy()
        ),
        dtype=torch.float32,
    )
    save_projection_plot(tsne_points, labels, label_names, "Game Embeddings t-SNE", output_dir / "embedding_tsne.png")
    print(f"saved {output_dir / 'embedding_tsne.png'}")

    umap_points = try_umap(embeddings)
    if umap_points is None:
        print("UMAP is not installed; skipped embedding_umap.png")
    else:
        save_projection_plot(umap_points, labels, label_names, "Game Embeddings UMAP", output_dir / "embedding_umap.png")
        print(f"saved {output_dir / 'embedding_umap.png'}")


if __name__ == "__main__":
    main()
