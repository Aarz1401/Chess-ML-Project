"""Shared evaluation utilities: retrieval metrics and player-level aggregation.

Single-game retrieval is the hard setting; the aggregated protocol here is the
standard one from behavioral stylometry: average k query-game embeddings into
one player vector and match it against per-player reference centroids. Accuracy
as a function of k is the headline result.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every game in a loader into normalized metric-space embeddings."""

    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for board, mask, y in loader:
            board = board.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            embeddings.append(model.encode_metric(board, mask).float().cpu())
            labels.append(torch.as_tensor(y).cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    top_k: int,
    id_to_label: dict[int, str],
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
    for label_id, label_name in id_to_label.items():
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


def player_centroids(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return one normalized reference centroid per player."""

    embeddings = F.normalize(embeddings.float(), dim=1)
    centroids = torch.zeros((num_classes, embeddings.shape[1]))
    for label_id in range(num_classes):
        mask = labels.eq(label_id)
        if not bool(mask.any()):
            raise ValueError(f"No reference embeddings for label id {label_id}.")
        centroids[label_id] = embeddings[mask].mean(dim=0)
    return F.normalize(centroids, dim=1)


def aggregated_player_accuracy(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    centroids: torch.Tensor,
    ks: list[int],
    trials: int,
    seed: int,
    id_to_label: dict[int, str],
) -> dict[str, object]:
    """Identify players from k aggregated query games against centroids.

    For every k and every trial, sample k games per player without
    replacement, average their normalized embeddings, and classify by nearest
    centroid. Returns overall and per-player accuracy plus the confusion
    counts for the largest k.
    """

    query_embeddings = F.normalize(query_embeddings.float(), dim=1)
    rng = np.random.default_rng(seed)
    num_classes = centroids.shape[0]

    indices_by_label = {
        label_id: torch.nonzero(query_labels.eq(label_id), as_tuple=False).flatten().numpy()
        for label_id in range(num_classes)
    }
    for label_id, indices in indices_by_label.items():
        if indices.size == 0:
            raise ValueError(f"No query embeddings for label id {label_id}.")

    results: dict[str, object] = {"ks": ks, "trials": trials, "per_k": {}}
    max_k = max(ks)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for k in ks:
        correct_by_label = np.zeros(num_classes, dtype=np.int64)
        total_by_label = np.zeros(num_classes, dtype=np.int64)

        for _ in range(trials):
            for label_id, indices in indices_by_label.items():
                sample_size = min(k, indices.size)
                chosen = rng.choice(indices, size=sample_size, replace=False)
                pooled = F.normalize(query_embeddings[chosen].mean(dim=0, keepdim=True), dim=1)
                pred = int((pooled @ centroids.T).argmax(dim=1))
                correct_by_label[label_id] += int(pred == label_id)
                total_by_label[label_id] += 1
                if k == max_k:
                    confusion[label_id, pred] += 1

        per_label = {
            id_to_label[label_id]: float(correct_by_label[label_id] / max(total_by_label[label_id], 1))
            for label_id in range(num_classes)
        }
        results["per_k"][str(k)] = {
            "overall": float(correct_by_label.sum() / max(total_by_label.sum(), 1)),
            "per_label": per_label,
        }

    results["confusion_at_max_k"] = confusion.tolist()
    results["confusion_labels"] = [id_to_label[i] for i in range(num_classes)]
    return results
