"""Frozen, pinned Maia-3 position/decision embeddings with train-only pooling transforms."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_identity import file_sha256


def pooled_features(sequences, masks):
    """Ordered segment means and standard deviations; padded decisions excluded."""
    rows = []
    for sequence, mask in zip(sequences, masks):
        valid = sequence[~mask].astype(np.float32)
        segments = np.array_split(valid, 3)
        rows.append(
            np.concatenate([np.concatenate([s.mean(0), s.std(0)]) for s in segments])
        )
    return np.asarray(rows, dtype=np.float32)


def reduce_on_train(features, train, destination, name):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[train])
    dim = min(128, scaled.shape[1], scaled.shape[0] - 1)
    reducer = PCA(
        n_components=dim, svd_solver="randomized", random_state=42, iterated_power=3
    )
    reducer.fit(scaled)
    vectors = reducer.transform(scaler.transform(features)).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    np.save(destination / f"{name}.npy", vectors)
    np.savez(
        destination / f"{name}_transform.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        pca_mean=reducer.mean_,
        components=reducer.components_,
    )
    return float(reducer.explained_variance_ratio_.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maia-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-manifest", type=Path, required=True)
    p.add_argument("--batch-positions", type=int, default=512)
    p.add_argument("--elo", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a new extraction directory")
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((args.corpus / "manifest.json").read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if file_sha256(args.checkpoint) != model_manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match its manifest")
    revision = subprocess.check_output(
        ["git", "-C", str(args.maia_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != model_manifest["code_revision"]:
        raise ValueError("Maia code revision mismatch")
    sys.path.insert(0, str(args.maia_source))
    from maia3.uci import parse_args
    from maia3.models import MAIA3Model

    cfg = parse_args(
        [
            "--model",
            "5m",
            "--checkpoint-path",
            str(args.checkpoint),
            "--device",
            args.device,
        ]
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for GPU extraction")
    torch.set_num_threads(4)
    model = MAIA3Model(cfg).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(
        {k.replace("smolgen", "gab"): v for k, v in state.items()}, strict=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows = [
        json.loads(line)
        for line in (args.corpus / "metadata.jsonl").read_text().splitlines()
    ]
    if file_sha256(args.corpus / "metadata.jsonl") != corpus["metadata_sha256"]:
        raise ValueError("Corpus metadata changed after preparation")
    histories = []
    actions = []
    masks = []
    plies = []
    targets = []
    for shard in corpus["shards"]:
        arrays = np.load(args.corpus / f"{shard['shard']}.npz")
        idx = shard["indices"]
        histories.append(arrays["histories"][idx])
        actions.append(arrays["actions"][idx])
        masks.append(arrays["masks"][idx])
        plies.append(arrays["plies"][idx])
        targets.append(arrays["targets"][idx])
    histories = np.concatenate(histories)
    actions = np.concatenate(actions)
    masks = np.concatenate(masks)
    plies = np.concatenate(plies)
    targets = np.concatenate(targets)
    assert len(rows) == len(histories) == corpus["samples"]
    if histories.max() > 12:
        raise ValueError("Invalid board codes")
    np.save(args.output / "masks.npy", masks)
    np.save(args.output / "plies.npy", plies)
    np.save(args.output / "targets.npy", targets)
    shutil.copy2(args.corpus / "metadata.jsonl", args.output / "metadata.jsonl")
    sequence = np.lib.format.open_memmap(
        args.output / "sequences.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), corpus["decisions"], cfg.dim_vit * 3),
    )
    indices = np.argwhere(~masks)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_positions):
            index = indices[start : start + args.batch_positions]
            a, b = index[:, 0], index[:, 1]
            codes = torch.from_numpy(histories[a, b].astype(np.int64)).to(device)
            # (B, history, square, 12) -> official chronological square tokens.
            tokens = (
                F.one_hot(codes, num_classes=13)[..., 1:]
                .permute(0, 2, 1, 3)
                .reshape(-1, 64, 96)
                .float()
            )
            elo = torch.full((len(index),), args.elo, device=device, dtype=torch.long)
            chosen = torch.from_numpy(actions[a, b].astype(np.int64)).to(device)
            context = (
                torch.autocast("cuda", dtype=dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                own = model.interpolate_elo(elo).unsqueeze(1).expand(-1, 64, -1)
                opponent = model.interpolate_elo(elo).unsqueeze(1).expand(-1, 64, -1)
                x = model.token_projection(torch.cat([tokens, own, opponent], -1))
                if hasattr(model, "abs_pe"):
                    x = model.abs_pe(x)
                x = model.transformer(x)
                batch = torch.arange(len(index), device=device)
                feature = torch.cat(
                    [x.mean(1), x[batch, chosen[:, 0]], x[batch, chosen[:, 1]]], 1
                )
            if not bool(torch.isfinite(feature).all()):
                raise RuntimeError("Non-finite encoder activations")
            sequence[a, b] = feature.float().cpu().numpy().astype(np.float16)
            if start % (args.batch_positions * 40) == 0:
                print(
                    f"{start}/{len(indices)} positions; {start/max(time.perf_counter()-started,1):.1f} positions/sec",
                    flush=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    sequence.flush()
    train = np.array([r["split"] == "train" for r in rows])
    if train.sum() < 3:
        raise ValueError("Insufficient training groups for PCA")
    print("Fitting train-only pooling transforms...", flush=True)
    variance = reduce_on_train(
        pooled_features(sequence, masks), train, args.output, "frozen_decision"
    )
    context_variance = reduce_on_train(
        pooled_features(sequence[:, :, : cfg.dim_vit], masks),
        train,
        args.output,
        "frozen_position",
    )
    spec = dict(
        schema_version=1,
        corpus_id=corpus["corpus_id"],
        model=model_manifest,
        fixed_elo=args.elo,
        dimensions=cfg.dim_vit * 3,
        decisions=corpus["decisions"],
        samples=len(rows),
        positions=len(indices),
        inference_seconds=inference_seconds,
        positions_per_second=len(indices) / inference_seconds,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        amp_dtype=str(dtype),
        peak_gpu_mib=(
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
        ),
        pca_variance=variance,
        position_pca_variance=context_variance,
        metadata_sha256=file_sha256(args.output / "metadata.jsonl"),
        feature_names=corpus["feature_names"],
        feature_blocks=corpus["feature_blocks"],
    )
    (args.output / "manifest.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2), flush=True)


if __name__ == "__main__":
    main()
