"""Train a temporal atlas encoder on grouped, frozen Maia decision embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atlas_model import AtlasEncoder, paired_contrastive, player_contrastive
from game_identity import file_sha256


class Decisions(Dataset):
    def __init__(self, directory, indices, target_mean, target_scale):
        self.sequence = np.load(directory / "sequences.npy", mmap_mode="r")
        self.masks = np.load(directory / "masks.npy")
        self.plies = np.load(directory / "plies.npy")
        self.targets = np.load(directory / "targets.npy")
        self.indices = indices
        self.mean = target_mean
        self.scale = target_scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        return (
            torch.from_numpy(np.array(self.sequence[i])),
            torch.from_numpy(self.masks[i]),
            torch.from_numpy(self.plies[i]),
            torch.from_numpy(
                np.clip((self.targets[i] - self.mean) / self.scale, -3, 3)
            ),
            int(i),
        )


def alternate_mask(mask, generator):
    noise = torch.rand(mask.shape, device=mask.device, generator=generator)
    dropped = mask | (noise < 0.2)
    # Every valid game supplies at least eight decisions; keep four regardless of randomness.
    for row in range(len(mask)):
        if int((~dropped[row]).sum()) < 4:
            valid = torch.where(~mask[row])[0]
            dropped[row, valid[:4]] = False
    return dropped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--objective", choices=["joint", "concept", "player"], default="joint"
    )
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if file_sha256(args.cache / "metadata.jsonl") != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum mismatch")
    rows = [
        json.loads(s) for s in (args.cache / "metadata.jsonl").read_text().splitlines()
    ]
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    if len(train) < 8 or len(val) < 2:
        raise ValueError("Insufficient train/validation groups")
    group_ids = {g: i for i, g in enumerate(sorted({r["game_id"] for r in rows}))}
    all_groups = torch.tensor([group_ids[r["game_id"]] for r in rows], device=device)
    targets = np.load(args.cache / "targets.npy")
    mean = targets[train].mean(0)
    scale = np.maximum(targets[train].std(0), 1e-5)
    loaders = {}
    for name, indices in [
        ("train", train),
        ("val", val),
        ("all", np.arange(len(rows))),
    ]:
        loaders[name] = DataLoader(
            Decisions(args.cache, indices, mean, scale),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
    player_names = sorted({r["player"] for r in rows})
    player_ids = torch.tensor(
        [player_names.index(r["player"]) for r in rows], device=device
    )
    player_counts = np.bincount(
        player_ids[train].cpu().numpy(), minlength=len(player_names)
    )
    class_weights = torch.tensor(
        1 / np.maximum(player_counts, 1), device=device, dtype=torch.float32
    )
    class_weights /= class_weights.mean()
    model = AtlasEncoder(
        input_dim=manifest["dimensions"],
        width=args.width,
        num_players=len(player_names) if args.objective == "player" else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-5
    )
    amp = device.type == "cuda"
    dtype = (
        torch.bfloat16
        if amp and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    scaler = torch.amp.GradScaler(enabled=amp and dtype == torch.float16)
    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / "last.pt"
    best_path = args.output / "best.pt"
    best = float("inf")
    start_epoch = 1
    history = []
    if last.exists():
        if not args.resume:
            raise FileExistsError("Run exists; use --resume or a new output directory")
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        if (
            checkpoint["cache_id"] != manifest["corpus_id"]
            or checkpoint["args"]["objective"] != args.objective
        ):
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best = checkpoint["best"]
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["rng"].cpu())
        if amp and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        stats = {}
        for phase in ("train", "val"):
            model.train(phase == "train")
            totals = np.zeros(3)
            count = 0
            generator = torch.Generator(device=device).manual_seed(
                args.seed + epoch if phase == "train" else 20261028
            )
            for seq, mask, plies, target, index in loaders[phase]:
                seq, mask, plies, target = (
                    seq.to(device),
                    mask.to(device),
                    plies.to(device),
                    target.to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(
                        device_type=device.type, dtype=dtype, enabled=amp
                    ):
                        a, pred = model(seq, alternate_mask(mask, generator), plies)
                        b, pred2 = model(seq, alternate_mask(mask, generator), plies)
                        concept = (
                            F.smooth_l1_loss(pred.float(), target.float())
                            + F.smooth_l1_loss(pred2.float(), target.float())
                        ) / 2
                        contrast = paired_contrastive(
                            a, b, all_groups[index.to(device)]
                        )
                        if args.objective == "player":
                            labels = player_ids[index.to(device)]
                            concept = (
                                F.cross_entropy(
                                    model.player_head(a).float(),
                                    labels,
                                    weight=class_weights,
                                )
                                + F.cross_entropy(
                                    model.player_head(b).float(),
                                    labels,
                                    weight=class_weights,
                                )
                            ) / 2
                            contrast = player_contrastive(a, b, labels)
                        loss = concept + (
                            0.25 * contrast
                            if args.objective in ("joint", "player")
                            else 0
                        )
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Non-finite loss")
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        scaler.step(optimizer)
                        scaler.update()
                totals += np.array(
                    [loss.item(), concept.item(), contrast.item()]
                ) * len(seq)
                count += len(seq)
            stats[phase] = (totals / count).tolist()
        scheduler.step()
        row = dict(
            epoch=epoch,
            train_loss=stats["train"][0],
            val_loss=stats["val"][0],
            val_concept_loss=stats["val"][1],
            val_view_loss=stats["val"][2],
        )
        if args.objective == "player":
            row["val_player_ce"] = row.pop("val_concept_loss")
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = row["val_loss"] < best
        best = min(best, row["val_loss"])
        payload = dict(
            model=model.state_dict(),
            config=model.config,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            best=best,
            history=history,
            args=vars(args) | {"cache": str(args.cache), "output": str(args.output)},
            cache_id=manifest["corpus_id"],
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if amp else None,
            target_mean=mean,
            target_scale=scale,
        )
        torch.save(payload, args.output / "last.tmp")
        (args.output / "last.tmp").replace(last)
        if improved:
            torch.save(payload, best_path)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    vectors = np.zeros((len(rows), 128), dtype=np.float32)
    with torch.inference_mode():
        for seq, mask, plies, _, index in loaders["all"]:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                embedding, _ = model(seq.to(device), mask.to(device), plies.to(device))
            vectors[index.numpy()] = embedding.float().cpu().numpy()
    if not np.isfinite(vectors).all():
        raise RuntimeError("Non-finite exported vectors")
    np.save(args.output / "vectors.npy", vectors)
    report = dict(
        seed=args.seed,
        objective=args.objective,
        cache_id=manifest["corpus_id"],
        epochs=args.epochs,
        best_epoch=checkpoint["epoch"],
        best_val_loss=checkpoint["best"],
        parameters=sum(p.numel() for p in model.parameters()),
        training_seconds=time.perf_counter() - started,
        gpu=torch.cuda.get_device_name(device) if amp else None,
        peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20 if amp else None,
        samples=len(rows),
        train_samples=len(train),
        val_samples=len(val),
        metadata_sha256=manifest["metadata_sha256"],
        config=model.config,
        player_names=player_names if args.objective == "player" else None,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
