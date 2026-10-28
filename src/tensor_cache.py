"""Disk cache for precomputed game tensors.

Building the 84-plane tensors is the slowest part of every run, so it is done
once by scripts/precompute_tensors.py and stored on disk. Training jobs then
memory-map the arrays and stream batches through DataLoader workers instead of
re-parsing PGN files.

Cache directory layout:

    manifest.json            settings, label names, per-split counts
    train_sequences.npy      (N, T, C, 8, 8) float16
    train_masks.npy          (N, T) bool, True means padded
    train_labels.npy         (N,) int64
    train_metadata.csv       provenance for every cached game
    val_*, test_*            same files per split
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from game_identity import file_sha256

MANIFEST_NAME = "manifest.json"
SPLIT_NAMES = ("train", "val", "test")


def manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def save_manifest(cache_dir: Path, manifest: dict) -> None:
    temporary = manifest_path(cache_dir).with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temporary.replace(manifest_path(cache_dir))


def load_manifest(cache_dir: Path) -> dict:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No tensor cache manifest at {path}. Run scripts/precompute_tensors.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_file(cache_dir: Path, split: str, kind: str) -> Path:
    return Path(cache_dir) / f"{split}_{kind}.npy"


def validate_cache(cache_dir: Path, data_dir: Path | None = None) -> dict:
    """Reject stale schemas, incomplete arrays, and cross-split game leakage."""
    cache_dir = Path(cache_dir)
    manifest = load_manifest(cache_dir)
    if manifest.get("schema_version") != 2 or not manifest.get("grouped_split"):
        raise ValueError("Legacy cache has no grouped game identities; rebuild it")
    if not manifest.get("source_hashes"):
        raise ValueError("Cache is missing source hashes")
    if data_dir is not None:
        for name, digest in manifest["source_hashes"].items():
            path = Path(data_dir) / name
            if not path.exists() or file_sha256(path) != digest:
                raise ValueError(f"Cache source changed: {name}")
    groups = {}
    for split in SPLIT_NAMES:
        sequence = np.load(split_file(cache_dir, split, "sequences"), mmap_mode="r")
        masks = np.load(split_file(cache_dir, split, "masks"), mmap_mode="r")
        labels = np.load(split_file(cache_dir, split, "labels"), mmap_mode="r")
        n = manifest["counts"][split]
        if sequence.shape != (
            n,
            manifest["sequence_length"],
            manifest["input_channels"],
            8,
            8,
        ):
            raise ValueError(f"Cache sequence shape mismatch: {split}")
        if masks.shape != (n, manifest["sequence_length"]) or labels.shape != (n,):
            raise ValueError(f"Cache labels/masks shape mismatch: {split}")
        if (
            masks.dtype != bool
            or sequence.dtype != np.float16
            or labels.dtype != np.int64
        ):
            raise ValueError(f"Cache dtype mismatch: {split}")
        if n and (labels.min() < 0 or labels.max() >= len(manifest["labels"])):
            raise ValueError(f"Cache contains unknown labels: {split}")
        if np.any(masks.all(axis=1)):
            raise ValueError(f"Cache contains empty games: {split}")
        with (cache_dir / f"{split}_metadata.csv").open() as source:
            records = list(csv.DictReader(source))
        if len(records) != n or any(not row.get("game_id") for row in records):
            raise ValueError(f"Missing canonical metadata: {split}")
        if len({(r["game_id"], r["target_color"]) for r in records}) != n:
            raise ValueError(f"Duplicate game perspectives: {split}")
        for row in records:
            old = groups.setdefault(row["game_id"], split)
            if old != split:
                raise ValueError("Canonical game appears in multiple splits")
    return manifest


class CachedChessDataset(Dataset):
    """Memory-mapped dataset over one cached split.

    Arrays are opened lazily per process so DataLoader workers each get their
    own file handles instead of inheriting one across fork boundaries.
    """

    def __init__(self, cache_dir: Path, split: str) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}, expected one of {SPLIT_NAMES}")
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.labels = np.load(split_file(self.cache_dir, split, "labels"))
        self.masks = np.load(split_file(self.cache_dir, split, "masks"))
        self._sequences: np.ndarray | None = None

    @property
    def sequences(self) -> np.ndarray:
        if self._sequences is None:
            self._sequences = np.load(
                split_file(self.cache_dir, self.split, "sequences"), mmap_mode="r"
            )
        return self._sequences

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sequence = torch.from_numpy(np.array(self.sequences[idx]))
        mask = torch.from_numpy(np.array(self.masks[idx]))
        return sequence, mask, int(self.labels[idx])

    def __getstate__(self) -> dict:
        # Drop the open memmap before pickling into DataLoader workers.
        state = dict(self.__dict__)
        state["_sequences"] = None
        return state
