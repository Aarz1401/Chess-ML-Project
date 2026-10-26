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
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_NAME = "manifest.json"
SPLIT_NAMES = ("train", "val", "test")


def manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / MANIFEST_NAME


def save_manifest(cache_dir: Path, manifest: dict) -> None:
    with open(manifest_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


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
            self._sequences = np.load(split_file(self.cache_dir, self.split, "sequences"), mmap_mode="r")
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
