"""Validate grouped neural-cache provenance and optionally seal immutable artifacts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_identity import file_sha256


def validate(cache, corpus, data_dir, seal=False):
    manifest = json.loads((cache / "manifest.json").read_text())
    source = json.loads((corpus / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or source["schema_version"] != 1:
        raise ValueError("Unsupported cache schema")
    if source["corpus_id"] != manifest["corpus_id"]:
        raise ValueError("Corpus identity mismatch")
    if file_sha256(cache / "metadata.jsonl") != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum mismatch")
    for name, digest in source["source_hashes"].items():
        if file_sha256(data_dir / name) != digest:
            raise ValueError(f"Source changed: {name}")
    rows = [
        json.loads(line) for line in (cache / "metadata.jsonl").read_text().splitlines()
    ]
    if len(rows) != manifest["samples"] or len({r["sample_id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Duplicate/missing sample identities")
    groups = {}
    for row in rows:
        old = groups.setdefault(row["game_id"], row["split"])
        if old != row["split"]:
            raise ValueError("Cross-split canonical-game leakage")
    n, t, d = manifest["samples"], manifest["decisions"], manifest["dimensions"]
    sequence = np.load(cache / "sequences.npy", mmap_mode="r")
    masks = np.load(cache / "masks.npy")
    if sequence.shape != (n, t, d) or sequence.dtype != np.float16:
        raise ValueError("Incomplete sequence cache")
    if masks.shape != (n, t) or masks.dtype != bool or masks.all(1).any():
        raise ValueError("Invalid masks")
    targets = np.load(cache / "targets.npy")
    if targets.shape != (n, 288) or not np.isfinite(targets).all():
        raise ValueError("Invalid concept targets")
    for name in ("frozen_position", "frozen_decision"):
        vectors = np.load(cache / f"{name}.npy")
        if len(vectors) != n or not np.isfinite(vectors).all():
            raise ValueError("Invalid frozen embeddings")
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)
    integrity = cache / "integrity.json"
    if seal:
        if integrity.exists():
            raise FileExistsError("A sealed cache cannot be resealed")
        names = [
            "manifest.json",
            "metadata.jsonl",
            "sequences.npy",
            "masks.npy",
            "plies.npy",
            "targets.npy",
            "frozen_position.npy",
            "frozen_decision.npy",
            "frozen_position_transform.npz",
            "frozen_decision_transform.npz",
        ]
        integrity.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    corpus_id=manifest["corpus_id"],
                    hashes={name: file_sha256(cache / name) for name in names},
                ),
                indent=2,
            )
        )
    if not integrity.exists():
        raise ValueError("Missing integrity seal")
    for name, digest in json.loads(integrity.read_text())["hashes"].items():
        if file_sha256(cache / name) != digest:
            raise ValueError(f"Sealed artifact changed: {name}")
    return dict(
        samples=n,
        unique_games=len(groups),
        canonical_overlap=0,
        integrity_sha256=file_sha256(integrity),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/pgn")
    p.add_argument("--seal", action="store_true")
    args = p.parse_args()
    print(json.dumps(validate(args.cache, args.corpus, args.data_dir, args.seal)))
