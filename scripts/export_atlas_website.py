"""Export real experiment vectors, provenance, and replayable games into the static atlas."""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys

import chess.pgn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "website/scripts"), str(ROOT / "src")]
from build_atlas import describe, FEATURES
from game_vectors import BLOCK_WEIGHTS
from game_identity import canonical_game_id, file_sha256


def project(vectors):
    points = PCA(
        n_components=min(30, len(vectors) - 1, vectors.shape[1]), random_state=42
    ).fit_transform(vectors)
    return TSNE(
        n_components=2,
        init="pca",
        perplexity=min(30, (len(points) - 1) / 3),
        learning_rate="auto",
        random_state=42,
    ).fit_transform(points)


def retention(vectors, points):
    k = min(10, len(points) - 1)

    def nearest(x):
        rows = (
            NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
            .fit(x)
            .kneighbors(x, return_distance=False)
        )
        return [list(j for j in row if j != i)[:k] for i, row in enumerate(rows)]

    return float(
        np.mean(
            [
                len(set(a) & set(b)) / k
                for a, b in zip(nearest(vectors), nearest(points))
            ]
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--per-player", type=int, default=125)
    args = p.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    report = json.loads(args.results.read_text())
    rows = [
        json.loads(line)
        for line in (args.cache / "metadata.jsonl").read_text().splitlines()
    ]
    if report["metadata_sha256"] != manifest["metadata_sha256"]:
        raise ValueError("Experiment/source mismatch")
    players = sorted({row["player"] for row in rows})
    by_id = {(r["game_id"], r["player"]): i for i, r in enumerate(rows)}
    chosen = []
    seen = set()
    counts = Counter()
    if args.previous and args.previous.exists():
        previous = json.loads(args.previous.read_text())
        for game in previous["games"]:
            i = by_id.get((game["id"], game["player"]))
            if i is not None and rows[i]["game_id"] not in seen:
                chosen.append(i)
                seen.add(rows[i]["game_id"])
                counts[rows[i]["player"]] += 1
    rng = np.random.default_rng(20261028)
    for player in players:
        candidates = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["player"] == player and r["split"] == "test"
            ]
        )
        rng.shuffle(candidates)
        for i in candidates:
            if counts[player] >= args.per_player:
                break
            if rows[i]["game_id"] in seen:
                continue
            chosen.append(int(i))
            seen.add(rows[i]["game_id"])
            counts[player] += 1
    raw_targets = np.load(args.cache / "targets.npy")
    strategic = np.load(args.cache / "engineered_trainfit.npy")[chosen]
    frozen = np.load(args.cache / "frozen_decision.npy")[chosen]
    available = {
        k: v for k, v in report["training"].items() if v["objective"] == "joint"
    }
    if not available:
        raise ValueError("No completed temporal model")
    selected_run = min(available, key=lambda key: available[key]["best_val_loss"])
    temporal = np.load(args.runs / selected_run / "vectors.npy")[chosen]
    player_vector = (
        np.load(args.runs / "player_42" / "vectors.npy")[chosen]
        if (args.runs / "player_42" / "vectors.npy").exists()
        else None
    )
    games = []
    for i in chosen:
        row = rows[i]
        game = chess.pgn.read_game(io.StringIO(row["pgn"]))
        record = describe(game, row["player"])
        if record is None or record["id"] != row["game_id"]:
            raise ValueError("Replay identity mismatch")
        record.update(
            split=row["split"],
            sample_id=row["sample_id"],
            timeControl=row["time_control"],
            strategicFeatures=np.round(raw_targets[i], 6).tolist(),
        )
        games.append(record)
    baseline = StandardScaler().fit_transform(
        np.asarray([g["features"] for g in games])
    )
    baseline_points = project(baseline)
    matrices = {
        "strategic": strategic,
        "frozen": frozen,
        "temporal": temporal,
        "baseline": baseline,
    }
    if player_vector is not None:
        matrices["player"] = player_vector
    projections = {
        key: project(matrix) if key != "baseline" else baseline_points
        for key, matrix in matrices.items()
    }
    fidelities = {key: retention(matrices[key], projections[key]) for key in matrices}
    for j, game in enumerate(games):
        game["vector"] = np.round(strategic[j], 6).tolist()
        game["position"] = np.round(projections["strategic"][j], 5).tolist()
        game["baselineVector"] = np.round(baseline[j], 6).tolist()
        game["baselinePosition"] = np.round(baseline_points[j], 5).tolist()
        game["frozenVector"] = np.round(frozen[j], 6).tolist()
        game["frozenPosition"] = np.round(projections["frozen"][j], 5).tolist()
        game["temporalVector"] = np.round(temporal[j], 6).tolist()
        game["temporalPosition"] = np.round(projections["temporal"][j], 5).tolist()
        if player_vector is not None:
            game["playerVector"] = np.round(player_vector[j], 6).tolist()
            game["playerPosition"] = np.round(projections["player"][j], 5).tolist()
    report["selected_temporal_run"] = selected_run
    report["selection_rule"] = (
        "Minimum validation loss among joint-objective seeds; test retrieval not used for selection"
    )
    report["map_neighbor_retention10"] = fidelities
    report["display_games"] = len(games)
    report["display_splits"] = dict(Counter(g["split"] for g in games))
    report["date"] = "2026-10-28"
    report["pipeline_source_hashes"] = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in [
            ROOT / "src/game_identity.py",
            ROOT / "src/atlas_model.py",
            ROOT / "scripts/prepare_atlas_corpus.py",
            ROOT / "scripts/extract_atlas_embeddings.py",
            ROOT / "scripts/train_atlas_encoder.py",
            ROOT / "scripts/evaluate_atlas_experiments.py",
            Path(__file__).resolve(),
        ]
    }
    # Retain measured metrics/provenance; omit detailed per-query IDs from the public summary.
    report.pop("examples", None)
    payload = dict(
        schemaVersion=3,
        method="Grouped-corpus neural game atlas",
        players=players,
        games=games,
        featureNames=FEATURES,
        experiment=dict(
            corpusId=manifest["corpus_id"],
            corpusSamples=len(rows),
            uniqueGames=report["unique_games"],
            trainingSamples=report["split_samples"]["train"],
            selectedRun=selected_run,
            model="Maia-3 5M",
            fixedElo=manifest["fixed_elo"],
            date="2026-10-28",
        ),
        representation=dict(
            dimensions=288,
            featureNames=manifest["feature_names"],
            blocks=manifest["feature_blocks"],
            blockWeights=BLOCK_WEIGHTS,
            neighborRetention10=fidelities["strategic"],
            baselineNeighborRetention10=fidelities["baseline"],
            frozenNeighborRetention10=fidelities["frozen"],
            temporalNeighborRetention10=fidelities["temporal"],
            playerNeighborRetention10=fidelities.get("player"),
            projection="PCA then t-SNE, shared settings, seed 42",
        ),
        resultsFile="experiments.json",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "atlas.json").write_text(json.dumps(payload, separators=(",", ":")))
    (args.output / "experiments.json").write_text(json.dumps(report, indent=2))
    from split_atlas_replays import split_replays

    split_replays(args.output)
    print(
        json.dumps(
            dict(
                display_games=len(games), selected_run=selected_run, fidelity=fidelities
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
