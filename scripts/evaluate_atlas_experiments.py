"""Held-out retrieval diagnostics for atlas representations; no human-style ground truth."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_identity import file_sha256

WEIGHTS = {
    "placement": 1.0,
    "pawns": 1.5,
    "activity": 1.25,
    "king": 1.0,
    "material": 0.5,
    "decisions": 1.0,
}


def engineered_vectors(targets, train, blocks):
    scaler = StandardScaler().fit(targets[train])
    scaled = np.clip(scaler.transform(targets), -3, 3)
    weights = np.array(
        [np.sqrt(WEIGHTS[group] / blocks.count(group)) for group in blocks]
    )
    vectors = scaled * weights
    return (
        vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    ).astype(np.float32)


def balanced_indices(rows, split, cap, seed=42):
    rng = np.random.default_rng(seed)
    indices = []
    for player in sorted({r["player"] for r in rows}):
        eligible = np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["split"] == split and r["player"] == player
            ]
        )
        rng.shuffle(eligible)
        indices.extend(eligible[:cap].tolist())
    return np.asarray(indices)


def bootstrap(values, groups, trials=500):
    grouped = {}
    for value, group in zip(values, groups):
        grouped.setdefault(group, []).append(value)
    means = np.array([np.mean(v) for v in grouped.values()])
    if not len(means):
        return [None, None]
    rng = np.random.default_rng(42)
    estimates = np.array(
        [rng.choice(means, len(means), replace=True).mean() for _ in range(trials)]
    )
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def retrieve(vectors, rows, reference, queries, cross_eco=False):
    predictions = []
    opening = []
    hit5 = []
    chance = []
    groups = []
    neighbors = []
    ref_labels = np.array([rows[i]["player"] for i in reference])
    ref_ids = np.array([rows[i]["game_id"] for i in reference])
    ref_eco = np.array([rows[i]["eco"] for i in reference])
    for start in range(0, len(queries), 128):
        query_indices = queries[start : start + 128]
        scores = vectors[query_indices] @ vectors[reference].T
        for local, i in enumerate(query_indices):
            row = rows[i]
            allowed = ref_ids != row["game_id"]
            if cross_eco:
                if not row["eco"]:
                    continue
                allowed &= (ref_eco != row["eco"]) & (ref_eco != "")
            if not allowed.any():
                continue
            scores[local, ~allowed] = -np.inf
            rank = np.argsort(-scores[local])[: min(5, int(allowed.sum()))]
            predictions.append(float(ref_labels[rank[0]] == row["player"]))
            hit5.append(float(np.any(ref_labels[rank] == row["player"])))
            chance.append(float(np.mean(ref_labels[allowed] == row["player"])))
            if row["eco"] and ref_eco[rank[0]]:
                opening.append(float(ref_eco[rank[0]] == row["eco"]))
            groups.append(row["game_id"])
            neighbors.append(
                dict(
                    query=row["sample_id"],
                    game_id=row["game_id"],
                    player_hit=predictions[-1],
                    neighbors=[rows[reference[j]]["sample_id"] for j in rank],
                )
            )
    return (
        dict(
            player_top1=float(np.mean(predictions)),
            player_top1_ci95=bootstrap(predictions, groups),
            player_hit5=float(np.mean(hit5)),
            random_player_top1=float(np.mean(chance)),
            same_eco_top1=float(np.mean(opening)) if opening else None,
            queries=len(predictions),
            unique_query_games=len(set(groups)),
        ),
        neighbors,
    )


def metadata_baseline(rows, train, test):
    def features(r):
        year = r["date"][:4]
        elo = r["white_elo"] if r["target_color"] else r["black_elo"]
        return dict(
            year=int(year) - 2000 if year.isdigit() else 0,
            elo=int(elo) - 2700 if elo.isdigit() else 0,
            eco=r["eco"] or "unknown",
            white=bool(r["target_color"]),
            plies=int(r["ply_count"]),
        )

    dictionaries = [features(r) for r in rows]
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform([dictionaries[i] for i in train])
    scaler = StandardScaler(with_mean=False)
    x = scaler.fit_transform(x)
    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(x, [rows[i]["player"] for i in train])
    accuracy = float(
        classifier.score(
            scaler.transform(vectorizer.transform([dictionaries[i] for i in test])),
            [rows[i]["player"] for i in test],
        )
    )
    return dict(
        accuracy=accuracy,
        iterations=int(classifier.n_iter_.max()),
        converged=bool(classifier.n_iter_.max() < classifier.max_iter),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if file_sha256(args.cache / "metadata.jsonl") != manifest["metadata_sha256"]:
        raise ValueError("Metadata mismatch")
    rows = [
        json.loads(s) for s in (args.cache / "metadata.jsonl").read_text().splitlines()
    ]
    split_groups = {
        split: {r["game_id"] for r in rows if r["split"] == split}
        for split in ("train", "val", "test")
    }
    assert not (
        split_groups["train"] & split_groups["test"]
        or split_groups["val"] & split_groups["test"]
        or split_groups["train"] & split_groups["val"]
    )
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    exploratory = balanced_indices(rows, "test", 200)
    observed_groups = {rows[i]["game_id"] for i in exploratory}
    audit_rows = [
        r if r["game_id"] not in observed_groups else dict(r, split="excluded")
        for r in rows
    ]
    test = balanced_indices(audit_rows, "test", 200)
    reference = balanced_indices(rows, "train", 300)
    targets = np.load(args.cache / "targets.npy")
    engineered = engineered_vectors(targets, train, manifest["feature_blocks"])
    np.save(args.cache / "engineered_trainfit.npy", engineered)
    representations = {
        "strategic": engineered,
        "frozen_position": np.load(args.cache / "frozen_position.npy"),
        "frozen_decision": np.load(args.cache / "frozen_decision.npy"),
    }
    training = {}
    for report_path in sorted(args.runs.glob("*/report.json")):
        if not (report_path.parent / "vectors.npy").exists():
            continue
        report = json.loads(report_path.read_text())
        if (
            report.get("cache_id") != manifest["corpus_id"]
            or report.get("metadata_sha256") != manifest["metadata_sha256"]
        ):
            continue
        key = f"{report['objective']}_{report['seed']}"
        representations[key] = np.load(report_path.parent / "vectors.npy")
        training[key] = report

    def modern_eligible(row):
        year = row["date"][:4]
        ratings = [row["white_elo"], row["black_elo"]]
        return (
            year.isdigit()
            and 2015 <= int(year) <= 2026
            and all(v.isdigit() and 2400 <= int(v) <= 3000 for v in ratings)
        )

    modern_rows = [
        (
            r
            if modern_eligible(r) and r["game_id"] not in observed_groups
            else dict(r, split="excluded")
        )
        for r in rows
    ]
    modern_reference = balanced_indices(modern_rows, "train", 300)
    modern_test = balanced_indices(modern_rows, "test", 200)
    labels_in_reference = {rows[i]["player"] for i in modern_reference}
    modern_test = np.array(
        [i for i in modern_test if rows[i]["player"] in labels_in_reference]
    )
    results = {}
    examples = {}
    outcomes = {}
    for name, vectors in representations.items():
        assert len(vectors) == len(rows) and np.isfinite(vectors).all()
        normal, normal_neighbors = retrieve(vectors, rows, reference, test)
        cross, neighbors = retrieve(vectors, rows, reference, test, cross_eco=True)
        modern, modern_neighbors = retrieve(
            vectors, rows, modern_reference, modern_test, cross_eco=True
        )
        results[name] = dict(
            standard=normal,
            cross_eco=cross,
            modern_cross_eco=modern,
            dimensions=vectors.shape[1],
        )
        outcomes[name] = {
            "standard": normal_neighbors,
            "cross_eco": neighbors,
            "modern_cross_eco": modern_neighbors,
        }
        examples[name] = neighbors[:80]
        print(name, json.dumps(results[name]), flush=True)
    comparisons = {}
    for name, views in outcomes.items():
        comparisons[name] = {}
        for kind, observations in views.items():
            base = {r["query"]: r for r in outcomes["strategic"][kind]}
            paired = [
                (r["player_hit"] - base[r["query"]]["player_hit"], r["game_id"])
                for r in observations
                if r["query"] in base
            ]
            differences = [value for value, _ in paired]
            groups = [group for _, group in paired]
            comparisons[name][kind] = dict(
                mean_difference=float(np.mean(differences)),
                ci95=bootstrap(differences, groups),
                queries=len(paired),
            )
    metadata_report = metadata_baseline(rows, train, test)
    output = dict(
        schema_version=1,
        corpus_id=manifest["corpus_id"],
        samples=len(rows),
        unique_games=len({r["game_id"] for r in rows}),
        split_samples=dict(Counter(r["split"] for r in rows)),
        reference_samples=len(reference),
        query_samples=len(test),
        metadata_sha256=manifest["metadata_sha256"],
        benchmark=dict(
            kind="fresh_grouped_audit",
            excluded_exploratory_queries=len(exploratory),
            excluded_exploratory_game_groups=len(observed_groups),
            audit_game_groups=len({rows[i]["game_id"] for i in test}),
        ),
        modern_cohort=dict(
            years=[2015, 2026],
            both_rating_range=[2400, 3000],
            reference_samples=len(modern_reference),
            query_samples=len(modern_test),
            players=sorted(labels_in_reference),
        ),
        extraction=manifest,
        results=results,
        training=training,
        metadata_player_classifier_accuracy=metadata_report["accuracy"],
        metadata_classifier=metadata_report,
        examples=examples,
        comparisons_to_strategic=comparisons,
        caveats=[
            "Player retrieval is a player-association diagnostic, not validated style similarity.",
            "Same-ECO retrieval diagnoses opening content; cross-ECO excludes exact ECO codes, not all opening similarity.",
            "The metadata classifier is a separate classification task, not a directly comparable retrieval method.",
            "Concept supervision uses engineered targets, so concept recovery is not independent validation.",
            "Time controls are unknown in the source PGNs; no time-control-matched claim is made.",
            "No human-judged semantic benchmark has been collected.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
