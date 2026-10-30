"""Build a bounded moment atlas from sealed Maia caches; no new model training."""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib
from importlib.metadata import version as package_version
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from multiprocessing.util import Finalize

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "website/scripts")]
from game_identity import file_sha256
from game_vectors import position_features, BLOCK_WEIGHTS
from moment_matching import facts, position_key, ranked, blind_trial

SEED = 20261030


def game_moments(task):
    index, row, wanted = task
    game = chess.pgn.read_game(io.StringIO(row["pgn"]))
    if not game or game.errors:
        raise ValueError("Invalid cached PGN")
    board = game.board()
    moves = list(game.mainline_moves())
    result = []
    for ply, move in enumerate(moves):
        if ply in wanted:
            if board.turn != row["target_color"]:
                raise ValueError("Position/cache target orientation mismatch")
            values, names, blocks = position_features(board, board.turn)
            result.append(
                dict(
                    row=index,
                    slot=wanted[ply],
                    ply=ply,
                    fen=board.fen(),
                    positionKey=position_key(board),
                    facts=facts(board, board.turn),
                    raw=values,
                    names=names,
                    blocks=blocks,
                )
            )
        board.push(move)
        if ply > 100:
            break
    return result


ENGINE = None


def start_engine(executable):
    global ENGINE
    ENGINE = chess.engine.SimpleEngine.popen_uci(executable)
    ENGINE.configure({"Threads": 1, "Hash": 16})
    Finalize(None, ENGINE.quit, exitpriority=10)


def annotate(task):
    fen, nodes = task
    board = chess.Board(fen)
    # Fresh game token clears search state, independent of worker scheduling.
    info = ENGINE.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
    score = info["score"].pov(board.turn)
    return dict(
        cp=score.score(mate_score=10000),
        mate=score.mate(),
        depth=info.get("depth"),
        nodes=info.get("nodes"),
        bestMove=info["pv"][0].uci(),
        bestSan=board.san(info["pv"][0]),
    )


def normalized(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--stockfish", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--nodes", type=int, default=8000)
    p.add_argument("--fit-games", type=int, default=1000)
    args = p.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a new versioned output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [
        "scripts/build_moments.py",
        "src/moment_matching.py",
        "website/scripts/game_vectors.py",
    ]
    source_hashes = {f: file_sha256(ROOT / f) for f in source_files}
    for f in source_files:
        saved = args.output / "source" / f
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, saved)
    started = time.perf_counter()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if manifest["dimensions"] != 768:
        raise ValueError("This preset requires the pinned 256D Maia-3 position tokens")
    integrity = json.loads((args.cache / "integrity.json").read_text())
    for name, digest in integrity["hashes"].items():
        if file_sha256(args.cache / name) != digest:
            raise ValueError(f"Sealed cache changed: {name}")
    rows = [
        json.loads(s) for s in (args.cache / "metadata.jsonl").read_text().splitlines()
    ]
    by_sample = {r["sample_id"]: i for i, r in enumerate(rows)}
    atlas = json.loads(args.atlas.read_text())
    display = [by_sample[g["sample_id"]] for g in atlas["games"]]
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    rng = np.random.default_rng(SEED)
    fit = sorted(
        rng.choice(train, min(args.fit_games, len(train)), replace=False).tolist()
    )
    all_rows = sorted(set(display + fit))
    plies = np.load(args.cache / "plies.npy")
    masks = np.load(args.cache / "masks.npy")
    sequence = np.load(args.cache / "sequences.npy", mmap_mode="r")
    tasks = []
    for i in all_rows:
        slots = np.where(~masks[i] & (plies[i] >= 24) & (plies[i] <= 100))[0]
        chosen = (
            slots[
                np.unique(
                    np.linspace(0, len(slots) - 1, min(6, len(slots))).astype(int)
                )
            ]
            if len(slots)
            else []
        )
        tasks.append((i, rows[i], {int(plies[i, s]): int(s) for s in chosen}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(game_moments, tasks))
    all_moments = [m for group in groups for m in group]
    if not all_moments:
        raise ValueError("No eligible middlegame positions")
    print(f"Extracted {len(all_moments)} display + fitting positions", flush=True)
    fit_set = set(fit)
    fit_mask = np.array([m["row"] in fit_set for m in all_moments])
    if any(
        rows[m["row"]]["split"] != "train"
        for m, use in zip(all_moments, fit_mask)
        if use
    ):
        raise ValueError("Transform fit leakage")
    raw = np.asarray([m["raw"] for m in all_moments], dtype=np.float32)
    scaler = StandardScaler().fit(raw[fit_mask])
    blocks = all_moments[0]["blocks"]
    weights = np.array([np.sqrt(BLOCK_WEIGHTS[b] / blocks.count(b)) for b in blocks])
    strategic = normalized(np.clip(scaler.transform(raw), -3, 3) * weights).astype(
        np.float32
    )
    # Only pooled square tokens: chosen from/to features are deliberately excluded.
    frozen = np.asarray(
        [sequence[m["row"], m["slot"], :256] for m in all_moments], dtype=np.float32
    )
    ns = StandardScaler().fit(frozen[fit_mask])
    pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED).fit(
        ns.transform(frozen[fit_mask])
    )
    neural = normalized(pca.transform(ns.transform(frozen))).astype(np.float32)
    np.savez(
        args.output / "transforms.npz",
        strategic_mean=scaler.mean_,
        strategic_scale=scaler.scale_,
        strategic_weights=weights,
        neural_mean=ns.mean_,
        neural_scale=ns.scale_,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
    )
    display_set = set(display)
    keep = [i for i, m in enumerate(all_moments) if m["row"] in display_set]
    moments = [all_moments[i] for i in keep]
    strategic, neural = strategic[keep], neural[keep]
    games = []
    game_index = {}
    old_by_sample = {g["sample_id"]: g for g in atlas["games"]}
    for i in display:
        row = rows[i]
        if not any(m["row"] == i for m in moments):
            continue
        old = old_by_sample[row["sample_id"]]
        game_index[i] = len(games)
        games.append(
            {
                k: old[k]
                for k in [
                    "id",
                    "white",
                    "black",
                    "date",
                    "eco",
                    "player",
                    "split",
                    "replayChunk",
                ]
            }
        )
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish)) as engine:
        engine_id = engine.id
    print(f"Annotating {len(moments)} positions with {engine_id}", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=start_engine,
        initargs=(str(args.stockfish),),
    ) as pool:
        annotations = list(
            pool.map(annotate, [(m["fen"], args.nodes) for m in moments], chunksize=8)
        )
    for i, m in enumerate(moments):
        m["game"] = game_index[m.pop("row")]
        m["id"] = f"{games[m['game']]['id']}:{m['ply']}"
        m["engine"] = annotations[i]
        for k in ["raw", "blocks", "names", "slot"]:
            m.pop(k)
    vectors = dict(strategic=strategic, neural=neural)
    reference = [
        i for i, m in enumerate(moments) if games[m["game"]]["split"] == "train"
    ]
    # One predeclared, deterministic query per held-out canonical game.
    by_game = {}
    for i, m in enumerate(moments):
        if games[m["game"]]["split"] == "test":
            by_game.setdefault(m["game"], []).append(i)
    queries = [
        v[len(v) // 2]
        for k, v in sorted(by_game.items(), key=lambda kv: games[kv[0]]["id"])
    ]
    diagnostics = {name: [] for name in ["strategic", "neural", "random"]}
    trials = []
    key = {}
    disagreed = 0
    for q in queries:
        rankings = {
            name: ranked(q, reference, moments, games, vec)
            for name, vec in vectors.items()
        }
        if not rankings["strategic"]:
            continue
        choice = {name: r[0] for name, r in rankings.items()}
        choice["random"] = int(rng.choice(rankings["strategic"]))
        for name, c in choice.items():
            a, b = moments[q], moments[c]
            diagnostics[name].append(
                dict(
                    engine_gap=abs(a["engine"]["cp"] - b["engine"]["cp"]),
                    material_gap=abs(a["facts"]["material"] - b["facts"]["material"]),
                    same_player=games[a["game"]]["player"]
                    == games[b["game"]]["player"],
                )
            )
        if choice["strategic"] != choice["neural"]:
            disagreed += 1
            if len(trials) < 100:
                trial, mapping = blind_trial(
                    q,
                    choice["strategic"],
                    choice["neural"],
                    games[moments[q]["game"]]["id"],
                )
                trial["partition"] = "development" if len(trials) < 20 else "audit"
                trials.append(trial)
                key[trial["id"]] = dict(
                    **mapping,
                    query_game=games[moments[q]["game"]]["id"],
                    partition=trial["partition"],
                )
    if len(trials) != 100:
        raise ValueError(f"Need 100 distinct-game disagreements; found {len(trials)}")
    for m in moments:
        m.pop("positionKey")
    diagnostic_report = {
        name: dict(
            queries=len(values),
            median_engine_gap_cp=float(np.median([r["engine_gap"] for r in values])),
            median_material_gap_pawns=float(
                np.median([r["material_gap"] for r in values])
            ),
            same_player_fraction=float(np.mean([r["same_player"] for r in values])),
        )
        for name, values in diagnostics.items()
    }
    version = hashlib.sha256(
        (
            file_sha256(args.atlas)
            + file_sha256(args.output / "transforms.npz")
            + str(SEED)
            + file_sha256(args.stockfish)
            + str(args.nodes)
        ).encode()
    ).hexdigest()[:20]
    data = dict(
        schemaVersion=1,
        version=version,
        games=games,
        moments=moments,
        vectors={
            name: np.round(v.astype(np.float64), 5).tolist()
            for name, v in vectors.items()
        },
        methods=dict(
            strategic="89 target-relative position features; train-only scaling and block weights",
            neural="Frozen Maia-3 pooled position tokens; train-only 64D PCA; no chosen move features",
        ),
    )
    review = dict(
        schemaVersion=1,
        version=version,
        trials=trials,
        instructions="Judge strategic relevance before viewing continuations. A/B methods and game identities are hidden. Tie and neither are valid. Export your anonymous local responses.",
    )
    report = dict(
        schemaVersion=1,
        version=version,
        date="2026-10-30",
        corpus_id=manifest["corpus_id"],
        cache_integrity_sha256=file_sha256(args.cache / "integrity.json"),
        atlas_sha256=file_sha256(args.atlas),
        games=len(games),
        moments=len(moments),
        fit_games=len(fit),
        fit_positions=int(fit_mask.sum()),
        fit_split="train",
        neural_dimensions=64,
        strategic_dimensions=len(blocks),
        pca_variance=float(pca.explained_variance_ratio_.sum()),
        sampling="Up to six cached target-to-move positions per game, plies 24–100. No chosen/future moves in vectors.",
        benchmark=dict(
            query_games=len(queries),
            reference_positions=len(reference),
            reference_split="train",
            query_split="test",
            constraints="Different canonical game, different exact ECO with both codes known, exclude exact/mirrored transpositions",
            disagreement_fraction=disagreed / len(queries),
            diagnostics=diagnostic_report,
        ),
        human_evaluation=dict(
            status="awaiting independent human judgments",
            prepared_trials=100,
            development_trials=20,
            audit_trials=80,
            responses=0,
            scope="Paired comparison conditional on methods disagreeing; not overall recommendation accuracy",
        ),
        engine=dict(
            id=engine_id,
            nodes_limit=args.nodes,
            threads_per_worker=1,
            workers=args.workers,
            binary_sha256=file_sha256(args.stockfish),
            median_depth=float(np.median([a["depth"] for a in annotations])),
        ),
        runtime=dict(
            seconds=time.perf_counter() - started,
            host=platform.node(),
            slurm_job_id=os.getenv("SLURM_JOB_ID"),
            python=platform.python_version(),
            machine=platform.machine(),
            execution="slurm-cpu" if os.getenv("SLURM_JOB_ID") else "local-cpu",
            packages={
                name: package_version(name)
                for name in ["numpy", "scikit-learn", "chess"]
            },
        ),
        source_hashes=source_hashes,
        limitations=[
            "Engine annotations use a small fixed node budget, not ground truth or coaching advice.",
            "Material and evaluation gaps are diagnostics, not independent semantic similarity judgments.",
            "Both position methods share the current board. Neural history contains previous legal positions.",
            "The corpus has eight curated player collections and unknown pretraining overlap.",
            "No new model was trained; this experiment reuses frozen cached features.",
        ],
    )
    if any(file_sha256(ROOT / f) != digest for f, digest in source_hashes.items()):
        raise RuntimeError(
            "Source changed during the analysis; retain the snapshot and start a fresh run"
        )
    for name, value in [
        ("moments.json", data),
        ("moment-review.json", review),
        ("moment-report.json", report),
        ("review-key.json", dict(version=version, trials=key)),
    ]:
        (args.output / name).write_text(json.dumps(value, separators=(",", ":")))
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                version=version,
                files={
                    f: file_sha256(args.output / f)
                    for f in [
                        "moments.json",
                        "moment-review.json",
                        "moment-report.json",
                        "review-key.json",
                        "transforms.npz",
                    ]
                },
            ),
            indent=2,
        )
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
