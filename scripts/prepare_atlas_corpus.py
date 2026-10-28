"""Deduplicate PGNs, group splits by canonical game, and cache compact legal histories."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "website" / "scripts")]
import chess
import chess.pgn
import numpy as np
from game_identity import canonical_game_id, target_color, split_for_game, file_sha256
from game_vectors import strategic_features

SCHEMA = 1


def board_codes(board):
    """Exactly the side-to-move rank/color convention of Maia-3 tokenize_board."""
    if board.turn == chess.BLACK:
        board = board.mirror()
    result = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        result[square] = piece.piece_type + (0 if piece.color else 6)
    return result


def compact_history(game, color, decisions):
    moves = list(game.mainline_moves())
    valid = [
        i
        for i in range(12, min(120, len(moves)))
        if (game.board().turn ^ bool(i % 2)) == color
    ]
    if len(valid) < 8:
        return None
    indices = np.unique(
        np.linspace(0, len(valid) - 1, min(decisions, len(valid))).astype(int)
    )
    wanted = {valid[i] for i in indices}
    histories = np.zeros((decisions, 8, 64), dtype=np.uint8)
    actions = np.zeros((decisions, 2), dtype=np.int16)
    plies = np.zeros(decisions, dtype=np.int16)
    mask = np.ones(decisions, dtype=bool)
    board = game.board()
    history = deque([board_codes(board)], maxlen=8)
    index = 0
    for ply, move in enumerate(moves):
        if ply in wanted:
            frames = list(history)
            frames = [frames[0]] * (8 - len(frames)) + frames
            histories[index] = np.stack(frames)
            from_sq, to_sq = move.from_square, move.to_square
            if not color:
                from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(
                    to_sq
                )
            actions[index] = [from_sq, to_sq]
            plies[index] = ply
            mask[index] = False
            index += 1
        board.push(move)
        if ply >= 119:
            break
        history.append(board_codes(board))
    return histories, actions, plies, mask


def prepare_file(task):
    path, destination, decisions, max_per_file = task
    path, destination = Path(path), Path(destination)
    records, history_rows, action_rows, ply_rows, mask_rows, targets = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    counts = Counter()
    seen = set()
    with path.open(encoding="utf-8", errors="replace") as source:
        game_index = 0
        while (game := chess.pgn.read_game(source)) is not None:
            game_index += 1
            counts["records"] += 1
            if game.errors or not game.board().is_valid():
                counts["invalid"] += 1
                continue
            color = target_color(game.headers, path.stem)
            if color is None:
                counts["ambiguous_target"] += 1
                continue
            game_id = canonical_game_id(game)
            if (game_id, color) in seen:
                counts["duplicates_within_collection"] += 1
                continue
            seen.add((game_id, color))
            compact = compact_history(game, color, decisions)
            if compact is None:
                counts["too_short"] += 1
                continue
            try:
                features, names, blocks = strategic_features(game, path.stem)
            except ValueError:
                counts["invalid_features"] += 1
                continue
            history, actions, plies, mask = compact
            h = game.headers
            rows = dict(
                game_id=game_id,
                sample_id=f"{game_id}-{'w' if color else 'b'}",
                player=path.stem,
                target_color=bool(color),
                split=split_for_game(game_id),
                white=h.get("White", "?"),
                black=h.get("Black", "?"),
                date=h.get("Date", "?"),
                event=h.get("Event", "?"),
                eco=h.get("ECO", ""),
                result=h.get("Result", "*"),
                white_elo=h.get("WhiteElo", ""),
                black_elo=h.get("BlackElo", ""),
                time_control=h.get("TimeControl") or None,
                source_file=path.name,
                source_index=game_index - 1,
                ply_count=game.end().ply(),
                pgn=game.accept(
                    chess.pgn.StringExporter(
                        headers=True, variations=False, comments=False
                    )
                ),
            )
            records.append(rows)
            history_rows.append(history)
            action_rows.append(actions)
            ply_rows.append(plies)
            mask_rows.append(mask)
            targets.append(features)
            if max_per_file and len(records) >= max_per_file:
                break
    if not records:
        raise ValueError(f"No usable games in {path}")
    shard = path.stem
    np.savez_compressed(
        destination / f"{shard}.npz",
        histories=np.asarray(history_rows),
        actions=np.asarray(action_rows),
        plies=np.asarray(ply_rows),
        masks=np.asarray(mask_rows),
        targets=np.asarray(targets, dtype=np.float32),
    )
    (destination / f"{shard}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    counts["accepted"] = len(records)
    return {
        "shard": shard,
        "counts": dict(counts),
        "source_sha256": file_sha256(path),
        "size": len(records),
        "feature_names": names,
        "feature_blocks": blocks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/pgn")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--decisions", type=int, default=24)
    p.add_argument("--max-per-file", type=int, default=0)
    args = p.parse_args()
    if not 8 <= args.decisions <= 54:
        p.error("decisions must be between 8 and 54")
    paths = sorted(args.data_dir.glob("*.pgn"))
    if not paths:
        p.error("No PGN files found")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a new versioned corpus directory")
    tasks = [
        (str(path), str(args.output), args.decisions, args.max_per_file)
        for path in paths
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(prepare_file, tasks))
    rows = []
    kept = []
    seen = set()
    for report in reports:
        shard = report["shard"]
        metadata = [
            json.loads(s)
            for s in (args.output / f"{shard}.jsonl").read_text().splitlines()
        ]
        indices = []
        for i, row in enumerate(metadata):
            if row["sample_id"] in seen:
                continue
            seen.add(row["sample_id"])
            row["shard"] = shard
            row["shard_index"] = i
            rows.append(row)
            indices.append(i)
        kept.append(dict(shard=shard, indices=indices))
    splits = {
        name: {r["game_id"] for r in rows if r["split"] == name}
        for name in ("train", "val", "test")
    }
    assert not (
        splits["train"] & splits["val"]
        or splits["train"] & splits["test"]
        or splits["val"] & splits["test"]
    )
    metadata_path = args.output / "metadata.jsonl"
    metadata_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = dict(
        schema_version=SCHEMA,
        split_seed=42,
        decisions=args.decisions,
        history=8,
        skip_plies=12,
        max_plies=120,
        source_reports=reports,
        shards=kept,
        source_hashes={path.name: file_sha256(path) for path in paths},
        samples=len(rows),
        unique_games=len({r["game_id"] for r in rows}),
        split_samples=dict(Counter(r["split"] for r in rows)),
        split_unique_games={k: len(v) for k, v in splits.items()},
        missing_time_control=sum(r["time_control"] is None for r in rows),
        metadata_sha256=file_sha256(metadata_path),
        feature_names=reports[0]["feature_names"],
        feature_blocks=reports[0]["feature_blocks"],
    )
    spec["corpus_id"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(spec, indent=2))
    print(
        json.dumps(
            {
                k: spec[k]
                for k in ("samples", "unique_games", "split_samples", "corpus_id")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
