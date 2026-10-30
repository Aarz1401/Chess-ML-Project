"""Check exported moments against legal replay, position facts and fitted vectors."""

import json
from pathlib import Path
import sys

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "website/scripts")]
from game_vectors import position_features
from moment_matching import facts, position_key, eligible


def validate(directory, transforms=None):
    data = json.loads((directory / "moments.json").read_text())
    review = json.loads((directory / "moment-review.json").read_text())
    report = json.loads((directory / "moment-report.json").read_text())
    assert data["schemaVersion"] == 1
    assert data["version"] == review["version"] == report["version"]
    games, moments = data["games"], data["moments"]
    assert len({m["id"] for m in moments}) == len(moments) == report["moments"]
    assert len({g["id"] for g in games}) == len(games) == report["games"]
    replay = {}
    raw = []
    for m in moments:
        g = games[m["game"]]
        if g["replayChunk"] not in replay:
            replay[g["replayChunk"]] = json.loads(
                (directory / g["replayChunk"]).read_text()
            )
        assert replay[g["replayChunk"]][g["id"]]["fens"][m["ply"]] == m["fen"]
        board = chess.Board(m["fen"])
        assert board.is_valid() and not board.is_game_over() and 24 <= m["ply"] <= 100
        assert facts(board, board.turn) == m["facts"]
        best = chess.Move.from_uci(m["engine"]["bestMove"])
        assert best in board.legal_moves and board.san(best) == m["engine"]["bestSan"]
        assert isinstance(m["engine"]["cp"], int) and m["engine"]["depth"] > 0
        raw.append(position_features(board, board.turn)[0])
    for key, dim in [("strategic", 89), ("neural", 64)]:
        array = np.asarray(data["vectors"][key])
        assert array.shape == (len(moments), dim) and np.isfinite(array).all()
        np.testing.assert_allclose(np.linalg.norm(array, axis=1), 1, atol=7e-5)
    if transforms is not None:
        transform = np.load(transforms)
        expected = (
            np.clip(
                (np.asarray(raw) - transform["strategic_mean"])
                / transform["strategic_scale"],
                -3,
                3,
            )
            * transform["strategic_weights"]
        )
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
        np.testing.assert_allclose(expected, data["vectors"]["strategic"], atol=1e-5)
    assert len(review["trials"]) == 100
    seen = set()
    for t in review["trials"]:
        q, a, b = (moments[t[k]] for k in ["query", "A", "B"])
        g = games[q["game"]]
        assert g["id"] not in seen and g["split"] == "test"
        seen.add(g["id"])
        assert t["A"] != t["B"]
        q = dict(q, positionKey=position_key(chess.Board(q["fen"])))
        for candidate in [a, b]:
            c = dict(candidate, positionKey=position_key(chess.Board(candidate["fen"])))
            assert eligible(q, c, games) and games[c["game"]]["split"] == "train"
        assert not any(name in t for name in ["strategic", "neural", "method"])
    assert sum(t["partition"] == "development" for t in review["trials"]) == 20
    assert sum(t["partition"] == "audit" for t in review["trials"]) == 80
    assert report["human_evaluation"]["responses"] == 0
    assert not (directory / "review-key.json").exists()
    print(
        f"PASS: {len(moments)} legal moments, facts, vectors, engine moves, 100 blinded cross-game trials and withheld method key"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT / "website/dist")
    p.add_argument("--transforms", type=Path)
    args = p.parse_args()
    validate(args.directory, args.transforms)
