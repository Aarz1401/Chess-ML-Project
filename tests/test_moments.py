import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moment_matching import facts, position_key, eligible, ranked, blind_trial

spec = importlib.util.spec_from_file_location(
    "judgments", ROOT / "scripts/evaluate_moment_judgments.py"
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


class MomentTests(unittest.TestCase):
    def test_preference_uncertainty_groups_by_query_instead_of_response_count(self):
        key = dict(version="fixture", trials={})
        records = []
        for i in range(5):
            key["trials"][str(i)] = dict(
                A="strategic" if i % 2 == 0 else "neural",
                B="neural" if i % 2 == 0 else "strategic",
                partition="audit",
                query_game=f"g{i}",
            )
            records.append(
                dict(
                    trialId=str(i),
                    choice="A",
                    partition="audit",
                    recordedAt="2026-10-30T00:00:00Z",
                )
            )
        first = dict(
            schemaVersion=1, version="fixture", browserId="first", responses=records
        )
        second = dict(
            schemaVersion=1,
            version="fixture",
            browserId="second",
            responses=[records[0]],
        )
        audit = scorer.evaluate(key, [first, second])["partitions"]["audit"]
        self.assertEqual(audit["responses"], 6)
        self.assertEqual(audit["query_games"], 5)
        self.assertAlmostEqual(audit["grouped_preference_difference"], 0.2)
        self.assertLessEqual(audit["ci95"][0], 0)
        self.assertGreater(audit["ci95"][1], 0)

    def test_isolated_passed_and_mirrored_facts(self):
        board = chess.Board("4k3/8/8/3P4/8/2P5/8/4K3 w - - 0 1")
        a = facts(board, chess.WHITE)
        self.assertEqual((a["isolated"], a["passed"], a["pawns"]), (0, 2, 2))
        self.assertEqual(a, facts(board.mirror(), chess.BLACK))
        board.remove_piece_at(chess.C3)
        self.assertEqual(facts(board, chess.WHITE)["isolated"], 1)

    def test_transposition_key_ignores_counters_and_mirrors(self):
        a = chess.Board()
        b = a.copy()
        b.halfmove_clock = 20
        b.fullmove_number = 10
        self.assertEqual(position_key(a), position_key(b))
        self.assertEqual(position_key(a), position_key(a.mirror()))
        a.push_uci("e2e4")
        self.assertNotEqual(position_key(a), position_key(b))

    def test_retrieval_rejects_same_game_transposition_and_same_opening(self):
        games = [
            dict(id="a", eco="A01"),
            dict(id="a", eco="B01"),
            dict(id="b", eco="A01"),
            dict(id="c", eco="B01"),
            dict(id="d", eco="C01"),
        ]
        moments = [
            dict(game=i, positionKey="same" if i in [0, 3] else str(i))
            for i in range(5)
        ]
        vectors = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]], dtype=float)
        self.assertEqual(ranked(0, range(5), moments, games, vectors), [4])
        games[4]["eco"] = ""
        self.assertFalse(eligible(moments[0], moments[4], games))

    def test_blinding_stable_and_contains_both_methods(self):
        trial, key = blind_trial(0, 1, 2, "example-game")
        self.assertEqual((trial, key), blind_trial(0, 1, 2, "example-game"))
        self.assertEqual({trial["A"], trial["B"]}, {1, 2})
        self.assertEqual(set(key.values()), {"strategic", "neural"})
        self.assertNotIn("strategic", trial)

    def test_scoring_deduplicates_exports_preserves_neither_and_rejects_versions(self):
        key = dict(
            version="fixture",
            trials={
                "t": dict(A="strategic", B="neural", partition="audit", query_game="g")
            },
        )
        export = dict(
            schemaVersion=1,
            version="fixture",
            browserId="test-reviewer",
            responses=[
                dict(
                    trialId="t",
                    choice="A",
                    partition="audit",
                    recordedAt="2026-10-30T00:00:00Z",
                )
            ],
        )
        later = copy.deepcopy(export)
        later["responses"][0].update(
            choice="neither", recordedAt="2026-10-30T01:00:00Z"
        )
        result = scorer.evaluate(key, [export, later, export])["partitions"]["audit"]
        self.assertEqual(result["responses"], 1)
        self.assertEqual(result["outcomes"]["neither"], 1)
        self.assertEqual(result["grouped_preference_difference"], 0)
        self.assertIsNone(result["ci95"])
        export["version"] = "wrong"
        with self.assertRaises(ValueError):
            scorer.evaluate(key, [export])


if __name__ == "__main__":
    unittest.main()
