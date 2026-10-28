import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "website/scripts")]
from game_identity import canonical_game_id, target_color, split_for_game, file_sha256
from board_dataset import (
    GameRecord,
    split_records,
    target_color_for_game,
    materialize_samples,
    TensorConfig,
)
from precompute_tensors import target_color_from_headers
from prepare_atlas_corpus import board_codes, compact_history
from atlas_model import AtlasEncoder, paired_contrastive
from tensor_cache import validate_cache


class IdentityTests(unittest.TestCase):
    def test_exact_player_selection_all_training_paths(self):
        game = chess.pgn.Game()
        game.headers["White"] = "Praggnanandhaa,R"
        game.headers["Black"] = "Anand,V"
        self.assertEqual(target_color(game.headers, "Anand"), chess.BLACK)
        self.assertEqual(target_color_for_game(game, "Anand"), chess.BLACK)
        self.assertEqual(target_color_from_headers(game.headers, "Anand"), chess.BLACK)
        game.headers["White"] = "Tseshkovsky,Vitaly"
        game.headers["Black"] = "Tal,M"
        self.assertEqual(target_color_for_game(game, "Tal"), chess.BLACK)

    def test_ambiguous_match_is_rejected(self):
        self.assertIsNone(
            target_color({"White": "Carlsen,M", "Black": "Carlsen,M"}, "Carlsen")
        )
        self.assertIsNone(target_color({"White": "Someone", "Black": "Other"}, "Tal"))

    def test_identity_independent_of_headers(self):
        game = chess.pgn.read_game(io.StringIO("1. e4 e5 2. Nf3 Nc6 *"))
        before = canonical_game_id(game)
        game.headers["White"] = "New name"
        self.assertEqual(before, canonical_game_id(game))

    def test_both_perspectives_share_split(self):
        record = GameRecord(Path("fake.pgn"), 0, "Tal", 0, True, 40, "7f1308c11575b04e")
        other = replace(record, label_name="Carlsen", label_id=1, target_color=False)
        splits = split_records([record, other], 42)
        self.assertEqual(len(splits[split_for_game(record.game_id)]), 2)
        with self.assertRaises(ValueError):
            split_for_game("")

    def test_materialized_metadata_remains_aligned_after_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Tal.pgn"
            fixtures = []
            for first, reply in [("e2e4", "e7e5"), ("d2d4", "d7d5")]:
                game = chess.pgn.Game()
                game.headers["White"] = "Tal,M"
                game.headers["Black"] = "Carlsen,M"
                node = game.add_variation(chess.Move.from_uci(first))
                node.add_variation(chess.Move.from_uci(reply))
                fixtures.append(str(game))
            path.write_text("\n\n".join(fixtures) + "\n")
            records = [GameRecord(path, i, "Tal", 0, True, 2, str(i)) for i in range(2)]

            def encode(game, *args, **kwargs):
                square = next(iter(game.mainline_moves())).to_square
                return torch.full((1, 1, 1, 1), float(square)), torch.tensor([False])

            kept = []
            with patch("board_dataset.game_to_model_tensors", side_effect=encode):
                samples = materialize_samples(
                    records, TensorConfig(), metadata_records=kept
                )
            self.assertEqual(len(samples), 2)
            for sample, record in zip(samples, kept):
                self.assertEqual(
                    int(sample[0].flatten()[0]), [chess.E4, chess.D4][record.game_index]
                )

    def test_board_encoding_mirrors_color_and_ranks(self):
        board = chess.Board()
        board.push_uci("e2e4")
        np.testing.assert_array_equal(board_codes(board), board_codes(board.mirror()))
        self.assertTrue(np.all(board_codes(board) <= 12))


class ModelTests(unittest.TestCase):
    def test_padding_cannot_change_embedding(self):
        torch.manual_seed(1)
        model = AtlasEncoder(
            input_dim=12, width=16, embedding_dim=8, target_dim=4, layers=1, dropout=0
        ).eval()
        sequence = torch.randn(2, 8, 12)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 6:] = True
        plies = torch.arange(8).expand(2, -1)
        with torch.no_grad():
            a, _ = model(sequence, mask, plies)
            sequence[:, 6:] = 10000
            b, _ = model(sequence, mask, plies)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a.norm(dim=1), torch.ones(2))
        with self.assertRaises(ValueError):
            model(sequence, torch.ones_like(mask), plies)

    def test_contrastive_excludes_alternate_perspectives(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        loss = paired_contrastive(a, a, torch.tensor([1, 1]))
        self.assertAlmostEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())


class CacheTests(unittest.TestCase):
    def test_cache_rejects_cross_split_duplicate_and_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Tal.pgn").write_text("source")
            manifest = dict(
                schema_version=2,
                grouped_split=True,
                source_hashes={"Tal.pgn": file_sha256(root / "Tal.pgn")},
                counts={s: 1 for s in ["train", "val", "test"]},
                sequence_length=2,
                input_channels=3,
                labels=["Tal"],
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            for split in ["train", "val", "test"]:
                np.save(
                    root / f"{split}_sequences.npy",
                    np.zeros((1, 2, 3, 8, 8), np.float16),
                )
                np.save(root / f"{split}_masks.npy", np.zeros((1, 2), bool))
                np.save(root / f"{split}_labels.npy", np.zeros(1, np.int64))
                with (root / f"{split}_metadata.csv").open("w") as source:
                    writer = csv.writer(source)
                    writer.writerow(["game_id", "target_color"])
                    writer.writerow([split, "white"])
            validate_cache(root, root)
            (root / "Tal.pgn").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                validate_cache(root, root)
            (root / "Tal.pgn").write_text("source")
            (root / "test_metadata.csv").write_text(
                "game_id,target_color\ntrain,black\n"
            )
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                validate_cache(root, root)


if __name__ == "__main__":
    unittest.main()
