"""Build a deterministic, label-free feature atlas from the repository's PGNs.

Run with the root project's Python environment. Only chess, numpy and sklearn
are needed. Player labels are metadata, never features or PCA inputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import io
import random
import sys

import chess
import chess.pgn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
from game_vectors import strategic_features, BLOCK_WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["Game length", "Capture rate", "Check rate", "Pawn move rate", "Knight move rate", "Bishop move rate", "Rook move rate", "Queen move rate", "King move rate", "Castling frequency", "Final material", "Early queen activity"]


def describe(game, player):
    board = game.board()
    moves = list(game.mainline_moves())
    if len(moves) < 20 or game.errors:
        return None
    fens, sans, counts = [board.fen()], [], np.zeros(6)
    captures = checks = castles = early_queen = 0
    for i, move in enumerate(moves):
        piece = board.piece_type_at(move.from_square)
        counts[piece - 1] += 1
        captures += board.is_capture(move)
        checks += board.gives_check(move)
        castles += board.is_castling(move)
        early_queen += piece == chess.QUEEN and i < 40
        sans.append(board.san(move))
        board.push(move)
        fens.append(board.fen())
    n = len(moves)
    material = sum(len(board.pieces(p, c)) * v for p, v in [(1, 1), (2, 3), (3, 3), (4, 5), (5, 9)] for c in chess.COLORS)
    features = [n / 2, captures / n, checks / n, *(counts / n).tolist(), castles / 2, material / 78, early_queen / min(n, 40)]
    identity = game.board().fen() + " " + " ".join(m.uci() for m in moves)
    header = game.headers
    return dict(id=hashlib.sha256(identity.encode()).hexdigest()[:16], player=player,
                white=header.get("White", "Unknown"), black=header.get("Black", "Unknown"),
                date=header.get("Date", "?"), event=header.get("Event", "Unknown event"),
                result=header.get("Result", "*"), eco=header.get("ECO", "—"),
                features=features, fens=fens, moves=sans,
                pgn=game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)))


def build(data_dir, per_player, seed):
    rng = random.Random(seed)
    games, seen = [], set()
    for path in sorted(data_dir.glob("*.pgn")):
        with path.open(encoding="utf-8", errors="replace") as source:
            offsets = []
            while True:
                pos = source.tell()
                if chess.pgn.read_headers(source) is None:
                    break
                offsets.append(pos)
            rng.shuffle(offsets)
            count = 0
            for pos in offsets:
                source.seek(pos)
                game = chess.pgn.read_game(source)
                record = describe(game, path.stem) if game else None
                if not record or record['id'] in seen:
                    continue
                seen.add(record['id']); games.append(record); count += 1
                if count >= per_player:
                    break
        print(f"{path.stem}: {count} games")
    if len(games) < 3:
        raise ValueError("Need at least three valid games")
    matrix = np.asarray([g['features'] for g in games])
    vectors = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(vectors)
    strategic = []
    for record in games:
        parsed = chess.pgn.read_game(io.StringIO(record['pgn']))
        values, strategic_names, blocks = strategic_features(parsed, record['player'])
        strategic.append(values)
    scaler = StandardScaler()
    scaled = np.clip(scaler.fit_transform(strategic), -3, 3)
    weights = np.array([np.sqrt(BLOCK_WEIGHTS[group] / blocks.count(group)) for group in blocks])
    weighted = scaled * weights
    weighted /= np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-12)
    reduced = PCA(n_components=min(30, len(games)-1, weighted.shape[1]), svd_solver="full").fit_transform(weighted)
    strategic_map = TSNE(n_components=2, perplexity=min(25, max(2, (len(games)-1)/3)),
                         init="pca", learning_rate="auto", random_state=seed).fit_transform(reduced)
    def neighbor_retention(vectors, points, k=10):
        k = min(k, len(points)-1)
        def nearest(array):
            distances = ((array[:, None] - array[None, :])**2).sum(axis=-1)
            np.fill_diagonal(distances, np.inf)
            return np.argsort(distances, axis=1)[:, :k]
        high, low = nearest(vectors), nearest(points)
        return float(np.mean([len(set(a)&set(b))/k for a,b in zip(high,low)]))
    for i, game in enumerate(games):
        game['position'] = np.round(strategic_map[i], 5).tolist()
        game['vector'] = np.round(weighted[i], 7).tolist()
        game['baselinePosition'] = np.round(coordinates[i], 5).tolist()
        game['baselineVector'] = np.round(vectors[i], 6).tolist()
        game['features'] = np.round(game['features'], 6).tolist()
        game['strategicFeatures'] = np.round(strategic[i], 7).tolist()
    return dict(schemaVersion=2, method="Target-relative strategic fingerprint + t-SNE", seed=seed,
                featureNames=FEATURES, explainedVariance=pca.explained_variance_ratio_.tolist(),
                representation=dict(dimensions=len(strategic_names), featureNames=strategic_names,
                    blocks=blocks, blockWeights=BLOCK_WEIGHTS, skipPlies=12, maxPlies=120,
                    description="Three chronological segments of target-relative positional and decision features; standardized, clipped, block-weighted, L2-normalized.",
                    projection="PCA to 30 dimensions, then t-SNE (seeded)",
                    neighborRetention10=neighbor_retention(weighted,strategic_map),
                    baselineNeighborRetention10=neighbor_retention(vectors,coordinates),
                    trustworthiness10=float(trustworthiness(weighted,strategic_map,n_neighbors=min(10,(len(games)-1)//2)))),
                players=sorted({g['player'] for g in games}), games=games)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'pgn')
    parser.add_argument('--output', type=Path, default=ROOT / 'website' / 'dist' / 'atlas.json')
    parser.add_argument('--per-player', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.per_player < 1:
        parser.error('--per-player must be positive')
    payload = build(args.data_dir, args.per_player, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    print(f"Wrote {len(payload['games'])} games to {args.output}")
