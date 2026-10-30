"""Position facts and deterministic cross-game retrieval, independent of the UI."""

import hashlib
import numpy as np
import chess


def position_key(board):
    """Merge exact transpositions and color/rank mirrored equivalents."""
    relative = board if board.turn else board.mirror()
    return " ".join(relative.fen().split()[:4])


def facts(board, color):
    own = board.pieces(chess.PAWN, color)
    other = board.pieces(chess.PAWN, not color)
    files = [sum(chess.square_file(s) == f for s in own) for f in range(8)]
    rank = lambda s: chess.square_rank(s) if color else 7 - chess.square_rank(s)
    isolated = sum(
        not any(
            files[t]
            for t in (chess.square_file(s) - 1, chess.square_file(s) + 1)
            if 0 <= t < 8
        )
        for s in own
    )
    passed = sum(
        not any(
            abs(chess.square_file(s) - chess.square_file(t)) <= 1 and rank(t) > rank(s)
            for t in other
        )
        for s in own
    )
    king = board.king(color)
    enemy_king = board.king(not color)
    zone = set(board.attacks(king)) | {king}
    attacked = sum(board.is_attacked_by(not color, s) for s in zone)
    shield = sum(
        chess.square_distance(king, s) <= 1 and rank(s) > rank(king) for s in own
    )
    balance = sum(
        (len(board.pieces(p, color)) - len(board.pieces(p, not color))) * v
        for p, v in [(1, 1), (2, 3), (3, 3), (4, 5), (5, 9)]
    )
    king_wing = lambda s: (
        "queenside"
        if chess.square_file(s) <= 2
        else "kingside" if chess.square_file(s) >= 5 else "central"
    )
    return dict(
        isolated=isolated,
        passed=passed,
        doubled=sum(max(0, n - 1) for n in files),
        pawns=len(own),
        shield=shield,
        kingAttacked=attacked,
        kingZone=len(zone),
        kingWing=king_wing(king),
        enemyKingWing=king_wing(enemy_king),
        material=balance,
        queens=sum(len(board.pieces(chess.QUEEN, c)) for c in chess.COLORS),
        center=sum(
            board.is_attacked_by(color, s)
            for s in (chess.D4, chess.E4, chess.D5, chess.E5)
        ),
    )


def eligible(query, candidate, games, different_eco=True):
    a, b = games[query["game"]], games[candidate["game"]]
    return (
        a["id"] != b["id"]
        and query["positionKey"] != candidate["positionKey"]
        and (not different_eco or bool(a["eco"] and b["eco"] and a["eco"] != b["eco"]))
    )


def ranked(query_index, indices, moments, games, vectors, different_eco=True):
    query = moments[query_index]
    keep = [i for i in indices if eligible(query, moments[i], games, different_eco)]
    if not keep:
        return []
    delta = vectors[keep] - vectors[query_index]
    distance = np.einsum("ij,ij->i", delta, delta)
    return [keep[i] for i in np.argsort(distance, kind="stable")]


def blind_trial(query, strategic, neural, query_game_id, seed=20261030):
    digest = hashlib.sha256(f"{seed}:{query_game_id}".encode()).hexdigest()
    flip = int(digest[:8], 16) % 2
    candidates = [strategic, neural] if not flip else [neural, strategic]
    key = dict(
        A="strategic" if not flip else "neural", B="neural" if not flip else "strategic"
    )
    return dict(id=digest[:16], query=query, A=candidates[0], B=candidates[1]), key
