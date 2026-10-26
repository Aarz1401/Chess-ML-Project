"""Classical style-feature experiments for Model 2.

This file contains the non-neural classifier pipeline. It can build three
feature sets:

- engine_agreement: a baseline that checks Stockfish top-move agreement. This
  is useful as an ablation, but it is circular for human-vs-Stockfish runs.
- style_trajectory: the main feature set. Stockfish is only used to evaluate
  the board before/after the played move; no top-move identity or rank is used.
- combined: both vectors concatenated, for diagnostics only.
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset import (
    LABELS,
    GameRecord,
    _balance_records_by_label,
    _open_pgn,
    _split_records_by_label,
    collect_game_records,
)
from src.experiment_utils import active_labels, remap_record
from src.pipeline import set_seed

PIECE_VALUES = {
    # Simple material scale used for style features such as sacrifices and
    # material swings. Kings are assigned 0 because they cannot be captured.
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

QUALITY_FEATURE_DIM = 14
STYLE_FEATURE_DIM = 33


def safe_ratio(num: float, den: float) -> float:
    """Divide safely when a game has no selected positions."""

    return float(num / den) if den else 0.0


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def trend_slope(values: list[float]) -> float:
    """Return the least-squares slope of a sequence.

    In the final style vector this measures whether the continuation's
    evaluation generally improves or worsens over the analyzed moves.
    """

    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    y = np.array(values, dtype=np.float32)
    x = x - x.mean()
    den = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / den) if den else 0.0


def material_balance(board: chess.Board, color: bool) -> float:
    """Compute material balance from one side's point of view."""

    balance = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        balance += value if piece.color == color else -value
    return balance


def destination_centrality(square: chess.Square) -> float:
    """Score how central the destination square is, from 0=edge/corner to 1=center."""

    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return 1.0 - ((abs(file_idx - 3.5) + abs(rank_idx - 3.5)) / 7.0)


def feature_dimension(feature_set: str) -> int:
    if feature_set == "engine_agreement":
        return QUALITY_FEATURE_DIM
    if feature_set == "style_trajectory":
        return STYLE_FEATURE_DIM
    if feature_set == "combined":
        return QUALITY_FEATURE_DIM + STYLE_FEATURE_DIM
    raise ValueError(f"Unknown feature_set: {feature_set}")


def record_with_stockfish_side(record: GameRecord, stockfish_side: str) -> GameRecord:
    """Optionally evaluate only one Stockfish side, matching human-player records."""

    if record.label_name != "stockfish" or stockfish_side == "both":
        return record

    if stockfish_side == "white":
        target_color = chess.WHITE
    elif stockfish_side == "black":
        target_color = chess.BLACK
    elif stockfish_side == "alternate":
        target_color = chess.WHITE if record.game_index % 2 == 0 else chess.BLACK
    else:
        raise ValueError(f"Unknown stockfish_side: {stockfish_side}")

    return GameRecord(
        pgn_path=record.pgn_path,
        game_index=record.game_index,
        label=record.label,
        label_name=record.label_name,
        ply_count=record.ply_count,
        target_color=target_color,
    )


def selected_positions(game: chess.pgn.Game, target_color: bool | None, skip_moves: int, max_moves: int):
    """Return the move decisions analyzed for one labeled example.

    For human labels, target_color selects only the labeled player's moves.
    For Stockfish-controlled games, target_color may be None, meaning use all
    moves unless --stockfish-side forces a single side.
    """

    board = game.board()
    selected = []
    for move in game.mainline_moves():
        mover = board.turn
        before = board.copy(stack=False)
        board.push(move)
        if target_color is None or mover == target_color:
            selected.append((before, move, mover))
    return selected[skip_moves : skip_moves + max_moves]


def score_to_cp(score: chess.engine.PovScore) -> float:
    cp = score.score(mate_score=100000)
    return float(cp if cp is not None else 0.0)


def cached_eval_cp(
    board: chess.Board,
    perspective: bool,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    cache: dict,
) -> float:
    """Evaluate one board once and reuse the result across feature sets/splits."""

    key = (board.fen(), perspective)
    if key not in cache:
        try:
            info = engine.analyse(board, limit)
            cache[key] = score_to_cp(info["score"].pov(perspective))
        except Exception:
            cache[key] = 0.0
    return cache[key]


def quality_features_for_game(game, record, engine, limit, max_moves, skip_moves, multipv, cache):
    """Old engine-agreement baseline feature vector.

    This intentionally remains in the code as an ablation. It asks whether the
    played move was Stockfish's top move/top-3 move and how far it is from the
    top line. Because the engine class is generated by Stockfish, this should
    be described as an engine-likeness baseline, not the main style model.
    """

    positions = selected_positions(game, record.target_color, skip_moves=skip_moves, max_moves=max_moves)
    if not positions:
        return np.zeros(14, dtype=np.float32)

    top1 = top3 = 0
    ranks = []
    gaps = []
    best_scores = []
    played_scores = []
    legal_counts = []
    checks = captures = 0

    for board, played, mover in positions:
        fen = board.fen()
        if fen not in cache:
            try:
                # multipv asks Stockfish for its top K candidate lines. This is
                # exactly what makes this feature set more circular than the
                # final style_trajectory features.
                info = engine.analyse(board, limit, multipv=multipv)
                if isinstance(info, dict):
                    info = [info]
                moves_scores = []
                for item in info:
                    pv = item.get('pv') or []
                    if not pv:
                        continue
                    moves_scores.append((pv[0], score_to_cp(item['score'].pov(mover))))
                cache[fen] = moves_scores
            except Exception:
                cache[fen] = []
        moves_scores = cache[fen]
        moves = [m for m, _ in moves_scores]
        scores = [s for _, s in moves_scores]

        legal_counts.append(board.legal_moves.count())
        captures += int(board.is_capture(played))
        checks += int(board.gives_check(played))

        if moves and played == moves[0]:
            top1 += 1
        if played in moves[:3]:
            top3 += 1
            ranks.append(moves.index(played) + 1)
        else:
            ranks.append(multipv + 1)

        if scores:
            best_scores.append(scores[0])
            if played in moves:
                played_score = scores[moves.index(played)]
                played_scores.append(played_score)
                gaps.append(max(scores[0] - played_score, 0.0))
            else:
                # Unknown outside top-k. Use the top-k spread as a conservative proxy.
                played_scores.append(scores[-1])
                gaps.append(max(scores[0] - scores[-1], 0.0) + 50.0)

    n = len(positions)
    arr = np.array(
        [
            safe_ratio(top1, n),
            safe_ratio(top3, n),
            float(np.mean(ranks)),
            float(np.std(ranks)),
            float(np.mean(gaps)) / 300.0 if gaps else 0.0,
            float(np.std(gaps)) / 300.0 if gaps else 0.0,
            float(np.mean(best_scores)) / 1000.0 if best_scores else 0.0,
            float(np.std(best_scores)) / 1000.0 if best_scores else 0.0,
            float(np.mean(played_scores)) / 1000.0 if played_scores else 0.0,
            float(np.std(played_scores)) / 1000.0 if played_scores else 0.0,
            float(np.mean(legal_counts)) / 50.0,
            float(np.std(legal_counts)) / 50.0,
            safe_ratio(captures, n),
            safe_ratio(checks, n),
        ],
        dtype=np.float32,
    )
    return arr


def style_trajectory_features_for_game(game, record, engine, limit, max_moves, skip_moves, cache):
    """Summarize style-oriented game trajectory without engine top-move matches.

    Stockfish is used only as a board evaluator before and after the played
    move. The features intentionally avoid best-move identity, top-k rank, and
    centipawn gap to Stockfish's preferred move.
    """

    positions = selected_positions(game, record.target_color, skip_moves=skip_moves, max_moves=max_moves)
    if not positions:
        return np.zeros(STYLE_FEATURE_DIM, dtype=np.float32)

    before_evals = []
    after_evals = []
    eval_deltas = []
    legal_counts = []
    material_before = []
    material_deltas = []
    center_scores = []
    piece_counts = {piece_type: 0 for piece_type in PIECE_TYPES}
    captures = checks = castles = promotions = en_passants = 0
    attacked_destinations = 0
    pawn_advances = 0

    for before, played, mover in positions:
        after = before.copy(stack=False)
        moved_piece = before.piece_at(played.from_square)

        # Evaluate the position before the player/engine decision. This is not
        # asking Stockfish for its preferred move; it is only scoring the board.
        before_eval = cached_eval_cp(before, mover, engine, limit, cache)
        before_material = material_balance(before, mover)

        # Non-engine behavioral features. These describe what kind of move was
        # played: tactical, centralizing, risky, forcing, etc.
        legal_counts.append(before.legal_moves.count())
        center_scores.append(destination_centrality(played.to_square))
        captures += int(before.is_capture(played))
        checks += int(before.gives_check(played))
        castles += int(before.is_castling(played))
        en_passants += int(before.is_en_passant(played))
        promotions += int(played.promotion is not None)

        if moved_piece is not None:
            piece_counts[moved_piece.piece_type] += 1
            if moved_piece.piece_type == chess.PAWN:
                rank_delta = chess.square_rank(played.to_square) - chess.square_rank(played.from_square)
                pawn_advances += int((rank_delta > 0 and mover == chess.WHITE) or (rank_delta < 0 and mover == chess.BLACK))

        # Push the actual PGN move and evaluate the resulting board. The
        # before/after difference gives a trajectory signal over time.
        after.push(played)
        after_eval = cached_eval_cp(after, mover, engine, limit, cache)
        after_material = material_balance(after, mover)

        before_evals.append(before_eval)
        after_evals.append(after_eval)
        eval_deltas.append(after_eval - before_eval)
        material_before.append(before_material)
        material_deltas.append(after_material - before_material)
        attacked_destinations += int(after.is_attacked_by(not mover, played.to_square))

    n = len(positions)
    negative_swings = [delta for delta in eval_deltas if delta < -100.0]
    big_negative_swings = [delta for delta in eval_deltas if delta < -300.0]
    sacrifices = [delta for delta in material_deltas if delta < -1.0]
    piece_rates = [safe_ratio(piece_counts[piece_type], n) for piece_type in PIECE_TYPES]

    arr = np.array(
        [
            # Coverage feature: useful when a short game has fewer selected
            # moves than max_moves.
            safe_ratio(n, max_moves),
            # Evaluation trajectory features.
            safe_mean(before_evals) / 1000.0,
            safe_std(before_evals) / 1000.0,
            safe_mean(after_evals) / 1000.0,
            safe_std(after_evals) / 1000.0,
            safe_mean(eval_deltas) / 300.0,
            safe_std(eval_deltas) / 300.0,
            safe_mean([abs(delta) for delta in eval_deltas]) / 300.0,
            safe_ratio(len(negative_swings), n),
            safe_ratio(len(big_negative_swings), n),
            trend_slope(after_evals) / 300.0,
            # Position complexity.
            safe_mean(legal_counts) / 50.0,
            safe_std(legal_counts) / 50.0,
            safe_mean(material_before) / 39.0,
            safe_std(material_before) / 39.0,
            safe_mean(material_deltas) / 9.0,
            safe_std(material_deltas) / 9.0,
            safe_ratio(len(sacrifices), n),
            # Move-behavior rates.
            safe_ratio(captures, n),
            safe_ratio(checks, n),
            safe_ratio(castles, n),
            safe_ratio(promotions, n),
            safe_ratio(en_passants, n),
            safe_ratio(attacked_destinations, n),
            safe_mean(center_scores),
            safe_std(center_scores),
            safe_ratio(pawn_advances, n),
            *piece_rates,
        ],
        dtype=np.float32,
    )
    return arr


def features_for_game(game, record, engine, limit, max_moves, skip_moves, multipv, cache, feature_set):
    """Dispatch to the requested feature family."""

    if feature_set == "engine_agreement":
        return quality_features_for_game(game, record, engine, limit, max_moves, skip_moves, multipv, cache)
    if feature_set == "style_trajectory":
        return style_trajectory_features_for_game(game, record, engine, limit, max_moves, skip_moves, cache)
    if feature_set == "combined":
        return np.concatenate(
            [
                quality_features_for_game(game, record, engine, limit, max_moves, skip_moves, multipv, cache),
                style_trajectory_features_for_game(game, record, engine, limit, max_moves, skip_moves, cache),
            ]
        ).astype(np.float32)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def materialize_features(records, engine, limit, max_moves, skip_moves, multipv, stockfish_side, feature_set):
    """Re-open selected games and convert each game into one feature row."""

    by_path = {}
    for record in records:
        by_path.setdefault(record.pgn_path, {})[record.game_index] = record
    cache = {}
    x, y, names = [], [], []
    done = 0
    total = len(records)
    for pgn_path, selected in by_path.items():
        with _open_pgn(pgn_path) as pgn:
            game_index = 0
            while selected:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                record = selected.pop(game_index, None)
                if record is not None:
                    record = record_with_stockfish_side(record, stockfish_side)
                    x.append(features_for_game(game, record, engine, limit, max_moves, skip_moves, multipv, cache, feature_set))
                    y.append(record.label)
                    names.append(record.label_name)
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(f"  analyzed {done}/{total} games", flush=True)
                game_index += 1
    return np.stack(x), np.array(y), names


def materialize_quality_features(records, engine, limit, max_moves, skip_moves, multipv, stockfish_side):
    return materialize_features(
        records,
        engine,
        limit,
        max_moves,
        skip_moves,
        multipv,
        stockfish_side,
        feature_set="engine_agreement",
    )


def fit_models(x_train, y_train, x_val, y_val, x_test, y_test):
    """Train the classical classifier bank on one task.

    For the report, focus on SVM and random forest. Logistic regression, GBDT,
    MLP, and KNN are kept as diagnostics/ablations.
    """

    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, random_state=0)),
        'svm_linear': make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=0)),
        'svm_rbf': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=0)),
        'forest': RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3, class_weight='balanced', random_state=0),
        'gbdt': GradientBoostingClassifier(random_state=0),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1000, random_state=0)),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights='distance')),
    }
    out = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        out[f'{name}_train_acc'] = accuracy_score(y_train, train_pred)
        out[f'{name}_val_acc'] = accuracy_score(y_val, val_pred)
        out[f'{name}_test_acc'] = accuracy_score(y_test, test_pred)

        # Keep the old field name as an alias so older plotting code still works.
        out[f'{name}_acc'] = out[f'{name}_test_acc']
        if hasattr(model, 'predict_proba'):
            try:
                classes = model.classes_ if hasattr(model, 'classes_') else model[-1].classes_
                out[f'{name}_train_loss'] = log_loss(y_train, model.predict_proba(x_train), labels=classes)
                out[f'{name}_val_loss'] = log_loss(y_val, model.predict_proba(x_val), labels=classes)
                out[f'{name}_test_loss'] = log_loss(y_test, model.predict_proba(x_test), labels=classes)

                # Keep the old field name as an alias so older plotting code still works.
                out[f'{name}_loss'] = out[f'{name}_test_loss']
            except Exception:
                out[f'{name}_train_loss'] = np.nan
                out[f'{name}_val_loss'] = np.nan
                out[f'{name}_test_loss'] = np.nan
                out[f'{name}_loss'] = np.nan
    return out


def majority_baseline(y_test: np.ndarray) -> float:
    if len(y_test) == 0:
        return 0.0
    counts = Counter(y_test.tolist())
    return max(counts.values()) / len(y_test)


def fit_task_row(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    inv: dict[int, str],
    feature_set: str,
    task_type: str,
    label_ids: list[int],
):
    """Fit one multiclass task or one pairwise slice and format one CSV row."""

    result = fit_models(x_train, y_train, x_val, y_val, x_test, y_test)
    class_names = ",".join(inv[label_id] for label_id in label_ids)
    row = {
        'task_type': task_type,
        'classes': class_names,
        'player_a': inv[label_ids[0]] if len(label_ids) >= 1 else '',
        'player_b': inv[label_ids[1]] if len(label_ids) == 2 else '',
        'n_classes': len(label_ids),
        'majority_baseline': majority_baseline(y_test),
        'feature_set': feature_set,
        'feature_dim': feature_dimension(feature_set),
        **result,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description='Human-vs-engine style feature experiments')
    parser.add_argument('--data-dir', default='data/controlled_engine_raw')
    parser.add_argument('--max-games-per-label', type=int, default=100)
    parser.add_argument('--include-label', action='append', default=[])
    parser.add_argument('--exclude-label', action='append', default=[])
    parser.add_argument('--stockfish-pairs-only', action='store_true')
    parser.add_argument('--stockfish-side', choices=['both', 'white', 'black', 'alternate'], default='alternate')
    parser.add_argument('--skip-moves', type=int, default=0)
    parser.add_argument('--max-analyzed-moves', type=int, default=12)
    parser.add_argument('--multipv', type=int, default=3)
    parser.add_argument(
        '--feature-set',
        choices=['engine_agreement', 'style_trajectory', 'combined'],
        default='style_trajectory',
        help='engine_agreement uses Stockfish top-move matching; style_trajectory avoids top-move identity/rank features.',
    )
    parser.add_argument(
        '--classification-mode',
        choices=['pairwise', 'multiclass', 'both'],
        default='pairwise',
        help='pairwise runs every 2-class slice; multiclass trains one classifier over all included labels.',
    )
    parser.add_argument('--engine-path', default='stockfish')
    parser.add_argument('--engine-depth', type=int, default=6)
    parser.add_argument('--results-path', default='outputs/style_trajectory_results.csv')
    args = parser.parse_args()

    set_seed()
    random.seed(0)
    np.random.seed(0)

    if args.include_label:
        excluded = set(active_labels(LABELS, set())) - set(args.include_label)
        excluded.update(args.exclude_label)
    else:
        excluded = set(args.exclude_label)
    labels = active_labels(LABELS, excluded)
    inv = {v: k for k, v in labels.items()}

    records = collect_game_records(data_dir=args.data_dir, window_size=1, skip_moves=0, exclude_labels=excluded)
    records = [remap_record(record, labels) for record in records]

    # Balance at the game level before splitting, so accuracy is not inflated
    # by whichever class has more PGN examples.
    records = _balance_records_by_label(records, max_games_per_label=args.max_games_per_label)
    splits = _split_records_by_label(records)

    print('Game counts:')
    for split_name, split_records in splits.items():
        print(f"  {split_name}: {dict(sorted(Counter(r.label_name for r in split_records).items()))}")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    limit = chess.engine.Limit(depth=args.engine_depth)
    try:
        print('Analyzing train features...')
        x_train, y_train, _ = materialize_features(splits['train'], engine, limit, args.max_analyzed_moves, args.skip_moves, args.multipv, args.stockfish_side, args.feature_set)
        print('Analyzing validation features...')
        x_val, y_val, _ = materialize_features(splits['val'], engine, limit, args.max_analyzed_moves, args.skip_moves, args.multipv, args.stockfish_side, args.feature_set)
        print('Analyzing test features...')
        x_test, y_test, _ = materialize_features(splits['test'], engine, limit, args.max_analyzed_moves, args.skip_moves, args.multipv, args.stockfish_side, args.feature_set)
    finally:
        engine.quit()

    label_ids = sorted(set(y_train.tolist()) & set(y_val.tolist()) & set(y_test.tolist()))
    rows = []

    if args.classification_mode in {'multiclass', 'both'}:
        rows.append(
            fit_task_row(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                inv=inv,
                feature_set=args.feature_set,
                task_type='multiclass',
                label_ids=label_ids,
            )
        )

    if args.classification_mode in {'pairwise', 'both'}:
        for i, a in enumerate(label_ids):
            for b in label_ids[i + 1:]:
                if args.stockfish_pairs_only and 'stockfish' not in {inv[a], inv[b]}:
                    continue
                train_mask = np.isin(y_train, [a, b])
                val_mask = np.isin(y_val, [a, b])
                test_mask = np.isin(y_test, [a, b])
                rows.append(
                    fit_task_row(
                        x_train[train_mask],
                        y_train[train_mask],
                        x_val[val_mask],
                        y_val[val_mask],
                        x_test[test_mask],
                        y_test[test_mask],
                        inv=inv,
                        feature_set=args.feature_set,
                        task_type='pairwise',
                        label_ids=[a, b],
                    )
                )

    metric_keys = [k for k in rows[0] if k.endswith('_acc')] if rows else []
    rows.sort(key=lambda r: max(r[k] for k in metric_keys), reverse=True)
    print('\nStyle feature results:')
    for row in rows:
        print(row)

    if args.results_path:
        Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
        priority = ['task_type', 'classes', 'player_a', 'player_b', 'n_classes', 'majority_baseline', 'feature_set', 'feature_dim']
        fieldnames = priority + sorted(k for row in rows for k in row if k not in set(priority))
        with open(args.results_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved {args.results_path}')


if __name__ == '__main__':
    main()
