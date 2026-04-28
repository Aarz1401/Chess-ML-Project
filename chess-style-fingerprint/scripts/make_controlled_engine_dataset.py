"""Build controlled human-vs-engine continuation datasets.

For each human game, this script chooses a midgame FEN after the labeled player
has made a fixed number of moves. It then saves two continuations from that
same position:

1. The real human continuation from the original PGN.
2. A Stockfish-generated continuation from the same FEN.

The same-FEN pairing controls the starting position distribution.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.engine
import chess.pgn

from src.dataset import LABELS, _open_pgn, _target_color_for_game


def iter_games(path: Path):
    """Yield every game from one PGN file."""

    with _open_pgn(str(path)) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def split_at_target_move(game: chess.pgn.Game, target_color: bool, target_moves: int):
    """Return the midgame board and the original continuation after target_moves.

    target_color is the labeled human player's side. For example, if the folder
    is Tal and Tal played black, we count only Black moves before selecting the
    starting FEN.
    """

    board = game.board()
    kept_moves = []
    selected = 0
    for move in game.mainline_moves():
        mover = board.turn
        board.push(move)
        if mover == target_color:
            selected += 1
        if selected >= target_moves:
            break
    else:
        return None, []

    start_board = board.copy(stack=False)

    # Replay the game a second time and collect the continuation after the
    # chosen starting FEN. This preserves the exact human moves after the split.
    board = game.board()
    selected = 0
    collect = False
    for move in game.mainline_moves():
        mover = board.turn
        if collect:
            kept_moves.append(move)
        board.push(move)
        if mover == target_color:
            selected += 1
            if selected >= target_moves:
                collect = True

    return start_board, kept_moves


def continuation_game_from_moves(source: chess.pgn.Game, board: chess.Board, moves: list[chess.Move], event: str) -> chess.pgn.Game | None:
    """Create a PGN continuation that starts from board and follows moves."""

    game = chess.pgn.Game.from_board(board)
    for key in ["White", "Black", "Date", "Round", "Site", "Result"]:
        if key in source.headers:
            game.headers[key] = source.headers[key]
    game.headers["Event"] = event
    game.headers["Result"] = "*"

    node = game
    replay = board.copy(stack=False)
    for move in moves:
        if move not in replay.legal_moves:
            return None
        node = node.add_variation(move)
        replay.push(move)
    return game


def engine_continuation(engine, board: chess.Board, plies: int, limit_time: float, skill: int) -> chess.pgn.Game:
    """Ask Stockfish to play a continuation from the same starting board."""

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Controlled Stockfish continuation"
    game.headers["White"] = "Stockfish"
    game.headers["Black"] = "Stockfish"
    game.headers["Result"] = "*"

    try:
        # Skill Level is supported by Stockfish but not guaranteed for every
        # UCI engine, so ignore it when unavailable.
        engine.configure({"Skill Level": skill})
    except chess.engine.EngineError:
        pass

    node = game
    replay = board.copy(stack=False)
    for _ in range(plies):
        if replay.is_game_over():
            break
        result = engine.play(replay, chess.engine.Limit(time=limit_time))
        node = node.add_variation(result.move)
        replay.push(result.move)
    game.headers["Result"] = replay.result() if replay.is_game_over() else "*"
    return game


def write_game(handle, game: chess.pgn.Game) -> None:
    print(game, file=handle, end="\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paired human/Stockfish continuations from the same human midgame FENs")
    parser.add_argument("--source-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/controlled_engine_raw")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Human labels to convert. Defaults to every non-Stockfish label known to the project.",
    )
    parser.add_argument("--games-per-label", type=int, default=60)
    parser.add_argument("--target-moves-before-fen", type=int, default=10)
    parser.add_argument("--continuation-plies", type=int, default=50)
    parser.add_argument("--engine-path", default="stockfish")
    parser.add_argument("--engine-time", type=float, default=0.03)
    parser.add_argument("--engine-skill", type=int, default=20)
    args = parser.parse_args()
    if args.labels is None:
        args.labels = [label for label in LABELS if label != "stockfish"]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stockfish").mkdir(parents=True, exist_ok=True)

    engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
    stockfish_path = out / "stockfish" / "controlled_stockfish.pgn"
    stockfish_written = 0

    with open(stockfish_path, "w", encoding="utf-8") as sf:
        for label in args.labels:
            if label not in LABELS:
                raise ValueError(f"Unknown label: {label}")
            label_dir = Path(args.source_dir) / label
            out_label_dir = out / label
            out_label_dir.mkdir(parents=True, exist_ok=True)
            human_path = out_label_dir / f"{label}_controlled.pgn"
            written = 0
            with open(human_path, "w", encoding="utf-8") as hf:
                for pgn_path in sorted(label_dir.glob("*.pgn")):
                    for game in iter_games(pgn_path):
                        target_color = _target_color_for_game(game, label)
                        if target_color is None:
                            continue
                        board, moves = split_at_target_move(game, target_color, args.target_moves_before_fen)
                        if board is None or len(moves) < args.continuation_plies // 3:
                            continue

                        # Human and engine examples share this exact board.
                        # This is what prevents opening/position distribution
                        # from becoming an easy shortcut.
                        human_game = continuation_game_from_moves(
                            game,
                            board,
                            moves[: args.continuation_plies],
                            event=f"Controlled human continuation: {label}",
                        )
                        if human_game is None:
                            continue
                        engine_game = engine_continuation(
                            engine,
                            board,
                            plies=min(args.continuation_plies, len(moves)),
                            limit_time=args.engine_time,
                            skill=args.engine_skill,
                        )
                        write_game(hf, human_game)
                        write_game(sf, engine_game)
                        written += 1
                        stockfish_written += 1
                        if written % 10 == 0:
                            print(f"{label}: wrote {written}/{args.games_per_label}", flush=True)
                        if written >= args.games_per_label:
                            break
                    if written >= args.games_per_label:
                        break
            print(f"{label}: final written {written}")

    engine.quit()
    print(f"Stockfish continuations written: {stockfish_written}")
    print(f"Output directory: {out}")


if __name__ == "__main__":
    main()
