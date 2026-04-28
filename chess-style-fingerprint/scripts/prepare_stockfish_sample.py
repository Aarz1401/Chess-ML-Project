"""Prepare a raw Stockfish PGN sample.

This is not the final controlled dataset builder. It is only useful when you
want a folder of generic Stockfish games under data/raw/stockfish.
"""

import argparse
import gzip
from pathlib import Path
from typing import TextIO

import chess.pgn


def open_pgn(path: Path) -> TextIO:
    """Read either plain PGN or gzipped PGN from the Fishtest archive."""

    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def download_fishtest_subset(pattern: str, output_dir: Path) -> None:
    """Download only files matching a Hugging Face allow pattern."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    snapshot_download(
        # Fishtest is Stockfish's public distributed testing PGN archive.
        repo_id="official-stockfish/fishtest_pgns",
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(output_dir),
    )


def sample_games(input_dir: Path, output_pgn: Path, max_games: int) -> int:
    """Combine the first max_games parsed games into one Stockfish PGN file."""

    pgn_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.name.lower().endswith(".pgn") or path.name.lower().endswith(".pgn.gz")
    )
    if not pgn_paths:
        raise SystemExit(f"No PGN files found under {input_dir}")

    output_pgn.parent.mkdir(parents=True, exist_ok=True)
    games_written = 0

    with open(output_pgn, "w", encoding="utf-8") as out_file:
        for pgn_path in pgn_paths:
            with open_pgn(pgn_path) as in_file:
                while games_written < max_games:
                    game = chess.pgn.read_game(in_file)
                    if game is None:
                        break

                    # Normalize headers so the dataset loader treats this file
                    # as Stockfish-vs-Stockfish data.
                    game.headers["Event"] = game.headers.get("Event", "Stockfish Fishtest")
                    game.headers["White"] = "Stockfish"
                    game.headers["Black"] = "Stockfish"
                    print(game, file=out_file, end="\n\n")
                    games_written += 1

            if games_written >= max_games:
                break

    return games_written


def main() -> None:
    """CLI entry point used after downloading or while downloading a subset."""

    parser = argparse.ArgumentParser(description="Prepare a balanced-size Stockfish PGN sample")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("stockfish_fishtest_raw"),
        help="Directory containing downloaded .pgn or .pgn.gz files",
    )
    parser.add_argument(
        "--output-pgn",
        type=Path,
        default=Path("data/raw/stockfish/stockfish_sample.pgn"),
        help="Combined PGN to write for the stockfish class",
    )
    parser.add_argument("--max-games", type=int, default=1000, help="Maximum games to write")
    parser.add_argument(
        "--download-pattern",
        type=str,
        default="",
        help='Optional Hugging Face allow pattern, for example "19-12-*/*/*.pgn.gz"',
    )
    args = parser.parse_args()

    if args.download_pattern:
        download_fishtest_subset(args.download_pattern, args.input_dir)

    games_written = sample_games(args.input_dir, args.output_pgn, args.max_games)
    print(f"Wrote {games_written} games to {args.output_pgn}")


if __name__ == "__main__":
    main()
