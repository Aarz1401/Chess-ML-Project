"""Reproducible Model 1 Tal-vs-Carlsen human-player ablation.

This wrapper keeps the exact command for a two-human Model 1 run in one place.
It uses the same hybrid neural architecture as the final Model 1 experiment,
but reads from data/raw and classifies Tal vs Carlsen instead of human vs
Stockfish.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULTS = OUT / "model1_pair_tal_carlsen_small_d2_m30_g200_e8.csv"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/run_hybrid_experiment.py",
        "--data-dir",
        "data/raw",
        "--include-label",
        "tal",
        "--include-label",
        "carlsen",
        "--max-games-per-label",
        "200",
        "--sequence-mode",
        "padded",
        "--window-length",
        "80",
        "--skip-moves",
        "0",
        "--max-analyzed-moves",
        "30",
        "--engine-depth",
        "2",
        "--feature-set",
        "style_trajectory",
        "--epochs",
        "8",
        "--batch-size",
        "8",
        "--model-scale",
        "small",
        "--lr",
        "3e-4",
        "--results-path",
        str(RESULTS),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
