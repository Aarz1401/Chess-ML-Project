#!/bin/bash
#SBATCH --job-name=chess-style-multiclass
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/multiclass_%j.out
#SBATCH --error=logs/multiclass_%j.err

set -euo pipefail

# Submit from the project root:
#   sbatch scripts/run_multiclass_hpc.slurm
#
# If your HPC uses different module names, edit only the environment block
# below. The training command at the bottom is intentionally plain Python.

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

module purge || true
module load python || true
module load cuda || true

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python3 scripts/train_multiclass.py \
  --max-games-per-player 500 \
  --sequence-length 80 \
  --stockfish-depth 2 \
  --multipv 3 \
  --epochs 10 \
  --batch-size 16 \
  --output-dir results/multiclass_proof
