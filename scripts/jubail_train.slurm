#!/bin/bash
#SBATCH --job-name=chess-style-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# GPU job: trains the style encoder from the precomputed tensor cache, then
# runs the aggregated player-level evaluation and figures.
#
# If your allocation has A100s behind a constraint, request one explicitly:
#   #SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${SCRATCH:?SCRATCH must be set}"
case "$(pwd -P)/" in
  "$(realpath "$SCRATCH")/"*) ;;
  *) echo "Submit from a scratch checkout"; exit 1 ;;
esac
mkdir -p logs results

module purge || true
module load "${PYTHON_MODULE:-python/3.11.3}"
module load cuda 2>/dev/null || true

if [ -d ".venv-jubail" ]; then
  source .venv-jubail/bin/activate
fi

CACHE_DIR="${CACHE_DIR:-$SCRATCH/chess_style/tensor_cache_full}"
RUN_NAME="${RUN_NAME:-full8_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="results/${RUN_NAME}"

nvidia-smi || true

python3 scripts/train_multiclass.py \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 60 \
  --warmup-epochs 4 \
  --lr 3e-4 \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --label-smoothing 0.05 \
  --batch-size 128 \
  --labels-per-batch 8 \
  --samples-per-label 32 \
  --classification-weight 0.25 \
  --contrastive-weight 1.0 \
  --contrastive-temperature 0.07 \
  --conv-channels 64 \
  --n-blocks 6 \
  --move-dim 256 \
  --transformer-dim 256 \
  --n-layers 4 \
  --n-heads 8 \
  --feedforward-dim 1024 \
  --metric-dim 128 \
  --pooling attention \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --eval-reference-cap 250 \
  --agg-ks 1 3 5 10 25 \
  --agg-trials 200 \
  --seed 42

python3 scripts/evaluate_aggregated.py \
  --checkpoint "${OUTPUT_DIR}/best_embedding_model.pt" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}/aggregated_eval" \
  --ks 1 3 5 10 25 \
  --trials 200 \
  --num-workers "${SLURM_CPUS_PER_TASK}"

echo "Training + evaluation complete. Results in ${OUTPUT_DIR}"
