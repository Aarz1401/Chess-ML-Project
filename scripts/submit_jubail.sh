#!/bin/bash
# One-command launcher for Jubail (NYUAD HPC).
#
# First time only, on a login node from the repository root:
#     module load python/3.11.3        # or: module load python
#     python3 -m venv .venv-jubail
#     source .venv-jubail/bin/activate
#     pip install --upgrade pip
#     pip install -r requirements.txt
#     pip install umap-learn         # optional, nicer embedding maps
#
# Then every run is just:
#     bash scripts/submit_jubail.sh
#
# The script submits the CPU precompute job (skipped automatically if the
# tensor cache already exists) and the GPU training job chained behind it.
# Override defaults with environment variables:
#     CACHE_DIR=$SCRATCH/my_cache RUN_NAME=exp1 bash scripts/submit_jubail.sh

set -euo pipefail
cd "$(dirname "$0")/.."
: "${SCRATCH:?SCRATCH must be set}"
case "$(pwd -P)/" in
  "$(realpath "$SCRATCH")/"*) ;;
  *) echo "Submit from a scratch checkout"; exit 1 ;;
esac
mkdir -p logs

export CACHE_DIR="${CACHE_DIR:-$SCRATCH/chess_style/tensor_cache_full}"
export RUN_NAME="${RUN_NAME:-full8_$(date +%Y%m%d_%H%M)}"

if [ ! -d ".venv-jubail" ]; then
  echo "ERROR: .venv-jubail not found. Run the one-time setup in the header of this script first."
  exit 1
fi

if [ -f "${CACHE_DIR}/manifest.json" ]; then
  .venv-jubail/bin/python scripts/validate_tensor_cache.py --cache-dir "${CACHE_DIR}" --data-dir data/pgn
  echo "Validated tensor cache at ${CACHE_DIR} - skipping precompute."
  train_id=$(sbatch --parsable --export=ALL scripts/jubail_train.slurm)
  echo "Submitted training job ${train_id}."
else
  pre_id=$(sbatch --parsable --export=ALL scripts/jubail_precompute.slurm)
  echo "Submitted precompute job ${pre_id} (cache -> ${CACHE_DIR})."
  train_id=$(sbatch --parsable --export=ALL --dependency=afterok:${pre_id} scripts/jubail_train.slurm)
  echo "Submitted training job ${train_id} (starts after precompute succeeds)."
fi

echo
echo "Run name: ${RUN_NAME}"
echo "Watch progress:   squeue -u \$USER"
echo "Training log:     tail -f logs/train_${train_id}.out"
echo "Results land in:  results/${RUN_NAME}/"
echo "Key outputs:      aggregated_metrics.json, aggregated_eval/aggregation_curve.png"
