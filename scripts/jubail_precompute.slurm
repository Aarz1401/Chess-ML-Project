#!/bin/bash
#SBATCH --job-name=chess-precompute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/precompute_%j.out
#SBATCH --error=logs/precompute_%j.err

# CPU-only job: builds the tensor cache once so GPU jobs never touch PGN
# parsing. Submit from the repository root (submit_jubail.sh
# does this for you).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge || true
# Jubail: `module avail python` to see exact names; adjust if needed.
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true

if [ -d ".venv-jubail" ]; then
  source .venv-jubail/bin/activate
fi

CACHE_DIR="${CACHE_DIR:-$SCRATCH/chess_style/tensor_cache_full}"

python3 scripts/precompute_tensors.py \
  --cache-dir "${CACHE_DIR}" \
  --max-games-per-player 4000 \
  --sequence-length 60 \
  --skip-plies 12 \
  --min-valid-plies 20 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --seed 42

echo "Tensor cache ready at ${CACHE_DIR}"
