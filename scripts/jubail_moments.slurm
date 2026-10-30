#!/bin/bash
#SBATCH --job-name=atlas-moments
#SBATCH --partition=compute
#SBATCH --account=students
#SBATCH --qos=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/moments_%j.out
#SBATCH --error=logs/moments_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SCRATCH:?}"
case "$(pwd -P)/" in "$(realpath "$SCRATCH")/"*) ;; *) echo "Run from scratch"; exit 1;; esac
module purge
module load python/3.11.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${ATLAS_CACHE:?Set the sealed October 28 cache directory}"
: "${ATLAS_PYTHON:?Set the prepared Python path}"
"$ATLAS_PYTHON" -u scripts/build_moments.py --cache "$ATLAS_CACHE" --atlas website/dist/atlas.json \
  --stockfish tools/stockfish/stockfish-ubuntu-x86-64-avx2 --output runs/moments_v1 \
  --workers "${SLURM_CPUS_PER_TASK}" --nodes 8000
