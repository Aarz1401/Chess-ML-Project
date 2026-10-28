#!/bin/bash
#SBATCH --job-name=atlas-corpus
#SBATCH --partition=compute
#SBATCH --account=students
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/atlas_cpu_%j.out
#SBATCH --error=logs/atlas_cpu_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SCRATCH:?}"
case "$(pwd -P)/" in "$(realpath "$SCRATCH")/"*) ;; *) echo "Run from scratch"; exit 1;; esac
module purge
module load python/3.11.3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
stage="${1:-full}"
limit=0
if [[ "$stage" == pilot ]]; then limit=125; fi
.venv/bin/python -u scripts/prepare_atlas_corpus.py \
  --output "runs/${stage}_corpus" --workers "${SLURM_CPUS_PER_TASK}" \
  --max-per-file "$limit" --decisions 24
