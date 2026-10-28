#!/bin/bash
#SBATCH --job-name=atlas-neural
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/atlas_gpu_%j.out
#SBATCH --error=logs/atlas_gpu_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SCRATCH:?}"
case "$(pwd -P)/" in "$(realpath "$SCRATCH")/"*) ;; *) echo "Run from scratch"; exit 1;; esac
module purge
module load python/3.11.3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/runs/matplotlib" HF_HUB_OFFLINE=1
nvidia-smi
stage="${1:?extract or train}"
case "$stage" in
  bootstrap)
    corpus="${2:-pilot}"
    limit=0
    if [[ "$corpus" == pilot ]]; then limit=125; fi
    if [[ ! -f "runs/${corpus}_corpus/manifest.json" ]]; then
      .venv/bin/python -u scripts/prepare_atlas_corpus.py --output "runs/${corpus}_corpus" --workers "${SLURM_CPUS_PER_TASK}" --max-per-file "$limit" --decisions 24
    fi
    bash scripts/jubail_atlas_gpu.slurm extract "$corpus"
    ;;
  extract)
    corpus="${2:-full}"
    .venv/bin/python -u scripts/extract_atlas_embeddings.py \
      --corpus "runs/${corpus}_corpus" --output "runs/${corpus}_cache" \
      --maia-source upstream-maia3 --checkpoint models/maia3-5m.pt \
      --model-manifest models/manifest.json --batch-positions 512
    ;;
  train)
    objective="${2:-joint}"
    seed="${3:-42}"
    .venv/bin/python -u scripts/train_atlas_encoder.py \
      --cache runs/full_cache --output "runs/${objective}_${seed}" \
      --seed "$seed" --objective "$objective" --epochs 12 --batch-size 128 --resume
    if [[ "${FINALIZE:-0}" == 1 ]]; then
      seal=()
      if [[ ! -f runs/full_cache/integrity.json ]]; then seal=(--seal); fi
      .venv/bin/python scripts/validate_atlas_cache.py --cache runs/full_cache --corpus runs/full_corpus "${seal[@]}"
      .venv/bin/python -u scripts/evaluate_atlas_experiments.py --cache runs/full_cache --runs runs --output runs/experiment_results.json
      .venv/bin/python -u scripts/export_atlas_website.py --cache runs/full_cache --runs runs --results runs/experiment_results.json --output runs/website_export --previous website/dist/atlas.json --per-player 125
    fi
    ;;
  *) echo "Unknown stage: $stage";exit 1;;
esac
