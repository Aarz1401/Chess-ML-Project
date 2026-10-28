#!/bin/bash
# Reproduce the bounded 5M / 24-decision experiment in a prepared scratch checkout.
# Outputs are versioned by checkout directory. Existing completed runs are validated.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SCRATCH:?Run on Jubail}"
case "$(pwd -P)/" in "$(realpath "$SCRATCH")/"*) ;; *) echo "Run from scratch";exit 1;; esac
test -x .venv/bin/python
test -f models/manifest.json
test -d upstream-maia3/.git
mkdir -p runs logs
export VENV_PATH="$PWD/.venv"
dependency=()
if [[ -f runs/full_cache/manifest.json ]]; then
  .venv/bin/python scripts/validate_atlas_cache.py --cache runs/full_cache --corpus runs/full_corpus
else
  if [[ ! -f runs/pilot_cache/manifest.json ]]; then
    probe=$(sbatch --parsable --gres=gpu:1 scripts/jubail_gpu_probe.slurm)
    pilot=$(sbatch --parsable --time=00:30:00 --dependency="afterok:$probe" scripts/jubail_atlas_gpu.slurm bootstrap pilot)
    dependency=(--dependency="afterok:$pilot")
  fi
  full=$(sbatch --parsable --time=01:00:00 "${dependency[@]}" scripts/jubail_atlas_gpu.slurm bootstrap full)
  dependency=(--dependency="afterok:$full")
fi
pending=()
for specification in "joint 7" "joint 21" "joint 42" "concept 42"; do
  read -r objective seed <<< "$specification"
  if [[ -f "runs/${objective}_${seed}/report.json" ]]; then
    .venv/bin/python - "runs/${objective}_${seed}/report.json" <<'PYCODE'
import json,sys
cache=json.load(open("runs/full_cache/manifest.json"))
report=json.load(open(sys.argv[1]))
if report["cache_id"]!=cache["corpus_id"] or report["metadata_sha256"]!=cache["metadata_sha256"]:
    raise ValueError("Completed run belongs to a different cache")
PYCODE
    echo "Completed run retained: ${objective}_${seed}"
    continue
  fi
  id=$(sbatch --parsable --time=00:45:00 "${dependency[@]}" scripts/jubail_atlas_gpu.slurm train "$objective" "$seed")
  pending+=("$id")
  echo "Submitted $objective seed $seed: $id"
done
if [[ -f runs/player_42/report.json && -f runs/website_export/experiments.json && "${#pending[@]}" == 0 ]]; then
  echo "All outputs are already complete: runs/website_export/"
  exit 0
fi
if [[ "${#pending[@]}" != 0 ]]; then
  joined=$(IFS=:; printf "%s" "${pending[*]}")
  dependency=(--dependency="afterok:$joined")
fi
final=$(sbatch --parsable --time=00:45:00 --export=ALL,FINALIZE=1 "${dependency[@]}" scripts/jubail_atlas_gpu.slurm train player 42)
echo "Player control + final evaluation/export: $final"
