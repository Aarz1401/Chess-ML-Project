#!/bin/bash
#SBATCH --job-name=atlas-gpu-probe
#SBATCH --partition=nvidia
#SBATCH --account=students
#SBATCH --qos=nvidias
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/atlas_probe_%j.out
#SBATCH --error=logs/atlas_probe_%j.err

# Run from a checkout on scratch. Creates only logs; no training or downloads.
# Override account/QOS/GPU requests on sbatch's command line if necessary.
set -euo pipefail
: "${SCRATCH:?SCRATCH is unset}"
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the scratch checkout}"
case "$(pwd -P)/" in
  "$(realpath "$SCRATCH")/"*) ;;
  *) echo "Submit from a checkout under SCRATCH, as required by CRC." >&2; exit 1 ;;
esac
module purge
module load "${PYTHON_MODULE:-python/3.11.3}"
VENV_PATH="${VENV_PATH:-$PWD/.venv-jubail}"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Create the project environment first: $VENV_PATH" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$VENV_PATH/bin/python" - <<'PY'
import json
import platform
import torch

if not torch.cuda.is_available():
    raise RuntimeError("This environment does not expose a working CUDA-enabled PyTorch build")
device = torch.device("cuda")
report = dict(python=platform.python_version(), pytorch=torch.__version__,
              torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
              bf16_supported=torch.cuda.is_bf16_supported())
a = torch.randn(1024, 1024, device=device)
for _ in range(3):
    result = a @ a
assert bool(torch.isfinite(result).all())
torch.cuda.synchronize()
report['allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
report['status'] = 'CUDA smoke check passed; model throughput remains unmeasured'
print(json.dumps(report, indent=2))
PY
