# Jubail workflow

For the implemented atlas pipeline, use the [current runbook](atlas-jubail-runbook.md)
and [measured results](atlas-results-2026-10-28.md). The older 84-plane workflow below
is retained as a separate comparison.

For the atlas's next learned representation, start with the
[representation roadmap and live readiness findings](representation-roadmap.md).
It includes a separate `scripts/jubail_gpu_probe.slurm` environment check and
explains which parts of the legacy launcher must change before large runs.

The existing neural workflow below predates that plan. Run it from the
repository root in a **scratch checkout**. Existing job scripts are in `scripts/`:

- `jubail_precompute.slurm`: CPU tensor generation.
- `jubail_train.slurm`: GPU training followed by aggregated evaluation.
- `submit_jubail.sh`: submit precompute, then training with an `afterok` dependency;
  reuse an existing cache when `manifest.json` exists.

The scripts were inherited from the project and have not been validated on a
live Jubail allocation in this cleanup. Confirm available modules and your job
resources on the cluster before submitting. In particular, the current training
script requests the `nvidia` partition and one GPU.

## Environment and submission

Use a fresh Linux environment on Jubail, not the local macOS `.venv`.
From the root of a checkout containing the reorganized files:

```bash
module avail python
# Load an available Python module, then:
python3 -m venv .venv-jubail
source .venv-jubail/bin/activate
python -m pip install -r requirements.txt
```

Ensure the installed PyTorch build supports the allocated CUDA GPU. Run a
small scheduled pilot before the full preset. Heavy preprocessing and training
belong on compute nodes, not login nodes.

Once the pilot and configuration checks pass:

```bash
CACHE_DIR="$SCRATCH/chess_style/tensor_cache_full" \
RUN_NAME="full8_baseline" bash scripts/submit_jubail.sh
```

Use a new run name for each experiment. Logs go to `logs/`, results to
`results/<RUN_NAME>/`, and the cache to the chosen scratch path. Results are
ignored by Git, so preserve selected reports and checkpoints separately.

The preset uses up to 4,000 games per player, 60 plies, 24 CPU workers for
preprocessing, and a 60-epoch GPU run. Its balanced training batch is
8 labels × 32 samples = 256 games; `--batch-size 128` governs evaluation,
not the balanced training batch. Measure GPU memory before using this preset.

Do not reuse a cache just because its manifest exists: check that its input
PGNs and preprocessing configuration match the intended experiment. A stronger
automated cache check is a priority in [the audit](project-audit.md).

## Official references

Checked 2026-10-26:

- [CRC introduction and compute-node policy](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html)
- [Jubail access and system](https://crc-docs.abudhabi.nyu.edu/hpc/system/index.html)
- [Job submission](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html)
- [Browser interface](https://crc-docs.abudhabi.nyu.edu/hpc/ood/index.html)
