# Reproduce the atlas experiments on Jubail

The October 28 run completed: a 1,000-game pilot, extraction for 36,234 target
perspectives of 34,740 unique games, four temporal/concept experiments, and a
player-supervised control. Read [the measured results](atlas-results-2026-10-28.md)
before choosing a follow-up objective.

## Fresh workspace

Use a new scratch directory for different input data or model configurations.
The scripts never overwrite completed corpus or extraction manifests.

~~~bash
ssh jubail
module load python/3.11.3
git clone https://github.com/Aarz1401/Chess-ML-Project.git "$SCRATCH/Chess-ML-Project"
cd "$SCRATCH/Chess-ML-Project"

python3 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements-jubail.txt
.venv/bin/python -m pip freeze > environment-freeze.txt

git clone https://github.com/CSSLab/maia3.git upstream-maia3
git -C upstream-maia3 checkout 1e13597c42d4858b7cfd7cfdae01e297263364b2
.venv/bin/python scripts/download_atlas_model.py
bash scripts/submit_atlas_jubail.sh
~~~

The downloaded backbone and source revisions are fixed and the checkpoint SHA
is verified. Maia-3 remains a separately licensed third-party dependency; its
source and weights are not copied into the website.

The submitted jobs request the verified students/nvidia association and one GPU
per neural run. Account, QOS, GPU type and walltime can be overridden at sbatch.
Modules and allocation limits may change; verify them on a new account.

## Stages

1. CUDA readiness check.
2. Pilot corpus and frozen extraction, bounded to 125 games per collection.
3. Full corpus preparation, exact game-ID grouping, and frozen extraction.
4. Joint concept/paired-view objectives at seeds 7, 21, 42; concept-only seed 42.
5. Player-supervised seed 42, final grouped audit, and website export.

The bounded bootstrap includes CPU preparation in the GPU job. This was used
after CPU-only jobs remained queued; for routine large runs, use
jubail_atlas_cpu.slurm for preprocessing before the GPU extraction stage to
avoid reserving an idle GPU during parsing.

The final stage seals the core cache files with hashes. Subsequent reuse
validates source PGNs, shapes, canonical groups, metadata, and the seal.
Completed model runs are checked against their cache IDs. An interrupted
trainer can resume from last.pt with optimizer, scheduler, scaler and RNG state.

## Outputs and monitoring

~~~bash
squeue -u "$USER"
tail -f logs/atlas_gpu_<job-id>.out
~~~

- runs/full_corpus/: compact uint8 board histories, actions and provenance.
- runs/full_cache/: frozen decision sequences, train-only transforms and vectors.
- runs/joint_*/ and runs/concept_42/: checkpoints, curves and vectors.
- runs/player_42/: explicit identity-supervised control.
- runs/experiment_results.json: fresh-audit results and paired uncertainty.
- runs/website_export/: map data, measured results and lazy replay chunks.

Player labels are supervision only in the player control. The other learned
game encoders use paired views and/or engineered chess-concept targets. Those
targets do not provide independent human validation of game similarity.

## Website export and local checks

Copy the exported files to website/dist, then:

~~~bash
python3 -m http.server 8765 --directory website/dist

# Separate terminal, root Python environment:
.venv/bin/python website/scripts/validate_atlas.py
.venv/bin/python -m unittest discover -s tests -v
npm ci --prefix website --ignore-scripts
npm test --prefix website
~~~

Replay data is split into chunks of 25 games and fetched on selection. The
browser map uses 1,000 sampled games; the benchmark uses separate training
reference and held-out query pools from the full corpus. Do not interpret
browser-neighbor precision as the published benchmark.

The website is static and uses exported arrays. It does not expose a GPU
service, connect to Jubail, or upload visitors' games. Native visual browser QA
was unavailable in the implementation session; DOM interaction checks and
legal-replay checks were run.

Keep checkpoints, transforms, reports and the package freeze in a backed-up
archive. The October 28 copy is in local results/atlas_jubail and a versioned
Jubail scratch workspace. Full sequence caches stay on scratch and can be
rebuilt. Scratch is temporary storage, not a durable backup.

## Primary references

- [CRC job submission](https://crc-docs.abudhabi.nyu.edu/hpc/training/quick_start.html)
- [CRC storage policy](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)
- [Maia-3 official source](https://github.com/CSSLab/maia3)
