# Chess Style Fingerprint

Learn player-associated chess embeddings from PGN games using an 84-channel
board representation, a CNN + Transformer, and supervised contrastive learning.

This is the active project in the
[Chess-ML-Project GitHub repository](https://github.com/Aarz1401/Chess-ML-Project).
Start here and run commands from this directory.

## Game Atlas website

Explore 240 real games on an interactive map, replay their moves, and retrieve
similar games. The default map uses 288-dimensional, target-relative strategic fingerprints;
the original 12-statistic baseline is available for comparison. Both are
engineered representations, separate from the neural model.

```bash
python3 -m http.server 8765 --directory website/dist
```

Open <http://localhost:8765>. See [website setup and methodology](website/README.md).
For the interview explanation, read [how Game Atlas works](docs/game-atlas-interview.md).
The new website code is MIT licensed; the existing PGN data is not relicensed.

## Layout

```text
src/                         Board tensors, model, cache, evaluation
scripts/                     Training, evaluation, visualization, Jubail jobs
data/pgn/                    Eight player PGN files (versioned)
notebooks/                   Explore saved results
docs/                        Workflow, repository audit, Jubail guidance
website/                     Game Atlas website and dataset export scripts
archive/chess-style-fingerprint/
                             Earlier human-vs-Stockfish research
results/                     Local checkpoints and generated reports (ignored)
```

The former `Aadil-Chess-Stylometry/` is now the root project. The former
`chess-style-fingerprint/` is archived because it solves a different task with
34-channel tensors, Stockfish-derived features, and different dependencies.
Both were ordinary folders in this one Git repository, not separate clones.
See the [audit and migration map](docs/project-audit.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Small local check

This checks the full cache → train → evaluate path with a deliberately tiny
model and dataset. Its scores are not research results.

```bash
python scripts/precompute_tensors.py \
  --cache-dir results/smoke_cache --labels Tal Carlsen \
  --max-games-per-player 10 --sequence-length 8 --min-valid-plies 4 --workers 2

python scripts/train_multiclass.py \
  --cache-dir results/smoke_cache --output-dir results/cleanup_smoke \
  --sequence-length 8 --epochs 1 --warmup-epochs 0 --num-workers 0 \
  --batch-size 4 --labels-per-batch 2 --samples-per-label 2 \
  --conv-channels 8 --n-blocks 1 --move-dim 16 --transformer-dim 16 \
  --n-layers 1 --n-heads 2 --feedforward-dim 32 --metric-dim 8 \
  --agg-ks 1 --agg-trials 2

python scripts/evaluate_aggregated.py \
  --checkpoint results/cleanup_smoke/best_embedding_model.pt \
  --cache-dir results/smoke_cache --output-dir results/cleanup_smoke/aggregated_eval \
  --ks 1 --trials 2 --num-workers 0
```

For larger experiments, use the [workflow reference](docs/workflow.md) and
[Jubail guide](docs/jubail.md). Keep the cache preprocessing settings consistent
with the training arguments recorded in checkpoints.

## Research status

A saved historical Tal-versus-Carlsen run reports 76% top-1 retrieval and
82% retrieval MAP on 100 held-out games. These results predate the current
residual model architecture; they have not been reproduced during this cleanup.
Era, opening, opponent, and duplicate-game effects still need controlled
validation. Player identification alone does not establish a measure of style.

The [project audit](docs/project-audit.md) lists the next improvements in order.
