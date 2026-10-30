# Game Atlas

An interactive chess-game atlas with reproducible representation-learning
experiments. Explore real games, replay their moves, and compare engineered,
pretrained, and learned game fingerprints.

## Explore the website

The atlas contains **1,000 games** sampled from a grouped corpus of **34,740
unique games**. It includes five selectable representations, different-ECO
neighbors, feature overlays, and real Jubail experiment results.

~~~bash
python3 -m http.server 8765 --directory website/dist
~~~

Open <http://localhost:8765>. Serving the site needs only a static HTTP server.
Node is optional and used for development tests.

The final focused round adds **Find a moment** at `/moments.html`: paired
position boards, transparent shared facts, historical continuations, and 100
prepared blinded relevance comparisons. Independent human judgments are pending.
See the [position-matching protocol](docs/moment-matching.md).
Read the [measured October 30 diagnostics](docs/moment-results-2026-10-30.md).

- [Measured October 28 results](docs/atlas-results-2026-10-28.md)
- [Jubail reproduction runbook](docs/atlas-jubail-runbook.md)
- [Website setup and methodology](website/README.md)

The learned models have not established a reliable gain over the handcrafted
baseline on held-out player retrieval. The results and uncertainty are published
alongside the map. Player identification and projection fidelity do not establish
strategic-style similarity.

The website code is MIT licensed. Existing PGN data, third-party models and
pre-existing project code are not relicensed by that website license.

## Layout

```text
src/                         Shared identity, temporal encoder, legacy board model
scripts/                     Corpus, pretrained extraction, training, evaluation, Jubail jobs
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

## Legacy board-model local check

This checks the full cache → train → evaluate path with a deliberately tiny
model and dataset. Its scores are not research results.

```bash
python scripts/precompute_tensors.py \
  --cache-dir results/smoke_cache --labels Tal Carlsen \
  --max-games-per-player 50 --sequence-length 8 --min-valid-plies 4 --workers 2

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

For the new atlas pipeline, use the [Jubail runbook](docs/atlas-jubail-runbook.md).
For older board-model experiments, use the [workflow reference](docs/workflow.md) and
[Jubail guide](docs/jubail.md). Keep the cache preprocessing settings consistent
with the training arguments recorded in checkpoints.

## Research status

A saved historical Tal-versus-Carlsen run reports 76% top-1 retrieval and
82% retrieval MAP on 100 held-out games. These results predate the current
residual model architecture; they have not been reproduced during this cleanup.
Era, opening, opponent, and duplicate-game effects still need controlled
validation. Player identification alone does not establish a measure of style.

The earlier [project audit](docs/project-audit.md) records the cleanup findings.
The [measured atlas report](docs/atlas-results-2026-10-28.md) records which issues
were fixed and the limits of the current learned representations.
