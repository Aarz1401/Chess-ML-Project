# Chess ML Project

This repository contains my chess machine-learning work, centered on learning
style fingerprints from PGN games.

The main public-facing project is:

```text
Aadil-Chess-Stylometry/
```

It builds neural embeddings for chess games using board-state and move-sequence
features, then evaluates whether games by the same player are close together in
embedding space.

## Featured Project: Chess Style Fingerprint

`Aadil-Chess-Stylometry` learns a similarity space for chess games without using
player names, Stockfish scores, or simple per-piece move counts. Each game is
converted into a sequence of `84 x 8 x 8` board tensors and encoded with a
spatial CNN plus Transformer. A supervised contrastive objective trains a
normalized metric embedding for cosine similarity search.

Proof-of-concept result:

| Experiment | Held-out result |
|---|---:|
| Tal vs Carlsen top-1 nearest-game retrieval | 76.0% |
| Tal vs Carlsen top-5 nearest-game retrieval | 85.0% |
| Retrieval mean average precision | 82.0% |
| Auxiliary classification accuracy | 73.0% |

The held-out split contained 100 games, and the first 12 plies were skipped to
reduce direct opening memorization. The binary baseline is 50%.

Start here:

- [Project README](Aadil-Chess-Stylometry/README.md)
- [Model code](Aadil-Chess-Stylometry/src/model.py)
- [Tensor construction](Aadil-Chess-Stylometry/src/board_dataset.py)
- [Training script](Aadil-Chess-Stylometry/scripts/train_multiclass.py)
- [Embedding visualization](Aadil-Chess-Stylometry/scripts/visualize_embeddings.py)

## Repository Structure

```text
Aadil-Chess-Stylometry/
  Current public project. CNN + Transformer chess style embeddings, metric
  learning, retrieval evaluation, and embedding visualization.

chess-style-fingerprint/
  Earlier research workflow and experiments comparing neural board-sequence
  models with feature-based classifiers, including human-vs-Stockfish style
  experiments.
```

## Quick Start

```bash
cd Aadil-Chess-Stylometry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the current embedding proof of concept:

```bash
python3 scripts/train_multiclass.py \
  --labels Tal Carlsen \
  --max-games-per-player 500 \
  --sequence-length 60 \
  --skip-plies 12 \
  --epochs 10 \
  --batch-size 32 \
  --samples-per-label 16 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --classification-weight 0.25 \
  --contrastive-weight 1.0 \
  --metric-dim 64 \
  --output-dir results/poc_tal_carlsen_embedding_500
```

## Notes

- Generated checkpoints, plots, local result folders, and outreach drafts are
  ignored by Git.
- The included PGN data is used for reproducible local experiments.
- The current proof of concept shows measurable player-associated signal, but
  it is not a claim that style has been fully isolated from era, opening trends,
  opponent pools, or game-selection effects.
