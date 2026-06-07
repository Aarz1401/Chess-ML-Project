# Chess Style Fingerprint

Chess Style Fingerprint is a machine-learning project for learning player-style
embeddings directly from PGN games. Instead of using player metadata, Stockfish
scores, or simple counts such as how often each piece moves, the model builds a
spatial representation of post-opening move sequences and learns a similarity
space where games by stylistically similar players should be close together.

## Proof Of Concept

A focused Tal-versus-Carlsen embedding experiment was run with a balanced
500-game-per-player dataset:

| Metric | Held-out result |
|---|---:|
| Top-1 nearest-game retrieval | 76.0% |
| Top-5 nearest-game retrieval | 85.0% |
| Retrieval mean average precision | 82.0% |
| Auxiliary classification accuracy | 73.0% |

The held-out split contained 100 games. The baseline for top-1 retrieval and
binary classification is 50%. The first 12 plies were skipped before tensor
recording to reduce direct opening memorization.

This is intentionally framed as a proof of concept. Tal and Carlsen played in
different eras, so era, opening trends, opponents, and tournament conditions may
contribute to the signal. A stronger validation would use same-era games matched
by time control, rating band, and opening distribution.

## Model

Each game is converted into a padded sequence of `84 x 8 x 8` move tensors.
The channels encode target-relative board and move information:

- target and opponent piece locations
- source and destination squares for each move
- legal destination heatmaps by piece type
- attack/control maps
- capture and threat maps
- king-zone pressure
- move-effect maps for attacks and defenses
- role planes for target-player and opponent moves
- game-level spatial summaries repeated across the sequence

The neural architecture is:

```text
84-channel move sequence
  -> spatial CNN move encoder
  -> Transformer sequence encoder
  -> pooled game embedding
  -> classifier head
  -> normalized metric projection for cosine similarity
```

The embedding run uses supervised contrastive learning with balanced batches,
so every batch contains same-player positives and different-player negatives.
The best embedding checkpoint is selected by validation retrieval mean average
precision, not by test performance.

## Repository Layout

```text
PGN_data/                    PGN files for the eight supported players
src/board_dataset.py         PGN parsing and 84-plane tensor construction
src/model.py                 CNN + Transformer + metric projection model
scripts/train_multiclass.py  Training, retrieval evaluation, checkpoints
scripts/export_embeddings.py Export learned metric embeddings
scripts/visualize_embeddings.py
                             PCA/t-SNE/UMAP embedding plots
scripts/predict_by_similarity.py
                             Query a PGN by cosine similarity
scripts/visualize_move_planes.py
scripts/visualize_game_planes.py
                             Debug visualizations for tensor channels
```

## Setup

```bash
cd Aadil-Chess-Stylometry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train An Embedding Model

Example Tal-versus-Carlsen embedding run:

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

Training saves:

```text
best_model.pt                 best validation classification checkpoint
best_embedding_model.pt       best validation retrieval checkpoint
metrics.csv                   epoch metrics
retrieval_metrics.json        top-k, MRR, MAP, cosine margin
classification_report.csv
confusion_matrix.png
```

## Export Embeddings

```bash
python3 scripts/export_embeddings.py \
  --checkpoint results/poc_tal_carlsen_embedding_500/best_embedding_model.pt \
  --max-games-per-player 500 \
  --sequence-length 60 \
  --skip-plies 12 \
  --batch-size 32 \
  --output-dir results/poc_tal_carlsen_embedding_500/embeddings
```

This writes one tensor file and one metadata CSV per split:

```text
embeddings_train.pt
embeddings_train_metadata.csv
embeddings_val.pt
embeddings_val_metadata.csv
embeddings_test.pt
embeddings_test_metadata.csv
```

## Visualize Embeddings

```bash
python3 scripts/visualize_embeddings.py \
  results/poc_tal_carlsen_embedding_500/embeddings/embeddings_test.pt \
  --output-dir results/poc_tal_carlsen_embedding_500/embedding_plots
```

The script saves PCA and t-SNE plots. If `umap-learn` is installed, it also
saves a UMAP plot.

## Predict By Similarity

After exporting reference embeddings, query a new PGN game:

```bash
python3 scripts/predict_by_similarity.py \
  --checkpoint results/poc_tal_carlsen_embedding_500/best_embedding_model.pt \
  --reference-embeddings results/poc_tal_carlsen_embedding_500/embeddings/embeddings_train.pt \
  --pgn path/to/query_game.pgn \
  --game-index 0 \
  --target-color white \
  --sequence-length 60 \
  --skip-plies 12
```

The script prints label-level cosine similarity scores and the nearest reference
games in embedding space.

## Notes

- The project currently targets eight PGN labels: Anand, Carlsen, Caruana,
  Duda, Erigaisi, Firouzja, Praggnanandhaa, and Tal.
- Generated checkpoints, plots, result CSVs, and local outreach drafts are
  ignored by Git.
- The reported POC result is not a claim of solved player identification. It is
  evidence that post-opening board-sequence embeddings contain measurable
  player-associated signal.
