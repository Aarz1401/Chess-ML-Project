# Archived human-versus-Stockfish experiments

This is the earlier research workflow, preserved with its notebook and results.
The active project is at the [repository root](../../README.md).
Run this workflow from `archive/chess-style-fingerprint/` with its own dependencies.
Its local `data/` is ignored by Git and is not included in a fresh clone.

This project tests whether chess games contain measurable style signals. The final workflow compares a neural board-sequence model against a simpler feature-based classifier on human-vs-Stockfish style classification.

## Quick Start

1. Install dependencies and activate venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install stockfish
```

2. Open and run:

```
final_workflow.ipynb
```

The notebook contains the final experiments, result tables, and figures used for the presentation.

## Final Models

**Model 1: Neural Hybrid Stylometry**

- Input: padded 80-move board sequence plus 33 style features.
- Shape: `(B, 80, 34, 8, 8)` for the board branch.
- Architecture: CNN move encoder, Transformer sequence encoder, feature MLP, fused classifier.
- Main motivation: learn style from both board patterns and evaluation-based trajectory features.

**Model 2: Style Feature Classifier**

- Input: 33 style-trajectory features.
- Architecture: classical classifiers, mainly Random Forest for the final comparison.
- Main motivation: keep the experiment interpretable and controlled by comparing human and Stockfish continuations from similar starting positions.

## Data

Raw PGNs are stored under:

```text
data/raw/<label>/*.pgn
```

Downloaded game counts:

| Player / Engine | PGN games |
|---|---:|
| Magnus Carlsen | 7,484 |
| Viswanathan Anand | 4,211 |
| Anatoly Karpov | 3,529 |
| Mikhail Tal | 2,431 |
| Garry Kasparov | 2,128 |
| Tigran Petrosian | 1,893 |
| Stockfish | 1,000 |
| Mikhail Botvinnik | 891 |
| Bobby Fischer | 827 |
| Jose Raul Capablanca | 597 |
| Paul Morphy | 211 |

Morphy was excluded from the final balanced all-pairs experiments because too few games survived preprocessing for the 200-games-per-class setup.

## Training, Ablations, and Tuning

The final experiments use an 80/10/10 game-level split. With 200 games per class, this gives 160 train, 20 validation, and 20 test games per class.

Main tuning choices explored:

- Learning rates: `3e-4` and `1e-4`.
- Batch sizes: `8` and `16`.
- Epochs: `8` and `16`.
- Model scale: `tiny`, `small`, and a defined but expensive `paper` configuration.
- Input format: 15-move windows vs padded 80-move whole-game sequences.
- Board orientation: mover-perspective board flipping vs no-flip ablation.
- Feature choice: direct `engine_agreement` baseline vs final `style_trajectory` features.

The final Model 1 all-pairs run used the `small` model, 200 games per class, padded 80-move inputs, 16 epochs, batch size 16, learning rate `1e-4`, Stockfish depth 2, and up to 30 analyzed moves per game.

Regularization included Dropout in the classifier, Transformer dropout, and AdamW weight decay. Validation curves were used to monitor overfitting.

Important ablations included 8-class human-player classification, pairwise human-vs-Stockfish classification, board flipping, short-window vs padded-game inputs, and Model 2 single-classifier vs best-per-pair classifier selection.

## Project Requirements Coverage

- **Problem and task:** supervised classification of chess playing style, mainly binary human-vs-Stockfish classification with a multiclass human-player ablation.
- **Example input/output:** input is a PGN game or controlled continuation; output is a predicted class label such as `tal` or `stockfish`.
- **Preprocessing/EDA:** PGNs are parsed, invalid/short games are filtered, games are balanced by class, player moves are selected from headers, and controlled same-FEN continuations are generated for Model 2.
- **Feature extraction:** Model 1 builds 34-channel board tensors; both models can use 33 style-trajectory features based on evaluation changes, material, move behavior, and piece usage.
- **Loss and metrics:** neural runs use cross-entropy loss; balanced classification results are reported with accuracy and loss.
- **Evaluation:** final tables compare Model 1 and Model 2 on all-pairs averages, with saved plots for training/validation/test curves and ablations in `outputs/figures/`.
- **Reproducibility:** scripts use fixed seeds where applicable, dependencies are listed in `requirements.txt`, and key CSV outputs are saved under `outputs/`.
- **Optional components covered:** literature/state-of-the-art discussion, ablation studies, controlled dataset construction. 

## Repository Layout

```text
final_workflow.ipynb         final notebook for results and figures
requirements.txt             Python dependencies

src/
  dataset.py                 PGN parsing and board tensor construction
  model.py                   neural model definitions
  pipeline.py                board-only training pipeline
  experiment_utils.py        shared experiment utilities

scripts/
  run_hybrid_experiment.py   Model 1 runner
  make_controlled_engine_dataset.py
                             controlled human/Stockfish continuation builder
  run_model1_tal_carlsen.py  human-vs-human Model 1 run
  model_size_report.py       parameter count helper

retrieval/
  run_engine_quality_baseline.py
                             Model 2 feature classifier

data/                        PGN data
outputs/                     result CSVs and generated figures
```

## Important Outputs

```text
outputs/model1_all_pairs_summary_g200_e16_b16_lr1e4.csv
outputs/model2_all_human_vs_stockfish_style_trajectory_summary_g200.csv
outputs/hybrid_padded80_tal_stockfish_style_small_d2_m30_g200_e8.csv
outputs/model1_multiclass8_human_padded_style_tiny_d1_m12_g40_e6.csv
outputs/figures/
```

## Notes

- Stockfish must be available on `PATH`.
- The final feature set is `style_trajectory`; direct Stockfish top-move agreement is kept only as a baseline.
- `.venv/`, caches, and local generated files are ignored by Git.
