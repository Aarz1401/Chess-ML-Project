Given chess games from different players or engines, I want to detect whether their move choices contain recognizable playing-style patterns.

## Compact Multiclass Classifier

This project now includes a small neural multiclass pipeline for the 8 PGN
players in `PGN_data`: Anand, Carlsen, Caruana, Duda, Erigaisi, Firouzja,
Praggnanandhaa, and Tal.

The classifier uses one padded full-game sequence per PGN game. Each move is
encoded as board-shaped channels:

- 34 paper-style before/after board and metadata channels.
- Role/context channels that mark target-player vs opponent moves.
- Stockfish evaluation channels mapped onto the board as scalar planes and
  destination-square heatmaps.

### Local Smoke Test

```bash
python3 scripts/train_multiclass.py \
  --max-games-per-player 5 \
  --epochs 1 \
  --stockfish-depth 1 \
  --output-dir results/smoke
```

### Default Proof Run

```bash
python3 scripts/train_multiclass.py \
  --max-games-per-player 500 \
  --sequence-length 80 \
  --stockfish-depth 2 \
  --multipv 3 \
  --epochs 10 \
  --batch-size 16 \
  --output-dir results/multiclass_proof
```

### HPC Run

```bash
sbatch scripts/run_multiclass_hpc.slurm
```

The training script saves metrics, a best checkpoint, a classification report,
and confusion matrix files under the chosen `results/` directory. Open
`notebooks/multiclass_results.ipynb` after training to inspect the saved
results.
