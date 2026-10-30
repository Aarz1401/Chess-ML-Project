# Position matching: measured results, October 30

The final focused round adds **5,963 sampled positions from 1,000 games**, two
position representations, paired replay boards, factual comparisons, and a
method-blinded evaluation workflow.

The neural vectors reuse the sealed Maia-3 cache computed on Jubail on October
28. Fitting and fixed-budget Stockfish annotations ran on a local four-worker
CPU fallback after compute and preempt jobs remained pending. No new GPU model
was trained. Both pending jobs were cancelled; no atlas job is left queued.

## Measured diagnostics

The comparison uses **792 test-game queries** and **1,084 training-reference
positions**. It excludes the same canonical game, exact/mirrored transpositions,
and matching exact ECO codes. Each query contributes once.

| Method | Median engine-evaluation gap | Median material gap | Same-player neighbor |
| --- | ---: | ---: | ---: |
| Strategic position, 89D | 1.325 pawns | 1 pawn unit | 13.89% |
| Frozen neural position, 64D | 1.130 pawns | 1 pawn unit | 14.90% |
| Random eligible position | 1.470 pawns | 1 pawn unit | 13.01% |

The two methods select different top recommendations for **97.73%** of query
games. The 64D PCA retains **92.86%** of training-position feature variance.
These are compression and descriptive retrieval measurements, not evidence of
human strategic relevance.

The neural method has a smaller median evaluation gap in this cohort. Similar
engine values do not imply similar plans, and no paired uncertainty test or
superiority claim is made for these medians. Material agreement is partly built
into the strategic features. The random control samples eligible positions,
so it is not uniform over reference games. These observations justify comparing
the methods with actual chess judgments rather than selecting a winner now.

Stockfish 17.1 uses one thread, 16 MiB hash and 8,000 nodes per position, with
search state reset each time. Median completed depth is 11. Mate scores use a
10,000-centipawn sentinel when computing gaps. These shallow annotations are
approximate and do not establish the best plan.

## Human evaluation status

**Zero independent human judgments are included.** The released workflow has
100 distinct-query-game trials where the two methods disagree: 20 development
and 80 reserved audit trials. Method assignments are in an offline key excluded
from the site. Player identities, method labels, engine scores and continuations
are omitted from the comparison interface.

Reviewers can choose A, B, tie or neither. Responses stay in their browser until
exported. The scoring script validates versions, deduplicates repeated exports,
keeps development/audit separate and bootstraps query-game preference scores.
The sample can compare preferences conditional on disagreement; it cannot
estimate overall recommendation accuracy.

## Artifacts and verification

- [Measured machine-readable report](../website/dist/moment-report.json)
- [Reproduction and evaluation protocol](moment-matching.md)
- Original frozen-cache seal SHA256:
  `3a2a97a1077497bb62982da23aed965a0f92a939872f610a0209df0d40f50d63`
- Local detailed artifacts: `results/moments_20261030/` (ignored by Git).
- The complete sealed cache is also backed up locally, including sequences.

Validation checks every exported FEN against the legal game replay, reconstructs
position facts and strategic vectors, verifies engine moves, and checks all 100
trial constraints. Python tests cover transpositions, color symmetry, retrieval
exclusions and judgment scoring. DOM interaction tests cover both site pages.
The local page serves HTTP 200. Native visual browser QA was unavailable because
the computer-use session was stopped.

The engineering round is complete. The remaining scientific evidence requires
independent relevance judgments. Keep the earlier game-level null result visible;
this round does not replace it with a claimed style-recognition breakthrough.
