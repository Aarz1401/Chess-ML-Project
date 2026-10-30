# Game Atlas: measured Jubail results, 2026-10-28

## What ran

The pipeline parsed 36,529 PGN records into **36,234 valid target-player
perspectives of 34,740 unique games**. Canonical move-sequence IDs
group duplicates and both target perspectives into one deterministic split.
There is zero canonical-game overlap across train, validation and test.

Training uses 29,064 perspectives; validation uses
3,548. The frozen encoder processes
836,664 post-opening target decisions. It uses the pinned
Maia-3 5M checkpoint at a shared Elo setting of 2000, with real eight-position
history and chosen from/to-square features.

Five small temporal training runs completed for 12 epochs each: three joint
concept/paired-view seeds, one concept-only ablation, and one explicitly
player-supervised control. The displayed temporal checkpoint is joint_21, selected
by validation loss within the joint-objective family.

## Fresh audit

An initial exploratory comparison used 1,600 test perspectives. After inspecting
that comparison, an explicit player-supervised control was added. The final
audit excludes all 1,576 game groups
used in that exploration, including their alternate perspectives.

The fresh audit uses **1,268 queries from
1,249 unique games**, against a class-balanced
pool of 2,400 training references. These are game-group
holdouts, not player holdouts: the same player labels occur in each split.

Random-reference player top-1 is 12.5% in the standard pool. Confidence intervals
bootstrap canonical query groups. Different-ECO removes exact ECO-code matches;
it does not remove every shared opening line or transposition.

| Representation | Player top-1 | 95% group interval | Different-ECO | Modern + different-ECO |
| --- | ---: | ---: | ---: | ---: |
| Strategic 288D | 14.75% | 12.93–16.71% | 13.18% | 14.88% |
| Frozen position 128D | 14.67% | 12.93–16.81% | 14.13% | 13.34% |
| Frozen position + move 128D | 14.59% | 12.63–16.57% | 13.65% | 14.59% |
| Concept-only temporal | 12.78% | 11.07–14.53% | 11.76% | 15.83% |
| Joint temporal, seed 21 (validation-selected) | 15.30% | 13.15–17.02% | 14.60% | 15.74% |
| Joint temporal, seed 42 | 14.20% | 12.25–16.05% | 13.65% | 16.89% |
| Joint temporal, seed 7 | 14.12% | 12.43–16.31% | 13.42% | 15.16% |
| Player-supervised control | 16.09% | 14.21–18.36% | 16.50% | 15.64% |

The modern cohort contains games from 2015–2026 with both header ratings between
2400 and 3000. It has 1,042 queries,
2,100 references, and seven player labels.
Its random-reference baseline is approximately
14.26%.
Time controls remain unknown; no time-control matching is claimed.

## What the evidence supports

The validation-selected joint model changes player top-1 by
**+0.55 percentage points** relative to the
strategic baseline. Its paired grouped 95% interval is
**-2.24 to +2.84 points**.
That interval includes zero. This experiment does not establish a retrieval gain.

A separate classifier using year, rating, ECO, target color and game length
reaches **50.79% player accuracy**.
It converged after 550 iterations. This is
a different classification task, not a directly comparable retrieval method.
It demonstrates why player association alone is insufficient evidence of style.

The small player-supervised control also does not establish a reliable gain over
the handcrafted baseline. Adding supervision, dimensions or GPU time did not
automatically solve the metric's task. A useful next experiment needs independent
judgments of strategic similarity and carefully constructed cross-game positives.

The current models may encode useful chess content even if they do not identify
players well. These diagnostics alone cannot establish that content quality.
The website makes positions, moves, feature overlays and neighboring games
inspectable so that better evaluation data can be collected.

## Resource measurements

Full frozen extraction covered 836,664 decisions in
46.1 seconds in the measured encoder loop on
Tesla V100-PCIE-32GB, with 252.7 MiB peak
PyTorch allocation. This excludes parsing, compact-history preparation, fitting
the pooling transforms, loading the checkpoint, and Slurm queue time.

Each temporal model has approximately
418,976 parameters. Existing measured training
loops took roughly 117
to 151 seconds per run,
excluding queue time. These measurements suggest data/objective/evaluation work
is a higher priority than a larger GPU for the next experiment.

## Website and reproducibility

The static site now exposes 1,000 replayable games and five representation modes:
simple statistics, strategic fingerprint, frozen Maia + moves, learned temporal,
and the player-supervised control. All neural vectors are real exported model
outputs. Player, opening, pawn-advancement and king-exposure overlays help inspect
what the maps contain. The displayed cohort includes preserved older links and
additional test games; each selected game shows its split.

Map projection uses the same PCA/t-SNE settings across modes. Projection fidelity
is reported separately from retrieval; neither cluster separation nor a high map
fidelity score is evidence of style recognition. Retrieval on the website is
within the displayed/filter-selected cohort, whereas this table uses the full
experiment's reference/query protocol.

- [Full measured report](../website/dist/experiments.json)
- [Public corpus manifest](../website/dist/corpus-manifest.json)
- [Jubail reproduction runbook](atlas-jubail-runbook.md)

The public report records the corpus/checkpoint IDs, source hashes, train/val/test
counts, GPU measurements, model selection rule, and cache-integrity seal. Local
checkpoints and detailed run artifacts are backed up under results/atlas_jubail
and excluded from Git. No new model weights are bundled into the website.

Known limits: eight curated player collections; missing time-control metadata;
unknown pretrained-data overlap; no human-judged semantic benchmark; unknown
licensing of the original PGN collections. Website software licensing does not
relicense the original data or Maia dependency. The older binary 76% result is
a different experiment and should not be compared directly with this eight-label
fresh audit.
