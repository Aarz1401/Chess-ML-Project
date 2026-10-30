# Understanding Game Atlas for an interview

## The problem

Given a chess game, retrieve games with related patterns and make the comparison
inspectable on a map and replay board. This is a representation-learning and
similarity-search project. Defining useful similarity is part of the problem.

## Follow one game through the current pipeline

1. Parse its PGN, match the target player exactly, and reconstruct legal boards.
2. Hash the starting FEN and move sequence. Keep duplicate games and both player
   perspectives in the same train/validation/test split.
3. Skip 12 opening plies, stop at ply 120, and sample up to 24 target decisions.
4. Encode each pre-move board and its preceding eight-position history with a
   frozen, pretrained Maia-3 5M model. Keep its pooled position features and the
   chosen move's from/to-square features: 768 numbers per decision.
5. Either pool these features directly, or train a small two-layer Transformer
   with position information and attention pooling to produce a normalized
   128-dimensional game vector.
6. Retrieve neighbors using the full vector. Separately project displayed games
   with PCA/t-SNE to make an interactive map.
7. Export vectors and replay chunks. The website serves static files and needs
   no live connection to the cluster.

The upstream Maia model was pretrained by its authors to predict human moves.
This project adapts those features and trains the small game-level encoder; it
does not claim to have trained the foundation model from scratch.

## Understand the experiments

| Mode | Where its representation comes from |
| --- | --- |
| Simple baseline | 12 whole-game chess statistics |
| Strategic fingerprint | 96 target-relative features, averaged in three segments |
| Frozen neural | Train-only reduction of pooled Maia position and move features |
| Learned temporal | Paired views plus engineered concept targets, with three seeds |
| Player control | Explicit player-label classification and contrastive supervision |

Names and results are not network inputs. Player names supply supervision in
the identity-control experiment. The concept targets are engineered labels,
so recovering those concepts is not independent semantic validation.

Black's ranks are mirrored so pawn advancement has the same meaning for either
color. Kingside/queenside identity is preserved. Actual legal histories are
retained: we do not shuffle moves or invent illegal chess augmentations.

Attention pooling learns how to combine decision tokens. Its weights are not
automatically faithful explanations of the model's decisions.

## Explain the metric and visualization

For vectors normalized to length one, squared Euclidean distance is
2 minus 2 times cosine similarity. The two metrics give the same neighbor order.

Standardization makes numerical feature scales comparable; block weights keep
48 placement dimensions from overwhelming seven king-safety dimensions.
Those weights are explicit design choices.

PCA compresses redundant dimensions. t-SNE attempts to preserve local
neighborhoods in two dimensions. Map gaps, cluster sizes and axes have no direct
chess interpretation. Benchmark retrieval uses full vectors, not screen distance.

## Explain the real result

The fresh eight-label audit gives 14.75% player top-1 for the strategic baseline
and 15.30% for the validation-selected joint model. The paired uncertainty
interval includes zero, so the experiment does not establish an improvement.

A separate metadata classifier reaches 50.79%. It is a different task, but
demonstrates how era, opening, rating, color and game length can reveal identity.

The main lesson is that more compute and richer embeddings do not define a
useful similarity objective. Player identification is a proxy; the atlas still
needs independent judgments of which games are strategically related.

Read the [measured report](atlas-results-2026-10-28.md) for all seeds, the
player-supervised control, the modern/different-ECO cohort, and uncertainty.

## What to be able to defend

The final focused round adds position-level retrieval. It reuses the sealed
Jubail Maia cache, compares 89D current-position features with 64D reduced neural
position tokens, and makes the match inspectable on paired replay boards.
Chosen and future moves are excluded from these position vectors. Its 100
prepared method-blinded trials separate 20 development from 80 audit queries;
independent human judgments remain pending. Read the
[focused-round protocol](moment-matching.md) for input boundaries, CPU fallback,
engine limits, and the scoring procedure.

- Trace a single game through IDs, orientation, sampled history, tokens and vectors.
- Explain why duplicate grouping happens before splitting.
- Distinguish pretrained features, learned pooling and engineered labels.
- Explain validation-based selection and the fresh audit after exploration.
- Describe the augmentation choices and possible false negatives.
- Separate player retrieval, strategic similarity and projection fidelity.
- Explain the next experiment: judged cross-game pairs and a meaningful metric,
  before investing in a larger backbone.

A defensible project description:

> Game Atlas adapts a pretrained chess encoder into game embeddings, trains and
> compares temporal objectives on Jubail, and publishes a replayable atlas with
> grouped evaluation and uncertainty. The first controlled results show why
> player identity alone is insufficient evidence of playing style.
