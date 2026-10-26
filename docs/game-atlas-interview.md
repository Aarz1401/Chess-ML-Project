# Understanding Game Atlas

## The problem

Given a chess game, find other games with comparable patterns and let a user
inspect them on a map and replay board. This is a similarity-search problem.
The current website uses engineered features. The repository's neural model
is a separate experiment and has not yet supplied the website's vectors.

## Follow one game

1. Parse its PGN and identify the target player's color from the player header.
2. Replay the moves legally. Keep positions after the target player's moves,
   starting after 12 plies and stopping at ply 120.
3. For each retained position/decision, compute 96 numerical features.
4. Divide the retained sequence into three chronological segments. Average
   each segment independently, then concatenate: 96 × 3 = 288 dimensions.
5. Standardize each dimension using the reference collection, clip extreme
   standardized values to [-3, 3], and apply block weights.
6. Normalize the vector to length one. Compare vectors to retrieve games.
7. Project the collection to two dimensions for browsing. Keep retrieval in
   the original 288-dimensional space.

## Explain the main choices

**Why target-relative?** A black pawn moving toward rank 1 and a white pawn
moving toward rank 8 both advance. Mirroring ranks makes those comparable.
Kingside and queenside remain distinct.

**Why three segments?** An average over a whole game loses how positions
change. Three summaries preserve coarse progression without requiring games
to have equal lengths. They still lose exact move order inside each segment.

**Why standardization?** A feature measured in large numerical units should
not dominate distance simply because of its units.

**Why block weights?** There are 48 placement dimensions but seven king-safety
dimensions per segment. Distribute a block's weight across its dimensions so
its dimension count alone does not control similarity. The chosen weights
are explicit heuristics and should be tested with retrieval judgments.

**Why normalize?** We compare direction in feature space. For unit vectors,
squared Euclidean distance equals `2 - 2 * cosine_similarity`, so Euclidean
and cosine similarity give the same neighbor ranking.

**Why PCA then t-SNE?** PCA compresses redundant dimensions, then t-SNE attempts
to preserve local neighborhoods in two dimensions. t-SNE can create visually
convincing groups, so evaluate neighborhood preservation and inspect retrieved
games. Map gaps, cluster sizes, and axes have no direct chess interpretation.

## Evidence and limitations

All 240 bundled games are legally replayed in validation. Both feature pipelines
are reconstructed, and target-relative color symmetry is checked. The website
reports how many high-dimensional neighbors survive the map projection.

These checks establish implementation consistency and projection fidelity.
They do not establish that the system captures human notions of chess style.
We need a similarity rubric, judged pairs or triplets, and controlled tests
for openings, era and playing strength. A larger number of dimensions alone
is not evidence of a better representation.

The current collection has no held-out evaluation for this similarity task.
Standardization and projection fit the entire browsing collection; that is
appropriate for displaying it, but is not a generalization benchmark.

## The next ML experiment

Use the engineered fingerprint as a baseline. Train the existing sequence
encoder on a clearly defined similarity objective and compare its nearest
games on the same held-out game identities. Same-player labels measure player
association; they are not automatic ground truth for strategic similarity.
Group duplicated games before splitting, keep a final test set untouched, and
compare feature groups and neural representations with the same evaluation.

An honest project description today:

> I built a chess-game explorer with reproducible, target-relative feature
> vectors and similarity search. I separated the representation from its 2D
> visualization, measured projection fidelity, and made the retrieved games
> inspectable. The next experiment compares the baseline with learned embeddings
> using a controlled similarity evaluation.
