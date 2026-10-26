# Game Atlas

A static, open-source chess game explorer. Browse 240 real games from eight
player collections, replay every move, and retrieve nearby games.

## Run locally

From the repository root:

```bash
python3 -m http.server 8765 --directory website/dist
```

Open <http://localhost:8765>. There is no Node build, backend, account, database,
or API key. The bundled JSON supports search, filtering, board replay, PGN
export, and shareable game links entirely in the browser. Google Fonts is an
optional external font request; system fonts are used if it is unavailable.

## Rebuild the collection

Use the root Python environment with the project requirements installed:

```bash
.venv/bin/python website/scripts/build_atlas.py
.venv/bin/python website/scripts/validate_atlas.py
node --check website/dist/app.js
```

The deterministic seed is 42, with 30 unique games per PGN collection. Games
are deduplicated using the starting FEN and move sequence. A game shared by
multiple source collections is assigned to the first collection sampled.
Invalid games and games shorter than 20 plies are skipped. This sample is a
browsing collection, not a train/test benchmark or an unbiased player census.

## Understand the implementation

The default representation is a **288-dimensional strategic fingerprint**:
96 features, averaged over three chronological segments of the target player's
post-opening positions and decisions. It skips 12 plies and stops at ply 120.
The segments are equal portions of the observed sequence; they are not labels
for opening, middlegame, or endgame.

| Feature block | Dimensions per segment | Weight | What it captures |
| --- | ---: | ---: | --- |
| Piece placement | 48 | 1.0 | Six piece types across eight target-relative board regions |
| Pawn structure | 16 | 1.5 | File occupancy, doubled/isolated/passed/connected pawns, islands and advancement |
| Activity | 12 | 1.25 | Center control, mobility, defended pieces and opponent-half control |
| King safety | 7 | 1.0 | King position, pawn shield, attacked/defended king zone and castling rights |
| Material | 6 | 0.5 | Material balance and remaining material |
| Decisions | 7 | 1.0 | Captures, checks, pawn moves, castling, destination and movement |

`game_vectors.py` contains the chess representation. Black's ranks are mirrored
so advancement has the same meaning for either color; kingside/queenside file
identity is preserved. A color/rank-symmetry check verifies this transformation.

Features are standardized across the sampled collection, clipped to [-3, 3],
then scaled by `sqrt(block_weight / dimensions_in_block)`. This distributes a
block's weight across its dimensions. Vectors are L2-normalized. Recommendations
use exact Euclidean distance in this space; for normalized vectors this gives
the same ranking as cosine similarity. The explicit weights are design choices,
not learned or validated optimal weights.

PCA reduces to 30 dimensions, then seeded t-SNE produces the 2D display.
Recommendations always use the full fingerprint. The UI reports local neighbor
preservation because the map is lossy. Cluster area and gap size are not
reliable similarity measures. Smaller distance means closer under the selected
representation; it is not a probability or calibrated style score.

The original **12-statistic baseline** remains selectable. It summarizes both
sides' whole-game move frequencies, length, captures, checks, castling, final
material and early queen activity, and displays them with PCA. It is useful
for comparing the effect of representation changes.

Names select the target side and label the game; names, dates, results and ECO
codes do not enter either vector. Features may still reflect openings, era,
strength, and opponents. “Closest blocks” lists the two blocks with the smallest
weighted-space differences after dividing out their explicit block weights;
it helps inspection but does not establish strategic equivalence.

## Measured projection quality

For the bundled 240-game collection, seed 42:

| Map | Average retention of the 10 nearest neighbors |
| --- | ---: |
| Strategic fingerprint + t-SNE | 40.0% |
| Simple baseline + PCA | 27.6% |

Strategic t-SNE trustworthiness at k=10 is 0.883. These numbers concern how well
each projection displays its own vector distances. They do **not** compare the
chess usefulness of the two representations. Both the features and projection
changed, so the difference cannot be attributed to either one alone.

Validation replays all games legally, reconstructs both vector pipelines,
checks unique game identities, finite coordinates, and color symmetry.

## Neural embeddings next

These are engineered, interpretable fingerprints. The existing CNN/Transformer
can supply a third representation once trained and evaluated for the atlas's
similarity task. Its original same-player objective does not automatically
make games with similar strategic content neighbors.

Keep the same game IDs when exporting neural embeddings. Evaluate retrieval
with held-out game groups, matched openings and an explicit similarity rubric.
A reproducible comparison should use the same projection settings for each
representation and separately score human judgments of the nearest games.
Do not claim learned strategic similarity based only on visually distinct
clusters or same-player retrieval.

## Deployment

`dist/` can be hosted by any static web host. No routing rewrites are needed:
selected games use URL fragments. `.openai/hosting.json` records the private
Sites preview identity; self-hosters do not need it and should create their own
hosting configuration. Hosting metadata is not a credential.

## Contributing and licensing

The code newly added under `website/` is MIT licensed; see `LICENSE`.
The bundled `dist/atlas.json` is derived from existing repository PGNs and is
excluded from that software license. Original collection download URLs and
redistribution terms were not documented in the source repository. Track those
in `DATA.md` before describing the dataset itself as openly licensed.
The software license does not relicense the pre-existing project or its data.

Useful first contributions: neural export integration, better game sampling,
opening labels, annotated feature comparisons, and dataset provenance.
