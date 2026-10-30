# Game Atlas website

A static explorer of 1,000 real games with five representation modes, lazy board
replays, different-ECO neighbors, feature overlays, and measured Jubail results.

`moments.html` adds position-level recommendations with paired boards, shared
facts, historical continuations, and 100 prepared method-blinded comparisons.
See the [focused-round protocol](../docs/moment-matching.md). Human relevance
judgments remain pending.

## Serve and test

From the repository root:

~~~bash
python3 -m http.server 8765 --directory website/dist
~~~

Open <http://localhost:8765>. Production needs no Node runtime, API key, account
database or model server. Node is only a development-test dependency:

~~~bash
npm ci --prefix website --ignore-scripts
npm test --prefix website
.venv/bin/python website/scripts/validate_atlas.py
.venv/bin/python website/scripts/validate_moments.py
.venv/bin/python -m unittest discover -s tests -v
~~~

The DOM tests exercise loading, replay, filtering, representation switching,
sharing, PGN downloads and measured-report rendering. They do not replace
browser layout or accessibility testing. Native browser QA was unavailable in
the implementation session.

## Representations

- Simple baseline: 12 whole-game statistics.
- Strategic fingerprint: 288 target-relative, phase-aware engineered features.
- Frozen Maia + moves: 128D, train-only pooling/reduction of pretrained features.
- Learned temporal: 128D, small game encoder with paired-view/concept objectives.
- Player-supervised control: 128D, explicitly trained for player identity.

The default joint model was selected by validation loss among three seeds.
Player retrieval has not demonstrated a reliable gain over the strategic
baseline. See [all measured results](../docs/atlas-results-2026-10-28.md), including
the paired confidence interval and metadata-confound check.

The map uses shared PCA/t-SNE settings. Neighbors use Euclidean distance in the
selected full vector space. Neural vectors are unit-normalized, so cosine gives
the same ranking. The different-ECO checkbox excludes exact code matches; it
does not eliminate all opening similarity.

Color overlays show collection, ECO family, king exposure or pawn advancement.
The chess-feature overlays are observable summaries, not causal explanations
of a neural match.

## Data delivery

dist/atlas.json contains the map, vectors, metadata and feature summaries.
dist/replays/ contains chunks of 25 games with SAN moves, FENs and PGN text.
A replay chunk loads on selection and is cached in the browser session.

dist/experiments.json is the measured grouped audit.
dist/moments.json contains the sampled position index and two vector types.
dist/moment-review.json contains the blinded trials, without method assignments.
dist/moment-report.json records the new diagnostics and actual execution host.
Anonymous judgments stay in local storage until exported by the reviewer; there
is no server-side collection. The offline review key is excluded from dist.
dist/corpus-manifest.json records source hashes, counts, missing metadata and
split sizes. The source has no time-control headers; no matched-time-control
claim is made. Browser neighbors are searched within the displayed/filter-
selected cohort, not the larger benchmark reference index.

The older 240 links are preserved where their game/perspective survived the
validity filter. Selected games disclose train/validation/test membership.
Additional displayed games are sampled from test groups.

## Rebuild the actual neural atlas

Use the repository checkout and the [Jubail runbook](../docs/atlas-jubail-runbook.md).
The scripts preserve canonical IDs through corpus preparation, extraction,
training, evaluation and export. No upstream model weights are bundled in dist.

For the original handcrafted-only experiment, the earlier generator remains:

~~~bash
.venv/bin/python website/scripts/build_atlas.py
~~~

This replaces atlas.json with the older feature-only schema. Use a backup or
separate output path when comparing it with the neural export.

## Hosting and licensing

dist/ works on a static web host. URL fragments identify selected games.
.openai/hosting.json records the existing private Sites preview identity;
self-hosters should create their own hosting configuration.

The newly added website software is MIT licensed; see LICENSE. That license
excludes PGN-derived data (including moment exports and replay chunks),
third-party fonts, pretrained models and pre-existing repository code. Original PGN collection download URLs and
redistribution terms are not documented; see DATA.md. Maia-3 is a separately
licensed upstream dependency with pinned source and checkpoint revisions.

Google Fonts is an optional external font request; system fonts are the fallback.
