# Find a moment: the final focused round

Game Atlas now compares specific positions as well as whole games. Open
`website/dist/moments.html` through a static server, or choose **Find a moment**
from the atlas. A selected game also links directly to its sampled positions.

Choose a source position, inspect related moments from other training-reference
games, and follow up to twelve historical plies on either board. Comparisons
show pawn structure, king placement, king-zone attacks, central control and
material. These are measured facts about the starting positions, not a generated
coaching explanation or an attribution of the neural model's decisions.

## What changed in the representation

The October 28 experiment pooled entire games. This round takes up to six
cached target-to-move positions per displayed game between plies 24 and 100.
This is a move-range sampling rule, not a rigorous middlegame classifier.

Two methods remain deliberately simple:

| Method | Inputs | Transform |
| --- | --- | --- |
| Strategic position | 89 current-board features relative to the side to move | Training-only standardization; clip at 3; block weights; unit normalization |
| Frozen neural position | 256 pooled square-token features from the pinned Maia-3 encoder and its previous legal position history | Training-only standardization; 64D PCA; unit normalization |

The chosen move's square features are excluded. Continuations, engine scores,
names, dates, results and ECO codes are not vector inputs. ECO is used only to
filter candidates. Scaling/PCA fit uses a deterministic sample of 1,000 training
perspectives; all perspectives of a canonical game remain in the original split.

The website computes full-vector Euclidean distances locally, shows at most one
position per reference game, and rejects the same canonical game and exact
transpositions, including color/rank mirrored equivalents. Different exact ECO
codes are required by default, with both codes known. Different ECO does not
guarantee completely different opening histories.

Boards use actual piece colors and orient the original side to move at the
bottom. Color/rank mirroring for the vectors preserves kingside/queenside file
identity. Neither method is trained or selected using the human audit.

## Engine diagnostics

Stockfish 17.1 annotates each exported position using one thread, 16 MiB hash and
8,000 nodes, resetting search state for every position. Engine values are from
the original side-to-move perspective. Best moves are displayed separately from
the historical continuation. This small budget produces approximate annotations,
not authoritative evaluations. Mate scores use a 10,000-centipawn sentinel for
the descriptive evaluation-gap calculation; do not interpret that sentinel as a
literal material value.

The report compares one deterministic query per displayed test game with
training-reference positions. It records median evaluation gap, median material
gap, same-player association and disagreement between the two methods. A random
eligible-position control is included. These diagnostics do not measure human
relevance. The strategic method contains material features, so its material
agreement is partly built in.

The frozen neural activations were computed on Jubail in the October 28 run.
This round attempted a four-CPU job in compute and then preempt; both remained
pending. The queued jobs were cancelled, and the final bounded position fitting
and engine annotations used a local CPU fallback with the sealed Jubail cache.
The report records the actual host, package versions, runtime and engine hash.
No additional GPU training is claimed.
Read the [measured October 30 results](moment-results-2026-10-30.md).

## A real human evaluation, ready to collect

The website includes 100 **method-blinded** A/B trials with different query
games. Each compares the best strategic and neural recommendations where those
recommendations disagree. The public trial file omits method assignments; the
offline `review-key.json` is retained in ignored run artifacts. This hides method
labels in the interface, not cryptographic access to publicly available vectors.

- The first 20 trials are for development.
- The other 80 are reserved for a final audit after freezing design decisions.
- Choose A, B, equally relevant, or neither useful.
- Compare pawn structures, activity, king safety and plausible plans.
- Use an independent chess player for the audit; record expertise honestly.
- Responses remain in browser local storage until explicitly exported.

Use `moments.html?review=1#review` for a review-only view that hides the explorer
and method diagnostics, reducing accidental exposure before judging.

No independent human judgments are included in this release. Test fixtures are
synthetic and never enter the published result files. A prepared evaluation is
not a completed relevance benchmark. The current Site's audience is preserved;
collect judgments in an authorized shared deployment or a local session.

The scoring command validates dataset versions, choices and trial partitions,
deduplicates repeated exports by anonymous browser ID and trial, and keeps the
latest timestamped response. It reports all four outcomes. For uncertainty it
averages +1 strategic / -1 neural / 0 tie-or-neither within each query game,
then bootstraps query games. Positive differences favor strategic. Anonymous
browser IDs do not prove different people; reviewer and reference-game
dependence are not fully accounted for by that interval.

~~~bash
.venv/bin/python scripts/evaluate_moment_judgments.py \
  --key results/moments_20261030/review-key.json \
  --responses /path/to/real-reviewer-export.json \
  --output results/moments_20261030/human-audit-v1.json
~~~

Do not tune the model on audit responses. The comparison is conditional on
method disagreement; it cannot estimate overall recommendation accuracy.

## Reproduce the bounded export

First reproduce the [sealed Jubail cache](atlas-jubail-runbook.md), or use its
verified local backup. Use the actual Python environment recorded in the report
for the same numerical implementation. Stockfish binaries are downloaded from
the [official 17.1 release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1).

~~~bash
# In a fresh scratch checkout with the prepared Python environment:
mkdir -p tools logs runs
curl -fL -o tools/stockfish-17.1.tar \
  https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64-avx2.tar
# Observed official Linux archive SHA256:
echo '09e953222dbe80aaea5c33dab265413b295cb8376418445c877708c61e31987d  tools/stockfish-17.1.tar' | sha256sum -c -
tar -xf tools/stockfish-17.1.tar -C tools
export ATLAS_CACHE=/path/to/sealed/full_cache
export ATLAS_PYTHON=/path/to/prepared/.venv/bin/python
sbatch scripts/jubail_moments.slurm

# Local CPU alternative (use the official Mac binary on Apple Silicon):
# Install requirements-moments.txt in a CPU environment if these packages are missing.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/build_moments.py \
  --cache results/atlas_jubail/runs/full_cache \
  --atlas website/dist/atlas.json \
  --stockfish /path/to/stockfish-macos-m1-apple-silicon \
  --output results/moments_20261030 --workers 4 --nodes 8000
~~~

Completed output directories are immutable. Use a new output path for a changed
configuration. The builder verifies all files in the cache's integrity seal.
It records source hashes, a source snapshot, transform artifacts and the measured report.

Copy only `moments.json`, `moment-review.json`, and `moment-report.json` into
`website/dist/`. Keep `review-key.json`, transforms and detailed artifacts in
ignored `results/`. Existing replay chunks must match the supplied atlas.

~~~bash
.venv/bin/python website/scripts/validate_moments.py \
  --transforms results/moments_20261030/transforms.npz
.venv/bin/python -m unittest discover -s tests -v
npm test --prefix website
~~~

This round completes the engineering scope. The next evidence requires human
judgments, not another backbone or infrastructure layer. Keep the product and
the original negative results together in the interview explanation.
