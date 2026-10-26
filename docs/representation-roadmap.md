# Better game representations and a Jubail experiment plan

Investigated 2026-10-26. This is a proposed experiment sequence, not a report
of trained-model results. Existing website representations remain unchanged.

## Recommendation

Keep the 288-dimensional strategic fingerprint as the interpretable baseline.
First compare it against a frozen pretrained chess encoder. Then train a small
sequence encoder over the cached position/decision embeddings, if the frozen
baseline is useful. Use one A100 for the pilot, followed by independent runs
across seeds. Larger models and multiple GPUs come after measured gains.

The goal is to retrieve games with similar chess content and decision patterns.
The current neural trainer instead pulls every pair of games by the same player
together. That can support player identification, but it is not an adequate
similarity definition for this atlas.

## Current evidence from this checkout

- `website/scripts/game_vectors.py` computes 96 features and averages three
  chronological segments. This loses ordering inside each segment, variation
  around the mean, fine square relationships, and interactions between concepts.
- `src/model.py` already provides a residual CNN, sequence Transformer, attention
  pooling, and a normalized metric head. `scripts/train_multiclass.py` optimizes
  player cross-entropy plus same-player supervised contrastive loss.
- The training cache includes 84 × 8 × 8 planes per ply. Some channels repeat
  summaries from the whole recorded sequence. If training on cropped views,
  these summaries must be recomputed from the visible view; otherwise the
  view still contains information about moves that were supposedly removed.
- Header audit: 36,529 PGN records across eight player collections. They are
  records, not deduplicated games. The smallest collection is Tal (2,431), so
  the current equal-player balancing caps the corpus at 19,448 records before
  filtering, despite the 4,000-per-player Slurm setting.
- Every record lacks a usable `TimeControl` header. Some dates and ratings are
  incomplete. We cannot honestly claim a time-control-matched benchmark using
  these PGNs alone.
- The legacy training-side substring matcher disagrees with exact token matching
  on eight unambiguous headers (three Anand, five Tal). The website has the
  tighter matcher; the training and cache code still need the equivalent fix.
- Nine records have absent or ambiguous target matches under that audit policy.
  These need inspection or exclusion before producing training labels.
- Export metadata currently contains only row index, split, and label. It cannot
  reliably join to the website's canonical game IDs. Fix this before integrating
  any model trained by the current scripts.

Reproduce the header audit:

```bash
.venv/bin/python scripts/audit_pgn_corpus.py --output results/corpus_audit.json
```

This audit deliberately does not parse moves or assert zero duplicate leakage.

## Representation experiments

| ID | Representation | Purpose | Compute |
| --- | --- | --- | --- |
| R0 | Existing 288D handcrafted fingerprint | Explainable baseline | CPU |
| R1 | Frozen pretrained chess embeddings, pooled over sampled target decisions | Test whether pretrained chess knowledge helps | GPU inference |
| R2 | Same frozen encoder + small trainable temporal aggregator, 128D output | Learn which positions and decisions matter across a game | Single-GPU training |
| R3 | Unfreeze a small part of the position encoder, or compare a larger backbone | Improve a validated R2 | Larger single-GPU experiments |

**R1 candidate: Maia-3 5M.** Its official project releases 5M, 23M and 79M
human-move prediction models. Start with the 5M checkpoint; the maintained
Maia-2 project itself recommends Maia-3 for new projects. This recommendation
is an engineering hypothesis, not evidence its vectors already solve game
similarity. Maia-2 is an alternative if its activation tooling is more useful.

For each target decision after the opening, encode the **pre-move board with
its real preceding history**, and retain the chosen move in that same board
orientation. Start by comparing pooled board-square activations with features
of the selected from/to squares. A vector of expected position features alone
cannot distinguish two players who reach the same position and choose different
moves. The upstream Maia-3 model exposes square-token activations before its
policy heads; an adapter or forward hook is needed to export them.

Pool a fixed number of decisions across the game, preserving ply indices and
material phase. Compare mean pooling against mean plus dispersion and ordered
segment pooling. Whiten or reduce dimensions using training games only. Keep
canonical game IDs and target color in metadata, not as network input.

Maia conditions on player/opponent rating. Compare a fixed shared rating setting
against actual ratings where available, recording the setting in the manifest.
A fixed setting avoids explicitly encoding Elo differences, but does not prove
that the representation is independent of strength. Match or stratify the
benchmark by rating as a separate control. Missing historic ratings are not
reliable values to invent.

**R2: a small temporal model.** Feed the cached decision embeddings to a two-layer
Transformer with position information and attention pooling, then project to a
normalized 128D game vector. Initially freeze the position encoder so this step
trains quickly and its contribution is easy to isolate. Keep the current CNN
model as an independent comparison rather than assuming it must be replaced.

Use a combination of:

- Paired, legal views of the same observed game: modest position subsampling or
  token masking, with board states/history left intact. Preserve enough overlap
  and context to represent the same game. Do not shuffle moves, invent moves,
  or apply image flips without transforming chess rules and move orientation.
- Auxiliary chess-concept targets: pawn structure, king exposure, material
  transitions, and tactical events. Weak labels are training signals, not
  independent evidence the resulting metric understands those concepts.
- A small, judged set of cross-game similarity pairs/triplets to tune or assess
  semantic alignment. Reserve an untouched subset for the final comparison.

Same-game contrastive learning alone can memorize openings or game identity;
it can also treat genuinely similar games as negatives. Compare it with
concept-only training, inspect false negatives, and avoid treating same-player
labels as the main strategic-similarity target. Start with a small ablation
matrix instead of combining every objective at once.

A later behavior-specific representation could encode how chosen moves differ
from a rating-conditioned policy's expectations. That is a separate style
hypothesis and requires consistent move probabilities, rating controls, and
held-out players; it should not be called validated style from the outset.

## Data and evaluation before scale

1. Hash canonical initial FEN + legal UCI sequence. Group duplicate games,
   both player perspectives, and all generated views before any train/val/test
   split. Persist IDs, PGN provenance and exact target matches.
2. Fix the training-side name matcher and reject malformed/ambiguous records.
   Version cache schemas and include source hashes, preprocessing settings,
   backbone commit/checkpoint hashes, rating settings, and tokenizer versions.
3. Keep the 240 displayed games as a browsing set, not a selection benchmark.
   Use grouped held-out games for retrieval evaluation. Exclude self matches
   and alternate perspectives of the same game from the candidate index.
4. Build an initial 100–200 judged triplets: query, more related game, less
   related game. Hide names and representation sources while judging. Include
   examples matched on opening/rating where both fields are available and
   different-opening examples with shared structures/plans. Allow “uncertain”;
   report agreement and skip rate, not only a forced preference score.
5. Score triplet agreement, graded relevance at k=5/10, and confidence intervals
   bootstrapped by query/game. Keep game-level groups intact in resampling.
   Concept probes and same-player retrieval are secondary diagnostics.
6. Test shortcuts: opening family, rating band, game length, color, era. Run
   opening-held-out and player-held-out evaluations as distinct stress tests.
   A model can encode these properties legitimately; the question is whether
   it retrieves meaningful games after matching those obvious cues.
7. Report map fidelity separately. Use the same projection settings across
   representations when comparing them, inspect several projection seeds,
   and compute retrieval in full-dimensional space. Do not optimize for
   visually separated player colors.

For expansion, use a bounded, filtered sample of Lichess's CC0 game exports
with recorded source month and metadata. A proposed next scale is 50,000–100,000
same-era standard games in a chosen time-control/rating band, after the pilot.
Start with a streamed bounded sample, not an unbounded archive download.
Pretrained-model training overlap with public games may be unknown; record this
and prefer post-training-date evaluation when a verified cutoff is available.

## Jubail: verified readiness versus proposed requests

Read-only checks through the existing `jubail` SSH alias succeeded. A login
shell (`bash -lc`) is needed to load Slurm into the noninteractive SSH PATH.
The account has the `students` association on `nvidia` with `nvidias` QOS.
`python/3.11.3` and CUDA modules including `cuda/12.2.0` are available. A100,
H100, H200 and V100 resource types are present. These are visible resources,
not a guarantee of an immediate allocation; the inspected GPU nodes were busy
or otherwise not fully idle.

Scratch usage was about 402 GB of a 5 TB quota. No `$SCRATCH/chess_style`
workspace existed. Archive file-count quota was already exceeded: preserve
selected runs as a small number of packed archives rather than many files.
This is a point-in-time snapshot, not a capacity reservation.

CRC requires data and job execution on scratch; compute work belongs in Slurm
jobs, not the login node. The previous launcher does not enforce a scratch
working directory and silently falls back when module loading fails. Its
`python/3.11` module name does not match the verified `python/3.11.3`. Correct
those before reusing it for long runs. It also reuses a cache based only on the
presence of a manifest, so do not treat it as an atlas experiment launcher yet.

Proposed stages and **resource requests**, not runtime predictions:

| Stage | Proposed request | Work and exit condition |
| --- | --- | --- |
| Environment probe | 1 A100, 4 CPUs, 16 GB RAM, 15 min limit | PyTorch detects CUDA; tiny finite matmul passes |
| Extraction pilot | 1 A100, 4–8 CPUs, 32 GB RAM, 1 hour limit | About 1,000 games; measure positions/sec, peak VRAM and IO |
| R1 extraction | 1 A100 per shard, 8 CPUs, 32 GB RAM | Size walltime from the pilot; reuse cached embeddings |
| R2 pilot | 1 A100, 4–8 CPUs, 32 GB RAM, 1 hour limit | Small batch, finite loss, validation/export/checkpoint round trip |
| Controlled runs | One GPU per run, three seeds | Compare agreed objectives and baselines before scaling |

CPU parsing, deduplication and optional engine annotations are separate CPU
jobs. Keep GPU jobs focused on batched neural inference/training. Do not start
with multi-node distributed training; independent runs use the resources more
simply and provide uncertainty estimates. Later allocation sizes depend on
measured throughput and account limits.

If a cached decision vector has 256 float16 values and there are 48 decisions
per game, it costs about 24 KiB/game, or 2.29 GiB for 100,000 games before
metadata. This is a planning example; actual backbone dimensions and chosen
move features determine the real cache size. The existing 84-plane, 60-ply
cache costs 0.615 MiB/game and about 19.2 GiB for 32,000 records before overhead.

Use a fresh environment and pin its validated package versions. Download the
chosen public checkpoint once, then use explicit offline cache paths in jobs.
Record the upstream code and checkpoint revisions. Maia-3's source repository
is AGPL-3.0; retain dependency notices and inspect the selected weights' terms.
The existing website MIT license does not replace upstream licenses.

## Prepared environment probe

`scripts/jubail_gpu_probe.slurm` is a small CUDA readiness job. It does not
train a model, download weights, or implement the proposed R1/R2 experiments.
It has been syntax-checked locally, but not submitted or GPU-tested.

Once the committed project is copied/cloned into a scratch checkout and a
CUDA-enabled PyTorch environment exists there:

```bash
ssh jubail
cd "$SCRATCH/Chess-ML-Project"
module load python/3.11.3
# Create and populate .venv-jubail using a validated CUDA-enabled environment.
mkdir -p logs
sbatch scripts/jubail_gpu_probe.slurm
```

The script requests the verified account/QOS and one A100. Command-line sbatch
flags can override them. Passing this probe is a prerequisite for timing the
real encoder; it is not a throughput benchmark. No jobs were submitted during
this investigation, and existing jobs were left untouched.

## Deliverables for the next implementation phase

- A deduplicated, grouped corpus manifest with exact target matching.
- A pinned frozen-encoder extraction adapter and content-addressed cache.
- R1 vectors for the same website game IDs, shown as a third selectable mode.
- A retrieval benchmark with an untouched judged test subset.
- A small R2 trainer with resume support, full preprocessing provenance and
  peak-memory/throughput logging; then three-seed Slurm experiments.
- A static website export with embeddings and optional neighbors. Jubail is
  an offline compute resource; the deployed website needs no live HPC connection.

## Primary references

Verified 2026-10-26:

- [Maia-3 official repository and released models](https://github.com/CSSLab/maia3)
- [Maia-3 square-token encoder and policy heads](https://github.com/CSSLab/maia3/blob/main/maia3/models.py)
- [Maia-2 maintained documentation](https://github.com/CSSLab/maia2)
- [SimCLR: contrastive learning and the importance of augmentations](https://arxiv.org/abs/2002.05709)
- [Lichess open database and CC0 terms](https://database.lichess.org/)
- [CRC GPU requests and partitions](https://crc-docs.abudhabi.nyu.edu/hpc/training/quick_start.html)
- [CRC storage and scratch execution policy](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)
- [CRC system hardware](https://crc-docs.abudhabi.nyu.edu/hpc/training/system.html)
- [CRC login-node compute policy](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html)
