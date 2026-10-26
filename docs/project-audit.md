# Project audit and cleanup

Reviewed 2026-10-26.

## GitHub relationship

There is one `.git` directory at the repository root and one configured remote:
`origin = https://github.com/Aarz1401/Chess-ML-Project.git`.
GitHub confirms this is a public repository with default branch `main`.
At the start of cleanup, both local HEAD and live GitHub main were
`2ed5311ded6565819e94811ed497e0c1827a6207`.
Neither subfolder was a nested Git repository or a submodule. Editing either
folder contributes changes to the same repository. This cleanup is local until
committed and pushed; it does not create or rename a GitHub repository.

## Why there were two projects

| Former folder | Purpose | Representation | Destination |
| --- | --- | --- | --- |
| Aadil-Chess-Stylometry | Player embedding and retrieval | 84 channels, CNN + Transformer, contrastive loss | Repository root |
| chess-style-fingerprint | Earlier human-vs-Stockfish classification | 34 channels plus 33 style features; neural and classical models | archive/chess-style-fingerprint/ |

Their identically titled READMEs and duplicated `src/model.py` names obscured
these differences. They are not interchangeable implementations. The archived
workflow keeps its own requirements, data layout, scripts, notebook, and saved
outputs. It is not imported by the active project.

Git history includes delete/re-extract/restore commits in April, a cleanup in
May, embedding polishing in June, and residual-model/cache/Jubail changes in
July. These are generations of research retained together, rather than a
technical requirement to maintain two peer projects.

## Migration map

| Old path | New path |
| --- | --- |
| Aadil-Chess-Stylometry/src/ | src/ |
| Aadil-Chess-Stylometry/scripts/ | scripts/ |
| Aadil-Chess-Stylometry/PGN_data/ | data/pgn/ |
| Aadil-Chess-Stylometry/notebooks/ | notebooks/ |
| Aadil-Chess-Stylometry/README.md | docs/workflow.md |
| Aadil-Chess-Stylometry/requirements.txt | requirements.txt |
| Aadil-Chess-Stylometry/results/ | results/ |
| Aadil-Chess-Stylometry/deliverables/ | deliverables/ |
| Aadil-Chess-Stylometry/.venv/ | .venv/ |
| chess-style-fingerprint/ | archive/chess-style-fingerprint/ |

Local private notes and plotting cache also moved to the root and remain
ignored. Installed environment launcher/activation paths were repaired for the
new location; new machines should create their own environment. Saved run
configs and notebook outputs retain historical paths as provenance. Rebuild
caches or explicitly select new paths when rerunning old commands.

The tracked macOS `.DS_Store` files were removed; a root ignore policy now covers
OS files, environments, bytecode, checkpoints, and generated active results.
The archived CSVs and figures remain versioned. Archived raw data is local and
ignored, so a fresh clone alone cannot reproduce that older workflow.

## Findings to address after organization

1. **Trustworthy splits first.** Current cached splitting shuffles records
   independently per player. It does not group by canonical game identity.
   The same game can occur in multiple player PGNs or repeat in one file.
   Hash and deduplicate games, group both perspectives into the same split,
   and measure overlap before interpreting scores.
2. **Separate historical evidence from current performance.** The local
   `results/poc_tal_carlsen_embedding_500/retrieval_metrics.json` supports the
   historical 76% top-1 / 85% top-5 / 82% MAP claim. The current residual tower
   differs from that earlier model. Reproduce a versioned baseline before
   claiming improvements or assuming older checkpoints still load.
3. **Make cache/checkpoint contracts explicit.** The launcher treats a manifest
   as proof that a cache is reusable. Validate schema, complete arrays, input
   hashes, labels, sequence length, and preprocessing configuration. Training
   loads labels from the cache but records CLI preprocessing arguments, which
   can disagree with the cache and affect later export/inference.
4. **Control the research question.** Use same-era players and matched openings,
   ratings, and time controls. Compare simple baselines, report several seeds
   and uncertainty, and keep the final test set untouched during selection.
   The archived notebook contains test-based ranking/epoch summaries; those
   historical comparisons should not be treated as unbiased model selection.
5. **Then scale on Jubail.** First run a small scheduled CPU/GPU pilot, measure
   memory and runtime, then expand. At 60 plies, 84 planes, and float16, the
   sequence cache costs about 0.615 MiB per game, up to 19.2 GiB for 32,000
   games before overhead and temporary compaction files. The actual balanced
   count is capped by the smallest player's available dataset.
6. **Reproducibility and maintainability.** Pin a validated environment, add
   focused tests for tensors/splits/cache/checkpoint round trips, and consolidate
   shared checkpoint loading. These belong to the next development phase;
   the cleanup leaves model training behavior unchanged.

## Cleanup verification

Passed locally:

- Integrity comparison: 275 files unchanged immediately after relocation;
  no missing research files. Documentation, ignore rules, and path references
  were the intentional edits at that point.
- All 92 preserved versioned files exist at their new paths and are not ignored.
- Python syntax for 24 source/scripts, shell syntax for all three job scripts,
  notebook JSON, relative documentation links, and all nine active CLI help commands.
- A 20-game CPU smoke run: 16 train / 2 validation / 2 test, one epoch,
  saved checkpoints, retrieval metrics, standalone aggregated evaluation and figures.
- Existing local environment Python and pip work at their new location.

Standalone evaluation initially failed on the two-game smoke split because its
t-SNE fallback required perplexity of at least 5. The plotting function now uses
PCA for fewer than four points and a valid smaller perplexity otherwise. This
is a plotting fix only; it does not change training or evaluation metrics.
Full training, archived Stockfish experiments, and live Jubail execution are
outside this organization pass.
