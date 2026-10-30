# Demo data provenance

The bundled `dist/atlas.json` is generated from the eight `data/pgn/*.pgn`
files already versioned in this repository. The source revision before this
website was added is `d356d84`. Game headers, moves, and results are preserved;
comments and side variations are omitted. The current atlas displays 1,000 unique games, drawn from the grouped corpus.
See the public corpus manifest and experiment report in dist/.

`dist/moments.json` and `dist/moment-review.json` derive positions and blind
comparisons from that same selected collection. `dist/moment-report.json`
contains this project's measured diagnostics, with zero independent human
judgments included. Engine annotations come from the official Stockfish 17.1
release at a fixed 8,000-node budget. See
[the position protocol](../docs/moment-matching.md).

The original download URLs and collection-specific redistribution terms were
not recorded in the repository. This document does not assert an open data
license. The MIT license for the website covers the new software only.

To replace the demo with your own collection:

```bash
.venv/bin/python website/scripts/build_atlas.py --data-dir /path/to/pgns
```

Use one PGN file per collection. Each file's stem becomes its display label.
The browser can work with different counts; the current color palette cycles
after eight collections. Rebuild and validate before hosting the replacement.

Before a broader dataset release, add original source URLs, attribution,
collection dates, and applicable terms here. Do not infer a data license from
the repository's public visibility.
