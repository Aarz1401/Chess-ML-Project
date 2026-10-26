"""Header-only corpus audit before representation learning; no game parsing/cache writes.

This counts records, not unique games. Canonical move-sequence deduplication is
still required before splitting. No player identities beyond collection labels
are written to the report.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def matches(name: str, label: str) -> bool:
    key = normalized(label)
    return bool(key) and (key in re.findall(r"[a-z0-9]+", name.lower()) or key == normalized(name))


def audit(data_dir: Path) -> dict:
    rows = []
    for path in sorted(data_dir.glob("*.pgn")):
        row = dict(label=path.stem, records=0, missing_either_rating=0,
                   missing_time_control=0, missing_date=0,
                   target_absent_or_ambiguous=0, legacy_side_disagreements=0)
        with path.open(encoding="utf-8", errors="replace") as source:
            while (headers := chess.pgn.read_headers(source)) is not None:
                row["records"] += 1
                ratings = [headers.get(k, "") for k in ("WhiteElo", "BlackElo")]
                row["missing_either_rating"] += any(not value.isdigit() for value in ratings)
                row["missing_time_control"] += headers.get("TimeControl", "") in ("", "?", "-")
                row["missing_date"] += not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", headers.get("Date", ""))
                white, black = headers.get("White", ""), headers.get("Black", "")
                sides = [color for name, color in ((white, chess.WHITE), (black, chess.BLACK)) if matches(name, path.stem)]
                if len(sides) != 1:
                    row["target_absent_or_ambiguous"] += 1
                    continue
                legacy = (chess.WHITE if normalized(path.stem) in normalized(white)
                          else chess.BLACK if normalized(path.stem) in normalized(black) else None)
                row["legacy_side_disagreements"] += legacy != sides[0]
        rows.append(row)
    if not rows:
        raise ValueError(f"No PGNs found in {data_dir}")
    return dict(audit_kind="headers_only_not_deduplicated", collections=rows,
                total_records=sum(row["records"] for row in rows),
                max_balanced_records_before_filtering=min(row["records"] for row in rows)*len(rows),
                deduplication_checked=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "pgn")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_dir)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
