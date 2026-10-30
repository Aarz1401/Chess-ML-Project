"""Score exported blinded judgments with query-game grouped uncertainty.

Run only after collecting real exports; this script never creates responses.
The method key is an offline artifact and is not included in the website.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from game_identity import file_sha256


def evaluate(key, exports):
    latest = {}
    for export in exports:
        if export.get("schemaVersion") != 1 or export.get("version") != key["version"]:
            raise ValueError("Judgment dataset version mismatch")
        browser = export.get("browserId")
        if not isinstance(browser, str) or not 1 <= len(browser) <= 100:
            raise ValueError("Missing anonymous reviewer ID")
        for response in export["responses"]:
            trial_id = response["trialId"]
            if trial_id not in key["trials"] or response["choice"] not in [
                "A",
                "B",
                "tie",
                "neither",
            ]:
                raise ValueError("Unknown trial or invalid choice")
            if response["partition"] != key["trials"][trial_id]["partition"]:
                raise ValueError("Trial partition mismatch")
            date = datetime.fromisoformat(response["recordedAt"].replace("Z", "+00:00"))
            if date.tzinfo is None:
                raise ValueError("Judgment timestamp needs a timezone")
            identity = (browser, trial_id)
            if identity not in latest or date >= latest[identity][0]:
                latest[identity] = (date, response)
    result = {}
    for partition in ["development", "audit"]:
        counts = Counter()
        scores = defaultdict(list)
        raters = set()
        for (browser, trial_id), (_, r) in latest.items():
            trial = key["trials"][trial_id]
            if trial["partition"] != partition:
                continue
            outcome = trial[r["choice"]] if r["choice"] in ["A", "B"] else r["choice"]
            counts[outcome] += 1
            scores[trial["query_game"]].append(
                {"strategic": 1, "neural": -1, "tie": 0, "neither": 0}[outcome]
            )
            raters.add(browser)
        means = np.asarray([np.mean(s) for _, s in sorted(scores.items())])
        ci = None
        if len(means) >= 5:
            rng = np.random.default_rng(20261030)
            estimates = rng.choice(means, (2000, len(means)), replace=True).mean(1)
            ci = np.quantile(estimates, [0.025, 0.975]).tolist()
        result[partition] = dict(
            responses=sum(counts.values()),
            anonymous_browser_ids=len(raters),
            query_games=len(scores),
            outcomes={k: counts[k] for k in ["strategic", "neural", "tie", "neither"]},
            grouped_preference_difference=float(means.mean()) if len(means) else None,
            ci95=ci,
        )
    return dict(
        schemaVersion=1,
        version=key["version"],
        partitions=result,
        interpretation="Positive difference favors strategic; negative favors neural. Tie/neither contribute zero. Average per query game, then bootstrap query games.",
        limitations=[
            "Conditional on top-match disagreement, not overall recommendation accuracy.",
            "Anonymous browser IDs do not prove different people or independent expertise.",
            "Repeated reviewers and shared reference games induce dependence not captured by query-only bootstrap.",
            "Use development judgments for iteration; do not tune on the reserved audit.",
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--responses", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Retain previous results; choose a new output file")
    report = evaluate(
        json.loads(args.key.read_text()),
        [json.loads(f.read_text()) for f in args.responses],
    )
    report["source_hashes"] = {
        str(f): file_sha256(f) for f in [args.key, *args.responses]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
