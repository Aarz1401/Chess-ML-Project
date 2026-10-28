"""Download the exact public Maia-3 5M checkpoint used in the atlas experiments."""

import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from game_identity import file_sha256

SPEC = dict(
    repo="UofTCSSLab/Maia3-5M",
    revision="b6559de2398d7140b985f28fd2c19fb5e47ddabe",
    checkpoint_sha256="ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f",
    code_revision="1e13597c42d4858b7cfd7cfdae01e297263364b2",
)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("models"))
    directory = p.parse_args().directory
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "maia3-5m.pt"
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".download")
        urllib.request.urlretrieve(
            f"https://huggingface.co/{SPEC['repo']}/resolve/{SPEC['revision']}/maia3-5m.pt",
            temporary,
        )
        if file_sha256(temporary) != SPEC["checkpoint_sha256"]:
            raise ValueError("Downloaded checkpoint checksum mismatch")
        temporary.replace(checkpoint)
    if file_sha256(checkpoint) != SPEC["checkpoint_sha256"]:
        raise ValueError("Existing checkpoint checksum mismatch")
    (directory / "manifest.json").write_text(json.dumps(SPEC, indent=2))
    print(json.dumps(SPEC))
