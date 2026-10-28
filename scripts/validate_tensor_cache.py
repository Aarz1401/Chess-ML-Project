"""Validate a versioned, grouped tensor cache before reusing it."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensor_cache import validate_cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    manifest = validate_cache(args.cache_dir, args.data_dir)
    print(f"Validated schema {manifest['schema_version']} cache: {manifest['counts']}")
