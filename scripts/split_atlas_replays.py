"""Keep map data small and fetch replay chunks only when a game is selected."""

import argparse
import json
from pathlib import Path


def split_replays(directory: Path, chunk_size: int = 25):
    path = directory / "atlas.json"
    data = json.loads(path.read_text())
    replays = directory / "replays"
    replays.mkdir(exist_ok=True)
    if any("replayChunk" in g for g in data["games"]):
        raise ValueError("Atlas is already packaged with lazy replays")
    for start in range(0, len(data["games"]), chunk_size):
        chunk = {}
        relative = f"replays/{start//chunk_size:03d}.json"
        for game in data["games"][start : start + chunk_size]:
            game["plies"] = len(game["moves"])
            chunk[game["id"]] = {key: game.pop(key) for key in ["pgn", "moves", "fens"]}
            game["replayChunk"] = relative
        (directory / relative).write_text(json.dumps(chunk, separators=(",", ":")))
    data["delivery"] = dict(replays="lazy_chunks", chunkSize=chunk_size)
    path.write_text(json.dumps(data, separators=(",", ":")))
    return dict(
        map_bytes=path.stat().st_size, replay_chunks=len(list(replays.glob("*.json")))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(split_replays(p.parse_args().directory)))
