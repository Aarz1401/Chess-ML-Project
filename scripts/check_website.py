"""Validate the static Pages artifact, including relative links below a project URL."""

from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / "website/dist"
BASE = "https://example.github.io/Chess-ML-Project/"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            if attributes["id"] in self.ids:
                raise ValueError(f"Duplicate element ID: {attributes['id']}")
            self.ids.add(attributes["id"])
        for name in ("href", "src"):
            if name in attributes:
                self.links.append(attributes[name])


def main():
    pages = {}
    for path in ROOT.glob("*.html"):
        page = Page()
        page.feed(path.read_text())
        pages[path.name] = page
    for name, page in pages.items():
        for link in page.links:
            url = urlsplit(urljoin(BASE + name, link))
            if url.netloc != "example.github.io" or url.scheme != "https":
                continue
            if not url.path.startswith("/Chess-ML-Project/"):
                raise ValueError(f"Link escapes the Pages project path: {name}: {link}")
            relative = (
                unquote(url.path.removeprefix("/Chess-ML-Project/")) or "index.html"
            )
            target = ROOT / relative
            if not target.is_file():
                raise ValueError(f"Missing website asset: {name}: {link}")
            if (
                url.fragment
                and relative in pages
                and url.fragment not in pages[relative].ids
            ):
                raise ValueError(f"Missing anchor: {name}: {link}")
    for color in "wb":
        for piece in "KQRBNP":
            root = ET.parse(ROOT / f"assets/pieces/{color}{piece}.svg").getroot()
            if not root.tag.endswith("svg"):
                raise ValueError("Invalid piece SVG")
    atlas = json.loads((ROOT / "atlas.json").read_text())
    moments = json.loads((ROOT / "moments.json").read_text())
    review = json.loads((ROOT / "moment-review.json").read_text())
    report = json.loads((ROOT / "moment-report.json").read_text())
    if not moments["version"] == review["version"] == report["version"]:
        raise ValueError("Position asset versions disagree")
    for game in atlas["games"] + moments["games"]:
        if not (ROOT / game["replayChunk"]).is_file():
            raise ValueError(f"Replay missing for {game['id']}")
    if (ROOT / "review-key.json").exists():
        raise ValueError("The offline review key must not be published")
    print(
        f"PASS: {len(pages)} pages, local links, project path, 12 SVG pieces and dataset assets"
    )


if __name__ == "__main__":
    main()
