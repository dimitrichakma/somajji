"""Download the three WHO guides into data/sources/ and check each file's sha256.

Run: uv run python scripts/download_sources.py
The PDFs are not committed (see .gitignore); this script and sources.yaml are.
"""
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"
SOURCES_FILE = SOURCES_DIR / "sources.yaml"


class DownloadError(Exception):
    """A download failed or a file did not match its expected hash."""


@dataclass(frozen=True)
class Source:
    name: str  # short name used in metadata: pfa, mhgap, pmplus
    title: str
    url: str  # direct PDF link
    landing_page: str  # WHO page to download from by hand
    sha256: str
    licence: str


def load_sources(path: Path = SOURCES_FILE) -> list[Source]:
    entries = yaml.safe_load(path.read_text())["sources"]
    sources = []
    for entry in entries:
        try:
            sources.append(Source(**entry))
        except TypeError as error:
            raise DownloadError(f"Bad entry in {path.name} ({entry.get('name')}): {error}") from error
    return sources


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _by_hand_hint(source: Source, destination: Path) -> str:
    return (
        f"Download it by hand from {source.landing_page} "
        f"and save it as {destination}."
    )


def _fetch(source: Source, destination: Path, client: httpx.Client) -> bytes:
    try:
        response = client.get(source.url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise DownloadError(
            f"Could not download {source.name} ({error}). {_by_hand_hint(source, destination)}"
        ) from error
    return response.content


def download_one(source: Source, directory: Path, client: httpx.Client) -> Path:
    destination = directory / f"{source.name}.pdf"
    if destination.exists() and _sha256_of(destination.read_bytes()) == source.sha256:
        print(f"{source.name}: already downloaded")
        return destination

    data = _fetch(source, destination, client)
    actual = _sha256_of(data)
    if actual != source.sha256:
        raise DownloadError(
            f"{source.name}: sha256 mismatch (expected {source.sha256}, got {actual}). "
            "WHO may have updated the file. Check it, then update sha256 in sources.yaml."
        )
    destination.write_bytes(data)
    print(f"{source.name}: downloaded ({len(data) // 1024} KB)")
    return destination


def download_all(sources: list[Source], directory: Path, client: httpx.Client) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return [download_one(source, directory, client) for source in sources]


def main() -> int:
    headers = {"User-Agent": "somajji-demo/0.1 (portfolio project)"}
    try:
        with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
            download_all(load_sources(), SOURCES_DIR, client)
    except DownloadError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
