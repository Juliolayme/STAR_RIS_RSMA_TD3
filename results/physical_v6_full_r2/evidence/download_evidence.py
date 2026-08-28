from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "EVIDENCE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "v6-evidence/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(target)


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    available = payload["assets"]
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify the permanent physical V6 r2 evidence"
    )
    parser.add_argument(
        "--asset",
        action="append",
        choices=sorted(available),
        help="Asset key to fetch; repeat as needed. Defaults to all four.",
    )
    parser.add_argument(
        "--destination", type=Path, default=HERE / "downloaded"
    )
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    selected = args.asset or list(available)
    args.destination.mkdir(parents=True, exist_ok=True)
    for key in selected:
        item = available[key]
        target = args.destination / item["file"]
        if not target.is_file() or sha256(target) != item["sha256"]:
            print(f"Downloading {key}: {item['download_url']}", flush=True)
            download(item["download_url"], target)
        observed = sha256(target)
        if observed != item["sha256"]:
            raise RuntimeError(
                f"{target}: SHA-256 {observed} != expected {item['sha256']}"
            )
        print(f"PASS {key}: {observed}", flush=True)
        if args.extract:
            safe_extract(target, args.destination / target.stem)


if __name__ == "__main__":
    main()
