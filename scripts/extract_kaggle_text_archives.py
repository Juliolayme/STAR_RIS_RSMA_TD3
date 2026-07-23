from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath


ALLOWED_SUFFIXES = {".csv", ".json", ".txt", ".log", ".yaml", ".yml", ".md"}
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_DEPTH = 4


def safe_name(name: str) -> Path:
    parts = [part for part in PurePosixPath(name).parts if part not in {"", ".", ".."}]
    return Path(*parts)


def write_member(data: bytes, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "path": destination.as_posix(),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def extract_zip_bytes(
    payload: bytes,
    source_label: str,
    output: Path,
    records: list[dict[str, object]],
    depth: int,
) -> None:
    if depth > MAX_DEPTH:
        raise RuntimeError(f"Nested ZIP depth exceeds {MAX_DEPTH}: {source_label}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                continue
            member = safe_name(info.filename)
            suffix = member.suffix.lower()
            data = archive.read(info)
            if suffix == ".zip":
                nested_label = f"{source_label}__{member.as_posix().replace('/', '__')}"
                extract_zip_bytes(data, nested_label, output, records, depth + 1)
            elif suffix in ALLOWED_SUFFIXES:
                target = output / source_label / member
                record = write_member(data, target)
                record.update({"archive": source_label, "member": info.filename, "depth": depth})
                records.append(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    zip_paths = sorted(args.input.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No ZIP files found below {args.input}")

    for zip_path in zip_paths:
        label = zip_path.relative_to(args.input).as_posix().replace("/", "__")
        extract_zip_bytes(zip_path.read_bytes(), label, args.output, records, depth=0)

    (args.output / "EXTRACTION_INDEX.json").write_text(
        json.dumps({"archives": len(zip_paths), "files": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Extracted {len(records)} compact files from {len(zip_paths)} ZIP files")


if __name__ == "__main__":
    main()
