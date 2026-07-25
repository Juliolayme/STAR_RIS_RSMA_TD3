#!/usr/bin/env python3
"""Download only legally open reference PDFs declared in the thesis library.

The inventory deliberately leaves paywalled papers without a ``pdf_url``. Those
entries are reported as metadata-only rather than being fetched from unofficial
mirrors. Every successful PDF is checked for the PDF magic header and recorded
with SHA-256 provenance.

For PubMed Central articles, an inventory value such as ``pmc-oa:PMC11397873``
uses NCBI's official Open Access Web Service, downloads the declared OA package,
and extracts the PDF from that package. This avoids browser challenge pages and
keeps the source traceable to the public NCBI archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


USER_AGENT = (
    "Mozilla/5.0 (compatible; STAR-RIS-RSMA-thesis-reference-downloader/1.0; "
    "+https://github.com/Juliolayme/STAR_RIS_RSMA_TD3)"
)
PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_bytes(
    url: str,
    *,
    accept: str,
    timeout: int = 120,
) -> tuple[bytes, str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return (
            response.read(),
            response.headers.get("Content-Type", ""),
            response.geturl(),
        )


def write_validated_pdf(
    data: bytes,
    destination: Path,
    *,
    source: str,
    content_type: str,
    attempt: int,
) -> dict[str, object]:
    if not data.startswith(b"%PDF-"):
        prefix = data[:120].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Downloaded content is not a PDF; content-type={content_type!r}, "
            f"prefix={prefix!r}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return {
        "status": "downloaded",
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "content_type": content_type,
        "resolved_source": source,
        "attempt": attempt,
    }


def download(url: str, destination: Path, retries: int = 4) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            data, content_type, resolved_url = request_bytes(
                url,
                accept="application/pdf,*/*;q=0.8",
            )
            return write_validated_pdf(
                data,
                destination,
                source=resolved_url,
                content_type=content_type,
                attempt=attempt,
            )
        except (OSError, RuntimeError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(min(20, 2**attempt))

    assert last_error is not None
    raise RuntimeError(f"Failed after {retries} attempts: {url}: {last_error}")


def _https_ftp_url(url: str) -> str:
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return "https://ftp.ncbi.nlm.nih.gov/" + url.removeprefix(
            "ftp://ftp.ncbi.nlm.nih.gov/"
        )
    return url


def download_pmc_oa_package(
    pmcid: str,
    destination: Path,
    retries: int = 4,
) -> dict[str, object]:
    """Resolve and extract one PDF from an official PMC OA package."""

    normalized = pmcid.strip().upper()
    if not normalized.startswith("PMC") or not normalized[3:].isdigit():
        raise ValueError(f"Invalid PMCID: {pmcid!r}")

    api_url = PMC_OA_API.format(pmcid=normalized)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            xml_data, xml_type, resolved_api = request_bytes(
                api_url,
                accept="application/xml,text/xml,*/*;q=0.5",
            )
            root = ET.fromstring(xml_data)
            error = root.find(".//error")
            if error is not None:
                raise RuntimeError(
                    f"PMC OA API error for {normalized}: "
                    f"{''.join(error.itertext()).strip()}"
                )

            package_links = [
                element.attrib.get("href", "").strip()
                for element in root.findall(".//link")
                if element.attrib.get("format", "").lower() in {"tgz", "tar.gz"}
                and element.attrib.get("href")
            ]
            if not package_links:
                raise RuntimeError(
                    f"PMC OA API returned no tgz package for {normalized}; "
                    f"content-type={xml_type!r}, endpoint={resolved_api}"
                )

            package_url = _https_ftp_url(package_links[0])
            package_data, package_type, resolved_package = request_bytes(
                package_url,
                accept="application/gzip,application/x-gzip,application/octet-stream,*/*;q=0.5",
            )

            with tarfile.open(fileobj=io.BytesIO(package_data), mode="r:gz") as archive:
                pdf_members = sorted(
                    (
                        member
                        for member in archive.getmembers()
                        if member.isfile() and member.name.lower().endswith(".pdf")
                    ),
                    key=lambda member: member.size,
                    reverse=True,
                )
                if not pdf_members:
                    raise RuntimeError(
                        f"PMC OA package contains no PDF: {resolved_package}"
                    )
                extracted = archive.extractfile(pdf_members[0])
                if extracted is None:
                    raise RuntimeError(
                        f"Could not read PDF member {pdf_members[0].name!r} "
                        f"from {resolved_package}"
                    )
                pdf_data = extracted.read()

            result = write_validated_pdf(
                pdf_data,
                destination,
                source=resolved_package,
                content_type=package_type,
                attempt=attempt,
            )
            result.update(
                {
                    "pmcid": normalized,
                    "pmc_oa_api": resolved_api,
                    "package_member": pdf_members[0].name,
                }
            )
            return result
        except (
            OSError,
            RuntimeError,
            tarfile.TarError,
            urllib.error.URLError,
            ValueError,
            ET.ParseError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(20, 2**attempt))

    assert last_error is not None
    raise RuntimeError(
        f"Failed after {retries} attempts to retrieve PMC OA package "
        f"for {normalized}: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("references/open_access_pdfs.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reference_library/pdfs"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reference_library/DOWNLOAD_MANIFEST.json"),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed downloads instead of exiting immediately.",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.inventory.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"key", "filename", "pdf_url", "landing_url", "access_note"}
        if set(reader.fieldnames or []) != required:
            raise RuntimeError(
                f"Unexpected inventory columns: {reader.fieldnames}; expected {sorted(required)}"
            )
        rows = list(reader)

    if len(rows) < 25:
        raise RuntimeError(f"Reference inventory is unexpectedly small: {len(rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for index, row in enumerate(rows, start=1):
        key = row["key"].strip()
        filename = row["filename"].strip()
        pdf_url = row["pdf_url"].strip()
        destination = args.output_dir / filename

        base: dict[str, object] = {
            "index": index,
            "key": key,
            "filename": filename,
            "pdf_url": pdf_url or None,
            "landing_url": row["landing_url"].strip(),
            "access_note": row["access_note"].strip(),
        }

        if not pdf_url:
            base["status"] = "metadata_only"
            report_rows.append(base)
            print(f"[{index:02d}/{len(rows)}] metadata only: {key}")
            continue

        print(f"[{index:02d}/{len(rows)}] download: {key} -> {destination}")
        try:
            if pdf_url.lower().startswith("pmc-oa:"):
                pmcid = pdf_url.split(":", 1)[1].strip()
                result = download_pmc_oa_package(pmcid, destination)
            else:
                result = download(pdf_url, destination)
            base.update(result)
        except Exception as exc:  # noqa: BLE001 - report exact external failure
            base["status"] = "failed"
            base["error"] = repr(exc)
            failures.append(key)
            if not args.continue_on_error:
                report_rows.append(base)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(
                    json.dumps(
                        {
                            "inventory": str(args.inventory),
                            "references": report_rows,
                            "failures": failures,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise
        report_rows.append(base)

    summary = {
        "inventory": str(args.inventory),
        "reference_count": len(report_rows),
        "downloaded_count": sum(
            row.get("status") == "downloaded" for row in report_rows
        ),
        "metadata_only_count": sum(
            row.get("status") == "metadata_only" for row in report_rows
        ),
        "failed_count": sum(row.get("status") == "failed" for row in report_rows),
        "failures": failures,
        "references": report_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "reference_count",
                    "downloaded_count",
                    "metadata_only_count",
                    "failed_count",
                )
            },
            indent=2,
        )
    )

    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
