# Reference Library Download Status

- Curated references: **30**
- Verified open PDFs: **27**
- Metadata/full-text-only papers: **3**
- Failed downloads: **0**
- Validation workflow: `Build open-access thesis reference library`
- Validation run ID: `30143092170`
- Validation job conclusion: `success`
- Artifact ID: `8615114985`
- Artifact digest: `sha256:797c626dfd2547d7d01f222e079c426e2cc269e6583ff11c9f858a9891f4d68b`
- Artifact size: `47,166,045` bytes as stored by GitHub Actions
- Uncompressed PDF payload: `55,056,154` bytes

All 27 downloaded files were independently checked for the `%PDF-` magic header. The generated `DOWNLOAD_MANIFEST.json` contains the SHA-256 digest and source URL for every downloaded paper.

The three entries without bundled PDFs are:

1. `Meng2024PPOSTARRSMA` — publisher/institutional metadata retained; publisher PDF not redistributed.
2. `Liu2024DiscretePhaseRSMARate` — official open full text retained through PubMed Central; no stable direct PDF endpoint was available to CI.
3. `Maghrebi2024CooperativeActiveSTAR` — metadata retained; no unrestricted manuscript PDF was identified.

Regenerate the library locally or in CI with:

```bash
python scripts/download_open_access_references.py \
  --inventory references/open_access_pdfs.tsv \
  --output-dir reference_library/pdfs \
  --report reference_library/DOWNLOAD_MANIFEST.json \
  --continue-on-error
```

The canonical citation library is `references/references.bib`; the chapter-by-chapter reading map is `references/README.md`.
