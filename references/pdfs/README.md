# In-Repository Thesis PDF Library

This directory contains the **27 legally open PDF files** selected for the STAR-RIS–RSMA TD3 thesis.

## Canonical companion files

- `../README.md`: reviewer-curated list of all 30 references and chapter mapping.
- `../references.bib`: canonical BibTeX entries.
- `../open_access_pdfs.tsv`: source URLs, landing pages, and access notes.
- `../DOWNLOAD_MANIFEST.json`: downloaded size, resolved source, SHA-256, and status for every reference.
- `../../artifacts/reference_library/AUDIT.json`: machine-readable audit verdict.

## References without bundled PDFs

Three of the 30 references remain metadata/full-text-only because no stable redistributable PDF endpoint was available:

1. `Meng2024PPOSTARRSMA`
2. `Liu2024DiscretePhaseRSMARate`
3. `Maghrebi2024CooperativeActiveSTAR`

Use their DOI or official landing page from `../open_access_pdfs.tsv`. Do not replace them with unofficial mirrors.

## Reproducibility

The PDFs were generated from the checked-in inventory and downloader, then validated for `%PDF-` headers and SHA-256 consistency before commit:

```bash
python scripts/download_open_access_references.py \
  --inventory references/open_access_pdfs.tsv \
  --output-dir references/pdfs \
  --report references/DOWNLOAD_MANIFEST.json \
  --continue-on-error
```

Commit containing the binary PDF library: `a13a59968d23747aa552d47a7dfb8eef0e6b72a6`.
