from __future__ import annotations

"""Compact the generated TD3 paired-test longtable without changing its content."""

from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "generated" / "table_td3_paired_tests_holm.tex"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    marker = r"\begin{landscape}" + "\n" + r"\begin{longtable}{r l r r c c}"
    replacement = (
        r"\begin{landscape}" + "\n"
        + r"\small" + "\n"
        + r"\renewcommand{\arraystretch}{0.92}" + "\n"
        + r"\setlength{\LTpre}{2pt}" + "\n"
        + r"\setlength{\LTpost}{2pt}" + "\n"
        + r"\begin{longtable}{r l r r c c}"
    )
    if replacement in text:
        print("paired-test table already compact")
        return
    if text.count(marker) != 1:
        raise RuntimeError("Could not locate paired-test longtable marker exactly once")
    TARGET.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
    print("compacted TD3 paired-test table layout")


if __name__ == "__main__":
    main()
