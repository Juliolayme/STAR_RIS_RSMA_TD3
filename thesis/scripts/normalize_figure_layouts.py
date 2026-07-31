from __future__ import annotations

r"""Normalize authored Draw.io figure environments for the four-chapter thesis.

The 11 Draw.io figures used through ``\ThesisFigure`` have portrait-oriented
canvases. Earlier sources wrapped them in ``pdflscape``, which made the figures
occupy only the center of a landscape page. This script removes only landscape
wrappers that directly contain a ``\ThesisFigure`` block. Landscape longtables
and the generated 2x2 six-method performance figure remain unchanged.

The transformation is intentionally idempotent: a source tree in which all 11
figures have already been normalized is accepted without modification.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / f"chapter{i}.tex" for i in range(1, 5)]

PATTERN = re.compile(
    r"\\begin\{landscape\}\s*\n"
    r"(?P<body>\\ThesisFigure\{.*?\\figuresource\{.*?\}\s*\n)"
    r"\\end\{landscape\}\s*\n",
    flags=re.DOTALL,
)


def normalize(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    updated, removed = PATTERN.subn(lambda match: match.group("body") + "\n", text)
    path.write_text(updated, encoding="utf-8")
    figure_count = len(re.findall(r"\\ThesisFigure\{", updated))
    return removed, figure_count


def main() -> None:
    removed_total = 0
    figure_total = 0
    for path in TARGETS:
        removed, figures = normalize(path)
        removed_total += removed
        figure_total += figures
        print(
            f"{path.name}: removed {removed} landscape wrapper(s); "
            f"found {figures} ThesisFigure block(s)"
        )

    if figure_total != 11:
        raise SystemExit(f"Expected 11 ThesisFigure blocks, found {figure_total}")

    if removed_total == 0:
        print("All ThesisFigure blocks were already normalized")
    else:
        print(f"Normalized {removed_total} landscape wrapper(s)")


if __name__ == "__main__":
    main()
