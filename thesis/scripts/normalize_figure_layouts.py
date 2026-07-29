from __future__ import annotations

"""Normalize authored Draw.io figure environments for the four-chapter thesis.

The 11 Draw.io figures used through ``\ThesisFigure`` have portrait-oriented
canvases. Earlier sources wrapped them in ``pdflscape``, which made the figures
occupy only the center of a landscape page. This script removes only landscape
wrappers that directly contain a ``\ThesisFigure`` block. Landscape longtables
and the generated 2x2 six-method performance figure remain unchanged.
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


def normalize(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    updated, count = PATTERN.subn(lambda match: match.group("body") + "\n", text)
    path.write_text(updated, encoding="utf-8")
    return count


def main() -> None:
    total = 0
    for path in TARGETS:
        count = normalize(path)
        total += count
        print(f"{path.name}: removed {count} landscape wrapper(s)")
    if total != 11:
        raise SystemExit(f"Expected 11 ThesisFigure landscape wrappers, found {total}")


if __name__ == "__main__":
    main()
