from __future__ import annotations

r"""Normalize Draw.io figure environments and repair stable LaTeX identifiers.

The 11 Draw.io figures used through ``\ThesisFigure`` have portrait-oriented
canvases. Earlier sources wrapped them in ``pdflscape``, which made the figures
occupy only the center of a landscape page. This script removes only landscape
wrappers that directly contain a ``\ThesisFigure`` block. Landscape longtables
and the generated 2x2 six-method performance figure remain unchanged.

A previous language-cleanup pass accidentally translated several internal
LaTeX labels and one figure filename. Those identifiers are not visible thesis
text and must remain stable. The repair map below restores them idempotently
before every build.
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

REPAIRS = {
    r"eq:người dùng-rate-ch2": r"eq:user-rate-ch2",
    r"eq:qos-frhành động-ch2": r"eq:qos-fraction-ch2",
    r"eq:hành động-dimension-ch3": r"eq:action-dimension-ch3",
    r"eq:hành động-decoder-ch3": r"eq:action-decoder-ch3",
    r"figures/pdf/chapter3_fig02_hành động_decoder.pdf":
        r"figures/pdf/chapter3_fig02_action_decoder.pdf",
    r"fig:ch3-hành động-decoder": r"fig:ch3-action-decoder",
    "Bộ giải mã bộ giải mã vật lý": "Bộ giải mã vật lý",
    "on-chính sách": "on-policy",
}


def normalize(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    repair_count = 0
    for old, new in REPAIRS.items():
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            repair_count += count

    updated, removed = PATTERN.subn(lambda match: match.group("body") + "\n", text)
    path.write_text(updated, encoding="utf-8")
    figure_count = len(re.findall(r"\\ThesisFigure\{", updated))
    return removed, figure_count, repair_count


def main() -> None:
    removed_total = 0
    figure_total = 0
    repair_total = 0
    for path in TARGETS:
        removed, figures, repaired = normalize(path)
        removed_total += removed
        figure_total += figures
        repair_total += repaired
        print(
            f"{path.name}: removed {removed} landscape wrapper(s); "
            f"repaired {repaired} identifier(s); "
            f"found {figures} ThesisFigure block(s)"
        )

    if figure_total != 11:
        raise SystemExit(f"Expected 11 ThesisFigure blocks, found {figure_total}")

    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        for forbidden in REPAIRS:
            if forbidden in text:
                raise SystemExit(f"Unrepaired identifier in {path.name}: {forbidden}")

    if removed_total == 0:
        print("All ThesisFigure blocks were already normalized")
    else:
        print(f"Normalized {removed_total} landscape wrapper(s)")
    print(f"Repaired {repair_total} accidental translated identifier(s)")


if __name__ == "__main__":
    main()
