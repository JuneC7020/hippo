"""Render a terminal transcript (.txt) into an animated SVG for the README.

    python scripts/render_demo_svg.py docs/demo2.txt docs/demo2.svg

Lines appear one after another (SMIL animation, plays in GitHub's README renderer).
No dependencies; deterministic output so the SVG diffs cleanly in git.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

W, LINE_H, PAD, FONT = 880, 19, 16, 13
BG, FG, DIM, GREEN, YELLOW, RED, BLUE = (
    "#0d1117",
    "#c9d1d9",
    "#8b949e",
    "#3fb950",
    "#d29922",
    "#f85149",
    "#58a6ff",
)


def color_for(line: str) -> str:
    s = line.strip()
    if s.startswith("$ "):
        return BLUE
    if s.startswith("#"):
        return DIM
    if "APPROVE" in s or "passed" in s and "failed" not in s:
        return GREEN
    if "REVISE" in s or "replans=" in s:
        return YELLOW
    if s.startswith("FAILED") or "failed" in s or s.startswith("- "):
        return RED
    if s.startswith("+ "):
        return GREEN
    return FG


MAX_COLS = 104  # what fits in W at FONT px for a monospace face


def wrap(line: str) -> list[str]:
    """Soft-wrap long lines at spaces; continuation lines are indented like shell output."""
    if len(line) <= MAX_COLS:
        return [line]
    out, cur = [], ""
    for word in line.split(" "):
        if cur and len(cur) + 1 + len(word) > MAX_COLS:
            out.append(cur)
            cur = "    " + word
        else:
            cur = f"{cur} {word}" if cur else word
    out.append(cur)
    return out


def render(lines: list[str], *, step_s: float = 0.5) -> str:
    rows: list[tuple[str, str]] = []  # (text, colour) - wrapped rows share the source colour
    for raw in lines:
        colour = color_for(raw)
        rows.extend((part, colour) for part in wrap(raw.rstrip("\n")))
    height = PAD * 2 + LINE_H * len(rows)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, '
        f'monospace" font-size="{FONT}">',
        f'<rect width="100%" height="100%" rx="8" fill="{BG}"/>',
    ]
    for i, (text, colour) in enumerate(rows):
        y = PAD + LINE_H * (i + 1) - 5
        begin = f"{step_s * i:.2f}"
        escaped = html.escape(text).replace(" ", "\u00a0")
        out.append(
            f'<text x="{PAD}" y="{y}" fill="{colour}" opacity="0">{escaped}'
            f'<animate attributeName="opacity" from="0" to="1" begin="{begin}s" dur="0.15s" '
            f'fill="freeze"/></text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    lines = src.read_text(encoding="utf-8").splitlines()
    dst.write_text(render(lines), encoding="utf-8")
    print(f"wrote {dst} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
