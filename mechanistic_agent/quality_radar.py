"""Radar chart (plain SVG, no plotting dependency) comparing a baseline and a harness on the
six ``quality_v2`` axes. Used by ``results_publish`` for the harness-vs-baseline section."""
from __future__ import annotations

import math
from html import escape
from typing import Mapping, Optional, Sequence, Tuple

from mechanistic_agent.quality_v2 import AXES, AXIS_LABELS

BASELINE_COLOR = "#eb6834"
HARNESS_COLOR = "#2a78d6"
INK = "#2b2b28"
MUTED = "#77766f"
GRID = "#dddcd6"
SURFACE = "#fcfcfb"

WIDTH, HEIGHT = 440, 440
CX, CY, R = 220, 190, 120


def _point(i: int, value: float) -> Tuple[float, float]:
    angle = -math.pi / 2 + 2 * math.pi * i / len(AXES)
    return CX + R * value * math.cos(angle), CY + R * value * math.sin(angle)


def _polygon(values: Sequence[float]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in (_point(i, v) for i, v in enumerate(values)))


def _label_lines(text: str) -> Sequence[str]:
    words = text.split(" ")
    if len(words) == 1:
        return words
    mid = (len(words) + 1) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def radar_svg(
    series: Sequence[Tuple[str, str, Mapping[str, float], Optional[str]]],
    *,
    title: str,
) -> str:
    """``series`` is ``(name, color, axes, note)``; ``note`` is printed under the legend entry."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
        f'font-family="Helvetica, Arial, sans-serif" role="img" aria-label="{escape(title)}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
        f'<text x="{CX}" y="22" text-anchor="middle" font-size="14" fill="{INK}">{escape(title)}</text>',
    ]
    for ring in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<polygon points="{_polygon([ring] * len(AXES))}" fill="none" stroke="{GRID}" stroke-width="1"/>')
    for i, axis in enumerate(AXES):
        x, y = _point(i, 1.0)
        parts.append(f'<line x1="{CX}" y1="{CY}" x2="{x:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        lx, ly = _point(i, 1.2)
        anchor = "middle" if abs(lx - CX) < 10 else ("start" if lx > CX else "end")
        lines = _label_lines(AXIS_LABELS[axis])
        y0 = ly - 6 * (len(lines) - 1) + 4
        for k, line in enumerate(lines):
            parts.append(f'<text x="{lx:.1f}" y="{y0 + 12 * k:.1f}" text-anchor="{anchor}" font-size="11" '
                         f'fill="{MUTED}">{escape(line)}</text>')
    for _, color, axes, _ in series:
        values = [float(axes.get(k) or 0.0) for k in AXES]
        parts.append(f'<polygon points="{_polygon(values)}" fill="{color}" fill-opacity="0.12" stroke="{color}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(values):
            x, y = _point(i, v)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
    legend_y = HEIGHT - 52
    for k, (name, color, _, note) in enumerate(series):
        x = 24 + k * (WIDTH // max(1, len(series)))
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="14" height="4" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y - 3}" font-size="12" fill="{INK}">{escape(name)}</text>')
        for j, line in enumerate((note or "").split("\n")):
            if line:
                parts.append(f'<text x="{x + 20}" y="{legend_y + 13 + 14 * j}" font-size="11" fill="{MUTED}">'
                             f'{escape(line)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
