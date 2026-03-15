#!/usr/bin/env python3
"""Render PNG panels for reaction_type_templates.json (generic templates).

Output: training_data/rxn_templates/ (outside version control)
Layout: same as flower mechanism PNGs — overall transformation + each step with labels.
Generic species are drawn as text (no RDKit); layout matches flower_rendering.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TRAINING_DATA = PROJECT_ROOT / "training_data"
DEFAULT_INPUT = _TRAINING_DATA / "reaction_type_templates.json"
DEFAULT_OUTPUT = _TRAINING_DATA / "rxn_templates"

# Match flower_rendering layout
MOL_IMG_SIZE = (300, 220)
ARROW_SPAN = 90
ARROW_GAP = 18
MARGIN = 24
TITLE_H = 56
SECTION_H = 42
STEP_HEADER_H = 78

# Text block size for generic state (no molecules)
TEXT_BLOCK_MIN_W = 280
TEXT_BLOCK_MIN_H = 180
TEXT_LINE_HEIGHT = 28
TEXT_PAD = 12


def _get_font(size: int):
    if ImageFont is None:
        return None
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_arrow(
    draw,
    x1: int,
    y: int,
    x2: int,
    *,
    color: str = "black",
    width: int = 5,
    head: int = 18,
) -> None:
    shaft_end = x2 - head
    if shaft_end > x1:
        draw.line([(x1, y), (shaft_end, y)], fill=color, width=width)
    draw.polygon(
        [(x2, y), (x2 - head, y - head // 2), (x2 - head, y + head // 2)],
        fill=color,
    )


def _wrap_line(draw, text: str, font, max_width: int) -> List[str]:
    """Wrap a long line into multiple lines that fit max_width."""
    if not text or not font:
        return [text] if text else []
    words = text.replace(".", " . ").split()
    lines: List[str] = []
    current: List[str] = []
    for w in words:
        current.append(w)
        line = " ".join(current)
        bbox = draw.textbbox((0, 0), line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            continue
        current.pop()
        if current:
            lines.append(" ".join(current))
        current = [w] if w != "." else []
    if current:
        lines.append(" ".join(current))
    return lines


def _make_text_state_image(species_list: List[str], *, title: str):
    """Create a PIL Image for a generic state (title + species as text)."""
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available")
    font_title = _get_font(20)
    font_body = _get_font(16)
    species_lines = [str(s).strip() for s in species_list if str(s).strip()]
    if not species_lines:
        species_lines = ["(generic)"]
    # Build text lines (wrap long ones)
    canvas_probe = Image.new("RGB", (TEXT_BLOCK_MIN_W, 400), "white")
    draw_probe = ImageDraw.Draw(canvas_probe)
    max_w = TEXT_BLOCK_MIN_W - 2 * TEXT_PAD
    all_lines: List[str] = []
    for raw in species_lines:
        for part in (raw.replace(" or ", " / ").split(" / ")):
            part = part.strip()
            if part:
                all_lines.extend(_wrap_line(draw_probe, part, font_body, max_w))
    line_h = TEXT_LINE_HEIGHT
    title_h = line_h + 8
    block_h = title_h + len(all_lines) * line_h + 2 * TEXT_PAD
    block_w = TEXT_BLOCK_MIN_W
    block_h = max(block_h, MOL_IMG_SIZE[1] // 2)
    img = Image.new("RGB", (block_w, block_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((TEXT_PAD, TEXT_PAD), title, fill="#183153", font=font_title)
    y = TEXT_PAD + title_h
    for line in all_lines:
        draw.text((TEXT_PAD, y), line, fill="#333333", font=font_body)
        y += line_h
    return img


def _parse_reaction_generic(reaction_generic: str) -> Tuple[List[str], List[str]]:
    """Split 'left>>right |dbe:...|' into ([left tokens], [right tokens])."""
    text = str(reaction_generic or "").strip().split("|", 1)[0].strip()
    if ">>" not in text:
        return ([text] if text else [], [])
    left, right = text.split(">>", 1)
    left_tokens = [p.strip() for p in left.split(".") if p.strip()]
    right_tokens = [p.strip() for p in right.split(".") if p.strip()]
    return (left_tokens, right_tokens)


def _render_template(template: Dict[str, Any]) -> Image.Image:
    """One PNG per template: title, overall transformation, then each step with labels."""
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available")

    type_id = str(template.get("type_id") or "unknown")
    label = str(template.get("label_exact") or "")
    current_generic = list(template.get("current_state_generic") or [])
    resulting_generic = list(template.get("resulting_state_generic") or [])
    steps = list(template.get("generic_mechanism_steps") or [])

    # Overall panel: current_state_generic -> resulting_state_generic
    overall_left = _make_text_state_image(current_generic, title="Starting / Reactants")
    overall_right = _make_text_state_image(resulting_generic, title="Products")
    overall_w = (
        MARGIN
        + overall_left.width
        + ARROW_GAP
        + ARROW_SPAN
        + ARROW_GAP
        + overall_right.width
        + MARGIN
    )
    overall_h = SECTION_H + max(overall_left.height, overall_right.height) + MARGIN
    overall = Image.new("RGB", (overall_w, overall_h), "white")
    draw = ImageDraw.Draw(overall)
    draw.text((MARGIN, 8), "Overall Transformation", fill="#183153", font=_get_font(24))
    img_y = SECTION_H
    overall.paste(overall_left, (MARGIN, img_y))
    arrow_x1 = MARGIN + overall_left.width + ARROW_GAP
    arrow_x2 = arrow_x1 + ARROW_SPAN
    arrow_y = img_y + max(overall_left.height, overall_right.height) // 2
    _draw_arrow(draw, arrow_x1, arrow_y, arrow_x2)
    overall.paste(overall_right, (arrow_x2 + ARROW_GAP, img_y))

    # Step panels (same layout as flower: Step N + label, current -> result)
    step_panels: List[Image.Image] = []
    for step in steps:
        step_index = int(step.get("step_index") or 0)
        reaction_generic = str(step.get("reaction_generic") or "")
        note = str(step.get("note") or "").strip()
        left_tokens, right_tokens = _parse_reaction_generic(reaction_generic)
        current_img = _make_text_state_image(left_tokens, title="Current state")
        resulting_img = _make_text_state_image(right_tokens, title="Resulting state")
        panel_w = (
            MARGIN
            + current_img.width
            + ARROW_GAP
            + ARROW_SPAN
            + ARROW_GAP
            + resulting_img.width
            + MARGIN
        )
        panel_h = STEP_HEADER_H + max(current_img.height, resulting_img.height) + MARGIN
        panel = Image.new("RGB", (panel_w, panel_h), "#f8f8f8")
        pdraw = ImageDraw.Draw(panel)
        pdraw.line([(0, 0), (panel_w, 0)], fill="#cccccc", width=2)
        pdraw.text((MARGIN, 8), f"Step {step_index}", fill="#183153", font=_get_font(24))
        if note:
            pdraw.text((MARGIN, 34), note[:120] + ("..." if len(note) > 120 else ""), fill="#444444", font=_get_font(16))
        img_y = STEP_HEADER_H
        panel.paste(current_img, (MARGIN, img_y))
        ax1 = MARGIN + current_img.width + ARROW_GAP
        ax2 = ax1 + ARROW_SPAN
        ay = img_y + max(current_img.height, resulting_img.height) // 2
        _draw_arrow(pdraw, ax1, ay, ax2)
        panel.paste(resulting_img, (ax2 + ARROW_GAP, img_y))
        step_panels.append(panel)

    # Full canvas: title + overall + steps
    total_w = max(overall.width, *(p.width for p in step_panels)) if step_panels else overall.width
    total_h = TITLE_H + overall.height + sum(p.height for p in step_panels)
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{type_id}  {label}".strip()
    draw.text((MARGIN, 10), title, fill="#000000", font=_get_font(28))
    y = TITLE_H
    canvas.paste(overall, (0, y))
    y += overall.height
    for panel in step_panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas


def render_pngs(
    *,
    input_path: Path,
    output_dir: Path,
    max_templates: int | None = None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    templates: List[Dict[str, Any]] = list(payload.get("templates") or [])
    if isinstance(max_templates, int) and max_templates > 0:
        templates = templates[:max_templates]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Any] = {
        "source": str(input_path),
        "output_dir": str(output_dir),
        "rendered_count": 0,
        "items": [],
    }

    for template in templates:
        type_id = str(template.get("type_id") or "unknown")
        slug = str(template.get("slug") or type_id)
        filename = f"{slug}.png"
        out_path = output_dir / filename
        image = _render_template(template)
        image.save(out_path)
        index["items"].append({
            "type_id": type_id,
            "slug": slug,
            "label_exact": template.get("label_exact"),
            "step_count": len(template.get("generic_mechanism_steps") or []),
            "png": filename,
        })

    index["rendered_count"] = len(index["items"])
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render reaction type template PNGs (overall + each step with labels)."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="reaction_type_templates.json path",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory (default: training_data/rxn_templates)",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=None,
        help="Optional limit on number of templates to render",
    )
    args = parser.parse_args()

    if Image is None or ImageDraw is None:
        print("Pillow is required for PNG rendering.", file=sys.stderr)
        return 2

    try:
        result = render_pngs(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            max_templates=args.max_templates,
        )
        print(f"Rendered {result['rendered_count']} template PNG(s) to {result['output_dir']}")
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
