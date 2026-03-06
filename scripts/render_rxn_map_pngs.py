#!/usr/bin/env python3
"""Render mechanism step panel PNGs from rxn_map_expanded.json."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover - optional runtime dependency
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - defensive
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

try:  # pragma: no cover - optional runtime dependency
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Draw
except Exception:  # pragma: no cover - defensive
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]
    Draw = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "training_data" / "rxn_map_expanded.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "training_data" / "pngs"

if RDLogger is not None:  # pragma: no cover - runtime-only
    RDLogger.DisableLog("rdApp.error")

# ── Layout constants ──────────────────────────────────────────────────────────
MOL_IMG_SIZE = (380, 280)   # per-molecule cell size in RDKit grid
ARROW_SPAN = 100             # horizontal length of reaction arrow
ARROW_GAP = 24               # gap between image edge and arrow shaft
MARGIN = 24
STEP_HEADER_H = 115          # vertical space above molecule row in each step panel
OVERALL_HEADER_H = 70        # vertical space above label row in overall panel
LABEL_H = 36                 # height of "Starting Material(s)" / "Product(s)" label


def _get_font(size: int):
    """Return a PIL font at approximately *size* pts, falling back gracefully."""
    if ImageFont is None:
        return None
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _strip_atom_maps(smiles: str) -> str:
    return re.sub(r":\d+\]", "]", str(smiles or ""))


def _candidate_tokens(raw: str) -> List[str]:
    """Split a SMILES string (possibly a reaction SMILES) into individual tokens."""
    text = str(raw or "").strip()
    if not text:
        return []
    text = text.split("|", 1)[0].strip()
    if ">>" in text:
        left, right = text.split(">>", 1)
        parts = [p.strip() for p in (left + "." + right).split(".") if p.strip()]
        return parts
    return [p.strip() for p in text.split(".") if p.strip()]


def _draw_arrow(
    draw,
    x1: int,
    y: int,
    x2: int,
    color: str = "black",
    width: int = 5,
    head_size: int = 22,
) -> None:
    """Draw a filled horizontal arrow from (x1, y) to (x2, y)."""
    shaft_end = x2 - head_size
    if shaft_end > x1:
        draw.line([(x1, y), (shaft_end, y)], fill=color, width=width)
    draw.polygon(
        [(x2, y), (x2 - head_size, y - head_size // 2), (x2 - head_size, y + head_size // 2)],
        fill=color,
    )


def _draw_state(smiles_list: List[str], *, title: str):
    """Render a list of SMILES as an RDKit molecule grid image."""
    if Chem is None or Draw is None or Image is None:
        raise RuntimeError("RDKit/Pillow not available")

    mols = []
    legends = []
    for smiles in smiles_list:
        for token in _candidate_tokens(smiles):
            core = _strip_atom_maps(token)
            if not core:
                continue
            try:
                mol = Chem.MolFromSmiles(core)
            except Exception:
                mol = None
            if mol is None:
                continue
            mols.append(mol)
            legends.append(core)

    if not mols:
        canvas = Image.new("RGB", MOL_IMG_SIZE, "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"{title}: no renderable molecules", fill="#888888", font=_get_font(18))
        return canvas

    image = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=min(4, max(1, len(mols))),
        subImgSize=MOL_IMG_SIZE,
    )
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def _draw_overall_reaction(reaction: Dict[str, Any]):
    """Render the overall starting-materials → products transformation."""
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available")

    source = reaction.get("source_reaction") or {}
    starting_materials = [str(s) for s in source.get("starting_materials") or []]
    products = [str(p) for p in source.get("products") or []]
    if not products and reaction.get("known_final_target"):
        products = [str(reaction["known_final_target"])]

    sm_img = _draw_state(starting_materials, title="Starting Materials")
    prod_img = _draw_state(products, title="Products")

    mechanism = str((reaction.get("mechanism_type") or {}).get("label_exact") or "")

    img_row_h = max(sm_img.height, prod_img.height)
    total_w = (
        MARGIN + sm_img.width
        + ARROW_GAP + ARROW_SPAN + ARROW_GAP
        + prod_img.width + MARGIN
    )
    total_h = OVERALL_HEADER_H + LABEL_H + img_row_h + MARGIN

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text((MARGIN, 8), "Overall Transformation", fill="#1a1a8c", font=_get_font(30))
    draw.text((MARGIN, 42), f"Mechanism: {mechanism}", fill="#444444", font=_get_font(22))

    # Starting materials
    sm_x = MARGIN
    label_y = OVERALL_HEADER_H
    img_y = label_y + LABEL_H
    draw.text((sm_x, label_y), "Starting Material(s)", fill="#006600", font=_get_font(24))
    canvas.paste(sm_img, (sm_x, img_y))

    # Arrow
    arrow_x1 = sm_x + sm_img.width + ARROW_GAP
    arrow_x2 = arrow_x1 + ARROW_SPAN
    arrow_y = img_y + img_row_h // 2
    _draw_arrow(draw, arrow_x1, arrow_y, arrow_x2, color="#000000", width=6, head_size=24)

    # Products
    prod_x = arrow_x2 + ARROW_GAP
    draw.text((prod_x, label_y), "Product(s)", fill="#880000", font=_get_font(24))
    canvas.paste(prod_img, (prod_x, img_y))

    return canvas


def _render_reaction_panel(reaction: Dict[str, Any]):
    """Render a full PNG for one reaction: overall transformation + all steps."""
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available")

    reaction_id = str(reaction.get("reaction_id") or "unknown")
    mechanism = str((reaction.get("mechanism_type") or {}).get("label_exact") or "unknown")
    steps = list(reaction.get("mechanistic_steps") or [])

    overall_img = _draw_overall_reaction(reaction)

    step_panels = []
    for step in steps:
        idx = int(step.get("step_index") or 0)
        concrete = step.get("concrete_step") or {}
        current_state = [str(x) for x in concrete.get("current_state") or []]
        resulting_state = [str(x) for x in concrete.get("resulting_state") or []]

        current_img = _draw_state(current_state, title=f"Step {idx} current")
        resulting_img = _draw_state(resulting_state, title=f"Step {idx} resulting")

        intermediate = ""
        try:
            intermediate = str(
                ((step.get("selected_candidate") or {}).get("intermediate_smiles") or "")
            ).strip()
        except Exception:
            intermediate = ""

        img_row_h = max(current_img.height, resulting_img.height)
        panel_w = (
            MARGIN + current_img.width
            + ARROW_GAP + ARROW_SPAN + ARROW_GAP
            + resulting_img.width + MARGIN
        )
        panel_h = STEP_HEADER_H + img_row_h + MARGIN

        panel = Image.new("RGB", (panel_w, panel_h), "#f8f8f8")
        draw = ImageDraw.Draw(panel)
        draw.line([(0, 0), (panel_w, 0)], fill="#cccccc", width=2)

        # Step header text
        draw.text((MARGIN, 10), f"Step {idx}", fill="#1a1a8c", font=_get_font(28))
        draw.text((MARGIN, 48), f"Mechanism: {mechanism}", fill="#444444", font=_get_font(22))
        if intermediate:
            draw.text((MARGIN, 80), f"Intermediate: {intermediate}", fill="#555555", font=_get_font(18))

        # Current state image
        cur_x = MARGIN
        img_y = STEP_HEADER_H
        panel.paste(current_img, (cur_x, img_y))

        # Arrow
        arrow_x1 = cur_x + current_img.width + ARROW_GAP
        arrow_x2 = arrow_x1 + ARROW_SPAN
        arrow_y = img_y + img_row_h // 2
        _draw_arrow(draw, arrow_x1, arrow_y, arrow_x2, color="#000000", width=6, head_size=24)

        # Resulting state image
        res_x = arrow_x2 + ARROW_GAP
        panel.paste(resulting_img, (res_x, img_y))

        step_panels.append(panel)

    if not step_panels:
        base = Image.new("RGB", (900, 160), "white")
        draw = ImageDraw.Draw(base)
        draw.text((MARGIN, 10), f"{reaction_id} ({mechanism})", fill="black", font=_get_font(24))
        draw.text((MARGIN, 50), "No mechanistic_steps available", fill="#888888", font=_get_font(22))
        step_panels = [base]

    # Title banner
    title_h = 54
    reaction_name = str(reaction.get("name") or "").strip()
    title_text = f"{reaction_id}  —  {reaction_name}" if reaction_name else reaction_id

    total_w = max(overall_img.width, *(p.width for p in step_panels))
    total_h = title_h + overall_img.height + sum(p.height for p in step_panels) + MARGIN

    combined = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(combined)
    draw.text((MARGIN, 10), title_text, fill="#000000", font=_get_font(32))

    y = title_h
    combined.paste(overall_img, (0, y))
    y += overall_img.height

    for panel in step_panels:
        combined.paste(panel, (0, y))
        y += panel.height

    return combined


def _filter_reactions(
    reactions: List[Dict[str, Any]],
    *,
    mechanism_label: str | None,
    max_reactions: int | None,
    reaction_id_prefix: str | None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in reactions:
        label = str((row.get("mechanism_type") or {}).get("label_exact") or "")
        rid = str(row.get("reaction_id") or "")
        if mechanism_label and label != mechanism_label:
            continue
        if reaction_id_prefix and not rid.startswith(reaction_id_prefix):
            continue
        selected.append(row)
    if isinstance(max_reactions, int) and max_reactions > 0:
        return selected[:max_reactions]
    return selected


def render_pngs(
    *,
    input_path: Path,
    output_dir: Path,
    mechanism_label: str | None,
    max_reactions: int | None,
    reaction_id_prefix: str | None,
) -> Dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    reactions = list(payload.get("reactions") or [])
    selected = _filter_reactions(
        reactions,
        mechanism_label=mechanism_label,
        max_reactions=max_reactions,
        reaction_id_prefix=reaction_id_prefix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Any] = {
        "source": str(input_path),
        "output_dir": str(output_dir),
        "requested_filters": {
            "mechanism_label": mechanism_label,
            "max_reactions": max_reactions,
            "reaction_id_prefix": reaction_id_prefix,
        },
        "total_reactions_in_input": len(reactions),
        "rendered_count": 0,
        "items": [],
    }

    for reaction in selected:
        reaction_id = str(reaction.get("reaction_id") or "unknown")
        image = _render_reaction_panel(reaction)
        filename = f"{reaction_id}.png"
        out_path = output_dir / filename
        image.save(out_path)

        index["items"].append(
            {
                "reaction_id": reaction_id,
                "name": reaction.get("name"),
                "mechanism_label": str((reaction.get("mechanism_type") or {}).get("label_exact") or ""),
                "step_count": len(list(reaction.get("mechanistic_steps") or [])),
                "png": filename,
            }
        )

    index["rendered_count"] = len(index["items"])
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Render reaction mechanism PNG panels.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input rxn_map_expanded.json path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for PNG files.")
    parser.add_argument("--mechanism-label", default=None, help="Exact mechanism label filter.")
    parser.add_argument("--max-reactions", type=int, default=None, help="Optional max number of reactions.")
    parser.add_argument("--reaction-id-prefix", default=None, help="Optional reaction_id prefix filter.")
    args = parser.parse_args()

    if Chem is None or Draw is None or Image is None or ImageDraw is None:
        print(
            "RDKit and Pillow are required for PNG rendering. Install rdkit and pillow.",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    index = render_pngs(
        input_path=input_path,
        output_dir=output_dir,
        mechanism_label=args.mechanism_label,
        max_reactions=args.max_reactions,
        reaction_id_prefix=args.reaction_id_prefix,
    )
    print(f"Rendered {index['rendered_count']} reaction PNG(s) to {output_dir}")
    print(f"Index: {output_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
