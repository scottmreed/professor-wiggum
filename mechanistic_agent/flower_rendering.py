"""Rendering helpers for FlowER-derived mechanism datasets and curriculum cases."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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

from mechanistic_agent.core.mechanism_moves import extract_mechanism_moves
from mechanistic_agent.flower_curriculum import (
    DEFAULT_FLOWER_INPUT,
    DEFAULT_INDEX_PATH,
    DEFAULT_LOOKUP_CACHE,
    load_curriculum_index,
    convert_mechanism_id_to_case,
)

if RDLogger is not None:  # pragma: no cover - runtime-only
    RDLogger.DisableLog("rdApp.error")


MOL_IMG_SIZE = (300, 220)
ARROW_SPAN = 90
ARROW_GAP = 18
MARGIN = 24
TITLE_H = 56
SECTION_H = 42
STEP_HEADER_H = 78
MAP_PATTERN = re.compile(r":\d+\]")


def _get_font(size: int):
    if ImageFont is None:
        return None
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _strip_atom_maps(text: str) -> str:
    return MAP_PATTERN.sub("]", str(text or ""))


def _candidate_tokens(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    text = text.split("|", 1)[0].strip()
    if ">>" in text:
        left, right = text.split(">>", 1)
        return [part.strip() for part in f"{left}.{right}".split(".") if part.strip()]
    return [part.strip() for part in text.split(".") if part.strip()]


def _draw_arrow(draw, x1: int, y: int, x2: int, *, color: str = "black", width: int = 5, head: int = 18) -> None:
    shaft_end = x2 - head
    if shaft_end > x1:
        draw.line([(x1, y), (shaft_end, y)], fill=color, width=width)
    draw.polygon(
        [(x2, y), (x2 - head, y - head // 2), (x2 - head, y + head // 2)],
        fill=color,
    )


def _draw_state(smiles_list: List[str], *, title: str):
    if Chem is None or Draw is None or Image is None:
        raise RuntimeError("RDKit/Pillow not available")

    mols = []
    legends = []
    for smiles in smiles_list:
        for token in _candidate_tokens(smiles):
            core = _strip_atom_maps(token)
            if not core:
                continue
            mol = Chem.MolFromSmiles(core)
            if mol is None:
                continue
            mols.append(mol)
            legends.append(core)

    if not mols:
        canvas = Image.new("RGB", MOL_IMG_SIZE, "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"{title}: no renderable molecules", fill="#777777", font=_get_font(16))
        return canvas

    image = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=min(3, max(1, len(mols))),
        subImgSize=MOL_IMG_SIZE,
    )
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def _move_summary(step: Dict[str, Any]) -> str:
    reaction_smirks = str(step.get("reaction_smirks") or "")
    _core, moves, details = extract_mechanism_moves(reaction_smirks)
    if moves:
        return "; ".join(str(move.as_dict().get("notation") or "") for move in moves)
    pushes = list(step.get("electron_pushes") or [])
    rendered = []
    for item in pushes:
        if not isinstance(item, dict):
            continue
        notation = str(item.get("notation") or "").strip()
        if notation:
            rendered.append(notation)
            continue
        kind = str(item.get("kind") or "")
        rendered.append(kind or "move")
    if rendered:
        return "; ".join(rendered)
    return str(details.get("error") or "no moves")


def _render_step_panel(step: Dict[str, Any]):
    current_state = [str(item) for item in step.get("current_state") or []]
    resulting_state = [str(item) for item in step.get("resulting_state") or []]
    current_img = _draw_state(current_state, title="Current")
    resulting_img = _draw_state(resulting_state, title="Result")
    panel_w = (
        MARGIN + current_img.width + ARROW_GAP + ARROW_SPAN + ARROW_GAP + resulting_img.width + MARGIN
    )
    panel_h = STEP_HEADER_H + max(current_img.height, resulting_img.height) + MARGIN
    panel = Image.new("RGB", (panel_w, panel_h), "#f8f8f8")
    draw = ImageDraw.Draw(panel)

    step_index = int(step.get("step_index") or 0)
    draw.text((MARGIN, 8), f"Step {step_index}", fill="#183153", font=_get_font(24))
    draw.text((MARGIN, 34), _move_summary(step), fill="#444444", font=_get_font(16))

    img_y = STEP_HEADER_H
    panel.paste(current_img, (MARGIN, img_y))
    arrow_x1 = MARGIN + current_img.width + ARROW_GAP
    arrow_x2 = arrow_x1 + ARROW_SPAN
    arrow_y = img_y + max(current_img.height, resulting_img.height) // 2
    _draw_arrow(draw, arrow_x1, arrow_y, arrow_x2)
    result_x = arrow_x2 + ARROW_GAP
    panel.paste(resulting_img, (result_x, img_y))
    return panel


def _render_case(case: Dict[str, Any]):
    starting = [str(item) for item in case.get("starting_materials") or []]
    products = [str(item) for item in case.get("products") or []]
    steps = list(((case.get("verified_mechanism") or {}).get("steps") or []))

    overall_left = _draw_state(starting, title="Starting Materials")
    overall_right = _draw_state(products, title="Products")
    overall_w = (
        MARGIN + overall_left.width + ARROW_GAP + ARROW_SPAN + ARROW_GAP + overall_right.width + MARGIN
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

    step_panels = [_render_step_panel(step) for step in steps]
    total_w = max(overall.width, *(panel.width for panel in step_panels)) if step_panels else overall.width
    total_h = TITLE_H + overall.height + sum(panel.height for panel in step_panels)
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{case.get('id', 'unknown')}  {case.get('name', '')}".strip()
    draw.text((MARGIN, 10), title, fill="#000000", font=_get_font(28))
    y = TITLE_H
    canvas.paste(overall, (0, y))
    y += overall.height
    for panel in step_panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas


def _load_existing_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        return {"items": []}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"items": []}


def _png_filename_for_case(case_id: str, metadata: Dict[str, Any]) -> str:
    step_count = metadata.get("step_count")
    rank_within_step_count = metadata.get("rank_within_step_count")
    if step_count is None or rank_within_step_count is None:
        return f"{case_id}.png"
    return f"{int(step_count)}{int(rank_within_step_count):04d}.png"


def render_cases(
    *,
    cases: Sequence[Dict[str, Any]],
    output_dir: Path,
    metadata_by_case_id: Optional[Dict[str, Dict[str, Any]]] = None,
    only_missing: bool = False,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    if Chem is None or Draw is None or Image is None or ImageDraw is None:
        raise RuntimeError("RDKit and Pillow are required for PNG rendering.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    existing = _load_existing_index(index_path)
    existing_items = {
        str(item.get("case_id") or item.get("reaction_id") or ""): dict(item)
        for item in existing.get("items") or []
        if isinstance(item, dict)
    }

    items: List[Dict[str, Any]] = []
    rendered_count = 0
    for case in cases:
        case_id = str(case.get("id") or "unknown")
        metadata = dict((metadata_by_case_id or {}).get(case_id) or {})
        out_path = output_dir / _png_filename_for_case(case_id, metadata)
        existing_item = existing_items.get(case_id) or {}
        previous_png = str(existing_item.get("png") or "").strip()
        if previous_png and previous_png != out_path.name:
            previous_path = output_dir / previous_png
            if previous_path.exists():
                previous_path.unlink()
        if not (only_missing and out_path.exists()):
            image = _render_case(case)
            image.save(out_path)
            rendered_count += 1

        item = {
            "case_id": case_id,
            "reaction_id": case_id,
            "name": case.get("name"),
            "step_count": int(case.get("n_mechanistic_steps") or len((((case.get("verified_mechanism") or {}).get("steps")) or []))),
            "png": out_path.name,
        }
        if "global_rank" in metadata:
            item["global_rank"] = int(metadata["global_rank"])
        if "rank_within_step_count" in metadata:
            item["rank_within_step_count"] = int(metadata["rank_within_step_count"])
        if metadata:
            item["metadata"] = metadata
        items.append(item)
        existing_items[case_id] = item

    payload = {
        "source": source or existing.get("source"),
        "output_dir": str(output_dir),
        "rendered_count": rendered_count,
        "items": sorted(existing_items.values(), key=lambda row: (int(row.get("global_rank") or 10**12), str(row.get("case_id") or ""))),
    }
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def render_pngs(*, input_path: Path, output_dir: Path, max_reactions: int | None = None) -> Dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cases = list(payload if isinstance(payload, list) else payload.get("cases") or [])
    if isinstance(max_reactions, int) and max_reactions > 0:
        cases = cases[:max_reactions]
    return render_cases(cases=cases, output_dir=output_dir, source=str(input_path))


def render_curriculum_pngs(
    *,
    input_path: Path = DEFAULT_FLOWER_INPUT,
    index_path: Path = DEFAULT_INDEX_PATH,
    cache_path: Path = DEFAULT_LOOKUP_CACHE,
    output_dir: Path,
    top_n: Optional[int] = 50,
    entries: Optional[Sequence[Dict[str, Any]]] = None,
    only_missing: bool = False,
) -> Dict[str, Any]:
    if entries is None:
        entries = load_curriculum_index(index_path)

    cases: List[Dict[str, Any]] = []
    metadata_by_case_id: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []
    for entry in entries:
        mechanism_id = int(entry["mechanism_id"])
        try:
            case = convert_mechanism_id_to_case(mechanism_id, input_path=input_path, cache_path=cache_path)
        except Exception as exc:
            skipped.append({
                "case_id": str(entry.get("case_id") or ""),
                "mechanism_id": mechanism_id,
                "error": str(exc),
            })
            continue
        cases.append(case)
        metadata_by_case_id[str(case.get("id") or "")] = dict(entry)
        if entries is not None and isinstance(top_n, int) and top_n > 0 and len(cases) >= top_n:
            break

    payload = render_cases(
        cases=cases,
        output_dir=output_dir,
        metadata_by_case_id=metadata_by_case_id,
        only_missing=only_missing,
        source=str(index_path),
    )
    if skipped:
        payload["skipped"] = skipped
        (Path(output_dir) / "index.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "render_cases",
    "render_curriculum_pngs",
    "render_pngs",
]
