from __future__ import annotations

import json
import re
from pathlib import Path

try:  # pragma: no cover - optional dependency in test runtime
    from rdkit import Chem
except Exception:  # pragma: no cover - defensive
    Chem = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "training_data"

RXN_MAP_EXPANDED = TRAINING_DIR / "rxn_map_expanded.json"
RXN_MAP_FLAT = TRAINING_DIR / "eval_mechanism_map.json"
EVAL_SET = TRAINING_DIR / "eval_set.json"
REACTION_TYPE_TEMPLATES = TRAINING_DIR / "reaction_type_templates.json"
RXN_NOTEBOOK = TRAINING_DIR / "rxn_map_visualizer.ipynb"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_dbe_total(reaction_smirks: str) -> int:
    match = re.search(r"\|dbe:([^|]+)\|", reaction_smirks)
    assert match, "reaction_smirks missing |dbe:...| block"
    total = 0
    for token in match.group(1).split(";"):
        token = token.strip()
        if not token:
            continue
        pair, delta = token.split(":", 1)
        assert re.match(r"^\d+-\d+$", pair), f"invalid dbe pair token: {pair}"
        total += int(delta)
    return total


def _strip_atom_maps(smiles: str) -> str:
    return re.sub(r":\d+\]", "]", str(smiles or ""))


def _normalize_smiles(smiles: str) -> str:
    token = _strip_atom_maps(smiles)
    if Chem is None:
        return token
    mol = Chem.MolFromSmiles(token)
    if mol is None:
        return token
    return Chem.MolToSmiles(mol, canonical=True)


def test_taxonomy_has_51_stable_entries_and_preserves_labels() -> None:
    template_payload = _load_json(REACTION_TYPE_TEMPLATES)
    templates = list(template_payload.get("templates") or [])
    templates.sort(key=lambda row: int(str(row.get("type_id") or "mt_999").split("_", 1)[1]))
    lines = [str(row.get("label_exact") or "") for row in templates]
    assert len(lines) == 51

    payload = _load_json(RXN_MAP_EXPANDED)
    taxonomy = payload["taxonomy"]
    assert len(taxonomy) == 51
    assert [entry["type_id"] for entry in taxonomy] == [f"mt_{idx:03d}" for idx in range(1, 52)]
    assert [entry["label_exact"] for entry in taxonomy] == lines


def test_first20_flat_mapping_is_complete_and_taxonomy_valid() -> None:
    payload = _load_json(RXN_MAP_EXPANDED)
    flat = _load_json(RXN_MAP_FLAT)

    taxonomy_by_id = {entry["type_id"]: entry["label_exact"] for entry in payload["taxonomy"]}
    taxonomy_norm = {label.lower(): label for label in taxonomy_by_id.values()}

    assert len(flat) == len(payload["reactions"])
    assert len(flat) >= 20
    ids = [row["reaction_id"] for row in flat]
    assert len(ids) == len(set(ids))

    for row in flat:
        assert row["mechanism_type_id"] in taxonomy_by_id
        label = str(row["mechanism_type_label"] or "").strip()
        canonical = taxonomy_by_id[row["mechanism_type_id"]]
        assert label
        assert label.lower() in taxonomy_norm or label == canonical
        assert isinstance(row["confidence"], float)
        assert row["rationale"]


def test_reaction_steps_have_contiguous_indices_dbe_and_aps() -> None:
    payload = _load_json(RXN_MAP_EXPANDED)
    template_payload = _load_json(REACTION_TYPE_TEMPLATES)

    assert payload["meta"]["scope"] in {"first_20_tier_reactions", "tier_30_reactions", "all_eval_reactions"}
    assert payload["meta"]["reaction_count"] == len(payload["reactions"])
    assert len(payload["reactions"]) >= 20

    template_rows = template_payload.get("templates", [])
    assert len(template_rows) == 51
    assert template_payload.get("example_mappings")
    by_label = {row["label_exact"]: row for row in template_rows}
    assert "SN2 reaction" in by_label
    assert by_label["SN2 reaction"]["suitable_step_count"] >= 1
    assert "R-Br" in by_label["SN2 reaction"]["generic_mechanism_steps"][0]["reaction_generic"]
    assert "R-Cl" in by_label["SN2 reaction"]["generic_mechanism_steps"][0]["reaction_generic"]
    assert by_label["Nitrile reduction"]["suitable_step_count"] >= 2

    for reaction in payload["reactions"]:
        steps = reaction["mechanistic_steps"]
        assert steps, f"{reaction['reaction_id']} has no mechanistic steps"
        assert [step["step_index"] for step in steps] == list(range(1, len(steps) + 1))

        final_target = reaction.get("known_final_target", "")
        last_state = steps[-1]["concrete_step"]["resulting_state"]
        assert last_state, f"{reaction['reaction_id']} has empty resulting state on last step"
        if final_target:
            normalized_target = _normalize_smiles(final_target)
            normalized_last = {_normalize_smiles(item) for item in last_state}
            if normalized_target not in normalized_last:
                assert "known_final_target_not_in_last_resulting_state" in (reaction.get("quality_flags") or [])

        for step in steps:
            rxn = step["concrete_step"]["reaction_smirks"]
            assert "|dbe:" in rxn
            assert _parse_dbe_total(rxn) == 0

            aps = step["annotation_suffix"]
            assert re.match(
                r"^\|aps:v1;src=[^;]+;snk=[^;]+;e=[12];tpl=[a-z0-9_]+;sc=\d\.\d{3}\|$",
                aps,
            )
            assert aps in step["annotated_reaction_smirks"]
            assert step["concrete_step"]["electron_pushes"]
            assert step["generic_step"]["reaction_generic"]
            assert step["generic_step"]["electron_pushes_generic"]
            assert step["generic_step"]["mechanism_label"] == reaction["mechanism_type"]["label_exact"]


def test_template_step_counts_cover_observed_eval_depth() -> None:
    flat = _load_json(RXN_MAP_FLAT)
    eval_set = _load_json(EVAL_SET)
    templates_payload = _load_json(REACTION_TYPE_TEMPLATES)

    steps_by_reaction: dict[str, int] = {}
    for case in eval_set:
        if not isinstance(case, dict):
            continue
        rid = str(case.get("id") or "")
        if not rid:
            continue
        km = case.get("known_mechanism") or {}
        steps_by_reaction[rid] = int(km.get("min_steps") or case.get("n_mechanistic_steps") or 0)

    template_by_id = {
        str(row.get("type_id")): int(row.get("suitable_step_count") or 0)
        for row in templates_payload.get("templates", [])
        if isinstance(row, dict)
    }

    observed_max_by_type: dict[str, int] = {}
    for row in flat:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("reaction_id") or "")
        type_id = str(row.get("mechanism_type_id") or "")
        if not rid or not type_id:
            continue
        observed = int(steps_by_reaction.get(rid, 0))
        observed_max_by_type[type_id] = max(observed, int(observed_max_by_type.get(type_id, 0)))

    assert observed_max_by_type
    for type_id, observed_max in observed_max_by_type.items():
        assert type_id in template_by_id
        assert template_by_id[type_id] >= observed_max


def test_notebook_smoke_and_required_cells() -> None:
    notebook = _load_json(RXN_NOTEBOOK)
    assert notebook.get("nbformat") == 4
    cells = notebook.get("cells", [])
    assert cells, "Notebook contains no cells"

    all_source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    assert "DATA_PATH" in all_source
    assert "rxn_map_expanded" in all_source
    assert "show_reaction" in all_source
    assert ("MECHANISM_FILTER" in all_source) or ("dd_mech" in all_source)
    assert (
        "Focused Verification: From Nitrile Reduction Onward" in all_source
        or "RXN Map Expanded Visualizer" in all_source
    )

    # Smoke check the payload expected by the notebook.
    payload = _load_json(RXN_MAP_EXPANDED)
    first = payload["reactions"][0]
    assert first["reaction_id"]
    assert first["mechanistic_steps"][0]["concrete_step"]["current_state"]
