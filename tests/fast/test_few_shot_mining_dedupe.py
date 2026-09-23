"""Few-shot mining dedupes identical outputs and caps examples per lane."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_evolve_module():
    spec = importlib.util.spec_from_file_location("evolve_harness_mod", _ROOT / "scripts" / "evolve_harness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("evolve_harness_mod", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _result(step_outputs):
    return {
        "passed": True,
        "run_status": "completed",
        "graded_details": {"final_product_reached": True},
        "score": 0.95,
        "input_payload": {"starting_materials": ["CCBr", "[Cl-]"], "products": ["CCCl", "[Br-]"]},
        "step_outputs": step_outputs,
    }


def _step(step_name: str, payload: dict, attempt: int = 1):
    return {
        "step_name": step_name,
        "attempt": attempt,
        "accepted_bool": True,
        "validation": {"passed": True},
        "output": {**payload, "schema_validation": {"status": "ok", "source": "tool_call"}},
    }


def test_mine_few_shots_dedupes_identical_outputs_and_respects_cap() -> None:
    mod = _load_evolve_module()
    config = mod.EvolutionConfig(model_name="gpt-5", max_few_shots_per_step=2)

    same = {"mapped_atoms": [{"product_atom": "CCCl#0"}], "confidence": 0.9}
    results = [
        _result([_step("atom_mapping", same), _step("atom_mapping", same, attempt=2)]),
        _result([_step("atom_mapping", {"mapped_atoms": [{"product_atom": "CCCl#1"}], "confidence": 0.8})]),
        _result([_step("atom_mapping", {"mapped_atoms": [{"product_atom": "CCCl#2"}], "confidence": 0.7})]),
    ]
    existing_hashes: dict = {}
    best: dict = {}
    mined = mod.mine_few_shots(results, config, existing_hashes, best)
    examples = mined["attempt_atom_mapping"]
    # 4 eligible steps, 1 exact duplicate removed -> 3 unique, capped to 2.
    assert len(examples) == 2
    assert len({ex["example_key"] for ex in examples}) == 2
    # Hashes kept in the tracker match exactly what was mined.
    assert existing_hashes["attempt_atom_mapping"] == {ex["example_key"] for ex in examples}


def test_mine_few_shots_skips_outputs_already_in_lane() -> None:
    mod = _load_evolve_module()
    import hashlib
    import json

    config = mod.EvolutionConfig(model_name="gpt-5", max_few_shots_per_step=5)
    payload = {"mapped_atoms": [{"product_atom": "CCCl#0"}], "confidence": 0.9}
    output = {**payload, "schema_validation": {"status": "ok", "source": "tool_call"}}
    known_hash = hashlib.sha256(json.dumps(output, indent=2, sort_keys=True).encode()).hexdigest()[:16]
    existing_hashes = {"attempt_atom_mapping": {known_hash}}
    mined = mod.mine_few_shots([_result([_step("atom_mapping", payload)])], config, existing_hashes, {})
    assert "attempt_atom_mapping" not in mined
