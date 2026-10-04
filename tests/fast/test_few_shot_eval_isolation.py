from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from mechanistic_agent import few_shot_isolation as fsi  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_no_few_shot_file_contains_an_eval_case() -> None:
    leaks = fsi.scan_few_shot_files(ROOT)
    report = "\n".join(f"{leak.path}:{leak.line} {leak.reason} {', '.join(leak.case_ids)}" for leak in leaks[:40])
    assert not leaks, (
        "few-shot examples leak eval cases; remove these lines "
        f"(python -m mechanistic_agent.few_shot_isolation --strip):\n{report}"
    )


def test_product_index_covers_every_tier_case() -> None:
    index = fsi.load_tier_index(ROOT)
    missing = sorted(fsi.tier_case_ids(ROOT) - set(index))
    assert not missing, (
        f"tier cases missing from training_data/{fsi.INDEX_FILE}: {missing[:10]} "
        "(python -m mechanistic_agent.few_shot_isolation --write-index)"
    )


def test_detects_main_product_and_case_id_but_not_small_byproducts() -> None:
    index = fsi.load_tier_index(ROOT)
    case_id, product = next((cid, smi) for cid, smi in sorted(index.items()) if smi)
    products = fsi.eval_case_products(ROOT, include_holdout=False)
    ids = fsi.tier_case_ids(ROOT)
    line = json.dumps({"input": {"current_state": ["CCO"]}, "output": {"resulting_state": [product, "O"]}})
    assert case_id in fsi.find_leaks(line, products, ids)["main_product"]
    assert fsi.find_leaks(json.dumps({"source": case_id}), products, ids)["case_id"] == {case_id}
    harmless = json.dumps({"output": {"resulting_state": ["O", "Cl", "CC(=O)O", "c1ccccc1"]}})
    assert fsi.find_leaks(harmless, products, ids) == {"case_id": set(), "main_product": set()}
