from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_calibrator_module():
    path = Path("training_data/template_confidence_calibrator.py")
    spec = importlib.util.spec_from_file_location("template_confidence_calibrator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["template_confidence_calibrator"] = module
    spec.loader.exec_module(module)
    return module


def test_stratified_allocation_includes_hard_tier() -> None:
    mod = _load_calibrator_module()
    allocation = mod._allocate_stratified_counts(
        tier_names=["easy", "medium", "hard"],
        available_by_tier={"easy": 10, "medium": 10, "hard": 10},
        max_mechanisms=5,
    )
    assert allocation["hard"] >= 1
    assert sum(allocation.values()) == 5


def test_failure_reason_timeout_classification() -> None:
    mod = _load_calibrator_module()
    reason = mod._classify_failure_reason(
        run_status="stopped",
        template_used=False,
        template_adhered=False,
        correct=False,
        product_match=False,
        step_count_match=False,
        notes=["timeout"],
        snapshot={},
    )
    assert reason == "timeout_pre_loop"


def test_failure_reason_template_not_adhered() -> None:
    mod = _load_calibrator_module()
    reason = mod._classify_failure_reason(
        run_status="failed",
        template_used=True,
        template_adhered=False,
        correct=False,
        product_match=False,
        step_count_match=False,
        notes=[],
        snapshot={},
    )
    assert reason == "template_not_adhered"


def test_failure_reason_reads_explicit_invalid_smiles_loop() -> None:
    mod = _load_calibrator_module()
    reason = mod._classify_failure_reason(
        run_status="failed",
        template_used=True,
        template_adhered=False,
        correct=False,
        product_match=False,
        step_count_match=False,
        notes=[],
        snapshot={"events": [{"event_type": "run_failed", "payload": {"reason": "proposal_invalid_smiles_loop"}}]},
    )
    assert reason == "proposal_invalid_smiles_loop"


def test_soft_signal_prefers_mapping_review_for_low_confidence_template_failure() -> None:
    mod = _load_calibrator_module()
    soft_signal_class, soft_signal_strength, action = mod._derive_soft_signal(
        failure_reason="proposal_incomplete_loop",
        prior_confidence=0.43,
        accepted_path_step_count=0,
        alignment_history_len=0,
        case_calibration_ready=True,
        template_used=True,
    )
    assert soft_signal_class == "template_mapping_likely_wrong"
    assert soft_signal_strength == "medium"
    assert action == "review_mapping"
