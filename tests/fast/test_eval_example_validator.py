from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_validator_module():
    path = Path("training_data/validate_eval_examples.py")
    spec = importlib.util.spec_from_file_location("validate_eval_examples", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_eval_examples"] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_smiles_uses_registry_mapping() -> None:
    mod = _load_validator_module()
    assert mod._normalize_smiles("Cl-", {"Cl-": "[Cl-]"}) == "[Cl-]"
    assert mod._normalize_smiles("[Cl-]", {"Cl-": "[Cl-]"}) == "[Cl-]"


def test_tag_case_detects_counterion_and_stereo() -> None:
    mod = _load_validator_module()
    tags = mod._tag_case(
        {
            "starting_materials": ["C/C=C\\CCl", "[Na+]", "[Cl-]"],
            "products": ["C/C=C\\CI"],
        }
    )
    assert tags["contains_counterion"] is True
    assert tags["stereochem_sensitive"] is True


def test_tag_case_flags_sulfuryl_and_suspect_mapping() -> None:
    mod = _load_validator_module()
    tags = mod._tag_case(
        {
            "starting_materials": ["CC(C(C1=CC=CC=C1)O)(C(O)=O)C", "FS(=O)(O)=O"],
            "products": ["C/C(C)=C(C(O)=O)/C1=CC=CC=C1"],
        },
        mapping_label="Aldol condensation",
    )
    assert tags["acid_fluoride_or_sulfuryl_halide_risk"] is True
    assert tags["template_mapping_suspect"] is True


def test_tag_case_flags_nitration_taxonomy_review() -> None:
    mod = _load_validator_module()
    tags = mod._tag_case(
        {
            "starting_materials": ["FC(C1=CC=CC=C1)(F)F", "O[N+]([O-])=O", "O=S(O)(O)=O"],
            "products": ["O=[N+](C1=CC(C(F)(F)F)=CC=C1)[O-]"],
        },
        mapping_label="Friedel Crafts acylation",
    )
    assert tags["strong_acid_eas_risk"] is True
    assert tags["requires_taxonomy_review"] is True
