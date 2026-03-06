"""Tool implementations exposed to the local mechanistic runtime."""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:  # pragma: no cover - optional import
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdmolfiles
except ImportError:  # pragma: no cover - handled at runtime
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    rdmolfiles = None  # type: ignore[assignment]

try:  # pragma: no cover - optional import
    from dimorphite_dl import DimorphiteDL
except ImportError:  # pragma: no cover - handled at runtime
    DimorphiteDL = None  # type: ignore[assignment]

from .llm import adapter_supports_forced_tools, extract_text_content, get_chat_model, get_model_api_key, get_provider_label, is_gemini_model, is_openrouter_model
from .model_registry import (
    build_reasoning_payload,
    get_default_model,
    get_fallback_model,
    get_model_catalog,
    get_model_family,
)
from .prompt_assets import compose_system_prompt, format_few_shot_block
from .smiles_utils import remove_mapping_and_canonicalize
from .tool_schemas import (
    ASSESS_CONDITIONS_TOOL,
    ATOM_MAPPING_TOOL,
    INTERMEDIATES_TOOL,
    MECHANISM_STEP_PROPOSAL_TOOL,
    MISSING_REAGENTS_TOOL,
    REACTION_TYPE_SELECTION_TOOL,
    build_tool_choice,
)
from .core.reaction_type_templates import (
    compact_template_for_prompt,
    list_reaction_type_choices,
    load_reaction_type_catalog_for_runtime,
)
from .core.mechanism_moves import (
    extract_mechanism_moves,
    implied_bond_deltas,
    normalize_electron_pushes,
    reaction_bond_deltas,
    repair_candidate_reaction_smirks,
    split_cxsmiles_metadata,
    synthesize_dbe_entries,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common LLM SMILES errors and their correct canonical forms.
# LLMs frequently produce molecular-formula-in-brackets notation or common
# abbreviations that are not valid SMILES.  This map lets us auto-correct the
# most predictable mistakes before any validation runs.
# ---------------------------------------------------------------------------
_COMMON_SMILES_CORRECTIONS: Dict[str, str] = {
    # Water
    "[H2O]": "O",
    "H2O": "O",
    # Sulfuric acid
    "[H2SO4]": "OS(=O)(=O)O",
    "H2SO4": "OS(=O)(=O)O",
    # Hydrochloric acid
    "[HCl]": "Cl",
    "HCl": "Cl",
    # Sodium hydroxide
    "[NaOH]": "[Na+].[OH-]",
    "NaOH": "[Na+].[OH-]",
    # Ammonia
    "[NH3]": "N",
    "NH3": "N",
    # Hydrobromic acid
    "[HBr]": "Br",
    "HBr": "Br",
    # Hydronium
    "[H3O+]": "[OH3+]",
    # Methanol
    "[CH3OH]": "CO",
    "CH3OH": "CO",
    "MeOH": "CO",
    "[MeOH]": "CO",
    # Ethanol
    "[EtOH]": "CCO",
    "EtOH": "CCO",
    "[C2H5OH]": "CCO",
    "C2H5OH": "CCO",
    # Hydrogen gas
    "[H2]": "[HH]",
    "H2": "[HH]",
    # Carbon dioxide
    "[CO2]": "O=C=O",
    "CO2": "O=C=O",
    # Nitric acid
    "[HNO3]": "[N+](=O)(O)[O-]",
    "HNO3": "[N+](=O)(O)[O-]",
    # Phosphoric acid
    "[H3PO4]": "OP(O)(O)=O",
    "H3PO4": "OP(O)(O)=O",
    # Acetic acid
    "[CH3COOH]": "CC(O)=O",
    "CH3COOH": "CC(O)=O",
    "AcOH": "CC(O)=O",
    "[AcOH]": "CC(O)=O",
    # Simple ionic species — LLM commonly writes shorthand that RDKit cannot parse
    "Cl-": "[Cl-]",
    "Br-": "[Br-]",
    "I-": "[I-]",
    "F-": "[F-]",
    "OH-": "[OH-]",
    "Na+": "[Na+]",
    "K+": "[K+]",
    "Li+": "[Li+]",
    "Mg2+": "[Mg+2]",
    "Ca2+": "[Ca+2]",
    "Zn2+": "[Zn+2]",
    # Ionic salts as dot-notation SMILES
    "NaCl": "[Na+].[Cl-]",
    "KCl": "[K+].[Cl-]",
    "LiCl": "[Li+].[Cl-]",
    "NaBr": "[Na+].[Br-]",
    "KBr": "[K+].[Br-]",
    "NaI": "[Na+].[I-]",
    "KI": "[K+].[I-]",
    "NaF": "[Na+].[F-]",
    "KOH": "[K+].[OH-]",
    "LiOH": "[Li+].[OH-]",
    # Common reducing agents
    "LiAlH4": "[Li+].[AlH4-]",
    "Li[AlH4]": "[Li+].[AlH4-]",
    "NaBH4": "[Na+].[BH4-]",
}

# Build a case-insensitive lookup for the correction map.
_SMILES_CORRECTIONS_LOWER: Dict[str, str] = {
    k.lower(): v for k, v in _COMMON_SMILES_CORRECTIONS.items()
}

_DEPRECATED_MODEL_ALIASES: Dict[str, str] = {
    "gemini-2.0-flash": "gemini-2.5-flash",
    "gemini-2.0-flash-thinking": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
}


def _apply_smiles_correction(smiles: str) -> Tuple[str, bool]:
    """Attempt to correct known LLM SMILES errors.

    Returns ``(corrected_smiles, was_corrected)``.
    """
    stripped = smiles.strip()
    # Exact match first (preserves case-sensitive keys like "[H2O]")
    if stripped in _COMMON_SMILES_CORRECTIONS:
        return _COMMON_SMILES_CORRECTIONS[stripped], True
    # Case-insensitive fallback
    lower = stripped.lower()
    if lower in _SMILES_CORRECTIONS_LOWER:
        return _SMILES_CORRECTIONS_LOWER[lower], True
    return stripped, False


def _resolve_step_model(step_name: str, env_var: str) -> str:
    """Resolve the model to use for a step.

    Priority: thread-local step model → thread-local active model → env var → default.
    """
    try:
        from mechanistic_agent.core.model_context import get_active_model, get_step_model
        model = get_step_model(step_name) or get_active_model()
    except Exception:
        model = None
    resolved = model or os.getenv(env_var) or os.getenv("MECHANISTIC_ACTIVE_MODEL") or get_default_model()
    replacement = _DEPRECATED_MODEL_ALIASES.get(str(resolved))
    if replacement and replacement in get_model_catalog():
        logger.warning(
            "Model '%s' is deprecated/unavailable; remapping to '%s' for step '%s'.",
            resolved,
            replacement,
            step_name,
        )
        return replacement
    return resolved


def _get_user_api_key_for_model(model_name: str) -> Optional[str]:
    """Return user-provided API key for a model's provider from thread-local context."""
    try:
        from mechanistic_agent.core.model_context import get_api_key
        if is_gemini_model(model_name):
            return get_api_key("gemini")
        if is_anthropic_model(model_name):
            return get_api_key("anthropic")
        if is_openrouter_model(model_name):
            return get_api_key("openrouter")
        return get_api_key("openai")
    except Exception:
        return None


@dataclass(frozen=True)
class ToolDescriptor:
    """Simple local descriptor for runtime tool registration."""

    name: str
    description: str
    func: Any


def _functional_group_analysis_enabled() -> bool:
    """Return True when functional group analysis is enabled via environment flag."""

    value = os.getenv("MECHANISTIC_FUNCTIONAL_GROUPS_ENABLED", "")
    return value.lower() in {"1", "true", "yes", "on"}


class ToolRuntimeError(RuntimeError):
    """Raised when a tool cannot execute due to missing optional dependencies."""


def _supports_temperature_parameter(model_name: Optional[str]) -> bool:
    """Return True when the OpenAI model accepts an explicit temperature override."""

    if not model_name:
        return True
    return "gpt-5" not in model_name.lower()


def _apply_reasoning_kwargs(
    llm_kwargs: Dict[str, Any],
    model_name: Optional[str],
    reasoning_level: Optional[str],
) -> None:
    if not model_name or not reasoning_level:
        return
    payload = build_reasoning_payload(model_name, reasoning_level)
    if not payload:
        return
    model_kwargs = dict(llm_kwargs.get("model_kwargs", {}))
    model_kwargs.update(payload)
    llm_kwargs["model_kwargs"] = model_kwargs


def _apply_output_token_cap(
    llm_kwargs: Dict[str, Any],
    model_name: Optional[str],
) -> None:
    """Apply conservative output token caps to avoid provider default overshoot."""
    if not model_name:
        return
    model_kwargs = dict(llm_kwargs.get("model_kwargs", {}))
    if "max_tokens" in model_kwargs or "max_output_tokens" in model_kwargs:
        return
    if is_openrouter_model(model_name):
        model_kwargs["max_tokens"] = max(
            512,
            int(os.getenv("MECHANISTIC_OPENROUTER_MAX_TOKENS", "12000")),
        )
    elif model_name.lower().startswith("gpt-4o"):
        model_kwargs["max_tokens"] = max(
            512,
            int(os.getenv("MECHANISTIC_GPT4O_MAX_TOKENS", "8192")),
        )
    if model_kwargs:
        llm_kwargs["model_kwargs"] = model_kwargs


def _require_rdkit() -> None:
    if Chem is None or Descriptors is None or rdMolDescriptors is None:
        raise ToolRuntimeError(
            "RDKit is required for this tool. Install `rdkit-pypi` (or conda rdkit)."
        )


_MECH_FORMAT_GUIDANCE = (
    "Append CXSMILES/SMIRKS with a '|mech:v1;...|' block using explicit move sources: "
    "'lp:a>b' for lone-pair attack, 'pi:a-b>c' for pi-bond donation, and "
    "'sigma:a-b>c' for sigma-bond attack."
)


_MECH_EXAMPLE = (
    "[CH3:1][CH2:2][Br:3].[Cl-:4]>>[CH3:1][CH2:2][Cl:4].[Br-:3] "
    "|mech:v1;lp:4>2;sigma:2-3>3|"
)


class BondElectronFormatError(ValueError):
    """Raised when bond-electron metadata cannot be parsed."""


@dataclass(frozen=True)
class BondElectronDelta:
    map_i: int
    map_j: int
    delta: int

    @property
    def entry_type(self) -> Literal["bond", "lone_pair"]:
        return "lone_pair" if self.map_i == self.map_j else "bond"

    def as_dict(self) -> Dict[str, object]:
        return {
            "map_i": self.map_i,
            "map_j": self.map_j,
            "pair": f"{self.map_i}-{self.map_j}",
            "delta": self.delta,
            "type": self.entry_type,
        }


def _parse_dbe_entries(entries: str, *, enforce_conservation: bool = True) -> List[BondElectronDelta]:
    if not entries:
        raise BondElectronFormatError("dbe metadata block is empty")
    parsed: List[BondElectronDelta] = []
    total_delta = 0
    for token in entries.split(";"):
        candidate = token.strip()
        if not candidate:
            continue
        if ":" not in candidate:
            raise BondElectronFormatError(f"Entry '{candidate}' is missing the ':' separator")
        pair_part, delta_part = candidate.split(":", 1)
        if "-" not in pair_part:
            raise BondElectronFormatError(f"Entry '{candidate}' must include a pair in the form mapI-mapJ")
        i_part, j_part = pair_part.strip().split("-", 1)
        if not re.fullmatch(r"\d+", i_part) or not re.fullmatch(r"\d+", j_part):
            raise BondElectronFormatError(f"Entry '{candidate}' must reference numeric atom-map indices")
        try:
            delta_value = int(delta_part.strip())
        except ValueError as exc:
            raise BondElectronFormatError(f"Entry '{candidate}' has a non-integer delta") from exc
        parsed.append(BondElectronDelta(map_i=int(i_part), map_j=int(j_part), delta=delta_value))
        total_delta += delta_value
    if not parsed:
        raise BondElectronFormatError("dbe metadata block did not contain any entries")
    if enforce_conservation and total_delta != 0:
        raise BondElectronFormatError(f"Bond-electron deltas must sum to zero (observed {total_delta}).")
    return parsed


def _extract_dbe_or_infer(expression: str, *, electron_pushes: Any) -> Tuple[str, List[BondElectronDelta], Dict[str, Any]]:
    core, metadata = split_cxsmiles_metadata(expression)
    details: Dict[str, Any] = {"raw": str(expression or "").strip(), "core": core, "metadata": metadata}
    dbe_entry = next((item for item in metadata if item.lower().startswith("dbe:")), None)
    if dbe_entry is not None:
        value = dbe_entry.split(":", 1)[1].strip() if ":" in dbe_entry else ""
        details["dbe"] = value
        try:
            deltas = _parse_dbe_entries(value)
            details["total_delta"] = sum(delta.delta for delta in deltas)
            details["source"] = "reaction_smirks"
            return core, deltas, details
        except BondElectronFormatError as exc:
            details["error"] = str(exc)
            try:
                deltas = _parse_dbe_entries(value, enforce_conservation=False)
                details["total_delta"] = sum(delta.delta for delta in deltas)
            except BondElectronFormatError:
                deltas = []
            return core, deltas, details
    inferred = synthesize_dbe_entries(electron_pushes)
    if inferred:
        details["dbe"] = inferred
        details["source"] = "inferred_from_electron_pushes"
        deltas = _parse_dbe_entries(inferred)
        details["total_delta"] = sum(delta.delta for delta in deltas)
        return core, deltas, details
    details["error"] = "Missing dbe metadata block and unable to infer one from electron_pushes"
    return core, [], details


@dataclass
class MoleculeReport:
    smiles: str
    formula: str
    mol_weight: float
    heavy_atom_count: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "smiles": self.smiles,
            "formula": self.formula,
            "molecular_weight": round(self.mol_weight, 4),
            "heavy_atom_count": self.heavy_atom_count,
        }


def _mol_from_smiles(smiles: str):
    _require_rdkit()
    candidate = str(smiles or "").strip()
    if not candidate:
        raise ToolRuntimeError("Invalid SMILES string: empty input")
    if not _looks_like_smiles(candidate):
        raise ToolRuntimeError(f"Invalid SMILES string: {candidate}")
    mol = Chem.MolFromSmiles(candidate)
    if mol is None:
        raise ToolRuntimeError(f"Invalid SMILES string: {candidate}")
    return mol


def _atom_counter(smiles_list: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        for atom in mol.GetAtoms():
            counts[atom.GetSymbol()] += 1
    return counts


def _atom_counter_with_hydrogens(smiles_list: Iterable[str]) -> Counter:
    """Count all atoms including hydrogens for proper stoichiometry."""
    counts: Counter = Counter()
    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts[symbol] += 1
            # Add hydrogens attached to this atom
            hydrogen_count = atom.GetTotalNumHs()
            if hydrogen_count > 0:
                counts['H'] += hydrogen_count
    return counts


def _molecule_report(smiles: str) -> MoleculeReport:
    mol = _mol_from_smiles(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    weight = Descriptors.MolWt(mol)
    heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return MoleculeReport(smiles=smiles, formula=formula, mol_weight=weight, heavy_atom_count=heavy)


def _strip_atom_mappings(mol: Any) -> Any:
    """Return a copy of the molecule without per-atom map annotations."""

    if Chem is None:
        return mol

    cleaned = Chem.Mol(mol)
    for atom in cleaned.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
        if atom.HasProp("_MolFileAtomMapNumber"):
            atom.ClearProp("_MolFileAtomMapNumber")

    for prop_name in ("molAtomMapNumber", "_MolFileAtomMapNumber"):
        if cleaned.HasProp(prop_name):
            cleaned.ClearProp(prop_name)

    return cleaned


def _mol_to_extended_smiles(
    mol: Any,
    *,
    canonical: bool = True,
    strip_atom_maps: bool = True,
) -> str:
    """Serialise an RDKit molecule, removing atom map annotations for readability."""

    _require_rdkit()
    export_mol = _strip_atom_mappings(mol) if strip_atom_maps else mol
    if rdmolfiles is not None and hasattr(rdmolfiles, "MolToCXSmiles"):
        try:
            return rdmolfiles.MolToCXSmiles(export_mol, canonical=canonical)
        except TypeError:
            # Older RDKit versions do not accept the canonical keyword.
            return rdmolfiles.MolToCXSmiles(export_mol)
    return Chem.MolToSmiles(export_mol, canonical=canonical)


def _mol_to_extended_smarts(mol: Any) -> str:
    """Serialise an RDKit molecule to SMARTS, preserving extended annotations."""

    _require_rdkit()
    if rdmolfiles is not None and hasattr(rdmolfiles, "MolToCXSmarts"):
        try:
            return rdmolfiles.MolToCXSmarts(mol)
        except TypeError:
            return rdmolfiles.MolToCXSmarts(mol)
    return Chem.MolToSmarts(mol)


def _serialise(data: Dict[str, object]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _build_missing_reagent_suggestions(delta: Counter) -> List[str]:
    """Heuristic suggestions for reagents that could supply missing atoms."""
    if not delta:
        return []

    suggestions: List[str] = []
    elems = {el for el, amount in delta.items() if amount > 0}
    if "O" in elems and "H" in elems:
        suggestions.append("H2O")
        elems.discard("O")
        elems.discard("H")
    if "O" in elems:
        suggestions.append("O2")
        elems.discard("O")
    if "H" in elems:
        suggestions.append("H2")
        elems.discard("H")
    for halogen in {"Cl", "Br", "I", "F"}.intersection(elems):
        suggestions.append(f"{halogen}2")
        elems.discard(halogen)
    if "C" in elems and "O" in delta and delta["O"] > 0 and delta["C"] <= delta["O"]:
        suggestions.append("CO2")
        elems.discard("C")
    suggestions.extend(sorted(elems))
    return suggestions


def _build_missing_product_suggestions(surplus: Counter) -> List[str]:
    """Heuristic suggestions for potential missing products from surplus atoms."""
    if not surplus:
        return []

    suggestions: List[str] = []
    elems = {el for el, amount in surplus.items() if amount > 0}
    
    # Common byproducts and side products
    if "H" in elems and "O" in elems:
        suggestions.append("H2O")
        elems.discard("H")
        elems.discard("O")
    if "H" in elems:
        suggestions.append("H2")
        elems.discard("H")
    if "O" in elems:
        suggestions.append("O2")
        elems.discard("O")
    for halogen in {"Cl", "Br", "I", "F"}.intersection(elems):
        suggestions.append(f"{halogen}2")
        elems.discard(halogen)
    if "C" in elems and "O" in elems:
        suggestions.append("CO2")
        elems.discard("C")
        elems.discard("O")
    if "N" in elems and "H" in elems:
        suggestions.append("NH3")
        elems.discard("N")
        elems.discard("H")
    
    # Add remaining elements as potential simple products
    suggestions.extend(sorted(elems))
    return suggestions


def _normalise_counter(counter: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    """Return a dictionary with canonical ordering and integer values."""

    normalised: Dict[str, int] = {}
    if not counter:
        return normalised

    for element, amount in counter.items():
        try:
            value = int(amount)
        except Exception:
            continue
        if value:
            normalised[str(element)] = value
    return dict(sorted(normalised.items()))


def _analyse_balance_rdkit_data(
    starting_materials: Sequence[str], products: Sequence[str]
) -> Dict[str, Any]:
    """Return RDKit-based stoichiometry analysis for reactants and products."""

    start_counts = _atom_counter(starting_materials)
    product_counts = _atom_counter(products)

    deficit: Counter[str] = Counter()
    surplus: Counter[str] = Counter()

    for element in set(start_counts) | set(product_counts):
        start_total = start_counts.get(element, 0)
        product_total = product_counts.get(element, 0)
        if start_total > product_total:
            deficit[element] = start_total - product_total
        elif product_total > start_total:
            surplus[element] = product_total - start_total

    balanced = not deficit and not surplus

    return {
        "balanced": balanced,
        "deficit": dict(deficit),
        "surplus": dict(surplus),
        "starting_materials": [r.as_dict() for r in map(_molecule_report, starting_materials)],
        "products": [r.as_dict() for r in map(_molecule_report, products)],
        "reactant_counts": dict(sorted(start_counts.items())),
        "product_counts": dict(sorted(product_counts.items())),
    }


def analyse_balance(starting_materials: List[str], products: List[str]) -> str:
    """Assess atom balance using the deterministic RDKit analysis."""

    rdkit_result = _analyse_balance_rdkit_data(starting_materials, products)
    payload: Dict[str, Any] = {
        "mode": "rdkit",
        "rdkit": rdkit_result,
    }
    return _serialise(payload)


def attempt_atom_mapping(starting_materials: List[str], products: List[str]) -> str:
    """Use an LLM to propose atom-to-atom mappings between reactants and products."""

    start_counts = _atom_counter(starting_materials)
    product_counts = _atom_counter(products)

    deficit: Dict[str, int] = {}
    surplus: Dict[str, int] = {}
    for element in set(start_counts) | set(product_counts):
        start_total = start_counts.get(element, 0)
        product_total = product_counts.get(element, 0)
        if start_total > product_total:
            deficit[element] = start_total - product_total
        elif product_total > start_total:
            surplus[element] = product_total - start_total

    stoichiometry = {
        "reactants": dict(sorted(start_counts.items())),
        "products": dict(sorted(product_counts.items())),
    }

    def _format_counts(counter: Dict[str, int]) -> str:
        if not counter:
            return "none"
        return ", ".join(f"{element}: {amount}" for element, amount in sorted(counter.items()))

    fg_enabled = _functional_group_analysis_enabled()
    functional_groups_start: Dict[str, Dict[str, int]] = {}
    functional_groups_products: Dict[str, Dict[str, int]] = {}
    start_source = "functional group analysis disabled"
    product_source = "functional group analysis disabled"

    if fg_enabled:
        functional_groups_start, start_source = _retrieve_functional_group_context(starting_materials)
        functional_groups_products, product_source = _retrieve_functional_group_context(products)

        def _format_functional_group_summary(summary: Dict[str, Dict[str, int]]) -> str:
            lines: List[str] = []
            for smiles, groups in summary.items():
                reactive_groups = [
                    f"{label} (×{count})" if count > 1 else label
                    for label, count in sorted(groups.items())
                    if count > 0
                ]
                if reactive_groups:
                    lines.append(f"    - {smiles}: {', '.join(reactive_groups)}")
                else:
                    lines.append(f"    - {smiles}: none detected")
            return "\n".join(lines) if lines else "    - none"

        functional_group_section = (
            "Functional group analysis (reactive sites) for context:\n"
            f"  Starting materials ({start_source}):\n{_format_functional_group_summary(functional_groups_start)}\n"
            f"  Products ({product_source}):\n{_format_functional_group_summary(functional_groups_products)}\n"
        )
    else:
        functional_group_section = ""
    functional_group_transformation = classify_functional_group_transformation(starting_materials, products)
    functional_group_section += (
        "Functional-group transformation summary:\n"
        f"  - Label: {functional_group_transformation.get('label')}\n"
    )
    if functional_group_transformation.get("uncertainty_note"):
        functional_group_section += (
            f"  - Uncertainty: {functional_group_transformation.get('uncertainty_note')}\n"
        )

    mapping_model = _resolve_step_model("atom_mapping", "MECHANISTIC_ATOM_MAPPING_MODEL")

    system_prompt = (
        "You are an expert organic chemist performing atom-to-atom mapping. "
        "Use knowledge of common reaction types, prioritise changes at the most "
        "reactive functional groups, and assume saturated alkyl fragments are "
        "typically conserved. Consider molecular symmetry when assigning origins "
        "so equivalent atoms remain consistently labelled. Provide a best-effort "
        "mapping and explicitly highlight any uncertainty. When proposing reaction "
        "steps or example transformations, express them as CXSMILES or SMIRKS "
        "followed by the sparse delta bond-electron block using atom-map indices, "
        "e.g. "
        f"{_MECH_EXAMPLE}."
    )
    system_prompt = compose_system_prompt(
        call_name="attempt_atom_mapping",
        dynamic_system_prompt=system_prompt,
    )

    # Get missing reagent suggestions from previous analysis
    missing_reagent_info = ""
    try:
        # Check if missing reagent analysis was performed previously
        # This context would usually be available from prior runtime tool outputs
        missing_reagent_info = "\nNote: Consider that missing reagents may be involved in the reaction. "
        missing_reagent_info += "If atoms cannot be mapped from starting materials to products, "
        missing_reagent_info += "they may originate from missing reagents or form missing products."
    except:
        pass

    human_prompt = (
        "Provide a likely atom mapping between reactants and products.\n"
        f"Starting materials: {starting_materials}\n"
        f"Products: {products}\n"
        "Stoichiometry summary (atom counts):\n"
        f"  Reactants: {_format_counts(dict(sorted(start_counts.items())))}\n"
        f"  Products: {_format_counts(dict(sorted(product_counts.items())))}\n"
        f"  Missing atoms vs reactants: {_format_counts(deficit)}\n"
        f"  Excess atoms vs products: {_format_counts(surplus)}\n"
        f"{functional_group_section}"
        f"{missing_reagent_info}\n"
        "It is not certain that every product atom is accounted for. If you cannot "
        "map atoms with confidence, return null for the mapping and explain why.\n"
        "Consider that unmapped atoms may originate from missing reagents or form missing products.\n"
        "Label atoms using zero-based indices in the order implied by the canonical "
        "SMILES for each molecule.\n"
        "Return JSON with the following fields:\n"
        "  mapped_atoms: list|null — each item is {\"product_atom\": \"<product_smiles>#<index>\", "
        "\"source\": {\"molecule_index\": int, \"smiles\": str, \"atom_index\": int}, \"notes\": str}.\n"
        "  unmapped_atoms: list of strings describing atoms without confident assignments.\n"
        "  confidence: numeric confidence score between 0.0 and 1.0.\n"
        "  reasoning: brief explanation referencing reactive sites or symmetry considerations.\n"
        "  missing_reagent_considerations: optional notes about potential missing reagents or products.\n"
        "Always mention when symmetry or assumed spectator regions influenced the mapping.\n"
        "Include at least one example CXSMILES + '|mech:v1;...|' block if you recommend a mechanistic step to ensure downstream steps follow the required format."
    )
    few_shot_block = format_few_shot_block("attempt_atom_mapping")
    if few_shot_block:
        human_prompt += f"\n\n{few_shot_block}\n"

    llm_response: Optional[Dict[str, object]] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    _atom_map_usage: Optional[Dict[str, int]] = None
    _use_forced_tools = adapter_supports_forced_tools(mapping_model)

    _mapping_user_key = _get_user_api_key_for_model(mapping_model)
    api_key = get_model_api_key(mapping_model, user_key=_mapping_user_key)
    if not api_key:
        provider_label = get_provider_label(mapping_model)
        error = f"{provider_label} API key not configured; unable to attempt atom mapping."
    else:
        try:
            llm_kwargs: Dict[str, Any] = {"model": mapping_model}
            if _supports_temperature_parameter(mapping_model):
                llm_kwargs["temperature"] = 0.0
            _apply_reasoning_kwargs(
                llm_kwargs,
                mapping_model,
                os.getenv("MECHANISTIC_ATOM_MAPPING_REASONING"),
            )
            _apply_output_token_cap(llm_kwargs, mapping_model)
            llm = get_chat_model(
                mapping_model,
                temperature=llm_kwargs.get("temperature"),
                model_kwargs=llm_kwargs.get("model_kwargs"),
                user_api_key=_mapping_user_key,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]
            if _use_forced_tools:
                ai_message = llm.invoke(
                    messages,
                    tools=[ATOM_MAPPING_TOOL],
                    tool_choice=build_tool_choice("atom_mapping_result"),
                )
            else:
                ai_message = llm.invoke(messages)

            _atom_map_usage = getattr(ai_message, "usage", None)

            # Try to extract from forced tool call first.
            if _use_forced_tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                try:
                    parsed_tc = json.loads(ai_message.tool_calls[0]["arguments"])
                    parsed_tc.pop("text", None)  # remove commentary field
                    llm_response = parsed_tc
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

            # Fall back to text parsing.
            if llm_response is None:
                raw_response = extract_text_content(ai_message)
                if raw_response:
                    try:
                        parsed = json.loads(raw_response)
                        if isinstance(parsed, dict):
                            llm_response = parsed
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:  # pragma: no cover - network or API failures
            error = f"Atom mapping LLM call failed: {exc}"

    schema_validation: Optional[Dict[str, str]] = None
    if isinstance(llm_response, dict):
        try:
            validated_mapping = AtomMappingPayload.model_validate(llm_response)
            llm_response = validated_mapping.model_dump(exclude_none=True)
            schema_validation = {"status": "ok", "validator": "AtomMappingPayload"}
        except ValidationError as exc:
            schema_validation = {
                "status": "fallback",
                "validator": "AtomMappingPayload",
                "error": _summarise_validation_error(exc),
            }

    payload: Dict[str, object] = {
        "stoichiometry": stoichiometry,
        "deficit": deficit,
        "surplus": surplus,
        "mapping_model": mapping_model,
        "tool_calling_used": _use_forced_tools,
        "guidance": "Mappings use zero-based atom indices; treat spectator regions as conserved.",
    }
    payload["functional_groups"] = {
        "starting_materials": functional_groups_start,
        "products": functional_groups_products,
    }
    payload["functional_group_context"] = {
        "starting_materials": {"source": start_source},
        "products": {"source": product_source},
    }
    payload["functional_group_transformation"] = functional_group_transformation
    if llm_response is not None:
        payload["llm_response"] = llm_response
    if schema_validation:
        payload["schema_validation"] = schema_validation
    if raw_response is not None and llm_response is None:
        payload["raw_response"] = raw_response
    if error is not None:
        payload["error"] = error

    payload["bond_electron_guidance"] = {
        "format": _MECH_FORMAT_GUIDANCE,
        "example": _MECH_EXAMPLE,
    }
    if _atom_map_usage:
        payload["_llm_usage"] = _atom_map_usage

    return _serialise(payload)


def _looks_like_smiles(s: str) -> bool:
    """Return True if *s* could plausibly be a SMILES string.

    Rejects plain English words regardless of capitalisation or trailing
    sentence punctuation (e.g. ``"group."``, ``"The"``, ``"classification"``).

    Key discriminators
    ------------------
    * Sentence-end punctuation (``.``, ``,``, ``;``, ``:`` etc.) stripped from
      both ends before testing — a leading/trailing ``.`` is never valid SMILES
      (the SMILES ``.`` separator must be *between* two components).
    * After stripping, if the core is purely alphabetic with **two or more
      consecutive lowercase letters** it is an English word, not a SMILES token.
      Valid unbracketed SMILES element symbols have at most one lowercase letter
      after an uppercase (``Cl``, ``Br``, ``Si`` …).
    * Any remaining non-alphabetic character (digit, bracket, ``=``, ``#`` …)
      signals a SMILES candidate and is allowed through.
    """
    import re

    if not s:
        return False

    core = s.strip().strip(".,;:!?")
    if not core or " " in core:
        return False

    lowered = core.lower()
    descriptor_tokens = {
        "acid",
        "acidic",
        "base",
        "basic",
        "neutral",
        "catalyst",
        "catalyzed",
        "catalysed",
        "reagent",
        "reactant",
        "product",
        "intermediate",
        "mechanism",
        "step",
    }
    if lowered in descriptor_tokens:
        return False

    # Reject pericyclic reaction notation like [4+2], [2+2], [3+3].
    # These are electron-count descriptors, never valid SMILES.
    if re.fullmatch(r"\[\d+\+\d+\]", core):
        return False

    # Reject parenthesised descriptive words like (diene), (dienophile), (solvent).
    # Real SMILES branch groups in parentheses contain atoms/bonds, not long English words.
    _inner_word = re.fullmatch(r"\(([A-Za-z]{4,})\)", core)
    if _inner_word and re.search(r"[a-z]{2,}", _inner_word.group(1)):
        return False

    # Reject natural-language hyphenated words such as "acid-catalyzed".
    if re.fullmatch(r"[A-Za-z-]+", core) and "-" in core and re.search(r"[a-z]{2,}", core):
        return False

    # Pure English words are not SMILES.
    if re.fullmatch(r"[A-Za-z]+", core) and re.search(r"[a-z]{2,}", core):
        return False

    # Structural punctuation/digits strongly indicate SMILES-like syntax.
    if re.search(r"[\d\[\]()=#@\\/\.:%*+]", core):
        return True

    # Allow explicit single bonds only when they connect plausible element tokens.
    if "-" in core:
        return bool(re.fullmatch(r"(?:[A-Z][a-z]?)(?:-(?:[A-Z][a-z]?))+", core))

    # Remaining alphabetic forms are plausible only if they don't look like words.
    return not bool(re.search(r"[a-z]{2,}", core))


def _clean_smiles_string(smiles: str) -> str:
    """Clean SMILES string by removing extra whitespace and common artifacts."""
    if not isinstance(smiles, str):
        return str(smiles)
    
    # Remove leading/trailing whitespace
    cleaned = smiles.strip()
    
    # Keep only the first line/token to avoid trailing explanations.
    lines = cleaned.split('\n')
    if lines:
        cleaned = lines[0].strip()
    if " " in cleaned:
        cleaned = cleaned.split(" ", 1)[0].strip()
    
    # Remove common artifacts that might be included
    cleaned = cleaned.replace('"', '').replace("'", '')

    # Remove any trailing punctuation or explanations
    # Keep only alphanumeric characters, brackets, parentheses, and common SMILES symbols
    cleaned = re.sub(r'[^A-Za-z0-9\[\]()=#@+\-\\/\.:%*]', '', cleaned)

    # Apply known LLM SMILES corrections (e.g. [H2O] → O)
    corrected, _ = _apply_smiles_correction(cleaned)
    return corrected


def _normalise_charged_brackets(smiles: str) -> Optional[str]:
    """Reorder bracketed charge expressions so the charge appears at the end."""

    if "[" not in smiles or "]" not in smiles:
        return None

    import re

    pattern = re.compile(r"\[([^\]]+)\]")
    changed = False

    def _reorder(match: re.Match[str]) -> str:
        nonlocal changed
        content = match.group(1)
        if "+" not in content and "-" not in content:
            return match.group(0)

        trailing_digits = ""
        base_content = content
        digit_match = re.search(r"(\d+)$", base_content)
        if digit_match and any(ch in "+-" for ch in base_content[: digit_match.start()]):
            trailing_digits = digit_match.group(1)
            base_content = base_content[: digit_match.start()]

        signs = "".join(ch for ch in base_content if ch in "+-")
        if not signs:
            return match.group(0)

        atoms = "".join(ch for ch in base_content if ch not in "+-")
        if not atoms:
            atoms = base_content.replace("+", "").replace("-", "")

        normalized = f"{atoms}{signs}{trailing_digits}"
        if normalized == content:
            return match.group(0)

        changed = True
        return f"[{normalized}]"

    normalised_smiles = pattern.sub(_reorder, smiles)
    return normalised_smiles if changed else None


def _canonicalise_candidate_smiles(smiles: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Try to validate a SMILES string, applying gentle normalisations and recovery when helpful."""

    details: Dict[str, Any] = {"raw": smiles}

    # Apply common LLM correction map before any other processing.
    corrected, was_corrected = _apply_smiles_correction(smiles)
    if was_corrected:
        details["correction_applied"] = True
        details["original_raw"] = smiles
        smiles = corrected

    base_cleaned = _clean_smiles_string(smiles)
    details["raw_cleaned"] = base_cleaned

    if not base_cleaned:
        details["error"] = "SMILES empty after cleaning"
        return None, details

    if not _looks_like_smiles(base_cleaned):
        details["error"] = f"Invalid SMILES string: {base_cleaned}"
        return None, details

    attempts: List[str] = [base_cleaned]
    normalized = _normalise_charged_brackets(base_cleaned)
    if normalized and normalized not in attempts:
        attempts.append(normalized)
        details["normalized"] = normalized
        details["normalization_applied"] = "reordered_charge"

    # Try SMILES recovery for common LLM errors
    from mechanistic_agent.smiles_utils import attempt_smiles_recovery
    recovered = attempt_smiles_recovery(base_cleaned)
    if recovered and recovered not in attempts:
        attempts.append(recovered)
        details["recovered"] = recovered
        details["recovery_applied"] = "common_fixes"

    last_error: Optional[str] = None

    for attempt in attempts:
        try:
            mol = _mol_from_smiles(attempt)
        except ToolRuntimeError as exc:
            last_error = str(exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive
            last_error = str(exc)
            continue

        canonical = _mol_to_extended_smiles(mol, canonical=True)
        details["cleaned"] = attempt
        if attempt != base_cleaned:
            details["validated_from"] = attempt
        else:
            details["validated_from"] = attempt
        details["canonical"] = canonical
        details["validated"] = True
        return canonical, details

    details["cleaned"] = base_cleaned
    if last_error:
        details["error"] = last_error
    else:
        details["error"] = "Unable to validate SMILES string"
    return None, details


def validate_proposed_reagents(
    proposed_reactants: List[str],
    proposed_products: List[str], 
    starting_materials: List[str], 
    products: List[str]
) -> str:
    """Validate proposed reagents and byproducts by checking SMILES validity and atomic balance."""
    
    # Combine all proposed molecules for validation
    all_proposed = proposed_reactants + proposed_products
    
    # Clean and validate SMILES strings with RDKit
    valid_reactants: List[str] = []
    valid_products: List[str] = []
    invalid_molecules: List[Dict[str, str]] = []
    
    for molecule in proposed_reactants:
        try:
            # Clean the SMILES string first
            cleaned_molecule = _clean_smiles_string(molecule)
            
            if not cleaned_molecule:
                invalid_molecules.append({
                    "molecule": molecule,
                    "cleaned_molecule": cleaned_molecule,
                    "error": "Empty SMILES string after cleaning"
                })
                continue
                
            mol = _mol_from_smiles(cleaned_molecule)
            if mol is not None:
                valid_reactants.append(cleaned_molecule)
            else:
                invalid_molecules.append({
                    "molecule": molecule,
                    "cleaned_molecule": cleaned_molecule,
                    "error": "Invalid SMILES string"
                })
        except Exception as e:
            invalid_molecules.append({
                "molecule": molecule,
                "cleaned_molecule": _clean_smiles_string(molecule),
                "error": f"RDKit validation failed: {str(e)}"
            })
    
    for molecule in proposed_products:
        try:
            # Clean the SMILES string first
            cleaned_molecule = _clean_smiles_string(molecule)
            
            if not cleaned_molecule:
                invalid_molecules.append({
                    "molecule": molecule,
                    "cleaned_molecule": cleaned_molecule,
                    "error": "Empty SMILES string after cleaning"
                })
                continue
                
            mol = _mol_from_smiles(cleaned_molecule)
            if mol is not None:
                valid_products.append(cleaned_molecule)
            else:
                invalid_molecules.append({
                    "molecule": molecule,
                    "cleaned_molecule": cleaned_molecule,
                    "error": "Invalid SMILES string"
                })
        except Exception as e:
            invalid_molecules.append({
                "molecule": molecule,
                "cleaned_molecule": _clean_smiles_string(molecule),
                "error": f"RDKit validation failed: {str(e)}"
            })
    
    if not valid_reactants and not valid_products:
        return _serialise({
            "status": "failed",
            "reason": "No valid molecules provided",
            "invalid_reagents": invalid_molecules,
            "validation_details": "All proposed molecules failed SMILES validation"
        })
    
    # Check atomic balance with the new molecules
    extended_starting_materials = starting_materials + valid_reactants
    extended_products = products + valid_products
    
    start_counts = _atom_counter_with_hydrogens(extended_starting_materials)
    product_counts = _atom_counter_with_hydrogens(extended_products)
    
    # Calculate remaining deficits and surpluses
    deficit: Counter = Counter()
    surplus: Counter = Counter()
    
    for element, amount in product_counts.items():
        deficit[element] = max(amount - start_counts.get(element, 0), 0)
    
    for element, amount in start_counts.items():
        surplus[element] = max(amount - product_counts.get(element, 0), 0)
    
    # Check if reaction is now balanced
    is_balanced = not any(deficit.values()) and not any(surplus.values())
    
    return _serialise({
        "status": "success" if is_balanced else "partial",
        "is_balanced": is_balanced,
        "valid_reactants": valid_reactants,
        "valid_products": valid_products,
        "valid_reagents": valid_reactants + valid_products,  # For backward compatibility
        "invalid_reagents": invalid_molecules,
        "remaining_deficit": dict({k: v for k, v in deficit.items() if v > 0}),
        "remaining_surplus": dict({k: v for k, v in surplus.items() if v > 0}),
        "validation_details": "All molecules passed SMILES validation" if not invalid_molecules else f"{len(invalid_molecules)} molecules failed validation"
    })


def _normalise_condition_candidates(
    items: Optional[Iterable[Any]],
    *,
    default_role: Literal["acid", "base", "buffer"],
) -> List[Dict[str, Optional[str]]]:
    """Convert heterogeneous candidate entries into a canonical structure."""

    if not items:
        return []

    normalised: List[Dict[str, Optional[str]]] = []

    for item in items:
        if item is None:
            continue

        entry: Dict[str, Optional[str]] = {
            "name": None,
            "smiles": None,
            "role": default_role,
            "justification": None,
        }

        if isinstance(item, Mapping):
            name = item.get("name") or item.get("label") or item.get("identifier")
            smiles = item.get("smiles")
            role = item.get("role") or item.get("type")
            reason = (
                item.get("reason")
                or item.get("justification")
                or item.get("note")
            )

            if name is not None:
                entry["name"] = str(name)
            elif isinstance(smiles, str) and smiles.strip():
                entry["name"] = smiles.strip()

            if isinstance(smiles, str) and smiles.strip():
                entry["smiles"] = smiles.strip()

            if isinstance(role, str) and role.strip():
                entry["role"] = role.strip().lower()

            if reason is not None:
                entry["justification"] = str(reason)
        elif isinstance(item, str):
            entry["name"] = item.strip()
        else:
            entry["name"] = str(item)

        if any(entry.values()):
            normalised.append(entry)

    return normalised


def _summarise_validation_error(exc: ValidationError) -> str:
    """Return a compact one-line summary for Pydantic validation failures."""

    try:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first.get("loc", []))
        msg = str(first.get("msg") or "validation error")
        if loc:
            return f"{loc}: {msg}"
        return msg
    except Exception:
        return str(exc)


def _validate_smiles_like_token(value: Any, *, field_name: str) -> str:
    """Validate that a value looks like a SMILES token."""

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty SMILES string")
    if not _looks_like_smiles(text):
        raise ValueError(f"{field_name} is not SMILES-like: {text}")
    return text


def _validate_smiles_with_rdkit(smiles: str, *, field_name: str) -> Optional[str]:
    """Validate a SMILES token using the correction map, heuristic, and RDKit.

    Returns the canonical SMILES if valid, ``None`` if invalid.  Unlike
    :func:`_validate_smiles_like_token` (which raises), this returns ``None``
    for graceful per-item filtering.
    """
    text = str(smiles or "").strip()
    if not text:
        return None

    # Apply correction map first (e.g. [H2O] → O).
    corrected, _ = _apply_smiles_correction(text)
    text = corrected

    # Quick heuristic filter.
    if not _looks_like_smiles(text):
        return None

    # RDKit validation (only when available).
    if Chem is not None:
        try:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                return None
            canonical = Chem.MolToSmiles(mol)
            return canonical or text
        except Exception:
            return None

    # Fallback when RDKit is not available: accept heuristic-passing tokens.
    return text


class ConditionCandidatePayload(BaseModel):
    """Structured candidate entry returned by condition assessment."""

    model_config = ConfigDict(extra="ignore")

    name: str
    smiles: Optional[str] = None
    role: Optional[str] = None
    justification: Optional[str] = None

    @field_validator("smiles")
    @classmethod
    def _validate_optional_smiles(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_smiles_like_token(value, field_name="smiles")


class AssessConditionsPayload(BaseModel):
    """Structured payload for assess_initial_conditions."""

    model_config = ConfigDict(extra="ignore")

    environment: Literal["acidic", "basic", "neutral"]
    representative_ph: float
    ph_range: Optional[List[float]] = None
    justification: Optional[str] = None
    acid_candidates: List[ConditionCandidatePayload] = Field(default_factory=list)
    base_candidates: List[ConditionCandidatePayload] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class MissingReagentsPayload(BaseModel):
    """Structured payload for predict_missing_reagents."""

    model_config = ConfigDict(extra="ignore")

    missing_reactants: List[str] = Field(default_factory=list)
    missing_products: List[str] = Field(default_factory=list)
    verification: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    @field_validator("missing_reactants", "missing_products")
    @classmethod
    def _validate_missing_lists(cls, values: List[str]) -> List[str]:
        # Per-item filtering: drop invalid SMILES rather than rejecting the
        # entire payload.  The correction map (e.g. [H2O]→O) and RDKit are
        # used when available so that predictable LLM errors are auto-fixed.
        validated: List[str] = []
        for value in values:
            result = _validate_smiles_with_rdkit(value, field_name="missing_reagent")
            if result is not None:
                validated.append(result)
        return validated


class AtomMappingSourcePayload(BaseModel):
    """Source atom reference for atom mapping output."""

    model_config = ConfigDict(extra="ignore")

    molecule_index: int
    smiles: str
    atom_index: int


class AtomMappingItemPayload(BaseModel):
    """One mapped-atom record from atom mapping output."""

    model_config = ConfigDict(extra="ignore")

    product_atom: str
    source: Optional[AtomMappingSourcePayload] = None
    notes: Optional[str] = None


def _normalise_mapping_confidence(value: Any, *, default: float = 0.0) -> float:
    """Coerce legacy mapping confidence formats to a numeric score in [0.0, 1.0]."""
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text:
            numeric = default
        elif text in {"high", "very_high", "strong"}:
            numeric = 0.9
        elif text in {"medium", "moderate"}:
            numeric = 0.6
        elif text in {"low", "weak"}:
            numeric = 0.3
        else:
            try:
                numeric = float(text)
            except ValueError:
                numeric = default
    else:
        numeric = default

    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


class AtomMappingPayload(BaseModel):
    """Structured payload for attempt_atom_mapping."""

    model_config = ConfigDict(extra="ignore")

    mapped_atoms: Optional[List[AtomMappingItemPayload]] = None
    unmapped_atoms: List[str] = Field(default_factory=list)
    confidence: float
    reasoning: str
    missing_reagent_considerations: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: Any) -> float:
        return _normalise_mapping_confidence(value, default=0.0)


def _extract_representative_ph(data: Mapping[str, Any]) -> Optional[float]:
    """Return a numeric representative pH from LLM output when available."""

    for key in ("representative_ph", "target_ph", "estimated_ph", "ph_value"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            numeric = float(value)
            if 0.0 <= numeric <= 14.0:
                return numeric
        elif isinstance(value, str):
            try:
                numeric = float(value.strip())
            except ValueError:
                continue
            if 0.0 <= numeric <= 14.0:
                return numeric

    for key in ("ph_range", "recommended_range", "ph_window"):
        value = data.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                lower = float(value[0])
                upper = float(value[1])
            except (TypeError, ValueError):
                continue
            if 0.0 <= lower <= 14.0 and 0.0 <= upper <= 14.0:
                return (lower + upper) / 2.0

    return None


def assess_initial_conditions(
    starting_materials: List[str],
    products: List[str],
    observed_ph: Optional[float] = None,
) -> str:
    """Assess likely reaction conditions and compatible acids/bases via LLM."""

    condition_model = _resolve_step_model("initial_conditions", "MECHANISTIC_CONDITION_MODEL")

    report: Dict[str, Any] = {
        "status": "pending",
        "model_used": condition_model,
        "starting_materials": list(starting_materials),
        "products": list(products),
    }

    if observed_ph is not None:
        report["observed_ph"] = observed_ph

    fg_enabled = _functional_group_analysis_enabled()
    functional_groups_start: Dict[str, Dict[str, int]] = {}
    functional_groups_products: Dict[str, Dict[str, int]] = {}
    start_source = "functional group analysis disabled"
    product_source = "functional group analysis disabled"

    if fg_enabled:
        functional_groups_start, start_source = _retrieve_functional_group_context(starting_materials)
        functional_groups_products, product_source = _retrieve_functional_group_context(products)

        report["functional_group_context"] = {
            "starting_materials": functional_groups_start,
            "products": functional_groups_products,
            "starting_materials_source": start_source,
            "products_source": product_source,
        }
    functional_group_transformation = classify_functional_group_transformation(starting_materials, products)
    report["functional_group_transformation"] = functional_group_transformation

    _condition_user_key = _get_user_api_key_for_model(condition_model)
    api_key = get_model_api_key(condition_model, user_key=_condition_user_key)
    if not api_key:
        provider_label = get_provider_label(condition_model)
        report.update(
            {
                "status": "failed",
                "error": f"{provider_label} API key not configured; unable to assess conditions.",
            }
        )
        return _serialise(report)

    def _format_functional_groups(summary: Mapping[str, Mapping[str, int]]) -> str:
        lines: List[str] = []
        for smiles, groups in summary.items():
            reactive = [
                f"{label} (×{count})" if count > 1 else f"{label}"
                for label, count in sorted(groups.items())
                if count > 0
            ]
            if reactive:
                lines.append(f"  - {smiles}: {', '.join(reactive)}")
            else:
                lines.append(f"  - {smiles}: none detected")
        return "\n".join(lines) if lines else "  - none"

    functional_group_section = ""
    if fg_enabled:
        functional_group_section = (
            "Functional group compatibility checks:\n"
            f"Starting materials ({start_source}):\n{_format_functional_groups(functional_groups_start)}\n"
            f"Products ({product_source}):\n{_format_functional_groups(functional_groups_products)}\n"
            "Functional-group transformation summary:\n"
            f"  - Label: {functional_group_transformation.get('label')}\n"
        )
        if functional_group_transformation.get("uncertainty_note"):
            functional_group_section += (
                f"  - Uncertainty: {functional_group_transformation.get('uncertainty_note')}\n"
            )

    system_prompt = (
        "You are an expert synthetic chemist evaluating reaction media."
        " You receive the starting materials (reactant SMILES), target products (product SMILES),"
        " an optional user-supplied pH, and optional functional group compatibility data."
        " Use these inputs to determine the required reaction environment and suggest appropriate reagents.\n\n"
        " Determine whether the reaction should proceed under acidic, basic, or neutral"
        " conditions and supply a representative numeric pH."
        " Propose EITHER acid candidates (if acidic) OR base candidates (if basic) — never both."
        " Avoid reagents that would obviously destroy key functional groups."
        " Respond with compact JSON including:\n"
        "- environment: one of 'acidic', 'basic', or 'neutral'\n"
        "- representative_ph: single float between 0 and 14 (always include)\n"
        "- ph_range: optional [lower, upper]\n"
        "- justification: <=12 words summarising your reasoning\n"
        "- acid_candidates: list of <=3 entries only when environment is acidic; each entry has name, smiles, role, justification (<=10 words)\n"
        "- base_candidates: list of <=3 entries only when environment is basic; each entry has name, smiles, role, justification (<=10 words)\n"
        "- warnings: optional list of phrases, each <=10 words\n"
        "Answer with terse language; no additional commentary outside the JSON structure."
    )
    system_prompt = compose_system_prompt(
        call_name="assess_initial_conditions",
        dynamic_system_prompt=system_prompt,
    )

    human_lines = [
        "Assess reaction conditions for the following transformation.",
        f"Starting materials: {starting_materials}",
        f"Products: {products}",
    ]

    if observed_ph is not None:
        human_lines.append(f"User-supplied pH: {observed_ph}")

    if functional_group_section:
        human_lines.append("")
        human_lines.append(functional_group_section)

    human_lines.extend(
        [
            "",
            "Output JSON only. Always provide a numeric representative_ph even when you also provide a range.",
            "Ensure every candidate object includes name (string), optional smiles, role, and a ≤10 word justification.",
            "Keep the overall JSON compact; do not add narrative sentences.",
        ]
    )

    human_prompt = "\n".join(human_lines)
    few_shot_block = format_few_shot_block("assess_initial_conditions")
    if few_shot_block:
        human_prompt += f"\n\n{few_shot_block}\n"

    raw_response: Optional[str] = None
    _use_forced_tools = adapter_supports_forced_tools(condition_model)
    try:
        llm_kwargs: Dict[str, Any] = {"model": condition_model}
        if _supports_temperature_parameter(condition_model):
            llm_kwargs["temperature"] = 0.1
        _apply_reasoning_kwargs(
            llm_kwargs,
            condition_model,
            os.getenv("MECHANISTIC_CONDITION_REASONING"),
        )
        _apply_output_token_cap(llm_kwargs, condition_model)
        llm = get_chat_model(
            condition_model,
            temperature=llm_kwargs.get("temperature"),
            model_kwargs=llm_kwargs.get("model_kwargs"),
            user_api_key=_condition_user_key,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
        if _use_forced_tools:
            ai_message = llm.invoke(
                messages,
                tools=[ASSESS_CONDITIONS_TOOL],
                tool_choice=build_tool_choice("assess_conditions_result"),
            )
        else:
            ai_message = llm.invoke(messages)
        raw_response = extract_text_content(ai_message)
        _llm_usage = getattr(ai_message, "usage", None)
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": f"LLM call failed: {exc}",
            }
        )
        return _serialise(report)

    if _llm_usage:
        report["_llm_usage"] = _llm_usage

    # Extract structured result from tool call or fall back to text parsing.
    parsed: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None

    if _use_forced_tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        try:
            parsed = json.loads(ai_message.tool_calls[0]["arguments"])
            llm_commentary = parsed.pop("text", None)
            if llm_commentary:
                report["llm_commentary"] = llm_commentary
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            parse_error = f"Failed to parse forced tool call arguments: {exc}"

    if parsed is None:
        # Text-based fallback (OLMo, Gemini, or tool call extraction failed)
        if raw_response is None:
            report.update(
                {
                    "status": "failed",
                    "error": "LLM returned no content when assessing conditions.",
                }
            )
            return _serialise(report)

        report["raw_response"] = raw_response
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(raw_response.replace("'", '"'))
            except Exception as exc:
                parse_error = (
                    f"Unable to parse LLM response as JSON: {exc}"
                    if str(exc)
                    else "Unable to parse LLM response as JSON"
                )

        if isinstance(parsed, list) and parsed:
            if all(isinstance(item, Mapping) for item in parsed):
                first = parsed[0]
                if isinstance(first, Mapping):
                    parsed = dict(first)

    report["model_used"] = condition_model
    report["tool_calling_used"] = _use_forced_tools

    if not isinstance(parsed, Mapping):
        report.update(
            {
                "status": "failed",
                "parse_error": parse_error or "LLM response was not a JSON object.",
            }
        )
        return _serialise(report)

    parsed_dict = dict(parsed)
    try:
        validated_conditions = AssessConditionsPayload.model_validate(parsed_dict)
        parsed_dict = validated_conditions.model_dump(exclude_none=True)
        report["schema_validation"] = {"status": "ok", "validator": "AssessConditionsPayload"}
    except ValidationError as exc:
        report["schema_validation"] = {
            "status": "fallback",
            "validator": "AssessConditionsPayload",
            "error": _summarise_validation_error(exc),
        }

    environment = str(parsed_dict.get("environment", "")).strip().lower() or None
    if environment not in {"acidic", "basic", "neutral"}:
        environment = None

    representative_ph = _extract_representative_ph(parsed_dict)

    ph_range: Optional[List[float]] = None
    for key in ("ph_range", "recommended_range", "ph_window"):
        value = parsed_dict.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                low = float(value[0])
                high = float(value[1])
            except (TypeError, ValueError):
                continue
            ph_range = [low, high]
            break

    acid_candidates = _normalise_condition_candidates(
        parsed_dict.get("acid_candidates") or parsed_dict.get("acidic_additives"),
        default_role="acid",
    )
    base_candidates = _normalise_condition_candidates(
        parsed_dict.get("base_candidates") or parsed_dict.get("basic_additives"),
        default_role="base",
    )

    if acid_candidates and base_candidates:
        logger.warning(
            "assess_initial_conditions: LLM returned both acid_candidates and base_candidates; "
            "expected only one to be populated based on environment=%r. "
            "Both will be stored but downstream steps will select the appropriate list.",
            parsed_dict.get("environment"),
        )

    warnings: List[str] = []
    raw_warnings = parsed_dict.get("warnings")
    if isinstance(raw_warnings, (list, tuple)):
        warnings = [str(item) for item in raw_warnings if item]
    elif isinstance(raw_warnings, str) and raw_warnings.strip():
        warnings = [raw_warnings.strip()]

    justification = parsed_dict.get("justification") or parsed_dict.get("rationale")
    if isinstance(justification, (list, tuple)):
        justification = " ".join(str(item) for item in justification if item)
    elif justification is not None:
        justification = str(justification)

    report.update(
        {
            "status": "success",
            "environment": environment or "undetermined",
            "representative_ph": representative_ph,
            "ph_range": ph_range,
            "acid_candidates": acid_candidates,
            "base_candidates": base_candidates,
            "warnings": warnings,
            "justification": justification,
        }
    )

    _record_initial_condition_summary(starting_materials, products, report, "tool")

    return _serialise(report)


def predict_missing_reagents(
    starting_materials: List[str],
    products: List[str],
    conditions_guidance: Optional[str] = None,
) -> str:
    """Use an LLM to suggest reagents that resolve any stoichiometric imbalance."""

    start_counts = _atom_counter_with_hydrogens(starting_materials)
    product_counts = _atom_counter_with_hydrogens(products)

    deficit: Counter = Counter()
    for element, amount in product_counts.items():
        deficit[element] = max(amount - start_counts.get(element, 0), 0)

    surplus: Counter = Counter()
    for element, amount in start_counts.items():
        surplus[element] = max(amount - product_counts.get(element, 0), 0)

    deficit_atoms = {k: v for k, v in deficit.items() if v > 0}
    surplus_atoms = {k: v for k, v in surplus.items() if v > 0}

    fg_enabled = _functional_group_analysis_enabled()

    functional_groups_start: Dict[str, Dict[str, int]] = {}
    functional_groups_products: Dict[str, Dict[str, int]] = {}
    start_source = "functional group analysis disabled"
    product_source = "functional group analysis disabled"

    if fg_enabled:
        functional_groups_start, start_source = _retrieve_functional_group_context(starting_materials)
        functional_groups_products, product_source = _retrieve_functional_group_context(products)
    functional_group_transformation = classify_functional_group_transformation(starting_materials, products)

    conditions_context: Optional[Dict[str, Any]] = None
    conditions_source: Optional[str] = None
    conditions_guidance_parse_error: Optional[str] = None

    if conditions_guidance:
        parsed_conditions: Optional[Any] = None
        try:
            parsed_conditions = json.loads(conditions_guidance)
        except json.JSONDecodeError:
            try:
                parsed_conditions = json.loads(conditions_guidance.replace("'", '"'))
            except Exception as exc:
                conditions_guidance_parse_error = (
                    f"Unable to parse conditions_guidance JSON: {exc}"
                    if str(exc)
                    else "Unable to parse conditions_guidance JSON"
                )

        if isinstance(parsed_conditions, list) and parsed_conditions:
            if all(isinstance(item, Mapping) for item in parsed_conditions):
                candidate = parsed_conditions[0]
                if isinstance(candidate, Mapping):
                    parsed_conditions = dict(candidate)

        if isinstance(parsed_conditions, Mapping):
            conditions_context = dict(parsed_conditions)
            conditions_source = "provided"
        elif parsed_conditions is not None and conditions_guidance_parse_error is None:
            conditions_guidance_parse_error = "conditions_guidance was not a JSON object"

    if conditions_context is None:
        cached_context, cached_source = _retrieve_initial_condition_context(starting_materials, products)
        if cached_context:
            conditions_context = cached_context
            conditions_source = cached_source or "cached"

    def _format_functional_group_summary(summary: Dict[str, Dict[str, int]]) -> str:
        lines: List[str] = []
        for smiles, groups in summary.items():
            reactive_groups = [
                f"{label} (×{count})" if count > 1 else label
                for label, count in sorted(groups.items())
                if count > 0
            ]
            if reactive_groups:
                lines.append(f"    - {smiles}: {', '.join(reactive_groups)}")
            else:
                lines.append(f"    - {smiles}: none detected")
        return "\n".join(lines) if lines else "    - none"

    if fg_enabled:
        functional_group_section = (
            "Functional group analysis:\n"
            f"  Starting materials ({start_source}):\n{_format_functional_group_summary(functional_groups_start)}\n"
            f"  Products ({product_source}):\n{_format_functional_group_summary(functional_groups_products)}\n\n"
            "Functional-group transformation summary:\n"
            f"  - Label: {functional_group_transformation.get('label')}\n"
        )
        if functional_group_transformation.get("uncertainty_note"):
            functional_group_section += (
                f"  - Uncertainty: {functional_group_transformation.get('uncertainty_note')}\n\n"
            )
    else:
        functional_group_section = ""

    def _format_conditions_summary(context: Mapping[str, Any]) -> str:
        lines: List[str] = []
        environment = context.get("environment")
        if isinstance(environment, str) and environment.strip():
            lines.append(f"  • Preferred environment: {environment.strip()}")

        rep_ph = context.get("representative_ph")
        numeric_ph = None
        try:
            if rep_ph is not None:
                numeric_ph = float(rep_ph)
        except (TypeError, ValueError):
            numeric_ph = None
        if numeric_ph is None:
            numeric_ph = _extract_representative_ph(context)
        if numeric_ph is not None:
            lines.append(f"  • Representative pH ≈ {numeric_ph:.2f}")

        ph_range = context.get("ph_range")
        if isinstance(ph_range, (list, tuple)) and len(ph_range) == 2:
            try:
                lower = float(ph_range[0])
                upper = float(ph_range[1])
            except (TypeError, ValueError):
                lower = upper = None
            if lower is not None and upper is not None:
                lines.append(f"  • Suggested pH span: {lower:.2f} – {upper:.2f}")

        def _candidate_summary(items: Any, label: str) -> None:
            if not isinstance(items, (list, tuple)) or not items:
                return
            snippets: List[str] = []
            for item in items[:3]:
                if isinstance(item, Mapping):
                    name = str(item.get("name") or item.get("smiles") or "unknown").strip()
                    smiles = item.get("smiles")
                    reason = item.get("justification")
                    details = [name]
                    if isinstance(smiles, str) and smiles.strip() and smiles.strip() != name:
                        details.append(smiles.strip())
                    if isinstance(reason, str) and reason.strip():
                        details.append(reason.strip())
                    snippets.append("; ".join(details))
                elif item is not None:
                    snippets.append(str(item))
            if snippets:
                lines.append(f"  • {label}: " + "; ".join(snippets))

        _candidate_summary(context.get("acid_candidates"), "Acid supports")
        _candidate_summary(context.get("base_candidates"), "Base supports")

        warnings = context.get("warnings")
        if isinstance(warnings, (list, tuple)):
            filtered = [str(item).strip() for item in warnings if str(item).strip()]
            if filtered:
                lines.extend(f"  • Warning: {item}" for item in filtered)
        elif isinstance(warnings, str) and warnings.strip():
            lines.append(f"  • Warning: {warnings.strip()}")

        return "\n".join(lines) if lines else "  • No additional context provided"

    # Prepare context for LLM
    stoichiometry_context = {
        "starting_materials": starting_materials,
        "products": products,
        "atom_counts": {
            "reactants": dict(sorted(start_counts.items())),
            "products": dict(sorted(product_counts.items()))
        },
        "deficit": deficit_atoms,
        "surplus": surplus_atoms,
    }

    reagent_model = _resolve_step_model("missing_reagents", "MECHANISTIC_REAGENT_MODEL")

    report: Dict[str, Any] = {
        "analysis_context": stoichiometry_context,
        "atom_delta": deficit_atoms,
        "surplus_atoms": surplus_atoms,
        "model_used": reagent_model,
        "suggested_reagents": [],
    }

    if conditions_context:
        report["conditions_guidance"] = conditions_context
    if conditions_source:
        report["conditions_guidance_source"] = conditions_source
    if conditions_guidance_parse_error:
        report["conditions_guidance_parse_error"] = conditions_guidance_parse_error

    if not deficit_atoms and not surplus_atoms:
        report.update(
            {
                "status": "balanced",
                "message": "Reaction is already atomically balanced; no additional reagents proposed.",
            }
        )
        return _serialise(report)

    if fg_enabled:
        report["functional_groups"] = {
            "starting_materials": functional_groups_start,
            "products": functional_groups_products,
            "starting_materials_source": start_source,
            "products_source": product_source,
        }
    report["functional_group_transformation"] = functional_group_transformation

    def _format_counts(counter: Dict[str, int]) -> str:
        if not counter:
            return "none"
        return ", ".join(f"{element}: {amount}" for element, amount in sorted(counter.items()))

    system_prompt = (
        "You are an expert organic chemist specializing in reaction stoichiometry and reagent selection. "
        "Analyze the supplied reaction and propose any missing reagents or byproducts required to balance the atomic counts. "
        "Follow this systematic plan to balance chemical equations:\n\n"
        "1. Start by considering the unbalanced equation: Envision writing the correct chemical formulas for the reactants (on the left) and products (on the right), separated by an arrow.\n"
        "2. Count the atoms: Tally the number of atoms for each element present on the reactant side and on the product side of the equation.\n"
        "3. Identify imbalances: Look for missing atoms (deficit) and excess atoms (surplus).\n"
        "4. Add missing reagents: For missing atoms in products, add reactants to the reactant side.\n"
        "5. Add missing byproducts: For excess atoms in reactants, add products (byproducts) to the product side.\n"
        "6. Never change the subscripts within a chemical formula, as this changes the substance itself.\n"
        "7. Balance one element at a time until all elements are balanced on both sides. Consider the symmetry of the reaction and places where two groups on one reagent might react. Consider both intermolecular reactions and intramolecular reactions.\n"
        "8. Balance elements that appear in only one reactant and one product first.\n"
        "9. Consider using elements in their elemental form (e.g., H₂, O₂).\n"
        "10. Balance hydrogen and oxygen last.\n"
        "11. VERIFY YOUR ANSWER: After proposing molecules, calculate the new atom counts:\n"
        "    - Reactant atoms = (starting_materials + missing_reactants)\n"
        "    - Product atoms = (products + missing_products)\n"
        "    - Verify that all elements balance exactly\n\n"
        "Key Rules:\n"
        "- Law of Conservation of Mass: The total number of atoms of each element must be equal on both sides of the equation.\n"
        "- Coefficients, not Subscripts: Only the coefficients (numbers in front of formulas) can be changed, not the subscripts (small numbers within formulas).\n"
        "- Whole Numbers: The coefficients in a balanced equation should be the simplest whole number ratio.\n"
        "- Excess atoms in reactants typically indicate missing byproducts (e.g., excess O often means missing H2O)\n"
        "- Missing atoms in products typically indicate missing reactants\n"
        "- Keep in mind that a common reason an organic reaction is unbalanced is incomplete consideration of acid/base pairs. For example, if an acid is listed only on one side of a reaction, consider whether the reaction will be balanced by adding the conjugate to the other side of the reaction. It is OK to have charges on reagents you add so long as the reaction remains balanced.\n\n"
        "Respond ONLY with valid JSON containing the keys 'missing_reactants' (list of SMILES for reactant side), 'missing_products' (list of SMILES for product side), optional 'verification' (showing your atom count calculations), and optional 'notes'. "
        "Return empty lists when no molecules are needed on that side."
    )
    system_prompt = compose_system_prompt(
        call_name="predict_missing_reagents",
        dynamic_system_prompt=system_prompt,
    )

    human_sections = [
        "Review the reaction details and propose missing molecules to balance the atomic counts.",
        f"Starting materials: {starting_materials}",
        f"Products: {products}",
        "Atomic stoichiometry analysis:",
        f"  Reactants: {_format_counts(dict(sorted(start_counts.items())))}",
        f"  Products: {_format_counts(dict(sorted(product_counts.items())))}",
        f"  Missing atoms (deficit): {_format_counts(deficit_atoms)}",
        f"  Excess atoms (surplus): {_format_counts(surplus_atoms)}",
        "",
        "Guidance:",
        "- For missing atoms (deficit): Add reactants to the reactant side",
        "- For excess atoms (surplus): Add products/byproducts to the product side",
        "- Example: If excess O=1, consider adding O (water, H2O) to products",
        "- Always verify your answer by calculating final atom counts",
    ]

    if conditions_context:
        human_sections.extend(
            [
                "",
                "Initial condition guidance from assess_initial_conditions (use as flexible context, not a hard constraint):",
                _format_conditions_summary(conditions_context),
                "If you choose reagents that deviate from this environment, briefly justify the adjustment in your notes.",
            ]
        )
    elif conditions_guidance_parse_error:
        human_sections.extend(
            [
                "",
                "A prior conditions_guidance payload could not be parsed; proceed using stoichiometry and functional group information alone.",
            ]
        )

    if fg_enabled and functional_group_section:
        human_sections.append(functional_group_section)

    human_sections.append(
        "List SMILES strings only; avoid names or commentary inside the list."
    )

    human_prompt = "\n".join(human_sections)
    few_shot_block = format_few_shot_block("predict_missing_reagents")
    if few_shot_block:
        human_prompt += f"\n\n{few_shot_block}\n"

    _reagent_user_key = _get_user_api_key_for_model(reagent_model)
    api_key = get_model_api_key(reagent_model, user_key=_reagent_user_key)
    if not api_key:
        provider_label = get_provider_label(reagent_model)
        report.update(
            {
                "status": "failed",
                "error": f"{provider_label} API key not configured; unable to request reagent suggestions.",
            }
        )
        return _serialise(report)

    raw_response: Optional[str] = None
    _use_forced_tools = adapter_supports_forced_tools(reagent_model)
    try:
        llm_kwargs: Dict[str, Any] = {"model": reagent_model}
        if _supports_temperature_parameter(reagent_model):
            llm_kwargs["temperature"] = 0.1
        _apply_reasoning_kwargs(
            llm_kwargs,
            reagent_model,
            os.getenv("MECHANISTIC_MISSING_REAGENTS_REASONING"),
        )
        _apply_output_token_cap(llm_kwargs, reagent_model)
        llm = get_chat_model(
            reagent_model,
            temperature=llm_kwargs.get("temperature"),
            model_kwargs=llm_kwargs.get("model_kwargs"),
            user_api_key=_reagent_user_key,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
        if _use_forced_tools:
            ai_message = llm.invoke(
                messages,
                tools=[MISSING_REAGENTS_TOOL],
                tool_choice=build_tool_choice("missing_reagents_result"),
            )
        else:
            ai_message = llm.invoke(messages)
        raw_response = extract_text_content(ai_message)
        _reagent_usage = getattr(ai_message, "usage", None)
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error": f"LLM call failed: {exc}",
            }
        )
        return _serialise(report)

    if _reagent_usage:
        report["_llm_usage"] = _reagent_usage
    report["tool_calling_used"] = _use_forced_tools

    # Extract structured result from tool call or fall back to text parsing.
    data: Any = None
    parse_error: Optional[str] = None

    if _use_forced_tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        try:
            data = json.loads(ai_message.tool_calls[0]["arguments"])
            llm_commentary = data.pop("text", None)
            if llm_commentary:
                report["llm_commentary"] = llm_commentary
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            parse_error = f"Failed to parse forced tool call arguments: {exc}"

    if data is None:
        # Text-based fallback (OLMo, Gemini, or tool call extraction failed)
        if raw_response is None:
            report.update(
                {
                    "status": "no_response",
                    "error": "LLM returned no content when asked for missing reagents.",
                }
            )
            return _serialise(report)

        report["raw_response"] = raw_response
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            try:
                data = json.loads(raw_response.replace("'", '"'))
            except Exception as exc:
                data = None
                parse_error = (
                    f"Unable to parse LLM response as JSON: {exc}"
                    if str(exc)
                    else "Unable to parse LLM response as JSON"
                )

    parsed_reactants: List[str] = []
    parsed_products: List[str] = []

    if isinstance(data, dict):
        data_dict = dict(data)
        try:
            validated_payload = MissingReagentsPayload.model_validate(data_dict)
            data_dict = validated_payload.model_dump(exclude_none=True)
            report["schema_validation"] = {"status": "ok", "validator": "MissingReagentsPayload"}
        except ValidationError as exc:
            report["schema_validation"] = {
                "status": "fallback",
                "validator": "MissingReagentsPayload",
                "error": _summarise_validation_error(exc),
            }

        # New format: separate reactants and products
        missing_reactants = data_dict.get("missing_reactants")
        missing_products = data_dict.get("missing_products")
        
        if isinstance(missing_reactants, list):
            parsed_reactants = [
                str(item).strip() for item in missing_reactants if isinstance(item, str) and str(item).strip()
            ]
        
        if isinstance(missing_products, list):
            parsed_products = [
                str(item).strip() for item in missing_products if isinstance(item, str) and str(item).strip()
            ]
        
        # Backward compatibility: check for old "missing_reagents" format
        if not parsed_reactants and not parsed_products:
            candidates = data_dict.get("missing_reagents")
            if isinstance(candidates, list):
                parsed_reactants = [
                    str(item).strip() for item in candidates if isinstance(item, str) and str(item).strip()
                ]
            # No recognised list keys → treat as "no additions needed" rather than hard-failing.
            # The caller will see total_molecules==0 and return status:"success" with empty lists.

        notes = data_dict.get("notes")
        if isinstance(notes, str) and notes.strip():
            report["notes"] = notes.strip()

        verification = data_dict.get("verification")
        if isinstance(verification, dict):
            report["verification"] = verification
            
    elif isinstance(data, list):
        # Backward compatibility: treat list as missing_reagents
        parsed_reactants = [str(item).strip() for item in data if isinstance(item, str) and str(item).strip()]
    elif parse_error is None:
        parse_error = "LLM response was not a JSON object or list."

    if parse_error:
        report["parse_error"] = parse_error

    # Check if we have any molecules to validate
    total_molecules = len(parsed_reactants) + len(parsed_products)
    
    if total_molecules == 0 and not parse_error:
        # LLM returned empty lists - this is valid when no molecules are needed
        report.update(
            {
                "status": "success",
                "message": "LLM determined no additional molecules are needed for this reaction.",
                "suggested_reagents": [],
            }
        )
        return _serialise(report)
    elif total_molecules == 0:
        report.update(
            {
                "status": "failed",
                "error": "LLM did not return any SMILES strings to evaluate.",
            }
        )
        return _serialise(report)

    # Store the parsed molecules for reporting
    report["suggested_reactants"] = parsed_reactants
    report["suggested_products"] = parsed_products
    # For backward compatibility, combine into suggested_reagents
    report["suggested_reagents"] = parsed_reactants + parsed_products

    # Validate proposed reagents with one retry opportunity.
    # If the first attempt fails due to invalid SMILES, re-prompt the LLM
    # with specific error feedback so it can correct the notation.
    _MAX_REAGENT_RETRIES = 2
    validation_payload: Optional[Dict[str, Any]] = None

    for _reagent_attempt in range(_MAX_REAGENT_RETRIES):
        try:
            validation_payload = json.loads(
                validate_proposed_reagents(parsed_reactants, parsed_products, starting_materials, products)
            )
        except Exception as exc:
            report.update(
                {
                    "status": "failed",
                    "error": f"Internal validation failed: {exc}",
                }
            )
            return _serialise(report)

        report["validation"] = validation_payload

        status = validation_payload.get("status")
        if status == "success" and validation_payload.get("is_balanced"):
            report.update(
                {
                    "status": "success",
                    "suggested_reagents": validation_payload.get("valid_reagents", parsed_reactants + parsed_products),
                    "message": "Proposed reagents validated and reaction is now balanced.",
                }
            )
            if _reagent_attempt > 0:
                report["reagent_retries"] = _reagent_attempt
            return _serialise(report)

        # Check if a retry with error feedback could help.
        invalid_reagents = validation_payload.get("invalid_reagents", [])
        can_retry = (
            _reagent_attempt < _MAX_REAGENT_RETRIES - 1
            and invalid_reagents
            and _use_forced_tools
        )
        if can_retry:
            # Build error feedback for the LLM.
            error_details = []
            for inv in invalid_reagents:
                mol = inv.get("molecule") or inv.get("cleaned_molecule") or "unknown"
                err = inv.get("error", "validation failed")
                error_details.append(f"  - '{mol}': {err}")

            retry_addendum = (
                "\n\nYour previous response contained invalid SMILES strings:\n"
                + "\n".join(error_details)
                + "\n\nPlease correct these. Remember: water='O' (not '[H2O]'), "
                "HCl='Cl', H2SO4='OS(=O)(=O)O'. Return RDKit-parseable SMILES only."
            )
            retry_human = human_prompt + retry_addendum
            report["reagent_retry_attempted"] = True

            try:
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_human},
                ]
                retry_response = llm.invoke(
                    retry_messages,
                    tools=[MISSING_REAGENTS_TOOL],
                    tool_choice=build_tool_choice("missing_reagents_result"),
                )
                # Re-parse the retry response.
                retry_data: Any = None
                if hasattr(retry_response, "tool_calls") and retry_response.tool_calls:
                    try:
                        retry_data = json.loads(retry_response.tool_calls[0]["arguments"])
                        retry_data.pop("text", None)
                    except Exception:
                        pass
                if retry_data is None:
                    retry_raw = extract_text_content(retry_response)
                    if retry_raw:
                        try:
                            retry_data = json.loads(retry_raw)
                        except Exception:
                            pass

                if isinstance(retry_data, dict):
                    try:
                        retry_validated = MissingReagentsPayload.model_validate(retry_data)
                        retry_dict = retry_validated.model_dump(exclude_none=True)
                    except ValidationError:
                        retry_dict = retry_data
                    parsed_reactants = [
                        str(item).strip() for item in retry_dict.get("missing_reactants", [])
                        if isinstance(item, str) and str(item).strip()
                    ]
                    parsed_products = [
                        str(item).strip() for item in retry_dict.get("missing_products", [])
                        if isinstance(item, str) and str(item).strip()
                    ]
                    report["suggested_reactants"] = parsed_reactants
                    report["suggested_products"] = parsed_products
                    report["suggested_reagents"] = parsed_reactants + parsed_products
                    continue  # Re-validate with corrected molecules.
            except Exception as retry_exc:
                report["reagent_retry_error"] = str(retry_exc)
            # If retry call itself failed, fall through to the abort path.

        # Final failure — no more retries.
        break

    # If we reach here, validation failed after all attempts.
    assert validation_payload is not None
    report.update(
        {
            "status": "failed",
            "error": validation_payload.get(
                "reason",
                "Proposed reagents did not yield a balanced reaction.",
            ),
        }
    )
    if _reagent_attempt > 0:
        report["reagent_retries"] = _reagent_attempt

    remaining_deficit = validation_payload.get("remaining_deficit", {})
    remaining_surplus = validation_payload.get("remaining_surplus", {})
    has_deficit = isinstance(remaining_deficit, dict) and bool(remaining_deficit)
    has_surplus = isinstance(remaining_surplus, dict) and bool(remaining_surplus)
    abort_severity = "hard" if has_deficit else "soft"
    report["abort_severity"] = abort_severity
    report["should_abort_mechanism"] = bool(has_deficit)
    report["balance_issues"] = {
        "remaining_deficit": remaining_deficit if isinstance(remaining_deficit, dict) else {},
        "remaining_surplus": remaining_surplus if isinstance(remaining_surplus, dict) else {},
    }
    if has_deficit:
        report["message"] = "Unable to balance required product-side atoms with suggested reagents; halt mechanism generation."
    elif has_surplus:
        report["message"] = "Suggested reagents leave surplus atoms; continuing with warning-level imbalance."
    else:
        report["message"] = "Suggested reagents could not be validated as balanced."

    if validation_payload.get("invalid_reagents"):
        report["invalid_reagents"] = validation_payload["invalid_reagents"]

    return _serialise(report)


def predict_missing_reagents_for_candidate(
    *,
    current_state: List[str],
    resulting_state: List[str],
    failed_checks: Optional[List[str]] = None,
    validation_details: Optional[Dict[str, Any]] = None,
) -> str:
    """Scoped reagent/byproduct rescue for a single mechanism candidate.

    This delegates to ``predict_missing_reagents`` using the candidate step's
    ``current_state -> resulting_state`` transition as the balancing problem.
    """
    raw = predict_missing_reagents(
        starting_materials=current_state,
        products=resulting_state,
        conditions_guidance=json.dumps(
            {
                "rescue_mode": True,
                "failed_checks": list(failed_checks or []),
                "validation_details": dict(validation_details or {}),
            }
        ),
    )
    try:
        parsed = json.loads(raw)
    except Exception:
        return _serialise(
            {
                "status": "failed",
                "error": "candidate_rescue_parse_failure",
                "raw": raw,
                "add_reactants": [],
                "add_products": [],
            }
        )
    if not isinstance(parsed, dict):
        return _serialise(
            {
                "status": "failed",
                "error": "candidate_rescue_invalid_payload",
                "payload": parsed,
                "add_reactants": [],
                "add_products": [],
            }
        )

    raw_reactants = [
        str(item).strip()
        for item in parsed.get("suggested_reactants", [])
        if isinstance(item, str) and str(item).strip()
    ]
    raw_products = [
        str(item).strip()
        for item in parsed.get("suggested_products", [])
        if isinstance(item, str) and str(item).strip()
    ]

    def _normalise_species(smiles: str) -> str:
        canonical = _validate_smiles_with_rdkit(smiles, field_name="candidate_rescue_species")
        return str(canonical or smiles).strip()

    current_norm = {_normalise_species(item) for item in current_state if str(item).strip()}
    resulting_norm = {_normalise_species(item) for item in resulting_state if str(item).strip()}

    # Matches bare radical-atom SMILES like [N], [O], [C], [S] — no implicit H, no charge.
    # Charged ions ([Br-], [K+]) and protonated atoms ([NH4+]) are excluded by the pattern.
    _radical_atom_re = re.compile(r"^\[[A-Z][a-z]?\]$")

    def _sanitize_additions(
        items: List[str],
        *,
        cap: int,
        existing: set[str],
        side: str,
    ) -> Tuple[List[str], List[Dict[str, str]]]:
        kept: List[str] = []
        dropped: List[Dict[str, str]] = []
        seen: set[str] = set()
        for raw_item in items:
            value = _normalise_species(raw_item)
            if not value:
                continue
            # Reject reaction-SMILES strings that slipped through (> is not valid in molecule SMILES)
            if ">" in value:
                dropped.append({"species": value, "reason": "reaction_smiles_not_molecule", "side": side})
                continue
            # Reject bare radical atoms like [N], [O], [C] — chemically meaningless as byproducts
            if _radical_atom_re.match(value):
                dropped.append({"species": value, "reason": "bare_radical_atom", "side": side})
                continue
            if value in seen:
                dropped.append({"species": value, "reason": "duplicate_suggestion", "side": side})
                continue
            if value in existing:
                dropped.append({"species": value, "reason": "already_present", "side": side})
                seen.add(value)
                continue
            if len(kept) >= cap:
                dropped.append({"species": value, "reason": "cap_exceeded", "side": side})
                seen.add(value)
                continue
            kept.append(value)
            seen.add(value)
        return kept, dropped

    add_reactants, dropped_reactants = _sanitize_additions(
        raw_reactants,
        cap=2,
        existing=current_norm,
        side="reactant",
    )
    add_products, dropped_products = _sanitize_additions(
        raw_products,
        cap=2,
        existing=resulting_norm,
        side="product",
    )
    dropped_additions = dropped_reactants + dropped_products
    hint = None
    checks = set(failed_checks or [])
    if "mechanism_moves" in checks:
        hint = "Adjust the explicit mechanism moves so they match the bond changes in reaction_smirks."

    return _serialise(
        {
            "status": str(parsed.get("status") or "success"),
            "add_reactants": add_reactants,
            "add_products": add_products,
            "dropped_additions": dropped_additions,
            "rescue_caps": {"reactants": 2, "products": 2},
            "dbe_adjustment_hint": hint,
            "source": "predict_missing_reagents",
            "raw": parsed,
        }
    )


def attempt_atom_mapping_for_step(
    *,
    current_state: List[str],
    resulting_state: List[str],
) -> str:
    """Map atoms between step-level reactant and product states for loop guidance."""
    raw = attempt_atom_mapping(current_state, resulting_state)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"status": "failed", "error": "step_mapping_parse_failure", "raw": raw}
    if not isinstance(parsed, dict):
        parsed = {"status": "failed", "error": "step_mapping_invalid_payload", "raw": parsed}

    mapped_atoms = parsed.get("llm_response", {}).get("mapped_atoms")
    if not isinstance(mapped_atoms, list):
        mapped_atoms = []
    compact = []
    for item in mapped_atoms[:12]:
        if not isinstance(item, dict):
            continue
        product_atom = item.get("product_atom")
        src = item.get("source") if isinstance(item.get("source"), dict) else {}
        compact.append(
            {
                "product_atom": product_atom,
                "source_smiles": src.get("smiles"),
                "source_atom_index": src.get("atom_index"),
            }
        )
    raw_confidence = parsed.get("llm_response", {}).get("confidence")
    confidence = _normalise_mapping_confidence(raw_confidence, default=0.0)

    return _serialise(
        {
            "status": parsed.get("status", "success"),
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "compact_mapped_atoms": compact,
            "unmapped_atoms": parsed.get("llm_response", {}).get("unmapped_atoms", []),
            "raw": parsed,
        }
    )


def select_reaction_type(
    *,
    starting_materials: List[str],
    products: List[str],
    balance_analysis: Optional[Dict[str, Any]] = None,
    functional_groups: Optional[Dict[str, Any]] = None,
    ph_recommendation: Optional[Dict[str, Any]] = None,
    initial_conditions: Optional[Dict[str, Any]] = None,
    missing_reagents: Optional[Dict[str, Any]] = None,
    atom_mapping: Optional[Dict[str, Any]] = None,
) -> str:
    """Map the reaction to one known mechanism type label or ``no_match``."""

    catalog = load_reaction_type_catalog_for_runtime()
    templates = list(catalog.get("templates") or [])
    taxonomy_by_label = dict(catalog.get("by_label") or {})
    taxonomy_by_id = dict(catalog.get("by_id") or {})
    choices = list_reaction_type_choices(catalog)

    reaction_type_model = _resolve_step_model(
        "reaction_type_mapping", "MECHANISTIC_REACTION_TYPE_MODEL"
    )
    user_api_key = _get_user_api_key_for_model(reaction_type_model)
    api_key = get_model_api_key(reaction_type_model, user_key=user_api_key)
    if not api_key:
        return _serialise(
            {
                "status": "fallback",
                "selected_label_exact": "no_match",
                "selected_type_id": None,
                "confidence": 0.0,
                "rationale": (
                    f"{get_provider_label(reaction_type_model)} API key not configured; "
                    "reaction type mapping skipped."
                ),
                "top_candidates": [],
                "selected_template": None,
                "available_reaction_type_count": len(templates),
                "model_used": reaction_type_model,
                "tool_calling_used": False,
            }
        )

    taxonomy_lines: List[str] = []
    for item in choices:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label_exact") or "").strip()
        type_id = str(item.get("type_id") or "").strip()
        template = taxonomy_by_id.get(type_id) or {}
        step_hint = int(template.get("suitable_step_count") or 0)
        group = str(item.get("canonical_group") or "").strip()
        taxonomy_lines.append(f"- {type_id}: {label} | group={group} | steps≈{step_hint}")

    system_prompt = (
        "You are an expert mechanistic chemist. Select the single best mechanism type from the "
        "provided taxonomy for the reaction. Use one exact label from the provided list, or "
        "return 'no_match' if none fit well. Do not invent new labels."
    )
    system_prompt = compose_system_prompt(
        call_name="select_reaction_type",
        dynamic_system_prompt=system_prompt,
    )

    context_payload = {
        "balance_analysis": balance_analysis or {},
        "functional_groups": functional_groups or {},
        "ph_recommendation": ph_recommendation or {},
        "initial_conditions": initial_conditions or {},
        "missing_reagents": missing_reagents or {},
        "atom_mapping": atom_mapping or {},
    }
    human_prompt = (
        "Map this reaction to one known mechanism type.\n\n"
        f"Starting materials: {starting_materials}\n"
        f"Products: {products}\n\n"
        "Context from prior analysis steps:\n"
        f"{json.dumps(context_payload, sort_keys=True)}\n\n"
        "Allowed taxonomy labels (exact text):\n"
        + "\n".join(taxonomy_lines)
        + "\n\nChoose one exact label or 'no_match'."
    )
    few_shot_block = format_few_shot_block("select_reaction_type")
    if few_shot_block:
        human_prompt += f"\n\n{few_shot_block}\n"

    use_forced_tools = adapter_supports_forced_tools(reaction_type_model)
    if not use_forced_tools:
        human_prompt += (
            "\n\nReturn JSON only with keys: selected_label_exact, selected_type_id, confidence, "
            "rationale, top_candidates."
        )

    model_used = reaction_type_model
    error_messages: List[str] = []
    response: Any = None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    try:
        llm_kwargs: Dict[str, Any] = {"model": reaction_type_model}
        _apply_reasoning_kwargs(
            llm_kwargs,
            reaction_type_model,
            os.getenv("MECHANISTIC_REACTION_TYPE_REASONING"),
        )
        _apply_output_token_cap(llm_kwargs, reaction_type_model)
        llm = get_chat_model(
            reaction_type_model,
            temperature=None,
            model_kwargs=llm_kwargs.get("model_kwargs"),
            user_api_key=user_api_key,
        )
        if use_forced_tools:
            response = llm.invoke(
                messages,
                tools=[REACTION_TYPE_SELECTION_TOOL],
                tool_choice=build_tool_choice("reaction_type_selection_result"),
            )
        else:
            response = llm.invoke(messages)
    except Exception as primary_error:
        error_messages.append(f"{reaction_type_model}: {primary_error}")
        if use_forced_tools:
            fallback_model = get_fallback_model(family=get_model_family(reaction_type_model))
            if fallback_model == reaction_type_model:
                fallback_model = get_fallback_model()
            if fallback_model != reaction_type_model:
                try:
                    fallback_user_key = _get_user_api_key_for_model(fallback_model)
                    fallback_kwargs: Dict[str, Any] = {"model": fallback_model}
                    _apply_output_token_cap(fallback_kwargs, fallback_model)
                    fallback_llm = get_chat_model(
                        fallback_model,
                        temperature=None,
                        model_kwargs=fallback_kwargs.get("model_kwargs"),
                        user_api_key=fallback_user_key,
                    )
                    response = fallback_llm.invoke(
                        messages,
                        tools=[REACTION_TYPE_SELECTION_TOOL],
                        tool_choice=build_tool_choice("reaction_type_selection_result"),
                    )
                    model_used = fallback_model
                except Exception as fallback_error:
                    error_messages.append(f"{fallback_model}: {fallback_error}")
                    response = None

    def _normalise_selection(parsed: ReactionTypeSelectionPayload) -> Dict[str, Any]:
        selected_label = str(parsed.selected_label_exact or "").strip()
        selected_type_id = parsed.selected_type_id
        if selected_label.lower() == "no_match":
            selected_label = "no_match"
            selected_type_id = None
        elif selected_label not in taxonomy_by_label and selected_type_id in taxonomy_by_id:
            selected_label = str(
                (taxonomy_by_id.get(selected_type_id) or {}).get("label_exact") or selected_label
            )
        elif selected_label not in taxonomy_by_label:
            for label in taxonomy_by_label:
                if label.casefold() == selected_label.casefold():
                    selected_label = label
                    break

        if selected_label != "no_match" and selected_label not in taxonomy_by_label:
            selected_label = "no_match"
            selected_type_id = None

        selected_template = (
            dict(taxonomy_by_label[selected_label])
            if selected_label != "no_match"
            else None
        )
        if selected_template is not None:
            selected_type_id = str(selected_template.get("type_id") or selected_type_id or "")

        top_candidates: List[Dict[str, Any]] = []
        for item in parsed.top_candidates[:3]:
            label = str(item.label_exact or "").strip()
            if not label:
                continue
            if label.lower() != "no_match" and label not in taxonomy_by_label:
                continue
            top_candidates.append(
                {
                    "label_exact": label,
                    "type_id": item.type_id,
                    "confidence": (
                        None if item.confidence is None else max(0.0, min(1.0, float(item.confidence)))
                    ),
                }
            )

        return {
            "selected_label_exact": selected_label,
            "selected_type_id": selected_type_id,
            "confidence": max(0.0, min(1.0, float(parsed.confidence))),
            "rationale": str(parsed.rationale or "").strip(),
            "top_candidates": top_candidates,
            "selected_template": (
                compact_template_for_prompt(selected_template)
                if selected_template is not None
                else None
            ),
        }

    try:
        if response is None:
            raise RuntimeError("; ".join(error_messages) if error_messages else "LLM returned no response")

        usage = getattr(response, "usage", None)
        raw_response_text = ""
        structured: Optional[Dict[str, Any]] = None
        schema_validation: Optional[Dict[str, str]] = None

        if use_forced_tools and hasattr(response, "tool_calls") and response.tool_calls:
            try:
                parsed_tc = json.loads(response.tool_calls[0]["arguments"])
                parsed_tc.pop("text", None)
                structured = _normalise_selection(
                    ReactionTypeSelectionPayload.model_validate(parsed_tc)
                )
                schema_validation = {
                    "status": "ok",
                    "validator": "ReactionTypeSelectionPayload",
                    "source": "tool_call",
                }
            except ValidationError as exc:
                schema_validation = {
                    "status": "fallback",
                    "validator": "ReactionTypeSelectionPayload",
                    "source": "tool_call",
                    "error": _summarise_validation_error(exc),
                }

        if structured is None:
            raw_response_text = extract_text_content(response) or ""
            if raw_response_text:
                try:
                    maybe_json = json.loads(raw_response_text)
                except json.JSONDecodeError:
                    maybe_json = None
                if isinstance(maybe_json, dict):
                    try:
                        structured = _normalise_selection(
                            ReactionTypeSelectionPayload.model_validate(maybe_json)
                        )
                        schema_validation = {
                            "status": "ok",
                            "validator": "ReactionTypeSelectionPayload",
                            "source": "text_json",
                        }
                    except ValidationError as exc:
                        schema_validation = {
                            "status": "fallback",
                            "validator": "ReactionTypeSelectionPayload",
                            "source": "text_json",
                            "error": _summarise_validation_error(exc),
                        }

        if structured is None:
            raw_text = (raw_response_text or "").strip()
            matched_label = None
            for label in taxonomy_by_label:
                if label and label in raw_text:
                    matched_label = label
                    break
            if matched_label is None and "no_match" in raw_text.lower():
                matched_label = "no_match"
            matched_label = matched_label or "no_match"
            template = taxonomy_by_label.get(matched_label) if matched_label != "no_match" else None
            structured = {
                "selected_label_exact": matched_label,
                "selected_type_id": (
                    str((template or {}).get("type_id") or "")
                    if template is not None
                    else None
                ),
                "confidence": 0.35,
                "rationale": "Fallback parse due to invalid structured response.",
                "top_candidates": [],
                "selected_template": (
                    compact_template_for_prompt(template) if template is not None else None
                ),
            }

        result: Dict[str, Any] = {
            "status": "success",
            "selected_label_exact": structured.get("selected_label_exact"),
            "selected_type_id": structured.get("selected_type_id"),
            "confidence": structured.get("confidence"),
            "rationale": structured.get("rationale"),
            "top_candidates": structured.get("top_candidates", []),
            "selected_template": structured.get("selected_template"),
            "available_reaction_type_count": len(templates),
            "taxonomy_labels": [str(item.get("label_exact") or "") for item in choices],
            "model_used": model_used,
            "tool_calling_used": use_forced_tools,
        }
        if raw_response_text:
            result["raw_response"] = raw_response_text
        if schema_validation:
            result["schema_validation"] = schema_validation
        if usage:
            result["_llm_usage"] = usage
        if error_messages and model_used != reaction_type_model:
            result["note"] = "; ".join(error_messages)
        return _serialise(result)
    except Exception as exc:
        return _serialise(
            {
                "status": "fallback",
                "selected_label_exact": "no_match",
                "selected_type_id": None,
                "confidence": 0.0,
                "rationale": f"Reaction type mapping fallback due to error: {exc}",
                "top_candidates": [],
                "selected_template": None,
                "available_reaction_type_count": len(templates),
                "model_used": model_used,
                "tool_calling_used": use_forced_tools,
                "error_details": error_messages,
            }
        )


def _score_protonation(smiles: str) -> Tuple[int, int]:
    """Score a SMILES string for acidic and basic character.
    
    Returns a tuple of (acid_score, base_score) where higher scores indicate
    stronger acidic or basic character.
    """
    _require_rdkit()
    
    mol = _mol_from_smiles(smiles)
    acid_score = 0
    base_score = 0
    
    # Common acidic functional groups
    acid_patterns = [
        # Carboxylic acids
        "[CX3](=[OX1])[OX2H1]",
        # Phenols
        "[cX3]1[cX3][cX3][cX3][cX3][cX3]1[OX2H1]",
        # Sulfonic acids
        "[SX4](=[OX1])(=[OX1])[OX2H1]",
        # Phosphoric acids
        "[PX4](=[OX1])([OX2H1])[OX2H1]",
        # Thiols (weak acids)
        "[SX2H1]",
    ]
    
    # Common basic functional groups
    base_patterns = [
        # Primary amines
        "[NX3H2]",
        # Secondary amines  
        "[NX3H1]",
        # Tertiary amines
        "[NX3H0]",
        # Pyridine-like nitrogens
        "[nX3]",
        # Guanidine
        "[NX3H1][CX3](=[NX3H1])[NX3H1]",
        # Simple ammonia/amine groups (more general)
        "[NX3]",
    ]
    
    for pattern in acid_patterns:
        try:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            acid_score += len(matches)
        except Exception:
            continue
    
    for pattern in base_patterns:
        try:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            base_score += len(matches)
        except Exception:
            continue
    
    return acid_score, base_score


def recommend_ph(starting_materials: List[str], products: List[str], fallback_ph: Optional[float]) -> str:
    """Estimate a pH window when none is supplied, optionally via Dimorphite-DL."""
    if fallback_ph is not None:
        return _serialise({
            "provided_ph": fallback_ph,
            "recommended": fallback_ph,
            "source": "user",
        })

    if DimorphiteDL is not None:  # pragma: no cover - depends on optional package
        runner = None
        for kwargs in (
            {"min_ph": 0.0, "max_ph": 14.0, "step": 0.5},
            {"min_ph": 0.0, "max_ph": 14.0, "step_size": 0.5},
            {"min_ph": 0.0, "max_ph": 14.0, "ph_increment": 0.5},
        ):
            try:
                runner = DimorphiteDL(**kwargs)
                break
            except TypeError:
                continue

        if runner is not None:
            protonation_profiles: Dict[str, List[Dict[str, str]]] = {}
            for smiles_list in (starting_materials, products):
                for smiles in smiles_list:
                    try:
                        protonation_profiles[smiles] = runner.protonate(smiles)
                    except Exception as exc:  # pragma: no cover - defensive
                        protonation_profiles[smiles] = [{"error": str(exc)}]
            return _serialise({
                "recommended": "see_profiles",
                "source": "dimorphite_dl",
                "profiles": protonation_profiles,
            })

    # Fallback heuristic when Dimorphite-DL is unavailable.
    acid_score = 0
    base_score = 0
    for smiles in starting_materials + products:
        acids, bases = _score_protonation(smiles)
        acid_score += acids
        base_score += bases

    if acid_score > base_score:
        recommended = (2.0, 5.0)
        rationale = "acidic functional groups dominate"
    elif base_score > acid_score:
        recommended = (9.0, 12.0)
        rationale = "basic amine-like groups dominate"
    else:
        recommended = (6.0, 8.0)
        rationale = "no clear acid/base dominance"

    return _serialise({
        "recommended_range": recommended,
        "source": "heuristic",
        "acidic_score": acid_score,
        "basic_score": base_score,
        "rationale": rationale,
    })


# Priority order matters: specific subclasses go before broad parent classes
FUNCTIONAL_GROUPS = [
    # very specific / high-priority
    ("nitro", ("[N+](=O)[O-]",), True),
    ("acyl_halide", ("[CX3](=O)[F,Cl,Br,I]",), True),
    ("anhydride", ("[CX3](=O)O[CX3](=O)",), True),

    # sulfur(VI/IV)
    ("sulfonic_acid", ("[SX4](=O)(=O)[OX2H,OX1-]",), True),
    ("sulfonate_ester", ("[SX4](=O)(=O)[OX2][#6]",), True),
    ("sulfonamide", ("[SX4](=O)(=O)[NX3;H0,H1,H2]",), True),
    ("sulfone", ("[SX4](=O)(=O)([!O])[!O]",), True),
    ("sulfoxide", ("[SX3](=O)([!O])[!O]",), True),

    # carbonyl-derived
    ("carboxylic_acid", ("[CX3](=O)[OX2H,OX1-]",), True),
    ("thioester", ("[CX3](=O)[SX2][#6]",), True),
    ("carbamate", ("[NX3][CX3](=O)[OX2][#6]",), True),
    ("lactone", ("[OX2;R][CX3;R](=O)",), True),
    ("ester", ("[CX3](=O)[OX2][#6]",), True),

    ("imide", ("[CX3](=O)[NX3][CX3](=O)",), True),
    ("urea", ("[NX3][CX3](=O)[NX3]",), True),
    ("lactam", ("[NX3;R][CX3;R](=O)",), True),
    ("amide", ("[CX3](=O)[NX3;H0,H1,H2]",), True),

    ("aldehyde", ("[CX3H1](=O)[#6,#1]",), True),
    ("ketone", ("[#6][CX3](=O)[#6]",), True),

    # cumulenes / C=N / C#N family
    ("isocyanate", ("[NX2]=[CX2]=[OX1]",), True),
    ("isothiocyanate", ("[NX2]=[CX2]=[SX1]",), True),
    ("amidine", ("[NX3][CX3]=[NX2]",), True),
    ("guanidine", ("[NX3][CX3](=[NX2])[NX3]",), True),
    ("imine", ("[CX3]=[NX2]",), True),
    ("nitrile", ("[CX2]#N",), True),
    ("azo", ("[NX2]=[NX2]",), True),

    # sulfur(II)
    ("thiol", ("[SX2H][#6]",), True),
    ("disulfide", ("[SX2][SX2]",), True),
    ("sulfide", ("[SX2]([#6])[#6]",), True),

    # oxygen
    ("phenol", ("[OX2H]c",), True),
    ("alcohol", ("[OX2H][CX4]",), True),
    ("ether", ("[OD2]([#6])[#6]",), True),
    ("epoxide", ("[OX2r3]1[#6r3][#6r3]1",), True),

    # nitrogen
    ("quaternary_ammonium", ("[NX4+]",), True),
    ("ammonium", ("[NX3+;H3,H2,H1]",), True),
    (
        "amine",
        (
            "[NX3;H2,H1,H0;+0;!$(N-[C,S,P]=O);!$(N-[S](=O)(=O));!$(N=C);!$(N#*)]",
        ),
        True,
    ),

    # halides / boron
    ("alkyl_halide", ("[CX4][F,Cl,Br,I]",), True),
    ("boronic_acid", ("[BX3]([OX2H])[OX2H]",), True),
    ("boronate_ester", ("[BX3]([OX2][#6])[OX2][#6]",), True),

    # overlapping scaffold/motif descriptors
    # keep these last and non-consuming
    ("aromatic_ring", ("a1aaaaa1",), False),  # benzene-like 6-membered aromatic ring
    ("alkene", ("[CX3]=[CX3]",), False),
    ("alkyne", ("[CX2]#[CX2]",), False),
]

_FUNCTIONAL_GROUP_LABELS: Dict[str, str] = {
    "acyl_halide": "acyl halide",
    "alcohol": "alcohol",
    "aldehyde": "aldehyde",
    "alkene": "alkene",
    "alkyl_halide": "alkyl halide",
    "alkyne": "alkyne",
    "amide": "amide",
    "amidine": "amidine",
    "amine": "amine",
    "ammonium": "ammonium",
    "anhydride": "anhydride",
    "aromatic_ring": "aromatic ring",
    "azo": "azo",
    "boronate_ester": "boronate ester",
    "boronic_acid": "boronic acid",
    "carbamate": "carbamate",
    "carboxylic_acid": "carboxylic acid",
    "disulfide": "disulfide",
    "epoxide": "epoxide",
    "ester": "ester",
    "ether": "ether",
    "guanidine": "guanidine",
    "imide": "imide",
    "imine": "imine",
    "isocyanate": "isocyanate",
    "isothiocyanate": "isothiocyanate",
    "ketone": "ketone",
    "lactam": "lactam",
    "lactone": "lactone",
    "nitrile": "nitrile",
    "nitro": "nitro",
    "phenol": "phenol",
    "quaternary_ammonium": "quaternary ammonium",
    "sulfide": "sulfide",
    "sulfonate_ester": "sulfonate ester",
    "sulfonamide": "sulfonamide",
    "sulfonic_acid": "sulfonic acid",
    "sulfone": "sulfone",
    "sulfoxide": "sulfoxide",
    "thioester": "thioester",
    "thiol": "thiol",
    "urea": "urea",
}

_FUNCTIONAL_GROUP_PRIORITY: Tuple[str, ...] = (
    "nitro",
    "acyl_halide",
    "anhydride",
    "sulfonic_acid",
    "sulfonate_ester",
    "sulfonamide",
    "sulfone",
    "sulfoxide",
    "carboxylic_acid",
    "thioester",
    "carbamate",
    "lactone",
    "ester",
    "imide",
    "urea",
    "lactam",
    "amide",
    "aldehyde",
    "ketone",
    "isocyanate",
    "isothiocyanate",
    "amidine",
    "guanidine",
    "imine",
    "nitrile",
    "azo",
    "thiol",
    "disulfide",
    "sulfide",
    "phenol",
    "alcohol",
    "ether",
    "epoxide",
    "quaternary_ammonium",
    "ammonium",
    "amine",
    "alkyl_halide",
    "boronic_acid",
    "boronate_ester",
    "aromatic_ring",
    "alkene",
    "alkyne",
)


_COMPILED_FG = None


@lru_cache(maxsize=1)
def _get_compiled_functional_groups() -> List[Tuple[str, List[Any], bool]]:
    """Compile SMARTS definitions once for functional-group fingerprinting.
    
    Returns list of (name, compiled_patterns_list, consume_flag) tuples.
    """
    global _COMPILED_FG
    if _COMPILED_FG is not None:
        return _COMPILED_FG

    _require_rdkit()
    compiled: List[Tuple[str, List[Any], bool]] = []
    for name, smarts_list, consume in FUNCTIONAL_GROUPS:
        compiled_patterns: List[Any] = []
        for smarts in smarts_list:
            try:
                pattern = Chem.MolFromSmarts(smarts)
            except Exception:
                pattern = None
            if pattern is not None:
                compiled_patterns.append(pattern)
        if compiled_patterns:
            compiled.append((name, compiled_patterns, consume))
        else:
            logger.warning("No valid SMARTS compiled for functional group '%s'", name)
    _COMPILED_FG = compiled
    return compiled


def _functional_group_fingerprint(smiles: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """Return raw functional group counts for each SMILES string provided.
    
    Uses priority-based atom consumption: specific subclasses consume atoms
    before broad parent classes, preventing double-counting.
    """
    _require_rdkit()
    compiled_groups = _get_compiled_functional_groups()
    summary: Dict[str, Dict[str, int]] = {}
    
    for s in smiles:
        source_smiles = str(s or "").strip()
        candidate = _normalize_smiles_for_fg_matching(source_smiles)
        mol = _mol_from_smiles(candidate)
        if mol is None:
            continue
            
        claimed_atoms = set()
        matches_by_group = defaultdict(list)
        seen_atomsets_by_group = defaultdict(set)
        
        for name, patterns, consume in compiled_groups:
            for patt in patterns:
                for match in mol.GetSubstructMatches(patt, uniquify=True):
                    atomset = frozenset(match)
                    
                    # dedupe same atom set within a label
                    if atomset in seen_atomsets_by_group[name]:
                        continue
                    
                    # priority-based exclusion for consuming groups
                    if consume and (set(match) & claimed_atoms):
                        continue
                    
                    matches_by_group[name].append(match)
                    seen_atomsets_by_group[name].add(atomset)
                    
                    if consume:
                        claimed_atoms.update(match)
        
        # Convert to count dict
        counts: Dict[str, int] = {name: len(matches) for name, matches in matches_by_group.items()}
        summary[source_smiles] = counts
    
    return summary


def _normalize_smiles_for_fg_matching(smiles: str) -> str:
    """Strip atom maps before SMARTS matching while preserving broad parseability."""

    candidate = str(smiles or "").strip()
    if not candidate:
        return ""
    try:
        cleaned = remove_mapping_and_canonicalize(candidate)
    except Exception:
        cleaned = candidate
    cleaned = str(cleaned or "").strip()
    return cleaned or candidate


def find_functional_groups(smiles: str) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Return accepted matches after priority-based atom consumption.

    Output format:
    {
        "amide": [(atom_idx1, atom_idx2, ...), ...],
        "aromatic_ring": [(...), (...)]
    }
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    compiled_groups = _get_compiled_functional_groups()
    claimed_atoms = set()
    matches_by_group = defaultdict(list)
    seen_atomsets_by_group = defaultdict(set)

    for name, patterns, consume in compiled_groups:
        for patt in patterns:
            for match in mol.GetSubstructMatches(patt, uniquify=True):
                atomset = frozenset(match)

                # dedupe same atom set within a label
                if atomset in seen_atomsets_by_group[name]:
                    continue

                # priority-based exclusion for consuming groups
                if consume and (set(match) & claimed_atoms):
                    continue

                matches_by_group[name].append(match)
                seen_atomsets_by_group[name].add(atomset)

                if consume:
                    claimed_atoms.update(match)

    return dict(matches_by_group)


def count_functional_groups(smiles: str) -> Dict[str, int]:
    """
    Count accepted matches after priority resolution.
    """
    found = find_functional_groups(smiles)
    return {name: len(matches) for name, matches in found.items()}


def _aggregate_functional_group_counts(summary: Mapping[str, Mapping[str, int]]) -> Dict[str, int]:
    totals: Counter = Counter()
    for groups in summary.values():
        for label, count in groups.items():
            totals[str(label)] += int(count or 0)
    return dict(totals)


def _ordered_functional_groups(labels: Iterable[str]) -> List[str]:
    preferred = {label: index for index, label in enumerate(_FUNCTIONAL_GROUP_PRIORITY)}
    return sorted(
        {str(label) for label in labels if str(label).strip()},
        key=lambda label: (preferred.get(label, len(preferred)), _FUNCTIONAL_GROUP_LABELS.get(label, label)),
    )


def _functional_group_display_name(label: str) -> str:
    return _FUNCTIONAL_GROUP_LABELS.get(label, str(label or "").replace("_", " "))


def _functional_group_transformation_candidates(consumed: Sequence[str], formed: Sequence[str]) -> List[str]:
    candidates: List[str] = []
    if consumed and formed:
        for left in consumed:
            for right in formed:
                phrase = f"{_functional_group_display_name(left)} -> {_functional_group_display_name(right)}"
                if phrase not in candidates:
                    candidates.append(phrase)
    elif formed:
        for right in formed:
            phrase = f"Formation of {_functional_group_display_name(right)}"
            if phrase not in candidates:
                candidates.append(phrase)
    elif consumed:
        for left in consumed:
            phrase = f"Loss of {_functional_group_display_name(left)}"
            if phrase not in candidates:
                candidates.append(phrase)
    return candidates


def classify_functional_group_transformation(
    starting_materials: Sequence[str],
    products: Sequence[str],
) -> Dict[str, Any]:
    """Describe the likely functional-group transformation for a reaction.

    This helper strips atom maps before matching SMARTS patterns so it is safe to
    use with mapped mechanism states and display-oriented example records.
    """

    start_clean = [item for item in (_normalize_smiles_for_fg_matching(s) for s in starting_materials) if item]
    product_clean = [item for item in (_normalize_smiles_for_fg_matching(s) for s in products) if item]

    start_summary = _functional_group_fingerprint(start_clean) if start_clean else {}
    product_summary = _functional_group_fingerprint(product_clean) if product_clean else {}
    start_counts = _aggregate_functional_group_counts(start_summary)
    product_counts = _aggregate_functional_group_counts(product_summary)

    all_groups = set(start_counts) | set(product_counts)
    consumed = _ordered_functional_groups(
        label for label in all_groups if start_counts.get(label, 0) > product_counts.get(label, 0)
    )
    formed = _ordered_functional_groups(
        label for label in all_groups if product_counts.get(label, 0) > start_counts.get(label, 0)
    )
    shared = _ordered_functional_groups(
        label
        for label in all_groups
        if start_counts.get(label, 0) > 0 and product_counts.get(label, 0) > 0
    )

    candidates = _functional_group_transformation_candidates(consumed, formed)
    uncertain = len(candidates) > 1 or (not candidates and len(shared) > 1)
    uncertainty_note = ""
    if len(candidates) > 1:
        uncertainty_note = (
            "Multiple functional-group transformations matched the stripped SMILES; "
            "treat the label as a best-effort description."
        )
    elif not candidates and len(shared) > 1:
        uncertainty_note = (
            "Multiple functional groups persist across the reaction, so the dominant "
            "transformation is unclear from SMARTS alone."
        )

    if candidates:
        label = candidates[0]
    elif shared:
        label = f"Retained {_functional_group_display_name(shared[0])}"
    else:
        label = "No clear functional-group change"

    llm_summary = f"Functional-group transformation label: {label}."
    if candidates and len(candidates) > 1:
        llm_summary += f" Alternatives: {', '.join(candidates[1:4])}."
    if uncertainty_note:
        llm_summary += f" {uncertainty_note}"

    return {
        "label": label,
        "label_candidates": candidates,
        "uncertain": uncertain,
        "uncertainty_note": uncertainty_note,
        "starting_materials_stripped": start_clean,
        "products_stripped": product_clean,
        "starting_group_counts": start_counts,
        "product_group_counts": product_counts,
        "consumed_groups": consumed,
        "formed_groups": formed,
        "shared_groups": shared,
        "llm_summary": llm_summary,
    }


_FUNCTIONAL_GROUP_HISTORY: deque[Tuple[Tuple[str, ...], Dict[str, Dict[str, int]], str]] = deque(maxlen=6)


def _record_functional_group_summary(
    smiles: Sequence[str], summary: Dict[str, Dict[str, int]], origin: str
) -> None:
    """Cache functional group fingerprints for reuse between tool calls."""

    snapshot = {sm: dict(groups) for sm, groups in summary.items()}
    _FUNCTIONAL_GROUP_HISTORY.append((tuple(smiles), snapshot, origin))


def _retrieve_functional_group_context(
    smiles: Sequence[str],
) -> Tuple[Dict[str, Dict[str, int]], str]:
    """Return cached functional group fingerprints when available."""

    smiles_list = list(smiles)
    if not smiles_list:
        return {}, "no molecules supplied"

    unique_smiles = list(dict.fromkeys(smiles_list))
    target_set = set(unique_smiles)

    aggregated: Dict[str, Dict[str, int]] = {}
    aggregated_origins: set[str] = set()

    for cached_smiles, summary, origin in reversed(_FUNCTIONAL_GROUP_HISTORY):
        available = set(summary.keys())
        if target_set.issubset(available):
            filtered = {sm: dict(summary[sm]) for sm in unique_smiles if sm in summary}
            source = (
                "from prior fingerprint_functional_groups tool call"
                if origin == "tool"
                else "from cached functional group analysis"
            )
            _record_functional_group_summary(unique_smiles, filtered, origin)
            return filtered, source

        used_entry = False
        for sm in unique_smiles:
            if sm in summary and sm not in aggregated:
                aggregated[sm] = dict(summary[sm])
                used_entry = True
        if used_entry:
            aggregated_origins.add(origin)
        if len(aggregated) == len(target_set):
            source = (
                "from prior fingerprint_functional_groups tool call"
                if aggregated_origins == {"tool"}
                else "from cached functional group analysis"
            )
            _record_functional_group_summary(unique_smiles, aggregated, "cached_combo")
            return aggregated, source

    computed = _functional_group_fingerprint(unique_smiles)
    _record_functional_group_summary(unique_smiles, computed, "computed")
    return computed, "computed directly for this atom mapping step"


_INITIAL_CONDITION_HISTORY: deque[Dict[str, Any]] = deque(maxlen=6)


def _record_initial_condition_summary(
    starting_materials: Sequence[str],
    products: Sequence[str],
    payload: Mapping[str, Any],
    source: str,
) -> None:
    """Store the latest initial condition assessment for reuse across tools."""

    try:
        canonical_payload: Dict[str, Any] = json.loads(_serialise(dict(payload)))
    except Exception:
        canonical_payload = dict(payload)

    entry = {
        "starting_materials": tuple(starting_materials),
        "products": tuple(products),
        "payload": canonical_payload,
        "source": source,
    }

    # Remove any existing entry for the same reaction snapshot
    existing_indices = [
        idx
        for idx, cached in enumerate(_INITIAL_CONDITION_HISTORY)
        if cached.get("starting_materials") == entry["starting_materials"]
        and cached.get("products") == entry["products"]
    ]
    for idx in reversed(existing_indices):
        try:
            del _INITIAL_CONDITION_HISTORY[idx]
        except IndexError:  # pragma: no cover - defensive removal
            continue

    _INITIAL_CONDITION_HISTORY.append(entry)


def _retrieve_initial_condition_context(
    starting_materials: Sequence[str],
    products: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return the most recent initial condition payload if available."""

    target_reactants = tuple(starting_materials)
    target_products = tuple(products)

    for cached in reversed(_INITIAL_CONDITION_HISTORY):
        if (
            cached.get("starting_materials") == target_reactants
            and cached.get("products") == target_products
        ):
            payload = cached.get("payload")
            if isinstance(payload, dict):
                return dict(payload), str(cached.get("source", "cached"))
            break

    return None, "unavailable"


def fingerprint_functional_groups(smiles: List[str]) -> str:
    """Return a qualitative functional group summary for each SMILES."""
    result = _functional_group_fingerprint(smiles)
    _record_functional_group_summary(smiles, result, "tool")
    return _serialise({"functional_groups": result})


class MechanismIntermediate(BaseModel):
    """Structured representation of a single mechanistic intermediate."""

    model_config = ConfigDict(extra="ignore")

    smiles: str = Field(..., description="SMILES string for the proposed intermediate.")
    type: Optional[str] = Field(
        default=None, description="Optional short label describing the intermediate type."
    )
    note: Optional[str] = Field(
        default=None, description="Optional short rationale for the intermediate."
    )

    @field_validator("smiles")
    @classmethod
    def _validate_smiles(cls, value: str) -> str:
        text = str(value or "").strip()
        corrected, _ = _apply_smiles_correction(text)
        text = corrected
        if not _looks_like_smiles(text):
            raise ValueError(f"intermediate.smiles is not SMILES-like: {text}")
        if Chem is not None:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                raise ValueError(f"intermediate.smiles failed RDKit validation: {text}")
            return Chem.MolToSmiles(mol) or text
        return text


class MechanismStepCandidate(BaseModel):
    """Structured ranked candidate for multi-candidate mechanism proposals."""

    model_config = ConfigDict(extra="ignore")

    rank: int
    intermediate_smiles: str
    reaction_description: str
    reaction_smirks: Optional[str] = None
    electron_pushes: List[Dict[str, object]] = Field(default_factory=list)
    resulting_state: Optional[List[str]] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    template_alignment: Optional[Literal["aligned", "partial", "not_aligned", "unknown"]] = None
    template_alignment_reason: Optional[str] = None
    intermediate_type: Optional[str] = None
    note: Optional[str] = None

    @field_validator("intermediate_smiles")
    @classmethod
    def _validate_candidate_smiles(cls, value: str) -> str:
        text = str(value or "").strip()
        corrected, _ = _apply_smiles_correction(text)
        text = corrected
        if not _looks_like_smiles(text):
            raise ValueError(f"candidate.intermediate_smiles is not SMILES-like: {text}")
        if Chem is not None:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                raise ValueError(f"candidate.intermediate_smiles failed RDKit validation: {text}")
            return Chem.MolToSmiles(mol) or text
        return text

    @field_validator("reaction_smirks")
    @classmethod
    def _validate_reaction_smirks(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text

    @field_validator("electron_pushes")
    @classmethod
    def _validate_electron_pushes(cls, value: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return [move.as_dict() for move in normalize_electron_pushes(value)]


class MechanismStepPrediction(BaseModel):
    """Structured output returned by the LLM when proposing the next step.

    Supports both the new ``candidates`` schema and the legacy
    ``intermediates`` list for backward compatibility.
    """

    model_config = ConfigDict(extra="ignore")

    classification: Literal["intermediate_step", "final_step"]
    candidates: List[MechanismStepCandidate] = Field(
        default_factory=list,
        description="Ranked mechanism candidates (new schema).",
    )
    intermediates: List[MechanismIntermediate] = Field(
        default_factory=list,
        description="Legacy intermediate list (backward compatibility).",
    )
    analysis: str = Field(
        ..., description="Brief explanation of the mechanistic reasoning for this step."
    )


class ReactionTypeTopCandidate(BaseModel):
    """Optional alternative reaction type candidate."""

    model_config = ConfigDict(extra="ignore")

    label_exact: str
    type_id: Optional[str] = None
    confidence: Optional[float] = None


class ReactionTypeSelectionPayload(BaseModel):
    """Structured payload for reaction type mapping."""

    model_config = ConfigDict(extra="ignore")

    selected_label_exact: str
    selected_type_id: Optional[str] = None
    confidence: float
    rationale: str
    top_candidates: List[ReactionTypeTopCandidate] = Field(default_factory=list)

    @field_validator("selected_label_exact")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("selected_label_exact must be a non-empty string")
        return text

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        score = float(value)
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score


def propose_intermediates(
    starting_materials: List[str],
    products: List[str],
    current_state: Optional[List[str]] = None,
    previous_intermediates: Optional[List[str]] = None,
    mapped_starting_materials: Optional[List[str]] = None,
    mapped_products: Optional[List[str]] = None,
    mapped_current_state: Optional[List[str]] = None,
    ph: Optional[float] = None,
    temperature: Optional[float] = None,
    step_index: Optional[int] = None,
    step_mapping_context: Optional[Dict[str, Any]] = None,
    template_guidance: Optional[Dict[str, Any]] = None,
) -> str:
    """Use LLM to propose intermediates for the next mechanistic step.
    
    Analyzes the overall transformation, considers pH and temperature context,
    and determines if the final product is reached or if more intermediates are needed.
    """
    
    # Get the model to use
    intermediate_model = _resolve_step_model("intermediates", "MECHANISTIC_INTERMEDIATE_MODEL")
    
    # Prepare context
    if current_state is None:
        current_state = starting_materials.copy()
    
    if previous_intermediates is None:
        previous_intermediates = []

    if mapped_starting_materials is None:
        mapped_starting_materials = []
    if mapped_products is None:
        mapped_products = []
    if mapped_current_state is None:
        mapped_current_state = []

    compaction_notes: List[str] = []
    prompt_char_cap = int(os.getenv("MECHANISTIC_PROMPT_CHAR_CAP", "60000"))
    if is_openrouter_model(intermediate_model):
        prompt_char_cap = int(os.getenv("MECHANISTIC_OPENROUTER_PROMPT_CHAR_CAP", "42000"))
    elif intermediate_model.lower().startswith("gpt-4o"):
        prompt_char_cap = int(os.getenv("MECHANISTIC_GPT4O_PROMPT_CHAR_CAP", "30000"))

    max_prev = max(2, int(os.getenv("MECHANISTIC_PREVIOUS_INTERMEDIATES_MAX", "12")))
    if len(previous_intermediates) > max_prev:
        dropped = len(previous_intermediates) - max_prev
        previous_intermediates = previous_intermediates[-max_prev:]
        compaction_notes.append(
            f"Compacted older intermediates: dropped {dropped}, kept {max_prev} most recent."
        )

    max_current_maps = max(2, int(os.getenv("MECHANISTIC_MAPPED_CURRENT_STATE_MAX", "8")))
    if len(mapped_current_state) > max_current_maps:
        dropped = len(mapped_current_state) - max_current_maps
        mapped_current_state = mapped_current_state[-max_current_maps:]
        compaction_notes.append(
            f"Compacted mapped current-state context: dropped {dropped}, kept {max_current_maps} most recent species."
        )

    def _compact_json_context(
        payload: Optional[Dict[str, Any]],
        *,
        label: str,
        max_chars: int,
    ) -> str:
        if not isinstance(payload, dict):
            return ""
        compact = dict(payload)
        if isinstance(compact.get("alignment_history"), list):
            history = list(compact.get("alignment_history") or [])
            keep = max(1, int(os.getenv("MECHANISTIC_ALIGNMENT_HISTORY_MAX", "4")))
            if len(history) > keep:
                compact["alignment_history"] = history[-keep:]
                compact["alignment_history_truncated"] = len(history) - keep
                compaction_notes.append(
                    f"Compacted {label}: dropped {len(history) - keep} older alignment history entries."
                )
        rendered = json.dumps(compact, sort_keys=True, separators=(",", ":"))
        if len(rendered) > max_chars:
            compaction_notes.append(
                f"Compacted {label}: trimmed verbose JSON context to fit prompt budget."
            )
            return rendered[: max(256, max_chars - 32)] + "...(truncated)"
        return rendered
    
    fg_enabled = _functional_group_analysis_enabled()

    if fg_enabled:
        functional_groups_start, start_source = _retrieve_functional_group_context(starting_materials)
        functional_groups_products, product_source = _retrieve_functional_group_context(products)
        functional_groups_current, current_source = _retrieve_functional_group_context(current_state)

        def _format_functional_group_summary(summary: Dict[str, Dict[str, int]]) -> str:
            lines: List[str] = []
            for smiles, groups in summary.items():
                reactive_groups = [
                    f"{label} (×{count})" if count > 1 else label
                    for label, count in sorted(groups.items())
                    if count > 0
                ]
                if reactive_groups:
                    lines.append(f"    - {smiles}: {', '.join(reactive_groups)}")
                else:
                    lines.append(f"    - {smiles}: none detected")
            return "\n".join(lines) if lines else "    - none"

        functional_group_section = (
            "Functional group analysis:\n"
            f"  Starting materials ({start_source}):\n{_format_functional_group_summary(functional_groups_start)}\n"
            f"  Target products ({product_source}):\n{_format_functional_group_summary(functional_groups_products)}\n"
            f"  Current state ({current_source}):\n{_format_functional_group_summary(functional_groups_current)}\n\n"
        )
    else:
        functional_group_section = ""
    
    # Check if final products are already present
    final_products_present = all(product in current_state for product in products)
    
    # Pull reagent suggestions from the cached conditions assessment.
    conditions_context, _conditions_source = _retrieve_initial_condition_context(starting_materials, products)
    reagent_candidates: List[Dict[str, Any]] = []
    reagent_label: Optional[str] = None
    if conditions_context:
        cond_env = str(conditions_context.get("environment") or "").strip().lower()
        cond_ph = conditions_context.get("representative_ph")
        cond_acid = conditions_context.get("acid_candidates") or []
        cond_base = conditions_context.get("base_candidates") or []

        if cond_acid and cond_base:
            logger.warning(
                "propose_intermediates: conditions cache contains both acid_candidates and "
                "base_candidates (environment=%r, ph=%r). Selecting based on environment.",
                cond_env,
                cond_ph,
            )

        # Select candidates: acid for acidic env, base for basic env, neither for neutral / pH≈7.
        if cond_env == "acidic" and cond_acid:
            reagent_candidates = list(cond_acid)
            reagent_label = "Suggested acid reagents from conditions assessment"
        elif cond_env == "basic" and cond_base:
            reagent_candidates = list(cond_base)
            reagent_label = "Suggested base reagents from conditions assessment"
        else:
            # Neutral or pH ~7: include neither; also fall back to pH check when env is unset.
            try:
                ph_float = float(cond_ph) if cond_ph is not None else None
            except (TypeError, ValueError):
                ph_float = None
            if ph_float is not None and ph_float != 7.0:
                if ph_float < 7.0 and cond_acid:
                    reagent_candidates = list(cond_acid)
                    reagent_label = "Suggested acid reagents from conditions assessment"
                elif ph_float > 7.0 and cond_base:
                    reagent_candidates = list(cond_base)
                    reagent_label = "Suggested base reagents from conditions assessment"

    system_prompt = (
        "You are an expert organic chemist specializing in reaction mechanism prediction. "
        "You receive: the original starting materials (reactant SMILES), the target products "
        "(product SMILES), the current mechanistic state (SMILES of species present after the "
        "last accepted step), a list of previously accepted intermediates, pH, temperature, and "
        "the step index (how many mechanistic steps have been accepted so far). "
        "When optional atom-mapped context is provided, use those atom-map indices when writing "
        "reaction_smirks and electron_pushes. If no mapped context is provided, do not invent "
        "atom maps; prefer chemically plausible candidates and let downstream mapping/validation handle the rest. "
        "Use all of this context to determine whether additional mechanistic steps are required "
        "and to propose candidates for the next mechanism step.\n\n"
        "Provide a classification of the current state as either 'intermediate_step' when further "
        "transformations are needed or 'final_step' when the target products have been reached.\n\n"
        "For each candidate, provide:\n"
        "- A rank (1 = most likely, 2 = next, 3 = least likely)\n"
        "- The intermediate product as a SMILES string\n"
        "- A balanced reaction description (bonds broken/formed, electron movement)\n"
        "- Confidence level (high, medium, low)\n"
        "- Optional type annotation and mechanistic note\n\n"
        "Propose up to 3 ranked candidates when multiple plausible pathways exist. "
        "If only one pathway is clearly dominant, a single candidate is acceptable. "
        "If the mechanism is complete, return an empty candidates list.\n\n"
        "Include a concise analysis explaining the mechanistic logic behind your decision. When "
        "charged intermediates are involved, use bracket notation with explicit hydrogens and place "
        "the charge indicator at the end of the bracket (e.g., [OH+], [O-], [NH4+])."
    )
    system_prompt = compose_system_prompt(
        call_name="propose_mechanism_step",
        dynamic_system_prompt=system_prompt,
    )

    human_prompt = (
        f"Analyze this chemical reaction and propose the next mechanistic step.\n\n"
        f"Overall Transformation:\n"
        f"  Starting materials: {starting_materials}\n"
        f"  Target products: {products}\n"
        f"  Current state: {current_state} "
        f"(species present after the last accepted step; equals starting materials at step 0)\n\n"
    )

    if ph is not None:
        human_prompt += f"Reaction conditions: pH = {ph}"
    if temperature is not None:
        human_prompt += f", Temperature = {temperature}°C"
    if ph is not None or temperature is not None:
        human_prompt += "\n\n"

    if step_index is not None:
        human_prompt += f"Step index: {step_index} (number of mechanistic steps accepted so far)\n\n"

    if mapped_starting_materials or mapped_products or mapped_current_state:
        human_prompt += (
            "Optional atom-mapped context for arrow-pushing notation. Use these atom-map indices if you provide "
            "reaction_smirks and electron_pushes:\n"
            f"  Atom-mapped starting materials: {mapped_starting_materials or []}\n"
            f"  Atom-mapped target products: {mapped_products or []}\n"
            f"  Atom-mapped current state: {mapped_current_state or []}\n\n"
        )

    human_prompt += functional_group_section

    if previous_intermediates:
        human_prompt += f"Previously accepted intermediates (do not repeat these): {previous_intermediates}\n\n"

    step_mapping_context_text = _compact_json_context(
        step_mapping_context,
        label="step_mapping_context",
        max_chars=max(1200, prompt_char_cap // 6),
    )
    if step_mapping_context_text:
        human_prompt += (
            "Recent atom-lineage context from the last accepted step (use to narrow plausible pathways):\n"
            f"{step_mapping_context_text}\n\n"
        )

    guidance_payload: Dict[str, Any] = dict(template_guidance or {})
    template_guidance_text = _compact_json_context(
        guidance_payload,
        label="template_guidance",
        max_chars=max(1200, prompt_char_cap // 6),
    )
    if template_guidance_text:
        guidance_strength = str(guidance_payload.get("guidance_strength") or "strong").strip().lower()
        guidance_blurb = (
            "The template may be followed when it aligns with observed chemistry and validation, "
            "but you may deviate whenever a different step is more chemically plausible."
        )
        if guidance_strength == "weak":
            guidance_blurb = (
                "Treat the template as a low-priority hint only. Prefer chemistry and validator-aligned "
                "step proposals when the template conflicts with plausible mechanism progression."
            )
        human_prompt += (
            "Optional reaction-type template guidance (advisory only, do NOT force-fit chemistry):\n"
            f"{guidance_blurb}\n"
            f"{template_guidance_text}\n\n"
        )

    # Incomplete-payload feedback from prior attempt (set by coordinator on reproposal)
    incomplete_payload_reasons = guidance_payload.get("incomplete_payload_reasons")
    if isinstance(incomplete_payload_reasons, list) and incomplete_payload_reasons:
        reason_str = ", ".join(str(r) for r in incomplete_payload_reasons)
        human_prompt += (
            "REPROPOSAL NOTICE — your previous candidates were rejected as structurally incomplete "
            f"({reason_str}). This means the validator could not find valid electron-push moves or "
            "a parseable reaction_smirks. Please ensure every candidate includes:\n"
            "  • reaction_smirks: CXSMILES/SMIRKS with a |mech:v1;...| block (e.g., "
            "[C:1][O:2]>>[C:1].[O:2] |mech:v1;lp:2>1|)\n"
            "  • electron_pushes: at least one move object; for lone_pair moves include "
            "kind, source_atom, target_atom, electrons=2; for pi_bond/sigma_bond moves include "
            "kind, source_bond ([bondStart, bondEnd]), through_atom (must equal source_bond[1]), "
            "target_atom, electrons=2.\n\n"
        )

    # Peer proposals appendix (decentralized topology, rounds 2+)
    peer_proposals = guidance_payload.get("peer_proposals")
    if isinstance(peer_proposals, list) and peer_proposals:
        peer_lines: List[str] = []
        for pp in peer_proposals[:9]:  # cap at 9 peer candidates
            if isinstance(pp, dict):
                smiles = str(pp.get("smiles") or "").strip()
                reaction = str(pp.get("reaction") or "").strip()
                if smiles:
                    line = smiles
                    if reaction:
                        line += f" — {reaction}"
                    peer_lines.append(f"  - {line}")
        if peer_lines:
            human_prompt += (
                "Other agents proposed the following candidates in the previous round:\n"
                + "\n".join(peer_lines)
                + "\nConsider these proposals when formulating your candidates. "
                "You may agree, refine, or propose alternatives.\n\n"
            )

    if reagent_candidates and reagent_label:
        def _fmt_reagent(item: Any) -> str:
            if isinstance(item, dict):
                parts = [str(item.get("name") or item.get("smiles") or "unknown")]
                if item.get("smiles"):
                    parts.append(str(item["smiles"]))
                if item.get("justification"):
                    parts.append(str(item["justification"]))
                return " — ".join(parts)
            return str(item)
        reagent_lines = "; ".join(_fmt_reagent(r) for r in reagent_candidates[:3])
        human_prompt += f"{reagent_label}: {reagent_lines}\n\n"
    
    if final_products_present:
        human_prompt += (
            f"ANALYSIS: All final products are already present in the current state.\n"
            f"This appears to be the final step of the mechanism.\n\n"
            f"Please confirm this is the final step and provide a brief summary of the transformation."
        )
    else:
        considerations = []
        if fg_enabled:
            considerations.append("The functional groups present and their reactivity")
        considerations.extend(
            [
                "The overall transformation pattern",
                "Common mechanistic steps for this type of reaction",
                "pH and temperature effects on reactivity",
                "Avoiding reverse reactions or previously explored intermediates",
            ]
        )
        human_prompt += (
            f"ANALYSIS: The reaction is not yet complete. Propose up to 3 ranked candidates "
            f"for the next mechanism step. Each candidate should include the intermediate "
            f"product (SMILES), a balanced description of the reaction step, reaction_smirks "
            f"(with '|mech:v1;...|' metadata), and electron_pushes.\n\n"
            f"Consider:\n"
            + "\n".join(f"{index + 1}. {item}" for index, item in enumerate(considerations))
            + "\n\n"
            "Rank candidates by likelihood (rank 1 = most likely). If only one pathway "
            "is clearly dominant, a single candidate is sufficient. Include a balanced "
            "reaction description for each candidate explaining bonds broken/formed. "
            "When atom balance appears inconsistent, include minimal additional reagents/byproducts in resulting_state. "
            "For each candidate provide reaction_smirks as CXSMILES/SMIRKS with a "
            "'|mech:v1;lp:a>b;pi:a-b>c;sigma:a-b>c|' block using atom-map indices and include "
            "electron_pushes entries with kind, target_atom, and either source_atom or source_bond/through_atom.\n\n"
            "When expressing charged intermediates, use bracket notation with explicit hydrogens: "
            "include 'H' if the bracketed atom is bonded to hydrogen (with a numeric count when more "
            "than one), followed by the charge sign (e.g., [NH4+], [OH-], [Ti+4]). Multiple charges can "
            "be shown with digits or repeated signs (e.g., [Co+3] or [Co+++])."
            " Ensure the charge sign appears at the end of the bracket after any atoms or hydrogens ("
            "e.g., [OH+], [C-], not [O+H])."
        )
    few_shot_block = format_few_shot_block("propose_mechanism_step")
    if few_shot_block:
        if len(human_prompt) + len(few_shot_block) <= prompt_char_cap:
            human_prompt += f"\n\n{few_shot_block}\n"
        else:
            compaction_notes.append(
                "Skipped older few-shot examples to preserve recent state and tool-call structure under prompt budget."
            )

    # For models that don't support tool calling, keep the JSON instruction.
    _use_forced_tools = adapter_supports_forced_tools(intermediate_model)
    if not _use_forced_tools:
        human_prompt += (
            "\n\nReturn JSON only with keys: "
            "classification ('intermediate_step' or 'final_step'), "
            "candidates (list of {rank, intermediate_smiles, reaction_description, confidence, "
            "template_alignment, template_alignment_reason, intermediate_type, note, reaction_smirks, electron_pushes, resulting_state}), "
            "and analysis (string)."
        )

    if len(human_prompt) > prompt_char_cap:
        # Preserve opening reaction context and final tool schema instructions; compact older middle context.
        head_keep = max(1500, int(prompt_char_cap * 0.45))
        tail_keep = max(2000, int(prompt_char_cap * 0.45))
        compact_notice = (
            "\n\n[context_compacted]\n"
            "Older context was truncated to keep prompt within provider limits. "
            "Most recent state and tool-call requirements were preserved.\n\n"
        )
        human_prompt = human_prompt[:head_keep] + compact_notice + human_prompt[-tail_keep:]
        compaction_notes.append(
            "Prompt exceeded provider cap; truncated older middle context and preserved tool-call instructions."
        )
    if compaction_notes:
        logger.warning(
            "propose_intermediates prompt compaction applied for model %s: %s",
            intermediate_model,
            " | ".join(compaction_notes[:4]),
        )

    _intermediate_user_key = _get_user_api_key_for_model(intermediate_model)
    api_key = get_model_api_key(intermediate_model, user_key=_intermediate_user_key)
    if not api_key:
        raise RuntimeError(
            f"{get_provider_label(intermediate_model)} API key not configured; "
            "unable to propose intermediates."
        )

    model_used = intermediate_model
    error_messages: List[str] = []
    response: Any = None

    try:
        llm_kwargs: Dict[str, Any] = {"model": intermediate_model}
        if _supports_temperature_parameter(intermediate_model):
            llm_kwargs["temperature"] = 0.3
        _apply_reasoning_kwargs(
            llm_kwargs,
            intermediate_model,
            os.getenv("MECHANISTIC_INTERMEDIATE_REASONING"),
        )
        _apply_output_token_cap(llm_kwargs, intermediate_model)
        llm = get_chat_model(
            intermediate_model,
            temperature=llm_kwargs.get("temperature"),
            model_kwargs=llm_kwargs.get("model_kwargs"),
            user_api_key=_intermediate_user_key,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
        if _use_forced_tools:
            response = llm.invoke(
                messages,
                tools=[MECHANISM_STEP_PROPOSAL_TOOL],
                tool_choice=build_tool_choice("mechanism_step_proposal_result"),
            )
        else:
            response = llm.invoke(messages)
    except Exception as primary_error:
        error_messages.append(f"{intermediate_model}: {primary_error}")
        # Try fallback model for tool-capable models.
        if _use_forced_tools:
            fallback_model = get_fallback_model(family=get_model_family(intermediate_model))
            if fallback_model == intermediate_model:
                fallback_model = get_fallback_model()
            if fallback_model != intermediate_model:
                try:
                    fallback_user_key = _get_user_api_key_for_model(fallback_model)
                    fallback_kwargs: Dict[str, Any] = {"model": fallback_model}
                    if _supports_temperature_parameter(fallback_model):
                        fallback_kwargs["temperature"] = llm_kwargs.get("temperature")
                    _apply_output_token_cap(fallback_kwargs, fallback_model)
                    fallback_llm = get_chat_model(
                        fallback_model,
                        temperature=fallback_kwargs.get("temperature"),
                        model_kwargs=fallback_kwargs.get("model_kwargs"),
                        user_api_key=fallback_user_key,
                    )
                    response = fallback_llm.invoke(
                        messages,
                        tools=[MECHANISM_STEP_PROPOSAL_TOOL],
                        tool_choice=build_tool_choice("mechanism_step_proposal_result"),
                    )
                    model_used = fallback_model
                except Exception as fallback_error:
                    error_messages.append(f"{fallback_model}: {fallback_error}")
                    response = None

    try:
        if response is None:
            aggregated_error = "; ".join(error_messages) if error_messages else "LLM returned no response"
            raise RuntimeError(aggregated_error)

        _intermediate_usage = getattr(response, "usage", None)

        raw_response_text = ""
        structured_payload: Optional[Dict[str, Any]] = None
        schema_validation: Optional[Dict[str, str]] = None

        # Try to extract from forced tool call first.
        if _use_forced_tools and hasattr(response, "tool_calls") and response.tool_calls:
            try:
                parsed_tc = json.loads(response.tool_calls[0]["arguments"])
                parsed_tc.pop("text", None)  # remove commentary field
                structured_payload = MechanismStepPrediction.model_validate(parsed_tc).model_dump()
                schema_validation = {"status": "ok", "validator": "MechanismStepPrediction", "source": "tool_call"}
                raw_response_text = json.dumps(structured_payload, indent=2)
            except ValidationError as exc:
                schema_validation = {
                    "status": "fallback",
                    "validator": "MechanismStepPrediction",
                    "source": "tool_call",
                    "error": _summarise_validation_error(exc),
                }
                structured_payload = None
            except Exception:
                structured_payload = None

        # Fall back to text parsing.
        if structured_payload is None:
            raw_response_text = extract_text_content(response) or ""
            if raw_response_text:
                try:
                    parsed = json.loads(raw_response_text)
                    if isinstance(parsed, dict):
                        structured_payload = MechanismStepPrediction.model_validate(parsed).model_dump()
                        schema_validation = {
                            "status": "ok",
                            "validator": "MechanismStepPrediction",
                            "source": "text_json",
                        }
                except ValidationError as exc:
                    if schema_validation is None:
                        schema_validation = {
                            "status": "fallback",
                            "validator": "MechanismStepPrediction",
                            "source": "text_json",
                            "error": _summarise_validation_error(exc),
                        }
                    structured_payload = None
                except Exception:
                    structured_payload = None

        if not raw_response_text and structured_payload is not None:
            raw_response_text = json.dumps(structured_payload, indent=2)

        if not structured_payload and raw_response_text:
            try:
                maybe_parsed = json.loads(raw_response_text)
            except json.JSONDecodeError:
                maybe_parsed = None
            if isinstance(maybe_parsed, dict):
                structured_payload = maybe_parsed

        def _salvage_from_raw(text: str) -> List[Dict[str, Any]]:
            import re

            if not text:
                return []

            candidates: List[Dict[str, Any]] = []

            fenced_json = re.findall(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
            for block in fenced_json:
                try:
                    snippet = json.loads(block)
                except json.JSONDecodeError:
                    continue
                if isinstance(snippet, dict):
                    maybe_ranked = snippet.get("candidates")
                    if isinstance(maybe_ranked, list):
                        for item in maybe_ranked:
                            if not isinstance(item, dict):
                                continue
                            smiles = item.get("intermediate_smiles")
                            if not smiles:
                                continue
                            record = {
                                "smiles": str(smiles),
                                "rank": item.get("rank"),
                                "reaction_description": item.get("reaction_description"),
                                "confidence": item.get("confidence"),
                                "type": item.get("intermediate_type"),
                                "note": item.get("note"),
                                "source": "json_block",
                            }
                            candidates.append(record)
                    maybe_items = snippet.get("intermediates")
                    if isinstance(maybe_items, list):
                        for item in maybe_items:
                            if isinstance(item, dict):
                                record = dict(item)
                            else:
                                record = {"smiles": str(item)}
                            record.setdefault("source", "json_block")
                            candidates.append(record)
                elif isinstance(snippet, list):
                    for item in snippet:
                        if isinstance(item, dict):
                            record = dict(item)
                        else:
                            record = {"smiles": str(item)}
                        record.setdefault("source", "json_block")
                        candidates.append(record)

            labelled_pattern = re.compile(
                r"(?i)smiles?\s*[:\-]\s*([A-Za-z0-9@+\-\[\]()=#\\/\.:%*]+)"
            )
            for match in labelled_pattern.finditer(text):
                candidate_smiles = match.group(1)
                if not _looks_like_smiles(candidate_smiles):
                    continue
                candidates.append(
                    {
                        "smiles": candidate_smiles,
                        "source": "labelled_line",
                    }
                )

            generic_pattern = re.compile(r'[A-Za-z0-9@+\-\[\]()=#\\/\.:%*]+')
            for token in generic_pattern.findall(text):
                if len(token) <= 2:
                    continue
                if not _looks_like_smiles(token):
                    continue
                lowered = token.lower()
                if lowered.startswith((
                    "ph", "temperature", "step", "analysis",
                    "diene", "dienophile", "cycloadd", "pericycl", "concerted",
                    "retro", "suprafacial", "antarafacial",
                )):
                    continue
                # Skip tokens that are substrings of common chemistry prose.
                if any(kw in lowered for kw in (
                    "catalyz", "protona", "deprotona", "nucleophil",
                    "electrophil", "eliminat", "rearrang", "substitut",
                    "cycloaddition", "pericyclic", "orbital", "suprafacial",
                    "frontier", "homo", "lumo",
                )):
                    continue
                candidates.append({"smiles": token, "source": "regex_scan"})

            deduped: List[Dict[str, Any]] = []
            seen_tokens: set[str] = set()
            for candidate in candidates:
                smiles_value = candidate.get("smiles")
                if not smiles_value:
                    continue
                if smiles_value in seen_tokens:
                    continue
                seen_tokens.add(smiles_value)
                deduped.append(candidate)

            return deduped

        candidate_specs: List[Dict[str, Any]] = []
        rejected_candidates: List[Dict[str, Any]] = []
        # Track ranked candidates from the new multi-candidate schema.
        ranked_candidates: List[Dict[str, Any]] = []
        def _candidate_ready_for_execution_payload(item: Dict[str, Any]) -> Tuple[bool, str]:
            repaired, repair_reason = repair_candidate_reaction_smirks(
                reaction_smirks=item.get("reaction_smirks"),
                electron_pushes=item.get("electron_pushes"),
            )
            if not repaired:
                return False, repair_reason or "reaction_smirks_invalid"
            item["reaction_smirks"] = repaired
            if repair_reason:
                item["mechanism_move_repair"] = repair_reason
            dbe_entries = synthesize_dbe_entries(item.get("electron_pushes"))
            if dbe_entries:
                item["dbe_repair"] = "synthesized_dbe_from_electron_pushes"
                item["inferred_dbe"] = dbe_entries

            valid_pushes = normalize_electron_pushes(item.get("electron_pushes"))
            if not valid_pushes:
                return False, "invalid_electron_pushes"
            return True, ""
        if structured_payload:
            # New schema: candidates[] with rank, intermediate_smiles, reaction_description
            maybe_candidates = structured_payload.get("candidates")
            if isinstance(maybe_candidates, list) and maybe_candidates:
                for item in maybe_candidates:
                    if not isinstance(item, dict):
                        continue
                    smiles = item.get("intermediate_smiles")
                    if not smiles:
                        continue
                    ready, reject_reason = _candidate_ready_for_execution_payload(item)
                    if not ready:
                        rejected_candidates.append(
                            {
                                "rank": item.get("rank"),
                                "intermediate_smiles": smiles,
                                "reason": reject_reason,
                                "mechanism_move_repair": item.get("mechanism_move_repair"),
                                "dbe_repair": item.get("dbe_repair"),
                            }
                        )
                        continue
                    ranked_candidates.append(item)
                    candidate_specs.append(
                        {
                            "smiles": smiles,
                            "type": item.get("intermediate_type"),
                            "note": item.get("note"),
                            "rank": item.get("rank"),
                            "reaction_description": item.get("reaction_description"),
                            "confidence": item.get("confidence"),
                            "template_alignment": item.get("template_alignment"),
                            "template_alignment_reason": item.get("template_alignment_reason"),
                            "reaction_smirks": item.get("reaction_smirks"),
                            "electron_pushes": item.get("electron_pushes"),
                            "resulting_state": item.get("resulting_state"),
                            "mechanism_move_repair": item.get("mechanism_move_repair"),
                            "dbe_repair": item.get("dbe_repair"),
                            "inferred_dbe": item.get("inferred_dbe"),
                            "source": "model_json",
                        }
                    )

            # Legacy schema: intermediates[] with smiles, type, note
            if not candidate_specs:
                maybe_intermediates = structured_payload.get("intermediates")
                if isinstance(maybe_intermediates, list):
                    for item in maybe_intermediates:
                        if isinstance(item, dict):
                            candidate_specs.append(
                                {
                                    "smiles": item.get("smiles"),
                                    "type": item.get("type"),
                                    "note": item.get("note"),
                                    "source": "model_json",
                                }
                            )
                        elif item is not None:
                            candidate_specs.append(
                                {
                                    "smiles": str(item),
                                    "source": "model_json",
                                }
                            )

        # Only salvage from raw text when structured parsing actually failed.
        # If structured_payload was successfully parsed (even with empty
        # candidates/intermediates), the LLM intentionally returned no
        # candidates — do not regex-mine the analysis text.
        _structured_parse_succeeded = (
            structured_payload is not None
            and schema_validation is not None
            and schema_validation.get("status") == "ok"
        )
        if not candidate_specs and not _structured_parse_succeeded:
            candidate_specs.extend(_salvage_from_raw(raw_response_text))

        validated_intermediates: List[str] = []
        intermediate_records: List[Dict[str, Any]] = []
        invalid_intermediates: List[Dict[str, Any]] = []

        processed_signatures: set[str] = set()
        raw_fallbacks: List[str] = []
        rdkit_errors: List[str] = []

        def _handle_candidate(smiles: Optional[str], metadata: Dict[str, Any]) -> None:
            if not smiles:
                return

            canonical, details = _canonicalise_candidate_smiles(smiles)
            details["source"] = metadata.get("source", "unknown")
            if metadata.get("type"):
                details["type"] = metadata["type"]
            if metadata.get("note"):
                details["note"] = metadata["note"]

            signature_candidates = [
                details.get("canonical"),
                details.get("cleaned"),
                details.get("normalized"),
                details.get("raw_cleaned"),
            ]
            for signature in signature_candidates:
                if isinstance(signature, str) and signature in processed_signatures:
                    return

            if canonical:
                intermediate_records.append(details)
                if canonical not in validated_intermediates:
                    validated_intermediates.append(canonical)
            else:
                invalid_intermediates.append(details)
                fallback_smiles = (
                    details.get("normalized")
                    or details.get("raw_cleaned")
                    or details.get("raw")
                )
                if isinstance(fallback_smiles, str) and fallback_smiles:
                    if fallback_smiles not in raw_fallbacks:
                        raw_fallbacks.append(fallback_smiles)
                error_text = details.get("error")
                if isinstance(error_text, str) and error_text:
                    rdkit_errors.append(error_text)

            for signature in signature_candidates:
                if isinstance(signature, str):
                    processed_signatures.add(signature)

        for candidate in candidate_specs:
            _handle_candidate(candidate.get("smiles"), candidate)

        if not validated_intermediates and not raw_fallbacks and not rejected_candidates:
            raise RuntimeError("LLM did not return a valid intermediate SMILES string")

        step_classification = "intermediate_step"
        classification_value: Optional[str] = None
        if structured_payload:
            raw_classification = structured_payload.get("classification")
            if isinstance(raw_classification, str):
                classification_value = raw_classification
        if classification_value:
            if classification_value.lower().startswith("final"):
                step_classification = "final_step"
        elif final_products_present or "final step" in raw_response_text.lower():
            step_classification = "final_step"

        analysis_text = None
        if structured_payload and isinstance(structured_payload.get("analysis"), str):
            analysis_text = structured_payload["analysis"]
        if not analysis_text:
            analysis_text = raw_response_text

        has_validated = bool(validated_intermediates)
        chosen_intermediates: List[str]
        status_value = "success"
        message_text: Optional[str] = None
        validation_status = "validated" if has_validated else "unvalidated"
        
        if has_validated:
            chosen_intermediates = validated_intermediates[:2]
            message_text = f"Proposed {len(validated_intermediates)} validated intermediate(s) for next step"
        else:
            chosen_intermediates = raw_fallbacks[:2]
            status_value = "warning"
            message_text = (
                "No intermediates passed RDKit validation; providing raw LLM proposals for review"
            )

        result: Dict[str, Any] = {
            "status": status_value,
            "step_classification": step_classification,
            "proposed_intermediates": chosen_intermediates,
            "llm_reasoning": analysis_text,
            "raw_response": raw_response_text,
            "final_products_present": final_products_present,
            "current_state": current_state,
            "model_used": model_used,
            "tool_calling_used": _use_forced_tools,
            "validation_status": validation_status,
        }

        # Include ranked candidates from multi-candidate schema.
        if ranked_candidates:
            result["candidates"] = ranked_candidates
        if rejected_candidates:
            result["rejected_candidates"] = rejected_candidates[:10]

        if structured_payload is not None:
            result["structured_response"] = structured_payload
        if schema_validation is not None:
            result["schema_validation"] = schema_validation

        result["validated_intermediates"] = intermediate_records[:5]
        if invalid_intermediates:
            result["invalid_intermediates"] = invalid_intermediates[:10]
        if raw_fallbacks:
            result["raw_intermediates"] = raw_fallbacks[:5]
        if rdkit_errors:
            result["rdkit_errors"] = rdkit_errors[:5]

        result["intermediate_count"] = len(chosen_intermediates)

        if step_classification == "final_step":
            result["message"] = "Final products reached - mechanism complete"
        else:
            result["message"] = message_text
        if compaction_notes:
            prior_msg = str(result.get("message") or "").strip()
            compact_msg = "Prompt context was compacted (older context omitted)."
            result["message"] = f"{prior_msg} {compact_msg}".strip()

        if error_messages and model_used != intermediate_model:
            result["note"] = "; ".join(error_messages)
        if not has_validated and invalid_intermediates:
            first_error = invalid_intermediates[0].get("error")
            if first_error and result.get("note"):
                result["note"] += f" | RDKit validation failed: {first_error}"
            elif first_error:
                result["note"] = f"RDKit validation failed: {first_error}"
        if _intermediate_usage:
            result["_llm_usage"] = _intermediate_usage
        if compaction_notes:
            result["prompt_compacted"] = True
            result["compaction_warning"] = (
                "Prompt context was compacted to respect model token limits. Older context may be omitted."
            )
            result["compaction_notes"] = compaction_notes[:8]

        return _serialise(result)
        
    except Exception as e:
        # Fallback to heuristic approach if LLM fails
        _require_rdkit()
        seen: set[str] = set()
        if current_state:
            seen.update(current_state)
        if previous_intermediates:
            seen.update(previous_intermediates)

        intermediates: List[Dict[str, object]] = []

        def _append_candidate(entry: Dict[str, object]) -> None:
            smiles_value = entry.get("smiles")
            if not smiles_value:
                return
            try:
                canonical = _mol_to_extended_smiles(
                    _mol_from_smiles(str(smiles_value)), canonical=True
                )
            except Exception:
                return
            if canonical in seen:
                return
            candidate = dict(entry)
            candidate["smiles"] = canonical
            intermediates.append(candidate)
            seen.add(canonical)

        for sm in starting_materials:
            mol = _mol_from_smiles(sm)
            for atom in mol.GetAtoms():
                if atom.GetDegree() >= 3 and atom.GetHybridization() == Chem.HybridizationType.SP3:
                    radical = Chem.RWMol(mol)
                    radical.RemoveAtom(atom.GetIdx())
                    radical_smiles = _mol_to_extended_smiles(
                        radical.GetMol(), canonical=True
                    )
                    if radical_smiles in seen:
                        continue
                    intermediates.append(
                        {
                            "derived_from": sm,
                            "type": "radical_fragment",
                            "smiles": radical_smiles,
                        }
                    )
                    seen.add(radical_smiles)
                    break

        start_counts = _atom_counter(starting_materials)
        product_counts = _atom_counter(products)
        delta: Dict[str, int] = {}
        all_elements = set(start_counts) | set(product_counts)
        for element in all_elements:
            delta[element] = product_counts.get(element, 0) - start_counts.get(element, 0)

        if delta.get("O", 0) > 0:
            _append_candidate(
                {
                    "type": "oxygen_source",
                    "smiles": "[O]",
                    "note": "Oxygen insertion suggested by product deficit",
                }
            )
        if delta.get("H", 0) < 0:
            _append_candidate(
                {
                    "type": "hydrogen_byproduct",
                    "smiles": "[HH]",
                    "note": "Excess hydrogen may leave as hydrogen gas",
                }
            )

        if previous_intermediates:
            for prior in reversed(previous_intermediates):
                _append_candidate(
                    {
                        "type": "recovered_intermediate",
                        "smiles": prior,
                        "note": "Reusing prior intermediate after LLM failure",
                    }
                )
                if len(intermediates) >= 5:
                    break

        current_counts = _atom_counter(current_state) if current_state else Counter()
        outstanding: Dict[str, int] = {}
        for element in set(product_counts) | set(current_counts):
            outstanding[element] = product_counts.get(element, 0) - current_counts.get(element, 0)

        element_templates: Dict[str, Dict[str, object]] = {
            "O": {
                "smiles": "[O]",
                "type": "oxygen_source",
                "note": "Products contain more oxygen than current state",
            },
            "H": {
                "smiles": "[OH3+]",
                "type": "proton_source",
                "note": "Products require additional hydrogen; consider hydronium-mediated transfer",
            },
            "N": {
                "smiles": "[NH4+]",
                "type": "ammonium_source",
                "note": "Nitrogen deficit suggests ammonium-like intermediate",
            },
            "Cl": {
                "smiles": "Cl",
                "type": "halide_source",
                "note": "Halogen balance indicates chloride participation",
            },
            "Br": {
                "smiles": "Br",
                "type": "halide_source",
                "note": "Halogen balance indicates bromide participation",
            },
            "F": {
                "smiles": "F",
                "type": "halide_source",
                "note": "Halogen balance indicates fluoride participation",
            },
            "I": {
                "smiles": "I",
                "type": "halide_source",
                "note": "Halogen balance indicates iodide participation",
            },
        }

        for element, amount in sorted(outstanding.items(), key=lambda item: item[1], reverse=True):
            if amount <= 0:
                continue
            template = element_templates.get(element)
            if template:
                _append_candidate(template)
            else:
                _append_candidate(
                    {
                        "type": "atom_balance_hint",
                        "smiles": element,
                        "note": f"Products require {amount} additional {element} atom(s)",
                    }
                )
            if len(intermediates) >= 5:
                break

        for product in products:
            if len(intermediates) >= 5:
                break
            _append_candidate(
                {
                    "type": "product_candidate",
                    "smiles": product,
                    "note": "Target product not yet present; treating as terminal intermediate",
                }
            )
        
        fallback_payload = {
            "status": "fallback",
            "step_classification": "intermediate_step",
            "intermediates": intermediates[:5],
            "non_executable_fallback": True,
            "error": f"LLM call failed: {str(e)}",
            "message": "Using heuristic fallback due to LLM error",
        }
        if compaction_notes:
            fallback_payload["prompt_compacted"] = True
            fallback_payload["compaction_warning"] = (
                "Prompt context was compacted to respect model token limits. Older context may be omitted."
            )
            fallback_payload["compaction_notes"] = compaction_notes[:8]
        if error_messages:
            fallback_payload["error_details"] = error_messages
        return _serialise(fallback_payload)


def predict_mechanistic_step(
    step_index: int,
    current_state: List[str],
    target_products: List[str],
    electron_pushes: List[Dict[str, object]],
    reaction_smirks: Optional[str] = None,
    predicted_intermediate: Optional[str] = None,
    resulting_state: Optional[List[str]] = None,
    previous_intermediates: Optional[List[str]] = None,
    note: Optional[str] = None,
    starting_materials: Optional[List[str]] = None,
) -> str:
    """Validate and record a single mechanistic electron-pushing step.

    Parameters
    ----------
    step_index:
        Zero-based index indicating the position of this step in the overall
        mechanism.
    current_state:
        Ordered list of species (SMILES) present prior to the step.
    target_products:
        Target products used to determine whether this step reaches the final
        outcome.
    electron_pushes:
        A list describing each explicit arrow-pushing move. Prefer the new
        schema with ``kind`` plus ``source_atom`` or ``source_bond`` and
        ``target_atom``; legacy ``start_atom``/``end_atom`` lone-pair entries
        are still accepted as a compatibility fallback.
    reaction_smirks:
        CXSMILES/SMIRKS representation of the step. LLM-facing paths should use
        a ``|mech:v1;...|`` block; deterministic validation will continue using
        `dbe` internally and may infer it from the explicit moves.
    predicted_intermediate:
        The intermediate produced by this step, if any.
    resulting_state:
        Optional explicit list of species after the step. When omitted the tool
        will append ``predicted_intermediate`` to ``current_state`` if supplied.
    previous_intermediates:
        A history of intermediates encountered so far used to guard against
        accidental reverse steps.
    note:
        Free-form explanatory text carried alongside the structured payload.
    starting_materials:
        Original starting materials used to detect unchanged returns and prevent
        no-change steps in the mechanism.
    """

    if step_index < 0:
        raise ValueError("step_index must be at least 0")
    if not electron_pushes:
        raise ValueError("At least one electron push must be provided")

    cleaned_pushes = [move.as_dict() for move in normalize_electron_pushes(electron_pushes)]
    if not cleaned_pushes:
        raise ValueError("At least one valid explicit electron push must be provided")

    raw_intermediate = predicted_intermediate
    intermediate_details: Dict[str, Any] = {}
    predicted_intermediate = None

    if raw_intermediate is not None:
        canonical, details = _canonicalise_candidate_smiles(str(raw_intermediate))
        intermediate_details.update(details)
        if canonical is not None:
            predicted_intermediate = canonical
        else:
            fallback_intermediate = (
                details.get("normalized")
                or details.get("raw_cleaned")
                or details.get("raw")
            )
            if (
                isinstance(fallback_intermediate, str)
                and fallback_intermediate
                and _looks_like_smiles(fallback_intermediate)
            ):
                predicted_intermediate = fallback_intermediate
                intermediate_details["validated"] = False
                intermediate_details["fallback_used"] = True

    def _dedupe_species(items: List[str]) -> List[str]:
        seen: set[str] = set()
        deduped: List[str] = []
        for raw in items:
            token = str(raw).strip()
            if not token:
                continue
            signature = token
            if Chem is not None:
                try:
                    mol = Chem.MolFromSmiles(token)
                    if mol is not None:
                        signature = Chem.MolToSmiles(mol) or token
                except Exception:
                    signature = token
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(token)
        return deduped

    resulting: List[str]
    if resulting_state is None:
        resulting = list(current_state)
        if predicted_intermediate and predicted_intermediate not in resulting:
            resulting.append(predicted_intermediate)
    else:
        resulting = [str(item) for item in resulting_state]
        if predicted_intermediate and predicted_intermediate not in resulting:
            resulting.append(predicted_intermediate)
    resulting_state = _dedupe_species(resulting)

    visited: set[str] = set(previous_intermediates or [])
    is_reverse = False
    if predicted_intermediate and predicted_intermediate in visited:
        is_reverse = True

    resulting_set = set(resulting_state)
    current_set = set(current_state)
    resulting_state_changed = resulting_set != current_set

    # Check for unchanged starting materials
    is_unchanged_starting_materials = False
    if starting_materials:
        starting_set = set(starting_materials)
        if starting_set == resulting_set:
            is_unchanged_starting_materials = True
    if not resulting_state_changed:
        is_unchanged_starting_materials = True

    contains_products = any(product in resulting_state for product in target_products)

    raw_reaction_smirks: Optional[str] = None
    reaction_smirks_core: Optional[str] = None
    bond_electron_validation: Dict[str, Any] = {}
    bond_electron_deltas_serialised: List[Dict[str, object]] = []

    if reaction_smirks is not None:
        raw_reaction_smirks = str(reaction_smirks).strip()
        if raw_reaction_smirks:
            mech_core, moves, mech_details = extract_mechanism_moves(raw_reaction_smirks)
            core, deltas, details = _extract_dbe_or_infer(raw_reaction_smirks, electron_pushes=cleaned_pushes)
            reaction_smirks_core = core
            actual_bond_deltas = reaction_bond_deltas(mech_core or core)
            implied_deltas = implied_bond_deltas(moves)
            bond_electron_validation = {
                "valid": details.get("error") is None,
                "message": "Delta bond-electron entries parsed successfully"
                if details.get("error") is None
                else None,
            }
            if deltas:
                bond_electron_deltas_serialised = [delta.as_dict() for delta in deltas]
            if details.get("metadata"):
                bond_electron_validation["metadata"] = details["metadata"]
            if details.get("dbe"):
                bond_electron_validation["dbe"] = details["dbe"]
            if details.get("total_delta") is not None:
                bond_electron_validation["total_delta"] = details["total_delta"]
            if mech_details.get("mech"):
                bond_electron_validation["mech"] = mech_details["mech"]
            if mech_details.get("moves"):
                bond_electron_validation["moves"] = mech_details["moves"]
            if details.get("source"):
                bond_electron_validation["dbe_source"] = details["source"]
            if implied_deltas:
                bond_electron_validation["expected_bond_deltas_from_moves"] = implied_deltas
                bond_electron_validation["observed_bond_deltas"] = actual_bond_deltas
            if mech_details.get("error"):
                bond_electron_validation["mech_warning"] = mech_details["error"]
            if details.get("error"):
                bond_electron_validation["valid"] = False
                bond_electron_validation["error"] = details["error"]
        else:
            bond_electron_validation = {
                "valid": False,
                "error": "reaction_smirks was provided but is empty after stripping whitespace",
            }
    else:
        bond_electron_validation = {
            "valid": False,
            "error": "reaction_smirks not provided; supply reaction_smirks plus electron_pushes or dbe metadata",
        }

    payload: Dict[str, object] = {
        "step_index": step_index,
        "current_state": current_state,
        "resulting_state": resulting_state,
        "predicted_intermediate": predicted_intermediate,
        "electron_pushes": cleaned_pushes,
        "contains_target_product": contains_products,
        "reverse_reaction_detected": is_reverse,
        "unchanged_starting_materials_detected": is_unchanged_starting_materials,
        "resulting_state_changed": resulting_state_changed,
    }
    payload["bond_electron_validation"] = bond_electron_validation
    if bond_electron_deltas_serialised:
        payload["bond_electron_deltas"] = bond_electron_deltas_serialised
    if reaction_smirks_core:
        payload["reaction_smirks"] = reaction_smirks_core
    if raw_reaction_smirks:
        payload["raw_reaction_smirks"] = raw_reaction_smirks
    if not bond_electron_validation.get("valid"):
        payload["bond_electron_guidance"] = {
            "format": _MECH_FORMAT_GUIDANCE,
            "example": _MECH_EXAMPLE,
        }
    if note:
        payload["note"] = note
    if intermediate_details:
        payload["intermediate_details"] = intermediate_details

    status: str = "accepted"
    reason: Optional[str] = None

    if is_reverse:
        status = "rejected"
        reason = "Intermediate was previously observed; treated as reverse reaction"
    elif is_unchanged_starting_materials and not contains_products:
        status = "rejected"
        reason = "Resulting state matches current state; propose a different mechanistic step"
    elif not contains_products:
        if not predicted_intermediate:
            status = "rejected"
            reason = "No intermediate SMILES provided; supply a valid intermediate for forward progress"
        elif not resulting_state_changed:
            status = "rejected"
            reason = "Resulting state unchanged; provide an intermediate distinct from the current state"

    if (
        status == "accepted"
        and not bond_electron_validation.get("valid", False)
        and bond_electron_validation.get("error")
    ):
        status = "accepted_with_warnings"
        bond_electron_validation["level"] = "warning"
    elif bond_electron_validation.get("valid"):
        bond_electron_validation["level"] = "ok"

    payload["status"] = status
    if reason:
        payload["reason"] = reason

    return _serialise(payload)


TOOLKIT = [
    ToolDescriptor(
        name="analyse_balance",
        description="Check atomic balance between starting materials and products.",
        func=analyse_balance,
    ),
    ToolDescriptor(
        name="fingerprint_functional_groups",
        description="Summarise functional groups present in the provided SMILES list.",
        func=fingerprint_functional_groups,
    ),
    ToolDescriptor(
        name="assess_initial_conditions",
        description="Estimate a compatible acidic/basic environment and suggest supportive additives.",
        func=assess_initial_conditions,
    ),
    ToolDescriptor(
        name="predict_missing_reagents",
        description="Use LLM to analyze reaction and suggest missing starting materials and/or products based on stoichiometry and functional groups.",
        func=predict_missing_reagents,
    ),
    ToolDescriptor(
        name="attempt_atom_mapping",
        description="Use an LLM to infer likely atom-to-atom mappings between reactants and products.",
        func=attempt_atom_mapping,
    ),
    ToolDescriptor(
        name="select_reaction_type",
        description="Use an LLM to map the reaction into a known mechanism taxonomy label or no_match.",
        func=select_reaction_type,
    ),
    ToolDescriptor(
        name="recommend_ph",
        description="Recommend pH if none provided, using Dimorphite-DL when available.",
        func=recommend_ph,
    ),
    ToolDescriptor(
        name="propose_intermediates",
        description="Use LLM to analyze reaction state and propose next intermediates or determine if final products are reached.",
        func=propose_intermediates,
    ),
    ToolDescriptor(
        name="predict_mechanistic_step",
        description="Validate a single electron-pushing step within the mechanism loop. REQUIRED: electron_pushes (explicit move objects with kind/source/target fields), step_index, current_state, target_products, reaction_smirks (CXSMILES/SMIRKS plus '|mech:v1;...|'). Optional: predicted_intermediate, resulting_state, previous_intermediates, note, starting_materials.",
        func=predict_mechanistic_step,
    ),
]


__all__ = ["TOOLKIT"]
