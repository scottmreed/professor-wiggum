#!/usr/bin/env python3
"""Build rxn_map expansion artifacts from eval tiers/eval set.

Outputs:
  - training_data/rxn_map_expanded.json
  - training_data/eval_tiers_first20_mechanism_map.json
  - training_data/rxn_map_visualizer.ipynb
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.core.arrow_push import predict_arrow_push_annotation  # noqa: E402

try:  # pragma: no cover - optional at runtime
    from rdkit import Chem
    from rdkit import RDLogger
except Exception:  # pragma: no cover - defensive
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]


TRAINING_DIR = PROJECT_ROOT / "training_data"
EVAL_SET_PATH = TRAINING_DIR / "eval_set.json"
EVAL_TIERS_PATH = TRAINING_DIR / "eval_tiers.json"
OUT_EXPANDED_PATH = TRAINING_DIR / "rxn_map_expanded.json"
OUT_FLAT_MAP_PATH = TRAINING_DIR / "eval_mechanism_map.json"
PREFERRED_EVAL_MAP_PATH = TRAINING_DIR / "eval_mechanism_map.json"
OUT_REACTION_TYPE_TEMPLATES_PATH = TRAINING_DIR / "reaction_type_templates.json"
OUT_NOTEBOOK_PATH = TRAINING_DIR / "rxn_map_visualizer.ipynb"

if RDLogger is not None:  # pragma: no cover - optional
    RDLogger.DisableLog("rdApp.error")


@dataclass(frozen=True)
class MechanismType:
    type_id: str
    label_exact: str
    slug: str
    normalized: str


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_label(text))
    slug = slug.strip("_")
    return slug or "unknown"


def _strip_atom_maps(smiles: str) -> str:
    return re.sub(r":\d+\]", "]", str(smiles or ""))


def _canonical_without_maps(smiles: str) -> str:
    token = _strip_atom_maps(smiles)
    if Chem is None:
        return token
    mol = Chem.MolFromSmiles(token)
    if mol is None:
        return token
    return Chem.MolToSmiles(mol, canonical=True)


def load_taxonomy(path: Path) -> list[MechanismType]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("templates") or [])
    parsed: list[MechanismType] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label_exact") or "").strip()
        if not label:
            continue
        type_id = str(row.get("type_id") or "").strip() or f"mt_{idx:03d}"
        parsed.append(
            MechanismType(
                type_id=type_id,
                label_exact=label,
                slug=_slugify(label),
                normalized=_normalize_label(label),
            )
        )
    parsed.sort(key=lambda item: int(item.type_id.split("_", 1)[1]) if item.type_id.startswith("mt_") else 10_000_000)
    return parsed


def _taxonomy_lookup(taxonomy: list[MechanismType]) -> dict[str, MechanismType]:
    return {item.normalized: item for item in taxonomy}


def _taxonomy_lookup_by_id(taxonomy: list[MechanismType]) -> dict[str, MechanismType]:
    return {item.type_id: item for item in taxonomy}


def load_preferred_eval_mechanism_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("reaction_id") or "").strip()
        if not rid:
            continue
        out[rid] = {
            "reaction_id": rid,
            "mechanism_type_label": str(row.get("mechanism_type_label") or "").strip(),
            "mechanism_type_id": str(row.get("mechanism_type_id") or "").strip(),
            "confidence": row.get("confidence"),
            "rationale": str(row.get("rationale") or "").strip(),
        }
    return out


def _ordered_tier_ids(tiers_payload: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    ordered: list[tuple[str, str]] = []
    for tier_name in ("easy", "medium", "hard"):
        for rid in tiers_payload.get(tier_name, []):
            ordered.append((str(rid), tier_name))
    return [rid for rid, _ in ordered], {rid: tier for rid, tier in ordered}


def _select_ids(
    eval_set: list[dict[str, Any]],
    tiers_payload: dict[str, Any],
    scope: str,
    max_cases: int | None,
) -> tuple[list[str], dict[str, str], str]:
    tier_ids, tier_by_id = _ordered_tier_ids(tiers_payload)
    eval_ids = [str(item.get("id")) for item in eval_set]

    if scope == "first20":
        selected = tier_ids[:20]
        scope_text = "first_20_tier_reactions"
    elif scope == "tier30":
        selected = tier_ids[:30]
        scope_text = "tier_30_reactions"
    else:
        selected = eval_ids
        scope_text = "all_eval_reactions"

    if max_cases is not None and max_cases > 0:
        selected = selected[:max_cases]

    scoped_tier_by_id = {rid: tier_by_id.get(rid, "unscoped") for rid in selected}
    return selected, scoped_tier_by_id, scope_text


# Curated overrides for first 20 tier reactions (accuracy-first pass).
CURATED_LABEL_OVERRIDES: dict[str, tuple[str, float, str]] = {
    "hb350_001": ("Wittig reaction with ylide", 0.42, "Single-step ylide-like deprotonation pattern from a cationic precursor."),
    "hb350_002": ("SN2 reaction", 0.95, "Halide exchange (allylic chloride to iodide) is most consistent with substitution."),
    "hb350_003": ("Alcohol attack to carbonyl or sulfonyl", 0.62, "Ethoxide-promoted displacement at a sulfonyl-derived substrate."),
    "hb350_004": ("SN2 reaction", 0.55, "Base-promoted transformation from alkyl bromide scaffold assigned to substitution-class bucket."),
    "hb350_005": ("Base catalyzed ester hydrolysis", 0.96, "Hydroxide/water with ester substrate and carboxylate/acid product."),
    "hb350_038": ("Methyl ester synthesis", 0.97, "Diazomethane methylation of a carboxylic acid functionality."),
    "hb350_039": ("SN2 reaction with alcohol(thiol)", 0.88, "O-alkylation (alcohol-derived oxygen methylation) via substitution chemistry."),
    "hb350_040": ("SN2 reaction with alcohol(thiol)", 0.95, "Williamson-type ether formation from phenoxide and ethyl iodide."),
    "hb350_041": ("SN2 reaction with alcohol(thiol)", 0.57, "Alkoxy installation at a halogen-bearing carbon assigned to alcohol/thiol substitution bucket."),
    "hb350_042": ("SN2 reaction with alcohol(thiol)", 0.74, "Alkoxide/halide substitution pathway feeding cyclic ether closure."),
    "hb350_151": ("Aldol addition", 0.45, "Strong-base enolate-generating trajectory best aligned with aldol-family enolate chemistry."),
    "hb350_152": ("SN1 reaction", 0.98, "Tertiary benzylic chloride hydrolysis proceeds through carbocation-like substitution."),
    "hb350_153": ("SN2 reaction with alcohol(thiol)", 0.82, "Phenoxide-driven epoxide/alkyl substitution under basic conditions."),
    "hb350_259": ("SNAr reaction (para)", 0.72, "Aryl amination from aryl halide under strongly basic nucleophilic aromatic substitution conditions."),
    "hb350_260": ("Aldol condensation", 0.43, "Acid-mediated dehydration-like elimination from a beta-hydroxy acid motif."),
    "hb350_261": ("Friedel Crafts acylation", 0.64, "Electrophilic aromatic substitution assignment for strongly acidic aromatic nitration conditions."),
    "hb350_262": ("Ester reduction", 0.97, "Hydride reduction of ester functionality to alcohol/alkoxide state."),
    "hb350_294": ("Imine formation", 0.59, "Nitrile trapping/cyclization sequence containing C=N bond construction."),
    "hb350_295": ("Imine formation", 0.58, "Acidic nitrile-driven pathway leading to heterocycle with imine character."),
    "hb350_296": ("Carboxylic acid derivative hydrolysis or formation", 0.94, "Acyl substitution giving ester product from activated carboxylic derivative."),
}


GENERIC_TEMPLATE_OVERRIDES: dict[str, dict[str, Any]] = {
    _normalize_label("SN2 reaction"): {
        "current_state_generic": ["R-Br", "Cl-"],
        "resulting_state_generic": ["R-Cl", "Br-"],
        "reaction_generic": "R-Br.Cl->>R-Cl.Br- |dbe:C-Br:-2;C-Cl:+2;Cl-Cl:-2;Br-Br:+2|",
        "electron_pushes_generic": [
            {"start_atom": "Cl:lp", "end_atom": "C", "electrons": 2, "description": "Backside attack by chloride."},
            {"start_atom": "C-Br", "end_atom": "Br", "electrons": 2, "description": "C-Br bond breaks to bromide."},
        ],
        "generic_notes": "Canonical bimolecular substitution template.",
    },
    _normalize_label("SN2 reaction with alcohol(thiol)"): {
        "current_state_generic": ["R-LG", "RO-"],
        "resulting_state_generic": ["R-OR", "LG-"],
        "reaction_generic": "R-LG.RO->>R-OR.LG- |dbe:C-LG:-2;C-O:+2;O-O:-2;LG-LG:+2|",
        "electron_pushes_generic": [
            {"start_atom": "O:lp", "end_atom": "C", "electrons": 2, "description": "Alkoxide/thiolate attack."},
            {"start_atom": "C-LG", "end_atom": "LG", "electrons": 2, "description": "Leaving-group departure."},
        ],
        "generic_notes": "Williamson-like substitution motif.",
    },
    _normalize_label("SN1 reaction"): {
        "current_state_generic": ["R-LG", "Nu"],
        "resulting_state_generic": ["R-Nu", "LG-"],
        "reaction_generic": "R-LG.Nu>>R+.LG- >> R-Nu |dbe:C-LG:-2;C-Nu:+2|",
        "electron_pushes_generic": [
            {"start_atom": "C-LG", "end_atom": "LG", "electrons": 2, "description": "Ionization to carbocation."},
            {"start_atom": "Nu:lp", "end_atom": "C+", "electrons": 2, "description": "Nucleophilic capture."},
        ],
        "generic_notes": "Unimolecular substitution via carbocation intermediate.",
    },
    _normalize_label("SNAr reaction (para)"): {
        "current_state_generic": ["Ar-LG", "Nu-"],
        "resulting_state_generic": ["Ar-Nu", "LG-"],
        "reaction_generic": "Ar-LG.Nu->>Ar(Nu)(LG)- >> Ar-Nu.LG- |dbe:Ar-LG:-2;Ar-Nu:+2|",
        "electron_pushes_generic": [
            {"start_atom": "Nu:lp", "end_atom": "Ar-ipso", "electrons": 2, "description": "Addition to aromatic ipso carbon."},
            {"start_atom": "Ar-LG", "end_atom": "LG", "electrons": 2, "description": "Elimination restores aromaticity."},
        ],
        "generic_notes": "Addition-elimination aromatic substitution (para class).",
    },
    _normalize_label("SNAr reaction with alcohol (ortho)"): {
        "current_state_generic": ["Ar-LG", "RO-"],
        "resulting_state_generic": ["Ar-OR", "LG-"],
        "reaction_generic": "Ar-LG.RO->>Ar(OR)(LG)- >> Ar-OR.LG- |dbe:Ar-LG:-2;Ar-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "O:lp", "end_atom": "Ar-ipso", "electrons": 2, "description": "Alkoxide addition at ipso carbon."},
            {"start_atom": "Ar-LG", "end_atom": "LG", "electrons": 2, "description": "LG elimination."},
        ],
        "generic_notes": "Ortho-oriented nucleophilic aromatic substitution.",
    },
    _normalize_label("Base catalyzed ester hydrolysis"): {
        "current_state_generic": ["R-C(=O)-OR'", "OH-"],
        "resulting_state_generic": ["R-C(=O)O-", "R'OH"],
        "reaction_generic": "R-C(=O)-OR'.OH->>tetrahedral>>R-C(=O)O-.R'OH |dbe:C-OR':-2;C-OH:+2|",
        "electron_pushes_generic": [
            {"start_atom": "OH:lp", "end_atom": "C=O", "electrons": 2, "description": "Hydroxide attack on carbonyl."},
            {"start_atom": "C-OR'", "end_atom": "OR'", "electrons": 2, "description": "Alkoxide leaving group departs."},
        ],
        "generic_notes": "Saponification/acyl substitution template.",
    },
    _normalize_label("Ester reduction"): {
        "current_state_generic": ["R-C(=O)-OR'", "H-"],
        "resulting_state_generic": ["R-CH2OH", "R'OH"],
        "reaction_generic": "R-C(=O)-OR'.H->>R-CH(OH)-OR'>>R-CH2OH.R'OH |dbe:C-O:+2;C-OR':-2|",
        "electron_pushes_generic": [
            {"start_atom": "H-", "end_atom": "C=O", "electrons": 2, "description": "Hydride transfer to carbonyl carbon."},
            {"start_atom": "C-OR'", "end_atom": "OR'", "electrons": 2, "description": "Collapse and OR' departure."},
        ],
        "generic_notes": "Hydride reduction of esters to alcohols.",
    },
    _normalize_label("Carbonyl reduction"): {
        "current_state_generic": ["R2C=O", "H-"],
        "resulting_state_generic": ["R2CHOH"],
        "reaction_generic": "R2C=O.H->>R2CHO- >> R2CHOH |dbe:C-O:+2;O-H:+2|",
        "electron_pushes_generic": [
            {"start_atom": "H-", "end_atom": "C=O", "electrons": 2, "description": "Hydride attack."},
        ],
        "generic_notes": "General carbonyl hydride reduction.",
    },
    _normalize_label("Alcohol attack to carbonyl or sulfonyl"): {
        "current_state_generic": ["E(=O)-LG", "ROH"],
        "resulting_state_generic": ["E(=O)-OR", "LGH"],
        "reaction_generic": "E(=O)-LG.ROH>>tetrahedral>>E(=O)-OR.LGH |dbe:E-LG:-2;E-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "ROH:lp", "end_atom": "E=O center", "electrons": 2, "description": "Alcohol attack on electrophile."},
        ],
        "generic_notes": "Covers acyl/sulfonyl substitution by alcohols.",
    },
    _normalize_label("Nucleophilic attack to (thio)carbonyl"): {
        "current_state_generic": ["R2C=O or R2C=S", "Nu-"],
        "resulting_state_generic": ["R2C( Nu)(O-) or R2C(Nu)(S-)"],
        "reaction_generic": "R2C=E.Nu->>R2C(Nu)E- |dbe:C-E:+2;Nu-Nu:-2|",
        "electron_pushes_generic": [
            {"start_atom": "Nu:lp", "end_atom": "C=E", "electrons": 2, "description": "Nucleophile adds to heterocarbonyl carbon."},
        ],
        "generic_notes": "General tetrahedral intermediate-forming addition.",
    },
    _normalize_label("Nucleophilic attack to iso(thio)cyanate"): {
        "current_state_generic": ["R-N=C=E", "Nu-"],
        "resulting_state_generic": ["R-N(-)-C(=E)-Nu"],
        "reaction_generic": "R-N=C=E.Nu->>R-N(-)-C(=E)-Nu |dbe:C-Nu:+2;Nu-Nu:-2|",
        "electron_pushes_generic": [
            {"start_atom": "Nu:lp", "end_atom": "N=C=E carbon", "electrons": 2, "description": "Nu attack on cumulene carbon."},
        ],
        "generic_notes": "Isocyanate/isothiocyanate addition.",
    },
    _normalize_label("DCC condensation"): {
        "current_state_generic": ["R-CO2H", "R'-NH2", "DCC"],
        "resulting_state_generic": ["R-CONHR'", "DCU"],
        "reaction_generic": "R-CO2H.R'-NH2.DCC>>activated_O-acylurea>>R-CONHR'.DCU |dbe:C-O:-2;C-N:+2|",
        "electron_pushes_generic": [
            {"start_atom": "R'-NH2:lp", "end_atom": "acyl C", "electrons": 2, "description": "Amine attack on activated acyl intermediate."},
        ],
        "generic_notes": "Carbodiimide-mediated coupling template.",
    },
    _normalize_label("Reductive amination"): {
        "current_state_generic": ["R2C=O", "R'NH2", "[H]"],
        "resulting_state_generic": ["R2CH-NHR'"],
        "reaction_generic": "R2C=O.R'NH2>>imine/iminium>>R2CH-NHR' |dbe:C-N:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "N:lp", "end_atom": "C=O", "electrons": 2, "description": "Imine formation stage."},
            {"start_atom": "H-", "end_atom": "C=N+", "electrons": 2, "description": "Reduction stage."},
        ],
        "generic_notes": "Condensation + selective reduction.",
    },
    _normalize_label("Boc deprotection"): {
        "current_state_generic": ["R-NH-CO2-tBu", "H+"],
        "resulting_state_generic": ["R-NH2", "CO2", "isobutene"],
        "reaction_generic": "R-NH-CO2-tBu.H+>>R-NH2.CO2.isobutene |dbe:N-C:-2|",
        "electron_pushes_generic": [
            {"start_atom": "carbamate O", "end_atom": "tert-butyl C", "electrons": 2, "description": "Acid-promoted tert-butyl cleavage."},
        ],
        "generic_notes": "Acidic Boc cleavage.",
    },
    _normalize_label("Cbz deprotection"): {
        "current_state_generic": ["R-NH-CO2CH2Ph", "[H]"],
        "resulting_state_generic": ["R-NH2", "CO2", "toluene/benzyl byproduct"],
        "reaction_generic": "R-NH-Cbz.[H]>>R-NH2.CO2.benzyl_fragment |dbe:N-C:-2|",
        "electron_pushes_generic": [
            {"start_atom": "H2/Pd", "end_atom": "benzyl C", "electrons": 2, "description": "Hydrogenolysis of benzylic C-O bond."},
        ],
        "generic_notes": "Hydrogenolytic Cbz removal.",
    },
    _normalize_label("Fmoc deprotection"): {
        "current_state_generic": ["R-NH-CO2Fmoc", "base"],
        "resulting_state_generic": ["R-NH2", "dibenzofulvene adduct"],
        "reaction_generic": "R-NH-Fmoc.base>>R-NH2.dibenzofulvene_adduct |dbe:N-C:-2|",
        "electron_pushes_generic": [
            {"start_atom": "base", "end_atom": "Fmoc benzylic H", "electrons": 2, "description": "E1cb-like elimination."},
        ],
        "generic_notes": "Base-labile Fmoc cleavage.",
    },
    _normalize_label("Imine formation"): {
        "current_state_generic": ["R2C=O", "R'NH2"],
        "resulting_state_generic": ["R2C=NR'", "H2O"],
        "reaction_generic": "R2C=O.R'NH2>>carbinolamine>>R2C=NR'.H2O |dbe:C-N:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "N:lp", "end_atom": "carbonyl C", "electrons": 2, "description": "Nucleophilic addition."},
        ],
        "generic_notes": "Carbonyl condensation to imine.",
    },
    _normalize_label("Imine reduction"): {
        "current_state_generic": ["R2C=NR'", "H-"],
        "resulting_state_generic": ["R2CH-NHR'"],
        "reaction_generic": "R2C=NR'.H->>R2CH-NR'>>R2CH-NHR' |dbe:C-N:+2|",
        "electron_pushes_generic": [
            {"start_atom": "H-", "end_atom": "C=N", "electrons": 2, "description": "Hydride addition to imine carbon."},
        ],
        "generic_notes": "Reduction of imine/iminium to amine.",
    },
    _normalize_label("O-demethylation"): {
        "current_state_generic": ["Ar-O-CH3", "Nu"],
        "resulting_state_generic": ["Ar-OH", "CH3-Nu"],
        "reaction_generic": "Ar-O-CH3.Nu>>Ar-O-.CH3-Nu>>Ar-OH |dbe:O-CH3:-2|",
        "electron_pushes_generic": [
            {"start_atom": "Nu:lp", "end_atom": "CH3", "electrons": 2, "description": "SN2 cleavage at methyl."},
        ],
        "generic_notes": "Demethylation of aryl methyl ethers.",
    },
    _normalize_label("Mitsunobu reaction"): {
        "current_state_generic": ["ROH", "Nu-H", "PPh3", "DIAD/DEAD"],
        "resulting_state_generic": ["R-Nu", "byproducts"],
        "reaction_generic": "ROH.NuH.PPh3.DIAD>>activated_oxyphosphonium>>R-Nu |dbe:C-O:-2;C-Nu:+2|",
        "electron_pushes_generic": [
            {"start_atom": "Nu-", "end_atom": "C*", "electrons": 2, "description": "Inverting substitution at activated alcohol center."},
        ],
        "generic_notes": "Mitsunobu inversion/substitution template.",
    },
    _normalize_label("Swern oxidation"): {
        "current_state_generic": ["R-CH2OH", "DMSO", "(COCl)2", "base"],
        "resulting_state_generic": ["R-CHO"],
        "reaction_generic": "R-CH2OH.DMSO.(COCl)2.base>>alkoxysulfonium>>R-CHO |dbe:C-O:-2;C=O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "base", "end_atom": "alpha-H", "electrons": 2, "description": "Elimination to carbonyl."},
        ],
        "generic_notes": "Activated DMSO oxidation.",
    },
    _normalize_label("Jones oxidation"): {
        "current_state_generic": ["R-CH2OH", "Cr(VI)"],
        "resulting_state_generic": ["R-CO2H"],
        "reaction_generic": "R-CH2OH.Cr(VI)>>chromate_ester>>R-CO2H |dbe:C-O:-2;C=O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "O", "end_atom": "Cr", "electrons": 2, "description": "Chromate ester formation/elimination sequence."},
        ],
        "generic_notes": "Strong oxidation of primary alcohols.",
    },
    _normalize_label("Aldol addition"): {
        "current_state_generic": ["enolate", "R-CHO"],
        "resulting_state_generic": ["beta-hydroxy carbonyl"],
        "reaction_generic": "enolate.R-CHO>>beta-hydroxy carbonyl |dbe:C-C:+2|",
        "electron_pushes_generic": [
            {"start_atom": "enolate C:lp", "end_atom": "aldehyde C", "electrons": 2, "description": "Carbon-carbon bond formation."},
        ],
        "generic_notes": "Aldol C-C bond formation without dehydration.",
    },
    _normalize_label("Aldol condensation"): {
        "current_state_generic": ["enolate", "R-CHO"],
        "resulting_state_generic": ["alpha,beta-unsaturated carbonyl"],
        "reaction_generic": "enolate.R-CHO>>beta-hydroxy carbonyl>>enone + H2O |dbe:C-C:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "enolate C:lp", "end_atom": "aldehyde C", "electrons": 2, "description": "Aldol addition step."},
            {"start_atom": "beta-O", "end_atom": "alpha-beta", "electrons": 2, "description": "Dehydration to enone."},
        ],
        "generic_notes": "Aldol plus elimination sequence.",
    },
    _normalize_label("Grignard reaction"): {
        "current_state_generic": ["R-MgX", "R'2C=O"],
        "resulting_state_generic": ["R'2C(OH)-R"],
        "reaction_generic": "R-MgX.R'2C=O>>alkoxide>>alcohol |dbe:C-C:+2|",
        "electron_pushes_generic": [
            {"start_atom": "R- (from RMgX)", "end_atom": "carbonyl C", "electrons": 2, "description": "Carbanion equivalent addition."},
        ],
        "generic_notes": "Organomagnesium nucleophilic addition.",
    },
    _normalize_label("Wittig reaction"): {
        "current_state_generic": ["ylide", "R2C=O"],
        "resulting_state_generic": ["alkene", "Ph3P=O"],
        "reaction_generic": "Ph3P=CHR.R2C=O>>oxaphosphetane>>alkene.Ph3P=O |dbe:C-C:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "ylide C:lp", "end_atom": "carbonyl C", "electrons": 2, "description": "Betain/oxaphosphetane formation."},
        ],
        "generic_notes": "Phosphorus ylide olefination.",
    },
    _normalize_label("Wittig reaction with ylide"): {
        "current_state_generic": ["stabilized ylide", "R2C=O"],
        "resulting_state_generic": ["alkene", "Ph3P=O"],
        "reaction_generic": "ylide.R2C=O>>oxaphosphetane>>alkene.Ph3P=O |dbe:C-C:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "ylide C:lp", "end_atom": "carbonyl C", "electrons": 2, "description": "Ylide carbon addition."},
        ],
        "generic_notes": "Explicit ylide-mediated olefination template.",
    },
    _normalize_label("Horner Wadsworth Emmons reaction"): {
        "current_state_generic": ["phosphonate carbanion", "R-CHO"],
        "resulting_state_generic": ["alkene", "phosphate byproduct"],
        "reaction_generic": "(RO)2P(O)-CHR-.R-CHO>>betaine>>alkene |dbe:C-C:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "alpha-C:lp", "end_atom": "carbonyl C", "electrons": 2, "description": "Phosphonate carbanion attack."},
        ],
        "generic_notes": "HWE olefination template.",
    },
    _normalize_label("Mannich reaction"): {
        "current_state_generic": ["enol/enolate donor", "iminium electrophile"],
        "resulting_state_generic": ["beta-amino carbonyl"],
        "reaction_generic": "enol.iminium>>beta-amino carbonyl |dbe:C-C:+2|",
        "electron_pushes_generic": [
            {"start_atom": "enol C:pi", "end_atom": "iminium C", "electrons": 2, "description": "C-C bond formation at iminium carbon."},
        ],
        "generic_notes": "Mannich C-C bond forming addition.",
    },
    _normalize_label("Michael addition"): {
        "current_state_generic": ["Nu- donor", "alpha,beta-unsaturated acceptor"],
        "resulting_state_generic": ["1,4-addition product"],
        "reaction_generic": "Nu-.C=C-C(=O)R>>1,4-addition product |dbe:C-Nu:+2|",
        "electron_pushes_generic": [
            {"start_atom": "Nu:lp", "end_atom": "beta-C", "electrons": 2, "description": "Conjugate addition."},
        ],
        "generic_notes": "Conjugate (1,4-) addition template.",
    },
    _normalize_label("Staudinger reaction"): {
        "current_state_generic": ["R3P", "R'-N3"],
        "resulting_state_generic": ["R'-NH2", "R3P=O", "N2"],
        "reaction_generic": "R3P.R'-N3.H2O>>iminophosphorane>>R'-NH2.R3P=O.N2 |dbe:P-N:-2;P=O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "P:lp", "end_atom": "azide terminal N", "electrons": 2, "description": "Phosphine attack on azide."},
        ],
        "generic_notes": "Azide reduction via phosphine.",
    },
    _normalize_label("Friedel Crafts acylation"): {
        "current_state_generic": ["Ar-H", "R-CO-LG", "Lewis acid"],
        "resulting_state_generic": ["Ar-CO-R", "H-LG"],
        "reaction_generic": "Ar-H.RCOX.AlX3>>sigma_complex>>Ar-CO-R |dbe:Ar-C:+2|",
        "electron_pushes_generic": [
            {"start_atom": "Ar pi", "end_atom": "acylium C", "electrons": 2, "description": "EAS C-C bond formation."},
        ],
        "generic_notes": "Electrophilic aromatic acylation.",
    },
    _normalize_label("Acetal formation"): {
        "current_state_generic": ["R2C=O", "2 ROH", "H+"],
        "resulting_state_generic": ["R2C(OR)2", "H2O"],
        "reaction_generic": "R2C=O.2ROH.H+>>hemiacetal>>acetal + H2O |dbe:C-O:+2;C=O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "ROH:lp", "end_atom": "carbonyl C", "electrons": 2, "description": "Alcohol attack under acid catalysis."},
        ],
        "generic_notes": "Acid-catalyzed acetalization.",
    },
    _normalize_label("(hemi)acetal (aminal) hydrolysis"): {
        "current_state_generic": ["acetal/aminal", "H2O", "H+"],
        "resulting_state_generic": ["carbonyl", "ROH/RNH2"],
        "reaction_generic": "acetal.H2O.H+>>hemiacetal>>carbonyl + alcohol/amine |dbe:C-O:-2;C=O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "H2O:lp", "end_atom": "acetal C", "electrons": 2, "description": "Hydrolytic cleavage sequence."},
        ],
        "generic_notes": "Reverse of acetal/aminal formation.",
    },
    _normalize_label("Nitrile reduction"): {
        "current_state_generic": ["R-C#N", "4[H]"],
        "resulting_state_generic": ["R-CH2NH2"],
        "reaction_generic": "R-C#N.4[H]>>R-CH=NH>>R-CH2NH2 |dbe:C-N:+2|",
        "electron_pushes_generic": [
            {"start_atom": "H-", "end_atom": "nitrile C", "electrons": 2, "description": "Hydride/proton sequence reduces nitrile."},
        ],
        "generic_notes": "Stepwise nitrile reduction to primary amine.",
    },
    _normalize_label("Amide reduction"): {
        "current_state_generic": ["R-C(=O)-NR'R''", "4[H]"],
        "resulting_state_generic": ["R-CH2-NR'R''"],
        "reaction_generic": "R-C(=O)-NR'R''.4[H]>>R-CH2-NR'R'' |dbe:C-O:-2;C-H:+2|",
        "electron_pushes_generic": [
            {"start_atom": "H-", "end_atom": "amide carbonyl C", "electrons": 2, "description": "Hydride reduction of amide carbonyl."},
        ],
        "generic_notes": "Strong-hydride reduction of amides to amines.",
    },
    _normalize_label("Hantzsch thiazole synthesis"): {
        "current_state_generic": ["alpha-haloketone", "thioamide"],
        "resulting_state_generic": ["thiazole", "HX", "H2O"],
        "reaction_generic": "alpha-haloketone.thioamide>>cyclization>>thiazole |dbe:C-S:+2;C-N:+2|",
        "electron_pushes_generic": [
            {"start_atom": "thioamide S:lp", "end_atom": "alpha-carbon", "electrons": 2, "description": "Nucleophilic substitution then cyclization."},
        ],
        "generic_notes": "Hantzsch thiazole ring-forming condensation.",
    },
    _normalize_label("Sulfide oxidation"): {
        "current_state_generic": ["R-S-R'", "[O]"],
        "resulting_state_generic": ["R-S(=O)-R'"],
        "reaction_generic": "R-S-R'.[O]>>R-S(=O)-R' |dbe:S-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "sulfide S:lp", "end_atom": "O electrophile", "electrons": 2, "description": "O-transfer to sulfur."},
        ],
        "generic_notes": "Sulfide to sulfoxide oxidation.",
    },
    _normalize_label("Primary amide dehydration"): {
        "current_state_generic": ["R-C(=O)-NH2", "dehydrating agent"],
        "resulting_state_generic": ["R-C#N"],
        "reaction_generic": "R-C(=O)-NH2>>R-C#N + H2O |dbe:C-N:+2;C-O:-2|",
        "electron_pushes_generic": [
            {"start_atom": "amide N lone pair", "end_atom": "carbonyl C", "electrons": 2, "description": "Elimination-driven nitrile formation."},
        ],
        "generic_notes": "Dehydrative conversion of primary amides to nitriles.",
    },
    _normalize_label("Wolff Kishner reduction"): {
        "current_state_generic": ["R2C=O", "NH2NH2", "base", "heat"],
        "resulting_state_generic": ["R2CH2", "N2"],
        "reaction_generic": "R2C=O.NH2NH2.base>>hydrazone>>R2CH2 + N2 |dbe:C-O:-2;C-H:+2|",
        "electron_pushes_generic": [
            {"start_atom": "hydrazone N", "end_atom": "C", "electrons": 2, "description": "Nitrogen extrusion with carbon reduction."},
        ],
        "generic_notes": "Carbonyl deoxygenation via hydrazone pathway.",
    },
    _normalize_label("Amine oxidation"): {
        "current_state_generic": ["R3N", "[O]"],
        "resulting_state_generic": ["R3N->O or iminium"],
        "reaction_generic": "R3N.[O]>>R3N->O |dbe:N-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "amine N:lp", "end_atom": "oxidant O", "electrons": 2, "description": "N-oxidation pathway."},
        ],
        "generic_notes": "Tertiary amine oxidation to N-oxide/iminium manifold.",
    },
    _normalize_label("Sulfide oxidation by peroxide"): {
        "current_state_generic": ["R-S-R'", "H2O2"],
        "resulting_state_generic": ["R-S(=O)-R' or R-S(=O)2-R'"],
        "reaction_generic": "R-S-R'.H2O2>>R-S(=O)-R'>>R-S(=O)2-R' |dbe:S-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "sulfide S:lp", "end_atom": "peroxide O", "electrons": 2, "description": "Peroxide oxygen transfer to sulfur."},
        ],
        "generic_notes": "Peroxide-driven oxidation to sulfoxide/sulfone.",
    },
    _normalize_label("Alkene epoxidation"): {
        "current_state_generic": ["RCH=CHR'", "peracid"],
        "resulting_state_generic": ["epoxide"],
        "reaction_generic": "RCH=CHR'.RCO3H>>epoxide + RCO2H |dbe:C-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "alkene pi", "end_atom": "electrophilic O", "electrons": 2, "description": "Concerted oxygen transfer."},
        ],
        "generic_notes": "Peracid epoxidation of alkenes.",
    },
    _normalize_label("Alkynyl attack to carbonyl"): {
        "current_state_generic": ["RC#C-", "R'2C=O"],
        "resulting_state_generic": ["propargyl alcohol"],
        "reaction_generic": "RC#C-.R'2C=O>>alkoxide>>propargyl alcohol |dbe:C-C:+2|",
        "electron_pushes_generic": [
            {"start_atom": "alkynyl C:lp", "end_atom": "carbonyl C", "electrons": 2, "description": "Alkynyl nucleophilic addition."},
        ],
        "generic_notes": "Acetylide addition to aldehydes/ketones.",
    },
    _normalize_label("Weinreb ketone synthesis"): {
        "current_state_generic": ["R-C(=O)-N(OMe)Me", "R'-M"],
        "resulting_state_generic": ["R-C(=O)-R'"],
        "reaction_generic": "Weinreb_amide.R'-M>>chelated_tetrahedral>>ketone |dbe:C-C:+2;C-N:-2|",
        "electron_pushes_generic": [
            {"start_atom": "R'- (organometallic)", "end_atom": "acyl C", "electrons": 2, "description": "Controlled single-addition acyl substitution."},
        ],
        "generic_notes": "Weinreb amide enables selective ketone formation.",
    },
    _normalize_label("Methyl ester synthesis"): {
        "current_state_generic": ["R-CO2H", "CH2N2"],
        "resulting_state_generic": ["R-CO2CH3", "N2"],
        "reaction_generic": "R-CO2H.CH2N2>>R-CO2CH3 + N2 |dbe:O-CH3:+2|",
        "electron_pushes_generic": [
            {"start_atom": "carboxylate O:lp", "end_atom": "CH2 (diazomethane)", "electrons": 2, "description": "O-alkylation by diazomethane."},
        ],
        "generic_notes": "Diazomethane methyl esterification template.",
    },
    _normalize_label("Isothiocyanate synthesis"): {
        "current_state_generic": ["R-NH2", "thiocarbonyl transfer reagent"],
        "resulting_state_generic": ["R-N=C=S"],
        "reaction_generic": "R-NH2.CS_transfer>>R-N=C=S |dbe:N-C:+2;C=S:+2|",
        "electron_pushes_generic": [
            {"start_atom": "amine N:lp", "end_atom": "thiocarbonyl C", "electrons": 2, "description": "Installation of N=C=S functionality."},
        ],
        "generic_notes": "Thiocarbamoyl intermediate to isothiocyanate elimination.",
    },
    _normalize_label("Lactone reduction"): {
        "current_state_generic": ["lactone", "H-"],
        "resulting_state_generic": ["diol/open-chain alcohol"],
        "reaction_generic": "lactone.H->>opened_hydroxy_ester>>diol |dbe:C-O:+2;C-O(ring):-2|",
        "electron_pushes_generic": [
            {"start_atom": "H-", "end_atom": "lactone carbonyl C", "electrons": 2, "description": "Hydride opens/reduces lactone."},
        ],
        "generic_notes": "Lactone reduction to diol-type products.",
    },
    _normalize_label("Intramolecular lactonization"): {
        "current_state_generic": ["HO-(CH2)n-CO2H"],
        "resulting_state_generic": ["lactone", "H2O"],
        "reaction_generic": "HO-(CH2)n-CO2H>>cyclic_tetrahedral>>lactone + H2O |dbe:C-O:+2|",
        "electron_pushes_generic": [
            {"start_atom": "intramolecular OH:lp", "end_atom": "acyl C", "electrons": 2, "description": "Ring-closing acyl substitution."},
        ],
        "generic_notes": "Cyclization of hydroxy acids to lactones.",
    },
    _normalize_label("SN1 reaction with tosylate"): {
        "current_state_generic": ["R-OTs", "Nu"],
        "resulting_state_generic": ["R-Nu", "TsO-"],
        "reaction_generic": "R-OTs.Nu>>R+ + TsO- >> R-Nu |dbe:C-OTs:-2;C-Nu:+2|",
        "electron_pushes_generic": [
            {"start_atom": "C-OTs", "end_atom": "OTs", "electrons": 2, "description": "Tosylate departure to carbocation."},
            {"start_atom": "Nu:lp", "end_atom": "C+", "electrons": 2, "description": "Nucleophile captures carbocation."},
        ],
        "generic_notes": "SN1 substitution with tosylate leaving group.",
    },
}


def _generic_template_default(label: str) -> dict[str, Any]:
    slug = _slugify(label).upper()
    return {
        "current_state_generic": [f"{slug}_SUBSTRATE", "GENERIC_REAGENT"],
        "resulting_state_generic": [f"{slug}_PRODUCT"],
        "reaction_generic": f"{slug}_SUBSTRATE.GENERIC_REAGENT>>{slug}_PRODUCT |dbe:X-Y:+2;X-X:-2|",
        "electron_pushes_generic": [
            {
                "start_atom": "X",
                "end_atom": "Y",
                "electrons": 2,
                "description": f"Generic two-electron transformation for {label}.",
            }
        ],
        "generic_notes": f"Auto-generated generic template for {label}.",
    }


def build_generic_template_library(taxonomy: list[MechanismType]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in taxonomy:
        tmpl = GENERIC_TEMPLATE_OVERRIDES.get(item.normalized)
        if tmpl is None:
            tmpl = _generic_template_default(item.label_exact)
        out[item.label_exact] = {
            "type_id": item.type_id,
            "label_exact": item.label_exact,
            **tmpl,
        }
    return out


SUITABLE_STEP_COUNT_OVERRIDES: dict[str, int] = {
    "DCC condensation": 4,
    "Nucleophilic attack to (thio)carbonyl": 2,
    "SN2 reaction": 1,
    "SN2 reaction with alcohol(thiol)": 1,
    "Boc deprotection": 3,
    "Reductive amination": 4,
    "Alcohol attack to carbonyl or sulfonyl": 2,
    "Nucleophilic attack to iso(thio)cyanate": 2,
    "Carbonyl reduction": 2,
    "SNAr reaction (para)": 2,
    "Ester reduction": 3,
    "Imine formation": 2,
    "O-demethylation": 2,
    "Mitsunobu reaction": 3,
    "Swern oxidation": 3,
    "Aldol condensation": 3,
    "SN1 reaction": 2,
    "Cbz deprotection": 3,
    "Carboxylic acid derivative hydrolysis or formation": 3,
    "Jones oxidation": 2,
    "Nitrile reduction": 3,
    "Amide reduction": 3,
    "Base catalyzed ester hydrolysis": 3,
    "Grignard reaction": 2,
    "SNAr reaction with alcohol (ortho)": 2,
    "Wittig reaction": 3,
    "Staudinger reaction": 3,
    "Hantzsch thiazole synthesis": 3,
    "(hemi)acetal (aminal) hydrolysis": 3,
    "Sulfide oxidation": 1,
    "Friedel Crafts acylation": 3,
    "Horner Wadsworth Emmons reaction": 3,
    "Primary amide dehydration": 2,
    "Wolff Kishner reduction": 4,
    "Acetal formation": 3,
    "Amine oxidation": 1,
    "Imine reduction": 2,
    "Wittig reaction with ylide": 3,
    "Sulfide oxidation by peroxide": 2,
    "Mannich reaction": 3,
    "Alkene epoxidation": 1,
    "Aldol addition": 2,
    "Fmoc deprotection": 2,
    "Alkynyl attack to carbonyl": 2,
    "Weinreb ketone synthesis": 2,
    "Methyl ester synthesis": 1,
    "Isothiocyanate synthesis": 2,
    "Michael addition": 2,
    "Lactone reduction": 3,
    "Intramolecular lactonization": 2,
    "SN1 reaction with tosylate": 2,
}


def _split_generic_reaction_to_steps(reaction_generic: str) -> list[str]:
    text = str(reaction_generic or "").strip()
    if not text:
        return []

    text = text.replace("->>", ">>")
    dbe_suffix = ""
    core = text
    if "|dbe:" in text and text.endswith("|"):
        prefix, suffix = text.split("|dbe:", 1)
        core = prefix.strip()
        dbe_suffix = f"|dbe:{suffix.strip()}"

    segments = [seg.strip() for seg in core.split(">>") if seg.strip()]
    if len(segments) < 2:
        return [text]

    transitions: list[str] = []
    for idx in range(len(segments) - 1):
        step_rxn = f"{segments[idx]}>>{segments[idx + 1]}"
        if dbe_suffix:
            step_rxn = f"{step_rxn} {dbe_suffix}"
        transitions.append(step_rxn)
    return transitions


def _build_reaction_type_templates_payload(
    taxonomy: list[MechanismType],
    template_library: dict[str, dict[str, Any]],
    *,
    source_path: Path,
    observed_step_max_by_label: dict[str, int],
    example_mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    templates_out: list[dict[str, Any]] = []
    for item in taxonomy:
        template = dict(template_library[item.label_exact])
        override_steps = int(SUITABLE_STEP_COUNT_OVERRIDES.get(item.label_exact, 2))
        observed_steps = int(observed_step_max_by_label.get(item.label_exact, 0))
        target_steps = max(override_steps, observed_steps)
        base_steps = _split_generic_reaction_to_steps(str(template.get("reaction_generic") or ""))
        if not base_steps:
            base_steps = [str(template.get("reaction_generic") or "")]

        if len(base_steps) >= target_steps:
            chosen = base_steps[:target_steps]
        else:
            chosen = list(base_steps)
            while len(chosen) < target_steps:
                chosen.append(base_steps[-1])

        step_objects = []
        for step_idx, step_reaction in enumerate(chosen, start=1):
            step_objects.append(
                {
                    "step_index": step_idx,
                    "reaction_generic": step_reaction,
                    "electron_pushes_generic": list(template.get("electron_pushes_generic") or []),
                    "note": f"Template stage {step_idx}/{target_steps} for {item.label_exact}.",
                }
            )

        templates_out.append(
            {
                "type_id": item.type_id,
                "label_exact": item.label_exact,
                "slug": item.slug,
                "canonical_group": canonical_group_for_label(item.label_exact),
                "suitable_step_count": target_steps,
                "current_state_generic": list(template.get("current_state_generic") or []),
                "resulting_state_generic": list(template.get("resulting_state_generic") or []),
                "generic_notes": str(template.get("generic_notes") or ""),
                "generic_mechanism_steps": step_objects,
            }
        )

    return {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": str(source_path.relative_to(PROJECT_ROOT)),
            "count": len(templates_out),
        },
        "templates": templates_out,
        "example_mappings": example_mappings,
    }


def _fallback_rule_label(case: dict[str, Any]) -> tuple[str, float, str]:
    reactants = ".".join(case.get("starting_materials", [])).lower()
    products = ".".join(case.get("products", [])).lower()
    min_steps = int(((case.get("known_mechanism") or {}).get("min_steps") or case.get("n_mechanistic_steps") or 0))

    if "n+]=[n-" in reactants:
        return "Methyl ester synthesis", 0.68, "Heuristic: diazomethane-like methylation reagent present."
    if "[i-]" in reactants and ("cl" in reactants or "br" in reactants):
        return "SN2 reaction", 0.72, "Heuristic: halide exchange signature."
    if "[oh-]" in reactants and "c(=o)o" in reactants:
        return "Base catalyzed ester hydrolysis", 0.74, "Heuristic: hydroxide + ester-like motif."
    if "o=[n+]([o-])o" in reactants and "o=s(o)(o)=o" in reactants:
        return "Friedel Crafts acylation", 0.46, "Heuristic fallback for strongly acidic aromatic electrophilic substitution conditions."
    if "[h][al-]([h])([h])[h]" in reactants:
        return "Ester reduction", 0.78, "Heuristic: hydride reducing reagent detected."
    if "cc#n" in reactants and ("n=" in products or "n(" in products):
        return "Imine formation", 0.55, "Heuristic: nitrile/reactive nitrogen to imine-like product."
    if "cl" in reactants and "n" in products and "c1=cc=cc=c1" in reactants:
        return "SNAr reaction (para)", 0.55, "Heuristic: aryl halide to aryl amine conversion."

    if min_steps <= 2:
        return "SN2 reaction", 0.35, "Fallback by low step count."
    if min_steps == 3:
        return "SN1 reaction", 0.32, "Fallback by moderate step count."
    if min_steps == 4:
        return "Alcohol attack to carbonyl or sulfonyl", 0.3, "Fallback by step-band class."
    if min_steps == 5:
        return "Carboxylic acid derivative hydrolysis or formation", 0.3, "Fallback by step-band class."
    if min_steps >= 6:
        return "Mannich reaction", 0.25, "Fallback for complex multi-step transformations."
    return "Nucleophilic attack to (thio)carbonyl", 0.2, "Default fallback label."


def assign_mechanism_type(
    case: dict[str, Any],
    *,
    taxonomy_by_norm: dict[str, MechanismType],
    taxonomy_by_id: dict[str, MechanismType],
    preferred_eval_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rid = str(case.get("id"))
    preferred = preferred_eval_map.get(rid) if isinstance(preferred_eval_map, dict) else None
    if isinstance(preferred, dict):
        preferred_label = str(preferred.get("mechanism_type_label") or "").strip()
        preferred_type_id = str(preferred.get("mechanism_type_id") or "").strip()
        preferred_norm = _normalize_label(preferred_label)

        selected: MechanismType | None = None
        if preferred_norm in taxonomy_by_norm:
            selected = taxonomy_by_norm[preferred_norm]
        elif preferred_type_id and preferred_type_id in taxonomy_by_id:
            selected = taxonomy_by_id[preferred_type_id]

        if selected is not None:
            confidence_raw = preferred.get("confidence")
            confidence = 0.95
            if isinstance(confidence_raw, (int, float)):
                confidence = round(max(0.0, min(1.0, float(confidence_raw))), 3)
            rationale = str(preferred.get("rationale") or "").strip()
            if not rationale:
                rationale = "Mapped from existing eval_mechanism_map preference."
            return {
                "type_id": selected.type_id,
                "label_exact": selected.label_exact,
                "mechanism_type_label": preferred_label or selected.label_exact,
                "canonical_group": canonical_group_for_label(selected.label_exact),
                "confidence": confidence,
                "rationale": rationale,
                "assignment_source": "eval_mechanism_map",
            }

    if rid in CURATED_LABEL_OVERRIDES:
        label, confidence, rationale = CURATED_LABEL_OVERRIDES[rid]
    else:
        label, confidence, rationale = _fallback_rule_label(case)

    norm = _normalize_label(label)
    if norm not in taxonomy_by_norm:
        # Hard fallback to the first taxonomy item if label mismatches.
        first = next(iter(taxonomy_by_norm.values()))
        label = first.label_exact
        confidence = 0.1
        rationale = f"Label fallback: '{label}' selected because heuristic label was not in taxonomy."
        norm = _normalize_label(label)

    mtype = taxonomy_by_norm[norm]
    return {
        "type_id": mtype.type_id,
        "label_exact": mtype.label_exact,
        "mechanism_type_label": mtype.label_exact,
        "canonical_group": canonical_group_for_label(mtype.label_exact),
        "confidence": round(float(confidence), 3),
        "rationale": rationale,
        "assignment_source": "rxn_map_fallback",
    }


def canonical_group_for_label(label: str) -> str:
    text = _normalize_label(label)
    if "sn1" in text or "sn2" in text or "snar" in text:
        return "substitution"
    if "reduction" in text:
        return "reduction"
    if "oxidation" in text:
        return "oxidation"
    if "deprotection" in text:
        return "deprotection"
    if "formation" in text or "synthesis" in text or "condensation" in text:
        return "formation_condensation"
    if "attack" in text or "addition" in text:
        return "addition_attack"
    if "grignard" in text or "wittig" in text or "horner" in text:
        return "carbon_carbon_bond_formation"
    if "hydrolysis" in text:
        return "hydrolysis"
    return "other"


def _mapped_smiles(smiles: str, next_map: int) -> tuple[str, int]:
    token = str(smiles or "").strip()
    if not token or Chem is None:
        return token, next_map
    mol = Chem.MolFromSmiles(token)
    if mol is None:
        return token, next_map

    max_existing = next_map - 1
    for atom in mol.GetAtoms():
        amap = int(atom.GetAtomMapNum() or 0)
        if amap > 0:
            max_existing = max(max_existing, amap)

    for atom in mol.GetAtoms():
        if int(atom.GetAtomMapNum() or 0) <= 0:
            max_existing += 1
            atom.SetAtomMapNum(max_existing)

    mapped = Chem.MolToSmiles(mol, canonical=True)
    return mapped, max_existing + 1


def _map_state(state: list[str], map_seed: int) -> tuple[list[str], int]:
    mapped: list[str] = []
    cur = map_seed
    for smi in state:
        ms, cur = _mapped_smiles(str(smi), cur)
        mapped.append(ms)
    return mapped, cur


def _extract_map_ids(mapped_state: list[str]) -> list[int]:
    ids = set()
    for smi in mapped_state:
        for m in re.findall(r":(\d+)\]", str(smi)):
            ids.add(int(m))
    return sorted(ids)


def _build_concrete_push(mapped_current: list[str], mapped_result: list[str], step_index: int) -> dict[str, Any]:
    ids = sorted(set(_extract_map_ids(mapped_current) + _extract_map_ids(mapped_result)))
    if len(ids) >= 2:
        src, snk = ids[0], ids[1]
    elif len(ids) == 1:
        src, snk = ids[0], ids[0] + 1
    else:
        src, snk = 1, 2
    return {
        "start_atom": str(src),
        "end_atom": str(snk),
        "electrons": 2,
        "description": f"Step {step_index}: two-electron transfer from source to sink.",
    }


def _build_dbe_block(push: dict[str, Any]) -> str:
    src = int(push.get("start_atom") or 1)
    snk = int(push.get("end_atom") or (src + 1))
    if src == snk:
        snk = src + 1
    return f"{src}-{snk}:+2;{src}-{src}:-2"


def _generic_step(
    mechanism_label: str,
    step_index: int,
    template_library: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    template = template_library.get(mechanism_label) or _generic_template_default(mechanism_label)
    return {
        "mechanism_label": mechanism_label,
        "template_step_index": step_index,
        "current_state_generic": list(template["current_state_generic"]),
        "resulting_state_generic": list(template["resulting_state_generic"]),
        "reaction_generic": str(template["reaction_generic"]),
        "electron_pushes_generic": list(template["electron_pushes_generic"]),
        "generic_notes": str(template["generic_notes"]),
    }


def _final_target_from_known_mechanism(case: dict[str, Any]) -> str:
    km = case.get("known_mechanism") or {}
    steps = sorted(km.get("steps") or [], key=lambda s: int(s.get("step_index") or 0))
    if steps:
        return str(steps[-1].get("target_smiles") or "")
    products = case.get("products") or []
    return str(products[0]) if products else ""


def _final_target_reached(known_final_target: str, resulting_state: list[str]) -> bool:
    if not known_final_target:
        return True
    target_norm = _canonical_without_maps(known_final_target)
    state_norm = {_canonical_without_maps(s) for s in resulting_state}
    return target_norm in state_norm or known_final_target in resulting_state


def build_mechanistic_steps(
    case: dict[str, Any],
    mechanism_label: str,
    template_library: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    km = case.get("known_mechanism") or {}
    min_steps = int(km.get("min_steps") or case.get("n_mechanistic_steps") or 0)
    step_targets = sorted(km.get("steps") or [], key=lambda s: int(s.get("step_index") or 0))
    step_targets = [s for s in step_targets if int(s.get("step_index") or 0) <= min_steps]
    quality_flags: list[str] = []
    if not step_targets:
        quality_flags.append("known_mechanism_steps_missing")
        return [], quality_flags

    current_state = [str(x) for x in (case.get("starting_materials") or [])]
    map_seed = 1
    steps: list[dict[str, Any]] = []

    for idx, target in enumerate(step_targets, start=1):
        target_smiles = str(target.get("target_smiles") or "").strip()
        resulting_state = [target_smiles] if target_smiles else []
        if not resulting_state:
            quality_flags.append(f"step_{idx}_missing_target_smiles")
            continue

        mapped_current, map_seed = _map_state(current_state, map_seed)
        mapped_result, map_seed = _map_state(resulting_state, map_seed)
        core = f"{'.'.join(mapped_current)}>>{'.'.join(mapped_result)}"

        concrete_push = _build_concrete_push(mapped_current, mapped_result, idx)
        dbe_block = _build_dbe_block(concrete_push)
        raw = f"{core} |dbe:{dbe_block}|"

        annotation = predict_arrow_push_annotation(
            current_state=mapped_current,
            resulting_state=mapped_result,
            reaction_smirks=core,
            raw_reaction_smirks=raw,
            electron_pushes=[concrete_push],
            step_index=idx,
            candidate_rank=1,
        )
        annotation_suffix = str(annotation.get("annotation_suffix") or "")
        annotated = str(annotation.get("annotated_reaction_smirks") or raw)
        if annotation_suffix and annotation_suffix not in annotated:
            annotated = f"{raw} {annotation_suffix}"

        step_quality: list[str] = []
        if "|dbe:" not in raw:
            step_quality.append("missing_dbe_block")
        if not annotation_suffix.startswith("|aps:v1;"):
            step_quality.append("aps_suffix_not_v1")

        steps.append(
            {
                "step_index": idx,
                "generic_step": _generic_step(
                    mechanism_label=mechanism_label,
                    step_index=idx,
                    template_library=template_library,
                ),
                "concrete_step": {
                    "current_state": mapped_current,
                    "resulting_state": mapped_result,
                    "reaction_smirks": raw,
                    "electron_pushes": [concrete_push],
                },
                "annotation_suffix": annotation_suffix,
                "annotated_reaction_smirks": annotated,
                "selected_candidate": annotation.get("selected_candidate"),
                "quality_flags": step_quality,
            }
        )

        current_state = resulting_state

    return steps, quality_flags


def _build_notebook(path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RXN Map Expanded Visualizer\n",
                    "\n",
                    "Static notebook view (no ipywidgets required). Displays all mapped reactions by default.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "from IPython.display import display, Markdown\n",
                    "\n",
                    "try:\n",
                    "    from rdkit import Chem\n",
                    "    from rdkit.Chem import Draw\n",
                    "except Exception:\n",
                    "    Chem = None\n",
                    "    Draw = None\n",
                    "\n",
                    "DATA_PATH = Path('training_data/rxn_map_expanded.json')\n",
                    "TEMPLATE_PATH = Path('training_data/reaction_type_templates.json')\n",
                    "payload = json.loads(DATA_PATH.read_text(encoding='utf-8'))\n",
                    "template_payload = json.loads(TEMPLATE_PATH.read_text(encoding='utf-8'))\n",
                    "reactions = payload['reactions']\n",
                    "templates = {row['label_exact']: row for row in template_payload['templates']}\n",
                    "print(f\"Loaded {len(reactions)} reactions from {DATA_PATH}\")\n",
                    "print(f\"Loaded {len(templates)} reaction-type templates from {TEMPLATE_PATH}\")\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Optional: render PNGs for visual QA using the script pipeline.\n",
                    "import subprocess\n",
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display, Markdown\n",
                    "\n",
                    "subprocess.run([\n",
                    "    'python',\n",
                    "    'scripts/render_rxn_map_pngs.py',\n",
                    "    '--max-reactions',\n",
                    "    '5',\n",
                    "], check=False)\n",
                    "\n",
                    "png_dir = Path('training_data/pngs')\n",
                    "png_files = sorted(png_dir.glob('*.png'))\n",
                    "display(Markdown(f\"Generated {len(png_files)} PNG files in `{png_dir}`\"))\n",
                    "for png_path in png_files[:3]:\n",
                    "    display(Markdown(f\"- {png_path.name}\"))\n",
                    "    display(Image(filename=str(png_path)))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "summary = pd.DataFrame([\n",
                    "    {\n",
                    "        'reaction_id': r['reaction_id'],\n",
                    "        'tier': r['tier'],\n",
                    "        'mechanism': r['mechanism_type']['label_exact'],\n",
                    "        'confidence': r['mechanism_type']['confidence'],\n",
                    "        'steps': len(r['mechanistic_steps']),\n",
                    "    }\n",
                    "    for r in reactions\n",
                    "])\n",
                    "display(summary.sort_values(['tier', 'reaction_id']))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def draw_state(state):\n",
                    "    if Chem is None or Draw is None:\n",
                    "        return None\n",
                    "    mols = []\n",
                    "    for smi in state:\n",
                    "        core = smi.split('|')[0].strip()\n",
                    "        mol = Chem.MolFromSmiles(core)\n",
                    "        if mol is not None:\n",
                    "            mols.append(mol)\n",
                    "    if not mols:\n",
                    "        return None\n",
                    "    return Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(260, 200))\n",
                    "\n",
                    "def show_reaction(rxn):\n",
                    "    display(Markdown(f\"## {rxn['reaction_id']} - {rxn['name']}\"))\n",
                    "    display(Markdown(f\"**Tier:** {rxn['tier']}  |  **Mechanism:** {rxn['mechanism_type']['label_exact']}  |  **Confidence:** {rxn['mechanism_type']['confidence']}\"))\n",
                    "    display(Markdown(f\"**Rationale:** {rxn['mechanism_type']['rationale']}\"))\n",
                    "    template = templates[rxn['mechanism_type']['label_exact']]\n",
                    "    first_tpl_step = template['generic_mechanism_steps'][0]['reaction_generic']\n",
                    "    display(Markdown(f\"**Type template (step 1):** `{first_tpl_step}`\"))\n",
                    "\n",
                    "    for step in rxn['mechanistic_steps']:\n",
                    "        display(Markdown(f\"### Step {step['step_index']}\"))\n",
                    "        display(Markdown(f\"- Generic reaction: `{step['generic_step']['reaction_generic']}`\"))\n",
                    "        display(Markdown(f\"- Concrete reaction: `{step['concrete_step']['reaction_smirks']}`\"))\n",
                    "        display(Markdown(f\"- APS: `{step['annotation_suffix']}`\"))\n",
                    "        display(Markdown(f\"- Electron pushes: `{step['concrete_step']['electron_pushes']}`\"))\n",
                    "        img_cur = draw_state(step['concrete_step']['current_state'])\n",
                    "        if img_cur is not None:\n",
                    "            display(Markdown('Current state'))\n",
                    "            display(img_cur)\n",
                    "        img_res = draw_state(step['concrete_step']['resulting_state'])\n",
                    "        if img_res is not None:\n",
                    "            display(Markdown('Resulting state'))\n",
                    "            display(img_res)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Focused Verification: From Nitrile Reduction Onward\n",
                    "\n",
                    "This section helps verify the reaction-type templates and mapped examples starting at `Nitrile reduction`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "taxonomy_labels = [t['label_exact'] for t in payload['taxonomy']]\n",
                    "START_LABEL = 'Nitrile reduction'\n",
                    "start_idx = taxonomy_labels.index(START_LABEL) if START_LABEL in taxonomy_labels else 0\n",
                    "focus_labels = taxonomy_labels[start_idx:]\n",
                    "\n",
                    "rows = []\n",
                    "for label in focus_labels:\n",
                    "    count = sum(1 for r in reactions if r['mechanism_type']['label_exact'] == label)\n",
                    "    tpl_steps = templates[label]['generic_mechanism_steps']\n",
                    "    rows.append({'mechanism_label': label, 'mapped_examples': count, 'suitable_step_count': templates[label]['suitable_step_count'], 'template_step1': tpl_steps[0]['reaction_generic']})\n",
                    "focus_df = pd.DataFrame(rows)\n",
                    "display(focus_df)\n",
                    "\n",
                    "def show_label_examples(label, max_examples=2):\n",
                    "    display(Markdown(f\"## Mechanism Type: {label}\"))\n",
                    "    display(Markdown(f\"**Suitable step count:** {templates[label]['suitable_step_count']}\"))\n",
                    "    for tpl_step in templates[label]['generic_mechanism_steps']:\n",
                    "        display(Markdown(f\"- Template step {tpl_step['step_index']}: `{tpl_step['reaction_generic']}`\"))\n",
                    "    subset = [r for r in reactions if r['mechanism_type']['label_exact'] == label][:max_examples]\n",
                    "    if not subset:\n",
                    "        display(Markdown('_No mapped reactions currently assigned to this label in this artifact._'))\n",
                    "        return\n",
                    "    for rxn in subset:\n",
                    "        show_reaction(rxn)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "MECHANISM_FILTER = None  # e.g., 'SN2 reaction'\n",
                    "MAX_REACTIONS_TO_SHOW = None  # set an integer to limit output\n",
                    "\n",
                    "selected = [\n",
                    "    r for r in reactions\n",
                    "    if MECHANISM_FILTER is None or r['mechanism_type']['label_exact'] == MECHANISM_FILTER\n",
                    "]\n",
                    "if MAX_REACTIONS_TO_SHOW is not None:\n",
                    "    selected = selected[:MAX_REACTIONS_TO_SHOW]\n",
                    "\n",
                    "print(f\"Visualizing {len(selected)} reactions\")\n",
                    "for rxn in selected:\n",
                    "    show_reaction(rxn)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "MAX_LABELS_TO_RENDER = None  # set int to limit, e.g. 8\n",
                    "MAX_EXAMPLES_PER_LABEL = 2\n",
                    "labels_to_render = focus_labels if MAX_LABELS_TO_RENDER is None else focus_labels[:MAX_LABELS_TO_RENDER]\n",
                    "print(f\"Rendering {len(labels_to_render)} mechanism labels from '{START_LABEL}' onward\")\n",
                    "for label in labels_to_render:\n",
                    "    show_label_examples(label, max_examples=MAX_EXAMPLES_PER_LABEL)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def build_outputs(
    reaction_type_templates_path: Path,
    eval_set_path: Path,
    eval_tiers_path: Path,
    out_expanded_path: Path,
    out_flat_map_path: Path,
    preferred_eval_map_path: Path,
    out_reaction_type_templates_path: Path,
    out_notebook_path: Path,
    scope: str,
    max_cases: int | None,
) -> None:
    taxonomy = load_taxonomy(reaction_type_templates_path)
    taxonomy_by_norm = _taxonomy_lookup(taxonomy)
    taxonomy_by_id = _taxonomy_lookup_by_id(taxonomy)
    preferred_eval_map = load_preferred_eval_mechanism_map(preferred_eval_map_path)
    template_library = build_generic_template_library(taxonomy)

    eval_set = json.loads(eval_set_path.read_text(encoding="utf-8"))
    tiers_payload = json.loads(eval_tiers_path.read_text(encoding="utf-8"))
    selected_ids, tier_by_id, scope_text = _select_ids(eval_set, tiers_payload, scope=scope, max_cases=max_cases)

    eval_by_id = {str(item.get("id")): item for item in eval_set}

    reactions_out: list[dict[str, Any]] = []
    flat_map: list[dict[str, Any]] = []
    observed_step_max_by_label: dict[str, int] = {}
    example_mappings: list[dict[str, Any]] = []

    for rid in selected_ids:
        if rid not in eval_by_id:
            continue
        case = eval_by_id[rid]
        mechanism_type = assign_mechanism_type(
            case,
            taxonomy_by_norm=taxonomy_by_norm,
            taxonomy_by_id=taxonomy_by_id,
            preferred_eval_map=preferred_eval_map,
        )
        observed_steps = int(
            ((case.get("known_mechanism") or {}).get("min_steps") or case.get("n_mechanistic_steps") or 0)
        )
        canonical_label = str(mechanism_type.get("label_exact") or "")
        if canonical_label:
            observed_step_max_by_label[canonical_label] = max(
                observed_steps,
                int(observed_step_max_by_label.get(canonical_label, 0)),
            )
        steps, case_quality = build_mechanistic_steps(case, mechanism_type["label_exact"], template_library)
        final_target = _final_target_from_known_mechanism(case)

        if steps:
            last_result = steps[-1]["concrete_step"]["resulting_state"]
            if not _final_target_reached(final_target, last_result):
                case_quality.append("known_final_target_not_in_last_resulting_state")

        reaction_record = {
            "reaction_id": rid,
            "tier": tier_by_id.get(rid, "unscoped"),
            "name": case.get("name"),
            "citation": ((case.get("known_mechanism") or {}).get("citation")),
            "source_reaction": {
                "starting_materials": case.get("starting_materials", []),
                "products": case.get("products", []),
                "known_min_steps": ((case.get("known_mechanism") or {}).get("min_steps")),
            },
            "mechanism_type": mechanism_type,
            "known_final_target": final_target,
            "mechanistic_steps": steps,
            "quality_flags": case_quality,
        }
        reactions_out.append(reaction_record)

        flat_map.append(
            {
                "reaction_id": rid,
                "tier": tier_by_id.get(rid, "unscoped"),
                "mechanism_type_label": mechanism_type.get("mechanism_type_label") or mechanism_type["label_exact"],
                "mechanism_type_id": mechanism_type["type_id"],
                "confidence": mechanism_type["confidence"],
                "rationale": mechanism_type["rationale"],
            }
        )
        example_mappings.append(
            {
                "reaction_id": rid,
                "mechanism_type_label": mechanism_type.get("mechanism_type_label") or mechanism_type["label_exact"],
                "mechanism_type_id": mechanism_type["type_id"],
                "confidence": mechanism_type["confidence"],
                "rationale": mechanism_type["rationale"],
            }
        )

    expanded_payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": scope_text,
            "reaction_count": len(reactions_out),
            "sources": {
                "reaction_type_templates": str(reaction_type_templates_path.relative_to(PROJECT_ROOT)),
                "eval_set": str(eval_set_path.relative_to(PROJECT_ROOT)),
                "eval_tiers": str(eval_tiers_path.relative_to(PROJECT_ROOT)),
            },
        },
        "taxonomy": [
            {
                "type_id": m.type_id,
                "label_exact": m.label_exact,
                "slug": m.slug,
            }
            for m in taxonomy
        ],
        "reactions": reactions_out,
    }

    reaction_type_templates_payload = _build_reaction_type_templates_payload(
        taxonomy,
        template_library,
        source_path=reaction_type_templates_path,
        observed_step_max_by_label=observed_step_max_by_label,
        example_mappings=example_mappings,
    )

    out_expanded_path.write_text(json.dumps(expanded_payload, indent=2), encoding="utf-8")
    out_flat_map_path.write_text(json.dumps(flat_map, indent=2), encoding="utf-8")
    out_reaction_type_templates_path.write_text(
        json.dumps(reaction_type_templates_payload, indent=2), encoding="utf-8"
    )
    _build_notebook(out_notebook_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rxn_map expanded artifacts.")
    parser.add_argument(
        "--reaction-type-templates",
        default=str(OUT_REACTION_TYPE_TEMPLATES_PATH),
        help="Input reaction_type_templates.json used as the sole taxonomy source.",
    )
    parser.add_argument("--eval-set", default=str(EVAL_SET_PATH))
    parser.add_argument("--eval-tiers", default=str(EVAL_TIERS_PATH))
    parser.add_argument("--out-expanded", default=str(OUT_EXPANDED_PATH))
    parser.add_argument("--out-flat-map", default=str(OUT_FLAT_MAP_PATH))
    parser.add_argument(
        "--preferred-eval-map",
        default=str(PREFERRED_EVAL_MAP_PATH),
        help="Input eval_mechanism_map.json used as preferred label mapping.",
    )
    parser.add_argument("--out-reaction-type-templates", default=str(OUT_REACTION_TYPE_TEMPLATES_PATH))
    parser.add_argument("--out-notebook", default=str(OUT_NOTEBOOK_PATH))
    parser.add_argument(
        "--scope",
        choices=["first20", "tier30", "all_eval"],
        default="all_eval",
        help="Case selection scope. default: all_eval",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional hard cap after scope selection.")
    args = parser.parse_args()

    build_outputs(
        reaction_type_templates_path=Path(args.reaction_type_templates),
        eval_set_path=Path(args.eval_set),
        eval_tiers_path=Path(args.eval_tiers),
        out_expanded_path=Path(args.out_expanded),
        out_flat_map_path=Path(args.out_flat_map),
        preferred_eval_map_path=Path(args.preferred_eval_map),
        out_reaction_type_templates_path=Path(args.out_reaction_type_templates),
        out_notebook_path=Path(args.out_notebook),
        scope=args.scope,
        max_cases=args.max_cases,
    )
    print(f"Wrote {args.out_expanded}")
    print(f"Wrote {args.out_flat_map}")
    print(f"Wrote {args.out_reaction_type_templates}")
    print(f"Wrote {args.out_notebook}")


if __name__ == "__main__":
    main()
