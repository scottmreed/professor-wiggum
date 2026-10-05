"""Mechanism quality rubric (``quality_v1``): one 1000-point score for harness and baseline runs.

Every mechanism is scored from its accepted path alone (each step's ``current_state``,
``resulting_state``, ``reaction_smirks`` and ``electron_pushes``), re-checked here with the same
deterministic code for every run type. Nothing the harness did internally (retries, its own
validation verdicts, rescue calls) is trusted or needed, so a one-shot baseline and a harness run
are measured the same way.

Components (points out of 1000; no speed, no points for reaching a product the prompt supplied):

========================  ======  ==========================================================
component                 points  what earns it
========================  ======  ==========================================================
step_validity               250   per step: atom/charge balance, arrows parse into a bond/
                                  electron delta, the SMIRKS sides match the stated species,
                                  and the step changes the state
sequence                    200   order of heavy-atom events matches the FlowER reference
                                  (proton-shuttle agnostic, ``skeleton_alignment``)
electron_conservation       100   per step the mapped SMIRKS conserves electrons in the
                                  bond-electron matrix (``bond_electron_view.v1``)
proton_bookkeeping          100   protons come from an explicit donor and go to an explicit
                                  acceptor (no bare ``[H+]``); net protons close at the end
protonation_states          100   no free strongly basic anion under acidic conditions and no
                                  free strongly acidic cation under basic conditions
reagents_and_solvent        100   every species entering a step was supplied (starting
                                  material, its conjugate, available water) or made earlier;
                                  the whole mechanism closes in mass and charge
efficiency                  100   no repeated or undone states (circular sequences) and no
                                  heavy-atom steps beyond the reference
intermolecular              50    a proton transfer uses an available shuttle (solvent, acid,
                                  base, conjugate) instead of an intramolecular shift
========================  ======  ==========================================================

The target product is given in the prompt, so reaching it earns nothing; it is a gate. A
mechanism that does not reach every target scores half and cannot pass. ``passed`` also needs
every step valid, mass/charge closure, no circular step and at least ``PASS_POINTS``.

Conditions are not part of FlowER cases, so they are read from the starting materials
(``classify_conditions``): strong or carboxylic acids make them acidic, hydroxide/alkoxide/hydride
salts, carbonates and aliphatic amines basic, both together buffered. The protonation-state rules
are deliberately narrow (pKa extremes only) so that zwitterions, ammonium ions and carboxylates
are never penalized.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

QUALITY_VERSION = "quality_v1"
WEIGHTS: Dict[str, int] = {
    "step_validity": 250,
    "sequence": 200,
    "electron_conservation": 100,
    "proton_bookkeeping": 100,
    "protonation_states": 100,
    "reagents_and_solvent": 100,
    "efficiency": 100,
    "intermolecular": 50,
}
assert sum(WEIGHTS.values()) == 1000
PASS_POINTS = 700
UNREACHED_FACTOR = 0.5
CLOSURE_SCORE = {"exact": 1.0, "reconciled": 0.9, "approximate": 0.3}
CIRCULAR_PENALTY = 0.5
EXCESS_STEP_PENALTY = 0.2

_PROTON_CARRIERS = {"O", "[OH3+]", "[OH-]"}

# SMARTS for the conditions classifier and the protonation-state rules.
_ACID_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "mineral_acid": "[Cl,Br,I;H1;X1]",
    "sulfuric_or_phosphoric": "[S,P](=O)([OX2H1])",
    "hydronium": "[OH3+]",
}
_BASE_SMARTS = {
    "hydroxide": "[OX1H1-]",
    "alkoxide": "[OX1-][CX4]",
    "hydride": "[H-]",
    "amide_anion": "[NX2-,NX1-]",
    "carbonate": "[OX1-]C(=O)[OX1-,OX2H1]",
    "aliphatic_amine": "[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-[a]);!$(N-S(=O)=O);!$(N#*);!$(N=*)]([CX4])",
    "pyridine_like": "[nX2;r6]",
}
# Free (net-charged) species that are implausible under the given conditions.
_STRONG_BASE_ANIONS = {
    "hydroxide": "[OX1H1-]",
    "alkoxide": "[OX1-][CX4]",
    "amide_anion": "[NX2-,NX1-;!$([N-][S](=O)=O);!$([N-]C=O)]",
    "carbanion": "[#6-;!$([#6-][N+,P+,S+])]",
}
_STRONG_ACID_CATIONS = {
    "hydronium": "[OH3+]",
    "oxonium": "[OX3+;H1,H2;!$([O+]=*)]",
    "protonated_carbonyl": "[OX2+;H1]=[#6]",
}


@dataclass
class QualityStep:
    step_index: int
    current_state: List[str]
    resulting_state: List[str]
    reaction_smirks: str = ""
    electron_pushes: Any = None


# --------------------------------------------------------------------------- chemistry helpers


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


def _tokens(species: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in species or []:
        for token in str(item or "").split("."):
            token = token.strip()
            if token:
                out.append(token)
    return out


def _mol(smiles: str):
    """Parsed species with atom maps cleared and mapped explicit hydrogens folded back in, so a
    SMIRKS written with ``[H:16]`` atoms and a state written without them canonicalize alike."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        return Chem.RemoveHs(mol)
    except Exception:  # noqa: BLE001 - keep the explicit form if hydrogens cannot be folded
        return mol


def canonical(smiles: str) -> Optional[str]:
    mol = _mol(smiles)
    return _rdkit().MolToSmiles(mol) if mol is not None else None


def _canon_list(species: Iterable[Any]) -> List[str]:
    return [c for c in (canonical(t) for t in _tokens(species)) if c]


def _neutral_parent(smiles: str) -> Optional[str]:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = _mol(smiles)
        if mol is None:
            return None
        return _rdkit().MolToSmiles(rdMolStandardize.Uncharger().uncharge(mol))
    except Exception:  # noqa: BLE001
        return canonical(smiles)


def _net_charge(smiles: str) -> int:
    mol = _mol(smiles)
    return sum(a.GetFormalCharge() for a in mol.GetAtoms()) if mol is not None else 0


_PATTERNS: Dict[str, Any] = {}
re_bare_proton = re.compile(r"\[H(?::\d+)?\+\]")


def _matches(smiles: str, smarts: str) -> bool:
    Chem = _rdkit()
    pattern = _PATTERNS.get(smarts)
    if pattern is None:
        pattern = _PATTERNS[smarts] = Chem.MolFromSmarts(smarts)
    mol = _mol(smiles)
    if mol is None or pattern is None:
        return False
    return mol.HasSubstructMatch(pattern) if "H-" not in smarts else Chem.AddHs(mol).HasSubstructMatch(pattern)


# --------------------------------------------------------------------------- conditions


def classify_conditions(starting_materials: Sequence[str]) -> Dict[str, Any]:
    """``{class: acidic|basic|buffered|neutral, acids, bases, water}`` from the starting materials."""
    species = _canon_list(starting_materials)
    acids = sorted({name for s in species for name, sm in _ACID_SMARTS.items() if _matches(s, sm)})
    bases = sorted({name for s in species for name, sm in _BASE_SMARTS.items() if _matches(s, sm)})
    water = any(s in _PROTON_CARRIERS for s in species)
    if acids and bases:
        cls = "buffered"
    elif acids:
        cls = "acidic"
    elif bases:
        cls = "basic"
    else:
        cls = "neutral"
    return {"class": cls, "acids": acids, "bases": bases, "water": water}


def implausible_species(species: Iterable[str], condition_class: str) -> List[Dict[str, str]]:
    """Free (net-charged) species whose protonation state contradicts the conditions."""
    rules = _STRONG_BASE_ANIONS if condition_class == "acidic" else (
        _STRONG_ACID_CATIONS if condition_class == "basic" else {}
    )
    found: List[Dict[str, str]] = []
    for smiles in species:
        charge = _net_charge(smiles)
        if (condition_class == "acidic" and charge >= 0) or (condition_class == "basic" and charge <= 0):
            continue
        for name, smarts in rules.items():
            if _matches(smiles, smarts):
                found.append({"species": smiles, "rule": name})
                break
    return found


# --------------------------------------------------------------------------- per-step checks


def normalize_smirks(reaction_smirks: str) -> str:
    """Reaction notation without representation quirks: surrounding whitespace, a CXSMILES/``mech:v1``
    suffix (``... |...|``) and an agents field (``reactants>agents>products``) are dropped, leaving
    ``reactants>>products`` (empty when there is no reaction arrow)."""
    text = str(reaction_smirks or "").strip().split(" ")[0].split("|")[0].strip()
    if ">>" in text:
        return text
    parts = text.split(">")
    if len(parts) == 3:
        return f"{parts[0]}>>{parts[2]}"
    return ""


def _smirks_sides(reaction_smirks: str) -> Optional[Tuple[List[str], List[str]]]:
    text = normalize_smirks(reaction_smirks)
    if not text:
        return None
    left, _, right = text.partition(">>")
    return _canon_list([left]), _canon_list([right])


def _fragments_in_state(side: str, state: Sequence[str]) -> bool:
    """Every SMIRKS fragment on one side (a whole species or a reacting core such as
    ``[C:1](=[O:2])[N:3]``) is a substructure of the stated species, charges included."""
    Chem = _rdkit()
    combined = _mol(".".join(_tokens(state)))
    if combined is None:
        return False
    combined = Chem.AddHs(combined)  # so explicit [H:n] atoms in a fragment have something to match
    fragments = [t for t in side.split(".") if t.strip()]
    for fragment in fragments:
        query = Chem.MolFromSmarts(fragment)
        if query is None:
            return False
        for atom in query.GetAtoms():
            atom.SetAtomMapNum(0)
        if not combined.HasSubstructMatch(query):
            return False
    return bool(fragments)


def _smirks_matches_states(step: QualityStep) -> bool:
    """The SMIRKS describes the stated step: its species are in the states (whole-species SMIRKS),
    or each fragment of a core-only SMIRKS is a substructure of the stated species."""
    sides = _smirks_sides(step.reaction_smirks)
    if sides is None:
        return False
    left, right = sides
    current = Counter(_canon_list(step.current_state))
    resulting = Counter(_canon_list(step.resulting_state))
    if left and right and not (Counter(left) - current) and not (Counter(right) - resulting):
        return True
    raw_left, _, raw_right = normalize_smirks(step.reaction_smirks).partition(">>")
    return _fragments_in_state(raw_left, step.current_state) and _fragments_in_state(raw_right, step.resulting_state)


def _bond_electron_valid(step: QualityStep) -> Tuple[bool, Optional[str]]:
    """The harness's own arrow check (``predict_mechanistic_step``'s ``bond_electron_validation``):
    explicit pushes are required and must parse, with the SMIRKS, into bond/electron deltas."""
    smirks = str(step.reaction_smirks or "").strip()
    if not smirks:
        return False, "reaction_smirks missing"
    from mechanistic_agent.tools import _extract_dbe_or_infer, extract_mechanism_moves, normalize_electron_pushes

    try:
        pushes = [move.as_dict() for move in normalize_electron_pushes(step.electron_pushes or [])]
        if not pushes:
            # Arrows written only in the SMIRKS ``mech:v1`` block are the same arrows (notation).
            _mech_core, moves, _details = extract_mechanism_moves(smirks)
            raw = [m.as_dict() if hasattr(m, "as_dict") else m for m in moves or []]
            pushes = [move.as_dict() for move in normalize_electron_pushes(raw)]
        if not pushes:
            return False, "no valid explicit electron pushes"
        _core, _deltas, details = _extract_dbe_or_infer(smirks, electron_pushes=pushes)
    except Exception as exc:  # noqa: BLE001 - a step the tools cannot read is invalid
        return False, f"unreadable step: {exc}"
    error = details.get("error")
    return error is None, error


def _fold_explicit_hydrogens(smirks: str) -> str:
    """The SMIRKS with every explicit hydrogen atom (mapped or not) folded into its heavy atom's
    implicit count, so a proton written ``[H:5]`` on one side and inside ``[OH2+:3]`` on the other
    is bookkept the same way on both sides. Bare ions such as ``[H+]`` are kept."""
    Chem = _rdkit()
    from rdkit.Chem import rdmolops

    params = rdmolops.RemoveHsParameters()
    params.removeMapped = True
    sides = []
    for side in smirks.split(">>"):
        mol = Chem.MolFromSmiles(side, sanitize=False)
        if mol is None:
            return smirks
        try:
            mol.UpdatePropertyCache(strict=False)
            mol = rdmolops.RemoveHs(mol, params, sanitize=False)
        except Exception:  # noqa: BLE001
            return smirks
        sides.append(Chem.MolToSmiles(mol, canonical=False))
    return ">>".join(sides)


def _bare_protons(side: str) -> int:
    return sum(1 for token in side.split(".") if re_bare_proton.fullmatch(token.strip()))


def _electron_conserved(step: QualityStep) -> Tuple[bool, Optional[str]]:
    """Electrons are conserved in the bond-electron matrix of the mapped SMIRKS.

    Notation never fails this check: the SMIRKS is also tried with explicit hydrogens folded in
    (mixed explicit/implicit H), and a residual of exactly 2 electrons per bare ``[H+]`` created or
    consumed is the proton itself (a bare proton is penalized once, under proton bookkeeping)."""
    from mechanistic_agent.core.bond_electron import build_bond_electron_view

    smirks = normalize_smirks(step.reaction_smirks)
    if not smirks:
        return False, "reaction_smirks missing"
    left, _, right = smirks.partition(">>")
    proton_residual = 2 * (_bare_protons(right) - _bare_protons(left))
    last_error: Optional[str] = None
    for candidate in (smirks, _fold_explicit_hydrogens(smirks)):
        try:
            view = build_bond_electron_view(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue
        if view.get("projection_error") or view.get("error"):
            last_error = str(view.get("projection_error") or view.get("error"))
            continue
        if view.get("conserved"):
            return True, None
        if proton_residual and view.get("electron_delta_sum") == proton_residual:
            return True, None
        last_error = f"electron_delta_sum={view.get('electron_delta_sum')}"
    return False, last_error


def _composition(species: Iterable[Any]) -> Optional[Counter]:
    """Element counts with hydrogens plus net charge (key ``+``); None if a species does not parse."""
    Chem = _rdkit()
    total: Counter = Counter()
    for token in _tokens(species):
        mol = _mol(token)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            total[atom.GetSymbol()] += 1
            total["+"] += atom.GetFormalCharge()
    return total


def _atom_balanced(step: QualityStep, starting_materials: Sequence[str] = ()) -> Tuple[bool, Dict[str, Any]]:
    """Atoms (H included) and net charge are identical on both sides of the step (pure RDKit,
    the same counts the atom-balance validator compares after repairing SMILES).

    As in the harness validator, a residual that is exactly whole equivalents of one starting
    material carried intact (a second TFA drawn from the solvent) is an excess reagent, not an
    imbalance (``mechanism_audit.excess_reagent_equivalents``)."""
    left, right = _composition(step.current_state), _composition(step.resulting_state)
    if left is None or right is None:
        return False, {"error": "unparseable species"}
    keys = set(left) | set(right)
    delta = {k: right.get(k, 0) - left.get(k, 0) for k in sorted(keys) if right.get(k, 0) != left.get(k, 0)}
    if delta and starting_materials:
        from mechanistic_agent.core.mechanism_audit import excess_reagent_equivalents

        pool = {smiles: "starting_material" for smiles in _canon_list(starting_materials)}
        excess = excess_reagent_equivalents(_canon_list(step.current_state), _canon_list(step.resulting_state), pool)
        if excess is not None and not excess.get("proton_residual"):
            return True, {"delta": delta, "excess_reagent": excess}
    return not delta, {"delta": delta}


def score_step(step: QualityStep, starting_materials: Sequence[str], products: Sequence[str]) -> Dict[str, Any]:
    balanced, balance_detail = _atom_balanced(step, starting_materials)
    arrows_ok, arrows_error = _bond_electron_valid(step)
    smirks_ok = _smirks_matches_states(step)
    progress = Counter(_canon_list(step.current_state)) != Counter(_canon_list(step.resulting_state))
    conserved, conserved_error = _electron_conserved(step)
    checks = {"atom_balance": balanced, "arrows": arrows_ok, "smirks_matches_states": smirks_ok, "state_progress": progress}
    return {
        "step_index": step.step_index,
        "checks": checks,
        "valid": all(checks.values()),
        "validity": sum(checks.values()) / len(checks),
        "electron_conserved": conserved,
        "errors": {k: v for k, v in {"atom_balance": None if balanced else balance_detail,
                                       "arrows": arrows_error, "electron_conservation": conserved_error}.items() if v},
    }


# --------------------------------------------------------------------------- path-level checks


def _introductions(steps: Sequence[QualityStep], starting_materials: Sequence[str]) -> List[Dict[str, Any]]:
    """Species entering a step's current state that the previous state did not hold."""
    out: List[Dict[str, Any]] = []
    previous = Counter(_canon_list(starting_materials))
    for step in steps:
        current = Counter(_canon_list(step.current_state))
        new = current - previous
        if new:
            out.append({"step_index": step.step_index, "species": sorted(new.elements())})
        previous = Counter(_canon_list(step.resulting_state))
    return out


def _supplied(smiles: str, pool_parents: set, conditions: Mapping[str, Any]) -> bool:
    if smiles in _PROTON_CARRIERS:
        return bool(conditions.get("water"))
    return _neutral_parent(smiles) in pool_parents


def _circular(steps: Sequence[QualityStep], starting_materials: Sequence[str]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    states = [Counter(_canon_list(starting_materials))] + [Counter(_canon_list(s.resulting_state)) for s in steps]
    seen: Dict[Tuple[str, ...], int] = {}
    for idx, state in enumerate(states):
        key = tuple(sorted(state.elements()))
        if key in seen and idx > 0:
            findings.append({"type": "repeated_state", "step_index": steps[idx - 1].step_index, "first_seen": seen[key]})
        seen.setdefault(key, idx)
    return findings


def _closure(steps: Sequence[QualityStep], starting_materials: Sequence[str], products: Sequence[str]) -> Dict[str, Any]:
    from mechanistic_agent.core.mechanism_audit import audit_mechanism

    audit = audit_mechanism(
        starting=list(starting_materials),
        targets=list(products),
        steps=[{"step_index": s.step_index, "current_state": s.current_state, "resulting_state": s.resulting_state}
               for s in steps],
    )
    return {"grade": audit.get("grade"), "balanced": audit.get("balanced"), "proton_reconciled": audit.get("proton_reconciled")}


def _targets_reached(steps: Sequence[QualityStep], products: Sequence[str], starting_materials: Sequence[str]) -> Dict[str, Any]:
    final = set(_canon_list(steps[-1].resulting_state)) if steps else set()
    start = set(_canon_list(starting_materials))
    targets = [t for t in _canon_list(products) if t not in start]
    main = max(targets, key=lambda t: _mol(t).GetNumHeavyAtoms(), default=None)
    final_parents = {_neutral_parent(s) for s in final}
    missing = []
    for target in targets:
        if target in final:
            continue
        if _mol(target).GetNumHeavyAtoms() <= 1 and _neutral_parent(target) in final_parents:
            continue  # water present as H3O+ etc.
        missing.append(target)
    return {"main_product": main, "main_reached": bool(main) and main in final, "missing": missing,
            "all_reached": bool(targets) and not missing}


# --------------------------------------------------------------------------- the rubric


def score_mechanism(
    steps: Sequence[QualityStep | Mapping[str, Any]],
    *,
    starting_materials: Sequence[str],
    products: Sequence[str],
    sequence_score: Optional[float] = None,
    reference_skeleton_steps: Optional[int] = None,
    predicted_skeleton_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Score one mechanism. ``sequence_score`` (0..1) is the reference alignment; when there is no
    reference it is ``None`` and the other components are rescaled to 1000."""
    from mechanistic_agent.core.proton_transfer import available_shuttles, classify_proton_transfer

    path = [s if isinstance(s, QualityStep) else QualityStep(
        step_index=int(s.get("step_index") or i + 1),
        current_state=list(s.get("current_state") or []),
        resulting_state=list(s.get("resulting_state") or []),
        reaction_smirks=str(s.get("reaction_smirks") or ""),
        electron_pushes=s.get("electron_pushes"),
    ) for i, s in enumerate(steps)]
    conditions = classify_conditions(starting_materials)
    if not path:
        return {"version": QUALITY_VERSION, "points": 0, "passed": False, "reason": "no_steps",
                "conditions": conditions, "components": {k: 0.0 for k in WEIGHTS}}

    per_step = [score_step(s, starting_materials, products) for s in path]
    n = len(path)
    ratios: Dict[str, Optional[float]] = {}
    ratios["step_validity"] = sum(s["validity"] for s in per_step) / n
    ratios["electron_conservation"] = sum(1 for s in per_step if s["electron_conserved"]) / n
    ratios["sequence"] = None if sequence_score is None else max(0.0, min(1.0, float(sequence_score)))

    closure = _closure(path, starting_materials, products)
    closure_factor = CLOSURE_SCORE.get(str(closure.get("grade")), 0.0)

    bare_proton_steps = [s.step_index for s in path
                         if any(t in {"[H+]", "[H]"} for t in _canon_list(s.current_state + s.resulting_state))]
    proton_closure = 1.0 if closure.get("grade") in {"exact", "reconciled"} or closure.get("proton_reconciled") else 0.5
    ratios["proton_bookkeeping"] = (1 - len(bare_proton_steps) / n) * proton_closure

    implausible = []
    for s in path:
        bad = implausible_species(_canon_list(s.resulting_state), conditions["class"])
        if bad:
            implausible.append({"step_index": s.step_index, "species": bad})
    ratios["protonation_states"] = 1 - len(implausible) / n

    pool_parents = {_neutral_parent(s) for s in _canon_list(starting_materials)}
    unexplained = []
    for item in _introductions(path, starting_materials):
        missing = [sp for sp in item["species"] if not _supplied(sp, pool_parents, conditions)]
        if missing:
            unexplained.append({"step_index": item["step_index"], "species": missing})
    ratios["reagents_and_solvent"] = 0.6 * (1 - len(unexplained) / n) + 0.4 * closure_factor

    circular = _circular(path, starting_materials)
    excess = 0
    if reference_skeleton_steps is not None and predicted_skeleton_steps is not None:
        excess = max(0, int(predicted_skeleton_steps) - int(reference_skeleton_steps))
    ratios["efficiency"] = max(0.0, 1 - CIRCULAR_PENALTY * len(circular) - EXCESS_STEP_PENALTY * excess)

    pt_steps = []
    for s in path:
        shuttles = available_shuttles(s.current_state, starting_materials)
        info = classify_proton_transfer(s.current_state, s.resulting_state, s.reaction_smirks, shuttles)
        if not info.get("is_proton_transfer") or info.get("mode") is None:
            continue
        other_shuttle = any(key not in set(info.get("changed_skeletons") or []) for key in shuttles)
        credited = info["mode"] == "intermolecular" or not other_shuttle
        pt_steps.append({"step_index": s.step_index, "mode": info["mode"], "shuttle": info.get("shuttle"),
                         "shuttle_available": other_shuttle, "credited": credited})
    ratios["intermolecular"] = (sum(1 for p in pt_steps if p["credited"]) / len(pt_steps)) if pt_steps else 1.0

    active = {k: w for k, w in WEIGHTS.items() if ratios.get(k) is not None}
    scale = 1000.0 / sum(active.values())
    components = {k: round(ratios[k] * w * scale, 1) for k, w in active.items()}
    raw_points = sum(components.values())
    targets = _targets_reached(path, products, starting_materials)
    points = raw_points if targets["all_reached"] else raw_points * UNREACHED_FACTOR
    all_valid = all(s["valid"] for s in per_step)
    passed = bool(targets["all_reached"] and all_valid and closure.get("grade") in {"exact", "reconciled"}
                  and not circular and points >= PASS_POINTS)
    return {
        "version": QUALITY_VERSION,
        "points": round(points, 1),
        "raw_points": round(raw_points, 1),
        "passed": passed,
        "components": components,
        "ratios": {k: (round(v, 4) if v is not None else None) for k, v in ratios.items()},
        "targets": targets,
        "conditions": conditions,
        "valid_steps": sum(1 for s in per_step if s["valid"]),
        "step_count": n,
        "closure": closure,
        "findings": {
            "bare_proton_steps": bare_proton_steps,
            "implausible_protonation": implausible,
            "unexplained_species": unexplained,
            "circular": circular,
            "excess_heavy_atom_steps": excess,
            "proton_transfers": pt_steps,
        },
        "steps": per_step,
    }


# --------------------------------------------------------------------------- run adapters


def steps_from_snapshot(snapshot: Mapping[str, Any]) -> List[QualityStep]:
    """Accepted path with SMIRKS/arrows, from a harness run snapshot or a baseline synthetic one.

    Uses the final chosen path (a re-acceptance at step k drops later steps). Older runs whose
    acceptance events lack SMIRKS fall back to the matching ``mechanism_synthesis`` output.
    """
    from mechanistic_agent.core.mechanism_audit import chosen_path_from_events

    events = list(snapshot.get("events") or [])
    chosen = chosen_path_from_events(events)
    outputs = [row for row in (snapshot.get("step_outputs") or [])
               if row.get("step_name") in {"mechanism_synthesis", "baseline_mechanism_step"}]
    path: List[QualityStep] = []
    for payload in chosen:
        index = int(payload.get("step_index") or 0)
        smirks = str(payload.get("reaction_smirks") or "")
        pushes = payload.get("electron_pushes")
        if not smirks:
            resulting = Counter(_canon_list(payload.get("resulting_state") or []))
            for row in reversed(outputs):
                out = row.get("output") or {}
                if int(row.get("attempt") or out.get("step_index") or 0) != index:
                    continue
                if Counter(_canon_list(out.get("resulting_state") or [])) == resulting or not resulting:
                    smirks = str(out.get("raw_reaction_smirks") or out.get("reaction_smirks") or "")
                    pushes = out.get("electron_pushes")
                    break
        path.append(QualityStep(index, list(payload.get("current_state") or []),
                                list(payload.get("resulting_state") or []), smirks, pushes))
    return path


def score_snapshot_quality(snapshot: Mapping[str, Any], expected: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """quality_v1 for a run snapshot against an eval case's expected payload."""
    from mechanistic_agent.scoring import score_snapshot_against_known

    inputs = next((snapshot[key] for key in ("input_payload", "input") if isinstance(snapshot.get(key), Mapping)), {})
    exp_starting, exp_products = _expected_inputs(expected)
    starting = list(snapshot.get("starting_materials") or inputs.get("starting_materials") or exp_starting)
    products = list(exp_products or snapshot.get("products") or inputs.get("products") or [])
    graded = score_snapshot_against_known(snapshot, expected, scoring_version="v3") if expected else {}
    skeleton = graded.get("proton_agnostic_alignment") or {}
    sequence = graded.get("known_alignment_component") if expected else None
    result = score_mechanism(
        steps_from_snapshot(snapshot),
        starting_materials=starting,
        products=products,
        sequence_score=sequence,
        reference_skeleton_steps=skeleton.get("reference_skeleton_steps"),
        predicted_skeleton_steps=skeleton.get("predicted_skeleton_steps"),
    )
    result["sequence_basis"] = graded.get("alignment_basis")
    return result


def quality_or_error(snapshot: Mapping[str, Any], expected: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``score_snapshot_quality`` that never raises: scoring must not break an eval."""
    try:
        return score_snapshot_quality(snapshot, expected)
    except Exception as exc:  # noqa: BLE001
        return {"version": QUALITY_VERSION, "error": f"{type(exc).__name__}: {exc}", "points": 0.0, "passed": False}


def summarize(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Mean points per component over cases, plus pass and target counts."""
    n = len(results)
    if not n:
        return {"version": QUALITY_VERSION, "cases": 0}
    components = {k: round(sum(float((r.get("components") or {}).get(k, 0.0)) for r in results) / n, 1) for k in WEIGHTS}
    return {
        "version": QUALITY_VERSION,
        "cases": n,
        "points": round(sum(float(r.get("points") or 0.0) for r in results) / n, 1),
        "components": components,
        "passed": sum(1 for r in results if r.get("passed")),
        "targets_reached": sum(1 for r in results if (r.get("targets") or {}).get("all_reached")),
        "valid_step_fraction": round(
            sum(int(r.get("valid_steps") or 0) for r in results) / max(1, sum(int(r.get("step_count") or 0) for r in results)), 4
        ),
    }


def _expected_inputs(expected: Optional[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
    """Starting materials and products of an eval case's expected payload (the reference
    mechanism's first and last states when the payload does not list them)."""
    expected = expected or {}
    steps = (expected.get("verified_mechanism") or {}).get("steps") if isinstance(expected.get("verified_mechanism"), Mapping) else None
    steps = sorted((s for s in steps or [] if isinstance(s, Mapping)), key=lambda s: int(s.get("step_index") or 0))
    starting = list(expected.get("starting_materials") or (steps[0].get("current_state") if steps else None) or [])
    products = list(expected.get("products") or [])
    return starting, products


def _baseline_snapshot(summary: Mapping[str, Any], expected: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    steps = summary.get("baseline_steps")
    if not isinstance(steps, list):
        return None
    from mechanistic_agent.core.baseline_runner import _steps_to_synthetic_snapshot

    starting, products = _expected_inputs(expected)
    return _steps_to_synthetic_snapshot(steps, starting, products)


def rescore_quality(store: Any, eval_run_ids: Sequence[str], *, write: bool = False) -> List[Dict[str, Any]]:
    """Compute quality_v1 for stored eval results and (with ``write``) save it as ``summary.quality``.

    Harness results are scored from their run snapshot; baseline results from ``baseline_steps``.
    A baseline stored before steps were kept has nothing to re-check and is reported as
    ``legacy`` (its old case score stays, labelled legacy on the leaderboard).
    """
    from mechanistic_agent.rescoring import default_expected_resolver

    resolver = default_expected_resolver(store)
    rows: List[Dict[str, Any]] = []
    for eval_run_id in eval_run_ids:
        run = store.get_eval_run(eval_run_id) or {}
        for result in store.list_eval_run_results(eval_run_id):
            summary = dict(result.get("summary") or {})
            expected = resolver(result, run)
            mode = str(summary.get("eval_mode") or ("baseline" if not result.get("run_id") else "harness"))
            snapshot = (
                store.get_run_snapshot(result["run_id"]) if result.get("run_id")
                else _baseline_snapshot(summary, expected)
            )
            row = {"eval_run_id": eval_run_id, "case_id": result.get("case_id"), "mode": mode}
            if not snapshot or not expected:
                row["status"] = "legacy" if mode == "baseline" else "no_snapshot"
                rows.append(row)
                continue
            quality = quality_or_error(snapshot, expected)
            row.update(status="scored", points=quality.get("points"), passed=quality.get("passed"))
            rows.append(row)
            if write:
                summary["quality"] = quality
                store.update_eval_run_result(
                    str(result["id"]), score=result.get("score"), passed=result.get("pass_bool"), summary=summary
                )
    return rows
