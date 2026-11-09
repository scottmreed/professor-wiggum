"""``catalytic_cycle`` teaching-asset spec: validation and accessible text.

Spec shape (JSON)::

    {
      "title": "Suzuki-Miyaura cross-coupling",
      "species": [
        {"id": "pd0", "label": "Pd(0)L2",
         "smiles": "c1ccc(cc1)P(c1ccccc1)(c1ccccc1)->[Pd]<-P(c1ccccc1)(c1ccccc1)c1ccccc1",
         "ox_state": 0}
      ],
      "steps": [
        {"from": "pd0", "to": "oa", "kind": "oxidative_addition",
         "label": "oxidative addition", "enters": ["Ar-Br"], "leaves": []}
      ],
      "legend": "L = PPh3; Ar = Ph",
      "accessibility": {"alt_text": "...", "long_description": "..."}
    }

* ``species[].smiles`` is optional and uses the dative house conventions in
  :mod:`organometallic_chem.organometallic_smiles`. When present it must lint clean, and
  the oxidation state and valence-electron count are *computed*
  (:mod:`organometallic_chem.organometallic_electron_count`); an authored ``ox_state`` (or
  ``electron_count``) that disagrees is an error. A species without SMILES may
  state ``ox_state`` / ``electron_count`` directly.
* ``steps`` are listed in cycle order and must form one closed cycle visiting
  every species exactly once: ``steps[i].to == steps[i+1].from`` and the last
  ``to`` is the first ``from``.
* ``steps[].kind`` is one of :data:`STEP_KINDS`. Each kind fixes the change in
  oxidation state and electron count across the step (covalent model; see
  :data:`organometallic_chem.organometallic_electron_count.STEP_DELTAS`): oxidative
  addition +2/+2, reductive elimination -2/-2, transmetalation 0/0, migratory
  insertion 0/-2, beta-hydride elimination 0/+2, ligand association 0/+2,
  ligand dissociation 0/-2; ``other`` (e.g. a trans-to-cis isomerization) is not
  checked. A delta is checked only when both species have a known value.
* ``enters`` / ``leaves`` items are display strings (a label such as
  ``"Ar'B(OH)2"``) or ``{"label": ..., "smiles": ...}`` objects; an object's
  SMILES must parse.

Labels are authored in plain ASCII digits ("L2", "B(OH)3"): the renderer
subscripts them, and the accessible text keeps them readable by a screen reader.

Pure RDKit + organometallic_chem, so the offline topic-package validator can import it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rdkit import Chem, rdBase

from .organometallic_electron_count import STEP_DELTAS, ElectronCount, try_count_electrons
from .organometallic_smiles import lint_organometallic_smiles

STEP_KINDS = tuple(STEP_DELTAS)

STEP_KIND_PHRASES: Dict[str, str] = {
    "oxidative_addition": "oxidative addition",
    "reductive_elimination": "reductive elimination",
    "transmetalation": "transmetalation",
    "migratory_insertion": "migratory insertion",
    "beta_hydride_elimination": "beta-hydride elimination",
    "ligand_association": "ligand association",
    "ligand_dissociation": "ligand dissociation",
    "other": "a step",
}

_TOP_KEYS = {"title", "species", "steps", "legend", "accessibility"}
_SPECIES_KEYS = {"id", "label", "smiles", "ox_state", "electron_count"}
_STEP_KEYS = {"from", "to", "kind", "label", "enters", "leaves"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def species_counts(spec: Dict[str, Any]) -> Dict[str, Optional[ElectronCount]]:
    """Electron count per species id (``None`` when it has no countable SMILES)."""
    counts: Dict[str, Optional[ElectronCount]] = {}
    for species in spec.get("species") or []:
        if not isinstance(species, dict) or not isinstance(species.get("id"), str):
            continue
        smiles = species.get("smiles")
        count = try_count_electrons(smiles)[0] if isinstance(smiles, str) and smiles.strip() else None
        counts[species["id"]] = count
    return counts


def species_values(spec: Dict[str, Any]) -> Dict[str, Dict[str, Optional[int]]]:
    """Resolved ``{"ox_state", "electron_count"}`` per species id."""
    counts = species_counts(spec)
    resolved: Dict[str, Dict[str, Optional[int]]] = {}
    for species in spec.get("species") or []:
        if not isinstance(species, dict) or not isinstance(species.get("id"), str):
            continue
        count = counts.get(species["id"])
        resolved[species["id"]] = {
            "ox_state": count.oxidation_state if count else _int_or_none(species.get("ox_state")),
            "electron_count": (
                count.valence_electrons if count else _int_or_none(species.get("electron_count"))
            ),
        }
    return resolved


def _int_or_none(value: Any) -> Optional[int]:
    return value if _is_int(value) else None


def _validate_reagent(item: Any, where: str, errors: List[str]) -> None:
    if isinstance(item, str):
        if not item.strip():
            errors.append(f"{where}: empty reagent label")
        return
    if isinstance(item, dict):
        label, smiles = item.get("label"), item.get("smiles")
        if not (isinstance(label, str) and label.strip()) and not (
            isinstance(smiles, str) and smiles.strip()
        ):
            errors.append(f"{where}: a reagent object needs a label or smiles")
        if isinstance(smiles, str) and smiles.strip():
            blocker = rdBase.BlockLogs()  # noqa: F841
            if Chem.MolFromSmiles(smiles) is None:
                errors.append(f"{where}: reagent SMILES {smiles!r} does not parse")
        return
    errors.append(f"{where}: reagent must be a string or an object")


def validate_catalytic_cycle(spec: Any) -> List[str]:
    """Every problem with a ``catalytic_cycle`` spec; ``[]`` when it is valid."""
    if not isinstance(spec, dict):
        return ["catalytic_cycle spec must be an object"]
    errors: List[str] = []
    unknown = sorted(set(spec) - _TOP_KEYS)
    if unknown:
        errors.append(f"unknown catalytic_cycle key(s): {', '.join(unknown)}")
    if not (isinstance(spec.get("title"), str) and spec["title"].strip()):
        errors.append("title is required")
    accessibility = spec.get("accessibility")
    if accessibility is not None and not isinstance(accessibility, dict):
        errors.append("accessibility must be an object")

    species_list = spec.get("species")
    steps = spec.get("steps")
    if not isinstance(species_list, list) or len(species_list) < 2:
        errors.append("species must be a list of at least two species")
        species_list = species_list if isinstance(species_list, list) else []
    if not isinstance(steps, list) or len(steps) < 2:
        errors.append("steps must be a list of at least two steps")
        steps = steps if isinstance(steps, list) else []

    ids: List[str] = []
    for index, species in enumerate(species_list):
        where = f"species[{index}]"
        if not isinstance(species, dict):
            errors.append(f"{where} must be an object")
            continue
        extra = sorted(set(species) - _SPECIES_KEYS)
        if extra:
            errors.append(f"{where}: unknown key(s) {', '.join(extra)}")
        species_id = species.get("id")
        if not (isinstance(species_id, str) and species_id.strip()):
            errors.append(f"{where}: id is required")
            continue
        where = f"species {species_id!r}"
        ids.append(species_id)
        if not (isinstance(species.get("label"), str) and species["label"].strip()):
            errors.append(f"{where}: label is required")
        for key in ("ox_state", "electron_count"):
            if key in species and species[key] is not None and not _is_int(species[key]):
                errors.append(f"{where}: {key} must be an integer")
        smiles = species.get("smiles")
        if smiles is None:
            continue
        if not (isinstance(smiles, str) and smiles.strip()):
            errors.append(f"{where}: smiles must be a non-empty string")
            continue
        problems = lint_organometallic_smiles(smiles)
        if problems:
            errors.extend(f"{where}: {problem}" for problem in problems)
            continue
        blocker = rdBase.BlockLogs()  # noqa: F841
        if Chem.MolFromSmiles(smiles) is None:
            errors.append(f"{where}: SMILES {smiles!r} does not parse")
            continue
        count, reason = try_count_electrons(smiles)
        if count is None:
            errors.append(f"{where}: cannot count electrons: {reason}")
            continue
        for key, computed in (
            ("ox_state", count.oxidation_state),
            ("electron_count", count.valence_electrons),
        ):
            authored = species.get(key)
            if _is_int(authored) and authored != computed:
                errors.append(
                    f"{where}: authored {key} {authored} disagrees with {computed} "
                    f"computed from its SMILES ({count.summary})"
                )

    duplicates = sorted({species_id for species_id in ids if ids.count(species_id) > 1})
    if duplicates:
        errors.append(f"duplicate species id(s): {', '.join(duplicates)}")
    known = set(ids)

    step_pairs: List[tuple] = []
    for index, step in enumerate(steps):
        where = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{where} must be an object")
            continue
        extra = sorted(set(step) - _STEP_KEYS)
        if extra:
            errors.append(f"{where}: unknown key(s) {', '.join(extra)}")
        source, target, kind = step.get("from"), step.get("to"), step.get("kind")
        for key, value in (("from", source), ("to", target)):
            if value not in known:
                errors.append(f"{where}: {key} {value!r} is not a species id")
        if kind not in STEP_KINDS:
            errors.append(f"{where}: kind {kind!r} must be one of {list(STEP_KINDS)}")
        if "label" in step and not isinstance(step["label"], str):
            errors.append(f"{where}: label must be a string")
        for key in ("enters", "leaves"):
            items = step.get(key, [])
            if not isinstance(items, list):
                errors.append(f"{where}: {key} must be a list")
                continue
            for item_index, item in enumerate(items):
                _validate_reagent(item, f"{where}.{key}[{item_index}]", errors)
        step_pairs.append((index, source, target, kind))

    # One closed cycle, in listed order, through every species exactly once.
    if step_pairs and known:
        sources = [source for _i, source, _t, _k in step_pairs]
        for index in range(len(step_pairs)):
            _i, _source, target, _kind = step_pairs[index]
            following = step_pairs[(index + 1) % len(step_pairs)]
            if target != following[1]:
                if index == len(step_pairs) - 1:
                    errors.append(
                        f"cycle is not closed: the last step ends at {target!r} but the "
                        f"first step starts from {following[1]!r}"
                    )
                else:
                    errors.append(
                        f"steps[{index}] ends at {target!r} but steps[{index + 1}] starts "
                        f"from {following[1]!r}; list steps in cycle order"
                    )
        repeated = sorted({s for s in sources if sources.count(s) > 1 and s in known})
        if repeated:
            errors.append(f"species visited more than once: {', '.join(map(str, repeated))}")
        missing = sorted(known - set(sources))
        if missing:
            errors.append(f"species not on the cycle: {', '.join(missing)}")

    # Oxidation-state and electron-count bookkeeping per step.
    values = species_values(spec)
    for index, source, target, kind in step_pairs:
        if kind not in STEP_DELTAS or source not in values or target not in values:
            continue
        expected_ox, expected_e = STEP_DELTAS[kind]
        for label, key, expected in (
            ("oxidation state", "ox_state", expected_ox),
            ("electron count", "electron_count", expected_e),
        ):
            before, after = values[source][key], values[target][key]
            if expected is None or before is None or after is None:
                continue
            if after - before != expected:
                errors.append(
                    f"steps[{index}] {kind} {source!r} -> {target!r}: {label} changes "
                    f"by {after - before:+d} ({before} -> {after}), expected {expected:+d}"
                )
    return errors


def reagent_label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("label") or item.get("smiles") or "").strip()
    return str(item or "").strip()


def _join(items: List[str]) -> str:
    items = [item for item in items if item]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _species_phrase(species: Dict[str, Any], values: Dict[str, Optional[int]], count) -> str:
    label = str(species.get("label") or species.get("id"))
    facts = []
    if count is not None:
        facts.append(f"{count.metal}({count.roman_oxidation_state}), {count.valence_electrons}-electron")
    elif values.get("ox_state") is not None:
        facts.append(f"oxidation state {values['ox_state']}")
    return f"{label} ({facts[0]})" if facts else label


def describe_catalytic_cycle(spec: Dict[str, Any]) -> str:
    """Ordered prose long description of a (valid) catalytic cycle."""
    species_by_id = {
        s["id"]: s for s in spec.get("species") or [] if isinstance(s, dict) and "id" in s
    }
    steps = [s for s in spec.get("steps") or [] if isinstance(s, dict)]
    counts = species_counts(spec)
    values = species_values(spec)

    mentioned: set = set()

    def phrase(species_id: str) -> str:
        # Oxidation state and electron count on first mention only; repeating
        # them every time makes the description much harder to listen to.
        species = species_by_id.get(species_id, {"id": species_id})
        if species_id in mentioned:
            return str(species.get("label") or species_id)
        mentioned.add(species_id)
        return _species_phrase(species, values.get(species_id, {}), counts.get(species_id))

    title = str(spec.get("title") or "Catalytic cycle").strip()
    sentences = [
        f"{title}: a catalytic cycle of {len(steps)} steps, read clockwise "
        f"starting from {phrase(steps[0]['from']) if steps else 'the catalyst'}."
    ]
    first = steps[0]["from"] if steps else None
    for index, step in enumerate(steps):
        kind = STEP_KIND_PHRASES.get(step.get("kind"), "a step")
        if step.get("kind") == "other" and step.get("label"):
            kind = str(step["label"])
        target_id = step.get("to")
        regenerates = target_id == first and index == len(steps) - 1
        enters = _join([reagent_label(item) for item in step.get("enters") or []])
        leaves = _join([reagent_label(item) for item in step.get("leaves") or []])
        sentence = f"{'then ' if index else ''}{phrase(step.get('from'))} undergoes {kind}"
        if enters:
            sentence += f" with {enters}"
        if leaves and regenerates:
            sentence += f", releasing {leaves} and regenerating {phrase(target_id)}."
        elif leaves:
            sentence += f", releasing {leaves}, to give {phrase(target_id)}."
        elif regenerates:
            sentence += f" and regenerates {phrase(target_id)}."
        else:
            sentence += f" to give {phrase(target_id)}."
        sentences.append(sentence[0].upper() + sentence[1:])
    legend = spec.get("legend")
    if isinstance(legend, str) and legend.strip():
        sentences.append(f"Key: {legend.strip()}.")
    return " ".join(sentences)
