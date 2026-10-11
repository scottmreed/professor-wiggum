"""Mechanism quality rubric ``quality_v2``: six axes, one score, the same for harness and baseline runs.

``quality_v1`` (``quality_scoring.py``) re-checks every step, but a one-shot baseline that jumps
between unconnected states can still score well because nothing checks that step k+1 starts where
step k ended. ``quality_v2`` adds that check and puts every measure on one 0..1 axis:

=====================  =====================================================================
axis                   what it measures (per case)
=====================  =====================================================================
product                1 when the product gate of ``quality_v1`` is met (all targets when the
                       product is given; the main product in any protonation state when hidden)
connectivity           fraction of steps whose start follows from what came before (below)
balance                mean over steps: atom/charge balanced **and** the state changes
notation               mean over steps: arrows parse into a bond/electron change **and** the
                       SMIRKS sides match the stated species
electrons              mean over steps: the mapped SMIRKS conserves electrons
sequence               heavy-atom events in the FlowER reference order (``quality_v1`` ratio)
=====================  =====================================================================

``score = 0.2·product + 0.3·connectivity + 0.5·Q`` with ``Q`` the mean of the four step axes.
The step axes are means over the steps the mechanism made, so a mechanism that stops early is
not penalized on them; it simply misses the product. A mechanism with no steps (a *failed
start*) scores 0 on every axis. Connectivity carries more weight than the product, so reaching
the product through disconnected steps scores below a connected mechanism that falls short.

Connectivity. Small species (at most one heavy atom: water, H3O+, halide, metal cation, H2) and
common solvents are ignored, because states routinely add or drop them. Other species are
compared by neutral parent, so a protonation change is not a break. A link is broken when

* the first step does not start from a starting material, or introduces a species that is neither
  a starting material nor a conjugate of one;
* a later step drops a species with more than four heavy atoms that the path itself made (an
  intermediate vanishes between steps; a small released byproduct such as SO2 or isobutylene may go);
* a later step introduces a species that is neither in the starting-material pool nor made by an
  earlier step (a product or intermediate appears from nowhere).

Dropping a spare starting material or reagent between steps is not a break.

Aggregates over cases: the mean of each axis, ``score`` (the mean case score), ``polygon_area``
(radar polygon area normalised to 1 for a perfect run: Σ v_i·v_{i+1} / 6 around the axes in
order), and ``failed_starts`` (fraction of cases with no step).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

QUALITY_V2_VERSION = "quality_v2"
AXES = ("product", "connectivity", "balance", "notation", "electrons", "sequence")
STEP_AXES = ("balance", "notation", "electrons", "sequence")
AXIS_LABELS = {
    "product": "Product correct",
    "connectivity": "Step connectivity",
    "balance": "Balanced, progressing",
    "notation": "Arrows + SMIRKS",
    "electrons": "Electron conservation",
    "sequence": "Matches reference",
}
SCORE_WEIGHTS = {"product": 0.2, "connectivity": 0.3, "step_quality": 0.5}
PASS_SCORE = 0.7

# Canonical SMILES of inert solvents that states add or drop freely (ignored for connectivity only).
# Solvents that are also common reagents or products (alcohols, acetone, ethyl acetate) are not listed.
SOLVENTS = frozenset({
    "C1CCOC1", "ClCCl", "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl", "CN(C)C=O", "CS(C)=O", "CC#N", "Cc1ccccc1",
    "c1ccccc1", "CCOCC", "C1COCCO1", "COCCOC", "COC(C)(C)C", "CCCCCC", "C1CCCCC1", "CN1CCCC1=O",
})


# A made species this small (SO2, CO2, isobutylene, B(OH)3 fragments) leaving the state between steps
# is a released byproduct, not a vanished intermediate.
BYPRODUCT_MAX_HEAVY = 4


def _heavy_atoms(smiles: str) -> int:
    from mechanistic_agent.quality_scoring import _mol

    mol = _mol(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _parents(species: Iterable[str], *, keep_solvents: bool = False) -> set:
    """Neutral parents of the species with more than one heavy atom (solvents dropped unless kept)."""
    from mechanistic_agent.quality_scoring import _canon_list, _mol, _neutral_parent

    out = set()
    for smiles in _canon_list(species):
        mol = _mol(smiles)
        if mol is None or mol.GetNumHeavyAtoms() <= 1:
            continue
        parent = _neutral_parent(smiles)
        if parent and (keep_solvents or parent not in SOLVENTS):
            out.add(parent)
    return out


def connectivity_links(steps: Sequence[Any], starting_materials: Sequence[str]) -> List[Dict[str, Any]]:
    """One link per step: whether the step's current state follows from what came before."""
    pool = _parents(starting_materials, keep_solvents=True)
    made: set = set()  # parents produced by the path so far
    links: List[Dict[str, Any]] = []
    previous: Optional[set] = None
    for step in steps:
        current = _parents(step.current_state)
        if previous is None:
            foreign = sorted(current - pool)
            ok = bool(current & pool) and not foreign
            links.append({"step_index": step.step_index, "connected": ok,
                          **({"appeared": foreign} if foreign else {}),
                          **({"reason": "no_starting_material"} if not current & pool else {})})
        else:
            vanished = sorted(p for p in (previous - current) & made - pool if _heavy_atoms(p) > BYPRODUCT_MAX_HEAVY)
            appeared = sorted(current - previous - pool - made)
            ok = not vanished and not appeared
            links.append({"step_index": step.step_index, "connected": ok,
                          **({"vanished": vanished} if vanished else {}),
                          **({"appeared": appeared} if appeared else {})})
        resulting = _parents(step.resulting_state)
        made |= resulting - pool
        previous = resulting
    return links


def score_case(
    steps: Sequence[Any],
    v1: Mapping[str, Any],
    *,
    starting_materials: Sequence[str],
) -> Dict[str, Any]:
    """quality_v2 for one case from its accepted path and its ``quality_v1`` result."""
    n = len(steps)
    per_step = list(v1.get("steps") or [])
    sequence = (v1.get("ratios") or {}).get("sequence")
    if not n or not per_step:
        axes = {k: 0.0 for k in AXES}
        return {"version": QUALITY_V2_VERSION, "axes": axes, "step_quality": 0.0, "score": 0.0,
                "points": 0.0, "failed_start": True, "step_count": 0, "links": [], "passed": False}
    links = connectivity_links(steps, starting_materials)
    axes = {
        "product": 1.0 if v1.get("product_correct") else 0.0,
        "connectivity": sum(1 for link in links if link["connected"]) / n,
        "balance": sum(1 for s in per_step if s["checks"]["atom_balance"] and s["checks"]["state_progress"]) / n,
        "notation": sum(1 for s in per_step if s["checks"]["arrows"] and s["checks"]["smirks_matches_states"]) / n,
        "electrons": sum(1 for s in per_step if s.get("electron_conserved")) / n,
    }
    step_axes = ["balance", "notation", "electrons"]
    if sequence is not None:
        axes["sequence"] = float(sequence)
        step_axes.append("sequence")
    step_quality = sum(axes[k] for k in step_axes) / len(step_axes)
    score = (SCORE_WEIGHTS["product"] * axes["product"] + SCORE_WEIGHTS["connectivity"] * axes["connectivity"]
             + SCORE_WEIGHTS["step_quality"] * step_quality)
    all_connected = all(link["connected"] for link in links)
    return {
        "version": QUALITY_V2_VERSION,
        "axes": {k: round(v, 4) for k, v in axes.items()},
        "step_quality": round(step_quality, 4),
        "score": round(score, 4),
        "points": round(score * 1000, 1),
        "failed_start": False,
        "step_count": n,
        "all_connected": all_connected,
        "links": links,
        "passed": bool(v1.get("passed") and all_connected and score >= PASS_SCORE),
    }


def polygon_area(axes: Mapping[str, float]) -> float:
    """Radar polygon area over ``AXES`` in order, normalised so a perfect run is 1."""
    vals = [float(axes.get(k) or 0.0) for k in AXES]
    return sum(vals[i] * vals[(i + 1) % len(vals)] for i in range(len(vals))) / len(vals)


def summarize(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Mean axes, mean score, polygon area and failed-start fraction over cases."""
    n = len(cases)
    if not n:
        return {"version": QUALITY_V2_VERSION, "cases": 0}
    axes = {k: round(sum(float((c.get("axes") or {}).get(k, 0.0)) for c in cases) / n, 4) for k in AXES}
    failed = sum(1 for c in cases if c.get("failed_start"))
    return {
        "version": QUALITY_V2_VERSION,
        "cases": n,
        "score": round(sum(float(c.get("score") or 0.0) for c in cases) / n, 4),
        "points": round(sum(float(c.get("points") or 0.0) for c in cases) / n, 1),
        "axes": axes,
        "step_quality": round(sum(float(c.get("step_quality") or 0.0) for c in cases) / n, 4),
        "polygon_area": round(polygon_area(axes), 4),
        "failed_starts": failed,
        "failed_start_fraction": round(failed / n, 4),
        "fully_connected": sum(1 for c in cases if c.get("all_connected")),
        "passed": sum(1 for c in cases if c.get("passed")),
    }
