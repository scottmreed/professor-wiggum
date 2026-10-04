#!/usr/bin/env python3
"""Plan (and optionally apply) a ChemIllusion runtime bump from this checkout.

ChemIllusion (scottmreed/chem-art-generator) imports the mechanism runtime
in-process from a pinned checkout of this repo: ``ARG WIGGUM_RUNTIME_REF=<sha>``
in ``backend/Dockerfile.api``. Its product model is ``PRODUCT_MODEL`` in
``backend/app/services/mechanism_predictor/product_model.py`` — a constant, not a
setting, so this process is the only thing that changes it (no Railway override).

A bump is proposed only when one of these holds between the current pin and ``--ref``:

* **harness change** — ``harness_versions/``, the shared prompts/few-shots in
  ``skills/mechanistic/``, the frontier model's own lane
  (``skills/mechanistic/<call>/models/<frontier slug>/``), or the harness engine
  (``mechanistic_agent/core/`` except storage and evolution tooling, ``tools.py``,
  ``tool_schemas.py``, ``llm.py``, ``smiles_utils.py``, ``prompt_assets.py``). Lanes
  of other models, leaderboard and CLI code, docs and results do not count;
* **frontier improvement** — the frontier model's best published harness score
  (``results/runs/*.json``, non-holdout, non-baseline) on the medium or hard tier
  is higher than it was at the pin;
* **new frontier** — one model now leads both the medium and hard tiers and it is
  not the product model; the bump then also rewrites ``PRODUCT_MODEL``.

The frontier model is the new leader when there is one, else the product model.
With ``--chemillusion-dir`` and ``--apply`` it rewrites the pin (and the model);
``.github/workflows/chemillusion-runtime-bump.yml`` then opens or refreshes one
PR there. It is never auto-merged: the evidence gate still decides what ships.

Stdlib only, so the workflow needs no dependency install.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DOCKERFILE = "backend/Dockerfile.api"
PRODUCT_MODEL_FILE = "backend/app/services/mechanism_predictor/product_model.py"
LEGACY_CONFIG = "backend/app/core/config.py"  # before PRODUCT_MODEL existed

REF_LINE = re.compile(r"^(ARG WIGGUM_RUNTIME_REF=)([0-9a-fA-F]{7,40})\s*$", re.M)
PRODUCT_MODEL_LINE = re.compile(r'^(PRODUCT_MODEL = ")([^"]+)(")\s*$', re.M)
LEGACY_MODEL_LINE = re.compile(r'^(\s*MECHANISM_PREDICTOR_MODEL: str = ")([^"]+)(")\s*$', re.M)

LANE = re.compile(r"^skills/mechanistic/[^/]+/models/([^/]+)/")
HARNESS_CODE = (
    "mechanistic_agent/core/",
    "mechanistic_agent/tools.py",
    "mechanistic_agent/tool_schemas.py",
    "mechanistic_agent/llm.py",
    "mechanistic_agent/smiles_utils.py",
    "mechanistic_agent/prompt_assets.py",
)
# Under core/ but not what a run executes: storage, baselines/evals and harness-evolution tooling.
NOT_HARNESS = (
    "mechanistic_agent/core/db.py",
    "mechanistic_agent/core/archive.py",
    "mechanistic_agent/core/lane_mutator.py",
    "mechanistic_agent/core/llm_mutator.py",
    "mechanistic_agent/core/baseline_runner.py",
    "mechanistic_agent/core/micro_eval_runner.py",
    "mechanistic_agent/core/experiment_ledger.py",
    "mechanistic_agent/core/overnight_ralph.py",
    "mechanistic_agent/core/ralph_orchestrator.py",
)
HARNESS_PATHS = ("harness_versions/", "skills/mechanistic/", *HARNESS_CODE)
KIND_LABELS = {
    "harness_config": "Harness configs (`harness_versions/`)",
    "prompts_fewshots": "Shared prompts / few-shots (`skills/mechanistic/`)",
    "frontier_lane": "Frontier model's prompt / few-shot lane",
    "harness_code": "Harness engine (`mechanistic_agent/core/`, tools, schemas)",
}
LEADER_TIERS = ("medium", "hard")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def _model_key(name: str) -> str:
    """`anthropic/claude-opus-5.5`, `anthropic__claude-opus-5.5`, `claude-opus-5-5 (note)` -> `claude-opus-5-5`."""
    return str(name or "").replace("__", "/").split("/")[-1].split(" (")[0].strip().lower().replace(".", "-")


def harness_kind(path: str, frontier_keys: set[str]) -> Optional[str]:
    """Harness kind of a changed path, or None when it does not move ChemIllusion."""
    lane = LANE.match(path)
    if lane:
        return "frontier_lane" if _model_key(lane.group(1)) in frontier_keys else None
    if path.startswith("harness_versions/"):
        return "harness_config"
    if path.startswith("skills/mechanistic/"):
        return "prompts_fewshots"
    if path.startswith(HARNESS_CODE) and path not in NOT_HARNESS:
        return "harness_code"
    return None


def read_pin(dockerfile_text: str) -> str:
    match = REF_LINE.search(dockerfile_text)
    if not match:
        raise ValueError(f"no `ARG WIGGUM_RUNTIME_REF=<sha>` line in {DOCKERFILE}")
    return match.group(2)


def read_product_model(model_text: str) -> Optional[str]:
    match = PRODUCT_MODEL_LINE.search(model_text) or LEGACY_MODEL_LINE.search(model_text)
    return match.group(2) if match else None


def harness_commits(repo: Path, old: str, new: str, frontier_keys: set[str]) -> List[Dict[str, Any]]:
    """Commits in old..new that change the harness ChemIllusion runs, with their kinds."""
    log = _git(repo, "log", "--format=@@%H%x09%s", "--name-only", f"{old}..{new}", "--", *HARNESS_PATHS)
    commits: List[Dict[str, Any]] = []
    for block in log.split("@@")[1:]:
        lines = [line for line in block.splitlines() if line.strip()]
        sha, _, subject = lines[0].partition("\t")
        kinds = sorted({k for k in (harness_kind(p, frontier_keys) for p in lines[1:]) if k})
        if kinds:
            commits.append({"sha": sha, "subject": subject, "kinds": kinds})
    return commits


def _effective_model(record: Dict[str, Any]) -> str:
    origin = record.get("origin") or {}
    if origin.get("responder") == "agent-bridge" and origin.get("declared_underlying_model"):
        return str(origin["declared_underlying_model"])
    return str(record.get("model") or "")


def harness_records(repo: Path, ref: str) -> List[Dict[str, Any]]:
    """Published non-holdout harness (not baseline) records at ``ref``."""
    try:
        names = _git(repo, "ls-tree", "--name-only", ref, "results/runs/").split()
    except subprocess.CalledProcessError:
        return []
    records = []
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            record = json.loads(_git(repo, "show", f"{ref}:{name}"))
        except (subprocess.CalledProcessError, ValueError):
            continue
        if record.get("holdout") or record.get("kind") == "baseline" or not record.get("tier"):
            continue
        records.append(
            {
                "tier": str(record["tier"]),
                "model": _effective_model(record),
                "points": int((record.get("summary") or {}).get("points") or 0),
                "date": str(record.get("date") or ""),
                "file": Path(name).name,
            }
        )
    return records


def leaders(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Best record per tier (same ranking as results_publish.best_by_tier: points, then date)."""
    best: Dict[str, Dict[str, Any]] = {}
    for record in records:
        current = best.get(record["tier"])
        if current is None or (record["points"], record["date"]) > (current["points"], current["date"]):
            best[record["tier"]] = record
    return best


def best_points(records: Sequence[Dict[str, Any]], tier: str, model: str) -> Optional[int]:
    scores = [r["points"] for r in records if r["tier"] == tier and _model_key(r["model"]) == _model_key(model)]
    return max(scores) if scores else None


def catalog_model_id(repo: Path, ref: str, name: str) -> Optional[str]:
    """Catalog id in model_pricing.json at ``ref`` matching ``name``, if any."""
    try:
        catalog = json.loads(_git(repo, "show", f"{ref}:mechanistic_agent/model_pricing.json"))
    except (subprocess.CalledProcessError, ValueError):
        return None
    models = catalog.get("models", catalog) if isinstance(catalog, dict) else {}
    wanted = _model_key(name)
    return next((model_id for model_id in models if _model_key(model_id) == wanted), None)


def model_recommendation(repo: Path, ref: str, records: Sequence[Dict[str, Any]], product_model: Optional[str]) -> Optional[Dict[str, Any]]:
    """A new product model when one model leads every LEADER_TIERS tier and differs from it."""
    top = leaders(records)
    if any(tier not in top for tier in LEADER_TIERS):
        return None
    keys = {_model_key(top[tier]["model"]) for tier in LEADER_TIERS}
    if len(keys) != 1 or (product_model and keys == {_model_key(product_model)}):
        return None
    model_id = catalog_model_id(repo, ref, top[LEADER_TIERS[0]]["model"])
    if model_id is None:
        return None
    return {"model": model_id, "leaders": {tier: top[tier] for tier in LEADER_TIERS}}


def plan_bump(repo: Path, new_ref: str, dockerfile_text: str, model_text: str) -> Dict[str, Any]:
    old_ref = read_pin(dockerfile_text)
    new_sha = _git(repo, "rev-parse", new_ref).strip()
    product_model = read_product_model(model_text)
    new_records = harness_records(repo, new_sha)
    recommendation = model_recommendation(repo, new_sha, new_records, product_model)
    frontier = recommendation["model"] if recommendation else product_model
    frontier_keys = {_model_key(m) for m in (frontier, product_model) if m}

    try:
        commits = harness_commits(repo, old_ref, new_sha, frontier_keys)
        old_records = harness_records(repo, old_ref)
        pin_reachable = True
    except subprocess.CalledProcessError:
        commits, old_records, pin_reachable = [], [], False

    improvements: Dict[str, Dict[str, Optional[int]]] = {}
    if frontier:
        for tier in LEADER_TIERS:
            before, after = best_points(old_records, tier, frontier), best_points(new_records, tier, frontier)
            if after is not None and (before is None or after > before):
                improvements[tier] = {"before": before, "after": after}

    reasons = []
    if commits:
        reasons.append("harness_change")
    if improvements:
        reasons.append("frontier_improvement")
    if recommendation:
        reasons.append("new_frontier_model")
    if not pin_reachable:
        reasons.append("pin_not_ancestor")
    changed = bool(reasons) and (new_sha != old_ref or recommendation is not None)
    return {
        "old_ref": old_ref,
        "new_ref": new_sha if changed else old_ref,
        "pin_reachable": pin_reachable,
        "reasons": reasons,
        "commits": commits,
        "kinds": sorted({k for c in commits for k in c["kinds"]}),
        "product_model": product_model,
        "frontier_model": frontier,
        "frontier_improvements": improvements,
        "model_recommendation": recommendation,
        "changed": changed,
    }


def apply_plan(plan: Dict[str, Any], dockerfile_text: str, model_text: str) -> tuple[str, str]:
    dockerfile = REF_LINE.sub(lambda m: m.group(1) + plan["new_ref"], dockerfile_text, count=1)
    model = model_text
    recommendation = plan.get("model_recommendation")
    if recommendation:
        pattern = PRODUCT_MODEL_LINE if PRODUCT_MODEL_LINE.search(model_text) else LEGACY_MODEL_LINE
        model = pattern.sub(lambda m: m.group(1) + recommendation["model"] + m.group(3), model_text, count=1)
    return dockerfile, model


def pr_body(plan: Dict[str, Any], repo_slug: str = "scottmreed/professor-wiggum") -> str:
    old, new = plan["old_ref"], plan["new_ref"]
    reason_text = {
        "harness_change": "harness change",
        "frontier_improvement": "frontier-model improvement on medium/hard",
        "new_frontier_model": "new frontier model",
        "pin_not_ancestor": "current pin is not an ancestor of the new ref",
    }
    lines = [
        "Automated by professor-wiggum's `chemillusion-runtime-bump` workflow "
        "(`scripts/chemillusion_runtime_bump.py`). Review, then merge to roll the embedded mechanism runtime.",
        "",
        f"- Why: {', '.join(reason_text.get(r, r) for r in plan['reasons'])}",
        f"- `WIGGUM_RUNTIME_REF`: `{old[:12]}` → `{new[:12]}` "
        f"([compare](https://github.com/{repo_slug}/compare/{old}...{new}))",
        f"- Frontier model: `{plan.get('frontier_model')}`",
    ]
    if plan["kinds"]:
        lines += ["", "### Harness changes", ""]
        lines += [f"- {KIND_LABELS.get(kind, kind)}" for kind in plan["kinds"]]
        lines += [""]
        lines += [
            f"- [`{c['sha'][:8]}`](https://github.com/{repo_slug}/commit/{c['sha']}) {c['subject']} "
            f"({', '.join(c['kinds'])})"
            for c in plan["commits"]
        ]
    if plan.get("frontier_improvements"):
        lines += ["", "### Frontier improvements (best published harness score)", ""]
        lines += [
            f"- {tier}: {d['before'] if d['before'] is not None else '—'} → {d['after']}/1000"
            for tier, d in plan["frontier_improvements"].items()
        ]
    recommendation = plan.get("model_recommendation")
    if recommendation:
        leaders_ = recommendation["leaders"]
        lines += [
            "",
            "### Product model",
            "",
            f"`{recommendation['model']}` now leads both {' and '.join(LEADER_TIERS)} harness tiers ("
            + ", ".join(f"{tier} {leaders_[tier]['points']}/1000" for tier in LEADER_TIERS)
            + f"). This PR changes `PRODUCT_MODEL` from `{plan['product_model']}`; it is not a setting, "
            "so this is the only place the product model changes.",
        ]
    lines += [
        "",
        "### Before merging",
        "",
        "- Harness, prompt, few-shot and model changes reached Wiggum `main` only through the evidence gate "
        "(`docs/change_evidence_policy.md`); check the linked commits' PRs if in doubt.",
        "- PR Backend Checks and No New Failures re-fetch the runtime at the new ref.",
    ]
    return "\n".join(lines) + "\n"


def model_file(chem: Path) -> Path:
    path = chem / PRODUCT_MODEL_FILE
    return path if path.exists() else chem / LEGACY_CONFIG


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wiggum-dir", default=".", help="this repo's checkout (full history)")
    parser.add_argument("--ref", default="HEAD", help="Wiggum ref to roll ChemIllusion to")
    parser.add_argument("--chemillusion-dir", required=True, help="chem-art-generator checkout")
    parser.add_argument("--apply", action="store_true", help="rewrite the pin (and product model) in place")
    parser.add_argument("--body-out", help="write the PR body markdown here")
    args = parser.parse_args(argv)

    repo = Path(args.wiggum_dir).resolve()
    chem = Path(args.chemillusion_dir).resolve()
    dockerfile_path, model_path = chem / DOCKERFILE, model_file(chem)
    dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    plan = plan_bump(repo, args.ref, dockerfile_text, model_text)
    if args.apply and plan["changed"]:
        dockerfile, model = apply_plan(plan, dockerfile_text, model_text)
        dockerfile_path.write_text(dockerfile, encoding="utf-8")
        model_path.write_text(model, encoding="utf-8")
    if args.body_out:
        Path(args.body_out).write_text(pr_body(plan), encoding="utf-8")
    json.dump(plan, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
