#!/usr/bin/env python3
"""Plan (and optionally apply) a ChemIllusion runtime bump from this checkout.

ChemIllusion (scottmreed/chem-art-generator) imports the mechanism runtime
in-process from a pinned checkout of this repo: ``ARG WIGGUM_RUNTIME_REF=<sha>``
in ``backend/Dockerfile.api``. Its product model is the default of
``MECHANISM_PREDICTOR_MODEL`` in ``backend/app/core/config.py``. Nothing there
tracks Wiggum ``main``, so this script works out what a bump would carry:

* the commits between the current pin and ``--ref`` that touch what the runtime
  ships (prompts/few-shots, harness configs, the model catalog, runtime code,
  FlowER asset manifest, dependencies), grouped by kind;
* whether the published leaderboard (``results/runs/*.json``) now has one model
  leading both the medium and hard harness tiers that is not the product model
  — if so, the bump also proposes that model as the new default.

With ``--chemillusion-dir`` and ``--apply`` it rewrites those two files in a
ChemIllusion checkout; the workflow ``.github/workflows/chemillusion-runtime-bump.yml``
then opens (or refreshes) a PR there. The PR is never auto-merged: the
evidence gate (docs/change_evidence_policy.md) still decides what ships.

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
CONFIG = "backend/app/core/config.py"

REF_LINE = re.compile(r"^(ARG WIGGUM_RUNTIME_REF=)([0-9a-fA-F]{7,40})\s*$", re.M)
MODEL_LINE = re.compile(r'^(\s*MECHANISM_PREDICTOR_MODEL: str = ")([^"]+)(")\s*$', re.M)

# What the embedded runtime reads, by kind. Order matters: first match wins.
RUNTIME_PATHS: List[tuple[str, str]] = [
    ("skills/mechanistic/", "prompts_fewshots"),
    ("harness_versions/", "harness"),
    ("mechanistic_agent/model_pricing.json", "model_catalog"),
    ("mechanistic_agent/", "runtime_code"),
    ("novelty_index/manifest.json", "flower_assets"),
    ("pyproject.toml", "dependencies"),
    ("requirements.txt", "dependencies"),
]
KIND_LABELS = {
    "prompts_fewshots": "Prompts / few-shots (`skills/mechanistic/`)",
    "harness": "Harness configs (`harness_versions/`)",
    "model_catalog": "Model catalog (`model_pricing.json`)",
    "runtime_code": "Runtime code (`mechanistic_agent/`)",
    "flower_assets": "FlowER reference assets (`novelty_index/manifest.json`)",
    "dependencies": "Dependencies",
}
LEADER_TIERS = ("medium", "hard")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def kind_of(path: str) -> Optional[str]:
    for prefix, kind in RUNTIME_PATHS:
        if path == prefix or path.startswith(prefix):
            return kind
    return None


def read_pin(dockerfile_text: str) -> str:
    match = REF_LINE.search(dockerfile_text)
    if not match:
        raise ValueError(f"no `ARG WIGGUM_RUNTIME_REF=<sha>` line in {DOCKERFILE}")
    return match.group(2)


def read_product_model(config_text: str) -> Optional[str]:
    match = MODEL_LINE.search(config_text)
    return match.group(2) if match else None


def runtime_commits(repo: Path, old: str, new: str) -> List[Dict[str, Any]]:
    """Commits in old..new touching runtime paths, each with the kinds it touches."""
    prefixes = sorted({prefix for prefix, _ in RUNTIME_PATHS})
    log = _git(repo, "log", "--format=@@%H%x09%s", "--name-only", f"{old}..{new}", "--", *prefixes)
    commits: List[Dict[str, Any]] = []
    for block in log.split("@@")[1:]:
        lines = [line for line in block.splitlines() if line.strip()]
        sha, _, subject = lines[0].partition("\t")
        kinds = sorted({k for k in (kind_of(p) for p in lines[1:]) if k})
        if kinds:
            commits.append({"sha": sha, "subject": subject, "kinds": kinds})
    return commits


def _model_key(name: str) -> str:
    """`anthropic/claude-opus-5.5`, `claude-opus-5-5 (note)` -> `claude-opus-5-5`."""
    return str(name or "").split("/")[-1].split(" (")[0].strip().lower().replace(".", "-")


def _effective_model(record: Dict[str, Any]) -> str:
    origin = record.get("origin") or {}
    if origin.get("responder") == "agent-bridge" and origin.get("declared_underlying_model"):
        return str(origin["declared_underlying_model"])
    return str(record.get("model") or "")


def catalog_model_id(repo: Path, name: str) -> Optional[str]:
    """Catalog id in this checkout's model_pricing.json matching ``name``, if any."""
    catalog_path = repo / "mechanistic_agent" / "model_pricing.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    models = catalog.get("models", catalog) if isinstance(catalog, dict) else {}
    wanted = _model_key(name)
    for model_id in models if isinstance(models, dict) else []:
        if _model_key(model_id) == wanted:
            return model_id
    return None


def leaderboard_leaders(repo: Path) -> Dict[str, Dict[str, Any]]:
    """Best published harness record per tier (same ranking as results_publish.best_by_tier)."""
    best: Dict[str, Dict[str, Any]] = {}
    for path in sorted((repo / "results" / "runs").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tier = str(record.get("tier") or "")
        if record.get("holdout") or record.get("kind") == "baseline" or not tier:
            continue
        key = (int((record.get("summary") or {}).get("points") or 0), str(record.get("date") or ""))
        current = best.get(tier)
        if current is None or key > current["key"]:
            best[tier] = {"key": key, "model": _effective_model(record), "points": key[0], "file": path.name}
    return best


def model_recommendation(repo: Path, product_model: Optional[str]) -> Optional[Dict[str, Any]]:
    """A new product default when one model leads every LEADER_TIERS tier and differs from it."""
    leaders = leaderboard_leaders(repo)
    if any(tier not in leaders for tier in LEADER_TIERS):
        return None
    keys = {_model_key(leaders[tier]["model"]) for tier in LEADER_TIERS}
    if len(keys) != 1 or (product_model and keys == {_model_key(product_model)}):
        return None
    model_id = catalog_model_id(repo, leaders[LEADER_TIERS[0]]["model"])
    if model_id is None:
        return None
    return {"model": model_id, "leaders": {tier: leaders[tier] for tier in LEADER_TIERS}}


def plan_bump(repo: Path, new_ref: str, dockerfile_text: str, config_text: str) -> Dict[str, Any]:
    old_ref = read_pin(dockerfile_text)
    new_sha = _git(repo, "rev-parse", new_ref).strip()
    product_model = read_product_model(config_text)
    try:
        commits = runtime_commits(repo, old_ref, new_sha)
        pin_reachable = True
    except subprocess.CalledProcessError:
        commits, pin_reachable = [], False
    recommendation = model_recommendation(repo, product_model)
    bump_ref = bool(commits) or not pin_reachable or recommendation is not None
    return {
        "old_ref": old_ref,
        "new_ref": new_sha if bump_ref else old_ref,
        "pin_reachable": pin_reachable,
        "commits": commits,
        "kinds": sorted({k for c in commits for k in c["kinds"]}),
        "product_model": product_model,
        "model_recommendation": recommendation,
        "changed": bump_ref and (new_sha != old_ref or recommendation is not None),
    }


def apply_plan(plan: Dict[str, Any], dockerfile_text: str, config_text: str) -> tuple[str, str]:
    dockerfile = REF_LINE.sub(lambda m: m.group(1) + plan["new_ref"], dockerfile_text, count=1)
    config = config_text
    recommendation = plan.get("model_recommendation")
    if recommendation:
        config = MODEL_LINE.sub(lambda m: m.group(1) + recommendation["model"] + m.group(3), config_text, count=1)
    return dockerfile, config


def pr_body(plan: Dict[str, Any], repo_slug: str = "scottmreed/professor-wiggum") -> str:
    old, new = plan["old_ref"], plan["new_ref"]
    lines = [
        "Automated by professor-wiggum's `chemillusion-runtime-bump` workflow "
        "(`scripts/chemillusion_runtime_bump.py`). Review, then merge to roll the embedded mechanism runtime.",
        "",
        f"- `WIGGUM_RUNTIME_REF`: `{old[:12]}` → `{new[:12]}` "
        f"([compare](https://github.com/{repo_slug}/compare/{old}...{new}))",
    ]
    if not plan.get("pin_reachable", True):
        lines.append("- ⚠️ The current pin is not an ancestor of the new ref; review the compare view.")
    if plan["kinds"]:
        lines += ["", "### What changes in the runtime", ""]
        lines += [f"- {KIND_LABELS.get(kind, kind)}" for kind in plan["kinds"]]
        lines += ["", "### Commits", ""]
        lines += [
            f"- [`{c['sha'][:8]}`](https://github.com/{repo_slug}/commit/{c['sha']}) {c['subject']} "
            f"({', '.join(c['kinds'])})"
            for c in plan["commits"]
        ]
    recommendation = plan.get("model_recommendation")
    if recommendation:
        leaders = recommendation["leaders"]
        lines += [
            "",
            "### Product model",
            "",
            f"The published Wiggum leaderboard has one model leading both "
            f"{' and '.join(LEADER_TIERS)} harness tiers: `{recommendation['model']}` ("
            + ", ".join(f"{tier} {leaders[tier]['points']}/1000" for tier in LEADER_TIERS)
            + f"). This PR changes the `MECHANISM_PREDICTOR_MODEL` default from `{plan['product_model']}`. "
            "A Railway variable of the same name on `spirited-liberation` overrides this default — update it too if set.",
        ]
    lines += [
        "",
        "### Before merging",
        "",
        "- Prompt, few-shot, model and harness changes reached Wiggum `main` only through the evidence gate "
        "(`docs/change_evidence_policy.md`); check the linked commits' PRs if in doubt.",
        "- PR Backend Checks and No New Failures re-fetch the runtime at the new ref.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wiggum-dir", default=".", help="this repo's checkout (full history)")
    parser.add_argument("--ref", default="HEAD", help="Wiggum ref to roll ChemIllusion to")
    parser.add_argument("--chemillusion-dir", required=True, help="chem-art-generator checkout")
    parser.add_argument("--apply", action="store_true", help="rewrite the pin (and model default) in place")
    parser.add_argument("--body-out", help="write the PR body markdown here")
    args = parser.parse_args(argv)

    repo = Path(args.wiggum_dir).resolve()
    chem = Path(args.chemillusion_dir).resolve()
    dockerfile_text = (chem / DOCKERFILE).read_text(encoding="utf-8")
    config_text = (chem / CONFIG).read_text(encoding="utf-8")
    plan = plan_bump(repo, args.ref, dockerfile_text, config_text)
    if args.apply and plan["changed"]:
        dockerfile, config = apply_plan(plan, dockerfile_text, config_text)
        (chem / DOCKERFILE).write_text(dockerfile, encoding="utf-8")
        (chem / CONFIG).write_text(config, encoding="utf-8")
    if args.body_out:
        Path(args.body_out).write_text(pr_body(plan), encoding="utf-8")
    json.dump(plan, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
