"""
Clawdiators Docker validator test runner.

Runs the same single-shot model call as clawdiators_test_run.py, then validates
each reaction step against the local Docker mechanistic validator. The validator
checks SMILES validity, atom balance, and charge balance — but has NO ground truth.

This is a chemical plausibility check, not a scoring check. Use it to verify
that your submission is chemically sound before submitting to the arena.

Usage:
    # 1. Start the validator (once):
    docker run -d -p 8080:8080 clawdiators/mechanistic-validator:1.0

    # 2. Run the validation:
    python local_contributions/clawdiators_docker_validate.py
    python local_contributions/clawdiators_docker_validate.py --model gpt-4o
    python local_contributions/clawdiators_docker_validate.py --validator-url http://localhost:8080

    # 3. Also score locally:
    python local_contributions/clawdiators_docker_validate.py --score

    # 4. Update leaderboard after scoring:
    python local_contributions/clawdiators_docker_validate.py --score --update-leaderboard

Output:
    Reaction-by-reaction chemical validity table, then score summary if --score.
    Outputs saved to: local_contributions/runs/<timestamp>_<model>_docker/

Requires: OPENAI_API_KEY, Docker running mechanistic-validator:1.0, pip install openai python-dotenv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

REACTIONS_DEFAULT = REPO_ROOT / "clawdiators-test" / "easy" / "test_reactions.json"
ARENA_REACTIONS   = REPO_ROOT / "clawdiators-submission" / "easy" / "workspace_reactions.json"
GROUND_TRUTH      = REPO_ROOT / "clawdiators-test" / "easy" / "test_ground_truth.json"
SCORING_SCRIPT    = REPO_ROOT / "clawdiators-test" / "scoring" / "score_submission.py"
RUNS_DIR          = REPO_ROOT / "local_contributions" / "runs"
LEADERBOARD_MD    = REPO_ROOT / "LEADERBOARD.md"

DEFAULT_VALIDATOR = "http://localhost:8080"


# ── Import shared prompt/call logic ───────────────────────────────────

def _import_test_runner():
    """Import shared functions from clawdiators_test_run.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    import clawdiators_test_run as tr
    return tr


# ── Docker validator client ────────────────────────────────────────────

def check_validator(base_url: str) -> bool:
    """Return True if the validator is reachable."""
    try:
        req = urllib.request.urlopen(f"{base_url}/health", timeout=3)
        return req.status == 200
    except Exception:
        # Try a no-op canonicalize call as fallback health check
        try:
            data = json.dumps({"smiles": ["C"]}).encode()
            req = urllib.request.Request(
                f"{base_url}/canonicalize",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=3)
            return True
        except Exception:
            return False


def canonicalize_via_docker(smiles_list: list[str], base_url: str) -> dict:
    """POST to /canonicalize, return response dict."""
    payload = json.dumps({"smiles": smiles_list}).encode()
    req = urllib.request.Request(
        f"{base_url}/canonicalize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def validate_step(from_smiles: list[str], to_smiles: list[str],
                  step_type: str, base_url: str) -> dict:
    """POST to /validate for a single reaction step."""
    payload = json.dumps({
        "steps": [
            {
                "from_smiles": from_smiles,
                "to_smiles": to_smiles,
                "step_type": step_type,
            }
        ]
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/validate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("results", [{}])[0]
    except Exception as e:
        return {"valid": None, "error": str(e)}


# ── Step type inference ────────────────────────────────────────────────

def infer_step_type(electron_pushes: list[str]) -> str:
    """
    Infer a step_type string for the validator from electron push types.
    The validator accepts: substitution, addition, elimination, oxidation,
    cycloaddition, proton_transfer, other.
    """
    types = set()
    for ep in electron_pushes:
        if ep.startswith("lp:"):
            types.add("lp")
        elif ep.startswith("sigma:"):
            types.add("sigma")
        elif ep.startswith("pi:"):
            types.add("pi")

    if "lp" in types and "sigma" in types and "pi" not in types:
        return "substitution"
    if "pi" in types and "sigma" in types:
        return "cycloaddition"
    if "lp" in types and "pi" not in types and "sigma" not in types:
        return "oxidation"
    if "lp" in types and "sigma" in types:
        return "proton_transfer"
    return "other"


# ── Validate a full submission ─────────────────────────────────────────

def validate_submission(
    reactions: list[dict],
    submission: dict,
    base_url: str,
) -> list[dict]:
    """
    For each reaction, validate each submitted step via the Docker validator.
    Returns a list of per-reaction validation results.
    """
    answer = submission.get("answer", submission)
    submitted_products = answer.get("final_products", [])
    submitted_steps = answer.get("steps", [])

    results = []

    for i, rxn in enumerate(reactions):
        from_state = rxn.get("starting_materials", [])
        rxn_result = {
            "reaction_index": i,
            "id": rxn.get("id", f"rxn-{i}"),
            "steps": [],
            "all_valid": True,
            "canonicalized_products": None,
        }

        rxn_steps = submitted_steps[i] if i < len(submitted_steps) else []
        if not isinstance(rxn_steps, list):
            rxn_steps = []

        for j, step in enumerate(rxn_steps):
            if not isinstance(step, dict):
                rxn_result["steps"].append({"step": j, "error": "not a dict"})
                rxn_result["all_valid"] = False
                continue

            raw_state = step.get("resulting_state", [])
            # Model may return a dot-joined string instead of a list
            if isinstance(raw_state, str):
                to_state = [s for s in raw_state.split(".") if s]
            else:
                to_state = list(raw_state) if raw_state else []
            ep = step.get("electron_pushes", [])
            step_type = infer_step_type(ep)

            vr = validate_step(from_state, to_state, step_type, base_url)
            rxn_result["steps"].append({
                "step": j,
                "from": from_state,
                "to": to_state,
                "step_type": step_type,
                "valid": vr.get("valid"),
                "atom_balance": vr.get("atom_balance"),
                "charge_balance": vr.get("charge_balance"),
                "error": vr.get("error"),
            })
            if not vr.get("valid"):
                rxn_result["all_valid"] = False
            from_state = to_state  # next step starts where this one ended

        # Canonicalize the final products via docker
        final_smi = submitted_products[i] if i < len(submitted_products) else ""
        if final_smi:
            try:
                canon = canonicalize_via_docker(final_smi.split("."), base_url)
                rxn_result["canonicalized_products"] = canon.get("canonical", [])
            except Exception as e:
                rxn_result["canonicalized_products"] = f"error: {e}"

        results.append(rxn_result)

    return results


# ── Print validation report ────────────────────────────────────────────

def print_validation_report(validation_results: list[dict], base_url: str) -> None:
    sep = "─" * 64
    print(sep)
    print(f"  Docker Validator: {base_url}")
    print(sep)

    n_rxn = len(validation_results)
    n_valid = sum(1 for r in validation_results if r.get("all_valid"))
    n_steps_total = sum(len(r["steps"]) for r in validation_results)
    n_steps_valid = sum(
        1 for r in validation_results
        for s in r["steps"]
        if s.get("valid") is True
    )

    for r in validation_results:
        mark = "✓" if r.get("all_valid") else "✗"
        n_s = len(r["steps"])
        n_sv = sum(1 for s in r["steps"] if s.get("valid") is True)
        print(f"  [{mark}] rxn {r['reaction_index']} ({r['id']})  — {n_sv}/{n_s} steps valid")
        for s in r["steps"]:
            v = s.get("valid")
            sb = s.get("atom_balance")
            cb = s.get("charge_balance")
            err = s.get("error")
            if v is None:
                print(f"        step {s['step']}: ERROR — {err}")
            elif v:
                print(f"        step {s['step']}: valid  (atoms={sb}, charge={cb})")
            else:
                print(f"        step {s['step']}: INVALID  (atoms={sb}, charge={cb})")
                if s.get("to"):
                    print(f"          from: {s['from']}")
                    print(f"          to  : {s['to']}")

    print(sep)
    print(f"  Reactions all-valid: {n_valid}/{n_rxn}")
    print(f"  Steps valid        : {n_steps_valid}/{n_steps_total}")
    print(sep)


# ── Leaderboard update ─────────────────────────────────────────────────

def update_leaderboard(result: dict, model: str, elapsed: float, reactions_file: str,
                       run_dir: Path, n_steps_valid: int = None,
                       n_steps_total: int = None) -> None:
    """Append a row to LEADERBOARD.md Zero-Shot Baselines table."""
    if "error" in result or not LEADERBOARD_MD.exists():
        return

    s = result.get("scores", {})
    n_correct = result.get("products_correct", "?")
    n_total = result.get("n_reactions", 10)
    push_ratio = s.get("electron_push_quality", {}).get("ratio", 0)
    total = result.get("total", "?")
    outcome = result.get("outcome", "?")
    test_set = Path(reactions_file).name
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_link = run_dir.name if run_dir else "—"
    docker_note = f"docker-validated steps={n_steps_valid}/{n_steps_total}" if n_steps_valid is not None else "docker-validated"

    new_row = (
        f"| {date_str} | `{model}` | {test_set} "
        f"| {n_correct}/{n_total} "
        f"| {push_ratio:.2f} "
        f"| {total} "
        f"| **{outcome}** "
        f"| {docker_note}, run={run_link} |"
    )

    text = LEADERBOARD_MD.read_text()
    placeholder = "| — | *(awaiting first run)*"
    if placeholder in text:
        lines = text.splitlines()
        new_lines = []
        replaced = False
        for line in lines:
            if not replaced and placeholder in line:
                new_lines.append(new_row)
                replaced = True
            else:
                new_lines.append(line)
        LEADERBOARD_MD.write_text("\n".join(new_lines) + "\n")
    else:
        marker = "| Date | Model | Test Set | Products Correct | Push Quality | Total Score | Predicted Outcome | Notes |"
        if marker in text:
            idx = text.index(marker)
            after_header = text.index("\n", text.index("\n", idx) + 1) + 1
            text = text[:after_header] + new_row + "\n" + text[after_header:]
            LEADERBOARD_MD.write_text(text)
        else:
            print("Warning: could not find leaderboard table to update.", file=sys.stderr)
            return

    print(f"\n  Leaderboard updated: {LEADERBOARD_MD.name}")
    print(f"    → {new_row}")


# ── Score locally ──────────────────────────────────────────────────────

def score_locally(submission: dict, ground_truth_path: Path) -> dict:
    scorer_path = SCORING_SCRIPT.parent
    sys.path.insert(0, str(scorer_path))
    try:
        from score_submission import score
    except ImportError:
        return {"error": "scorer not found"}
    if not ground_truth_path.exists():
        return {"error": f"ground truth not found: {ground_truth_path}"}
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    return score(submission, ground_truth)


def format_score_summary(result: dict, model: str, elapsed: float) -> str:
    if "error" in result:
        return f"Scoring error: {result['error']}"
    s = result.get("scores", {})
    sep = "=" * 60
    lines = [
        sep,
        f"  Score (via local ground truth) — {model}",
        sep,
        f"  Product Accuracy     : {result.get('products_correct', '?'):2}/{result.get('n_reactions', 10)} correct"
          f"  → {s.get('product_accuracy', {}).get('points', '?'):3} pts  (30%)",
        f"  Pathway Coverage     : avg {s.get('pathway_coverage', {}).get('ratio', 0):.3f}"
          f"           → {s.get('pathway_coverage', {}).get('points', '?'):3} pts  (30%)",
        f"  Electron Push Quality: avg {s.get('electron_push_quality', {}).get('ratio', 0):.3f}"
          f"           → {s.get('electron_push_quality', {}).get('points', '?'):3} pts  (20%)",
        f"  Speed                : local mode (full)"
          f"    → {s.get('speed', {}).get('points', '?'):3} pts  (10%)",
        f"  Methodology          : {'present' if s.get('methodology', {}).get('ratio', 0) == 1.0 else 'MISSING'}"
          f"              → {s.get('methodology', {}).get('points', '?'):3} pts  (10%)",
        sep,
        f"  TOTAL            : {result.get('total', '?'):4} / 1000",
        f"  PREDICTED OUTCOME: {result.get('outcome', '?')}  (win ≥700 / draw 400-699 / loss <400)",
        sep,
    ]
    return "\n".join(lines)


# ── Save outputs ───────────────────────────────────────────────────────

def save_run(run_dir: Path, user_prompt: str, raw: str, submission: dict,
             validation_results: list, score_result: dict, summary: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt_user.txt").write_text(user_prompt)
    (run_dir / "raw_response.txt").write_text(raw)
    (run_dir / "submission.json").write_text(json.dumps(submission, indent=2))
    (run_dir / "docker_validation.json").write_text(json.dumps(validation_results, indent=2))
    if score_result:
        (run_dir / "score_report.json").write_text(json.dumps(score_result, indent=2))
        (run_dir / "score_summary.txt").write_text(summary)
    print(f"\nOutputs saved to: {run_dir}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clawdiators Docker validator test runner — runs model and validates steps via Docker."
    )
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--reactions", type=Path, default=REACTIONS_DEFAULT,
                        help="Path to reactions JSON")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH,
                        help="Path to ground truth JSON (for local scoring)")
    parser.add_argument("--validator-url", default=DEFAULT_VALIDATOR,
                        help=f"Validator base URL (default: {DEFAULT_VALIDATOR})")
    parser.add_argument("--score", action="store_true",
                        help="Also score against local ground truth after validation")
    parser.add_argument("--update-leaderboard", action="store_true",
                        help="Append result row to LEADERBOARD.md (requires --score)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt only, no API call")
    args = parser.parse_args()

    # Check validator is up (skip for dry-run)
    if not args.dry_run:
        print(f"Checking validator at {args.validator_url}... ", end="", flush=True)
        if not check_validator(args.validator_url):
            print("NOT REACHABLE")
            print(f"\nError: Docker validator is not running at {args.validator_url}", file=sys.stderr)
            print("Start it with:", file=sys.stderr)
            print("  docker run -d -p 8080:8080 clawdiators/mechanistic-validator:1.0", file=sys.stderr)
            sys.exit(1)
        print("OK")

    # Load reactions
    if not args.reactions.exists():
        print(f"Error: reactions file not found: {args.reactions}", file=sys.stderr)
        sys.exit(1)

    with open(args.reactions) as f:
        data = json.load(f)
    reactions = data.get("reactions", data) if isinstance(data, dict) else data
    print(f"Loaded {len(reactions)} reactions from {args.reactions.name}")

    # Import shared prompt/API functions from clawdiators_test_run
    tr = _import_test_runner()
    user_prompt = tr.build_user_prompt(reactions)
    raw_response, elapsed = tr.call_model(args.model, tr.SYSTEM_PROMPT, user_prompt, args.dry_run)

    if args.dry_run:
        return

    submission = tr.parse_submission(raw_response)

    # Docker validation
    print("\nValidating steps via Docker...\n")
    validation_results = validate_submission(reactions, submission, args.validator_url)
    print_validation_report(validation_results, args.validator_url)

    n_steps_total = sum(len(r["steps"]) for r in validation_results)
    n_steps_valid = sum(1 for r in validation_results for s in r["steps"] if s.get("valid") is True)

    # Optional local scoring
    score_result = {}
    summary = ""
    if args.score and args.ground_truth.exists():
        score_result = score_locally(submission, args.ground_truth)
        summary = format_score_summary(score_result, args.model, elapsed)
        print("\n" + summary)
    elif args.score:
        print(f"\nNote: ground truth not found at {args.ground_truth} — skipping score.", file=sys.stderr)

    # Save run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{args.model.replace('/', '_').replace('-', '_')}_docker"
    run_dir = RUNS_DIR / run_name
    save_run(run_dir, user_prompt, raw_response, submission,
             validation_results, score_result, summary)

    # Optional leaderboard update
    if args.update_leaderboard and args.score and score_result:
        update_leaderboard(
            score_result, args.model, elapsed, str(args.reactions), run_dir,
            n_steps_valid=n_steps_valid, n_steps_total=n_steps_total,
        )


if __name__ == "__main__":
    main()
