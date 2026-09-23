"""
Clawdiators single-shot test runner.

Simulates a zero-shot agent competing in the mechanistic-easy challenge.
Sends the 10 reactions to an OpenAI model (default: gpt-4o-mini) and scores
the response against local ground truth using the same rubric as the arena.

Usage:
    python local_contributions/clawdiators_test_run.py
    python local_contributions/clawdiators_test_run.py --model gpt-4o
    python local_contributions/clawdiators_test_run.py --reactions clawdiators-submission/easy/workspace_reactions.json
    python local_contributions/clawdiators_test_run.py --dry-run   # print prompt only, no API call

Outputs are saved to: local_contributions/runs/<timestamp>/
  - prompt.txt              full prompt sent to the model
  - raw_response.txt        model's raw output
  - submission.json         parsed submission (final_products, intermediates, methodology)
  - score_report.json       full scoring breakdown
  - score_summary.txt       human-readable score table

Requires: OPENAI_API_KEY env var (or .env file in repo root)
          pip install openai python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path

from mechanistic_agent.core.db import RunStore
from mechanistic_agent.data_paths import db_path as resolve_db_path, local_contributions_runs_dir
from mechanistic_agent.eval_set_resolution import (
    EvalSetResolutionError,
    case_ids_hash,
    resolve_eval_set,
)

# ── Load .env from repo root ──────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # dotenv optional

# ── Paths ─────────────────────────────────────────────────────────────
REACTIONS_DEFAULT = REPO_ROOT / "clawdiators-test" / "easy" / "test_reactions.json"
ARENA_REACTIONS   = REPO_ROOT / "clawdiators-submission" / "easy" / "workspace_reactions.json"
GROUND_TRUTH      = REPO_ROOT / "clawdiators-test" / "easy" / "test_ground_truth.json"
SCORING_SCRIPT    = REPO_ROOT / "clawdiators-test" / "scoring" / "score_submission.py"
CHALLENGE_MD      = REPO_ROOT / "clawdiators-submission" / "easy" / "CHALLENGE.md"
WORKED_EXAMPLE    = REPO_ROOT / "clawdiators-submission" / "easy" / "worked_example.json"
RUNS_DIR          = local_contributions_runs_dir(REPO_ROOT)
LEADERBOARD_MD    = REPO_ROOT / "LEADERBOARD.md"


# ── Build prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert organic chemist specializing in reaction mechanism prediction.
You will be given 10 organic reactions with starting materials and target products in SMILES notation.
Each reaction includes an `n_steps` field telling you whether it is a 1-step (concerted) or 2-step mechanism.

For each reaction, predict:
1. The final product SMILES (usually identical to the given target_products, but expressed correctly)
2. The mechanistic steps — for each step: the state of the system after the step (SMILES) and the electron pushes
3. A methodology description

## Reaction types and electron pushes

1-step concerted mechanisms:
- SN2: nucleophile N or P lone pair attacks electrophilic C, halide departs. 2 pushes: lp + sigma.
  Electron pushes: ["lp:N>C_electrophile", "sigma:C-X>X"]
- N-oxidation: lone pair on aromatic N attacks electrophilic O of peracid. 1 push: lp.
  Electron pushes: ["lp:N>O"]
- Diels-Alder [4+2]: 3 concerted pushes: 2 pi + 1 sigma.
  Electron pushes: ["pi:C1-C2>new_sigma", "pi:C3-C4>new_sigma", "sigma:dienophile_pi>remaining_pi"]
- Ene reaction: 3 concerted pushes: pi + sigma + pi.
  Electron pushes: ["pi:C=C>new_bond", "sigma:C-H>O", "pi:C=O>C"]

2-step mechanisms:
- Epoxide ring opening by amine: Step 1 (SN2-like): N lone pair attacks C, ring C-O breaks (lp + sigma).
  Step 2 (proton transfer): O lone pair accepts H from N+ (lp + sigma).
- Arbuzov reaction: Step 1: P lone pair attacks C of alkyl halide (lp + sigma).
  Step 2: halide attacks another C on P-O (lp + sigma).

## Electron push notation
- "lp:SOURCE>TARGET" = lone pair from SOURCE atom flows toward TARGET (forms a bond)
- "sigma:A-B>C" = sigma bond A-B electrons flow toward C (bond breaks)
- "pi:A-B>C" = pi bond A-B electrons flow toward C
Atom indices: use your best guess based on SMILES atom order. Partial credit is given for correct push TYPES (lp/sigma/pi) even if atom indices are wrong.

You MUST respond with ONLY a valid JSON object matching the exact format below. No prose, no markdown, no explanation outside the JSON.

Required format:
{
  "answer": {
    "final_products": ["SMILES_0", "SMILES_1", "SMILES_2", "SMILES_3", "SMILES_4",
                       "SMILES_5", "SMILES_6", "SMILES_7", "SMILES_8", "SMILES_9"],
    "steps": [
      [{"resulting_state": ["prod_SMILES"], "electron_pushes": ["lp:1>2", "sigma:2-3>3"]}],
      ...10 entries, one per reaction...
    ],
    "methodology": "Your reasoning here..."
  }
}

Rules:
- final_products must have exactly 10 entries (one per reaction, in order)
- steps must have exactly 10 entries (arrays of step objects, one per reaction)
  - For 1-step reactions: steps[i] has 1 object where resulting_state = final products
  - For 2-step reactions: steps[i] has 2 objects where steps[i][0].resulting_state = intermediate
- Multi-species products: join with "." (e.g. "C[N+](C)(C)C.[I-]") or use list
- methodology must be a non-empty string"""


def build_user_prompt(reactions: list[dict]) -> str:
    lines = [
        "Here are the 10 reactions. Predict the mechanism for each.",
        "",
    ]
    for i, rxn in enumerate(reactions):
        rxn_id = rxn.get("id", f"reaction-{i}")
        sm = ", ".join(rxn.get("starting_materials", []))
        tp = ", ".join(rxn.get("target_products", []))
        cond = rxn.get("conditions", "not specified")
        mtype = rxn.get("mechanism_type", "")
        hint = rxn.get("hint", "")
        n_steps = rxn.get("n_steps", "")
        lines.append(f"Reaction {i} (id: {rxn_id})")
        lines.append(f"  Starting materials : {sm}")
        lines.append(f"  Target products    : {tp}")
        lines.append(f"  Conditions         : {cond}")
        if n_steps:
            lines.append(f"  Steps              : {n_steps}")
        if mtype:
            lines.append(f"  Mechanism type     : {mtype}")
        if hint:
            lines.append(f"  Hint               : {hint}")
        lines.append("")
    lines.append("Respond with ONLY the JSON object. No other text.")
    return "\n".join(lines)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _infer_n_steps_from_case(case: dict) -> int | None:
    expected = case.get("expected") or {}
    if not isinstance(expected, dict):
        return None
    direct = expected.get("n_mechanistic_steps")
    if isinstance(direct, (int, float)):
        return int(direct)
    known = expected.get("known_mechanism") or {}
    if isinstance(known, dict):
        min_steps = known.get("min_steps")
        if isinstance(min_steps, (int, float)):
            return int(min_steps)
        steps = known.get("steps")
        if isinstance(steps, list):
            return len(steps)
    verified = expected.get("verified_mechanism") or {}
    if isinstance(verified, dict):
        steps = verified.get("steps")
        if isinstance(steps, list):
            return len(steps)
    return None


def _reactions_from_eval_cases(cases: list[dict]) -> list[dict]:
    reactions: list[dict] = []
    for case in cases:
        case_id = str(case.get("case_id") or "").strip()
        input_payload = case.get("input") or {}
        starting = [str(item) for item in input_payload.get("starting_materials", [])]
        products = [str(item) for item in input_payload.get("products", [])]
        if not case_id or not starting or not products:
            continue
        n_steps = _infer_n_steps_from_case(case)
        reaction = {
            "id": case_id,
            "starting_materials": starting,
            "target_products": products,
            "conditions": "from_eval_set",
            "n_steps": n_steps if n_steps is not None else "",
        }
        reactions.append(reaction)
    return reactions


# ── OpenAI call ───────────────────────────────────────────────────────

def call_model(
    model: str,
    system: str,
    user: str,
    dry_run: bool,
    *,
    llm_seed: int | None,
    llm_temperature: float,
) -> tuple[str, float]:
    """Call OpenAI chat completion. Returns (response_text, elapsed_seconds)."""
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN — prompt preview:")
        print("="*60)
        print("SYSTEM:", system[:300], "...")
        print("\nUSER (first 500 chars):", user[:500], "...")
        print("="*60 + "\n")
        return '{"answer": {"final_products": [], "intermediates": [], "methodology": "dry run"}}', 0.0

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"Calling {model}... ", end="", flush=True)
    t0 = time.time()

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": llm_temperature,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    if llm_seed is not None:
        request_kwargs["seed"] = int(llm_seed)

    response = client.chat.completions.create(
        **request_kwargs,
    )

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")
    return response.choices[0].message.content or "", elapsed


# ── Parse submission ──────────────────────────────────────────────────

def parse_submission(raw: str) -> dict:
    """Parse model response into submission dict. Returns best-effort dict."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parse error: {e}", file=sys.stderr)
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r"```(?:json)?\n?(.*?)```", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                parsed = {"answer": {"final_products": [], "intermediates": [], "methodology": "parse error"}}
        else:
            parsed = {"answer": {"final_products": [], "intermediates": [], "methodology": "parse error"}}

    # Normalize: accept both wrapped {"answer": {...}} and bare {final_products: ...}
    if "answer" in parsed:
        return parsed
    if "final_products" in parsed:
        return {"answer": parsed}
    return {"answer": {"final_products": [], "steps": [[] for _ in range(10)], "methodology": str(parsed)}}


# ── Local scoring ─────────────────────────────────────────────────────

def score_locally(submission: dict, ground_truth_path: Path) -> dict:
    """Run score_submission.py logic inline (no subprocess needed)."""
    # Import scorer from clawdiators-test
    scorer_path = SCORING_SCRIPT.parent
    sys.path.insert(0, str(scorer_path))
    try:
        from score_submission import score
    except ImportError:
        print(f"Warning: could not import scorer from {scorer_path}", file=sys.stderr)
        return {"error": "scorer not found"}

    if not ground_truth_path.exists():
        return {"error": f"ground truth not found: {ground_truth_path}"}

    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    return score(submission, ground_truth)


def format_score_summary(result: dict, model: str, elapsed: float, reactions_file: str) -> str:
    if "error" in result:
        return f"Scoring error: {result['error']}"

    s = result.get("scores", {})
    lines = [
        "=" * 60,
        f"  Clawdiators Test Run — {model}",
        f"  Reactions: {Path(reactions_file).name}",
        f"  Elapsed  : {elapsed:.1f}s",
        f"  RDKit    : {'available' if result.get('rdkit_available') else 'NOT available (less accurate)'}",
        "=" * 60,
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
        "=" * 60,
        f"  TOTAL            : {result.get('total', '?'):4} / 1000",
        f"  PREDICTED OUTCOME: {result.get('outcome', '?')}  (win ≥700 / draw 400-699 / loss <400)",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── Leaderboard update ────────────────────────────────────────────────

def update_leaderboard(result: dict, model: str, elapsed: float, reactions_file: str,
                       run_dir: Path, harness: str = "zero-shot") -> None:
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
    rdkit = "rdkit" if result.get("rdkit_available") else "no-rdkit"

    new_row = (
        f"| {date_str} | `{model}` | {test_set} "
        f"| {n_correct}/{n_total} "
        f"| {push_ratio:.2f} "
        f"| {total} "
        f"| **{outcome}** "
        f"| {harness}, {rdkit}, run={run_link} |"
    )

    text = LEADERBOARD_MD.read_text()
    # Find the awaiting placeholder row and replace it, or append after the header row
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
        # Find the table header row and insert after the separator
        marker = "| Date | Model | Test Set | Products Correct | Push Quality | Total Score | Predicted Outcome | Notes |"
        if marker in text:
            idx = text.index(marker)
            # Skip past header + separator row
            after_header = text.index("\n", text.index("\n", idx) + 1) + 1
            text = text[:after_header] + new_row + "\n" + text[after_header:]
            LEADERBOARD_MD.write_text(text)
        else:
            print("Warning: could not find leaderboard table to update.", file=sys.stderr)
            return

    print(f"  Leaderboard updated: {LEADERBOARD_MD.name}")
    print(f"    → {new_row}")


# ── Save outputs ──────────────────────────────────────────────────────

def save_run(run_dir: Path, prompt: str, user_prompt: str, raw: str,
             submission: dict, score_result: dict, summary: str, metadata: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt_system.txt").write_text(prompt)
    (run_dir / "prompt_user.txt").write_text(user_prompt)
    (run_dir / "raw_response.txt").write_text(raw)
    (run_dir / "submission.json").write_text(json.dumps(submission, indent=2))
    (run_dir / "score_report.json").write_text(json.dumps(score_result, indent=2))
    (run_dir / "score_summary.txt").write_text(summary)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nOutputs saved to: {run_dir}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot clawdiators single-shot test runner."
    )
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--eval-set-id", default=None,
                        help="Resolve reactions from DB eval set id instead of a reactions JSON file.")
    parser.add_argument("--official-holdout", action="store_true",
                        help="Default to the latest purpose=leaderboard_holdout eval set.")
    parser.add_argument("--allow-non-holdout", action="store_true",
                        help="Allow non-holdout eval_set_id when --official-holdout is set.")
    parser.add_argument("--reactions", type=Path, default=REACTIONS_DEFAULT,
                        help=f"Path to reactions JSON (default: clawdiators-test/easy/test_reactions.json)\n"
                             f"Use --reactions {ARENA_REACTIONS.relative_to(REPO_ROOT)} for the actual eval set.")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH,
                        help="Path to ground truth JSON")
    parser.add_argument("--llm-seed", type=int, default=42,
                        help="Deterministic seed hint for providers that support it.")
    parser.add_argument("--llm-temperature", type=float, default=0.0,
                        help="Sampling temperature for the one-shot call.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt and exit without calling the API")
    parser.add_argument("--arena-reactions", action="store_true",
                        help=f"Use the actual eval reactions from clawdiators-submission/easy/workspace_reactions.json "
                             f"(no ground truth available for scoring)")
    parser.add_argument("--update-leaderboard", action="store_true",
                        help="Append result row to LEADERBOARD.md after scoring")
    args = parser.parse_args()

    resolved_eval_set_id = None
    resolved_eval_set_purpose = None
    resolved_case_ids_hash = None
    reactions_source_label = ""
    if args.eval_set_id or args.official_holdout:
        store = RunStore(resolve_db_path(REPO_ROOT))
        try:
            resolved = resolve_eval_set(
                store=store,
                requested_eval_set_id=args.eval_set_id,
                require_purpose=(
                    None if args.allow_non_holdout else ("leaderboard_holdout" if args.official_holdout else None)
                ),
                default_purpose=("leaderboard_holdout" if args.official_holdout and not args.eval_set_id else None),
            )
        except EvalSetResolutionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        reactions = _reactions_from_eval_cases(list(resolved.cases))
        if not reactions:
            print("Error: no valid reactions resolved from eval set", file=sys.stderr)
            sys.exit(1)
        resolved_eval_set_id = resolved.eval_set_id
        resolved_eval_set_purpose = resolved.purpose
        resolved_case_ids_hash = case_ids_hash([str(r.get("id") or "") for r in reactions])
        reactions_source_label = f"eval_set:{resolved_eval_set_id} ({resolved_eval_set_purpose})"
        ground_truth_path = None
        print(
            f"Loaded {len(reactions)} reactions from {reactions_source_label} "
            f"case_ids_hash={resolved_case_ids_hash}"
        )
    elif args.arena_reactions:
        reactions_path = ARENA_REACTIONS
        ground_truth_path = None
        print("Note: using arena eval reactions — no ground truth available for local scoring.")
    else:
        reactions_path = args.reactions
        ground_truth_path = args.ground_truth

    # Load reactions (file mode)
    if not (args.eval_set_id or args.official_holdout):
        if not reactions_path.exists():
            print(f"Error: reactions file not found: {reactions_path}", file=sys.stderr)
            sys.exit(1)

        with open(reactions_path) as f:
            data = json.load(f)
        reactions = data.get("reactions", data) if isinstance(data, dict) else data
        reactions_source_label = str(reactions_path)
        print(f"Loaded {len(reactions)} reactions from {reactions_path.name}")

    # Build prompt
    user_prompt = build_user_prompt(reactions)
    prompt_system_hash = _sha256_text(SYSTEM_PROMPT)
    prompt_user_hash = _sha256_text(user_prompt)
    prompt_hash = _sha256_text(f"{SYSTEM_PROMPT}\n\n{user_prompt}")

    # Call model
    raw_response, elapsed = call_model(
        args.model,
        SYSTEM_PROMPT,
        user_prompt,
        args.dry_run,
        llm_seed=args.llm_seed,
        llm_temperature=float(args.llm_temperature),
    )

    if args.dry_run:
        return

    # Parse submission
    submission = parse_submission(raw_response)

    # Score
    if ground_truth_path and ground_truth_path.exists():
        score_result = score_locally(submission, ground_truth_path)
        summary = format_score_summary(score_result, args.model, elapsed, reactions_source_label)
    else:
        score_result = {"note": "no ground truth available for arena reactions"}
        summary = f"Run complete in {elapsed:.1f}s. No ground truth to score against."
        print("  (No local ground truth available for this eval source.)")

    print("\n" + summary)

    # Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{args.model.replace('/', '_').replace('-', '_')}"
    run_dir = RUNS_DIR / run_name
    metadata = {
        "model": args.model,
        "elapsed_seconds": elapsed,
        "llm_seed": args.llm_seed,
        "llm_temperature": float(args.llm_temperature),
        "reactions_source": reactions_source_label,
        "eval_set_id": resolved_eval_set_id,
        "eval_set_purpose": resolved_eval_set_purpose,
        "eval_case_ids_hash": resolved_case_ids_hash,
        "prompt_hash": prompt_hash,
        "prompt_system_hash": prompt_system_hash,
        "prompt_user_hash": prompt_user_hash,
        "reaction_count": len(reactions),
    }
    save_run(run_dir, SYSTEM_PROMPT, user_prompt, raw_response,
             submission, score_result, summary, metadata)

    # Optionally update leaderboard
    if args.update_leaderboard and ground_truth_path:
        update_leaderboard(score_result, args.model, elapsed, reactions_source_label, run_dir)


if __name__ == "__main__":
    main()
