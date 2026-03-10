import {
  GROUND_TRUTH,
  NUM_REACTIONS,
  TIME_LIMIT_SECS,
  MAX_SCORE,
  normalizeDotJoined,
} from "./data.js";
import { exact_match_ratio, set_overlap, time_decay } from "../primitives/scoring.js";

// ── Submitted step type ───────────────────────────────────────────────

interface SubmittedStep {
  resulting_state: string[];
  electron_pushes: string[];
}

// ── Electron push type extraction and scoring ─────────────────────────

/**
 * Extract push type prefix from a push notation string.
 * "lp:7>1" → "lp", "sigma:1-2>2" → "sigma", "pi:3-4>7" → "pi"
 * Returns null for unrecognized formats.
 */
function extractPushType(push: string): string | null {
  if (!push || typeof push !== "string") return null;
  const colonIdx = push.indexOf(":");
  if (colonIdx === -1) return null;
  const prefix = push.slice(0, colonIdx).trim().toLowerCase();
  if (prefix === "lp" || prefix === "sigma" || prefix === "pi") return prefix;
  return null;
}

/**
 * Extract push types from an array of push notation strings.
 * Returns an array of type strings (may contain duplicates = multiset).
 */
function extractPushTypes(pushes: string[]): string[] {
  if (!Array.isArray(pushes)) return [];
  return pushes.map(extractPushType).filter((t): t is string => t !== null);
}

/**
 * Jaccard similarity on type multisets.
 * Intersection = sum of min counts for each type.
 * Union = sum of max counts for each type.
 */
function typeJaccard(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) return 1.0;
  if (a.length === 0 || b.length === 0) return 0.0;

  // Build count maps
  const countA = new Map<string, number>();
  const countB = new Map<string, number>();
  for (const t of a) countA.set(t, (countA.get(t) ?? 0) + 1);
  for (const t of b) countB.set(t, (countB.get(t) ?? 0) + 1);

  const allTypes = new Set([...countA.keys(), ...countB.keys()]);
  let intersection = 0;
  let union = 0;
  for (const t of allTypes) {
    const ca = countA.get(t) ?? 0;
    const cb = countB.get(t) ?? 0;
    intersection += Math.min(ca, cb);
    union += Math.max(ca, cb);
  }
  return union === 0 ? 1.0 : intersection / union;
}

// ── Scoring logic ─────────────────────────────────────────────────────

function scoreProducts(
  submittedProducts: string[],
  shuffleOrder: number[],
): number {
  const submittedNorm = submittedProducts.map(normalizeDotJoined);

  const expectedNorm = shuffleOrder.map((canonIdx) => {
    const gt = GROUND_TRUTH[canonIdx];
    return normalizeDotJoined(gt.finalProducts.join("."));
  });

  return exact_match_ratio(submittedNorm, expectedNorm);
}

/**
 * Score mechanism completeness: step count accuracy + intermediate species Jaccard.
 * For each reaction:
 *   - Step count score: 1 if submitted step count matches GT, else 0.5 if within 1, else 0
 *   - Intermediate Jaccard: set_overlap of submitted intermediate states vs GT intermediates
 *   - Reaction score: average of step count score and intermediate Jaccard
 * Returns a value in [0, 1].
 */
function scoreMechanism(
  submittedSteps: SubmittedStep[][],
  shuffleOrder: number[],
): number {
  let total = 0;
  let counted = 0;

  for (let i = 0; i < NUM_REACTIONS; i++) {
    const canonIdx = shuffleOrder[i];
    const gt = GROUND_TRUTH[canonIdx];
    const submitted = Array.isArray(submittedSteps[i]) ? submittedSteps[i] : [];

    // Step count score
    const gtStepCount = gt.steps.length;
    const subStepCount = submitted.length;
    let stepCountScore: number;
    if (subStepCount === gtStepCount) {
      stepCountScore = 1.0;
    } else if (Math.abs(subStepCount - gtStepCount) === 1) {
      stepCountScore = 0.5;
    } else {
      stepCountScore = 0.0;
    }

    // Intermediate species Jaccard (all states except final products)
    // GT intermediates = resultingState of all steps except the last
    const gtIntermediates = gt.steps.length > 1
      ? gt.steps.slice(0, -1).flatMap((s) => s.resultingState)
      : [];

    const submittedIntermediates = submitted.length > 1
      ? submitted.slice(0, -1).flatMap((s) =>
          Array.isArray(s?.resulting_state) ? s.resulting_state : []
        )
      : [];

    const intermediateJaccard = set_overlap(submittedIntermediates, gtIntermediates);

    // Reaction completeness score
    const reactionScore = (stepCountScore + intermediateJaccard) / 2;
    total += reactionScore;
    counted++;
  }

  return counted > 0 ? total / counted : 0;
}

/**
 * Score electron push quality across all reactions and steps.
 * For each reaction i, for each step j:
 *   - Extract push types from submitted electron_pushes
 *   - Compare type multiset with GT type multiset using Jaccard
 * Reaction push score = average over steps (only steps that exist in both submitted and GT).
 * Challenge push score = average over reactions that have at least one submitted step.
 * Returns a value in [0, 1].
 */
function scoreElectronPushes(
  submittedSteps: SubmittedStep[][],
  shuffleOrder: number[],
): number {
  let totalReactionScores = 0;
  let reactionsWithSubmission = 0;

  for (let i = 0; i < NUM_REACTIONS; i++) {
    const canonIdx = shuffleOrder[i];
    const gt = GROUND_TRUTH[canonIdx];
    const submitted = Array.isArray(submittedSteps[i]) ? submittedSteps[i] : [];

    if (submitted.length === 0) continue;
    reactionsWithSubmission++;

    let stepScoreSum = 0;
    const numSteps = Math.max(gt.steps.length, submitted.length);

    for (let j = 0; j < numSteps; j++) {
      const gtStep = gt.steps[j];
      const subStep = submitted[j];

      if (!gtStep || !subStep) {
        // One side has no step at this index — score 0 for this step
        stepScoreSum += 0;
        continue;
      }

      const gtTypes = extractPushTypes(gtStep.electronPushes);
      const subTypes = extractPushTypes(
        Array.isArray(subStep?.electron_pushes) ? subStep.electron_pushes : []
      );

      stepScoreSum += typeJaccard(subTypes, gtTypes);
    }

    totalReactionScores += stepScoreSum / numSteps;
  }

  return reactionsWithSubmission > 0 ? totalReactionScores / reactionsWithSubmission : 0;
}

export function score(input: any) {
  const { submission, groundTruth, startedAt, submittedAt } = input;

  const shuffleOrder: number[] = (groundTruth as any).shuffleOrder ?? Array.from({ length: NUM_REACTIONS }, (_, i) => i);

  const finalProducts = (submission.final_products ?? []) as string[];
  const stepsRaw = (submission.steps ?? []) as SubmittedStep[][];
  const methodology = submission.methodology as string | undefined;

  // Product accuracy (30%, max 300)
  const productRaw = scoreProducts(finalProducts, shuffleOrder);
  const productScore = Math.round(productRaw * 0.30 * MAX_SCORE);

  // Anti-gaming gate: completeness, electron_push, and speed are zeroed if no correct products
  const hasCorrectProduct = productRaw > 0;

  // Pathway coverage (30%, max 300)
  const completenessRaw = hasCorrectProduct ? scoreMechanism(stepsRaw, shuffleOrder) : 0;
  const completenessScore = Math.round(completenessRaw * 0.30 * MAX_SCORE);

  // Electron push quality (20%, max 200)
  const pushRaw = hasCorrectProduct ? scoreElectronPushes(stepsRaw, shuffleOrder) : 0;
  const pushScore = Math.round(pushRaw * 0.20 * MAX_SCORE);

  // Speed (10%, max 100)
  const elapsedSecs = (submittedAt.getTime() - startedAt.getTime()) / 1000;
  const speedRaw = hasCorrectProduct ? time_decay(elapsedSecs, TIME_LIMIT_SECS) : 0;
  const speedScore = Math.round(speedRaw * 0.10 * MAX_SCORE);

  // Methodology (10%, max 100) — awarded regardless of product accuracy
  const methodologyScore =
    typeof methodology === "string" && methodology.trim().length > 0 ? 100 : 0;

  const total = productScore + completenessScore + pushScore + speedScore + methodologyScore;

  return {
    breakdown: {
      correctness: productScore,
      completeness: completenessScore,
      precision: pushScore,  // Changed from electron_push to precision
      speed: speedScore,
      methodology: methodologyScore,
      total,
    },
  };
}