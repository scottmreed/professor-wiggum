import { describe, it, expect } from "@jest/globals";
import mod from "./index.js";
import { generateData, GROUND_TRUTH, NUM_REACTIONS } from "./data.js";
import { score } from "./scorer.js";

describe("mechanistic-easy challenge", () => {
  describe("determinism", () => {
    it("generateData() returns byte-identical JSON for same seed", () => {
      const seed = 42;
      const data1 = generateData(seed, {});
      const data2 = generateData(seed, {});

      // Convert to JSON strings for byte comparison
      const json1 = JSON.stringify(data1, Object.keys(data1).sort());
      const json2 = JSON.stringify(data2, Object.keys(data2).sort());

      expect(json1).toBe(json2);
    });
  });

  describe("solvability", () => {
    it("perfect submission scores ≥ 700/1000 (≥60%)", () => {
      const seed = 42;
      const data = generateData(seed, {});
      const shuffleOrder = (data.groundTruth as any).shuffleOrder;

      // Create perfect submission by passing through ground truth
      const finalProducts = shuffleOrder.map((canonIdx) =>
        GROUND_TRUTH[canonIdx].finalProducts.join(".")
      );

      const steps = shuffleOrder.map((canonIdx) =>
        GROUND_TRUTH[canonIdx].steps.map((step) => ({
          resulting_state: step.resultingState,
          electron_pushes: step.electronPushes,
        }))
      );

      const submission = {
        final_products: finalProducts,
        steps: steps,
        methodology: "Perfect ground truth submission for testing",
      };

      const result = score({
        submission,
        groundTruth: data.groundTruth,
        startedAt: new Date(Date.now() - 1000), // 1 second elapsed
        submittedAt: new Date(),
      });

      expect(result.breakdown.total).toBeGreaterThanOrEqual(700);
      expect(result.breakdown.correctness).toBe(300); // All products correct
      expect(result.breakdown.completeness).toBe(300); // All mechanisms correct
      expect(result.breakdown.precision).toBe(200); // All electron pushes correct
      expect(result.breakdown.speed).toBe(100); // Full speed score
      expect(result.breakdown.methodology).toBe(100); // Methodology present
    });
  });

  describe("anti-gaming", () => {
    it("all-wrong products submission scores ≤ 300 (<30%)", () => {
      const seed = 42;
      const data = generateData(seed, {});
      const shuffleOrder = (data.groundTruth as any).shuffleOrder;

      // Create submission with all wrong products (but correct steps and pushes)
      const wrongProducts = shuffleOrder.map(() => "CC"); // Wrong SMILES for all reactions
      const steps = shuffleOrder.map((canonIdx) =>
        GROUND_TRUTH[canonIdx].steps.map((step) => ({
          resulting_state: step.resultingState,
          electron_pushes: step.electronPushes,
        }))
      );

      const submission = {
        final_products: wrongProducts,
        steps: steps,
        methodology: "Anti-gaming test with wrong products",
      };

      const result = score({
        submission,
        groundTruth: data.groundTruth,
        startedAt: new Date(Date.now() - 1000),
        submittedAt: new Date(),
      });

      // With zero correct products, completeness, precision, and speed should be zeroed
      expect(result.breakdown.correctness).toBe(0); // No products correct
      expect(result.breakdown.completeness).toBe(0); // Zeroed by anti-gaming gate
      expect(result.breakdown.precision).toBe(0); // Zeroed by anti-gaming gate
      expect(result.breakdown.speed).toBe(0); // Zeroed by anti-gaming gate
      expect(result.breakdown.methodology).toBe(100); // Methodology still scores
      expect(result.breakdown.total).toBeLessThanOrEqual(300); // ≤ 30% of max score
    });
  });

  describe("validation", () => {
    it("rejects submissions with wrong final_products length", () => {
      const submission = {
        final_products: ["CC"], // Only 1 instead of 10
        steps: [],
        methodology: "test",
      };

      const warnings = mod.validateSubmission(submission);
      expect(warnings.some(w => w.field === "final_products" && w.severity === "error")).toBe(true);
    });

    it("rejects submissions with wrong steps length", () => {
      const submission = {
        final_products: Array(10).fill("CC"),
        steps: [[]], // Only 1 instead of 10
        methodology: "test",
      };

      const warnings = mod.validateSubmission(submission);
      expect(warnings.some(w => w.field === "steps" && w.severity === "error")).toBe(true);
    });

    it("accepts valid submissions", () => {
      const submission = {
        final_products: Array(10).fill("CC"),
        steps: Array(10).fill([]),
        methodology: "test methodology",
      };

      const warnings = mod.validateSubmission(submission);
      const errors = warnings.filter(w => w.severity === "error");
      expect(errors.length).toBe(0);
    });
  });
});