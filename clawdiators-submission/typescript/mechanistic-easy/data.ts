// ── Ground truth types ─────────────────────────────────────────────────

interface GroundTruthStep {
  resultingState: string[];    // SMILES of species after this step (intermediates or final products)
  electronPushes: string[];    // push notations: "lp:N>M", "sigma:N-M>P", "pi:N-M>P"
}

interface GroundTruthReaction {
  sourceId: string;
  finalProducts: string[];     // canonical SMILES for each product species (= last step resultingState)
  steps: GroundTruthStep[];    // 1 step for concerted, 2+ for multi-step
  description: string;
}

// ── Ground truth (canonical order, index 0-9) ────────────────────────
// Products are RDKit-canonical, dot-sorted for multi-species reactions.
// 1-step concerted reactions have steps.length === 1 with resultingState = finalProducts.
// 2-step reactions have steps[0].resultingState = ionic intermediate, steps[1].resultingState = finalProducts.

const GROUND_TRUTH: GroundTruthReaction[] = [
  {
    sourceId: "flower_024300",
    finalProducts: ["C[N+](C)(C)CC1CO1", "[Cl-]"],
    steps: [
      {
        resultingState: ["C[N+](C)(C)CC1CO1", "[Cl-]"],
        electronPushes: ["lp:7>1", "sigma:1-5>5"],
      },
    ],
    description: "SN2 substitution: chloromethyl oxetane + trimethylamine → trimethyl(oxetan-2-ylmethyl)ammonium chloride",
  },
  {
    sourceId: "flower_130926",
    finalProducts: ["[Br-]", "CCC[N+]1(C)CCCC1"],
    steps: [
      {
        resultingState: ["[Br-]", "CCC[N+]1(C)CCCC1"],
        electronPushes: ["lp:6>2", "sigma:2-1>1"],
      },
    ],
    description: "SN2 substitution: n-propyl bromide + N-methylpyrrolidine → 1-methyl-1-propylpyrrolidin-1-ium bromide",
  },
  {
    sourceId: "flower_222822",
    finalProducts: ["CC[N+]1(C2CCCCC2)CCCC1", "[I-]"],
    steps: [
      {
        resultingState: ["CC[N+]1(C2CCCCC2)CCCC1", "[I-]"],
        electronPushes: ["lp:5>1", "sigma:1-2>2"],
      },
    ],
    description: "SN2 substitution: ethyl iodide + 4-(pyrrolidin-1-yl)cyclohexane → N-ethyl quaternary ammonium iodide",
  },
  {
    sourceId: "flower_120680",
    finalProducts: ["Clc1ccc(NCC(O)C)cc1"],
    steps: [
      {
        resultingState: ["Clc1ccc([NH2+]CC([O-])C)cc1"],
        electronPushes: ["lp:6>12", "sigma:12-9>9"],
      },
      {
        resultingState: ["Clc1ccc(NCC(O)C)cc1"],
        electronPushes: ["lp:9>15", "sigma:15-6>6"],
      },
    ],
    description: "2-step: epoxide ring opening of propylene oxide by 4-chloroaniline — SN2 attack then proton transfer",
  },
  {
    sourceId: "flower_053068",
    finalProducts: ["CP(=O)(OCC)OCC", "CCI"],
    steps: [
      {
        resultingState: ["C[P+](OCC)(OCC)OCC", "[I-]"],
        electronPushes: ["lp:3>1", "sigma:1-2>2"],
      },
      {
        resultingState: ["CP(=O)(OCC)OCC", "CCI"],
        electronPushes: ["lp:2>5", "sigma:5-4>3"],
      },
    ],
    description: "2-step: Arbuzov reaction — SN2 methylation of triethyl phosphite, then demethylation by iodide",
  },
  {
    sourceId: "flower_135501",
    finalProducts: ["N#CCCC1C=CC=C1"],
    steps: [
      {
        resultingState: ["N#CCCC1C=CC=C1"],
        electronPushes: ["pi:3-4>7", "pi:7-8>9", "sigma:9-17>3"],
      },
    ],
    description: "Diels-Alder [4+2]: acrylonitrile (dienophile) + cyclopenta-1,3-diene (diene) → cyanoethyl-cyclopentadiene adduct",
  },
  {
    sourceId: "flower_160718",
    finalProducts: ["C=C(C)C(CO)C(C)=O"],
    steps: [
      {
        resultingState: ["C=C(C)C(CO)C(C)=O"],
        electronPushes: ["pi:1-2>13", "sigma:13-7>6", "pi:6-5>1"],
      },
    ],
    description: "Ene reaction: formaldehyde (enophile) + methyl isopropenyl ketone → homoallylic alcohol",
  },
  {
    sourceId: "flower_225090",
    finalProducts: ["[O-][n+]1cccc2ccccc21", "CC(=O)O"],
    steps: [
      {
        resultingState: ["[O-][n+]1cccc2ccccc21", "CC(=O)O"],
        electronPushes: ["lp:6>1"],
      },
    ],
    description: "N-oxidation: quinoline + peracetic acid → quinoline N-oxide + acetic acid",
  },
  {
    sourceId: "flower_105699",
    finalProducts: ["CCOC(=O)c1ccc(-c2cccc[n+]2[O-])cc1", "CC(=O)O"],
    steps: [
      {
        resultingState: ["CCOC(=O)c1ccc(-c2cccc[n+]2[O-])cc1", "CC(=O)O"],
        electronPushes: ["lp:13>18"],
      },
    ],
    description: "N-oxidation: ethyl 4-(pyridin-2-yl)benzoate + peracetic acid → pyridine N-oxide + acetic acid",
  },
  {
    sourceId: "flower_127589",
    finalProducts: ["CC1=CCOC(C(C)c2ccccc2)C1"],
    steps: [
      {
        resultingState: ["CC1=CCOC(C(C)c2ccccc2)C1"],
        electronPushes: ["pi:2-5>9", "pi:9-10>4", "pi:4-3>2"],
      },
    ],
    description: "Hetero Diels-Alder [4+2]: isoprene (diene) + 2-phenylpropanal (C=O dienophile) → dihydropyran",
  },
];

// ── Workspace reactions (display, no ground truth) ────────────────────

interface WorkspaceReaction {
  id: string;
  sourceId: string;
  startingMaterials: string[];
  targetProducts: string[];
  conditions: string;
  nSteps: number;
}

const WORKSPACE_REACTIONS: WorkspaceReaction[] = [
  {
    id: "SEED_PLACEHOLDER-0",
    sourceId: "flower_024300",
    startingMaterials: ["ClCC1CO1", "CN(C)C"],
    targetProducts: ["C[N+](C)(C)CC1CO1", "[Cl-]"],
    conditions: "aqueous acetonitrile, RT",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-1",
    sourceId: "flower_130926",
    startingMaterials: ["CCCBr", "CN1CCCC1"],
    targetProducts: ["[Br-]", "CCC[N+]1(C)CCCC1"],
    conditions: "acetonitrile, RT",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-2",
    sourceId: "flower_222822",
    startingMaterials: ["CCI", "C1CCC(N2CCCC2)CC1"],
    targetProducts: ["CC[N+]1(C2CCCCC2)CCCC1", "[I-]"],
    conditions: "acetonitrile, RT",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-3",
    sourceId: "flower_120680",
    startingMaterials: ["Clc1ccc(N)cc1", "CC1CO1"],
    targetProducts: ["Clc1ccc(NCC(O)C)cc1"],
    conditions: "aqueous, RT",
    nSteps: 2,
  },
  {
    id: "SEED_PLACEHOLDER-4",
    sourceId: "flower_053068",
    startingMaterials: ["CI", "CCOP(OCC)OCC"],
    targetProducts: ["CP(=O)(OCC)OCC", "CCI"],
    conditions: "neat, 100 degC",
    nSteps: 2,
  },
  {
    id: "SEED_PLACEHOLDER-5",
    sourceId: "flower_135501",
    startingMaterials: ["C=CC#N", "C1=CCC=C1"],
    targetProducts: ["N#CCCC1C=CC=C1"],
    conditions: "toluene, 150 degC, thermal",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-6",
    sourceId: "flower_160718",
    startingMaterials: ["C=O", "CC(=O)C=C(C)C"],
    targetProducts: ["C=C(C)C(CO)C(C)=O"],
    conditions: "neat, thermal",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-7",
    sourceId: "flower_225090",
    startingMaterials: ["c1ccc2ncccc2c1", "CC(=O)OO"],
    targetProducts: ["[O-][n+]1cccc2ccccc21", "CC(=O)O"],
    conditions: "acetic acid, RT",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-8",
    sourceId: "flower_105699",
    startingMaterials: ["CCOC(=O)c1ccc(-c2ccccn2)cc1", "CC(=O)OO"],
    targetProducts: ["CCOC(=O)c1ccc(-c2cccc[n+]2[O-])cc1", "CC(=O)O"],
    conditions: "acetic acid, RT",
    nSteps: 1,
  },
  {
    id: "SEED_PLACEHOLDER-9",
    sourceId: "flower_127589",
    startingMaterials: ["C=CC(=C)C", "CC(C=O)c1ccccc1"],
    targetProducts: ["CC1=CCOC(C(C)c2ccccc2)C1"],
    conditions: "toluene, 80 degC, thermal",
    nSteps: 1,
  },
];

// ── Constants ─────────────────────────────────────────────────────────

const TIME_LIMIT_SECS = 600;
const MAX_SCORE = 1000;
const NUM_REACTIONS = 10;

// ── Seeded PRNG (mulberry32) ──────────────────────────────────────────

function mulberry32(seed: number): () => number {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Shuffle indices using seeded PRNG (Fisher-Yates) ─────────────────

function shuffledIndices(seed: number): number[] {
  const rng = mulberry32(seed);
  const indices = Array.from({ length: NUM_REACTIONS }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

// ── SMILES normalization (pure TypeScript, no RDKit) ──────────────────
// Normalizes dot-joined multi-species strings by sorting fragments.

function normalizeDotJoined(smi: string): string {
  if (!smi || typeof smi !== "string") return "";
  return smi
    .trim()
    .split(".")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .sort()
    .join(".");
}

export {
  type GroundTruthStep,
  type GroundTruthReaction,
  type WorkspaceReaction,
  GROUND_TRUTH,
  WORKSPACE_REACTIONS,
  TIME_LIMIT_SECS,
  MAX_SCORE,
  NUM_REACTIONS,
  mulberry32,
  shuffledIndices,
  normalizeDotJoined,
  generateData,
  generateWorkspace,
};

// ── Data generation ───────────────────────────────────────────────────

function generateData(seed: number, _config: Record<string, unknown>) {
  const order = shuffledIndices(seed);

  const groundTruth = {
    shuffleOrder: order,
    reactions: order.map((canonIdx) => ({
      canonicalIndex: canonIdx,
      sourceId: GROUND_TRUTH[canonIdx].sourceId,
      finalProducts: GROUND_TRUTH[canonIdx].finalProducts,
      steps: GROUND_TRUTH[canonIdx].steps,
    })),
  };

  return {
    objective: `Predict the mechanism for 10 organic reactions drawn from the FlowER benchmark. Submit final product SMILES, mechanistic steps with electron push notations, and a methodology description. Concerted mechanisms (SN2, pericyclic, N-oxidation) have 1 step. Multi-step reactions (e.g., epoxide opening, Arbuzov) have 2+ steps. See reactions.json and example/worked_example.json in your workspace.`,
    groundTruth,
  };
}

function generateWorkspace(seed: number, _config: Record<string, unknown>) {
  const order = shuffledIndices(seed);

  const reactions = order.map((canonIdx, shuffledIdx) => {
    const r = WORKSPACE_REACTIONS[canonIdx];
    return {
      id: `mech-easy-${seed}-${shuffledIdx}`,
      starting_materials: r.startingMaterials,
      target_products: r.targetProducts,
      conditions: r.conditions,
      n_steps: r.nSteps,
    };
  });

  const reactionsJson = JSON.stringify({ reactions }, null, 2);

  const files: Record<string, string> = {
    "reactions.json": reactionsJson,
  };

  for (let shuffledIdx = 0; shuffledIdx < NUM_REACTIONS; shuffledIdx++) {
    const rxn = reactions[shuffledIdx];
    files[`reactions/mech-easy-${seed}-${shuffledIdx}.json`] = JSON.stringify(rxn, null, 2);
  }

  // Worked examples (outside eval set)
  const workedExample = {
    _note: "Two fully solved example reactions. NOT from the eval set. Shows the new submission format with steps and electron_pushes.",
    examples: [
      {
        reaction: {
          id: "example-sn2",
          starting_materials: ["CI", "[OH-]"],
          target_products: ["CO", "[I-]"],
          conditions: "aqueous, basic",
        },
        correct_submission: {
          final_products: ["CO.[I-]"],
          steps: [
            {
              resulting_state: ["CO", "[I-]"],
              electron_pushes: ["lp:O>C", "sigma:C-I>I"],
            },
          ],
          methodology: "SN2 concerted: hydroxide lone pair attacks methyl carbon. Backside attack, iodide leaves in single step. No discrete intermediate.",
        },
      },
      {
        reaction: {
          id: "example-epoxide-opening",
          starting_materials: ["C1CO1", "CCN"],
          target_products: ["CCNCC[OH]"],
          conditions: "aqueous, RT",
        },
        correct_submission: {
          final_products: ["CCNCCO"],
          steps: [
            {
              resulting_state: ["CC[NH2+]CC[O-]"],
              electron_pushes: ["lp:N>C_epoxide", "sigma:C-O>O"],
            },
            {
              resulting_state: ["CCNCCO"],
              electron_pushes: ["lp:O>H_nitrogen", "sigma:N-H>N"],
            },
          ],
          methodology: "SN2 ring opening: amine lone pair attacks less hindered epoxide carbon. Ring opens via backside attack. Then proton transfer from ammonium to alkoxide. 2 discrete steps.",
        },
      },
    ],
  };

  files["example/worked_example.json"] = JSON.stringify(workedExample, null, 2);

  return files;
}