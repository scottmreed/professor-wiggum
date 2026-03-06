// --- API origin helpers ---
function getApiBaseUrl() {
  return (window.MECH_CONFIG && window.MECH_CONFIG.API_BASE_URL) || "";
}
// Drop-in replacement for fetch() that prepends the configured API base URL.
function fetchApi(path, ...args) {
  return fetch(getApiBaseUrl() + path, ...args);
}
function sseUrl(path) {
  return getApiBaseUrl() + path;
}

let runId = null;
let eventSource = null;
let snapshotTimer = null;
let examples = [];
let _curriculumStatus = null;
let latestEvaluation = null;
let latestFlow = null;
let evalSets = [];
let evalTierData = { tiers: { easy: [], medium: [], hard: [] }, cases: [] };
let harnessVersions = [];
let modelCatalog = [];
let allFamilies = [];
let lastSelectedNodeId = null;
let selectedPromptVersion = "latest";
let versionTemplateCache = {};
let stepModelMap = {}; // { step_name: model_string }
let isDryRun = false;
let latestPauseInfo = null;
let activeKnownMechanism = null;
let knownMechanismRenderToken = 0;
let latestSnapshotData = null;
let showAtomNumbers = false;
let skipAtomMappingForSmirks = false;
let currentPrimaryView = "curriculum";
let latestCurriculumStatus = null;
let latestCurriculumHistory = [];

const runStatusEl = document.getElementById("runStatus");
const terminalOutput = document.getElementById("terminalOutput");
const evaluationOutput = document.getElementById("evaluationOutput");
const harnessOutput = document.getElementById("harnessOutput");
const flowDiagram = document.getElementById("flowDiagram");
const nodeDetailPanel = document.getElementById("nodeDetailPanel");
const nodeDetailTitle = document.getElementById("nodeDetailTitle");
const nodeDetailBody = document.getElementById("nodeDetailBody");
const runStepSummaries = document.getElementById("runStepSummaries");
const modalLeaderboardList = document.getElementById("modalLeaderboardList");
const activeStepBadge = document.getElementById("activeStepBadge");
const knownMechanismText = document.getElementById("knownMechanismText");
const knownMechanismCards = document.getElementById("knownMechanismCards");
const knownMechanismFinalProduct = document.getElementById("knownMechanismFinalProduct");
const predictedMechanismText = document.getElementById("predictedMechanismText");
const predictedMechanismCards = document.getElementById("predictedMechanismCards");
const curriculumStatusMessage = document.getElementById("curriculumStatusMessage");
const toastContainer = document.getElementById("toastContainer");

const FLOW_KIND_FILL = { llm: "#fff2d8", deterministic: "#e7f4ea", decision: "#e8eefb" };
const FLOW_KIND_STROKE = { llm: "#c26b00", deterministic: "#2f7d3b", decision: "#3355aa" };
const FLOW_KIND_BY_NODE_ID = {
  balance_analysis: "deterministic",
  functional_groups: "deterministic",
  ph_recommendation: "deterministic",
  initial_conditions: "llm",
  missing_reagents: "llm",
  atom_mapping: "llm",
  reaction_type_mapping: "llm",
  mechanism_step_proposal: "llm",
  mechanism_synthesis: "deterministic",
  bond_electron_validation: "deterministic",
  atom_balance_validation: "deterministic",
  state_progress_validation: "deterministic",
  retry_gate: "decision",
  backtrack_gate: "decision",
  completion_check: "decision",
  reflection: "deterministic",
  step_atom_mapping: "llm",
  run_failed: "decision",
  run_complete: "deterministic",
};

const NODE_DESCRIPTIONS = {
  balance_analysis: "RDKit counts atoms on each side of the reaction. Reports whether starting materials and products are stoichiometrically balanced, and lists any deficit or surplus atoms. Runs once at the start.",
  functional_groups: "SMARTS pattern matching identifies reactive functional groups (alcohols, ketones, amines, etc.) and leaving groups across all starting materials and products. Runs once; output is not passed to the LLM directly but informs atom-mapping context.",
  ph_recommendation: "Rule-based pH suggestion. If the user supplies a pH it is returned as-is. Otherwise, uses Dimorphite-DL protonation profiles or a heuristic acid/base scoring of the molecules. Runs once; the recommended pH is passed to the Assess Reaction Conditions LLM call.",
  initial_conditions: "LLM call that receives starting materials, products, the pH recommendation, and optional functional group data. Returns environment (acidic/basic/neutral), representative pH, pH range, and either acid candidates (acidic) or base candidates (basic) — never both. Its full JSON output is forwarded as conditions_guidance to the Predict Missing Reagents step and the reagent candidates are injected into Propose Next Intermediate.",
  missing_reagents: "LLM call that receives starting materials, products, and the full conditions_guidance JSON from the previous step (including representative pH and acid/base candidates). Returns any missing reactants or products needed to balance the reaction under the recommended conditions.",
  atom_mapping: "LLM call that receives starting materials and products. Returns a proposed atom-to-atom mapping, unmapped atoms, and confidence score. If functional group analysis is enabled, that context is included.",
  reaction_type_mapping: "LLM call that maps the reaction to one mechanism taxonomy label from training_data/reaction_type_templates.json, with confidence and rationale. Includes explicit 'no_match' when no taxonomy type fits.",
  mechanism_step_proposal: "Topology-aware LLM call that proposes ranked candidates for the next mechanism step. Dispatches via coordination_topology: SAS (1 call, 1 candidate), centralized MAS (1 call, up to 3 candidates), independent MAS (3 parallel calls, 2 each), or decentralized MAS (3 agents × 2 debate rounds, consensus merge). All returned candidates are validated independently by the same post-step validators. The top-ranked valid candidate proceeds; alternatives are stored as branch points for backtracking.",
  mechanism_synthesis: "Deterministic validation of the proposed mechanism step. Receives: step index, current state, target products, explicit electron moves, reaction SMIRKS with |mech:v1;...| metadata, the predicted intermediate, resulting state, and a note field. On retries (up to 3), the note field includes retry_feedback containing the names of failed checks and guidance text from the previous attempt. The three sub-checks (bond/electron, atom balance, state progress) run as part of this validation.",
  bond_electron_validation: "Parses the explicit mechanism-move tags (for example |mech:v1;lp:4>2;sigma:2-3>3|) from the reaction SMIRKS and verifies that the implied bond changes match the actual step. Part of the mechanism step validation; re-runs on each retry attempt.",
  atom_balance_validation: "RDKit verifies that every atom in the current state appears in the resulting state — no atoms created or destroyed. Part of the mechanism step validation; re-runs on each retry attempt.",
  state_progress_validation: "Checks that the resulting state differs from the current state and that starting materials are not simply returned unchanged. Part of the mechanism step validation; re-runs on each retry attempt.",
  reflection: "Deterministic aggregation of validation warnings from the accepted mechanism step (for example unchanged state detected or invalid mechanism-move metadata). Runs once per loop iteration after a step passes all three validation checks. Does not feed back into the current step — it records critique for observability.",
  retry_gate: "Routes based on validation outcome: if all 3 checks pass, proceeds to Target Products Reached; if any check fails and retries remain (up to 3 per candidate), loops back to Validate Mechanism Step with retry_feedback; if a repeating failure signature is detected, repropose routes back to Propose Mechanism Step with avoidance hints; if all candidates fail, routes to Backtrack.",
  backtrack_gate: "Activated when all candidates for a mechanism step fail validation and reproposal limits are exhausted. Searches branch points (most recent first) for untried alternatives. If a branch point with alternatives exists, reverts state to that point and tries the next alternative with a clean slate — no trace of the failed path is passed to the LLM. If no branch points remain, routes to Run Failed.",
  completion_check: "Checks the contains_target_product flag from the latest accepted mechanism step. If yes, routes to Run Complete (mechanism finished). If no, routes to Collect Validation Warnings and step atom mapping, then loops back to Propose Mechanism Step for the next step.",
  step_atom_mapping: "LLM atom mapping for the accepted mechanism step. Maps atoms between current_state and resulting_state after each successful step. Runs on the loop-back path after completion_check determines the mechanism is not yet complete.",
  run_failed: "Terminal node reached when all retry, reproposal, and backtracking options are exhausted. The run status is set to failed.",
  run_complete: "Terminal node reached when the target products are found in the resulting state. The run status is set to completed and no further mechanism steps are generated.",
};

const NODE_IO_SCHEMAS = {
  functional_groups: {
    inputs: ["starting_materials (SMILES[])", "products (SMILES[])"],
    outputs: ["functional_groups (object: group → count)", "leaving_groups (string[])"],
  },
  initial_conditions: {
    inputs: ["starting_materials (SMILES[])", "products (SMILES[])", "ph (from pH recommendation or user)", "functional_groups (if enabled)"],
    outputs: ["environment (acidic/basic/neutral)", "representative_ph", "ph_range", "acid_candidates (acidic only)", "base_candidates (basic only)", "warnings"],
  },
  missing_reagents: {
    inputs: ["starting_materials (SMILES[])", "products (SMILES[])", "conditions_guidance (full JSON from Assess Reaction Conditions, includes pH and acid/base candidates)"],
    outputs: ["missing_reactants (SMILES[])", "missing_products (SMILES[])", "verification"],
  },
  atom_mapping: {
    inputs: ["starting_materials (SMILES[])", "products (SMILES[])", "functional_groups (if enabled)"],
    outputs: ["mapped_atoms", "unmapped_atoms", "confidence", "reasoning"],
  },
  reaction_type_mapping: {
    inputs: ["starting_materials (SMILES[])", "products (SMILES[])", "balance_analysis", "functional_groups", "ph_recommendation", "initial_conditions", "missing_reagents", "atom_mapping"],
    outputs: ["selected_label_exact (taxonomy label or no_match)", "selected_type_id", "confidence (0-1)", "rationale", "top_candidates"],
  },
  mechanism_step_proposal: {
    inputs: ["starting_materials (SMILES[])", "products (SMILES[])", "current_state (SMILES[], updated after each accepted step)", "previous_intermediates (SMILES[], all prior accepted intermediates)", "ph", "temperature_celsius", "step_index", "acid_candidates or base_candidates (from conditions assessment, pH-selected)"],
    outputs: ["classification (intermediate_step/final_step)", "candidates [{rank, intermediate_smiles, reaction_description, confidence, intermediate_type, note}]", "analysis"],
  },
  mechanism_synthesis: {
    inputs: ["step_index", "current_state (SMILES[])", "target_products (SMILES[])", "predicted_intermediate (from Propose Next Intermediate)", "resulting_state (current_state + intermediate)", "previous_intermediates (SMILES[])", "starting_materials (SMILES[])", "note (on retry: includes retry_feedback with failed_checks[] and guidance text)"],
    outputs: ["reaction_smirks (with mech metadata)", "electron_pushes", "resulting_state", "contains_target_product", "bond_electron_validation", "atom_balance_check"],
  },
};

const CALL_TO_STEP_NAMES = {
  assess_initial_conditions: ["initial_conditions"],
  predict_missing_reagents: ["missing_reagents"],
  attempt_atom_mapping: ["atom_mapping"],
  select_reaction_type: ["reaction_type_mapping"],
  propose_mechanism_step: ["intermediates", "mechanism_step_proposal"],
  evaluate_run_judge: ["evaluation_judge"],
};

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function parseSmilesList(raw) {
  return String(raw || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseJsonInput(raw, fallback = null) {
  try {
    return JSON.parse(String(raw || "").trim());
  } catch (_) {
    return fallback;
  }
}

function _selectedLabel(selectId) {
  const el = document.getElementById(selectId);
  if (!el) return "-";
  const idx = typeof el.selectedIndex === "number" ? el.selectedIndex : -1;
  if (idx < 0 || !el.options || !el.options[idx]) return String(el.value || "-");
  return String(el.options[idx].textContent || el.value || "-");
}

function _onOff(value) {
  return value ? "On" : "Off";
}

function _setReadOnlyValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function getHarnessSelection() {
  const raw = document.getElementById("harnessSelectionInput")?.value || "default";
  if (raw === "new_harness") {
    return { type: "new", version: null };
  }
  if (raw.startsWith("family:")) {
    return { type: "family", familyId: raw.slice("family:".length) || null, version: null };
  }
  if (raw.startsWith("saved:")) {
    return { type: "existing", version: raw.slice("saved:".length) || null };
  }
  return { type: "existing", version: null };
}

function getSelectedHarnessName() {
  const selected = getHarnessSelection();
  if (selected.type === "existing" && selected.version) {
    return selected.version;
  }
  return "default";
}

function isNewHarnessSelection() {
  return getHarnessSelection().type === "new";
}

function updateOrchestrationModeUi() {
  const mode = document.getElementById("orchestrationModeInput")?.value || "standard";
  const panel = document.getElementById("ralphConfigPanel");
  if (panel) panel.style.display = mode === "ralph" ? "" : "none";
  const curriculumPanel = document.getElementById("curriculumPanel");
  const toggleBtn = document.getElementById("toggleCurriculumBtn");
  if (mode !== "ralph") {
    if (curriculumPanel) curriculumPanel.style.display = "none";
    if (toggleBtn) toggleBtn.style.display = "none";
  } else {
    if (toggleBtn) toggleBtn.style.display = "";
  }
  const ralphIcon = document.getElementById("ralphIcon");
  if (ralphIcon) ralphIcon.style.display = mode === "ralph" ? "" : "none";
}

function updateHarnessReadOnlySummary() {
  _setReadOnlyValue("readonlyModelFamily", _selectedLabel("modelFamilyInput"));
  _setReadOnlyValue("readonlyModelName", _selectedLabel("modelNameInput"));
  _setReadOnlyValue("readonlyReasoning", _selectedLabel("reasoningInput"));
}

function getSelectedModelMeta() {
  const modelName = document.getElementById("modelNameInput")?.value || "";
  return modelCatalog.find((item) => item.id === modelName) || null;
}

async function loadModelCatalog() {
  const response = await fetchApi("/api/catalog/models");
  if (!response.ok) return;
  const data = await response.json();
  modelCatalog = Array.isArray(data) ? data : [];
  populateModelOptions();
}

function populateModelOptions(preferredModel = null) {
  const family = document.getElementById("modelFamilyInput")?.value || "openai";
  const select = document.getElementById("modelNameInput");
  if (!select) return;
  const familyModels = modelCatalog.filter((item) => item.family === family && item.best_in_class);
  const selectedBefore = preferredModel || select.value;
  select.innerHTML = "";
  if (!familyModels.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = modelCatalog.length
      ? "No models available for this family"
      : "No models available";
    select.appendChild(opt);
    select.value = "";
    select.disabled = true;
    updateThinkingVisibility();
    return;
  }
  select.disabled = false;
  familyModels.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = item.label || item.id;
    select.appendChild(opt);
  });
  if ([...select.options].some((opt) => opt.value === selectedBefore)) {
    select.value = selectedBefore;
  } else if (select.options.length) {
    select.selectedIndex = 0;
  }
  updateThinkingVisibility();
}

function updateThinkingVisibility() {
  const reasoningRow = document.getElementById("reasoningRow");
  const reasoningInput = document.getElementById("reasoningInput");
  const meta = getSelectedModelMeta();
  const levels = Array.isArray(meta?.reasoning_levels) ? meta.reasoning_levels : [];
  const supportsThinking = levels.includes("low") || levels.includes("high");
  if (reasoningRow) reasoningRow.style.display = supportsThinking ? "" : "none";
  if (reasoningInput && !supportsThinking) reasoningInput.value = "";
}

function setPrimaryView(viewName) {
  currentPrimaryView = viewName;
  document.querySelectorAll(".view-tab").forEach((btn) => {
    const isActive = btn.id === `view${viewName.charAt(0).toUpperCase()}${viewName.slice(1)}Btn`;
    btn.classList.toggle("active", isActive);
  });
  const curriculumSection = document.getElementById("curriculumDashboardSection");
  const historySection = document.getElementById("historyDashboardSection");
  if (curriculumSection) curriculumSection.style.display = viewName === "curriculum" ? "" : "none";
  if (historySection) historySection.style.display = viewName === "history" ? "" : "none";
}

function setCurriculumPanelVisible(isVisible) {
  const panel = document.getElementById("curriculumPanel");
  if (panel) panel.style.display = isVisible ? "" : "none";
  const toggleBtn = document.getElementById("toggleCurriculumBtn");
  if (toggleBtn) toggleBtn.textContent = isVisible ? "Hide Curriculum" : "Show Curriculum";
}

async function refreshCurriculumStatus() {
  const response = await fetchApi("/api/curriculum/status");
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to load curriculum status");
  latestCurriculumStatus = data;
  renderCurriculumStatus();
  return data;
}

async function refreshCurriculumHistory() {
  const response = await fetchApi("/api/curriculum/history");
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to load curriculum history");
  latestCurriculumHistory = Array.isArray(data.items) ? data.items : [];
  renderCurriculumHistory();
  return latestCurriculumHistory;
}

function renderCurriculumStatus() {
  const status = latestCurriculumStatus || {};
  const module = status.current_module || {};
  const slot = status.today_slot || {};
  const nextSlot = status.next_slot || {};
  const nextCountdown = nextSlot.countdown || {};
  const queued = status.queued_release || {};
  const leaderboard = status.latest_leaderboard_row || {};
  const week = Array.isArray(status.weekly_checklist) ? status.weekly_checklist : [];
  const history = Array.isArray(status.history) ? status.history : [];

  const moduleLabel = document.getElementById("curriculumModuleLabel");
  if (moduleLabel) moduleLabel.textContent = `Module ${module.number || "?"}: ${module.label || "n/a"}`;
  const todayLabel = document.getElementById("curriculumTodayLabel");
  if (todayLabel) {
    if (nextSlot.label) {
      todayLabel.textContent = `${nextSlot.label} on ${nextSlot.release_date}`;
    } else if (slot.label) {
      todayLabel.textContent = `${slot.label} on ${slot.release_date}`;
    } else {
      todayLabel.textContent = "No scheduled release";
    }
  }
  const queuedStatus = document.getElementById("curriculumQueuedStatus");
  if (queuedStatus) {
    if (queued.id) {
      queuedStatus.textContent = `Queued status: ${queued.status || "queued"}`;
    } else if (nextSlot.scheduled_publish_at_iso) {
      queuedStatus.textContent = `Countdown: ${nextCountdown.label || "0d 0h"} until ${nextSlot.scheduled_publish_at_iso}`;
    } else {
      queuedStatus.textContent = "No queued release for today";
    }
  }
  const leaderboardMetric = document.getElementById("curriculumLeaderboardMetric");
  if (leaderboardMetric) leaderboardMetric.textContent = leaderboard.mean_quality_score ? `${Number(leaderboard.mean_quality_score).toFixed(3)} quality` : "n/a";
  const leaderboardMeta = document.getElementById("curriculumLeaderboardMeta");
  if (leaderboardMeta) leaderboardMeta.textContent = leaderboard.run_group_name
    ? `${Math.round(Number(leaderboard.deterministic_pass_rate || 0) * 100)}% pass • ${leaderboard.run_group_name}`
    : "No published trainee leaderboard row yet";

  const weekChecklist = document.getElementById("curriculumWeekChecklist");
  if (weekChecklist) {
    weekChecklist.innerHTML = week.length
      ? week.map((item) => `<div class="curriculum-list-item"><strong>${escapeHtml(item.date || "")}</strong> &nbsp;${escapeHtml(item.label || "")} <span class="muted"> &mdash; ${escapeHtml(item.status || "scheduled")}</span></div>`).join("")
      : "<div class='muted'>No weekly checklist available.</div>";
  }

  const checkpointLinks = document.getElementById("curriculumCheckpointLinks");
  if (checkpointLinks) {
    checkpointLinks.innerHTML = history.length
      ? history.slice(0, 5).map((item) => {
          const summary = item.summary || {};
          return `<div class="curriculum-list-item">
            <strong>${escapeHtml(item.release_date || "")}</strong>
            <span>${escapeHtml(item.release_kind || "")} • Module ${escapeHtml(String(summary.module_number || item.module_id || "?"))}</span>
            <span class="muted">tag=${escapeHtml(item.git_tag || "n/a")} • commit=${escapeHtml((item.commit_sha || "n/a").slice(0, 12))}</span>
          </div>`;
        }).join("")
      : "<div class='muted'>No published checkpoints yet.</div>";
  }

  if (curriculumStatusMessage) {
    curriculumStatusMessage.textContent = queued.id
      ? `Queued release ${queued.id} is waiting for publish time.`
      : "Curriculum status loaded.";
  }
}

function renderCurriculumHistory() {
  const container = document.getElementById("curriculumHistoryList");
  if (!container) return;
  if (!latestCurriculumHistory.length) {
    container.innerHTML = "<div class='muted'>No curriculum checkpoints yet.</div>";
    return;
  }
  container.innerHTML = latestCurriculumHistory.map((item) => {
    const summary = item.summary || {};
    const pr = item.pr || {};
    return `<div class="history-card">
      <div class="history-card-header">
        <strong>${escapeHtml(item.release_date || "")}</strong>
        <span class="history-badge">${escapeHtml(item.release_kind || "")}</span>
      </div>
      <div>Module ${escapeHtml(String(summary.module_number || item.module_id || "?"))}</div>
      <div class="muted">Quality ${Number(summary.mean_quality_score || 0).toFixed(3)} • Pass ${escapeHtml(String(summary.pass_count || 0))}/${escapeHtml(String(summary.case_count || 0))}</div>
      <div class="muted">branch=${escapeHtml(item.git_branch || "n/a")} • tag=${escapeHtml(item.git_tag || "n/a")}</div>
      <div class="muted">commit=${escapeHtml((item.commit_sha || "n/a").slice(0, 12))} • manifest=${escapeHtml(item.manifest_path || "n/a")}</div>
      <div class="muted">pr=${escapeHtml(pr.url || pr.number || "n/a")}</div>
    </div>`;
  }).join("");
}

async function submitCurriculumRun() {
  const response = await fetchApi("/api/curriculum/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: "anthropic/claude-opus-4.6" }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to submit curriculum run");
  await refreshCurriculumStatus();
  await refreshCurriculumHistory();
  return data;
}

async function publishCurriculumRuns() {
  const response = await fetchApi("/api/curriculum/publish-due", { method: "POST" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to publish curriculum releases");
  await refreshCurriculumStatus();
  await refreshCurriculumHistory();
  return data;
}


function showHarnessSettingsModal() {
  const modal = document.getElementById("harnessSettingsModal");
  if (modal) modal.style.display = "";
}

function hideHarnessSettingsModal() {
  const modal = document.getElementById("harnessSettingsModal");
  if (modal) modal.style.display = "none";
}

// ---------------------------------------------------------------------------
// Pipeline Editor
// ---------------------------------------------------------------------------
let currentHarnessConfig = null;

async function loadHarnessConfig() {
  try {
    const resp = await fetchApi("/api/harness/config?name=default");
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

function showPipelineEditor() {
  const modal = document.getElementById("pipelineEditorModal");
  if (modal) modal.style.display = "";
  loadHarnessConfig().then((config) => {
    if (config) {
      currentHarnessConfig = config;
      renderHarnessEditor();
    }
  });
}

function hidePipelineEditor() {
  const modal = document.getElementById("pipelineEditorModal");
  if (modal) modal.style.display = "none";
}

function renderHarnessEditor() {
  if (!currentHarnessConfig) return;
  renderModuleList("preLoopModuleList", currentHarnessConfig.pre_loop_modules || [], "pre_loop");
  renderModuleList("postStepModuleList", currentHarnessConfig.post_step_modules || [], "post_step");
  document.getElementById("pipelineEditorStatus").textContent = "";
}

function renderModuleList(containerId, modules, phase) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  modules.forEach((mod, idx) => {
    const card = document.createElement("div");
    card.className = `module-card kind-${mod.kind || "deterministic"}${mod.enabled === false ? " disabled" : ""}`;
    const isValidator = (mod.group_key === "validators");

    let actionsHtml = "";
    if (mod.movable !== false) {
      actionsHtml += `<button class="btn-move-up" data-idx="${idx}" data-phase="${phase}" title="Move up">&uarr;</button>`;
      actionsHtml += `<button class="btn-move-down" data-idx="${idx}" data-phase="${phase}" title="Move down">&darr;</button>`;
    }
    if (mod.removable !== false) {
      actionsHtml += `<button class="btn-remove" data-idx="${idx}" data-phase="${phase}" title="Remove">&times;</button>`;
    }

    card.innerHTML = `
      <span class="module-kind-badge ${mod.kind || "deterministic"}">${mod.kind || "det"}</span>
      <span class="module-label">${mod.label || mod.id}</span>
      ${isValidator ? '<span class="validator-warning" title="Removing validators is risky">&#9888;</span>' : ""}
      <span class="module-actions">${actionsHtml}</span>
    `;
    container.appendChild(card);
  });

  // Attach move/remove handlers
  container.querySelectorAll(".btn-move-up").forEach((btn) =>
    btn.addEventListener("click", () => moveModule(phase, +btn.dataset.idx, -1))
  );
  container.querySelectorAll(".btn-move-down").forEach((btn) =>
    btn.addEventListener("click", () => moveModule(phase, +btn.dataset.idx, 1))
  );
  container.querySelectorAll(".btn-remove").forEach((btn) =>
    btn.addEventListener("click", () => removeModule(phase, +btn.dataset.idx))
  );
}

function getModuleArray(phase) {
  if (!currentHarnessConfig) return [];
  return phase === "pre_loop"
    ? currentHarnessConfig.pre_loop_modules
    : currentHarnessConfig.post_step_modules;
}

function moveModule(phase, idx, direction) {
  const arr = getModuleArray(phase);
  const newIdx = idx + direction;
  if (newIdx < 0 || newIdx >= arr.length) return;
  // Don't move past non-movable modules
  if (arr[newIdx].movable === false) return;
  [arr[idx], arr[newIdx]] = [arr[newIdx], arr[idx]];
  renderHarnessEditor();
}

function removeModule(phase, idx) {
  const arr = getModuleArray(phase);
  const mod = arr[idx];
  if (mod && mod.group_key === "validators") {
    if (!confirm(`Removing validator "${mod.label}" may reduce mechanism quality. Continue?`)) return;
  }
  arr.splice(idx, 1);
  renderHarnessEditor();
}

async function saveHarnessConfig() {
  if (!currentHarnessConfig) return;
  const statusEl = document.getElementById("pipelineEditorStatus");
  try {
    const resp = await fetchApi("/api/harness/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentHarnessConfig),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      statusEl.textContent = `Error: ${err.detail || resp.statusText}`;
      return;
    }
    const result = await resp.json();
    statusEl.textContent = `Saved as "${result.name}" (version: ${(result.version || "").slice(0, 12)})`;
  } catch (e) {
    statusEl.textContent = `Save failed: ${e.message}`;
  }
}

async function resetHarnessToDefault() {
  const config = await loadHarnessConfig();
  if (config) {
    currentHarnessConfig = config;
    renderHarnessEditor();
    document.getElementById("pipelineEditorStatus").textContent = "Reset to default.";
  }
}

// ---------------------------------------------------------------------------
// Add Module Wizard
// ---------------------------------------------------------------------------
let wizardState = {};

function showAddModuleWizard() {
  wizardState = { name: "", kind: "llm", phase: "pre_loop", inputs: [], text: "" };
  document.getElementById("addModuleModal").style.display = "";
  document.getElementById("wizardStep1").style.display = "";
  document.getElementById("wizardStep2").style.display = "none";
  document.getElementById("wizardStep3").style.display = "none";
  document.getElementById("wizModuleName").value = "";
  document.getElementById("wizModuleKind").value = "llm";
  document.getElementById("wizModulePhase").value = "pre_loop";
}

function hideAddModuleWizard() {
  document.getElementById("addModuleModal").style.display = "none";
}

function wizardNextStep1() {
  const name = (document.getElementById("wizModuleName").value || "").trim();
  if (!name) { alert("Module name is required."); return; }
  wizardState.name = name;
  wizardState.kind = document.getElementById("wizModuleKind").value;
  wizardState.phase = document.getElementById("wizModulePhase").value;

  // Build input checkboxes from upstream modules
  const upstream = getModuleArray(wizardState.phase);
  const container = document.getElementById("wizInputCheckboxes");
  container.innerHTML = "";
  upstream.forEach((m) => {
    if (m.enabled === false) return;
    const lbl = document.createElement("label");
    lbl.innerHTML = `<input type="checkbox" value="${m.id}"> ${m.label || m.id}`;
    container.appendChild(lbl);
  });

  document.getElementById("wizardStep1").style.display = "none";
  document.getElementById("wizardStep2").style.display = "";
}

function wizardNextStep2() {
  const checks = document.querySelectorAll("#wizInputCheckboxes input:checked");
  wizardState.inputs = Array.from(checks).map((c) => c.value);

  document.getElementById("wizStep3Title").textContent =
    wizardState.kind === "llm" ? "Write Prompt" : "Write Code";
  document.getElementById("wizModuleText").placeholder =
    wizardState.kind === "llm"
      ? "Enter the system prompt for this LLM module..."
      : "Enter Python code. Use `context` dict for inputs, return a dict.";
  document.getElementById("wizModuleText").value = "";

  document.getElementById("wizardStep2").style.display = "none";
  document.getElementById("wizardStep3").style.display = "";
}

function wizardFinish() {
  const text = (document.getElementById("wizModuleText").value || "").trim();
  const newModule = {
    id: wizardState.name.replace(/\s+/g, "_").toLowerCase(),
    label: wizardState.name,
    kind: wizardState.kind,
    phase: wizardState.phase,
    enabled: true,
    step_name: wizardState.name.replace(/\s+/g, "_").toLowerCase(),
    inputs: wizardState.inputs,
    outputs: [wizardState.name.replace(/\s+/g, "_").toLowerCase()],
    movable: true,
    removable: true,
    custom: true,
    description: `Custom ${wizardState.kind} module`,
  };
  if (wizardState.kind === "llm") {
    newModule.prompt_text = text;
  } else {
    newModule.code_text = text;
  }

  const arr = getModuleArray(wizardState.phase);
  arr.push(newModule);
  renderHarnessEditor();
  hideAddModuleWizard();
}

function setStatus(text) {
  runStatusEl.textContent = text;
}

function appendTerminalLine(kind, stepName, message, level = "info") {
  if (!terminalOutput) return;
  const ts = new Date().toLocaleTimeString();
  const line = document.createElement("div");
  const lvlClass = level === "error" ? "t-error" : level === "warn" ? "t-warn" : level === "success" ? "t-success" : "";
  line.className = `terminal-line ${lvlClass}`;
  line.innerHTML = `<span class="t-ts">${ts}</span><span class="t-kind">${escapeHtml(kind)}</span>${stepName ? `<span class="t-step">${escapeHtml(stepName)}</span>` : ""}<span class="t-msg">${escapeHtml(message)}</span>`;
  terminalOutput.appendChild(line);
  terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function appendEventToTerminal(event) {
  const kind = event.event_type || "event";
  const step = event.step_name || "";
  const payload = event.payload || {};
  let msg = "";
  if (payload.message) msg = payload.message;
  else if (payload.error) msg = payload.error;
  else if (Object.keys(payload).length) msg = JSON.stringify(payload);

  const errorKinds = ["run_failed", "step_failed", "mechanism_retry_failed", "mechanism_retry_exhausted"];
  const warnKinds = ["run_paused", "ralph_budget_warning", "runtime_limit"];
  const successKinds = ["run_completed", "target_products_detected"];

  let level = "info";
  if (errorKinds.includes(kind)) level = "error";
  else if (warnKinds.includes(kind)) level = "warn";
  else if (successKinds.includes(kind)) level = "success";

  appendTerminalLine(kind, step, msg, level);

  if (successKinds.includes(kind)) showToast(kind.replace(/_/g, " "), "success");
  else if (errorKinds.includes(kind)) showToast(`${kind.replace(/_/g, " ")}${msg ? ": " + msg.slice(0, 80) : ""}`, "error");
  else if (kind === "run_paused") showToast("Run paused — awaiting decision", "info");
}

function showToast(message, type = "info") {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 3600);
}

function updateButtons(snapshotStatus = "") {
  const hasRun = Boolean(runId);
  const terminal = ["completed", "failed", "stopped"];
  const paused = snapshotStatus === "paused";
  const isTerminal = hasRun && terminal.includes(snapshotStatus);

  // Determine pause type from the latest pause details
  const pauseDetails = latestPauseInfo && latestPauseInfo.details ? latestPauseInfo.details : {};
  const hasAlternative = paused && Boolean(pauseDetails.has_alternative);
  const isDeadEnd = paused && pauseDetails.has_alternative === false;

  document.getElementById("createBtn").disabled = false;
  document.getElementById("startBtn").disabled = !hasRun || ["running", "paused", ...terminal].includes(snapshotStatus);
  document.getElementById("stopBtn").disabled = !hasRun || snapshotStatus !== "running";

  // Continue only shown when paused with an alternative available
  const resumeContinueBtn = document.getElementById("resumeContinueBtn");
  resumeContinueBtn.disabled = !hasRun || !hasAlternative;
  resumeContinueBtn.style.display = (paused && !isDeadEnd) || !paused ? "" : "none";

  // Cancel always shown when paused
  document.getElementById("resumeStopBtn").disabled = !hasRun || !paused;

  document.getElementById("evaluateBtn").disabled = !isTerminal;
  document.getElementById("saveEvaluationBtn").disabled = !hasRun || !latestEvaluation;
  document.getElementById("applyHarnessBtn").disabled = !hasRun || !latestEvaluation;
  document.getElementById("createPrBtn").disabled = !hasRun;
  const storeFewShotBtn = document.getElementById("storeFewShotBtn");
  if (storeFewShotBtn) storeFewShotBtn.disabled = !isTerminal;
  const evaluationSection = document.getElementById("evaluationSection");
  const dryRunPanel = document.getElementById("dryRunPanel");
  if (evaluationSection) evaluationSection.style.display = isTerminal ? "" : "none";
  if (dryRunPanel) dryRunPanel.style.display = isTerminal && isDryRun ? "" : "none";
  // Disable PR creation during dry runs — evidence will be discarded
  const createPrBtn = document.getElementById("createPrBtn");
  if (createPrBtn) createPrBtn.disabled = !hasRun || isDryRun;

  // Pause notification banner
  const pauseNotification = document.getElementById("pauseNotification");
  if (pauseNotification) {
    const pauseDetails = (latestPauseInfo && latestPauseInfo.details) || {};
    const failedChecks = Array.isArray(pauseDetails.failed_checks) ? pauseDetails.failed_checks : [];
    const rescueAttempted = !!pauseDetails.rescue_attempted;
    const rescueOutcome = pauseDetails.rescue_outcome || "none";
    const checksText = failedChecks.length ? ` Failed checks: ${failedChecks.join(", ")}.` : "";
    const rescueText = rescueAttempted ? ` Rescue: ${rescueOutcome}.` : "";
    if (paused && hasAlternative) {
      pauseNotification.textContent = "Validation failed on all candidates. One alternative reaction is available as a last resort — click \"Try Next Best Reaction\" to attempt it, or \"Cancel Run\" to stop.";
      pauseNotification.style.display = "";
    } else if (paused && isDeadEnd) {
      pauseNotification.textContent = `No viable reactions remain. The run has reached a dead end and cannot continue.${checksText}${rescueText}`;
      pauseNotification.style.display = "";
    } else {
      pauseNotification.textContent = "";
      pauseNotification.style.display = "none";
    }
  }
}

function showNodeDetail(nodeId) {
  if (!latestFlow || !Array.isArray(latestFlow.nodes)) {
    nodeDetailPanel.style.display = "none";
    return;
  }
  const node = latestFlow.nodes.find((n) => n.id === nodeId);
  if (!node) {
    nodeDetailPanel.style.display = "none";
    return;
  }

  nodeDetailPanel.style.display = "block";
  const kindLabel = node.kind === "llm" ? "LLM Call" : node.kind === "decision" ? "Decision Gate" : "Deterministic - no LLM";
  nodeDetailTitle.textContent = `${node.label} (${kindLabel})`;

  if (node.kind === "llm") {
    const ref = node.prompt_ref;
    const io = NODE_IO_SCHEMAS[nodeId] || { inputs: [], outputs: [] };
    let html = "";
    if (ref) {
      html += `<div class="step-meta">${ref.name || nodeId}</div>`;
    } else {
      html += `<div class="step-meta muted">No prompt registered for this step.</div>`;
    }
    html += `<div class="node-io-grid">`;
    html += `<div><strong>Inputs</strong><ul>${io.inputs.map((i) => `<li class="muted">${escapeHtml(i)}</li>`).join("")}</ul></div>`;
    html += `<div><strong>Outputs</strong><ul>${io.outputs.map((o) => `<li class="muted">${escapeHtml(o)}</li>`).join("")}</ul></div>`;
    html += `</div>`;
    const template = getTemplateForNode(nodeId, ref);
    if (template) {
      html += `<pre class="code">${escapeHtml(template)}</pre>`;
    }
    nodeDetailBody.innerHTML = html;
  } else {
    const desc = NODE_DESCRIPTIONS[nodeId];
    nodeDetailBody.innerHTML = `<p class="muted">${desc || "No description available for this step."}</p>`;
  }
}

function getTemplateForNode(nodeId, promptRef) {
  if (selectedPromptVersion !== "latest" && versionTemplateCache[nodeId]) {
    return versionTemplateCache[nodeId];
  }
  return promptRef ? promptRef.template : null;
}

function getFlowNodeStyle(kind, state) {
  const resolvedKind = kind || "deterministic";
  const resolvedState = state || "pending";
  let fill = FLOW_KIND_FILL[resolvedKind] || "#f0f4f8";
  let stroke = FLOW_KIND_STROKE[resolvedKind] || "#999";
  let strokeWidth = 1;
  let opacity = resolvedState === "pending" ? 0.6 : 1;

  if (resolvedState === "active") {
    strokeWidth = 3;
  } else if (resolvedState === "completed") {
    strokeWidth = 2;
  } else if (resolvedState === "retrying") {
    fill = "#fff7d6";
    stroke = "#a16207";
    strokeWidth = 2;
  } else if (resolvedState === "failed") {
    fill = "#fee2e2";
    stroke = "#b91c1c";
    strokeWidth = 2;
  } else if (resolvedState === "paused") {
    fill = "#f3e8ff";
    stroke = "#7c3aed";
    strokeWidth = 2;
  }

  return { fill, stroke, strokeWidth, opacity };
}

function resolveNodeKind(node) {
  if (!node || typeof node !== "object") return "deterministic";
  return node.kind || FLOW_KIND_BY_NODE_ID[node.id] || "deterministic";
}

function findNodeLabel(stepName) {
  if (!latestFlow || !Array.isArray(latestFlow.nodes)) return stepName;
  const direct = latestFlow.nodes.find((node) => node.id === stepName || node.step_name === stepName);
  return direct ? direct.label : stepName;
}

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatList(list, limit = 3) {
  const values = Array.isArray(list) ? list.filter(Boolean).map((item) => String(item)) : [];
  if (!values.length) return "";
  return values.slice(0, limit).join(", ");
}

function candidateName(item) {
  if (!item) return "";
  if (typeof item === "string") return item;
  if (typeof item === "object") return String(item.name || item.smiles || "").trim();
  return "";
}

function summariseStepOutput(step) {
  const output = step && typeof step.output === "object" && step.output ? step.output : {};
  const name = String(step.step_name || "");
  const validation = step.validation && typeof step.validation === "object" ? step.validation : null;

  // Helper function to extract error details from validation checks
  function getValidationError(checkName) {
    if (!validation || !Array.isArray(validation.checks)) return null;
    const check = validation.checks.find(c => c.name === checkName);
    if (!check || check.passed) return null;
    const details = check.details || {};
    return details.error || details.message || "Validation failed";
  }

  // Helper function to format deficit/surplus info
  function formatBalanceInfo(details) {
    if (!details) return "";
    const deficit = details.deficit || {};
    const surplus = details.surplus || {};
    const deficitItems = Object.entries(deficit).map(([atom, count]) => `${atom}:${count}`).join(", ");
    const surplusItems = Object.entries(surplus).map(([atom, count]) => `${atom}:${count}`).join(", ");
    const parts = [];
    if (deficitItems) parts.push(`deficit: ${deficitItems}`);
    if (surplusItems) parts.push(`surplus: ${surplusItems}`);
    return parts.length ? ` (${parts.join(", ")})` : "";
  }

  if (name === "ph_recommendation") {
    // Handle single pH value
    const ph = asNumber(output.recommended ?? output.representative_ph ?? output.recommended_ph ?? output.ph);
    if (ph !== null) {
      return `pH ${ph.toFixed(2)} recommended.`;
    }

    // Handle pH range
    if (Array.isArray(output.recommended_range) && output.recommended_range.length >= 2) {
      const [min, max] = output.recommended_range;
      return `pH ${min.toFixed(1)}-${max.toFixed(1)} range recommended.`;
    }

    return "pH recommendation generated.";
  }

  if (name === "initial_conditions") {
    const env = output.environment ? String(output.environment) : "undetermined";
    const ph = asNumber(output.representative_ph);
    const acids = Array.isArray(output.acid_candidates) ? output.acid_candidates.map(candidateName).filter(Boolean) : [];
    const bases = Array.isArray(output.base_candidates) ? output.base_candidates.map(candidateName).filter(Boolean) : [];
    const candidateSummary = acids.length
      ? `Acid supports: ${acids.slice(0, 2).join(", ")}.`
      : bases.length
        ? `Base supports: ${bases.slice(0, 2).join(", ")}.`
        : "No additive candidates returned.";
    const phSummary = ph !== null ? `pH ${ph.toFixed(2)}.` : "No pH value returned.";
    return `${env} environment. ${phSummary} ${candidateSummary}`;
  }

  if (name === "missing_reagents") {
    const reactants = output.suggested_reactants || output.missing_reactants || [];
    const products = output.suggested_products || output.missing_products || [];
    const reactantText = formatList(reactants, 3);
    const productText = formatList(products, 3);
    if (!reactantText && !productText) return "No additional molecules required for atom balance.";
    return `Reactant additions: ${reactantText || "none"}. Product additions: ${productText || "none"}.`;
  }

  if (name === "atom_mapping") {
    const mapping = output.llm_response && typeof output.llm_response === "object" ? output.llm_response : output;
    const confidence = mapping.confidence ? String(mapping.confidence) : "unknown";
    const mappedCount = Array.isArray(mapping.mapped_atoms) ? mapping.mapped_atoms.length : 0;
    const unmappedCount = Array.isArray(mapping.unmapped_atoms) ? mapping.unmapped_atoms.length : 0;
    return `Atom mapping confidence: ${confidence}. Mapped atoms: ${mappedCount}. Unmapped atoms: ${unmappedCount}.`;
  }

  if (name === "reaction_type_mapping") {
    const label = output.selected_label_exact ? String(output.selected_label_exact) : "no_match";
    const confidence = asNumber(output.confidence);
    const mode = output.selected_type_id ? String(output.selected_type_id) : "n/a";
    const confidenceText = confidence !== null ? confidence.toFixed(2) : "n/a";
    return `Reaction type: ${label} (${mode}), confidence ${confidenceText}.`;
  }

  if (name === "mechanism_step_proposal") {
    const candidates = Array.isArray(output.candidates) ? output.candidates : [];
    if (candidates.length) {
      const ranked = [...candidates].sort((a, b) => Number(a.rank || 99) - Number(b.rank || 99));
      const top = ranked[0] || {};
      const topSmiles = top.intermediate_smiles ? String(top.intermediate_smiles) : "n/a";
      const topDescription = top.reaction_description ? String(top.reaction_description) : "no description";
      return `${candidates.length} candidate(s) proposed. Top candidate: ${topSmiles} (${topDescription}).`;
    }
    const proposed = formatList(output.proposed_intermediates || [], 2);
    return proposed ? `Intermediate proposal: ${proposed}.` : "No mechanism candidate proposed.";
  }

  if (name === "mechanism_synthesis") {
    const targetReached = Boolean(output.contains_target_product);
    const intermediate = output.predicted_intermediate ? String(output.predicted_intermediate) : "n/a";

    // Extract failed checks from validation
    const failedChecks = [];
    if (validation && Array.isArray(validation.checks)) {
      for (const check of validation.checks) {
        if (!check.passed) {
          failedChecks.push(check.name);
        }
      }
    }

    // Check for retry feedback with guidance
    const retryFeedback = output.retry_feedback || {};
    const guidance = retryFeedback.guidance || "";
    const failedCheckNames = retryFeedback.failed_checks || failedChecks;

    let statusText = targetReached ? "Target product reached." : "Mechanism advanced.";
    if (failedCheckNames.length) {
      statusText += ` Failed: ${failedCheckNames.join(", ")}.`;
    }
    if (guidance) {
      statusText += ` Guidance: ${guidance}`;
    }

    return `Predicted intermediate: ${intermediate}. ${statusText}`.trim();
  }

  if (name === "bond_electron_validation") {
    if (validation && validation.passed) {
      return "Bond/electron conservation check passed.";
    }
    const error = getValidationError("dbe_metadata");
    const details = validation && Array.isArray(validation.checks) ?
      validation.checks.find(c => c.name === "dbe_metadata")?.details : null;
    const source = details?.dbe_source ? ` (${details.dbe_source})` : "";
    return error ? `Bond/electron conservation failed: ${error}${source}` : "Bond/electron conservation check failed.";
  }

  if (name === "atom_balance_validation") {
    if (validation && validation.passed) {
      return "Step-level atom balance check passed.";
    }
    const details = validation && Array.isArray(validation.checks) ?
      validation.checks.find(c => c.name === "atom_balance")?.details : null;
    const balanceInfo = formatBalanceInfo(details);
    return `Step-level atom balance failed${balanceInfo}.`;
  }

  if (name === "state_progress_validation") {
    if (validation && validation.passed) {
      return "State-progress check passed (forward progress detected).";
    }
    const details = validation && Array.isArray(validation.checks) ?
      validation.checks.find(c => c.name === "state_progress")?.details : null;
    if (details) {
      const reasons = [];
      if (details.unchanged_starting_materials_detected) reasons.push("unchanged starting materials");
      if (!details.resulting_state_changed) reasons.push("no state change");
      if (reasons.length) return `State-progress failed: ${reasons.join(", ")}.`;
    }
    return "State-progress check failed (no forward progress).";
  }

  if (name === "reflection") {
    const warnings = Array.isArray(output.validation_warnings) ? output.validation_warnings : [];
    if (!warnings.length) return "No validation warnings recorded.";
    // Show actual warning messages, not just count
    const warningTexts = warnings.slice(0, 3).map(w => String(w)).filter(Boolean);
    return `Warnings: ${warningTexts.join("; ")}${warnings.length > 3 ? "..." : ""}`;
  }

  if (name === "balance_analysis") {
    const balanced = output.balanced;
    if (balanced === true) return "Reaction atom balance verified.";
    if (balanced === false) {
      const details = output.details || {};
      const balanceInfo = formatBalanceInfo(details);
      return `Reaction atom imbalance detected${balanceInfo}.`;
    }
    return "Atom balance analysis completed.";
  }

  if (name === "functional_groups") {
    const groups = output.functional_groups || {};
    const groupCount = Object.keys(groups).length;
    const leaving = Array.isArray(output.leaving_groups) ? output.leaving_groups.length : 0;
    return `Identified ${groupCount} functional group type(s) and ${leaving} leaving group(s).`;
  }

  if (output.message) return String(output.message);
  if (output.status) return `Status: ${String(output.status)}.`;

  // Try to extract meaningful info from any available fields
  if (output.error) return `Error: ${String(output.error)}`;
  if (output.result) return `Result: ${String(output.result)}`;
  if (Object.keys(output).length > 0) {
    const keys = Object.keys(output).slice(0, 2);
    return `${name}: ${keys.join(", ")} data recorded.`;
  }

  return "Step output recorded.";
}

// Global variable to track which step detail is currently open
let currentlyOpenStepIndex = -1;

function renderRunStepSummaries(snapshot) {
  if (!runStepSummaries) return;
  const steps = Array.isArray(snapshot.step_outputs) ? snapshot.step_outputs : [];
  if (nodeDetailPanel) nodeDetailPanel.style.display = "block";
  if (!steps.length) {
    runStepSummaries.classList.add("muted");
    runStepSummaries.innerHTML = "Start a run to see step-by-step summaries.";
    return;
  }

  // Helper function to determine step status
  function getStepStatus(step) {
    const name = String(step.step_name || "");
    const validation = step.validation && typeof step.validation === "object" ? step.validation : null;
    const output = step && typeof step.output === "object" && step.output ? step.output : {};

    // Check validation status
    if (validation && Array.isArray(validation.checks)) {
      const hasFailedChecks = validation.checks.some(check => !check.passed);
      if (hasFailedChecks) return "error";
    }

    // Check for specific error conditions
    if (name === "reflection" && Array.isArray(output.validation_warnings) && output.validation_warnings.length > 0) {
      return "warning";
    }

    // Check for validation passed
    if (validation && validation.passed) {
      return "success";
    }

    // Check output for error indicators
    if (output.error || output.failed_checks) {
      return "error";
    }

    return "neutral";
  }

  runStepSummaries.classList.remove("muted");
  const total = steps.length;

  // Create the HTML for all details elements
  const detailsHtml = steps
    .map((step, index) => {
      const label = findNodeLabel(String(step.step_name || ""));
      const attempt = Number(step.attempt || 0);
      const retry = Number(step.retry_index || 0);
      const retrySuffix = retry > 0 ? `, retry ${retry}` : "";
      const meta = attempt > 0 ? `attempt ${attempt}${retrySuffix}` : `entry ${index + 1}`;
      const summary = summariseStepOutput(step);
      const status = getStepStatus(step);
      const statusClass = status !== "neutral" ? ` has-${status}` : "";
      const open = index === currentlyOpenStepIndex ? " open" : "";
      return `
        <details class="run-summary-item${statusClass}" data-step-index="${index}"${open}>
          <summary>
            <span class="run-summary-step">${escapeHtml(label)}</span>
            <span class="run-summary-meta">${escapeHtml(meta)}</span>
          </summary>
          <div class="run-summary-text">${escapeHtml(summary)}</div>
        </details>
      `;
    })
    .join("");

  runStepSummaries.innerHTML = detailsHtml;

  // Set up accordion behavior - only allow one details element to be open at a time
  const allDetails = runStepSummaries.querySelectorAll("details");
  allDetails.forEach((details, index) => {
    details.addEventListener("toggle", () => {
      if (details.open) {
        // Close all other details elements
        allDetails.forEach((otherDetails, otherIndex) => {
          if (otherDetails !== details && otherDetails.open) {
            otherDetails.open = false;
          }
        });
        // Update the currently open index
        currentlyOpenStepIndex = index;
      } else {
        // If this details was closed and it was the currently open one, reset the index
        if (currentlyOpenStepIndex === index) {
          currentlyOpenStepIndex = -1;
        }
      }
    });
  });
}

async function updateMermaidPreview() {
  const familyEl = document.getElementById("modelFamilyInput");
  const modelNameEl = document.getElementById("modelNameInput");
  const reasoningEl = document.getElementById("reasoningInput");

  // Exit early if elements don't exist yet (during bootstrap)
  if (!familyEl || !modelNameEl || !reasoningEl) return;

  const family = familyEl.value;
  const modelName = modelNameEl.value;
  const reasoning = reasoningEl.value;

  updateThinkingVisibility();

  // Don't override model map while a run is active
  if (runId) return;

  const params = new URLSearchParams({ model_name: modelName });
  if (reasoning) params.set("thinking_level", reasoning);

  try {
    const [previewResp, flowResp] = await Promise.all([
      fetch(`/api/catalog/preview_step_models?${params}`),
      fetchApi("/api/catalog/flow_template"),
    ]);
    if (!previewResp.ok || !flowResp.ok) return;

    const previewData = await previewResp.json();
    const flowData = await flowResp.json();

    // Make nodes clickable in preview mode (before any run)
    latestFlow = flowData;

    // Populate stepModelMap from preview
    stepModelMap = {};
    const previewModels = previewData.step_models || {};
    Object.entries(previewModels).forEach(([step, model]) => {
      stepModelMap[step] = model;
    });

    // Render mermaid from template flow
    const nodes = Array.isArray(flowData.nodes) ? flowData.nodes : [];
    const edges = Array.isArray(flowData.edges) ? flowData.edges : [];
    if (!window.mermaid || !nodes.length) return;

    const MODEL_SYMBOLS = ["\u00b9", "\u00b2", "\u00b3", "\u2074", "\u2075", "\u2076", "\u2077", "\u2078"];
    const uniqueModels = [];
    nodes.forEach((node) => {
      const model = stepModelMap[node.id];
      if (model && !uniqueModels.includes(model)) uniqueModels.push(model);
    });
    const modelSymbolMap = {};
    uniqueModels.forEach((model, i) => {
      modelSymbolMap[model] = MODEL_SYMBOLS[i] || `[${i + 1}]`;
    });

    const lines = ["flowchart TD"];
    nodes.forEach((node) => {
      const model = stepModelMap[node.id];
      const sym = model ? ` ${modelSymbolMap[model]}` : "";
      lines.push(`  ${node.id}["${node.label}${sym}"]`);
    });
    edges.forEach((edge) => {
      const label = edge.label ? `|${edge.label}|` : "";
      lines.push(`  ${edge.source} -->${label} ${edge.target}`);
    });

    nodes.forEach((node) => {
      const style = getFlowNodeStyle(resolveNodeKind(node), "pending");
      lines.push(`  style ${node.id} fill:${style.fill},stroke:${style.stroke},stroke-width:${style.strokeWidth}px,opacity:${style.opacity}`);
    });

    const graphText = lines.join("\n");
    const renderId = `preview-${Date.now()}`;
    const rendered = await window.mermaid.render(renderId, graphText);
    flowDiagram.innerHTML = rendered.svg;

    // Set cursor and re-apply selection highlight
    nodes.forEach((node) => {
      const svgEl = flowDiagram.querySelector(`[id^="flowchart-${node.id}-"]`);
      if (svgEl) svgEl.style.cursor = "pointer";
    });
    if (lastSelectedNodeId) {
      const prevGroup = flowDiagram.querySelector(`[id^="flowchart-${lastSelectedNodeId}-"]`);
      if (prevGroup) prevGroup.classList.add("flow-node-selected");
    }

    // Update model key below the diagram
    const modelKeyEl = document.getElementById("modelKey");
    if (modelKeyEl) {
      const entries = Object.entries(modelSymbolMap);
      if (entries.length) {
        const shortName = (m) => (m.includes("/") ? m.split("/").pop() : m);
        modelKeyEl.innerHTML = entries
          .map(([model, sym]) => `<span class="model-key-item" title="${model}"><strong>${sym}</strong>\u00a0${shortName(model)}</span>`)
          .join("");
        modelKeyEl.style.display = "flex";
      } else {
        modelKeyEl.style.display = "none";
      }
    }
  } catch (_) {
    /* ignore preview errors */
  }
}

async function renderFlow() {
  if (!runId) return;
  const response = await fetch(`/api/runs/${runId}/flow`);
  if (!response.ok) return;
  latestFlow = await response.json();

  const nodes = Array.isArray(latestFlow.nodes) ? latestFlow.nodes : [];
  const edges = Array.isArray(latestFlow.edges) ? latestFlow.edges : [];
  if (!window.mermaid || !nodes.length) {
    flowDiagram.innerHTML = "<div class='muted'>Flow chart unavailable.</div>";
    return;
  }

  // Build model → superscript symbol map from stepModelMap
  const MODEL_SYMBOLS = ["\u00b9", "\u00b2", "\u00b3", "\u2074", "\u2075", "\u2076", "\u2077", "\u2078"];
  const uniqueModels = [];
  nodes.forEach((node) => {
    const model = stepModelMap[node.id];
    if (model && !uniqueModels.includes(model)) uniqueModels.push(model);
  });
  const modelSymbolMap = {};
  uniqueModels.forEach((model, i) => {
    modelSymbolMap[model] = MODEL_SYMBOLS[i] || `[${i + 1}]`;
  });

  const lines = ["flowchart TD"];
  nodes.forEach((node) => {
    const model = stepModelMap[node.id];
    const sym = model ? ` ${modelSymbolMap[model]}` : "";
    lines.push(`  ${node.id}["${node.label}${sym}"]`);
  });
  edges.forEach((edge) => {
    const label = edge.label ? `|${edge.label}|` : "";
    lines.push(`  ${edge.source} -->${label} ${edge.target}`);
  });

  nodes.forEach((node) => {
    const style = getFlowNodeStyle(resolveNodeKind(node), node.state);
    lines.push(`  style ${node.id} fill:${style.fill},stroke:${style.stroke},stroke-width:${style.strokeWidth}px,opacity:${style.opacity}`);
  });

  const graphText = lines.join("\n");
  const renderId = `flow-${Date.now()}`;
  const rendered = await window.mermaid.render(renderId, graphText);
  flowDiagram.innerHTML = rendered.svg;

  // Set cursor on all node groups
  nodes.forEach((node) => {
    const svgEl = flowDiagram.querySelector(`[id^="flowchart-${node.id}-"]`);
    if (svgEl) svgEl.style.cursor = "pointer";
  });

  // Re-apply highlight if a node was previously selected
  if (lastSelectedNodeId) {
    const prevGroup = flowDiagram.querySelector(`[id^="flowchart-${lastSelectedNodeId}-"]`);
    if (prevGroup) prevGroup.classList.add("flow-node-selected");
  }

  // Update model key below the diagram
  const modelKeyEl = document.getElementById("modelKey");
  if (modelKeyEl) {
    const entries = Object.entries(modelSymbolMap);
    if (entries.length) {
      const shortName = (m) => (m.includes("/") ? m.split("/").pop() : m);
      modelKeyEl.innerHTML = entries
        .map(([model, sym]) => `<span class="model-key-item" title="${model}"><strong>${sym}</strong>\u00a0${shortName(model)}</span>`)
        .join("");
      modelKeyEl.style.display = "flex";
    } else {
      modelKeyEl.style.display = "none";
    }
  }
}

function createMoleculeCard(mol) {
  const card = document.createElement("div");
  card.className = "molecule-card";
  const image = mol.image_data ? `<img src="data:image/png;base64,${mol.image_data}" alt="${mol.smiles}" />` : "<div class='muted'>No image</div>";
  card.innerHTML = `
    ${image}
    <div class="molecule-title">${mol.smiles || "n/a"}</div>
    <div class="molecule-meta">${mol.formula || ""} ${mol.molecular_weight ? `- ${mol.molecular_weight} Da` : ""}</div>
  `;
  return card;
}

async function renderMoleculeCardsFromSmiles(smilesList, options = {}) {
  const smiles = Array.isArray(smilesList)
    ? smilesList.map((item) => String(item || "").trim())
    : [];
  if (!smiles.length) return [];
  const response = await fetchApi("/api/molecules/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      smiles,
      show_atom_numbers: options.showAtomNumbers === true,
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to render molecules (${response.status})`);
  }
  const data = await response.json();
  return Array.isArray(data.items) ? data.items : [];
}

async function _cardsForDisplay(molecules) {
  const items = Array.isArray(molecules) ? molecules : [];
  if (!showAtomNumbers) return items;
  const rendered = await renderMoleculeCardsFromSmiles(
    items.map((item) => item?.smiles || ""),
    { showAtomNumbers: true },
  );
  return items.map((item, index) => rendered[index] || item);
}

async function _cardForDisplay(molecule) {
  if (!molecule) return null;
  if (!showAtomNumbers) return molecule;
  const rendered = await renderMoleculeCardsFromSmiles([molecule.smiles || ""], { showAtomNumbers: true });
  return rendered[0] || molecule;
}

async function renderMoleculesFromInputs() {
  const startingCards = document.getElementById("startingCards");
  const productCards = document.getElementById("productCards");
  if (!startingCards || !productCards) return;
  const startingSmiles = parseSmilesList(document.getElementById("startingInput")?.value || "");
  const productSmiles = parseSmilesList(document.getElementById("productsInput")?.value || "");
  startingCards.innerHTML = "";
  productCards.innerHTML = "";
  if (!startingSmiles.length && !productSmiles.length) return;
  try {
    const [startRendered, prodRendered] = await Promise.all([
      renderMoleculeCardsFromSmiles(startingSmiles, { showAtomNumbers }),
      renderMoleculeCardsFromSmiles(productSmiles, { showAtomNumbers }),
    ]);
    startRendered.forEach((mol) => startingCards.appendChild(createMoleculeCard(mol)));
    prodRendered.forEach((mol) => productCards.appendChild(createMoleculeCard(mol)));
  } catch (_) {
    startingCards.innerHTML = "<div class='muted'>Could not render molecules.</div>";
  }
}

async function renderReactionVisuals(snapshot) {
  const visuals = snapshot.reaction_visuals || {};
  const startingCards = document.getElementById("startingCards");
  const productCards = document.getElementById("productCards");
  const mechanismCards = document.getElementById("mechanismCards");

  startingCards.innerHTML = "";
  productCards.innerHTML = "";
  mechanismCards.innerHTML = "";

  const startingVisuals = await _cardsForDisplay(visuals.starting_materials || []);
  const productVisuals = await _cardsForDisplay(visuals.products || []);
  startingVisuals.forEach((mol) => startingCards.appendChild(createMoleculeCard(mol)));
  productVisuals.forEach((mol) => productCards.appendChild(createMoleculeCard(mol)));

  const mechanismSummary = Array.isArray(snapshot.mechanism_summary) ? snapshot.mechanism_summary : [];
  if (!mechanismSummary.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "Mechanism will appear after mechanism steps are generated.";
    mechanismCards.appendChild(empty);
    return;
  }

  for (const entry of mechanismSummary) {
    const card = document.createElement("div");
    card.className = "step-card";
    const current = Array.isArray(entry.current_state_cards) ? entry.current_state_cards : [];
    const resulting = Array.isArray(entry.resulting_state_cards) ? entry.resulting_state_cards : [];
    const currentCards = await _cardsForDisplay(current);
    const resultingCards = await _cardsForDisplay(resulting);
    const intermediateCard = await _cardForDisplay(entry.intermediate_card);
    const currentHtml = currentCards.map((mol) => createMoleculeCard(mol).outerHTML).join("");
    const resultingHtml = resultingCards.map((mol) => createMoleculeCard(mol).outerHTML).join("");
    const intermediateHtml = intermediateCard ? createMoleculeCard(intermediateCard).outerHTML : "<div class='muted'>none</div>";

    card.innerHTML = `
      <div class="step-title">Mechanism attempt ${entry.attempt}</div>
      <div class="step-meta">contains target: ${entry.contains_target_product}</div>
      <div class="grid two">
        <div><div class="muted">Current state</div><div class="molecule-grid">${currentHtml}</div></div>
        <div><div class="muted">Resulting state</div><div class="molecule-grid">${resultingHtml}</div></div>
      </div>
      <div><div class="muted">Predicted intermediate</div><div class="molecule-grid">${intermediateHtml}</div></div>
    `;
    mechanismCards.appendChild(card);
  }

  // Render failed paths (from backtracking) in red
  const failedPaths = Array.isArray(snapshot.failed_paths) ? snapshot.failed_paths : [];
  failedPaths.forEach((fp) => {
    const card = document.createElement("div");
    card.className = "step-card failed-path";
    card.innerHTML = `
      <div class="step-title failed-label">Failed Path (branched at step ${fp.branch_step_index}, candidate rank ${fp.candidate_rank})</div>
      <div class="step-meta">${escapeHtml(fp.failure_reason || "validation failed")}</div>
      <div class="muted">Steps explored: ${fp.steps_in_path || 0}</div>
    `;
    mechanismCards.appendChild(card);
  });
}


async function loadCurriculumStatus() {
  const modelName = document.getElementById("modelNameInput")?.value || "";
  if (!modelName) {
    _curriculumStatus = null;
    populateExampleStepFilter();
    return;
  }
  try {
    const response = await fetchApi(`/api/curriculum/status?model_name=${encodeURIComponent(modelName)}`);
    _curriculumStatus = response.ok ? await response.json() : null;
  } catch (_) {
    _curriculumStatus = null;
  }
  populateExampleStepFilter();
}

async function loadExamples() {
  const response = await fetchApi("/api/examples");
  if (!response.ok) return;
  examples = await response.json();
  await loadCurriculumStatus();

  populateExampleOptions();

  const exampleSelect = document.getElementById("exampleInput");
  const firstOption = exampleSelect && exampleSelect.options.length > 1
    ? exampleSelect.options[1]
    : null;
  if (firstOption && firstOption.value) {
    exampleSelect.value = firstOption.value;
    applyExample(firstOption.value);
  }
}

function _exampleKnownMechanism(example) {
  const known = example && example.known_mechanism && typeof example.known_mechanism === "object"
    ? example.known_mechanism
    : null;
  if (known) return known;
  const verified = example && example.verified_mechanism && typeof example.verified_mechanism === "object"
    ? example.verified_mechanism
    : null;
  const steps = Array.isArray(verified?.steps) ? verified.steps : [];
  if (!steps.length) return null;
  return {
    source: example?.source || "FlowER 100",
    citation: "Derived from FlowER verified mechanism",
    min_steps: Number(example?.n_mechanistic_steps || steps.length || 0),
    steps: steps.map((step, index) => {
      const currentState = Array.isArray(step?.current_state) ? step.current_state : [];
      const resultingState = Array.isArray(step?.resulting_state) ? step.resulting_state : [];
      const resulting = Array.isArray(step?.resulting_state) ? step.resulting_state : [];
      const target = resulting.find((item) => String(item || "").trim()) || String(step?.predicted_intermediate || "").trim();
      return {
        step_index: Number(step?.step_index || index + 1),
        target_smiles: target,
        current_state: currentState,
        resulting_state: resultingState,
        predicted_intermediate: String(step?.predicted_intermediate || "").trim(),
      };
    }),
    final_products: Array.isArray(example?.products) ? example.products : [],
  };
}

function _exampleStepCount(example) {
  const known = _exampleKnownMechanism(example);
  if (known) {
    const minSteps = Number(
      known.min_steps
      || (Array.isArray(known.steps) ? known.steps.length : 0)
      || 0,
    );
    if (minSteps > 0) return minSteps;
  }
  const direct = Number(example?.n_mechanistic_steps || 0);
  if (direct > 0) return direct;
  const verified = example?.verified_mechanism && typeof example.verified_mechanism === "object"
    ? example.verified_mechanism
    : null;
  return Array.isArray(verified?.steps) ? verified.steps.length : 0;
}

function _isValidatedExample(example) {
  return String(example?.source || "").startsWith("Eval set:");
}

function _updateStepCountWarning(selectedStep) {
  const warning = document.getElementById("stepCountWarning");
  if (!warning) return;
  const maxStep = _curriculumStatus?.current_module?.max_step_count ?? null;
  const moduleNumber = _curriculumStatus?.current_module?.number ?? null;
  const selected = selectedStep !== "all" ? Number(selectedStep) : null;
  if (maxStep !== null && selected !== null && selected > maxStep) {
    warning.textContent = `This model is currently on module ${moduleNumber} (up to ${maxStep}-step reactions). Results above this level may be unreliable.`;
    warning.style.display = "";
  } else {
    warning.textContent = "";
    warning.style.display = "none";
  }
}

function populateExampleStepFilter() {
  const select = document.getElementById("exampleStepFilter");
  if (!select) return;
  const selectedBefore = select.value || "all";
  const maxStep = _curriculumStatus?.current_module?.max_step_count ?? null;
  const counts = [...new Set(
    examples
      .map((example) => _exampleStepCount(example))
      .filter((count) => Number.isFinite(count) && count > 0),
  )].sort((a, b) => a - b);
  select.innerHTML = '<option value="all">any</option>';
  counts.forEach((count) => {
    const option = document.createElement("option");
    option.value = String(count);
    const hasValidated = examples.some(
      (ex) => _exampleStepCount(ex) === count && _isValidatedExample(ex),
    );
    const aheadOfCurriculum = maxStep !== null && count > maxStep;
    option.textContent = `${count} step${count === 1 ? "" : "s"}${aheadOfCurriculum ? " \u26a0" : ""}`;
    if (!hasValidated) option.style.color = "#94a3b8";
    select.appendChild(option);
  });
  const newValue = counts.map(String).includes(selectedBefore) || selectedBefore === "all"
    ? selectedBefore
    : "all";
  select.value = newValue;
  _updateStepCountWarning(newValue);
}

function populateExampleOptions() {
  const select = document.getElementById("exampleInput");
  const selectedBefore = select.value || "";
  select.innerHTML = '<option value="">Select example</option>';
  const stepFilter = document.getElementById("exampleStepFilter")?.value || "all";

  if (stepFilter === "all") {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.disabled = true;
    placeholder.textContent = "— select # of steps first —";
    select.appendChild(placeholder);
    return;
  }

  // Group by exact step count for the FlowER-backed default menu.
  const groups = {};
  examples.forEach((example) => {
    const stepCount = _exampleStepCount(example);
    if (stepFilter !== "all" && String(stepCount) !== String(stepFilter)) {
      return;
    }
    const key = stepCount > 0 ? String(stepCount) : "other";
    if (!groups[key]) groups[key] = [];
    groups[key].push(example);
  });

  Object.entries(groups)
    .sort(([a], [b]) => {
      if (a === "other") return 1;
      if (b === "other") return -1;
      return Number(a) - Number(b);
    })
    .forEach(([stepKey, items]) => {
    const group = document.createElement("optgroup");
    const groupCount = Number(stepKey);
    group.label = Number.isFinite(groupCount) && groupCount > 0
      ? `${groupCount} step${groupCount === 1 ? "" : "s"}`
      : "Other";
    items
      .sort((a, b) => String(a.name || "").localeCompare(String(b.name || "")))
      .forEach((example) => {
      const option = document.createElement("option");
      option.value = example.id;
      const known = _exampleKnownMechanism(example);
      const knownSteps = _exampleStepCount(example);
      const derivedLabel = String(
        example?.derived_reaction_label
        || (known && known.reaction_label)
        || ""
      ).trim();
      const basicName = String(example.name || "").trim();
      const fallbackDescriptor = `${(example.starting_materials || [])[0] || "reaction"} -> ${(example.products || [])[0] || "product"}`;
      const firstReaction = _firstReactionText(example, known);
      const descriptor = firstReaction || fallbackDescriptor || derivedLabel || (basicName && !basicName.toLowerCase().startsWith("humanbenchmark reaction")
        ? basicName
        : "reaction");
      option.textContent = knownSteps > 0
        ? `# of steps ${knownSteps} | ${descriptor}`
        : (basicName || fallbackDescriptor);
      if (!_isValidatedExample(example)) option.style.color = "#94a3b8";
      group.appendChild(option);
    });
    select.appendChild(group);
  });
  if (selectedBefore && [...select.options].some((opt) => opt.value === selectedBefore)) {
    select.value = selectedBefore;
  }
}

function applyExample(exampleId) {
  const found = examples.find((item) => item.id === exampleId);
  if (!found) return;
  document.getElementById("startingInput").value = (found.starting_materials || []).join(", ");
  document.getElementById("productsInput").value = (found.products || []).join(", ");
  if (found.temperature_celsius !== undefined) {
    document.getElementById("tempInput").value = String(found.temperature_celsius);
  }
  if (found.ph !== undefined && found.ph !== null) {
    document.getElementById("phInput").value = String(found.ph);
  } else {
    document.getElementById("phInput").value = "";
  }
  const known = _exampleKnownMechanism(found);
  renderKnownMechanismPanel(known, { openPanel: true });
  renderMoleculesFromInputs();
}

function _knownStepsForDisplay(knownMechanism) {
  if (!knownMechanism || typeof knownMechanism !== "object") return [];
  const raw = Array.isArray(knownMechanism.steps) ? knownMechanism.steps : [];
  const byIndex = new Map();
  raw.forEach((step) => {
    const idx = Number(step.step_index || 0);
    if (idx > 0) byIndex.set(idx, step);
  });
  const minSteps = Number(knownMechanism.min_steps || 0);
  if (minSteps > 0) {
    const displaySteps = [];
    for (let i = 1; i <= minSteps; i += 1) {
      displaySteps.push(byIndex.get(i) || { step_index: i, target_smiles: "", missing: true });
    }
    return displaySteps;
  }
  return raw;
}

function _nonEmptyStringList(items) {
  return Array.isArray(items)
    ? items.map((item) => String(item || "").trim()).filter(Boolean)
    : [];
}

function _knownFinalProducts(knownMechanism) {
  if (!knownMechanism || typeof knownMechanism !== "object") return [];
  const explicitFinal = _nonEmptyStringList(knownMechanism.final_products);
  if (explicitFinal.length) return explicitFinal;
  const raw = Array.isArray(knownMechanism.steps) ? knownMechanism.steps : [];
  const sorted = [...raw].sort((a, b) => Number(a.step_index || 0) - Number(b.step_index || 0));
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const resultingState = _nonEmptyStringList(sorted[i].resulting_state);
    if (resultingState.length) return resultingState;
    const target = String(sorted[i].target_smiles || "").trim();
    if (target) return [target];
  }
  return [];
}

function _firstReactionText(example, knownMechanism) {
  const knownSteps = _knownStepsForDisplay(knownMechanism);
  const firstStep = Array.isArray(knownSteps) && knownSteps.length ? knownSteps[0] : null;
  const firstCurrent = _nonEmptyStringList(firstStep?.current_state);
  const firstResulting = _nonEmptyStringList(firstStep?.resulting_state);
  const from = (firstCurrent[0] || (example?.starting_materials || [])[0] || "").trim();
  const to = (
    firstResulting[0]
    || String(firstStep?.target_smiles || "").trim()
    || (example?.products || [])[0]
    || ""
  ).trim();
  if (!from && !to) return "";
  return `${from || "?"} -> ${to || "?"}`;
}

function renderKnownMechanismPanel(knownMechanism, options = {}) {
  const openPanel = options.openPanel === true;
  activeKnownMechanism = knownMechanism && typeof knownMechanism === "object"
    ? knownMechanism
    : null;
  const knownSteps = _knownStepsForDisplay(activeKnownMechanism);
  if (!activeKnownMechanism || !knownSteps.length) {
    knownMechanismText.textContent = "No established benchmark mechanism loaded.";
    knownMechanismCards.innerHTML = "<div class='muted'>Choose an eval reaction to view known steps.</div>";
    if (knownMechanismFinalProduct) knownMechanismFinalProduct.textContent = "No final product available.";
    return;
  }
  const lines = [`# of steps: ${knownSteps.length}`];
  const firstReaction = _firstReactionText({}, activeKnownMechanism);
  if (firstReaction) lines.push(`First reaction: ${firstReaction}`);
  const expectedCount = Number(activeKnownMechanism.min_steps || 0);
  if (expectedCount > 0 && expectedCount !== knownSteps.length) {
    lines.push(`Expected steps from benchmark metadata: ${expectedCount}`);
  }
  knownMechanismText.textContent = lines.join("\n");
  const finalProducts = _knownFinalProducts(activeKnownMechanism);
  if (knownMechanismFinalProduct) {
    knownMechanismFinalProduct.textContent = finalProducts.length
      ? finalProducts.join("\n")
      : "No final product available.";
  }
  if (openPanel) {
    const benchmarkTile = document.getElementById("benchmarkTile");
    if (benchmarkTile) benchmarkTile.classList.remove("collapsed");
  }
  renderKnownMechanismCards(knownSteps);
  renderPredictedMechanismComparison(latestSnapshotData || {});
}

async function _renderKnownStateBlock(parent, title, smilesList) {
  const block = document.createElement("div");
  block.className = "known-state-block";
  const heading = document.createElement("div");
  heading.className = "muted";
  heading.textContent = title;
  block.appendChild(heading);

  const cleaned = _nonEmptyStringList(smilesList);
  if (!cleaned.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "none";
    block.appendChild(empty);
    parent.appendChild(block);
    return;
  }

  try {
    const rendered = await renderMoleculeCardsFromSmiles(cleaned, { showAtomNumbers });
    const grid = document.createElement("div");
    grid.className = "known-state-grid";
    rendered.forEach((mol) => grid.appendChild(createMoleculeCard(mol)));
    block.appendChild(grid);
  } catch (_) {
    const fallback = document.createElement("pre");
    fallback.className = "code";
    fallback.textContent = cleaned.join("\n");
    block.appendChild(fallback);
  }
  parent.appendChild(block);
}

async function renderKnownMechanismCards(steps) {
  const renderToken = ++knownMechanismRenderToken;
  knownMechanismCards.innerHTML = "";
  for (const step of steps) {
    if (renderToken !== knownMechanismRenderToken) return;
    const currentState = _nonEmptyStringList(step.current_state);
    const resultingState = _nonEmptyStringList(step.resulting_state);
    const target = String(step.target_smiles || "").trim();
    const predictedIntermediate = String(step.predicted_intermediate || "").trim();
    const card = document.createElement("div");
    card.className = "step-card";
    const title = document.createElement("div");
    title.className = "step-title";
    title.textContent = `Known Step ${Number(step.step_index || 0)}`;
    card.appendChild(title);
    if (step.reaction_label) {
      const meta = document.createElement("div");
      meta.className = "step-meta";
      meta.textContent = step.uncertain
        ? `${step.reaction_label} (uncertain)`
        : step.reaction_label;
      card.appendChild(meta);
    }
    if (!currentState.length && !resultingState.length && !target) {
      const missing = document.createElement("div");
      missing.className = "muted";
      missing.textContent = "Missing mechanism-step display data.";
      card.appendChild(missing);
      knownMechanismCards.appendChild(card);
      continue;
    }

    const reactionLayout = document.createElement("div");
    reactionLayout.className = "known-step-reaction";
    await _renderKnownStateBlock(reactionLayout, "Current state", currentState);

    const arrow = document.createElement("div");
    arrow.className = "known-step-arrow";
    arrow.textContent = "->";
    reactionLayout.appendChild(arrow);

    await _renderKnownStateBlock(reactionLayout, "Resulting state", resultingState.length ? resultingState : (target ? [target] : []));
    card.appendChild(reactionLayout);

    if (predictedIntermediate && !resultingState.includes(predictedIntermediate)) {
      const intermediate = document.createElement("pre");
      intermediate.className = "code";
      intermediate.textContent = `Intermediate: ${predictedIntermediate}`;
      card.appendChild(intermediate);
    }
    if (step.reaction_smirks) {
      const smirks = document.createElement("pre");
      smirks.className = "code";
      smirks.textContent = step.reaction_smirks;
      card.appendChild(smirks);
    }
    if (renderToken !== knownMechanismRenderToken) return;
    knownMechanismCards.appendChild(card);
  }
}

async function renderPredictedMechanismComparison(snapshot) {
  if (!predictedMechanismCards || !predictedMechanismText) return;
  const summary = Array.isArray(snapshot?.mechanism_summary) ? snapshot.mechanism_summary : [];
  predictedMechanismCards.innerHTML = "";
  if (!summary.length) {
    predictedMechanismText.textContent = "No mechanism steps generated yet.";
    predictedMechanismCards.innerHTML = "<div class='muted'>Run the reaction to compare against the benchmark steps.</div>";
    return;
  }
  predictedMechanismText.textContent = `Accepted steps so far: ${summary.length}`;
  for (const entry of summary) {
    const card = document.createElement("div");
    card.className = "step-card";
    const stepIndex = Number(entry.step_index || entry.attempt || 0);
    const containsTarget = Boolean(entry.contains_target_product);
    const displayIntermediateCard = await _cardForDisplay(entry.intermediate_card);
    const intermediateCard = displayIntermediateCard ? createMoleculeCard(displayIntermediateCard).outerHTML : "<div class='muted'>none</div>";
    card.innerHTML = `
      <div class="step-title">Predicted Step ${stepIndex}</div>
      <div class="step-meta">contains target: ${containsTarget}</div>
      <div><div class="muted">Predicted intermediate</div><div class="molecule-grid">${intermediateCard}</div></div>
    `;
    predictedMechanismCards.appendChild(card);
  }
}

function applyFrontendMode() {
  renderPredictedMechanismComparison(latestSnapshotData || {});
}

function applyHarnessTestingMode() {
  const selection = getHarnessSelection();
  const isExisting = selection.type === "existing";

  // Seed model fields when a per-family preset is selected.
  if (selection.type === "family" && selection.familyId) {
    const family = allFamilies.find((f) => f.id === selection.familyId);
    if (family) {
      const familyEl = document.getElementById("modelFamilyInput");
      if (familyEl) {
        familyEl.value = family.id;
        familyEl.dispatchEvent(new Event("change"));
      }
      // Wait a tick for modelNameInput to be populated, then select top_model.
      setTimeout(() => {
        const modelEl = document.getElementById("modelNameInput");
        if (modelEl && family.top_model) {
          modelEl.value = family.top_model;
        }
        const reasoningEl = document.getElementById("reasoningInput");
        if (reasoningEl) {
          reasoningEl.value = family.supports_reasoning ? "high" : "";
        }
        updateHarnessReadOnlySummary();
      }, 0);
    }
  }

  const lockFields = false;
  const fieldIds = ["modelFamilyInput", "modelNameInput", "reasoningInput"];
  fieldIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = lockFields;
  });
  const showBtn = document.getElementById("showHarnessSettingsBtn");
  if (showBtn) showBtn.style.display = "";

  const editable = document.getElementById("harnessConfigEditable");
  if (editable) editable.style.display = lockFields ? "none" : "";

  const advancedInputDetails = document.getElementById("advancedInputDetails");
  if (advancedInputDetails) {
    // Keep Advanced controls visible for all harness selections (including default).
    advancedInputDetails.style.display = "";
    if (!isExisting) advancedInputDetails.open = true;
  }
  updateHarnessReadOnlySummary();
}

async function loadHarnessVersions() {
  // Load saved harness configs.
  const cfgResponse = await fetchApi("/api/harness/configs");
  if (cfgResponse.ok) {
    const data = await cfgResponse.json();
    harnessVersions = Array.isArray(data.items) ? data.items : [];
  }

  // Load model families for per-family presets.
  const famResponse = await fetchApi("/api/catalog/families");
  if (famResponse.ok) {
    allFamilies = await famResponse.json();
  }

  const select = document.getElementById("harnessSelectionInput");
  if (!select) return;
  const selectedBefore = select.value || "default";
  select.innerHTML = "";

  // Baseline options.
  const latestOpt = document.createElement("option");
  latestOpt.value = "default";
  latestOpt.textContent = "default harness";
  select.appendChild(latestOpt);

  const newOpt = document.createElement("option");
  newOpt.value = "new_harness";
  newOpt.textContent = "test new harness";
  select.appendChild(newOpt);

  // Per-family presets: one option per family, seeds top model + highest thinking.
  if (allFamilies.length > 0) {
    const groupLabel = document.createElement("optgroup");
    groupLabel.label = "Quick start — by model family";
    allFamilies.forEach((family) => {
      const opt = document.createElement("option");
      opt.value = `family:${family.id}`;
      const thinkingNote = family.supports_reasoning ? " · high thinking" : "";
      opt.textContent = `Latest — ${family.label || family.id}${thinkingNote}`;
      groupLabel.appendChild(opt);
    });
    select.appendChild(groupLabel);
  }

  // Saved harness configs.
  const savedConfigs = harnessVersions.filter((item) => item.name !== "default");
  if (savedConfigs.length > 0) {
    const savedGroup = document.createElement("optgroup");
    savedGroup.label = "Saved harness configs";
    savedConfigs.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = `saved:${item.name}`;
      opt.textContent = `saved: ${item.name}`;
      savedGroup.appendChild(opt);
    });
    select.appendChild(savedGroup);
  }

  if ([...select.options].some((opt) => opt.value === selectedBefore)) {
    select.value = selectedBefore;
  } else {
    select.value = "default";
  }
}

async function parseSmirksInput() {
  const smirks = document.getElementById("smirksInput").value;
  if (!smirks.trim()) {
    setStatus("Provide SMIRKS text first");
    return;
  }
  const response = await fetchApi("/api/parse_smirks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smirks }),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.detail || "Failed to parse SMIRKS");
    return;
  }
  document.getElementById("startingInput").value = data.reactants.join(", ");
  document.getElementById("productsInput").value = data.products.join(", ");
  if (data.mapping_detected) {
    skipAtomMappingForSmirks = true;
    setStatus("SMIRKS parsed. Atom mapping detected; mapping tool disabled for this run.");
  } else {
    skipAtomMappingForSmirks = false;
    setStatus("SMIRKS parsed successfully.");
  }
}

function collectOptionalTools() {
  const options = ["predict_missing_reagents"];
  if (!skipAtomMappingForSmirks) options.push("attempt_atom_mapping");
  return options;
}


async function createRun() {
  const modelFamily = document.getElementById("modelFamilyInput").value;
  const modelName = document.getElementById("modelNameInput").value;
  const reasoning = document.getElementById("reasoningInput").value || null;
  const orchestrationMode = document.getElementById("orchestrationModeInput")?.value || "standard";
  const harnessStrategy = document.getElementById("harnessStrategyInput")?.value || "latest";
  const harnessName = getSelectedHarnessName();
  const harnessList = parseSmilesList(document.getElementById("harnessListInput")?.value || "");

  const apiKeys = {};
  const openaiKey = document.getElementById("openaiKeyInput")?.value.trim();
  const openrouterKey = document.getElementById("openrouterKeyInput")?.value.trim();
  const geminiKey = document.getElementById("geminiKeyInput")?.value.trim();
  if (openaiKey) apiKeys.openai = openaiKey;
  if (openrouterKey) apiKeys.openrouter = openrouterKey;
  if (geminiKey) apiKeys.gemini = geminiKey;

  const dryRunChecked = document.getElementById("dryRunToggle")?.checked ?? false;
  const ralphPayload = {
    max_iterations: 0,
    completion_promise: "target_products_reached && flow_node:run_complete",
    max_runtime_seconds: Number(document.getElementById("ralphMaxRuntimeInput")?.value || 6000),
    max_cost_usd: document.getElementById("maxCostUsdInput")?.value
      ? Number(document.getElementById("maxCostUsdInput").value)
      : null,
    repeat_failure_signature_limit: Number(document.getElementById("repeatFailureSignatureLimitInput")?.value || 2),
    harness_strategy: harnessStrategy,
    harness_list: harnessList,
    babysit_mode: document.getElementById("babysitModeInput")?.value || "off",
    allow_validator_mutation: true,
  };
  const body = {
    mode: "unverified",
    orchestration_mode: orchestrationMode,
    model_name: modelName,
    thinking_level: reasoning || null,
    example_id: document.getElementById("exampleInput")?.value || null,
    starting_materials: parseSmilesList(document.getElementById("startingInput").value),
    products: parseSmilesList(document.getElementById("productsInput").value),
    temperature_celsius: Number(document.getElementById("tempInput").value || 25),
    ph: document.getElementById("phInput").value ? Number(document.getElementById("phInput").value) : null,
    api_keys: apiKeys,
    optional_llm_tools: collectOptionalTools(),
    functional_groups_enabled: true,
    intermediate_prediction_enabled: true,
    max_steps: Number(document.getElementById("maxStepsInput").value || 6),
    harness_name: harnessName,
    coordination_topology: document.getElementById("coordinationTopologyInput")?.value || "centralized_mas",
    ralph: orchestrationMode === "ralph" ? ralphPayload : null,
    dry_run: dryRunChecked,
  };

  const response = await fetchApi("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to create run");

  runId = data.run_id;
  isDryRun = dryRunChecked;
  latestEvaluation = null;
  evaluationOutput.textContent = "";
  harnessOutput.textContent = "";
  document.getElementById("dryRunPanel").style.display = "none";
  document.getElementById("dryRunDiscardStatus").textContent = "";
  if (terminalOutput) terminalOutput.innerHTML = "";
  _lastTerminalStepCount = 0;
  appendTerminalLine("system", "", `Run created: ${data.run_id}${isDryRun ? " [DRY RUN]" : ""}`, "info");
  setStatus(`Run created: ${runId}${isDryRun ? " [DRY RUN]" : ""}`);
  updateButtons("pending");
  await refreshSnapshot();
}

function openEventStream() {
  if (!runId) return;
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  eventSource = new EventSource(sseUrl(`/api/runs/${runId}/events`));
  const eventKinds = [
    "run_created",
    "run_start_requested",
    "run_started",
    "step_output",
    "target_products_detected",
    "verification_decision",
    "run_failed",
    "run_completed",
    "awaiting_verification",
    "run_stopped",
    "run_stop_requested",
    "feedback_recorded",
    "runtime_limit",
    "step_started",
    "step_completed",
    "step_failed",
    "loop_iteration_started",
    "loop_iteration_completed",
    "completion_check",
    "mechanism_retry_started",
    "mechanism_retry_failed",
    "mechanism_retry_exhausted",
    "run_paused",
    "run_resumed",
    "awaiting_manual_steps",
    "backtrack",
    "branch_point_created",
    "failed_path_recorded",
    "ralph_iteration_started",
    "ralph_harness_mutated",
    "ralph_completion_promise_check",
    "ralph_budget_warning",
    "ralph_stopped",
    "ralph_vote_recorded",
    "evaluation_completed",
    "evaluation_saved",
    "harness_updated",
    "stream_end",
  ];

  eventKinds.forEach((kind) => {
    eventSource.addEventListener(kind, (ev) => {
      if (kind === "stream_end") return;
      try {
        appendEventToTerminal(JSON.parse(ev.data));
      } catch (_) {
        // no-op
      }
    });
  });
}

async function startRun() {
  if (!runId) return;
  const response = await fetch(`/api/runs/${runId}/start`, { method: "POST" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to start run");
  setStatus(`Run started: ${runId}`);
  openEventStream();
  if (snapshotTimer) clearInterval(snapshotTimer);
  snapshotTimer = setInterval(refreshSnapshot, 1500);
  await refreshSnapshot();
}

async function stopRun() {
  if (!runId) return;
  await fetch(`/api/runs/${runId}/stop`, { method: "POST" });
  await refreshSnapshot();
}

async function resumeRun(decision) {
  if (!runId) return;
  const response = await fetch(`/api/runs/${runId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to resume run");
  if (decision === "continue") {
    openEventStream();
  }
  await refreshSnapshot();
}

// Step feedback, run feedback, and advisory votes removed from UI.
// Use the API directly or the CONTRIBUTING workflow for feedback.

async function evaluateRun() {
  if (!runId) return;
  const response = await fetch(`/api/runs/${runId}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ judge_model: "gpt-5.4" }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to evaluate run");
  latestEvaluation = data.evaluation;
  evaluationOutput.textContent = JSON.stringify(data.evaluation, null, 2);
  const recommendations = data.evaluation?.harness_recommendations;
  if (typeof recommendations === "string") {
    document.getElementById("harnessRecommendationInput").value = recommendations;
  } else if (Array.isArray(recommendations)) {
    document.getElementById("harnessRecommendationInput").value = recommendations.join("\n");
  }
  updateButtons("completed");
}

async function saveEvaluation() {
  if (!runId || !latestEvaluation) return;
  const response = await fetch(`/api/runs/${runId}/evaluation/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ evaluation: latestEvaluation, judge_model: "gpt-5.4" }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to save evaluation");
  harnessOutput.textContent = JSON.stringify(data.record, null, 2);
}

async function refreshFewShotStatus() {
  const statusEl = document.getElementById("fewShotStatus");
  if (!statusEl) return;
  const response = await fetchApi("/api/few_shot?approved_only=false&limit=1000");
  if (!response.ok) return;
  const data = await response.json();
  const items = Array.isArray(data.items) ? data.items : [];
  statusEl.textContent = `Few-shot stored candidates: ${items.length}`;
}

async function storeLatestFewShotCandidate() {
  if (!runId) throw new Error("Create and run a prediction first.");
  const traceResp = await fetch(`/api/traces?run_id=${encodeURIComponent(runId)}&step_name=mechanism_step_proposal&limit=50`);
  const traceData = await traceResp.json();
  if (!traceResp.ok) throw new Error(traceData.detail || "Failed to load traces");
  const items = Array.isArray(traceData.items) ? traceData.items : [];
  if (!items.length) throw new Error("No mechanism proposal traces found for this run.");
  const best = items[0];
  const resp = await fetchApi("/api/few_shot/from_trace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id: best.id, approved: true }),
  });
  const payload = await resp.json();
  if (!resp.ok) throw new Error(payload.detail || "Failed to store few-shot candidate");
  harnessOutput.textContent = JSON.stringify(payload, null, 2);
  await refreshFewShotStatus();
}

async function applyHarnessUpdate() {
  if (!runId) return;
  const recommendation = document.getElementById("harnessRecommendationInput").value || null;
  const response = await fetch(`/api/runs/${runId}/harness/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      call_name: "propose_mechanism_step",
      component: "base",
      recommendation,
      append_mode: true,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to apply harness update");
  harnessOutput.textContent = JSON.stringify(data, null, 2);
}

async function createHarnessPR() {
  const trackSelection = isNewHarnessSelection()
    ? "new_harness"
    : "existing_harness";
  const commitMessage = trackSelection === "new_harness"
    ? "Create new harness variant from local run config"
    : "Update existing harness prompts from local evidence";
  const prTitle = trackSelection === "new_harness"
    ? "New harness variant from local evaluation"
    : "Existing harness update from local evaluation";
  const prBody = trackSelection === "new_harness"
    ? "Creates a new harness variant. Run easy/medium eval tier for validation."
    : "Updates existing harness with approved evidence traces and prompt updates.";
  const response = await fetchApi("/api/harness/pr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pr_track: trackSelection,
      call_names: ["propose_mechanism_step"],
      evidence_trace_ids: {},
      commit_message: commitMessage,
      pr_title: prTitle,
      pr_body: prBody,
      push: true,
      open_pr: true,
      run_id: runId,
      include_evidence: trackSelection === "existing_harness",
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Failed to create PR");
  harnessOutput.textContent = JSON.stringify(data, null, 2);
}

async function loadEvalSets() {
  const response = await fetchApi("/api/eval_sets");
  if (!response.ok) return;
  const data = await response.json();
  evalSets = Array.isArray(data.items) ? data.items : [];

  // Populate both the main eval section selector and the modal selector.
  const selects = [
    document.getElementById("evalSetSelect"),
    document.getElementById("modalEvalSetSelect"),
  ].filter(Boolean);

  selects.forEach((select) => {
    select.innerHTML = "";
    if (!evalSets.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No eval sets";
      select.appendChild(option);
      return;
    }
    evalSets.forEach((setItem) => {
      const option = document.createElement("option");
      option.value = setItem.id;
      option.textContent = `${setItem.name} (${setItem.version})`;
      select.appendChild(option);
    });
  });

  await loadEvalTierMetadata();
}

function _caseSort(a, b) {
  const aSteps = Number(a.min_steps || 0);
  const bSteps = Number(b.min_steps || 0);
  if (aSteps !== bSteps) return aSteps - bSteps;
  return String(a.case_id || "").localeCompare(String(b.case_id || ""));
}

function applyEvalCaseToInputs(caseId) {
  const selected = (evalTierData.cases || []).find((item) => String(item.case_id) === String(caseId));
  if (!selected) return;
  document.getElementById("startingInput").value = (selected.starting_materials || []).join(", ");
  document.getElementById("productsInput").value = (selected.products || []).join(", ");
  const known = selected.known_mechanism && typeof selected.known_mechanism === "object"
    ? selected.known_mechanism
    : null;
  renderKnownMechanismPanel(known, { openPanel: true });
}

function pickTierPreviewCaseId(tierName) {
  const tierCaseIds = evalTierData.tiers?.[tierName] || [];
  if (!tierCaseIds.length) return "";
  const scored = tierCaseIds.map((caseId) => {
    const meta = (evalTierData.cases || []).find((item) => String(item.case_id) === String(caseId));
    return {
      caseId: String(caseId),
      minSteps: Number(meta?.min_steps || 0),
    };
  });
  scored.sort((a, b) => {
    if (a.minSteps !== b.minSteps) return b.minSteps - a.minSteps;
    return a.caseId.localeCompare(b.caseId);
  });
  return scored[0]?.caseId || String(tierCaseIds[0] || "");
}

async function loadEvalTierMetadata() {
  const evalSetId = document.getElementById("evalSetSelect")?.value
    || document.getElementById("modalEvalSetSelect")?.value || "";
  if (!evalSetId) return;
  const response = await fetch(`/api/evals/tiers?eval_set_id=${encodeURIComponent(evalSetId)}`);
  if (!response.ok) return;
  const data = await response.json();
  evalTierData = {
    tiers: data.tiers || { easy: [], medium: [], hard: [] },
    cases: Array.isArray(data.cases) ? data.cases : [],
  };
  const caseSelect = document.getElementById("evalCaseSelect");
  if (!caseSelect) return;
  caseSelect.innerHTML = '<option value="">Select reaction</option>';
  [...evalTierData.cases].sort(_caseSort).forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.case_id;
    opt.textContent = `${item.case_id} (steps: ${item.min_steps ?? "n/a"})`;
    caseSelect.appendChild(opt);
  });
  const selectionMode = document.getElementById("evalSelectionMode")?.value || "tier";
  const tierName = document.getElementById("evalTierSelect")?.value || "easy";
  if (selectionMode === "tier") {
    const firstTierCaseId = pickTierPreviewCaseId(tierName);
    if (firstTierCaseId) {
      caseSelect.value = firstTierCaseId;
      applyEvalCaseToInputs(firstTierCaseId);
    }
  } else if (caseSelect.value) {
    applyEvalCaseToInputs(caseSelect.value);
  }
}

function _formatTimestamp(ts) {
  if (!ts) return "n/a";
  try {
    return new Date(Number(ts) * 1000).toLocaleString();
  } catch (_) {
    return String(ts);
  }
}

function _renderLeaderboardInto(container, items) {
  container.innerHTML = "";
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "No leaderboard rows yet.";
    container.appendChild(empty);
    return;
  }
  rows.forEach((row) => {
    const card = document.createElement("div");
    const isBaseline = !!row.is_baseline;
    const isSimulated = !!row.is_simulated;
    card.className = "step-card" + (isBaseline ? " step-card-baseline" : "") + (isSimulated ? " step-card-simulated" : "");

    const hashShort = row.harness_bundle_hash ? row.harness_bundle_hash.slice(0, 8) : "n/a";
    const modelLabel = row.thinking_level
      ? `${row.model_name || row.model || "unknown model"} (${row.thinking_level})`
      : (row.model_name || row.model || "unknown model");

    // Mode badge: harness vs baseline (vs simulated).
    const modeBadge = isBaseline
      ? `<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:0.78em;font-weight:600;background:#e8f4fd;color:#1a6fa8;margin-left:6px;">Baseline</span>`
      : `<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:0.78em;font-weight:600;background:#edf7ed;color:#276621;margin-left:6px;">Harness</span>`;
    const simulatedBadge = isSimulated
      ? `<span style="display:inline-block;padding:1px 6px;border-radius:3px;font-size:0.78em;font-weight:600;background:#fff3cd;color:#856404;margin-left:4px;">⚠ Simulated</span>`
      : "";

    // Build per-subagent table if data is available.
    const perSubagent = row.per_subagent_scores || {};
    const subagentIds = Object.keys(perSubagent);
    let subagentHtml = "";
    if (subagentIds.length) {
      const rows_html = subagentIds
        .map((id) => {
          const entry = perSubagent[id];
          const q = Number(entry.quality_score || 0).toFixed(3);
          const p = Number(entry.pass_rate || 0).toFixed(3);
          const n = entry.case_count || 0;
          return `<tr><td style="padding:2px 8px 2px 0;">${id}</td><td style="padding:2px 8px;">${q}</td><td style="padding:2px 8px;">${p}</td><td style="padding:2px 0;">${n}</td></tr>`;
        })
        .join("");
      const tableLabel = isBaseline ? "Single-shot breakdown" : "Per-subagent breakdown";
      subagentHtml = `
        <details style="margin-top:0.5rem;">
          <summary class="muted" style="cursor:pointer;">${tableLabel}</summary>
          <table style="font-size:0.85em;margin-top:0.4rem;border-collapse:collapse;">
            <thead><tr>
              <th style="text-align:left;padding:2px 8px 2px 0;">${isBaseline ? "Step" : "Subagent"}</th>
              <th style="padding:2px 8px;">Quality</th>
              <th style="padding:2px 8px;">Pass rate</th>
              <th style="padding:2px 0;">Cases</th>
            </tr></thead>
            <tbody>${rows_html}</tbody>
          </table>
        </details>`;
    }

    card.innerHTML = `
      <div class="step-title">${modelLabel}${modeBadge}${simulatedBadge}</div>
      <div class="step-meta">eval run: ${row.eval_run_id} &middot; ${_formatTimestamp(row.created_at)}</div>
      <div class="grid three">
        <div><strong>Quality</strong><div>${Number(row.mean_quality_score || 0).toFixed(3)}</div></div>
        <div><strong>Pass rate</strong><div>${Number(row.deterministic_pass_rate || 0).toFixed(3)}</div></div>
        <div><strong>Cases</strong><div>${row.case_count || 0}</div></div>
      </div>
      <div class="muted">group=${row.run_group_name || "n/a"} &middot; harness=${hashShort}</div>
      ${subagentHtml}
    `;
    container.appendChild(card);
  });
}

async function refreshLeaderboardModal() {
  if (!modalLeaderboardList) return;
  const selectEl = document.getElementById("modalEvalSetSelect");

  // Auto-select first available eval set if none is selected.
  if (selectEl && (!selectEl.value || selectEl.value === "") && evalSets.length > 0) {
    selectEl.value = evalSets[0].id;
  }

  const evalSetId = selectEl?.value || "";
  if (!evalSetId) {
    modalLeaderboardList.innerHTML = `
      <div class="muted" style="padding:1rem 0;">
        No eval sets found. Use <strong>Import HumanBenchmark Eval Set</strong> in the
        Evaluation panel, then come back to refresh.
      </div>`;
    return;
  }

  modalLeaderboardList.innerHTML = "<div class='muted'>Loading…</div>";
  const response = await fetch(`/api/evals/leaderboard?eval_set_id=${encodeURIComponent(evalSetId)}&limit=20`);
  const data = await response.json();
  if (!response.ok) {
    modalLeaderboardList.innerHTML = `<div class='muted'>Error: ${data.detail || "failed"}</div>`;
    return;
  }
  const items = data.items || [];
  if (!items.length) {
    modalLeaderboardList.innerHTML = `
      <div class="muted" style="padding:1rem 0;">
        No leaderboard results yet for this eval set. Run an eval set to populate the board.
      </div>`;
    return;
  }
  _renderLeaderboardInto(modalLeaderboardList, items);
}

function renderProgress(snapshot) {
  const progress = snapshot.progress || {};
  const percent = progress.progress_percentage || 0;
  const completed = progress.completed_count || 0;
  const total = progress.total_count || 0;
  const activeStep = progress.active_step_name || null;
  const loopCount = Array.isArray(snapshot.mechanism_summary) ? snapshot.mechanism_summary.length : 0;
  const maxLoops = (snapshot.config && snapshot.config.max_steps) ? snapshot.config.max_steps : null;
  document.getElementById("progressPercent").textContent = `${percent}%`;
  document.getElementById("progressMeta").textContent = `${completed} / ${total} steps`;
  const loopCounterEl = document.getElementById("loopCounter");
  if (loopCounterEl) {
    const loopTotal = maxLoops !== null ? ` / ${maxLoops}` : "";
    loopCounterEl.textContent = `${loopCount}${loopTotal} "Propose Next Mechanism Step" loop${loopCount !== 1 ? "s" : ""} completed`;
  }
  if (activeStep) {
    activeStepBadge.innerHTML = `Active: <strong>${activeStep}</strong> <span class="spinner"></span>`;
  } else {
    activeStepBadge.textContent = "No active step.";
  }

  // Populate model map for mermaid node annotations (diagram renders this)
  stepModelMap = {};
  const steps = Array.isArray(progress.steps) ? progress.steps : [];
  steps.forEach((step) => {
    if (step.model && step.model !== "n/a") stepModelMap[step.name] = step.model;
  });
}

function renderRalphStatus(snapshot) {
  const budgetEl = document.getElementById("ralphBudgetBar");
  const timelineEl = document.getElementById("ralphAttemptTimeline");
  const cfg = snapshot.config || {};
  const mode = cfg.orchestration_mode || "standard";
  if (!budgetEl || !timelineEl) return;
  if (mode !== "ralph") {
    budgetEl.style.display = "none";
    timelineEl.style.display = "none";
    return;
  }

  const attempts = Array.isArray(snapshot.ralph_attempts) ? snapshot.ralph_attempts : [];
  const totalCost = attempts.reduce((sum, item) => sum + Number(item.cost_usd || 0), 0);
  const maxCost = cfg.max_cost_usd !== null && cfg.max_cost_usd !== undefined
    ? Number(cfg.max_cost_usd)
    : null;
  budgetEl.style.display = "";
  if (maxCost && maxCost > 0) {
    const pct = Math.min(100, Math.round((totalCost / maxCost) * 100));
    budgetEl.textContent = `RAlph budget: $${totalCost.toFixed(3)} / $${maxCost.toFixed(3)} (${pct}%)`;
  } else {
    budgetEl.textContent = `RAlph budget: $${totalCost.toFixed(3)} (no cap)`;
  }

  timelineEl.style.display = "";
  if (!attempts.length) {
    timelineEl.textContent = "RAlph attempts: none";
    return;
  }
  const labels = attempts.map((item) => {
    const idx = item.attempt_index;
    const reason = item.stop_reason || "n/a";
    const met = item.completion_promise_met ? "met" : "miss";
    return `#${idx}:${reason}:${met}`;
  });
  timelineEl.textContent = `RAlph attempts: ${labels.join(" | ")}`;
}

let _lastTerminalStepCount = 0;

function renderStepOutputsToTerminal(snapshot) {
  const steps = Array.isArray(snapshot.step_outputs) ? snapshot.step_outputs : [];
  if (steps.length <= _lastTerminalStepCount) return;

  for (let i = _lastTerminalStepCount; i < steps.length; i++) {
    const step = steps[i];
    const summary = summariseStepOutput(step);
    const validation = step.validation;
    let level = "info";
    if (validation && !validation.passed) level = "error";
    else if (validation && validation.passed) level = "success";
    const retry = step.retry_index ? ` retry=${step.retry_index}` : "";
    appendTerminalLine(
      step.step_name || "step",
      `attempt ${step.attempt || 0}${retry}`,
      summary,
      level,
    );
  }
  _lastTerminalStepCount = steps.length;
}

async function refreshSnapshot() {
  if (!runId) return;
  const verbose = document.getElementById("verboseToggle").checked;
  const response = await fetch(`/api/runs/${runId}?verbose=${verbose ? "true" : "false"}`);
  if (!response.ok) return;
  const data = await response.json();
  latestSnapshotData = data;
  if (verbose && terminalOutput) {
    appendTerminalLine("snapshot", "", JSON.stringify(data, null, 2).slice(0, 2000));
  }
  renderProgress(data);
  renderRalphStatus(data);
  renderStepOutputsToTerminal(data);
  await renderReactionVisuals(data);
  await renderPredictedMechanismComparison(data);
  renderRunStepSummaries(data);
  await renderFlow();

  if (data.latest_evaluation && data.latest_evaluation.summary) {
    latestEvaluation = data.latest_evaluation.summary;
    evaluationOutput.textContent = JSON.stringify(latestEvaluation, null, 2);
  }

  latestPauseInfo = data.latest_pause || null;

  if (data.status === "paused" && data.latest_pause) {
    const reason = data.latest_pause.reason || "unknown";
    const hasAlt = (data.latest_pause.details || {}).has_alternative;
    const failedChecks = (data.latest_pause.details || {}).failed_checks || [];
    const checksSuffix = Array.isArray(failedChecks) && failedChecks.length
      ? ` checks=${failedChecks.join(",")}`
      : "";
    const qualifier = hasAlt ? "last chance" : "dead end";
    const rt = data.reaction_type_selection || {};
    const tg = data.template_guidance_state || {};
    const rtLabel = rt.selected_label_exact ? String(rt.selected_label_exact) : "n/a";
    const rtConf = asNumber(rt.confidence);
    const tgMode = tg.mode ? String(tg.mode) : "n/a";
    const tgReason = tg.disable_reason ? ` (${String(tg.disable_reason)})` : "";
    const rtText = ` | type=${rtLabel}${rtConf !== null ? `@${rtConf.toFixed(2)}` : ""}, guidance=${tgMode}${tgReason}`;
    setStatus(`Run ${runId}: paused (${data.mode}) — ${qualifier} [${reason}]${checksSuffix}${rtText}`);
  } else {
    const rt = data.reaction_type_selection || {};
    const tg = data.template_guidance_state || {};
    const rtLabel = rt.selected_label_exact ? String(rt.selected_label_exact) : "n/a";
    const rtConf = asNumber(rt.confidence);
    const tgMode = tg.mode ? String(tg.mode) : "n/a";
    const tgReason = tg.disable_reason ? ` (${String(tg.disable_reason)})` : "";
    const rtText = ` | type=${rtLabel}${rtConf !== null ? `@${rtConf.toFixed(2)}` : ""}, guidance=${tgMode}${tgReason}`;
    setStatus(`Run ${runId}: ${data.status} (${data.mode})${rtText}`);
  }
  updateButtons(data.status);
}

async function loadPromptVersions() {
  const select = document.getElementById("promptVersionSelect");
  if (!select) return;
  try {
    const resp = await fetchApi("/api/catalog/prompt_versions");
    if (!resp.ok) return;
    const data = await resp.json();
    const items = Array.isArray(data.items) ? data.items : [];
    select.innerHTML = '<option value="latest">latest</option>';
    const seen = new Set();
    for (const item of items) {
      if (seen.has(item.sha256)) continue;
      seen.add(item.sha256);
      const dateStr = item.created_at ? new Date(item.created_at * 1000).toLocaleDateString() : "";
      const opt = document.createElement("option");
      opt.value = item.sha256;
      opt.textContent = `v${item.version} (${(item.sha256 || "").slice(0, 8)}) ${dateStr}`;
      select.appendChild(opt);
    }
  } catch (_) {
    // endpoint may not exist yet
  }
}

async function onPromptVersionChange(sha256) {
  selectedPromptVersion = sha256;
  versionTemplateCache = {};
  if (sha256 === "latest") {
    if (lastSelectedNodeId) showNodeDetail(lastSelectedNodeId);
    return;
  }
  try {
    const resp = await fetch(`/api/catalog/prompt_versions?sha256=${encodeURIComponent(sha256)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    const items = Array.isArray(data.items) ? data.items : [];
    for (const item of items) {
      const mapped = item.call_name ? (CALL_TO_STEP_NAMES[item.call_name] || []) : [];
      const targetSteps = mapped.length ? mapped : (item.step_name ? [item.step_name] : []);
      for (const step of targetSteps) {
        versionTemplateCache[step] = item.template_text || "";
      }
    }
  } catch (_) {
    // ignore
  }
  if (lastSelectedNodeId) showNodeDetail(lastSelectedNodeId);
}

async function bootstrap() {
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "default",
    });
  }

  showAtomNumbers = Boolean(document.getElementById("displayAtomNumbersToggle")?.checked);

  const leaderboardModal = document.getElementById("leaderboardModal");
  const leaderboardModalBackdrop = document.getElementById("leaderboardModalBackdrop");
  const openLeaderboardModal = async () => {
    if (leaderboardModal) leaderboardModal.style.display = "";
    try {
      await refreshLeaderboardModal();
    } catch (err) {
      if (modalLeaderboardList) {
        modalLeaderboardList.innerHTML = `<div class='muted'>Error loading leaderboard: ${err.message || err}</div>`;
      }
    }
  };
  const closeLeaderboardModal = () => {
    if (leaderboardModal) leaderboardModal.style.display = "none";
  };

  const openLeaderboardBtn = document.getElementById("openLeaderboardModalBtn");
  if (openLeaderboardBtn) openLeaderboardBtn.onclick = openLeaderboardModal;
  const closeLeaderboardBtn = document.getElementById("closeLeaderboardModalBtn");
  if (closeLeaderboardBtn) closeLeaderboardBtn.onclick = closeLeaderboardModal;
  if (leaderboardModalBackdrop) leaderboardModalBackdrop.onclick = closeLeaderboardModal;
  const modalRefreshLeaderboardBtn = document.getElementById("modalRefreshLeaderboardBtn");
  if (modalRefreshLeaderboardBtn) {
    modalRefreshLeaderboardBtn.onclick = async () => {
      try {
        await refreshLeaderboardModal();
      } catch (err) {
        if (modalLeaderboardList) {
          modalLeaderboardList.innerHTML = `<div class='muted'>${err.message || err}</div>`;
        }
      }
    };
  }
  document.getElementById("modalEvalSetSelect")?.addEventListener("change", async () => {
    try {
      await refreshLeaderboardModal();
    } catch (_) {}
  });

  // API Keys modal
  const apiKeysBtn = document.getElementById("apiKeysBtn");
  const apiKeysModal = document.getElementById("apiKeysModal");
  const apiKeysBackdrop = document.getElementById("apiKeysBackdrop");
  const closeApiKeysBtn = document.getElementById("closeApiKeysBtn");
  if (apiKeysBtn) apiKeysBtn.onclick = () => { if (apiKeysModal) apiKeysModal.style.display = ""; };
  if (closeApiKeysBtn) closeApiKeysBtn.onclick = () => { if (apiKeysModal) apiKeysModal.style.display = "none"; };
  if (apiKeysBackdrop) apiKeysBackdrop.onclick = () => { if (apiKeysModal) apiKeysModal.style.display = "none"; };

  // Tile toggling (click header to collapse/expand tile body)
  document.querySelectorAll(".tile-header[data-tile]").forEach((header) => {
    header.addEventListener("click", (e) => {
      if (e.target.closest("label") || e.target.closest("input")) return;
      const tileId = header.getAttribute("data-tile");
      const body = document.getElementById(tileId);
      if (body) body.classList.toggle("collapsed");
    });
  });

  const showHarnessSettingsBtn = document.getElementById("showHarnessSettingsBtn");
  if (showHarnessSettingsBtn) showHarnessSettingsBtn.addEventListener("click", showHarnessSettingsModal);
  const closeHarnessSettingsBtn = document.getElementById("closeHarnessSettingsBtn");
  if (closeHarnessSettingsBtn) closeHarnessSettingsBtn.addEventListener("click", hideHarnessSettingsModal);
  const harnessSettingsBackdrop = document.getElementById("harnessSettingsBackdrop");
  if (harnessSettingsBackdrop) harnessSettingsBackdrop.addEventListener("click", hideHarnessSettingsModal);

  const safeInit = async (label, action) => {
    try {
      await action();
    } catch (err) {
      console.error(`bootstrap:${label}`, err);
      setStatus(`UI warning: ${label} failed to load (${err.message || err})`);
    }
  };

  await safeInit("model catalog", loadModelCatalog);
  populateModelOptions();
  await safeInit("examples", loadExamples);
  await safeInit("eval sets", loadEvalSets);
  await safeInit("harness configs", loadHarnessVersions);
  applyFrontendMode();
  applyHarnessTestingMode();
  updateOrchestrationModeUi();
  updateHarnessReadOnlySummary();
  await safeInit("few-shot status", refreshFewShotStatus);
  await safeInit("prompt versions", loadPromptVersions);
  await safeInit("curriculum status", refreshCurriculumStatus);
  await safeInit("curriculum history", refreshCurriculumHistory);
  renderKnownMechanismPanel(null);
  updateButtons();
  await safeInit("step preview", updateMermaidPreview);
  await safeInit("initial molecules", renderMoleculesFromInputs);
  setPrimaryView("curriculum");
  setCurriculumPanelVisible(false);
  const verboseToggle = document.getElementById("verboseToggle");
  if (verboseToggle) {
    verboseToggle.closest("label")?.addEventListener("click", (e) => e.stopPropagation());
    verboseToggle.addEventListener("change", async () => {
      await refreshSnapshot();
    });
  }

  document.getElementById("modelFamilyInput").addEventListener("change", async () => {
    populateModelOptions();
    await updateMermaidPreview();
  });
  document.getElementById("modelNameInput").addEventListener("change", async () => {
    await loadCurriculumStatus();
    await updateMermaidPreview();
  });
  document.getElementById("reasoningInput").addEventListener("change", updateMermaidPreview);
  ["modelFamilyInput", "modelNameInput", "reasoningInput"]
    .forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", () => {
          updateHarnessReadOnlySummary();
        });
      }
    });

  document.getElementById("exampleInput").addEventListener("change", (event) => {
    applyExample(event.target.value);
  });

  document.getElementById("parseSmirksBtn").onclick = async () => {
    try {
      await parseSmirksInput();
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("createBtn").onclick = async () => {
    try {
      await createRun();
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("startBtn").onclick = async () => {
    try {
      await startRun();
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("stopBtn").onclick = async () => {
    try {
      await stopRun();
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("resumeContinueBtn").onclick = async () => {
    try {
      await resumeRun("continue");
      setStatus("Run resumed");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("resumeStopBtn").onclick = async () => {
    try {
      await resumeRun("stop");
      setStatus("Run stopped from pause");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("evaluateBtn").onclick = async () => {
    try {
      await evaluateRun();
      setStatus("Evaluation complete");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("saveEvaluationBtn").onclick = async () => {
    try {
      await saveEvaluation();
      setStatus("Evaluation saved");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("applyHarnessBtn").onclick = async () => {
    try {
      await applyHarnessUpdate();
      setStatus("Harness updated locally");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };

  document.getElementById("createPrBtn").onclick = async () => {
    try {
      await createHarnessPR();
      setStatus("PR workflow executed");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };
  document.getElementById("storeFewShotBtn").onclick = async () => {
    try {
      await storeLatestFewShotCandidate();
      setStatus("Few-shot candidate stored");
    } catch (err) {
      setStatus(String(err.message || err));
    }
  };
  document.getElementById("refreshFewShotBtn").onclick = async () => {
    await refreshFewShotStatus();
  };

  document.getElementById("dryRunDiscardBtn").onclick = async () => {
    if (!runId) return;
    const statusEl = document.getElementById("dryRunDiscardStatus");
    const btn = document.getElementById("dryRunDiscardBtn");
    if (!confirm(`Discard all evidence for dry run ${runId}? This cannot be undone.`)) return;
    btn.disabled = true;
    statusEl.textContent = "Discarding…";
    try {
      const resp = await fetch(`/api/runs/${runId}/discard`, { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Discard failed");
      // Clear all run state
      const discardedId = runId;
      runId = null;
      isDryRun = false;
      if (eventSource) { eventSource.close(); eventSource = null; }
      if (snapshotTimer) { clearInterval(snapshotTimer); snapshotTimer = null; }
      document.getElementById("dryRunPanel").style.display = "none";
      setStatus(`Dry run ${discardedId} discarded — all evidence deleted.`);
      updateButtons("");
      if (terminalOutput) terminalOutput.innerHTML = "";
      _lastTerminalStepCount = 0;
    } catch (err) {
      statusEl.textContent = `Error: ${err.message || err}`;
      btn.disabled = false;
    }
  };

  // Delegated click handler for Mermaid flow nodes
  flowDiagram.addEventListener("click", (event) => {
    const group = event.target.closest('[id^="flowchart-"]');
    if (!group) return;
    const match = group.id.match(/^flowchart-(.+)-\d+$/);
    if (!match) return;

    const nodeId = match[1];
    lastSelectedNodeId = nodeId;

    // Clear previous highlight and apply new one
    const prev = flowDiagram.querySelector(".flow-node-selected");
    if (prev) prev.classList.remove("flow-node-selected");
    group.classList.add("flow-node-selected");

    showNodeDetail(nodeId);
  });

  // Version selector
  document.getElementById("promptVersionSelect").addEventListener("change", (e) => {
    onPromptVersionChange(e.target.value);
  });

  document.getElementById("orchestrationModeInput")?.addEventListener("change", () => {
    updateOrchestrationModeUi();
  });
  document.getElementById("frontendModeInput").addEventListener("change", () => {
    applyFrontendMode();
    applyHarnessTestingMode();
  });
  document.getElementById("harnessSelectionInput").addEventListener("change", () => {
    applyHarnessTestingMode();
    if (getHarnessSelection().type === "existing") {
      showHarnessSettingsModal();
    }
  });

  // Pipeline Editor
  document.getElementById("editPipelineBtn").addEventListener("click", showPipelineEditor);
  document.getElementById("closePipelineEditorBtn").addEventListener("click", hidePipelineEditor);
  document.getElementById("pipelineEditorBackdrop").addEventListener("click", hidePipelineEditor);
  document.getElementById("pipelineSaveBtn").addEventListener("click", saveHarnessConfig);
  document.getElementById("pipelineResetBtn").addEventListener("click", resetHarnessToDefault);

  // Add Module Wizard
  document.getElementById("addModuleBtn").addEventListener("click", showAddModuleWizard);
  document.getElementById("closeAddModuleBtn").addEventListener("click", hideAddModuleWizard);
  document.getElementById("addModuleBackdrop").addEventListener("click", hideAddModuleWizard);
  document.getElementById("wizNextStep1").addEventListener("click", wizardNextStep1);
  document.getElementById("wizBackStep2").addEventListener("click", () => {
    document.getElementById("wizardStep2").style.display = "none";
    document.getElementById("wizardStep1").style.display = "";
  });
  document.getElementById("wizNextStep2").addEventListener("click", wizardNextStep2);
  document.getElementById("wizBackStep3").addEventListener("click", () => {
    document.getElementById("wizardStep3").style.display = "none";
    document.getElementById("wizardStep2").style.display = "";
  });
  document.getElementById("wizFinishBtn").addEventListener("click", wizardFinish);
  document.getElementById("exampleStepFilter").addEventListener("change", () => {
    const stepFilter = document.getElementById("exampleStepFilter")?.value || "all";
    _updateStepCountWarning(stepFilter);
    populateExampleOptions();
  });
  const atomToggle = document.getElementById("displayAtomNumbersToggle");
  if (atomToggle) {
    atomToggle.closest("label")?.addEventListener("click", (e) => e.stopPropagation());
    atomToggle.addEventListener("change", async (event) => {
      showAtomNumbers = Boolean(event.target?.checked);
      await renderMoleculesFromInputs();
      if (latestSnapshotData) {
        await renderReactionVisuals(latestSnapshotData);
        await renderPredictedMechanismComparison(latestSnapshotData);
      }
      if (activeKnownMechanism) {
        await renderKnownMechanismCards(_knownStepsForDisplay(activeKnownMechanism));
      }
    });
  }

  document.getElementById("toggleCurriculumBtn")?.addEventListener("click", () => {
    const panel = document.getElementById("curriculumPanel");
    const nextVisible = panel?.style.display === "none";
    setCurriculumPanelVisible(Boolean(nextVisible));
  });
  document.getElementById("hideCurriculumBtn")?.addEventListener("click", () => {
    setCurriculumPanelVisible(false);
  });
  document.getElementById("viewCurriculumBtn")?.addEventListener("click", () => {
    setCurriculumPanelVisible(true);
    setPrimaryView("curriculum");
  });
  document.getElementById("viewHistoryBtn")?.addEventListener("click", () => {
    setCurriculumPanelVisible(true);
    setPrimaryView("history");
  });
  document.getElementById("curriculumSubmitBtn")?.addEventListener("click", async () => {
    try {
      await submitCurriculumRun();
      if (curriculumStatusMessage) curriculumStatusMessage.textContent = "Submitted today's curriculum run.";
    } catch (err) {
      if (curriculumStatusMessage) curriculumStatusMessage.textContent = String(err.message || err);
    }
  });
  document.getElementById("curriculumPublishBtn")?.addEventListener("click", async () => {
    try {
      const result = await publishCurriculumRuns();
      if (curriculumStatusMessage) curriculumStatusMessage.textContent = `Published ${Array.isArray(result.items) ? result.items.length : 0} checkpoint(s).`;
    } catch (err) {
      if (curriculumStatusMessage) curriculumStatusMessage.textContent = String(err.message || err);
    }
  });
  document.getElementById("curriculumRefreshBtn")?.addEventListener("click", async () => {
    try {
      await refreshCurriculumStatus();
      await refreshCurriculumHistory();
    } catch (err) {
      if (curriculumStatusMessage) curriculumStatusMessage.textContent = String(err.message || err);
    }
  });

}

// ---- Verification UI ----

async function loadVerificationEvalSets() {
  try {
    const response = await fetchApi("/api/eval_sets");
    if (!response.ok) return;
    const data = await response.json();
    const select = document.getElementById("verificationEvalSetSelect");
    if (!select) return;
    select.innerHTML = "";
    const items = data.items || data;
    if (!Array.isArray(items) || !items.length) {
      select.innerHTML = '<option value="">No eval sets</option>';
      return;
    }
    items.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = `${item.name} (${item.version})`;
      select.appendChild(opt);
    });
  } catch (_) { /* ignore */ }
}

async function startVerification() {
  const family = document.getElementById("verificationFamilySelect").value;
  const evalSetSelect = document.getElementById("verificationEvalSetSelect");
  const evalSetId = evalSetSelect ? evalSetSelect.value : "";
  const statusEl = document.getElementById("verificationJobStatus");

  if (!evalSetId) {
    if (statusEl) statusEl.textContent = "Please select an eval set first.";
    return;
  }

  if (statusEl) statusEl.textContent = "Starting verification...";

  try {
    const response = await fetchApi("/api/verification/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ eval_set_id: evalSetId, model_family: family }),
    });
    if (!response.ok) {
      if (statusEl) statusEl.textContent = "Failed to start verification.";
      return;
    }
    const data = await response.json();
    const jobId = data.job_id;
    if (statusEl) statusEl.textContent = `Verification job started: ${jobId}`;

    // Poll for progress
    const pollInterval = setInterval(async () => {
      try {
        const jobResp = await fetch(`/api/verification/jobs/${jobId}`);
        if (!jobResp.ok) { clearInterval(pollInterval); return; }
        const job = await jobResp.json();
        const progress = job.progress || {};
        if (statusEl) {
          statusEl.textContent = `[${job.status}] ${progress.message || ""} (step ${progress.current_step || 0}/${progress.total_steps || "?"})`;
        }
        if (job.status === "completed" || job.status === "failed") {
          clearInterval(pollInterval);
          await loadVerificationResults();
        }
      } catch (_) { clearInterval(pollInterval); }
    }, 3000);
  } catch (_) {
    if (statusEl) statusEl.textContent = "Error starting verification.";
  }
}

async function loadVerificationResults() {
  const family = document.getElementById("verificationFamilySelect").value;
  const tableEl = document.getElementById("verificationTable");
  const diagramEl = document.getElementById("verificationDiagram");
  if (!tableEl) return;

  try {
    const response = await fetch(`/api/verification/history?model_family=${family}`);
    if (!response.ok) { tableEl.innerHTML = "<div class='muted'>No results.</div>"; return; }
    const data = await response.json();
    const items = data.items || [];
    if (!items.length) {
      tableEl.innerHTML = "<div class='muted'>No verification results yet. Run a verification to populate.</div>";
      if (diagramEl) diagramEl.innerHTML = "";
      return;
    }

    // Group by harness_version, then by step
    const versions = {};
    const allSteps = new Set();
    items.forEach((item) => {
      const ver = (item.harness_version || "").substring(0, 12);
      if (!versions[ver]) versions[ver] = {};
      versions[ver][item.step_name] = item.verified_model;
      allSteps.add(item.step_name);
    });

    const versionKeys = Object.keys(versions);
    const stepNames = Array.from(allSteps).sort();

    let html = '<table class="verification-table"><thead><tr><th>Step</th>';
    versionKeys.forEach((v) => { html += `<th title="${v}">${v}</th>`; });
    html += "</tr></thead><tbody>";
    stepNames.forEach((step) => {
      html += `<tr><td>${step}</td>`;
      versionKeys.forEach((v) => {
        const model = versions[v][step] || "-";
        const shortModel = model.includes("/") ? model.split("/").pop() : model;
        html += `<td title="${model}">${shortModel}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    tableEl.innerHTML = html;

    // Render verification diagram for the latest version
    if (diagramEl && window.mermaid && versionKeys.length) {
      const latestVersion = versionKeys[0];
      const latestModels = versions[latestVersion];
      await renderVerificationDiagram(diagramEl, latestModels);
    }
  } catch (_) {
    tableEl.innerHTML = "<div class='muted'>Failed to load verification results.</div>";
  }
}

async function renderVerificationDiagram(container, stepModels) {
  try {
    const flowResp = await fetchApi("/api/catalog/flow_template");
    if (!flowResp.ok) return;
    const flowData = await flowResp.json();
    const nodes = flowData.nodes || [];
    const edges = flowData.edges || [];
    if (!nodes.length) return;

    const MODEL_SYMBOLS = ["\u00b9", "\u00b2", "\u00b3", "\u2074", "\u2075", "\u2076", "\u2077", "\u2078"];
    const uniqueModels = [];
    nodes.forEach((node) => {
      const model = stepModels[node.id];
      if (model && !uniqueModels.includes(model)) uniqueModels.push(model);
    });
    const modelSymbolMap = {};
    uniqueModels.forEach((model, i) => {
      modelSymbolMap[model] = MODEL_SYMBOLS[i] || `[${i + 1}]`;
    });

    const lines = ["flowchart TD"];
    nodes.forEach((node) => {
      const model = stepModels[node.id];
      const sym = model ? ` ${modelSymbolMap[model]}` : "";
      lines.push(`  ${node.id}["${node.label}${sym}"]`);
    });
    edges.forEach((edge) => {
      const label = edge.label ? `|${edge.label}|` : "";
      lines.push(`  ${edge.source} -->${label} ${edge.target}`);
    });

    nodes.forEach((node) => {
      const style = getFlowNodeStyle(resolveNodeKind(node), "completed");
      lines.push(`  style ${node.id} fill:${style.fill},stroke:${style.stroke},stroke-width:1px,opacity:1`);
    });

    const graphText = lines.join("\n");
    const renderId = `verify-${Date.now()}`;
    const rendered = await window.mermaid.render(renderId, graphText);
    container.innerHTML = rendered.svg;

    // Model key
    const entries = Object.entries(modelSymbolMap);
    if (entries.length) {
      const shortName = (m) => (m.includes("/") ? m.split("/").pop() : m);
      const keyHtml = entries
        .map(([model, sym]) => `<span class="model-key-item" title="${model}"><strong>${sym}</strong>\u00a0${shortName(model)}</span>`)
        .join("");
      container.innerHTML += `<div class="model-key" style="display:flex;gap:1rem;margin-top:0.5rem;">${keyHtml}</div>`;
    }
  } catch (_) { /* ignore */ }
}

bootstrap().catch((err) => {
  console.error("bootstrap failed", err);
  setStatus(`UI failed to initialize: ${err.message || err}`);
});
