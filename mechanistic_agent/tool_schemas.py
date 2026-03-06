"""OpenAI-format tool schemas for forced tool calling.

Some schemas include an optional ``text`` field for supplemental commentary.
That field is never required and is handled separately from the structured
payload used for deterministic validation.
"""
from __future__ import annotations

from typing import Any, Dict


def build_tool_choice(tool_name: str) -> Dict[str, Any]:
    """Build a forced ``tool_choice`` dict for a single tool.

    Works with both the OpenAI Chat Completions API and OpenRouter (which
    accepts OpenAI-compatible payloads).
    """
    return {"type": "function", "function": {"name": tool_name}}


# ---------------------------------------------------------------------------
# Candidate item schema (shared across acid/base lists)
# ---------------------------------------------------------------------------

_CANDIDATE_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Reagent name."},
        "smiles": {
            "type": "string",
            "description": (
                "Valid SMILES representation (e.g., 'OS(=O)(=O)O' for H2SO4, "
                "'O' for water, 'Cl' for HCl). Must be RDKit-parseable. "
                "Never use molecular formulas in brackets like '[H2SO4]'."
            ),
        },
        "role": {"type": "string", "description": "Role in the reaction (acid or base)."},
        "justification": {"type": "string", "description": "10 words or fewer explaining the choice."},
    },
    "required": ["name"],
}

# ---------------------------------------------------------------------------
# 1. assess_initial_conditions
# ---------------------------------------------------------------------------

ASSESS_CONDITIONS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "assess_conditions_result",
        "description": (
            "Return the assessed reaction conditions including environment, pH, "
            "and compatible acid or base candidates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Free-form reasoning or commentary about the assessment. "
                        "Use this field for any explanatory text you want to provide."
                    ),
                },
                "environment": {
                    "type": "string",
                    "enum": ["acidic", "basic", "neutral"],
                    "description": "Whether the reaction proceeds under acidic, basic, or neutral conditions.",
                },
                "representative_ph": {
                    "type": "number",
                    "description": "Single float between 0 and 14 representing the estimated pH.",
                },
                "ph_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional [lower, upper] pH range as two numbers.",
                },
                "justification": {
                    "type": "string",
                    "description": "12 words or fewer summarising your reasoning.",
                },
                "acid_candidates": {
                    "type": "array",
                    "items": _CANDIDATE_ITEM,
                    "description": "Up to 3 acid reagent suggestions (name, smiles, role, justification). Only populate when environment is acidic.",
                },
                "base_candidates": {
                    "type": "array",
                    "items": _CANDIDATE_ITEM,
                    "description": "Up to 3 base reagent suggestions (name, smiles, role, justification). Only populate when environment is basic.",
                },
                "warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional warning phrases, each 10 words or fewer.",
                },
            },
            "required": ["environment", "representative_ph"],
        },
    },
}

# ---------------------------------------------------------------------------
# 2. predict_missing_reagents
# ---------------------------------------------------------------------------

MISSING_REAGENTS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "missing_reagents_result",
        "description": (
            "Return missing reactants and products needed to balance the reaction "
            "stoichiometry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Free-form reasoning about the balancing analysis. "
                        "Use this field for any explanatory text."
                    ),
                },
                "missing_reactants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "SMILES strings for molecules to add to the reactant side. "
                        "Use valid SMILES only (e.g., water='O' NOT '[H2O]', "
                        "H2SO4='OS(=O)(=O)O', HCl='Cl', NaOH='[Na+].[OH-]'). "
                        "Never use molecular formulas in brackets as SMILES."
                    ),
                },
                "missing_products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "SMILES strings for molecules to add to the product side. "
                        "Use valid SMILES only (e.g., water='O' NOT '[H2O]', "
                        "CO2='O=C=O', ethanol='CCO'). "
                        "Never use molecular formulas in brackets as SMILES."
                    ),
                },
                "verification": {
                    "type": "object",
                    "description": "Optional atom count verification showing your calculations.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the balancing.",
                },
            },
            "required": ["missing_reactants", "missing_products"],
        },
    },
}

# ---------------------------------------------------------------------------
# 3. attempt_atom_mapping
# ---------------------------------------------------------------------------

_MAPPED_ATOM_ITEM = {
    "type": "object",
    "properties": {
        "product_atom": {
            "type": "string",
            "description": "Product atom reference as '<product_smiles>#<index>'.",
        },
        "source": {
            "type": "object",
            "properties": {
                "molecule_index": {"type": "integer"},
                "smiles": {"type": "string"},
                "atom_index": {"type": "integer"},
            },
            "description": "Source atom in the starting materials.",
        },
        "notes": {"type": "string", "description": "Notes on this mapping."},
    },
}

ATOM_MAPPING_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atom_mapping_result",
        "description": (
            "Return atom-to-atom mapping between reactants and products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Free-form reasoning about the mapping strategy. "
                        "Use this field for any explanatory text."
                    ),
                },
                "mapped_atoms": {
                    "type": "array",
                    "items": _MAPPED_ATOM_ITEM,
                    "description": (
                        "List of mapped atom entries. Use null/omit if unable to map."
                    ),
                },
                "unmapped_atoms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Descriptions of atoms without confident assignments.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score in [0.0, 1.0].",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation referencing reactive sites or symmetry.",
                },
                "missing_reagent_considerations": {
                    "type": "string",
                    "description": "Notes about potential missing reagents or products.",
                },
            },
            "required": ["confidence", "reasoning"],
        },
    },
}

# ---------------------------------------------------------------------------
# 4. select_reaction_type
# ---------------------------------------------------------------------------

_REACTION_TYPE_TOP_CANDIDATE = {
    "type": "object",
    "properties": {
        "label_exact": {"type": "string"},
        "type_id": {"type": "string"},
        "confidence": {"type": "number"},
    },
}

REACTION_TYPE_SELECTION_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "reaction_type_selection_result",
        "description": (
            "Select the most likely mechanism type label from the provided taxonomy "
            "or return 'no_match' when no taxonomy entry fits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Optional free-form reasoning.",
                },
                "selected_label_exact": {
                    "type": "string",
                    "description": "Exact taxonomy label text or 'no_match'.",
                },
                "selected_type_id": {
                    "type": ["string", "null"],
                    "description": "Stable type id such as mt_001; null when no_match.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score in [0.0, 1.0].",
                },
                "rationale": {
                    "type": "string",
                    "description": "Short rationale for the selection.",
                },
                "top_candidates": {
                    "type": "array",
                    "items": _REACTION_TYPE_TOP_CANDIDATE,
                    "description": "Up to 3 alternative candidates.",
                },
            },
            "required": ["selected_label_exact", "confidence", "rationale"],
        },
    },
}

# ---------------------------------------------------------------------------
# 5. propose_intermediates
# ---------------------------------------------------------------------------

_INTERMEDIATE_ITEM = {
    "type": "object",
    "properties": {
        "smiles": {
            "type": "string",
            "description": (
                "Valid SMILES string for the intermediate. Must be parseable by RDKit. "
                "Use proper SMILES notation (e.g., water='O', not '[H2O]'). "
                "Never use natural-language descriptors, molecular formulas, or prose."
            ),
        },
        "type": {"type": "string", "description": "Optional type annotation."},
        "note": {"type": "string", "description": "Optional note about this intermediate."},
    },
    "required": ["smiles"],
}

INTERMEDIATES_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "intermediates_result",
        "description": (
            "Return proposed intermediates for the next mechanistic step "
            "or confirm the reaction is complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Free-form mechanistic reasoning. "
                        "Use this field for any explanatory text."
                    ),
                },
                "classification": {
                    "type": "string",
                    "enum": ["intermediate_step", "final_step"],
                    "description": "Whether more steps are needed or the target products are reached.",
                },
                "intermediates": {
                    "type": "array",
                    "items": _INTERMEDIATE_ITEM,
                    "description": "Proposed intermediate species as SMILES (empty list if final step).",
                },
                "analysis": {
                    "type": "string",
                    "description": "Concise analysis explaining the mechanistic logic.",
                },
            },
            "required": ["classification", "intermediates", "analysis"],
        },
    },
}


# ---------------------------------------------------------------------------
# 6. propose_mechanism_step (multi-candidate)
# ---------------------------------------------------------------------------

_MECHANISM_STEP_CANDIDATE = {
    "type": "object",
    "properties": {
        "rank": {
            "type": "integer",
            "description": "Rank of this candidate. 1 = most likely, 2 = next, 3 = least likely.",
        },
        "intermediate_smiles": {
            "type": "string",
            "description": (
                "Valid SMILES string for the proposed intermediate product of this step. "
                "Must be parseable by RDKit. Use standard SMILES (e.g., 'O' for water, "
                "'CCO' for ethanol). Never use molecular formula notation like '[H2O]' "
                "or natural-language descriptors like 'acid-catalyzed'."
            ),
        },
        "intermediate_type": {
            "type": "string",
            "description": "Type annotation (e.g., tetrahedral_intermediate, carbocation, enolate).",
        },
        "reaction_description": {
            "type": "string",
            "description": (
                "Brief balanced description of what happens in this step "
                "(bonds broken/formed, electron movement)."
            ),
        },
        "reaction_smirks": {
            "type": "string",
            "description": (
                "CXSMILES/SMIRKS for this step including a '|mech:v1;...|' block. "
                "Use lp:a>b for lone-pair attack, pi:a-b>c for pi-bond donation, "
                "and sigma:a-b>c for sigma-bond attack."
            ),
        },
        "electron_pushes": {
            "type": "array",
            "description": "Explicit electron movements for this step.",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["lone_pair", "pi_bond", "sigma_bond"],
                        "description": "Source type for the electron move.",
                    },
                    "source_atom": {
                        "type": "string",
                        "description": "Atom-map index holding the attacking lone pair.",
                    },
                    "source_bond": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Ordered atom-map pair for a pi or sigma bond source.",
                    },
                    "through_atom": {
                        "type": "string",
                        "description": "Which end of source_bond delivers the electrons.",
                    },
                    "target_atom": {
                        "type": "string",
                        "description": "Atom-map index where electrons end.",
                    },
                    "electrons": {
                        "type": "integer",
                        "enum": [2],
                        "description": "Number of electrons moved; use 2 for arrow-pushing pairs.",
                    },
                    "notation": {
                        "type": "string",
                        "description": "Optional compact move token such as 'lp:4>2' or 'pi:1-2>2'.",
                    },
                    "description": {"type": "string", "description": "Optional short description."},
                },
                "required": ["kind", "target_atom", "electrons"],
            },
        },
        "resulting_state": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional explicit species list after this step. "
                "Prefer including both newly formed species and consumed/remaining species, "
                "plus any minimal byproducts/reagents needed for atom-balance consistency."
            ),
        },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence in this candidate being the correct next step.",
                },
                "template_alignment": {
                    "type": "string",
                    "enum": ["aligned", "partial", "not_aligned", "unknown"],
                    "description": "Optional alignment of this candidate to the selected reaction-type template.",
                },
                "template_alignment_reason": {
                    "type": "string",
                    "description": "Optional reason for the template alignment classification.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional mechanistic reasoning for this candidate.",
                },
            },
    "required": [
        "rank",
        "intermediate_smiles",
        "reaction_description",
        "reaction_smirks",
        "electron_pushes",
    ],
}

MECHANISM_STEP_PROPOSAL_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "mechanism_step_proposal_result",
        "description": (
            "Propose up to 3 ranked candidates for the next mechanism step. "
            "Each candidate includes the intermediate product SMILES and a "
            "balanced description of the reaction step. Rank 1 is the most "
            "likely candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Free-form mechanistic reasoning. "
                        "Use this field for any explanatory text."
                    ),
                },
                "classification": {
                    "type": "string",
                    "enum": ["intermediate_step", "final_step"],
                    "description": (
                        "Whether more steps are needed or the target products are reached."
                    ),
                },
                "candidates": {
                    "type": "array",
                    "items": _MECHANISM_STEP_CANDIDATE,
                    "description": (
                        "Up to 3 candidates ranked by likelihood. "
                        "Each includes intermediate SMILES and a balanced reaction description. "
                        "Empty list if final step."
                    ),
                },
                "analysis": {
                    "type": "string",
                    "description": "Concise analysis explaining the mechanistic logic.",
                },
            },
            "required": ["classification", "candidates", "analysis"],
        },
    },
}


# ---------------------------------------------------------------------------
# 6. predict_full_mechanism  (baseline / harness-free single-shot tool)
# ---------------------------------------------------------------------------

_ELECTRON_PUSH_ITEM: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["lone_pair", "pi_bond", "sigma_bond"],
            "description": "Source type for the electron move.",
        },
        "source_atom": {
            "type": "integer",
            "description": "Atom-map number of the lone-pair source atom.",
        },
        "source_bond": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Ordered atom-map pair for a pi or sigma bond source.",
        },
        "through_atom": {
            "type": "integer",
            "description": "Which end of source_bond delivers the electrons.",
        },
        "target_atom": {
            "type": "integer",
            "description": "Atom-map number of the electron destination atom.",
        },
        "electrons": {
            "type": "integer",
            "enum": [2],
            "description": "Number of electrons moved (use 2 for a lone pair).",
        },
    },
    "required": ["kind", "target_atom", "electrons"],
}

_MECHANISM_STEP_ITEM: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "step_index": {
            "type": "integer",
            "description": "1-based step index.",
        },
        "step_label": {
            "type": "string",
            "description": "Short human-readable label (e.g. 'proton transfer', 'nucleophilic addition').",
        },
        "current_state": {
            "type": "array",
            "items": {"type": "string"},
            "description": "SMILES list of all species present before this step.",
        },
        "resulting_state": {
            "type": "array",
            "items": {"type": "string"},
            "description": "SMILES list of all species present after this step (including byproducts).",
        },
        "predicted_intermediate": {
            "type": "string",
            "description": "SMILES of the focal intermediate produced or consumed in this step.",
        },
        "reaction_smirks": {
            "type": "string",
            "description": "SMIRKS/CXSMILES for the elementary transformation.",
        },
        "electron_pushes": {
            "type": "array",
            "items": _ELECTRON_PUSH_ITEM,
            "description": "Arrow-push descriptors for this step.",
        },
        "contains_target_product": {
            "type": "boolean",
            "description": "True when resulting_state contains the final target product.",
        },
    },
    "required": ["step_index", "current_state", "resulting_state"],
}

PREDICT_FULL_MECHANISM_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "predict_full_mechanism",
        "description": (
            "Return the complete ordered stepwise mechanism for a reaction in a single call. "
            "Each step must be a single elementary transformation. "
            "The final step's resulting_state must contain the target product SMILES."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Free-form reasoning about the overall mechanism strategy. "
                        "Use this for any explanatory commentary."
                    ),
                },
                "mechanism_type": {
                    "type": "string",
                    "description": "Brief name of the overall mechanism class (e.g. SN2, E2, aldol condensation).",
                },
                "steps": {
                    "type": "array",
                    "items": _MECHANISM_STEP_ITEM,
                    "description": "Ordered list of all elementary mechanism steps from starting materials to products.",
                },
            },
            "required": ["steps"],
        },
    },
}


__all__ = [
    "build_tool_choice",
    "ASSESS_CONDITIONS_TOOL",
    "MISSING_REAGENTS_TOOL",
    "ATOM_MAPPING_TOOL",
    "INTERMEDIATES_TOOL",
    "MECHANISM_STEP_PROPOSAL_TOOL",
    "PREDICT_FULL_MECHANISM_TOOL",
]
