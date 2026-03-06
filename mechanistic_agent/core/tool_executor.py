"""Uniform adapters around deterministic and LLM-backed tools."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mechanistic_agent.smiles_utils import strip_atom_mapping_list, strip_atom_mapping_optional
from mechanistic_agent.tools import (
    analyse_balance,
    assess_initial_conditions,
    attempt_atom_mapping,
    attempt_atom_mapping_for_step,
    fingerprint_functional_groups,
    predict_mechanistic_step,
    predict_missing_reagents,
    predict_missing_reagents_for_candidate,
    propose_intermediates,
    recommend_ph,
    select_reaction_type,
)


class ToolExecutor:
    """Executes tool calls and normalizes outputs into dictionaries."""

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except json.JSONDecodeError:
            return {"raw_output": raw}

    @staticmethod
    def _sanitize_species_list(items: List[str]) -> List[str]:
        return strip_atom_mapping_list(items)

    @staticmethod
    def _sanitize_species(item: Optional[str]) -> Optional[str]:
        return strip_atom_mapping_optional(item)

    def run_balance(self, starting: List[str], products: List[str]) -> Dict[str, Any]:
        return self._parse(analyse_balance(starting, products))

    def run_ph_recommendation(
        self,
        starting: List[str],
        products: List[str],
        ph: Optional[float],
    ) -> Dict[str, Any]:
        return self._parse(recommend_ph(starting, products, ph))

    def run_conditions(
        self,
        starting: List[str],
        products: List[str],
        ph: Optional[float],
    ) -> Dict[str, Any]:
        return self._parse(
            assess_initial_conditions(
                self._sanitize_species_list(starting),
                self._sanitize_species_list(products),
                ph,
            )
        )

    def run_mapping(self, starting: List[str], products: List[str]) -> Dict[str, Any]:
        return self._parse(
            attempt_atom_mapping(
                self._sanitize_species_list(starting),
                self._sanitize_species_list(products),
            )
        )

    def run_functional_groups(self, smiles: List[str]) -> Dict[str, Any]:
        return self._parse(fingerprint_functional_groups(smiles))

    def run_missing_reagents(
        self,
        *,
        starting: List[str],
        products: List[str],
        conditions_guidance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        guidance = json.dumps(conditions_guidance) if conditions_guidance else None
        return self._parse(
            predict_missing_reagents(
                starting_materials=self._sanitize_species_list(starting),
                products=self._sanitize_species_list(products),
                conditions_guidance=guidance,
            )
        )

    def run_intermediates(
        self,
        *,
        starting: List[str],
        products: List[str],
        current_state: List[str],
        previous_intermediates: List[str],
        ph: Optional[float],
        temperature: Optional[float],
        step_index: int,
        step_mapping_context: Optional[Dict[str, Any]] = None,
        template_guidance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._parse(
            propose_intermediates(
                starting_materials=self._sanitize_species_list(starting),
                products=self._sanitize_species_list(products),
                current_state=self._sanitize_species_list(current_state),
                previous_intermediates=self._sanitize_species_list(previous_intermediates),
                mapped_starting_materials=list(starting),
                mapped_products=list(products),
                mapped_current_state=list(current_state),
                ph=ph,
                temperature=temperature,
                step_index=step_index,
                step_mapping_context=step_mapping_context,
                template_guidance=template_guidance,
            )
        )

    def run_reaction_type_mapping(
        self,
        *,
        starting: List[str],
        products: List[str],
        balance_analysis: Optional[Dict[str, Any]] = None,
        functional_groups: Optional[Dict[str, Any]] = None,
        ph_recommendation: Optional[Dict[str, Any]] = None,
        initial_conditions: Optional[Dict[str, Any]] = None,
        missing_reagents: Optional[Dict[str, Any]] = None,
        atom_mapping: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._parse(
            select_reaction_type(
                starting_materials=self._sanitize_species_list(starting),
                products=self._sanitize_species_list(products),
                balance_analysis=balance_analysis,
                functional_groups=functional_groups,
                ph_recommendation=ph_recommendation,
                initial_conditions=initial_conditions,
                missing_reagents=missing_reagents,
                atom_mapping=atom_mapping,
            )
        )

    def run_candidate_rescue(
        self,
        *,
        current_state: List[str],
        resulting_state: List[str],
        failed_checks: Optional[List[str]] = None,
        validation_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._parse(
            predict_missing_reagents_for_candidate(
                current_state=self._sanitize_species_list(current_state),
                resulting_state=self._sanitize_species_list(resulting_state),
                failed_checks=failed_checks,
                validation_details=validation_details,
            )
        )

    def run_step_mapping(
        self,
        *,
        current_state: List[str],
        resulting_state: List[str],
    ) -> Dict[str, Any]:
        return self._parse(
            attempt_atom_mapping_for_step(
                current_state=self._sanitize_species_list(current_state),
                resulting_state=self._sanitize_species_list(resulting_state),
            )
        )

    def run_mechanism_step(
        self,
        *,
        step_index: int,
        current_state: List[str],
        target_products: List[str],
        predicted_intermediate: Optional[str],
        resulting_state: Optional[List[str]],
        electron_pushes: Optional[List[Dict[str, object]]],
        reaction_smirks: Optional[str],
        previous_intermediates: List[str],
        starting_materials: List[str],
        note: Optional[str],
    ) -> Dict[str, Any]:
        # Keep a deterministic fallback only when proposal output is incomplete.
        pushes = electron_pushes or [{"kind": "lone_pair", "source_atom": "0", "target_atom": "1", "electrons": 2}]
        return self._parse(
            predict_mechanistic_step(
                step_index=step_index,
                current_state=current_state,
                target_products=target_products,
                electron_pushes=pushes,
                reaction_smirks=reaction_smirks,
                predicted_intermediate=predicted_intermediate,
                resulting_state=resulting_state,
                previous_intermediates=previous_intermediates,
                note=note,
                starting_materials=starting_materials,
            )
        )
