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
        functional_groups_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return self._parse(
            assess_initial_conditions(
                self._sanitize_species_list(starting),
                self._sanitize_species_list(products),
                ph,
                functional_groups_enabled=functional_groups_enabled,
            )
        )

    def run_mapping(
        self,
        starting: List[str],
        products: List[str],
        functional_groups_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return self._parse(
            attempt_atom_mapping(
                self._sanitize_species_list(starting),
                self._sanitize_species_list(products),
                functional_groups_enabled=functional_groups_enabled,
            )
        )

    def run_functional_groups(self, smiles: List[str]) -> Dict[str, Any]:
        return self._parse(fingerprint_functional_groups(self._sanitize_species_list(smiles)))

    def run_missing_reagents(
        self,
        *,
        starting: List[str],
        products: List[str],
        conditions_guidance: Optional[Dict[str, Any]] = None,
        functional_groups_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        guidance = json.dumps(conditions_guidance) if conditions_guidance else None
        return self._parse(
            predict_missing_reagents(
                starting_materials=self._sanitize_species_list(starting),
                products=self._sanitize_species_list(products),
                conditions_guidance=guidance,
                functional_groups_enabled=functional_groups_enabled,
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
        mapped_loop_current_state: Optional[List[str]] = None,
        mapped_starting_materials: Optional[List[str]] = None,
        mapped_products: Optional[List[str]] = None,
        mapped_current_state: Optional[List[str]] = None,
        functional_groups_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # loop_state_mapping="mapped" (opt-in): the loop's mapped copy replaces
        # the stripped current_state; pre-loop inputs stay stripped.
        loop_current_state = (
            [str(s) for s in mapped_loop_current_state]
            if mapped_loop_current_state
            else self._sanitize_species_list(current_state)
        )
        # mapped_* carry atom maps on purpose (rendered from the global
        # atom_mapping output) and are therefore not sanitized.
        return self._parse(
            propose_intermediates(
                starting_materials=self._sanitize_species_list(starting),
                products=self._sanitize_species_list(products),
                current_state=loop_current_state,
                previous_intermediates=self._sanitize_species_list(previous_intermediates),
                mapped_starting_materials=list(mapped_starting_materials or []),
                mapped_products=list(mapped_products or []),
                mapped_current_state=list(mapped_current_state or []),
                ph=ph,
                temperature=temperature,
                step_index=step_index,
                step_mapping_context=step_mapping_context,
                template_guidance=template_guidance,
                functional_groups_enabled=functional_groups_enabled,
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

    def run_reaction_type_mapping_jev(
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
        jev_config: Any = None,
        client: Any = None,
    ) -> Dict[str, Any]:
        """Reaction-type Choice via Jev (decision_policy.reaction_type == "jev").

        Species are map-stripped like every other model input. On a Jev
        failure the configured fallback may call the LLM selector above.
        """
        from .reaction_type_jev import select_reaction_type_jev

        context = dict(
            balance_analysis=balance_analysis,
            functional_groups=functional_groups,
            ph_recommendation=ph_recommendation,
            initial_conditions=initial_conditions,
            missing_reagents=missing_reagents,
            atom_mapping=atom_mapping,
        )
        return select_reaction_type_jev(
            starting_materials=self._sanitize_species_list(starting),
            products=self._sanitize_species_list(products),
            jev_config=jev_config,
            client=client,
            llm_fallback=lambda: self.run_reaction_type_mapping(
                starting=starting, products=products, **context
            ),
            **context,
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
        allowed_extra_species: Optional[List[str]] = None,
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
                allowed_extra_species=allowed_extra_species,
            )
        )
