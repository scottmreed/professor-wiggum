"""Plugin interface placeholder for future external mechanism validation providers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


class ExternalValidator(Protocol):
    """Contract for external validation plugins (future extension point)."""

    name: str

    def validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass(slots=True)
class ExternalValidatorRegistry:
    """Runtime registry for optional external validators."""

    validators: List[ExternalValidator] = field(default_factory=list)

    def register(self, validator: ExternalValidator) -> None:
        self.validators.append(validator)

    def run_all(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for validator in self.validators:
            try:
                result = validator.validate(payload)
            except Exception as exc:  # pragma: no cover - defensive plugin guard
                result = {"validator": getattr(validator, "name", "unknown"), "error": str(exc)}
            else:
                result = {"validator": getattr(validator, "name", "unknown"), **dict(result)}
            results.append(result)
        return results
