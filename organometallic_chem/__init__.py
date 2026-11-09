"""Shared organometallic chemistry: house SMILES conventions, MLX electron counting,
catalytic-cycle validation and the covalent <-> dative notation bridge.

RDKit-only, with no imports from Wiggum or ChemIllusion, so the directory can be
split into its own distribution unchanged. Both apps pin ``__version__``;
``HOUSE_CONVENTIONS`` changes only with a major version bump
(docs/PRD_catalysis_handling.md, R1.1-R1.3).

Extracted from ChemIllusion ``backend/app/utils`` at chem-art-generator c46cef522.
"""

__version__ = "1.0.0"
HOUSE_CONVENTIONS = "v1"

__all__ = ["__version__", "HOUSE_CONVENTIONS"]
