"""Shared reaction signatures: submission hashes, corpus keys and fingerprints.

This module is the single implementation of the reaction identity recipe used
by Professor Wiggum's FlowER novelty index and by ChemIllusion's mechanism
predictor submissions (ChemIllusion imports it from the pinned Wiggum runtime;
it must not keep its own copy). It is pure RDKit: no database, no network, no
model calls, no logging, and no dependency on either product's schemas.

Three families of output, each with its own version string:

``RECIPE_VERSION`` (``mechanism_submission.v2``)
    :func:`reaction_hashes` — the role-aware submission hashes, a byte-for-byte
    port of ChemIllusion's ``submission_normalizer`` at ``6c45cc8b``. Every
    participant is parsed, atom maps are cleared, and each ``.``-fragment is
    written as canonical isomeric SMILES. Keys are
    ``sha256("mechanism_submission.v2|<kind>|<canonical text>")`` hex digests
    (kinds ``role``, ``stoich``, ``union``, ``core``, ``major``,
    ``role_inchikey``, ``cond``); the analogue cluster id is
    ``"mpc_" + major_key_hash[:20]``. Parity vectors live in
    ``tests/fast/fixtures/reaction_signature_vectors.json``.

``CORPUS_RECIPE_VERSION`` (``reaction_corpus.v1``)
    :func:`corpus_keys` / :func:`corpus_keys_from_submission` — four 64-bit
    keys computed over **standard InChIKeys** of the atom-map-free fragments
    (InChIKeys are stable across RDKit releases; canonical SMILES are not).
    Each key is ``int(sha256("reaction_corpus.v1|<kind>|<text>")[:16], 16)``,
    an unsigned 64-bit integer. Kinds:

    * ``exact`` — sorted set of reactant-side fragment InChIKeys ``>>`` sorted
      set of product-side fragment InChIKeys. Direction matters; order,
      duplicates, atom maps and SMILES spelling do not.
    * ``core`` — the reacting species only: (1) fragments whose InChIKey
      appears on both sides are dropped (regenerated catalysts, spectators),
      then (2) on each side fragments with at most one heavy atom are dropped
      while that side still has a larger fragment (H+, H2O, halide and alkali
      counterions, ...), then sorted set left ``>>`` sorted set right. A side
      that step (1) or (2) would empty falls back to its set from the step
      before. This is what lets a FlowER state that carries ``[H+]`` on both
      sides and releases water match a user's bare substrate → product.
    * ``endpoint`` — every left-side species (for a submission: reactants,
      reagents, catalysts and solvents) ``>>`` products. For a FlowER
      reaction the left side already holds everything, so it equals ``exact``.
    * ``family`` — the heaviest fragment(s) among each side's ``core``
      fragments (the fragments with the maximum heavy-atom count), mirroring
      the ``major`` submission key.

    Submission mapping (:func:`corpus_keys_from_submission`): ``exact``,
    ``core`` and ``family`` use reactants against products only; ``endpoint``
    uses reactants + reagents + catalysts + solvents against products. So a
    user submission and a FlowER start→final-state reaction of the same
    chemistry give equal ``core``/``family`` keys whenever the user's reactants
    and products name the same reacting species, and equal ``exact`` keys only
    when they also list every catalyst, spectator and small byproduct FlowER
    carries in its initial and final states.

Fingerprints (``FINGERPRINT_SPEC``)
    :func:`reaction_fingerprints` returns two 256-bit (32-byte) fingerprints
    built from the sorted *set* of canonical, atom-map-free fragments per side:

    * the reaction **difference** fingerprint — RDKit
      ``rdChemReactions.CreateDifferenceFingerprintForReaction`` with
      ``fpType=MorganFP`` (RDKit's fixed radius 2), ``fpSize=256`` and
      ``includeAgents=False``; a bit is set when the product-minus-reactant
      count for that folded feature is non-zero (the sign is discarded).
      Species present on both sides cancel exactly.
    * the **product** fingerprint — Morgan radius-2, 256-bit
      (``rdFingerprintGenerator``) bit vectors OR-ed over the product
      fragments that do not also appear on the reactant side (all product
      fragments when every one is shared).

    Bit ``i`` lives in byte ``i // 8`` at bit position ``i % 8`` (LSB first,
    ``numpy.packbits(..., bitorder="little")``). Fingerprints are only
    comparable when built with the same RDKit version; the novelty index
    manifest records it. :func:`tanimoto` compares two packed fingerprints.
    :func:`corpus_keys_and_fingerprints` computes keys and fingerprints for a
    FlowER-shaped reaction while parsing each species once (the index builder
    uses it; results equal the separate calls).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions, rdMolDescriptors

try:  # InChI support is an optional RDKit build component.
    from rdkit.Chem import inchi as _inchi
except Exception:  # pragma: no cover - depends on the RDKit build
    _inchi = None

try:
    from rdkit.Chem import rdFingerprintGenerator as _fpgen
except Exception:  # pragma: no cover - very old RDKit
    _fpgen = None

__all__ = [
    "RECIPE_VERSION",
    "CORPUS_RECIPE_VERSION",
    "RDKIT_VERSION",
    "FINGERPRINT_BITS",
    "FINGERPRINT_BYTES",
    "FINGERPRINT_SPEC",
    "ROLE_ORDER",
    "CorpusKeyError",
    "SpeciesSignature",
    "ParticipantSignature",
    "ReactionHashes",
    "CorpusKeys",
    "normalize_species",
    "parse_reaction_smiles",
    "reaction_hashes",
    "corpus_keys",
    "corpus_keys_from_submission",
    "reaction_fingerprints",
    "corpus_keys_and_fingerprints",
    "tanimoto",
]

RECIPE_VERSION = "mechanism_submission.v2"
CORPUS_RECIPE_VERSION = "reaction_corpus.v1"
RDKIT_VERSION: str = rdkit.__version__

FINGERPRINT_BITS = 256
FINGERPRINT_BYTES = FINGERPRINT_BITS // 8
FINGERPRINT_SPEC: Dict[str, Any] = {
    "bits": FINGERPRINT_BITS,
    "bit_order": "little",
    "difference": {
        "function": "rdChemReactions.CreateDifferenceFingerprintForReaction",
        "fpType": "MorganFP",
        "fpSize": FINGERPRINT_BITS,
        "includeAgents": False,
        "binarize": "count != 0",
        "input": "sorted set of canonical map-free fragments per side",
    },
    "product": {
        "function": "rdFingerprintGenerator.GetMorganGenerator",
        "radius": 2,
        "fpSize": FINGERPRINT_BITS,
        "combine": "OR over product fragments not also on the reactant side",
    },
}

# (roles-dict key, participant role) in ChemIllusion's display order.
ROLE_ORDER: Tuple[Tuple[str, str], ...] = (
    ("reactants", "reactant"),
    ("products", "product"),
    ("reagents", "reagent"),
    ("catalysts", "catalyst"),
    ("solvents", "solvent"),
)
_ROLE_KEY_PARTS: Tuple[Tuple[str, str], ...] = (
    ("R", "reactants"),
    ("G", "reagents"),
    ("C", "catalysts"),
    ("S", "solvents"),
    ("P", "products"),
)
_CORPUS_KINDS: Tuple[str, ...] = ("exact", "core", "endpoint", "family")


class CorpusKeyError(ValueError):
    """A reaction cannot be keyed: an empty side, an unparsable species, or a
    fragment without a standard InChIKey."""


def _h(kind: str, text: str) -> str:
    return hashlib.sha256(f"{RECIPE_VERSION}|{kind}|{text}".encode("utf-8")).hexdigest()


def _corpus_hash(kind: str, text: str) -> int:
    digest = hashlib.sha256(f"{CORPUS_RECIPE_VERSION}|{kind}|{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# species
# ---------------------------------------------------------------------------


@dataclass
class SpeciesSignature:
    """One species as RDKit sees it. ``error`` is set when it did not parse."""

    input: str
    canonical_smiles: Optional[str] = None
    inchikey: Optional[str] = None
    formula: Optional[str] = None
    formal_charge: Optional[int] = None
    heavy_atoms: Optional[int] = None
    atom_counts: Dict[str, int] = field(default_factory=dict)
    fragments: List[str] = field(default_factory=list)
    fragment_heavy_atoms: List[int] = field(default_factory=list)
    fragment_inchikeys: List[Optional[str]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.error is None


def _inchikey(mol: Chem.Mol) -> Optional[str]:
    if _inchi is None:
        return None
    try:
        key = _inchi.MolToInchiKey(mol)
    except Exception:
        return None
    return key or None


def _atom_counts(mol: Chem.Mol) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        hydrogens = atom.GetTotalNumHs()
        if hydrogens:
            counts["H"] += hydrogens
    return dict(counts)


def normalize_species(text: str) -> SpeciesSignature:
    """Parse one SMILES; never raises. A failure keeps the raw input and an error.

    Atom maps are cleared before anything is written, so a mapped FlowER
    species and its unmapped spelling give the same signature.
    """
    raw = (text or "").strip()
    info = SpeciesSignature(input=raw)
    if not raw:
        info.error = "Enter a structure (SMILES)."
        return info
    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromSmiles(raw)
        except Exception:
            mol = None
        if mol is None:
            info.error = "Could not parse SMILES"
            return info
        try:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            info.canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            info.formula = rdMolDescriptors.CalcMolFormula(mol)
            info.formal_charge = int(Chem.GetFormalCharge(mol))
            info.heavy_atoms = int(mol.GetNumHeavyAtoms())
            info.atom_counts = _atom_counts(mol)
            info.inchikey = _inchikey(mol)
            for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
                info.fragments.append(Chem.MolToSmiles(frag, isomericSmiles=True))
                info.fragment_heavy_atoms.append(int(frag.GetNumHeavyAtoms()))
                info.fragment_inchikeys.append(_inchikey(frag))
        except Exception:
            return SpeciesSignature(input=raw, error="Could not interpret this structure")
    return info


# ---------------------------------------------------------------------------
# reaction SMILES
# ---------------------------------------------------------------------------


def parse_reaction_smiles(text: str) -> Dict[str, List[str]]:
    """Split ``A.B>>C.D`` or ``A.B>R1.R2>C.D`` into reactants / reagents / products.

    Species are the ``.``-separated components; middle agents become reagents.
    A trailing CXSMILES block (`` |...|``) is ignored. Structures are not parsed
    here, so an invalid species surfaces later as a participant error. Raises
    ``ValueError`` with a user-facing message for a malformed reaction.
    """
    body = (text or "").strip().split()
    if not body:
        raise ValueError("Enter a reaction SMILES such as CCBr.[Cl-]>>CCCl.[Br-].")
    parts = body[0].split(">")
    if len(parts) != 3:
        raise ValueError(
            "A reaction SMILES needs exactly two '>' characters: reactants>>products "
            "or reactants>reagents>products."
        )

    def species(part: str) -> List[str]:
        return [piece for piece in part.split(".") if piece]

    reactants, reagents, products = (species(part) for part in parts)
    if not reactants and not products:
        raise ValueError("The reaction SMILES has no reactants and no products.")
    return {"reactants": reactants, "reagents": reagents, "products": products}


# ---------------------------------------------------------------------------
# submission hashes (mechanism_submission.v2)
# ---------------------------------------------------------------------------

RoleEntry = Union[str, Tuple[str, Optional[float]], Sequence[Any]]
Roles = Mapping[str, Sequence[RoleEntry]]


@dataclass
class ParticipantSignature:
    """One participant: its place in the submission and its species signature."""

    list_name: str
    role: str
    index: int
    coefficient: Optional[float]
    species: SpeciesSignature


@dataclass
class ReactionHashes:
    """The ``mechanism_submission.v2`` hashes for one submission."""

    exact_reaction_hash: str
    stoich_hash: str
    endpoint_hash: str
    core_hash: str
    major_key_hash: str
    inchikey_reaction_hash: Optional[str]
    conditions_hash: str
    similarity_cluster_id: str
    executable: bool
    participants: List[ParticipantSignature]
    recipe_version: str = RECIPE_VERSION
    rdkit_version: str = RDKIT_VERSION

    def as_dict(self) -> Dict[str, Optional[str]]:
        """The seven hashes keyed as ChemIllusion stores them."""
        return {
            "exact_reaction_hash": self.exact_reaction_hash,
            "stoich_hash": self.stoich_hash,
            "endpoint_hash": self.endpoint_hash,
            "core_hash": self.core_hash,
            "major_key_hash": self.major_key_hash,
            "inchikey_reaction_hash": self.inchikey_reaction_hash,
            "conditions_hash": self.conditions_hash,
        }


def _entry_text_and_coefficient(entry: RoleEntry) -> Tuple[str, Optional[float]]:
    if isinstance(entry, str):
        return entry, None
    text, coefficient = entry[0], (entry[1] if len(entry) > 1 else None)
    return str(text or ""), (float(coefficient) if coefficient is not None else None)


def _role_tokens(entries: Iterable[ParticipantSignature], *, inchikey: bool = False) -> Optional[List[str]]:
    """Fragment tokens for one role; an invalid participant contributes ``?<raw>``.

    With ``inchikey`` the tokens are InChIKeys, and ``None`` is returned when any
    fragment has none (the InChIKey key is then not computable).
    """
    tokens: List[str] = []
    for entry in entries:
        s = entry.species
        if not s.valid:
            if inchikey:
                return None
            tokens.append(f"?{s.input}")
            continue
        if inchikey:
            if any(key is None for key in s.fragment_inchikeys):
                return None
            tokens.extend(key for key in s.fragment_inchikeys if key)
        else:
            tokens.extend(s.fragments)
    return tokens


def _role_key(by_role: Dict[str, List[ParticipantSignature]], *, inchikey: bool = False) -> Optional[str]:
    parts = []
    for prefix, name in _ROLE_KEY_PARTS:
        tokens = _role_tokens(by_role[name], inchikey=inchikey)
        if tokens is None:
            return None
        parts.append(f"{prefix}:" + ".".join(sorted(set(tokens))))
    return ">".join(parts)


def _format_coefficient(value: float) -> str:
    return f"{value:g}"


def _stoich_key(by_role: Dict[str, List[ParticipantSignature]]) -> str:
    parts = []
    for prefix, name in _ROLE_KEY_PARTS:
        totals: Dict[str, float] = defaultdict(float)
        for entry in by_role[name]:
            coef = entry.coefficient if entry.coefficient is not None else 1.0
            s = entry.species
            for token in (s.fragments if s.valid else [f"?{s.input}"]):
                totals[token] += coef
        parts.append(
            f"{prefix}:" + ".".join(f"{_format_coefficient(totals[t])}*{t}" for t in sorted(totals))
        )
    return ">".join(parts)


def _heaviest(entries: Iterable[ParticipantSignature]) -> List[str]:
    frags: List[Tuple[int, str]] = []
    for entry in entries:
        if entry.species.valid:
            frags.extend(zip(entry.species.fragment_heavy_atoms, entry.species.fragments))
    if not frags:
        return []
    if any(heavy > 1 for heavy, _ in frags):
        frags = [(heavy, smi) for heavy, smi in frags if heavy > 1]
    top = max(heavy for heavy, _ in frags)
    return sorted({smi for heavy, smi in frags if heavy == top})


def _condition_value(value: Any) -> Optional[float]:
    # ChemIllusion's Conditions model declares ph / temperature_celsius as
    # Optional[float], so an int arrives as a float (7 -> 7.0) before hashing.
    return float(value) if value is not None else None


def _participants(roles: Roles) -> Dict[str, List[ParticipantSignature]]:
    by_role: Dict[str, List[ParticipantSignature]] = {name: [] for name, _ in ROLE_ORDER}
    for name, role in ROLE_ORDER:
        for index, raw_entry in enumerate(roles.get(name) or ()):
            text, coefficient = _entry_text_and_coefficient(raw_entry)
            by_role[name].append(ParticipantSignature(name, role, index, coefficient, normalize_species(text)))
    return by_role


def reaction_hashes(roles: Roles, conditions: Optional[Mapping[str, Any]] = None) -> ReactionHashes:
    """The ``mechanism_submission.v2`` hashes for a role-tagged submission.

    ``roles`` maps ``reactants``/``products``/``reagents``/``catalysts``/
    ``solvents`` to lists of ``(smiles, coefficient_or_None)`` (a bare SMILES
    string means coefficient ``None``); missing roles are empty.
    ``conditions`` supplies ``ph`` and ``temperature_celsius`` (either may be
    absent or ``None``). Callers that accept a raw reaction SMILES split it
    with :func:`parse_reaction_smiles` first when no participants were given,
    exactly as ChemIllusion's ``_participant_lists`` does.
    """
    conditions = conditions or {}
    by_role = _participants(roles)
    entries = [entry for name, _ in ROLE_ORDER for entry in by_role[name]]

    has_reactant = any(e.species.valid for e in by_role["reactants"])
    has_product = any(e.species.valid for e in by_role["products"])
    executable = has_reactant and has_product and all(e.species.valid for e in entries)

    role_text = _role_key(by_role)
    inchikey_text = _role_key(by_role, inchikey=True) if executable else None
    left = [e for name in ("reactants", "reagents", "catalysts", "solvents") for e in by_role[name]]
    left_tokens = _role_tokens(left) or []
    right_tokens = _role_tokens(by_role["products"]) or []
    endpoint_text = ".".join(sorted(set(left_tokens))) + ">>" + ".".join(sorted(set(right_tokens)))
    core_text = (
        "R:" + ".".join(sorted(set(_role_tokens(by_role["reactants"]) or [])))
        + ">P:" + ".".join(sorted(set(right_tokens)))
    )
    major_text = ".".join(_heaviest(by_role["reactants"])) + ">>" + ".".join(_heaviest(by_role["products"]))

    ph = _condition_value(conditions.get("ph"))
    temperature = _condition_value(conditions.get("temperature_celsius"))
    major_key_hash = _h("major", major_text)
    return ReactionHashes(
        exact_reaction_hash=_h("role", role_text or ""),
        stoich_hash=_h("stoich", _stoich_key(by_role)),
        endpoint_hash=_h("union", endpoint_text),
        core_hash=_h("core", core_text),
        major_key_hash=major_key_hash,
        inchikey_reaction_hash=_h("role_inchikey", inchikey_text) if inchikey_text else None,
        conditions_hash=_h("cond", f"{ph!r}|{temperature!r}"),
        similarity_cluster_id="mpc_" + major_key_hash[:20],
        executable=executable,
        participants=entries,
    )


# ---------------------------------------------------------------------------
# corpus keys (reaction_corpus.v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusKeys:
    """Four unsigned 64-bit ``reaction_corpus.v1`` keys for one reaction."""

    exact: int
    core: int
    endpoint: int
    family: int

    def as_dict(self) -> Dict[str, int]:
        return {kind: getattr(self, kind) for kind in _CORPUS_KINDS}


@dataclass(frozen=True)
class _Frag:
    inchikey: str
    heavy_atoms: int
    smiles: str


def _side_fragments(species: Iterable[str], *, require_inchikey: bool = True) -> List[_Frag]:
    frags: List[_Frag] = []
    for text in species:
        info = normalize_species(text)
        if not info.valid:
            raise CorpusKeyError(f"could not parse species {info.input!r}")
        for key, heavy, smi in zip(info.fragment_inchikeys, info.fragment_heavy_atoms, info.fragments):
            if not key and require_inchikey:
                raise CorpusKeyError(f"no standard InChIKey for fragment {smi!r}")
            frags.append(_Frag(key or "", int(heavy), smi))
    return frags


def _key_set(frags: Iterable[_Frag]) -> str:
    return ".".join(sorted({frag.inchikey for frag in frags}))


def _core_fragments(frags: List[_Frag], other: List[_Frag]) -> List[_Frag]:
    """One side's reacting fragments (see the ``core`` kind in the module docstring)."""
    shared = {frag.inchikey for frag in other}
    own = [frag for frag in frags if frag.inchikey not in shared] or list(frags)
    if any(frag.heavy_atoms > 1 for frag in own):
        own = [frag for frag in own if frag.heavy_atoms > 1]
    return own


def _heaviest_keys(core: List[_Frag]) -> List[str]:
    top = max(frag.heavy_atoms for frag in core)
    return sorted({frag.inchikey for frag in core if frag.heavy_atoms == top})


def _keys_from_fragments(left: List[_Frag], endpoint_left: List[_Frag], right: List[_Frag]) -> CorpusKeys:
    if not left or not right:
        raise CorpusKeyError("a reaction needs at least one species on each side")
    core_left = _core_fragments(left, right)
    core_right = _core_fragments(right, left)
    exact_text = _key_set(left) + ">>" + _key_set(right)
    core_text = _key_set(core_left) + ">>" + _key_set(core_right)
    endpoint_text = _key_set(endpoint_left) + ">>" + _key_set(right)
    family_text = ".".join(_heaviest_keys(core_left)) + ">>" + ".".join(_heaviest_keys(core_right))
    return CorpusKeys(
        exact=_corpus_hash("exact", exact_text),
        core=_corpus_hash("core", core_text),
        endpoint=_corpus_hash("endpoint", endpoint_text),
        family=_corpus_hash("family", family_text),
    )


def corpus_keys(reactants: Sequence[str], products: Sequence[str]) -> CorpusKeys:
    """Corpus keys for a FlowER-shaped reaction (left side ``>>`` right side).

    Atom-mapped SMILES are accepted; maps are cleared. Every left-side species
    counts as a reactant, so ``endpoint`` equals ``exact`` in text. Raises
    :class:`CorpusKeyError` when a side is empty, a species does not parse, or
    a fragment has no standard InChIKey.
    """
    left = _side_fragments(reactants)
    return _keys_from_fragments(left, left, _side_fragments(products))


def _role_texts(roles: Roles, name: str) -> List[str]:
    return [_entry_text_and_coefficient(entry)[0] for entry in (roles.get(name) or ())]


def corpus_keys_from_submission(roles: Roles) -> CorpusKeys:
    """Corpus keys for a role-tagged submission (same ``roles`` shape as
    :func:`reaction_hashes`; coefficients are ignored).

    ``exact``, ``core`` and ``family`` use reactants against products only
    (core drops shared and one-heavy-atom fragments exactly as for FlowER);
    ``endpoint`` adds reagents, catalysts and solvents to the left side.
    Raises :class:`CorpusKeyError` like :func:`corpus_keys`.
    """
    left = _side_fragments(_role_texts(roles, "reactants"))
    auxiliaries = [text for name in ("reagents", "catalysts", "solvents") for text in _role_texts(roles, name)]
    right = _side_fragments(_role_texts(roles, "products"))
    return _keys_from_fragments(left, left + _side_fragments(auxiliaries), right)


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def _pack(bits: Iterable[int]) -> bytes:
    buf = bytearray(FINGERPRINT_BYTES)
    for bit in bits:
        bit = int(bit) % FINGERPRINT_BITS
        buf[bit // 8] |= 1 << (bit % 8)
    return bytes(buf)


def _difference_bits(left: Sequence[str], right: Sequence[str]) -> List[int]:
    params = rdChemReactions.ReactionFingerprintParams()
    params.fpSize = FINGERPRINT_BITS
    params.fpType = rdChemReactions.FingerprintType.MorganFP
    params.includeAgents = False
    rxn = rdChemReactions.ChemicalReaction()
    for smi in left:
        rxn.AddReactantTemplate(Chem.MolFromSmiles(smi))
    for smi in right:
        rxn.AddProductTemplate(Chem.MolFromSmiles(smi))
    fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn, params)
    return sorted(idx for idx, count in fp.GetNonzeroElements().items() if count != 0)


def _morgan_bits(smiles: Sequence[str]) -> List[int]:
    bits: set = set()
    if _fpgen is not None:
        generator = _fpgen.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)
        for smi in smiles:
            bits.update(generator.GetFingerprint(Chem.MolFromSmiles(smi)).GetOnBits())
    else:  # pragma: no cover - RDKit < 2022.03
        from rdkit.Chem import AllChem

        for smi in smiles:
            bits.update(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=FINGERPRINT_BITS).GetOnBits())
    return sorted(bits)


def _fingerprints_from_fragments(left_frags: List[_Frag], right_frags: List[_Frag]) -> Tuple[bytes, bytes]:
    left = sorted({frag.smiles for frag in left_frags})
    right = sorted({frag.smiles for frag in right_frags})
    if not left or not right:
        raise CorpusKeyError("a reaction needs at least one species on each side")
    left_set = set(left)
    product_frags = [smi for smi in right if smi not in left_set] or right
    with rdBase.BlockLogs():
        return _pack(_difference_bits(left, right)), _pack(_morgan_bits(product_frags))


def reaction_fingerprints(reactants: Sequence[str], products: Sequence[str]) -> Tuple[bytes, bytes]:
    """``(difference_fp, product_fp)``, each 32 bytes (256 bits), see ``FINGERPRINT_SPEC``.

    Atom-mapped SMILES are accepted. For a submission pass reactants and
    products only (auxiliaries are context, not structural change). Raises
    :class:`CorpusKeyError` when a side is empty or a species does not parse.
    """
    return _fingerprints_from_fragments(
        _side_fragments(reactants, require_inchikey=False),
        _side_fragments(products, require_inchikey=False),
    )


def corpus_keys_and_fingerprints(
    reactants: Sequence[str], products: Sequence[str], *, fingerprints: bool = True
) -> Tuple[CorpusKeys, Optional[Tuple[bytes, bytes]]]:
    """:func:`corpus_keys` and (optionally) :func:`reaction_fingerprints` for
    one FlowER-shaped reaction, parsing each species once. Used by the index
    builder; the results equal the two separate calls."""
    left = _side_fragments(reactants)
    right = _side_fragments(products)
    keys = _keys_from_fragments(left, left, right)
    return keys, (_fingerprints_from_fragments(left, right) if fingerprints else None)


def tanimoto(a: bytes, b: bytes) -> float:
    """Tanimoto similarity of two packed fingerprints of equal length.

    Two all-zero fingerprints score ``0.0`` (no shared evidence).
    """
    if len(a) != len(b):
        raise ValueError("fingerprints differ in length")
    ia = int.from_bytes(a, "little")
    ib = int.from_bytes(b, "little")
    union = bin(ia | ib).count("1")
    if union == 0:
        return 0.0
    return bin(ia & ib).count("1") / union
