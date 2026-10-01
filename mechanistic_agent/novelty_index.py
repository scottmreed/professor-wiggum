"""FlowER novelty index and train-only reference mechanism library (loaders).

Built by ``scripts/build_reaction_novelty_index.py``; see
``training_data/REGENERATE.md`` ("Reaction novelty index").

Novelty index artifact (``reaction_novelty_index.npz``)
    A zip of ``.npy`` arrays, readable with ``numpy.load`` (written by
    :func:`write_npz_deterministic`: fixed timestamps, sorted member order,
    so the same input and the same RDKit/numpy give the same SHA-256). It holds
    **only** split labels, 64-bit ``reaction_corpus.v1`` keys and 256-bit
    fingerprints — no structures, no mechanisms, no FlowER ids:

    * ``corpus_recipe_version`` — 0-d unicode array, must equal
      :data:`~mechanistic_agent.reaction_signatures.CORPUS_RECIPE_VERSION`;
    * ``<kind>_keys`` (uint64, sorted ascending) and ``<kind>_splits`` (uint8,
      ``0`` = train, ``1`` = test) for each kind in ``exact``, ``core``,
      ``endpoint``, ``family`` — the unique ``(key, split)`` pairs, sorted by
      key then split, so :meth:`NoveltyIndex.match` is a binary search;
    * ``fingerprints`` (uint8 ``[rows, 64]``: bytes 0–31 the reaction
      difference fingerprint, bytes 32–63 the product fingerprint) and
      ``fingerprint_splits`` (uint8 ``[rows]``) — one row per unique
      ``(split, exact key)`` reaction, absent when built with
      ``--no-fingerprints``.

    The sidecar ``reaction_novelty_index.manifest.json`` records the recipe
    versions, RDKit/numpy versions, source file SHA-256s, row counts, and the
    artifact's ``bytes`` and ``sha256``. :meth:`NoveltyIndex.load` refuses an
    artifact whose SHA-256 or corpus recipe differs (:class:`NoveltyIndexMismatch`).

Reference mechanism library (``flower_train_mechanisms.sqlite[.gz]``)
    FlowER **train** mechanisms only, keyed by the 64-bit ``core`` key. Table
    ``mechanisms(core_key INTEGER, mechanism_id INTEGER, step_count INTEGER,
    steps_json TEXT)`` indexed on ``core_key``; ``steps_json`` is the compact
    JSON of ``verified_mechanism.steps`` exactly as
    ``flower_curriculum._convert_group`` produces it. SQLite integers are
    signed, so a key ``>= 2**63`` is stored as ``key - 2**64``
    (:func:`to_sqlite_int`); :meth:`ReferenceMechanismLibrary.lookup` takes the
    unsigned key. A ``meta`` table records ``split = train`` and the corpus
    recipe; the loader refuses anything else.

Committed manifest (``novelty_index/manifest.json``)
    ``{"schema": "novelty_index_manifest.v1", "assets": null, ...}`` until the
    first build is uploaded. Populated shape::

        {
          "schema": "novelty_index_manifest.v1",
          "release_tag": "novelty-index-reaction_corpus.v1",
          "corpus_recipe_version": "reaction_corpus.v1",
          "recipe_version": "mechanism_submission.v2",
          "rdkit_version": "2023.09.1",
          "assets": [
            {"name": "reaction_novelty_index.npz", "kind": "novelty_index",
             "url": "https://github.com/.../releases/download/<tag>/reaction_novelty_index.npz",
             "sha256": "...", "bytes": 0},
            {"name": "reaction_novelty_index.manifest.json", "kind": "novelty_index_manifest", ...},
            {"name": "flower_train_mechanisms.sqlite.gz", "kind": "train_mechanisms", ...}
          ]
        }

    :func:`load_from_manifest` returns ``None`` when ``assets`` is null or a
    named file is missing (reference screening unavailable), and raises
    :class:`NoveltyIndexMismatch` when a file is present but its SHA-256
    differs from the committed manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from mechanistic_agent.reaction_signatures import (
    CORPUS_RECIPE_VERSION,
    FINGERPRINT_BYTES,
    CorpusKeys,
)

__all__ = [
    "NOVELTY_INDEX_ARTIFACT",
    "NOVELTY_INDEX_MANIFEST",
    "MECHANISM_LIBRARY_ARTIFACT",
    "COMMITTED_MANIFEST_SCHEMA",
    "KINDS",
    "SPLITS",
    "NoveltyIndexMismatch",
    "NoveltyIndex",
    "ReferenceMechanismLibrary",
    "file_sha256",
    "write_npz_deterministic",
    "to_sqlite_int",
    "load_from_manifest",
    "load_reference_library_from_manifest",
]

NOVELTY_INDEX_ARTIFACT = "reaction_novelty_index.npz"
NOVELTY_INDEX_MANIFEST = "reaction_novelty_index.manifest.json"
MECHANISM_LIBRARY_ARTIFACT = "flower_train_mechanisms.sqlite"
COMMITTED_MANIFEST_SCHEMA = "novelty_index_manifest.v1"
COMMITTED_MANIFEST_RELATIVE = Path("novelty_index") / "manifest.json"
KINDS: Tuple[str, ...] = ("exact", "core", "endpoint", "family")
SPLITS: Tuple[str, ...] = ("train", "test")  # index = stored uint8 label
_FP_ROW_BYTES = 2 * FINGERPRINT_BYTES
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

PathLike = Union[str, Path]


class NoveltyIndexMismatch(RuntimeError):
    """An artifact does not match its manifest or this code's recipe version."""


def file_sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz_deterministic(path: PathLike, arrays: Mapping[str, np.ndarray]) -> None:
    """Write ``arrays`` as a compressed ``.npz`` with no timestamps.

    ``numpy.savez_compressed`` stamps each zip member with the current time,
    which would change the artifact SHA-256 on every build.
    """
    with zipfile.ZipFile(Path(path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            array = np.asarray(arrays[name])
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def to_sqlite_int(key: int) -> int:
    """Map an unsigned 64-bit key onto SQLite's signed INTEGER range."""
    key = int(key)
    return key - (1 << 64) if key >= (1 << 63) else key


# ---------------------------------------------------------------------------
# popcount
# ---------------------------------------------------------------------------

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


def _popcount_rows(words: np.ndarray) -> np.ndarray:
    """Per-row popcount of a ``[n, w]`` uint64 matrix, as int32."""
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(words).sum(axis=1, dtype=np.int32)
    x = words - ((words >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    x = (x * _H01) >> np.uint64(56)
    return x.sum(axis=1, dtype=np.int32)


def _as_words(fp: bytes) -> np.ndarray:
    if len(fp) != FINGERPRINT_BYTES:
        raise ValueError(f"fingerprint must be {FINGERPRINT_BYTES} bytes")
    return np.frombuffer(fp, dtype="<u8").astype(np.uint64)


def _tanimoto_column(words: np.ndarray, pops: np.ndarray, query: np.ndarray) -> np.ndarray:
    inter = _popcount_rows(words & query[None, :])
    union = pops + int(_popcount_rows(query[None, :])[0]) - inter
    out = np.zeros(len(words), dtype=np.float64)
    nonzero = union > 0
    out[nonzero] = inter[nonzero] / union[nonzero]
    return out


# ---------------------------------------------------------------------------
# novelty index
# ---------------------------------------------------------------------------


@dataclass
class NoveltyIndex:
    """Loaded novelty index. Returns only split labels and similarities."""

    manifest: Dict[str, Any]
    keys: Dict[str, np.ndarray]
    key_splits: Dict[str, np.ndarray]
    fingerprints: Optional[np.ndarray] = None
    fingerprint_splits: Optional[np.ndarray] = None
    _diff_words: Optional[np.ndarray] = field(default=None, repr=False)
    _prod_words: Optional[np.ndarray] = field(default=None, repr=False)
    _diff_pops: Optional[np.ndarray] = field(default=None, repr=False)
    _prod_pops: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def load(cls, npz_path: PathLike, manifest_path: PathLike) -> "NoveltyIndex":
        npz_path = Path(npz_path)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        recipe = manifest.get("corpus_recipe_version")
        if recipe != CORPUS_RECIPE_VERSION:
            raise NoveltyIndexMismatch(
                f"novelty index recipe {recipe!r} does not match this code's {CORPUS_RECIPE_VERSION!r}"
            )
        actual = file_sha256(npz_path)
        if actual != manifest.get("sha256"):
            raise NoveltyIndexMismatch(
                f"novelty index sha256 {actual} does not match manifest {manifest.get('sha256')}"
            )
        with np.load(npz_path, allow_pickle=False) as data:
            embedded = None
            if "corpus_recipe_version" in data.files and data["corpus_recipe_version"].size == 1:
                embedded = str(data["corpus_recipe_version"].reshape(()).item())
            if embedded != CORPUS_RECIPE_VERSION:
                raise NoveltyIndexMismatch(f"novelty index embeds recipe {embedded!r}")
            keys = {kind: np.asarray(data[f"{kind}_keys"], dtype=np.uint64) for kind in KINDS}
            key_splits = {kind: np.asarray(data[f"{kind}_splits"], dtype=np.uint8) for kind in KINDS}
            fingerprints = None
            fingerprint_splits = None
            if "fingerprints" in data.files:
                fingerprints = np.ascontiguousarray(data["fingerprints"], dtype=np.uint8)
                fingerprint_splits = np.asarray(data["fingerprint_splits"], dtype=np.uint8)
        index = cls(manifest, keys, key_splits, fingerprints, fingerprint_splits)
        if fingerprints is not None:
            if fingerprints.ndim != 2 or fingerprints.shape[1] != _FP_ROW_BYTES:
                raise NoveltyIndexMismatch(f"fingerprint matrix has shape {fingerprints.shape}")
            words = fingerprints.view("<u8").astype(np.uint64)
            half = FINGERPRINT_BYTES // 8
            index._diff_words = np.ascontiguousarray(words[:, :half])
            index._prod_words = np.ascontiguousarray(words[:, half:])
            index._diff_pops = _popcount_rows(index._diff_words)
            index._prod_pops = _popcount_rows(index._prod_words)
        return index

    @property
    def fingerprints_available(self) -> bool:
        return self.fingerprints is not None

    @property
    def rdkit_version(self) -> Optional[str]:
        """RDKit that built the fingerprints; compare before trusting ``nearest``."""
        return self.manifest.get("rdkit_version")

    @property
    def row_count(self) -> int:
        return int(len(self.fingerprint_splits)) if self.fingerprint_splits is not None else int(
            sum((self.manifest.get("rows") or {}).values())
        )

    def match(self, keys: CorpusKeys) -> List[Dict[str, str]]:
        """Every ``{"split", "kind"}`` whose key equals the query's, in kind
        order (exact, core, endpoint, family) then split order (train, test)."""
        out: List[Dict[str, str]] = []
        for kind in KINDS:
            array = self.keys[kind]
            needle = np.uint64(int(getattr(keys, kind)))
            lo = int(np.searchsorted(array, needle, side="left"))
            hi = int(np.searchsorted(array, needle, side="right"))
            if hi <= lo:
                continue
            for code in sorted({int(code) for code in self.key_splits[kind][lo:hi]}):
                out.append({"split": SPLITS[code], "kind": kind})
        return out

    def nearest(
        self, fps: Tuple[bytes, bytes], k: int = 8, min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Top-``k`` rows by ``similarity`` = mean of the difference and product
        Tanimoto similarities (ties broken by row order). Returns ``[]`` when
        the index has no fingerprints."""
        if self._diff_words is None or k <= 0:
            return []
        diff_fp, prod_fp = fps
        diff = _tanimoto_column(self._diff_words, self._diff_pops, _as_words(diff_fp))
        prod = _tanimoto_column(self._prod_words, self._prod_pops, _as_words(prod_fp))
        combined = (diff + prod) / 2.0
        candidates = np.flatnonzero(combined >= min_similarity)
        if candidates.size == 0:
            return []
        if candidates.size > k:
            part = np.argpartition(-combined[candidates], k - 1)[:k]
            # Include every row tied with the k-th score so tie-breaking is by row.
            threshold = combined[candidates[part]].min()
            candidates = candidates[combined[candidates] >= threshold]
        order = candidates[np.lexsort((candidates, -combined[candidates]))][:k]
        return [
            {
                "split": SPLITS[int(self.fingerprint_splits[row])],
                "similarity": float(combined[row]),
                "diff_similarity": float(diff[row]),
                "product_similarity": float(prod[row]),
            }
            for row in order
        ]


# ---------------------------------------------------------------------------
# reference mechanism library (train only)
# ---------------------------------------------------------------------------


class ReferenceMechanismLibrary:
    """Read-only access to ``flower_train_mechanisms.sqlite`` (or ``.sqlite.gz``)."""

    def __init__(self, conn: sqlite3.Connection, meta: Dict[str, str], tempdir: Optional[str] = None):
        self._conn = conn
        self.meta = meta
        self._tempdir = tempdir

    @classmethod
    def load(cls, path: PathLike) -> "ReferenceMechanismLibrary":
        path = Path(path)
        tempdir: Optional[str] = None
        if path.suffix == ".gz":
            tempdir = tempfile.mkdtemp(prefix="wiggum_mechanisms_")
            target = Path(tempdir) / path.with_suffix("").name
            with gzip.open(path, "rb") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            path = target
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise NoveltyIndexMismatch(f"not a reference mechanism library: {exc}") from exc
        if meta.get("split") != "train":
            conn.close()
            raise NoveltyIndexMismatch(f"reference mechanism library split is {meta.get('split')!r}, not 'train'")
        if meta.get("corpus_recipe_version") != CORPUS_RECIPE_VERSION:
            conn.close()
            raise NoveltyIndexMismatch(
                f"reference mechanism library recipe {meta.get('corpus_recipe_version')!r} "
                f"does not match {CORPUS_RECIPE_VERSION!r}"
            )
        return cls(conn, meta, tempdir)

    def lookup(self, core_key: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT mechanism_id, step_count, steps_json FROM mechanisms WHERE core_key = ? ORDER BY mechanism_id",
            (to_sqlite_int(core_key),),
        ).fetchall()
        return [
            {"mechanism_id": int(mid), "step_count": int(count), "steps": json.loads(steps)}
            for mid, count, steps in rows
        ]

    reference_mechanisms = lookup

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
        if self._tempdir:
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None


# ---------------------------------------------------------------------------
# committed manifest
# ---------------------------------------------------------------------------


def _committed_assets(base_dir: PathLike) -> Optional[List[Dict[str, Any]]]:
    path = Path(base_dir) / COMMITTED_MANIFEST_RELATIVE
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != COMMITTED_MANIFEST_SCHEMA:
        raise NoveltyIndexMismatch(f"unknown committed manifest schema {manifest.get('schema')!r}")
    assets = manifest.get("assets")
    return list(assets) if assets else None


def _asset_file(assets: Sequence[Mapping[str, Any]], kind: str, assets_dir: PathLike) -> Optional[Path]:
    for asset in assets:
        if asset.get("kind") != kind:
            continue
        path = Path(assets_dir) / str(asset["name"])
        if not path.is_file():
            return None
        expected = asset.get("sha256")
        if expected and file_sha256(path) != expected:
            raise NoveltyIndexMismatch(f"{path.name} sha256 does not match novelty_index/manifest.json")
        return path
    return None


def load_from_manifest(base_dir: PathLike, assets_dir: PathLike) -> Optional[NoveltyIndex]:
    """Load the index named by ``<base_dir>/novelty_index/manifest.json`` from
    ``assets_dir``. ``None`` means unavailable (null assets or missing files)."""
    assets = _committed_assets(base_dir)
    if not assets:
        return None
    npz = _asset_file(assets, "novelty_index", assets_dir)
    manifest = _asset_file(assets, "novelty_index_manifest", assets_dir)
    if npz is None or manifest is None:
        return None
    return NoveltyIndex.load(npz, manifest)


def load_reference_library_from_manifest(
    base_dir: PathLike, assets_dir: PathLike
) -> Optional[ReferenceMechanismLibrary]:
    """The ``train_mechanisms`` asset, or ``None`` when it is not published."""
    assets = _committed_assets(base_dir)
    if not assets:
        return None
    path = _asset_file(assets, "train_mechanisms", assets_dir)
    return ReferenceMechanismLibrary.load(path) if path is not None else None
