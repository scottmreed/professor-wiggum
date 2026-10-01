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

Reference mechanism library, ``train_mechanisms.v2`` (``flower_train_mechanisms.v2.sqlite``)
    FlowER **train** mechanisms only, keyed by the 64-bit ``core`` key. Tables
    ``meta(key, value)`` (``format = train_mechanisms.v2``, ``split = train``,
    ``corpus_recipe_version``, ``source_sha256``, ``rows``, ``built_at``) and
    ``mechanisms(core_key INTEGER, mechanism_id INTEGER, step_count INTEGER,
    payload BLOB)`` indexed on ``core_key``, rows in ``(core_key,
    mechanism_id)`` order. ``payload`` is ``zlib.compress(level 9)`` of the
    compact, key-sorted JSON ``{"initial_state": [...], "steps":
    [{"resulting_state": [...], "electron_pushes": ["<notation>", ...]}]}``
    (:func:`compact_mechanism`): ``initial_state`` is the first step's
    ``current_state``, and the derivable or constant v1 fields
    (``current_state``, ``target_products``, ``predicted_intermediate``,
    ``reaction_smirks``, ``note``, ``confidence``, structured pushes) are
    dropped. The file is not gzipped (each row already is). ``built_at`` is
    ``SOURCE_DATE_EPOCH`` or the source file's mtime, so the same input gives
    the same bytes.

    The original **v1** (``flower_train_mechanisms.sqlite[.gz]``, no ``format``
    meta) has ``steps_json TEXT`` instead of ``payload``: the full
    ``verified_mechanism.steps`` of ``flower_curriculum._convert_group``,
    about 11 KB per mechanism. ``scripts/compact_reference_mechanisms.py``
    converts v1 to v2.

    SQLite integers are signed, so a key ``>= 2**63`` is stored as
    ``key - 2**64`` (:func:`to_sqlite_int`); :meth:`ReferenceMechanismLibrary.lookup`
    takes the unsigned key and returns the same normalized shape for both
    formats. The loader refuses a library whose ``split`` is not ``train`` or
    whose corpus recipe differs.

Committed manifest (``novelty_index/manifest.json``)
    ``"assets": null`` means nothing is published. Populated shape::

        {
          "schema": "novelty_index_manifest.v1",
          "release_tag": "novelty-index-reaction_corpus.v1",
          "assets": [
            {"name": "reaction_novelty_index.npz", "kind": "novelty_index",
             "url": "https://github.com/.../releases/download/<tag>/reaction_novelty_index.npz",
             "sha256": "...", "bytes": 0},
            {"name": "reaction_novelty_index.manifest.json", "kind": "novelty_index_manifest", ...},
            {"name": "flower_train_mechanisms.v2.sqlite", "kind": "train_mechanisms", ...}
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
import os
import shutil
import sqlite3
import tempfile
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

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
    "MECHANISM_LIBRARY_ARTIFACT_V1",
    "MECHANISM_LIBRARY_FORMAT_V1",
    "MECHANISM_LIBRARY_FORMAT_V2",
    "COMMITTED_MANIFEST_SCHEMA",
    "KINDS",
    "SPLITS",
    "NoveltyIndexMismatch",
    "NoveltyIndex",
    "ReferenceMechanismLibrary",
    "file_sha256",
    "write_npz_deterministic",
    "to_sqlite_int",
    "compact_mechanism",
    "encode_mechanism_payload",
    "decode_mechanism_payload",
    "normalize_mechanism",
    "default_built_at",
    "write_mechanism_library_v2",
    "load_from_manifest",
    "load_reference_library_from_manifest",
]

NOVELTY_INDEX_ARTIFACT = "reaction_novelty_index.npz"
NOVELTY_INDEX_MANIFEST = "reaction_novelty_index.manifest.json"
MECHANISM_LIBRARY_ARTIFACT = "flower_train_mechanisms.v2.sqlite"
MECHANISM_LIBRARY_ARTIFACT_V1 = "flower_train_mechanisms.sqlite"
MECHANISM_LIBRARY_FORMAT_V1 = "train_mechanisms.v1"
MECHANISM_LIBRARY_FORMAT_V2 = "train_mechanisms.v2"
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


def _push_notation(push: Any) -> str:
    """The ``notation`` string of one ``electron_pushes`` entry."""
    if isinstance(push, str):
        return push
    notation = push.get("notation") if isinstance(push, Mapping) else None
    if not isinstance(notation, str) or not notation:
        raise ValueError(f"electron push without a notation string: {push!r}")
    return notation


def compact_mechanism(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """v1 ``verified_mechanism.steps`` -> the v2 payload object.

    Keeps the first step's ``current_state`` once, then each step's
    ``resulting_state`` and push notations. Raises :class:`ValueError` when a
    step's ``current_state`` is not the previous ``resulting_state`` (the
    dropped field would not be derivable, so the conversion would lose data).
    """
    if not steps:
        raise ValueError("mechanism has no steps")
    out_steps: List[Dict[str, Any]] = []
    previous: Optional[List[str]] = None
    for position, step in enumerate(steps, start=1):
        current = list(step["current_state"])
        if previous is not None and current != previous:
            raise ValueError(f"step {position} current_state is not step {position - 1} resulting_state")
        resulting = list(step["resulting_state"])
        out_steps.append(
            {
                "resulting_state": resulting,
                "electron_pushes": [_push_notation(push) for push in step.get("electron_pushes") or []],
            }
        )
        previous = resulting
    return {"initial_state": list(steps[0]["current_state"]), "steps": out_steps}


def encode_mechanism_payload(compact: Mapping[str, Any]) -> bytes:
    """zlib (level 9) of the compact, key-sorted JSON of a v2 payload object."""
    text = json.dumps(compact, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return zlib.compress(text.encode("utf-8"), 9)


def decode_mechanism_payload(payload: bytes) -> Dict[str, Any]:
    return json.loads(zlib.decompress(payload).decode("utf-8"))


def normalize_mechanism(mechanism_id: int, step_count: int, compact: Mapping[str, Any]) -> Dict[str, Any]:
    """The lookup shape shared by v1 and v2 (see the Plan B API contract)."""
    return {
        "mechanism_id": int(mechanism_id),
        "step_count": int(step_count),
        "initial_state": list(compact["initial_state"]),
        "steps": [
            {
                "index": position,
                "resulting_state": list(step["resulting_state"]),
                "electron_pushes": [str(push) for push in step["electron_pushes"]],
            }
            for position, step in enumerate(compact["steps"], start=1)
        ],
    }


def default_built_at(source: PathLike) -> str:
    """``built_at`` for a library: ``SOURCE_DATE_EPOCH`` when set, otherwise the
    source file's modification time. Never "now", so the same input gives the
    same library bytes (and SHA-256)."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    seconds = int(epoch) if epoch and epoch.strip().isdigit() else int(Path(source).stat().st_mtime)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="seconds")


def write_mechanism_library_v2(
    path: PathLike,
    rows: Iterable[Tuple[int, int, int, bytes]],
    *,
    source_sha256: str,
    built_at: str,
    extra_meta: Optional[Mapping[str, str]] = None,
) -> int:
    """Write a ``train_mechanisms.v2`` library from ``rows`` and return the row count.

    ``rows`` are ``(core_key, mechanism_id, step_count, payload)`` with the
    core key already mapped by :func:`to_sqlite_int`, in ascending
    ``(core_key, mechanism_id)`` order (enforced, so output order is
    deterministic); it may be a generator. ``path`` is replaced. The caller is
    responsible for passing train-split rows only.
    """
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    count = 0
    try:
        conn.executescript(
            """
            PRAGMA journal_mode = DELETE;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE mechanisms (
                core_key INTEGER NOT NULL,
                mechanism_id INTEGER NOT NULL,
                step_count INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            """
        )
        previous: Optional[Tuple[int, int]] = None

        def checked() -> Iterator[Tuple[int, int, int, bytes]]:
            nonlocal previous, count
            for core, mid, steps, payload in rows:
                ident = (int(core), int(mid))
                if previous is not None and ident <= previous:
                    raise ValueError(f"mechanism rows out of (core_key, mechanism_id) order at {ident}")
                previous = ident
                count += 1
                yield ident[0], ident[1], int(steps), sqlite3.Binary(payload)

        conn.executemany(
            "INSERT INTO mechanisms(core_key, mechanism_id, step_count, payload) VALUES (?, ?, ?, ?)", checked()
        )
        conn.execute("CREATE INDEX idx_mechanisms_core_key ON mechanisms(core_key)")
        meta = {
            "format": MECHANISM_LIBRARY_FORMAT_V2,
            "split": "train",
            "corpus_recipe_version": CORPUS_RECIPE_VERSION,
            "source_sha256": source_sha256,
            "rows": str(count),
            "built_at": built_at,
            "key": "core (reaction_corpus.v1), unsigned 64-bit stored as signed",
            "payload": "zlib(level 9) of compact JSON {initial_state, steps: [{resulting_state, electron_pushes}]}",
            **dict(extra_meta or {}),
        }
        conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", sorted(meta.items()))
        conn.commit()
        conn.execute("VACUUM")
    except BaseException:
        conn.close()
        path.unlink(missing_ok=True)
        raise
    conn.close()
    return count


def _open_library(path: Path) -> Tuple[sqlite3.Connection, Optional[str]]:
    """Read-only connection; a ``.gz`` is decompressed into a temp directory."""
    tempdir: Optional[str] = None
    if path.suffix == ".gz":
        tempdir = tempfile.mkdtemp(prefix="wiggum_mechanisms_")
        target = Path(tempdir) / path.with_suffix("").name
        with gzip.open(path, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        path = target
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False), tempdir


def _library_format(conn: sqlite3.Connection, meta: Mapping[str, str]) -> str:
    declared = meta.get("format")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(mechanisms)")}
    if declared == MECHANISM_LIBRARY_FORMAT_V2 and "payload" in columns:
        return MECHANISM_LIBRARY_FORMAT_V2
    if declared in (None, MECHANISM_LIBRARY_FORMAT_V1) and "steps_json" in columns:
        return MECHANISM_LIBRARY_FORMAT_V1
    raise NoveltyIndexMismatch(f"unknown reference mechanism library format {declared!r} (columns {sorted(columns)})")


class ReferenceMechanismLibrary:
    """Read-only access to the train-only reference mechanism library.

    Reads ``train_mechanisms.v2`` (``flower_train_mechanisms.v2.sqlite``) and
    the original v1 (``flower_train_mechanisms.sqlite[.gz]``, no ``format``
    meta, ``steps_json`` column); :meth:`lookup` returns the same normalized
    shape for both.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        meta: Dict[str, str],
        tempdir: Optional[str] = None,
        library_format: str = MECHANISM_LIBRARY_FORMAT_V1,
    ):
        self._conn = conn
        self.meta = meta
        self._tempdir = tempdir
        self.format = library_format

    @classmethod
    def load(cls, path: PathLike) -> "ReferenceMechanismLibrary":
        conn, tempdir = _open_library(Path(path))

        def refuse(message: str) -> NoveltyIndexMismatch:
            conn.close()
            if tempdir:
                shutil.rmtree(tempdir, ignore_errors=True)
            return NoveltyIndexMismatch(message)

        try:
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            library_format = _library_format(conn, meta)
        except sqlite3.DatabaseError as exc:
            raise refuse(f"not a reference mechanism library: {exc}") from exc
        except NoveltyIndexMismatch as exc:
            raise refuse(str(exc)) from exc
        if meta.get("split") != "train":
            raise refuse(f"reference mechanism library split is {meta.get('split')!r}, not 'train'")
        if meta.get("corpus_recipe_version") != CORPUS_RECIPE_VERSION:
            raise refuse(
                f"reference mechanism library recipe {meta.get('corpus_recipe_version')!r} "
                f"does not match {CORPUS_RECIPE_VERSION!r}"
            )
        return cls(conn, meta, tempdir, library_format)

    def lookup(self, core_key: int) -> List[Dict[str, Any]]:
        """Train mechanisms with this unsigned ``core`` key, by mechanism id:
        ``[{"mechanism_id", "step_count", "initial_state", "steps": [{"index"
        (1-based), "resulting_state", "electron_pushes": [notation, ...]}]}]``."""
        column = "payload" if self.format == MECHANISM_LIBRARY_FORMAT_V2 else "steps_json"
        rows = self._conn.execute(
            f"SELECT mechanism_id, step_count, {column} FROM mechanisms WHERE core_key = ? ORDER BY mechanism_id",
            (to_sqlite_int(core_key),),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for mid, count, body in rows:
            if self.format == MECHANISM_LIBRARY_FORMAT_V2:
                compact = decode_mechanism_payload(body)
            else:
                compact = compact_mechanism(json.loads(body))
            out.append(normalize_mechanism(mid, count, compact))
        return out

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
    """The first asset of ``kind`` present in ``assets_dir`` (SHA-256 checked)."""
    for asset in assets:
        if asset.get("kind") != kind:
            continue
        path = Path(assets_dir) / str(asset["name"])
        if not path.is_file():
            continue
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
    """The ``train_mechanisms`` asset (v1 or v2), or ``None`` when it is not
    published or not downloaded."""
    assets = _committed_assets(base_dir)
    if not assets:
        return None
    path = _asset_file(assets, "train_mechanisms", assets_dir)
    return ReferenceMechanismLibrary.load(path) if path is not None else None
