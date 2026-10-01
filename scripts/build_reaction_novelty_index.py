#!/usr/bin/env python3
"""Build the FlowER reaction novelty index (and, optionally, the train-only
reference mechanism library).

Reads FlowER ``train.txt`` and ``test.txt`` (lines ``<mapped reaction>|<id>``,
grouped by id exactly as ``flower_curriculum.build_lookup_cache`` does: file
order within an id, trivial ``A>>A`` rows skipped). Each mechanism becomes one
reaction: the unique species of its first step's left side ``>>`` the unique
species of its last step's right side (``_convert_group``'s
``starting_materials`` → ``products``); atom maps are ignored.

Outputs in ``--out``:

* ``reaction_novelty_index.npz`` — split labels, 64-bit corpus keys and
  256-bit fingerprints only (layout: ``mechanistic_agent/novelty_index.py``);
* ``reaction_novelty_index.manifest.json`` — provenance, row counts, size, SHA-256;
* with ``--with-mechanisms``: ``flower_train_mechanisms.v2.sqlite`` — TRAIN
  mechanisms only, ``train_mechanisms.v2`` format (per-row zlib, so the file
  is not gzipped).

Exit 2 when the index exceeds ``--max-bytes`` (rebuild with ``--no-fingerprints``).

    python scripts/build_reaction_novelty_index.py \\
        --flower-dir ../FlowER/data/flower_new_dataset --with-mechanisms --out dist/novelty
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.flower_curriculum import (  # noqa: E402
    ConversionError,
    _convert_group,
    _is_trivial_reaction,
    _parse_line,
    _split_reaction_sides,
    _tokenize_species,
    _unique_preserving_order,
)
from mechanistic_agent.novelty_index import (  # noqa: E402
    KINDS,
    MECHANISM_LIBRARY_ARTIFACT,
    NOVELTY_INDEX_ARTIFACT,
    NOVELTY_INDEX_MANIFEST,
    compact_mechanism,
    default_built_at,
    encode_mechanism_payload,
    file_sha256,
    to_sqlite_int,
    write_mechanism_library_v2,
    write_npz_deterministic,
)
from mechanistic_agent.reaction_signatures import (  # noqa: E402
    CORPUS_RECIPE_VERSION,
    FINGERPRINT_SPEC,
    RDKIT_VERSION,
    RECIPE_VERSION,
    CorpusKeyError,
    corpus_keys_and_fingerprints,
)

BUILDER_VERSION = "build_reaction_novelty_index.v1"
DEFAULT_MAX_BYTES = 40_000_000
SPLIT_CODES = {"train": 0, "test": 1}
RELEASE_TAG = f"novelty-index-{CORPUS_RECIPE_VERSION}"
RELEASE_URL = f"https://github.com/scottmreed/professor-wiggum/releases/download/{RELEASE_TAG}"


# ---------------------------------------------------------------------------
# reading FlowER
# ---------------------------------------------------------------------------


def iter_flower_mechanisms(path: Path) -> Iterator[Tuple[int, List[str]]]:
    """Yield ``(mechanism_id, [mapped reaction, ...])`` in ascending id order.

    Same grouping rules as ``flower_curriculum.build_lookup_cache``: lines that
    do not end in ``|<digits>`` are ignored, trivial rows are skipped, and the
    remaining rows keep file order within an id (ids may be non-contiguous).
    """
    offsets: Dict[int, List[int]] = {}
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            parsed = _parse_line(raw.decode("utf-8"))
            if parsed is None or _is_trivial_reaction(parsed.mapped_reaction):
                continue
            offsets.setdefault(parsed.mechanism_id, []).append(offset)
    with path.open("rb") as handle:
        for mechanism_id in sorted(offsets):
            reactions: List[str] = []
            for offset in offsets[mechanism_id]:
                handle.seek(offset)
                parsed = _parse_line(handle.readline().decode("utf-8"))
                if parsed is not None:
                    reactions.append(parsed.mapped_reaction)
            yield mechanism_id, reactions


def mechanism_endpoints(reactions: Sequence[str]) -> Tuple[List[str], List[str]]:
    """First state → final state, as ``_convert_group`` reports them."""
    left, _ = _split_reaction_sides(reactions[0])
    _, right = _split_reaction_sides(reactions[-1])
    return _unique_preserving_order(_tokenize_species(left)), _unique_preserving_order(_tokenize_species(right))


# ---------------------------------------------------------------------------
# per-mechanism work (runs in worker processes)
# ---------------------------------------------------------------------------


def _process(task: Tuple[str, int, List[str], bool, bool]) -> Dict[str, Any]:
    split, mechanism_id, reactions, with_fps, with_mechanism = task
    out: Dict[str, Any] = {"split": split}
    try:
        starting, products = mechanism_endpoints(reactions)
        keys, fps = corpus_keys_and_fingerprints(starting, products, fingerprints=with_fps)
    except ConversionError as exc:
        out["skip"] = f"endpoints_{exc.reason}"
        return out
    except CorpusKeyError:
        out["skip"] = "unkeyable"
        return out
    except Exception:  # pragma: no cover - RDKit edge cases
        out["skip"] = "rdkit_error"
        return out
    out["keys"] = (keys.exact, keys.core, keys.endpoint, keys.family)
    out["fp"] = (fps[0] + fps[1]) if fps else b""
    if with_mechanism:
        if split != "train":  # the library is train-only, always
            raise AssertionError("mechanism conversion requested for a non-train split")
        try:
            case = _convert_group(mechanism_id, reactions)
        except ConversionError as exc:
            out["mechanism_skip"] = exc.reason
        except Exception:  # pragma: no cover - conversion edge cases
            out["mechanism_skip"] = "conversion_error"
        else:
            steps = case["verified_mechanism"]["steps"]
            out["mechanism"] = (
                keys.core,
                int(mechanism_id),
                len(steps),
                encode_mechanism_payload(compact_mechanism(steps)),
            )
    return out


def _tasks(sources: Sequence[Tuple[str, Path]], with_fps: bool, with_mechanisms: bool, limit: Optional[int]):
    for split, path in sources:
        for count, (mechanism_id, reactions) in enumerate(iter_flower_mechanisms(path)):
            if limit is not None and count >= limit:
                break
            if reactions:
                yield (split, mechanism_id, reactions, with_fps, with_mechanisms and split == "train")


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def assemble_index_arrays(rows: Sequence[Tuple[int, Tuple[int, int, int, int], bytes]], with_fps: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Dedupe ``(split, exact)`` rows and build the npz arrays."""
    ordered = sorted(rows, key=lambda row: (row[0], row[1], row[2]))
    unique: List[Tuple[int, Tuple[int, int, int, int], bytes]] = []
    seen: set = set()
    for row in ordered:
        ident = (row[0], row[1][0])
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(row)

    arrays: Dict[str, np.ndarray] = {"corpus_recipe_version": np.array(CORPUS_RECIPE_VERSION)}
    splits = np.array([row[0] for row in unique], dtype=np.uint8)
    for position, kind in enumerate(KINDS):
        keys = np.array([row[1][position] for row in unique], dtype=np.uint64)
        pairs = sorted(set(zip(keys.tolist(), splits.tolist())))
        arrays[f"{kind}_keys"] = np.array([key for key, _ in pairs], dtype=np.uint64)
        arrays[f"{kind}_splits"] = np.array([code for _, code in pairs], dtype=np.uint8)
    if with_fps:
        matrix = np.frombuffer(b"".join(row[2] for row in unique), dtype=np.uint8)
        arrays["fingerprints"] = matrix.reshape(len(unique), -1) if unique else np.zeros((0, 64), dtype=np.uint8)
        arrays["fingerprint_splits"] = splits
    counts = {split: int((splits == code).sum()) for split, code in SPLIT_CODES.items()}
    return arrays, counts


def write_mechanism_library(
    out_dir: Path,
    records: Sequence[Tuple[str, Tuple[int, int, int, bytes]]],
    *,
    source_sha256: str = "",
    built_at: str = "1970-01-01T00:00:00+00:00",
) -> Dict[str, Any]:
    """Write the train-only ``train_mechanisms.v2`` library; returns size stats.

    ``records`` are ``(split, (core_key, mechanism_id, step_count, payload))``
    with an unsigned core key and an :func:`encode_mechanism_payload` payload;
    any non-train record is a hard error.
    """
    for split, _ in records:
        if split != "train":
            raise AssertionError("refusing to write a non-train mechanism into the reference library")
    path = out_dir / MECHANISM_LIBRARY_ARTIFACT
    rows = sorted((to_sqlite_int(core), mid, count, payload) for _, (core, mid, count, payload) in records)
    count = write_mechanism_library_v2(
        path,
        rows,
        source_sha256=source_sha256,
        built_at=built_at,
        extra_meta={"builder_version": BUILDER_VERSION, "source": "FlowER train.txt"},
    )
    size = path.stat().st_size
    steps_total = sum(row[2] for row in rows)
    return {
        "artifact": path.name,
        "format": "train_mechanisms.v2",
        "split": "train",
        "mechanisms": count,
        "steps": steps_total,
        "bytes": size,
        "sha256": file_sha256(path),
        "avg_steps_per_mechanism": round(steps_total / count, 3) if count else 0.0,
        "avg_bytes_per_mechanism": round(size / count, 1) if count else 0.0,
        "avg_payload_bytes_per_mechanism": round(sum(len(row[3]) for row in rows) / count, 1) if count else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flower-dir", type=Path, help="Directory containing train.txt and test.txt")
    parser.add_argument("--train-file", type=Path, help="FlowER train.txt (instead of --flower-dir)")
    parser.add_argument("--test-file", type=Path, help="FlowER test.txt (instead of --flower-dir)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory (not under training_data/)")
    parser.add_argument("--no-fingerprints", action="store_true", help="Keys only (smaller artifact)")
    parser.add_argument("--with-mechanisms", action="store_true", help="Also write the train-only mechanism library")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Fail above this index size")
    parser.add_argument("--limit", type=int, default=None, help="Mechanisms per split (smoke runs)")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (output is order-independent)")
    args = parser.parse_args(argv)
    if args.flower_dir:
        args.train_file = args.train_file or args.flower_dir / "train.txt"
        args.test_file = args.test_file or args.flower_dir / "test.txt"
    if not args.train_file or not args.test_file:
        parser.error("give --flower-dir, or both --train-file and --test-file")
    for path in (args.train_file, args.test_file):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    out = args.out.resolve()
    training = (PROJECT_ROOT / "training_data").resolve()
    if out == training or training in out.parents:
        # Holdout isolation: build output never lands beside committed or holdout eval data.
        parser.error("--out must not be inside training_data/ (use e.g. dist/novelty)")
    return args


def _asset(name: str, kind: str, path: Path) -> Dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "url": f"{RELEASE_URL}/{name}",
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with_fps = not args.no_fingerprints
    sources = [("train", args.train_file), ("test", args.test_file)]

    rows: List[Tuple[int, Tuple[int, int, int, int], bytes]] = []
    mechanisms: List[Tuple[str, Tuple[int, int, int, str]]] = []
    read = Counter()
    skipped: Dict[str, Counter] = {"train": Counter(), "test": Counter()}
    mechanism_skipped: Counter = Counter()

    tasks = _tasks(sources, with_fps, args.with_mechanisms, args.limit)
    pool = Pool(args.workers) if args.workers > 1 else None
    try:
        results = pool.imap(_process, tasks, chunksize=256) if pool else map(_process, tasks)
        for result in results:
            split = result["split"]
            read[split] += 1
            if "skip" in result:
                skipped[split][result["skip"]] += 1
                continue
            rows.append((SPLIT_CODES[split], result["keys"], result["fp"]))
            if "mechanism" in result:
                mechanisms.append((split, result["mechanism"]))
            elif "mechanism_skip" in result:
                mechanism_skipped[result["mechanism_skip"]] += 1
    finally:
        if pool:
            pool.close()
            pool.join()

    arrays, row_counts = assemble_index_arrays(rows, with_fps)
    npz_path = out_dir / NOVELTY_INDEX_ARTIFACT
    manifest_path = out_dir / NOVELTY_INDEX_MANIFEST
    write_npz_deterministic(npz_path, arrays)
    size = npz_path.stat().st_size
    if size > args.max_bytes:
        npz_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        hint = " Rebuild with --no-fingerprints." if with_fps else ""
        print(
            f"error: {NOVELTY_INDEX_ARTIFACT} is {size:,} bytes, over --max-bytes {args.max_bytes:,}.{hint}",
            file=sys.stderr,
        )
        return 2

    manifest: Dict[str, Any] = {
        "artifact": NOVELTY_INDEX_ARTIFACT,
        "corpus_recipe_version": CORPUS_RECIPE_VERSION,
        "recipe_version": RECIPE_VERSION,
        "rdkit_version": RDKIT_VERSION,
        "numpy_version": np.__version__,
        "builder_version": BUILDER_VERSION,
        "source": {
            "train_sha256": file_sha256(args.train_file),
            "test_sha256": file_sha256(args.test_file),
            "limit": args.limit,
        },
        "mechanisms_read": {split: int(read[split]) for split in SPLIT_CODES},
        "skipped": {split: dict(sorted(skipped[split].items())) for split in SPLIT_CODES},
        "rows": row_counts,
        "kind_entries": {kind: int(len(arrays[f"{kind}_keys"])) for kind in KINDS},
        "fingerprints": with_fps,
        "fingerprint_spec": FINGERPRINT_SPEC if with_fps else None,
        "bytes": size,
        "sha256": file_sha256(npz_path),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    library: Optional[Dict[str, Any]] = None
    if args.with_mechanisms:
        library = write_mechanism_library(
            out_dir,
            mechanisms,
            source_sha256=manifest["source"]["train_sha256"],
            built_at=default_built_at(args.train_file),
        )
        library["skipped"] = dict(sorted(mechanism_skipped.items()))
        manifest["train_mechanisms"] = library
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total_rows = sum(row_counts.values())
    print(f"{NOVELTY_INDEX_ARTIFACT}: {size:,} bytes, sha256 {manifest['sha256']}")
    print(
        f"  rows: train {row_counts['train']:,}, test {row_counts['test']:,}"
        f" ({size / total_rows:.1f} bytes/row)" if total_rows else "  rows: 0"
    )
    print(f"  skipped: {json.dumps(manifest['skipped'], sort_keys=True)}")
    if library is not None:
        print(
            f"{library['artifact']}: {library['bytes']:,} bytes; {library['mechanisms']:,} train mechanisms, "
            f"{library['avg_steps_per_mechanism']} steps and {library['avg_bytes_per_mechanism']} bytes "
            f"per mechanism (skipped: {json.dumps(library['skipped'], sort_keys=True)})"
        )
    assets = [
        _asset(NOVELTY_INDEX_ARTIFACT, "novelty_index", npz_path),
        _asset(NOVELTY_INDEX_MANIFEST, "novelty_index_manifest", manifest_path),
    ]
    if library is not None:
        assets.append(_asset(library["artifact"], "train_mechanisms", out_dir / library["artifact"]))
    print(f"\nRelease tag {RELEASE_TAG}; novelty_index/manifest.json \"assets\":")
    print(json.dumps(assets, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
