#!/usr/bin/env python3
"""Convert a v1 train-only reference mechanism library to ``train_mechanisms.v2``.

v1 is ``flower_train_mechanisms.sqlite`` (or its ``.sqlite.gz``) from
``build_reaction_novelty_index.py --with-mechanisms`` before the v2 format:
full ``verified_mechanism.steps`` JSON per row, about 11 KB per mechanism.
v2 keeps the initial state once plus each step's resulting state and push
notations, zlib-compressed per row (layout: ``mechanistic_agent/novelty_index.py``).
No FlowER rebuild is needed.

    python scripts/compact_reference_mechanisms.py \\
        --input dist/novelty/flower_train_mechanisms.sqlite.gz --out dist/novelty

Writes ``<out>/flower_train_mechanisms.v2.sqlite`` and prints its size,
SHA-256 and the ``novelty_index/manifest.json`` asset entry to paste. Refuses
a v1 file whose meta is not ``split = train`` with this code's corpus recipe,
and any mechanism whose step states are not continuous (the dropped
``current_state`` would not be derivable). Rows are streamed one at a time.
The same input gives the same bytes: ``built_at`` is ``SOURCE_DATE_EPOCH``
when set, else the input file's mtime.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mechanistic_agent.novelty_index import (  # noqa: E402
    MECHANISM_LIBRARY_ARTIFACT,
    MECHANISM_LIBRARY_FORMAT_V1,
    compact_mechanism,
    default_built_at,
    encode_mechanism_payload,
    file_sha256,
    write_mechanism_library_v2,
)
from mechanistic_agent.reaction_signatures import CORPUS_RECIPE_VERSION  # noqa: E402

CONVERTER_VERSION = "compact_reference_mechanisms.v1"
RELEASE_TAG = f"novelty-index-{CORPUS_RECIPE_VERSION}"
RELEASE_URL = f"https://github.com/scottmreed/professor-wiggum/releases/download/{RELEASE_TAG}"


class ConversionRefused(RuntimeError):
    """The input is not a convertible train-only v1 library."""


def _check_v1(conn: sqlite3.Connection) -> Dict[str, str]:
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        columns = {row[1] for row in conn.execute("PRAGMA table_info(mechanisms)")}
    except sqlite3.DatabaseError as exc:
        raise ConversionRefused(f"not a reference mechanism library: {exc}") from exc
    if meta.get("format") not in (None, MECHANISM_LIBRARY_FORMAT_V1) or "steps_json" not in columns:
        raise ConversionRefused(f"input is not a v1 library (format {meta.get('format')!r}, columns {sorted(columns)})")
    if meta.get("split") != "train":
        raise ConversionRefused(f"input split is {meta.get('split')!r}, not 'train'; refusing to convert")
    if meta.get("corpus_recipe_version") != CORPUS_RECIPE_VERSION:
        raise ConversionRefused(
            f"input corpus recipe {meta.get('corpus_recipe_version')!r} does not match {CORPUS_RECIPE_VERSION!r}"
        )
    return meta


def _iter_rows(conn: sqlite3.Connection, stats: Dict[str, int]) -> Iterator[Tuple[int, int, int, bytes]]:
    """v2 rows in ``(core_key, mechanism_id)`` order, one v1 row in memory at a time."""
    order = conn.execute("SELECT rowid FROM mechanisms ORDER BY core_key, mechanism_id").fetchall()
    for (rowid,) in order:
        core, mid, count, steps_json = conn.execute(
            "SELECT core_key, mechanism_id, step_count, steps_json FROM mechanisms WHERE rowid = ?", (rowid,)
        ).fetchone()
        steps = json.loads(steps_json)
        if int(count) != len(steps):
            raise ConversionRefused(f"mechanism {mid}: step_count {count} but {len(steps)} steps")
        try:
            payload = encode_mechanism_payload(compact_mechanism(steps))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionRefused(f"mechanism {mid}: {exc}") from exc
        stats["steps"] += len(steps)
        stats["v1_json_bytes"] += len(steps_json.encode("utf-8"))
        stats["payload_bytes"] += len(payload)
        yield int(core), int(mid), int(count), payload


def convert(input_path: Path, out_dir: Path, tmp_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Convert ``input_path`` (v1 ``.sqlite`` or ``.sqlite.gz``) into ``out_dir``."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_sha256 = file_sha256(input_path)
    built_at = default_built_at(input_path)
    tempdir: Optional[str] = None
    sqlite_path = input_path
    try:
        if input_path.suffix == ".gz":
            tempdir = tempfile.mkdtemp(prefix="wiggum_v1_", dir=str(tmp_dir or out_dir))
            sqlite_path = Path(tempdir) / input_path.with_suffix("").name
            with gzip.open(input_path, "rb") as src, sqlite_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            _check_v1(conn)
            stats = {"steps": 0, "v1_json_bytes": 0, "payload_bytes": 0}
            out_path = out_dir / MECHANISM_LIBRARY_ARTIFACT
            rows = write_mechanism_library_v2(
                out_path,
                _iter_rows(conn, stats),
                source_sha256=source_sha256,
                built_at=built_at,
                extra_meta={"converter_version": CONVERTER_VERSION, "source": f"{MECHANISM_LIBRARY_FORMAT_V1} {input_path.name}"},
            )
        finally:
            conn.close()
    finally:
        if tempdir:
            shutil.rmtree(tempdir, ignore_errors=True)
    size = out_path.stat().st_size
    sha = file_sha256(out_path)
    return {
        "path": str(out_path),
        "rows": rows,
        "steps": stats["steps"],
        "bytes": size,
        "sha256": sha,
        "source_sha256": source_sha256,
        "built_at": built_at,
        "seconds": round(time.perf_counter() - started, 1),
        "avg_steps_per_mechanism": round(stats["steps"] / rows, 3) if rows else 0.0,
        "avg_bytes_per_mechanism": round(size / rows, 1) if rows else 0.0,
        "avg_payload_bytes_per_mechanism": round(stats["payload_bytes"] / rows, 1) if rows else 0.0,
        "avg_v1_json_bytes_per_mechanism": round(stats["v1_json_bytes"] / rows, 1) if rows else 0.0,
        "asset": {
            "name": MECHANISM_LIBRARY_ARTIFACT,
            "kind": "train_mechanisms",
            "url": f"{RELEASE_URL}/{MECHANISM_LIBRARY_ARTIFACT}",
            "sha256": sha,
            "bytes": size,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="v1 flower_train_mechanisms.sqlite or .sqlite.gz")
    parser.add_argument("--out", type=Path, required=True, help="Output directory (not under training_data/)")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Where to decompress a .gz input (default: --out)")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"not a file: {args.input}")
    out = args.out.resolve()
    training = (PROJECT_ROOT / "training_data").resolve()
    if out == training or training in out.parents:
        parser.error("--out must not be inside training_data/ (use e.g. dist/novelty)")
    try:
        result = convert(args.input, args.out, args.tmp_dir)
    except ConversionRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{MECHANISM_LIBRARY_ARTIFACT}: {result['bytes']:,} bytes, sha256 {result['sha256']}")
    print(
        f"  rows: {result['rows']:,} train mechanisms, {result['steps']:,} steps "
        f"({result['avg_steps_per_mechanism']} per mechanism)"
    )
    print(
        f"  per mechanism: {result['avg_bytes_per_mechanism']} file bytes, "
        f"{result['avg_payload_bytes_per_mechanism']} payload bytes (v1 JSON {result['avg_v1_json_bytes_per_mechanism']})"
    )
    print(f"  source sha256 {result['source_sha256']}; built_at {result['built_at']}; {result['seconds']} s")
    print(f"\nRelease tag {RELEASE_TAG}; add to novelty_index/manifest.json \"assets\":")
    print(json.dumps(result["asset"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
