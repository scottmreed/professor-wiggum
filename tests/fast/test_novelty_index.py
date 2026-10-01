"""Novelty index builder, loader and train-only mechanism library.

Uses a SYNTHETIC FlowER-format corpus written to ``tmp_path``: the committed
``training_data/eval_set.json`` cases (themselves FlowER train mechanisms)
rewritten as ``<mapped reaction>|<id>`` lines, plus a few hand-built mapped
reactions. Nothing here reads real FlowER files or the leaderboard holdout.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("rdkit")
np = pytest.importorskip("numpy")

from mechanistic_agent import novelty_index as ni  # noqa: E402
from mechanistic_agent import reaction_signatures as rs  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SET = PROJECT_ROOT / "training_data" / "eval_set.json"

# Two-step mapped mechanism (hydroxide adds to formaldehyde, then protonation)
# with a carried-through [H+] spectator in step 1.
TWO_STEP = [
    "[O-:1][H:5].[CH2:2]=[O:3].[H+:4]>>[H:5][O:1][CH2:2][O-:3].[H+:4]",
    "[H:5][O:1][CH2:2][O-:3].[H+:4]>>[H:5][O:1][CH2:2][O:3][H:4]",
]
TWO_STEP_ID = 900001
# A reaction that exists only in the synthetic test split.
TEST_ONLY = "[Cl-:1].[CH3:2][CH2:4][Br:3]>>[CH3:2][CH2:4][Cl:1].[Br-:3]"
TEST_ONLY_ID = 800001
# A reaction present in both splits.
SHARED = "[Cl-:1].[CH3:2][Br:3]>>[CH3:2][Cl:1].[Br-:3]"


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_script("build_reaction_novelty_index")
converter = _load_script("compact_reference_mechanisms")


def _eval_cases() -> List[Dict[str, Any]]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


def _case_lines(case: Dict[str, Any], mechanism_id: int) -> List[str]:
    return [
        ".".join(step["current_state"]) + ">>" + ".".join(step["resulting_state"]) + f"|{mechanism_id}\n"
        for step in case["verified_mechanism"]["steps"]
    ]


def _strip_maps(smiles: str) -> str:
    return re.sub(r":\d+\]", "]", smiles)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Dict[str, Any]:
    root = tmp_path_factory.mktemp("flower")
    cases = _eval_cases()
    train_cases, test_cases = cases[:60], cases[60:90]
    train_lines: List[str] = []
    for case in train_cases:
        train_lines.extend(_case_lines(case, int(case["id"].split("_")[1])))
    # Non-contiguous two-step mechanism, a trivial row, and noise lines.
    train_lines.insert(3, TWO_STEP[0] + f"|{TWO_STEP_ID}\n")
    train_lines.append("[OH2:1]>>[OH2:1]|" + str(TWO_STEP_ID) + "\n")
    train_lines.append(TWO_STEP[1] + f"|{TWO_STEP_ID}\n")
    train_lines.append("not a flower line\n")
    train_lines.append(SHARED + "|700001\n")
    test_lines: List[str] = []
    for offset, case in enumerate(test_cases):
        test_lines.extend(_case_lines(case, 500000 + offset))
    test_lines.append(TEST_ONLY + f"|{TEST_ONLY_ID}\n")
    test_lines.append(SHARED + "|800002\n")
    # Duplicate reaction under another id: deduplicated by (split, exact key).
    test_lines.append(SHARED + "|800003\n")
    train = root / "train.txt"
    test = root / "test.txt"
    train.write_text("".join(train_lines), encoding="utf-8")
    test.write_text("".join(test_lines), encoding="utf-8")
    return {"dir": root, "train": train, "test": test, "train_cases": train_cases, "test_cases": test_cases}


def _build(corpus: Dict[str, Any], out: Path, *extra: str) -> int:
    return builder.main(["--flower-dir", str(corpus["dir"]), "--out", str(out), *extra])


@pytest.fixture(scope="module")
def built(corpus, tmp_path_factory) -> Dict[str, Any]:
    out = tmp_path_factory.mktemp("novelty")
    assert _build(corpus, out, "--with-mechanisms") == 0
    return {
        "out": out,
        "npz": out / ni.NOVELTY_INDEX_ARTIFACT,
        "manifest": out / ni.NOVELTY_INDEX_MANIFEST,
        "index": ni.NoveltyIndex.load(out / ni.NOVELTY_INDEX_ARTIFACT, out / ni.NOVELTY_INDEX_MANIFEST),
    }


def _manifest(built) -> Dict[str, Any]:
    return json.loads(built["manifest"].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def test_grouping_matches_flower_curriculum(corpus) -> None:
    groups = dict(builder.iter_flower_mechanisms(corpus["train"]))
    assert groups[TWO_STEP_ID] == TWO_STEP  # non-contiguous rows joined, trivial row dropped
    starting, products = builder.mechanism_endpoints(groups[TWO_STEP_ID])
    from mechanistic_agent.flower_curriculum import _convert_group

    case = _convert_group(TWO_STEP_ID, groups[TWO_STEP_ID])
    assert (starting, products) == (case["starting_materials"], case["products"])


# ---------------------------------------------------------------------------
# artifact
# ---------------------------------------------------------------------------


def test_manifest_contents(built, corpus) -> None:
    manifest = _manifest(built)
    assert manifest["artifact"] == ni.NOVELTY_INDEX_ARTIFACT
    assert manifest["corpus_recipe_version"] == rs.CORPUS_RECIPE_VERSION
    assert manifest["recipe_version"] == rs.RECIPE_VERSION
    assert manifest["rdkit_version"] == rs.RDKIT_VERSION
    assert manifest["numpy_version"] == np.__version__
    assert manifest["fingerprints"] is True
    assert manifest["fingerprint_spec"]["bits"] == 256
    assert manifest["bytes"] == built["npz"].stat().st_size
    assert manifest["sha256"] == ni.file_sha256(built["npz"])
    assert manifest["source"]["train_sha256"] == ni.file_sha256(corpus["train"])
    assert manifest["source"]["test_sha256"] == ni.file_sha256(corpus["test"])
    assert manifest["mechanisms_read"] == {"train": 62, "test": 33}
    # 30 test cases + TEST_ONLY + SHARED (its duplicate collapses).
    assert manifest["rows"]["test"] == 32
    assert manifest["skipped"] == {"train": {}, "test": {}}
    train_exact = {rs.corpus_keys(*_case_endpoints(c)).exact for c in corpus["train_cases"]}
    train_exact |= {rs.corpus_keys(["[OH-]", "C=O", "[H+]"], ["OCO"]).exact, rs.corpus_keys(["CBr", "[Cl-]"], ["CCl", "[Br-]"]).exact}
    assert manifest["rows"]["train"] == len(train_exact)
    assert manifest["built_at"]


def test_npz_holds_no_structures(built, corpus) -> None:
    with np.load(built["npz"], allow_pickle=False) as data:
        names = set(data.files)
        assert names == {
            "corpus_recipe_version",
            *(f"{kind}_{part}" for kind in ni.KINDS for part in ("keys", "splits")),
            "fingerprints",
            "fingerprint_splits",
        }
        for name in names - {"corpus_recipe_version"}:
            assert data[name].dtype.kind == "u", name
        assert data["fingerprints"].shape[1] == 64
        for kind in ni.KINDS:
            keys = data[f"{kind}_keys"]
            assert np.all(keys[:-1] <= keys[1:])
    raw = b"".join(zipfile.ZipFile(built["npz"]).read(n) for n in zipfile.ZipFile(built["npz"]).namelist())
    manifest_text = built["manifest"].read_text(encoding="utf-8")
    for species in (TEST_ONLY, SHARED, "[Cl-"):
        assert species.encode() not in raw
        assert species not in manifest_text
    assert not re.search(r"flower_\d", manifest_text)
    assert str(TEST_ONLY_ID) not in manifest_text


def test_build_is_reproducible(corpus, tmp_path) -> None:
    assert _build(corpus, tmp_path / "a") == 0
    assert _build(corpus, tmp_path / "b", "--workers", "2") == 0
    a = tmp_path / "a" / ni.NOVELTY_INDEX_ARTIFACT
    b = tmp_path / "b" / ni.NOVELTY_INDEX_ARTIFACT
    assert ni.file_sha256(a) == ni.file_sha256(b)
    assert a.read_bytes() == b.read_bytes()


def test_size_cap_fails_with_exit_2(corpus, tmp_path, capsys) -> None:
    assert _build(corpus, tmp_path, "--max-bytes", "1000") == 2
    err = capsys.readouterr().err
    assert "--max-bytes" in err and "--no-fingerprints" in err
    assert not (tmp_path / ni.NOVELTY_INDEX_ARTIFACT).exists()
    assert not (tmp_path / ni.NOVELTY_INDEX_MANIFEST).exists()


def test_no_fingerprints_build(corpus, tmp_path) -> None:
    assert _build(corpus, tmp_path, "--no-fingerprints") == 0
    index = ni.NoveltyIndex.load(tmp_path / ni.NOVELTY_INDEX_ARTIFACT, tmp_path / ni.NOVELTY_INDEX_MANIFEST)
    assert not index.fingerprints_available
    assert index.nearest(rs.reaction_fingerprints(["CBr", "[Cl-]"], ["CCl", "[Br-]"])) == []
    assert index.match(rs.corpus_keys(["CBr", "[Cl-]"], ["CCl", "[Br-]"]))


def test_limit_and_out_dir_guard(corpus, tmp_path) -> None:
    assert _build(corpus, tmp_path, "--limit", "5") == 0
    manifest = json.loads((tmp_path / ni.NOVELTY_INDEX_MANIFEST).read_text())
    assert manifest["mechanisms_read"] == {"train": 5, "test": 5}
    with pytest.raises(SystemExit):
        _build(corpus, PROJECT_ROOT / "training_data" / "novelty_should_not_exist")
    assert not (PROJECT_ROOT / "training_data" / "novelty_should_not_exist").exists()


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_loader_rejects_sha_and_recipe_mismatch(built, tmp_path) -> None:
    manifest = _manifest(built)
    bad_sha = tmp_path / "bad_sha.json"
    bad_sha.write_text(json.dumps({**manifest, "sha256": "0" * 64}))
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.NoveltyIndex.load(built["npz"], bad_sha)
    bad_recipe = tmp_path / "bad_recipe.json"
    bad_recipe.write_text(json.dumps({**manifest, "corpus_recipe_version": "reaction_corpus.v0"}))
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.NoveltyIndex.load(built["npz"], bad_recipe)
    # An npz that embeds another recipe is refused even with a matching manifest.
    arrays = dict(np.load(built["npz"], allow_pickle=False))
    arrays["corpus_recipe_version"] = np.array("reaction_corpus.v0")
    other = tmp_path / "other.npz"
    ni.write_npz_deterministic(other, arrays)
    other_manifest = tmp_path / "other.json"
    other_manifest.write_text(json.dumps({**manifest, "sha256": ni.file_sha256(other)}))
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.NoveltyIndex.load(other, other_manifest)


def _case_endpoints(case: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    return list(case["starting_materials"]), list(case["products"])


def test_match_by_split_and_kind(built, corpus) -> None:
    index = built["index"]
    left, right = _case_endpoints(corpus["train_cases"][0])
    result = index.match(rs.corpus_keys(left, right))
    assert {"split": "train", "kind": "exact"} in result
    assert {"split": "train", "kind": "core"} in result
    assert all(set(item) == {"split", "kind"} for item in result)

    # The same chemistry typed by a user: unmapped, reordered, as a submission.
    user = rs.corpus_keys_from_submission(
        {"reactants": [_strip_maps(s) for s in left[::-1]], "products": [_strip_maps(s) for s in right]}
    )
    assert {"split": "train", "kind": "exact"} in index.match(user)

    shared = index.match(rs.corpus_keys(["CBr", "[Cl-]"], ["CCl", "[Br-]"]))
    assert {"split": "train", "kind": "exact"} in shared and {"split": "test", "kind": "exact"} in shared

    test_only = index.match(rs.corpus_keys(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"]))
    assert {item["split"] for item in test_only} == {"test"}

    assert index.match(rs.corpus_keys(["C=CC=C", "C=C"], ["C1=CCCCC1"])) == []
    # Reversed direction is not an exact match.
    reverse = index.match(rs.corpus_keys(right, left))
    assert not any(item["kind"] == "exact" for item in reverse)


def test_nearest_returns_splits_and_similarities_only(built, corpus) -> None:
    index = built["index"]
    left, right = _case_endpoints(corpus["test_cases"][0])
    hits = index.nearest(rs.reaction_fingerprints(left, right), k=3)
    assert len(hits) == 3
    assert hits[0]["split"] == "test" and hits[0]["similarity"] == pytest.approx(1.0)
    assert hits[0]["diff_similarity"] == pytest.approx(1.0)
    assert hits[0]["product_similarity"] == pytest.approx(1.0)
    assert [h["similarity"] for h in hits] == sorted((h["similarity"] for h in hits), reverse=True)
    for hit in hits:
        assert set(hit) == {"split", "similarity", "diff_similarity", "product_similarity"}
        assert 0.0 <= hit["similarity"] <= 1.0
    # Matches the scalar reference implementation.
    fps = rs.reaction_fingerprints(left, right)
    matrix = index.fingerprints
    best = max(
        (rs.tanimoto(bytes(row[:32]), fps[0]) + rs.tanimoto(bytes(row[32:]), fps[1])) / 2 for row in matrix
    )
    assert hits[0]["similarity"] == pytest.approx(best)
    assert index.nearest(fps, k=5, min_similarity=1.01) == []


def test_nearest_latency_at_300k_rows(tmp_path) -> None:
    rng = np.random.default_rng(7)
    rows = 300_000
    fingerprints = (rng.random((rows, 64 * 8)) < 0.1).astype(np.uint8)
    packed = np.packbits(fingerprints, axis=1, bitorder="little")
    keys = np.sort(rng.integers(0, 2**63, size=rows, dtype=np.uint64))
    splits = (rng.random(rows) < 0.1).astype(np.uint8)
    arrays = {"corpus_recipe_version": np.array(rs.CORPUS_RECIPE_VERSION), "fingerprints": packed, "fingerprint_splits": splits}
    for kind in ni.KINDS:
        arrays[f"{kind}_keys"] = keys
        arrays[f"{kind}_splits"] = splits
    npz = tmp_path / "big.npz"
    ni.write_npz_deterministic(npz, arrays)
    manifest = tmp_path / "big.json"
    manifest.write_text(json.dumps({"corpus_recipe_version": rs.CORPUS_RECIPE_VERSION, "sha256": ni.file_sha256(npz)}))
    index = ni.NoveltyIndex.load(npz, manifest)
    query = (bytes(packed[123, :32]), bytes(packed[123, 32:]))
    index.nearest(query)  # warm-up
    started = time.perf_counter()
    for _ in range(5):
        hits = index.nearest(query, k=8)
    elapsed_ms = (time.perf_counter() - started) / 5 * 1000
    print(f"nearest() over {rows:,} rows: {elapsed_ms:.1f} ms")
    assert hits[0]["similarity"] == pytest.approx(1.0)
    assert elapsed_ms < 1000


# ---------------------------------------------------------------------------
# reference mechanism library (train only)
# ---------------------------------------------------------------------------


def _expected_normalized(mechanism_id: int, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The lookup shape, derived independently from raw v1 steps."""
    return {
        "mechanism_id": mechanism_id,
        "step_count": len(steps),
        "initial_state": steps[0]["current_state"],
        "steps": [
            {
                "index": i,
                "resulting_state": step["resulting_state"],
                "electron_pushes": [push["notation"] for push in step["electron_pushes"]],
            }
            for i, step in enumerate(steps, start=1)
        ],
    }


def test_mechanism_library_is_train_only(built, corpus) -> None:
    out = built["out"]
    path = out / ni.MECHANISM_LIBRARY_ARTIFACT
    assert path.name == "flower_train_mechanisms.v2.sqlite" and path.exists()
    assert not (out / ni.MECHANISM_LIBRARY_ARTIFACT_V1).exists()
    assert not list(out.glob("*.gz"))

    conn = sqlite3.connect(path)
    try:
        ids = {row[0] for row in conn.execute("SELECT mechanism_id FROM mechanisms")}
        order = [row for row in conn.execute("SELECT core_key, mechanism_id FROM mechanisms ORDER BY rowid")]
        indexes = [row[1] for row in conn.execute("PRAGMA index_list(mechanisms)")]
        columns = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(mechanisms)")}
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        payload = conn.execute("SELECT payload FROM mechanisms LIMIT 1").fetchone()[0]
    finally:
        conn.close()
    train_ids = {int(c["id"].split("_")[1]) for c in corpus["train_cases"]} | {TWO_STEP_ID, 700001}
    test_ids = {500000 + i for i in range(len(corpus["test_cases"]))} | {TEST_ONLY_ID, 800002, 800003}
    assert ids and ids <= train_ids
    assert not ids & test_ids
    assert order == sorted(order)
    assert columns == {"core_key": "INTEGER", "mechanism_id": "INTEGER", "step_count": "INTEGER", "payload": "BLOB"}
    assert meta["format"] == ni.MECHANISM_LIBRARY_FORMAT_V2
    assert meta["split"] == "train"
    assert meta["corpus_recipe_version"] == rs.CORPUS_RECIPE_VERSION
    assert meta["source_sha256"] == ni.file_sha256(corpus["train"])
    assert meta["rows"] == str(len(ids))
    assert meta["built_at"]
    assert "idx_mechanisms_core_key" in indexes
    assert set(ni.decode_mechanism_payload(payload)) == {"initial_state", "steps"}

    library = ni.ReferenceMechanismLibrary.load(path)
    try:
        assert library.format == ni.MECHANISM_LIBRARY_FORMAT_V2
        case = corpus["train_cases"][0]
        left, right = _case_endpoints(case)
        hits = library.lookup(rs.corpus_keys(left, right).core)
        mid = int(case["id"].split("_")[1])
        expected = _expected_normalized(mid, case["verified_mechanism"]["steps"])
        assert expected in hits
        assert all(set(h) == {"mechanism_id", "step_count", "initial_state", "steps"} for h in hits)
        two = library.reference_mechanisms(rs.corpus_keys(["[OH-]", "C=O", "[H+]"], ["OCO"]).core)
        assert [h["mechanism_id"] for h in two] == [TWO_STEP_ID] and two[0]["step_count"] == 2
        assert [s["index"] for s in two[0]["steps"]] == [1, 2]
        assert two[0]["initial_state"] == TWO_STEP[0].split(">>")[0].split(".")
        assert two[0]["steps"][-1]["resulting_state"] == TWO_STEP[1].split(">>")[1].split(".")
        # The user's bare substrate -> product (no hydroxide, no proton) finds it too.
        bare = library.lookup(rs.corpus_keys_from_submission({"reactants": ["C=O"], "products": ["OCO"]}).core)
        assert [h["mechanism_id"] for h in bare] == [TWO_STEP_ID]
        # A test-split reaction is never served, even though it is in the novelty index.
        assert library.lookup(rs.corpus_keys(["CCBr", "[Cl-]"], ["CCCl", "[Br-]"]).core) == []
        assert len(library) == len(ids)
    finally:
        library.close()

    manifest = _manifest(built)
    assert manifest["train_mechanisms"]["mechanisms"] == len(ids)
    assert manifest["train_mechanisms"]["format"] == ni.MECHANISM_LIBRARY_FORMAT_V2
    assert manifest["train_mechanisms"]["sha256"] == ni.file_sha256(path)
    assert manifest["train_mechanisms"]["bytes"] == path.stat().st_size


def test_mechanism_writer_refuses_test_rows(tmp_path) -> None:
    record = ("test", (1, 2, 1, b"x"))
    with pytest.raises(AssertionError):
        builder.write_mechanism_library(tmp_path, [record])
    assert not (tmp_path / ni.MECHANISM_LIBRARY_ARTIFACT).exists()


def test_library_loader_refuses_non_train(tmp_path) -> None:
    path = tmp_path / "lib.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE meta (key TEXT, value TEXT); CREATE TABLE mechanisms (core_key INTEGER, mechanism_id INTEGER, step_count INTEGER, steps_json TEXT);"
    )
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [("split", "test"), ("corpus_recipe_version", rs.CORPUS_RECIPE_VERSION)])
    conn.commit()
    conn.close()
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.ReferenceMechanismLibrary.load(path)

    v2 = tmp_path / "v2.sqlite"
    ni.write_mechanism_library_v2(v2, [], source_sha256="0" * 64, built_at="x", extra_meta={"split": "test"})
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.ReferenceMechanismLibrary.load(v2)
    # A declared format that does not match the columns is refused too.
    odd = tmp_path / "odd.sqlite"
    ni.write_mechanism_library_v2(odd, [], source_sha256="0" * 64, built_at="x", extra_meta={"format": "train_mechanisms.v9"})
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.ReferenceMechanismLibrary.load(odd)


def test_sqlite_key_mapping_roundtrip() -> None:
    for key in (0, 1, 2**63 - 1, 2**63, 2**64 - 1):
        mapped = ni.to_sqlite_int(key)
        assert -(2**63) <= mapped < 2**63
        assert mapped % 2**64 == key


# ---------------------------------------------------------------------------
# library format v2 and the v1 -> v2 converter
# ---------------------------------------------------------------------------


def _v1_records(cases: List[Dict[str, Any]]) -> List[Tuple[int, int, int, str]]:
    """Old-format rows (as ``write_mechanism_library`` wrote them before v2)."""
    records = []
    for case in cases:
        steps = case["verified_mechanism"]["steps"]
        core = rs.corpus_keys(*_case_endpoints(case)).core
        steps_json = json.dumps(steps, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
        records.append((ni.to_sqlite_int(core), int(case["id"].split("_")[1]), len(steps), steps_json))
    return sorted(records)


def _write_v1(path: Path, cases: List[Dict[str, Any]], split: str = "train") -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE mechanisms (core_key INTEGER NOT NULL, mechanism_id INTEGER NOT NULL,
                                 step_count INTEGER NOT NULL, steps_json TEXT NOT NULL);
        """
    )
    conn.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [("split", split), ("corpus_recipe_version", rs.CORPUS_RECIPE_VERSION), ("builder_version", "build_reaction_novelty_index.v1")],
    )
    conn.executemany("INSERT INTO mechanisms VALUES (?, ?, ?, ?)", _v1_records(cases))
    conn.execute("CREATE INDEX idx_mechanisms_core_key ON mechanisms(core_key)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="module")
def v1_library(corpus, tmp_path_factory) -> Path:
    return _write_v1(tmp_path_factory.mktemp("v1") / ni.MECHANISM_LIBRARY_ARTIFACT_V1, corpus["train_cases"])


def test_v2_write_read_roundtrip(corpus, tmp_path) -> None:
    cases = corpus["train_cases"][:10]
    rows = sorted(
        (
            ni.to_sqlite_int(rs.corpus_keys(*_case_endpoints(c)).core),
            int(c["id"].split("_")[1]),
            len(c["verified_mechanism"]["steps"]),
            ni.encode_mechanism_payload(ni.compact_mechanism(c["verified_mechanism"]["steps"])),
        )
        for c in cases
    )
    path = tmp_path / ni.MECHANISM_LIBRARY_ARTIFACT
    assert ni.write_mechanism_library_v2(path, iter(rows), source_sha256="a" * 64, built_at="2026-10-01T00:00:00+00:00") == 10
    library = ni.ReferenceMechanismLibrary.load(path)
    try:
        assert library.meta["source_sha256"] == "a" * 64 and library.meta["rows"] == "10"
        for case in cases:
            hits = library.lookup(rs.corpus_keys(*_case_endpoints(case)).core)
            expected = _expected_normalized(int(case["id"].split("_")[1]), case["verified_mechanism"]["steps"])
            assert expected in hits
    finally:
        library.close()
    # Payloads are compact: no derivable or constant v1 fields survive.
    decoded = ni.decode_mechanism_payload(rows[0][3])
    flat = json.dumps(decoded)
    for dropped in ("current_state", "target_products", "predicted_intermediate", "reaction_smirks", "note", "confidence"):
        assert dropped not in flat, dropped
    pushes = decoded["steps"][0]["electron_pushes"]
    assert pushes and all(isinstance(push, str) for push in pushes)
    # Out-of-order rows are refused and leave no file behind.
    with pytest.raises(ValueError):
        ni.write_mechanism_library_v2(tmp_path / "bad.sqlite", rows[::-1], source_sha256="", built_at="x")
    assert not (tmp_path / "bad.sqlite").exists()


def test_compact_mechanism_refuses_state_discontinuity() -> None:
    steps = [
        {"current_state": ["A"], "resulting_state": ["B"], "electron_pushes": [{"notation": "lp:1>2"}]},
        {"current_state": ["C"], "resulting_state": ["D"], "electron_pushes": []},
    ]
    with pytest.raises(ValueError):
        ni.compact_mechanism(steps)
    steps[1]["current_state"] = ["B"]
    assert ni.compact_mechanism(steps) == {
        "initial_state": ["A"],
        "steps": [{"resulting_state": ["B"], "electron_pushes": ["lp:1>2"]}, {"resulting_state": ["D"], "electron_pushes": []}],
    }


def test_v1_and_v2_lookups_are_identical(v1_library, corpus, tmp_path) -> None:
    gz = tmp_path / f"{ni.MECHANISM_LIBRARY_ARTIFACT_V1}.gz"
    with v1_library.open("rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    assert converter.main(["--input", str(gz), "--out", str(tmp_path / "out")]) == 0
    v2_path = tmp_path / "out" / ni.MECHANISM_LIBRARY_ARTIFACT
    assert not [p for p in (tmp_path / "out").iterdir() if p != v2_path]  # temp dir cleaned up
    v1 = ni.ReferenceMechanismLibrary.load(v1_library)
    v1_gz = ni.ReferenceMechanismLibrary.load(gz)
    v2 = ni.ReferenceMechanismLibrary.load(v2_path)
    try:
        assert (v1.format, v2.format) == (ni.MECHANISM_LIBRARY_FORMAT_V1, ni.MECHANISM_LIBRARY_FORMAT_V2)
        assert len(v1) == len(v2) == len(corpus["train_cases"])
        assert v2.meta["source_sha256"] == ni.file_sha256(gz)
        for case in corpus["train_cases"]:
            core = rs.corpus_keys(*_case_endpoints(case)).core
            assert v1.lookup(core) == v2.lookup(core) == v1_gz.lookup(core)
            expected = _expected_normalized(int(case["id"].split("_")[1]), case["verified_mechanism"]["steps"])
            assert expected in v2.lookup(core)
    finally:
        for library in (v1, v1_gz, v2):
            library.close()


def test_converter_matches_builder_output(built, v1_library, corpus, tmp_path) -> None:
    assert converter.main(["--input", str(v1_library), "--out", str(tmp_path)]) == 0
    converted = ni.ReferenceMechanismLibrary.load(tmp_path / ni.MECHANISM_LIBRARY_ARTIFACT)
    direct = ni.ReferenceMechanismLibrary.load(built["out"] / ni.MECHANISM_LIBRARY_ARTIFACT)
    try:
        for case in corpus["train_cases"]:
            core = rs.corpus_keys(*_case_endpoints(case)).core
            mid = int(case["id"].split("_")[1])
            pick = lambda hits: [h for h in hits if h["mechanism_id"] == mid]  # noqa: E731
            assert pick(converted.lookup(core)) == pick(direct.lookup(core))
    finally:
        converted.close()
        direct.close()


def test_converter_refuses_non_train_and_non_v1(corpus, tmp_path, capsys) -> None:
    test_lib = _write_v1(tmp_path / "test_split.sqlite", corpus["train_cases"][:3], split="test")
    assert converter.main(["--input", str(test_lib), "--out", str(tmp_path / "out")]) == 2
    assert "not 'train'" in capsys.readouterr().err
    assert not (tmp_path / "out" / ni.MECHANISM_LIBRARY_ARTIFACT).exists()
    with pytest.raises(converter.ConversionRefused):
        converter.convert(test_lib, tmp_path / "out")
    # Already v2: refused.
    v2 = tmp_path / "already.sqlite"
    ni.write_mechanism_library_v2(v2, [], source_sha256="", built_at="x")
    with pytest.raises(converter.ConversionRefused):
        converter.convert(v2, tmp_path / "out2")
    # Discontinuous states cannot be compacted losslessly: refused, no output.
    broken = json.loads(json.dumps(corpus["train_cases"][:1]))
    broken[0]["verified_mechanism"]["steps"] = TWO_STEP_BROKEN
    _write_v1(tmp_path / "broken.sqlite", broken)
    with pytest.raises(converter.ConversionRefused):
        converter.convert(tmp_path / "broken.sqlite", tmp_path / "out3")
    assert not (tmp_path / "out3" / ni.MECHANISM_LIBRARY_ARTIFACT).exists()
    with pytest.raises(SystemExit):
        converter.main(["--input", str(test_lib), "--out", str(PROJECT_ROOT / "training_data" / "novelty_should_not_exist")])
    assert not (PROJECT_ROOT / "training_data" / "novelty_should_not_exist").exists()


TWO_STEP_BROKEN = [
    {"current_state": ["[CH3:1][OH:2]"], "resulting_state": ["[CH3:1][O-:2]", "[H+:3]"], "electron_pushes": [{"notation": "sigma:2-3>2"}]},
    {"current_state": ["[CH4:1]"], "resulting_state": ["[CH4:1]"], "electron_pushes": []},
]


def test_converter_is_deterministic(v1_library, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    first = converter.convert(v1_library, tmp_path / "a")
    second = converter.convert(v1_library, tmp_path / "b")
    assert first["sha256"] == second["sha256"] == ni.file_sha256(tmp_path / "a" / ni.MECHANISM_LIBRARY_ARTIFACT)
    assert (tmp_path / "a" / ni.MECHANISM_LIBRARY_ARTIFACT).read_bytes() == (tmp_path / "b" / ni.MECHANISM_LIBRARY_ARTIFACT).read_bytes()
    assert first["asset"] == {
        "name": "flower_train_mechanisms.v2.sqlite",
        "kind": "train_mechanisms",
        "url": "https://github.com/scottmreed/professor-wiggum/releases/download/novelty-index-reaction_corpus.v1/flower_train_mechanisms.v2.sqlite",
        "sha256": first["sha256"],
        "bytes": (tmp_path / "a" / ni.MECHANISM_LIBRARY_ARTIFACT).stat().st_size,
    }
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    pinned = converter.convert(v1_library, tmp_path / "c")
    assert pinned["built_at"] == "2023-11-14T22:13:20+00:00"
    assert pinned["sha256"] == converter.convert(v1_library, tmp_path / "d")["sha256"]


def test_converter_prints_manifest_entry(v1_library, tmp_path, capsys) -> None:
    assert converter.main(["--input", str(v1_library), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    entry = json.loads(out.strip().splitlines()[-1])
    assert entry["kind"] == "train_mechanisms" and entry["name"] == ni.MECHANISM_LIBRARY_ARTIFACT
    assert entry["sha256"] == ni.file_sha256(tmp_path / ni.MECHANISM_LIBRARY_ARTIFACT)
    assert "rows:" in out and "per mechanism" in out


# ---------------------------------------------------------------------------
# committed manifest
# ---------------------------------------------------------------------------

RELEASE_ASSETS = [
    {
        "name": "reaction_novelty_index.npz",
        "kind": "novelty_index",
        "url": "https://github.com/scottmreed/professor-wiggum/releases/download/novelty-index-reaction_corpus.v1/reaction_novelty_index.npz",
        "sha256": "374ad393a85ed1ac35fe8bd01342f8a05b822e60dd2a9d1af17c5675f514ffd2",
        "bytes": 20184490,
    },
    {
        "name": "reaction_novelty_index.manifest.json",
        "kind": "novelty_index_manifest",
        "url": "https://github.com/scottmreed/professor-wiggum/releases/download/novelty-index-reaction_corpus.v1/reaction_novelty_index.manifest.json",
        "sha256": "9d5b5d4f6864cd7df7ae1dde1887a82c43e3168bf60202cc66e95dcfdcb0dc34",
        "bytes": 2332,
    },
    {
        "name": "flower_train_mechanisms.v2.sqlite",
        "kind": "train_mechanisms",
        "url": "https://github.com/scottmreed/professor-wiggum/releases/download/novelty-index-reaction_corpus.v1/flower_train_mechanisms.v2.sqlite",
        "sha256": "4dd5c2e8745e77fddf5220dbf73d9b0761a4e572ed7ff7ba66da5fd414c90354",
        "bytes": 60297216,
    },
]


def test_committed_manifest_names_the_release_assets(tmp_path) -> None:
    committed = json.loads((PROJECT_ROOT / "novelty_index" / "manifest.json").read_text(encoding="utf-8"))
    assert committed["schema"] == ni.COMMITTED_MANIFEST_SCHEMA
    assert committed["release_tag"] == "novelty-index-reaction_corpus.v1"
    assert committed["assets"] == RELEASE_ASSETS
    for asset in committed["assets"]:
        assert asset["url"].endswith(f"/releases/download/{committed['release_tag']}/{asset['name']}")
    # Files not downloaded: unavailable, not an error.
    assert ni.load_from_manifest(PROJECT_ROOT, tmp_path) is None
    assert ni.load_reference_library_from_manifest(PROJECT_ROOT, tmp_path) is None
    assert ni.load_from_manifest(tmp_path / "no_repo", tmp_path) is None
    # A downloaded file that is not the released one is refused.
    (tmp_path / "reaction_novelty_index.npz").write_bytes(b"not the release")
    (tmp_path / "reaction_novelty_index.manifest.json").write_text("{}")
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.load_from_manifest(PROJECT_ROOT, tmp_path)
    (tmp_path / "flower_train_mechanisms.v2.sqlite").write_bytes(b"not the release")
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.load_reference_library_from_manifest(PROJECT_ROOT, tmp_path)


def test_null_assets_manifest_is_unavailable(tmp_path) -> None:
    base = tmp_path / "wiggum"
    (base / "novelty_index").mkdir(parents=True)
    (base / "novelty_index" / "manifest.json").write_text(json.dumps({"schema": ni.COMMITTED_MANIFEST_SCHEMA, "assets": None}))
    assert ni.load_from_manifest(base, tmp_path) is None
    assert ni.load_reference_library_from_manifest(base, tmp_path) is None


_REAL_ASSETS_DIR = Path(os.environ.get("WIGGUM_NOVELTY_ASSETS_DIR") or PROJECT_ROOT / "dist" / "novelty")


@pytest.mark.skipif(
    not all((_REAL_ASSETS_DIR / a["name"]).is_file() for a in RELEASE_ASSETS),
    reason="released novelty index not downloaded (set WIGGUM_NOVELTY_ASSETS_DIR)",
)
def test_committed_manifest_loads_the_released_index() -> None:
    index = ni.load_from_manifest(PROJECT_ROOT, _REAL_ASSETS_DIR)
    assert isinstance(index, ni.NoveltyIndex) and index.fingerprints_available
    assert index.manifest["rows"] == {"train": 250408, "test": 28739}
    assert index.manifest["sha256"] == RELEASE_ASSETS[0]["sha256"]
    # The committed eval cases are FlowER train mechanisms.
    case = _eval_cases()[0]
    assert {"split": "train", "kind": "exact"} in index.match(rs.corpus_keys(*_case_endpoints(case)))
    # No train_mechanisms entry is committed yet.
    assert ni.load_reference_library_from_manifest(PROJECT_ROOT, _REAL_ASSETS_DIR) is None


def test_load_from_populated_manifest(built, tmp_path) -> None:
    base = tmp_path / "wiggum"
    (base / "novelty_index").mkdir(parents=True)
    out = built["out"]
    assets = [
        {"name": ni.NOVELTY_INDEX_ARTIFACT, "kind": "novelty_index", "sha256": ni.file_sha256(built["npz"])},
        {"name": ni.NOVELTY_INDEX_MANIFEST, "kind": "novelty_index_manifest", "sha256": ni.file_sha256(built["manifest"])},
        {
            "name": ni.MECHANISM_LIBRARY_ARTIFACT,
            "kind": "train_mechanisms",
            "sha256": ni.file_sha256(out / ni.MECHANISM_LIBRARY_ARTIFACT),
        },
    ]
    manifest_path = base / "novelty_index" / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": ni.COMMITTED_MANIFEST_SCHEMA, "release_tag": "t", "assets": assets}))
    index = ni.load_from_manifest(base, out)
    assert isinstance(index, ni.NoveltyIndex) and index.fingerprints_available
    library = ni.load_reference_library_from_manifest(base, out)
    assert library is not None and len(library) > 0 and library.format == ni.MECHANISM_LIBRARY_FORMAT_V2
    library.close()
    assert ni.load_from_manifest(base, tmp_path / "empty") is None  # files missing
    assert ni.load_reference_library_from_manifest(base, tmp_path / "empty") is None

    assets[0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps({"schema": ni.COMMITTED_MANIFEST_SCHEMA, "assets": assets}))
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.load_from_manifest(base, out)
