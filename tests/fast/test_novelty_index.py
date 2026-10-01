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


def _load_builder():
    path = PROJECT_ROOT / "scripts" / "build_reaction_novelty_index.py"
    spec = importlib.util.spec_from_file_location("build_reaction_novelty_index", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


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


def test_mechanism_library_is_train_only(built, corpus) -> None:
    out = built["out"]
    sqlite_path = out / f"{ni.MECHANISM_LIBRARY_ARTIFACT}"
    gz_path = out / f"{ni.MECHANISM_LIBRARY_ARTIFACT}.gz"
    assert sqlite_path.exists() and gz_path.exists()
    with gzip.open(gz_path, "rb") as handle:
        assert handle.read() == sqlite_path.read_bytes()

    conn = sqlite3.connect(sqlite_path)
    try:
        ids = {row[0] for row in conn.execute("SELECT mechanism_id FROM mechanisms")}
        indexes = [row[1] for row in conn.execute("PRAGMA index_list(mechanisms)")]
        meta = dict(conn.execute("SELECT key, value FROM meta"))
    finally:
        conn.close()
    train_ids = {int(c["id"].split("_")[1]) for c in corpus["train_cases"]} | {TWO_STEP_ID, 700001}
    test_ids = {500000 + i for i in range(len(corpus["test_cases"]))} | {TEST_ONLY_ID, 800002, 800003}
    assert ids and ids <= train_ids
    assert not ids & test_ids
    assert meta["split"] == "train"
    assert "idx_mechanisms_core_key" in indexes

    library = ni.ReferenceMechanismLibrary.load(gz_path)
    try:
        left, right = _case_endpoints(corpus["train_cases"][0])
        hits = library.lookup(rs.corpus_keys(left, right).core)
        assert hits and hits[0]["steps"] == corpus["train_cases"][0]["verified_mechanism"]["steps"]
        assert hits[0]["step_count"] == len(hits[0]["steps"])
        two = library.reference_mechanisms(rs.corpus_keys(["[OH-]", "C=O", "[H+]"], ["OCO"]).core)
        assert [h["mechanism_id"] for h in two] == [TWO_STEP_ID] and two[0]["step_count"] == 2
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
    assert manifest["train_mechanisms"]["gz_sha256"] == ni.file_sha256(gz_path)


def test_mechanism_writer_refuses_test_rows(tmp_path) -> None:
    record = ("test", (1, 2, 1, "[]"))
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


def test_sqlite_key_mapping_roundtrip() -> None:
    for key in (0, 1, 2**63 - 1, 2**63, 2**64 - 1):
        mapped = ni.to_sqlite_int(key)
        assert -(2**63) <= mapped < 2**63
        assert mapped % 2**64 == key


# ---------------------------------------------------------------------------
# committed manifest
# ---------------------------------------------------------------------------


def test_committed_manifest_is_null_and_loads_as_unavailable(tmp_path) -> None:
    committed = json.loads((PROJECT_ROOT / "novelty_index" / "manifest.json").read_text(encoding="utf-8"))
    assert committed["schema"] == ni.COMMITTED_MANIFEST_SCHEMA
    assert committed["assets"] is None
    assert ni.load_from_manifest(PROJECT_ROOT, tmp_path) is None
    assert ni.load_reference_library_from_manifest(PROJECT_ROOT, tmp_path) is None
    assert ni.load_from_manifest(tmp_path / "no_repo", tmp_path) is None


def test_load_from_populated_manifest(built, tmp_path) -> None:
    base = tmp_path / "wiggum"
    (base / "novelty_index").mkdir(parents=True)
    out = built["out"]
    assets = [
        {"name": ni.NOVELTY_INDEX_ARTIFACT, "kind": "novelty_index", "sha256": ni.file_sha256(built["npz"])},
        {"name": ni.NOVELTY_INDEX_MANIFEST, "kind": "novelty_index_manifest", "sha256": ni.file_sha256(built["manifest"])},
        {
            "name": f"{ni.MECHANISM_LIBRARY_ARTIFACT}.gz",
            "kind": "train_mechanisms",
            "sha256": ni.file_sha256(out / f"{ni.MECHANISM_LIBRARY_ARTIFACT}.gz"),
        },
    ]
    manifest_path = base / "novelty_index" / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": ni.COMMITTED_MANIFEST_SCHEMA, "assets": assets}))
    index = ni.load_from_manifest(base, out)
    assert isinstance(index, ni.NoveltyIndex) and index.fingerprints_available
    library = ni.load_reference_library_from_manifest(base, out)
    assert library is not None and len(library) > 0
    library.close()
    assert ni.load_from_manifest(base, tmp_path / "empty") is None  # files missing

    assets[0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps({"schema": ni.COMMITTED_MANIFEST_SCHEMA, "assets": assets}))
    with pytest.raises(ni.NoveltyIndexMismatch):
        ni.load_from_manifest(base, out)
