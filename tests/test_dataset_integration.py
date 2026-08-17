"""Data-contract tests against the actual generated dataset artifacts.
Unlike the rest of the suite (pure functions), these check invariants on
the real files under output_v3/ — skipped if a given file hasn't been
generated in this environment (e.g. a clean checkout before running
build_behavior_dataset.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from build_behavior_dataset import ROOT, sha256_of

DATASET_PATH = ROOT / "output_v3/patch=16.14/behavior_windows_v3.parquet"
MANIFEST_PATH = ROOT / "output_v3/patch=16.14/MANIFEST.json"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} not generated in this environment")


def test_no_puuid_or_match_split_leakage():
    _require(DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH, columns=["puuid", "match_id", "split"])
    assert df.groupby("puuid")["split"].nunique().max() <= 1, "a puuid appears on both sides of the split"
    assert df.groupby("match_id")["split"].nunique().max() <= 1, "a match appears on both sides of the split"


def test_no_duplicate_windows():
    _require(DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH, columns=["match_id", "participant_id", "feature_cutoff_minute"])
    assert not df.duplicated(subset=["match_id", "participant_id", "feature_cutoff_minute"]).any()


def test_manifest_file_hashes_match_actual_files():
    _require(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    file_hash_pairs = [
        ("dataset_file", "dataset_file_sha256"),
        ("baseline_file", "baseline_file_sha256"),
        ("rate_baseline_file", "rate_baseline_file_sha256"),
    ]
    for file_key, hash_key in file_hash_pairs:
        if file_key not in manifest:
            continue  # older manifest format, nothing to check
        path = ROOT / manifest[file_key]
        _require(path)
        assert sha256_of(path) == manifest[hash_key], f"{file_key} on disk does not match the manifest's recorded hash"

    for name, expected_hash in manifest.get("source_files_sha256", {}).items():
        path = ROOT / name
        _require(path)
        assert sha256_of(path) == expected_hash, f"{name} has changed since the manifest was generated"
