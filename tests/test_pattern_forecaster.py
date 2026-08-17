import json
from pathlib import Path

import pandas as pd
import pytest

from build_behavior_dataset import ROOT, sha256_of
from train_pattern_forecaster import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    prepare_features,
    should_train,
    split_train_test,
)

MANIFEST_PATH = ROOT / "output_v3/patch=16.14/pattern_forecaster_manifest.json"


def test_should_train_rejects_low_train_positives():
    trained, reason = should_train(n_train_positive=5, n_test_positive=50, min_train_positives=30, min_test_positives=20)
    assert trained is False
    assert "train positives" in reason


def test_should_train_rejects_low_test_positives():
    trained, reason = should_train(n_train_positive=100, n_test_positive=5, min_train_positives=30, min_test_positives=20)
    assert trained is False
    assert "test positives" in reason


def test_should_train_passes_when_both_thresholds_met():
    trained, reason = should_train(n_train_positive=30, n_test_positive=20, min_train_positives=30, min_test_positives=20)
    assert trained is True
    assert reason is None


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "some_label_h5": [True, False, None, True, False],
            "split": ["train", "train", "train", "test", "test"],
        }
    )


def test_split_train_test_drops_null_label_rows_before_splitting():
    train, test = split_train_test(_toy_frame(), "some_label_h5")
    assert len(train) == 2  # the 3rd train row has a NULL label and must be dropped
    assert len(test) == 2
    assert set(train["split"]) == {"train"}
    assert set(test["split"]) == {"test"}


def test_prepare_features_selects_columns_and_casts_categoricals():
    df = pd.DataFrame({col: [0, 1] for col in FEATURE_COLUMNS})
    features = prepare_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    for column in CATEGORICAL_FEATURE_COLUMNS:
        assert isinstance(features[column].dtype, pd.CategoricalDtype)


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} not generated in this environment")


def test_manifest_model_file_hashes_match_actual_files():
    _require(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for label, entry in manifest["labels"].items():
        if not entry["trained"]:
            assert entry["model_file"] is None and entry["model_file_sha256"] is None
            continue
        model_path = ROOT / entry["model_file"]
        _require(model_path)
        assert sha256_of(model_path) == entry["model_file_sha256"], f"{label} model file does not match manifest hash"


def test_manifest_source_dataset_hash_matches_actual_dataset_file():
    _require(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset_path = ROOT / manifest["source_dataset_file"]
    _require(dataset_path)
    assert sha256_of(dataset_path) == manifest["source_dataset_file_sha256"]


def test_skipped_labels_stayed_below_threshold():
    _require(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in manifest["labels"].values():
        if entry["trained"]:
            continue
        below_train = entry["n_train_positive"] < manifest["min_train_positives"]
        below_test = entry["n_test_positive"] < manifest["min_test_positives"]
        assert below_train or below_test, "a label was skipped without actually failing either threshold"
