import pandas as pd

from train_review_gated_forecaster import BASE_CATEGORICAL, BASE_NUMERIC, EXPERIMENTS, build_estimator, prepare_features, should_train


def test_future_labels_are_not_features():
    features = {feature for numeric, categorical in EXPERIMENTS.values() for feature in numeric + categorical}
    assert not any(feature.endswith("_h5") for feature in features)


def test_positive_count_gate():
    ok, reason = should_train(pd.Series([True] * 30 + [False]), pd.Series([True] * 20 + [False]))
    assert ok and reason is None
    ok, reason = should_train(pd.Series([True] * 29 + [False]), pd.Series([True] * 20 + [False]))
    assert not ok and "train positives" in reason


def test_prepare_features_casts_categoricals_and_leaves_numeric_alone():
    frame = pd.DataFrame({column: [0, 1] for column in BASE_NUMERIC + BASE_CATEGORICAL})
    features = prepare_features(frame, BASE_NUMERIC, BASE_CATEGORICAL)
    assert list(features.columns) == BASE_NUMERIC + BASE_CATEGORICAL
    for column in BASE_CATEGORICAL:
        assert isinstance(features[column].dtype, pd.CategoricalDtype)
    for column in BASE_NUMERIC:
        assert not isinstance(features[column].dtype, pd.CategoricalDtype)


def test_build_estimator_declares_categorical_features_from_dtype():
    # Regression test for the v4 review-fix: role/champion_id/baseline_scope
    # must be split on as categories, not ordinal-encoded integers treated
    # as a numeric scale.
    estimator = build_estimator()
    assert estimator.categorical_features == "from_dtype"


def test_estimator_fits_on_categorical_dtype_features():
    n = 60  # large enough for HistGradientBoostingClassifier's internal early-stopping split
    frame = pd.DataFrame(
        {
            "feature_cutoff_minute": [3 + (i % 10) for i in range(n)],
            "role": (["TOP", "JUNGLE", "MIDDLE", "BOTTOM"] * ((n // 4) + 1))[:n],
            "champion_id": [i % 8 for i in range(n)],
        }
    )
    label = pd.Series([i % 2 == 0 for i in range(n)])
    features = prepare_features(frame, ["feature_cutoff_minute"], ["role", "champion_id"])
    estimator = build_estimator()
    estimator.fit(features, label)
    probability = estimator.predict_proba(features)
    assert probability.shape == (n, 2)
