from train_review_gated_forecaster import EXPERIMENTS, should_train


def test_future_labels_are_not_features():
    features = {feature for numeric, categorical in EXPERIMENTS.values() for feature in numeric + categorical}
    assert not any(feature.endswith("_h5") for feature in features)


def test_positive_count_gate():
    import pandas as pd

    ok, reason = should_train(pd.Series([True] * 30 + [False]), pd.Series([True] * 20 + [False]))
    assert ok and reason is None
    ok, reason = should_train(pd.Series([True] * 29 + [False]), pd.Series([True] * 20 + [False]))
    assert not ok and "train positives" in reason
