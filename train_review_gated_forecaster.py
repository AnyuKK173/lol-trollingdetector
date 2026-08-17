"""Run A/B/C ablations and train calibrated event-risk forecasters.

A = original v3 current-state features
B = A + timeline teacher value/trend/age
C = B + replay-free review state/confidence gate

Each future ``*_h5`` label is an independent binary target. No future label is
ever included as an input to another head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from build_timeline_teacher_dataset import sha256_of

BASE_NUMERIC = [
    "feature_cutoff_minute", "current_p_score", "current_gap_z_vs_lane_opponent",
    "past_gold_rate", "past_xp_rate", "past_cs_rate", "past_movement",
    "past_team_participation", "past_death_count", "teammate_avg_gold_now",
    "baseline_fallback_level", "baseline_n", "baseline_match_n", "baseline_player_n",
]
BASE_CATEGORICAL = ["role", "champion_id", "baseline_scope"]
TEACHER_NUMERIC = [
    "teacher_probability", "timeline_teacher_score", "timeline_teacher_trend_3m",
    "teacher_score_age_minutes", "teacher_baseline_n", "p_score_trend_3m",
]
GATE_NUMERIC = ["review_gate_high_confidence"]
GATE_CATEGORICAL = ["review_state_candidate", "review_evidence_confidence"]
EXPERIMENTS = {
    "A_v3": (BASE_NUMERIC, BASE_CATEGORICAL),
    "B_teacher": (BASE_NUMERIC + TEACHER_NUMERIC, BASE_CATEGORICAL),
    "C_teacher_review": (BASE_NUMERIC + TEACHER_NUMERIC + GATE_NUMERIC, BASE_CATEGORICAL + GATE_CATEGORICAL),
}
MIN_TRAIN_POSITIVES = 30
MIN_TEST_POSITIVES = 20


def _match_bucket(value: str, modulus: int = 10) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest(), 16) % modulus


def build_estimator(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    categorical_mask = [False] * (len(numeric) * 2) + [True] * len(categorical)
    # SimpleImputer only adds indicators for columns that are missing during
    # fit, so a fixed mask cannot safely account for them. Keep the estimator
    # numeric; ordinal category IDs are merely stable codes, not an ordering
    # claim, and the A/B/C comparison uses the same encoding throughout.
    _ = categorical_mask
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def should_train(train_y: pd.Series, test_y: pd.Series) -> tuple[bool, str | None]:
    train_positive, test_positive = int(train_y.sum()), int(test_y.sum())
    if train_positive < MIN_TRAIN_POSITIVES:
        return False, f"train positives {train_positive} < {MIN_TRAIN_POSITIVES}"
    if test_positive < MIN_TEST_POSITIVES:
        return False, f"test positives {test_positive} < {MIN_TEST_POSITIVES}"
    if train_y.nunique() < 2 or test_y.nunique() < 2:
        return False, "both train and test need positive and negative examples"
    return True, None


def _calibration_table(y: pd.Series, probability: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": y.astype(int).to_numpy(), "p": probability})
    frame["bin"] = pd.cut(frame["p"], np.linspace(0, 1, 11), include_lowest=True)
    return [
        {
            "bin": str(interval),
            "n": len(group),
            "mean_predicted": round(float(group["p"].mean()), 5),
            "observed_rate": round(float(group["y"].mean()), 5),
        }
        for interval, group in frame.groupby("bin", observed=True)
    ]


def evaluate(y: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    return {
        "n": len(y),
        "positive_n": int(y.sum()),
        "positive_rate": round(float(y.mean()), 5),
        "roc_auc": round(float(roc_auc_score(y, probability)), 5),
        "pr_auc": round(float(average_precision_score(y, probability)), 5),
        "pr_auc_no_skill": round(float(y.mean()), 5),
        "brier": round(float(brier_score_loss(y, probability)), 5),
        "log_loss": round(float(log_loss(y, probability)), 5),
        "calibration_table": _calibration_table(y, probability),
    }


def grouped_cv(frame: pd.DataFrame, label: str, numeric: list[str], categorical: list[str], folds: int = 3) -> dict[str, Any]:
    groups = frame["match_id"]
    n_splits = min(folds, groups.nunique())
    if n_splits < 2:
        return {"folds": 0, "reason": "fewer than two train matches"}
    fold_metrics = []
    for train_index, valid_index in GroupKFold(n_splits=n_splits).split(frame, frame[label], groups):
        train_fold, valid_fold = frame.iloc[train_index], frame.iloc[valid_index]
        if train_fold[label].nunique() < 2 or valid_fold[label].nunique() < 2:
            continue
        estimator = build_estimator(numeric, categorical)
        estimator.fit(train_fold[numeric + categorical], train_fold[label].astype(bool))
        probability = estimator.predict_proba(valid_fold[numeric + categorical])[:, 1]
        fold_metrics.append(
            {
                "roc_auc": float(roc_auc_score(valid_fold[label], probability)),
                "pr_auc": float(average_precision_score(valid_fold[label], probability)),
            }
        )
    if not fold_metrics:
        return {"folds": 0, "reason": "no fold had both classes"}
    return {
        "folds": len(fold_metrics),
        "roc_auc_mean": round(float(np.mean([x["roc_auc"] for x in fold_metrics])), 5),
        "roc_auc_std": round(float(np.std([x["roc_auc"] for x in fold_metrics])), 5),
        "pr_auc_mean": round(float(np.mean([x["pr_auc"] for x in fold_metrics])), 5),
        "pr_auc_std": round(float(np.std([x["pr_auc"] for x in fold_metrics])), 5),
    }


def fit_calibrated(
    train: pd.DataFrame, label: str, numeric: list[str], categorical: list[str]
) -> dict[str, Any]:
    calibration_mask = train["match_id"].astype(str).map(_match_bucket).ge(8)
    fit, calibration = train.loc[~calibration_mask], train.loc[calibration_mask]
    if fit[label].nunique() < 2 or calibration[label].nunique() < 2:
        raise ValueError("whole-match fit/calibration split lacks both classes")
    estimator = build_estimator(numeric, categorical)
    columns = numeric + categorical
    estimator.fit(fit[columns], fit[label].astype(bool))
    raw = np.clip(estimator.predict_proba(calibration[columns])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(np.log(raw / (1 - raw)).reshape(-1, 1), calibration[label].astype(bool))
    return {"estimator": estimator, "calibrator": calibrator, "features": columns, "label": label}


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    raw = np.clip(bundle["estimator"].predict_proba(frame[bundle["features"]])[:, 1], 1e-6, 1 - 1e-6)
    return bundle["calibrator"].predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train v4 A/B/C review-gated forecasters.")
    parser.add_argument("--review-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    data = pd.read_parquet(args.review_windows)
    labels = sorted(column for column in data.columns if column.endswith("_h5"))
    if not labels:
        raise ValueError("no *_h5 targets found")
    if data.groupby("match_id")["split"].nunique().max() > 1:
        raise ValueError("mixed-split match in review dataset")
    all_feature_names = {name for numeric, categorical in EXPERIMENTS.values() for name in numeric + categorical}
    missing = all_feature_names - set(data.columns)
    if missing:
        raise ValueError(f"review dataset missing feature columns: {sorted(missing)}")
    if any(label in all_feature_names for label in labels):
        raise AssertionError("future target leaked into feature contract")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_results: dict[str, Any] = {}
    for label in labels:
        labeled = data[data[label].notna()].copy()
        train = labeled[labeled["split"] == "train"].copy()
        test = labeled[labeled["split"] == "test"].copy()
        trained, reason = should_train(train[label].astype(bool), test[label].astype(bool))
        manifest_results[label] = {
            "n_train": len(train), "n_train_positive": int(train[label].sum()),
            "n_test": len(test), "n_test_positive": int(test[label].sum()),
            "experiments": {},
        }
        if not trained:
            manifest_results[label]["skip_reason"] = reason
            print(f"[skip] {label}: {reason}")
            continue
        for experiment, (numeric, categorical) in EXPERIMENTS.items():
            entry: dict[str, Any] = {
                "features": numeric + categorical,
                "grouped_cv": grouped_cv(train, label, numeric, categorical, args.cv_folds),
            }
            try:
                bundle = fit_calibrated(train, label, numeric, categorical)
            except ValueError as exc:
                entry.update(trained=False, reason=str(exc))
                manifest_results[label]["experiments"][experiment] = entry
                continue
            probability = predict_bundle(bundle, test)
            model_path = args.output_dir / f"{label}__{experiment}.joblib"
            joblib.dump(bundle, model_path)
            entry.update(
                trained=True,
                model_file=model_path.name,
                model_sha256=sha256_of(model_path),
                test=evaluate(test[label].astype(bool), probability),
            )
            if experiment == "C_teacher_review":
                gate = test["review_gate_high_confidence"].fillna(False).astype(bool)
                if gate.any() and test.loc[gate, label].nunique() == 2:
                    entry["high_confidence_gate_test"] = evaluate(test.loc[gate, label].astype(bool), probability[gate.to_numpy()])
                else:
                    entry["high_confidence_gate_test"] = {"n": int(gate.sum()), "reason": "gate subset lacks both classes"}
            manifest_results[label]["experiments"][experiment] = entry
            print(f"[trained] {label} {experiment}: PR-AUC={entry['test']['pr_auc']:.3f}")

    manifest = {
        "release": "v4_teacher_review",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.review_windows),
        "source_sha256": sha256_of(args.review_windows),
        "split_contract": "reuse frozen v3 match split; GroupKFold groups by match_id within train",
        "calibration": "Platt/logistic calibration on whole-match holdout within train",
        "experiments": {name: {"numeric": n, "categorical": c} for name, (n, c) in EXPERIMENTS.items()},
        "labels": manifest_results,
        "semantic": "future event-risk models; review state is weak supervision, not trolling truth",
    }
    manifest_path = args.output_dir / "review_gated_forecaster_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
