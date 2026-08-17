"""Train calibrated, role-specific v4 checkpoint outcome teachers.

The teacher estimates the probability that a player's team eventually wins
from information available at a checkpoint. It is an outcome proxy, not a
skill score, behavior label, or proof of intent.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from build_timeline_teacher_dataset import FEATURE_COLUMNS, KEYS, ROLES, sha256_of

MIN_ROLE_TRAIN_ROWS = 500
MIN_MINUTE_BASELINE_ROWS = 100


def match_bucket(match_id: str, modulus: int = 10) -> int:
    return int(hashlib.sha256(str(match_id).encode("utf-8")).hexdigest(), 16) % modulus


def split_fit_calibration(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve whole matches from the training side for Platt calibration."""
    mask = train["match_id"].astype(str).map(match_bucket).ge(8)
    fit, calibration = train.loc[~mask].copy(), train.loc[mask].copy()
    if fit["final_win"].nunique() < 2 or calibration["final_win"].nunique() < 2:
        raise ValueError("fit/calibration split must contain both outcomes")
    return fit, calibration


def percentile_from_quantiles(value: float, p25: float, p50: float, p75: float) -> float | None:
    if any(pd.isna(x) for x in (value, p25, p50, p75)):
        return None
    value, p25, p50, p75 = map(float, (value, p25, p50, p75))
    if p25 == p50 == p75:
        return 50.0
    if value <= p25:
        width = max(p50 - p25, 1e-6)
        return max(0.0, 25.0 - 25.0 * (p25 - value) / width)
    if value <= p50:
        return 50.0 if p50 == p25 else 25.0 + 25.0 * (value - p25) / (p50 - p25)
    if value <= p75:
        return 75.0 if p75 == p50 else 50.0 + 25.0 * (value - p50) / (p75 - p50)
    width = max(p75 - p50, 1e-6)
    return min(100.0, 75.0 + 25.0 * (value - p75) / width)


def fit_quantile_baseline(train_scores: pd.DataFrame, min_rows: int = MIN_MINUTE_BASELINE_ROWS) -> pd.DataFrame:
    grouped = train_scores.groupby(["role", "feature_cutoff_minute"])["teacher_probability"]
    count = grouped.size().rename("teacher_baseline_n")
    q = grouped.quantile([0.25, 0.50, 0.75]).unstack().rename(columns={0.25: "teacher_p25", 0.5: "teacher_p50", 0.75: "teacher_p75"})
    result = count.to_frame().join(q).reset_index()
    return result[result["teacher_baseline_n"] >= min_rows].copy()


def calibration_table(y_true: pd.Series, probability: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": probability})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True, duplicates="drop")
    rows = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append({
            "bin": str(interval),
            "n": len(group),
            "mean_predicted": round(float(group["p"].mean()), 5),
            "observed_rate": round(float(group["y"].mean()), 5),
        })
    return rows


def safe_metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": len(y_true),
        "positive_n": int(y_true.astype(int).sum()),
        "positive_rate": round(float(y_true.astype(int).mean()), 5),
        "brier": round(float(brier_score_loss(y_true, probability)), 5),
        "log_loss": round(float(log_loss(y_true, probability, labels=[False, True])), 5),
        "calibration_table": calibration_table(y_true, probability),
    }
    if y_true.nunique() == 2:
        result["roc_auc"] = round(float(roc_auc_score(y_true, probability)), 5)
        result["pr_auc"] = round(float(average_precision_score(y_true, probability)), 5)
    else:
        result["roc_auc"] = result["pr_auc"] = None
    return result


def train_role(role_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    train = role_frame[role_frame["split"] == "train"].copy()
    test = role_frame[role_frame["split"] == "test"].copy()
    if len(train) < MIN_ROLE_TRAIN_ROWS or train["final_win"].nunique() < 2:
        raise ValueError(f"insufficient training data: n={len(train)} outcomes={train['final_win'].nunique()}")
    fit, calibration = split_fit_calibration(train)
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
    model.fit(fit[FEATURE_COLUMNS], fit["final_win"].astype(bool))
    calibration_raw = np.clip(model.predict_proba(calibration[FEATURE_COLUMNS])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(np.log(calibration_raw / (1 - calibration_raw)).reshape(-1, 1), calibration["final_win"].astype(bool))

    scored_parts = []
    metrics: dict[str, Any] = {}
    for split_name, part in (("train", train), ("test", test)):
        if part.empty:
            metrics[split_name] = {"n": 0}
            continue
        raw = np.clip(model.predict_proba(part[FEATURE_COLUMNS])[:, 1], 1e-6, 1 - 1e-6)
        calibrated = calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
        scored = part[KEYS + ["puuid", "split", "role"]].copy()
        scored["teacher_raw_probability"] = raw
        scored["teacher_probability"] = calibrated
        scored_parts.append(scored)
        metrics[split_name] = safe_metrics(part["final_win"].astype(bool), calibrated)
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "features": FEATURE_COLUMNS,
        "semantic": "checkpoint outcome proxy; not skill, intent, or trolling truth",
    }
    return bundle, pd.concat(scored_parts, ignore_index=True), metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train role-specific calibrated timeline teachers.")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-minute-baseline-rows", type=int, default=MIN_MINUTE_BASELINE_ROWS)
    args = parser.parse_args()

    data = pd.read_parquet(args.checkpoints)
    missing = set(FEATURE_COLUMNS + KEYS + ["puuid", "split", "role", "final_win"]) - set(data.columns)
    if missing:
        raise ValueError(f"checkpoint dataset missing columns: {sorted(missing)}")
    if "final_win" in FEATURE_COLUMNS:
        raise AssertionError("target leaked into FEATURE_COLUMNS")
    if data.groupby("match_id")["split"].nunique().max() > 1:
        raise ValueError("mixed-split match in checkpoint dataset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_all = []
    role_manifest: dict[str, Any] = {}
    for role in ROLES:
        role_frame = data[data["role"] == role].copy()
        if role_frame.empty:
            role_manifest[role] = {"trained": False, "reason": "no rows"}
            continue
        try:
            bundle, scored, metrics = train_role(role_frame)
        except ValueError as exc:
            role_manifest[role] = {"trained": False, "reason": str(exc)}
            continue
        model_path = args.output_dir / f"{role.lower()}_timeline_teacher_v2.joblib"
        joblib.dump(bundle, model_path)
        role_manifest[role] = {
            "trained": True,
            "model_file": model_path.name,
            "model_sha256": sha256_of(model_path),
            "metrics": metrics,
        }
        scored_all.append(scored)

    if not scored_all:
        raise RuntimeError("no role teacher could be trained")
    scored = pd.concat(scored_all, ignore_index=True)
    train_scores = scored[scored["split"] == "train"]
    baseline = fit_quantile_baseline(train_scores, args.min_minute_baseline_rows)
    scored = scored.merge(baseline, on=["role", "feature_cutoff_minute"], how="left", validate="many_to_one")
    scored["timeline_teacher_score"] = [
        percentile_from_quantiles(*values)
        for values in scored[["teacher_probability", "teacher_p25", "teacher_p50", "teacher_p75"]].itertuples(index=False, name=None)
    ]
    scored = scored.sort_values(KEYS).reset_index(drop=True)

    checkpoints_path = args.output_dir / "timeline_teacher_checkpoints_v2.parquet"
    baseline_path = args.output_dir / "timeline_teacher_quantiles_train_v2.parquet"
    scored.to_parquet(checkpoints_path, index=False, compression="zstd")
    baseline.to_parquet(baseline_path, index=False, compression="zstd")
    manifest = {
        "model_version": "timeline_teacher_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.checkpoints),
        "input_sha256": sha256_of(args.checkpoints),
        "features": FEATURE_COLUMNS,
        "target": "final_win",
        "target_excluded_from_features": "final_win" not in FEATURE_COLUMNS,
        "calibration": "Platt/logistic calibration on whole-match holdout from the train split",
        "baseline_fit_scope": "train split only; role x cutoff minute",
        "min_minute_baseline_rows": args.min_minute_baseline_rows,
        "roles": role_manifest,
        "checkpoints_file": checkpoints_path.name,
        "checkpoints_sha256": sha256_of(checkpoints_path),
        "baseline_file": baseline_path.name,
        "baseline_sha256": sha256_of(baseline_path),
        "semantic": "checkpoint outcome proxy; not skill, intent, or trolling truth",
    }
    (args.output_dir / "timeline_teacher_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(scored):,} scored checkpoints to {checkpoints_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
