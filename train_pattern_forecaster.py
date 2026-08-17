"""Trains one independent binary classifier per weak label produced by
build_behavior_dataset.py (the *_h5 columns), instead of a single mutually
exclusive state classifier — the labels can co-occur in the same window
(e.g. death_risk_h5 and resource_collapse_h5 both firing at once), so a
single multi-class head would force an artificial tie-break between them.

Rules on top of the leakage prevention build_behavior_dataset.py already
does on the dataset itself:

1. The train/test split is never recomputed here — it reuses the dataset's
   own `split` column, which is already match/puuid-isolated.
2. Every OTHER *_h5 label is excluded from the feature set for a given
   label's model. They are all derived from the same (t, t+5] future
   window as the target, so using one as a feature for another would leak
   future information no production caller would have at prediction time.
3. A label is only trained once it clears MIN_TRAIN_POSITIVES/MIN_TEST_POSITIVES
   positive examples on both sides of the split — the same "insufficient
   samples means no curve" stance build_baselines.py takes for champion-
   level quantiles. Below that, roc_auc/pr_auc are noise, not a usable
   reliability estimate.

Output is one *.joblib model per trained label under
output_v3/patch={patch}/models/, plus pattern_forecaster_manifest.json
documenting what was trained, what was skipped and why, and per-label
metrics with SHA-256 provenance for both the source dataset and the
resulting model files.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from build_behavior_dataset import ROOT, sha256_of

NUMERIC_FEATURE_COLUMNS = [
    "feature_cutoff_minute",
    "current_p_score",
    "current_gap_z_vs_lane_opponent",
    "past_gold_rate",
    "past_xp_rate",
    "past_cs_rate",
    "past_movement",
    "past_team_participation",
    "past_death_count",
    "teammate_avg_gold_now",
    "baseline_fallback_level",
    "baseline_n",
    "baseline_match_n",
    "baseline_player_n",
]
# champion_id is nominal (an ID), not ordinal — cast to category, not left
# as int, or the model would treat champion 266 as "greater than" champion 1.
CATEGORICAL_FEATURE_COLUMNS = ["role", "champion_id", "baseline_scope"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

MIN_TRAIN_POSITIVES = 30
MIN_TEST_POSITIVES = 20
PERMUTATION_IMPORTANCE_SAMPLE = 3000
TOP_FEATURES_KEPT = 10


# ---------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_pattern_forecaster.py)
# ---------------------------------------------------------------------------


def should_train(
    n_train_positive: int,
    n_test_positive: int,
    min_train_positives: int = MIN_TRAIN_POSITIVES,
    min_test_positives: int = MIN_TEST_POSITIVES,
) -> tuple[bool, str | None]:
    """Gates on absolute positive counts, not rates — a label at 20%
    positive with only 200 rows is just as unreliable as one at 0.1%
    positive with 50,000 rows. Both sides must clear independently."""
    if n_train_positive < min_train_positives:
        return False, f"train positives {n_train_positive} < minimum {min_train_positives}"
    if n_test_positive < min_test_positives:
        return False, f"test positives {n_test_positive} < minimum {min_test_positives}"
    return True, None


def split_train_test(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drops rows where this label is NULL (scope-inconsistent trajectory,
    no lane opponent matched, etc.) before splitting on the dataset's own
    `split` column."""
    labeled = df[df[label].notna()]
    train = labeled[labeled["split"] == "train"]
    test = labeled[labeled["split"] == "test"]
    return train, test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects the model's feature columns and casts the nominal ones to
    pandas' category dtype, which HistGradientBoostingClassifier's
    categorical_features="from_dtype" expects. Numeric columns keep their
    native NaNs — the model splits on "is missing" directly, which matters
    here since several features (e.g. current_gap_z_vs_lane_opponent) are
    NULL for a specific reason (no lane opponent, scope-inconsistency gate)
    rather than missing at random, so imputing a value would erase that
    signal."""
    features = df[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURE_COLUMNS:
        features[column] = features[column].astype("category")
    return features


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        class_weight="balanced",
        max_iter=300,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    # Windows consoles default to cp1252, which can't encode the full-width
    # "：" used below — force UTF-8 so the summary print doesn't crash.
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    target_patch = os.getenv("TARGET_PATCH", "").strip()
    if not target_patch:
        print("缺少 TARGET_PATCH。", file=sys.stderr)
        return 1

    out_dir = ROOT / f"output_v3/patch={target_patch}"
    dataset_path = out_dir / "behavior_windows_v3.parquet"
    dataset_manifest_path = out_dir / "MANIFEST.json"
    if not dataset_path.exists() or not dataset_manifest_path.exists():
        print(
            f"找不到 {dataset_path} 或 {dataset_manifest_path}——先运行 build_behavior_dataset.py。",
            file=sys.stderr,
        )
        return 1

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    actual_dataset_hash = sha256_of(dataset_path)
    if actual_dataset_hash != dataset_manifest.get("dataset_file_sha256"):
        print(
            "behavior_windows_v3.parquet 的哈希与 MANIFEST.json 记录的不一致——"
            "数据集可能在 build_behavior_dataset.py 重新生成后没有同步更新，"
            "先重新运行 build_behavior_dataset.py 再训练。",
            file=sys.stderr,
        )
        return 2

    df = pd.read_parquet(dataset_path)
    label_columns = sorted(c for c in df.columns if c.endswith("_h5"))

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for label in label_columns:
        train_df, test_df = split_train_test(df, label)
        n_train, n_test = len(train_df), len(test_df)
        n_train_positive = int(train_df[label].sum()) if n_train else 0
        n_test_positive = int(test_df[label].sum()) if n_test else 0
        test_positive_rate_pct = round(100 * n_test_positive / n_test, 3) if n_test else None

        entry: dict[str, Any] = {
            "n_train": n_train,
            "n_train_positive": n_train_positive,
            "n_test": n_test,
            "n_test_positive": n_test_positive,
            "test_positive_rate_pct": test_positive_rate_pct,
        }

        trained, skip_reason = should_train(n_train_positive, n_test_positive)
        if not trained:
            entry.update(
                trained=False,
                skip_reason=skip_reason,
                roc_auc=None,
                pr_auc=None,
                model_file=None,
                model_file_sha256=None,
                top_features_by_permutation_importance=None,
            )
            results[label] = entry
            print(f"[skip]    {label}: {skip_reason}")
            continue

        X_train = prepare_features(train_df)
        y_train = train_df[label].astype(bool)
        X_test = prepare_features(test_df)
        y_test = test_df[label].astype(bool)

        model = build_model()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, proba))
        pr_auc = float(average_precision_score(y_test, proba))

        sample_n = min(len(X_test), PERMUTATION_IMPORTANCE_SAMPLE)
        sample_index = X_test.sample(n=sample_n, random_state=42).index
        importance = permutation_importance(
            model,
            X_test.loc[sample_index],
            y_test.loc[sample_index],
            scoring="average_precision",
            n_repeats=5,
            random_state=42,
        )
        top_features = sorted(
            zip(FEATURE_COLUMNS, importance.importances_mean), key=lambda pair: pair[1], reverse=True
        )[:TOP_FEATURES_KEPT]

        model_path = models_dir / f"{label}.joblib"
        joblib.dump(model, model_path)

        entry.update(
            trained=True,
            skip_reason=None,
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            # average_precision's no-skill baseline is the positive rate
            # itself (unlike ROC-AUC's fixed 0.5) — recorded alongside
            # pr_auc so it's clear what "good" means at this label's rate.
            pr_auc_no_skill_baseline=round((y_test.mean()), 4),
            model_file=str(model_path.relative_to(ROOT)).replace("\\", "/"),
            model_file_sha256=sha256_of(model_path),
            top_features_by_permutation_importance=[
                {"feature": feature, "importance_mean": round(float(value), 5)}
                for feature, value in top_features
            ],
        )
        results[label] = entry
        print(
            f"[trained] {label}: roc_auc={roc_auc:.3f} pr_auc={pr_auc:.3f} "
            f"(test positive rate {test_positive_rate_pct}%, n_test={n_test:,})"
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "git_note": "project directory is not a git repository; no commit hash available",
        "patch": target_patch,
        "source_dataset_file": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
        "source_dataset_file_sha256": actual_dataset_hash,
        "source_dataset_version": dataset_manifest.get("dataset_version"),
        "feature_columns": FEATURE_COLUMNS,
        "model_type": "sklearn.ensemble.HistGradientBoostingClassifier(categorical_features='from_dtype', class_weight='balanced', max_iter=300)",
        "min_train_positives": MIN_TRAIN_POSITIVES,
        "min_test_positives": MIN_TEST_POSITIVES,
        "permutation_importance_sample_n": PERMUTATION_IMPORTANCE_SAMPLE,
        "labels": results,
        "source_files_sha256": {
            name: sha256_of(ROOT / name)
            for name in ("train_pattern_forecaster.py", "build_behavior_dataset.py", "p_score.py")
        },
        "known_limitations": [
            "Skipped labels are not merely 'lower priority' — below MIN_TRAIN_POSITIVES/MIN_TEST_POSITIVES, an AUC would be estimated from too few positive examples to mean anything; more collection (not a lowered threshold) is the fix.",
            "Feature importance is via permutation_importance on a capped test subsample (see permutation_importance_sample_n), scored with average_precision — it reflects what the model currently leans on, not a causal claim about what drives the underlying behavior.",
            "class_weight='balanced' reweights the loss for imbalance but does not change what counts as a positive; pr_auc_no_skill_baseline is the number to compare pr_auc against, not 0.5.",
        ],
    }
    manifest_path = out_dir / "pattern_forecaster_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest：{manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
