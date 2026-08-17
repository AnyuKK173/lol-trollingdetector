"""Merge the v4 timeline teacher into Gold-subject behavior windows.

Only the most recent checkpoint at or before a window may be used. The default
maximum age is two minutes because the teacher is scored every three minutes.
The output remains weak supervision for triage/intervention gating; future
``*_h5`` event fields remain the forecasting targets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_timeline_teacher_dataset import sha256_of

WINDOW_KEYS = ["match_id", "participant_id", "feature_cutoff_minute"]
TEACHER_REQUIRED = set(WINDOW_KEYS + ["timeline_teacher_score", "teacher_probability", "teacher_baseline_n"])


def add_exact_trend(frame: pd.DataFrame, value: str, minutes: int, output: str) -> pd.DataFrame:
    previous = frame[["match_id", "participant_id", "feature_cutoff_minute", value]].copy()
    previous["feature_cutoff_minute"] += minutes
    previous = previous.rename(columns={value: f"{value}_previous"})
    result = frame.merge(previous, on=WINDOW_KEYS, how="left", validate="one_to_one")
    result[output] = result[value] - result[f"{value}_previous"]
    return result.drop(columns=f"{value}_previous")


def align_past_teacher(windows: pd.DataFrame, teacher: pd.DataFrame, max_age_minutes: int = 2) -> pd.DataFrame:
    """Past-only groupwise as-of merge; never backfills from a future checkpoint."""
    pieces = []
    teacher_columns = [
        "feature_cutoff_minute",
        "timeline_teacher_score",
        "teacher_probability",
        "teacher_baseline_n",
        "timeline_teacher_trend_3m",
    ]
    teacher_groups = {
        key: group.sort_values("feature_cutoff_minute")
        for key, group in teacher.groupby(["match_id", "participant_id"], sort=False)
    }
    for key, group in windows.groupby(["match_id", "participant_id"], sort=False):
        left = group.sort_values("feature_cutoff_minute").copy()
        right = teacher_groups.get(key)
        if right is None:
            for column in teacher_columns[1:]:
                left[column] = pd.NA
            left["teacher_checkpoint_minute"] = pd.NA
            pieces.append(left)
            continue
        right = right[teacher_columns].rename(columns={"feature_cutoff_minute": "teacher_checkpoint_minute"})
        merged = pd.merge_asof(
            left,
            right,
            left_on="feature_cutoff_minute",
            right_on="teacher_checkpoint_minute",
            direction="backward",
            tolerance=max_age_minutes,
            allow_exact_matches=True,
        )
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    result["teacher_score_age_minutes"] = result["feature_cutoff_minute"] - result["teacher_checkpoint_minute"]
    invalid = result["teacher_score_age_minutes"].dropna().lt(0)
    if invalid.any():
        raise AssertionError("future teacher checkpoint aligned to a current window")
    return result


def classify_review_state(row: pd.Series) -> tuple[str, str, str]:
    def number(name: str, default: float = 0.0) -> float:
        value = row.get(name)
        return default if value is None or pd.isna(value) else float(value)

    evidence: list[str] = []
    p_score = row.get("current_p_score")
    p_trend = row.get("p_score_trend_3m")
    teacher_score = row.get("timeline_teacher_score")
    teacher_trend = row.get("timeline_teacher_trend_3m")
    p_low = pd.notna(p_score) and float(p_score) <= 25
    p_falling = pd.notna(p_trend) and float(p_trend) <= -15
    teacher_low = pd.notna(teacher_score) and float(teacher_score) <= 25
    teacher_falling = pd.notna(teacher_trend) and float(teacher_trend) <= -15
    for condition, label in (
        (p_low, "p_score_bottom_quartile"),
        (p_falling, "p_score_drop_3m"),
        (teacher_low, "teacher_bottom_quartile"),
        (teacher_falling, "teacher_drop_3m"),
    ):
        if condition:
            evidence.append(label)

    teacher_available = pd.notna(teacher_score) and pd.notna(row.get("teacher_baseline_n"))
    scope = row.get("baseline_scope")
    scope_reliable = pd.notna(scope) and scope != "global"
    exact_checkpoint = number("teacher_score_age_minutes", default=-1) == 0
    agreement = (p_low or p_falling) and (teacher_low or teacher_falling)
    recent_risk = number("past_death_count") >= 1 or (
        pd.notna(row.get("current_gap_z_vs_lane_opponent")) and float(row["current_gap_z_vs_lane_opponent"]) <= -15
    )
    if agreement:
        if recent_risk:
            evidence.append("recent_death_or_large_lane_gap")
            state = "critical_risk"
        else:
            state = "struggling"
        confidence = "high" if teacher_available and scope_reliable and exact_checkpoint else "medium"
        return state, confidence, ";".join(evidence)

    movement_p25 = row.get("movement_p25")
    inactive = (
        pd.notna(row.get("past_movement"))
        and pd.notna(movement_p25)
        and float(row["past_movement"]) <= float(movement_p25)
        and number("past_team_participation") == 0
        and number("past_death_count") == 0
    )
    if inactive:
        evidence.append("low_movement_no_participation_no_death")
        return "disengagement_candidate", "low", ";".join(evidence)
    if p_low or p_falling or teacher_low or teacher_falling:
        return "struggling", "medium" if teacher_available else "low", ";".join(evidence)
    if not teacher_available:
        evidence.append("teacher_missing_or_unreliable")
        return "uncertain", "low", ";".join(evidence)
    return "stable", "medium", ";".join(evidence)


def build_review_windows(windows: pd.DataFrame, teacher: pd.DataFrame, max_age_minutes: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = TEACHER_REQUIRED - set(teacher.columns)
    if missing:
        raise ValueError(f"teacher checkpoints missing columns: {sorted(missing)}")
    if windows.duplicated(WINDOW_KEYS).any() or teacher.duplicated(WINDOW_KEYS).any():
        raise ValueError("inputs must contain one row per checkpoint key")
    if windows.groupby("match_id")["split"].nunique().max() > 1:
        raise ValueError("mixed-split match in windows")

    teacher = teacher.copy().sort_values(WINDOW_KEYS)
    teacher = add_exact_trend(teacher, "timeline_teacher_score", 3, "timeline_teacher_trend_3m")
    review = add_exact_trend(windows.copy(), "current_p_score", 3, "p_score_trend_3m")
    review = align_past_teacher(review, teacher, max_age_minutes)

    movement = (
        review[review["split"] == "train"]
        .groupby(["role", "feature_cutoff_minute"])["past_movement"]
        .quantile(0.25)
        .rename("movement_p25")
        .reset_index()
    )
    review = review.merge(movement, on=["role", "feature_cutoff_minute"], how="left", validate="many_to_one")
    states = review.apply(classify_review_state, axis=1, result_type="expand")
    states.columns = ["review_state_candidate", "review_evidence_confidence", "review_evidence"]
    review[states.columns] = states
    review["review_gate_high_confidence"] = review["review_evidence_confidence"].eq("high")
    review = review.sort_values(WINDOW_KEYS).reset_index(drop=True)

    metadata = {
        "dataset_version": "auto_review_windows_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_teacher_age_minutes": max_age_minutes,
        "teacher_alignment": "latest checkpoint at or before current minute; future backfill forbidden",
        "teacher_baseline_fit_scope": "teacher role x minute quantiles fitted on train only by train_timeline_teacher.py",
        "movement_baseline_fit_scope": "train split only; role x minute",
        "state_counts": review["review_state_candidate"].value_counts(dropna=False).to_dict(),
        "confidence_counts": review["review_evidence_confidence"].value_counts(dropna=False).to_dict(),
        "semantic": "weak-supervision review state; never an intent/trolling ground truth",
    }
    return review, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v4 replay-free review windows.")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--teacher-checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-teacher-age-minutes", type=int, default=2)
    args = parser.parse_args()
    windows = pd.read_parquet(args.windows)
    teacher = pd.read_parquet(args.teacher_checkpoints)
    review, manifest = build_review_windows(windows, teacher, args.max_teacher_age_minutes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    review.to_parquet(args.output, index=False, compression="zstd")
    manifest.update(
        {
            "source_windows": str(args.windows),
            "source_windows_sha256": sha256_of(args.windows),
            "teacher_checkpoints": str(args.teacher_checkpoints),
            "teacher_checkpoints_sha256": sha256_of(args.teacher_checkpoints),
            "output_sha256": sha256_of(args.output),
            "window_n": len(review),
        }
    )
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(review):,} review windows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
