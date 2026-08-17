import pandas as pd

from build_auto_review_dataset import align_past_teacher, build_review_windows


def _windows():
    rows = []
    for i, minute in enumerate(range(3, 9)):
        rows.append(
            {
                "match_id": "m1", "participant_id": 1, "puuid": "p1", "split": "train",
                "role": "TOP", "champion_id": 1, "feature_cutoff_minute": minute,
                "current_p_score": 60 - i * 10, "current_gap_z_vs_lane_opponent": -20,
                "past_death_count": 1, "past_movement": 100, "past_team_participation": 1,
                "baseline_scope": "role", "baseline_fallback_level": 2, "baseline_n": 1000,
                "baseline_match_n": 100, "baseline_player_n": 100,
            }
        )
    return pd.DataFrame(rows)


def _teacher():
    return pd.DataFrame(
        [
            {"match_id": "m1", "participant_id": 1, "feature_cutoff_minute": 3, "timeline_teacher_score": 60.0, "teacher_probability": 0.6, "teacher_baseline_n": 200},
            {"match_id": "m1", "participant_id": 1, "feature_cutoff_minute": 6, "timeline_teacher_score": 20.0, "teacher_probability": 0.2, "teacher_baseline_n": 200},
        ]
    )


def test_past_only_alignment_and_age():
    teacher = _teacher().copy()
    teacher["timeline_teacher_trend_3m"] = [None, -40]
    result = align_past_teacher(_windows(), teacher, max_age_minutes=2)
    row5 = result[result.feature_cutoff_minute == 5].iloc[0]
    row6 = result[result.feature_cutoff_minute == 6].iloc[0]
    assert row5.teacher_checkpoint_minute == 3
    assert row5.teacher_score_age_minutes == 2
    assert row6.teacher_checkpoint_minute == 6
    assert not (result.teacher_checkpoint_minute > result.feature_cutoff_minute).any()


def test_high_confidence_requires_exact_checkpoint_and_non_global_scope():
    review, _ = build_review_windows(_windows(), _teacher(), max_age_minutes=2)
    row6 = review[review.feature_cutoff_minute == 6].iloc[0]
    assert row6.review_state_candidate == "critical_risk"
    assert row6.review_evidence_confidence == "high"
    assert bool(row6.review_gate_high_confidence)


def test_test_score_does_not_change_train_movement_threshold():
    windows = _windows()
    test = windows.iloc[[0]].copy()
    test["match_id"] = "m2"
    test["participant_id"] = 2
    test["puuid"] = "p2"
    test["split"] = "test"
    test["past_movement"] = 999999
    teacher = pd.concat([_teacher(), _teacher().assign(match_id="m2", participant_id=2)], ignore_index=True)
    review, _ = build_review_windows(pd.concat([windows, test], ignore_index=True), teacher)
    train_threshold = review[(review.match_id == "m1") & (review.feature_cutoff_minute == 3)].movement_p25.iloc[0]
    assert train_threshold == 100
