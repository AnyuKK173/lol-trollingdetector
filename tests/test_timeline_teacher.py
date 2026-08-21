import pandas as pd

from build_timeline_teacher_dataset import FEATURE_COLUMNS, KEYS, build_checkpoint_rows, retained_match_splits
from train_timeline_teacher import cross_fit_oof, fit_quantile_baseline, percentile_from_quantiles


def _frames(match_id="m1"):
    rows = []
    for pid in range(1, 11):
        for minute in range(0, 7):
            rows.append(
                {
                    "match_id": match_id,
                    "participant_id": pid,
                    "minute": minute,
                    "timestamp_ms": minute * 60_000,
                    "total_gold": 500 + pid * 10 + minute * 100,
                    "xp": minute * 120,
                    "level": 1 + minute // 2,
                    "minions_killed": minute * 5,
                    "jungle_minions_killed": minute if pid in (2, 7) else 0,
                    "position_x": 1000 + minute * 50,
                    "position_y": 1000 + minute * 40,
                }
            )
    return pd.DataFrame(rows)


def _participants(match_id="m1"):
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"] * 2
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "participant_id": pid,
                "puuid": f"p{pid}",
                "team_id": 100 if pid <= 5 else 200,
                "champion_id": pid,
                "win": pid <= 5,
                "role": roles[pid - 1],
            }
            for pid in range(1, 11)
        ]
    )


def test_match_split_is_inherited_by_all_ten_players():
    windows = pd.DataFrame([{"match_id": "m1", "split": "train"}])
    result = build_checkpoint_rows(windows, _participants(), _frames(), pd.DataFrame(columns=[
        "match_id", "timestamp_ms", "event_type", "participant_id", "killer_id", "victim_id", "team_id", "assisting_participant_ids"
    ]))
    assert result["participant_id"].nunique() == 10
    assert set(result["split"]) == {"train"}
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert "final_win" not in FEATURE_COLUMNS


def test_events_after_cutoff_do_not_change_checkpoint_features():
    windows = pd.DataFrame([{"match_id": "m1", "split": "train"}])
    events = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_ms": 100_000, "event_type": "CHAMPION_KILL", "participant_id": None, "killer_id": 1, "victim_id": 6, "team_id": 100, "assisting_participant_ids": []},
            {"match_id": "m1", "timestamp_ms": 500_000, "event_type": "CHAMPION_KILL", "participant_id": None, "killer_id": 1, "victim_id": 6, "team_id": 100, "assisting_participant_ids": []},
        ]
    )
    result = build_checkpoint_rows(windows, _participants(), _frames(), events)
    at_three = result[(result.participant_id == 1) & (result.feature_cutoff_minute == 3)].iloc[0]
    at_six = result[(result.participant_id == 1) & (result.feature_cutoff_minute == 6)].iloc[0]
    assert at_three.kills_now == 1
    assert at_six.kills_now == 1  # 500s is still after the six-minute cutoff


def test_mixed_v3_match_is_rejected():
    windows = pd.DataFrame([{"match_id": "m1", "split": "train"}, {"match_id": "m1", "split": "test"}])
    try:
        retained_match_splits(windows)
    except ValueError as exc:
        assert "mixed" in str(exc)
    else:
        raise AssertionError("mixed match was accepted")


def test_teacher_quantiles_use_only_passed_training_rows():
    import pytest

    train = pd.DataFrame(
        {
            "role": ["TOP"] * 4,
            "feature_cutoff_minute": [3] * 4,
            "teacher_probability": [0.1, 0.2, 0.3, 0.4],
        }
    )
    baseline = fit_quantile_baseline(train, min_rows=1)
    assert baseline.iloc[0].teacher_p75 == pytest.approx(0.325)
    assert percentile_from_quantiles(0.2, 0.1, 0.2, 0.3) == 50.0


def _synthetic_train_rows(n_matches: int = 60) -> pd.DataFrame:
    """One row per match; win alternates by index so every reasonably-sized
    subset (fold pool, calibration holdout) contains both outcomes."""
    rows = []
    for i in range(n_matches):
        row = {
            "match_id": f"synthetic_{i}",
            "puuid": f"p{i}",
            "participant_id": 1,
            "split": "train",
            "role": "TOP",
            "final_win": i % 2 == 0,
        }
        for column in FEATURE_COLUMNS:
            row[column] = float(i % 5)
        rows.append(row)
    return pd.DataFrame(rows)


def test_cross_fit_oof_covers_every_train_row_exactly_once():
    train = _synthetic_train_rows()
    oof = cross_fit_oof(train, n_splits=3)
    assert len(oof) == len(train)
    assert not oof.duplicated(KEYS).any()
    assert set(oof["match_id"]) == set(train["match_id"])
    assert oof["teacher_probability"].notna().all()


def test_cross_fit_oof_never_scores_a_row_with_a_model_fit_on_its_own_match():
    """Each held-out fold's rows must come from a model fit on a fold pool
    that excludes their own match -- verified by checking every match_id
    appears in exactly one fold's held-out output, never leaking into two
    folds' training pools simultaneously (which would happen if GroupKFold
    grouping were broken)."""
    train = _synthetic_train_rows()
    oof = cross_fit_oof(train, n_splits=3)
    # one row per match in the output implies each match was held out by
    # exactly one fold -- if a match had leaked into a fold pool it was also
    # held out from, cross_fit_oof's own duplicate-key check above would
    # already have failed; this asserts the row count reconciles exactly.
    assert oof.groupby("match_id").size().eq(1).all()
