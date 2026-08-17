from build_behavior_dataset import (
    ENEMY_GAP_EXPAND_THRESHOLD,
    assign_split,
    find_mixed_split_matches,
    label_activity_drop,
    label_death_risk,
    label_enemy_gap_expand,
    label_multi_death,
    label_performance_collapse,
    label_recovery,
    label_resource_collapse,
    label_worthless_death,
)


def test_assign_split_is_deterministic_and_stable_per_puuid():
    puuid = "abc-123"
    assert assign_split(puuid) == assign_split(puuid)


def test_assign_split_roughly_respects_train_fraction():
    puuids = [f"player-{i}" for i in range(500)]
    train_count = sum(1 for p in puuids if assign_split(p, train_fraction=0.8) == "train")
    # Hash-based split won't be exact, but should land in a sane band.
    assert 350 <= train_count <= 450


def test_find_mixed_split_matches_flags_matches_with_both_sides():
    match_ids = ["m1", "m1", "m2", "m3", "m3"]
    splits = ["train", "test", "train", "test", "test"]
    assert find_mixed_split_matches(match_ids, splits) == {"m1"}


def test_find_mixed_split_matches_empty_when_all_clean():
    match_ids = ["m1", "m1", "m2"]
    splits = ["train", "train", "test"]
    assert find_mixed_split_matches(match_ids, splits) == set()


def test_performance_collapse_requires_high_start_and_low_end():
    assert label_performance_collapse(60, [55, 20, 15]) is True
    assert label_performance_collapse(60, [55, 50, 45]) is False
    assert label_performance_collapse(40, [10, 5]) is False  # precondition (>=50) not met
    assert label_performance_collapse(None, [10, 5]) is None
    assert label_performance_collapse(60, []) is None


def test_resource_collapse_needs_two_metrics_below_and_declining():
    rate_p25 = {"total_gold": 50, "xp": 40, "minions_killed": 1}
    past = {"total_gold": 60, "xp": 45, "minions_killed": 1.2}
    future_bad = {"total_gold": 30, "xp": 20, "minions_killed": 1.3}
    future_ok = {"total_gold": 60, "xp": 45, "minions_killed": 1.2}
    assert label_resource_collapse(past, future_bad, rate_p25) is True
    assert label_resource_collapse(past, future_ok, rate_p25) is False


def test_resource_collapse_none_when_no_baseline_available():
    assert label_resource_collapse({"total_gold": 1}, {"total_gold": 1}, {}) is None


def test_enemy_gap_expand_thresholds_at_configured_percentile_points():
    assert label_enemy_gap_expand(0.0, ENEMY_GAP_EXPAND_THRESHOLD) is True
    assert label_enemy_gap_expand(0.0, ENEMY_GAP_EXPAND_THRESHOLD + 0.1) is False
    assert label_enemy_gap_expand(None, ENEMY_GAP_EXPAND_THRESHOLD) is None


def test_death_risk_is_simple_count_check():
    assert label_death_risk(0) is False
    assert label_death_risk(1) is True


def test_multi_death_requires_at_least_two():
    assert label_multi_death(0) is False
    assert label_multi_death(1) is False
    assert label_multi_death(2) is True


def test_worthless_death_true_when_no_team_payoff_nearby():
    deaths = [{"timestamp_ms": 100_000, "team_id": 100, "position_x": 1000, "position_y": 1000}]
    events_with_payoff = [
        {
            "timestamp_ms": 110_000,
            "event_type": "CHAMPION_KILL",
            "team_id": 100,
            "position_x": 1200,
            "position_y": 1000,
        }
    ]
    events_without_payoff = [
        {
            "timestamp_ms": 300_000,  # outside the +-30s time window
            "event_type": "CHAMPION_KILL",
            "team_id": 100,
            "position_x": 1200,
            "position_y": 1000,
        }
    ]
    assert label_worthless_death(deaths, events_with_payoff) is False
    assert label_worthless_death(deaths, events_without_payoff) is True
    assert label_worthless_death([], events_without_payoff) is False


def test_worthless_death_missing_coordinates_never_default_to_a_payoff():
    # Death has no recorded position -> spatial proximity can never be
    # confirmed, so a same-team/same-time kill must NOT count as a payoff.
    death_missing_position = [{"timestamp_ms": 100_000, "team_id": 100}]
    candidate_event = [
        {
            "timestamp_ms": 105_000,
            "event_type": "CHAMPION_KILL",
            "team_id": 100,
            "position_x": 1000,
            "position_y": 1000,
        }
    ]
    assert label_worthless_death(death_missing_position, candidate_event) is True

    # Same thing if the death has a position but the candidate event doesn't.
    death_with_position = [{"timestamp_ms": 100_000, "team_id": 100, "position_x": 1000, "position_y": 1000}]
    event_missing_position = [{"timestamp_ms": 105_000, "event_type": "CHAMPION_KILL", "team_id": 100}]
    assert label_worthless_death(death_with_position, event_missing_position) is True


def test_worthless_death_ignores_payoff_that_happened_elsewhere_on_the_map():
    deaths = [{"timestamp_ms": 100_000, "team_id": 100, "position_x": 1000, "position_y": 1000}]
    far_away_payoff = [
        {
            "timestamp_ms": 105_000,
            "event_type": "CHAMPION_KILL",
            "team_id": 100,
            "position_x": 14000,
            "position_y": 14000,
        }
    ]
    nearby_payoff = [
        {
            "timestamp_ms": 105_000,
            "event_type": "CHAMPION_KILL",
            "team_id": 100,
            "position_x": 1500,
            "position_y": 1000,
        }
    ]
    assert label_worthless_death(deaths, far_away_payoff) is True
    assert label_worthless_death(deaths, nearby_payoff) is False


def test_recovery_requires_sustained_rise_and_resource_recovery():
    assert label_recovery(15, [20, 55, 60], True) is True
    assert label_recovery(15, [20, 55, 60], False) is False
    assert label_recovery(30, [55, 60], True) is False  # precondition (<25) not met
    assert label_recovery(15, [20], True) is None  # not enough future points


def test_activity_drop_excludes_windows_with_death_or_purchase():
    past = {"movement": 100, "resource_rate": 50, "team_participation": 3}
    future_all_down = {"movement": 10, "resource_rate": 5, "team_participation": 0}
    assert label_activity_drop(True, past, future_all_down) is None
    assert label_activity_drop(False, past, future_all_down) is True
    future_mixed = {"movement": 10, "resource_rate": 50, "team_participation": 0}
    assert label_activity_drop(False, past, future_mixed) is False
