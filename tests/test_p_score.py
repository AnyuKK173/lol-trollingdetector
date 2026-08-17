import pandas as pd

from p_score import choose_baseline, piecewise_percentile, score_observation


def test_piecewise_percentile_hits_anchor_points():
    assert piecewise_percentile(100, 100, 200, 300) == 25
    assert piecewise_percentile(200, 100, 200, 300) == 50
    assert piecewise_percentile(300, 100, 200, 300) == 75
    assert piecewise_percentile(400, 100, 200, 300) == 100


def test_piecewise_percentile_all_quantiles_tied_anchors_at_median():
    assert piecewise_percentile(3, 3, 3, 3) == 50


def test_piecewise_percentile_p25_equals_p50_tied():
    # value at the tied p25/p50 point should resolve to 50, not 25.
    assert piecewise_percentile(5, 5, 5, 10) == 50


def test_piecewise_percentile_p50_equals_p75_tied():
    # value at the tied p50/p75 point should resolve to 75, not 50.
    assert piecewise_percentile(10, 5, 10, 10) == 75


def test_baseline_falls_back_from_champion_to_role():
    baselines = pd.DataFrame(
        [
            {
                "scope": "role",
                "role": "TOP",
                "champion_id": pd.NA,
                "minute": 10,
                "sample_count": 500,
            },
            {
                "scope": "global",
                "role": pd.NA,
                "champion_id": pd.NA,
                "minute": 10,
                "sample_count": 1000,
            },
        ]
    )
    row = choose_baseline(baselines, "TOP", 266, 10)
    assert row["scope"] == "role"


def test_score_observation_aggregates_available_metrics():
    row = pd.Series(
        {
            "scope": "role",
            "sample_count": 500,
            "total_gold_p25": 3000,
            "total_gold_p50": 3500,
            "total_gold_p75": 4000,
            "xp_p25": 4000,
            "xp_p50": 4500,
            "xp_p75": 5000,
            "level_p25": 6,
            "level_p50": 7,
            "level_p75": 8,
            "minions_killed_p25": 55,
            "minions_killed_p50": 65,
            "minions_killed_p75": 75,
        }
    )
    result = score_observation(
        row,
        {"total_gold": 3500, "xp": 4500, "level": 7, "minions_killed": 65},
        "TOP",
    )
    assert result["p_score"] == 50
    assert result["baseline_scope"] == "role"
    assert result["baseline_fallback_level"] == 2
    # Old-format baseline rows without match_n/player_n degrade to None
    # rather than raising.
    assert result["baseline_match_n"] is None
    assert result["baseline_player_n"] is None


def test_score_observation_passes_through_match_and_player_counts():
    row = pd.Series(
        {
            "scope": "champion_role",
            "sample_count": 60,
            "match_n": 55,
            "player_n": 40,
            "total_gold_p25": 3000,
            "total_gold_p50": 3500,
            "total_gold_p75": 4000,
        }
    )
    result = score_observation(row, {"total_gold": 3500}, "TOP")
    assert result["baseline_fallback_level"] == 1
    assert result["baseline_match_n"] == 55
    assert result["baseline_player_n"] == 40
