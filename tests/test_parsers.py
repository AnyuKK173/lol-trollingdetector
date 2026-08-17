from datetime import datetime, timedelta, timezone

from parsers import (
    check_patch_match,
    parse_match_metadata,
    parse_participant_frames,
    parse_timeline_events,
    rank_drift_days,
)


def test_match_metadata_keeps_queue_patch_and_duration():
    match = {
        "metadata": {"matchId": "NA1_123", "dataVersion": "2"},
        "info": {
            "platformId": "NA1",
            "queueId": 420,
            "gameVersion": "16.14.702.1234",
            "gameStartTimestamp": 1_700_000_000_000,
            "gameEndTimestamp": 1_700_001_800_000,
            "gameDuration": 1800,
            "mapId": 11,
            "gameMode": "CLASSIC",
            "gameType": "MATCHED_GAME",
        },
    }
    parsed = parse_match_metadata(match, "americas")
    assert parsed["match_id"] == "NA1_123"
    assert parsed["queue_id"] == 420
    assert parsed["patch"] == "16.14"
    assert parsed["duration_seconds"] == 1800
    assert parsed["regional_route"] == "AMERICAS"


def test_participant_frames_capture_economy_position_and_nested_stats():
    timeline = {
        "info": {
            "frames": [
                {
                    "timestamp": 60000,
                    "participantFrames": {
                        "1": {
                            "participantId": 1,
                            "currentGold": 400,
                            "totalGold": 900,
                            "level": 2,
                            "xp": 500,
                            "minionsKilled": 8,
                            "jungleMinionsKilled": 0,
                            "position": {"x": 100, "y": 200},
                            "championStats": {"armor": 35},
                            "damageStats": {"totalDamageDone": 700},
                        }
                    },
                }
            ]
        }
    }
    row = parse_participant_frames(timeline, "NA1_123")[0]
    assert row["minute"] == 1
    assert row["total_gold"] == 900
    assert row["position_x"] == 100
    assert row["champion_stats"]["armor"] == 35
    assert row["damage_stats"]["totalDamageDone"] == 700


def test_event_uses_its_own_timestamp_not_frame_timestamp():
    timeline = {
        "info": {
            "frames": [
                {
                    "timestamp": 120000,
                    "events": [
                        {
                            "timestamp": 73456,
                            "type": "CHAMPION_KILL",
                            "killerId": 1,
                            "victimId": 6,
                        }
                    ],
                }
            ]
        }
    }
    row = parse_timeline_events(timeline, "NA1_123")[0]
    assert row["timestamp_ms"] == 73456
    assert row["event_type"] == "CHAMPION_KILL"


def test_check_patch_match_accepts_exact_target_patch():
    assert check_patch_match("16.14", "16.14") is None


def test_check_patch_match_rejects_other_patches():
    reason = check_patch_match("16.13", "16.14")
    assert reason is not None
    assert "16.13" in reason and "16.14" in reason


def test_rank_drift_days_returns_none_when_game_start_unknown():
    observed_at = datetime(2026, 7, 20, tzinfo=timezone.utc)
    assert rank_drift_days(observed_at, None) is None


def test_rank_drift_days_within_window():
    observed_at = datetime(2026, 7, 20, tzinfo=timezone.utc)
    game_start = observed_at - timedelta(days=21)
    assert rank_drift_days(observed_at, game_start) == 21.0


def test_rank_drift_days_beyond_window():
    observed_at = datetime(2026, 7, 20, tzinfo=timezone.utc)
    game_start = observed_at - timedelta(days=22)
    assert rank_drift_days(observed_at, game_start) == 22.0
