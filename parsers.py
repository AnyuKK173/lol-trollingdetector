"""Pure Riot Match-V5 parsers.

These functions do not touch the network or database, which makes schema
changes and regression testing much safer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc_from_ms(value: Any) -> datetime | None:
    if value in (None, 0, ""):
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def _patch_from_version(game_version: str | None) -> str | None:
    if not game_version:
        return None
    parts = game_version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else game_version


def check_patch_match(actual_patch: str | None, target_patch: str) -> str | None:
    """None if the match's patch is the one we're collecting; otherwise a
    skip reason string suitable for matches.collection_error."""
    if actual_patch != target_patch:
        return f"patch_mismatch:expected={target_patch}:actual={actual_patch}"
    return None


def rank_drift_days(observed_at: datetime, game_start: datetime | None) -> float | None:
    """How many days separate the rank observation from when the match was
    actually played. None if game_start is unknown (can't be judged)."""
    if game_start is None:
        return None
    delta = observed_at - game_start
    return abs(delta.total_seconds()) / 86400.0


def parse_match_metadata(match: dict[str, Any], regional_route: str) -> dict[str, Any]:
    metadata = match.get("metadata") or {}
    info = match.get("info") or {}

    duration = info.get("gameDuration")
    if duration is not None:
        duration = int(duration)
        # Old payloads sometimes used milliseconds. Current Match-V5 uses seconds.
        if duration > 100_000:
            duration //= 1000

    game_version = info.get("gameVersion")
    return {
        "match_id": metadata.get("matchId"),
        "data_version": metadata.get("dataVersion"),
        "platform_id": info.get("platformId"),
        "regional_route": regional_route.upper(),
        "queue_id": int(info.get("queueId") or 0),
        "game_version": game_version,
        "patch": _patch_from_version(game_version),
        "map_id": info.get("mapId"),
        "game_mode": info.get("gameMode"),
        "game_type": info.get("gameType"),
        "game_start": _utc_from_ms(
            info.get("gameStartTimestamp") or info.get("gameCreation")
        ),
        "game_end": _utc_from_ms(info.get("gameEndTimestamp")),
        "duration_seconds": duration,
    }


def parse_participants(match: dict[str, Any]) -> list[dict[str, Any]]:
    match_id = (match.get("metadata") or {}).get("matchId")
    rows: list[dict[str, Any]] = []

    for player in (match.get("info") or {}).get("participants") or []:
        rows.append(
            {
                "match_id": match_id,
                "participant_id": player.get("participantId"),
                "puuid": player.get("puuid"),
                "team_id": player.get("teamId"),
                "team_position": player.get("teamPosition") or None,
                "individual_position": player.get("individualPosition") or None,
                "champion_id": player.get("championId"),
                "champion_name": player.get("championName"),
                "win": player.get("win"),
                "kills": player.get("kills"),
                "deaths": player.get("deaths"),
                "assists": player.get("assists"),
                "gold_earned": player.get("goldEarned"),
                "total_minions_killed": player.get("totalMinionsKilled"),
                "neutral_minions_killed": player.get("neutralMinionsKilled"),
                "vision_score": player.get("visionScore"),
                "wards_placed": player.get("wardsPlaced"),
                "wards_killed": player.get("wardsKilled"),
                "damage_to_champions": player.get("totalDamageDealtToChampions"),
                "damage_taken": player.get("totalDamageTaken"),
                "damage_to_objectives": player.get("damageDealtToObjectives"),
                "damage_to_turrets": player.get("damageDealtToTurrets"),
                "time_ccing_others": player.get("timeCCingOthers"),
                "total_time_cc_dealt": player.get("totalTimeCCDealt"),
            }
        )
    return rows


def parse_participant_frames(
    timeline: dict[str, Any], match_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in (timeline.get("info") or {}).get("frames") or []:
        timestamp_ms = int(frame.get("timestamp") or 0)
        minute = timestamp_ms // 60_000
        participant_frames = frame.get("participantFrames") or {}

        for participant_key, participant_frame in participant_frames.items():
            position = participant_frame.get("position") or {}
            participant_id = int(
                participant_frame.get("participantId") or participant_key
            )
            rows.append(
                {
                    "match_id": match_id,
                    "participant_id": participant_id,
                    "minute": minute,
                    "timestamp_ms": timestamp_ms,
                    "current_gold": participant_frame.get("currentGold"),
                    "total_gold": participant_frame.get("totalGold"),
                    "level": participant_frame.get("level"),
                    "xp": participant_frame.get("xp"),
                    "minions_killed": participant_frame.get("minionsKilled"),
                    "jungle_minions_killed": participant_frame.get(
                        "jungleMinionsKilled"
                    ),
                    "position_x": position.get("x"),
                    "position_y": position.get("y"),
                    "time_enemy_spent_controlled": participant_frame.get(
                        "timeEnemySpentControlled"
                    ),
                    "champion_stats": participant_frame.get("championStats") or {},
                    "damage_stats": participant_frame.get("damageStats") or {},
                }
            )
    return rows


def parse_timeline_events(
    timeline: dict[str, Any], match_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in (timeline.get("info") or {}).get("frames") or []:
        frame_timestamp = int(frame.get("timestamp") or 0)
        for event in frame.get("events") or []:
            position = event.get("position") or {}
            rows.append(
                {
                    "match_id": match_id,
                    # Important: events inside one frame have their own timestamps.
                    "timestamp_ms": int(event.get("timestamp") or frame_timestamp),
                    "event_type": event.get("type") or "UNKNOWN",
                    "participant_id": event.get("participantId"),
                    "killer_id": event.get("killerId"),
                    "victim_id": event.get("victimId"),
                    "creator_id": event.get("creatorId"),
                    "assisting_participant_ids": event.get(
                        "assistingParticipantIds"
                    )
                    or [],
                    "team_id": event.get("teamId"),
                    "item_id": event.get("itemId"),
                    "ward_type": event.get("wardType"),
                    "monster_type": event.get("monsterType"),
                    "monster_sub_type": event.get("monsterSubType"),
                    "building_type": event.get("buildingType"),
                    "lane_type": event.get("laneType"),
                    "tower_type": event.get("towerType"),
                    "position_x": position.get("x"),
                    "position_y": position.get("y"),
                    "raw_event": event,
                }
            )
    return rows
