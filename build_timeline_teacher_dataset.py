"""Build all-player, cutoff-safe checkpoints for the v4 outcome teacher.

The retained matches and their train/test membership come from the frozen v3
Gold-subject windows. Every valid participant in those matches inherits the
match split. The final result is used only as ``final_win``; every model input
is reconstructed from frames/events at or before ``feature_cutoff_minute``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
MIN_DURATION_SECONDS = 900
CHECKPOINT_STEP = 3
KEYS = ["match_id", "participant_id", "feature_cutoff_minute"]
FEATURE_COLUMNS = [
    "feature_cutoff_minute",
    "gold_now",
    "xp_now",
    "level_now",
    "cs_now",
    "jungle_cs_now",
    "gold_rate_3m",
    "xp_rate_3m",
    "cs_rate_3m",
    "jungle_cs_rate_3m",
    "kills_now",
    "deaths_now",
    "assists_now",
    "wards_placed_now",
    "team_objectives_now",
    "lane_gold_gap",
    "lane_xp_gap",
    "lane_cs_gap",
    "team_gold_gap",
    "distance_3m",
]

PARTICIPANTS_QUERY = """
SELECT p.match_id, p.participant_id, p.puuid, p.team_id, p.champion_id, p.win,
       COALESCE(NULLIF(p.team_position, ''), NULLIF(p.individual_position, '')) AS role
FROM participants p
JOIN matches m USING (match_id)
WHERE m.collection_status = 'complete'
  AND m.patch = %(patch)s AND m.queue_id = %(queue)s
  AND m.duration_seconds >= %(min_duration)s
"""

FRAMES_QUERY = """
SELECT pf.match_id, pf.participant_id, pf.minute, pf.timestamp_ms, pf.total_gold,
       pf.xp, pf.level, pf.minions_killed, pf.jungle_minions_killed,
       pf.position_x, pf.position_y
FROM participant_frames pf
JOIN matches m USING (match_id)
WHERE m.collection_status = 'complete'
  AND m.patch = %(patch)s AND m.queue_id = %(queue)s
  AND m.duration_seconds >= %(min_duration)s
"""

EVENTS_QUERY = """
SELECT te.match_id, te.timestamp_ms, te.event_type, te.participant_id,
       te.killer_id, te.victim_id, te.team_id, te.assisting_participant_ids
FROM timeline_events te
JOIN matches m USING (match_id)
WHERE m.collection_status = 'complete'
  AND m.patch = %(patch)s AND m.queue_id = %(queue)s
  AND m.duration_seconds >= %(min_duration)s
  AND te.event_type IN (
      'CHAMPION_KILL', 'CHAMPION_SPECIAL_KILL', 'WARD_PLACED',
      'ELITE_MONSTER_KILL', 'BUILDING_KILL'
  )
"""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_match_splits(windows: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "split"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"v3 windows missing columns: {sorted(missing)}")
    split_n = windows.groupby("match_id")["split"].nunique()
    mixed = split_n[split_n > 1]
    if not mixed.empty:
        raise ValueError(f"v3 contains {len(mixed)} mixed-split matches")
    result = windows[["match_id", "split"]].drop_duplicates()
    invalid = set(result["split"]) - {"train", "test"}
    if invalid:
        raise ValueError(f"unknown split values: {sorted(invalid)}")
    return result


def _last_frame_per_minute(frames: pd.DataFrame) -> pd.DataFrame:
    return (
        frames.sort_values("timestamp_ms")
        .drop_duplicates(["match_id", "participant_id", "minute"], keep="last")
        .sort_values(["match_id", "participant_id", "minute"])
    )


def _rate(frame_by_minute: dict[int, dict[str, Any]], minute: int, column: str) -> float | None:
    start = frame_by_minute.get(minute - 3)
    end = frame_by_minute.get(minute)
    if start is None or end is None or pd.isna(start.get(column)) or pd.isna(end.get(column)):
        return None
    return (float(end[column]) - float(start[column])) / 3.0


def _distance_3m(frame_by_minute: dict[int, dict[str, Any]], minute: int) -> float | None:
    points = []
    for t in range(minute - 3, minute + 1):
        row = frame_by_minute.get(t)
        if row and pd.notna(row.get("position_x")) and pd.notna(row.get("position_y")):
            points.append((float(row["position_x"]), float(row["position_y"])))
    if len(points) < 2:
        return None
    return sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 for (x1, y1), (x2, y2) in zip(points, points[1:]))


def _assist_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [int(x) for x in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return list(value) if hasattr(value, "__iter__") else []


def _event_state(events: pd.DataFrame, participant_id: int, team_id: int, cutoff_ms: int) -> tuple[int, int, int, int, int]:
    kills = deaths = assists = wards = objectives = 0
    for event in events[events["timestamp_ms"] <= cutoff_ms].itertuples(index=False):
        if event.event_type in {"CHAMPION_KILL", "CHAMPION_SPECIAL_KILL"}:
            kills += int(event.killer_id == participant_id)
            deaths += int(event.victim_id == participant_id)
            assists += int(participant_id in _assist_ids(event.assisting_participant_ids))
        elif event.event_type == "WARD_PLACED" and event.participant_id == participant_id:
            wards += 1
        elif event.event_type in {"ELITE_MONSTER_KILL", "BUILDING_KILL"} and event.team_id == team_id:
            objectives += 1
    return kills, deaths, assists, wards, objectives


def build_checkpoint_rows(
    windows: pd.DataFrame,
    participants: pd.DataFrame,
    frames: pd.DataFrame,
    events: pd.DataFrame,
    step: int = CHECKPOINT_STEP,
) -> pd.DataFrame:
    """Pure builder used by production and leakage regression tests."""
    if step < 1:
        raise ValueError("step must be >= 1")
    match_splits = retained_match_splits(windows)
    people = participants.merge(match_splits, on="match_id", how="inner", validate="many_to_one")
    people = people[people["role"].isin(ROLES)].copy()
    retained = set(people["match_id"])
    frames = _last_frame_per_minute(frames[frames["match_id"].isin(retained)].copy())
    events = events[events["match_id"].isin(retained)].sort_values(["match_id", "timestamp_ms"]).copy()

    frame_maps = {
        key: group.set_index("minute").to_dict("index")
        for key, group in frames.groupby(["match_id", "participant_id"], sort=False)
    }
    event_maps = {match_id: group for match_id, group in events.groupby("match_id", sort=False)}
    people_by_match = {match_id: group for match_id, group in people.groupby("match_id", sort=False)}
    rows: list[dict[str, Any]] = []

    for person in people.itertuples(index=False):
        own = frame_maps.get((person.match_id, person.participant_id), {})
        match_people = people_by_match[person.match_id]
        opponent_rows = match_people[(match_people["team_id"] != person.team_id) & (match_people["role"] == person.role)]
        opponent = opponent_rows.iloc[0] if len(opponent_rows) == 1 else None
        opponent_frames = frame_maps.get((person.match_id, int(opponent["participant_id"])), {}) if opponent is not None else {}
        teammate_ids = match_people.loc[
            (match_people["team_id"] == person.team_id) & (match_people["participant_id"] != person.participant_id),
            "participant_id",
        ].astype(int).tolist()
        enemy_ids = match_people.loc[match_people["team_id"] != person.team_id, "participant_id"].astype(int).tolist()
        match_events = event_maps.get(person.match_id, events.iloc[0:0])

        for minute in sorted(t for t in own if t >= 3 and t % step == 0):
            current = own[minute]
            cutoff_ms = int(current["timestamp_ms"])
            kills, deaths, assists, wards, objectives = _event_state(
                match_events, int(person.participant_id), int(person.team_id), cutoff_ms
            )
            opponent_now = opponent_frames.get(minute)
            teammate_gold = [frame_maps.get((person.match_id, pid), {}).get(minute, {}).get("total_gold") for pid in teammate_ids]
            enemy_gold = [frame_maps.get((person.match_id, pid), {}).get(minute, {}).get("total_gold") for pid in enemy_ids]
            teammate_gold = [float(x) for x in teammate_gold if x is not None and pd.notna(x)]
            enemy_gold = [float(x) for x in enemy_gold if x is not None and pd.notna(x)]

            def gap(column: str) -> float | None:
                if opponent_now is None or pd.isna(current.get(column)) or pd.isna(opponent_now.get(column)):
                    return None
                return float(current[column]) - float(opponent_now[column])

            rows.append(
                {
                    "match_id": person.match_id,
                    "puuid": person.puuid,
                    "participant_id": int(person.participant_id),
                    "split": person.split,
                    "role": person.role,
                    "champion_id": int(person.champion_id) if pd.notna(person.champion_id) else None,
                    "feature_cutoff_minute": int(minute),
                    "feature_cutoff_timestamp_ms": cutoff_ms,
                    "final_win": bool(person.win),
                    "gold_now": current.get("total_gold"),
                    "xp_now": current.get("xp"),
                    "level_now": current.get("level"),
                    "cs_now": current.get("minions_killed"),
                    "jungle_cs_now": current.get("jungle_minions_killed"),
                    "gold_rate_3m": _rate(own, minute, "total_gold"),
                    "xp_rate_3m": _rate(own, minute, "xp"),
                    "cs_rate_3m": _rate(own, minute, "minions_killed"),
                    "jungle_cs_rate_3m": _rate(own, minute, "jungle_minions_killed"),
                    "kills_now": kills,
                    "deaths_now": deaths,
                    "assists_now": assists,
                    "wards_placed_now": wards,
                    "team_objectives_now": objectives,
                    "lane_gold_gap": gap("total_gold"),
                    "lane_xp_gap": gap("xp"),
                    "lane_cs_gap": gap("minions_killed"),
                    "team_gold_gap": (sum(teammate_gold) + float(current["total_gold"]) - sum(enemy_gold)) if teammate_gold and enemy_gold else None,
                    "distance_3m": _distance_3m(own, minute),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no checkpoint rows were constructed")
    if result.duplicated(KEYS).any():
        raise AssertionError("duplicate checkpoint keys")
    if result.groupby("match_id")["split"].nunique().max() > 1:
        raise AssertionError("mixed-split match leaked into teacher dataset")
    return result.sort_values(KEYS).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all-player timeline-safe teacher checkpoints.")
    parser.add_argument("--v3-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=CHECKPOINT_STEP)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    patch = os.getenv("TARGET_PATCH", "").strip()
    if not database_url or not patch:
        print("DATABASE_URL and TARGET_PATCH are required")
        return 1
    params = {
        "patch": patch,
        "queue": int(os.getenv("TARGET_QUEUE_ID", "420")),
        "min_duration": MIN_DURATION_SECONDS,
    }
    windows = pd.read_parquet(args.v3_windows)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            participants = pd.read_sql_query(PARTICIPANTS_QUERY, connection, params=params)
            frames = pd.read_sql_query(FRAMES_QUERY, connection, params=params)
            events = pd.read_sql_query(EVENTS_QUERY, connection, params=params)
    finally:
        engine.dispose()

    result = build_checkpoint_rows(windows, participants, frames, events, args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "dataset_version": "timeline_teacher_training_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patch": patch,
        "queue_id": params["queue"],
        "checkpoint_step_minutes": args.step,
        "row_n": len(result),
        "match_n": result["match_id"].nunique(),
        "player_n": result["puuid"].nunique(),
        "all_players_in_retained_matches": True,
        "source_v3_windows": str(args.v3_windows),
        "source_v3_windows_sha256": sha256_of(args.v3_windows),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "final_win",
        "feature_contract": "all model inputs use timestamp <= feature_cutoff_timestamp_ms; final_win is target only",
        "output_sha256": sha256_of(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(result):,} rows from {manifest['match_n']:,} retained matches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
