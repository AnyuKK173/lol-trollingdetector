from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


ROOT = Path(__file__).resolve().parent
METRICS = [
    "total_gold",
    "xp",
    "level",
    "minions_killed",
    "jungle_minions_killed",
]


MIN_DURATION_SECONDS = 180  # excludes remakes/early-surrender games


BASE_QUERY = """
WITH candidate_ranks AS (
    -- Do NOT filter to tier='GOLD' before picking the nearest snapshot —
    -- that could skip past a closer non-GOLD snapshot that should actually
    -- invalidate this match. Rank the nearest snapshot to game_start first
    -- (by absolute time difference, not just "observed after the match" —
    -- a snapshot taken a few days BEFORE the match is equally valid
    -- evidence and was being wrongly excluded by a one-directional window),
    -- then check its tier afterwards.
    SELECT
        p.match_id, p.participant_id, p.puuid, r.tier,
        ROW_NUMBER() OVER (
            PARTITION BY p.match_id, p.puuid
            ORDER BY ABS(EXTRACT(EPOCH FROM (r.observed_at - m.game_start)))
        ) AS rn
    FROM participants p
    JOIN matches m ON m.match_id = p.match_id
    JOIN rank_snapshots r ON r.puuid = p.puuid AND r.queue_type = 'RANKED_SOLO_5x5'
    WHERE m.collection_status = 'complete'
      AND m.patch = %(target_patch)s
      AND m.queue_id = %(target_queue)s
      AND m.duration_seconds >= %(min_duration_seconds)s
      AND ABS(EXTRACT(EPOCH FROM (r.observed_at - m.game_start))) <= %(max_rank_age_days)s * 86400
),
nearest_verified_rank AS (
    SELECT match_id, participant_id, puuid FROM candidate_ranks WHERE rn = 1 AND tier = 'GOLD'
),
minute_frames AS (
    SELECT DISTINCT ON (match_id, participant_id, minute)
        match_id, participant_id, minute, timestamp_ms, total_gold, xp, level,
        minions_killed, jungle_minions_killed
    FROM participant_frames
    ORDER BY match_id, participant_id, minute, timestamp_ms DESC
)
SELECT
    f.match_id,
    p.participant_id,
    p.puuid,
    COALESCE(NULLIF(p.team_position, ''), NULLIF(p.individual_position, '')) AS role,
    p.champion_id,
    f.minute,
    f.total_gold,
    f.xp,
    f.level,
    f.minions_killed,
    f.jungle_minions_killed
FROM minute_frames f
JOIN participants p
  ON p.match_id = f.match_id AND p.participant_id = f.participant_id
JOIN matches m ON m.match_id = f.match_id
JOIN nearest_verified_rank r
  ON r.match_id = f.match_id AND r.participant_id = f.participant_id AND r.puuid = p.puuid
WHERE m.collection_status = 'complete'
  -- Defense in depth: collector.py should already guarantee patch/queue
  -- correctness, but this file must not trust that blindly.
  AND m.patch = %(target_patch)s
  AND m.queue_id = %(target_queue)s
  AND m.duration_seconds >= %(min_duration_seconds)s
  AND f.minute BETWEEN 1 AND 60
  AND COALESCE(NULLIF(p.team_position, ''), NULLIF(p.individual_position, ''))
      IN ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
"""


def load_gold_frames(
    engine,
    target_patch: str,
    target_queue: int,
    max_rank_age_days: float,
    min_duration_seconds: int = MIN_DURATION_SECONDS,
    match_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Loads identity-verified Gold participant_frames for one patch.

    match_ids, when given, restricts the query to that set — this is how the
    behavior-window builder fits a baseline on the train split only, so
    labels for test-split matches are never scored against a baseline that
    included those same matches."""
    query = BASE_QUERY
    params: dict[str, object] = {
        "target_patch": target_patch,
        "target_queue": target_queue,
        "min_duration_seconds": min_duration_seconds,
        "max_rank_age_days": max_rank_age_days,
    }
    if match_ids is not None:
        query += " AND f.match_id = ANY(%(match_ids)s)"
        params["match_ids"] = list(match_ids)
    frame = pd.read_sql_query(query, engine, params=params)
    frame[METRICS] = frame[METRICS].apply(pd.to_numeric, errors="coerce")
    return frame


def aggregate_quantiles(
    frame: pd.DataFrame,
    group_columns: list[str],
    scope: str,
    minimum_samples: int,
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False, observed=True)
    counts = grouped.size().rename("sample_count")
    match_n = grouped["match_id"].nunique().rename("match_n")
    player_n = grouped["puuid"].nunique().rename("player_n")
    quantiles = grouped[METRICS].quantile([0.25, 0.50, 0.75]).unstack(-1)
    quantiles.columns = [
        f"{metric}_p{int(quantile * 100)}" for metric, quantile in quantiles.columns
    ]
    result = counts.to_frame().join([match_n, player_n, quantiles]).reset_index()
    result = result[result["sample_count"] >= minimum_samples].copy()
    result.insert(0, "scope", scope)
    if "champion_id" not in result.columns:
        result["champion_id"] = pd.NA
    if "role" not in result.columns:
        result["role"] = pd.NA
    return result


SCOPE_FALLBACK_LEVEL = {"champion_role": 1, "role": 2, "global": 3}


def compute_baselines(
    frame: pd.DataFrame,
    min_champion_samples: int,
    min_role_samples: int,
    min_global_samples: int,
) -> pd.DataFrame:
    """Runs the champion_role -> role -> global aggregation over an
    already-loaded frame (see load_gold_frames). Kept separate from the
    query so build_behavior_dataset.py can fit this on a train-only frame."""
    baselines = [
        aggregate_quantiles(
            frame, ["role", "champion_id", "minute"], "champion_role", min_champion_samples
        ),
        aggregate_quantiles(frame, ["role", "minute"], "role", min_role_samples),
        aggregate_quantiles(frame, ["minute"], "global", min_global_samples),
    ]
    result = pd.concat(baselines, ignore_index=True, sort=False)
    result["fallback_level"] = result["scope"].map(SCOPE_FALLBACK_LEVEL)
    ordered = [
        "scope",
        "fallback_level",
        "role",
        "champion_id",
        "minute",
        "sample_count",
        "match_n",
        "player_n",
    ] + [f"{metric}_p{q}" for metric in METRICS for q in (25, 50, 75)]
    return result.reindex(columns=ordered)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="从已采集 Gold 玩家帧构建 P25/P50/P75 经验表现曲线。"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-champion-samples", type=int, default=200)
    parser.add_argument("--min-role-samples", type=int, default=500)
    parser.add_argument("--min-global-samples", type=int, default=1000)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("缺少 DATABASE_URL。", file=sys.stderr)
        return 1

    target_patch = os.getenv("TARGET_PATCH", "").strip()
    if not target_patch:
        print("缺少 TARGET_PATCH，不知道该为哪个版本建基线。", file=sys.stderr)
        return 1
    target_queue = int(os.getenv("TARGET_QUEUE_ID", "420"))
    max_rank_age_days = float(os.getenv("MAX_RANK_AGE_DAYS", "21"))

    engine = create_engine(database_url)
    try:
        frame = load_gold_frames(
            engine, target_patch, target_queue, max_rank_age_days, MIN_DURATION_SECONDS
        )
    finally:
        engine.dispose()

    if frame.empty:
        print(
            f"没有可用于基线的 Gold participant_frames（target_patch={target_patch}）。",
            file=sys.stderr,
        )
        return 2

    result = compute_baselines(
        frame, args.min_champion_samples, args.min_role_samples, args.min_global_samples
    )

    output_arg = args.output or f"./output_v3/patch={target_patch}/baselines/gold_quantiles.parquet"
    target = Path(output_arg)
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False, compression="zstd")

    print(f"target_patch：{target_patch}")
    print(f"输入帧：{len(frame):,}")
    print("输出基线：")
    print(result.groupby("scope").size().to_string())
    print(f"文件：{target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
