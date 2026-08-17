"""Integration tests against a live Postgres for the rank-verification query
in build_baselines.BASE_QUERY (via load_gold_frames). These are regression
tests for two bugs the query previously had:

1. The window used to be one-directional (snapshot had to be AFTER
   game_start), silently rejecting valid pre-match snapshots.
2. The old query filtered to tier='GOLD' before picking the nearest
   snapshot, which could skip past a closer non-GOLD snapshot that should
   have invalidated the match.

Each test inserts synthetic rows with a distinctly-namespaced match_id/
puuid directly into the real tables, runs the actual production query via
load_gold_frames, and rolls back the transaction — nothing is persisted.
Skipped automatically when DATABASE_URL isn't configured (e.g. CI without a
database available).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from build_baselines import load_gold_frames

load_dotenv(".env")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="requires a live DATABASE_URL")


def _insert_match_and_participant(conn, match_id: str, puuid: str, game_start_sql: str) -> None:
    conn.execute(
        text(
            f"""
            INSERT INTO matches (match_id, regional_route, queue_id, patch, game_version,
                                  game_start, duration_seconds, collection_status)
            VALUES (:mid, 'AMERICAS', 420, '16.14', '16.14.1.1',
                    TIMESTAMPTZ '{game_start_sql}', 1800, 'complete')
            """
        ),
        {"mid": match_id},
    )
    conn.execute(
        text(
            """
            INSERT INTO participants (match_id, participant_id, puuid, team_id, team_position,
                                       champion_id, champion_name)
            VALUES (:mid, 1, :puuid, 100, 'TOP', 266, 'Aatrox')
            """
        ),
        {"mid": match_id, "puuid": puuid},
    )
    conn.execute(
        text(
            """
            INSERT INTO participant_frames (match_id, participant_id, minute, timestamp_ms,
                                             total_gold, xp, level, minions_killed, jungle_minions_killed)
            VALUES (:mid, 1, 1, 60000, 500, 200, 1, 5, 0)
            """
        ),
        {"mid": match_id},
    )


def test_rank_verification_picks_nearest_snapshot_across_all_tiers_before_checking_gold():
    """The nearest snapshot to game_start is SILVER; a farther snapshot IS
    Gold. Correct behavior: reject the match — the nearest snapshot is
    checked for tier, the farther Gold snapshot must not be used as a
    fallback."""
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        match_id, puuid = "TESTMATCH_RANKWINDOW_TIER", "TESTPUUID_RANKWINDOW_TIER"
        _insert_match_and_participant(conn, match_id, puuid, "2026-01-10 00:00:00+00")
        conn.execute(
            text(
                """
                INSERT INTO rank_snapshots (puuid, queue_type, tier, division, league_points, wins, losses, observed_at)
                VALUES
                    (:puuid, 'RANKED_SOLO_5x5', 'SILVER', 'I', 50, 5, 5, TIMESTAMPTZ '2026-01-11 00:00:00+00'),
                    (:puuid, 'RANKED_SOLO_5x5', 'GOLD', 'IV', 10, 5, 5, TIMESTAMPTZ '2026-01-15 00:00:00+00')
                """
            ),
            {"puuid": puuid},
        )
        frame = load_gold_frames(conn, "16.14", 420, max_rank_age_days=21, match_ids=[match_id])
        assert frame.empty, "nearest snapshot is SILVER -- must reject even though a farther GOLD snapshot exists"
    finally:
        trans.rollback()
        conn.close()


def test_rank_verification_accepts_gold_snapshot_observed_before_the_match():
    """Regression test for the directionality bug: a Gold snapshot observed
    2 days BEFORE game_start must be accepted. A one-directional
    (observed_at >= game_start) window would have wrongly rejected this."""
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        match_id, puuid = "TESTMATCH_RANKWINDOW_BEFORE", "TESTPUUID_RANKWINDOW_BEFORE"
        _insert_match_and_participant(conn, match_id, puuid, "2026-01-10 00:00:00+00")
        conn.execute(
            text(
                """
                INSERT INTO rank_snapshots (puuid, queue_type, tier, division, league_points, wins, losses, observed_at)
                VALUES (:puuid, 'RANKED_SOLO_5x5', 'GOLD', 'IV', 10, 5, 5, TIMESTAMPTZ '2026-01-08 00:00:00+00')
                """
            ),
            {"puuid": puuid},
        )
        frame = load_gold_frames(conn, "16.14", 420, max_rank_age_days=21, match_ids=[match_id])
        assert not frame.empty, "a GOLD snapshot 2 days before game_start must be accepted (bidirectional window)"
    finally:
        trans.rollback()
        conn.close()


def test_rank_verification_rejects_snapshot_outside_the_window():
    """A Gold snapshot 30 days away (outside a 21-day window) must be
    rejected regardless of direction."""
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    trans = conn.begin()
    try:
        match_id, puuid = "TESTMATCH_RANKWINDOW_STALE", "TESTPUUID_RANKWINDOW_STALE"
        _insert_match_and_participant(conn, match_id, puuid, "2026-01-10 00:00:00+00")
        conn.execute(
            text(
                """
                INSERT INTO rank_snapshots (puuid, queue_type, tier, division, league_points, wins, losses, observed_at)
                VALUES (:puuid, 'RANKED_SOLO_5x5', 'GOLD', 'IV', 10, 5, 5, TIMESTAMPTZ '2026-02-09 00:00:00+00')
                """
            ),
            {"puuid": puuid},
        )
        frame = load_gold_frames(conn, "16.14", 420, max_rank_age_days=21, match_ids=[match_id])
        assert frame.empty, "a snapshot 30 days away must be rejected (outside the 21-day window)"
    finally:
        trans.rollback()
        conn.close()
