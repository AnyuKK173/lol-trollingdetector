from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import psycopg2
from psycopg2.extras import Json, execute_values


class Storage:
    def __init__(self, database_url: str) -> None:
        self.connection = psycopg2.connect(database_url)

    def close(self) -> None:
        self.connection.close()

    def apply_schema(self, schema_path: Path) -> None:
        sql = schema_path.read_text(encoding="utf-8")
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)

    def is_complete(self, match_id: str) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM matches WHERE match_id = %s AND collection_status = 'complete'",
                (match_id,),
            )
            return cursor.fetchone() is not None

    def save_rank_snapshots(
        self, entries: Iterable[dict[str, Any]], observed_at: datetime
    ) -> int:
        values = [
            (
                entry.get("puuid"),
                entry.get("queueType") or "RANKED_SOLO_5x5",
                entry.get("tier") or "GOLD",
                entry.get("rank"),
                entry.get("leaguePoints") or 0,
                entry.get("wins") or 0,
                entry.get("losses") or 0,
                observed_at,
            )
            for entry in entries
            if entry.get("puuid") and entry.get("rank")
        ]
        if not values:
            return 0
        with self.connection:
            with self.connection.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    INSERT INTO rank_snapshots (
                        puuid, queue_type, tier, division, league_points,
                        wins, losses, observed_at
                    ) VALUES %s
                    """,
                    values,
                    page_size=500,
                )
        return len(values)

    def save_match_bundle(
        self,
        metadata: dict[str, Any],
        participants: list[dict[str, Any]],
        frames: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> None:
        match_id = metadata["match_id"]
        participant_ids = {
            int(row["participant_id"])
            for row in participants
            if row.get("participant_id") is not None
        }
        frames = [row for row in frames if row["participant_id"] in participant_ids]

        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO matches (
                        match_id, data_version, platform_id, regional_route,
                        queue_id, game_version, patch, map_id, game_mode,
                        game_type, game_start, game_end, duration_seconds,
                        collection_status, collection_error
                    ) VALUES (
                        %(match_id)s, %(data_version)s, %(platform_id)s,
                        %(regional_route)s, %(queue_id)s, %(game_version)s,
                        %(patch)s, %(map_id)s, %(game_mode)s, %(game_type)s,
                        %(game_start)s, %(game_end)s, %(duration_seconds)s,
                        'collecting', NULL
                    )
                    ON CONFLICT (match_id) DO UPDATE SET
                        data_version = EXCLUDED.data_version,
                        platform_id = EXCLUDED.platform_id,
                        regional_route = EXCLUDED.regional_route,
                        queue_id = EXCLUDED.queue_id,
                        game_version = EXCLUDED.game_version,
                        patch = EXCLUDED.patch,
                        map_id = EXCLUDED.map_id,
                        game_mode = EXCLUDED.game_mode,
                        game_type = EXCLUDED.game_type,
                        game_start = EXCLUDED.game_start,
                        game_end = EXCLUDED.game_end,
                        duration_seconds = EXCLUDED.duration_seconds,
                        collected_at = NOW(),
                        collection_status = 'collecting',
                        collection_error = NULL
                    """,
                    metadata,
                )

                # Retrying a failed match is idempotent: replace its child rows.
                cursor.execute("DELETE FROM timeline_events WHERE match_id = %s", (match_id,))
                cursor.execute("DELETE FROM participant_frames WHERE match_id = %s", (match_id,))
                cursor.execute("DELETE FROM participants WHERE match_id = %s", (match_id,))

                self._insert_participants(cursor, participants)
                self._insert_frames(cursor, frames)
                self._insert_events(cursor, events)

                cursor.execute(
                    """
                    UPDATE matches
                    SET collection_status = 'complete', collection_error = NULL
                    WHERE match_id = %s
                    """,
                    (match_id,),
                )

    def mark_failed(
        self,
        match_id: str,
        regional_route: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO matches (
                        match_id, regional_route, queue_id, collection_status,
                        collection_error
                    ) VALUES (%s, %s, %s, 'failed', %s)
                    ON CONFLICT (match_id) DO UPDATE SET
                        collection_status = 'failed',
                        collection_error = EXCLUDED.collection_error,
                        collected_at = NOW()
                    """,
                    (
                        match_id,
                        regional_route.upper(),
                        int(metadata.get("queue_id") or 0),
                        error[:2000],
                    ),
                )

    def mark_skipped(
        self,
        match_id: str,
        regional_route: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Records a deliberate rejection (patch mismatch, rank drift) as
        distinct from a real fetch/parse failure, so operators can tell
        'we saw it and it didn't qualify' from 'something broke'."""
        metadata = metadata or {}
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO matches (
                        match_id, regional_route, queue_id, game_version, patch,
                        game_start, collection_status, collection_error
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'skipped', %s)
                    ON CONFLICT (match_id) DO UPDATE SET
                        collection_status = 'skipped',
                        collection_error = EXCLUDED.collection_error,
                        collected_at = NOW()
                    """,
                    (
                        match_id,
                        regional_route.upper(),
                        int(metadata.get("queue_id") or 0),
                        metadata.get("game_version"),
                        metadata.get("patch"),
                        metadata.get("game_start"),
                        reason[:2000],
                    ),
                )

    def count_complete_matches(self, patch: str, queue_id: int) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*) FROM matches
                WHERE collection_status = 'complete' AND patch = %s AND queue_id = %s
                """,
                (patch, queue_id),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def upsert_collection_subjects(
        self, entries: Iterable[dict[str, Any]], discovered_at: datetime
    ) -> int:
        values = [
            (
                entry.get("puuid"),
                entry.get("tier") or "GOLD",
                entry.get("rank"),
                discovered_at,
            )
            for entry in entries
            if entry.get("puuid") and entry.get("rank")
        ]
        if not values:
            return 0
        with self.connection:
            with self.connection.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    INSERT INTO collection_subjects (
                        puuid, tier, division, discovered_at
                    ) VALUES %s
                    ON CONFLICT (puuid) DO NOTHING
                    """,
                    values,
                    page_size=500,
                )
        return len(values)

    def fetch_pending_subjects(self, limit: int) -> list[dict[str, Any]]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT puuid, tier, division FROM collection_subjects
                WHERE status != 'exhausted'
                ORDER BY last_attempted_at NULLS FIRST
                LIMIT %s
                """,
                (limit,),
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def update_subject_progress(
        self,
        puuid: str,
        accepted_delta: int,
        status: str,
        attempted_at: datetime,
    ) -> None:
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE collection_subjects
                    SET accepted_matches = accepted_matches + %s,
                        status = %s,
                        last_attempted_at = %s
                    WHERE puuid = %s
                    """,
                    (accepted_delta, status, attempted_at, puuid),
                )

    def start_collection_run(
        self, target_patch: str, target_queue: int, target_verified_matches: int
    ) -> int:
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO collection_runs (
                        target_patch, target_queue, target_verified_matches
                    ) VALUES (%s, %s, %s)
                    RETURNING run_id
                    """,
                    (target_patch, target_queue, target_verified_matches),
                )
                row = cursor.fetchone()
                return int(row[0])

    def finish_collection_run(
        self,
        run_id: int,
        accepted_matches: int,
        patch_mismatch_matches: int,
        failed_matches: int,
    ) -> None:
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE collection_runs
                    SET completed_at = NOW(),
                        accepted_matches = %s,
                        patch_mismatch_matches = %s,
                        failed_matches = %s
                    WHERE run_id = %s
                    """,
                    (accepted_matches, patch_mismatch_matches, failed_matches, run_id),
                )

    @staticmethod
    def _insert_participants(cursor: Any, rows: list[dict[str, Any]]) -> None:
        values = [
            (
                row.get("match_id"), row.get("participant_id"), row.get("puuid"),
                row.get("team_id"), row.get("team_position"),
                row.get("individual_position"), row.get("champion_id"),
                row.get("champion_name"), row.get("win"), row.get("kills"),
                row.get("deaths"), row.get("assists"), row.get("gold_earned"),
                row.get("total_minions_killed"), row.get("neutral_minions_killed"),
                row.get("vision_score"), row.get("wards_placed"),
                row.get("wards_killed"), row.get("damage_to_champions"),
                row.get("damage_taken"), row.get("damage_to_objectives"),
                row.get("damage_to_turrets"), row.get("time_ccing_others"),
                row.get("total_time_cc_dealt"),
            )
            for row in rows
            if row.get("participant_id") is not None
        ]
        execute_values(
            cursor,
            """
            INSERT INTO participants (
                match_id, participant_id, puuid, team_id, team_position,
                individual_position, champion_id, champion_name, win, kills,
                deaths, assists, gold_earned, total_minions_killed,
                neutral_minions_killed, vision_score, wards_placed, wards_killed,
                damage_to_champions, damage_taken, damage_to_objectives,
                damage_to_turrets, time_ccing_others, total_time_cc_dealt
            ) VALUES %s
            """,
            values,
            page_size=200,
        )

    @staticmethod
    def _insert_frames(cursor: Any, rows: list[dict[str, Any]]) -> None:
        values = [
            (
                row["match_id"], row["participant_id"], row["minute"],
                row["timestamp_ms"], row.get("current_gold"), row.get("total_gold"),
                row.get("level"), row.get("xp"), row.get("minions_killed"),
                row.get("jungle_minions_killed"), row.get("position_x"),
                row.get("position_y"), row.get("time_enemy_spent_controlled"),
                Json(row.get("champion_stats") or {}),
                Json(row.get("damage_stats") or {}),
            )
            for row in rows
        ]
        if not values:
            return
        execute_values(
            cursor,
            """
            INSERT INTO participant_frames (
                match_id, participant_id, minute, timestamp_ms, current_gold,
                total_gold, level, xp, minions_killed, jungle_minions_killed,
                position_x, position_y, time_enemy_spent_controlled,
                champion_stats, damage_stats
            ) VALUES %s
            """,
            values,
            page_size=1000,
        )

    @staticmethod
    def _insert_events(cursor: Any, rows: list[dict[str, Any]]) -> None:
        values = [
            (
                row["match_id"], row["timestamp_ms"], row["event_type"],
                row.get("participant_id"), row.get("killer_id"),
                row.get("victim_id"), row.get("creator_id"),
                Json(row.get("assisting_participant_ids") or []), row.get("team_id"),
                row.get("item_id"), row.get("ward_type"), row.get("monster_type"),
                row.get("monster_sub_type"), row.get("building_type"),
                row.get("lane_type"), row.get("tower_type"), row.get("position_x"),
                row.get("position_y"), Json(row.get("raw_event") or {}),
            )
            for row in rows
        ]
        if not values:
            return
        execute_values(
            cursor,
            """
            INSERT INTO timeline_events (
                match_id, timestamp_ms, event_type, participant_id, killer_id,
                victim_id, creator_id, assisting_participant_ids, team_id,
                item_id, ward_type, monster_type, monster_sub_type,
                building_type, lane_type, tower_type, position_x, position_y,
                raw_event
            ) VALUES %s
            """,
            values,
            page_size=1000,
        )
