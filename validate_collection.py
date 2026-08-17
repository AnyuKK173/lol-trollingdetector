from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def scalar(cursor, sql: str) -> int:
    cursor.execute(sql)
    return int(cursor.fetchone()[0])


def main() -> int:
    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("缺少 DATABASE_URL。", file=sys.stderr)
        return 1

    connection = psycopg2.connect(database_url)
    try:
        with connection.cursor() as cursor:
            counts = {}
            for table in (
                "rank_snapshots",
                "matches",
                "participants",
                "participant_frames",
                "timeline_events",
            ):
                counts[table] = scalar(cursor, f"SELECT COUNT(*) FROM {table}")

            complete = scalar(
                cursor,
                "SELECT COUNT(*) FROM matches WHERE collection_status = 'complete'",
            )
            failed = scalar(
                cursor,
                "SELECT COUNT(*) FROM matches WHERE collection_status = 'failed'",
            )
            bad_queue = scalar(
                cursor,
                """
                SELECT COUNT(*) FROM matches
                WHERE collection_status = 'complete' AND queue_id <> 420
                """,
            )
            bad_metadata = scalar(
                cursor,
                """
                SELECT COUNT(*) FROM matches
                WHERE collection_status = 'complete'
                  AND (patch IS NULL OR duration_seconds IS NULL OR duration_seconds <= 0)
                """,
            )
            bad_participant_matches = scalar(
                cursor,
                """
                SELECT COUNT(*) FROM (
                    SELECT m.match_id
                    FROM matches m
                    LEFT JOIN participants p ON p.match_id = m.match_id
                    WHERE m.collection_status = 'complete'
                    GROUP BY m.match_id
                    HAVING COUNT(p.participant_id) <> 10
                ) bad
                """,
            )
            participants_without_frames = scalar(
                cursor,
                """
                SELECT COUNT(*) FROM (
                    SELECT p.match_id, p.participant_id
                    FROM participants p
                    JOIN matches m ON m.match_id = p.match_id
                    LEFT JOIN participant_frames f
                      ON f.match_id = p.match_id
                     AND f.participant_id = p.participant_id
                    WHERE m.collection_status = 'complete'
                    GROUP BY p.match_id, p.participant_id
                    HAVING COUNT(f.minute) = 0
                ) bad
                """,
            )

            cursor.execute(
                """
                SELECT patch, COUNT(*)
                FROM matches
                WHERE collection_status = 'complete'
                GROUP BY patch ORDER BY patch DESC
                """
            )
            patches = cursor.fetchall()

            print("数据表行数：")
            for table, count in counts.items():
                print(f"  {table}: {count:,}")
            print(f"\n比赛状态：complete={complete:,}, failed={failed:,}")
            print("版本分布：" + ", ".join(f"{p}={n}" for p, n in patches))
            print("\n硬性质量检查：")
            print(f"  complete 但 queue_id != 420: {bad_queue}")
            print(f"  缺 patch 或有效 duration: {bad_metadata}")
            print(f"  参与者数量不是 10 的比赛: {bad_participant_matches}")
            print(f"  没有任何局内帧的参与者: {participants_without_frames}")

            severe = (
                bad_queue
                + bad_metadata
                + bad_participant_matches
                + participants_without_frames
            )
            if complete == 0:
                print("\n结论：还没有完整比赛。", file=sys.stderr)
                return 2
            if severe:
                print("\n结论：数据未通过硬性检查。", file=sys.stderr)
                return 3
            print("\n结论：采集数据通过硬性检查。failed 记录可在下次运行时重试。")
            return 0
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
