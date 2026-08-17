from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from parsers import (
    check_patch_match,
    parse_match_metadata,
    parse_participant_frames,
    parse_participants,
    parse_timeline_events,
    rank_drift_days,
)
from riot_api import RiotAPI
from storage import Storage


LOGGER = logging.getLogger("riot_gold_collector")
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    api_key: str
    platform: str
    region: str
    database_url: str
    raw_dir: Path
    request_interval_seconds: float
    max_retries: int
    target_patch: str
    target_queue_id: int
    target_verified_matches: int
    target_patch_start: int | None
    max_rank_age_days: float
    no_progress_warning_seconds: float

    @classmethod
    def from_environment(cls, require_api_key: bool = True) -> "Settings":
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("RIOT_API_KEY", "").strip()
        if require_api_key and (not api_key or api_key == "RGAPI-replace-me"):
            raise ValueError("请在 .env 中设置有效的 RIOT_API_KEY。")

        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError("请在 .env 中设置 DATABASE_URL。")

        raw_dir_value = os.getenv("RAW_DIR", "./raw_samples")
        raw_dir = Path(raw_dir_value)
        if not raw_dir.is_absolute():
            raw_dir = ROOT / raw_dir

        target_patch = os.getenv("TARGET_PATCH", "").strip()
        if require_api_key and not target_patch:
            # Deliberately no hardcoded fallback (e.g. "current version") —
            # locking the patch is a decision the operator has to make.
            raise ValueError("请在 .env 中设置 TARGET_PATCH（例如 16.14）。")

        patch_start_value = os.getenv("TARGET_PATCH_START", "").strip()
        target_patch_start: int | None = None
        if patch_start_value:
            target_patch_start = int(
                datetime.fromisoformat(patch_start_value.replace("Z", "+00:00")).timestamp()
            )

        return cls(
            api_key=api_key,
            platform=os.getenv("RIOT_PLATFORM", "NA1").upper(),
            region=os.getenv("RIOT_REGION", "AMERICAS").upper(),
            database_url=database_url,
            raw_dir=raw_dir.resolve(),
            request_interval_seconds=float(
                os.getenv("REQUEST_INTERVAL_SECONDS", "1.25")
            ),
            max_retries=int(os.getenv("MAX_RETRIES", "5")),
            target_patch=target_patch,
            target_queue_id=int(os.getenv("TARGET_QUEUE_ID", "420")),
            target_verified_matches=int(os.getenv("TARGET_VERIFIED_MATCHES", "500")),
            target_patch_start=target_patch_start,
            max_rank_age_days=float(os.getenv("MAX_RANK_AGE_DAYS", "21")),
            no_progress_warning_seconds=float(
                os.getenv("NO_PROGRESS_WARNING_SECONDS", "300")
            ),
        )


def save_raw_pair(
    raw_dir: Path,
    match_id: str,
    match: dict[str, Any],
    timeline: dict[str, Any],
) -> None:
    targets = (
        (raw_dir / "matches" / f"{match_id}.json.gz", match),
        (raw_dir / "timelines" / f"{match_id}.json.gz", timeline),
    )
    for target, payload in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def select_gold_entries(
    api: RiotAPI,
    divisions: list[str],
    players_per_division: int,
    max_pages: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for division in divisions:
        division_entries: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            page_entries = api.get_gold_entries(division, page=page)
            if not page_entries:
                break
            division_entries.extend(page_entries)

        random.Random(f"{seed}:{division}").shuffle(division_entries)
        chosen = division_entries[:players_per_division]
        LOGGER.info("Gold %s：选中 %s 名玩家", division, len(chosen))
        selected.extend(chosen)

    # A player should only appear once, but dedupe defensively.
    unique: dict[str, dict[str, Any]] = {}
    for entry in selected:
        puuid = entry.get("puuid")
        if puuid:
            unique.setdefault(puuid, entry)
    return list(unique.values())


class ProgressWatchdog:
    """Tracks wall-clock time since the last confirmed unit of work and logs
    a visibility warning if the collector goes quiet for too long. This does
    not cancel in-flight requests — it only makes silent stalls observable."""

    def __init__(self, warning_seconds: float) -> None:
        self.warning_seconds = warning_seconds
        self._last_progress_at = time.monotonic()
        self._warned = False

    def note(self) -> None:
        self._last_progress_at = time.monotonic()
        self._warned = False

    def check(self, context: str) -> None:
        idle = time.monotonic() - self._last_progress_at
        if idle > self.warning_seconds and not self._warned:
            LOGGER.warning(
                "已 %.0f 秒没有采集进展（%s），可能是网络卡住", idle, context
            )
            self._warned = True


def collect(args: argparse.Namespace) -> int:
    if args.players_per_division < 1 or args.pages < 1:
        raise ValueError("--players-per-division 和 --pages 必须大于 0。")
    if not 1 <= args.matches_per_player <= 100:
        raise ValueError("--matches-per-player 必须在 1 到 100 之间。")
    if args.raw_sample_limit < 0 or args.stop_after_matches < 0:
        raise ValueError("sample/stop 参数不能为负数。")

    settings = Settings.from_environment(require_api_key=not args.schema_only)
    storage = Storage(settings.database_url)
    try:
        storage.apply_schema(ROOT / "schema.sql")
        LOGGER.info("数据库 schema 已就绪")
        if args.schema_only:
            return 0

        already_accepted = storage.count_complete_matches(
            settings.target_patch, settings.target_queue_id
        )
        if already_accepted >= settings.target_verified_matches:
            LOGGER.info(
                "目标已达成：patch=%s 已有 %s/%s 场完成，无需继续采集",
                settings.target_patch,
                already_accepted,
                settings.target_verified_matches,
            )
            return 0

        api = RiotAPI(
            settings.api_key,
            settings.platform,
            settings.region,
            settings.request_interval_seconds,
            settings.max_retries,
        )
        entries = select_gold_entries(
            api,
            divisions=args.divisions,
            players_per_division=args.players_per_division,
            max_pages=args.pages,
            seed=args.seed,
        )
        if not entries:
            LOGGER.error("没有拿到 Gold 玩家。请检查平台路由、API Key 和限流状态。")
            return 2

        discovered_at = datetime.now(timezone.utc)
        snapshot_count = storage.save_rank_snapshots(entries, discovered_at)
        LOGGER.info("已写入 %s 条段位快照", snapshot_count)
        new_subjects = storage.upsert_collection_subjects(entries, discovered_at)
        LOGGER.info("本次新发现 %s 名玩家（已有的玩家保留原有进度）", new_subjects)

        run_id = storage.start_collection_run(
            settings.target_patch, settings.target_queue_id,
            settings.target_verified_matches,
        )

        subjects = storage.fetch_pending_subjects(limit=args.subject_batch_limit)
        LOGGER.info("本轮工作队列：%s 名待处理/在办玩家", len(subjects))

        accepted = 0
        patch_mismatch = 0
        rank_drift_skipped = 0
        failed = 0
        raw_saved = 0
        seen_match_ids: set[str] = set()
        watchdog = ProgressWatchdog(settings.no_progress_warning_seconds)

        def target_reached() -> bool:
            return already_accepted + accepted >= settings.target_verified_matches

        for subject in subjects:
            if target_reached():
                break
            puuid = subject["puuid"]
            watchdog.check(f"subject={puuid}")
            try:
                match_ids = api.get_match_ids(
                    puuid, args.matches_per_player, start_time=settings.target_patch_start
                )
                watchdog.note()
            except Exception as exc:
                failed += 1
                LOGGER.exception("玩家比赛列表拉取失败：%s", exc)
                storage.update_subject_progress(
                    puuid, 0, "failed", datetime.now(timezone.utc)
                )
                continue

            subject_accepted = 0
            for match_id in match_ids:
                if target_reached():
                    break
                watchdog.check(f"subject={puuid} match={match_id}")
                if match_id in seen_match_ids:
                    continue
                seen_match_ids.add(match_id)

                if storage.is_complete(match_id):
                    watchdog.note()
                    continue

                metadata: dict[str, Any] | None = None
                try:
                    match = api.get_match(match_id)
                    watchdog.note()
                    metadata = parse_match_metadata(match, settings.region)
                    metadata["match_id"] = metadata.get("match_id") or match_id
                    if metadata["match_id"] != match_id:
                        raise ValueError(
                            f"Match payload ID={metadata['match_id']} 与请求 ID={match_id} 不一致"
                        )

                    mismatch_reason = check_patch_match(
                        metadata["patch"], settings.target_patch
                    )
                    if mismatch_reason:
                        storage.mark_skipped(
                            match_id, settings.region, mismatch_reason, metadata
                        )
                        patch_mismatch += 1
                        watchdog.note()
                        continue  # deliberately skip the Timeline request

                    if metadata["queue_id"] != settings.target_queue_id:
                        storage.mark_skipped(
                            match_id,
                            settings.region,
                            f"queue_mismatch:expected={settings.target_queue_id}"
                            f":actual={metadata['queue_id']}",
                            metadata,
                        )
                        watchdog.note()
                        continue

                    drift_days = rank_drift_days(discovered_at, metadata["game_start"])
                    if drift_days is not None and drift_days > settings.max_rank_age_days:
                        storage.mark_skipped(
                            match_id,
                            settings.region,
                            f"rank_drift:{drift_days:.1f}d>{settings.max_rank_age_days}d",
                            metadata,
                        )
                        rank_drift_skipped += 1
                        watchdog.note()
                        continue

                    participants = parse_participants(match)
                    if len(participants) != 10:
                        raise ValueError(f"参与者数量为 {len(participants)}，预期为 10")

                    timeline = api.get_timeline(match_id)
                    watchdog.note()
                    timeline_match_id = (timeline.get("metadata") or {}).get("matchId")
                    if timeline_match_id and timeline_match_id != match_id:
                        raise ValueError(
                            f"Timeline payload ID={timeline_match_id} 与请求 ID={match_id} 不一致"
                        )

                    frames = parse_participant_frames(timeline, match_id)
                    events = parse_timeline_events(timeline, match_id)
                    if not frames:
                        raise ValueError("timeline 中没有 participantFrames")

                    if raw_saved < args.raw_sample_limit:
                        save_raw_pair(settings.raw_dir, match_id, match, timeline)
                        raw_saved += 1

                    storage.save_match_bundle(metadata, participants, frames, events)
                    watchdog.note()
                    accepted += 1
                    subject_accepted += 1
                    LOGGER.info(
                        "target_patch=%s verified=%s/%s patch_mismatch=%s "
                        "failed=%s match=%s",
                        settings.target_patch,
                        already_accepted + accepted,
                        settings.target_verified_matches,
                        patch_mismatch,
                        failed,
                        match_id,
                    )
                except Exception as exc:
                    failed += 1
                    LOGGER.exception("采集 %s 失败：%s", match_id, exc)
                    try:
                        storage.mark_failed(match_id, settings.region, str(exc), metadata)
                    except Exception:
                        LOGGER.exception("写入失败状态也失败：%s", match_id)
                    watchdog.note()

                if args.stop_after_matches and accepted >= args.stop_after_matches:
                    LOGGER.info(
                        "达到 --stop-after-matches=%s，停止", args.stop_after_matches
                    )
                    storage.update_subject_progress(
                        puuid, subject_accepted, "in_progress", datetime.now(timezone.utc)
                    )
                    storage.finish_collection_run(run_id, accepted, patch_mismatch, failed)
                    LOGGER.info(
                        "汇总：verified=%s/%s(target) patch_mismatch=%s "
                        "rank_drift=%s failed=%s",
                        already_accepted + accepted,
                        settings.target_verified_matches,
                        patch_mismatch,
                        rank_drift_skipped,
                        failed,
                    )
                    return 0

            storage.update_subject_progress(
                puuid, subject_accepted, "exhausted", datetime.now(timezone.utc)
            )

        storage.finish_collection_run(run_id, accepted, patch_mismatch, failed)
        LOGGER.info(
            "汇总：verified=%s/%s(target) patch_mismatch=%s rank_drift=%s failed=%s",
            already_accepted + accepted,
            settings.target_verified_matches,
            patch_mismatch,
            rank_drift_skipped,
            failed,
        )
        return 0 if accepted or already_accepted else 3
    finally:
        storage.close()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="采集 Gold I-IV 单双排 Match-V5、Timeline 与段位快照。"
    )
    parser.add_argument(
        "--divisions",
        nargs="+",
        choices=["I", "II", "III", "IV"],
        default=["I", "II", "III", "IV"],
    )
    parser.add_argument("--players-per-division", type=int, default=25)
    parser.add_argument("--matches-per-player", type=int, default=10)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-sample-limit", type=int, default=5)
    parser.add_argument(
        "--subject-batch-limit",
        type=int,
        default=300,
        help="本轮从 collection_subjects 里取多少名待处理玩家。",
    )
    parser.add_argument(
        "--stop-after-matches",
        type=int,
        default=0,
        help="测试用：本次运行成功写入指定比赛数后停止；0 表示不限（只受 TARGET_VERIFIED_MATCHES 约束）。",
    )
    parser.add_argument("--schema-only", action="store_true")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        return collect(build_argument_parser().parse_args())
    except (ValueError, OSError) as exc:
        LOGGER.error("启动失败：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
