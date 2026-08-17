"""Turns per-minute participant_frames + timeline_events into a five-minute
prediction-horizon, multi-label weak-labeled dataset for a future state
model. See .claude/plans/smooth-munching-meerkat.md for the full design
rationale and the leakage-prevention rules this file follows:

1. Features for a window ending at minute t only use data from [0, t].
2. Labels for that window only use data from (t, t+5].
3. The baseline (P25/P50/P75) used to score every window — train AND test —
   is fit on the TRAIN split only, and persisted alongside the dataset so
   the file that gets shipped is provably the one that was actually used.
4. A match is entirely train or entirely test — never split across both,
   even when it has more than one Gold-verified subject.

5. A P-score-derived label (performance_collapse_h5/recovery_h5/
   enemy_gap_expand_h5) is only computed when every p_score it depends on
   — current and future — resolved to the SAME non-"global" baseline
   scope. Global mixes all five roles' metric distributions together, so a
   role-specific value scored against it is not on a comparable scale (a
   support's near-zero CS reads very differently against a global CS
   distribution than against a support-only one) — verified empirically,
   not just a coverage concern. Mixing scopes within one window's own
   trajectory has the same problem. Rows that don't qualify get NULL, not
   a silently-biased number.

Output is one row per (match_id, participant_id, minute) for Gold-verified
subjects, written to output_v3/patch={patch}/behavior_windows_v3.parquet.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from build_baselines import METRICS, MIN_DURATION_SECONDS, compute_baselines, load_gold_frames
from p_score import choose_baseline, score_observation

ROOT = Path(__file__).resolve().parent
DATASET_VERSION = "behavior_windows_v3"
PREDICTION_HORIZON = 5
PAST_WINDOW = 5
WARMUP_MINUTES = 3  # need at least this much history before scoring a window
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
TEAM_PAYOFF_EVENT_TYPES = {"CHAMPION_KILL", "CHAMPION_SPECIAL_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL"}
DEATH_NEARBY_WINDOW_MS = 30_000
# Rough map-scale judgment call (Summoner's Rift is ~15000x15000 units) —
# a same-team kill/objective only counts as a payoff for a given death if
# it happened in roughly the same fight, not just the same 30-second window
# anywhere on the map. Adjust if this doesn't match intuition once reviewed.
DEATH_PROXIMITY_UNITS = 3000.0
RATE_METRICS = ["total_gold", "xp", "minions_killed"]
# Percentile-point drop in the subject-vs-opponent p_score gap over the
# horizon. p_score is 0-100 with P25/P50/P75 anchored at 25/50/75 by
# construction (see p_score.piecewise_percentile), so this is on the same
# scale as the labels' own P25/P50 language — unlike a raw z-score, which
# doesn't map onto "percentile" in a way that lines up with those anchors.
ENEMY_GAP_EXPAND_THRESHOLD = -15.0


# ---------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_behavior_dataset.py)
# ---------------------------------------------------------------------------


def assign_split(puuid: str, train_fraction: float = 0.8) -> str:
    """Deterministic hash-based train/test split so reruns are stable and a
    single player's matches never straddle both sides."""
    digest = hashlib.md5(puuid.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 100
    return "train" if bucket < int(train_fraction * 100) else "test"


def find_mixed_split_matches(match_ids: list[str], splits: list[str]) -> set[str]:
    """A match with more than one Gold-verified subject can have subjects on
    both sides of the split (PUUID-level splitting doesn't prevent this).
    Such a match must be dropped entirely — not just the minority-side rows
    — otherwise it can still contribute train-side frames to the baseline
    while also having a test-side subject scored against that same
    baseline, which is the leak this exists to prevent."""
    counts = pd.Series(splits, index=pd.Index(match_ids, name="match_id")).groupby("match_id").nunique()
    return set(counts[counts > 1].index)


def label_performance_collapse(
    current_p_score: float | None, future_p_scores: list[float]
) -> bool | None:
    if current_p_score is None or not future_p_scores:
        return None
    if current_p_score < 50:
        return False
    tail = future_p_scores[-2:] if len(future_p_scores) >= 2 else future_p_scores
    return (sum(tail) / len(tail)) < 25


def label_resource_collapse(
    past_rates: dict[str, float], future_rates: dict[str, float], rate_p25: dict[str, float]
) -> bool | None:
    checked = 0
    below_baseline = 0
    down_from_past = 0
    for metric in RATE_METRICS:
        p25 = rate_p25.get(metric)
        past_rate = past_rates.get(metric)
        future_rate = future_rates.get(metric)
        if p25 is None or past_rate is None or future_rate is None:
            continue
        checked += 1
        if future_rate < p25:
            below_baseline += 1
        if future_rate < past_rate:
            down_from_past += 1
    if checked == 0:
        return None
    return below_baseline >= 2 and down_from_past >= 2


def label_enemy_gap_expand(gap_now: float | None, gap_future: float | None) -> bool | None:
    if gap_now is None or gap_future is None:
        return None
    return (gap_future - gap_now) <= ENEMY_GAP_EXPAND_THRESHOLD


def label_death_risk(future_death_count: int) -> bool:
    return future_death_count >= 1


def label_multi_death(future_death_count: int) -> bool:
    """Stricter companion to death_risk_h5 — two or more deaths in the
    horizon is a much higher-confidence risk signal than "at least one",
    which overlapping sliding windows make very common on its own."""
    return future_death_count >= 2


def _euclidean_distance(x1: float | None, y1: float | None, x2: float | None, y2: float | None) -> float | None:
    if None in (x1, y1, x2, y2):
        return None
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def label_worthless_death(
    future_deaths: list[dict[str, Any]],
    match_events: list[dict[str, Any]],
    proximity_units: float = DEATH_PROXIMITY_UNITS,
) -> bool:
    """future_deaths: [{'timestamp_ms', 'team_id', 'position_x', 'position_y'}]
    for the subject's deaths in the horizon. match_events: same-match events
    with timestamp_ms/event_type/team_id/position_x/position_y. worthless=True
    if ANY single death in the horizon lacks a same-team, nearby-in-time-AND-
    space payoff — this is a per-death AND, not an OR across all deaths."""
    if not future_deaths:
        return False
    for death in future_deaths:
        window_start = death["timestamp_ms"] - DEATH_NEARBY_WINDOW_MS
        window_end = death["timestamp_ms"] + DEATH_NEARBY_WINDOW_MS
        payoff = False
        for event in match_events:
            if event.get("team_id") != death["team_id"]:
                continue
            if event.get("event_type") not in TEAM_PAYOFF_EVENT_TYPES:
                continue
            if not (window_start <= event.get("timestamp_ms", -1) <= window_end):
                continue
            distance = _euclidean_distance(
                death.get("position_x"), death.get("position_y"),
                event.get("position_x"), event.get("position_y"),
            )
            # Missing position on either side means we cannot confirm spatial
            # proximity, so it must NOT default to counting as a payoff —
            # only a confirmed close-by event clears the death.
            if distance is not None and distance <= proximity_units:
                payoff = True
                break
        if not payoff:
            return True
    return False


def label_recovery(
    current_p_score: float | None,
    future_p_scores: list[float],
    future_resource_recovering: bool | None,
) -> bool | None:
    if current_p_score is None or len(future_p_scores) < 2 or future_resource_recovering is None:
        return None
    if current_p_score >= 25:
        return False
    tail = future_p_scores[-2:]
    return all(score >= 50 for score in tail) and future_resource_recovering


def label_activity_drop(
    window_has_death_or_purchase: bool,
    past_activity: dict[str, float],
    future_activity: dict[str, float],
) -> bool | None:
    if window_has_death_or_purchase:
        return None  # excluded window per the definition, not a negative
    checked = 0
    drops = 0
    for key in ("movement", "resource_rate", "team_participation"):
        past_v = past_activity.get(key)
        future_v = future_activity.get(key)
        if past_v is None or future_v is None:
            continue
        checked += 1
        if future_v < past_v:
            drops += 1
    if checked == 0:
        return None
    return drops == checked


# ---------------------------------------------------------------------------
# Bulk data loading
# ---------------------------------------------------------------------------

MATCHES_QUERY = """
SELECT match_id, patch, queue_id, game_start, duration_seconds
FROM matches
WHERE collection_status = 'complete' AND patch = %(target_patch)s AND queue_id = %(target_queue)s
  AND duration_seconds >= %(min_duration_seconds)s
"""

PARTICIPANTS_QUERY = """
SELECT p.match_id, p.participant_id, p.puuid, p.team_id, p.champion_id,
       COALESCE(NULLIF(p.team_position, ''), NULLIF(p.individual_position, '')) AS role
FROM participants p
JOIN matches m ON m.match_id = p.match_id
WHERE m.collection_status = 'complete' AND m.patch = %(target_patch)s AND m.queue_id = %(target_queue)s
  AND m.duration_seconds >= %(min_duration_seconds)s
"""

FRAMES_QUERY = """
SELECT pf.match_id, pf.participant_id, pf.minute, pf.timestamp_ms, pf.total_gold, pf.xp,
       pf.level, pf.minions_killed, pf.jungle_minions_killed, pf.position_x, pf.position_y
FROM participant_frames pf
JOIN matches m ON m.match_id = pf.match_id
WHERE m.collection_status = 'complete' AND m.patch = %(target_patch)s AND m.queue_id = %(target_queue)s
  AND m.duration_seconds >= %(min_duration_seconds)s
"""

EVENTS_QUERY = """
SELECT te.match_id, te.timestamp_ms, te.event_type, te.participant_id, te.killer_id,
       te.victim_id, te.team_id, te.assisting_participant_ids, te.position_x, te.position_y
FROM timeline_events te
JOIN matches m ON m.match_id = te.match_id
WHERE m.collection_status = 'complete' AND m.patch = %(target_patch)s AND m.queue_id = %(target_queue)s
  AND m.duration_seconds >= %(min_duration_seconds)s
  AND te.event_type IN (
      'CHAMPION_KILL', 'CHAMPION_SPECIAL_KILL', 'ELITE_MONSTER_KILL', 'BUILDING_KILL',
      'ITEM_PURCHASED', 'WARD_PLACED'
  )
"""


def rate(frames: pd.DataFrame, minute_from: int, minute_to: int, metric: str) -> float | None:
    start = frames.loc[frames["minute"] == minute_from, metric]
    end = frames.loc[frames["minute"] == minute_to, metric]
    if start.empty or end.empty or minute_to == minute_from:
        return None
    return float(end.iloc[0] - start.iloc[0]) / (minute_to - minute_from)


def compute_rate_baseline(train_frame: pd.DataFrame) -> pd.DataFrame:
    """P25 of each metric's per-minute growth rate, grouped by role+minute.
    Kept separate from build_baselines.compute_baselines because rate
    baselines don't need champion granularity for this purpose."""
    sorted_frame = train_frame.sort_values(["match_id", "puuid", "role", "minute"])
    rates = sorted_frame.groupby(["match_id", "puuid", "role"])[RATE_METRICS].diff()
    rates.columns = [f"{metric}_rate" for metric in RATE_METRICS]
    combined = pd.concat([sorted_frame[["role", "minute"]], rates], axis=1).dropna()
    grouped = combined.groupby(["role", "minute"])
    p25 = grouped[[f"{metric}_rate" for metric in RATE_METRICS]].quantile(0.25)
    p25.columns = [f"{metric}_rate_p25" for metric in RATE_METRICS]
    return p25.reset_index()


def p_score_for_minute(
    baseline: pd.DataFrame, role: str, champion_id: int, minute: int, frame_row
) -> tuple[float | None, pd.Series | None]:
    """Full multi-metric P-score (gold+xp+level+cs, role-appropriate
    weights) via p_score.py's own choose_baseline/score_observation —
    reused rather than re-implemented so P25/P50/P75 mean exactly what
    piecewise_percentile defines them to mean (this file previously had its
    own z-score-based formula where "P25" did not actually land at 25)."""
    try:
        baseline_row = choose_baseline(baseline, role, int(champion_id), int(minute))
    except LookupError:
        return None, None
    observation = {
        "total_gold": frame_row["total_gold"],
        "xp": frame_row["xp"],
        "level": frame_row["level"],
        "minions_killed": frame_row["minions_killed"],
        "jungle_minions_killed": frame_row["jungle_minions_killed"],
    }
    try:
        result = score_observation(baseline_row, observation, role)
    except ValueError:
        return None, None
    return result["p_score"], baseline_row


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    target_patch = os.getenv("TARGET_PATCH", "").strip()
    if not database_url or not target_patch:
        print("缺少 DATABASE_URL 或 TARGET_PATCH。", file=sys.stderr)
        return 1
    target_queue = int(os.getenv("TARGET_QUEUE_ID", "420"))
    max_rank_age_days = float(os.getenv("MAX_RANK_AGE_DAYS", "21"))

    engine = create_engine(database_url)
    try:
        params = {
            "target_patch": target_patch,
            "target_queue": target_queue,
            "min_duration_seconds": MIN_DURATION_SECONDS,
        }
        matches = pd.read_sql_query(MATCHES_QUERY, engine, params=params)
        participants = pd.read_sql_query(PARTICIPANTS_QUERY, engine, params=params)
        frames = pd.read_sql_query(FRAMES_QUERY, engine, params=params)
        events = pd.read_sql_query(EVENTS_QUERY, engine, params=params)

        # Single source of truth for "who is a rank-verified Gold subject for
        # THIS match": the same per-match observed_at/game_start window join
        # that the baseline itself is built from (load_gold_frames), instead
        # of a separate looser "have they ever been seen as Gold" query.
        gold_frame_full = load_gold_frames(
            engine, target_patch, target_queue, max_rank_age_days, MIN_DURATION_SECONDS, match_ids=None,
        )
    finally:
        engine.dispose()

    if gold_frame_full.empty:
        print("没有找到任何 Gold 验证过的参与者。", file=sys.stderr)
        return 2

    subject_participants = gold_frame_full.loc[
        gold_frame_full["role"].isin(ROLES),
        ["match_id", "participant_id", "puuid", "role", "champion_id"],
    ].drop_duplicates().reset_index(drop=True)

    subject_participants["split"] = subject_participants["puuid"].map(assign_split)
    mixed_match_ids = find_mixed_split_matches(
        subject_participants["match_id"].tolist(), subject_participants["split"].tolist()
    )
    excluded_subject_rows = len(subject_participants[subject_participants["match_id"].isin(mixed_match_ids)])
    if mixed_match_ids:
        subject_participants = subject_participants[
            ~subject_participants["match_id"].isin(mixed_match_ids)
        ].reset_index(drop=True)

    train_match_ids = subject_participants.loc[
        subject_participants["split"] == "train", "match_id"
    ].unique().tolist()
    train_gold_frame = gold_frame_full[gold_frame_full["match_id"].isin(train_match_ids)]

    baseline = compute_baselines(
        train_gold_frame, min_champion_samples=40, min_role_samples=400, min_global_samples=800
    )
    rate_baseline = compute_rate_baseline(train_gold_frame)
    rate_baseline_lookup = {
        (row["role"], int(row["minute"])): {m: row[f"{m}_rate_p25"] for m in RATE_METRICS}
        for _, row in rate_baseline.iterrows()
    }

    matches_by_id = matches.set_index("match_id")
    frames_by_match = {mid: g for mid, g in frames.groupby("match_id")}
    events_by_match = {mid: g for mid, g in events.groupby("match_id")}
    participants_by_match = {mid: g for mid, g in participants.groupby("match_id")}

    rows: list[dict[str, Any]] = []
    for _, subject in subject_participants.iterrows():
        match_id = subject["match_id"]
        if match_id not in matches_by_id.index:
            continue
        match_frames = frames_by_match.get(match_id)
        match_events = events_by_match.get(match_id, pd.DataFrame()).to_dict("records")
        match_participants = participants_by_match.get(match_id)
        if match_frames is None or match_participants is None:
            continue

        subject_pid = subject["participant_id"]
        subject_role = subject["role"]
        subject_champion = subject["champion_id"]
        subject_team = match_participants.loc[
            match_participants["participant_id"] == subject_pid, "team_id"
        ].iloc[0]
        subject_frames = match_frames[match_frames["participant_id"] == subject_pid].sort_values("minute")
        max_minute = int(subject_frames["minute"].max()) if not subject_frames.empty else 0

        # CHAMPION_KILL/ELITE_MONSTER_KILL rows never carry team_id in Riot's
        # raw payload (only BUILDING_KILL does) — resolve it from the
        # killer's own team. NOTE: events was loaded as one query mixing
        # event types that do and don't carry team_id, so pandas upcast the
        # column to float64 — missing values arrive as NaN, not Python None,
        # so this must use pd.isna(), not `is None`.
        pid_to_team = dict(zip(match_participants["participant_id"], match_participants["team_id"]))
        for event in match_events:
            if pd.isna(event.get("team_id")) and not pd.isna(event.get("killer_id")):
                event["team_id"] = pid_to_team.get(event["killer_id"])

        opponent = match_participants[
            (match_participants["team_id"] != subject_team) & (match_participants["role"] == subject_role)
        ]
        opponent_pid = opponent.iloc[0]["participant_id"] if not opponent.empty else None
        opponent_champion = opponent.iloc[0]["champion_id"] if not opponent.empty else None
        opponent_frames = (
            match_frames[match_frames["participant_id"] == opponent_pid].sort_values("minute")
            if opponent_pid is not None else pd.DataFrame()
        )

        teammate_frames = match_frames[
            match_frames["participant_id"].isin(
                match_participants.loc[match_participants["team_id"] == subject_team, "participant_id"]
            )
            & (match_frames["participant_id"] != subject_pid)
        ]

        # Riot's raw Timeline occasionally repeats the exact same
        # CHAMPION_KILL event twice within a frame (confirmed: 1 instance in
        # 214,921 kill events across this dataset) — a real death can't
        # happen twice at the same timestamp, so dedupe defensively to keep
        # multi_death_h5 from double-counting a single death.
        seen_death_timestamps: set[int] = set()
        deaths = []
        for e in match_events:
            if e.get("event_type") != "CHAMPION_KILL" or e.get("victim_id") != subject_pid:
                continue
            ts = int(e["timestamp_ms"])
            if ts in seen_death_timestamps:
                continue
            seen_death_timestamps.add(ts)
            deaths.append(
                {
                    "timestamp_ms": ts,
                    "team_id": subject_team,
                    "position_x": e.get("position_x"),
                    "position_y": e.get("position_y"),
                }
            )
        purchases = [e for e in match_events if e.get("event_type") == "ITEM_PURCHASED" and e.get("participant_id") == subject_pid]

        for t in range(WARMUP_MINUTES, max_minute - PREDICTION_HORIZON + 1):
            past_start = max(0, t - PAST_WINDOW)
            current_row = subject_frames[subject_frames["minute"] == t]
            if current_row.empty:
                continue
            current = current_row.iloc[0]

            future_frames = subject_frames[(subject_frames["minute"] > t) & (subject_frames["minute"] <= t + PREDICTION_HORIZON)]
            if future_frames.empty:
                continue

            current_p_score, baseline_row = p_score_for_minute(baseline, subject_role, subject_champion, t, current)
            if baseline_row is None:
                continue  # no usable baseline for this role/champion/minute at all
            current_scope = baseline_row["scope"]

            future_p_scores = []
            future_scopes = []
            for _, f in future_frames.iterrows():
                p, row = p_score_for_minute(baseline, subject_role, subject_champion, int(f["minute"]), f)
                if p is not None:
                    future_p_scores.append(p)
                    future_scopes.append(row["scope"])

            # performance_collapse_h5/recovery_h5 require every p_score used
            # — current and future — to share the SAME non-"global" scope.
            # global mixes all roles' metric distributions, so a role's
            # value scored against it isn't comparable to the same role's
            # value scored against its own role/champion_role tier; mixing
            # scopes within one trajectory has the same comparability
            # problem. Passing [] reuses each label function's own
            # "not enough future points" -> None handling.
            trajectory_scope_ok = (
                bool(future_scopes)
                and current_scope != "global"
                and all(s == current_scope for s in future_scopes)
            )
            scored_future_p_scores = future_p_scores if trajectory_scope_ok else []

            past_rates = {m: rate(subject_frames, past_start, t, m) for m in RATE_METRICS}
            future_rates = {m: rate(subject_frames, t, t + PREDICTION_HORIZON, m) for m in RATE_METRICS}
            rate_p25 = rate_baseline_lookup.get((subject_role, t), {})

            gap_now = gap_future = None
            if not opponent_frames.empty:
                opp_now = opponent_frames[opponent_frames["minute"] == t]
                opp_future = opponent_frames[opponent_frames["minute"] == t + PREDICTION_HORIZON]
                subj_future_row = subject_frames[subject_frames["minute"] == t + PREDICTION_HORIZON]
                opp_now_scope = opp_future_scope = subj_future_scope = None
                if not opp_now.empty:
                    # Opponent is scored against their OWN champion's
                    # baseline — champion_role baselines are champion-
                    # specific, reusing subject_champion would corrupt it.
                    opp_p_score, opp_now_row = p_score_for_minute(baseline, subject_role, opponent_champion, t, opp_now.iloc[0])
                    if opp_p_score is not None:
                        gap_now = current_p_score - opp_p_score
                        opp_now_scope = opp_now_row["scope"]
                if not opp_future.empty and not subj_future_row.empty:
                    subj_future_p, subj_future_row_b = p_score_for_minute(baseline, subject_role, subject_champion, t + PREDICTION_HORIZON, subj_future_row.iloc[0])
                    opp_future_p, opp_future_row_b = p_score_for_minute(baseline, subject_role, opponent_champion, t + PREDICTION_HORIZON, opp_future.iloc[0])
                    if subj_future_p is not None and opp_future_p is not None:
                        gap_future = subj_future_p - opp_future_p
                        subj_future_scope = subj_future_row_b["scope"]
                        opp_future_scope = opp_future_row_b["scope"]
                # Each side's own now/future trajectory must be internally
                # scope-consistent and non-global (subject and opponent are
                # NOT required to share a tier — p_score is comparable
                # across tiers by construction, it's mixing WITHIN one
                # side's own trajectory that breaks comparability).
                gap_scope_ok = (
                    gap_now is not None and gap_future is not None
                    and current_scope not in (None, "global") and current_scope == subj_future_scope
                    and opp_now_scope not in (None, "global") and opp_now_scope == opp_future_scope
                )
                if not gap_scope_ok:
                    gap_now = gap_future = None

            past_death_count = len([d for d in deaths if d["timestamp_ms"] <= current["timestamp_ms"]])
            future_deaths = [d for d in deaths if current["timestamp_ms"] < d["timestamp_ms"] <= current["timestamp_ms"] + PREDICTION_HORIZON * 60_000]
            has_purchase_in_window = any(
                current["timestamp_ms"] < p["timestamp_ms"] <= current["timestamp_ms"] + PREDICTION_HORIZON * 60_000
                for p in purchases
            )

            # Resource "recovery" means beating the role-minute P25 growth
            # rate, not merely being non-negative — cumulative stats
            # (gold/xp/cs) almost never go down, so ">=0" was true almost by
            # definition and added no real signal (verified: 0/71,494 rows
            # had a negative past_gold_rate in v1).
            future_resource_recovering = None
            if future_rates and rate_p25:
                checked = above_p25 = 0
                for m in RATE_METRICS:
                    p25 = rate_p25.get(m)
                    fr = future_rates.get(m)
                    if p25 is None or fr is None:
                        continue
                    checked += 1
                    if fr >= p25:
                        above_p25 += 1
                if checked > 0:
                    future_resource_recovering = above_p25 >= 2

            def movement(frame_slice: pd.DataFrame) -> float | None:
                if len(frame_slice) < 2:
                    return None
                dx = frame_slice["position_x"].diff().abs()
                dy = frame_slice["position_y"].diff().abs()
                return float((dx + dy).sum())

            past_frame_slice = subject_frames[(subject_frames["minute"] >= past_start) & (subject_frames["minute"] <= t)]
            future_frame_slice = subject_frames[(subject_frames["minute"] > t) & (subject_frames["minute"] <= t + PREDICTION_HORIZON)]
            past_team_participation = len([
                e for e in match_events
                if e.get("event_type") in TEAM_PAYOFF_EVENT_TYPES
                and (e.get("killer_id") == subject_pid or e.get("participant_id") == subject_pid or subject_pid in (e.get("assisting_participant_ids") or []))
                and current["timestamp_ms"] - PAST_WINDOW * 60_000 <= e["timestamp_ms"] <= current["timestamp_ms"]
            ])
            future_team_participation = len([
                e for e in match_events
                if e.get("event_type") in TEAM_PAYOFF_EVENT_TYPES
                and (e.get("killer_id") == subject_pid or e.get("participant_id") == subject_pid or subject_pid in (e.get("assisting_participant_ids") or []))
                and current["timestamp_ms"] < e["timestamp_ms"] <= current["timestamp_ms"] + PREDICTION_HORIZON * 60_000
            ])

            past_activity = {
                "movement": movement(past_frame_slice),
                "resource_rate": past_rates.get("total_gold"),
                "team_participation": past_team_participation,
            }
            future_activity = {
                "movement": movement(future_frame_slice),
                "resource_rate": future_rates.get("total_gold"),
                "team_participation": future_team_participation,
            }
            window_has_death_or_purchase = bool(future_deaths) or has_purchase_in_window

            # Current-minute teammate average is a legitimate [0,t] feature
            # (team economy context). A future-minute version would leak —
            # deliberately not computed.
            teammate_avg_gold_now = None
            current_teammate_frames = teammate_frames[teammate_frames["minute"] == t]
            if not current_teammate_frames.empty:
                teammate_avg_gold_now = float(current_teammate_frames["total_gold"].mean())

            rows.append(
                {
                    "match_id": match_id,
                    "participant_id": int(subject_pid),
                    "puuid": subject["puuid"],
                    "patch": target_patch,
                    "queue_id": target_queue,
                    "role": subject_role,
                    "champion_id": int(subject_champion) if pd.notna(subject_champion) else None,
                    "feature_cutoff_minute": t,
                    "prediction_horizon": PREDICTION_HORIZON,
                    "split": subject["split"],
                    "rank_verified": True,
                    "current_p_score": round(current_p_score, 2),
                    "current_gap_z_vs_lane_opponent": gap_now,
                    "past_gold_rate": past_rates.get("total_gold"),
                    "past_xp_rate": past_rates.get("xp"),
                    "past_cs_rate": past_rates.get("minions_killed"),
                    "past_movement": past_activity["movement"],
                    "past_team_participation": past_team_participation,
                    "past_death_count": past_death_count,
                    "teammate_avg_gold_now": teammate_avg_gold_now,
                    "baseline_scope": baseline_row["scope"],
                    "baseline_fallback_level": int(baseline_row["fallback_level"]),
                    "baseline_n": int(baseline_row["sample_count"]),
                    "baseline_match_n": int(baseline_row["match_n"]),
                    "baseline_player_n": int(baseline_row["player_n"]),
                    "baseline_patch_scope": target_patch,
                    "performance_collapse_h5": label_performance_collapse(current_p_score, scored_future_p_scores),
                    "resource_collapse_h5": label_resource_collapse(past_rates, future_rates, rate_p25),
                    "enemy_gap_expand_h5": label_enemy_gap_expand(gap_now, gap_future),
                    "death_risk_h5": label_death_risk(len(future_deaths)),
                    "multi_death_h5": label_multi_death(len(future_deaths)),
                    "worthless_death_h5": label_worthless_death(future_deaths, match_events),
                    "recovery_h5": label_recovery(current_p_score, scored_future_p_scores, future_resource_recovering),
                    "activity_drop_h5": label_activity_drop(window_has_death_or_purchase, past_activity, future_activity),
                }
            )

    result = pd.DataFrame(rows)
    label_columns = [c for c in result.columns if c.endswith("_h5")]
    # Force a clean nullable-boolean dtype right away — a freshly-built
    # DataFrame from a list of dicts can leave True/False/None columns as
    # unconsolidated object blocks, where pandas' own .mean() has been
    # observed to silently misbehave.
    for col in label_columns:
        result[col] = result[col].astype("boolean")

    out_dir = ROOT / f"output_v3/patch={target_patch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "behavior_windows_v3.parquet"
    result.to_parquet(dataset_path, index=False, compression="zstd")

    baselines_dir = out_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = baselines_dir / "gold_quantiles_train_v3.parquet"
    rate_baseline_path = baselines_dir / "rate_quantiles_train_v3.parquet"
    baseline.to_parquet(baseline_path, index=False, compression="zstd")
    rate_baseline.to_parquet(rate_baseline_path, index=False, compression="zstd")

    print(f"target_patch：{target_patch}")
    print(f"mixed-split 比赛已排除：{len(mixed_match_ids)} 场，{excluded_subject_rows} 个 subject 候选")
    print(f"subject 参与者数：{len(subject_participants):,}")
    print(f"输出窗口行数：{len(result):,}")
    print("split 分布：")
    print(result["split"].value_counts().to_string())
    assert result.groupby("match_id")["split"].nunique().max() <= 1, "mixed-split match leaked into output"
    print("\nbaseline_scope 分布（按 role）：")
    print((100 * result.groupby("role")["baseline_scope"].value_counts(normalize=True)).round(1).to_string())
    print("\ncurrent_p_score 中位数（按 role）：")
    print(result.groupby("role")["current_p_score"].median().round(2).to_string())
    print("\n标签正例率（忽略 NULL）：")
    label_rates = {}
    for col in label_columns:
        valid = result[col].dropna()
        rate_pct = 100 * valid.mean() if len(valid) else float("nan")
        n_null = int(result[col].isna().sum())
        label_rates[col] = {"positive_pct": round(float(rate_pct), 2), "n_valid": int(len(valid)), "n_null": n_null}
        print(f"  {col}: {rate_pct:.1f}% (n={len(valid):,}, null={n_null:,})")
    print(f"数据集：{dataset_path}")
    print(f"train baseline：{baseline_path}")
    print(f"rate baseline：{rate_baseline_path}")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "git_note": "project directory is not a git repository; no commit hash available",
        "patch": target_patch,
        "queue_id": target_queue,
        "match_n": int(result["match_id"].nunique()),
        "window_n": int(len(result)),
        "column_n": int(len(result.columns)),
        "subject_player_n": int(result["puuid"].nunique()),
        "train_window_n": int((result["split"] == "train").sum()),
        "test_window_n": int((result["split"] == "test").sum()),
        "mixed_split_matches_excluded": len(mixed_match_ids),
        "mixed_split_excluded_subject_rows": excluded_subject_rows,
        "split_strategy": "deterministic md5(puuid) hash bucketing, 80/20 train/test; matches with subjects on both sides are dropped entirely (see mixed_split_matches_excluded)",
        "baseline_file": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "baseline_file_sha256": sha256_of(baseline_path),
        "rate_baseline_file": str(rate_baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "rate_baseline_file_sha256": sha256_of(rate_baseline_path),
        "baseline_fit_scope": "TRAIN split only, computed in this same run and saved to the two files above — baseline_match_n/player_n in the dataset are guaranteed to match those files exactly (verify via the SHA-256)",
        "baseline_thresholds": {"min_champion_samples": 40, "min_role_samples": 400, "min_global_samples": 800},
        "baseline_scope_by_role_pct": {
            role: {k: round(float(v), 2) for k, v in sub.items()}
            for role, sub in (100 * result.groupby("role")["baseline_scope"].value_counts(normalize=True)).unstack(fill_value=0).to_dict("index").items()
        },
        "current_p_score_median_by_role": result.groupby("role")["current_p_score"].median().round(2).to_dict(),
        "empirical_unvalidated_parameters": {
            "DEATH_PROXIMITY_UNITS": {"value": DEATH_PROXIMITY_UNITS, "unit": "map position units (Summoner's Rift ~15000x15000)", "note": "judgment-call heuristic, not a validated map-mechanic constant"},
            "DEATH_NEARBY_WINDOW_MS": {"value": DEATH_NEARBY_WINDOW_MS, "unit": "milliseconds", "note": "judgment call, not validated"},
            "ENEMY_GAP_EXPAND_THRESHOLD": {"value": ENEMY_GAP_EXPAND_THRESHOLD, "unit": "p_score percentile points", "note": "judgment call, not validated"},
        },
        "label_definitions": {
            "performance_collapse_h5": "current multi-metric P-score >= P50 AND mean of last 2 future minutes < P25; NULL unless every p_score used (current + future) shares the same non-global baseline scope",
            "resource_collapse_h5": "future Gold/XP/CS growth rate: >=2 of 3 metrics below role-minute P25 rate baseline AND >=2 below their own past-5-min rate",
            "enemy_gap_expand_h5": f"subject-vs-lane-opponent P-score gap widens by >= {abs(ENEMY_GAP_EXPAND_THRESHOLD)} percentile points over the horizon; NULL when no lane opponent could be matched, or when either side's own now/future p_score scope isn't consistent and non-global",
            "death_risk_h5": "at least 1 death in (t, t+5]",
            "multi_death_h5": "at least 2 deaths in (t, t+5]",
            "worthless_death_h5": "at least one death in (t, t+5] with no same-team payoff within +-30s AND <=3000 units; missing coordinates never default to counting as a payoff",
            "recovery_h5": "current P-score < P25 AND last 2 future minutes both >= P50 AND >=2/3 future resource rates >= role-minute P25 rate; same scope-consistency NULL rule as performance_collapse_h5",
            "activity_drop_h5": "EXPERIMENTAL — excludes windows with a death or purchase event, then requires movement + resource rate + team participation to ALL drop vs. the player's own past 5 minutes",
        },
        "label_positive_rates": label_rates,
        "known_limitations": [
            "activity_drop_h5 sample size is small (windows with neither a death nor a purchase are inherently uncommon) — not recommended as a core training head",
            "champion_role baseline tier still covers a minority of (role, champion, minute) combinations at the lowered n>=40 threshold; most fall back to role or global",
            "TOP/UTILITY/JUNGLE roles fall back to the global baseline tier more often than BOTTOM/MIDDLE (see baseline_scope_by_role_pct); performance_collapse_h5/recovery_h5/enemy_gap_expand_h5 are NULL for those rows rather than scored against the not-comparable global scale — this trades sample size for correctness, more collection is the actual fix for the underlying coverage gap",
        ],
        "tests_passing": None,
        "predecessor": "behavior_windows_v2 — superseded due to 3 confirmed issues found in a second independent review: one-directional rank-verification time window silently rejecting valid pre-match snapshots (verified 260/260 rejected pairs were observed BEFORE game_start, not stale), a tie-handling bug in piecewise_percentile that mis-scored degenerate quantile distributions, and P-score-derived labels being computed against a role-mixed global baseline that isn't comparable to role-specific values",
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest：{manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
