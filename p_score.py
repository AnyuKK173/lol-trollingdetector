from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_baselines import SCOPE_FALLBACK_LEVEL


LANE_WEIGHTS = {
    "total_gold": 0.35,
    "xp": 0.25,
    "level": 0.15,
    "minions_killed": 0.25,
}
JUNGLE_WEIGHTS = {
    "total_gold": 0.35,
    "xp": 0.25,
    "level": 0.15,
    "jungle_minions_killed": 0.25,
}


def piecewise_percentile(value: float, p25: float, p50: float, p75: float) -> float:
    """Map three empirical quantiles to a clipped 0-100 percentile-like score.

    Branches are checked top-down with >=, not bottom-up with <=: when
    quantiles are tied (common with thin samples or discrete metrics like
    level), checking value<=p25 first would catch a value at a tied p50/p75
    too and silently under-score it. Checking from the top means a tied
    value always resolves to its highest matching anchor instead."""
    points = sorted(float(point) for point in (p25, p50, p75))
    p25, p50, p75 = points
    value = float(value)
    epsilon = 1e-9

    if p75 - p25 < epsilon:
        # All three quantiles coincide — no information to discriminate
        # above/below, so anchor at the median rather than let branch order
        # produce an arbitrary answer.
        return 50.0

    if value >= p75:
        score = 75.0 + 25.0 * (value - p75) / max(p75 - p50, epsilon)
    elif value >= p50:
        score = 50.0 + 25.0 * (value - p50) / max(p75 - p50, epsilon)
    elif value >= p25:
        score = 25.0 + 25.0 * (value - p25) / max(p50 - p25, epsilon)
    else:
        score = 25.0 - 25.0 * (p25 - value) / max(abs(p25), epsilon)
    return max(0.0, min(100.0, score))


def choose_baseline(
    baselines: pd.DataFrame,
    role: str,
    champion_id: int,
    minute: int,
) -> pd.Series:
    common = baselines[baselines["minute"] == int(minute)]
    candidates = (
        common[
            (common["scope"] == "champion_role")
            & (common["role"] == role)
            & (common["champion_id"].astype("Int64") == int(champion_id))
        ],
        common[(common["scope"] == "role") & (common["role"] == role)],
        common[common["scope"] == "global"],
    )
    for candidate in candidates:
        if not candidate.empty:
            return candidate.iloc[0]
    raise LookupError(
        f"找不到 role={role}, champion={champion_id}, minute={minute} 的基线"
    )


def score_observation(
    baseline_row: pd.Series,
    observation: dict[str, float],
    role: str,
) -> dict[str, Any]:
    weights = JUNGLE_WEIGHTS if role == "JUNGLE" else LANE_WEIGHTS
    metric_scores: dict[str, float] = {}
    weighted_sum = 0.0
    used_weight = 0.0

    for metric, weight in weights.items():
        if metric not in observation:
            continue
        p25 = baseline_row.get(f"{metric}_p25")
        p50 = baseline_row.get(f"{metric}_p50")
        p75 = baseline_row.get(f"{metric}_p75")
        if pd.isna(p25) or pd.isna(p50) or pd.isna(p75):
            continue
        score = piecewise_percentile(observation[metric], p25, p50, p75)
        metric_scores[metric] = round(score, 2)
        weighted_sum += score * weight
        used_weight += weight

    if used_weight == 0:
        raise ValueError("观测值与基线没有可共同评分的指标。")
    scope = baseline_row["scope"]
    match_n = baseline_row.get("match_n")
    player_n = baseline_row.get("player_n")
    return {
        "p_score": round(weighted_sum / used_weight, 2),
        "baseline_scope": scope,
        "sample_count": int(baseline_row["sample_count"]),
        # Raw provenance, not a synthesized confidence score: sample_count
        # over-counts reliability because same-match/same-player minutes are
        # correlated, so match_n/player_n are the more honest reliability
        # signals for anything downstream that needs to judge trust.
        "baseline_match_n": int(match_n) if pd.notna(match_n) else None,
        "baseline_player_n": int(player_n) if pd.notna(player_n) else None,
        "baseline_fallback_level": SCOPE_FALLBACK_LEVEL.get(scope),
        "metric_scores": metric_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="用 Gold 经验分位数计算单分钟 P-score。")
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--role", required=True, choices=["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    )
    parser.add_argument("--champion-id", type=int, required=True)
    parser.add_argument("--minute", type=int, required=True)
    parser.add_argument("--total-gold", type=float, required=True)
    parser.add_argument("--xp", type=float, required=True)
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--minions-killed", type=float, default=0)
    parser.add_argument("--jungle-minions-killed", type=float, default=0)
    args = parser.parse_args()

    baselines = pd.read_parquet(Path(args.baseline))
    row = choose_baseline(
        baselines, args.role, args.champion_id, args.minute
    )
    observation = {
        "total_gold": args.total_gold,
        "xp": args.xp,
        "level": args.level,
        "minions_killed": args.minions_killed,
        "jungle_minions_killed": args.jungle_minions_killed,
    }
    result = score_observation(row, observation, args.role)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
