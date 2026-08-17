from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote

import requests


LOGGER = logging.getLogger(__name__)


class RiotAPIError(RuntimeError):
    pass


class RiotAPI:
    def __init__(
        self,
        api_key: str,
        platform: str,
        region: str,
        request_interval_seconds: float = 1.25,
        max_retries: int = 5,
    ) -> None:
        self.platform = platform.lower()
        self.region = region.lower()
        self.request_interval_seconds = max(0.0, request_interval_seconds)
        self.max_retries = max(1, max_retries)
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-Riot-Token": api_key,
                "User-Agent": "riot-gold-collector-v2/1.0",
            }
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=(5, 30))
                self._last_request_at = time.monotonic()
            except requests.RequestException as exc:
                last_error = exc
                wait_seconds = min(30, 2**attempt)
                LOGGER.warning("网络请求失败，%s 秒后重试：%s", wait_seconds, exc)
                time.sleep(wait_seconds)
                continue

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "2"))
                wait_seconds = max(retry_after, 2**attempt)
                LOGGER.warning("触发 Riot 限流，等待 %.1f 秒", wait_seconds)
                time.sleep(wait_seconds)
                continue

            if response.status_code in {500, 502, 503, 504}:
                wait_seconds = min(30, 2**attempt)
                LOGGER.warning(
                    "Riot API %s，%s 秒后重试", response.status_code, wait_seconds
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 403:
                raise RiotAPIError("Riot API 返回 403：请更新 RIOT_API_KEY。")
            if response.status_code == 404:
                raise RiotAPIError(f"Riot API 返回 404：{url}")
            if not response.ok:
                raise RiotAPIError(
                    f"Riot API 返回 {response.status_code}：{response.text[:300]}"
                )
            return response.json()

        raise RiotAPIError(f"Riot API 重试耗尽：{last_error or url}")

    def get_gold_entries(self, division: str, page: int = 1) -> list[dict[str, Any]]:
        url = (
            f"https://{self.platform}.api.riotgames.com"
            f"/lol/league/v4/entries/RANKED_SOLO_5x5/GOLD/{division.upper()}"
        )
        return self._get(url, {"page": page})

    def get_match_ids(
        self, puuid: str, count: int, start_time: int | None = None
    ) -> list[str]:
        url = (
            f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/"
            f"{quote(puuid, safe='')}/ids"
        )
        # queue=420 is ranked solo/duo. type=ranked is a second defensive filter.
        params: dict[str, Any] = {
            "start": 0,
            "count": min(max(count, 1), 100),
            "queue": 420,
            "type": "ranked",
        }
        if start_time is not None:
            # Optimization only: narrows the match-ID window to reduce wasted
            # requests on other patches. Never relied on for correctness —
            # the actual patch is always re-verified from Match Detail.
            params["startTime"] = int(start_time)
        return self._get(url, params)

    def get_match(self, match_id: str) -> dict[str, Any]:
        url = (
            f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/"
            f"{quote(match_id, safe='')}"
        )
        return self._get(url)

    def get_timeline(self, match_id: str) -> dict[str, Any]:
        url = (
            f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/"
            f"{quote(match_id, safe='')}/timeline"
        )
        return self._get(url)
