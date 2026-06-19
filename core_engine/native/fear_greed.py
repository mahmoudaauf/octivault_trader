"""
Fear & Greed Index fetcher (alternative.me — free, no auth, updates daily).

Provides a cached sentiment score (0–100) and classification:
  0–24  → EXTREME_FEAR   (contrarian BUY signal)
  25–44 → FEAR
  45–55 → NEUTRAL
  56–74 → GREED
  75–100 → EXTREME_GREED (contrarian SELL signal)

The score is also normalised to [-1.0, +1.0] so it can be consumed by
MLForecaster's existing Sentiment filter without modification.

Usage::

    fetcher = FearGreedFetcher()
    await fetcher.start()
    score = fetcher.score          # int 0–100, or None if never fetched
    norm  = fetcher.normalised     # float -1..+1 (50 → 0.0)
    label = fetcher.classification # "Extreme Fear" | "Fear" | ... | None
    await fetcher.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import aiohttp

_PAUSE_FLAG_PATH = os.path.join("logs", "pause_buys.flag")
_MANUAL_OVERRIDES_PATH = os.path.join("logs", "manual_overrides.json")
# Auto-resume when F&G rises by this many points across consecutive hourly reads
_AUTO_RESUME_RISE_THRESHOLD = 3
# Number of profitable trades before restoring full position size after resume
_HALF_SIZE_TRADE_COUNT = 5

logger = logging.getLogger(__name__)

_API_URL = "https://api.alternative.me/fng/?limit=1&format=json"
_REFRESH_INTERVAL_SEC = 3600.0  # F&G updates once per day; check every hour


class FearGreedFetcher:
    def __init__(
        self,
        refresh_interval_sec: float = _REFRESH_INTERVAL_SEC,
        shared_state: Optional[object] = None,
        exchange_client: Optional[object] = None,
    ) -> None:
        self._interval = max(60.0, float(refresh_interval_sec))
        self._score: Optional[int] = None
        self._prev_score: Optional[int] = None  # score from the previous fetch
        self._classification: Optional[str] = None
        self._fetched_at: float = 0.0
        self._task: Optional[asyncio.Task] = None
        self._shared_state = shared_state  # NativeSharedState | None
        self._exchange_client = exchange_client  # for BTC 1h reversal confirmation
        # BTC reversal confirmation cache (1h candles only close hourly; cache briefly)
        self._btc_rev_cached: bool = False
        self._btc_rev_checked_at: float = 0.0
        self._btc_rev_cache_ttl: float = 300.0  # 5 minutes

    # ── public read properties ──────────────────────────────────────────

    @property
    def score(self) -> Optional[int]:
        """Raw F&G score 0–100, or None if never successfully fetched."""
        return self._score

    @property
    def classification(self) -> Optional[str]:
        return self._classification

    @property
    def normalised(self) -> float:
        """
        Map score 0–100 → -1.0..+1.0 where 50=neutral, 0=extreme fear=-1, 100=extreme greed=+1.
        Returns 0.0 (neutral) when score is unavailable.
        """
        if self._score is None:
            return 0.0
        return (self._score - 50.0) / 50.0

    @property
    def is_extreme_fear(self) -> bool:
        return self._score is not None and self._score < 25

    @property
    def is_fear(self) -> bool:
        return self._score is not None and self._score < 45

    @property
    def age_sec(self) -> float:
        return time.time() - self._fetched_at if self._fetched_at > 0 else float("inf")

    # ── lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        # Fetch once immediately before entering background loop
        await self._fetch_once()
        self._task = asyncio.create_task(self._run(), name="fear-greed-fetcher")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ── internals ───────────────────────────────────────────────────────

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._fetch_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("FearGreedFetcher background refresh failed: %s", e)

    async def _fetch_once(self) -> None:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(_API_URL) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            entry = data["data"][0]
            new_score = int(entry["value"])
            self._prev_score = self._score
            self._score = new_score
            self._classification = str(entry.get("value_classification", ""))
            self._fetched_at = time.time()
            # Push into shared_state so _resolve_mode sees it without polling
            if self._shared_state is not None:
                self._shared_state.fear_greed_score = self._score
                self._shared_state.fear_greed_label = self._classification
            logger.info(
                "Fear & Greed Index: %d (%s) — normalised=%.2f",
                self._score, self._classification, self.normalised,
            )
            # Auto-manage buy pause flag based on F&G trend
            await self._auto_manage_pause_flag()
        except Exception as e:
            logger.warning("FearGreedFetcher: fetch failed (%s) — retaining last value", e)

    async def _btc_confirmed_reversal(self) -> bool:
        """Check if BTC has 2 consecutive green 1h candles — confirms market reversal.

        Fetches BTC 1h candles via REST on-demand (cached 5min) because the market-data
        WebSocket only streams 1m candles. Also mirrors candles + result into shared_state
        so the capital allocator can size fear-time positions globally.
        """
        now = time.time()
        # Serve from cache — 1h candles only close hourly, no need to refetch each cycle
        if (now - self._btc_rev_checked_at) < self._btc_rev_cache_ttl and self._btc_rev_checked_at > 0:
            return self._btc_rev_cached

        confirmed = False
        try:
            candles = None
            # Primary: fetch fresh 1h candles via REST
            ec = self._exchange_client
            if ec is not None and hasattr(ec, "get_klines"):
                raw = await ec.get_klines("BTCUSDT", interval="1h", limit=3)
                if raw and isinstance(raw, list):
                    candles = [
                        {"open": float(r[1]), "high": float(r[2]),
                         "low": float(r[3]), "close": float(r[4])}
                        for r in raw if isinstance(r, (list, tuple)) and len(r) >= 5
                    ]
                    # Mirror into shared_state for any other consumer
                    if candles and self._shared_state is not None:
                        try:
                            self._shared_state.market_data[("BTCUSDT", "1h")] = candles
                        except Exception:
                            pass
            # Fallback: whatever may already be in shared_state
            if not candles and self._shared_state is not None:
                md = getattr(self._shared_state, "market_data", {}) or {}
                candles = md.get(("BTCUSDT", "1h")) or md.get(("BTCUSDT", "1H"))

            if not candles or len(candles) < 2:
                logger.info("BTC 1h candles unavailable — reversal=False (protect capital)")
                confirmed = False
            else:
                last_two = candles[-2:]
                green = [
                    float(c.get("close", 0) or 0) > float(c.get("open", 0) or 0)
                    for c in last_two
                ]
                confirmed = all(green)
                logger.info(
                    "BTC reversal check: candle[-2] %s candle[-1] %s → confirmed=%s",
                    "🟢" if green[0] else "🔴",
                    "🟢" if green[1] else "🔴",
                    confirmed,
                )
        except Exception as e:
            logger.warning("BTC reversal check failed: %s", e)
            confirmed = False

        self._btc_rev_cached = confirmed
        self._btc_rev_checked_at = now
        # Publish for the capital allocator's fear-sizing logic
        if self._shared_state is not None:
            try:
                self._shared_state.btc_reversal_confirmed = confirmed
            except Exception:
                pass
        return confirmed

    def _set_manual_override(self, size_multiplier: float) -> None:
        """Write SIZE_MULTIPLIER to manual_overrides.json — picked up by main.py each cycle."""
        try:
            import json
            data = {"SIZE_MULTIPLIER": size_multiplier, "set_at": time.time(),
                    "resume_trade_count": 0, "half_size_trade_count": _HALF_SIZE_TRADE_COUNT}
            with open(_MANUAL_OVERRIDES_PATH, "w") as f:
                json.dump(data, f)
            logger.info("📐 Manual override: SIZE_MULTIPLIER=%.2f written", size_multiplier)
        except Exception as e:
            logger.warning("Failed to write manual overrides: %s", e)

    async def _auto_manage_pause_flag(self) -> None:
        """Auto-pause BUYs when F&G is falling; auto-resume when F&G rising + BTC confirmed."""
        if self._score is None:
            return
        flag_exists = os.path.exists(_PAUSE_FLAG_PATH)
        prev = self._prev_score
        curr = self._score

        if prev is None:
            # First fetch — only auto-pause if deeply in fear with no prior reading
            if curr <= 15 and not flag_exists:
                try:
                    open(_PAUSE_FLAG_PATH, "w").close()
                    logger.warning("🛑 AUTO-PAUSE: F&G=%d (Extreme Fear) — BUYs paused until recovery", curr)
                    if self._shared_state is not None:
                        try:
                            self._shared_state.buy_paused_fear_greed = True
                        except Exception:
                            pass
                except Exception:
                    pass
            elif self._shared_state is not None:
                # Ensure flag matches state on every restart
                try:
                    self._shared_state.buy_paused_fear_greed = flag_exists
                except Exception:
                    pass
            return

        rise = curr - prev
        if flag_exists:
            if rise >= _AUTO_RESUME_RISE_THRESHOLD:
                # F&G is rising — now require BTC 2-green-candle confirmation
                if await self._btc_confirmed_reversal():
                    try:
                        os.remove(_PAUSE_FLAG_PATH)
                        logger.info(
                            "✅ AUTO-RESUME: F&G rose %d→%d (+%d pts) + BTC confirmed 🟢🟢 — BUYs unpaused at HALF SIZE",
                            prev, curr, rise,
                        )
                        if self._shared_state is not None:
                            try:
                                self._shared_state.buy_paused_fear_greed = False
                            except Exception:
                                pass
                        # Enter half-size mode for first 5 profitable trades
                        self._set_manual_override(size_multiplier=0.5)
                    except Exception:
                        pass
                else:
                    logger.info(
                        "⏸️  F&G rose %d→%d (+%d pts) but BTC not confirmed — staying paused",
                        prev, curr, rise,
                    )
            else:
                logger.info(
                    "⏸️  BUY PAUSE holds: F&G=%d→%d (need +%d pts rise to resume, got %+d)",
                    prev, curr, _AUTO_RESUME_RISE_THRESHOLD, rise,
                )
        else:
            # Auto-pause if F&G drops significantly (falling knife protection)
            if rise <= -5 and curr <= 20:
                try:
                    open(_PAUSE_FLAG_PATH, "w").close()
                    logger.warning(
                        "🛑 AUTO-PAUSE: F&G fell %d→%d (%d pts drop) — BUYs paused",
                        prev, curr, rise,
                    )
                    if self._shared_state is not None:
                        try:
                            self._shared_state.buy_paused_fear_greed = True
                        except Exception:
                            pass
                except Exception:
                    pass
