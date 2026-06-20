"""
Symbol Universe Rotator

Every ROTATION_INTERVAL_SEC (default 2h), scores all symbols that have
a trained MLForecaster model and picks the top TOP_N by:
  1. Regime gate    — TRENDING/RANGING only (CHOPPY/DOWNTREND excluded)
  2. ATR floor veto — ATR% < 0.4% excluded (can't cover round-trip fees)
  3. Liquidity veto — score < 0.10 excluded (too thin to trade cleanly)
  4. Momentum       — 20-bar price change % from klines cache
  5. Win-rate       — from SymbolPerformanceTracker (soft-blocked symbols penalised)
  6. PnL history    — last 10 trades avg PnL
  7. Liquidity      — log-scale quote volume score
  8. Preferred bonus — +0.05 for large-cap pairs (BTC/ETH/SOL/XRP etc.)

Applies the result by calling shared_state.set_accepted_symbols(top_n).
MLForecaster reads accepted_symbols on every run_once() call so the
watchlist updates automatically without a restart.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger("SymbolRotator")

ROTATION_INTERVAL_SEC = float(os.getenv("SYMBOL_ROTATION_INTERVAL_SEC", "7200"))  # 2 hours — matches immunity window
TOP_N = int(os.getenv("SYMBOL_ROTATION_TOP_N", "8"))
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MIN_SELLABLE_NOTIONAL = 5.0  # Binance min notional — below this we can't sell
# Confirmed liquidity score below this is too thin to trade cleanly → excluded from rotation.
_LIQUIDITY_MIN_SCORE = float(os.getenv("SYMBOL_ROTATION_MIN_LIQUIDITY", "0.10"))
# ATR% floor: symbols whose 14-bar ATR is below this fraction of price are excluded.
# 0.4% is roughly 2× the round-trip fee (0.2%) — can't generate net-positive trades below it.
_ATR_MIN_PCT = float(os.getenv("SYMBOL_ROTATION_MIN_ATR_PCT", "0.004"))
# Preferred high-quality symbols get a bonus in scoring to break ties in flat markets.
# These are the most liquid, most data-rich pairs — models train/calibrate better on them.
_PREFERRED_SYMBOLS = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
}

# Regimes that qualify for entry
_ALLOWED_REGIMES = {"UPTREND", "RANGING"}


def _available_model_symbols() -> list[str]:
    """Return all symbols that have a trained mlforecaster_*_5m.keras model."""
    syms = []
    for p in MODELS_DIR.glob("mlforecaster_*_5m.keras"):
        name = p.stem  # e.g. mlforecaster_BNBUSDT_5m
        parts = name.split("_")
        # format: mlforecaster_<SYMBOL>_5m
        if len(parts) >= 3:
            sym = "_".join(parts[1:-1])  # handles symbols without underscore
            if sym.endswith("USDT"):
                syms.append(sym)
    return syms


def _get_momentum(symbol: str, market_data: Any) -> float:
    """20-bar price-change % from klines cache. Returns 0.0 if unavailable."""
    if market_data is None:
        return 0.0
    klines_cache = getattr(market_data, "_klines", {})
    for key, (_ts, data) in klines_cache.items():
        if isinstance(key, tuple) and key[0] == symbol and data and len(data) >= 5:
            try:
                closes = [float(row[4]) for row in data if row and len(row) > 4]
                if len(closes) >= 2:
                    lookback = min(20, len(closes))
                    return (closes[-1] - closes[-lookback]) / closes[-lookback]
            except Exception:
                pass
    return 0.0


def _get_atr_pct(symbol: str, market_data: Any, period: int = 14) -> float:
    """ATR as a fraction of price (True Range based). Returns -1.0 if unavailable.

    A return of -1.0 means 'unknown — don't veto', so symbols with no klines data
    are not excluded. Only explicitly low ATR triggers the filter.
    """
    if market_data is None:
        return -1.0
    klines_cache = getattr(market_data, "_klines", {})
    for key, (_ts, data) in klines_cache.items():
        if isinstance(key, tuple) and key[0] == symbol and data and len(data) >= period + 1:
            try:
                rows = data[-(period + 1):]
                trs: list[float] = []
                for i in range(1, len(rows)):
                    h = float(rows[i][2])
                    l = float(rows[i][3])
                    pc = float(rows[i - 1][4])
                    tr = max(h - l, abs(h - pc), abs(l - pc))
                    trs.append(tr)
                if not trs:
                    return -1.0
                atr = sum(trs) / len(trs)
                close = float(rows[-1][4])
                return atr / close if close > 0 else -1.0
            except Exception:
                pass
    return -1.0


# Deep, always-liquid pairs. Used as a fair-liquidity floor so blue-chips aren't
# starved of a liquidity score when their klines aren't currently being streamed
# (which created a self-reinforcing bias toward whatever was already in the universe).
_LIQUID_MAJORS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "TRX",
    "DOT", "MATIC", "LTC", "BCH", "ATOM", "UNI", "NEAR", "APT", "ARB", "OP",
    "FIL", "ETC", "XLM", "ICP", "HBAR", "VET", "INJ", "SUI", "SEI", "TIA",
}


def _get_liquidity_score(symbol: str, market_data: Any) -> float:
    """0..1 liquidity from recent kline quote-volume, with a known-major floor.

    Binance kline row index 7 = quote-asset volume (USDT turnover per bar). Map on a
    log scale: ~$10k/bar → 0 (too thin to trade), ~$500k/bar → 1 (liquid). When kline
    volume is unavailable, fall back to a static tier (blue-chips → 0.85) so deep pairs
    aren't unfairly scored as 'unknown' relative to whatever is already streamed.
    Returns -1.0 only when both signals are unavailable (caller treats as neutral).
    """
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    static = 0.85 if base in _LIQUID_MAJORS else -1.0

    if market_data is not None:
        klines_cache = getattr(market_data, "_klines", {})
        for key, (_ts, data) in klines_cache.items():
            if isinstance(key, tuple) and key[0] == symbol and data and len(data) >= 5:
                try:
                    vols = [float(row[7]) for row in data[-20:] if row and len(row) > 7]
                    if vols:
                        avg_qv = sum(vols) / len(vols)
                        vol_score = max(0.0, min(1.0, (math.log10(max(avg_qv, 1.0)) - 4.0) / 1.7))
                        # Never score a known blue-chip below its static floor.
                        return max(vol_score, static) if static > 0 else vol_score
                except Exception:
                    pass
    return static  # majors → 0.85; unknown alts → -1.0 (neutral)


def _get_regime(symbol: str, regime_detector: Any, shared_state: Any) -> str:
    if regime_detector is not None:
        try:
            return str(regime_detector.get_symbol_regime(symbol)).upper()
        except Exception:
            pass
    if shared_state is not None:
        stored = (getattr(shared_state, "metrics", {}) or {}).get("symbol_regimes", {})
        return str(stored.get(symbol, "RANGING")).upper()
    return "RANGING"


def _get_win_rate(symbol: str, perf_tracker: Any) -> float:
    if perf_tracker is None:
        return 0.5
    try:
        info = perf_tracker.symbol_info(symbol)
        if info and info.get("win_rate") is not None:
            return float(info["win_rate"])
    except Exception:
        pass
    return 0.5


class SymbolRotator:
    """
    Periodic symbol universe rotator.

    Wires into the native stack via:
      shared_state     — writes accepted_symbols
      regime_detector  — per-symbol regime classification
      market_data      — klines cache for momentum
      perf_tracker     — win_rate for each symbol
    """

    def __init__(
        self,
        shared_state: Any,
        regime_detector: Any | None = None,
        market_data: Any | None = None,
        perf_tracker: Any | None = None,
        fallback_symbols: list[str] | None = None,
    ) -> None:
        self._ss = shared_state
        self._regime_detector = regime_detector
        self._market_data = market_data
        self._perf_tracker = perf_tracker
        self._fallback = fallback_symbols or [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "LINKUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT",
        ]
        self._last_rotation: float = 0.0
        self._startup_rotate_after: float = time.monotonic() + 120.0  # re-run after 2 min (klines ready)
        self._current_universe: list[str] = list(self._fallback)
        # Pending exits: symbols rotated out with sellable positions — emitted to main loop
        self._pending_exits: dict[str, float] = {}  # symbol -> qty to sell
        # Known un-sellable dust (notional < MIN_SELLABLE_NOTIONAL) we've already
        # acknowledged. Tracked so we log each crumb ONCE rather than warning every
        # rotation. We deliberately do NOT auto-liquidate dust — automated dust
        # liquidation caused the 2026-05-04 incident ($86 loss). Best practice for
        # tiny un-sellable balances is to leave them parked; the user can convert
        # them to BNB manually in the Binance UI if they ever accumulate.
        self._stranded_dust: set[str] = set()
        # Buy timestamp registry: track when each symbol was last bought so immunity window works
        # even when position entry_time is 0 (fill lag / position hydration race).
        # Persisted in shared_state.metrics["rotator_buy_ts"] so it survives restarts.
        self._buy_ts: dict[str, float] = self._load_buy_ts()

    def _load_buy_ts(self) -> dict[str, float]:
        """Load persisted buy timestamps from shared_state so immunity survives restarts."""
        if self._ss is None:
            return {}
        try:
            stored = (getattr(self._ss, "metrics", {}) or {}).get("rotator_buy_ts", {})
            now = time.time()
            # Drop entries older than 3 hours (well past immunity window)
            return {sym: ts for sym, ts in (stored or {}).items()
                    if isinstance(ts, (int, float)) and (now - float(ts)) < 10800.0}
        except Exception:
            return {}

    def _persist_buy_ts(self) -> None:
        """Write current buy timestamps to shared_state.metrics for snapshot persistence."""
        if self._ss is None:
            return
        try:
            metrics = getattr(self._ss, "metrics", None)
            if metrics is not None:
                metrics["rotator_buy_ts"] = dict(self._buy_ts)
        except Exception:
            pass

    # ── Public ───────────────────────────────────────────────────────

    def register_buy(self, symbol: str) -> None:
        """Call immediately after a BUY fill so the immunity window is anchored correctly."""
        self._buy_ts[symbol] = time.time()
        self._persist_buy_ts()

    def maybe_rotate(self) -> bool:
        """Call every cycle. Returns True if rotation was performed."""
        now = time.monotonic()
        # Startup re-rotation: fire once ~2 min after boot so klines are populated
        if self._startup_rotate_after and now >= self._startup_rotate_after:
            self._startup_rotate_after = 0.0
            self._last_rotation = now
            self._rotate()
            return True
        if now - self._last_rotation < ROTATION_INTERVAL_SEC:
            return False
        self._last_rotation = now
        self._rotate()
        return True

    def force_rotate(self) -> None:
        """Force an immediate rotation (e.g. on startup)."""
        self._last_rotation = 0.0
        self.maybe_rotate()

    @property
    def current_universe(self) -> list[str]:
        return list(self._current_universe)

    def pop_pending_exits(self) -> dict[str, float]:
        """
        Return and clear the dict of symbols that were rotated out while
        holding a sellable position (notional >= MIN_SELLABLE_NOTIONAL).
        Main loop should issue a SELL for each.
        Returns {symbol: qty}.
        """
        exits = dict(self._pending_exits)
        self._pending_exits.clear()
        return exits

    # ── Internal ─────────────────────────────────────────────────────

    def _rotate(self) -> None:
        candidates = _available_model_symbols()
        if not candidates:
            _log.warning("[SymbolRotator] No model files found — keeping current universe")
            return

        _log.info("[SymbolRotator] Scoring %d candidates for universe rotation", len(candidates))
        scores: list[tuple[float, str]] = []
        trending: list[str] = []
        ranging: list[str] = []
        regime_counts: dict[str, int] = {}

        for sym in candidates:
            regime = _get_regime(sym, self._regime_detector, self._ss)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            if regime not in _ALLOWED_REGIMES:
                continue

            momentum = _get_momentum(sym, self._market_data)
            win_rate = _get_win_rate(sym, self._perf_tracker)
            liq_raw = _get_liquidity_score(sym, self._market_data)

            # Fix A — ATR floor veto: exclude symbols whose volatility can't cover round-trip fees.
            # -1.0 = no klines data yet → don't gate (give benefit of the doubt on startup).
            atr_pct = _get_atr_pct(sym, self._market_data)
            if atr_pct >= 0.0 and atr_pct < _ATR_MIN_PCT:
                _log.debug(
                    "[SymbolRotator] %s excluded: ATR%.4f%% < min %.4f%%",
                    sym, atr_pct * 100, _ATR_MIN_PCT * 100,
                )
                continue

            # Fix B — Liquidity hard veto (unchanged logic, now also logs vetoed symbol).
            # liq_raw < 0 means "unknown" → don't gate.
            if 0.0 <= liq_raw < _LIQUIDITY_MIN_SCORE:
                _log.debug(
                    "[SymbolRotator] %s excluded: liquidity_score=%.2f < min %.2f",
                    sym, liq_raw, _LIQUIDITY_MIN_SCORE,
                )
                continue
            liquidity = 0.5 if liq_raw < 0.0 else liq_raw  # unknown → neutral

            # Normalize EVERY component to 0..1 so the intended weights actually apply.
            mom_norm = max(0.0, min(1.0, 0.5 + momentum * 5.0))     # ±10% move spans 0..1
            regime_score = 1.0 if regime == "UPTREND" else 0.5      # RANGING neutral, UPTREND best

            avg_pnl = 0.0
            if self._ss is not None:
                _hist = (getattr(self._ss, "trade_history", None) or {}).get(sym, [])
                if _hist:
                    _recent = _hist[-10:]  # last 10 trades per symbol
                    avg_pnl = sum(float(t.get("pnl_pct", 0.0)) for t in _recent) / len(_recent)
            pnl_score = max(0.0, min(1.0, 0.5 + avg_pnl * 10.0))    # ±5% avg pnl spans 0..1

            # Regime-aware momentum: in a non-trending name momentum is mostly noise, so
            # dampen it toward neutral and lean on win-rate / pnl / liquidity instead.
            if regime != "UPTREND":
                mom_norm = 0.5 + (mom_norm - 0.5) * 0.4

            # Fix B — Preferred-symbol quality bonus: large-cap pairs have deeper order books,
            # better model calibration, and more historical data.  +0.05 bonus (≈ a 10pp
            # liquidity advantage) breaks ties in flat markets in favour of quality names
            # over micro-caps like LUNC/PEPE whose micro-price makes spread cost proportionally huge.
            preferred_bonus = 0.05 if sym in _PREFERRED_SYMBOLS else 0.0

            # Weighted blend (weights sum to 1.0; all terms 0..1 → real influence).
            score = (
                0.30 * mom_norm
                + 0.30 * win_rate
                + 0.15 * regime_score
                + 0.10 * pnl_score
                + 0.15 * liquidity
                + preferred_bonus
            )

            scores.append((score, sym))
            if regime == "UPTREND":
                trending.append(sym)
            else:
                ranging.append(sym)

        if not scores:
            _log.warning(
                "[SymbolRotator] No TRENDING/RANGING symbols found among %d candidates "
                "(regimes: %s) — keeping current universe", len(candidates), regime_counts
            )
            return

        scores.sort(key=lambda x: x[0], reverse=True)
        top = [sym for _, sym in scores[:TOP_N]]

        # Always keep at least one trending symbol if any exist
        if trending and not any(s in trending for s in top):
            top[-1] = trending[0]

        prev = set(self._current_universe)
        added = set(top) - prev
        removed = prev - set(top)

        self._current_universe = top
        # A symbol back in the active set is no longer "stranded dust" — clear its tag
        # so it re-logs if it strands again after a future rotation.
        self._stranded_dust.difference_update(set(top))
        if self._ss is not None:
            self._ss.set_accepted_symbols(top)

        # Check removed symbols for sellable positions — queue exits
        exits_queued = []
        dust_stranded = []
        if removed and self._ss is not None:
            positions = dict(getattr(self._ss, "positions", {}) or {})
            balance = dict(getattr(self._ss, "balance", {}) or {})
            prices = dict(getattr(self._ss, "prices", {}) or {})

            for sym in removed:
                qty = 0.0
                # Check positions dict first
                pos = positions.get(sym)
                if pos is not None:
                    qty = float(getattr(pos, "qty", None) or (pos.get("qty", 0) if isinstance(pos, dict) else 0))
                # Fall back to balance
                if qty <= 1e-8 and sym.endswith("USDT"):
                    base = sym[:-4]
                    qty = float(balance.get(base, 0.0) or 0.0)

                if qty <= 1e-8:
                    continue

                price = float(prices.get(sym, 0.0) or 0.0)
                notional = qty * price if price > 0 else 0.0

                if notional >= MIN_SELLABLE_NOTIONAL:
                    # Post-buy immunity: never rotation-exit a position entered within 2 hours.
                    # Gives signal exit the chance to fire and generate profit before rotation takes over.
                    # Use _buy_ts registry first (set at fill time), fall back to position entry_time.
                    _now = time.time()
                    _buy_ts = self._buy_ts.get(sym, 0.0)
                    if _buy_ts == 0.0 and pos is not None:
                        _buy_ts = float(getattr(pos, "entry_time", None)
                                        or (pos.get("entry_time", 0) if isinstance(pos, dict) else 0) or 0.0)
                    if _buy_ts == 0.0:
                        # No timestamp available — entry time unknown, assume position is fresh and
                        # grant immunity to prevent rotating out a just-entered position we can't date.
                        _log.info(
                            "[SymbolRotator] EXIT deferred: %s has no entry timestamp — granting immunity to avoid rotating fresh position",
                            sym,
                        )
                        self._current_universe.append(sym)
                        continue
                    _pos_age = _now - _buy_ts
                    if _pos_age < 7200.0:  # 2-hour immunity window
                        _log.info(
                            "[SymbolRotator] EXIT deferred: %s rotated out but only %.0fm old — "
                            "holding for signal exit (immunity window=2h)",
                            sym, _pos_age / 60,
                        )
                        # Keep in universe temporarily so it isn't re-rotated immediately
                        self._current_universe.append(sym)
                        continue
                    self._pending_exits[sym] = qty
                    exits_queued.append(f"{sym}(qty={qty:.6f} ~${notional:.2f})")
                    _log.info(
                        "[SymbolRotator] EXIT queued: %s rotated out with sellable position qty=%.6f notional=$%.2f",
                        sym, qty, notional,
                    )
                else:
                    # Un-sellable dust. Keep it pinned in the watchlist so it stays
                    # priced/managed (no orphan / NAV under-count) and log ONCE — never
                    # auto-liquidate (see 2026-05-04 incident).
                    if sym not in self._current_universe:
                        self._current_universe.append(sym)
                    if sym not in self._stranded_dust:
                        self._stranded_dust.add(sym)
                        dust_stranded.append(f"{sym}(~${notional:.2f})")
                        _log.info(
                            "[SymbolRotator] DUST parked: %s notional=$%.2f < $%.0f min — "
                            "un-sellable, pinned to watchlist (not auto-liquidated). "
                            "Convert to BNB manually in Binance UI if it accumulates.",
                            sym, notional, MIN_SELLABLE_NOTIONAL,
                        )

        _log.info(
            "[SymbolRotator] Universe updated → %s | +%s -%s | exits_queued=%s dust_stranded=%s | trending=%d ranging=%d total_candidates=%d",
            top, list(added), list(removed),
            exits_queued or "none", dust_stranded or "none",
            len(trending), len(ranging), len(candidates),
        )
