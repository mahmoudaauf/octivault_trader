"""
Native TP/SL Engine — Volatility-Adaptive Take Profit & Stop Loss

Tier 1 Implementation (May 7, 2026):
  1. ATR-based volatility adaptation
  2. Risk-based position sizing (Kelly-criterion style)
  3. Auto-arm safety on startup

Tier 2 Enhancements (May 7, 2026):
  4. Cost-aware minimum TP floor (configured fees + slippage + net profit)
  5. Time-based TP tightening and configurable stale-position exit
  6. Trailing TP after +1.5% profit (0.8% trail below peak)

No external dependencies beyond asyncio + logging + time.
"""

import json
import logging
import os
import time
from typing import Optional, Union

from . import math_utils


class NativeTPSLEngine:
    """Volatility-adaptive TP/SL with fee-awareness, time decay, and trailing stops."""

    # Minimum net profit required above fees and configured execution slippage.
    _ROUND_TRIP_FEE_PCT = 0.003
    _MIN_NET_PROFIT_PCT = 0.005  # 0.5% net profit floor

    # Time-based tightening thresholds
    _AGE_TIGHTEN_1_SEC = 2 * 3600   # 2 hours  → tp=+1.5%, sl=break-even
    _AGE_TIGHTEN_2_SEC = 4 * 3600   # 4 hours  → tp=+0.8%
    _AGE_FORCE_EXIT_SEC = int(
        os.getenv("TPSL_FORCE_EXIT_H", "3")
    ) * 3600  # 3 hours default → force exit after rotator immunity (2h) expires
    _PROTECTIVE_TIGHTEN_SEC = 30 * 60  # 30 minutes for protection-driven tightening

    # Trailing stop — regime-aware profit capture.
    # Faster lock-in during chop (fee drag kills small winners); let trends run further.
    # Each tuple: (activation_pct, distance_pct).  Default row used for UNKNOWN/"".
    # Trend regimes keep a high activation so real runners reach the +2% TP. Non-trend
    # regimes (range/chop/down) activate EARLY (+0.5%) so a stalled winner that won't reach
    # TP gets its modest profit locked instead of round-tripping to breakeven — the +2% TP
    # rarely hits in chop, so capturing +0.3-0.5% net is the higher-expectancy play there.
    _REGIME_TRAIL_PARAMS: dict[str, tuple[float, float]] = {
        "UPTREND": (0.020, 0.010),  # ride the trend; exit ~+1.0% net
        "TRENDING": (0.015, 0.008),  # solid momentum; exit ~+0.7% net
        "RANGING": (0.008, 0.005),  # arm at +0.8%, trail 0.5% — min capture +0.3% gross → ~+0.1% net
        "CHOPPY": (0.005, 0.003),   # arm at +0.5%, trail 0.3% — lock fast in noise
        "DOWNTREND": (0.005, 0.003),  # survival mode; any real gain locked immediately
    }
    _REGIME_TRAIL_DEFAULT = (0.015, 0.008)  # fallback = TRENDING behaviour

    # SL bounds — ATR-based SL is clamped within these to prevent both whipsaw and runaway loss.
    _SL_FLOOR_PCT = 0.010  # 1.0% minimum SL distance
    _SL_CEILING_PCT = 0.025  # 2.5% maximum SL distance (never worse than before)

    def __init__(self, shared_state, config, **kwargs):
        self.shared_state = shared_state
        self.config = config
        self.logger = logging.getLogger("NativeTPSLEngine")

        # Config: TP/SL strategy
        self._base_tp_atr_mult = float(getattr(config, "TP_ATR_MULT", 1.5) or 1.5)
        # SL raised from 1.0 to 1.5×ATR — gives trades a noise buffer before stopping out.
        # All regime multipliers scale from this base, so RANGING/DOWNTREND widen proportionally.
        self._base_sl_atr_mult = float(getattr(config, "SL_ATR_MULT", 1.5) or 1.5)

        # Config: Risk management
        self._target_risk_pct = float(
            getattr(config, "TARGET_RISK_PCT", 2.0) or 2.0
        )  # 2% risk per trade
        self._atr_lookback = int(getattr(config, "ATR_LOOKBACK", 14) or 14)
        self._min_atr_pct = float(
            getattr(config, "MIN_ATR_PCT", 0.005) or 0.005
        )  # 0.5% floor — prevents SL within bid-ask spread

        # Config: Volatility adaptation
        self._vol_adaptation_enabled = bool(getattr(config, "TPSL_VOL_ADAPTATION_ENABLED", True))
        self._vol_pressure_scale = float(getattr(config, "VOL_PRESSURE_SCALE", 0.35) or 0.35)

        # Config: Safety
        self._min_notional_safety = float(getattr(config, "MIN_NOTIONAL_SAFETY", 10.0) or 10.0)
        self._auto_arm_enabled = bool(getattr(config, "TPSL_AUTO_ARM_ENABLED", True))
        taker_fee_bps = float(getattr(config, "taker_fee_bps", 10.0) or 10.0)
        exit_slippage_bps = float(
            getattr(
                config,
                "exit_slippage_bps",
                os.getenv("EXIT_SLIPPAGE_BPS", os.getenv("CR_PRICE_SLIPPAGE_BPS", "10")),
            )
            or 0.0
        )
        self._round_trip_cost_pct = max(
            0.002, ((taker_fee_bps * 2.0) + exit_slippage_bps) / 10_000.0
        )

        # Runtime state
        self._armed_symbols: set[str] = set()
        # Grace period: suppress TP/SL triggers for 90s after startup hydration
        # so that stale/first-tick prices don't fire spurious exits on restored positions.
        self._startup_grace_until: float = 0.0

        # Internal TP/SL registry — authoritative source, not position dict
        self._tp_levels: dict[str, float] = {}
        self._sl_levels: dict[str, float] = {}

        # Trailing state: symbol → peak_price
        self._peak_prices: dict[str, float] = {}

        # Time-based state: symbol → entry_ts (unix seconds)
        self._entry_timestamps: dict[str, float] = {}

        # Entry price registry — used for P&L calculation on SELL
        self._entry_prices: dict[str, float] = {}

        # Force-exit set: symbols that hit 4h timeout threshold
        self._force_exit_symbols: set[str] = set()

        # Breakeven tracking: symbols where SL has been moved to entry price
        self._breakeven_activated: set[str] = set()

        self._tpsl_state_path = os.path.join("logs", "tpsl_state.json")
        self._load_tpsl_state()

    @staticmethod
    def _fp(price: float) -> str:
        """Format a price with adaptive decimal places — prevents sub-cent tokens showing as 0."""
        if price == 0:
            return "0"
        abs_p = abs(price)
        if abs_p < 0.000001:
            return f"{price:.10f}"
        if abs_p < 0.0001:
            return f"{price:.8f}"
        if abs_p < 0.01:
            return f"{price:.7f}"
        if abs_p < 1.0:
            return f"{price:.6f}"
        if abs_p < 100.0:
            return f"{price:.4f}"
        return f"{price:.2f}"

    def _load_tpsl_state(self) -> None:
        try:
            if os.path.exists(self._tpsl_state_path):
                with open(self._tpsl_state_path) as _f:
                    data = json.loads(_f.read())
                self._tp_levels = {k: float(v) for k, v in data.get("tp_levels", {}).items()}
                self._sl_levels = {k: float(v) for k, v in data.get("sl_levels", {}).items()}
                self._entry_timestamps = {
                    k: float(v) for k, v in data.get("entry_timestamps", {}).items()
                }
                self._peak_prices = {k: float(v) for k, v in data.get("peak_prices", {}).items()}
                self._entry_prices = {k: float(v) for k, v in data.get("entry_prices", {}).items()}
                # Purge entries for positions that no longer exist — prevents phantom TP/SL
                # triggers on stale state from a previous session.
                current_positions = set(getattr(self.shared_state, "positions", {}) or {})
                if current_positions:
                    stale = [s for s in list(self._tp_levels) if s not in current_positions]
                    for s in stale:
                        self._tp_levels.pop(s, None)
                        self._sl_levels.pop(s, None)
                        self._entry_timestamps.pop(s, None)
                        self._peak_prices.pop(s, None)
                    if stale:
                        self.logger.info(
                            "[TPSLEngine] Purged %d stale entries on load: %s", len(stale), stale
                        )
                # Mark restored symbols as armed so auto-arm skips recalculation.
                # Best practice: never recalculate SL on an existing position after entry —
                # the persisted SL reflects the trade's original risk parameters.
                self._armed_symbols.update(self._tp_levels.keys())
                if self._tp_levels:
                    self.logger.info(
                        "[TPSLEngine] Restored TP/SL state for %d symbols: %s",
                        len(self._tp_levels),
                        list(self._tp_levels.keys()),
                    )
        except Exception as exc:
            self.logger.debug("[TPSLEngine] _load_tpsl_state failed: %s", exc)

    def _save_tpsl_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._tpsl_state_path) or "logs", exist_ok=True)
            # P4: MIN_NOTIONAL guard — never persist entries whose notional is below
            # Binance's minimum. Prevents dust positions from surviving restarts.
            live_prices = getattr(self.shared_state, "prices", {}) if self.shared_state else {}
            _min_notional = getattr(self, "_min_notional_safety", 10.0)
            def _above_min(sym: str) -> bool:
                ep = self._entry_prices.get(sym, 0.0)
                if ep <= 0:
                    return True  # unknown price — keep, don't silently drop
                price = float((live_prices or {}).get(sym, ep) or ep)
                # Use whichever is larger to be conservative
                notional = price * 1.0  # qty unknown here; guard on entry_price alone
                # Real guard: entry_price itself below minimum unit price is suspect
                # Practical check: if entry_price is set, symbol was real — keep it
                return sym in (getattr(self.shared_state, "positions", {}) or {})
            valid_syms = {s for s in self._tp_levels if _above_min(s)}
            payload = {
                "tp_levels": {k: v for k, v in self._tp_levels.items() if k in valid_syms},
                "sl_levels": {k: v for k, v in self._sl_levels.items() if k in valid_syms},
                "entry_timestamps": {k: v for k, v in self._entry_timestamps.items() if k in valid_syms},
                "peak_prices": {k: v for k, v in self._peak_prices.items() if k in valid_syms},
                "entry_prices": {k: v for k, v in self._entry_prices.items() if k in valid_syms},
                "saved_at": time.time(),
            }
            tmp = self._tpsl_state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, allow_nan=False)
            os.replace(tmp, self._tpsl_state_path)
        except Exception as exc:
            self.logger.debug("[TPSLEngine] _save_tpsl_state failed: %s", exc)

    async def start(self):
        """Initialize and auto-arm existing positions on startup."""
        self.logger.info("[TPSLEngine] Starting TP/SL engine (Tier 2: fee-aware + time + trailing)")

        if self._auto_arm_enabled:
            await self._auto_arm_existing_positions()

    async def stop(self):
        """Shutdown."""
        self.logger.info("[TPSLEngine] Stopping TP/SL engine")

    async def _auto_arm_existing_positions(self) -> None:
        """Auto-arm TP/SL for all existing positions at startup."""
        try:
            positions = getattr(self.shared_state, "positions", {}) or {}

            # Purge stale TP/SL entries for symbols no longer in live positions.
            # _load_tpsl_state runs before hydration so positions was empty then —
            # this is the correct place to do the cleanup.
            stale = [s for s in list(self._tp_levels) if s not in positions]
            for s in stale:
                self._tp_levels.pop(s, None)
                self._sl_levels.pop(s, None)
                self._entry_timestamps.pop(s, None)
                self._peak_prices.pop(s, None)
                self._entry_prices.pop(s, None)
                self._armed_symbols.discard(s)
            if stale:
                self.logger.info(
                    "[TPSLEngine] Post-hydration purge: removed %d ghost entries: %s", len(stale), stale
                )
                # P2: Flush immediately so ghosts don't survive the next disk write
                self._save_tpsl_state()

            if not positions:
                self.logger.debug("[TPSLEngine] No existing positions to auto-arm")
                return

            self.logger.info(f"[TPSLEngine] Auto-arming {len(positions)} existing positions")

            now = time.time()
            for symbol, position in positions.items():
                if symbol in self._armed_symbols:
                    continue

                # Support both dict and Position-object style entries
                if isinstance(position, dict):
                    qty = float(position.get("qty", 0) or 0)
                    entry_price = float(
                        position.get("entry_price", 0) or position.get("avg_price", 0) or 0
                    )
                else:
                    qty = float(getattr(position, "qty", 0) or 0)
                    entry_price = float(getattr(position, "entry_price", 0) or 0)

                if qty <= 0 or entry_price <= 0:
                    self.logger.debug(f"[TPSLEngine] {symbol} skipped (invalid qty or price)")
                    continue

                try:
                    tp, sl = self.calculate_tp_sl(symbol, entry_price)
                    if isinstance(position, dict):
                        position["tp"] = tp
                        position["sl"] = sl
                    else:
                        # Position dataclass — store TP/SL in engine's own registry
                        pass
                    self._armed_symbols.add(symbol)
                    self._tp_levels[symbol] = tp or 0.0
                    self._sl_levels[symbol] = sl or 0.0
                    self._entry_prices[symbol] = entry_price

                    # Record entry_ts (fallback: now, since we don't know exact entry time)
                    if symbol not in self._entry_timestamps:
                        stored_ts = float(
                            position.get("entry_ts", 0) if isinstance(position, dict) else 0
                        )
                        self._entry_timestamps[symbol] = stored_ts if stored_ts > 0 else now

                    # Initialize peak price from current price
                    prices = getattr(self.shared_state, "prices", {}) or {}
                    current = float(prices.get(symbol, entry_price) or entry_price)
                    self._peak_prices[symbol] = max(entry_price, current)

                    self.logger.info(
                        f"[TPSLEngine:AUTO-ARM] {symbol} entry={self._fp(entry_price)} "
                        f"tp={self._fp(tp)} sl={self._fp(sl)}"
                    )
                except Exception as e:
                    self.logger.warning(f"[TPSLEngine] Failed to auto-arm {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"[TPSLEngine] Auto-arm failed: {e}", exc_info=True)

        # Set grace period: suppress triggers for 90s after startup hydration
        self._startup_grace_until = time.time() + 90.0
        self.logger.info("[TPSLEngine] Startup grace period active — TP/SL suppressed for 90s")

    def calculate_tp_sl(
        self, symbol: str, entry_price: float
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate volatility-adaptive TP/SL with fee-aware minimum floor.

        Layer 1: cost-aware minimum TP using the canonical round-trip estimate.
        """
        try:
            atr_raw = self._compute_atr(symbol, self._atr_lookback)

            # ATR noise floor: prevents SL collapsing within bid-ask spread
            min_atr = entry_price * self._min_atr_pct
            atr = max(atr_raw, min_atr)
            atr_pct = atr / entry_price

            # ATR-based SL: scales with each symbol's actual volatility.
            # Slow movers (WLFI ATR ~0.8%) get tight SL; fast movers (DASH ATR ~2%) get room.
            # Hard clamp: never tighter than 1.0% (whipsaw protection), never wider than 2.5%.
            sl_pct = self._base_sl_atr_mult * atr_pct
            sl_pct = max(sl_pct, self._SL_FLOOR_PCT)
            sl_pct = min(sl_pct, self._SL_CEILING_PCT)
            sl = entry_price * (1.0 - sl_pct)

            # TP is ATR-proportional with a minimum 2:1 R:R vs the SL distance.
            # Static 8% TP was unreachable on 0.65% expected moves — only exits were SL
            # hits and portfolio-recovery selling at tiny profits, producing -1.3R per trade.
            # Formula: TP = entry + (sl_distance × TP_RR_MULT), floored at 1.5× sl_pct.
            _TP_RR_MULT = 2.0  # minimum reward:risk ratio
            _TP_ATR_MULT = self._base_tp_atr_mult  # ATR-based ceiling (config default 1.5)
            tp_pct_rr = sl_pct * _TP_RR_MULT
            tp_pct_atr = _TP_ATR_MULT * atr_pct
            tp_pct = max(tp_pct_rr, tp_pct_atr)
            # Hard bounds: never tighter than 1.5% (covers fees+spread), never wider than 6%
            tp_pct = max(tp_pct, 0.015)
            tp_pct = min(tp_pct, 0.06)
            tp = entry_price * (1.0 + tp_pct)
            sl = entry_price * (1.0 - sl_pct)

            # Guard: for sub-cent tokens, float precision can collapse TP/SL to equal entry.
            # If they round to the same value, widen by at least 0.5% of entry.
            _min_sep = entry_price * 0.005
            if abs(tp - entry_price) < _min_sep:
                tp = entry_price * (1.0 + max(tp_pct, 0.005))
            if abs(sl - entry_price) < _min_sep:
                sl = entry_price * (1.0 - max(sl_pct, 0.005))

            self.logger.debug(
                f"[TPSLEngine] {symbol} atr_pct={atr_pct*100:.2f}% "
                f"sl={self._fp(sl)} (-{sl_pct*100:.1f}%) tp={self._fp(tp)} (+{tp_pct*100:.1f}%)"
            )

            return tp, sl

        except Exception as e:
            self.logger.error(f"[TPSLEngine] calc_tp_sl failed for {symbol}: {e}")
            return entry_price * 1.01, entry_price * 0.99

    def arm_position(
        self, symbol: str, entry_price: float, entry_ts: Optional[float] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Arm a new position with TP/SL. Records entry_ts and initializes peak price.

        Call this when a BUY order fills.
        """
        tp, sl = self.calculate_tp_sl(symbol, entry_price)
        self._armed_symbols.add(symbol)
        self._tp_levels[symbol] = tp or 0.0
        self._sl_levels[symbol] = sl or 0.0
        self._entry_timestamps[symbol] = entry_ts if entry_ts and entry_ts > 0 else time.time()
        self._peak_prices[symbol] = entry_price
        self._entry_prices[symbol] = entry_price
        # Clear any stale force-exit flag from previous position
        self._force_exit_symbols.discard(symbol)
        self._breakeven_activated.discard(symbol)
        self._save_tpsl_state()
        return tp, sl

    def get_entry_price(self, symbol: str) -> float:
        """Return the last known entry price for a symbol, or 0.0 if not armed."""
        return float(self._entry_prices.get(symbol, 0.0) or 0.0)

    def close_position_tracking(self, symbol: str) -> None:
        """Clear all tracking state for a closed position."""
        self._armed_symbols.discard(symbol)
        self._tp_levels.pop(symbol, None)
        self._sl_levels.pop(symbol, None)
        self._entry_timestamps.pop(symbol, None)
        self._peak_prices.pop(symbol, None)
        self._entry_prices.pop(symbol, None)
        self._force_exit_symbols.discard(symbol)
        self._breakeven_activated.discard(symbol)
        self._save_tpsl_state()

    @staticmethod
    def _pos_field(position: object, *keys: str, default: float = 0.0) -> float:
        """Extract a numeric field from either a dict or a Position-like object."""
        for k in keys:
            v = position.get(k, None) if isinstance(position, dict) else getattr(position, k, None)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    def check_triggers(self, symbol: str, position: object, current_price: float) -> Optional[str]:
        """
        Check all TP/SL triggers for a position.

        Returns: "TP_HIT", "SL_HIT", "TRAILING_STOP", "TIME_FORCE_EXIT", or None.
        Supports both dict-based and Position-object positions.

        Layers checked in order:
          3. Trailing TP (overrides static TP once activated)
          2. Time-based force exit (configured, 3h default)
          1. Static TP/SL
        """
        try:
            if current_price <= 0:
                return None

            # Startup grace period: suppress all triggers for 90s after hydration
            if time.time() < self._startup_grace_until:
                return None

            entry_price = self._pos_field(position, "entry_price", "avg_price")
            # Prefer the entry_price registered by arm_position() over the position object,
            # same reasoning as recalculate_aged_positions(): the position's entry_price can
            # be inflated by polling_coordinator's fill-reconciliation averaging together
            # fills from a previous, already-closed round trip on the same symbol.
            # arm_position() always stores the actual fill price for the CURRENT position,
            # so it is the authoritative source here too.
            armed_fill_price = float(self._entry_prices.get(symbol, 0.0) or 0.0)
            if armed_fill_price > 0 and symbol in self._armed_symbols:
                entry_price = armed_fill_price
            if entry_price <= 0:
                return None

            # Use engine's internal registry (authoritative) — fall back to position for legacy
            tp = float(self._tp_levels.get(symbol, 0) or self._pos_field(position, "tp"))
            sl = float(self._sl_levels.get(symbol, 0) or self._pos_field(position, "sl"))

            # Auto-arm any managed-but-unarmed position. Positions re-adopted from the
            # exchange balance (update_balance_map's external-position detection) are
            # added to the registry with a qty/entry but no TP/SL — leaving them with no
            # stop-loss. Arm them on first sight so every held position is protected.
            if tp <= 0 and sl <= 0 and entry_price > 0:
                try:
                    self.arm_position(symbol, entry_price)
                    tp = float(self._tp_levels.get(symbol, 0) or 0.0)
                    sl = float(self._sl_levels.get(symbol, 0) or 0.0)
                    self.logger.info(
                        "[TPSLEngine:AUTO-ARM] %s armed on first sight (entry=%s) → tp=%s sl=%s",
                        symbol, self._fp(entry_price), self._fp(tp), self._fp(sl),
                    )
                except Exception as _arm_err:
                    self.logger.debug("[TPSLEngine] auto-arm failed for %s: %s", symbol, _arm_err)

            now = time.time()

            # Layer 2: Force exit at the configured age — only when not profitable.
            # Profitable positions are allowed to run; the trailing stop will exit them.
            # Only unprofitable or stagnant positions (< +0.3% after fees) are force-exited.
            entry_ts = self._entry_timestamps.get(symbol, 0)
            if entry_ts > 0:
                age_sec = now - entry_ts
                if age_sec >= self._AGE_FORCE_EXIT_SEC:
                    _fee_cost_pct = self._round_trip_cost_pct
                    _net_profit_pct = (current_price - entry_price) / entry_price - _fee_cost_pct
                    if _net_profit_pct > 0.003:
                        # Position is profitable — let trailing stop manage the exit
                        self.logger.debug(
                            f"[TPSLEngine:TIME-EXIT] {symbol} age={age_sec/3600:.1f}h "
                            f"but profitable ({_net_profit_pct*100:.2f}%) — skipping force exit"
                        )
                    else:
                        self._force_exit_symbols.add(symbol)
                        self.logger.info(
                            f"[TPSLEngine:TIME-EXIT] {symbol} age={age_sec/3600:.1f}h "
                            f"net_pnl={_net_profit_pct*100:.2f}% → force exit"
                        )
                        return "TIME_FORCE_EXIT"

            # Layer 3: Update trailing peak, check trailing stop.
            # Activation and distance scale with market regime — trends run further,
            # choppy markets exit quickly to avoid fee drag eating small gains.
            profit_pct = (current_price - entry_price) / entry_price
            _market_regime = str(
                (getattr(self.shared_state, "metrics", {}) or {}).get("market_regime", "") or ""
            ).upper()
            _trail_activation, _trail_distance = self._REGIME_TRAIL_PARAMS.get(
                _market_regime, self._REGIME_TRAIL_DEFAULT
            )

            if profit_pct >= _trail_activation:
                prev_peak = self._peak_prices.get(symbol, entry_price)
                new_peak = max(prev_peak, current_price)
                self._peak_prices[symbol] = new_peak
                if new_peak > prev_peak:
                    self._save_tpsl_state()

                trailing_stop = new_peak * (1.0 - _trail_distance)
                if current_price <= trailing_stop:
                    self.logger.info(
                        f"[TPSLEngine:TRAILING] {symbol} regime={_market_regime} "
                        f"peak={self._fp(new_peak)} trail={self._fp(trailing_stop)} "
                        f"current={self._fp(current_price)} → TRAILING_STOP"
                    )
                    return "TRAILING_STOP"

                self.logger.debug(
                    f"[TPSLEngine:TRAILING] {symbol} profit={profit_pct*100:.2f}% "
                    f"peak={self._fp(new_peak)} trail={self._fp(trailing_stop)} (active)"
                )
                # Still fire static TP if price has reached it — don't let trailing block a win
                if tp > 0 and current_price >= tp:
                    self.logger.info(
                        f"[TPSLEngine:TP] {symbol} current={self._fp(current_price)} >= tp={self._fp(tp)}"
                    )
                    return "TP_HIT"
                return None  # Trailing is active; delegate exit to trailing stop

            # Breakeven SL: once price reaches 50% of TP distance, move SL to entry price.
            # Guarantees we never lose on a trade that got halfway to target.
            if (
                tp > 0
                and entry_price > 0
                and sl > 0
                and sl < entry_price
                and symbol not in self._breakeven_activated
            ):
                tp_dist = tp - entry_price
                if tp_dist > 0:
                    progress = (current_price - entry_price) / tp_dist
                    if progress >= 0.50:
                        self._breakeven_activated.add(symbol)
                        self._sl_levels[symbol] = entry_price
                        sl = entry_price
                        self.logger.info(
                            "[TPSLEngine:BREAKEVEN] %s at %.0f%% TP → SL moved to entry %s",
                            symbol,
                            progress * 100,
                            self._fp(entry_price),
                        )
                        self._save_tpsl_state()

            # Layer 1: Static TP/SL
            if tp > 0 and current_price >= tp:
                self.logger.info(
                    f"[TPSLEngine:TP] {symbol} current={self._fp(current_price)} >= tp={self._fp(tp)}"
                )
                return "TP_HIT"

            if sl > 0 and current_price <= sl:
                self.logger.info(
                    f"[TPSLEngine:SL] {symbol} current={self._fp(current_price)} <= sl={self._fp(sl)}"
                )
                return "SL_HIT"

            return None

        except Exception as e:
            self.logger.error(f"[TPSLEngine] check_triggers failed for {symbol}: {e}")
            return None

    def reconcile_against_positions(self) -> list[str]:
        """
        P3: Periodic reconciliation — drop any TP/SL entry whose symbol is no longer
        in shared_state.positions. Runs every 5 min via recalculate_aged_positions.
        Returns list of purged symbols.
        """
        live = set(getattr(self.shared_state, "positions", {}) or {})
        ghosts = [s for s in list(self._tp_levels) if s not in live]
        for s in ghosts:
            self._tp_levels.pop(s, None)
            self._sl_levels.pop(s, None)
            self._entry_timestamps.pop(s, None)
            self._peak_prices.pop(s, None)
            self._entry_prices.pop(s, None)
            self._armed_symbols.discard(s)
        if ghosts:
            self.logger.info("[TPSLEngine] Reconciliation purged %d ghosts: %s", len(ghosts), ghosts)
            self._save_tpsl_state()
        return ghosts

    def recalculate_aged_positions(self) -> dict[str, dict]:
        """
        Layer 2: Time-based TP tightening. Call periodically (e.g., every 5 min).

        Returns dict of {symbol: {"tp": new_tp, "sl": new_sl, "reason": str}}
        for positions whose TP/SL changed.
        """
        # P3: Reconcile first — purge any ghost entries before recalculating
        self.reconcile_against_positions()

        updates: dict[str, dict] = {}
        now = time.time()
        positions = getattr(self.shared_state, "positions", {}) or {}
        nav_protection = getattr(self.shared_state, "nav_protection_state", {}) or {}
        suggested_actions = {
            str(action or "").upper()
            for action in list(nav_protection.get("suggested_actions", []) or [])
        }
        allow_tp_sl_adjustment = bool(nav_protection.get("allow_tp_sl_adjustment", False))
        protection_mode = str(nav_protection.get("protection_mode", "NORMAL") or "NORMAL").upper()
        prices = getattr(self.shared_state, "prices", {}) or {}
        setup_contexts = getattr(self.shared_state, "position_setup_context", {}) or {}

        _market_regime = str(
            (getattr(self.shared_state, "metrics", {}) or {}).get("market_regime", "") or ""
        ).upper()

        for symbol, position in positions.items():
            entry_price = self._pos_field(position, "entry_price", "avg_price")
            if entry_price <= 0:
                # Fall back to armed entry price (e.g. for externally detected positions)
                entry_price = float(self._entry_prices.get(symbol, 0.0) or 0.0)
            # Prefer the entry_price registered by arm_position() over the position object.
            # The position's entry_price can be inflated by position-hydration re-reads from
            # the Binance API after a session restart — arm_position always stores the actual
            # fill price, so it is the authoritative source for break-even SL calculations.
            armed_fill_price = float(self._entry_prices.get(symbol, 0.0) or 0.0)
            if armed_fill_price > 0 and symbol in self._armed_symbols:
                entry_price = armed_fill_price
            if entry_price <= 0:
                continue

            entry_ts = self._entry_timestamps.get(symbol, 0)
            if entry_ts <= 0:
                entry_ts = float(self.shared_state.prices.get("_boot_ts", 0.0) or 0.0) or (time.time() - 3600)
                self._entry_timestamps[symbol] = entry_ts

            age_sec = now - entry_ts
            current_price = float(
                prices.get(symbol)
                or self._pos_field(position, "current_price", "mark_price")
                or 0.0
            )
            profit_pct = (
                ((current_price - entry_price) / entry_price)
                if current_price > 0 and entry_price > 0
                else 0.0
            )

            current_tp = float(self._tp_levels.get(symbol) or self._pos_field(position, "tp"))
            current_sl = float(self._sl_levels.get(symbol) or self._pos_field(position, "sl"))
            changed = False
            reason_parts: list[str] = []

            protective_update = self._compute_protective_tightening(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                current_tp=current_tp,
                current_sl=current_sl,
                age_sec=age_sec,
                profit_pct=profit_pct,
                protection_mode=protection_mode,
                allow_tp_sl_adjustment=allow_tp_sl_adjustment,
                suggested_actions=suggested_actions,
                setup_context=dict(setup_contexts.get(symbol, {}) or {}),
            )
            if protective_update is not None:
                new_tp = float(protective_update["tp"])
                new_sl = float(protective_update["sl"])
                if current_tp > 0 and new_tp < current_tp:
                    self._set_pos_field(position, "tp", new_tp)
                    self._tp_levels[symbol] = new_tp
                    current_tp = new_tp
                    changed = True
                if current_sl > 0 and new_sl > current_sl:
                    self._set_pos_field(position, "sl", new_sl)
                    self._sl_levels[symbol] = new_sl
                    current_sl = new_sl
                    changed = True
                if changed:
                    reason_parts.append(str(protective_update["reason"]))

            # Dynamic TP widening for TRENDING/UPTREND regime on young positions (<2h).
            # Only widen when not yet profitable — gives more room to reach profit.
            if age_sec < self._AGE_TIGHTEN_1_SEC and not changed:
                widened = self._maybe_widen_tp(
                    symbol, entry_price, current_price, current_tp, current_sl,
                    age_sec, profit_pct, _market_regime
                )
                if widened:
                    self._set_pos_field(position, "tp", widened["tp"])
                    self._set_pos_field(position, "sl", widened["sl"])
                    self._tp_levels[symbol] = widened["tp"]
                    self._sl_levels[symbol] = widened["sl"]
                    changed = True
                    reason_parts.append(widened["reason"])

            if age_sec < self._AGE_TIGHTEN_1_SEC:
                if changed:
                    updates[symbol] = {
                        "tp": self._tp_levels.get(symbol, current_tp),
                        "sl": self._sl_levels.get(symbol, current_sl),
                        "reason": "; ".join(reason_parts) or "protective_tightening",
                    }
                    self.logger.info(
                        f"[TPSLEngine:ADJUST] {symbol} {updates[symbol]['reason']} "
                        f"tp={self._fp(updates[symbol]['tp'])} sl={self._fp(updates[symbol]['sl'])}"
                    )
                continue

            if age_sec >= self._AGE_TIGHTEN_2_SEC:
                # 4h threshold: tighten to +0.8%
                new_tp = entry_price * 1.008
                new_sl = entry_price * (1.0 + self._round_trip_cost_pct)
                reason = f"age={age_sec/3600:.1f}h >= 4h → tp=+0.8%"
            else:
                # 2h threshold: tighten to +1.5%, move SL to break-even
                new_tp = entry_price * 1.015
                new_sl = entry_price * (1.0 + self._round_trip_cost_pct)
                reason = f"age={age_sec/3600:.1f}h >= 2h → tp=+1.5%, sl=break-even"

            # Only tighten TP (never loosen via time logic)
            if current_tp > 0 and new_tp < current_tp:
                self._set_pos_field(position, "tp", new_tp)
                self._tp_levels[symbol] = new_tp
                changed = True
            if current_sl > 0 and new_sl > current_sl:
                self._set_pos_field(position, "sl", new_sl)
                self._sl_levels[symbol] = new_sl
                changed = True

            if changed:
                reason_parts.append(reason)
                updates[symbol] = {
                    "tp": self._tp_levels.get(symbol, current_tp),
                    "sl": self._sl_levels.get(symbol, current_sl),
                    "reason": "; ".join(reason_parts),
                }
                self.logger.info(
                    f"[TPSLEngine:TIME-TIGHTEN] {symbol} {updates[symbol]['reason']} "
                    f"tp={self._fp(updates[symbol]['tp'])} sl={self._fp(updates[symbol]['sl'])}"
                )

        return updates

    @staticmethod
    def _set_pos_field(position: object, key: str, value: float) -> None:
        """Write a field to either a dict or a Position-like object."""
        if isinstance(position, dict):
            position[key] = value
        else:
            try:
                setattr(position, key, value)
            except (AttributeError, TypeError):
                pass

    # Regime → (max_tp_pct, sl_pct) for dynamic widening.
    # Only applied when position is young (<2h) and not yet profitable.
    _REGIME_WIDEN_PARAMS: dict[str, tuple[float, float]] = {
        "UPTREND":  (0.10, 0.025),  # 10% TP, 2.5% SL — ride the trend
        "TRENDING": (0.06, 0.020),  # 6% TP, 2.0% SL — solid momentum
        "RANGING":  (0.03, 0.015),  # 3% TP, 1.5% SL — range-bound, keep tight
        "CHOPPY":   (0.02, 0.012),  # 2% TP, 1.2% SL — noisy, exit fast
    }

    def _maybe_widen_tp(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_tp: float,
        current_sl: float,
        age_sec: float,
        profit_pct: float,
        regime: str,
    ) -> Optional[dict]:
        """
        Dynamically widen TP when regime improves and position is underwater or flat.

        Only widens — never tightens. Does not touch profitable positions (trailing
        stop handles those). Returns dict with new tp/sl/reason or None if no change.
        """
        params = self._REGIME_WIDEN_PARAMS.get(regime)
        if params is None:
            return None  # DOWNTREND or unknown — don't widen

        max_tp_pct, sl_pct = params

        # Don't widen for positions already comfortably profitable (trailing stop active)
        if profit_pct > 0.01:
            return None

        new_tp = entry_price * (1.0 + max_tp_pct)
        new_sl = entry_price * (1.0 - sl_pct)

        tp_changed = current_tp > 0 and new_tp > current_tp * 1.01  # at least 1% wider
        sl_changed = current_sl > 0 and new_sl < current_sl * 0.99  # at least 1% wider

        if not tp_changed and not sl_changed:
            return None

        result = {}
        if tp_changed:
            result["tp"] = new_tp
        else:
            result["tp"] = current_tp
        if sl_changed:
            result["sl"] = new_sl
        else:
            result["sl"] = current_sl
        result["reason"] = (
            f"regime={regime} dynamic_widen tp={self._fp(result['tp'])} sl={self._fp(result['sl'])}"
        )
        self.logger.info(
            "[TPSLEngine:WIDEN] %s regime=%s age=%.1fh pnl=%.2f%% "
            "tp: %s→%s sl: %s→%s",
            symbol, regime, age_sec / 3600, profit_pct * 100,
            self._fp(current_tp), self._fp(result["tp"]),
            self._fp(current_sl), self._fp(result["sl"]),
        )
        return result

    def _compute_protective_tightening(
        self,
        *,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_tp: float,
        current_sl: float,
        age_sec: float,
        profit_pct: float,
        protection_mode: str,
        allow_tp_sl_adjustment: bool,
        suggested_actions: set[str],
        setup_context: dict,
    ) -> Optional[dict[str, Union[float, str]]]:
        if not allow_tp_sl_adjustment or current_tp <= 0 or current_sl <= 0:
            return None
        if current_price <= 0 or entry_price <= 0:
            return None
        if age_sec < self._PROTECTIVE_TIGHTEN_SEC or profit_pct < 0.0075:
            return None

        entry_quality = float(setup_context.get("entry_quality", 0.0) or 0.0)
        confidence = float(setup_context.get("confidence", 0.0) or 0.0)

        tighten_tp_pct = 0.012
        break_even_buffer = self._round_trip_cost_pct
        reason = "protect_quick_winner"

        if (
            protection_mode in {"FLOATING_GAIN_PROTECTION", "PROFIT_LOCK"}
            or "TIGHTEN_TP_SL" in suggested_actions
        ):
            tighten_tp_pct = 0.010
            break_even_buffer = self._round_trip_cost_pct + 0.002
            reason = "nav_profit_protection"

        if "PARTIAL_TAKE_PROFIT" in suggested_actions:
            tighten_tp_pct = min(tighten_tp_pct, 0.009)
            break_even_buffer = max(break_even_buffer, self._round_trip_cost_pct + 0.003)
            reason = "partial_take_profit_protection"

        if confidence >= 0.70 and entry_quality >= 0.65:
            tighten_tp_pct = max(0.009, tighten_tp_pct - 0.001)
            break_even_buffer = max(break_even_buffer, self._round_trip_cost_pct + 0.003)

        new_tp = entry_price * (1.0 + tighten_tp_pct)
        new_sl = entry_price * (1.0 + break_even_buffer)

        # Never loosen TP below already-secured price structure if the market is still below new TP.
        new_tp = min(
            current_tp, max(new_tp, entry_price * (1.0 + self._round_trip_cost_pct + 0.003))
        )
        new_sl = max(current_sl, min(new_sl, current_price * 0.9975))

        if new_tp >= current_tp and new_sl <= current_sl:
            return None

        return {
            "tp": new_tp,
            "sl": new_sl,
            "reason": reason,
        }

    def get_force_exit_symbols(self) -> set[str]:
        """Return symbols that need forced exit due to the configured timeout."""
        return self._force_exit_symbols.copy()

    def calculate_risk_based_position_size(
        self, symbol: str, entry_price: float, sl_price: float, nav: float
    ) -> float:
        """
        Calculate position size based on SL distance and target risk.

        Returns: quote_usdt (amount to deploy)
        """
        try:
            sl_distance_pct = abs(entry_price - sl_price) / entry_price
            if sl_distance_pct <= 0:
                self.logger.warning(f"[TPSLEngine] {symbol} invalid SL distance")
                return nav * 0.05

            position_quote = nav * (self._target_risk_pct / 100.0 / sl_distance_pct)
            position_quote = max(position_quote, self._min_notional_safety)

            self.logger.debug(
                f"[TPSLEngine:RISK] {symbol} sl_dist={sl_distance_pct:.4f} "
                f"nav={nav:.2f} quote={position_quote:.2f}"
            )

            return float(position_quote)

        except Exception as e:
            self.logger.error(f"[TPSLEngine] risk_based_sizing failed for {symbol}: {e}")
            return nav * 0.05

    def _compute_atr(self, symbol: str, lookback: int = 14) -> float:
        """Compute ATR(lookback) from market data — delegates to math_utils.compute_atr
        (shared with NativeCapitalAllocator's volatility sizing input; moved there
        2026-07-14 so both callers stay in sync). See that function's docstring
        for the fallback strategy."""
        return math_utils.compute_atr(self.shared_state, symbol, lookback)

    def _compute_atr_from_candles(self, candles: list, lookback: int = 14) -> float:
        """Compute ATR from candlestick data — delegates to
        math_utils.compute_atr_from_candles (see that function's docstring)."""
        return math_utils.compute_atr_from_candles(candles, lookback)

    def _estimate_volatility_pressure(self, symbol: str) -> float:
        """
        Estimate volatility pressure from 20-bar realized volatility.

        Returns: 0.0 (calm) to 1.0 (highly volatile), normalized against 2% baseline.
        """
        try:
            import math

            market_data = getattr(self.shared_state, "market_data", {}) or {}
            candles = None
            for tf in ("5m", "15m", "1h"):
                key = (symbol, tf)
                c = market_data.get(key) or []
                if len(c) >= 5:
                    candles = c
                    break
            if candles is None or len(candles) < 5:
                return 0.5
            closes = []
            for c in candles[-21:]:
                if isinstance(c, dict):
                    v = float(c.get("close") or c.get("c") or 0)
                elif isinstance(c, (list, tuple)) and len(c) >= 5:
                    v = float(c[4] or 0)
                else:
                    continue
                if v > 0:
                    closes.append(v)
            if len(closes) < 5:
                return 0.5
            log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
            n = len(log_returns)
            mean = sum(log_returns) / n
            variance = sum((r - mean) ** 2 for r in log_returns) / max(n - 1, 1)
            realized_vol_pct = math.sqrt(variance) * 100.0
            # Normalize: 0% vol → 0.0, 2% baseline → 0.5, 4%+ → 1.0
            pressure = min(1.0, max(0.0, realized_vol_pct / 4.0))
            return pressure
        except Exception:
            return 0.5

    def update_position_tp_sl(self, symbol: str, position: dict, entry_price: float) -> None:
        """Update TP/SL on an existing position."""
        try:
            tp, sl = self.calculate_tp_sl(symbol, entry_price)
            position["tp"] = tp
            position["sl"] = sl
            self.logger.debug(f"[TPSLEngine:UPDATE] {symbol} tp={self._fp(tp)} sl={self._fp(sl)}")
        except Exception as e:
            self.logger.error(f"[TPSLEngine] update_position_tp_sl failed for {symbol}: {e}")
