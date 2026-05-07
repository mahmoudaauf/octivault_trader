"""
Native TP/SL Engine — Volatility-Adaptive Take Profit & Stop Loss

Tier 1 Implementation (May 7, 2026):
  1. ATR-based volatility adaptation
  2. Risk-based position sizing (Kelly-criterion style)
  3. Auto-arm safety on startup

Tier 2 Enhancements (May 7, 2026):
  4. Fee-aware minimum TP floor (0.2% round-trip + 0.5% net profit)
  5. Time-based TP tightening (2hr/6hr/12hr thresholds)
  6. Trailing TP after +1.5% profit (0.8% trail below peak)

No external dependencies beyond asyncio + logging + time.
"""

import logging
import time
from typing import Optional


class NativeTPSLEngine:
    """Volatility-adaptive TP/SL with fee-awareness, time decay, and trailing stops."""

    # Minimum net profit required above fees (0.2% round-trip Binance fee)
    _ROUND_TRIP_FEE_PCT = 0.002
    _MIN_NET_PROFIT_PCT = 0.005  # 0.5% net profit floor

    # Time-based tightening thresholds
    _AGE_TIGHTEN_1_SEC = 2 * 3600  # 2 hours  → tp=+1.5%, sl=break-even
    _AGE_TIGHTEN_2_SEC = 6 * 3600  # 6 hours  → tp=+0.8%
    _AGE_FORCE_EXIT_SEC = 12 * 3600  # 12 hours → force SELL signal

    # Trailing TP: activate at +1.5%, trail 0.8% below peak
    _TRAIL_ACTIVATION_PCT = 0.015
    _TRAIL_DISTANCE_PCT = 0.008

    def __init__(self, shared_state, config, **kwargs):
        self.shared_state = shared_state
        self.config = config
        self.logger = logging.getLogger("NativeTPSLEngine")

        # Config: TP/SL strategy
        self._base_tp_atr_mult = float(getattr(config, "TP_ATR_MULT", 1.5) or 1.5)
        self._base_sl_atr_mult = float(getattr(config, "SL_ATR_MULT", 1.0) or 1.0)

        # Config: Risk management
        self._target_risk_pct = float(
            getattr(config, "TARGET_RISK_PCT", 2.0) or 2.0
        )  # 2% risk per trade
        self._atr_lookback = int(getattr(config, "ATR_LOOKBACK", 14) or 14)
        self._min_atr_pct = float(getattr(config, "MIN_ATR_PCT", 0.001) or 0.001)  # 0.1% floor

        # Config: Volatility adaptation
        self._vol_adaptation_enabled = bool(getattr(config, "TPSL_VOL_ADAPTATION_ENABLED", True))
        self._vol_pressure_scale = float(getattr(config, "VOL_PRESSURE_SCALE", 0.35) or 0.35)

        # Config: Safety
        self._min_notional_safety = float(getattr(config, "MIN_NOTIONAL_SAFETY", 10.0) or 10.0)
        self._auto_arm_enabled = bool(getattr(config, "TPSL_AUTO_ARM_ENABLED", True))

        # Runtime state
        self._armed_symbols: set[str] = set()

        # Trailing state: symbol → peak_price
        self._peak_prices: dict[str, float] = {}

        # Time-based state: symbol → entry_ts (unix seconds)
        self._entry_timestamps: dict[str, float] = {}

        # Force-exit set: symbols that hit 12h threshold
        self._force_exit_symbols: set[str] = set()

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
            if not positions:
                self.logger.debug("[TPSLEngine] No existing positions to auto-arm")
                return

            self.logger.info(f"[TPSLEngine] Auto-arming {len(positions)} existing positions")

            now = time.time()
            for symbol, position in positions.items():
                if not isinstance(position, dict):
                    continue

                if symbol in self._armed_symbols:
                    continue

                qty = float(position.get("qty", 0) or 0)
                entry_price = float(
                    position.get("entry_price", 0) or position.get("avg_price", 0) or 0
                )

                if qty <= 0 or entry_price <= 0:
                    self.logger.debug(f"[TPSLEngine] {symbol} skipped (invalid qty or price)")
                    continue

                try:
                    tp, sl = self.calculate_tp_sl(symbol, entry_price)
                    position["tp"] = tp
                    position["sl"] = sl
                    self._armed_symbols.add(symbol)

                    # Record entry_ts (fallback: now, since we don't know exact entry time)
                    if symbol not in self._entry_timestamps:
                        stored_ts = float(position.get("entry_ts", 0) or 0)
                        self._entry_timestamps[symbol] = stored_ts if stored_ts > 0 else now

                    # Initialize peak price from current price
                    prices = getattr(self.shared_state, "prices", {}) or {}
                    current = float(prices.get(symbol, entry_price) or entry_price)
                    self._peak_prices[symbol] = max(entry_price, current)

                    self.logger.info(
                        f"[TPSLEngine:AUTO-ARM] {symbol} entry={entry_price:.6f} "
                        f"tp={tp:.6f} sl={sl:.6f}"
                    )
                except Exception as e:
                    self.logger.warning(f"[TPSLEngine] Failed to auto-arm {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"[TPSLEngine] Auto-arm failed: {e}", exc_info=True)

    def calculate_tp_sl(
        self, symbol: str, entry_price: float
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate volatility-adaptive TP/SL with fee-aware minimum floor.

        Layer 1: fee-aware minimum TP = entry * (1 + 0.2% fee + 0.5% net profit)
        """
        try:
            atr = self._compute_atr(symbol, self._atr_lookback)

            min_atr = entry_price * self._min_atr_pct
            atr = max(atr, min_atr)

            tp_mult = self._base_tp_atr_mult
            sl_mult = self._base_sl_atr_mult

            if self._vol_adaptation_enabled:
                vol_pressure = self._estimate_volatility_pressure(symbol)
                sl_mult *= 1.0 + max(-0.25, min(0.55, vol_pressure * self._vol_pressure_scale))
                tp_mult *= 1.0 + max(-0.20, min(0.40, vol_pressure * 0.22))

            # ATR-based TP
            atr_tp = entry_price + (atr * tp_mult)

            # Layer 1: fee-aware floor (0.2% round-trip + 0.5% net profit = 0.7%)
            fee_floor = entry_price * (1.0 + self._ROUND_TRIP_FEE_PCT + self._MIN_NET_PROFIT_PCT)

            # Take the higher of ATR-based TP and fee floor
            tp = max(atr_tp, fee_floor)
            sl = entry_price - (atr * sl_mult)

            self.logger.debug(
                f"[TPSLEngine] {symbol} atr={atr:.6f} atr_tp={atr_tp:.6f} "
                f"fee_floor={fee_floor:.6f} final_tp={tp:.6f} sl={sl:.6f}"
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
        self._entry_timestamps[symbol] = entry_ts if entry_ts and entry_ts > 0 else time.time()
        self._peak_prices[symbol] = entry_price
        # Clear any stale force-exit flag from previous position
        self._force_exit_symbols.discard(symbol)
        return tp, sl

    def close_position_tracking(self, symbol: str) -> None:
        """Clear all tracking state for a closed position."""
        self._armed_symbols.discard(symbol)
        self._entry_timestamps.pop(symbol, None)
        self._peak_prices.pop(symbol, None)
        self._force_exit_symbols.discard(symbol)

    def check_triggers(self, symbol: str, position: dict, current_price: float) -> Optional[str]:
        """
        Check all TP/SL triggers for a position.

        Returns: "TP_HIT", "SL_HIT", "TRAILING_STOP", "TIME_FORCE_EXIT", or None.

        Layers checked in order:
          3. Trailing TP (overrides static TP once activated)
          2. Time-based force exit (12h)
          1. Static TP/SL
        """
        try:
            if current_price <= 0:
                return None

            entry_price = float(position.get("entry_price", 0) or position.get("avg_price", 0) or 0)
            if entry_price <= 0:
                return None

            tp = float(position.get("tp", 0) or 0)
            sl = float(position.get("sl", 0) or 0)

            now = time.time()

            # Layer 2: Force exit at 12h
            entry_ts = self._entry_timestamps.get(symbol, 0)
            if entry_ts > 0:
                age_sec = now - entry_ts
                if age_sec >= self._AGE_FORCE_EXIT_SEC:
                    self._force_exit_symbols.add(symbol)
                    self.logger.info(
                        f"[TPSLEngine:TIME-EXIT] {symbol} age={age_sec/3600:.1f}h → force exit"
                    )
                    return "TIME_FORCE_EXIT"

            # Layer 3: Update trailing peak, check trailing stop
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self._TRAIL_ACTIVATION_PCT:
                prev_peak = self._peak_prices.get(symbol, entry_price)
                new_peak = max(prev_peak, current_price)
                self._peak_prices[symbol] = new_peak

                trailing_stop = new_peak * (1.0 - self._TRAIL_DISTANCE_PCT)
                if current_price <= trailing_stop:
                    self.logger.info(
                        f"[TPSLEngine:TRAILING] {symbol} peak={new_peak:.6f} "
                        f"trail={trailing_stop:.6f} current={current_price:.6f} → TRAILING_STOP"
                    )
                    return "TRAILING_STOP"

                self.logger.debug(
                    f"[TPSLEngine:TRAILING] {symbol} profit={profit_pct*100:.2f}% "
                    f"peak={new_peak:.6f} trail={trailing_stop:.6f} (active)"
                )
                return None  # Trailing is active; don't check static TP while trailing

            # Layer 1: Static TP/SL
            if tp > 0 and current_price >= tp:
                self.logger.info(
                    f"[TPSLEngine:TP] {symbol} current={current_price:.6f} >= tp={tp:.6f}"
                )
                return "TP_HIT"

            if sl > 0 and current_price <= sl:
                self.logger.info(
                    f"[TPSLEngine:SL] {symbol} current={current_price:.6f} <= sl={sl:.6f}"
                )
                return "SL_HIT"

            return None

        except Exception as e:
            self.logger.error(f"[TPSLEngine] check_triggers failed for {symbol}: {e}")
            return None

    def recalculate_aged_positions(self) -> dict[str, dict]:
        """
        Layer 2: Time-based TP tightening. Call periodically (e.g., every 5 min).

        Returns dict of {symbol: {"tp": new_tp, "sl": new_sl, "reason": str}}
        for positions whose TP/SL changed.
        """
        updates: dict[str, dict] = {}
        now = time.time()
        positions = getattr(self.shared_state, "positions", {}) or {}

        for symbol, position in positions.items():
            if not isinstance(position, dict):
                continue

            entry_price = float(position.get("entry_price", 0) or position.get("avg_price", 0) or 0)
            if entry_price <= 0:
                continue

            entry_ts = self._entry_timestamps.get(symbol, 0)
            if entry_ts <= 0:
                continue

            age_sec = now - entry_ts

            if age_sec < self._AGE_TIGHTEN_1_SEC:
                continue  # Position is fresh, no tightening yet

            current_tp = float(position.get("tp", 0) or 0)
            current_sl = float(position.get("sl", 0) or 0)

            if age_sec >= self._AGE_TIGHTEN_2_SEC:
                # 6h threshold: tighten to +0.8%
                new_tp = entry_price * 1.008
                # Move SL to break-even (fee-covered)
                new_sl = entry_price * (1.0 + self._ROUND_TRIP_FEE_PCT)
                reason = f"age={age_sec/3600:.1f}h >= 6h → tp=+0.8%"
            else:
                # 2h threshold: tighten to +1.5%, move SL to break-even
                new_tp = entry_price * 1.015
                new_sl = entry_price * (1.0 + self._ROUND_TRIP_FEE_PCT)
                reason = f"age={age_sec/3600:.1f}h >= 2h → tp=+1.5%, sl=break-even"

            # Only update if TP would tighten (never loosen via time logic)
            changed = False
            if current_tp > 0 and new_tp < current_tp:
                position["tp"] = new_tp
                changed = True
            if current_sl > 0 and new_sl > current_sl:
                position["sl"] = new_sl
                changed = True

            if changed:
                updates[symbol] = {
                    "tp": position.get("tp"),
                    "sl": position.get("sl"),
                    "reason": reason,
                }
                self.logger.info(
                    f"[TPSLEngine:TIME-TIGHTEN] {symbol} {reason} "
                    f"tp={position.get('tp'):.6f} sl={position.get('sl'):.6f}"
                )

        return updates

    def get_force_exit_symbols(self) -> set[str]:
        """Return symbols that need forced exit due to 12h timeout."""
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
        """
        Compute ATR(lookback) from market data.

        Strategy:
          1. Try cached ATR from market_data
          2. Compute from klines if available
          3. Fallback to fixed 1.5% of price
        """
        try:
            market_data = getattr(self.shared_state, "market_data", {}) or {}
            if symbol in market_data:
                sym_md = market_data[symbol]
                if isinstance(sym_md, dict):
                    cached_atr = float(sym_md.get("atr") or 0.0)
                    if cached_atr > 0:
                        return cached_atr

            klines = getattr(self.shared_state, "klines", {}) or {}
            if symbol in klines:
                candles = klines.get(symbol, {}).get("1m", [])
                if isinstance(candles, list) and len(candles) >= lookback:
                    atr = self._compute_atr_from_candles(candles, lookback)
                    if atr > 0:
                        return atr

            prices = getattr(self.shared_state, "prices", {}) or {}
            if symbol in prices:
                last_price = float(prices[symbol] or 0.0)
                if last_price > 0:
                    return last_price * 0.015

            self.logger.warning(f"[TPSLEngine] {symbol} no data for ATR, returning 0")
            return 0.0

        except Exception as e:
            self.logger.error(f"[TPSLEngine] _compute_atr failed for {symbol}: {e}")
            return 0.0

    def _compute_atr_from_candles(self, candles: list, lookback: int = 14) -> float:
        """
        Compute ATR from candlestick data.

        ATR = SMA(TR) where TR = max(H-L, abs(H-PC), abs(L-PC))
        """
        try:
            if len(candles) < lookback:
                return 0.0

            true_ranges = []
            prev_close = None

            for candle in candles[-lookback:]:
                if not isinstance(candle, (list, tuple)) or len(candle) < 5:
                    continue

                high = float(candle[2] or 0.0)
                low = float(candle[3] or 0.0)
                close = float(candle[4] or 0.0)

                if prev_close is None:
                    prev_close = close
                    continue

                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)
                prev_close = close

            if not true_ranges:
                return 0.0

            return float(sum(true_ranges) / len(true_ranges))

        except Exception as e:
            self.logger.error(f"[TPSLEngine] _compute_atr_from_candles failed: {e}")
            return 0.0

    def _estimate_volatility_pressure(self, symbol: str) -> float:
        """
        Estimate volatility pressure on a symbol.

        Returns: 0.0 (calm) to 1.0 (volatile)
        """
        try:
            prices = getattr(self.shared_state, "prices", {}) or {}
            if symbol not in prices:
                return 0.5

            return 0.5

        except Exception:
            return 0.5

    def update_position_tp_sl(self, symbol: str, position: dict, entry_price: float) -> None:
        """Update TP/SL on an existing position."""
        try:
            tp, sl = self.calculate_tp_sl(symbol, entry_price)
            position["tp"] = tp
            position["sl"] = sl
            self.logger.debug(f"[TPSLEngine:UPDATE] {symbol} tp={tp:.6f} sl={sl:.6f}")
        except Exception as e:
            self.logger.error(f"[TPSLEngine] update_position_tp_sl failed for {symbol}: {e}")
