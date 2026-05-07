"""
Native TP/SL Engine — Volatility-Adaptive Take Profit & Stop Loss

Tier 1 Implementation (May 7, 2026):
  1. ATR-based volatility adaptation
  2. Risk-based position sizing (Kelly-criterion style)
  3. Auto-arm safety on startup

Features:
  • Computes ATR(14) from klines or cached market data
  • Adapts TP/SL distances based on volatility
  • Derives position size from SL distance (ensures fixed risk per trade)
  • Auto-arms existing positions on startup (safety)
  • Minimal: 300 lines, focused on core value

No external dependencies beyond asyncio + logging.
"""

import logging
from typing import Optional


class NativeTPSLEngine:
    """Volatility-adaptive TP/SL with risk-based position sizing."""

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

    async def start(self):
        """Initialize and auto-arm existing positions on startup."""
        self.logger.info("[TPSLEngine] Starting TP/SL engine")

        if self._auto_arm_enabled:
            await self._auto_arm_existing_positions()

    async def stop(self):
        """Shutdown."""
        self.logger.info("[TPSLEngine] Stopping TP/SL engine")

    async def _auto_arm_existing_positions(self) -> None:
        """Auto-arm TP/SL for all existing positions at startup (safety)."""
        try:
            positions = getattr(self.shared_state, "positions", {}) or {}
            if not positions:
                self.logger.debug("[TPSLEngine] No existing positions to auto-arm")
                return

            self.logger.info(f"[TPSLEngine] Auto-arming {len(positions)} existing positions")

            for symbol, position in positions.items():
                if not isinstance(position, dict):
                    continue

                # Skip if already armed
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
        Calculate volatility-adaptive TP/SL.

        Returns (tp_price, sl_price).
        """
        try:
            # Get ATR for volatility
            atr = self._compute_atr(symbol, self._atr_lookback)

            # Floor: at least X% of entry price
            min_atr = entry_price * self._min_atr_pct
            atr = max(atr, min_atr)

            # Base multipliers
            tp_mult = self._base_tp_atr_mult
            sl_mult = self._base_sl_atr_mult

            # Volatility adaptation: scale multipliers with volatility
            if self._vol_adaptation_enabled:
                vol_pressure = self._estimate_volatility_pressure(symbol)
                # Higher vol → wider SL, slightly wider TP (preserve edge)
                sl_mult *= 1.0 + max(-0.25, min(0.55, vol_pressure * self._vol_pressure_scale))
                tp_mult *= 1.0 + max(-0.20, min(0.40, vol_pressure * 0.22))

            # Calculate TP/SL
            tp = entry_price + (atr * tp_mult)
            sl = entry_price - (atr * sl_mult)

            self.logger.debug(
                f"[TPSLEngine] {symbol} atr={atr:.6f} tp_mult={tp_mult:.2f} "
                f"sl_mult={sl_mult:.2f} tp={tp:.6f} sl={sl:.6f}"
            )

            return tp, sl

        except Exception as e:
            self.logger.error(f"[TPSLEngine] calc_tp_sl failed for {symbol}: {e}")
            # Fallback: simple fixed %
            return entry_price * 1.01, entry_price * 0.99

    def calculate_risk_based_position_size(
        self, symbol: str, entry_price: float, sl_price: float, nav: float
    ) -> float:
        """
        Calculate position size based on SL distance and target risk.

        Ensures consistent risk across all trades (e.g., always 2% risk).

        Returns: quote_usdt (amount to deploy)
        """
        try:
            # SL distance as percentage of entry
            sl_distance_pct = abs(entry_price - sl_price) / entry_price
            if sl_distance_pct <= 0:
                self.logger.warning(f"[TPSLEngine] {symbol} invalid SL distance")
                # Fallback: 5% of NAV
                return nav * 0.05

            # Kelly-style sizing: position_quote = nav * (target_risk / sl_distance)
            position_quote = nav * (self._target_risk_pct / 100.0 / sl_distance_pct)

            # Safety floor: never deploy less than minimum notional
            position_quote = max(position_quote, self._min_notional_safety)

            self.logger.debug(
                f"[TPSLEngine:RISK] {symbol} sl_dist={sl_distance_pct:.4f} "
                f"nav={nav:.2f} quote={position_quote:.2f}"
            )

            return float(position_quote)

        except Exception as e:
            self.logger.error(f"[TPSLEngine] risk_based_sizing failed for {symbol}: {e}")
            return nav * 0.05  # Fallback

    def _compute_atr(self, symbol: str, lookback: int = 14) -> float:
        """
        Compute ATR(lookback) from market data.

        Strategy:
          1. Try cached ATR from market_data
          2. Compute from klines if available
          3. Fallback to fixed 1% of entry price

        Returns: ATR value
        """
        try:
            # Strategy 1: Check cached ATR in market_data
            market_data = getattr(self.shared_state, "market_data", {}) or {}
            if symbol in market_data:
                sym_md = market_data[symbol]
                if isinstance(sym_md, dict):
                    cached_atr = float(sym_md.get("atr") or 0.0)
                    if cached_atr > 0:
                        self.logger.debug(
                            f"[TPSLEngine] {symbol} using cached ATR={cached_atr:.6f}"
                        )
                        return cached_atr

            # Strategy 2: Compute from klines
            klines = getattr(self.shared_state, "klines", {}) or {}
            if symbol in klines:
                candles = klines.get(symbol, {}).get("1m", [])  # Use 1m candles
                if isinstance(candles, list) and len(candles) >= lookback:
                    atr = self._compute_atr_from_candles(candles, lookback)
                    if atr > 0:
                        self.logger.debug(f"[TPSLEngine] {symbol} computed ATR={atr:.6f}")
                        return atr

            # Strategy 3: Fallback to price-based estimate
            prices = getattr(self.shared_state, "prices", {}) or {}
            if symbol in prices:
                last_price = float(prices[symbol] or 0.0)
                if last_price > 0:
                    estimated_atr = last_price * 0.015  # Estimate: 1.5% of price
                    self.logger.debug(
                        f"[TPSLEngine] {symbol} using estimated ATR={estimated_atr:.6f}"
                    )
                    return estimated_atr

            # Final fallback
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

            # Extract OHLC from candles (assume format: [time, open, high, low, close, ...])
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

                # TR = max(H-L, abs(H-PC), abs(L-PC))
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)
                prev_close = close

            if not true_ranges:
                return 0.0

            # ATR = SMA of TR
            atr = sum(true_ranges) / len(true_ranges)
            return float(atr)

        except Exception as e:
            self.logger.error(f"[TPSLEngine] _compute_atr_from_candles failed: {e}")
            return 0.0

    def _estimate_volatility_pressure(self, symbol: str) -> float:
        """
        Estimate volatility pressure on a symbol.

        Returns: 0.0 (calm) to 1.0 (volatile)

        Simple heuristic: compare recent price range to ATR
        """
        try:
            prices = getattr(self.shared_state, "prices", {}) or {}
            if symbol not in prices:
                return 0.5  # Neutral

            # For now, return neutral (0.5)
            # TODO: Compute from price history or use volatility_state
            return 0.5

        except Exception:
            return 0.5  # Default to neutral

    def update_position_tp_sl(self, symbol: str, position: dict, entry_price: float) -> None:
        """Update TP/SL on an existing position."""
        try:
            tp, sl = self.calculate_tp_sl(symbol, entry_price)
            position["tp"] = tp
            position["sl"] = sl
            self.logger.debug(f"[TPSLEngine:UPDATE] {symbol} tp={tp:.6f} sl={sl:.6f}")
        except Exception as e:
            self.logger.error(f"[TPSLEngine] update_position_tp_sl failed for {symbol}: {e}")
