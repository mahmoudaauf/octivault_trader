"""
Native L6: Adaptive Capital Allocator (Phase 8.2.8 + Feedback)

Allocates trading capital for buy decisions based on available balance,
with optional feedback from AdaptiveCapitalEngine and runtime overrides
from ObjectiveFeedbackController.

Design choices
--------------
* Adaptive allocation: uses ACE to compute dynamic risk_fraction per trade
* Feedback-aware: reads SIZE_MULTIPLIER from OFC runtime_overrides
* Market-driven: incorporates volatility, drawdown, win rate, fee efficiency
* Graceful degradation: falls back to flat % if ACE unavailable
"""

from __future__ import annotations

import logging
from typing import Any

from . import math_utils
from .capital_policy import compute_spendable_quote
from .daily_compounding import DailyCompoundingPolicy

logger = logging.getLogger(__name__)


class NativeCapitalAllocator:
    """
    Adaptive capital allocator for trading decisions.

    Integrates AdaptiveCapitalEngine (ACE) for dynamic sizing based on
    performance history, and reads SIZE_MULTIPLIER from ObjectiveFeedbackController.

    Usage::

        allocator = NativeCapitalAllocator(
            portfolio_manager=pm,
            market_data=md,
            allocation_pct=5.0,
            adaptive_engine=ace,       # NativeAdaptiveCapitalEngine
            shared_state=ss,           # NativeSharedState
        )
        quantity = await allocator.allocate_for_buy(symbol="BTCUSDT")
        # ⇒ float (quantity to buy)
    """

    def __init__(
        self,
        *,
        portfolio_manager: Any,
        market_data: Any | None = None,
        allocation_pct: float = 5.0,
        adaptive_engine: Any | None = None,
        shared_state: Any | None = None,
        exchange_client: Any | None = None,
        default_planned_quote: float = 12.0,
        quote_reserve_ratio: float = 0.10,
        quote_min_reserve_usdt: float = 0.0,
        daily_compounding_enabled: bool = True,
        daily_compounding_state_path: str | None = None,
    ) -> None:
        """
        Initialize adaptive capital allocator.

        Args:
            portfolio_manager: NativePortfolioManager with get_nav() method
            market_data: NativeMarketData with get_price() method (optional)
            allocation_pct: Base percentage of available USDT per buy signal (default 5%)
            adaptive_engine: NativeAdaptiveCapitalEngine for dynamic sizing (optional)
            shared_state: NativeSharedState for feedback loop data (optional)
            exchange_client: NativeExchangeClient for symbol filters (optional)
            default_planned_quote: Fixed USDT quote for small accounts (<$100); scales to %-based when account grows
        """
        self._pm = portfolio_manager
        self._md = market_data
        self._allocation_pct = max(0.1, min(100.0, float(allocation_pct)))
        self._ace = adaptive_engine
        self._ss = shared_state
        self._exchange_client = exchange_client
        self._default_planned_quote = max(1.0, float(default_planned_quote))
        self._quote_reserve_ratio = max(0.0, float(quote_reserve_ratio))
        self._quote_min_reserve_usdt = max(0.0, float(quote_min_reserve_usdt))
        self._daily_compounding = DailyCompoundingPolicy(
            enabled=daily_compounding_enabled,
            state_path=daily_compounding_state_path,
        )
        self._symbol_filters_cache: dict[str, dict[str, Any]] = {}

    async def allocate_for_buy(self, symbol: str) -> float:
        """
        Allocate quantity for a buy signal.

        First attempts to use AdaptiveCapitalEngine if available, then applies
        SIZE_MULTIPLIER from OFC runtime_overrides. Falls back to flat percentage
        if ACE is unavailable.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Quantity to buy (0.0 if allocation fails or insufficient capital)
        """
        if not self._pm:
            logger.debug("Portfolio manager unavailable; no allocation")
            return 0.0

        try:
            # Lazy-load symbol filters if not in cache (for symbols discovered per-cycle)
            if symbol not in self._symbol_filters_cache and self._exchange_client:
                try:
                    logger.debug("🔍 Fetching filters for new symbol: %s", symbol)
                    exchange_info = await self._exchange_client.get_exchange_info()
                    symbol_data = {s["symbol"]: s for s in exchange_info.get("symbols", [])}
                    sym_info = symbol_data.get(symbol, {})
                    filters = sym_info.get("filters", [])

                    min_notional = 10.0
                    step_size = 0.00000001
                    min_qty = 0.0

                    for f in filters:
                        # Binance renamed MIN_NOTIONAL → NOTIONAL; accept both.
                        if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                            min_notional = float(f.get("minNotional", min_notional) or min_notional)
                        elif f.get("filterType") == "LOT_SIZE":
                            step_size = float(f.get("stepSize", 0.00000001))
                            min_qty = float(f.get("minQty", 0.0) or 0.0)

                    self._symbol_filters_cache[symbol] = {
                        "min_notional": min_notional,
                        "step_size": step_size,
                        "min_qty": min_qty,
                    }
                    logger.debug(
                        "✅ %s filters cached: min_notional=%.2f step_size=%.8f min_qty=%.8f",
                        symbol, min_notional, step_size, min_qty,
                    )
                except Exception as e:
                    logger.warning("Failed to fetch filters for %s: %s (using defaults)", symbol, e)
                    self._symbol_filters_cache[symbol] = {
                        "min_notional": 10.0,
                        "step_size": 0.00000001,
                        "min_qty": 0.0,
                    }

            # Get current NAV
            live_nav = await self._pm.get_nav()
            if not live_nav or live_nav <= 0:
                logger.debug("NAV %s too low to allocate", live_nav)
                return 0.0
            _nav_prot_state = (getattr(self._ss, "nav_protection_state", {}) or {}) if self._ss else {}
            nav = self._daily_compounding.sizing_nav(
                float(live_nav),
                has_open_positions=self._has_open_positions(),
                protection_floor_usdt=float(_nav_prot_state.get("protection_floor_usdt", 0.0) or 0.0),
            )
            if nav < float(live_nav):
                logger.info(
                    "[compounding] sizing from committed NAV $%.2f (live NAV $%.2f)",
                    nav,
                    float(live_nav),
                )

            free_capital = (
                float(getattr(self._ss, "free_balance_usdt", 0.0) or 0.0) if self._ss else nav
            )
            reserved_quote = (
                float(getattr(self._ss, "reserved_quote_total", lambda _asset: 0.0)("USDT") or 0.0)
                if self._ss
                else 0.0
            )
            spendable_capital = compute_spendable_quote(
                free_capital,
                reserve_ratio=self._quote_reserve_ratio,
                min_reserve=self._quote_min_reserve_usdt,
                reserved_quote=reserved_quote,
            )
            if spendable_capital <= 0:
                logger.debug("Spendable capital %.2f too low to allocate", spendable_capital)
                return 0.0

            # Get price from market_data if available
            price = None
            if self._md and hasattr(self._md, "get_price"):
                price = self._md.get_price(symbol)

            # Price missing from market_data. In LIVE mode we must NEVER size off a fake
            # price (the old mock table is stale by 30-50% — BTC 45k vs 62k, DOGE 0.35 vs
            # 0.085 — and would badly mis-size real orders). Try a fresh REST quote; if that
            # also fails, skip the allocation. Mock prices are paper-mode only.
            if not price or price <= 0:
                # First: a real REST quote (same source as the executor's freshness guard).
                if self._exchange_client is not None and hasattr(self._exchange_client, "get_prices"):
                    try:
                        quotes = await self._exchange_client.get_prices([symbol])
                        price = float(quotes.get(symbol, 0.0) or 0.0)
                        if price > 0 and self._ss is not None and hasattr(self._ss, "update_price"):
                            self._ss.update_price(symbol, price)
                    except Exception as e:
                        logger.debug("REST price fetch failed for %s: %s", symbol, e)

                if not price or price <= 0:
                    _is_paper = (
                        getattr(self._exchange_client, "api_key", "") == "paper_key"
                        or not getattr(self._exchange_client, "api_secret", None)
                    ) if self._exchange_client is not None else True
                    if _is_paper:
                        mock_prices = {
                            "BTCUSDT": 45000.0, "ETHUSDT": 2500.0, "BNBUSDT": 600.0,
                            "SOLUSDT": 180.0, "XRPUSDT": 2.5, "ADAUSDT": 0.9,
                            "LINKUSDT": 25.0, "DOGEUSDT": 0.35, "AVAXUSDT": 80.0,
                            "PEPEUSDT": 0.000015,
                        }
                        price = mock_prices.get(symbol, 10.0)
                        logger.debug("[paper] mock price for %s: $%.8f", symbol, price)
                    else:
                        logger.warning(
                            "⛔ %s real price unavailable (market_data + REST both empty) — "
                            "skipping allocation rather than sizing off a stale mock", symbol,
                        )
                        return 0.0

            # Apply runtime_overrides from ObjectiveFeedbackController if present
            overrides = getattr(self._ss, "runtime_overrides", {}) if self._ss else {}
            size_mult = float(overrides.get("SIZE_MULTIPLIER", 1.0))
            if size_mult != 1.0:
                logger.info(
                    "📊 OFC SIZE_MULTIPLIER active: %.2f (from runtime_overrides)", size_mult
                )

            # Hybrid allocation: NAV-percentage for all accounts, floored by min_notional
            # At $58 NAV: 15% = $8.70; floor ensures we always clear Binance minimum.
            # Auto-compounds: as NAV grows, position size grows proportionally.
            if nav < 100.0:
                sym_filters = self._symbol_filters_cache.get(symbol, {})
                min_notional = float(sym_filters.get("min_notional", 10.0))
                # 15% of NAV, floored at min_notional + 20% safety buffer, capped at 20% of NAV
                nav_pct_usdt = nav * 0.15 * size_mult
                # Fear-time half-size: in fear (F&G ≤ 25) without confirmed BTC reversal,
                # halve position size. Reversal is market-wide so this gates globally.
                fear_factor = 1.0
                try:
                    fg = getattr(self._ss, "fear_greed_score", None) if self._ss else None
                    btc_rev = bool(getattr(self._ss, "btc_reversal_confirmed", False)) if self._ss else False
                    if fg is not None and float(fg) <= 25 and not btc_rev:
                        fear_factor = 0.5
                except Exception:
                    fear_factor = 1.0
                nav_pct_usdt *= fear_factor
                floor_usdt = min_notional * 1.2  # 20% above min_notional to absorb slippage
                allocation_usdt = min(max(nav_pct_usdt, floor_usdt), nav * 0.20, spendable_capital)
                alloc_reason = (
                    f"nav-pct-15% (nav=${nav:.2f}, floor=${floor_usdt:.2f}, fear×{fear_factor:.1f})"
                )
            else:
                # Larger account: switch to percentage-based allocation
                # ACE will further refine this if enabled
                if self._ace and getattr(self._ace, "enabled", False):
                    # Build ACE inputs from SharedState
                    try:
                        trade_history = self._ss.trade_history.get(symbol, []) if self._ss else []
                        positions = self._ss.get_all_positions() if self._ss else {}
                        drawdown_pct = self._compute_drawdown_pct()
                        volatility_pct = self._compute_volatility_pct(symbol)

                        decision = self._ace.evaluate(
                            symbol=symbol,
                            nav=nav,
                            free_capital=spendable_capital,
                            base_risk_fraction=self._allocation_pct / 100.0,
                            volatility_pct=volatility_pct,
                            drawdown_pct=drawdown_pct,
                            fee_bps=float(self._ss.metrics.get("avg_fee_bps", 10.0))
                            if self._ss
                            else 10.0,
                            slippage_bps=float(self._ss.metrics.get("avg_slippage_bps", 5.0))
                            if self._ss
                            else 5.0,
                            min_notional=10.0,
                            slot_utilization=len(positions) / max(1, 5),
                            throughput_per_hour=float(
                                overrides.get("TARGET_THROUGHPUT_PER_HOUR", 10.0)
                            ),
                            target_throughput_per_hour=10.0,
                            trade_history=trade_history,
                        )
                        risk_fraction = decision.risk_fraction * size_mult
                        allocation_usdt = min(nav * risk_fraction, spendable_capital)
                        alloc_reason = f"ACE-adaptive (risk={decision.risk_fraction:.3f})"

                        logger.debug(
                            "ACE allocation for %s: risk=%.3f mult=%.2f reason=%s",
                            symbol,
                            decision.risk_fraction,
                            size_mult,
                            " | ".join(decision.reasons[:2]),
                        )
                    except Exception as e:
                        logger.warning("ACE evaluation failed; falling back to flat: %s", e)
                        risk_fraction = (self._allocation_pct / 100.0) * size_mult
                        allocation_usdt = min(nav * risk_fraction, spendable_capital)
                        alloc_reason = "flat-pct (%.1f%%)" % self._allocation_pct
                else:
                    # Flat percentage allocation with OFC multiplier
                    risk_fraction = (self._allocation_pct / 100.0) * size_mult
                    allocation_usdt = min(nav * risk_fraction, spendable_capital)
                    alloc_reason = "flat-pct (%.1f%%)" % self._allocation_pct

            # ── Exchange-minimum enforcement ───────────────────────────────────
            # A sub-minimum order (allocation too small for the symbol's min_notional/
            # min_qty — common on high-priced assets like ETH/BTC, or when spendable
            # dips) just gets rejected by the exchange and wastes the attempt. If we can
            # afford the minimum, bump the allocation up to it; otherwise skip cleanly.
            _sf = self._symbol_filters_cache.get(symbol, {})
            _min_notional = float(_sf.get("min_notional", 10.0) or 10.0)
            _min_qty = float(_sf.get("min_qty", 0.0) or 0.0)
            _min_order_usdt = max(_min_notional, _min_qty * price) * 1.02  # 2% headroom for fees/rounding
            if allocation_usdt < _min_order_usdt:
                if _min_order_usdt <= spendable_capital and _min_order_usdt <= nav * 0.25:
                    logger.info(
                        "[allocator] %s allocation $%.4f < exchange min $%.2f — bumped to min",
                        symbol, allocation_usdt, _min_order_usdt,
                    )
                    allocation_usdt = _min_order_usdt
                else:
                    logger.info(
                        "[allocator] %s skip: min order $%.2f exceeds budget (spendable $%.2f) — no sub-min order",
                        symbol, _min_order_usdt, spendable_capital,
                    )
                    return 0.0

            quantity = allocation_usdt / price

            # Round down to step_size to meet Binance LOT_SIZE requirement
            if quantity > 0:
                quantity = self._round_quantity_for_exchange_sync(symbol, quantity, price)

            logger.info(
                "[allocator] %s: nav=%.2f mult=%.3f spendable=%.2f usdt=%.2f price=%.4f qty=%.8f (%s)",
                symbol, nav, size_mult, spendable_capital, allocation_usdt, price, quantity, alloc_reason,
            )

            # The executor expects a BUY decision.quantity to be a USD quote intent
            # (it converts to base qty via current price itself). Returning the base
            # `quantity` here caused a double division by price downstream, producing
            # sub-minimum dust orders (e.g. SOL: $6 alloc → $0.09 order → LOT_SIZE
            # reject). `quantity` above is computed only to validate step-size
            # feasibility; the USD allocation is the value the executor needs.
            if quantity <= 0:
                return 0.0
            return float(max(0.0, allocation_usdt))

        except Exception as e:
            logger.warning("Capital allocation failed for %s: %s", symbol, e)
            return 0.0

    async def allocate_for_sell(self, symbol: str, position_quantity: float) -> float:
        """
        Allocate quantity for a sell signal.

        Args:
            symbol: Trading pair
            position_quantity: Current position size

        Returns:
            Quantity to sell (clamped to position_quantity, default all)
        """
        # For MVP: sell all (100% of position)
        return float(max(0.0, position_quantity))

    def _compute_drawdown_pct(self) -> float:
        """Compute current drawdown percentage from peak NAV."""
        if not self._ss:
            return 0.0
        peak = self._ss.metrics.get("peak_nav", self._ss.nav_usdt)
        if not peak or peak <= 0:
            return 0.0
        return float(max(0.0, (peak - self._ss.nav_usdt) / peak * 100.0))

    def _has_open_positions(self) -> bool:
        """Return whether any non-zero position prevents a realized-NAV rollover."""
        if self._ss is None:
            return False
        positions = getattr(self._ss, "positions", {}) or {}
        for position in positions.values():
            if isinstance(position, dict):
                qty = position.get("qty", position.get("quantity", 0.0))
            else:
                qty = getattr(position, "qty", getattr(position, "quantity", 0.0))
            try:
                if abs(float(qty or 0.0)) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _compute_volatility_pct(self, symbol: str) -> float:
        """Compute volatility percentage for symbol from real ATR/candle data.

        Delegates to math_utils.compute_atr_pct — the same ATR computation
        NativeTPSLEngine uses for TP/SL sizing, moved to math_utils.py so both
        callers share one implementation. Falls back to 0.008 (0.8% —
        mid-range volatility) only when no candle/price data is available at
        all (cold start), matching this method's prior always-0.008 behavior
        for that specific case rather than silently sizing off a 0.0 input.
        """
        pct = math_utils.compute_atr_pct(self._ss, symbol) if self._ss else 0.0
        return pct if pct > 0 else 0.008

    def _round_quantity_for_exchange_sync(
        self, symbol: str, quantity: float, price: float
    ) -> float:
        """
        Round quantity DOWN using the symbol's actual step-size from cache,
        or conservative default if cache misses.
        """
        if quantity <= 0:
            return quantity

        try:
            from decimal import ROUND_DOWN, Decimal

            # Try to get real step-size from cache; fall back to conservative default
            step_size = 0.00000001
            if symbol in self._symbol_filters_cache:
                step_size = float(self._symbol_filters_cache[symbol].get("step_size", step_size))

            qty = Decimal(str(quantity))
            step = Decimal(str(step_size))
            rounded_qty = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
            result = float(rounded_qty)

            if result < quantity:
                logger.debug(
                    "Step-size adjusted %s: %.8f → %.8f (step=%.8f)",
                    symbol,
                    quantity,
                    result,
                    step_size,
                )

            return result

        except Exception as e:
            logger.warning("Step rounding failed for %s: %s", symbol, e)
            return quantity
