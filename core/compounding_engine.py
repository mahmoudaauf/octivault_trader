import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("CompoundingEngine")

# Optional deps (keep runtime lightweight and safe)
try:
    from binance.exceptions import BinanceAPIException  # type: ignore
except Exception:  # pragma: no cover
    class BinanceAPIException(Exception):  # fallback
        pass
try:
    import aiohttp  # type: ignore
    _NetError = aiohttp.ClientConnectionError
except Exception:  # pragma: no cover
    class _NetError(Exception):
        pass
try:
    from tenacity import RetryError  # type: ignore
except Exception:  # pragma: no cover
    class RetryError(Exception):
        pass

class CompoundingEngine:
    """
    Generates exposure directives (proposals) to MetaController instead of executing directly.
    
    No longer an autonomous executor. Instead:
    1. Analyzes market conditions
    2. Passes protective gates (volatility, edge, economic)
    3. Proposes exposure directives to MetaController
    4. MetaController validates signals and issues trace_id
    5. ExecutionManager executes with trace_id (rejects without)
    
    This aligns CompoundingEngine with the coordinated system design.
    """

    def __init__(
        self,
        shared_state: Any,
        exchange_client: Optional[Any],
        config: Any,
        execution_manager: Any,
        **kwargs
    ):
        if execution_manager is None:
            raise ValueError("execution_manager must not be None")

        self.shared_state = shared_state
        # Deprecated dependency: CompoundingEngine no longer talks to ExchangeClient directly.
        # Keep the attribute for wiring/back-compat only.
        self.exchange_client = exchange_client
        self.config = config
        self.execution_manager = execution_manager

        # Tunables (with safe defaults)
        # Lazy initialization via properties below
        self.base_currency = str(getattr(config, "BASE_CURRENCY", "USDT")).upper()

        self.running = True
        self.just_ran = False
        self._task: Optional[asyncio.Task] = None  # ✅ let Phase 9 manage a background task
        logger.info("✅ CompoundingEngine initialized.")

    # ---------- runnable entrypoints for Phase 9 ----------
    async def start(self) -> None:
        """Called by Phase 9 if present."""
        if self._task and not self._task.done():
            logger.info("CompoundingEngine already running.")
            return
        self.running = True
        self._task = asyncio.create_task(self.run_loop(), name="CompoundingEngine.run_loop")
        logger.info("🚀 CompoundingEngine start() launched background loop.")

    async def run_loop(self) -> None:
        """Alias that Phase 9’s scheduler looks for."""
        # Optional: wait for market data readiness if your SharedState exposes it
        ready_event = getattr(self.shared_state, "market_data_ready_event", None)
        if ready_event and hasattr(ready_event, "wait"):
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=120)
            except asyncio.TimeoutError:
                logger.warning("CompoundingEngine starting without market_data_ready_event.")
        await self.run()  # reuse your existing loop

    async def shutdown(self) -> None:
        """Graceful stop hook (if your Phase 9 calls it)."""
        self.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 CompoundingEngine shutdown complete.")

    # ---------- core ----------
    async def run(self) -> None:
        logger.info("💰 CompoundingEngine started.")
        try:
            while self.running:
                try:
                    await self._check_and_compound()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("❌ Error in compounding loop: %s", e)
                await asyncio.sleep(self.check_interval)
        finally:
            logger.info("CompoundingEngine stopped.")

    async def propose_compounding(self) -> None:
        """
        Compatibility hook used by MetaController Tier-1 reinvest path.
        """
        await self._check_and_compound()

    # ---------- helpers ----------
    async def _maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value) or asyncio.isfuture(value):
            return await value
        return value

    async def _get_free_quote(self) -> float:
        """
        Read free quote balance from SharedState only.
        CompoundingEngine must not fetch balances from ExchangeClient directly.
        """
        # Preferred: canonical shared-state free-balance getter
        try:
            if hasattr(self.shared_state, "get_free_balance"):
                bal = await self._maybe_await(self.shared_state.get_free_balance(self.base_currency))
                return float(bal or 0.0)
        except Exception:
            logger.debug("shared_state.get_free_balance failed", exc_info=True)

        # Fallback: full balance object from SharedState
        try:
            if hasattr(self.shared_state, "get_balance"):
                bal = await self._maybe_await(self.shared_state.get_balance(self.base_currency))
                if isinstance(bal, dict):
                    return float(bal.get("free", 0.0))
                return float(bal or 0.0)
        except Exception:
            logger.debug("shared_state.get_balance failed", exc_info=True)

        # Final fallback to shared_state snapshot
        try:
            bmap = getattr(self.shared_state, "wallet_balances", {}) or getattr(self.shared_state, "balances", {}) or {}
            v = bmap.get(self.base_currency) or bmap.get(self.base_currency.upper()) or 0.0
            if isinstance(v, dict):
                return float(v.get("free", 0.0))
            return float(v or 0.0)
        except Exception:
            logger.debug("shared_state balance read failed", exc_info=True)

        return 0.0

    async def _get_realized_pnl_total(self) -> float:
        """
        Resolve cumulative realized PnL with fallbacks.

        Primary source is metrics['realized_pnl'], but some runtime paths may lag
        this value briefly. In that case, use shared-state fallbacks so compounding
        gates are driven by the best available accounting signal.
        """
        try:
            metrics = getattr(self.shared_state, "metrics", {}) or {}
            primary = float(metrics.get("realized_pnl", 0.0) or 0.0)
            if abs(primary) > 1e-12:
                return primary
        except Exception:
            pass

        # Common direct attribute fallback
        try:
            attr_val = float(getattr(self.shared_state, "realized_pnl", 0.0) or 0.0)
            if abs(attr_val) > 1e-12:
                return attr_val
        except Exception:
            pass

        # History/deque fallback
        try:
            rp = getattr(self.shared_state, "_realized_pnl", None)
            if rp:
                return float(sum(float(item[1] or 0.0) for item in rp))
        except Exception:
            pass

        # MetaController KPI fallback (restart-safe view used by runtime health dashboards).
        # This helps keep compounding eligibility aligned when accounting events were
        # captured by meta KPI tracking but not yet mirrored into metrics['realized_pnl'].
        try:
            meta = getattr(self.shared_state, "meta_controller", None)
            if meta is None:
                app_ref = getattr(self.shared_state, "app", None)
                meta = getattr(app_ref, "meta_controller", None) if app_ref is not None else None
            kpi = getattr(meta, "_kpi_metrics", None) if meta is not None else None
            if isinstance(kpi, dict):
                val = float(kpi.get("total_realized_pnl", 0.0) or 0.0)
                if abs(val) > 1e-12:
                    return val
        except Exception:
            pass

        return 0.0

    async def _has_min_ohlcv_bars(self, symbol: str, timeframe: str, min_bars: int) -> bool:
        """Best-effort OHLCV depth check for a (symbol, timeframe) pair."""
        sym = str(symbol or "").upper()
        tf = str(timeframe or "").strip()
        need = int(min_bars or 0)
        if not sym or not tf or need <= 0:
            return False

        try:
            has_fn = getattr(self.shared_state, "has_ohlcv", None)
            if callable(has_fn):
                ready = has_fn(sym, tf, need)
                return bool(await self._maybe_await(ready))
        except Exception:
            logger.debug("has_ohlcv check failed for %s %s", sym, tf, exc_info=True)

        try:
            count_fn = getattr(self.shared_state, "get_ohlcv_count", None)
            if callable(count_fn):
                count = count_fn(sym, tf)
                count = await self._maybe_await(count)
                return int(count or 0) >= need
        except Exception:
            logger.debug("get_ohlcv_count check failed for %s %s", sym, tf, exc_info=True)

        try:
            rows_fn = getattr(self.shared_state, "get_market_data", None)
            if callable(rows_fn):
                rows = rows_fn(sym, tf)
                rows = await self._maybe_await(rows)
                return len(rows or []) >= need
        except Exception:
            logger.debug("get_market_data check failed for %s %s", sym, tf, exc_info=True)

        try:
            md = getattr(self.shared_state, "market_data", {}) or {}
            rows = md.get((sym, tf)) or []
            return len(rows) >= need
        except Exception:
            return False

    def _atr_readiness_requirements(self) -> Tuple[int, int, List[str]]:
        """Resolve ATR period/min bars/timeframe chain used before affordability checks."""
        period = max(2, int(self._cfg("COMPOUNDING_ATR_PERIOD", 14) or 14))
        min_bars = int(self._cfg("COMPOUNDING_ATR_MIN_BARS", period + 1) or (period + 1))
        min_bars = max(period + 1, min_bars)

        primary_tf = str(self._cfg("COMPOUNDING_ATR_TIMEFRAME_PRIMARY", "5m") or "5m").strip()
        raw_fallbacks = str(self._cfg("COMPOUNDING_ATR_TIMEFRAME_FALLBACKS", "1h,1m") or "1h,1m")
        fallback_tfs = [t.strip() for t in raw_fallbacks.split(",") if t.strip()]

        timeframes: List[str] = []
        for tf in [primary_tf, *fallback_tfs, "5m", "1h", "1m"]:
            tf_u = str(tf or "").strip()
            if tf_u and tf_u not in timeframes:
                timeframes.append(tf_u)

        return period, min_bars, timeframes

    async def _atr_data_ready_for_symbol(
        self,
        symbol: str,
        *,
        period: int,
        min_bars: int,
        timeframes: List[str],
    ) -> Tuple[bool, Optional[str], float]:
        """
        Return True when symbol has enough candles and a positive ATR on at least one timeframe.
        """
        calc_atr = getattr(self.shared_state, "calc_atr", None)
        sym = str(symbol or "").upper()

        for tf in timeframes:
            if not await self._has_min_ohlcv_bars(sym, tf, min_bars):
                continue

            # If calc_atr is unavailable, depth alone is the best readiness signal.
            if not callable(calc_atr):
                return True, tf, 0.0

            try:
                atr_val = calc_atr(sym, tf, period)
                atr_val = await self._maybe_await(atr_val)
                if float(atr_val or 0.0) > 0:
                    return True, tf, float(atr_val)
            except Exception:
                logger.debug("calc_atr readiness check failed for %s %s", sym, tf, exc_info=True)

        return False, None, 0.0

    # ========== PROTECTIVE GATES FOR FEE CHURN ELIMINATION ==========
    
    def _get_volatility_filter(self) -> float:
        """
        Gate 1: VOLATILITY FILTER
        
        Returns the minimum required 24h volatility for a symbol to be eligible
        for compounding. Prevents buying calm symbols where fees exceed edge.
        
        Fee structure costs ~0.225% per buy (Binance fee + spread + slippage).
        We require volatility > 2x this cost (0.45%) to ensure we have recovery space.
        
        Returns: Minimum required volatility (e.g., 0.0045 for 0.45%)
        """
        return float(self._cfg("COMPOUNDING_MIN_VOLATILITY", 0.0045))  # 0.45% default
    
    async def _validate_volatility_gate(self, symbol: str) -> bool:
        """
        Check if symbol has sufficient volatility for compounding entry.
        
        Args:
            symbol: Trading pair symbol (e.g., "ETHUSDT")
            
        Returns:
            True if volatility meets minimum threshold, False otherwise
        """
        min_vol = self._get_volatility_filter()
        
        # Try to get volatility from various sources
        volatility = None
        
        # Option 1: From OHLCV data in shared_state
        try:
            if hasattr(self.shared_state, "get_symbol_ohlcv"):
                ohlcv = await self._maybe_await(self.shared_state.get_symbol_ohlcv(symbol, "1h", limit=24))
                if ohlcv and len(ohlcv) >= 5:
                    closes = [float(candle[4]) for candle in ohlcv]
                    returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]
                    volatility = float(np.std(returns)) if returns else 0.0
        except Exception:
            logger.debug(f"Failed to get OHLCV volatility for {symbol}", exc_info=True)
        
        # Option 2: From market data snapshot if available
        if volatility is None:
            try:
                market_data = getattr(self.shared_state, "symbol_market_data", {}) or {}
                if symbol in market_data:
                    data = market_data[symbol]
                    volatility = float(data.get("volatility_24h", 0.0))
            except Exception:
                logger.debug(f"Failed to get market data volatility for {symbol}", exc_info=True)
        
        # Option 3: Use exchange client if available (fallback)
        if volatility is None and self.exchange_client:
            try:
                klines = await self._maybe_await(
                    self.exchange_client.get_klines(symbol, interval="1h", limit=24)
                )
                if klines and len(klines) >= 5:
                    closes = [float(k[4]) for k in klines]
                    returns = [(closes[i] / closes[i-1] - 1) for i in range(1, len(closes))]
                    volatility = float(np.std(returns)) if returns else 0.0
            except Exception:
                logger.debug(f"Failed to get exchange volatility for {symbol}", exc_info=True)
        
        # If we couldn't calculate volatility, skip this symbol (conservative approach)
        if volatility is None:
            logger.debug(f"⚠️ Could not calculate volatility for {symbol}, skipping (conservative)")
            return False
        
        # Check against threshold
        if volatility >= min_vol:
            logger.debug(f"✅ {symbol} volatility {volatility:.4%} >= {min_vol:.4%} (Gate 1: PASS)")
            return True
        else:
            logger.debug(f"❌ {symbol} volatility {volatility:.4%} < {min_vol:.4%} (Gate 1: FAIL - too calm)")
            return False
    
    async def _validate_edge_gate(
        self,
        symbol: str,
        *,
        local_high_buffer: float = 0.001,
        momentum_limit: float = 0.005,
        allow_inconclusive: bool = False,
        gate_label: str = "Gate 2",
    ) -> bool:
        """
        Gate 2: EDGE VALIDATION
        
        Prevents buying:
        - At local highs (within 0.1% of 20-candle high)
        - After momentum has already fired (within last 5 candles)
        - When price action is flat
        
        Args:
            symbol: Trading pair symbol (e.g., "ETHUSDT")
            
        Returns:
            True if entry is valid, False if entry appears poor-timing
        """
        try:
            # Get recent OHLCV data
            ohlcv = None
            if hasattr(self.shared_state, "get_symbol_ohlcv"):
                ohlcv = await self._maybe_await(
                    self.shared_state.get_symbol_ohlcv(symbol, "1h", limit=25)
                )
            
            if not ohlcv or len(ohlcv) < 5:
                # Can't validate. Strict mode stays conservative; relaxed mode can opt in.
                if allow_inconclusive:
                    logger.info(
                        "⚠️ %s edge validation inconclusive (insufficient OHLCV) but allowed by config",
                        symbol,
                    )
                    return True
                logger.debug(
                    "⚠️ Insufficient OHLCV for %s, edge validation inconclusive (%s: SKIP)",
                    symbol,
                    gate_label,
                )
                return False
            
            # Check 1: Not at local high (within 0.1% of 20-candle high)
            high_prices = [float(c[2]) for c in ohlcv[-20:]]
            current_price = float(ohlcv[-1][4])  # Close of last candle
            local_high = max(high_prices)
            distance_from_high = (local_high - current_price) / current_price
            
            if distance_from_high < float(local_high_buffer):  # Too close to local high
                logger.debug(
                    "❌ %s at local high (current=%.8f, high=%.8f, dist=%.4f%%, min_dist=%.4f%%) (%s: FAIL)",
                    symbol,
                    current_price,
                    local_high,
                    distance_from_high * 100.0,
                    float(local_high_buffer) * 100.0,
                    gate_label,
                )
                return False
            
            # Check 2: Not after recent momentum (uptrend in last 5 candles)
            recent_closes = [float(c[4]) for c in ohlcv[-6:]]
            recent_momentum = (recent_closes[-1] / recent_closes[0] - 1)
            
            if recent_momentum > float(momentum_limit):  # Too much short-term uptrend
                logger.debug(
                    "❌ %s momentum fired recently (move=%.4f%% in last 5 candles, max=%.4f%%) (%s: FAIL)",
                    symbol,
                    recent_momentum * 100.0,
                    float(momentum_limit) * 100.0,
                    gate_label,
                )
                return False

            logger.debug("✅ %s edge is valid - not at high, momentum clear (%s: PASS)", symbol, gate_label)
            return True
            
        except Exception as e:
            logger.debug("⚠️ Edge validation failed for %s: %s, being conservative", symbol, e, exc_info=True)
            return False
    
    async def _validate_economic_gate(self, amount: float, num_symbols: int) -> bool:
        """
        Gate 3: ECONOMIC THRESHOLD
        
        Prevents compounding when profit is insufficient to cover:
        - Trading fees (~0.225% per buy)
        - Safety buffer ($50 minimum remaining)
        
        This ensures compounding won't eat away ALL profit via fee churn.
        
        Args:
            amount: Total USDT available to deploy
            num_symbols: Number of symbols we'd buy
            
        Returns:
            True if profit has room for compounding, False if too thin
        """
        realized_pnl = float(await self._get_realized_pnl_total())
        
        # Estimate total fees for this compounding cycle
        # Each $10 order costs ~$0.0225 (0.225% friction)
        per_symbol = amount / num_symbols if num_symbols > 0 else amount
        fee_per_order = per_symbol * 0.00225  # 0.225% total friction
        estimated_total_fees = fee_per_order * num_symbols
        
        # Safety buffer to ensure compounding doesn't eat all profit.
        # Default reduced to 5.0 — $50 was effectively disabling compounding on small accounts.
        # Micro-account dynamic path below further adapts this based on NAV.
        safety_buffer = float(self._cfg("COMPOUNDING_ECONOMIC_BUFFER", 5.0))
        if str(self._cfg("COMPOUNDING_ECONOMIC_BUFFER_DYNAMIC", "true")).lower() == "true":
            try:
                metrics = getattr(self.shared_state, "metrics", {}) or {}
                nav_candidates: List[float] = []

                # 1) Metrics (fast path)
                for key in ("nav", "total_nav", "total_equity", "total_value"):
                    try:
                        val = float(metrics.get(key, 0.0) or 0.0)
                        if val > 0.0:
                            nav_candidates.append(val)
                    except Exception:
                        pass

                # 2) SharedState attributes (common in live runtime)
                for key in ("nav", "total_value"):
                    try:
                        val = float(getattr(self.shared_state, key, 0.0) or 0.0)
                        if val > 0.0:
                            nav_candidates.append(val)
                    except Exception:
                        pass

                # 3) SharedState async getters
                for fn_name in ("get_nav_quote", "get_nav"):
                    fn = getattr(self.shared_state, fn_name, None)
                    if callable(fn):
                        try:
                            val = float(await self._maybe_await(fn()) or 0.0)
                            if val > 0.0:
                                nav_candidates.append(val)
                        except Exception:
                            pass

                # 4) Portfolio snapshot fallback
                snap_fn = getattr(self.shared_state, "get_portfolio_snapshot", None)
                if callable(snap_fn):
                    try:
                        snap = await self._maybe_await(snap_fn())
                        if isinstance(snap, dict):
                            for key in ("nav", "total_nav", "total_equity", "total_value"):
                                try:
                                    val = float(snap.get(key, 0.0) or 0.0)
                                    if val > 0.0:
                                        nav_candidates.append(val)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # 5) Final floor for micro accounts with sparse NAV wiring
                # Use free quote / spendable amount as a conservative proxy.
                if not nav_candidates:
                    try:
                        free_quote = float(await self._get_free_quote() or 0.0)
                        if free_quote > 0.0:
                            nav_candidates.append(free_quote)
                    except Exception:
                        pass
                    if amount > 0.0:
                        nav_candidates.append(float(amount))

                nav = max(nav_candidates) if nav_candidates else 0.0
                # Use a dedicated compounding micro threshold so trading bracket
                # knobs (e.g. CAPITAL_MICRO_THRESHOLD=40) don't accidentally
                # disable compounding adaptation for small accounts.
                micro_threshold = float(self._cfg("COMPOUNDING_MICRO_NAV_THRESHOLD", 1000.0))
                if micro_threshold <= 0.0:
                    micro_threshold = 1000.0
                # Treat nav==0 (cold-start / NAV not yet computed) as micro-account
                # so the $5 default buffer is not blocked by a missing NAV read.
                if nav <= micro_threshold:
                    micro_floor = float(self._cfg("COMPOUNDING_ECONOMIC_BUFFER_MICRO_MIN", 0.05))
                    micro_nav_pct = float(self._cfg("COMPOUNDING_ECONOMIC_BUFFER_MICRO_NAV_PCT", 0.001))
                    legacy_fee_mult = self._cfg("COMPOUNDING_ECONOMIC_BUFFER_MICRO_FEE_MULT", None)
                    if legacy_fee_mult is None:
                        micro_fee_mult = float(
                            self._cfg("COMPOUNDING_ECONOMIC_BUFFER_MICRO_EXTRA_FEE_MULT", 0.10)
                        )
                    else:
                        micro_fee_mult = float(legacy_fee_mult)
                    micro_fee_mult = max(0.0, micro_fee_mult)
                    adaptive_buffer = max(
                        micro_floor,
                        nav * micro_nav_pct,
                        estimated_total_fees * micro_fee_mult,
                    )
                    safety_buffer = min(safety_buffer, adaptive_buffer)
                    logger.info(
                        "Compounding economic buffer adapted for micro account: "
                        "nav=%.2f threshold=%.2f fixed=%.2f adaptive=%.4f selected=%.4f fee_mult=%.3f",
                        nav,
                        micro_threshold,
                        float(self._cfg("COMPOUNDING_ECONOMIC_BUFFER", 50.0)),
                        adaptive_buffer,
                        safety_buffer,
                        micro_fee_mult,
                    )
            except Exception:
                logger.debug("Compounding adaptive economic buffer resolution failed", exc_info=True)
        
        # Calculate remaining profit after fees and buffer
        available_for_compounding = realized_pnl - estimated_total_fees - safety_buffer
        
        if available_for_compounding > 0:
            logger.info(
                f"✅ Economic gate PASS: realized_pnl=${realized_pnl:.2f} - "
                f"fees=${estimated_total_fees:.2f} - buffer=${safety_buffer:.2f} = "
                f"available=${available_for_compounding:.2f} (Gate 3: PASS)"
            )
            return True
        else:
            logger.info(
                f"❌ Economic gate FAIL: profit too thin (realized_pnl=${realized_pnl:.2f} < "
                f"fees+buffer=${estimated_total_fees + safety_buffer:.2f}) (Gate 3: FAIL - would churn)"
            )
            return False

    async def _estimate_available_position_capacity(self) -> Tuple[int, int, int]:
        """
        Estimate allocation-layer capacity for NEW positions.
        Returns: (available_slots, max_positions_total, open_positions_count)
        """
        max_positions = int(
            self._cfg(
                "MAX_POSITIONS_TOTAL",
                self._cfg("MAX_OPEN_POSITIONS_TOTAL", 2),
            )
            or 2
        )
        max_positions = max(1, max_positions)

        open_positions = 0
        try:
            count_fn = getattr(self.shared_state, "open_positions_count", None)
            if callable(count_fn):
                open_positions = int(await self._maybe_await(count_fn()) or 0)
        except Exception:
            open_positions = 0

        if open_positions <= 0:
            try:
                get_open = getattr(self.shared_state, "get_open_positions", None)
                if callable(get_open):
                    snap = await self._maybe_await(get_open())
                    if isinstance(snap, dict):
                        open_positions = len(snap)
            except Exception:
                open_positions = open_positions or 0

        open_positions = max(0, int(open_positions))
        available_slots = max(0, int(max_positions - open_positions))
        return available_slots, max_positions, open_positions

    async def _pick_symbols(self, limit: Optional[int] = None) -> List[str]:
        """
        Choose up to max_symbols from active/accepted symbols, preferring highest score if available.
        Ensures USDT quote and filters out obvious non-tradables.
        
        🚨 INTEGRATED PROTECTIVE GATES:
        - Gate 1: Volatility Filter (skip calm symbols where fees > edge)
        - Gate 2: Edge Validation (skip poor entries at local highs or after momentum)
        """
        # Preferred: snapshot of accepted symbols (dict keys)
        syms = []
        try:
            if hasattr(self.shared_state, "get_accepted_symbols_snapshot"):
                snap = self.shared_state.get_accepted_symbols_snapshot()
                if isinstance(snap, dict):
                    syms = list(snap.keys())
        except Exception:
            logger.debug("get_accepted_symbols_snapshot failed", exc_info=True)

        # Fallback to active symbols (list)
        if not syms:
            try:
                if hasattr(self.shared_state, "get_active_symbols"):
                    syms = list(self.shared_state.get_active_symbols() or [])
            except Exception:
                logger.debug("get_active_symbols failed", exc_info=True)

        if not syms:
            return []

        # USDT quote only, and avoid doubled quote (e.g., USDTUSDT)
        syms = [s for s in syms if isinstance(s, str) and s.endswith(self.base_currency) and not s.endswith(self.base_currency * 2)]

        # 1. Use unified scoring from SharedState
        try:
            scores = self.shared_state.get_symbol_scores()
            if scores:
                # ONLY compound into symbols with POSITIVE scores
                # This prevents fighting MetaController's exit decisions.
                syms = [s for s in syms if float(scores.get(s, 0.0)) > 0]
                syms.sort(key=lambda x: float(scores.get(x, 0.0)), reverse=True)
        except Exception:
            logger.debug("Unified scoring failed in CompoundingEngine", exc_info=True)

        # 2. Coordinate with PortfolioBalancer: only buy symbols that the balancer wants to KEEP
        # This prevents the "Buy -> Balancer Sells" loop.
        targets = getattr(self.shared_state, "rebalance_targets", set())
        if targets:
            filtered = [s for s in syms if s in targets]
            if filtered:
                syms = filtered
            else:
                logger.debug("No intersection between Compounding candidates and Balancer targets.")

        # ========== APPLY PROTECTIVE GATES ==========
        # Gate 1: Volatility Filter
        filtered_syms = []
        for symbol in syms:
            if await self._validate_volatility_gate(symbol):
                filtered_syms.append(symbol)
        syms = filtered_syms
        
        if not syms:
            logger.warning("⚠️ All symbols filtered by volatility gate (none volatile enough for compounding)")
            return []
        
        # Gate 2: Edge Validation
        strict_local_high_buffer = float(self._cfg("COMPOUNDING_EDGE_LOCAL_HIGH_BUFFER", 0.001))
        strict_momentum_limit = float(self._cfg("COMPOUNDING_EDGE_MOMENTUM_LIMIT", 0.005))
        strict_allow_inconclusive = str(
            self._cfg("COMPOUNDING_EDGE_ALLOW_INCONCLUSIVE", "false")
        ).lower() == "true"
        pre_edge_syms = list(syms)
        filtered_syms = []
        for symbol in syms:
            if await self._validate_edge_gate(
                symbol,
                local_high_buffer=strict_local_high_buffer,
                momentum_limit=strict_momentum_limit,
                allow_inconclusive=strict_allow_inconclusive,
                gate_label="Gate 2 strict",
            ):
                filtered_syms.append(symbol)
        syms = filtered_syms

        if not syms:
            relax_enabled = str(self._cfg("COMPOUNDING_EDGE_RELAX_WHEN_EMPTY", "true")).lower() == "true"
            if relax_enabled and pre_edge_syms:
                relaxed_local_high_buffer = float(
                    self._cfg("COMPOUNDING_EDGE_RELAXED_LOCAL_HIGH_BUFFER", 0.0002)
                )
                relaxed_momentum_limit = float(
                    self._cfg("COMPOUNDING_EDGE_RELAXED_MOMENTUM_LIMIT", 0.015)
                )
                relaxed_allow_inconclusive = str(
                    self._cfg("COMPOUNDING_EDGE_RELAXED_ALLOW_INCONCLUSIVE", "false")
                ).lower() == "true"

                relaxed_syms = []
                for symbol in pre_edge_syms:
                    if await self._validate_edge_gate(
                        symbol,
                        local_high_buffer=relaxed_local_high_buffer,
                        momentum_limit=relaxed_momentum_limit,
                        allow_inconclusive=relaxed_allow_inconclusive,
                        gate_label="Gate 2 relaxed",
                    ):
                        relaxed_syms.append(symbol)

                if relaxed_syms:
                    logger.warning(
                        "⚠️ Strict edge gate filtered all symbols; relaxed gate recovered %d candidate(s) "
                        "(high_buffer=%.4f%% momentum_limit=%.4f%%).",
                        len(relaxed_syms),
                        relaxed_local_high_buffer * 100.0,
                        relaxed_momentum_limit * 100.0,
                    )
                    syms = relaxed_syms
                else:
                    score_fallback_enabled = str(
                        self._cfg("COMPOUNDING_EDGE_SCORE_FALLBACK_WHEN_EMPTY", "true")
                    ).lower() == "true"
                    if score_fallback_enabled:
                        fallback_cap = int(
                            self._cfg("COMPOUNDING_EDGE_SCORE_FALLBACK_MAX_SYMBOLS", 1) or 1
                        )
                        fallback_cap = max(1, fallback_cap)
                        min_score = float(
                            self._cfg("COMPOUNDING_EDGE_SCORE_FALLBACK_MIN_SCORE", 0.0) or 0.0
                        )

                        scores: Dict[str, float] = {}
                        try:
                            raw_scores = self.shared_state.get_symbol_scores()
                            if isinstance(raw_scores, dict):
                                scores = {
                                    str(k): float(v or 0.0)
                                    for k, v in raw_scores.items()
                                }
                        except Exception:
                            logger.debug("Score fallback: get_symbol_scores failed", exc_info=True)

                        fallback_syms: List[str] = []
                        for symbol in pre_edge_syms:
                            score_val = float(scores.get(symbol, 0.0) or 0.0) if scores else 0.0
                            if score_val >= min_score:
                                fallback_syms.append(symbol)

                        # If scores are unavailable and floor is non-positive, preserve ranking order.
                        if not fallback_syms and min_score <= 0.0:
                            fallback_syms = list(pre_edge_syms)

                        fallback_syms = fallback_syms[:fallback_cap]
                        if fallback_syms:
                            logger.warning(
                                "⚠️ All symbols failed strict+relaxed edge gates; score fallback activated "
                                "for %d symbol(s) (min_score=%.4f).",
                                len(fallback_syms),
                                min_score,
                            )
                            syms = fallback_syms
                        else:
                            logger.warning(
                                "⚠️ All symbols filtered by edge validation gate (strict + relaxed + score fallback). "
                                "No compounding candidates this cycle."
                            )
                            return []
                    else:
                        logger.warning(
                            "⚠️ All symbols filtered by edge validation gate (strict + relaxed). "
                            "No compounding candidates this cycle."
                        )
                        return []
            else:
                logger.warning("⚠️ All symbols filtered by edge validation gate (poor entry timing for all)")
                return []

        cap = int(self.max_symbols)
        if limit is not None and int(limit) > 0:
            cap = min(cap, int(limit))
        if cap <= 0:
            return []
        return syms[:cap]

    async def _check_and_compound(self) -> None:
        # --- Circuit Breaker Invariant ---
        if hasattr(self.shared_state, "is_circuit_breaker_open") and await self.shared_state.is_circuit_breaker_open():
            logger.warning("🛑 Compounding frozen: Circuit Breaker is OPEN.")
            return

        # --- Profit Lock Invariant: Only compound if we have realized profit ---
        realized_pnl = float(await self._get_realized_pnl_total())
        if realized_pnl <= 0:
            logger.info(
                "⏸ Compounding skipped: realized_pnl=%.4f — "
                "check that execution_manager writes metrics['realized_pnl'] on each trade close.",
                realized_pnl,
            )
            return

        free_balance = await self._get_free_quote()
        logger.debug("🔎 Available %s balance: %.2f", self.base_currency, free_balance)

        # Reserve model (capital-quality first):
        # - Keep a strategic quote ratio (default 20%) so the engine always has optionality.
        # - Preserve legacy absolute reserve as a bounded floor (capped by reserve ratio).
        shared_cfg = getattr(self.shared_state, "config", None)
        reserve_ratio = float(
            self._cfg(
                "COMPOUNDING_RESERVE_RATIO",
                getattr(shared_cfg, "quote_reserve_ratio", 0.20),
            )
            or 0.20
        )
        reserve_ratio = max(0.0, min(0.95, reserve_ratio))

        reserve_from_ratio = free_balance * reserve_ratio
        legacy_abs_reserve = float(self._cfg("COMPOUNDING_RESERVE_USDT", 25.0) or 0.0)
        legacy_cap_ratio = float(self._cfg("COMPOUNDING_RESERVE_USDT_CAP_RATIO", reserve_ratio) or reserve_ratio)
        legacy_cap_ratio = max(0.0, min(1.0, legacy_cap_ratio))
        bounded_legacy_reserve = min(max(0.0, legacy_abs_reserve), free_balance * legacy_cap_ratio)
        reserve = max(reserve_from_ratio, bounded_legacy_reserve)
        spendable = max(0.0, free_balance - reserve)

        if spendable <= self.min_compound_threshold:
            logger.debug(
                "No compounding: spendable %.2f (reserve=%.2f ratio=%.2f%%) below threshold %.2f.",
                spendable,
                reserve,
                reserve_ratio * 100.0,
                self.min_compound_threshold,
            )
            return

        logger.info(
            "📈 Compounding opportunity: %.2f %s available (reserve %.2f / %.1f%%, spendable %.2f)",
            free_balance,
            self.base_currency,
            reserve,
            reserve_ratio * 100.0,
            spendable,
        )
        
        # ========== Allocation Capacity (separate from universe breadth) ==========
        available_slots, max_positions, open_positions = await self._estimate_available_position_capacity()
        if available_slots <= 0:
            logger.info(
                "Compounding skipped: allocation full (open_positions=%d max_positions=%d).",
                open_positions,
                max_positions,
            )
            return

        # ========== GATE 3: ECONOMIC THRESHOLD ==========
        # Fee estimate should reflect likely executable orders, not universe breadth.
        estimated_symbols = max(1, min(int(self.max_symbols), int(available_slots)))
        
        # Apply economic gate BEFORE executing strategy
        if not await self._validate_economic_gate(spendable, estimated_symbols):
            logger.info("⚠️ Compounding blocked by economic gate (profit insufficient to cover fee churn)")
            return
        
        await self._execute_compounding_strategy(spendable, max_new_positions=available_slots)

    async def _execute_compounding_strategy(self, amount: float, max_new_positions: Optional[int] = None) -> None:
        """
        Generate exposure directives (proposals) instead of executing directly.
        
        PHASE 2 CHANGE: No longer executes trades autonomously.
        Instead proposes directives to MetaController for validation.
        
        🚨 NOTE: Symbol selection includes protective gates (volatility + edge validation)
                 Economic threshold is checked in _check_and_compound before calling this.
        """
        symbols = await self._pick_symbols(limit=max_new_positions)
        if not symbols:
            logger.warning("No eligible symbols available for compounding.")
            return

        # Per-symbol allocation with cap
        per = min(amount / len(symbols), self.max_allocation_per_symbol)
        if per < self.min_compound_threshold:
            logger.info(
                "⚠️ Skipping compounding: per-symbol allocation %.2f below threshold %.2f.",
                per, self.min_compound_threshold
            )
            return

        logger.info("📊 Selected symbols for directives: %s", symbols)
        logger.info("💸 Target directive amount per symbol: %.2f %s", per, self.base_currency)

        directives_generated = 0
        atr_period, atr_min_bars, atr_timeframes = self._atr_readiness_requirements()

        for symbol in symbols:
            # Refresh free balance each iteration to avoid overspend
            free_quote = await self._get_free_quote()
            remaining = max(0.0, amount - (directives_generated * per))
            if remaining < self.min_compound_threshold or free_quote < self.min_compound_threshold:
                logger.info("Stopping directive generation: remaining=%.2f, free=%.2f", remaining, free_quote)
                break

            planned = min(per, remaining, free_quote)

            try:
                # ATR readiness guard: do not ask EM affordability gates before symbol ATR is computable.
                atr_ready, atr_tf, atr_val = await self._atr_data_ready_for_symbol(
                    symbol,
                    period=atr_period,
                    min_bars=atr_min_bars,
                    timeframes=atr_timeframes,
                )
                if not atr_ready:
                    logger.info(
                        "⏭️ Skipping directive for %s: ATR not ready (need >=%d bars for ATR-%d on %s).",
                        symbol,
                        atr_min_bars,
                        atr_period,
                        ",".join(atr_timeframes),
                    )
                    continue

                atr_pct_hint = 0.0
                try:
                    px = float((getattr(self.shared_state, "latest_prices", {}) or {}).get(symbol, 0.0) or 0.0)
                    if px > 0 and atr_val > 0:
                        atr_pct_hint = float(atr_val) / px
                except Exception:
                    atr_pct_hint = 0.0

                afford_policy_ctx: Dict[str, Any] = {
                    "source": "CompoundingEngine",
                    "atr_timeframe": str(atr_tf or ""),
                    # Compounding pre-check should validate sizing/liquidity only.
                    # Tradeability/policy gates are evaluated later by MetaController.
                    "affordability_probe": True,
                    "probe_source": "compounding_directive",
                }
                if atr_pct_hint > 0:
                    # Feed deterministic expected-move context so affordability gates
                    # do not regress to ATR=0 if another path misses candle state.
                    afford_policy_ctx["tradeability_expected_move_pct"] = float(atr_pct_hint)

                # Quick affordability + minNotional pre-check via EM
                ok, gap, why = await self.execution_manager.can_afford_market_buy(
                    symbol,
                    planned,
                    policy_context=afford_policy_ctx,
                )
                if not ok:
                    logger.info("⏭️ Skipping directive for %s: cannot afford planned %.2f (%s, gap=%.2f).", symbol, planned, why, gap)
                    continue

                # PHASE 2: Generate directive instead of executing
                directive = self._generate_directive(
                    symbol=symbol,
                    amount=planned,
                    reason="compounding",
                    atr_timeframe=atr_tf,
                    atr_value=atr_val,
                    atr_pct=atr_pct_hint,
                )
                
                # Propose to MetaController
                await self._propose_exposure_directive(directive)
                directives_generated += 1
                logger.info(
                    "📋 Generated exposure directive for %s: %.2f %s (atr_tf=%s atr=%.8f atr_pct=%.4f%%)",
                    symbol,
                    planned,
                    self.base_currency,
                    atr_tf or "unknown",
                    float(atr_val or 0.0),
                    float(atr_pct_hint) * 100.0,
                )

            except BinanceAPIException as api_error:
                logger.error("Binance API error during directive generation for %s: %s", symbol, api_error)
            except _NetError as net_error:
                logger.error("Network error during directive generation for %s: %s", symbol, net_error)
            except RetryError as retry_error:
                logger.error("RetryError generating directive for %s: %s", symbol, retry_error)
            except Exception as unknown_error:
                logger.exception("Unexpected error generating directive for %s: %s", symbol, unknown_error)

        if directives_generated > 0:
            self.just_ran = True

        logger.info("🧮 Directive generation cycle finished. Generated %d directives.", directives_generated)

    # ---------- PHASE 2: Exposure Directive Generation ----------
    def _generate_directive(
        self,
        symbol: str,
        amount: float,
        reason: str = "compounding",
        atr_timeframe: Optional[str] = None,
        atr_value: float = 0.0,
        atr_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate exposure directive (proposal) without executing.
        
        Returns:
            {
                "source": "CompoundingEngine",
                "symbol": symbol,
                "action": "buy",
                "amount": amount,
                "reason": reason,
                "timestamp": float,
                "gates_status": {
                    "volatility": bool,
                    "edge": bool,
                    "economic": bool,
                }
            }
        """
        return {
            "source": "CompoundingEngine",
            "symbol": symbol,
            "action": "buy",
            "amount": float(amount),
            "reason": reason,
            "timestamp": time.time(),
            "atr_timeframe": str(atr_timeframe or ""),
            "atr_value": float(atr_value or 0.0),
            "atr_pct": float(atr_pct or 0.0),
            "gates_status": {
                "volatility": True,  # Already passed gates before here
                "edge": True,        # Symbol selection enforces these
                "economic": True,    # Economic gate checked in _check_and_compound
            }
        }

    async def _propose_exposure_directive(
        self,
        directive: Dict[str, Any]
    ) -> None:
        """
        Send exposure directive to MetaController for decision.
        
        MetaController will:
        1. Validate directive against all signal sources
        2. Generate trace_id if approved
        3. Issue execute_trade call with trace_id
        """
        try:
            meta_controller = getattr(self, "meta_controller", None)
            if meta_controller is None:
                meta_controller = getattr(self.shared_state, "meta_controller", None)
            if meta_controller is None:
                app_ref = getattr(self.shared_state, "app", None)
                meta_controller = getattr(app_ref, "meta_controller", None) if app_ref is not None else None
            if meta_controller is None and hasattr(self.shared_state, "get"):
                with_context_get = getattr(self.shared_state, "get")
                if callable(with_context_get):
                    meta_controller = with_context_get("meta_controller")

            if not meta_controller:
                logger.warning(
                    "MetaController not available for directive %s, directive cached locally",
                    directive.get("symbol")
                )
                return
            
            # Proposal to MetaController
            await meta_controller.propose_exposure_directive(directive)
            logger.info(
                "✅ Proposed exposure directive: %s buy %.2f %s",
                directive["symbol"],
                directive["amount"],
                self.base_currency
            )
        except Exception as e:
            logger.error(
                "Failed to propose exposure directive for %s: %s",
                directive.get("symbol"),
                e
            )

    # ---------- dynamic properties ----------
    def _cfg(self, key: str, default: Any = None) -> Any:
        # 1. Check SharedState for live/dynamic overrides
        if hasattr(self.shared_state, "dynamic_config"):
            val = self.shared_state.dynamic_config.get(key)
            if val is not None:
                return val

        # 2. Fallback to static config
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @property
    def check_interval(self) -> int:
        return int(self._cfg("COMPOUNDING_INTERVAL", 60))

    @property
    def min_compound_threshold(self) -> float:
        return float(self._cfg("COMPOUNDING_THRESHOLD", 10.0))

    @property
    def max_allocation_per_symbol(self) -> float:
        return float(self._cfg("MAX_ALLOCATION_PER_SYMBOL", 100.0))

    @property
    def max_symbols(self) -> int:
        return int(self._cfg("MAX_COMPOUND_SYMBOLS", 5))

    # ---------- control ----------
    def stop(self) -> None:
        self.running = False
        logger.info("🛑 CompoundingEngine stop requested.")
