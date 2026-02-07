# core/risk_manager.py
import logging
import contextlib
import inspect
import math
import time
from typing import Dict, Any, Optional, Tuple, Union
import asyncio as _asyncio

from core.component_status_logger import ComponentStatusLogger


class RiskManager:
    """
    - Snapshots config → fewer getattr() on hot paths
    - Flexible validate_order(): works with both dict order AND EM's kwargs form
      returns (ok, reason, adj_qty, adj_quote)
    - Stricter safety gates: halt on daily loss, restrict buys on high exposure,
      per-trade min/max quote enforcement, sanity caps for SELL size
    - Leaner logging & health pings
    """

    def __init__(
        self,
        shared_state,
        config,
        execution_manager: Optional[object] = None,
        exchange_client: Optional[object] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        P9 Canon: RiskManager بوابة استشارية—لا تُنفّذ أوامر. يمكنها الاستعلام
        من ExecutionManager لكن لا تعتمد عليه أثناء التهيئة لتفادي دورات الاعتماد.
        """
        self.shared_state = shared_state
        self.config = config
        self.execution_manager = execution_manager
        self.logger = logger or logging.getLogger("RiskManager")
        # استخدم ExchangeClient من EM إن لم يُمرَّر صراحةً
        self.exchange_client = exchange_client or (
            getattr(execution_manager, "exchange_client", None) if execution_manager else None
        )

        # --- config snapshot (hot path friendly) ---
        # Lazy initialization via properties below
        self.base_currency = str(getattr(config, "BASE_CURRENCY", "USDT")).upper()

        # --- state ---
        self._initialized = False
        self._health_task: Optional[_asyncio.Task] = None
        self._running: bool = False
        self._p6_task: Optional[_asyncio.Task] = None
        self.metrics: Dict[str, Any] = {}
        self._last_utc_day_key: Optional[int] = None
        
        # Kill-Switch: Global freeze flag (survives daily reset, requires manual unfreeze)
        self._global_freeze: bool = False
        self._global_freeze_reason: Optional[str] = None

        # time helpers
        self._mono = time.monotonic
        self._epoch = time.time

        # expose to shared_state for convenience
        with contextlib.suppress(Exception):
            setattr(self.shared_state, "risk_manager", self)

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

    def _exit_fee_bps(self) -> float:
        return float(self._cfg("EXIT_FEE_BPS", self._cfg("CR_FEE_BPS", 10.0)) or 0.0)

    def _exit_slippage_bps(self) -> float:
        return float(self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)) or 0.0)

    async def _get_exit_floor_info(self, symbol: str, price: Optional[float] = None) -> Dict[str, float]:
        if hasattr(self.shared_state, "compute_symbol_exit_floor"):
            return await self.shared_state.compute_symbol_exit_floor(
                symbol,
                price=price,
                fee_bps=self._exit_fee_bps(),
                slippage_bps=self._exit_slippage_bps(),
            )
        return {"min_exit_quote": 0.0}

    @property
    def risk_eval_sec(self) -> float:
        return float(self._cfg("RISK_EVALUATION_INTERVAL", 5.0))

    @property
    def shutdown_timeout(self) -> float:
        return float(self._cfg("SHUTDOWN_TIMEOUT_SECONDS", 10.0))

    @property
    def max_drawdown_pct(self) -> float:
        return self._normalize_pct(self._cfg("MAX_DRAWDOWN_PCT", 0.20))

    @property
    def max_daily_loss_pct(self) -> float:
        return self._normalize_pct(self._cfg("MAX_DAILY_LOSS_PCT", 0.10))

    @property
    def max_pos_exposure_pct(self) -> float:
        return self._normalize_pct(self._cfg("MAX_POSITION_EXPOSURE_PCT", 0.20))

    @property
    def max_total_exposure_pct(self) -> float:
        return self._normalize_pct(self._cfg("MAX_TOTAL_EXPOSURE_PCT", 0.60))

    @property
    def min_trade_quote(self) -> float:
        return float(self._cfg("MIN_TRADE_QUOTE", 5.0))

    @property
    def max_trade_quote(self) -> float:
        return float(self._cfg("MAX_TRADE_QUOTE", 50.0))

    @property
    def tier_b_max_quote(self) -> float:
        """Micro-sizing cap for Tier B trades (throughput/probing)."""
        return float(self._cfg("TIER_B_MAX_QUOTE", 3.0))

    # ---------- small utils ----------

    @staticmethod
    def _normalize_pct(v: float) -> float:
        try:
            v = float(v)
            return v / 100.0 if v > 1.0 else max(0.0, v)
        except Exception:
            return 0.0

    async def pre_check(
        self,
        symbol: str,
        side: str,
        planned_quote: Optional[float] = None,
        tier: Optional[str] = None,
        is_liquidation: bool = False,
    ) -> Tuple[bool, str]:
        """
        Tier-aware permission check with FIX #4: liquidation bypass support.
        
        RiskManager answers: "Is this intent allowed to exist at all?"
        
        Does NOT:
        - Calculate order size
        - Check minNotional
        - Touch prices
        
        Only returns:
        - ✅ (True, "ok") - allowed
        - ❌ (False, reason) - denied with reason
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            planned_quote: Intended quote amount (for BUY)
            tier: "A" (normal) or "B" (micro)
            is_liquidation: If True, bypass min_trade_quote for dust recovery
        
        Returns:
            (ok: bool, reason: str)
        """
        await self._ensure_initialized()
        
        s = (side or "").upper()
        if s not in ("BUY", "SELL"):
            return False, "invalid_side"
        
        # Global halts
        if self.metrics.get("daily_trading_halt"):
            # G027: DailyTradingHalt gate - ELEVATED to INFO
            self.logger.info(f"[EXEC_BLOCK] gate=DAILY_TRADING_HALT reason=SYSTEM_HALTED component=RiskManager action=DENY_EXECUTION symbol={symbol} side={side}")
            return False, "daily_halt"
        
        if s == "BUY" and self.is_buy_freeze_active():
            return False, "buy_freeze_active"
        
        # Tier-aware quote caps (BUY only)
        if s == "BUY" and planned_quote is not None:
            q = float(planned_quote)

            # Exit-feasibility floor (symbol-aware)
            try:
                if hasattr(self.shared_state, "compute_min_entry_quote"):
                    base_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", 0.0)) or 0.0)
                    min_entry = await self.shared_state.compute_min_entry_quote(
                        symbol,
                        default_quote=base_quote,
                        fee_bps=self._exit_fee_bps(),
                        slippage_bps=self._exit_slippage_bps(),
                    )
                    if q < float(min_entry or 0.0):
                        return False, "below_exit_floor"
            except Exception:
                pass
            
            # Tier B: Micro-sizing cap
            if tier == "B":
                if q > self.tier_b_max_quote:
                    return False, f"tier_b_exceeds_micro_cap_{self.tier_b_max_quote}"

            # Enforce minimum trade quote for all tiers (unless liquidation)
            if q < self.min_trade_quote:
                if not is_liquidation:
                    self.logger.debug(f"[Risk:pre_check] {symbol} {side} ${q:.2f} below min ${self.min_trade_quote:.2f}")
                    return False, "below_min_trade_quote"
                self.logger.info(f"[Risk:pre_check] {symbol} {side} ${q:.2f} LIQUIDATION_BYPASS min_trade_quote")

            if self.max_trade_quote > 0 and q > self.max_trade_quote:
                return False, "exceeds_max_trade_quote"
        
        # FIX #4: Liquidation SELL bypasses min_trade_quote entirely
        if s == "SELL" and is_liquidation and planned_quote is not None:
            q = float(planned_quote)
            if q < self.min_trade_quote:
                self.logger.info(f"[Risk:pre_check] {symbol} SELL ${q:.2f} LIQUIDATION_BYPASS (normal min=${self.min_trade_quote:.2f})")

        if s == "SELL" and not is_liquidation and planned_quote is not None:
            try:
                exit_info = await self._get_exit_floor_info(symbol)
                min_exit = float(exit_info.get("min_exit_quote", 0.0) or 0.0)
                if min_exit > 0 and float(planned_quote) < min_exit:
                    return False, "sell_below_exit_floor"
            except Exception:
                pass
        
        return True, "ok"

    @staticmethod
    def _utc_day_key(ts: Optional[float] = None) -> int:
        return int((ts if ts is not None else time.time()) // 86400)

    @staticmethod
    def _free_amount(entry: Any) -> float:
        if entry is None:
            return 0.0
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, str):
            with contextlib.suppress(Exception):
                return float(entry)
            return 0.0
        if isinstance(entry, dict):
            for k in ("free", "available", "balance", "freeBalance"):
                if k in entry:
                    with contextlib.suppress(Exception):
                        return float(entry[k])
        return 0.0

    @classmethod
    def _balances_to_map(cls, balances_raw: Any) -> Dict[str, float]:
        if isinstance(balances_raw, dict):
            return {str(a).upper(): cls._free_amount(v) for a, v in balances_raw.items()}
        if isinstance(balances_raw, list):
            out: Dict[str, float] = {}
            for item in balances_raw:
                if not isinstance(item, dict):
                    continue
                asset = (item.get("asset") or item.get("currency") or item.get("coin") or item.get("symbol") or "")
                asset = str(asset).upper()
                if asset:
                    out[asset] = cls._free_amount(item)
            return out
        return {}

    # ---------- init / metrics ----------

    async def _get_portfolio_value(self) -> float:
        # Try method then attribute; support sync or async; fallback to metrics API if present
        val_fn = getattr(self.shared_state, "total_value", None)
        if callable(val_fn):
            try:
                res = val_fn()
                return await res if inspect.isawaitable(res) else float(res)
            except Exception:
                pass
        with contextlib.suppress(Exception):
            return float(getattr(self.shared_state, "portfolio_value", 0.0))
        pm_fn = getattr(self.shared_state, "get_portfolio_metrics", None)
        if callable(pm_fn):
            with contextlib.suppress(Exception):
                pm = pm_fn()
                pm = await pm if inspect.isawaitable(pm) else pm
                return float((pm or {}).get("total_value", 0.0))
        return 0.0

    async def _ensure_initialized(self):
        if self._initialized:
            return
        initial_value = await self._get_portfolio_value()
        self.metrics = {
            "start_of_day_value": initial_value,
            "current_portfolio_value": initial_value,
            "peak_portfolio_value": initial_value,
            "daily_loss": 0.0,
            "overall_drawdown": 0.0,
            "trades_count_daily": 0,
            "wins_count_daily": 0,
            "losses_count_daily": 0,
            "last_reset_time": self._epoch(),
            "daily_trading_halt": False,
            "trading_restricted": False,
        }
        self._last_utc_day_key = self._utc_day_key()
        self._initialized = True
        self.logger.info("✅ RiskManager initialized.")
        with contextlib.suppress(Exception):
            ComponentStatusLogger.log_status("RiskManager", "Initialized", "Baseline metrics ready")

    async def initialize(self):
        """Public initializer for phased startup."""
        await self._ensure_initialized()
        await self._safe_health("RiskManager", "Initialized", "Baseline metrics ready")

    async def _reset_daily_metrics_if_needed(self):
        cur_key = self._utc_day_key()
        if self._last_utc_day_key != cur_key:
            self.logger.info("New UTC day → resetting daily risk metrics")
            self.metrics.update({
                "start_of_day_value": self.metrics.get("current_portfolio_value", 0.0),
                "daily_loss": 0.0,
                "trades_count_daily": 0,
                "wins_count_daily": 0,
                "losses_count_daily": 0,
                "last_reset_time": self._epoch(),
                "daily_trading_halt": False,
                "trading_restricted": False,
            })
            self._last_utc_day_key = cur_key
            await self._safe_health("RiskManager", "Operational", "Daily metrics reset")

    async def _update_metrics(self):
        total = await self._get_portfolio_value()
        self.metrics["current_portfolio_value"] = total

        peak = self.metrics.get("peak_portfolio_value") or 0.0
        if total > peak:
            peak = total
            self.metrics["peak_portfolio_value"] = total

        self.metrics["overall_drawdown"] = ((peak - total) / peak) if peak > 0 else 0.0
        self.metrics["daily_loss"] = max(0.0, self.metrics.get("start_of_day_value", 0.0) - total)

        # Sync to SharedState for Governance (MetaController) visibility
        if hasattr(self.shared_state, "metrics"):
            self.shared_state.metrics["drawdown_pct"] = self.metrics["overall_drawdown"]
            self.shared_state.metrics["daily_loss_usdt"] = self.metrics["daily_loss"]

        self.logger.debug(
            "Risk Metrics | PV: %.2f | DailyLoss: %.2f | Drawdown: %.2f%%",
            total, self.metrics["daily_loss"], self.metrics["overall_drawdown"] * 100.0
        )

    # ---------- core periodic evaluation ----------

    async def evaluate_risk(self):
        await self._ensure_initialized()
        await self._reset_daily_metrics_if_needed()
        await self._update_metrics()

        # 1) Global drawdown breach
        if self.max_drawdown_pct > 0 and self.metrics["overall_drawdown"] >= self.max_drawdown_pct:
            msg = f"Overall Drawdown {self.metrics['overall_drawdown']:.2%} >= {self.max_drawdown_pct:.2%}"
            self.logger.critical("🛑 %s", msg)
            await self._set_cot_safe("Application", f"Veto: drawdown {self.metrics['overall_drawdown']:.2%} > {self.max_drawdown_pct:.2%}")
            await self._emergency_shutdown(msg)
            return

        # 2) Daily loss breach
        sod = self.metrics.get("start_of_day_value", 0.0)
        daily_loss_pct = (self.metrics["daily_loss"] / sod) if sod > 0 else 0.0
        if self.max_daily_loss_pct > 0 and daily_loss_pct >= self.max_daily_loss_pct:
            msg = f"Max Daily Loss {daily_loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}"
            self.logger.warning("🚨 %s", msg)
            await self._set_cot_safe("Application", f"Veto: daily loss {daily_loss_pct:.2%} > {self.max_daily_loss_pct:.2%}")
            self.metrics["daily_trading_halt"] = True
            await self._emergency_shutdown(msg)
            return

        # 3 & 4) Exposure checks
        account_value = self.metrics["current_portfolio_value"]
        total_exposure_pct = 0.0
        if account_value > 0:
            open_trades = self.shared_state.get_all_open_trades() or {}
            latest_prices = getattr(self.shared_state, "latest_prices", {}) or {}

            def _px(s: str, tr: Dict[str, Any]) -> Optional[float]:
                return latest_prices.get(s) or tr.get("entry_price")

            total_open_value = 0.0
            for symbol, tr in open_trades.items():
                price = _px(symbol, tr)
                if not price:
                    self.logger.debug("Skip exposure calc for %s: missing price", symbol)
                    continue
                position_value = float(tr.get("quantity", 0.0)) * float(price)
                total_open_value += position_value
                exposure_pct = (position_value / account_value) if account_value > 0 else 0.0
                if self.max_pos_exposure_pct > 0 and exposure_pct > self.max_pos_exposure_pct:
                    warn = f"High exposure {symbol}: {exposure_pct:.2%} > {self.max_pos_exposure_pct:.2%}"
                    self.logger.warning("⚠️ %s", warn)
                    await self._set_cot_safe(symbol, f"Veto: position exposure {exposure_pct:.2%} > {self.max_pos_exposure_pct:.2%}")
                    await self._safe_health("RiskManager", "Warning", warn)

            total_exposure_pct = (total_open_value / account_value) if account_value > 0 else 0.0
            if self.max_total_exposure_pct > 0 and total_exposure_pct > self.max_total_exposure_pct:
                warn = f"Total exposure {total_exposure_pct:.2%} > {self.max_total_exposure_pct:.2%}. Restricting trades."
                self.logger.warning("⚠️ %s", warn)
                self.metrics["trading_restricted"] = True
                await self._set_cot_safe("Portfolio", f"Veto: total exposure {total_exposure_pct:.2%} > {self.max_total_exposure_pct:.2%}")
                await self._safe_health("RiskManager", "Warning", warn)
            elif self.metrics.get("trading_restricted"):
                self.logger.info("Total exposure within limits. Lifting restrictions.")

        await self._safe_health("RiskManager", "Operational", "Risk evaluation complete.")
        self.logger.debug(
            "RiskManager: DailyLossPct=%s TotalExposurePct=%s",
            f"{daily_loss_pct:.2%}" if sod > 0 else "N/A",
            f"{total_exposure_pct:.2%}" if account_value > 0 else "N/A"
        )

    # ---------- order-level gate (ExecutionManager calls this) ----------

    async def validate_order(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        qty: Optional[float] = None,
        quote_qty: Optional[float] = None,
        order_type: str = "MARKET",
        order: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[float], Optional[float]]:
        """
        Flexible API:
          - EM style: validate_order(symbol=..., side=..., qty=..., quote_qty=..., order_type="MARKET")
          - Dict style: validate_order(order={"symbol": ..., "side": ..., "quantity": ..., "planned_quote": ...})
        Returns: (ok, reason, adj_qty, adj_quote)
        """
        await self._ensure_initialized()

        # dict-style extraction
        if order and isinstance(order, dict):
            symbol = symbol or order.get("symbol")
            side = side or order.get("side")
            qty = order.get("quantity") if qty is None else qty
            quote_qty = order.get("planned_quote") if quote_qty is None else quote_qty
        is_liquidation = False
        if order and isinstance(order, dict):
            if order.get("is_liquidation"):
                is_liquidation = True
            tag = str(order.get("tag", "") or "")
            if "liquidation" in tag.lower():
                is_liquidation = True

        sym = (symbol or "").replace("/", "").upper()
        s = (side or "").upper()

        # Light input validation
        if not sym or s not in ("BUY", "SELL"):
            return False, "invalid_side_or_symbol", None, None

        # global halts / restrictions / freeze
        if self.metrics.get("daily_trading_halt"):
            return False, "daily_halt", None, None
        if s == "BUY" and self.is_buy_freeze_active():
            return False, "buy_freeze_active", None, None

        # enforce per-trade quote bounds for BUY
        adj_qty, adj_quote = None, None
        if s == "BUY":
            if quote_qty is not None and quote_qty > 0:
                q = float(quote_qty)
                if q < self.min_trade_quote:
                    return False, "below_min_trade_quote", None, None
                if q > self.max_trade_quote > 0:
                    adj_quote = float(self.max_trade_quote)  # cap spend
                else:
                    adj_quote = q
            # if no quote path, leave sizing to EM/hygiene guards; still pass ok
        else:  # SELL safety: cap by free balance if available
            # Exit-feasibility floor for SELLs (skip for liquidation)
            if not is_liquidation:
                try:
                    price = None
                    if hasattr(self.shared_state, "get_latest_price_safe"):
                        price = await self.shared_state.get_latest_price_safe(sym)
                    elif hasattr(self.shared_state, "get_latest_price"):
                        price = await self.shared_state.get_latest_price(sym)
                    exit_info = await self._get_exit_floor_info(sym, price=price)
                    min_exit = float(exit_info.get("min_exit_quote", 0.0) or 0.0)
                    est_quote = float(quote_qty or 0.0)
                    if est_quote <= 0 and price and qty:
                        est_quote = float(qty) * float(price)
                    if min_exit > 0 and est_quote > 0 and est_quote < min_exit:
                        return False, "sell_below_exit_floor", None, None
                except Exception:
                    pass
            base_ccy = sym[:-len(self.base_currency)] if sym.endswith(self.base_currency) else None
            free_ok = None
            try:
                if base_ccy and self.exchange_client:
                    if hasattr(self.exchange_client, "get_account_balance"):
                        bal = await self.exchange_client.get_account_balance(base_ccy)
                        free_ok = float((bal or {}).get("free", 0.0))
                    elif hasattr(self.exchange_client, "get_balances"):
                        bals = await self.exchange_client.get_balances()
                        if isinstance(bals, dict):
                            # Assuming get_balances returns {'ASSET': {'free': X, ...}}
                            free_ok = float((bals.get(base_ccy) or {}).get("free", 0.0))
                        elif isinstance(bals, list):
                            # Assuming get_balances returns [{'asset': 'ASSET', 'free': X, ...}]
                            for it in bals:
                                if str(it.get("asset", "")).upper() == base_ccy:
                                    free_ok = float((it or {}).get("free", 0.0)); break
            except Exception:
                free_ok = None
            if free_ok is not None and qty is not None:
                adj_qty = min(float(qty), max(0.0, free_ok))

        return True, "ok", adj_qty, adj_quote

    @staticmethod
    def _floor_to_step(value: Optional[float], step: Optional[float]) -> float:
        if value is None:
            return 0.0
        if not step:
            return float(value)
        return math.floor(float(value) / float(step)) * float(step)

    # ---- EM compatibility: canonical approve + legacy allow + should_allow ----
    def _order_to_kwargs(self, order):
        if order is None:
            return {}
        if isinstance(order, dict):
            return {
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "qty": order.get("quantity") or order.get("qty"),
                "quote_qty": order.get("planned_quote") or order.get("quote_quantity") or order.get("quote"),
                "order_type": order.get("type") or order.get("order_type") or "MARKET",
            }
        return {
                "symbol": getattr(order, "symbol", None),
                "side": getattr(order, "side", None),
                "qty": getattr(order, "quantity", None),
                "quote_qty": getattr(order, "planned_quote", None) or getattr(order, "quote_quantity", None),
                "order_type": getattr(order, "order_type", None) or getattr(order, "type", None) or "MARKET",
        }

    def _apply_adjustments_to_order(self, order, adj_qty, adj_quote):
        if order is None:
            return
        if adj_qty is not None:
            if isinstance(order, dict):
                order["quantity"] = float(adj_qty)
            else:
                with contextlib.suppress(Exception):
                    setattr(order, "quantity", float(adj_qty))

        if adj_quote is not None:
            if isinstance(order, dict):
                order["planned_quote"] = float(adj_quote)
                order["quote_quantity"] = float(adj_quote)
            else:
                # set BOTH if they exist; never rely on only one name
                with contextlib.suppress(Exception):
                    if hasattr(order, "planned_quote"):
                        setattr(order, "planned_quote", float(adj_quote))
                with contextlib.suppress(Exception):
                    if hasattr(order, "quote_quantity"):
                        setattr(order, "quote_quantity", float(adj_quote))

    async def approve(self, order=None, **kwargs) -> Tuple[bool, str]:
        params = self._order_to_kwargs(order)
        params.update({k: v for k, v in kwargs.items() if v is not None})
        ok, reason, adj_qty, adj_quote = await self.validate_order(
            symbol=params.get("symbol"),
            side=params.get("side"),
            qty=params.get("qty"),
            quote_qty=params.get("quote_qty"),
            order_type=params.get("order_type") or "MARKET",
            order=order if isinstance(order, dict) else None,
        )
        if ok and order is not None:
            self._apply_adjustments_to_order(order, adj_qty, adj_quote)
        return bool(ok), str(reason or "ok")

    async def allow(self, order) -> bool:
        ok, _ = await self.approve(order)
        return bool(ok)

    async def should_allow(self, order) -> Tuple[bool, str]:
        # EM prefers this if present
        return await self.approve(order)

    # ---------- diagnostics you wrote ----------

    def evaluate_trade_risk(self, symbol: str, price: float) -> float:
        sentiment = self.shared_state.sentiment_score.get(symbol, 0.0)
        volatility = self.shared_state.volatility_state.get(symbol, "normal")
        base = 0.5
        base += 0.2 if sentiment < -0.5 else (-0.2 if sentiment > 0.5 else 0.0)
        base += 0.2 if volatility == "high" else (-0.1 if volatility == "low" else 0.0)
        risk = min(1.0, max(0.0, base))
        self.logger.debug("Risk score %s @ %.8f → %.2f | Sent=%.2f Vol=%s", symbol, price, risk, sentiment, volatility)
        return risk

    async def run_diagnostics(self, symbol: str):
        await self._ensure_initialized()
        price = await self.shared_state.get_latest_price_safe(symbol)
        if price is None:
            self.logger.warning("⚠️ Risk diagnostics skipped: no price for %s", symbol)
            return
        score = self.evaluate_trade_risk(symbol, price)
        self.logger.info("Diagnostic risk score for %s: %.2f", symbol, score)

    def is_buy_freeze_active(self) -> bool:
        """
        Checks if a portfolio-wide buy freeze is active due to daily trading halt,
        general trading restrictions, or global kill-switch.
        """
        return bool(
            self._global_freeze or 
            self.metrics.get("daily_trading_halt") or 
            self.metrics.get("trading_restricted")
        )

    def freeze_trading(self, reason: str = "Manual freeze") -> None:
        """
        Kill-Switch: Freeze all trading immediately. Requires manual unfreeze.
        This is a graceful freeze - system stays running but no new orders are placed.
        """
        self._global_freeze = True
        self._global_freeze_reason = reason
        self.logger.critical("🛑 KILL-SWITCH ACTIVATED: %s", reason)
        with contextlib.suppress(Exception):
            ComponentStatusLogger.log_status("RiskManager", "FROZEN", f"Kill-switch: {reason}")

    def unfreeze_trading(self) -> None:
        """
        Manually unfreeze trading after kill-switch was activated.
        """
        if self._global_freeze:
            self.logger.info("✅ KILL-SWITCH DEACTIVATED. Trading resumed.")
        self._global_freeze = False
        self._global_freeze_reason = None
        with contextlib.suppress(Exception):
            ComponentStatusLogger.log_status("RiskManager", "Operational", "Kill-switch released")

    def is_frozen(self) -> bool:
        """Check if kill-switch is active."""
        return self._global_freeze

    # ---------- shutdown + health ----------

    async def _emergency_shutdown(self, reason: str):
        self.logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED! Reason: %s", reason)

        open_positions = self.shared_state.get_all_open_trades() or {}
        if open_positions:
            self.logger.warning("Attempting to close %d open positions...", len(open_positions))
            close_tasks = [
                _asyncio.create_task(self.execution_manager.close_trade(sym, reason="Emergency Shutdown"))
                for sym in open_positions.keys()
            ]
            done, pending = await _asyncio.wait(
                close_tasks,
                timeout=self.shutdown_timeout,
                return_when=_asyncio.ALL_COMPLETED
            )
            for t in done:
                if t.exception():
                    self.logger.error("Error closing position during emergency shutdown: %s", t.exception(), exc_info=True)
            if pending:
                for t in pending:
                    t.cancel()
                self.logger.warning("%d positions not closed within %ss.", len(pending), self.shutdown_timeout)

        # Notify
        if getattr(self.execution_manager, "alert_callback", None):
            with contextlib.suppress(Exception):
                await self.execution_manager.alert_callback(
                    f"🔴 OCTIVAULT TRADER SHUTDOWN: {reason}", level="CRITICAL"
                )

        await self._safe_health("RiskManager", "CRITICAL", f"Emergency shutdown: {reason}")
        await self._safe_health("Application", "SHUTDOWN", "Risk limit breached, application terminating.")
        raise SystemExit(f"Risk limit breached: {reason}")

    async def _safe_health(self, component: str, status: str, detail: str):
        with contextlib.suppress(Exception):
            await self.shared_state.update_system_health(component, status, detail)
        with contextlib.suppress(Exception):
            ComponentStatusLogger.log_status(component, status, detail)

    async def _safe_timestamp(self, component: str):
        with contextlib.suppress(Exception):
            await self.shared_state.update_timestamp(component)
        with contextlib.suppress(Exception):
            ComponentStatusLogger.log_status(component, "Ping", "Timestamp updated")

    async def _set_cot_safe(self, scope: str, message: str):
        """
        Best-effort hook to notify CoT/Meta components about vetoes.
        """
        try:
            cot = getattr(self.shared_state, "cot_assistant", None) or getattr(self, "cot_assistant", None)
            if cot and hasattr(cot, "set_veto"):
                res = cot.set_veto(scope, message)
                if inspect.isawaitable(res):
                    await res
                return
            add_veto = getattr(self.shared_state, "add_veto", None)
            if add_veto:
                res = add_veto(scope, message)
                if inspect.isawaitable(res):
                    await res
        except Exception as e:
            self.logger.debug("set_cot_safe noop: %s", e)

    async def _p6_periodic_check_loop(self, interval_sec: float):
        """
        Lightweight liveness/health loop for Phase 6.
        Does minimal work and is safe to run alongside full run_loop() when promoted later.
        """
        # Initial announce
        try:
            await self._safe_health("RiskManager", "STARTING", f"P6 check loop @ {interval_sec:.1f}s")
        except Exception:
            pass

        while self._running:
            try:
                # Minimal periodic evaluation; tolerate failures without escalation.
                await self._reset_daily_metrics_if_needed()
                await self._update_metrics()
                await self._safe_health("RiskManager", "Healthy", "P6 liveness OK")
            except _asyncio.CancelledError:
                break
            except Exception as e:
                # Non-fatal; emit warning-level health but keep loop alive
                self.logger.debug("P6 periodic check failed: %s", e, exc_info=True)
                with contextlib.suppress(Exception):
                    await self._safe_health("RiskManager", "Warning", f"P6 check error: {e}")
            finally:
                try:
                    await _asyncio.sleep(max(5.0, float(interval_sec)))
                except _asyncio.CancelledError:
                    break

        # Shutdown announce
        with contextlib.suppress(Exception):
            await self._safe_health("RiskManager", "SHUTDOWN", "P6 check loop stopped")

    async def start(self):
        """
        Lightweight start() for P6: emit HealthStatus and schedule periodic checks.
        Idempotent and non-blocking. Safe to call multiple times.
        """
        if self._running:
            return
        self._running = True

        # Determine heartbeat interval (default 15s for P6)
        default_beat = 15.0
        try:
            beat = float(getattr(self.config, "RISK_HEALTH_BEAT_SECS", default_beat))
        except Exception:
            beat = default_beat

        # Kick a health reporter if not already running
        if self._health_task is None or self._health_task.done():
            self._health_task = _asyncio.create_task(self.report_health_loop(), name="RiskManager:health")

        # Launch the lightweight periodic check loop
        self._p6_task = _asyncio.create_task(self._p6_periodic_check_loop(beat), name="RiskManager:P6")
        with contextlib.suppress(Exception):
            await self._safe_health("RiskManager", "Operational", "P6 start scheduled")

    async def report_health_loop(self):
        interval = min(30, max(5, float(getattr(self.config, "RISK_HEALTH_BEAT_SECS", 10))))
        try:
            while True:
                try:
                    await self._safe_timestamp("RiskManager")
                    await self._safe_health("RiskManager", "Healthy", "Running normally.")
                except Exception as e:
                    self.logger.warning("⚠️ RiskManager health update failed: %s", e)
                await _asyncio.sleep(interval)
        except _asyncio.CancelledError:
            self.logger.info("RiskManager health loop cancelled.")

    async def stop(self):
        """Stop the lightweight P6 tasks."""
        self._running = False
        # Cancel P6 loop
        t = getattr(self, "_p6_task", None)
        if t:
            try:
                t.cancel()
            except Exception:
                pass
            self._p6_task = None
        # Cancel health loop if it was started by start()
        if self._health_task and not self._health_task.done():
            try:
                self._health_task.cancel()
            except Exception:
                pass
            with contextlib.suppress(Exception):
                await self._health_task
            self._health_task = None

    async def run_loop(self):
        """Preferred entrypoint for scheduler."""
        self.logger.info("🛡️ RiskManager started.")
        with contextlib.suppress(Exception):
            ComponentStatusLogger.log_status("RiskManager", "Running", "Risk loop active")
        self._health_task = _asyncio.create_task(self.report_health_loop())
        await self._safe_health("RiskManager", "Operational", "Monitoring risk.")

        try:
            while True:
                try:
                    await self.evaluate_risk()
                except _asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("❌ RiskManager error: %s", e, exc_info=True)
                    await self._safe_health("RiskManager", "Error", f"Run loop error: {e}")
                await _asyncio.sleep(self.risk_eval_sec)
        except _asyncio.CancelledError:
            self.logger.info("RiskManager loop cancelled.")
        finally:
            if self._health_task and not self._health_task.done():
                self._health_task.cancel()
                with contextlib.suppress(Exception):
                    await self._health_task

    async def run(self):
        """Backward-compat alias."""
        await self.run_loop()

    # --- Attachment hook per canon ---
    def attach_execution_manager(self, execution_manager: object):
        """Late-bind ExecutionManager after construction to avoid init-order cycles."""
        self.execution_manager = execution_manager
        # backfill exchange_client if still None
        if self.exchange_client is None:
            with contextlib.suppress(Exception):
                self.exchange_client = getattr(execution_manager, "exchange_client", None)
        if self.logger:
            self.logger.info("[RiskManager] ExecutionManager attached")


# ===== Imports =====

import time
from datetime import datetime
from typing import Optional, Dict, Any



# ===== Exceptions =====

class RiskViolation(Exception): ...
class ExposureViolation(RiskViolation): ...
class BuyFreezeActive(RiskViolation): ...
class IntegrityError(RiskViolation): ...



# ===== Helpers =====

def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _get_cfg_val(cfg, key: str, default):
    try:
        return getattr(cfg, key)
    except Exception:
        return default
