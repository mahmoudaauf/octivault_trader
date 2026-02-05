"""
ExecutionManager subsystem extracted from MetaController.
Handles trade execution, order management, and related logic.
"""


import time
import asyncio
from collections import deque, defaultdict
from typing import Optional
from core.meta_controller import ExecutionError
from core.meta_controller import classify_execution_error as _classify_execution_error

class ExecutionLogic:
    def __init__(self, shared_state, execution_manager, config, logger, meta_controller):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.config = config
        self.logger = logger
        self.meta_controller = meta_controller
        # Initialize trade tracking attributes
        self._trade_timestamps = deque(maxlen=1000)
        self._trade_timestamps_sym = defaultdict(lambda: deque(maxlen=100))
        self._trade_timestamps_agent = defaultdict(lambda: deque(maxlen=100))
        self._max_trades_per_hour = int(getattr(config, 'MAX_TRADES_PER_HOUR', 10))
        self._max_trades_per_day = int(getattr(config, 'MAX_TRADES_PER_DAY', 0) or 0)
        self._trade_timestamps_day = deque(maxlen=5000)
        self._last_buy_ts = {}
        # Add any other needed initializations here

    async def execute_decision(self, intent: dict) -> dict:
        try:
            action = str(intent.get("action", "")).upper()
            symbol = self._normalize_symbol(intent.get("symbol", ""))

            if action not in {"BUY", "SELL"} or not symbol:
                return {"ok": False, "status": "REJECTED", "reason": "invalid_intent"}

            sig = {
                "action": action,
                "symbol": symbol,
                "confidence": 1.0,
                "agent": intent.get("agent", "Meta"),
                "quote": intent.get("quote", intent.get("planned_quote")),
                "quantity": intent.get("quantity"),
                "bypass_conf": True,
            }

            acc = await self._safe_await(getattr(self.shared_state, "get_accepted_symbols_snapshot", lambda: [])())
            accepted = set(acc) if acc else {symbol}
            await self._execute_decision(symbol, action, sig, accepted)
            return {"ok": True, "status": "ACCEPTED"}
        except Exception as e:
            try:
                self.logger.error("execute_decision failed: %s", e, exc_info=True)
            except Exception:
                pass
            return {"ok": False, "status": "REJECTED", "reason": "internal_error"}

    def _extract_decision_id(self, signal: dict) -> Optional[str]:
        if not isinstance(signal, dict):
            return None
        for key in (
            "decision_id",
            "decisionId",
            "signal_id",
            "signalId",
            "id",
            "cache_key",
            "intent_id",
            "intentId",
        ):
            val = signal.get(key)
            if val:
                return str(val)
        return None

    async def _execute_decision(self, symbol: str, side: str, signal: dict, accepted_symbols_set: set):
        try:
            # Gating and trade limit logic
            now = time.time()
            agent_name = signal.get("agent", "Meta")
            for dq in [self._trade_timestamps, self._trade_timestamps_sym[symbol], self._trade_timestamps_agent[agent_name]]:
                while dq and (now - dq[0] > 3600):
                    dq.popleft()
            while self._trade_timestamps_day and (now - self._trade_timestamps_day[0] > 86400):
                self._trade_timestamps_day.popleft()
            is_bootstrap = "bootstrap" in str(signal.get("reason", "")).lower()
            is_flat_init = signal.get("_flat_init_buy", False)
            if not (is_bootstrap or is_flat_init):
                max_hourly = int(getattr(self.config, "MAX_TRADES_PER_HOUR", self._max_trades_per_hour) or 0)
                max_daily = int(getattr(self.config, "MAX_TRADES_PER_DAY", self._max_trades_per_day) or 0)
                if max_daily > 0 and len(self._trade_timestamps_day) >= max_daily:
                    self.logger.info("[Meta] Skip %s BUY: Global daily trade limit (%d) reached.", symbol, max_daily)
                    return {"ok": False, "status": "skipped", "reason": "global_daily_limit", "reason_detail": "global_daily_limit_reached"}
                if max_hourly > 0 and len(self._trade_timestamps) >= max_hourly:
                    self.logger.info("[Meta] Skip %s BUY: Global hourly trade limit (%d) reached.", symbol, max_hourly)
                    return {"ok": False, "status": "skipped", "reason": "global_limit", "reason_detail": "global_hourly_limit_reached"}
                max_sym_hourly = int(getattr(self.config, "MAX_TRADES_PER_SYMBOL_PER_HOUR", 2) or 0)
                if max_sym_hourly > 0 and len(self._trade_timestamps_sym[symbol]) >= max_sym_hourly:
                    self.logger.info("[Meta] Skip %s BUY: Symbol hourly trade limit reached.", symbol)
                    return {"ok": False, "status": "skipped", "reason": "symbol_limit", "reason_detail": "symbol_hourly_limit_reached"}
                agent_limit = max(1, int(max_hourly * 0.75))
                if len(self._trade_timestamps_agent[agent_name]) >= agent_limit:
                    self.logger.info("[Meta] Skip %s BUY: Agent %s hourly limit reached.", symbol, agent_name)
                    return {"ok": False, "status": "skipped", "reason": "agent_limit", "reason_detail": f"agent_limit_{agent_name}_reached"}
            else:
                self.logger.info(
                    "[Meta:FIX#2] Bootstrap BUY %s bypassing hourly limits (is_bootstrap=%s, is_flat_init=%s)",
                    symbol, is_bootstrap, is_flat_init
                )

            if symbol not in accepted_symbols_set:
                # SELL always bypasses accepted_symbols check
                if side == "SELL":
                    self.logger.info(
                        "[Meta:P9] SELL bypass: %s not in accepted set but SELL must execute (P9 Rule: Exits always allowed). Proceeding.",
                        symbol
                    )
                else:
                    is_bootstrap = "bootstrap" in str(signal.get("reason", "")).lower()
                    if side == "BUY" and is_bootstrap:
                        self.logger.info("[Meta:Bootstrap] Gating bypass: %s is not in accepted set but is a bootstrap BUY. Proceeding.", symbol)
                    else:
                        self.logger.warning("Skipping unaccepted symbol: %s", symbol)
                        await self.meta_controller._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "symbol_not_accepted"})
                        return {"ok": False, "status": "skipped", "reason": "symbol_not_accepted"}

            # Ensure per-symbol market data is ready if SharedState provides a hook
            try:
                fn = getattr(self.shared_state, "is_symbol_data_ready", None)
                if callable(fn):
                    ok = fn(symbol)
                    if asyncio.iscoroutine(ok):
                        ok = await ok
                    if not ok:
                        await self.meta_controller._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "symbol_data_not_ready"})
                        return {"ok": False, "status": "skipped", "reason": "symbol_data_not_ready"}
            except Exception:
                self.logger.debug("symbol readiness check failed for %s", symbol, exc_info=True)

            if side == "BUY":
                planned_quote = float(signal.get("_planned_quote", 0.0))
                if planned_quote <= 0:
                    planned_quote = await self.meta_controller._planned_quote_for(symbol, signal)

                intent_owner = signal.get("_intent_owner", "")
                if side == "SELL" and hasattr(self.shared_state, "get_pending_intent"):
                    intent = self.shared_state.get_pending_intent(symbol, "BUY")
                    if intent and intent.state == "ACCUMULATING":
                        is_protected_sell = any(tag in str(intent_owner).lower() for tag in ["risk", "liquidation", "tp_sl", "emergency"])
                        if not is_protected_sell:
                            self.logger.info(
                                "[Meta:AccumGuard] Blocking SELL on %s: position is ACCUMULATING (intent_owner=%s). Only risk/liquidation/tp_sl can override.",
                                symbol, intent_owner
                            )
                            await self.shared_state.record_rejection(
                                symbol, "SELL", "ACCUMULATING_PROTECTION", source="MetaController"
                            )
                            if hasattr(self.shared_state, "record_policy_conflict"):
                                self.shared_state.record_policy_conflict("accumulating_protection_blocks")
                            return {"ok": False, "status": "skipped", "reason": "accumulating_protection"}

                min_capital_floor = float(self.meta_controller._cfg("CAPITAL_PRESERVATION_FLOOR", 50.0))
                free_quote = 0.0
                try:
                    if hasattr(self.shared_state, "get_free_quote"):
                        maybe = self.shared_state.get_free_quote()
                        if asyncio.iscoroutine(maybe):
                            free_quote = await maybe
                        else:
                            free_quote = float(maybe or 0.0)
                except Exception:
                    free_quote = 0.0

                policy_flags = {"POLICY_SINGLE_AUTHORITY": True}
                decision_id = self._extract_decision_id(signal)
                econ_ok, econ_reason, econ_metrics = self.meta_controller._check_economic_profitability(symbol, signal)
                if not econ_ok:
                    self.meta_controller._log_reason("INFO", symbol, f"economic_guard:{econ_reason}")
                    if hasattr(self.shared_state, "record_rejection"):
                        await self.shared_state.record_rejection(
                            symbol, "BUY", "ECONOMIC_PROFITABILITY_BLOCK", source="MetaController"
                        )
                    await self.meta_controller._log_execution_result(
                        symbol,
                        side,
                        signal,
                        {"status": "skipped", "reason": "economic_guard", "details": econ_metrics},
                    )
                    return {"ok": False, "status": "skipped", "reason": "economic_guard"}
                policy_flags["ECONOMIC_PROFITABILITY_INVARIANT"] = True

                bootstrap_policy_ctx = None
                if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                    bootstrap_policy_ctx = self.meta_controller._build_policy_context(
                        symbol,
                        side,
                        extra={"bootstrap_bypass": True, "decision_id": decision_id}
                    )
                    self.logger.info("[Meta:BOOTSTRAP] Using bootstrap_bypass context for affordability check: %s", symbol)

                can_ex, _, reason = await self.execution_manager.can_afford_market_buy(symbol, planned_quote, policy_context=bootstrap_policy_ctx)

                if not can_ex:
                    self.logger.warning("⚡ [Escalation] Signal %s for %s has zero executable qty (%s). Triggering Rule 5 Escalation.", symbol, side, reason)
                    agent = signal.get("agent", "Meta")
                    if hasattr(self.shared_state, "report_agent_capital_failure"):
                        self.shared_state.report_agent_capital_failure(agent)
                    if hasattr(self.shared_state, "ops_plane_ready_event"):
                        self.shared_state.ops_plane_ready_event.clear()
                        self.logger.info("[Meta] Readiness = FALSE (Escalation Triggered)")
                    if self.meta_controller.liquidation_agent and hasattr(self.meta_controller.liquidation_agent, "_free_usdt_now"):
                        target_usdt = float(self.meta_controller._cfg("MIN_NOTIONAL_FLOOR", 15.0))
                        try:
                            if hasattr(self.shared_state, "compute_min_entry_quote"):
                                target_usdt = await self.shared_state.compute_min_entry_quote(
                                    symbol,
                                    default_quote=target_usdt,
                                )
                        except Exception:
                            pass
                        await self.meta_controller.liquidation_agent._free_usdt_now(
                            target=target_usdt,
                            reason=f"rule5_escalation_{symbol}"
                        )
                    elif self.meta_controller.liquidation_agent:
                        self.logger.warning(f"[Meta] LiquidationAgent doesn't have _free_usdt_now method. Using propose_liquidations instead.")
                        try:
                            target_usdt = float(self.meta_controller._cfg("MIN_NOTIONAL_FLOOR", 15.0))
                            if hasattr(self.shared_state, "compute_min_entry_quote"):
                                target_usdt = await self.shared_state.compute_min_entry_quote(
                                    symbol,
                                    default_quote=target_usdt,
                                )
                            proposals = await self.meta_controller.liquidation_agent.propose_liquidations(
                                gap_usdt=target_usdt,
                                reason=f"rule5_escalation_{symbol}",
                                force=True
                            )
                            if proposals:
                                self.logger.info(f"[Meta] Generated {len(proposals)} liquidation proposals for escalation")
                        except Exception as e:
                            self.logger.warning(f"[Meta] Failed to generate liquidation proposals: {e}")
                    else:
                        self.logger.warning(f"[Meta] No liquidation agent available for escalation")
                    if hasattr(self.shared_state, "replan_request_event"):
                        self.shared_state.replan_request_event.set()
                    await self.meta_controller._log_execution_result(symbol, side, signal, {"status": "failed", "reason": f"rule5_escalation_{reason}"})
                    return {"ok": False, "status": "failed", "reason": f"rule5_escalation_{reason}"}

                if signal.get("_need_liquidity"):
                    success = await self.meta_controller._attempt_liquidity_healing(symbol, signal)
                    if not success:
                        await self.meta_controller._log_execution_result(
                            symbol, side, signal, {"status": "skipped", "reason": "liquidity_healing_failed"}
                        )
                        return {"ok": False, "status": "skipped", "reason": "liquidity_healing"}

                if self.meta_controller.risk_manager and hasattr(self.meta_controller.risk_manager, "pre_check"):
                    tier = signal.get("_tier", "A")
                    is_liq = signal.get("_is_starvation_sell") or signal.get("_quote_based") or signal.get("_batch_sell")
                    tag = signal.get("_tag") or signal.get("tag") or ""
                    is_liq = is_liq or ("liquidation" in str(tag))
                    ok, r_reason = await self._safe_await(self.meta_controller.risk_manager.pre_check(
                        symbol=symbol, side="BUY", planned_quote=planned_quote, tier=tier, is_liquidation=is_liq
                    ))
                    if not ok:
                        self.meta_controller._log_reason("INFO", symbol, f"risk_precheck:{r_reason}")
                        await self.meta_controller._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": f"risk:{r_reason}"})
                        return {"ok": False, "status": "skipped", "reason": f"risk:{r_reason}"}

                if hasattr(self.shared_state, "is_intent_valid"):
                    if not self.shared_state.is_intent_valid(symbol, "BUY"):
                        self.logger.warning("[Meta] Signal no longer valid at firing time for %s. Skipping.", symbol)
                        await self.meta_controller._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": "signal_invalid_at_firing"})
                        return {"ok": False, "status": "skipped", "reason": "signal_invalid"}

                tier = signal.get("_tier", "A")
                extra_ctx = {"economic_guard": econ_metrics}
                if decision_id:
                    extra_ctx["decision_id"] = decision_id
                if signal.get("_accumulate_mode"):
                    extra_ctx["_accumulate_mode"] = True
                    self.logger.info("[Meta:P0] Passing ACCUMULATE_MODE flag to ExecutionManager for %s", symbol)
                
                # BOOTSTRAP_MODE: If signal is marked with _bootstrap or _bootstrap_override, pass it through policy context
                if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                    extra_ctx["bootstrap_bypass"] = True
                    self.logger.info("[Meta:BOOTSTRAP] Passing bootstrap_bypass flag to ExecutionManager for %s", symbol)

                policy_ctx = self.meta_controller._build_policy_context(
                    symbol,
                    "BUY",
                    policies=[p for p, enabled in policy_flags.items() if enabled],
                    extra=extra_ctx,
                )
                self.meta_controller.increment_execution_attempts()
                result = await self.execution_manager.execute_trade(
                    symbol=symbol,
                    side="buy",
                    quantity=None,
                    planned_quote=planned_quote,
                    tag=f"meta/{signal.get('agent', 'Meta')}",
                    tier=tier,
                    policy_context=policy_ctx,
                )

                ec = result.get("error_code") or result.get("reason")
                if not result.get("ok") and ec in ("InsufficientBalance", "INSUFFICIENT_QUOTE", "RESERVE_FLOOR", "QUOTE_LT_MIN_NOTIONAL", "MIN_NOTIONAL_VIOLATION"):
                    agent = signal.get("agent", "Meta")
                    if hasattr(self.shared_state, "report_agent_capital_failure"):
                        self.shared_state.report_agent_capital_failure(agent)
                    if self.meta_controller.liquidation_agent and hasattr(self.meta_controller.liquidation_agent, "_free_usdt_now"):
                        self.logger.info("⚡ [Escalation] Insufficient funds for %s. Triggering forced liquidation.", symbol)
                        await self.meta_controller.liquidation_agent._free_usdt_now(
                            target=float(planned_quote),
                            reason=f"escalation_{symbol}"
                        )
                    elif self.meta_controller.liquidation_agent:
                        self.logger.info("⚡ [Escalation] Insufficient funds for %s. Using propose_liquidations fallback.", symbol)
                        try:
                            proposals = await self.meta_controller.liquidation_agent.propose_liquidations(
                                gap_usdt=float(planned_quote),
                                reason=f"escalation_{symbol}",
                                force=True
                            )
                            if proposals:
                                self.logger.info(f"[Meta] Generated {len(proposals)} liquidation proposals for insufficient funds")
                        except Exception as e:
                            self.logger.warning(f"[Meta] Failed to generate liquidation proposals: {e}")
                    else:
                        self.logger.warning(f"[Meta] No liquidation agent available for insufficient funds escalation")
                        if hasattr(self.shared_state, "replan_request_event"):
                            self.shared_state.replan_request_event.set()
                        await asyncio.sleep(float(self.meta_controller._cfg("ESCALATION_RETRY_DELAY_SEC", default=2.0)))
                        self.logger.info("🔄 [Escalation] Retrying BUY for %s after liquidation.", symbol)
                        retry_extra = {
                            "economic_guard": econ_metrics,
                            "retry": "post_liquidation",
                        }
                        if decision_id:
                            retry_extra["decision_id"] = decision_id
                        if signal.get("_bootstrap") or signal.get("_bootstrap_override"):
                            retry_extra["bootstrap_bypass"] = True
                            self.logger.info("[Meta:BOOTSTRAP] Passing bootstrap_bypass flag to ExecutionManager for %s (RETRY)", symbol)

                        retry_policy_ctx = self.meta_controller._build_policy_context(
                            symbol,
                            "BUY",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra=retry_extra,
                        )
                        self.meta_controller.increment_execution_attempts()
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="buy",
                            quantity=None,
                            planned_quote=planned_quote,
                            tag=f"meta/{signal.get('agent', 'Meta')}",
                            policy_context=retry_policy_ctx,
                        )

                if str(result.get("status", "")).lower() in {"placed", "executed", "filled"}:
                    ts = time.time()
                    self._trade_timestamps.append(ts)
                    self._trade_timestamps_day.append(ts)
                    self._trade_timestamps_sym[symbol].append(ts)
                    self._trade_timestamps_agent[agent_name].append(ts)
                    self._last_buy_ts[symbol] = ts
                    try:
                        if hasattr(self.shared_state, "set_cooldown"):
                            cooldown_sec = float(self.meta_controller._cfg("META_DECISION_COOLDOWN_SEC", default=15))
                            await self.shared_state.set_cooldown(symbol, cooldown_sec)
                    except Exception:
                        self.logger.debug("Failed to set SharedState cooldown for %s", symbol)
                    try:
                        await self.meta_controller._health_set("Healthy", f"Executed BUY {symbol}")
                    except Exception:
                        pass

            else:  # SELL
                policy_flags = {"POLICY_SINGLE_AUTHORITY": True}
                decision_id = self._extract_decision_id(signal)
                qty = signal.get("quantity")
                tag = signal.get("_tag") or signal.get("tag") or ""
                is_starvation_sell = bool(signal.get("_is_starvation_sell"))
                is_quote_based_signal = bool(signal.get("_quote_based"))
                is_batch_sell = bool(signal.get("_batch_sell"))
                is_liq_signal = bool(
                    is_starvation_sell
                    or is_quote_based_signal
                    or is_batch_sell
                    or signal.get("_force_dust_liquidation")
                    or ("liquidation" in str(tag).lower())
                )
                sell_policy_extra = {
                    "liquidation_signal": is_liq_signal,
                    "dust_value": signal.get("_dust_value"),
                }
                if signal.get("_partial_pct") is not None:
                    sell_policy_extra["_partial_pct"] = signal.get("_partial_pct")
                if signal.get("partial_pct") is not None:
                    sell_policy_extra["_partial_pct"] = signal.get("partial_pct")
                if decision_id:
                    sell_policy_extra["decision_id"] = decision_id
                if signal.get("_phase2_guard"):
                    policy_flags["PHASE_2_GRACE_PERIOD"] = True
                    sell_policy_extra["phase2_guard"] = signal["_phase2_guard"]
                if is_liq_signal:
                    policy_flags["UNIFIED_SELL_AUTHORITY"] = True
                    self.logger.info(f"[Meta:UnifiedSell] {symbol} liquidation SELL will assert UNIFIED_SELL_AUTHORITY")
                if not is_liq_signal:
                    is_cold = False
                    try:
                        cold_attr = getattr(self.shared_state, "is_cold_bootstrap", None)
                        if callable(cold_attr):
                            cold_state = cold_attr()
                            if asyncio.iscoroutine(cold_state):
                                cold_state = await cold_state
                            is_cold = bool(cold_state)
                        elif cold_attr is not None:
                            is_cold = bool(cold_attr)
                    except Exception:
                        is_cold = False
                    if is_cold:
                        self.meta_controller._log_reason("INFO", symbol, "sell_blocked_cold_bootstrap")
                        if hasattr(self.shared_state, "record_rejection"):
                            await self.shared_state.record_rejection(
                                symbol, "SELL", "COLD_BOOTSTRAP_BLOCK", source="MetaController"
                            )
                        await self.meta_controller._log_execution_result(
                            symbol,
                            side,
                            signal,
                            {"status": "blocked", "reason": "cold_bootstrap_no_sell"},
                        )
                        return {"ok": False, "status": "blocked", "reason": "cold_bootstrap_no_sell"}
                qty = float(self.shared_state.get_position_qty(symbol) or 0.0)
                if not qty or qty <= 0:
                    self.meta_controller._log_reason("INFO", symbol, "sell_no_position_qty")
                    await self.meta_controller._log_execution_result(
                        symbol, side, signal, {"status": "skipped", "reason": "no_position_quantity"}
                    )
                    return
                if self.meta_controller.risk_manager and hasattr(self.meta_controller.risk_manager, "pre_check"):
                    ok, r_reason = await self._safe_await(self.meta_controller.risk_manager.pre_check(
                        symbol=symbol, side="SELL", is_liquidation=is_liq_signal
                    ))
                    if not ok:
                        self.meta_controller._log_reason("INFO", symbol, f"risk_precheck:{r_reason}")
                        await self.meta_controller._log_execution_result(symbol, side, signal, {"status": "skipped", "reason": f"risk:{r_reason}"})
                        return
                if is_quote_based_signal and is_starvation_sell:
                    quote_value = signal.get("_target_usdt", 0.0)
                    if quote_value > 0:
                        self.logger.warning(
                            "[Meta:QuoteLiq:Execute] Executing QUOTE-BASED liquidation SELL. Symbol: %s, TargetUSDT: %.2f, PositionQty: %.8f. Using quoteOrderQty (bypasses min-notional).",
                            symbol, quote_value, qty
                        )
                        policy_ctx = self.meta_controller._build_policy_context(
                            symbol,
                            "SELL",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra={**sell_policy_extra, "mode": "quote_based"},
                        )
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="sell",
                            quantity=None,
                            planned_quote=quote_value,
                            tag=f"meta/{signal.get('agent', 'Meta')}",
                            policy_context=policy_ctx,
                        )
                    else:
                        self.logger.warning("[Meta:QuoteLiq:Execute] Quote-based liquidation flag set but _target_usdt is invalid. Falling back to quantity-based sell.")
                        policy_ctx = self.meta_controller._build_policy_context(
                            symbol,
                            "SELL",
                            policies=[p for p, enabled in policy_flags.items() if enabled],
                            extra=sell_policy_extra,
                        )
                        result = await self.execution_manager.execute_trade(
                            symbol=symbol,
                            side="sell",
                            quantity=qty,
                            tag=f"meta/{signal.get('agent', 'Meta')}",
                            policy_context=policy_ctx,
                        )
                else:
                    if is_starvation_sell:
                        self.logger.warning(
                            "[Meta:LiquidationHardPath:Execute] Executing LIQUIDATION SELL (batch or fallback). Symbol: %s, Qty: %.8f. Bypassing starvation/affordability gates.",
                            symbol, qty
                        )
                    policy_ctx = self.meta_controller._build_policy_context(
                        symbol,
                        "SELL",
                        policies=[p for p, enabled in policy_flags.items() if enabled],
                        extra={**sell_policy_extra, "mode": "quantity"},
                    )
                    self.meta_controller.increment_execution_attempts()
                    result = await self.execution_manager.execute_trade(
                        symbol=symbol,
                        side="sell",
                        quantity=qty,
                        tag=f"meta/{signal.get('agent', 'Meta')}",
                        policy_context=policy_ctx,
                    )
                if str(result.get("status", "")).lower() in {"placed", "executed", "filled"}:
                    if getattr(self, "_focus_mode_active", False):
                        self._focus_mode_trade_executed = True
                        self._focus_mode_trade_executed_count += 1
                        self.logger.info(f"[FOCUS_MODE] Trade executed: {symbol} {side} (count={self._focus_mode_trade_executed_count})")
                    try:
                        if hasattr(self.shared_state, "set_cooldown"):
                            cooldown_sec = float(self.meta_controller._cfg("META_DECISION_COOLDOWN_SEC", default=15))
                            await self.shared_state.set_cooldown(symbol, cooldown_sec)
                    except Exception:
                        self.logger.debug("Failed to set SharedState cooldown for %s", symbol)
                    try:
                        await self.meta_controller._health_set("Healthy", f"Executed SELL {symbol}")
                    except Exception:
                        pass

            await self.meta_controller._log_execution_result(symbol, side, signal, result)
            await self.meta_controller._update_kpi_metrics("execution")
            return result

        except Exception as e:
            classified_error = _classify_execution_error(e, symbol)
            self.logger.error("Decision execution failed for %s: %s", symbol, classified_error, exc_info=True)
            if classified_error.error_type in (ExecutionError.Type.INSUFFICIENT_BALANCE, ExecutionError.Type.MIN_NOTIONAL_VIOLATION):
                agent = signal.get("agent", "Meta")
                if hasattr(self.shared_state, "report_agent_capital_failure"):
                    self.shared_state.report_agent_capital_failure(agent)
            await self.meta_controller._update_kpi_metrics("error", classified_error.error_type)
            await self.meta_controller._health_set("Critical", f"Execution error for {symbol}: {classified_error}")

    async def _safe_await(self, maybe_coro):
        if asyncio.iscoroutine(maybe_coro):
            return await maybe_coro
        return maybe_coro

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol).upper()
