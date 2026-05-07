"""
Core Engine Real Method Implementations
────────────────────────────────────────

PHASE 2: Implement actual methods that call L0-L8 components

This module provides concrete implementations for each engine method.
Methods are designed to be swapped into the engine classes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Await if coroutine, otherwise return value as-is.

    Bridges sync legacy methods with async façade engines.
    """
    if inspect.iscoroutine(value) or inspect.isawaitable(value):
        return await value
    return value


# ═════════════════════════════════════════════════════════════════════════════
# MARKET_ACCOUNT_ENGINE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════


class MarketAccountEngineImpl:
    """Real implementations for MarketAccountEngine methods."""

    @staticmethod
    async def get_account_state(app_ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Fetch account state from exchange_client (L1).
        """
        exchange_client = app_ctx.get("exchange_client")

        account_data = {
            "balances": {},
            "positions": {},
            "open_orders": [],
            "timestamp": asyncio.get_event_loop().time(),
        }

        if not exchange_client:
            logger.warning("⚠️ exchange_client not available")
            return account_data

        try:
            # Call exchange_client methods
            if hasattr(exchange_client, "get_account"):
                account = await exchange_client.get_account()
                # Transform to standard format
                account_data["balances"] = {
                    b["asset"]: float(b["free"]) for b in account.get("balances", [])
                }

            if hasattr(exchange_client, "get_open_orders"):
                orders = await exchange_client.get_open_orders()
                account_data["open_orders"] = orders

        except Exception as e:
            logger.error(f"❌ Error getting account state: {e}")

        return account_data

    @staticmethod
    async def get_market_prices(
        app_ctx: dict[str, Any], symbols: list[str] | None = None
    ) -> dict[str, float]:
        """
        Fetch prices from market_data_feed or exchange_client (L1/L2).
        """
        market_data_feed = app_ctx.get("market_data_feed")
        exchange_client = app_ctx.get("exchange_client")

        prices = {}

        # Try market_data_feed first (cached, faster)
        if market_data_feed and hasattr(market_data_feed, "get_prices"):
            try:
                prices = await market_data_feed.get_prices(symbols)
                return prices
            except Exception as e:
                logger.debug(f"market_data_feed unavailable: {e}")

        # Fall back to exchange_client
        if exchange_client:
            try:
                if hasattr(exchange_client, "get_prices"):
                    prices = await exchange_client.get_prices(symbols)
                elif hasattr(exchange_client, "get_symbol_price_ticker"):
                    tickers = await exchange_client.get_symbol_price_ticker()
                    prices = {t["symbol"]: float(t["price"]) for t in tickers}
            except Exception as e:
                logger.error(f"❌ Error getting prices: {e}")

        return prices

    @staticmethod
    async def get_wallet_balance(app_ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Get wallet balance from balance_manager (L2).
        """
        balance_manager = app_ctx.get("balance_manager")
        exchange_client = app_ctx.get("exchange_client")

        wallet = {
            "total_usdt": 0.0,
            "available_usdt": 0.0,
            "locked_usdt": 0.0,
            "by_symbol": {},
            "timestamp": asyncio.get_event_loop().time(),
        }

        if balance_manager and hasattr(balance_manager, "get_balance"):
            try:
                balance = await balance_manager.get_balance()
                wallet.update(balance)
                return wallet
            except Exception as e:
                logger.debug(f"balance_manager unavailable: {e}")

        # Fallback: derive from account
        if exchange_client:
            try:
                account = (
                    await exchange_client.get_account()
                    if hasattr(exchange_client, "get_account")
                    else None
                )
                if account:
                    balances = account.get("balances", [])
                    for b in balances:
                        symbol = b.get("asset")
                        free = float(b.get("free", 0))
                        locked = float(b.get("locked", 0))
                        if free > 0 or locked > 0:
                            wallet["by_symbol"][symbol] = free

                            # Sum USDT
                            if symbol == "USDT":
                                wallet["available_usdt"] = free
                                wallet["locked_usdt"] = locked
                                wallet["total_usdt"] = free + locked
            except Exception as e:
                logger.error(f"❌ Error getting wallet balance: {e}")

        return wallet


# ═════════════════════════════════════════════════════════════════════════════
# SITUATION_ENGINE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════


class SituationEngineImpl:
    """Real implementations for SituationEngine methods."""

    @staticmethod
    async def get_portfolio_snapshot(app_ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Get portfolio state from portfolio_manager (L3).
        """
        portfolio_manager = app_ctx.get("portfolio_manager")

        snapshot = {
            "nav_usdt": 0.0,
            "available_capital": 0.0,
            "locked_capital": 0.0,
            "active_positions": 0,
            "total_p_and_l": 0.0,
            "total_p_and_l_pct": 0.0,
            "timestamp": asyncio.get_event_loop().time(),
        }

        if not portfolio_manager:
            logger.debug("portfolio_manager not available — using balance fallbacks")
        else:
            try:
                # Call portfolio_manager methods
                if hasattr(portfolio_manager, "get_nav"):
                    snapshot["nav_usdt"] = await _maybe_await(portfolio_manager.get_nav())

                if hasattr(portfolio_manager, "get_positions"):
                    positions = await _maybe_await(portfolio_manager.get_positions())
                    snapshot["active_positions"] = len(positions) if positions else 0

                if hasattr(portfolio_manager, "get_pnl"):
                    pnl = await _maybe_await(portfolio_manager.get_pnl())
                    snapshot["total_p_and_l"] = pnl
                    if snapshot["nav_usdt"] > 0:
                        snapshot["total_p_and_l_pct"] = (pnl / snapshot["nav_usdt"]) * 100

                if hasattr(portfolio_manager, "get_capital_allocated"):
                    snapshot["locked_capital"] = await _maybe_await(
                        portfolio_manager.get_capital_allocated()
                    )

                if hasattr(portfolio_manager, "get_capital_available"):
                    snapshot["available_capital"] = await _maybe_await(
                        portfolio_manager.get_capital_available()
                    )

            except Exception as e:
                logger.error(f"❌ Error getting portfolio snapshot: {e}")

        # ── NAV fallback chain (production bridge): if portfolio_manager
        # didn't yield a NAV, fall through to balance_manager → shared_state.
        if snapshot["nav_usdt"] <= 0:
            balance_manager = app_ctx.get("balance_manager")
            if balance_manager is not None:
                nav_candidate = getattr(balance_manager, "last_nav", 0.0) or 0.0
                if nav_candidate > 0:
                    snapshot["nav_usdt"] = float(nav_candidate)

        if snapshot["nav_usdt"] <= 0:
            shared_state = app_ctx.get("shared_state")
            if shared_state is not None:
                nav_candidate = getattr(shared_state, "nav", 0.0) or 0.0
                if nav_candidate > 0:
                    snapshot["nav_usdt"] = float(nav_candidate)

        return snapshot

    @staticmethod
    async def get_all_signals(
        app_ctx: dict[str, Any], symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get signals from signal_manager (L5).

        For native signal engine (NativeSignalEngine), integrates with
        market_data to evaluate signals on-demand.
        """
        signal_manager = app_ctx.get("signal_manager")

        signals = []

        if not signal_manager:
            logger.warning("⚠️ signal_manager not available")
            return signals

        try:
            # Native signal engine optimization: evaluate with market_data
            if hasattr(signal_manager, "evaluate_with_market_data"):
                market_data = app_ctx.get("market_data_feed")
                symbols_to_eval = [symbol] if symbol else None
                signals = signal_manager.evaluate_with_market_data(market_data, symbols_to_eval)
            # Legacy signal_manager exposes sync get_all_signals() and
            # get_signals_for_symbol(); newer impls may be async — handle both.
            elif symbol and hasattr(signal_manager, "get_signals_for_symbol"):
                signals = await _maybe_await(signal_manager.get_signals_for_symbol(symbol))
            elif hasattr(signal_manager, "get_signals"):
                signals = await _maybe_await(signal_manager.get_signals(symbol))
            elif hasattr(signal_manager, "get_all_signals"):
                signals = await _maybe_await(signal_manager.get_all_signals())
                if symbol and signals:
                    signals = [s for s in signals if s.get("symbol") == symbol]

        except Exception as e:
            logger.error(f"❌ Error getting signals: {e}")

        return signals or []

    @staticmethod
    async def get_fused_signal(app_ctx: dict[str, Any], symbol: str) -> dict[str, Any] | None:
        """
        Get fused signal from signal_fusion (L5).
        """
        signal_fusion = app_ctx.get("signal_fusion")

        if not signal_fusion:
            logger.warning("⚠️ signal_fusion not available")
            return None

        try:
            if hasattr(signal_fusion, "fuse_signal"):
                fused = await signal_fusion.fuse_signal(symbol)
                return fused

        except Exception as e:
            logger.error(f"❌ Error fusing signal for {symbol}: {e}")

        return None

    @staticmethod
    async def get_market_regime(app_ctx: dict[str, Any]) -> dict[str, str]:
        """
        Get market regime from regime_detector (L2).
        """
        regime_detector = app_ctx.get("market_regime_detector")

        regime = {
            "volatility_regime": "NORMAL",
            "trend_regime": "RANGING",
            "nav_regime": "GROWTH",
            "overall_health": "OK",
        }

        if not regime_detector:
            logger.warning("⚠️ market_regime_detector not available")
            return regime

        try:
            if hasattr(regime_detector, "get_regime"):
                detected = await regime_detector.get_regime()
                regime.update(detected)

        except Exception as e:
            logger.error(f"❌ Error detecting regime: {e}")

        return regime


# ═════════════════════════════════════════════════════════════════════════════
# DECISION_ENGINE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════


class DecisionEngineImpl:
    """Real implementations for DecisionEngine methods."""

    @staticmethod
    async def get_current_mode(app_ctx: dict[str, Any]) -> str:
        """
        Get current trading mode from mode_manager (L5).
        """
        mode_manager = app_ctx.get("mode_manager")

        if not mode_manager:
            logger.warning("⚠️ mode_manager not available, defaulting to PROTECTIVE")
            return "PROTECTIVE"

        try:
            if hasattr(mode_manager, "get_current_mode"):
                mode = await mode_manager.get_current_mode()
                return mode
            elif hasattr(mode_manager, "mode"):
                return mode_manager.mode

        except Exception as e:
            logger.error(f"❌ Error getting mode: {e}")
            return "PROTECTIVE"

    @staticmethod
    async def evaluate_signal(
        app_ctx: dict[str, Any], symbol: str, signal_type: str, edge_score: float
    ) -> dict[str, Any]:
        """
        Evaluate signal through 6-layer arbitration gates.
        Uses arbitration_engine (L5).
        """
        arbitration_engine = app_ctx.get("arbitration_engine")

        result = {
            "passed": False,
            "gates_status": {},
            "blocking_gates": [],
            "reason": "",
        }

        if not arbitration_engine:
            logger.warning("⚠️ arbitration_engine not available; defaulting to pass=True for MVP")
            result["passed"] = True
            result["reason"] = "Arbitration engine unavailable; MVP default-pass"
            return result

        try:
            if hasattr(arbitration_engine, "evaluate"):
                result = await arbitration_engine.evaluate(symbol, signal_type, edge_score)

        except Exception as e:
            logger.error(f"❌ Error evaluating signal: {e}")
            result["reason"] = str(e)

        return result

    @staticmethod
    async def make_buy_decision(
        app_ctx: dict[str, Any], symbol: str, edge_score: float
    ) -> dict[str, Any] | None:
        """
        Make buy decision with capital allocation.
        Coordinates: arbitration_engine (L5), capital_allocator (L6).
        """
        # Step 1: Arbitrate
        arb_result = await DecisionEngineImpl.evaluate_signal(app_ctx, symbol, "BUY", edge_score)

        if not arb_result.get("passed"):
            logger.debug(f"🚫 BUY rejected for {symbol}: {arb_result.get('blocking_gates')}")
            return None

        # Step 2: Allocate capital
        capital_allocator = app_ctx.get("capital_allocator")
        quantity = 0.0

        if capital_allocator and hasattr(capital_allocator, "allocate_for_buy"):
            try:
                quantity = await capital_allocator.allocate_for_buy(symbol)
            except Exception as e:
                logger.warning(f"⚠️ Error allocating capital: {e}")

        # Step 3: Build decision
        decision = {
            "symbol": symbol,
            "action": "BUY",
            "quantity": quantity,
            "price_target": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": f"Signal edge: {edge_score:.3f}",
            "confidence": abs(edge_score),
            "timestamp": asyncio.get_event_loop().time(),
            "mode": await DecisionEngineImpl.get_current_mode(app_ctx),
        }

        logger.info(f"✅ BUY decision: {symbol} x{quantity:.4f}")
        return decision

    @staticmethod
    async def make_sell_decision(
        app_ctx: dict[str, Any], symbol: str, edge_score: float, source: str = "signal"
    ) -> dict[str, Any] | None:
        """
        Make sell decision for an open position — only if profitable after fees.
        Coordinates: arbitration_engine (L5), position_manager (L3), shared_state (L0).

        Args:
            app_ctx: Application context
            symbol: Trading pair
            edge_score: Signal confidence
            source: Where signal came from ("signal", "tp", "sl", etc.)

        Returns:
            SELL decision dict if position is profitable after fees, else None
        """
        # Step 1: Arbitrate
        arb_result = await DecisionEngineImpl.evaluate_signal(app_ctx, symbol, "SELL", edge_score)

        if not arb_result.get("passed"):
            logger.debug(f"🚫 SELL rejected for {symbol}: {arb_result.get('blocking_gates')}")
            return None

        # Step 2: Get position details
        position_manager = app_ctx.get("position_manager")
        quantity = 0.0
        entry_price = 0.0
        current_price = 0.0

        if position_manager and hasattr(position_manager, "get_position"):
            try:
                pos = await _maybe_await(position_manager.get_position(symbol))
                if pos is None:
                    logger.debug(f"⚠️ No open position for {symbol}")
                    return None

                quantity = float(getattr(pos, "qty", 0.0) or 0.0)
                entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
                current_price = float(getattr(pos, "mark_price", 0.0) or 0.0)
            except Exception as e:
                logger.warning(f"⚠️ Could not get position details: {e}")
                return None

        if quantity <= 0 or entry_price <= 0 or current_price <= 0:
            logger.debug(
                f"⚠️ Invalid position for {symbol}: qty={quantity}, entry={entry_price}, current={current_price}"
            )
            return None

        # Step 3: Calculate profit after fees
        # Binance trading fee: typically 0.1% (0.001) per side
        # Total round-trip: BUY fee + SELL fee = 0.2% (0.002)
        fee_pct = 0.002  # 0.2% round-trip fee

        # Unrealized P&L (before fees)
        pnl_before_fees = (current_price - entry_price) * quantity
        pnl_before_fees_pct = (
            ((current_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
        )

        # P&L after fees
        # Formula: (current_price * (1 - fee_pct) - entry_price * (1 + fee_pct)) * qty
        # Simplified: sell_value = qty * current_price * (1 - fee_pct)
        #            cost_basis = qty * entry_price * (1 + fee_pct)
        sell_value = quantity * current_price * (1.0 - fee_pct / 2.0)  # 0.1% sell fee
        cost_basis = quantity * entry_price * (1.0 + fee_pct / 2.0)  # 0.1% buy fee already paid
        pnl_after_fees = sell_value - cost_basis
        pnl_after_fees_pct = (pnl_after_fees / cost_basis * 100.0) if cost_basis > 0 else 0.0

        # Gate: Only sell if profitable after fees
        if pnl_after_fees <= 0:
            logger.debug(
                f"🚫 SELL skipped {symbol}: unprofitable after fees. "
                f"Entry={entry_price:.4f}, Current={current_price:.4f}, "
                f"P&L before fees: {pnl_before_fees_pct:.2f}%, "
                f"P&L after fees: {pnl_after_fees_pct:.2f}%"
            )
            return None

        # Step 4: Build SELL decision
        decision = {
            "symbol": symbol,
            "action": "SELL",
            "quantity": quantity,
            "price_target": current_price,
            "stop_loss": None,
            "take_profit": None,
            "reason": f"SELL {source} @ {current_price:.4f} (entry={entry_price:.4f}, "
            f"profit={pnl_after_fees:.2f} USDT, {pnl_after_fees_pct:.2f}% after fees)",
            "confidence": max(0.5, abs(edge_score)),  # At least 50% confidence since profitable
            "timestamp": asyncio.get_event_loop().time(),
            "mode": await DecisionEngineImpl.get_current_mode(app_ctx),
        }

        logger.info(
            f"✅ SELL decision: {symbol} x{quantity:.4f} @ {current_price:.4f} "
            f"(profit={pnl_after_fees:.2f} USDT, {pnl_after_fees_pct:.2f}% after fees)"
        )
        return decision


# ═════════════════════════════════════════════════════════════════════════════
# SAFE_EXECUTION_ENGINE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════


class SafeExecutionEngineImpl:
    """Real implementations for SafeExecutionEngine methods."""

    @staticmethod
    async def validate_order(
        _app_ctx: dict[str, Any],
        symbol: str,
        _action: str,
        quantity: float,
        price: float | None = None,
    ) -> dict[str, Any]:
        """
        Validate order with comprehensive checks.
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check 1: Symbol format
        if not symbol or not symbol.endswith("USDT"):
            validation["valid"] = False
            validation["errors"].append(f"Invalid symbol format: {symbol}")
            return validation

        # Check 2: Quantity
        if quantity <= 0:
            validation["valid"] = False
            validation["errors"].append(f"Quantity must be > 0, got: {quantity}")
            return validation

        # Check 3: Price
        if price is not None and price <= 0:
            validation["valid"] = False
            validation["errors"].append(f"Price must be > 0, got: {price}")
            return validation

        # Check 4: Notional floor
        if price is not None:
            notional = quantity * price
            min_notional = 10.0  # Binance minimum

            if notional < min_notional:
                validation["valid"] = False
                validation["errors"].append(
                    f"Notional {notional:.2f} USDT < minimum {min_notional} USDT"
                )

        return validation

    @staticmethod
    async def place_buy_order(
        app_ctx: dict[str, Any],
        symbol: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        """
        Place BUY order via execution_manager (L4).
        """
        # Validate
        validation = await SafeExecutionEngineImpl.validate_order(
            app_ctx, symbol, "BUY", quantity, price
        )

        result = {
            "success": False,
            "order_id": None,
            "symbol": symbol,
            "action": "BUY",
            "quantity": quantity,
            "filled_quantity": 0.0,
            "average_price": 0.0,
            "status": "FAILED",
            "error_message": None,
            "timestamp": asyncio.get_event_loop().time(),
        }

        if not validation["valid"]:
            result["error_message"] = "; ".join(validation["errors"])
            return result

        # Place order via exchange_client (native) or execution_manager (legacy)
        exchange_client = app_ctx.get("exchange_client")
        execution_manager = app_ctx.get("execution_manager")

        try:
            order = None

            # Try native exchange client first
            if exchange_client and hasattr(exchange_client, "place_order"):
                order = await exchange_client.place_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                )
            # Fallback to legacy execution_manager
            elif execution_manager and hasattr(execution_manager, "place_order"):
                order = await execution_manager.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    action="BUY",
                    order_type=order_type,
                )

            if order:
                result["success"] = True
                result["order_id"] = order.get("orderId")
                result["status"] = "FILLED"
                result["average_price"] = order.get("price", price or 0)
                result["filled_quantity"] = order.get("executedQty", quantity)
                logger.info(f"✅ BUY order placed: {symbol} x{quantity}")
            else:
                result["error_message"] = "No exchange client or execution manager available"
                logger.error(f"❌ No order executor available for BUY {symbol}")

        except Exception as e:
            result["error_message"] = str(e)
            logger.error(f"❌ Error placing BUY order: {e}")

        return result

    @staticmethod
    async def place_sell_order(
        app_ctx: dict[str, Any],
        symbol: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        """
        Place SELL order with FIX #2 guard.
        """
        # Validate
        validation = await SafeExecutionEngineImpl.validate_order(
            app_ctx, symbol, "SELL", quantity, price
        )

        result = {
            "success": False,
            "order_id": None,
            "symbol": symbol,
            "action": "SELL",
            "quantity": quantity,
            "filled_quantity": 0.0,
            "average_price": 0.0,
            "status": "FAILED",
            "error_message": None,
            "timestamp": asyncio.get_event_loop().time(),
        }

        if not validation["valid"]:
            result["error_message"] = "; ".join(validation["errors"])
            return result

        # FIX #2: Check idempotent guard
        bounded_cache = app_ctx.get("bounded_cache")
        order_id_key = f"{symbol}_{asyncio.get_event_loop().time()}"
        cache_key = f"sell_finalize_{symbol}_{order_id_key}"

        if bounded_cache and hasattr(bounded_cache, "get"):
            try:
                if await bounded_cache.get(cache_key):
                    logger.warning(f"⚠️ SELL already finalized for {symbol}, skipping")
                    result["status"] = "ALREADY_FINALIZED"
                    result["error_message"] = "Duplicate SELL prevented by FIX #2 guard"
                    return result
            except Exception as e:
                logger.warning(f"⚠️ FIX #2 guard check failed: {e}")

        # Place order via exchange_client (native) or execution_manager (legacy)
        exchange_client = app_ctx.get("exchange_client")
        execution_manager = app_ctx.get("execution_manager")

        try:
            order = None

            # Try native exchange client first
            if exchange_client and hasattr(exchange_client, "place_order"):
                order = await exchange_client.place_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                )
            # Fallback to legacy execution_manager
            elif execution_manager and hasattr(execution_manager, "place_order"):
                order = await execution_manager.place_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    action="SELL",
                    order_type=order_type,
                )

            if order:
                result["success"] = True
                result["order_id"] = order.get("orderId")
                result["status"] = "FILLED"
                result["average_price"] = order.get("price", price or 0)
                result["filled_quantity"] = order.get("executedQty", quantity)

                # Mark in FIX #2 cache
                if bounded_cache and hasattr(bounded_cache, "set"):
                    try:
                        await bounded_cache.set(cache_key, True, ttl=300)
                    except Exception as e:
                        logger.warning(f"⚠️ FIX #2 cache mark failed: {e}")

                logger.info(f"✅ SELL order placed: {symbol} x{quantity}")
            else:
                result["error_message"] = "No exchange client or execution manager available"
                logger.error(f"❌ No order executor available for SELL {symbol}")

        except Exception as e:
            result["error_message"] = str(e)
            logger.error(f"❌ Error placing SELL order: {e}")

        return result


# ═════════════════════════════════════════════════════════════════════════════
# OPERATIONS_ENGINE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════


class OperationsEngineImpl:
    """Real implementations for OperationsEngine methods."""

    @staticmethod
    async def startup_system(app_ctx: dict[str, Any]) -> bool:
        """
        Execute system startup (L0→L8).

        Tries native orchestrator first (Phase 8.2.8), falls back to legacy
        startup_orchestrator for backward compatibility.
        """
        # Try native orchestrator first (L8 — Phase 8.2.8)
        native_orch = app_ctx.get("_native_orchestrator")
        if native_orch and hasattr(native_orch, "start"):
            try:
                await native_orch.start()
                logger.info("✅ Native orchestrator started (market_data + balance_sync)")
                return True
            except Exception as e:
                logger.error(f"❌ Native orchestrator startup failed: {e}")
                return False

        # Fall back to legacy startup_orchestrator
        startup_orchestrator = app_ctx.get("startup_orchestrator")
        if not startup_orchestrator:
            logger.warning("⚠️ Neither native nor legacy orchestrator available")
            return False

        try:
            if hasattr(startup_orchestrator, "startup"):
                result = await startup_orchestrator.startup()
                return result

        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            return False

    @staticmethod
    async def get_health_report(app_ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Get health status from health_monitor (L7).
        """
        health_monitor = app_ctx.get("health_monitor")

        report = {
            "timestamp": asyncio.get_event_loop().time(),
            "overall_status": "OK",
            "components": {},
            "critical_issues": [],
            "warnings": [],
            "suggestions": [],
        }

        if not health_monitor:
            logger.warning("⚠️ health_monitor not available")
            return report

        try:
            if hasattr(health_monitor, "get_report"):
                report = await health_monitor.get_report()

        except Exception as e:
            logger.error(f"❌ Error getting health report: {e}")

        return report
