import logging
import asyncio
from typing import Any, Dict, List, Optional

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
    Periodically deploys idle quote balance (e.g., USDT) across a small basket
    of active symbols using quote-based market buys via ExecutionManager.
    """

    def __init__(
        self,
        shared_state: Any,
        exchange_client: Any,
        config: Any,
        execution_manager: Any,
        **kwargs
    ):
        if exchange_client is None:
            raise ValueError("exchange_client must not be None")
        if execution_manager is None:
            raise ValueError("execution_manager must not be None")

        self.shared_state = shared_state
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

    # ---------- helpers ----------

    async def _get_free_quote(self) -> float:
        """
        Robustly read free quote balance (USDT by default) from client or shared_state.
        """
        # Prefer a direct single-asset getter if present
        try:
            if hasattr(self.exchange_client, "get_account_balance"):
                bal = await self.exchange_client.get_account_balance(self.base_currency)
                if isinstance(bal, dict):
                    return float(bal.get("free", 0.0))
                return float(bal or 0.0)
        except Exception:
            logger.debug("get_account_balance failed", exc_info=True)

        # Fallback to map of balances
        try:
            if hasattr(self.exchange_client, "get_account_balances"):
                wallet = await self.exchange_client.get_account_balances()
                if isinstance(wallet, dict):
                    v = wallet.get(self.base_currency)
                    if isinstance(v, dict):
                        return float(v.get("free", 0.0))
                    return float(v or 0.0)
        except Exception:
            logger.debug("get_account_balances failed", exc_info=True)

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

    def _pick_symbols(self) -> List[str]:
        """
        Choose up to max_symbols from active/accepted symbols, preferring highest score if available.
        Ensures USDT quote and filters out obvious non-tradables.
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

        return syms[: self.max_symbols]

    async def _check_and_compound(self) -> None:
        # --- Circuit Breaker Invariant ---
        if hasattr(self.shared_state, "is_circuit_breaker_open") and await self.shared_state.is_circuit_breaker_open():
            logger.warning("🛑 Compounding frozen: Circuit Breaker is OPEN.")
            return

        # --- Profit Lock Invariant: Only compound if we have realized profit ---
        realized_pnl = float(self.shared_state.metrics.get("realized_pnl", 0.0))
        if realized_pnl <= 0:
            logger.debug("Compounding skipped: No realized PnL (%.2f).", realized_pnl)
            return

        free_balance = await self._get_free_quote()
        logger.debug("🔎 Available %s balance: %.2f", self.base_currency, free_balance)

        reserve = float(self._cfg("COMPOUNDING_RESERVE_USDT", 25.0))
        spendable = max(0.0, free_balance - reserve)

        if spendable <= self.min_compound_threshold:
            logger.debug("No compounding: spendable %.2f (after %s reserve) below threshold %.2f.", 
                         spendable, reserve, self.min_compound_threshold)
            return

        logger.info("📈 Compounding opportunity: %.2f %s available (Spendable after reserve: %.2f)", 
                    free_balance, self.base_currency, spendable)
        await self._execute_compounding_strategy(spendable)

    async def _execute_compounding_strategy(self, amount: float) -> None:
        """
        Spend quote by using ExecutionManager's quote-order path (planned_quote).
        This avoids manual lot-size rounding and minNotional guesswork.
        """
        symbols = self._pick_symbols()
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

        logger.info("📊 Selected symbols: %s", symbols)
        logger.info("💸 Target allocation per symbol: %.2f %s", per, self.base_currency)

        spent_total = 0.0

        for symbol in symbols:
            # Refresh free balance each iteration to avoid overspend
            free_quote = await self._get_free_quote()
            remaining = max(0.0, amount - spent_total)
            if remaining < self.min_compound_threshold or free_quote < self.min_compound_threshold:
                logger.info("Stopping compounding: remaining=%.2f, free=%.2f", remaining, free_quote)
                break

            planned = min(per, remaining, free_quote)

            try:
                # Quick affordability + minNotional pre-check via EM
                # Using 'buy' action directly as this module always compounds by buying base_currency
                ok, gap, why = await self.execution_manager.can_afford_market_buy(symbol, planned)
                if not ok:
                    logger.info("⏭️ Skipping %s: cannot afford planned %.2f (%s, gap=%.2f).", symbol, planned, why, gap)
                    continue

                # Place a BUY using quote route
                res = await self.execution_manager.execute_trade(
                    symbol=symbol,
                    side="buy", # Changed from 'action' to 'side' as per the new code
                    planned_quote=planned,
                    tag="meta/CompoundingEngine",
                )

                status = (res or {}).get("status", "unknown")
                if status in {"executed", "filled", "placed"}:
                    # Use executed_qty * price when available; otherwise trust 'planned'
                    spent_total += planned
                    logger.info("✅ Compounded into %s with %.2f %s (status=%s).", symbol, planned, self.base_currency, status)
                    # Optional: refresh balances after each fill to keep loop precise
                    if hasattr(self.exchange_client, "refresh_balances"):
                        try:
                            await self.exchange_client.refresh_balances()
                        except Exception:
                            logger.debug("refresh_balances failed", exc_info=True)
                elif status == "skipped":
                    logger.info("⏭️ Skipped %s (reason=%s).", symbol, (res or {}).get("reason"))
                else:
                    logger.warning("⚠️ Compounding result for %s unclear: %s", symbol, res)

            except BinanceAPIException as api_error:
                logger.error("Binance API error during compounding %s: %s", symbol, api_error)
            except _NetError as net_error:
                logger.error("Network error during compounding %s: %s", symbol, net_error)
            except RetryError as retry_error:
                logger.error("RetryError placing order for %s: %s", symbol, retry_error)
            except Exception as unknown_error:
                logger.exception("Unexpected error placing order for %s: %s", symbol, unknown_error)

        if spent_total > 0:
            self.just_ran = True

        logger.info("🧮 Compounding cycle finished. Spent ≈ %.2f %s.", spent_total, self.base_currency)
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
