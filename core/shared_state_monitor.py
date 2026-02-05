# core/shared_state_monitor.py
import asyncio
import logging

async def shared_state_monitor(shared_state, interval: int = 60):
    logger = logging.getLogger("SharedStateMonitor")
    while True:
        try:
            # Prefer lock-free snapshots for hot paths
            try:
                symbols = shared_state.get_accepted_symbols_snapshot()
            except AttributeError:
                # Fallback to async API if snapshot isn't available
                symbols = await shared_state.get_accepted_symbols()

            sym_count = len(symbols) if symbols else 0

            # PnL
            try:
                pnl = shared_state.get_total_pnl()
            except AttributeError:
                pnl = float(getattr(shared_state, "realized_pnl", 0.0)) + float(getattr(shared_state, "unrealized_pnl", 0.0))

            # Equity & ROI (safe fallbacks)
            try:
                equity = shared_state.get_total_equity()
            except Exception:
                equity = 0.0

            try:
                roi = shared_state.get_roi()
            except Exception:
                roi = 0.0

            # Trades (use your new alias if present)
            try:
                trade_count = shared_state.get_total_trade_count()
            except Exception:
                # fallback to whatever is available
                trade_count = int(getattr(shared_state, "trade_count", 0)) \
                              or len(getattr(shared_state, "active_trades", {}) or {}) \
                              + len(getattr(shared_state, "closed_trades", {}) or {})

            logger.info(
                f"📈 SharedState: symbols={sym_count} | trades={trade_count} | "
                f"PnL={pnl:.2f} | equity={equity:.2f} | ROI={roi:.2f}%"
            )

        except Exception as e:
            logger.exception(f"⚠️ SharedStateMonitor error: {e}")

        await asyncio.sleep(interval)
