# core/diagnostics/system_summary.py
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("SystemSummary")
logger.setLevel(logging.INFO)

# ---------- helpers ----------

async def _maybe_await(value):
    """Await if coroutine/function, else return value directly."""
    if asyncio.iscoroutine(value):
        return await value
    if callable(value):
        result = value()
        if asyncio.iscoroutine(result):
            return await result
        return result
    return value

def _extract_open_symbols(open_trades):
    """
    Accepts:
      - dict: {symbol: ...}
      - list: [trade_obj_or_dict, ...]
      - None / other
    Returns:
      - list[str] symbols (may be empty)
    """
    try:
        if open_trades is None:
            return []
        if isinstance(open_trades, dict):
            return list(open_trades.keys())
        if isinstance(open_trades, list):
            syms = []
            for t in open_trades:
                if isinstance(t, dict):
                    sym = t.get("symbol") or t.get("pair") or t.get("asset")
                else:
                    sym = (
                        getattr(t, "symbol", None)
                        or getattr(t, "pair", None)
                        or getattr(t, "asset", None)
                    )
                if sym:
                    syms.append(sym)
            return syms
        # fallback to stringified single entry
        return [str(open_trades)]
    except Exception:
        return []

async def _get_open_trades(shared_state):
    """
    Try multiple shapes:
      - shared_state.get_all_open_trades() (sync or async)
      - shared_state.get_open_trades_snapshot() (sync or async)
      - shared_state.open_trades (attr)
    """
    # 1) get_all_open_trades
    getter = getattr(shared_state, "get_all_open_trades", None)
    if getter is not None:
        return await _maybe_await(getter)

    # 2) get_open_trades_snapshot
    getter = getattr(shared_state, "get_open_trades_snapshot", None)
    if getter is not None:
        return await _maybe_await(getter)

    # 3) raw attr
    if hasattr(shared_state, "open_trades"):
        return getattr(shared_state, "open_trades")

    return None

async def _get_total_equity(shared_state):
    # Prefer method if exists
    getter = getattr(shared_state, "get_total_equity", None)
    if getter is not None:
        v = await _maybe_await(getter)
        if v is not None:
            return float(v)
    # Fallback to attribute
    return float(getattr(shared_state, "total_equity", 0.0))

async def _get_trade_counts(shared_state):
    v = getattr(shared_state, "trade_counts", None)
    if v is None:
        getter = getattr(shared_state, "get_trade_counts", None)
        if getter is not None:
            v = await _maybe_await(getter)
    return v or {}

async def _get_total_pnl(shared_state):
    getter = getattr(shared_state, "get_total_pnl", None)
    if getter is not None:
        return await _maybe_await(getter)
    # Could be attribute
    if hasattr(shared_state, "total_pnl"):
        return getattr(shared_state, "total_pnl")
    return "N/A"

async def _get_realized_pnl(shared_state):
    """
    Try to read realized PnL (cumulative).
    """
    # Prefer method if exists
    getter = getattr(shared_state, "get_realized_pnl", None)
    if getter is not None:
        v = await _maybe_await(getter)
        if v is not None:
            return float(v)
    # Fallback to attribute
    return float(getattr(shared_state, "realized_pnl", 0.0))

async def _get_realized_pnl_60m(shared_state):
    """
    Prefer rolling computation if available; otherwise try attribute `pnl_realized_60m`.
    """
    # Preferred: rolling accessor used by ProfitTargetGuard
    rolling = getattr(shared_state, "get_rolling_realized_pnl", None)
    if rolling is not None:
        try:
            v = await _maybe_await(lambda: rolling(minutes=60))
            return float(v or 0.0)
        except Exception:
            pass
    # Fallback: attribute that may be maintained elsewhere
    return float(getattr(shared_state, "pnl_realized_60m", 0.0))

# ---------- main task ----------

async def system_summary_logger(shared_state, config, interval=60):
    """
    Asynchronously logs a summary of the system's state at a given interval.

    Args:
        shared_state: central state (sync/async getters supported)
        config: reserved for future use
        interval: seconds between summaries
    """
    while True:
        try:
            now = datetime.utcnow().isoformat()

            # Gather
            open_trades = await _get_open_trades(shared_state)
            open_symbols = _extract_open_symbols(open_trades)
            open_trade_count = len(open_symbols)

            total_equity = await _get_total_equity(shared_state)
            realized_pnl = await _get_realized_pnl(shared_state)
            pnl_60m = await _get_realized_pnl_60m(shared_state)
            pnl_60m_per_hour = pnl_60m  # 60m window = hourly

            trade_counts = await _get_trade_counts(shared_state)
            sorted_agents = sorted(trade_counts.items(), key=lambda x: -x[1]) if trade_counts else []

            total_pnl = await _get_total_pnl(shared_state)

            # Log
            logger.info("\n\n📊 [System Summary] @ %s", now)
            logger.info("----------------------------------------------------")
            logger.info("📈 Open Trades (%d): %s", open_trade_count, open_symbols)
            logger.info(
                "📊 SystemSummary | equity=%.2f | realized=%.2f | 60m_realized=%.2f USDT/h",
                float(total_equity), float(realized_pnl), float(pnl_60m_per_hour)
            )
            logger.info("💹 Total PnL: %s", total_pnl)
            logger.info("🧠 Agent Trade Counts:")
            for agent, count in sorted_agents:
                logger.info("    - %s: %d trades", agent, count)
            logger.info("----------------------------------------------------\n")

        except Exception as e:
            logger.warning("⚠️ System Summary Logger failed: %s", e, exc_info=True)

        await asyncio.sleep(interval)
