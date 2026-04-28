import asyncio
from typing import Optional, Any


def _maybe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


async def post_exit_bookkeeping(shared_state, config, logger, symbol: str, exit_reason: str, source: str) -> None:
    """Record exit reason and apply standard post-exit cooldown."""
    try:
        if hasattr(shared_state, "record_exit_reason"):
            shared_state.record_exit_reason(symbol, exit_reason, source=source)
    except Exception as exc:
        try:
            logger.debug("[ExitBookkeeping] record_exit_reason failed for %s: %s", symbol, exc)
        except Exception:
            pass

    try:
        if hasattr(shared_state, "set_cooldown"):
            cooldown_sec = _maybe_float(getattr(config, "META_DECISION_COOLDOWN_SEC", 15), 15)
            res = shared_state.set_cooldown(symbol, cooldown_sec)
            if asyncio.iscoroutine(res):
                await res
    except Exception as exc:
        try:
            logger.debug("[ExitBookkeeping] set_cooldown failed for %s: %s", symbol, exc)
        except Exception:
            pass
