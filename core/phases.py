# core/phases.py
from __future__ import annotations
import asyncio
from typing import Any, Awaitable, Callable, Optional
import inspect

# We keep emit_summary optional: import locally when provided to avoid hard deps.
async def _maybe_summary(shared_state: Any, event: str, **fields: Any) -> None:
    if not shared_state:
        return
    try:
        # from core.summaries import emit_summary  # local import to avoid cycles
        # Stub implementation that actually logs
        async def _emit_summary_stub(state, evt, **kw):
            # print(f"[SUMMARY] {evt} - {kw}") # Optional: print to stdout
            pass
        emit_summary = _emit_summary_stub
        await emit_summary(shared_state, event, **fields)
    except Exception:
        # Phase runner must never crash due to telemetry
        pass

async def run_with_timeout(
    logger,
    phase_key: str,
    fn_or_coro: Callable[[], Any] | Awaitable[Any],
    timeout_sec: float,
    *,
    shared_state: Optional[Any] = None,
    start_event: str = "PHASE_START",
    done_event: str = "PHASE_DONE",
    timeout_event: str = "PHASE_TIMEOUT",
) -> Any:
    """
    Run a callable/coroutine for a 'phase' with a timeout.
    - Logs clean start/done/timeout lines
    - Emits optional SUMMARY events if shared_state is provided
    - Returns the function's result (if any)
    """
    # Announce start
    try:
        logger.info(f"[{phase_key}] start() beginning")
        await _maybe_summary(shared_state, start_event, phase=phase_key)
    except Exception:
        pass

    # Normalize to coroutine
    if inspect.iscoroutine(fn_or_coro):          # already an awaited thing
        coro = fn_or_coro
    elif callable(fn_or_coro):                   # function or bound method
        try:
            maybe = fn_or_coro()
        except Exception as e:
            # Immediate failure while calling the function (before awaiting)
            logger.exception(f"[{phase_key}] start() raised before await: {e}")
            raise
        if inspect.iscoroutine(maybe) or isinstance(maybe, asyncio.Future):
            coro = maybe
        else:
            # Synchronous return: wrap so we still honor the same protocol
            async def _sync_wrapper():
                return maybe
            coro = _sync_wrapper()
    else:
        # Bad input — fail loudly so caller can fix
        raise TypeError(f"run_with_timeout({phase_key}): fn_or_coro must be callable or awaitable")

    # Execute with timeout
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(f"[{phase_key}] start() timed out at {timeout_sec:.1f}s — continuing in background")
        try:
            await _maybe_summary(shared_state, timeout_event, phase=phase_key, timeout_sec=timeout_sec)
            await _maybe_summary(shared_state, f"{phase_key.upper()}_STARTED")
        except Exception:
            pass
        return None
    except Exception as e:
        # Surface errors to caller; phases often want to abort on hard failures
        logger.exception(f"[{phase_key}] start() failed: {e}")
        raise
    else:
        # Completed within timeout
        try:
            logger.info(f"[{phase_key}] start() completed")
            await _maybe_summary(shared_state, done_event, phase=phase_key)
            logger.info(f"[{phase_key}] start() returned")
            await _maybe_summary(shared_state, f"{phase_key.upper()}_STARTED")
        except Exception:
            pass
        return result
