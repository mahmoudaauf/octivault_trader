# agents/signal_utils.py
import logging
import time
from typing import Optional, Dict, Any

def normalize_symbol(s: str) -> str:
    return (s or "").replace("/", "").upper()

async def emit_to_meta(
    meta,
    agent_name: str,
    symbol: str,
    *,
    action: str,
    confidence: float,
    quote: Optional[float] = None,
    quantity: Optional[float] = None,
    horizon_hours: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    sym = normalize_symbol(symbol)
    payload: Dict[str, Any] = dict(extra or {})
    payload["action"] = (action or "").upper()
    payload["confidence"] = float(confidence)
    if quote is not None:
        payload["quote"] = float(quote)
    if quantity is not None:
        payload["quantity"] = float(quantity)
    if horizon_hours is not None:
        payload["horizon_hours"] = float(horizon_hours)

    # Prefer shim if it exists; fall back to receive_signal
    try:
        if hasattr(meta, "submit_signal"):
            await meta.submit_signal(agent_name, sym, payload, payload.get("confidence"))
        else:
            await meta.receive_signal(agent_name, sym, payload)
        logging.info(
            f"[Agent:{agent_name}] Signal {sym} {payload['action']} "
            f"conf={payload['confidence']:.2f}"
            + (f" quote={payload.get('quote'):.2f}" if 'quote' in payload else "")
            + (f" qty={payload.get('quantity'):.6f}" if 'quantity' in payload else "")
        )
    except Exception as e:
        logging.warning(f"[Agent:{agent_name}] Submit failed for {sym}: {e}", exc_info=True)

async def is_fresh(shared_state, symbol: str, max_age_sec: float = 120.0) -> bool:
    """Lightweight freshness guard before emitting a signal."""
    try:
        if hasattr(shared_state, "is_market_data_ready"):
            ready = await shared_state.is_market_data_ready()
            if not ready:
                return False
        # Optional: if you expose last-bar timestamps
        if hasattr(shared_state, "get_last_bar_ts"):
            sym = normalize_symbol(symbol)
            ts = await shared_state.get_last_bar_ts(sym, timeframe="1m")
            if ts is None:
                return False
            return (time.time() - float(ts)) <= max_age_sec
    except Exception:
        return False
    return True
