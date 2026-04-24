"""
Unified holding utility scoring for portfolio authorities.

This module provides a single, reusable utility model so rotation/recycle
decisions are based on the same notion of "position usefulness":
forward edge, liquidity, pnl efficiency, age decay, tradability, and
opportunity-cost pressure.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from utils.shared_state_tools import fee_bps


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _norm_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").upper()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return float(default)


def _infer_liquidity_component(
    symbol: str,
    position: Dict[str, Any],
    shared_state: Any = None,
) -> float:
    """
    Returns liquidity score in [0, 1].
    Prefers explicit position fields, then shared-state symbol metadata.
    """
    pos = position if isinstance(position, dict) else {}
    for key in (
        "liquidity_score",
        "market_liquidity",
        "liquidity",
        "_liquidity_score",
    ):
        if key in pos and pos.get(key) is not None:
            return _clamp(_safe_float(pos.get(key), 0.5))

    quote_volume = 0.0
    spread = 0.0
    if shared_state is not None:
        try:
            sym = _norm_symbol(symbol)
            accepted = getattr(shared_state, "accepted_symbols", {}) or {}
            meta = accepted.get(sym, {}) if isinstance(accepted, dict) else {}
            quote_volume = _safe_float(
                meta.get("quoteVolume")
                or meta.get("quote_volume")
                or meta.get("quote_volume_24h")
                or meta.get("volume_quote")
            )
            spread = _safe_float(meta.get("spread") or meta.get("spread_pct"))
        except Exception:
            quote_volume = 0.0
            spread = 0.0

    if quote_volume > 0:
        # Practical 24h quote-volume normalization:
        #  100k USDT ~= full liquidity score (cap at 1.0)
        vol_score = _clamp(quote_volume / 100000.0)
        spread_penalty = _clamp(spread / 0.05)  # 5% spread => full penalty
        return _clamp(vol_score * (1.0 - spread_penalty))
    return 0.5


def _infer_forward_edge_component(position: Dict[str, Any]) -> float:
    pos = position if isinstance(position, dict) else {}
    for key in (
        "continuation_score",
        "continuation_strength",
        "signal_strength",
        "trend_strength",
        "_continuation_confidence",
        "confidence",
    ):
        if key in pos and pos.get(key) is not None:
            return _clamp(_safe_float(pos.get(key), 0.5))
    return 0.5


def _infer_position_value_usdt(symbol: str, position: Dict[str, Any], shared_state: Any = None) -> float:
    pos = position if isinstance(position, dict) else {}
    value = _safe_float(pos.get("value_usdt"), 0.0)
    if value > 0:
        return value

    qty = _safe_float(pos.get("quantity") or pos.get("qty"), 0.0)
    if qty <= 0:
        return 0.0

    px = _safe_float(
        pos.get("mark_price")
        or pos.get("avg_price")
        or pos.get("entry_price")
        or pos.get("price"),
        0.0,
    )
    if px <= 0 and shared_state is not None:
        try:
            prices = getattr(shared_state, "latest_prices", {}) or {}
            px = _safe_float((prices.get(_norm_symbol(symbol)) if isinstance(prices, dict) else 0.0), 0.0)
        except Exception:
            px = 0.0
    return qty * px if px > 0 else 0.0


def _round_trip_fee_pct(shared_state: Any = None, config: Any = None) -> float:
    try:
        taker_bps = float(fee_bps(shared_state, "taker") or 10.0)
    except Exception:
        taker_bps = 10.0
    slippage_bps = 0.0
    try:
        if config is not None:
            slippage_bps = _safe_float(
                getattr(config, "EXIT_SLIPPAGE_BPS", getattr(config, "CR_PRICE_SLIPPAGE_BPS", 0.0))
            )
    except Exception:
        slippage_bps = 0.0
    return ((taker_bps * 2.0) + slippage_bps) / 10000.0


def compute_holding_utility(
    symbol: str,
    position: Dict[str, Any],
    *,
    best_opp_score: float = 0.0,
    shared_state: Any = None,
    config: Any = None,
    now_ts: float = 0.0,
) -> Dict[str, float]:
    """
    Compute unified utility and rotation pressure for a held position.

    Returns dict with:
      utility (0..1), rotation_pressure (higher => more exit-worthy), and components.
    """
    pos = position if isinstance(position, dict) else {}
    now = float(now_ts or time.time())

    entry_ts = _safe_float(pos.get("entry_time") or pos.get("opened_at"), now)
    age_sec = max(0.0, now - entry_ts)
    age_hours = age_sec / 3600.0

    max_hold_sec = 1800.0
    stale_mult = 4.0
    target_pnl_pct = 0.03
    opportunity_scale = 0.05
    min_significant = 25.0
    if config is not None:
        max_hold_sec = _safe_float(getattr(config, "MAX_HOLD_SEC", max_hold_sec), max_hold_sec)
        stale_mult = _safe_float(getattr(config, "HOLDING_UTILITY_STALE_MULT", stale_mult), stale_mult)
        target_pnl_pct = _safe_float(
            getattr(config, "HOLDING_UTILITY_TARGET_PNL_PCT", target_pnl_pct),
            target_pnl_pct,
        )
        opportunity_scale = _safe_float(
            getattr(config, "HOLDING_UTILITY_OPPORTUNITY_SCALE", opportunity_scale),
            opportunity_scale,
        )
        min_significant = _safe_float(
            getattr(config, "MIN_SIGNIFICANT_POSITION_USDT", min_significant),
            min_significant,
        )

    forward_edge_component = _infer_forward_edge_component(pos)
    liquidity_component = _infer_liquidity_component(symbol, pos, shared_state=shared_state)

    gross_pnl_pct = _safe_float(pos.get("unrealized_pnl_pct"), 0.0)
    net_pnl_pct = gross_pnl_pct - _round_trip_fee_pct(shared_state=shared_state, config=config)
    # Map [-target, +target] -> [0, 1]
    pnl_eff_component = _clamp((net_pnl_pct + target_pnl_pct) / max(2.0 * target_pnl_pct, 1e-9))

    value_usdt = _infer_position_value_usdt(symbol, pos, shared_state=shared_state)
    floor_usdt = _safe_float(pos.get("significant_floor_usdt"), min_significant)
    if floor_usdt <= 0:
        floor_usdt = min_significant
    tradability_component = _clamp(value_usdt / max(floor_usdt, 1e-9))

    max_useful_age_sec = max_hold_sec * max(stale_mult, 1.0)
    age_component = _clamp(1.0 - (age_sec / max(max_useful_age_sec, 1.0)))

    # Opportunity-cost pressure: if best opportunity materially exceeds held profile,
    # utility should decline.
    held_expectancy = (forward_edge_component * 0.6) + (_clamp((net_pnl_pct + 0.02) / 0.04) * 0.4)
    opportunity_cost = max(0.0, float(best_opp_score or 0.0) - held_expectancy)
    opportunity_penalty = _clamp(opportunity_cost / max(opportunity_scale, 1e-9))

    # Utility weighting (sums to 1.0 before penalty).
    utility_pre_penalty = (
        (0.30 * forward_edge_component)
        + (0.25 * pnl_eff_component)
        + (0.15 * liquidity_component)
        + (0.15 * tradability_component)
        + (0.15 * age_component)
    )
    utility = _clamp(utility_pre_penalty - (0.25 * opportunity_penalty))

    rotation_pressure = _clamp((1.0 - utility) + (0.20 * opportunity_penalty), 0.0, 1.5)

    return {
        "utility": float(utility),
        "rotation_pressure": float(rotation_pressure),
        "forward_edge_component": float(forward_edge_component),
        "pnl_efficiency_component": float(pnl_eff_component),
        "liquidity_component": float(liquidity_component),
        "tradability_component": float(tradability_component),
        "age_component": float(age_component),
        "opportunity_penalty": float(opportunity_penalty),
        "opportunity_cost": float(opportunity_cost),
        "net_pnl_pct": float(net_pnl_pct),
        "gross_pnl_pct": float(gross_pnl_pct),
        "age_hours": float(age_hours),
        "value_usdt": float(value_usdt),
    }
