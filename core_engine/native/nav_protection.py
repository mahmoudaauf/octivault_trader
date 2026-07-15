from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NAVAttribution:
    current_nav_usdt: float
    previous_nav_usdt: float
    peak_nav_usdt: float
    nav_delta_usdt: float
    nav_delta_pct: float
    realized_pnl_delta_usdt: float
    unrealized_pnl_delta_usdt: float
    deposit_withdrawal_delta_usdt: float
    fee_delta_usdt: float
    market_beta_estimate_usdt: float
    alpha_estimate_usdt: float
    largest_contributor_symbol: str
    largest_contributor_delta_usdt: float
    concentration_ratio: float
    attribution_type: str
    confidence: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class NAVProtectionState:
    protection_mode: str
    peak_nav_usdt: float
    protection_floor_usdt: float
    drawdown_from_peak_pct: float
    locked_profit_usdt: float
    available_profit_to_risk_usdt: float
    allow_buy: bool
    allow_sell: bool
    allow_tp_sl_adjustment: bool
    suggested_size_multiplier: float
    suggested_confidence_floor_delta: float
    suggested_reserve_floor_ratio: float
    suggested_actions: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NAVAttributionEngine:
    def __init__(self, *, concentration_gain_trim_threshold: float = 0.50) -> None:
        self.concentration_gain_trim_threshold = max(
            0.0, min(1.0, float(concentration_gain_trim_threshold))
        )

    def evaluate(self, shared_state: Any) -> NAVAttribution:
        current_nav = float(getattr(shared_state, "nav_usdt", 0.0) or 0.0)
        previous_nav = float(getattr(shared_state, "previous_nav_usdt", 0.0) or 0.0)
        # Cap peak_nav to session_anchor_nav — prevents prior session highs from
        # dominating drawdown calculations in a new session.
        _raw_peak = max(
            float(getattr(shared_state, "peak_nav_usdt", 0.0) or 0.0),
            float(getattr(shared_state, "metrics", {}).get("peak_nav", 0.0) or 0.0),
            current_nav,
        )
        _session_anchor_for_peak = float(getattr(shared_state, "session_anchor_nav", 0.0) or 0.0)
        peak_nav = min(_raw_peak, _session_anchor_for_peak) if _session_anchor_for_peak > 0 else _raw_peak
        peak_nav = max(peak_nav, current_nav)
        metrics = getattr(shared_state, "metrics", {}) or {}
        current_realized = float(metrics.get("realized_pnl", 0.0) or 0.0)
        current_unrealized = float(metrics.get("unrealized_pnl", 0.0) or 0.0)
        prev_attr = getattr(shared_state, "last_nav_attribution", {}) or {}
        prev_realized = float(prev_attr.get("realized_pnl_total_usdt", current_realized) or 0.0)
        prev_free = float(
            prev_attr.get("free_usdt", getattr(shared_state, "previous_nav_usdt", 0.0) or 0.0)
            or 0.0
        )
        current_free = float(getattr(shared_state, "free_balance_usdt", 0.0) or 0.0)
        nav_delta = current_nav - previous_nav
        nav_delta_pct = (nav_delta / previous_nav) if previous_nav > 0 else 0.0
        realized_delta = current_realized - prev_realized
        if "unrealized_pnl_total_usdt" in prev_attr:
            prev_unrealized = float(prev_attr.get("unrealized_pnl_total_usdt", current_unrealized) or 0.0)
        else:
            inferred_unrealized_delta = nav_delta - realized_delta
            if abs(current_free - prev_free) > max(5.0, abs(nav_delta) * 0.5):
                inferred_unrealized_delta = current_unrealized
            prev_unrealized = max(0.0, current_unrealized - inferred_unrealized_delta)
        unrealized_delta = current_unrealized - prev_unrealized
        deposit_delta = 0.0
        fee_delta = 0.0

        contributor_symbol = ""
        contributor_delta = 0.0
        total_positive_contrib = 0.0
        positive_count = 0
        if hasattr(shared_state, "get_all_positions"):
            positions = shared_state.get_all_positions() or {}
        else:
            positions = getattr(shared_state, "positions", {}) or {}
        for symbol, pos in positions.items():
            qty = _pos_float(pos, "qty")
            mark = _pos_float(pos, "mark_price", "current_price", "avg_price")
            entry = _pos_float(pos, "entry_price", "avg_price")
            if qty <= 0 or mark <= 0 or entry <= 0:
                continue
            delta = qty * (mark - entry)
            if delta > 0:
                total_positive_contrib += delta
                positive_count += 1
            if abs(delta) > abs(contributor_delta):
                contributor_symbol = str(symbol or "")
                contributor_delta = delta
        concentration_ratio = (
            min(1.0, abs(contributor_delta) / max(abs(unrealized_delta), 1e-9))
            if abs(unrealized_delta) > 0
            else 0.0
        )
        market_beta_estimate = max(0.0, unrealized_delta) if positive_count > 1 else 0.0
        alpha_estimate = nav_delta - market_beta_estimate - deposit_delta - fee_delta

        attribution_type = "UNKNOWN"
        confidence = "UNKNOWN"
        reason = "insufficient_nav_history"

        if previous_nav <= 0 or current_nav <= 0:
            attribution_type = "UNKNOWN"
            confidence = "LOW"
            reason = "missing_current_or_previous_nav"
        elif abs(nav_delta) <= 1e-9 and abs(realized_delta) <= 1e-9 and abs(unrealized_delta) <= 1e-9:
            attribution_type = "UNKNOWN"
            confidence = "LOW"
            reason = "no_material_nav_change"
        elif (
            abs(nav_delta - realized_delta - unrealized_delta) > max(5.0, current_nav * 0.10)
            and abs(current_free - prev_free) > max(5.0, abs(nav_delta) * 0.5)
        ):
            deposit_delta = nav_delta - realized_delta - unrealized_delta
            attribution_type = "DEPOSIT_WITHDRAWAL"
            confidence = "MEDIUM"
            reason = "nav_change_unexplained_by_trade_pnl"
        elif nav_delta < 0 and current_nav < peak_nav:
            attribution_type = "NAV_DECAY"
            confidence = "HIGH"
            reason = "nav_below_peak"
        elif realized_delta > 0 and nav_delta > 0:
            attribution_type = "REALIZED_PROFIT"
            confidence = "HIGH"
            reason = "realized_pnl_increased_with_nav"
        elif contributor_symbol and concentration_ratio > self.concentration_gain_trim_threshold and nav_delta > 0:
            attribution_type = "CONCENTRATION_GAIN"
            confidence = "MEDIUM" if unrealized_delta > 0 else "LOW"
            reason = f"single_symbol_drove_nav:{contributor_symbol}"
        elif unrealized_delta > 0 and nav_delta > 0:
            if positive_count > 1:
                attribution_type = "UNREALIZED_MARKET_BETA"
                confidence = "MEDIUM"
                reason = "multiple_holdings_contributed_to_floating_gain"
            else:
                attribution_type = "UNREALIZED_ALPHA"
                confidence = "LOW"
                reason = "single_position_floating_gain_without_realized_profit"
        elif nav_delta < 0 and unrealized_delta < 0 and abs(fee_delta) >= abs(nav_delta) * 0.5:
            attribution_type = "FEE_DRAG"
            confidence = "LOW"
            reason = "fee_drag_dominates_nav_change"
        elif abs(realized_delta) > 0 or abs(unrealized_delta) > 0:
            attribution_type = "MIXED"
            confidence = "LOW"
            reason = "mixed_realized_and_unrealized_change"

        return NAVAttribution(
            current_nav_usdt=current_nav,
            previous_nav_usdt=previous_nav,
            peak_nav_usdt=peak_nav,
            nav_delta_usdt=nav_delta,
            nav_delta_pct=nav_delta_pct,
            realized_pnl_delta_usdt=realized_delta,
            unrealized_pnl_delta_usdt=unrealized_delta,
            deposit_withdrawal_delta_usdt=deposit_delta,
            fee_delta_usdt=fee_delta,
            market_beta_estimate_usdt=market_beta_estimate,
            alpha_estimate_usdt=alpha_estimate,
            largest_contributor_symbol=contributor_symbol,
            largest_contributor_delta_usdt=contributor_delta,
            concentration_ratio=concentration_ratio,
            attribution_type=attribution_type,
            confidence=confidence,
            reason=reason,
        )


class NAVProtectionEngine:
    def __init__(
        self,
        *,
        profit_lock_trigger_pct: float = 0.03,
        floating_gain_trigger_pct: float = 0.02,
        drawdown_defensive_pct: float = 0.02,
        drawdown_freeze_buy_pct: float = 0.04,  # gap6: staggered — OFC kill-switch fires at 6%
        drawdown_recovery_pct: float = 0.08,
        profit_lock_ratio: float = 0.50,
        minimum_protection_floor_ratio: float = 0.95,
        market_beta_buy_size_multiplier: float = 0.50,
        market_beta_confidence_floor_delta: float = 0.10,
        concentration_gain_trim_threshold: float = 0.50,
    ) -> None:
        self.profit_lock_trigger_pct = max(0.0, float(profit_lock_trigger_pct))
        self.floating_gain_trigger_pct = max(0.0, float(floating_gain_trigger_pct))
        self.drawdown_defensive_pct = max(0.0, float(drawdown_defensive_pct))
        self.drawdown_freeze_buy_pct = max(0.0, float(drawdown_freeze_buy_pct))
        self.drawdown_recovery_pct = max(0.0, float(drawdown_recovery_pct))
        self.profit_lock_ratio = max(0.0, min(1.0, float(profit_lock_ratio)))
        self.minimum_protection_floor_ratio = max(0.0, min(1.0, float(minimum_protection_floor_ratio)))
        self.market_beta_buy_size_multiplier = max(0.0, min(1.0, float(market_beta_buy_size_multiplier)))
        self.market_beta_confidence_floor_delta = max(0.0, min(1.0, float(market_beta_confidence_floor_delta)))
        self.concentration_gain_trim_threshold = max(
            0.0, min(1.0, float(concentration_gain_trim_threshold))
        )

    # Rolling peak window: 7 days. Peak older than this decays toward current NAV.
    _PEAK_ROLLING_WINDOW_SEC = 7 * 24 * 3600
    # Intraday decay: if peak hasn't been refreshed for >24h, start drifting it toward current NAV.
    _PEAK_INTRADAY_DECAY_SEC = 24 * 3600
    # Per-day drift factor: peak decays 3% of the gap toward current NAV per day without new highs.
    _PEAK_DAILY_DRIFT = 0.03

    def evaluate(self, shared_state: Any, attribution: NAVAttribution) -> NAVProtectionState:
        current_nav = attribution.current_nav_usdt

        # Session anchor: on startup, peak is capped to session_anchor_nav so a stale
        # cross-session unrealized spike doesn't lock the new session into DEFENSIVE immediately.
        session_anchor = float(
            getattr(shared_state, "session_anchor_nav", 0.0) or 0.0
        )
        _snapshot_peak = max(
            float(getattr(shared_state, "peak_nav_usdt", 0.0) or 0.0),
            attribution.peak_nav_usdt,
            current_nav,
        )
        _peak_ts = float((getattr(shared_state, "metrics", {}) or {}).get("peak_nav_ts", 0.0) or 0.0)
        _now = time.time()
        # 2026-07-14 fix: the two "cap peak to session_anchor" guards below exist
        # to stop a STALE peak from a PRIOR session dominating this one -- but
        # they previously fired unconditionally on magnitude alone, so a
        # genuine, freshly-set peak from real intra-session growth (e.g. NAV
        # climbed to $1500 this session, then pulled back to $1400) was
        # silently collapsed down to current_nav every cycle after the pullback,
        # discarding the true peak. Only apply the caps when the peak predates
        # this session's start (or session-start info is unavailable, in which
        # case fall back to the original conservative behavior).
        _session_start_ts = float(getattr(shared_state, "_session_start_ts", 0.0) or 0.0)
        _peak_is_fresh_this_session = (
            _session_start_ts > 0 and _peak_ts > 0 and _peak_ts >= _session_start_ts
        )
        # Cap stale snapshot peak: if it exceeds the session anchor by >10%, treat session anchor
        # as the effective starting peak so a cross-session spike doesn't dominate.
        if (
            not _peak_is_fresh_this_session
            and session_anchor > 0
            and _snapshot_peak > session_anchor * 1.10
        ):
            _snapshot_peak = max(session_anchor, current_nav)

        _raw_peak = _snapshot_peak

        if _peak_ts > 0:
            _age_sec = _now - _peak_ts
            if _age_sec > self._PEAK_ROLLING_WINDOW_SEC:
                # Peak is stale (>7 days) — blend toward current NAV (50% weight per expired window)
                _windows_expired = _age_sec / self._PEAK_ROLLING_WINDOW_SEC
                _decay = 0.5 ** _windows_expired
                _raw_peak = current_nav + (_raw_peak - current_nav) * _decay
            elif _age_sec > self._PEAK_INTRADAY_DECAY_SEC:
                # Peak hasn't refreshed in >24h — drift it 3% of the gap per day toward current NAV
                _days_stale = _age_sec / (24 * 3600)
                _drift_factor = max(0.0, 1.0 - self._PEAK_DAILY_DRIFT * _days_stale)
                _raw_peak = current_nav + (_raw_peak - current_nav) * _drift_factor

        # Cap peak_nav to session_anchor — prevents prior session highs from dominating.
        # When session_anchor is set, the effective peak cannot exceed it. Skipped
        # when the peak is fresh (set this session) -- see _peak_is_fresh_this_session
        # above; a genuine intra-session peak above the anchor must be preserved.
        if not _peak_is_fresh_this_session and session_anchor > 0 and _raw_peak > session_anchor:
            _raw_peak = session_anchor
        peak_nav = max(_raw_peak, current_nav)
        prev_floor = float(getattr(shared_state, "protection_floor_usdt", 0.0) or 0.0)

        # Use session_anchor_nav as the drawdown baseline (session-relative).
        # This prevents past sessions' peaks from permanently blocking a new session.
        # Drawdown resets to 0% on every restart — each session protects its own capital.
        drawdown_baseline = session_anchor if session_anchor > 0 else current_nav
        drawdown = 0.0
        if drawdown_baseline > 0:
            drawdown = max(0.0, (drawdown_baseline - current_nav) / drawdown_baseline)

        # Auto-reset anchor when fully in cash after RECOVERY threshold is breached.
        # If all positions have been stopped out (exposure=0, no open positions) and drawdown
        # still exceeds the recovery threshold, the SL engine already protected all capital —
        # keeping the old anchor just freezes the bot indefinitely with nothing to recover.
        # Reset anchor + peak to current NAV so trading can resume from a clean baseline.
        if drawdown >= self.drawdown_recovery_pct and current_nav > 0:
            _positions = getattr(shared_state, "positions", {}) or {}
            _invested = float(getattr(shared_state, "invested_capital_usdt", 0.0) or 0.0)
            _fully_in_cash = (len(_positions) == 0 or _invested <= 0.0)
            if _fully_in_cash:
                import logging as _logging
                _logging.getLogger("NAVProtectionEngine").warning(
                    "[NAVProtect:ANCHOR-RESET] Fully in cash after RECOVERY (drawdown=%.2f%% "
                    "anchor=%.2f→%.2f) — resetting anchor to current NAV to unblock trading.",
                    drawdown * 100.0, drawdown_baseline, current_nav,
                )
                if hasattr(shared_state, "session_anchor_nav"):
                    shared_state.session_anchor_nav = current_nav
                if hasattr(shared_state, "peak_nav_usdt"):
                    shared_state.peak_nav_usdt = current_nav
                metrics = getattr(shared_state, "metrics", {})
                if isinstance(metrics, dict):
                    metrics["peak_nav"] = current_nav
                    metrics["peak_nav_ts"] = time.time()
                drawdown_baseline = current_nav
                drawdown = 0.0
                peak_nav = current_nav

        mode = "NORMAL"
        reason = "capital_stable"
        allow_buy = True
        allow_sell = True
        allow_tp_sl_adjustment = False
        size_multiplier = 1.0
        confidence_delta = 0.0
        reserve_floor_ratio = 0.0
        actions: list[str] = []
        locked_profit = 0.0
        available_profit_to_risk = max(0.0, current_nav - prev_floor)

        profit_over_anchor = max(
            0.0,
            current_nav - float(getattr(shared_state, "session_anchor_nav", current_nav) or current_nav),
        )
        floating_gain_pct = max(0.0, attribution.nav_delta_pct)
        free_usdt = float(getattr(shared_state, "free_balance_usdt", 0.0) or 0.0)
        free_ratio = (free_usdt / current_nav) if current_nav > 0 else 0.0

        # 2026-07-15 fix: this used to unconditionally ratchet to peak*ratio (95%)
        # on EVERY cycle regardless of mode -- so a perfectly healthy, flat/cash
        # account sitting at its own all-time peak (the normal state for a small
        # account that has never drawn down) had ~95% of NAV permanently walled
        # off from new position sizing via daily_compounding.sizing_nav()'s
        # protection_floor_usdt connection, leaving ~5% of peak "spendable" --
        # far below Binance's minimum order size for any realistic small NAV.
        # Confirmed live: peak=$57.87, ratio=0.95 -> floor=$54.97 -> only $2.87
        # "available", blocking a signal that had already passed confidence and
        # regime gates. The floor should only ratchet up within an ACTUAL
        # protective trigger below (real drawdown or a realized-profit lock),
        # matching every other branch's own bespoke floor logic -- not in the
        # default/NORMAL, no-event case.
        new_floor = prev_floor

        recovery_state = getattr(shared_state, "recovery_state", {}) or {}
        if recovery_state.get("recovery_mode_active", False):
            mode = "RECOVERY"
            allow_buy = False
            size_multiplier = 0.0
            confidence_delta = 0.20
            reserve_floor_ratio = max(
                reserve_floor_ratio,
                float(recovery_state.get("reserve_floor_ratio", 0.0) or 0.0),
            )
            actions.append("ALLOW_RECOVERY_SELLS")
            reason = "portfolio_recovery_active"
        elif drawdown >= self.drawdown_recovery_pct:
            mode = "RECOVERY"
            allow_buy = False
            size_multiplier = 0.0
            confidence_delta = 0.20
            new_floor = max(new_floor, peak_nav * self.minimum_protection_floor_ratio)
            actions.append("RECOVERY_ONLY")
            reason = "nav_drawdown_recovery_threshold"
        elif drawdown >= self.drawdown_freeze_buy_pct:
            mode = "FREEZE_BUY"
            allow_buy = False
            size_multiplier = 0.0
            confidence_delta = 0.15
            allow_tp_sl_adjustment = True
            new_floor = max(new_floor, peak_nav * self.minimum_protection_floor_ratio)
            actions.extend(["FREEZE_NEW_BUYS", "TIGHTEN_TP_SL"])
            reason = "nav_drawdown_freeze_buy_threshold"
        elif drawdown >= self.drawdown_defensive_pct:
            mode = "DEFENSIVE"
            size_multiplier = 0.50
            confidence_delta = 0.10
            allow_tp_sl_adjustment = True
            new_floor = max(new_floor, peak_nav * self.minimum_protection_floor_ratio)
            actions.extend(["REDUCE_BUY_SIZE", "TIGHTEN_TP_SL"])
            reason = "nav_drawdown_defensive_threshold"
        elif attribution.attribution_type == "REALIZED_PROFIT" and floating_gain_pct >= self.profit_lock_trigger_pct:
            mode = "PROFIT_LOCK"
            locked_profit = profit_over_anchor * self.profit_lock_ratio
            available_profit_to_risk = max(0.0, profit_over_anchor - locked_profit)
            new_floor = max(new_floor, current_nav - locked_profit)
            size_multiplier = 0.85
            confidence_delta = 0.05
            actions.extend(["LOCK_REALIZED_PROFIT", "KEEP_CONTROLLED_REINVESTMENT"])
            reason = "realized_profit_nav_increase"
        elif attribution.attribution_type in {"UNREALIZED_MARKET_BETA", "UNREALIZED_ALPHA"} and floating_gain_pct >= self.floating_gain_trigger_pct:
            mode = "FLOATING_GAIN_PROTECTION"
            size_multiplier = self.market_beta_buy_size_multiplier
            confidence_delta = self.market_beta_confidence_floor_delta
            reserve_floor_ratio = 0.20
            allow_tp_sl_adjustment = True
            actions.extend(["TIGHTEN_TP_SL", "TRAIL_STOP"])
            reason = "floating_gain_requires_protection"
        elif attribution.attribution_type == "CONCENTRATION_GAIN":
            mode = "FLOATING_GAIN_PROTECTION"
            size_multiplier = self.market_beta_buy_size_multiplier
            confidence_delta = max(self.market_beta_confidence_floor_delta, 0.12)
            reserve_floor_ratio = 0.20
            allow_tp_sl_adjustment = True
            actions.extend(["TIGHTEN_TP_SL", "PARTIAL_TAKE_PROFIT", "BLOCK_ADD_TO_CONCENTRATED_SYMBOL"])
            reason = "concentration_gain_risk"
        elif (
            attribution.confidence in {"LOW", "UNKNOWN"}
            and drawdown < self.drawdown_defensive_pct
            and free_ratio >= max(0.20, float(recovery_state.get("reserve_floor_ratio", 0.0) or 0.0))
        ):
            # Low-confidence attribution but drawdown is below defensive threshold and capital is healthy
            # — treat as NORMAL rather than DEFENSIVE to avoid paralysing the system on routine small losses
            mode = "NORMAL"
            size_multiplier = 1.0
            confidence_delta = 0.0
            reserve_floor_ratio = 0.0
            reason = "stable_nav_healthy_capital_attribution_uncertain"

        return NAVProtectionState(
            protection_mode=mode,
            peak_nav_usdt=peak_nav,
            protection_floor_usdt=max(prev_floor, new_floor),
            drawdown_from_peak_pct=drawdown * 100.0,
            locked_profit_usdt=locked_profit,
            available_profit_to_risk_usdt=available_profit_to_risk,
            allow_buy=allow_buy,
            allow_sell=allow_sell,
            allow_tp_sl_adjustment=allow_tp_sl_adjustment,
            suggested_size_multiplier=size_multiplier,
            suggested_confidence_floor_delta=confidence_delta,
            suggested_reserve_floor_ratio=reserve_floor_ratio,
            suggested_actions=actions,
            reason=reason,
        )


def evaluate_nav_protection(
    shared_state: Any,
    *,
    attribution_engine: NAVAttributionEngine | None = None,
    protection_engine: NAVProtectionEngine | None = None,
) -> tuple[NAVAttribution, NAVProtectionState]:
    attribution_engine = attribution_engine or NAVAttributionEngine()
    protection_engine = protection_engine or NAVProtectionEngine()
    attribution = attribution_engine.evaluate(shared_state)
    protection = protection_engine.evaluate(shared_state, attribution)
    attr_dict = attribution.to_dict()
    attr_dict["realized_pnl_total_usdt"] = float(
        getattr(shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0
    )
    attr_dict["unrealized_pnl_total_usdt"] = float(
        getattr(shared_state, "metrics", {}).get("unrealized_pnl", 0.0) or 0.0
    )
    attr_dict["free_usdt"] = float(getattr(shared_state, "free_balance_usdt", 0.0) or 0.0)
    attr_dict["evaluated_at"] = time.time()
    if hasattr(shared_state, "update_nav_protection"):
        shared_state.update_nav_protection(
            attribution=attr_dict,
            protection_state=protection.to_dict(),
        )
    return attribution, protection


def _pos_float(pos: Any, *keys: str) -> float:
    for key in keys:
        if isinstance(pos, dict):
            value = pos.get(key)
        else:
            value = getattr(pos, key, None)
        if value is None:
            continue
        try:
            return float(value or 0.0)
        except Exception:
            continue
    return 0.0
