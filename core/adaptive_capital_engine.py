"""
Adaptive Capital Engine
=======================

Computes a bounded adaptive sizing envelope for BUY planning:
- dynamic risk fraction
- dynamic economic min trade quote
- optional cooldown / TP hints for downstream components
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import time

# Sentinel for "no trades on record": guaranteed >= any finite idle_time_sec threshold.
_IDLE_SEC_NO_TRADES: float = float("inf")


@dataclass
class AdaptiveSizingDecision:
    risk_fraction: float
    min_trade_quote: float
    cooldown_mult: float
    tp_bias_mult: float
    win_rate: float
    avg_r_multiple: float
    avg_hold_sec: float
    win_streak: int
    loss_streak: int
    fee_to_gross_ratio: float
    reasons: List[str] = field(default_factory=list)


class AdaptiveCapitalEngine:
    """Adaptive economic sizing policy used by ScalingManager."""

    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("AdaptiveCapitalEngine")
        self.enabled = bool(getattr(config, "ADAPTIVE_CAPITAL_ENGINE_ENABLED", False))
        self._perf_review_sec = float(getattr(config, "ADAPTIVE_PERF_REVIEW_SEC", 3600.0) or 3600.0)
        self._perf_lookback = int(getattr(config, "ADAPTIVE_PERF_LOOKBACK_TRADES", 200) or 200)
        self._risk_min = float(getattr(config, "ADAPTIVE_RISK_FRACTION_MIN", 0.05) or 0.05)
        self._risk_max = float(getattr(config, "ADAPTIVE_RISK_FRACTION_MAX", 0.35) or 0.35)
        self._drawdown_soft = float(getattr(config, "ADAPTIVE_DRAWDOWN_SOFT_PCT", 2.0) or 2.0)
        self._drawdown_hard = float(getattr(config, "ADAPTIVE_DRAWDOWN_HARD_PCT", 4.0) or 4.0)
        self._high_vol_pct = float(getattr(config, "ADAPTIVE_HIGH_VOL_PCT", 0.015) or 0.015)
        self._low_vol_pct = float(getattr(config, "ADAPTIVE_LOW_VOL_PCT", 0.004) or 0.004)
        self._throughput_low_ratio = float(
            getattr(config, "ADAPTIVE_THROUGHPUT_LOW_RATIO", 0.50) or 0.50
        )
        self._idle_free_capital_pct = float(
            getattr(config, "ADAPTIVE_IDLE_FREE_CAPITAL_PCT", 0.60) or 0.60
        )
        self._idle_time_sec = float(getattr(config, "ADAPTIVE_IDLE_TIME_SEC", 1800.0) or 1800.0)
        self._win_streak_trades = int(getattr(config, "ADAPTIVE_WIN_STREAK_TRADES", 3) or 3)
        self._loss_streak_trades = int(getattr(config, "ADAPTIVE_LOSS_STREAK_TRADES", 3) or 3)
        self._win_streak_bonus = float(
            getattr(config, "ADAPTIVE_WIN_STREAK_RISK_BONUS", 0.10) or 0.10
        )
        self._loss_streak_penalty = float(
            getattr(config, "ADAPTIVE_LOSS_STREAK_RISK_PENALTY", 0.18) or 0.18
        )
        self._win_rate_bonus_threshold = float(
            getattr(config, "ADAPTIVE_WIN_RATE_BONUS_THRESHOLD", 0.60) or 0.60
        )
        self._win_rate_penalty_threshold = float(
            getattr(config, "ADAPTIVE_WIN_RATE_PENALTY_THRESHOLD", 0.45) or 0.45
        )
        self._win_rate_bonus = float(getattr(config, "ADAPTIVE_WIN_RATE_BONUS", 0.08) or 0.08)
        self._win_rate_penalty = float(
            getattr(config, "ADAPTIVE_WIN_RATE_PENALTY", 0.10) or 0.10
        )
        self._fee_gross_threshold = float(
            getattr(config, "ADAPTIVE_FEE_GROSS_THRESHOLD", 0.35) or 0.35
        )
        self._fee_gross_bonus = float(getattr(config, "ADAPTIVE_FEE_GROSS_BONUS", 0.06) or 0.06)
        self._econ_min_notional_mult = float(
            getattr(config, "ADAPTIVE_ECON_MIN_NOTIONAL_MULT", 1.20) or 1.20
        )
        self._econ_target_profit_pct = float(
            getattr(
                config,
                "ADAPTIVE_ECON_TARGET_PROFIT_PCT",
                getattr(config, "MIN_NET_PROFIT_AFTER_FEES", 0.004),
            )
            or 0.004
        )

        # Per-symbol performance cache: {symbol: snapshot_dict} and {symbol: cache_ts}
        self._perf_cache: Dict[str, Dict[str, Any]] = {}
        self._perf_cache_ts: Dict[str, float] = {}

        # Dynamic sizing state
        self.dynamic_min_trade_quote: float = float(getattr(config, "DEFAULT_PLANNED_QUOTE", 10.0))

    def _extract_realized(self, trade: Dict[str, Any]) -> float:
        for key in ("realized_delta", "realized_pnl", "net_pnl", "pnl", "profit"):
            try:
                if key in trade and trade[key] is not None:
                    return float(trade[key] or 0.0)
            except Exception:
                continue
        return 0.0

    def _extract_fee(self, trade: Dict[str, Any]) -> float:
        for key in ("fee_quote", "commission", "fee"):
            try:
                if key in trade and trade[key] is not None:
                    return abs(float(trade[key] or 0.0))
            except Exception:
                continue
        return 0.0

    def _extract_hold_sec(self, trade: Dict[str, Any]) -> float:
        for key in ("hold_sec", "hold_time_sec", "holding_time_sec", "duration_sec"):
            try:
                if key in trade and trade[key] is not None:
                    return max(0.0, float(trade[key] or 0.0))
            except Exception:
                continue
        return 0.0

    def _compute_streaks(self, realized_series: List[float]) -> Dict[str, int]:
        win_streak = 0
        loss_streak = 0
        for pnl in reversed(realized_series):
            if pnl > 0:
                if loss_streak == 0:
                    win_streak += 1
                else:
                    break
            elif pnl < 0:
                if win_streak == 0:
                    loss_streak += 1
                else:
                    break
            else:
                # Exactly zero PnL: breakeven trade — neutral, does not break a streak.
                continue
        return {"win_streak": int(win_streak), "loss_streak": int(loss_streak)}

    def _compute_performance_snapshot(self, trade_history: List[Dict[str, Any]], now_ts: float) -> Dict[str, Any]:
        if not trade_history:
            return {
                "win_rate": 0.50,
                "avg_r_multiple": 0.0,
                "avg_hold_sec": 0.0,
                "win_streak": 0,
                "loss_streak": 0,
                "fee_to_gross_ratio": 0.0,
                "idle_sec": _IDLE_SEC_NO_TRADES,
            }

        realized_vals: List[float] = []
        fees: List[float] = []
        holds: List[float] = []
        last_trade_ts = 0.0
        for t in trade_history[-self._perf_lookback:]:
            if not isinstance(t, dict):
                continue
            pnl = self._extract_realized(t)
            if abs(pnl) > 1e-12:
                realized_vals.append(pnl)
            fee_val = self._extract_fee(t)
            if fee_val > 0:
                fees.append(fee_val)
            hold = self._extract_hold_sec(t)
            if hold > 0:
                holds.append(hold)
            try:
                ts = float(t.get("ts", 0.0) or 0.0)
                if ts > last_trade_ts:
                    last_trade_ts = ts
            except Exception:
                pass

        idle_sec = float(max(0.0, now_ts - last_trade_ts)) if last_trade_ts > 0 else _IDLE_SEC_NO_TRADES
        avg_hold = float(sum(holds) / len(holds)) if holds else 0.0

        # No trades with non-zero PnL yet: return neutral stats rather than a 0.0 win-rate
        # that would erroneously trigger the win_rate_penalty branch in evaluate().
        if not realized_vals:
            return {
                "win_rate": 0.50,
                "avg_r_multiple": 0.0,
                "avg_hold_sec": avg_hold,
                "win_streak": 0,
                "loss_streak": 0,
                "fee_to_gross_ratio": 0.0,
                "idle_sec": idle_sec,
            }

        wins = sum(1 for p in realized_vals if p > 0)
        losses = sum(1 for p in realized_vals if p < 0)
        denom = max(1, wins + losses)
        win_rate = float(wins / denom)
        avg_r = float(sum(realized_vals) / max(1, len(realized_vals)))
        gross_profit = float(sum(max(0.0, p) for p in realized_vals))
        total_fees = float(sum(fees))
        fee_to_gross = float(total_fees / gross_profit) if gross_profit > 0 else 0.0
        streaks = self._compute_streaks(realized_vals)
        return {
            "win_rate": win_rate,
            "avg_r_multiple": avg_r,
            "avg_hold_sec": avg_hold,
            "win_streak": streaks["win_streak"],    # int, not float
            "loss_streak": streaks["loss_streak"],   # int, not float
            "fee_to_gross_ratio": fee_to_gross,
            "idle_sec": idle_sec,
        }

    def _get_perf_snapshot(self, symbol: str, trade_history: List[Dict[str, Any]], now_ts: float) -> Dict[str, Any]:
        """Return cached performance snapshot for this symbol, recomputing when stale."""
        cache_ts = self._perf_cache_ts.get(symbol, 0.0)
        if (now_ts - cache_ts) < self._perf_review_sec and symbol in self._perf_cache:
            return dict(self._perf_cache[symbol])
        snap = self._compute_performance_snapshot(trade_history, now_ts)
        self._perf_cache[symbol] = dict(snap)
        self._perf_cache_ts[symbol] = now_ts
        return snap

    def evaluate(
        self,
        *,
        symbol: str,
        nav: float,
        free_capital: float,
        base_risk_fraction: float,
        volatility_pct: float,
        drawdown_pct: float,
        fee_bps: float,
        slippage_bps: float,
        min_notional: float,
        slot_utilization: float,
        throughput_per_hour: float,
        target_throughput_per_hour: float,
        trade_history: List[Dict[str, Any]],
        now_ts: Optional[float] = None,
    ) -> AdaptiveSizingDecision:
        now_ts = float(now_ts or time.time())
        base_risk = max(0.0, float(base_risk_fraction or 0.0))
        nav_val = max(0.0, float(nav or 0.0))
        free_cap = max(0.0, float(free_capital or 0.0))
        free_ratio = (free_cap / nav_val) if nav_val > 0 else 0.0
        slot_use = min(1.0, max(0.0, float(slot_utilization or 0.0)))
        min_notional_val = max(0.0, float(min_notional or 0.0))

        if not self.enabled:
            return AdaptiveSizingDecision(
                risk_fraction=base_risk,
                min_trade_quote=min_notional_val,
                cooldown_mult=1.0,
                tp_bias_mult=1.0,
                win_rate=0.5,
                avg_r_multiple=0.0,
                avg_hold_sec=0.0,
                win_streak=0,
                loss_streak=0,
                fee_to_gross_ratio=0.0,
                reasons=["engine_disabled"],
            )

        perf = self._get_perf_snapshot(symbol, trade_history, now_ts)
        win_rate = float(perf.get("win_rate", 0.5) or 0.5)
        avg_r = float(perf.get("avg_r_multiple", 0.0) or 0.0)
        avg_hold = float(perf.get("avg_hold_sec", 0.0) or 0.0)
        win_streak = int(perf.get("win_streak", 0) or 0)
        loss_streak = int(perf.get("loss_streak", 0) or 0)
        fee_to_gross = float(perf.get("fee_to_gross_ratio", 0.0) or 0.0)
        idle_sec = perf.get("idle_sec", _IDLE_SEC_NO_TRADES)
        if idle_sec is None:
            idle_sec = _IDLE_SEC_NO_TRADES

        reasons: List[str] = []
        risk_mult = 1.0
        cooldown_mult = 1.0
        tp_bias_mult = 1.0

        if win_streak >= self._win_streak_trades:
            risk_mult *= (1.0 + self._win_streak_bonus)
            reasons.append("win_streak_up")
        if loss_streak >= self._loss_streak_trades:
            risk_mult *= max(0.25, 1.0 - self._loss_streak_penalty)
            cooldown_mult *= 1.10
            reasons.append("loss_streak_down")

        if drawdown_pct >= self._drawdown_hard:
            risk_mult *= 0.70
            cooldown_mult *= 1.15
            reasons.append("drawdown_hard")
        elif drawdown_pct >= self._drawdown_soft:
            risk_mult *= 0.85
            reasons.append("drawdown_soft")

        vol_pct = max(0.0, float(volatility_pct or 0.0))
        if vol_pct >= self._high_vol_pct:
            risk_mult *= 0.85
            reasons.append("high_vol_down")
        elif vol_pct <= self._low_vol_pct:
            risk_mult *= 1.05
            reasons.append("low_vol_up")

        target_throughput = max(0.0, float(target_throughput_per_hour or 0.0))
        throughput = max(0.0, float(throughput_per_hour or 0.0))
        if target_throughput > 0:
            throughput_ratio = throughput / target_throughput
            if throughput_ratio < self._throughput_low_ratio:
                risk_mult *= 1.08
                cooldown_mult *= 0.92
                reasons.append("throughput_low_up")

        if free_ratio > self._idle_free_capital_pct and idle_sec >= self._idle_time_sec:
            risk_mult *= 1.15
            cooldown_mult *= 0.90
            reasons.append("idle_capital_boost")

        if slot_use >= 0.90:
            risk_mult *= 0.90
            reasons.append("slot_tight_down")

        if win_rate > self._win_rate_bonus_threshold:
            risk_mult *= (1.0 + self._win_rate_bonus)
            reasons.append("win_rate_up")
        elif win_rate < self._win_rate_penalty_threshold:
            risk_mult *= max(0.25, 1.0 - self._win_rate_penalty)
            reasons.append("win_rate_down")

        # When fee/gross is high, trades are undersized relative to fees.
        # Upsize risk_fraction so future trades amortise fees more effectively.
        # The complementary fee_eff_quote below raises the minimum notional floor
        # for the same reason; the two levers are independent and additive.
        if fee_to_gross > self._fee_gross_threshold:
            risk_mult *= (1.0 + self._fee_gross_bonus)
            reasons.append("fee_efficiency_upsize")

        if avg_hold > float(getattr(self.config, "MAX_HOLD_TIME_SEC", 1800.0) or 1800.0):
            tp_bias_mult *= 0.95
            reasons.append("hold_time_tighten_tp")

        risk_fraction = min(self._risk_max, max(self._risk_min, base_risk * risk_mult))

        # Economic min quote: fee/slippage-aware and NAV-bounded.
        # fee_bps is assumed to be one-way taker fee; multiply by 2 for round-trip.
        round_trip_cost_pct = max(0.0, ((float(fee_bps or 0.0) * 2.0) + float(slippage_bps or 0.0)) / 10000.0)
        target_profit_pct = max(0.0005, float(self._econ_target_profit_pct))
        fee_eff_quote = 0.0
        if round_trip_cost_pct > 0:
            # Scale up min_notional so that round-trip cost is a small fraction of target net profit.
            # net_target: required net profit rate above costs; floor prevents division explosion.
            net_target = max(0.0002, target_profit_pct - (round_trip_cost_pct * 0.5))
            fixed_cost_proxy = min_notional_val * round_trip_cost_pct
            fee_eff_quote = fixed_cost_proxy / max(net_target, 1e-9)

        min_trade_quote = max(
            min_notional_val * self._econ_min_notional_mult,
            fee_eff_quote,
            min_notional_val,
        )

        max_position_pct = float(
            getattr(self.config, "MAX_POSITION_EXPOSURE_PERCENTAGE", 0.20) or 0.20
        )
        nav_cap = nav_val * max(0.01, max_position_pct) if nav_val > 0 else 0.0
        if nav_cap > 0:
            min_trade_quote = min(min_trade_quote, nav_cap)
        min_trade_quote = max(min_trade_quote, min_notional_val)

        # Update dynamic minimum for external access
        self.dynamic_min_trade_quote = float(min_trade_quote)

        self.logger.debug(
            "[ACE] %s risk_frac=%.4f min_quote=%.2f cooldown=%.2f reasons=%s",
            symbol, risk_fraction, min_trade_quote, cooldown_mult, reasons,
        )

        return AdaptiveSizingDecision(
            risk_fraction=float(risk_fraction),
            min_trade_quote=float(min_trade_quote),
            cooldown_mult=float(max(0.5, min(1.5, cooldown_mult))),
            tp_bias_mult=float(max(0.8, min(1.2, tp_bias_mult))),
            win_rate=win_rate,
            avg_r_multiple=avg_r,
            avg_hold_sec=avg_hold,
            win_streak=win_streak,
            loss_streak=loss_streak,
            fee_to_gross_ratio=fee_to_gross,
            reasons=reasons,
        )
