from __future__ import annotations

from .schemas import AIOfficeSnapshot, NAVInsight


class PerformanceAnalystAgent:
    name = "PerformanceAnalystAgent"

    async def analyze(self, snapshot: AIOfficeSnapshot) -> NAVInsight:
        nav_delta = float(snapshot.nav_usdt - snapshot.previous_nav_usdt)
        realized = float(snapshot.realized_pnl_usdt)
        unrealized = float(snapshot.unrealized_pnl_usdt)

        attribution_type = "UNKNOWN"
        action = "keep_current_policy"
        reason = "insufficient_nav_movement"
        confidence = 0.35
        severity = "INFO"

        if nav_delta > 0 and realized > 0:
            attribution_type = "REALIZED_PROFIT"
            action = "keep_current_policy"
            reason = "nav_growth_supported_by_realized_profit"
            confidence = 0.8
        elif nav_delta > 0 and realized <= 0 and unrealized > 0:
            attribution_type = "UNREALIZED_MARKET_BETA"
            action = "protect_unrealized_gains"
            reason = "nav_growth_is_floating_not_secured"
            confidence = 0.78
            severity = "WARN"
        elif nav_delta < 0 or snapshot.drawdown_from_peak_pct > 0.02:
            attribution_type = "NAV_DECAY"
            action = "tighten_tp_sl"
            reason = "nav_decay_detected_from_peak"
            confidence = 0.74
            severity = "WARN"

        return NAVInsight(
            source=self.name,
            recommendation_type="nav_insight",
            confidence=confidence,
            reason=reason,
            severity=severity,
            target_component="StrategyManager",
            suggested_action=action,
            ttl_sec=28800,
            attribution_type=attribution_type,
            metadata={
                "nav_usdt": snapshot.nav_usdt,
                "previous_nav_usdt": snapshot.previous_nav_usdt,
                "realized_pnl_usdt": realized,
                "unrealized_pnl_usdt": unrealized,
            },
        )
