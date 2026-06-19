from __future__ import annotations

from .schemas import AIOfficeSnapshot, LiquidityRecommendation


class LiquiditySupervisorAgent:
    name = "LiquiditySupervisorAgent"

    async def analyze(self, snapshot: AIOfficeSnapshot) -> LiquidityRecommendation:
        action = "keep_current_policy"
        severity = "INFO"
        confidence = 0.45
        reason = "liquidity_stable"
        liquidity_state = "NORMAL"

        if snapshot.free_usdt <= 0.0 or snapshot.capital_locked_ratio >= 0.80:
            action = "pause_new_buys"
            severity = "WARN"
            confidence = 0.86
            reason = "insufficient_quote_liquidity"
            liquidity_state = "STARVED"
        elif snapshot.capital_locked_ratio < 0.70 and snapshot.free_usdt > 10.0:
            # Conditions recovered — explicitly resume so the pause flag clears
            action = "resume_new_buys"
            severity = "INFO"
            confidence = 0.80
            reason = "liquidity_recovered"
            liquidity_state = "NORMAL"
        elif snapshot.dust_ratio >= 0.03:
            action = "request_dust_cleanup"
            severity = "WARN"
            confidence = 0.82
            reason = "dust_ratio_above_cleanup_threshold"
            liquidity_state = "DUSTY"
        elif snapshot.stale_positions_count > 0 or snapshot.weak_positions_count > 0:
            action = "request_liquidity"
            severity = "WARN"
            confidence = 0.72
            reason = "stale_or_weak_positions_locking_capital"
            liquidity_state = "LOCKED"

        return LiquidityRecommendation(
            source=self.name,
            recommendation_type="liquidity_policy",
            confidence=confidence,
            reason=reason,
            severity=severity,
            target_component="PortfolioBalancer",
            suggested_action=action,
            ttl_sec=28800,
            liquidity_state=liquidity_state,
            metadata={
                "free_usdt": snapshot.free_usdt,
                "locked_usdt": snapshot.locked_usdt,
                "dust_ratio": snapshot.dust_ratio,
                "stale_positions_count": snapshot.stale_positions_count,
            },
        )
