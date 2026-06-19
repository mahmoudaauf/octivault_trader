from __future__ import annotations

from .schemas import AIOfficeSnapshot, RiskPolicyRecommendation


class RiskAuditorAgent:
    name = "RiskAuditorAgent"

    async def analyze(self, snapshot: AIOfficeSnapshot) -> RiskPolicyRecommendation:
        action = "keep_current_policy"
        severity = "INFO"
        confidence = 0.4
        reason = "risk_profile_stable"
        risk_state = "NORMAL"

        if snapshot.drawdown_from_peak_pct >= 0.08:
            action = "freeze_new_buys"
            severity = "CRITICAL"
            confidence = 0.9
            reason = "drawdown_breach_requires_defensive_freeze"
            risk_state = "CRITICAL"
        elif snapshot.concentration_ratio >= 0.50:
            action = "reduce_exposure"
            severity = "WARN"
            confidence = 0.8
            reason = "portfolio_concentration_too_high"
            risk_state = "CONCENTRATED"
        elif snapshot.capital_locked_ratio >= 0.60:
            action = "request_risk_review"
            severity = "WARN"
            confidence = 0.75
            reason = "capital_locked_ratio_elevated"
            risk_state = "LOCKED"

        return RiskPolicyRecommendation(
            source=self.name,
            recommendation_type="risk_policy",
            confidence=confidence,
            reason=reason,
            severity=severity,
            target_component="RiskManager",
            suggested_action=action,
            ttl_sec=28800,
            risk_state=risk_state,
            metadata={
                "drawdown_from_peak_pct": snapshot.drawdown_from_peak_pct,
                "concentration_ratio": snapshot.concentration_ratio,
                "capital_locked_ratio": snapshot.capital_locked_ratio,
            },
        )
