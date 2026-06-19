from __future__ import annotations

import logging
from typing import Any

from src.l0_core.stubs import maybe_call

from .schemas import AIOfficeDecision, AIOfficeRecommendation

logger = logging.getLogger("AIOffice.PolicyBus")

UNSAFE_ACTIONS = {
    "place_order",
    "cancel_order",
    "direct_exchange_call",
    "override_risk",
    "bypass_execution_manager",
    "override_hyg",
    "edit_live_code",
}

SAFE_ACTIONS = {
    "adjust_agent_weight",
    "adjust_exposure_target",
    "request_liquidity",
    "request_dust_cleanup",
    "pause_new_buys",
    "resume_new_buys",
    "trigger_recovery_check",
    "emit_alert",
    "keep_current_policy",
    "request_risk_review",
    "tighten_tp_sl",
    "reduce_exposure",
    "freeze_new_buys",
}


class PolicyBus:
    def __init__(self, shared_state: Any | None = None) -> None:
        self._ss = shared_state

    async def validate(self, rec: AIOfficeRecommendation) -> AIOfficeDecision:
        action = str(rec.suggested_action or "").strip()
        if action in UNSAFE_ACTIONS:
            return await self._reject(rec, f"unsafe_action:{action}")
        if action not in SAFE_ACTIONS:
            return await self._reject(rec, f"unknown_action:{action}")

        return AIOfficeDecision(
            source="PolicyBus",
            trace_id=rec.trace_id,
            recommendation_type=rec.recommendation_type,
            accepted=True,
            reason="validated",
            severity=rec.severity,
            target_component=rec.target_component,
            suggested_action=action,
            ttl_sec=int(rec.ttl_sec or 0),
            metadata=dict(rec.metadata or {}),
        )

    async def _reject(self, rec: AIOfficeRecommendation, reason: str) -> AIOfficeDecision:
        payload = {
            "event": "AI_OFFICE_RECOMMENDATION_REJECTED",
            "trace_id": rec.trace_id,
            "source": rec.source,
            "recommendation_type": rec.recommendation_type,
            "suggested_action": rec.suggested_action,
            "reason": reason,
            "target_component": rec.target_component,
            "metadata": dict(rec.metadata or {}),
        }
        logger.warning("AI_OFFICE_RECOMMENDATION_REJECTED %s", payload)
        await maybe_call(self._ss, "emit_event", "SummaryLog", payload)
        await maybe_call(self._ss, "emit_event", "ErrorEvent", payload)
        return AIOfficeDecision(
            source="PolicyBus",
            trace_id=rec.trace_id,
            recommendation_type=rec.recommendation_type,
            accepted=False,
            reason=reason,
            severity="WARN",
            target_component=rec.target_component,
            suggested_action=str(rec.suggested_action or ""),
            ttl_sec=int(rec.ttl_sec or 0),
            metadata=dict(rec.metadata or {}),
        )
