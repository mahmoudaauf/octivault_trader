from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _trace_id() -> str:
    return str(uuid.uuid4())


@dataclass
class _Serializable:
    version: str = "1.0"
    trace_id: str = field(default_factory=_trace_id)
    ts: float = field(default_factory=time.time)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AIOfficeSnapshot(_Serializable):
    nav_usdt: float = 0.0
    previous_nav_usdt: float = 0.0
    peak_nav_usdt: float = 0.0
    realized_pnl_usdt: float = 0.0
    unrealized_pnl_usdt: float = 0.0
    free_usdt: float = 0.0
    locked_usdt: float = 0.0
    capital_locked_ratio: float = 0.0
    dust_ratio: float = 0.0
    positions_count: int = 0
    stale_positions_count: int = 0
    weak_positions_count: int = 0
    concentration_ratio: float = 0.0
    drawdown_from_peak_pct: float = 0.0
    market_regime: str = "UNKNOWN"
    health_status: str = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIOfficeRecommendation(_Serializable):
    recommendation_type: str = ""
    confidence: float = 0.0
    reason: str = ""
    severity: str = "INFO"
    target_component: str = ""
    suggested_action: str = "keep_current_policy"
    ttl_sec: int = 28800
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NAVInsight(AIOfficeRecommendation):
    attribution_type: str = "UNKNOWN"


@dataclass
class RiskPolicyRecommendation(AIOfficeRecommendation):
    risk_state: str = "NORMAL"


@dataclass
class LiquidityRecommendation(AIOfficeRecommendation):
    liquidity_state: str = "NORMAL"


@dataclass
class AIOfficeDecision(_Serializable):
    recommendation_type: str = ""
    accepted: bool = False
    reason: str = ""
    severity: str = "INFO"
    target_component: str = ""
    suggested_action: str = "keep_current_policy"
    ttl_sec: int = 28800
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIOfficeHealth(_Serializable):
    status: str = "OK"
    recommendation_type: str = "health"
    confidence: float = 1.0
    reason: str = ""
    severity: str = "INFO"
    target_component: str = "HealthStatus"
    suggested_action: str = "keep_current_policy"
    ttl_sec: int = 28800
    metadata: dict[str, Any] = field(default_factory=dict)
