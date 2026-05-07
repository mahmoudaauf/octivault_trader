"""
Quant-terminal reasoning helpers.

Small, transparent helpers for classifying situation state, selecting a
playbook, and converting signal quality into a probability score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Playbook:
    name: str
    allow_buy: bool
    allow_sell: bool
    allow_rebalance: bool
    allow_dust_cleanup: bool
    max_trade_size_usdt: float
    confidence_floor: float
    reason: str


def classify_market_regime(regime: dict[str, Any], system_state: str) -> str:
    if system_state == "CRITICAL":
        return "CRISIS"
    trend = str(regime.get("trend_regime", "") or "").upper()
    vol = str(regime.get("volatility_regime", "") or "").upper()
    health = str(regime.get("overall_health", "") or "").upper()
    if health == "CRISIS":
        return "CRISIS"
    if vol == "HIGH":
        return "VOLATILE"
    if trend in {"UPTREND", "DOWNTREND"}:
        return "TRENDING"
    if trend == "RANGING":
        return "CHOPPY"
    return "UNKNOWN"


def select_playbook(situation: Any) -> Playbook:
    metrics = getattr(situation, "metrics", {}) or {}
    system_state = str(getattr(situation, "system_state", "HEALTHY") or "HEALTHY")
    risk_state = str(getattr(situation, "risk_state", "NORMAL") or "NORMAL")
    portfolio_state = str(getattr(situation, "portfolio_state", "BALANCED") or "BALANCED")
    market_regime = str(getattr(situation, "market_regime", "UNKNOWN") or "UNKNOWN")
    nav_usdt = float(metrics.get("nav_usdt", 0.0) or 0.0)
    free_usdt = float(metrics.get("free_usdt", 0.0) or 0.0)

    if system_state != "HEALTHY":
        return Playbook("SYSTEM_PAUSE", False, False, False, False, 0.0, 1.0, system_state)
    if portfolio_state == "LOW_USDT":
        return Playbook(
            "LOW_USDT_RECOVERY",
            False,
            True,
            True,
            False,
            max(5.0, free_usdt),
            0.70,
            "low free usdt",
        )
    if portfolio_state == "OVEREXPOSED":
        return Playbook(
            "OVEREXPOSED_PROTECTION",
            False,
            True,
            True,
            False,
            max(5.0, nav_usdt * 0.02),
            0.75,
            "exposure too high",
        )
    if portfolio_state == "DUST_HEAVY":
        return Playbook(
            "DUST_CLEANUP",
            False,
            True,
            True,
            True,
            max(5.0, nav_usdt * 0.01),
            0.60,
            "dust concentration high",
        )
    if risk_state == "DEFENSIVE":
        return Playbook(
            "DEFENSIVE_TRADING",
            True,
            True,
            False,
            False,
            max(10.0, nav_usdt * 0.03),
            0.70,
            "defensive risk posture",
        )
    if market_regime == "TRENDING" and portfolio_state == "CASH_HEAVY":
        return Playbook(
            "CASH_DEPLOYMENT",
            True,
            True,
            False,
            False,
            max(10.0, nav_usdt * 0.08),
            0.55,
            "cash available in trending regime",
        )
    if str(getattr(situation, "capital_state", "HEALTHY")) == "HEALTHY" and float(
        metrics.get("unrealized_pnl_usdt", 0.0) or 0.0
    ) > max(25.0, nav_usdt * 0.02):
        return Playbook(
            "PROFIT_PROTECTION",
            True,
            True,
            True,
            False,
            max(10.0, nav_usdt * 0.04),
            0.65,
            "protect open profits",
        )
    return Playbook(
        "NORMAL_TRADING",
        True,
        True,
        False,
        False,
        max(10.0, nav_usdt * 0.05),
        0.55,
        "balanced healthy trading",
    )


def compute_probability_score(
    *,
    signal_confidence: float,
    edge_score: float,
    market_fit: float,
    portfolio_fit: float,
    agent_quality: float,
    market_regime: str,
    risk_state: str,
    system_state: str,
) -> float:
    normalized_edge = max(0.0, min(1.0, abs(edge_score)))
    probability = (
        0.40 * max(0.0, min(1.0, signal_confidence))
        + 0.25 * normalized_edge
        + 0.15 * max(0.0, min(1.0, market_fit))
        + 0.10 * max(0.0, min(1.0, portfolio_fit))
        + 0.10 * max(0.0, min(1.0, agent_quality))
    )
    if market_regime == "VOLATILE":
        probability *= 0.85
    if risk_state == "DEFENSIVE":
        probability *= 0.90
    if system_state != "HEALTHY":
        probability *= 0.0
    return max(0.0, min(1.0, probability))


def default_telemetry() -> dict[str, Any]:
    return {
        "market_fit": 0.5,
        "portfolio_fit": 0.5,
        "agent_quality": 0.5,
    }


__all__ = [
    "Playbook",
    "classify_market_regime",
    "select_playbook",
    "compute_probability_score",
    "default_telemetry",
]
