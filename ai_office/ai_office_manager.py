from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.l0_core.stubs import maybe_call

from .liquidity_supervisor_agent import LiquiditySupervisorAgent
from .performance_analyst_agent import PerformanceAnalystAgent
from .policy_bus import PolicyBus
from .risk_auditor_agent import RiskAuditorAgent
from .schemas import (
    AIOfficeDecision,
    AIOfficeHealth,
    AIOfficeRecommendation,
    AIOfficeSnapshot,
)

logger = logging.getLogger("AIOffice.Manager")


class AIOfficeManager:
    def __init__(
        self,
        *,
        shared_state: Any,
        portfolio_manager: Any | None = None,
        performance_monitor: Any | None = None,
        risk_manager: Any | None = None,
        strategy_manager: Any | None = None,
        liquidation_agent: Any | None = None,
        portfolio_balancer: Any | None = None,
        review_interval_sec: int = 28800,
    ) -> None:
        self._ss = shared_state
        self._pm = portfolio_manager
        self._perf = performance_monitor
        self._risk = risk_manager
        self._strategy = strategy_manager
        self._liq = liquidation_agent
        self._balancer = portfolio_balancer
        self._interval = max(60, int(review_interval_sec or 28800))
        self._running = False
        self._task: asyncio.Task | None = None
        self._bus = PolicyBus(shared_state)
        self._agents = [
            PerformanceAnalystAgent(),
            RiskAuditorAgent(),
            LiquiditySupervisorAgent(),
        ]

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="ai_office_manager")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._emit_summary("AI_OFFICE_LOOP_START", {"interval_sec": self._interval})
                snapshot = await self.build_snapshot()
                await self._emit_summary("AI_OFFICE_SNAPSHOT_BUILT", snapshot.to_dict())
                recommendations = []
                for agent in self._agents:
                    rec = await agent.analyze(snapshot)
                    recommendations.append(rec)
                    await self._emit_summary("AI_OFFICE_RECOMMENDATION", rec.to_dict())
                    decision = await self._bus.validate(rec)
                    if not decision.accepted:
                        continue
                    await self._route(decision)
                await self._emit_health(
                    AIOfficeHealth(
                        source="AIOfficeManager",
                        status="OK",
                        reason="loop_completed",
                        metadata={"recommendations": len(recommendations)},
                    )
                )
                await self._emit_summary("AI_OFFICE_LOOP_DONE", {"recommendations": len(recommendations)})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("AI_OFFICE_ERROR %s", exc)
                await self._emit_summary("AI_OFFICE_ERROR", {"error": repr(exc)})
                await self._emit_health(
                    AIOfficeHealth(
                        source="AIOfficeManager",
                        status="ERROR",
                        reason=repr(exc),
                        severity="ERROR",
                    )
                )
            await asyncio.sleep(self._interval)

    async def build_snapshot(self) -> AIOfficeSnapshot:
        metrics = dict(getattr(self._ss, "metrics", {}) or {})
        nav = float(getattr(self._ss, "nav_usdt", 0.0) or 0.0)
        peak_nav = float(getattr(self._ss, "peak_nav_usdt", nav) or nav)
        free_usdt = float(getattr(self._ss, "free_balance_usdt", 0.0) or 0.0)
        locked_usdt = max(0.0, nav - free_usdt)
        recovery = dict(getattr(self._ss, "position_recovery", {}) or {})
        positions_count = len(getattr(self._ss, "positions", {}) or {})
        stale_positions = sum(1 for p in recovery.values() if str(p.get("status", "")).upper() == "STALE")
        weak_positions = sum(1 for p in recovery.values() if str(p.get("status", "")).upper() == "WEAK")
        dust_value = sum(
            float(p.get("notional_usdt", 0.0) or 0.0)
            for p in recovery.values()
            if str(p.get("status", "")).upper() == "DUST"
        )
        # Compute concentration_ratio from live positions rather than relying on
        # last_nav_attribution which is rarely populated.
        _positions = dict(getattr(self._ss, "positions", {}) or {})
        _prices = dict(getattr(self._ss, "prices", {}) or {})
        if _positions and nav > 0:
            _notionals = []
            for _sym, _pos in _positions.items():
                _qty = float(
                    getattr(_pos, "qty", None)
                    if not isinstance(_pos, dict)
                    else _pos.get("qty", 0) or 0
                ) or 0.0
                _price = float(_prices.get(_sym, 0.0) or 0.0)
                if _qty > 0 and _price > 0:
                    _notionals.append(_qty * _price)
            largest_contributor = max(_notionals) / nav if _notionals else 0.0
        else:
            largest_contributor = 0.0
        drawdown = 0.0
        if peak_nav > 0:
            drawdown = max(0.0, (peak_nav - nav) / peak_nav)
        return AIOfficeSnapshot(
            source="AIOfficeManager",
            nav_usdt=nav,
            previous_nav_usdt=float(getattr(self._ss, "previous_nav_usdt", nav) or nav),
            peak_nav_usdt=peak_nav,
            realized_pnl_usdt=float(metrics.get("realized_pnl", 0.0) or 0.0),
            unrealized_pnl_usdt=float(metrics.get("unrealized_pnl", 0.0) or 0.0),
            free_usdt=free_usdt,
            locked_usdt=locked_usdt,
            capital_locked_ratio=(locked_usdt / nav) if nav > 0 else 0.0,
            dust_ratio=(dust_value / nav) if nav > 0 else 0.0,
            positions_count=positions_count,
            stale_positions_count=stale_positions,
            weak_positions_count=weak_positions,
            concentration_ratio=largest_contributor,
            drawdown_from_peak_pct=drawdown,
            market_regime=str(metrics.get("trend_regime", "UNKNOWN") or "UNKNOWN"),
            health_status=str(getattr(self._ss, "system_state", "NORMAL") or "NORMAL"),
            metadata={
                "performance_monitor_available": bool(self._perf),
                "risk_manager_available": bool(self._risk),
                "strategy_manager_available": bool(self._strategy),
                "liquidation_agent_available": bool(self._liq),
                "portfolio_balancer_available": bool(self._balancer),
            },
        )

    async def _route(self, decision: AIOfficeDecision) -> None:
        action = decision.suggested_action
        target = decision.target_component
        routed = False

        # --- SharedState-level routing (always available if _ss is set) ---
        if self._ss is not None:
            if action == "tighten_tp_sl":
                # Try calling tp_sl_engine.tighten_all if available, else write metric
                _tpsl = getattr(self._ss, "_app_ctx", {}) if hasattr(self._ss, "_app_ctx") else {}
                _tpsl_eng = _tpsl.get("tp_sl_engine") if isinstance(_tpsl, dict) else None
                if _tpsl_eng is not None and hasattr(_tpsl_eng, "tighten_all"):
                    try:
                        _tpsl_eng.tighten_all(factor=0.8)
                    except Exception:
                        pass
                _m = getattr(self._ss, "metrics", None)
                if _m is not None:
                    _m["tpsl_tighten_factor"] = 0.8
                routed = True
                logger.warning("AI_OFFICE: tighten_tp_sl action applied (factor=0.8)")
            elif action == "reduce_exposure":
                _m = getattr(self._ss, "metrics", None)
                if _m is not None:
                    _m["ai_office_reduce_exposure"] = True
                routed = True
                logger.warning("AI_OFFICE: reduce_exposure — blocking new BUYs via ai_office_reduce_exposure flag")
            elif action in {"pause_new_buys", "freeze_new_buys"}:
                _m = getattr(self._ss, "metrics", None)
                if _m is not None:
                    _m["ai_office_pause_buys"] = True
                routed = True
                logger.warning("AI_OFFICE: %s — new BUYs paused via ai_office_pause_buys flag", action)
            elif action == "resume_new_buys":
                _m = getattr(self._ss, "metrics", None)
                if _m is not None:
                    _m["ai_office_pause_buys"] = False
                    _m["ai_office_reduce_exposure"] = False
                routed = True
                logger.info("AI_OFFICE: resume_new_buys — BUY pause flag cleared")
            elif action == "request_risk_review":
                _m = getattr(self._ss, "metrics", None)
                if _m is not None:
                    _m["ai_office_risk_review_requested"] = True
                    _m["ai_office_risk_review_reason"] = str(
                        (decision.metadata or {}).get("reason", decision.reason or "governance")
                    )
                routed = True
                logger.warning(
                    "AI_OFFICE: request_risk_review — governance alert raised, reason=%s",
                    (decision.metadata or {}).get("reason", decision.reason),
                )

        if target == "RiskManager" and self._risk is not None:
            if action == "pause_new_buys" and hasattr(self._risk, "freeze_trading"):
                self._risk.freeze_trading("AI Office pause_new_buys")
                routed = True
            elif action == "resume_new_buys" and hasattr(self._risk, "unfreeze_trading"):
                self._risk.unfreeze_trading()
                routed = True
        elif target == "StrategyManager" and self._strategy is not None:
            if action == "adjust_agent_weight" and hasattr(self._strategy, "set_weight"):
                meta = dict(decision.metadata or {})
                agent = str(meta.get("agent", "") or "")
                weight = meta.get("weight")
                if agent and weight is not None:
                    await maybe_call(self._strategy, "set_weight", agent, weight)
                    routed = True
        elif target == "PortfolioBalancer" and self._balancer is not None:
            if action in {"request_dust_cleanup", "request_liquidity"} and hasattr(self._balancer, "run_once"):
                await maybe_call(self._balancer, "run_once")
                routed = True
        elif target == "LiquidationAgent" and self._liq is not None:
            if action == "request_liquidity" and hasattr(self._liq, "propose_liquidations"):
                await maybe_call(self._liq, "propose_liquidations", None, {})
                routed = True

        await self._emit_summary(
            "AI_OFFICE_ACTION_ROUTED",
            {
                "trace_id": decision.trace_id,
                "target_component": target,
                "suggested_action": action,
                "routed": routed,
                "reason": decision.reason,
            },
        )

    async def _emit_summary(self, event: str, payload: dict[str, Any]) -> None:
        body = {"event": event, **dict(payload or {})}
        logger.info("%s %s", event, body)
        await maybe_call(self._ss, "emit_event", "SummaryLog", body)

    async def _emit_health(self, health: AIOfficeHealth) -> None:
        await maybe_call(self._ss, "emit_event", "HealthStatus", health.to_dict())
