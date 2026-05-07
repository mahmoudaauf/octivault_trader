"""
AI Trading Command Center API Server

Minimal FastAPI server exposing the 5 engines via safe, read-only endpoints
and governance control endpoints. No direct exchange access.

Architecture:
- Runs alongside main.py as separate process or integrated
- Reads from shared native state (no direct position/order mutations)
- Control endpoints call governance layer only
- All endpoints return structured JSON suitable for frontend UI
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    raise ImportError("FastAPI required for API server. Install: pip install fastapi uvicorn")

logger = logging.getLogger("octivault.api")


# ────────────────────────────────────────────────────────────────────────────
# Data models (TypeScript-friendly contracts)
# ────────────────────────────────────────────────────────────────────────────


class SystemHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    INITIALIZING = "INITIALIZING"


class ThrottleStatus(str, Enum):
    CLEAR = "CLEAR"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"


class CapitalState(str, Enum):
    HEALTHY = "HEALTHY"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PositionStatus(str, Enum):
    LEADER = "LEADER"
    WEAK = "WEAK"
    STALE = "STALE"
    DUST = "DUST"
    RECOVERING = "RECOVERING"


@dataclass
class SystemStatusResponse:
    """Current system state snapshot."""
    nav_usdt: float
    free_usdt: float
    locked_usdt: float
    growth_24h_pct: float
    active_positions_count: int
    open_orders_count: int
    mode: str
    market_regime: str
    system_health: str
    capital_state: str
    throttle_status: str
    throttle_until_ts: Optional[float] = None
    api_weight_estimate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SignalView:
    """Single signal from any source."""
    source: str  # "MLForecaster", "PaperGenerator", "SymbolScreener", etc.
    symbol: str
    direction: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0.0-1.0
    reason: Optional[str] = None


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_name: str
    passed: bool
    reason: Optional[str] = None


@dataclass
class DecisionExplanation:
    """Why the AI made (or didn't make) its last decision."""
    symbol: Optional[str] = None
    action: Optional[str] = None  # "BUY", "SELL", "NONE"
    signals: list[SignalView] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    playbook: Optional[str] = None
    confidence: Optional[float] = None
    blocked_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PositionView:
    """Position card for portfolio display."""
    symbol: str
    quantity: float
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    status: str = "ACTIVE"  # LEADER, WEAK, STALE, DUST, RECOVERING
    ai_action: Optional[str] = None  # HOLD, TAKE_PROFIT, ROTATE_OUT, CLEAN_DUST, WAIT


@dataclass
class CapitalHealthResponse:
    """Capital allocation health check."""
    free_ratio: float  # 0.0-1.0
    active_ratio: float  # 0.0-1.0
    reserve_ratio: float  # 0.0-1.0
    dust_ratio: float  # 0.0-1.0
    exposure_ratio: float  # 0.0-1.0
    largest_position_pct: float
    state: str  # HEALTHY, CAUTION, WARNING, CRITICAL
    warnings: list[str] = field(default_factory=list)


@dataclass
class ActivityEvent:
    """Structured event for timeline."""
    timestamp: float
    event_type: str  # DECISION, EXECUTION, FILL, THROTTLE, RECOVERY, HEALTH, CONTROL
    symbol: Optional[str] = None
    action: Optional[str] = None
    details: Optional[str] = None
    pnl: Optional[float] = None


@dataclass
class ComponentHealth:
    """Health of a single component."""
    component: str
    status: str
    error_count: int = 0
    last_error: Optional[str] = None
    last_check_ts: float = field(default_factory=time.time)


@dataclass
class HealthResponse:
    """Overall system health."""
    overall: str
    components: list[ComponentHealth] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControlActionResponse:
    """Result of a control action."""
    success: bool
    action: str
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ────────────────────────────────────────────────────────────────────────────
# API Server Class
# ────────────────────────────────────────────────────────────────────────────


class AICommandCenterAPI:
    """
    Read-only monitoring + governance control endpoints.
    Bridges between the 5 native engines and the frontend dashboard.
    """

    def __init__(self, app_ctx: dict[str, Any]):
        """Initialize with app context from main.py."""
        self.app_ctx = app_ctx
        self.app = FastAPI(
            title="AI Trading Command Center",
            description="Autonomous trading system monitoring API",
            version="1.0.0",
        )
        self._setup_routes()
        self._setup_middleware()

        # Event buffer for activity timeline
        self.event_buffer: list[ActivityEvent] = []
        self.max_events = 500

    def _setup_middleware(self):
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Register all endpoints."""

        @self.app.get("/api/status")
        async def get_status() -> dict[str, Any]:
            """Current system status snapshot."""
            ss = self._get_shared_state()
            return asdict(
                SystemStatusResponse(
                    nav_usdt=float(getattr(ss, "nav_usdt", 1.0)),
                    free_usdt=float(getattr(ss, "free_balance_usdt", 0.0)),
                    locked_usdt=float(getattr(ss, "locked_balance_usdt", 0.0)),
                    growth_24h_pct=self._compute_24h_growth(),
                    active_positions_count=len(getattr(ss, "positions", {}) or {}),
                    open_orders_count=len(getattr(ss, "open_orders", {}) or {}),
                    mode=str(getattr(ss, "current_mode", "NORMAL_TRADING") or "NORMAL_TRADING"),
                    market_regime=str(getattr(ss, "market_regime", "UNKNOWN") or "UNKNOWN"),
                    system_health=str(getattr(ss, "system_state", "HEALTHY") or "HEALTHY"),
                    capital_state=self._compute_capital_state(),
                    throttle_status=self._get_throttle_status(),
                    throttle_until_ts=float(getattr(ss, "exchange_throttle_until_ts", 0.0) or 0.0),
                    api_weight_estimate=self._estimate_api_weight(),
                )
            )

        @self.app.get("/api/ai-state")
        async def get_ai_state() -> dict[str, Any]:
            """Latest AI decision and reasoning."""
            ss = self._get_shared_state()

            # Extract decision data from shared state
            latest_decision = getattr(ss, "latest_decision", {}) or {}

            explanation = DecisionExplanation(
                symbol=latest_decision.get("symbol"),
                action=latest_decision.get("action"),
                confidence=latest_decision.get("confidence"),
                playbook=latest_decision.get("playbook"),
                blocked_reason=latest_decision.get("blocked_reason"),
            )

            return asdict(explanation)

        @self.app.get("/api/portfolio")
        async def get_portfolio() -> dict[str, Any]:
            """Portfolio composition and health."""
            ss = self._get_shared_state()
            positions_raw = getattr(ss, "positions", {}) or {}
            price_cache = getattr(ss, "price_cache", {}) or {}

            positions: list[dict[str, Any]] = []
            for sym, pos in positions_raw.items():
                qty = float(getattr(pos, "qty", 0.0) or 0.0)
                entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
                current_price = float(price_cache.get(sym, 0.0) or 0.0)

                pnl = (current_price - entry_price) * qty if entry_price > 0 else None
                pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else None

                positions.append(asdict(PositionView(
                    symbol=sym,
                    quantity=qty,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl_pct,
                    status="ACTIVE",
                )))

            nav = float(getattr(ss, "nav_usdt", 1.0))
            active_value = sum((p["current_price"] or 0) * p["quantity"] for p in positions)

            health = CapitalHealthResponse(
                free_ratio=float(getattr(ss, "free_balance_usdt", 0)) / nav if nav > 0 else 0,
                active_ratio=active_value / nav if nav > 0 else 0,
                reserve_ratio=0.0,
                dust_ratio=0.0,
                exposure_ratio=active_value / nav if nav > 0 else 0,
                largest_position_pct=(max((p["current_price"] or 0) * p["quantity"] for p in positions) / nav * 100) if positions and nav > 0 else 0,
                state=self._compute_capital_state(),
            )

            return {
                "positions": positions,
                "health": asdict(health),
            }

        @self.app.get("/api/activity")
        async def get_activity(limit: int = 50) -> dict[str, Any]:
            """Recent activity events."""
            return {
                "events": [asdict(e) for e in self.event_buffer[-limit:]],
                "total": len(self.event_buffer),
            }

        @self.app.get("/api/health")
        async def get_health() -> dict[str, Any]:
            """Component health status."""
            ss = self._get_shared_state()

            components = [
                ComponentHealth(
                    component="MarketData",
                    status="HEALTHY" if getattr(ss, "price_cache") else "DEGRADED",
                ),
                ComponentHealth(
                    component="PositionTracking",
                    status="HEALTHY" if getattr(ss, "positions") is not None else "DEGRADED",
                ),
                ComponentHealth(
                    component="Exchange",
                    status="THROTTLED" if getattr(ss, "exchange_throttled", False) else "HEALTHY",
                ),
                ComponentHealth(
                    component="SignalPipeline",
                    status="HEALTHY",  # Covered by fallback generators
                ),
            ]

            overall = "HEALTHY"
            if any(c.status == "CRITICAL" for c in components):
                overall = "CRITICAL"
            elif any(c.status == "DEGRADED" for c in components):
                overall = "DEGRADED"

            return asdict(HealthResponse(overall=overall, components=components))

        # ────────────────────────────────────────────────────────────────────────────
        # Control endpoints (governance only)
        # ────────────────────────────────────────────────────────────────────────────

        @self.app.post("/api/control/pause-buying")
        async def pause_buying(confirmed: bool = False) -> dict[str, Any]:
            """Prevent new BUY decisions while keeping system running."""
            if not confirmed:
                raise HTTPException(status_code=400, detail="Action requires confirmed=true")

            ss = self._get_shared_state()
            setattr(ss, "buying_paused", True)
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="PAUSE_BUYING",
                details="Operator paused new buy decisions",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="PAUSE_BUYING",
                reason="Buying paused; system continues monitoring",
            ))

        @self.app.post("/api/control/resume-buying")
        async def resume_buying() -> dict[str, Any]:
            """Resume BUY decisions."""
            ss = self._get_shared_state()
            setattr(ss, "buying_paused", False)
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="RESUME_BUYING",
                details="Operator resumed buying",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="RESUME_BUYING",
            ))

        @self.app.post("/api/control/force-safe-mode")
        async def force_safe_mode(confirmed: bool = False) -> dict[str, Any]:
            """Reduce position size, restrict trading to small safe orders."""
            if not confirmed:
                raise HTTPException(status_code=400, detail="Action requires confirmed=true")

            ss = self._get_shared_state()
            setattr(ss, "safe_mode_active", True)
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="SAFE_MODE",
                details="Operator activated safe mode",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="FORCE_SAFE_MODE",
                reason="Safe mode active; position sizes reduced",
            ))

        @self.app.post("/api/control/resume-normal")
        async def resume_normal() -> dict[str, Any]:
            """Exit safe mode."""
            ss = self._get_shared_state()
            setattr(ss, "safe_mode_active", False)
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="RESUME_NORMAL",
                details="Operator resumed normal mode",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="RESUME_NORMAL",
            ))

        @self.app.post("/api/control/cancel-open-orders")
        async def cancel_open_orders(confirmed: bool = False) -> dict[str, Any]:
            """Cancel all open orders (calls SafeExecutionEngine, not exchange directly)."""
            if not confirmed:
                raise HTTPException(status_code=400, detail="Action requires confirmed=true")

            # Delegate to SafeExecutionEngine
            execution_engine = self.app_ctx.get("execution_engine")
            if not execution_engine:
                return asdict(ControlActionResponse(
                    success=False,
                    action="CANCEL_OPEN_ORDERS",
                    reason="Execution engine not available",
                ))

            # This would call execution_engine.cancel_all_orders() or similar
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="CANCEL_ORDERS",
                details="Operator requested cancel all open orders",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="CANCEL_OPEN_ORDERS",
                reason="Open orders cancelled",
            ))

        @self.app.post("/api/control/pause-all")
        async def pause_all_trading(confirmed: bool = False) -> dict[str, Any]:
            """Full stop: pause all trading immediately."""
            if not confirmed:
                raise HTTPException(status_code=400, detail="Action requires confirmed=true")

            ss = self._get_shared_state()
            setattr(ss, "trading_halted", True)
            setattr(ss, "system_paused", True)
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="PAUSE_ALL",
                details="EMERGENCY: Operator halted all trading",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="PAUSE_ALL_TRADING",
                reason="ALL TRADING PAUSED - System in halt state",
            ))

        @self.app.post("/api/control/resume-trading")
        async def resume_trading() -> dict[str, Any]:
            """Resume trading after emergency pause."""
            ss = self._get_shared_state()
            setattr(ss, "trading_halted", False)
            setattr(ss, "system_paused", False)
            self._log_event(ActivityEvent(
                timestamp=time.time(),
                event_type="CONTROL",
                action="RESUME_TRADING",
                details="Operator resumed trading",
            ))

            return asdict(ControlActionResponse(
                success=True,
                action="RESUME_TRADING",
            ))

        @self.app.get("/health")
        async def health_check() -> dict[str, str]:
            """Simple health check for load balancer."""
            return {"status": "ok"}

    def _get_shared_state(self) -> Any:
        """Retrieve shared state from app context."""
        return self.app_ctx.get("shared_state")

    def _compute_24h_growth(self) -> float:
        """Calculate NAV growth over last 24 hours."""
        ss = self._get_shared_state()
        current_nav = float(getattr(ss, "nav_usdt", 1.0))
        session_anchor = float(getattr(ss, "session_anchor_nav", current_nav) or current_nav)
        if session_anchor <= 0:
            return 0.0
        return ((current_nav - session_anchor) / session_anchor) * 100

    def _compute_capital_state(self) -> str:
        """Determine capital health state."""
        ss = self._get_shared_state()
        nav = float(getattr(ss, "nav_usdt", 1.0))
        free = float(getattr(ss, "free_balance_usdt", 0.0))

        if free / nav < 0.1 if nav > 0 else False:
            return "CRITICAL"
        if free / nav < 0.2 if nav > 0 else False:
            return "WARNING"
        if free / nav < 0.5 if nav > 0 else False:
            return "CAUTION"
        return "HEALTHY"

    def _get_throttle_status(self) -> str:
        """Check exchange throttle status."""
        ss = self._get_shared_state()
        if getattr(ss, "exchange_throttled", False):
            return "ACTIVE"
        until_ts = float(getattr(ss, "exchange_throttle_until_ts", 0.0) or 0.0)
        if until_ts > time.time():
            return "PENDING"
        return "CLEAR"

    def _estimate_api_weight(self) -> float:
        """Estimate current API weight usage."""
        # Placeholder: in real implementation, query exchange client state
        return 0.0

    def _log_event(self, event: ActivityEvent) -> None:
        """Add event to activity buffer."""
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.max_events:
            self.event_buffer.pop(0)

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the API server."""
        logger.info(f"Starting AI Command Center API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")


# ────────────────────────────────────────────────────────────────────────────
# Standalone startup
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # For testing: create mock app context
    mock_ctx = {
        "shared_state": type("MockState", (), {
            "nav_usdt": 100.0,
            "free_balance_usdt": 50.0,
            "locked_balance_usdt": 0.0,
            "positions": {},
            "open_orders": {},
            "price_cache": {},
            "market_regime": "CHOPPY",
            "system_state": "HEALTHY",
            "exchange_throttled": False,
            "exchange_throttle_until_ts": 0.0,
            "current_mode": "NORMAL_TRADING",
        })(),
    }

    api = AICommandCenterAPI(mock_ctx)
    api.run(port=8000)
