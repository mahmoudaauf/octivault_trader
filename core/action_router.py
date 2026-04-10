"""
ActionRouter: Decision Governance Layer

The BRAIN of the trading system. Routes intents from multiple sources
(MetaController, LiquidationAgent, PortfolioBalancer) to ExecutionManager
with conflict resolution, priority system, and comprehensive audit trail.

Architecture:
    MetaController (signals)
         ↓
    Portfolio Engine (state + classification)
         ↓
    🧠 ActionRouter (THIS COMPONENT - Decision Governance)
         ↓
    ExecutionManager (execution)
         ↓
    Binance

Priority System:
    100 - Liquidation (capital/risk protection)
    90  - Risk Exit (loss limits)
    80  - TP/SL (position management)
    50  - MetaController/Agents (strategy signals)
    40  - Portfolio Balancer (rebalancing)
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class IntentSource(Enum):
    """Source of trading intent"""
    LIQUIDATION = "liquidation"        # LiquidationAgent
    RISK_EXIT = "risk_exit"           # RiskManager
    TP_SL = "tp_sl"                   # TPSLEngine
    META_CONTROLLER = "meta/controller"  # MetaController (strategy signals)
    AGENT = "meta/agent"              # Discovery agents
    PORTFOLIO_BALANCER = "portfolio/balancer"  # Rebalancing
    MANUAL = "manual"                 # Manual intervention


class IntentAction(Enum):
    """Action to take"""
    BUY = "BUY"
    SELL = "SELL"
    LIQUIDATE = "LIQUIDATE"  # Urgent sell


@dataclass
class TradeIntent:
    """Standard trade intent"""
    symbol: str
    action: IntentAction
    source: IntentSource
    quantity: Optional[float] = None
    priority: int = field(default=50)
    reason: str = ""
    confidence: float = 0.5
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.symbol, self.action.value, self.source.value))


class ConflictResolution(Enum):
    """How to resolve conflicting intents"""
    PRIORITY_WINS = "priority"      # Higher priority keeps, lower cancelled
    CANCEL_BOTH = "cancel"          # Both rejected
    QUEUE_SEQUENTIAL = "queue"      # Execute in priority order
    MERGE = "merge"                 # Combine compatible intents


@dataclass
class RoutingDecision:
    """Result of routing a single intent"""
    intent: TradeIntent
    decision: str  # "ACCEPTED", "REJECTED", "QUEUED"
    reason: str = ""
    replaced_intent: Optional[TradeIntent] = None
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())


class ActionRouter:
    """
    Decision Governance Layer

    Responsibilities:
    1. Validate intent sources
    2. Check for conflicts (same symbol, opposite actions)
    3. Apply priority system
    4. Forward to ExecutionManager
    5. Log all decisions for audit

    Design Principles:
    - Single source of truth for routing logic
    - Explicit priority system (no implicit ordering)
    - Comprehensive conflict detection
    - Complete audit trail
    - Non-blocking (errors don't crash system)
    """

    # Priority system (higher = more urgent)
    PRIORITY = {
        IntentSource.LIQUIDATION: 100,        # Capital/risk protection (highest)
        IntentSource.RISK_EXIT: 90,           # Loss limits
        IntentSource.TP_SL: 80,               # Position management
        IntentSource.META_CONTROLLER: 50,     # Strategy signals
        IntentSource.AGENT: 50,               # Discovery agents
        IntentSource.PORTFOLIO_BALANCER: 40,  # Rebalancing
        IntentSource.MANUAL: 95,              # Manual (just below liquidation)
    }

    # Conflict resolution modes per source
    CONFLICT_MODES = {
        IntentSource.LIQUIDATION: ConflictResolution.PRIORITY_WINS,  # Always wins
        IntentSource.RISK_EXIT: ConflictResolution.PRIORITY_WINS,
        IntentSource.TP_SL: ConflictResolution.PRIORITY_WINS,
        IntentSource.META_CONTROLLER: ConflictResolution.PRIORITY_WINS,
        IntentSource.AGENT: ConflictResolution.PRIORITY_WINS,
        IntentSource.PORTFOLIO_BALANCER: ConflictResolution.PRIORITY_WINS,
        IntentSource.MANUAL: ConflictResolution.PRIORITY_WINS,
    }

    def __init__(self, shared_state=None, execution_manager=None):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # State tracking
        self._active_intents: Dict[str, List[TradeIntent]] = defaultdict(list)  # By symbol
        self._intent_history: List[RoutingDecision] = []
        self._conflicts_resolved: int = 0
        self._intents_accepted: int = 0
        self._intents_rejected: int = 0
        self._lock = asyncio.Lock()

    def _intent_action_name(self, intent: Any) -> str:
        action = getattr(intent, "action", None)
        if isinstance(action, IntentAction):
            return action.value
        if action is not None:
            return str(action).upper()
        side = getattr(intent, "side", None)
        if side is not None:
            return str(side).upper()
        return ""

    def _intent_source(self, intent: Any) -> IntentSource:
        source = getattr(intent, "source", None)
        if isinstance(source, IntentSource):
            return source

        policy_ctx = getattr(intent, "policy_context", None)
        policy_ctx = policy_ctx if isinstance(policy_ctx, dict) else {}
        agent_name = str(getattr(intent, "agent", "") or "").strip().lower()
        tag = str(getattr(intent, "tag", "") or "").strip().lower()
        authority = str(
            policy_ctx.get("authority")
            or policy_ctx.get("governor")
            or policy_ctx.get("policy_authority")
            or ""
        ).strip().lower()

        if authority == "metacontroller" or "meta" in tag:
            return IntentSource.META_CONTROLLER
        if "liquidat" in tag or authority == "liquidation":
            return IntentSource.LIQUIDATION
        if "risk" in tag or authority == "risk":
            return IntentSource.RISK_EXIT
        if "tp" in tag or "sl" in tag:
            return IntentSource.TP_SL
        if "balanc" in tag:
            return IntentSource.PORTFOLIO_BALANCER
        if "manual" in tag or authority == "manual":
            return IntentSource.MANUAL
        if agent_name:
            return IntentSource.AGENT
        return IntentSource.META_CONTROLLER

    def _intent_quantity(self, intent: Any) -> Optional[float]:
        for field_name in ("quantity", "qty_hint"):
            value = getattr(intent, field_name, None)
            if value is None:
                continue
            try:
                qty = float(value)
            except Exception:
                continue
            if qty > 0:
                return qty
        return None

    def _intent_budget(self, intent: Any) -> Optional[float]:
        for field_name in ("planned_quote", "quote_hint"):
            value = getattr(intent, field_name, None)
            if value is None:
                continue
            try:
                quote = float(value)
            except Exception:
                continue
            if quote > 0:
                return quote
        return None

    def _intent_priority(self, intent: Any) -> int:
        return self._get_priority(self._intent_source(intent))

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    async def route(self, intent: TradeIntent) -> RoutingDecision:
        """
        Main entry point: Route a single intent

        Returns: RoutingDecision with accepted/rejected status
        """
        async with self._lock:
            decision = await self._route_internal(intent)
            self._intent_history.append(decision)

            # Update metrics
            if decision.decision == "ACCEPTED":
                self._intents_accepted += 1
            elif decision.decision == "REJECTED":
                self._intents_rejected += 1

            # Log decision
            self._log_routing_decision(decision)

            return decision

    async def route_batch(self, intents: List[TradeIntent]) -> List[RoutingDecision]:
        """
        Route multiple intents efficiently

        Returns: List of RoutingDecisions
        """
        decisions = []
        for intent in intents:
            decision = await self.route(intent)
            decisions.append(decision)
        return decisions

    def get_active_intents(self, symbol: Optional[str] = None) -> Dict[str, List[TradeIntent]]:
        """Get currently active/queued intents"""
        if symbol:
            return {symbol: self._active_intents.get(symbol, [])}
        return dict(self._active_intents)

    def get_routing_history(self, limit: int = 100) -> List[RoutingDecision]:
        """Get recent routing decisions"""
        return self._intent_history[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        return {
            "intents_accepted": self._intents_accepted,
            "intents_rejected": self._intents_rejected,
            "conflicts_resolved": self._conflicts_resolved,
            "active_intents_count": sum(len(v) for v in self._active_intents.values()),
            "active_symbols": list(self._active_intents.keys()),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Internal Routing Logic
    # ═══════════════════════════════════════════════════════════════════

    async def _route_internal(self, intent: TradeIntent) -> RoutingDecision:
        """
        Internal routing logic with full validation

        Steps:
        1. Validate source
        2. Check for conflicts with active intents
        3. Apply priority system
        4. Forward to ExecutionManager (if accepted)
        5. Update active intents tracking
        """
        try:
            # Step 1: Validate source
            validation_error = self._validate_intent(intent)
            if validation_error:
                return RoutingDecision(
                    intent=intent,
                    decision="REJECTED",
                    reason=validation_error,
                )

            # Step 2: Check conflicts
            conflict_result = await self._check_conflicts(intent)
            if not conflict_result["can_route"]:
                return RoutingDecision(
                    intent=intent,
                    decision="REJECTED",
                    reason=conflict_result["reason"],
                    replaced_intent=conflict_result.get("replaced_intent"),
                )

            # Step 3: Apply priority & get slot
            replaced_intent = conflict_result.get("replaced_intent")
            if replaced_intent:
                self._active_intents[intent.symbol].remove(replaced_intent)
                self._conflicts_resolved += 1

            # Step 4: Update tracking. Execution is owned by MetaController after governance.
            self._active_intents[intent.symbol].append(intent)

            return RoutingDecision(
                intent=intent,
                decision="ACCEPTED",
                reason=f"Priority {self._intent_priority(intent)}",
            )

        except Exception as e:
            self.logger.error(f"[ActionRouter] Routing error: {e}", exc_info=True)
            return RoutingDecision(
                intent=intent,
                decision="REJECTED",
                reason=f"Router error: {str(e)}",
            )

    async def _check_conflicts(self, intent: TradeIntent) -> Dict[str, Any]:
        """
        Check for conflicting intents on same symbol

        Conflict = same symbol, opposite action (BUY vs SELL)

        Returns:
            {
                "can_route": bool,
                "reason": str,
                "replaced_intent": Optional[TradeIntent],
            }
        """
        active = self._active_intents.get(intent.symbol, [])
        if not active:
            return {"can_route": True}

        for existing in active:
            # Check for opposite action
            is_opposite = (
                (self._intent_action_name(intent) == "BUY" and self._intent_action_name(existing) == "SELL") or
                (self._intent_action_name(intent) == "SELL" and self._intent_action_name(existing) == "BUY") or
                (self._intent_action_name(intent) == "LIQUIDATE")
            )

            if not is_opposite:
                # Same action, not a conflict
                continue

            # We have a conflict - apply resolution logic
            new_priority = self._intent_priority(intent)
            existing_priority = self._intent_priority(existing)

            if new_priority > existing_priority:
                # New intent wins
                return {
                    "can_route": True,
                    "reason": f"Higher priority ({new_priority} > {existing_priority})",
                    "replaced_intent": existing,
                }
            else:
                # Existing intent wins
                return {
                    "can_route": False,
                    "reason": f"Conflict: {self._intent_source(existing).value} already active (priority {existing_priority} > {new_priority})",
                    "replaced_intent": None,
                }

        return {"can_route": True}

    def _validate_intent(self, intent: TradeIntent) -> Optional[str]:
        """
        Validate intent structure

        Returns error message if invalid, None if valid
        """
        if not getattr(intent, "symbol", None):
            return "Missing symbol"

        action = self._intent_action_name(intent)
        if action not in {"BUY", "SELL", "LIQUIDATE"}:
            return f"Invalid action: {action or None}"

        if action in {"BUY", "SELL"}:
            qty = self._intent_quantity(intent)
            budget = self._intent_budget(intent)
            if (qty is None or qty <= 0) and (budget is None or budget <= 0):
                return f"Invalid size: quantity={qty} planned_quote={budget}"

        try:
            confidence = float(getattr(intent, "confidence", 0.0) or 0.0)
        except Exception:
            return f"Invalid confidence: {getattr(intent, 'confidence', None)}"
        if not 0.0 <= confidence <= 1.0:
            return f"Invalid confidence: {confidence}"

        return None  # Valid

    def _get_priority(self, source: IntentSource) -> int:
        """Get priority value for source"""
        return self.PRIORITY.get(source, 0)

    def _log_routing_decision(self, decision: RoutingDecision):
        """Log routing decision with full context"""
        self.logger.info(
            "[ACTION_ROUTER] symbol=%s action=%s source=%s decision=%s reason=%s priority=%d",
            decision.intent.symbol,
            self._intent_action_name(decision.intent),
            self._intent_source(decision.intent).value,
            decision.decision,
            decision.reason,
            self._intent_priority(decision.intent),
        )

    # ═══════════════════════════════════════════════════════════════════
    # Diagnostic API
    # ═══════════════════════════════════════════════════════════════════

    async def get_status(self) -> Dict[str, Any]:
        """Get full router status"""
        return {
            "status": "operational",
            "metrics": self.get_metrics(),
            "active_intents": {
                symbol: [
                    {
                        "action": self._intent_action_name(intent),
                        "source": self._intent_source(intent).value,
                        "priority": self._intent_priority(intent),
                        "confidence": float(getattr(intent, "confidence", 0.0) or 0.0),
                        "timestamp": intent.timestamp,
                    }
                    for intent in intents
                ]
                for symbol, intents in self._active_intents.items()
            },
            "recent_decisions": [
                {
                    "symbol": d.intent.symbol,
                    "decision": d.decision,
                    "reason": d.reason,
                    "timestamp": d.timestamp,
                }
                for d in self._intent_history[-10:]
            ],
        }

    async def clear_symbol_intents(self, symbol: str):
        """
        Force clear all intents for a symbol (use with caution)
        Useful after fatal errors or manual intervention
        """
        async with self._lock:
            count = len(self._active_intents.get(symbol, []))
            if count > 0:
                self.logger.warning(
                    "[ACTION_ROUTER] Force clearing %d intents for %s",
                    count,
                    symbol,
                )
                self._active_intents[symbol] = []
            return count
