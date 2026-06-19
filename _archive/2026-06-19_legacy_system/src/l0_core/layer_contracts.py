# core/layer_contracts.py
# Professional Three-Layer Capital Accounting Architecture
# Formal contracts and boundaries for Wallet Layer → Portfolio Layer → Strategy Layer

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("LayerContracts")


class LayerName(Enum):
    """
    Layer hierarchy.

    Legacy 3-layer aliases (WALLET_LAYER / PORTFOLIO_LAYER / STRATEGY_LAYER) are
    kept for backwards compatibility. The authoritative model is the 8-layer
    stack defined in LOGICAL_LAYERED_ARCHITECTURE.md (L0–L8).
    """

    # Legacy 3-layer names (kept for backwards compat)
    WALLET_LAYER = "WALLET_LAYER"  # → L2
    PORTFOLIO_LAYER = "PORTFOLIO_LAYER"  # → L3
    STRATEGY_LAYER = "STRATEGY_LAYER"  # → L5

    # Authoritative 8-layer names
    L0_CROSS_CUTTING = "L0_CROSS_CUTTING"
    L1_EXCHANGE_IO = "L1_EXCHANGE_IO"
    L2_WALLET_MARKETDATA = "L2_WALLET_MARKETDATA"
    L3_PORTFOLIO_STATE = "L3_PORTFOLIO_STATE"
    L4_EXECUTION = "L4_EXECUTION"
    L5_STRATEGY = "L5_STRATEGY"
    L6_GOVERNANCE = "L6_GOVERNANCE"
    L7_OBSERVABILITY = "L7_OBSERVABILITY"
    L8_LIFECYCLE = "L8_LIFECYCLE"


@dataclass
class LayerInput:
    """Contract: What data a layer receives as input."""

    source_layer: str  # Which layer provides this input
    timestamp: float  # When data was generated
    data: dict[str, Any]  # Payload
    validated: bool = True  # Input meets contract requirements


@dataclass
class LayerOutput:
    """Contract: What data a layer produces as output."""

    source_layer: str  # Which layer produced this output
    timestamp: float  # When output was generated
    data: dict[str, Any]  # Payload
    verified: bool = True  # Output meets contract requirements


class WalletLayerContract:
    """
    Layer 1: Wallet Synchronization Contract

    Input (from Exchange):
      - Raw account balances: {asset: {free: float, locked: float}}
      - Raw open positions: [{symbol, quantity, avg_price, ...}]

    Responsibilities:
      - Periodically sync wallet balances from exchange
      - Periodically sync open positions from exchange
      - Classify assets (EXTERNAL_POSITION, STABLE, DUST)
      - Maintain wallet_snapshot for emergency access

    Output (to Portfolio Layer):
      - Classified wallet assets: {asset: ClassifiedPosition}
      - Exchange-verified positions: {symbol: Position}
      - wallet_last_updated: timestamp

    Invariants:
      - All balances come directly from exchange (no computed values)
      - All positions are from exchange_open_positions (verified trades)
      - EXTERNAL_POSITION is never modified by bot (read-only)
      - STABLE assets maintain accurate quantities

    Error Handling:
      - Failed sync: retry with exponential backoff
      - Exchange timeout: use last known snapshot
      - Data corruption: log and skip corrupted records
    """

    def __init__(self):
        self.logger = logging.getLogger("WalletLayerContract")
        self.contract_name = "WalletLayerContract"
        self.required_output_fields = {
            "assets": dict,  # {symbol: ClassifiedPosition}
            "positions": dict,  # {symbol: Position}
            "last_updated": float,  # Timestamp of last sync
        }

    def validate_output(self, output: dict[str, Any]) -> bool:
        """Verify Layer 1 output meets contract requirements."""
        try:
            # Check required fields exist
            for field, expected_type in self.required_output_fields.items():
                if field not in output:
                    self.logger.error(f"[{self.contract_name}] Missing required field: {field}")
                    return False
                if not isinstance(output[field], expected_type):
                    self.logger.error(
                        f"[{self.contract_name}] Field {field} has wrong type: "
                        f"expected {expected_type}, got {type(output[field])}"
                    )
                    return False

            # Validate asset classifications
            assets = output.get("assets", {})
            for symbol, asset_data in assets.items():
                if "classification" not in asset_data:
                    self.logger.warning(
                        f"[{self.contract_name}] Asset {symbol} missing classification"
                    )
                    return False
                if asset_data["classification"] not in (
                    "BOT_POSITION",
                    "EXTERNAL_POSITION",
                    "STABLE",
                    "DUST",
                ):
                    self.logger.error(
                        f"[{self.contract_name}] Invalid classification for {symbol}: "
                        f"{asset_data.get('classification')}"
                    )
                    return False

            self.logger.info(
                f"[{self.contract_name}] Output validation PASSED "
                f"(assets={len(assets)}, positions={len(output.get('positions', {}))})"
            )
            return True

        except Exception as e:
            self.logger.error(f"[{self.contract_name}] Validation error: {e}", exc_info=True)
            return False


class PortfolioLayerContract:
    """
    Layer 2: Portfolio Management Contract

    Input (from Wallet Layer):
      - Classified wallet assets
      - Exchange-verified positions
      - wallet_last_updated timestamp

    Input (from Strategy Layer):
      - Trade execution results
      - Position updates (opened, closed, liquidated)

    Responsibilities:
      - Maintain authoritative position registry
      - Compute Net Asset Value (NAV)
      - Classify all positions (BOT_POSITION, EXTERNAL_POSITION, DUST, STABLE)
      - Detect and track dust positions
      - Enforce capital accounting rules (double-entry bookkeeping)

    Output (to Strategy Layer):
      - Current portfolio: {symbol: Position with classification}
      - Portfolio NAV: Total value in quote currency
      - Capital available: USDT/quote ready for new trades
      - Risk metrics: exposure, concentration, etc.

    Output (to Wallet Layer feedback):
      - Position status updates
      - Rebalancing suggestions (optional)

    Invariants:
      - Sum of all positions = NAV (double-entry check)
      - EXTERNAL_POSITION quantity never changes (read-only)
      - Classification is deterministic and consistent
      - All positions have origin, created_at, created_by_agent
      - Dust positions tracked separately for cleanup

    Error Handling:
      - Missing price: use last known price or mark price
      - Malformed position: log and skip
      - Classification conflicts: resolve by origin (exchange > trade history)
    """

    def __init__(self):
        self.logger = logging.getLogger("PortfolioLayerContract")
        self.contract_name = "PortfolioLayerContract"
        self.required_output_fields = {
            "portfolio": dict,  # {symbol: Position}
            "nav": float,  # Net asset value
            "capital_available": float,  # Quote available for trades
            "risk_metrics": dict,  # exposure, concentration, etc.
            "dust_positions": dict,  # Positions flagged as dust
            "last_computed": float,  # Timestamp of computation
        }

    def validate_output(self, output: dict[str, Any]) -> bool:
        """Verify Layer 2 output meets contract requirements."""
        try:
            # Check required fields
            for field, expected_type in self.required_output_fields.items():
                if field not in output:
                    self.logger.error(f"[{self.contract_name}] Missing required field: {field}")
                    return False
                if not isinstance(output[field], expected_type):
                    self.logger.error(
                        f"[{self.contract_name}] Field {field} has wrong type: "
                        f"expected {expected_type}, got {type(output[field])}"
                    )
                    return False

            # Validate position data
            portfolio = output.get("portfolio", {})
            for symbol, position in portfolio.items():
                required_pos_fields = {"quantity", "price", "classification", "origin"}
                for field in required_pos_fields:
                    if field not in position:
                        self.logger.error(
                            f"[{self.contract_name}] Position {symbol} missing {field}"
                        )
                        return False

            # Validate NAV is non-negative
            nav = output.get("nav", 0.0)
            if nav < 0:
                self.logger.warning(
                    f"[{self.contract_name}] NAV is negative: {nav} (possible liquidation)"
                )

            # Validate capital_available is reasonable
            capital = output.get("capital_available", 0.0)
            if capital < 0:
                self.logger.error(
                    f"[{self.contract_name}] Capital available is negative: {capital}"
                )
                return False

            self.logger.info(
                f"[{self.contract_name}] Output validation PASSED "
                f"(portfolio={len(portfolio)}, nav={nav:.2f}, capital={capital:.2f})"
            )
            return True

        except Exception as e:
            self.logger.error(f"[{self.contract_name}] Validation error: {e}", exc_info=True)
            return False


class StrategyLayerContract:
    """
    Layer 3: Strategy & Trading Contract

    Input (from Portfolio Layer):
      - Current portfolio with classifications
      - Portfolio NAV
      - Capital available
      - Risk metrics

    Responsibilities:
      - Analyze market signals (agents)
      - Generate trade recommendations
      - Execute trades (BUY/SELL) on BOT_POSITION assets
      - Never touch EXTERNAL_POSITION (read-only)
      - Respect capital limits and risk constraints
      - Generate audit trail of all decisions

    Output (to Portfolio Layer):
      - Trade execution results: {symbol: {filled_qty, filled_price, timestamp}}
      - Position lifecycle events: opened, closed, liquidated
      - PnL updates: realized_pnl, unrealized_pnl

    Outputs (to external):
      - Trade audit log (for compliance)
      - Performance metrics (PnL, Sharpe, drawdown, etc.)

    Invariants:
      - Total open positions cannot exceed capital limit
      - Concentration per symbol limited (e.g., max 20% of NAV)
      - Only BOT_POSITION assets can be traded (exit by strategy)
      - EXTERNAL_POSITION can never be traded (only displayed)
      - Every trade has clear audit trail: agent, signal, timestamp, reason

    Error Handling:
      - Rejected by validator: log reason and skip
      - Insufficient capital: queue for next cycle
      - Network error: retry with circuit breaker
      - Position lock: skip (already in operation)
    """

    def __init__(self):
        self.logger = logging.getLogger("StrategyLayerContract")
        self.contract_name = "StrategyLayerContract"
        self.required_output_fields = {
            "trades": list,  # [{symbol, side, qty, price, timestamp}]
            "pnl": dict,  # {realized, unrealized, total}
            "audit_log": list,  # Trade audit entries
            "execution_timestamp": float,  # When trades were executed
        }

    def validate_operation(
        self, operation_type: str, symbol: str, classification: Optional[str], quantity: float
    ) -> bool:
        """
        Pre-operation validation.

        Args:
          - operation_type: "ENTRY" (BUY) or "EXIT" (SELL)
          - symbol: Trading symbol
          - classification: Position classification from portfolio
          - quantity: Quantity to trade

        Returns: True if operation is allowed
        """
        try:
            # Entry (BUY) restrictions
            if operation_type == "ENTRY":
                # Can't entry on EXTERNAL_POSITION
                if classification == "EXTERNAL_POSITION":
                    self.logger.warning(
                        f"[{self.contract_name}] Cannot entry on EXTERNAL_POSITION: {symbol}"
                    )
                    return False

                # Can entry on new positions, BOT_POSITION, or DUST
                if classification in (None, "BOT_POSITION", "DUST"):
                    return True

            # Exit (SELL) restrictions
            elif operation_type == "EXIT":
                # Can exit BOT_POSITION
                if classification == "BOT_POSITION":
                    return True

                # Cannot exit EXTERNAL_POSITION (user holding)
                if classification == "EXTERNAL_POSITION":
                    self.logger.warning(
                        f"[{self.contract_name}] Cannot exit EXTERNAL_POSITION: {symbol}"
                    )
                    return False

                # Can exit DUST
                if classification == "DUST":
                    return True

            return False

        except Exception as e:
            self.logger.error(f"[{self.contract_name}] Validation error: {e}", exc_info=True)
            return False


class LayerContractManager:
    """
    Manager for all three-layer contracts.
    Provides unified validation and enforcement.
    """

    def __init__(self):
        self.logger = logging.getLogger("LayerContractManager")
        self.wallet_contract = WalletLayerContract()
        self.portfolio_contract = PortfolioLayerContract()
        self.strategy_contract = StrategyLayerContract()

    def validate_wallet_output(self, output: dict[str, Any]) -> bool:
        """Validate Wallet Layer output."""
        return self.wallet_contract.validate_output(output)

    def validate_portfolio_output(self, output: dict[str, Any]) -> bool:
        """Validate Portfolio Layer output."""
        return self.portfolio_contract.validate_output(output)

    def validate_strategy_operation(
        self, operation_type: str, symbol: str, classification: Optional[str], quantity: float
    ) -> bool:
        """Validate Strategy Layer operation."""
        return self.strategy_contract.validate_operation(
            operation_type, symbol, classification, quantity
        )


# ============================================================================
# 8-LAYER MODEL — Authoritative contracts (see LOGICAL_LAYERED_ARCHITECTURE.md)
# ============================================================================
# Aliases that map the legacy 3-layer contracts onto the 8-layer model:
#   WalletLayerContract     -> L2WalletContract
#   PortfolioLayerContract  -> L3PortfolioContract
#   StrategyLayerContract   -> L5StrategyContract

L2WalletContract = WalletLayerContract
L3PortfolioContract = PortfolioLayerContract
L5StrategyContract = StrategyLayerContract


class _BaseLayerContract:
    """Common skeleton for layer contracts (L1, L4, L6, L7, L8)."""

    contract_name: str = "BaseLayerContract"
    required_output_fields: dict[str, type] = {}

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.contract_name)

    def validate_output(self, output: dict[str, Any]) -> bool:
        try:
            for field, expected_type in self.required_output_fields.items():
                if field not in output:
                    self.logger.error(f"[{self.contract_name}] Missing required field: {field}")
                    return False
                if not isinstance(output[field], expected_type):
                    self.logger.error(
                        f"[{self.contract_name}] Field {field} has wrong type: "
                        f"expected {expected_type}, got {type(output[field])}"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.error(f"[{self.contract_name}] Validation error: {e}", exc_info=True)
            return False


class L1ExchangeContract(_BaseLayerContract):
    """
    Layer 1: Exchange I/O Contract.

    Single chokepoint for every byte that crosses the network to/from the
    exchange. Translates raw REST/WS responses into typed L0 objects.

    Input  : raw HTTP/WS frames (bytes / dict).
    Output : typed dicts consumed by L2 (balances, klines) and L4 (orders).

    Invariants:
      - No business logic; only protocol translation + retry/backoff.
      - All retries happen here; upper layers see success or typed ExchangeError.
      - OrderCacheManager is the sole writer of local order state.
    """

    contract_name = "L1ExchangeContract"
    required_output_fields = {
        "balances": dict,  # {asset: {free, locked}}
        "open_positions": list,  # [{symbol, qty, ...}]
        "exchange_time_ms": int,  # server time at sample
        "rate_limit_remaining": int,
    }


class L4ExecutionContract(_BaseLayerContract):
    """
    Layer 4: Execution & Order Management Contract.

    Turns approved TradeIntents (from L6) into actual orders, monitors them to
    completion, and emits fill events.

    Input (from L6) : ApprovedOrder = (intent, sized_qty, reservation_token).
    Output (to L3/L5/L7):
      - tickets   : list of ExecutionTicket dicts
      - fills     : list of Fill dicts (symbol, qty, price, ts, fees)
      - cancels   : list of cancelled order ids
      - timestamp : float

    Invariants:
      - L4 only spends capital reserved by L3 (ReservationToken required).
      - Every order has a journal entry before hitting the wire.
      - L4 never reads raw balances; reservation availability comes from L3.
    """

    contract_name = "L4ExecutionContract"
    required_output_fields = {
        "tickets": list,
        "fills": list,
        "cancels": list,
        "timestamp": float,
    }

    def validate_intent(
        self,
        reservation_token: Optional[str],
        symbol: str,
        side: str,
        quantity: float,
    ) -> bool:
        if not reservation_token:
            self.logger.error(
                f"[{self.contract_name}] {symbol} {side} {quantity}: "
                f"no reservation_token (L3 must reserve before L4 spends)"
            )
            return False
        if quantity <= 0:
            self.logger.error(
                f"[{self.contract_name}] {symbol} {side}: non-positive qty {quantity}"
            )
            return False
        if side not in ("BUY", "SELL"):
            self.logger.error(f"[{self.contract_name}] invalid side: {side}")
            return False
        return True


class L6PolicyContract(_BaseLayerContract):
    """
    Layer 6: Governance & Policy Contract.

    Final approver between intent (L5) and order (L4). Owns risk caps, sizing,
    and rule overrides.

    Input (from L5) : TradeIntent.
    Output (to L4)  : ApprovedOrder | GovernanceVeto.

    Invariants:
      - Veto authority: any intent breaching a cap is rejected with a typed
        GovernanceVeto and a recorded reason — never silently downsized.
      - All overrides are versioned + journaled via L3.
      - L6 does not know about exchanges; sees only TradeIntent + L3 state.
    """

    contract_name = "L6PolicyContract"
    required_output_fields = {
        "approved": list,  # [ApprovedOrder]
        "vetoed": list,  # [{intent_id, reason, cap_breached}]
        "caps": dict,  # current risk caps
        "timestamp": float,
    }

    def validate_decision(
        self,
        intent: dict[str, Any],
        approved: bool,
        veto_reason: Optional[str],
    ) -> bool:
        if approved and veto_reason:
            self.logger.error(f"[{self.contract_name}] inconsistent: approved with veto_reason")
            return False
        if not approved and not veto_reason:
            self.logger.error(
                f"[{self.contract_name}] vetoed intent {intent.get('id')} "
                f"with no recorded reason"
            )
            return False
        return True


class L7ObservabilityContract(_BaseLayerContract):
    """
    Layer 7: Observability & UX Contract.

    Read-only subscriber to all lower layers. Emits metrics, alerts, traces.

    Invariants:
      - L7 must not call any mutating method on L1–L6.
      - A failure in L7 must never break trading (degraded observability OK).
      - Subscribes to events; never tight-loops on business state.
    """

    contract_name = "L7ObservabilityContract"
    required_output_fields = {
        "metrics_emitted": int,
        "alerts_emitted": int,
        "last_scrape_ts": float,
    }


class L8LifecycleContract(_BaseLayerContract):
    """
    Layer 8: Lifecycle & Recovery Contract.

    Owns time. Boots layers in deterministic order, supervises them, restarts
    on failure, performs chaos drills, and shuts down gracefully.

    Deterministic boot order (BOOT_ORDER below):
      L0 → L1 → L2 → L3 → L4 → L6 → L5 → L7

    L6 starts before L5 so the policy gate exists before any TradeIntent can
    be produced.

    Invariants:
      - L8 is the only layer allowed to call start_layer / stop_layer.
      - Watchdog can restart any single layer without restarting the process.
    """

    contract_name = "L8LifecycleContract"
    required_output_fields = {
        "boot_order": list,  # ordered LayerName values
        "layer_health": dict,  # {layer_name: "OK"|"DEGRADED"|"DOWN"}
        "uptime_s": float,
    }

    # Authoritative boot order (mirrors LOGICAL_LAYERED_ARCHITECTURE.md §11.4)
    BOOT_ORDER: list[LayerName] = [
        LayerName.L0_CROSS_CUTTING,
        LayerName.L1_EXCHANGE_IO,
        LayerName.L2_WALLET_MARKETDATA,
        LayerName.L3_PORTFOLIO_STATE,
        LayerName.L4_EXECUTION,
        LayerName.L6_GOVERNANCE,
        LayerName.L5_STRATEGY,
        LayerName.L7_OBSERVABILITY,
    ]


# ----------------------------------------------------------------------------
# Allowed call graph (LOGICAL_LAYERED_ARCHITECTURE.md §13).
# Maps caller_layer -> set of callee_layers it is permitted to import.
# Used by scripts/check_layer_imports.py as the CI guard.
# ----------------------------------------------------------------------------
ALLOWED_DEPENDENCIES: dict[str, set] = {
    "L0": set(),  # pure
    "L1": {"L0"},
    "L2": {"L0", "L1"},
    "L3": {"L0", "L2"},
    "L4": {"L0", "L1", "L3"},  # skips L2 (cached in L3)
    "L5": {"L0", "L3"},  # pure decisions
    "L6": {"L0", "L3", "L5"},
    "L7": {"L0", "L1", "L2", "L3", "L4", "L5", "L6"},  # read-only
    "L8": {"L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"},  # boots everything
}


class EightLayerContractManager:
    """
    Manager for the full 8-layer model.

    Wraps the legacy LayerContractManager (L2/L3/L5) and adds L1/L4/L6/L7/L8.
    Existing code that uses LayerContractManager continues to work unchanged.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("EightLayerContractManager")

        # Legacy three contracts (re-exposed under L2/L3/L5 names)
        self.l2 = WalletLayerContract()
        self.l3 = PortfolioLayerContract()
        self.l5 = StrategyLayerContract()

        # New five contracts
        self.l1 = L1ExchangeContract()
        self.l4 = L4ExecutionContract()
        self.l6 = L6PolicyContract()
        self.l7 = L7ObservabilityContract()
        self.l8 = L8LifecycleContract()

    # Convenience pass-throughs
    def validate(self, layer: LayerName, output: dict[str, Any]) -> bool:
        mapping = {
            LayerName.L1_EXCHANGE_IO: self.l1.validate_output,
            LayerName.L2_WALLET_MARKETDATA: self.l2.validate_output,
            LayerName.WALLET_LAYER: self.l2.validate_output,
            LayerName.L3_PORTFOLIO_STATE: self.l3.validate_output,
            LayerName.PORTFOLIO_LAYER: self.l3.validate_output,
            LayerName.L4_EXECUTION: self.l4.validate_output,
            # L5 has operation-level validation; treat output validation as no-op
            LayerName.L5_STRATEGY: lambda _o: True,
            LayerName.STRATEGY_LAYER: lambda _o: True,
            LayerName.L6_GOVERNANCE: self.l6.validate_output,
            LayerName.L7_OBSERVABILITY: self.l7.validate_output,
            LayerName.L8_LIFECYCLE: self.l8.validate_output,
        }
        validator = mapping.get(layer)
        if validator is None:
            self.logger.error(f"No validator registered for layer {layer}")
            return False
        return validator(output)

    @staticmethod
    def is_call_allowed(caller_layer: str, callee_layer: str) -> bool:
        """Check the §13 call graph. caller_layer/callee_layer are 'L0'..'L8'."""
        allowed = ALLOWED_DEPENDENCIES.get(caller_layer, set())
        return callee_layer in allowed or callee_layer == caller_layer
