# core/layer_contracts.py
# Professional Three-Layer Capital Accounting Architecture
# Formal contracts and boundaries for Wallet Layer → Portfolio Layer → Strategy Layer

import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("LayerContracts")


class LayerName(Enum):
    """Three-layer hierarchy."""
    WALLET_LAYER = "WALLET_LAYER"          # Layer 1: Exchange wallet synchronization
    PORTFOLIO_LAYER = "PORTFOLIO_LAYER"    # Layer 2: Asset registry and classification
    STRATEGY_LAYER = "STRATEGY_LAYER"      # Layer 3: Bot trading decisions


@dataclass
class LayerInput:
    """Contract: What data a layer receives as input."""
    source_layer: str                       # Which layer provides this input
    timestamp: float                        # When data was generated
    data: Dict[str, Any]                    # Payload
    validated: bool = True                  # Input meets contract requirements


@dataclass
class LayerOutput:
    """Contract: What data a layer produces as output."""
    source_layer: str                       # Which layer produced this output
    timestamp: float                        # When output was generated
    data: Dict[str, Any]                    # Payload
    verified: bool = True                   # Output meets contract requirements


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
            "assets": dict,           # {symbol: ClassifiedPosition}
            "positions": dict,        # {symbol: Position}
            "last_updated": float,    # Timestamp of last sync
        }
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Verify Layer 1 output meets contract requirements."""
        try:
            # Check required fields exist
            for field, expected_type in self.required_output_fields.items():
                if field not in output:
                    self.logger.error(
                        f"[{self.contract_name}] Missing required field: {field}"
                    )
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
                if asset_data["classification"] not in ("BOT_POSITION", "EXTERNAL_POSITION", "STABLE", "DUST"):
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
            self.logger.error(
                f"[{self.contract_name}] Validation error: {e}",
                exc_info=True
            )
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
            "portfolio": dict,              # {symbol: Position}
            "nav": float,                   # Net asset value
            "capital_available": float,     # Quote available for trades
            "risk_metrics": dict,           # exposure, concentration, etc.
            "dust_positions": dict,         # Positions flagged as dust
            "last_computed": float,         # Timestamp of computation
        }
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Verify Layer 2 output meets contract requirements."""
        try:
            # Check required fields
            for field, expected_type in self.required_output_fields.items():
                if field not in output:
                    self.logger.error(
                        f"[{self.contract_name}] Missing required field: {field}"
                    )
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
            self.logger.error(
                f"[{self.contract_name}] Validation error: {e}",
                exc_info=True
            )
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
            "trades": list,                 # [{symbol, side, qty, price, timestamp}]
            "pnl": dict,                    # {realized, unrealized, total}
            "audit_log": list,              # Trade audit entries
            "execution_timestamp": float,   # When trades were executed
        }
    
    def validate_operation(
        self,
        operation_type: str,
        symbol: str,
        classification: Optional[str],
        quantity: float
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
            self.logger.error(
                f"[{self.contract_name}] Validation error: {e}",
                exc_info=True
            )
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
    
    def validate_wallet_output(self, output: Dict[str, Any]) -> bool:
        """Validate Wallet Layer output."""
        return self.wallet_contract.validate_output(output)
    
    def validate_portfolio_output(self, output: Dict[str, Any]) -> bool:
        """Validate Portfolio Layer output."""
        return self.portfolio_contract.validate_output(output)
    
    def validate_strategy_operation(
        self,
        operation_type: str,
        symbol: str,
        classification: Optional[str],
        quantity: float
    ) -> bool:
        """Validate Strategy Layer operation."""
        return self.strategy_contract.validate_operation(
            operation_type, symbol, classification, quantity
        )
