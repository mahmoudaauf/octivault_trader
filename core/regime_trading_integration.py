"""
Regime Trading Integration Layer

Adapter that bridges the new universe-ready live trading system with the existing
Octivault codebase (SharedState, ExecutionManager, MarketDataFeed).

Key responsibilities:
  1. Initialize regime system from existing components
  2. Synchronize regime state with SharedState
  3. Route trade signals through ExecutionManager
  4. Coordinate with existing risk management
  5. Provide monitoring and status

Design principle: Non-invasive integration via feature flags
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

# New live trading system components
from live_trading_system_architecture import (
    LiveTradingOrchestrator,
    RegimeDetectionEngine,
    ExposureController,
    PositionSizer,
    UniverseManager,
    SymbolConfig,
    RegimeState,
)
from live_data_pipeline import LiveDataFetcher, LivePositionManager

# Existing Octivault components
from core.shared_state import SharedState, Component, HealthCode
from core.execution_manager import ExecutionManager
from core.market_data_feed import MarketDataFeed

logger = logging.getLogger(__name__)


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

@dataclass
class RegimeTradingConfig:
    """Configuration for regime trading integration"""
    enabled: bool = False
    paper_trading: bool = True
    symbols: Dict[str, SymbolConfig] = None
    
    # Integration behavior
    sync_interval_seconds: float = 60.0
    regime_history_size: int = 1000
    position_sync_delay_seconds: float = 2.0
    
    # Risk guards
    max_concurrent_positions: int = 3
    max_total_exposure: float = 5.0  # 5x combined leverage
    emergency_stop_loss_pct: float = 0.50  # 50% portfolio loss
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = {}


# ============================================================================
# REGIME TRADING ADAPTER
# ============================================================================

class RegimeTradingAdapter:
    """
    Main integration class that coordinates between:
    - LiveTradingOrchestrator (new system)
    - SharedState (existing system)
    - ExecutionManager (existing system)
    - MarketDataFeed (existing system)
    """
    
    def __init__(
        self,
        shared_state: SharedState,
        execution_manager: ExecutionManager,
        market_data_feed: MarketDataFeed,
        config: RegimeTradingConfig,
    ):
        """Initialize the integration adapter"""
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.market_data_feed = market_data_feed
        self.config = config
        
        # Initialize new system components
        self.live_trader = None
        self.regime_engine = None
        self.data_fetcher = None
        self.position_manager = None
        
        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.regime_history: Dict[str, List[RegimeState]] = {}
        self.trade_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Control flags
        self._running = False
        self._sync_task = None
        self.last_sync_time = None
        
        logger.info(f"RegimeTradingAdapter initialized | paper_trading={config.paper_trading}")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def initialize(self) -> bool:
        """
        Initialize all regime trading components.
        Called once at system startup.
        """
        try:
            logger.info("Initializing regime trading system...")
            
            # 1. Initialize data pipeline
            self.data_fetcher = LiveDataFetcher(
                exchange_client=self.shared_state.exchange_client,
                symbols=list(self.config.symbols.keys()),
            )
            
            # 2. Initialize position manager
            self.position_manager = LivePositionManager(
                exchange_client=self.shared_state.exchange_client,
                paper_trading=self.config.paper_trading,
            )
            
            # 3. Initialize regime detection engine
            self.regime_engine = RegimeDetectionEngine()
            
            # 4. Initialize full orchestrator
            self.live_trader = LiveTradingOrchestrator(
                data_fetcher=self.data_fetcher,
                position_manager=self.position_manager,
                symbol_configs=self.config.symbols,
                paper_trading=self.config.paper_trading,
            )
            
            # 5. Initialize history tracking
            for symbol in self.config.symbols.keys():
                self.regime_history[symbol] = []
            
            # 6. Sync initial state with SharedState
            await self._sync_state_to_shared_state()
            
            self._running = True
            logger.info("✅ Regime trading system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize regime trading system: {e}", exc_info=True)
            self.shared_state.set_component_status(
                Component.AGENT_MANAGER, HealthCode.ERROR, f"Regime init failed: {e}"
            )
            return False
    
    # ========================================================================
    # MAIN EXECUTION LOOP
    # ========================================================================
    
    async def run_iteration(self) -> Dict[str, Any]:
        """
        Execute one trading iteration.
        
        This is the main loop called by main.py or main_live.py.
        
        Returns:
            Dictionary with iteration results:
            {
                "success": bool,
                "regime_states": {symbol: RegimeState},
                "trades_executed": [trade_records],
                "positions": {symbol: PositionState},
                "metrics": performance_metrics,
                "errors": [error_messages],
            }
        """
        if not self._running:
            logger.error("Adapter not running. Call initialize() first.")
            return {
                "success": False,
                "error": "Adapter not initialized",
                "regime_states": {},
                "trades_executed": [],
                "positions": {},
                "metrics": {},
                "errors": ["Adapter not initialized"],
            }
        
        iteration_result = {
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "regime_states": {},
            "trades_executed": [],
            "positions": {},
            "metrics": {},
            "errors": [],
        }
        
        try:
            # 1. Fetch latest market data
            market_data = await self.data_fetcher.fetch_ohlcv_batch(
                symbols=list(self.config.symbols.keys()),
                timeframe="1h",
            )
            
            if not market_data:
                iteration_result["errors"].append("Failed to fetch market data")
                return iteration_result
            
            # 2. Detect regimes for each symbol
            regime_states = {}
            for symbol, ohlcv_data in market_data.items():
                try:
                    regime = await self.regime_engine.detect_regime(
                        symbol=symbol,
                        ohlcv=ohlcv_data,
                        config=self.config.symbols.get(symbol),
                    )
                    regime_states[symbol] = regime
                    
                    # Track history
                    if len(self.regime_history[symbol]) >= self.config.regime_history_size:
                        self.regime_history[symbol].pop(0)
                    self.regime_history[symbol].append(regime)
                    
                except Exception as e:
                    logger.error(f"Regime detection failed for {symbol}: {e}")
                    iteration_result["errors"].append(f"Regime detection {symbol}: {e}")
            
            iteration_result["regime_states"] = regime_states
            
            # 3. Generate signals and size positions
            trades = await self.live_trader.run_iteration(regime_states)
            
            if trades:
                iteration_result["trades_executed"] = trades
            
            # 4. Execute trades through existing ExecutionManager
            executed_trades = []
            for trade in trades:
                try:
                    result = await self._execute_trade(trade)
                    executed_trades.append(result)
                except Exception as e:
                    logger.error(f"Trade execution failed: {e}")
                    iteration_result["errors"].append(f"Trade execution: {e}")
            
            # 5. Fetch current positions
            positions = await self.position_manager.get_positions()
            iteration_result["positions"] = positions
            
            # 6. Sync with SharedState
            await self._sync_state_to_shared_state(
                regime_states=regime_states,
                positions=positions,
            )
            
            # 7. Calculate performance metrics
            await self._calculate_metrics(regime_states, positions)
            iteration_result["metrics"] = self.performance_metrics
            
            iteration_result["success"] = len(iteration_result["errors"]) == 0
            
            self.last_sync_time = datetime.utcnow()
            logger.info(f"Iteration complete | regimes={len(regime_states)} | trades={len(executed_trades)}")
            
            return iteration_result
            
        except Exception as e:
            logger.error(f"❌ Iteration failed: {e}", exc_info=True)
            iteration_result["errors"].append(str(e))
            return iteration_result
    
    # ========================================================================
    # SIGNAL GENERATION & EXECUTION
    # ========================================================================
    
    async def _execute_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade through the existing ExecutionManager.
        
        Trade format (from LiveTradingOrchestrator):
        {
            "symbol": "ETHUSDT",
            "side": "BUY" or "SELL",
            "exposure": 2.0,  # leverage
            "reason": "LOW_VOL_TRENDING",
            "regime_state": RegimeState,
        }
        """
        symbol = trade.get("symbol")
        side = trade.get("side", "").upper()
        exposure = trade.get("exposure", 1.0)
        reason = trade.get("reason", "REGIME_SIGNAL")
        
        try:
            # 1. Get current position
            current_qty = await self._get_position_size(symbol)
            
            # 2. Calculate target position from exposure
            target_qty = await self._calculate_target_quantity(
                symbol=symbol,
                exposure=exposure,
                side=side,
            )
            
            # 3. Determine order type
            if side == "BUY":
                if current_qty > 0:
                    # Already long, adjust size
                    qty_delta = max(0, target_qty - current_qty)
                    if qty_delta > 0:
                        order_qty = qty_delta
                else:
                    order_qty = target_qty
            else:  # SELL
                if current_qty > 0:
                    order_qty = current_qty
                else:
                    # Already short or flat, don't short
                    return {
                        "symbol": symbol,
                        "side": side,
                        "status": "SKIPPED",
                        "reason": "No position to sell",
                    }
            
            if order_qty <= 0:
                return {
                    "symbol": symbol,
                    "side": side,
                    "status": "SKIPPED",
                    "reason": "Order quantity <= 0",
                }
            
            # 4. Execute through existing ExecutionManager
            if self.config.paper_trading:
                # Paper trading: simulate execution
                result = await self._simulate_order(
                    symbol=symbol,
                    side=side,
                    qty=order_qty,
                )
            else:
                # Live trading: use real ExecutionManager
                result = await self.execution_manager.execute_order(
                    symbol=symbol,
                    side=side,
                    qty=order_qty,
                    order_type="MARKET",
                )
            
            # 5. Log trade
            trade_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "qty": order_qty,
                "exposure": exposure,
                "reason": reason,
                "status": result.get("status", "EXECUTED"),
                "price": result.get("price"),
                "cost": result.get("cost"),
            }
            
            self.trade_log.append(trade_record)
            logger.info(f"Trade executed: {symbol} {side} {order_qty} @ {result.get('price', 'N/A')}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "side": side,
                "status": "FAILED",
                "error": str(e),
            }
    
    # ========================================================================
    # STATE SYNCHRONIZATION
    # ========================================================================
    
    async def _sync_state_to_shared_state(
        self,
        regime_states: Optional[Dict[str, RegimeState]] = None,
        positions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Synchronize regime trading state with SharedState.
        
        This ensures the existing system can monitor and coordinate
        with the new regime trading system.
        """
        try:
            if regime_states:
                # Store regime states in SharedState metadata
                for symbol, regime in regime_states.items():
                    key = f"regime_{symbol}"
                    state_dict = {
                        "timestamp": regime.timestamp.isoformat(),
                        "symbol": regime.symbol,
                        "volatility_regime": regime.volatility_regime,
                        "trend_regime": regime.trend_regime,
                        "macro_trend": regime.macro_trend,
                        "is_alpha": regime.is_alpha_regime(),
                    }
                    # Store in metadata (implementation depends on SharedState API)
                    # self.shared_state.set_metadata(key, state_dict)
            
            if positions:
                # Store position states in SharedState
                for symbol, pos_info in positions.items():
                    # Update SharedState position tracking
                    pass
            
            # Update component health
            self.shared_state.set_component_status(
                Component.AGENT_MANAGER,
                HealthCode.OK,
                f"Regime trading active | {len(regime_states or {})} symbols",
            )
            
        except Exception as e:
            logger.error(f"State synchronization failed: {e}")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _get_position_size(self, symbol: str) -> float:
        """Get current position size from SharedState or ExecutionManager"""
        try:
            # Try to get from SharedState first
            # This depends on your SharedState implementation
            positions = await self.position_manager.get_positions()
            return positions.get(symbol, {}).get("size", 0.0)
        except Exception as e:
            logger.error(f"Failed to get position size for {symbol}: {e}")
            return 0.0
    
    async def _calculate_target_quantity(
        self,
        symbol: str,
        exposure: float,
        side: str,
    ) -> float:
        """
        Calculate target position size based on exposure and account size.
        
        Uses position sizer from new system, constrained by risk limits
        from existing system.
        """
        try:
            # Get account balance from SharedState
            account_balance = self.shared_state.account_state.get("balance", 0.0)
            
            # Get symbol config
            symbol_config = self.config.symbols.get(symbol)
            if not symbol_config:
                return 0.0
            
            # Use PositionSizer from new system
            position_sizer = PositionSizer(config=symbol_config)
            target_qty = await position_sizer.calculate_position_size(
                symbol=symbol,
                account_balance=account_balance,
                exposure=exposure,
                side=side,
            )
            
            return target_qty
            
        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
            return 0.0
    
    async def _simulate_order(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Dict[str, Any]:
        """Simulate order execution for paper trading"""
        try:
            # Get latest price from market data
            price = await self.market_data_feed.get_latest_price(symbol)
            
            if price is None:
                price = 0.0
            
            cost = qty * price
            
            return {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "cost": cost,
                "status": "EXECUTED",
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Paper trading simulation failed: {e}")
            return {
                "symbol": symbol,
                "side": side,
                "status": "FAILED",
                "error": str(e),
            }
    
    async def _calculate_metrics(
        self,
        regime_states: Dict[str, RegimeState],
        positions: Dict[str, Any],
    ) -> None:
        """Calculate performance metrics"""
        try:
            self.performance_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "regime_frequency": {},
                "position_count": len(positions),
                "total_exposure": 0.0,
                "trade_count": len(self.trade_log),
            }
            
            # Calculate regime frequency
            for symbol, history in self.regime_history.items():
                if history:
                    alpha_count = sum(1 for r in history if r.is_alpha_regime())
                    freq = alpha_count / len(history) if history else 0.0
                    self.performance_metrics["regime_frequency"][symbol] = freq
            
            # Calculate total exposure
            for symbol, pos in positions.items():
                exposure = pos.get("exposure", 0.0)
                self.performance_metrics["total_exposure"] += exposure
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
    
    # ========================================================================
    # STATUS & MONITORING
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self._running,
            "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "active_positions": len(self.active_positions),
            "trade_count": len(self.trade_log),
            "performance_metrics": self.performance_metrics,
            "regime_history_sizes": {
                symbol: len(history)
                for symbol, history in self.regime_history.items()
            },
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the adapter"""
        try:
            logger.info("Shutting down regime trading adapter...")
            self._running = False
            
            # Close positions if needed
            if self.live_trader:
                await self.live_trader.shutdown()
            
            logger.info("✅ Regime trading adapter shut down")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

async def create_regime_trading_adapter(
    shared_state: SharedState,
    execution_manager: ExecutionManager,
    market_data_feed: MarketDataFeed,
    config: Optional[RegimeTradingConfig] = None,
) -> Optional[RegimeTradingAdapter]:
    """
    Factory function to create and initialize a RegimeTradingAdapter.
    
    Usage in main.py:
    
        adapter = await create_regime_trading_adapter(
            shared_state=shared_state,
            execution_manager=execution_manager,
            market_data_feed=market_data_feed,
            config=regime_config,
        )
        
        if adapter:
            result = await adapter.run_iteration()
    """
    if config is None:
        config = RegimeTradingConfig(enabled=False)
    
    if not config.enabled:
        logger.info("Regime trading disabled via configuration")
        return None
    
    try:
        adapter = RegimeTradingAdapter(
            shared_state=shared_state,
            execution_manager=execution_manager,
            market_data_feed=market_data_feed,
            config=config,
        )
        
        if await adapter.initialize():
            return adapter
        else:
            logger.error("Failed to initialize regime trading adapter")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create regime trading adapter: {e}", exc_info=True)
        return None
