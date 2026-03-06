"""
Market Regime Detector Integration Module

This module integrates MarketRegimeDetector with all components:
- MetaController: Weighted signal voting based on regime
- AgentManager: Enable/disable agents based on regime
- CapitalAllocator: Adjust capital deployment based on regime
- ExecutionManager/MakerExecutor: Adjust execution style based on regime

Architecture
-----------
The regime detector sits between market data and agent/execution decision-making:

MarketDataFeed
      ↓
MarketRegimeDetector (this integration)
      ├─→ MetaController (weighted voting)
      ├─→ AgentManager (enable/disable agents)
      ├─→ CapitalAllocator (capital deployment)
      └─→ ExecutionManager (execution style)

Usage Example
-----------
# In your AppContext or startup:
from core.market_regime_integration import RegimeAwareMediator

mediator = RegimeAwareMediator(
    config=config,
    market_regime_detector=detector,
    meta_controller=meta_controller,
    agent_manager=agent_manager,
    capital_allocator=capital_allocator,
    execution_manager=execution_manager,
)

# In your main loop:
regime_metrics = await mediator.update_regime(symbol, ohlcv, bid, ask)
if regime_metrics:
    await mediator.apply_regime_adaptations(symbol, regime_metrics)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

logger = logging.getLogger("RegimeAwareMediator")


@dataclass
class RegimeAdaptation:
    """Result of applying regime-based adaptations to system."""
    regime: str
    confidence: float
    agents_active: List[str]
    agent_weights: Dict[str, float]
    execution_style: Dict[str, Any]
    capital_allocation: Dict[str, Any]
    timestamp: float


class RegimeAwareMediator:
    """
    Coordinates MarketRegimeDetector with all system components.
    
    Responsibilities:
    1. Update regime detection from market data
    2. Broadcast regime to MetaController for weighted voting
    3. Update AgentManager with active agents for regime
    4. Adjust CapitalAllocator deployment based on regime
    5. Configure ExecutionManager with regime-appropriate execution style
    """
    
    def __init__(
        self,
        config: Any,
        market_regime_detector: Any,
        meta_controller: Optional[Any] = None,
        agent_manager: Optional[Any] = None,
        capital_allocator: Optional[Any] = None,
        execution_manager: Optional[Any] = None,
        maker_executor: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize RegimeAwareMediator.
        
        Args:
            config: Configuration object
            market_regime_detector: MarketRegimeDetector instance
            meta_controller: MetaController for weighted signal voting
            agent_manager: AgentManager for enabling/disabling agents
            capital_allocator: CapitalAllocator for deployment adjustment
            execution_manager: ExecutionManager for execution style
            maker_executor: Optional MakerExecutor for maker-specific config
            logger: Optional logger
        """
        self.config = config
        self.detector = market_regime_detector
        self.meta_controller = meta_controller
        self.agent_manager = agent_manager
        self.capital_allocator = capital_allocator
        self.execution_manager = execution_manager
        self.maker_executor = maker_executor
        self.logger = logger or globals()["logger"]
        
        # Cache last regime per symbol
        self._regime_cache: Dict[str, Any] = {}
        self._last_adaptation: Dict[str, RegimeAdaptation] = {}
    
    async def update_regime(
        self,
        symbol: str,
        ohlcv: List[Dict[str, float]],
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Detect and cache regime for a symbol.
        
        Args:
            symbol: Trading symbol
            ohlcv: List of OHLCV candles
            bid_price: Current bid
            ask_price: Current ask
        
        Returns:
            RegimeMetrics if successful, None otherwise
        """
        if not self.detector:
            return None
        
        try:
            metrics = self.detector.detect(symbol, ohlcv, bid_price, ask_price)
            self._regime_cache[symbol] = metrics
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error updating regime for {symbol}: {e}", exc_info=True)
            return None
    
    async def apply_regime_adaptations(
        self,
        symbol: str,
        regime_metrics: Any,
    ) -> Optional[RegimeAdaptation]:
        """
        Apply regime-based adaptations to all components.
        
        This is the main coordination method that broadcasts regime to:
        1. MetaController → weighted signal voting
        2. AgentManager → enable/disable agents
        3. CapitalAllocator → adjust deployment ratio
        4. ExecutionManager → adjust execution style
        
        Args:
            symbol: Trading symbol
            regime_metrics: RegimeMetrics from detector
        
        Returns:
            RegimeAdaptation describing what was applied
        """
        if not regime_metrics:
            return None
        
        try:
            regime = regime_metrics.regime
            confidence = regime_metrics.confidence
            
            # Get regime-specific guidance
            agent_weights = self.detector.get_agent_weights(regime)
            execution_style = self.detector.get_execution_style(regime)
            capital_allocation = self.detector.get_capital_allocation(regime)
            
            # Determine active agents (those in weights map)
            agents_active = list(agent_weights.keys())
            
            # Apply to MetaController
            if self.meta_controller:
                await self._apply_to_meta_controller(
                    symbol, regime, agent_weights, confidence
                )
            
            # Apply to AgentManager
            if self.agent_manager:
                await self._apply_to_agent_manager(symbol, agents_active, regime)
            
            # Apply to CapitalAllocator
            if self.capital_allocator:
                await self._apply_to_capital_allocator(
                    symbol, capital_allocation, regime, confidence
                )
            
            # Apply to ExecutionManager
            if self.execution_manager or self.maker_executor:
                await self._apply_to_execution(symbol, execution_style, regime)
            
            # Build adaptation record
            adaptation = RegimeAdaptation(
                regime=regime.value,
                confidence=confidence,
                agents_active=agents_active,
                agent_weights=agent_weights,
                execution_style=execution_style,
                capital_allocation=capital_allocation,
                timestamp=time.time(),
            )
            
            self._last_adaptation[symbol] = adaptation
            
            self.logger.info(
                f"[RegimeAdaptation] {symbol}: regime={regime.value} "
                f"confidence={confidence:.2f} agents={len(agents_active)} "
                f"exec_ratio={execution_style.get('maker_ratio', 0.5):.1%}"
            )
            
            return adaptation
        
        except Exception as e:
            self.logger.error(
                f"Error applying regime adaptations for {symbol}: {e}",
                exc_info=True
            )
            return None
    
    async def _apply_to_meta_controller(
        self,
        symbol: str,
        regime: Any,
        agent_weights: Dict[str, float],
        confidence: float,
    ) -> None:
        """
        Update MetaController with regime-aware weighted signal voting.
        
        This allows MetaController to weight signals from agents based on regime:
        - In trending markets, weight TrendHunter higher
        - In ranging markets, weight DipSniper higher
        - Etc.
        """
        if not self.meta_controller:
            return
        
        try:
            # Store regime context in meta_controller for use during signal arbitration
            if not hasattr(self.meta_controller, "_regime_context"):
                self.meta_controller._regime_context = {}
            
            self.meta_controller._regime_context[symbol] = {
                "regime": regime.value,
                "confidence": confidence,
                "agent_weights": agent_weights,
                "timestamp": time.time(),
            }
            
            # Log for debugging
            self.logger.debug(
                f"[MetaController] Set regime context for {symbol}: "
                f"regime={regime.value} weights={agent_weights}"
            )
        
        except Exception as e:
            self.logger.warning(
                f"Failed to apply regime to MetaController: {e}"
            )
    
    async def _apply_to_agent_manager(
        self,
        symbol: str,
        agents_active: List[str],
        regime: Any,
    ) -> None:
        """
        Update AgentManager with active agents for this regime.
        
        In production, this would enable/disable agents in the agent pool
        based on regime specialization. For now, it stores context.
        """
        if not self.agent_manager:
            return
        
        try:
            # Store regime-aware agent activation context
            if not hasattr(self.agent_manager, "_regime_active_agents"):
                self.agent_manager._regime_active_agents = {}
            
            self.agent_manager._regime_active_agents[symbol] = {
                "regime": regime.value,
                "agents_active": agents_active,
                "timestamp": time.time(),
            }
            
            self.logger.debug(
                f"[AgentManager] Set active agents for {symbol} in {regime.value}: "
                f"{agents_active}"
            )
        
        except Exception as e:
            self.logger.warning(
                f"Failed to apply regime to AgentManager: {e}"
            )
    
    async def _apply_to_capital_allocator(
        self,
        symbol: str,
        capital_allocation: Dict[str, Any],
        regime: Any,
        confidence: float,
    ) -> None:
        """
        Update CapitalAllocator with regime-based deployment adjustments.
        
        Based on regime, adjust:
        - deploy_ratio: How much capital to deploy (0.0-1.0)
        - risk_adjustment: Risk multiplier (0.5 = half normal risk)
        """
        if not self.capital_allocator:
            return
        
        try:
            # Store regime-aware deployment context
            if not hasattr(self.capital_allocator, "_regime_deployment"):
                self.capital_allocator._regime_deployment = {}
            
            deploy_ratio = capital_allocation.get("deploy_ratio", 0.5)
            risk_adjustment = capital_allocation.get("risk_adjustment", 1.0)
            
            self.capital_allocator._regime_deployment[symbol] = {
                "regime": regime.value,
                "confidence": confidence,
                "deploy_ratio": deploy_ratio,
                "risk_adjustment": risk_adjustment,
                "timestamp": time.time(),
            }
            
            self.logger.debug(
                f"[CapitalAllocator] Set deployment for {symbol}: "
                f"regime={regime.value} deploy_ratio={deploy_ratio:.1%} "
                f"risk_adj={risk_adjustment:.2f}x"
            )
        
        except Exception as e:
            self.logger.warning(
                f"Failed to apply regime to CapitalAllocator: {e}"
            )
    
    async def _apply_to_execution(
        self,
        symbol: str,
        execution_style: Dict[str, Any],
        regime: Any,
    ) -> None:
        """
        Update ExecutionManager with regime-appropriate execution style.
        
        Based on regime, adjust:
        - maker_ratio: How aggressive to be with maker orders
        - limit_order_timeout_sec: How long to wait for limit fills
        - spread_placement_ratio: How deep inside spread to place orders
        """
        if not (self.execution_manager or self.maker_executor):
            return
        
        try:
            # Apply to MakerExecutor if available
            if self.maker_executor:
                maker_ratio = execution_style.get("maker_ratio", 0.5)
                
                # Store regime context in maker_executor
                if not hasattr(self.maker_executor, "_regime_context"):
                    self.maker_executor._regime_context = {}
                
                self.maker_executor._regime_context[symbol] = {
                    "regime": regime.value,
                    "maker_ratio": maker_ratio,
                    "timeout_sec": execution_style.get("limit_order_timeout_sec", 5.0),
                    "spread_placement": execution_style.get("spread_placement_ratio", 0.2),
                    "timestamp": time.time(),
                }
            
            self.logger.debug(
                f"[ExecutionManager] Set execution style for {symbol}: "
                f"regime={regime.value} maker_ratio={execution_style.get('maker_ratio', 0.5):.1%}"
            )
        
        except Exception as e:
            self.logger.warning(
                f"Failed to apply regime to ExecutionManager: {e}"
            )
    
    def get_last_regime(self, symbol: str) -> Optional[Any]:
        """Get cached regime metrics for a symbol."""
        return self._regime_cache.get(symbol)
    
    def get_last_adaptation(self, symbol: str) -> Optional[RegimeAdaptation]:
        """Get last applied adaptation for a symbol."""
        return self._last_adaptation.get(symbol)
    
    async def get_regime_report(self) -> Dict[str, Any]:
        """
        Get comprehensive regime report across all symbols.
        
        Returns dict with current regime, confidence, and active agents per symbol.
        """
        report = {
            "timestamp": time.time(),
            "symbols": {},
        }
        
        for symbol, metrics in self._regime_cache.items():
            adaptation = self._last_adaptation.get(symbol)
            report["symbols"][symbol] = {
                "regime": metrics.regime.value,
                "confidence": metrics.confidence,
                "adx": metrics.adx,
                "atr_pct": metrics.atr_pct,
                "rsi": metrics.rsi,
                "spread_pct": metrics.spread_pct,
                "agents_active": adaptation.agents_active if adaptation else [],
                "timestamp": metrics.timestamp,
            }
        
        return report
