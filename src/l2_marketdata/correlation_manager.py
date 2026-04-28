"""
Correlation and Portfolio Concentration Manager

This module manages portfolio correlation analysis and enforces position concentration
limits to prevent correlated trades that amplify losses. Uses rolling correlation
matrices and concentration thresholds.

Issue #10: Portfolio Correlation Checks
- Calculate correlation matrix across active positions
- Prevent excessive concentration in correlated assets
- Track sector/exchange exposure
- Alert on correlation-driven concentration risks
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import threading
import statistics

logger = logging.getLogger(__name__)


class CorrelationStatus(Enum):
    """Position correlation status indicators."""
    ACCEPTED = "ACCEPTED"
    REJECTED_HIGH_CORRELATION = "REJECTED_HIGH_CORRELATION"
    REJECTED_CONCENTRATION_RISK = "REJECTED_CONCENTRATION_RISK"
    REJECTED_SECTOR_CONCENTRATION = "REJECTED_SECTOR_CONCENTRATION"
    REJECTED_EXCHANGE_CONCENTRATION = "REJECTED_EXCHANGE_CONCENTRATION"
    WARNING_ELEVATED_CORRELATION = "WARNING_ELEVATED_CORRELATION"


@dataclass
class Position:
    """Active trading position."""
    symbol: str
    allocation: float  # Percentage of portfolio (0-100)
    sector: str  # e.g., "TECH", "FINANCE", "CRYPTO"
    exchange: str  # e.g., "NYSE", "NASDAQ", "BINANCE"
    entry_price: float
    entry_time: datetime
    historical_returns: List[float] = field(default_factory=list)


@dataclass
class CorrelationPair:
    """Correlation between two positions."""
    symbol_a: str
    symbol_b: str
    correlation: float  # -1.0 to 1.0
    lookback_days: int
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CorrelationCheckResult:
    """Result of correlation validation."""
    status: CorrelationStatus
    symbol: str
    allocation: float
    timestamp: datetime
    reason: str = ""
    correlation_pairs: List[CorrelationPair] = field(default_factory=list)
    sector_exposure: float = 0.0  # Current sector exposure after adding position
    exchange_exposure: float = 0.0  # Current exchange exposure after adding position
    confidence: float = 0.95


@dataclass
class PortfolioCorrelationMatrix:
    """Correlation matrix for portfolio."""
    symbols: List[str]
    correlations: Dict[Tuple[str, str], float]  # Keyed by (symbol_a, symbol_b)
    lookback_days: int
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """Get correlation between two symbols (handles symmetry)."""
        key1 = (symbol_a, symbol_b) if symbol_a < symbol_b else (symbol_b, symbol_a)
        return self.correlations.get(key1)


class CorrelationCalculator:
    """Calculates correlation between asset returns."""
    
    @staticmethod
    def calculate_correlation(returns_a: List[float], returns_b: List[float]) -> Optional[float]:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            returns_a: Return series for asset A
            returns_b: Return series for asset B
            
        Returns:
            Correlation coefficient (-1 to 1), or None if insufficient data
        """
        if len(returns_a) < 3 or len(returns_b) < 3:
            return None
        
        if len(returns_a) != len(returns_b):
            return None
        
        try:
            mean_a = statistics.mean(returns_a)
            mean_b = statistics.mean(returns_b)
            
            numerator = sum((returns_a[i] - mean_a) * (returns_b[i] - mean_b) for i in range(len(returns_a)))
            std_a = statistics.stdev(returns_a)
            std_b = statistics.stdev(returns_b)
            
            if std_a == 0 or std_b == 0:
                return None
            
            denominator = std_a * std_b * len(returns_a)
            correlation = numerator / denominator
            
            return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
        except (ValueError, ZeroDivisionError):
            return None


class PortfolioConcentrationManager:
    """
    Manages portfolio concentration and correlation risks.
    
    Tracks:
    - Position concentration (max 25% per symbol)
    - Sector concentration (max 40% per sector)
    - Exchange concentration (max 50% per exchange)
    - Position correlation (reject >0.8 correlation with existing positions)
    - High correlation groups (alert at >0.7 group correlation)
    """
    
    def __init__(
        self,
        max_position_allocation: float = 25.0,  # 25% per symbol
        max_sector_allocation: float = 40.0,     # 40% per sector
        max_exchange_allocation: float = 50.0,   # 50% per exchange
        max_position_correlation: float = 0.8,    # Reject new positions correlated >0.8
        warning_correlation_threshold: float = 0.7,  # Warn at >0.7
        correlation_lookback_days: int = 30
    ):
        self.max_position_allocation = max_position_allocation
        self.max_sector_allocation = max_sector_allocation
        self.max_exchange_allocation = max_exchange_allocation
        self.max_position_correlation = max_position_correlation
        self.warning_correlation_threshold = warning_correlation_threshold
        self.correlation_lookback_days = correlation_lookback_days
        
        self.active_positions: Dict[str, Position] = {}
        self.correlation_matrix: Optional[PortfolioCorrelationMatrix] = None
        self.validation_log: List[CorrelationCheckResult] = []
        self.lock = threading.Lock()
        
        logger.info(
            f"CorrelationManager initialized: "
            f"max_position={max_position_allocation}%, "
            f"max_sector={max_sector_allocation}%, "
            f"max_exchange={max_exchange_allocation}%, "
            f"max_correlation={max_position_correlation}"
        )
    
    def add_position(self, symbol: str, allocation: float, sector: str, exchange: str,
                    entry_price: float, returns: List[float]) -> None:
        """Add active position to tracking."""
        position = Position(
            symbol=symbol,
            allocation=allocation,
            sector=sector,
            exchange=exchange,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            historical_returns=returns
        )
        
        with self.lock:
            self.active_positions[symbol] = position
        
        logger.debug(f"Added position: {symbol} ({allocation:.1f}%, {sector}, {exchange})")
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from tracking."""
        with self.lock:
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                # Invalidate correlation matrix
                self.correlation_matrix = None
        
        logger.debug(f"Removed position: {symbol}")
    
    def _update_correlation_matrix(self) -> None:
        """Calculate correlation matrix for all active positions."""
        with self.lock:
            symbols = list(self.active_positions.keys())
        
        if len(symbols) < 2:
            return
        
        correlations: Dict[Tuple[str, str], float] = {}
        
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                pos_a = self.active_positions[sym_a]
                pos_b = self.active_positions[sym_b]
                
                corr = CorrelationCalculator.calculate_correlation(
                    pos_a.historical_returns,
                    pos_b.historical_returns
                )
                
                if corr is not None:
                    key = (sym_a, sym_b) if sym_a < sym_b else (sym_b, sym_a)
                    correlations[key] = corr
        
        with self.lock:
            self.correlation_matrix = PortfolioCorrelationMatrix(
                symbols=symbols,
                correlations=correlations,
                lookback_days=self.correlation_lookback_days,
                updated_at=datetime.utcnow()
            )
        
        logger.debug(f"Updated correlation matrix for {len(symbols)} positions")
    
    def validate_new_position(self, symbol: str, allocation: float, sector: str,
                             exchange: str, returns: List[float]) -> CorrelationCheckResult:
        """
        Validate new position for concentration and correlation risks.
        
        Args:
            symbol: Trading symbol
            allocation: Proposed allocation as % of portfolio
            sector: Sector classification
            exchange: Exchange listing
            returns: Historical returns for correlation calculation
            
        Returns:
            CorrelationCheckResult with validation details
        """
        # Check individual position concentration
        if allocation > self.max_position_allocation:
            result = CorrelationCheckResult(
                status=CorrelationStatus.REJECTED_CONCENTRATION_RISK,
                symbol=symbol,
                allocation=allocation,
                timestamp=datetime.utcnow(),
                reason=f"Position size {allocation:.1f}% exceeds limit {self.max_position_allocation}%"
            )
            self._log_result(result)
            logger.warning(f"CONCENTRATION: {symbol} {allocation:.1f}% exceeds max {self.max_position_allocation}%")
            return result
        
        # Check sector concentration
        current_sector_exposure = self._calculate_sector_exposure(sector)
        new_sector_exposure = current_sector_exposure + allocation
        
        if new_sector_exposure > self.max_sector_allocation:
            result = CorrelationCheckResult(
                status=CorrelationStatus.REJECTED_SECTOR_CONCENTRATION,
                symbol=symbol,
                allocation=allocation,
                sector_exposure=new_sector_exposure,
                timestamp=datetime.utcnow(),
                reason=f"Sector {sector} concentration {new_sector_exposure:.1f}% exceeds limit {self.max_sector_allocation}%"
            )
            self._log_result(result)
            logger.warning(f"SECTOR_CONCENTRATION: {sector} {new_sector_exposure:.1f}% exceeds max {self.max_sector_allocation}%")
            return result
        
        # Check exchange concentration
        current_exchange_exposure = self._calculate_exchange_exposure(exchange)
        new_exchange_exposure = current_exchange_exposure + allocation
        
        if new_exchange_exposure > self.max_exchange_allocation:
            result = CorrelationCheckResult(
                status=CorrelationStatus.REJECTED_EXCHANGE_CONCENTRATION,
                symbol=symbol,
                allocation=allocation,
                exchange_exposure=new_exchange_exposure,
                timestamp=datetime.utcnow(),
                reason=f"Exchange {exchange} concentration {new_exchange_exposure:.1f}% exceeds limit {self.max_exchange_allocation}%"
            )
            self._log_result(result)
            logger.warning(f"EXCHANGE_CONCENTRATION: {exchange} {new_exchange_exposure:.1f}% exceeds max {self.max_exchange_allocation}%")
            return result
        
        # Check correlation with existing positions
        self._update_correlation_matrix()
        correlation_pairs: List[CorrelationPair] = []
        
        with self.lock:
            if self.correlation_matrix:
                for existing_symbol in self.correlation_matrix.symbols:
                    existing_pos = self.active_positions[existing_symbol]
                    
                    corr = CorrelationCalculator.calculate_correlation(
                        existing_pos.historical_returns,
                        returns
                    )
                    
                    if corr is not None:
                        correlation_pairs.append(CorrelationPair(
                            symbol_a=symbol,
                            symbol_b=existing_symbol,
                            correlation=corr,
                            lookback_days=self.correlation_lookback_days
                        ))
                        
                        # Check if correlation exceeds limit
                        if corr > self.max_position_correlation:
                            result = CorrelationCheckResult(
                                status=CorrelationStatus.REJECTED_HIGH_CORRELATION,
                                symbol=symbol,
                                allocation=allocation,
                                timestamp=datetime.utcnow(),
                                correlation_pairs=correlation_pairs,
                                reason=f"High correlation with {existing_symbol} (ρ={corr:.2f}, max={self.max_position_correlation})"
                            )
                            self._log_result(result)
                            logger.warning(f"HIGH_CORRELATION: {symbol} vs {existing_symbol} ρ={corr:.2f}")
                            return result
                        
                        # Check if correlation warrants warning
                        if corr > self.warning_correlation_threshold:
                            logger.info(f"ELEVATED_CORRELATION: {symbol} vs {existing_symbol} ρ={corr:.2f} - consider risk")
        
        # All checks passed
        result = CorrelationCheckResult(
            status=CorrelationStatus.ACCEPTED,
            symbol=symbol,
            allocation=allocation,
            sector_exposure=new_sector_exposure,
            exchange_exposure=new_exchange_exposure,
            correlation_pairs=correlation_pairs,
            timestamp=datetime.utcnow(),
            reason="Position passes all concentration and correlation checks"
        )
        self._log_result(result)
        logger.info(f"CORRELATION_ACCEPTED: {symbol} allocation={allocation:.1f}%, sector_exp={new_sector_exposure:.1f}%, exchange_exp={new_exchange_exposure:.1f}%")
        return result
    
    def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to sector."""
        with self.lock:
            return sum(
                pos.allocation for pos in self.active_positions.values()
                if pos.sector == sector
            )
    
    def _calculate_exchange_exposure(self, exchange: str) -> float:
        """Calculate current exposure to exchange."""
        with self.lock:
            return sum(
                pos.allocation for pos in self.active_positions.values()
                if pos.exchange == exchange
            )
    
    def _log_result(self, result: CorrelationCheckResult) -> None:
        """Log validation result."""
        with self.lock:
            self.validation_log.append(result)
            if len(self.validation_log) > 1000:
                self.validation_log = self.validation_log[-1000:]
    
    def get_portfolio_risk_profile(self) -> Dict:
        """Get current portfolio risk metrics."""
        with self.lock:
            positions = dict(self.active_positions)
            correlation_matrix = self.correlation_matrix
        
        sector_exposure = {}
        exchange_exposure = {}
        total_allocation = 0
        
        for pos in positions.values():
            total_allocation += pos.allocation
            sector_exposure[pos.sector] = sector_exposure.get(pos.sector, 0) + pos.allocation
            exchange_exposure[pos.exchange] = exchange_exposure.get(pos.exchange, 0) + pos.allocation
        
        high_correlations = []
        if correlation_matrix:
            for (sym_a, sym_b), corr in correlation_matrix.correlations.items():
                if corr > self.warning_correlation_threshold:
                    high_correlations.append({
                        "pair": f"{sym_a}-{sym_b}",
                        "correlation": corr
                    })
        
        return {
            "total_positions": len(positions),
            "total_allocation": total_allocation,
            "sector_exposure": sector_exposure,
            "exchange_exposure": exchange_exposure,
            "high_correlations": high_correlations,
            "updated_at": datetime.utcnow().isoformat()
        }
    
    def get_validation_log(self, limit: int = 100) -> List[Dict]:
        """Get recent validation results."""
        with self.lock:
            logs = self.validation_log[-limit:]
        
        results = []
        for log in logs:
            results.append({
                "status": log.status.value,
                "symbol": log.symbol,
                "allocation": log.allocation,
                "sector_exposure": log.sector_exposure,
                "exchange_exposure": log.exchange_exposure,
                "timestamp": log.timestamp.isoformat(),
                "reason": log.reason,
                "correlation_pairs": [
                    {
                        "symbols": f"{p.symbol_a}-{p.symbol_b}",
                        "correlation": p.correlation
                    } for p in log.correlation_pairs
                ]
            })
        
        return results


# Global instance
_correlation_manager: Optional[PortfolioConcentrationManager] = None
_manager_lock = threading.Lock()


def get_correlation_manager() -> PortfolioConcentrationManager:
    """Get or create global correlation manager instance."""
    global _correlation_manager
    if _correlation_manager is None:
        with _manager_lock:
            if _correlation_manager is None:
                _correlation_manager = PortfolioConcentrationManager(
                    max_position_allocation=25.0,
                    max_sector_allocation=40.0,
                    max_exchange_allocation=50.0,
                    max_position_correlation=0.8
                )
    return _correlation_manager


def initialize_correlation_management() -> PortfolioConcentrationManager:
    """Initialize correlation management system at startup."""
    manager = get_correlation_manager()
    logger.info("Portfolio correlation manager initialized")
    return manager


if __name__ == "__main__":
    # Example usage
    manager = get_correlation_manager()
    
    # Simulate some historical returns
    returns_btc = [0.01, -0.005, 0.015, 0.002, -0.01] * 3  # Bitcoin returns
    returns_eth = [0.008, -0.003, 0.012, 0.001, -0.008] * 3  # Ethereum returns (correlated)
    returns_appl = [0.002, 0.001, -0.001, 0.003, 0.0] * 3  # Apple returns (uncorrelated)
    
    # Add positions
    manager.add_position("BTC", 20.0, "CRYPTO", "BINANCE", 45000, returns_btc)
    manager.add_position("ETH", 15.0, "CRYPTO", "BINANCE", 3000, returns_eth)
    
    # Check new position (correlated)
    result = manager.validate_new_position("AAPL", 18.0, "TECH", "NASDAQ", returns_appl)
    print(f"AAPL validation: {result.status.value}")
    
    # Get risk profile
    profile = manager.get_portfolio_risk_profile()
    print(f"Portfolio profile: {profile}")
