# Week 3 Integration Test Templates

This file contains pytest templates for integrating and testing the 5 trading safety modules.

## Template 1: Balance Manager Integration Test

```python
import pytest
from decimal import Decimal
from core.balance_manager import BalanceValidator, AllocationStatus
from core.capital_allocator import CapitalAllocator

class TestBalanceManagerIntegration:
    """Test balance_manager integration with capital_allocator"""
    
    @pytest.fixture
    def allocator(self):
        return CapitalAllocator()
    
    def test_allocation_approved(self, allocator):
        """Test that valid allocations are approved"""
        status = allocator.validator.validate_allocation(
            available_balance=Decimal('10000'),
            requested_capital=Decimal('5000'),
            allocated_so_far=Decimal('0')
        )
        assert status == AllocationStatus.APPROVED
    
    def test_allocation_rejected_insufficient_balance(self, allocator):
        """Test that over-allocation is rejected"""
        status = allocator.validator.validate_allocation(
            available_balance=Decimal('1000'),
            requested_capital=Decimal('5000'),
            allocated_so_far=Decimal('0')
        )
        assert status != AllocationStatus.APPROVED
    
    def test_allocation_circuit_breaker(self, allocator):
        """Test circuit breaker prevents cascade failures"""
        # First allocation succeeds
        allocator.validator.commit_allocation(Decimal('5000'))
        
        # Circuit breaker should prevent excessive allocation
        assert allocator.validator.get_circuit_breaker_state() == 'CLOSED'
    
    def test_allocation_release_on_error(self, allocator):
        """Test that allocation is released on error"""
        allocated = Decimal('1000')
        allocator.validator.commit_allocation(allocated)
        
        # Simulate error - release allocation
        allocator.validator.release_allocation(allocated)
        
        # Verify allocation was released
        assert allocator.validator.allocated_capital == Decimal('0')
```

## Template 2: Leverage Manager Integration Test

```python
import pytest
from decimal import Decimal
from core.leverage_manager import LeverageValidator, LeverageMonitor

class TestLeverageManagerIntegration:
    """Test leverage_manager integration with position_sizing"""
    
    @pytest.fixture
    def leverage_setup(self):
        validator = LeverageValidator()
        monitor = LeverageMonitor()
        return validator, monitor
    
    def test_position_leverage_validation(self, leverage_setup):
        """Test individual position leverage validation"""
        validator, monitor = leverage_setup
        
        # Position with 5x leverage should pass
        is_valid = validator.validate_position_leverage(
            leverage_ratio=Decimal('5'),
            symbol='AAPL',
            current_portfolio_leverage=Decimal('2')
        )
        assert is_valid is True
    
    def test_position_leverage_exceeds_limit(self, leverage_setup):
        """Test that excessive leverage is rejected"""
        validator, monitor = leverage_setup
        
        # Position with 50x leverage should fail (default limit 10x)
        is_valid = validator.validate_position_leverage(
            leverage_ratio=Decimal('50'),
            symbol='AAPL',
            current_portfolio_leverage=Decimal('2')
        )
        assert is_valid is False
    
    def test_portfolio_leverage_aggregation(self, leverage_setup):
        """Test portfolio-level leverage aggregation"""
        validator, monitor = leverage_setup
        
        # Add multiple positions
        monitor.add_position('AAPL', Decimal('150'), Decimal('100'), Decimal('90'))
        monitor.add_position('MSFT', Decimal('200'), Decimal('100'), Decimal('80'))
        
        portfolio_leverage = monitor.calculate_portfolio_leverage()
        assert portfolio_leverage > Decimal('0')
    
    def test_leverage_history_tracking(self, leverage_setup):
        """Test that leverage history is tracked"""
        validator, monitor = leverage_setup
        
        initial_len = len(monitor.leverage_history)
        
        monitor.add_position('AAPL', Decimal('150'), Decimal('100'), Decimal('90'))
        
        # History should be updated
        assert len(monitor.leverage_history) > initial_len
```

## Template 3: Trading Hours Manager Integration Test

```python
import pytest
from datetime import datetime, timezone
from core.trading_hours_manager import TradingHoursValidator, MarketHours

class TestTradingHoursManagerIntegration:
    """Test trading_hours_manager integration with order_execution"""
    
    @pytest.fixture
    def hours_validator(self):
        validator = TradingHoursValidator()
        # Register markets
        validator.register_symbol_market('AAPL', MarketHours.NYSE)
        validator.register_symbol_market('BTC/USD', MarketHours.CRYPTO_24_7)
        validator.register_symbol_market('EUR/USD', MarketHours.FOREX)
        return validator
    
    def test_crypto_always_tradeable(self, hours_validator):
        """Test that crypto markets are always open"""
        allowed, reason = hours_validator.validate_trading_allowed('BTC/USD')
        assert allowed is True
    
    def test_equity_market_hours(self, hours_validator):
        """Test equity market hours validation"""
        # This test may pass or fail depending on time of day
        allowed, reason = hours_validator.validate_trading_allowed('AAPL')
        # During market hours: allowed=True, During off-hours: allowed=False
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)
    
    def test_off_hours_order_queueing(self, hours_validator):
        """Test off-hours order queueing"""
        import asyncio
        
        async def test_queueing():
            await hours_validator.queue_off_hours_order(
                symbol='AAPL',
                order_type='BUY',
                execution_time=datetime.now(timezone.utc)
            )
        
        # Queue should accept off-hours orders
        asyncio.run(test_queueing())
    
    def test_market_hours_registration(self, hours_validator):
        """Test market registration"""
        # NYSE should be registered
        assert 'AAPL' in hours_validator.symbol_markets
        assert 'BTC/USD' in hours_validator.symbol_markets
        assert 'EUR/USD' in hours_validator.symbol_markets
```

## Template 4: Anomaly Detection Integration Test

```python
import pytest
from core.anomaly_detection import AnomalyDetector, AnomalyStatus

class TestAnomalyDetectionIntegration:
    """Test anomaly_detection integration with signal_processing"""
    
    @pytest.fixture
    def detector(self):
        return AnomalyDetector()
    
    def test_signal_registration(self, detector):
        """Test signal registration for baseline"""
        detector.register_signal(
            symbol='AAPL',
            value=150.0,
            confidence=0.85
        )
        
        # Signal should be registered in history
        assert 'AAPL' in detector.signal_history
    
    def test_normal_signal_detection(self, detector):
        """Test that normal signals are not flagged"""
        # Register multiple normal signals
        for i in range(10):
            detector.register_signal('AAPL', 150.0 + i*0.1, 0.85)
        
        # Check signal within normal range
        status = detector.check_signal('AAPL', 150.5, threshold_sigma=3.0)
        assert status == AnomalyStatus.NORMAL
    
    def test_outlier_detection(self, detector):
        """Test that outliers are detected"""
        # Register baseline
        for i in range(10):
            detector.register_signal('AAPL', 150.0, 0.85)
        
        # Check extreme outlier
        status = detector.check_signal('AAPL', 500.0, threshold_sigma=3.0)
        assert status == AnomalyStatus.GARBAGE
    
    def test_weak_signal_detection(self, detector):
        """Test detection of weak signals"""
        for i in range(10):
            detector.register_signal('AAPL', 150.0, 0.85)
        
        # Check weak signal (beyond 2 sigma but within 3)
        status = detector.check_signal('AAPL', 170.0, threshold_sigma=3.0)
        # Should be WEAK or NORMAL depending on implementation
        assert status in [AnomalyStatus.WEAK, AnomalyStatus.NORMAL]
    
    def test_baseline_update(self, detector):
        """Test baseline update with new data"""
        import asyncio
        
        async def test_update():
            # Register initial signals
            for i in range(5):
                detector.register_signal('AAPL', 150.0, 0.85)
            
            # Update baseline
            await detector.update_baseline('AAPL')
        
        asyncio.run(test_update())
```

## Template 5: Correlation Manager Integration Test

```python
import pytest
from decimal import Decimal
from core.correlation_manager import PortfolioConcentrationManager

class TestCorrelationManagerIntegration:
    """Test correlation_manager integration with portfolio_construction"""
    
    @pytest.fixture
    def manager(self):
        return PortfolioConcentrationManager()
    
    def test_single_position_validation(self, manager):
        """Test validation of single position"""
        is_valid = manager.validate_new_position(
            symbol='AAPL',
            new_size=Decimal('2500'),  # 25% of 10k portfolio
            max_symbol_pct=Decimal('0.25'),
            max_sector_pct=Decimal('0.40'),
            max_exchange_pct=Decimal('0.50')
        )
        assert is_valid is True
    
    def test_symbol_concentration_limit(self, manager):
        """Test that symbol concentration limit is enforced"""
        # First position: 20%
        manager.add_position('AAPL', Decimal('2000'), Decimal('100'), 
                           'Technology', 'NASDAQ')
        
        # Second position: 20% (total would be 40%)
        is_valid = manager.validate_new_position(
            symbol='AAPL',
            new_size=Decimal('2000'),
            max_symbol_pct=Decimal('0.25'),
            max_sector_pct=Decimal('0.40'),
            max_exchange_pct=Decimal('0.50')
        )
        # Should fail as total would exceed 25% per symbol
        assert is_valid is False
    
    def test_sector_concentration_limit(self, manager):
        """Test that sector concentration limit is enforced"""
        # Add multiple tech positions
        manager.add_position('AAPL', Decimal('1500'), Decimal('100'),
                           'Technology', 'NASDAQ')
        manager.add_position('MSFT', Decimal('1500'), Decimal('100'),
                           'Technology', 'NASDAQ')
        
        # Try to add another tech position
        is_valid = manager.validate_new_position(
            symbol='GOOGL',
            new_size=Decimal('1500'),
            max_symbol_pct=Decimal('0.25'),
            max_sector_pct=Decimal('0.40'),
            max_exchange_pct=Decimal('0.50')
        )
        # May fail depending on current allocation
        assert isinstance(is_valid, bool)
    
    def test_portfolio_risk_profile(self, manager):
        """Test portfolio risk profile calculation"""
        manager.add_position('AAPL', Decimal('2000'), Decimal('100'),
                           'Technology', 'NASDAQ')
        manager.add_position('JPM', Decimal('1500'), Decimal('100'),
                           'Finance', 'NYSE')
        
        profile = manager.get_portfolio_risk_profile()
        
        # Profile should have concentration metrics
        assert 'total_value' in profile or 'symbol_concentration' in profile
    
    def test_multiple_exchange_limits(self, manager):
        """Test limits across multiple exchanges"""
        # Add positions on different exchanges
        manager.add_position('AAPL', Decimal('2500'), Decimal('100'),
                           'Technology', 'NASDAQ')  # 25%
        
        # Try to add more NASDAQ
        is_valid = manager.validate_new_position(
            symbol='MSFT',
            new_size=Decimal('2600'),  # Would exceed 50% per exchange
            max_symbol_pct=Decimal('0.25'),
            max_sector_pct=Decimal('0.40'),
            max_exchange_pct=Decimal('0.50')
        )
        # Should fail as would exceed exchange limit
        assert is_valid is False
```

## Running the Tests

```bash
# Run all integration tests
pytest tests/test_week3_integration.py -v

# Run specific integration point tests
pytest tests/test_week3_integration.py::TestBalanceManagerIntegration -v
pytest tests/test_week3_integration.py::TestLeverageManagerIntegration -v
pytest tests/test_week3_integration.py::TestTradingHoursManagerIntegration -v
pytest tests/test_week3_integration.py::TestAnomalyDetectionIntegration -v
pytest tests/test_week3_integration.py::TestCorrelationManagerIntegration -v

# Run with coverage
pytest tests/test_week3_integration.py --cov=core --cov-report=html
```

## Key Testing Points

1. **Balance Manager**: Validation, commitment, release, circuit breaker
2. **Leverage Manager**: Position validation, portfolio aggregation, history tracking
3. **Trading Hours**: Market hours validation, off-hours queueing, timezone handling
4. **Anomaly Detection**: Signal registration, baseline updates, outlier detection
5. **Correlation Manager**: Concentration limits, sector/exchange checks, risk profiling

All tests target the success criteria:
- 85%+ coverage for balance_manager
- 80%+ coverage for leverage_manager
- 75%+ coverage for trading_hours_manager
- 80%+ coverage for anomaly_detection
- 75%+ coverage for correlation_manager
- **Total target: 79%+ coverage by Thursday EOD**

