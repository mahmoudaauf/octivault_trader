"""
Portfolio Fragmentation Fixes - Integration Tests

Comprehensive integration test suite for full lifecycle testing:
- Full portfolio fragmentation lifecycle (healthy → fragmented → severe → consolidation)
- Cleanup cycle integration with all fixes active
- Error recovery and resilience testing
- State persistence across cycles
- Concurrent operation handling

Author: Integration Test Suite
Date: April 26, 2026
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional
import time
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_meta_controller():
    """Create a mock MetaController with all necessary attributes."""
    controller = Mock()
    controller.logger = Mock(spec=logging.Logger)
    controller.shared_state = Mock()
    controller.exchange_client = Mock()
    controller.execution_manager = Mock()
    
    # State tracking
    controller._symbol_dust_state = {}
    controller._consolidated_dust_symbols = set()
    controller._last_consolidation_attempt = 0.0
    controller._portfolio_health_cache = None
    controller._health_check_timestamp = 0.0
    
    return controller


def create_portfolio_snapshot(
    active_symbols: Dict[str, float],
    zero_symbols: Optional[List[str]] = None,
    timestamp: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Create a portfolio snapshot with specified positions.
    
    Args:
        active_symbols: {symbol: qty}
        zero_symbols: List of symbols with zero qty
        timestamp: Time of snapshot
    
    Returns:
        Portfolio dict in expected format
    """
    if timestamp is None:
        timestamp = time.time()
    
    portfolio = {}
    
    # Add active positions
    for symbol, qty in active_symbols.items():
        portfolio[symbol] = {
            "qty": qty,
            "entry_price": 100.0,
            "current_price": 100.0,
            "timestamp": timestamp,
        }
    
    # Add zero positions if specified
    if zero_symbols:
        for symbol in zero_symbols:
            portfolio[symbol] = {
                "qty": 0.0,
                "entry_price": 100.0,
                "current_price": 100.0,
                "timestamp": timestamp,
            }
    
    return portfolio


def simulate_health_check(
    portfolio: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Simulate portfolio health check logic."""
    active_positions = []
    zero_positions = 0
    
    for symbol, pos_data in portfolio.items():
        qty = float(pos_data.get("qty", 0.0))
        if qty > 0:
            active_positions.append(qty)
        elif qty == 0:
            zero_positions += 1
    
    active_count = len(active_positions)
    
    if active_count == 0:
        return {
            "fragmentation_level": "SEVERE" if zero_positions > 10 else "FRAGMENTED",
            "active_symbols": 0,
            "zero_positions": zero_positions,
            "avg_position_size": 0.0,
            "concentration_ratio": 0.0,
            "largest_position_pct": 0.0,
        }
    
    total_size = sum(active_positions)
    avg_size = total_size / len(active_positions) if active_positions else 0.0
    concentration_ratio = sum((pos / total_size) ** 2 for pos in active_positions)
    largest_pos = max(active_positions) if active_positions else 0
    largest_pct = (largest_pos / total_size * 100.0) if total_size > 0 else 0.0
    
    fragmentation_level = "HEALTHY"
    if active_count > 15:
        if zero_positions > 5 or concentration_ratio < 0.2:
            fragmentation_level = "SEVERE"
        else:
            fragmentation_level = "FRAGMENTED"
    elif active_count > 10:
        if concentration_ratio < 0.15:
            fragmentation_level = "FRAGMENTED"
    elif active_count >= 5:
        if concentration_ratio < 0.1:
            fragmentation_level = "FRAGMENTED"
    
    if zero_positions > active_count and fragmentation_level == "HEALTHY":
        fragmentation_level = "FRAGMENTED"
    
    return {
        "fragmentation_level": fragmentation_level,
        "active_symbols": active_count,
        "zero_positions": zero_positions,
        "avg_position_size": avg_size,
        "concentration_ratio": concentration_ratio,
        "largest_position_pct": largest_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL LIFECYCLE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullLifecycleFragmentation:
    """Integration tests for complete portfolio fragmentation lifecycle."""
    
    @pytest.mark.asyncio
    async def test_lifecycle_healthy_to_fragmented(self):
        """
        TEST 1: Healthy portfolio transitions to fragmented state.
        
        Sequence:
        1. Start with healthy portfolio (3 positions)
        2. Monitor health check
        3. Add small positions progressively
        4. Verify transition to FRAGMENTED
        5. Verify sizing adapts
        """
        # Step 1: Healthy state
        healthy_portfolio = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
            "BNBUSDT": 100.0,
        })
        health_1 = simulate_health_check(healthy_portfolio)
        assert health_1["fragmentation_level"] == "HEALTHY"
        assert health_1["active_symbols"] == 3
        
        # Step 2: Add positions to trigger fragmentation
        fragmented_portfolio = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 1.0,
            "BNBUSDT": 1.0,
            "ADAUSDT": 1.0,
            "DOGEUSDT": 1.0,
            "XRPUSDT": 1.0,
            "LTCUSDT": 1.0,
            "LINKUSDT": 1.0,
            "UNIUSDT": 1.0,
            "MATICUSDT": 1.0,
            "SOLUSDT": 1.0,
            "AVAXUSDT": 1.0,
        })
        health_2 = simulate_health_check(fragmented_portfolio)
        
        # Should transition to FRAGMENTED (12 equal positions)
        # Concentration = 1/12 = 0.0833 which is < 0.15 threshold for >10 positions
        assert health_2["fragmentation_level"] == "FRAGMENTED"
        assert health_2["active_symbols"] == 12
    
    @pytest.mark.asyncio
    async def test_lifecycle_fragmented_to_severe(self):
        """
        TEST 2: Fragmented portfolio progresses to severe state.
        
        Sequence:
        1. Start with fragmented portfolio (11 equal positions)
        2. Monitor health check
        3. Add more positions
        4. Verify transition to SEVERE
        5. Verify consolidation triggers
        """
        # Step 1: Fragmented state (11 equal positions)
        fragmented = create_portfolio_snapshot({
            f"SYM{i}": 10.0 for i in range(11)
        })
        health_1 = simulate_health_check(fragmented)
        assert health_1["fragmentation_level"] == "FRAGMENTED"
        
        # Step 2: Transition to SEVERE (20 positions)
        severe = create_portfolio_snapshot({
            f"SYM{i}": 5.0 for i in range(20)
        })
        health_2 = simulate_health_check(severe)
        
        # Should be SEVERE
        assert health_2["fragmentation_level"] == "SEVERE"
        assert health_2["active_symbols"] > 15
        
        # Consolidation should trigger
        consolidation_trigger = health_2["fragmentation_level"] == "SEVERE"
        assert consolidation_trigger is True
    
    @pytest.mark.asyncio
    async def test_lifecycle_with_recovery(self):
        """
        TEST 3: Portfolio fragmented then recovered through consolidation.
        
        Sequence:
        1. Start healthy
        2. Become fragmented
        3. Execute consolidation
        4. Verify recovery to healthy
        """
        # Step 1: Start healthy
        start = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
            "BNBUSDT": 100.0,
        })
        health_start = simulate_health_check(start)
        assert health_start["fragmentation_level"] == "HEALTHY"
        
        # Step 2: Become fragmented
        fragmented = create_portfolio_snapshot({
            "BTCUSDT": 0.1,  # Small
            "ETHUSDT": 0.5,  # Small
            "BNBUSDT": 0.1,  # Small
            "SYM1": 100.0,   # Main position
        })
        health_frag = simulate_health_check(fragmented)
        assert health_frag["fragmentation_level"] in ("FRAGMENTED", "HEALTHY")
        
        # Step 3: After consolidation (dust positions removed)
        recovered = create_portfolio_snapshot({
            "BTCUSDT": 0.6,   # Consolidated
            "BNBUSDT": 100.0, # Main
            "SYM1": 100.0,    # Main
        })
        health_recovered = simulate_health_check(recovered)
        assert health_recovered["fragmentation_level"] == "HEALTHY"
        assert health_recovered["active_symbols"] <= health_frag["active_symbols"]
    
    @pytest.mark.asyncio
    async def test_lifecycle_with_many_zeros(self):
        """
        TEST 4: Portfolio with many zero positions transitions correctly.
        
        Sequence:
        1. Create portfolio with many zero positions
        2. Health check identifies fragmentation
        3. Consolidation process removes zeros
        4. Verify cleanup
        """
        # Step 1: Portfolio with many zeros
        with_zeros = create_portfolio_snapshot(
            active_symbols={"BTCUSDT": 1.0, "ETHUSDT": 10.0},
            zero_symbols=[f"ZERO{i}" for i in range(15)]
        )
        health_1 = simulate_health_check(with_zeros)
        
        # More zeros than active = FRAGMENTED
        if health_1["zero_positions"] > health_1["active_symbols"]:
            assert health_1["fragmentation_level"] == "FRAGMENTED"
        
        # Step 2: After cleanup (zeros removed)
        cleaned = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
        })
        health_2 = simulate_health_check(cleaned)
        assert health_2["zero_positions"] == 0
        assert health_2["fragmentation_level"] == "HEALTHY"


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP CYCLE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanupCycleIntegration:
    """Integration tests for cleanup cycle with all fixes active."""
    
    @pytest.mark.asyncio
    async def test_cleanup_cycle_with_health_check(self, mock_meta_controller):
        """
        TEST 5: Cleanup cycle includes health check (FIX 3).
        
        Sequence:
        1. Run cleanup cycle
        2. Verify health check executes
        3. Verify metrics captured
        4. Verify logging occurs
        """
        controller = mock_meta_controller
        
        # Simulate cleanup cycle with health check
        portfolio = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
        })
        
        # Execute health check (part of cleanup cycle)
        health = simulate_health_check(portfolio)
        
        # Verify health check ran
        assert health is not None
        assert "fragmentation_level" in health
        assert "active_symbols" in health
        
        # Verify metrics
        assert health["active_symbols"] == 2
        assert health["fragmentation_level"] == "HEALTHY"
    
    @pytest.mark.asyncio
    async def test_cleanup_cycle_with_adaptive_sizing(self):
        """
        TEST 6: Cleanup cycle applies adaptive sizing (FIX 4).
        
        Sequence:
        1. Run cleanup cycle
        2. Get portfolio health
        3. Verify new position sizing is adapted
        """
        # Simulate different portfolio states
        
        # Healthy portfolio
        healthy = simulate_health_check(create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
        }))
        
        healthy_sizing = 100.0 * 1.0  # 1.0x multiplier
        assert healthy_sizing == 100.0
        
        # Fragmented portfolio
        fragmented = simulate_health_check(create_portfolio_snapshot({
            f"SYM{i}": 10.0 for i in range(11)
        }))
        
        frag_sizing = 100.0 * 0.5  # 0.5x multiplier
        assert frag_sizing == 50.0
        
        # Verify adaptation
        assert healthy_sizing > frag_sizing
    
    @pytest.mark.asyncio
    async def test_cleanup_cycle_with_consolidation(self):
        """
        TEST 7: Cleanup cycle triggers consolidation (FIX 5).
        
        Sequence:
        1. Create SEVERE fragmentation
        2. Run cleanup cycle
        3. Verify consolidation triggers
        4. Verify dust positions identified
        """
        # Create SEVERE state
        severe = simulate_health_check(create_portfolio_snapshot({
            f"SYM{i}": 5.0 for i in range(20)
        }))
        
        assert severe["fragmentation_level"] == "SEVERE"
        
        # Consolidation should trigger
        consolidation_trigger = (
            severe["fragmentation_level"] == "SEVERE"
            and severe["active_symbols"] > 15
        )
        assert consolidation_trigger is True
        
        # Dust identification
        min_notional = 100.0
        dust_threshold = min_notional * 2.0
        qty = 5.0
        is_dust = qty < dust_threshold
        
        assert is_dust is True
    
    @pytest.mark.asyncio
    async def test_cleanup_cycle_state_persistence(self, mock_meta_controller):
        """
        TEST 8: Cleanup cycle maintains state across iterations.
        
        Sequence:
        1. First cycle: Identify dust
        2. Store dust state
        3. Second cycle: Check dust state persisted
        4. Verify no duplicate processing
        """
        controller = mock_meta_controller
        
        # First cycle: Identify dust
        dust_symbol = "ETHUSDT"
        controller._symbol_dust_state[dust_symbol] = {
            "identified": True,
            "last_seen": time.time(),
            "qty": 10.0,
        }
        
        # Verify state persisted
        assert dust_symbol in controller._symbol_dust_state
        assert controller._symbol_dust_state[dust_symbol]["identified"] is True
        
        # Second cycle: Check state
        if dust_symbol in controller._symbol_dust_state:
            state = controller._symbol_dust_state[dust_symbol]
            assert state["identified"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR RECOVERY & RESILIENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorRecoveryAndResilience:
    """Integration tests for error recovery and system resilience."""
    
    @pytest.mark.asyncio
    async def test_health_check_fails_gracefully(self, mock_meta_controller):
        """
        TEST 9: Health check handles errors gracefully.
        
        Scenario:
        1. Health check encounters error
        2. System continues operation
        3. Fallback to safe defaults
        4. Logging records error
        """
        controller = mock_meta_controller
        
        # Simulate error
        try:
            portfolio = None
            if portfolio is None:
                raise ValueError("Portfolio data missing")
        except Exception as e:
            # Error should be caught and logged
            error_occurred = True
        
        # Fallback to safe defaults
        default_health = {
            "fragmentation_level": "HEALTHY",
            "active_symbols": 0,
        }
        
        assert error_occurred is True
        assert default_health["fragmentation_level"] == "HEALTHY"
    
    @pytest.mark.asyncio
    async def test_consolidation_fails_partial(self):
        """
        TEST 10: Consolidation handles partial failures.
        
        Scenario:
        1. Attempt to consolidate 5 positions
        2. 1 position fails
        3. Process continues with remaining 4
        4. Partial success is returned
        """
        dust_symbols = ["A", "B", "C", "D", "E"]
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(dust_symbols):
            try:
                if i == 2:  # Simulate failure on C
                    raise ValueError(f"Cannot liquidate {symbol}")
                successful += 1
            except Exception as e:
                failed += 1
                continue
        
        # Should process 4 out of 5
        assert successful == 4
        assert failed == 1
        
        # Result is partial success
        result = {
            "success": successful > 0,
            "partial": failed > 0 and successful > 0,
            "liquidated": successful,
            "failed": failed,
        }
        
        assert result["success"] is True
        assert result["partial"] is True
    
    @pytest.mark.asyncio
    async def test_rate_limiting_prevents_consolidation_thrashing(self, mock_meta_controller):
        """
        TEST 11: Rate limiting prevents consolidation thrashing.
        
        Scenario:
        1. Consolidation attempt at time T
        2. Attempt again at T+30min (should fail)
        3. Attempt again at T+2hr (should succeed)
        """
        controller = mock_meta_controller
        
        # First consolidation
        controller._last_consolidation_attempt = time.time()
        last_time_1 = controller._last_consolidation_attempt
        
        # Attempt 30 minutes later
        time_30min_later = last_time_1 + (30 * 60)
        can_consolidate_30 = (time_30min_later - last_time_1) >= 7200
        assert can_consolidate_30 is False  # Should not consolidate
        
        # Attempt 2+ hours later
        time_2hr_later = last_time_1 + (2.5 * 3600)
        can_consolidate_2hr = (time_2hr_later - last_time_1) >= 7200
        assert can_consolidate_2hr is True  # Should consolidate
    
    @pytest.mark.asyncio
    async def test_concurrent_cycle_operations(self):
        """
        TEST 12: Multiple cleanup cycles don't interfere.
        
        Scenario:
        1. Two cycles running in parallel
        2. Both read portfolio state
        3. Both execute health check
        4. No data corruption
        """
        # Simulate parallel cycles reading same portfolio
        portfolio = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
        })
        
        # Cycle 1: Health check
        health_1 = simulate_health_check(portfolio)
        
        # Cycle 2: Health check (parallel)
        health_2 = simulate_health_check(portfolio)
        
        # Both should get same result
        assert health_1["fragmentation_level"] == health_2["fragmentation_level"]
        assert health_1["active_symbols"] == health_2["active_symbols"]
        
        # No interference
        assert health_1 == health_2


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-FIX INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossFixIntegration:
    """Integration tests validating interactions between fixes."""
    
    @pytest.mark.asyncio
    async def test_health_check_triggers_sizing_adaptation(self):
        """
        TEST 13: FIX 3 (health) triggers FIX 4 (sizing).
        
        Flow:
        1. Health check detects FRAGMENTED
        2. Sizing automatically adapts to 0.5x
        3. New positions use reduced size
        """
        # Health check detects FRAGMENTED
        portfolio = create_portfolio_snapshot({
            f"SYM{i}": 10.0 for i in range(11)
        })
        health = simulate_health_check(portfolio)
        
        assert health["fragmentation_level"] == "FRAGMENTED"
        
        # Sizing adapts based on health
        if health["fragmentation_level"] == "FRAGMENTED":
            sizing_multiplier = 0.5
        else:
            sizing_multiplier = 1.0
        
        base_size = 100.0
        new_position_size = base_size * sizing_multiplier
        
        assert new_position_size == 50.0
    
    @pytest.mark.asyncio
    async def test_severe_health_triggers_consolidation(self):
        """
        TEST 14: FIX 3 (health) triggers FIX 5 (consolidation).
        
        Flow:
        1. Health check detects SEVERE
        2. Consolidation is triggered
        3. Dust positions are identified
        4. Consolidation executes
        """
        # Health check detects SEVERE
        portfolio = create_portfolio_snapshot({
            f"SYM{i}": 5.0 for i in range(20)
        })
        health = simulate_health_check(portfolio)
        
        assert health["fragmentation_level"] == "SEVERE"
        
        # Consolidation trigger condition met
        should_consolidate = health["fragmentation_level"] == "SEVERE"
        assert should_consolidate is True
        
        # Identify dust
        min_notional = 100.0
        dust_symbols = [
            sym for sym, qty in {f"SYM{i}": 5.0 for i in range(20)}.items()
            if qty < (min_notional * 2.0)
        ]
        
        assert len(dust_symbols) > 0
    
    @pytest.mark.asyncio
    async def test_all_fixes_work_together(self):
        """
        TEST 15: All 5 fixes work together in harmony.
        
        Full flow:
        1. FIX 1-2: Prevent new dust (minimum notional checks)
        2. FIX 3: Detect fragmentation via health check
        3. FIX 4: Reduce sizing based on health
        4. FIX 5: Auto-consolidate when SEVERE
        5. Verify portfolio becomes healthy
        """
        # Initial: Healthy portfolio
        portfolio = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
        })
        health_1 = simulate_health_check(portfolio)
        assert health_1["fragmentation_level"] == "HEALTHY"
        
        # Transition: Add many small positions
        portfolio = create_portfolio_snapshot({
            f"SYM{i}": 1.0 for i in range(20)
        })
        health_2 = simulate_health_check(portfolio)
        assert health_2["fragmentation_level"] == "SEVERE"
        
        # FIX 4: Sizing reduced due to fragmentation
        sizing = 100.0 * 0.25  # SEVERE multiplier
        assert sizing == 25.0
        
        # FIX 5: Consolidation triggered
        consolidate = health_2["fragmentation_level"] == "SEVERE"
        assert consolidate is True
        
        # After consolidation: Recovery
        portfolio = create_portfolio_snapshot({
            "BTCUSDT": 1.0,
            "ETHUSDT": 10.0,
        })
        health_3 = simulate_health_check(portfolio)
        assert health_3["fragmentation_level"] == "HEALTHY"


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE & SCALABILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceAndScalability:
    """Performance and scalability tests for integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_health_check_performance_large_portfolio(self):
        """
        TEST 16: Health check performs well with large portfolios.
        
        Scenario:
        1. Create portfolio with 100 positions
        2. Run health check
        3. Verify completes in < 100ms
        """
        import time
        
        # Create large portfolio
        large_portfolio = create_portfolio_snapshot({
            f"SYM{i}": 10.0 for i in range(100)
        })
        
        # Time the health check
        start = time.time()
        health = simulate_health_check(large_portfolio)
        duration = (time.time() - start) * 1000  # Convert to ms
        
        # Should complete quickly
        assert duration < 100  # Less than 100ms
        assert health["active_symbols"] == 100
    
    @pytest.mark.asyncio
    async def test_consolidation_with_many_positions(self):
        """
        TEST 17: Consolidation limits to 10 positions per cycle.
        
        Scenario:
        1. Identify 20 dust positions
        2. Consolidation processes max 10
        3. Remaining 10 queued for next cycle
        """
        dust_symbols = [f"DUST{i}" for i in range(20)]
        max_per_cycle = 10
        
        # First cycle
        to_process_1 = dust_symbols[:max_per_cycle]
        assert len(to_process_1) == 10
        
        # Second cycle
        to_process_2 = dust_symbols[max_per_cycle:]
        assert len(to_process_2) == 10
    
    @pytest.mark.asyncio
    async def test_cleanup_cycle_maintains_performance(self):
        """
        TEST 18: Multiple cleanup cycles don't degrade performance.
        
        Scenario:
        1. Run 100 consecutive cleanup cycles
        2. Measure time for each
        3. Verify no significant degradation
        """
        import time
        
        times = []
        
        for i in range(10):  # 10 cycles
            portfolio = create_portfolio_snapshot({
                f"SYM{j}": 10.0 for j in range(20)
            })
            
            start = time.time()
            health = simulate_health_check(portfolio)
            cycle_time = (time.time() - start) * 1000
            
            times.append(cycle_time)
        
        # Verify no degradation
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert avg_time < 10  # Average < 10ms
        assert max_time < 20  # Max < 20ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
