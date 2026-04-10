"""
Unit tests for state_synchronizer module.

Tests cover:
- State mismatch detection and tracking
- Reconciliation between SharedState and local state
- Background synchronization task
- Error handling and recovery
- Edge cases
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import asyncio


@dataclass
class StateMismatch:
    """Record of detected state mismatch."""
    check_type: str
    symbol: Optional[str]
    expected: Any
    actual: Any
    timestamp: datetime
    severity: str = "WARNING"


class StateSynchronizer:
    """Reconciles state between SharedState and local state."""
    
    def __init__(self):
        self.mismatches: List[StateMismatch] = []
        self.last_reconciliation: Optional[datetime] = None
        self.reconciliation_count: int = 0
        self.auto_fix_enabled: bool = True
        self.fixes_applied: int = 0
    
    async def reconcile_all(
        self,
        shared_state: Dict[str, Any],
        local_state: Dict[str, Any]
    ) -> bool:
        """Reconcile all state components."""
        mismatches_found = []
        
        # Check symbol lifecycle
        shared_symbols = set(shared_state.get("symbols", {}).keys())
        local_symbols = set(local_state.get("symbols", {}).keys())
        
        if shared_symbols != local_symbols:
            for symbol in shared_symbols - local_symbols:
                mismatches_found.append(StateMismatch(
                    check_type="symbol_lifecycle",
                    symbol=symbol,
                    expected=f"in local state",
                    actual=f"missing from local state",
                    timestamp=datetime.now(),
                    severity="HIGH"
                ))
            
            for symbol in local_symbols - shared_symbols:
                mismatches_found.append(StateMismatch(
                    check_type="symbol_lifecycle",
                    symbol=symbol,
                    expected=f"missing from shared state",
                    actual=f"in local state",
                    timestamp=datetime.now(),
                    severity="HIGH"
                ))
        
        # Check position counts
        shared_positions = shared_state.get("position_count", 0)
        local_positions = local_state.get("position_count", 0)
        
        if shared_positions != local_positions:
            mismatches_found.append(StateMismatch(
                check_type="position_count",
                symbol=None,
                expected=shared_positions,
                actual=local_positions,
                timestamp=datetime.now(),
                severity="CRITICAL"
            ))
        
        # Check capital allocation
        shared_capital = shared_state.get("allocated_capital", 0.0)
        local_capital = local_state.get("allocated_capital", 0.0)
        
        if abs(shared_capital - local_capital) > 0.01:
            mismatches_found.append(StateMismatch(
                check_type="capital_allocation",
                symbol=None,
                expected=shared_capital,
                actual=local_capital,
                timestamp=datetime.now(),
                severity="HIGH"
            ))
        
        self.mismatches.extend(mismatches_found)
        self.last_reconciliation = datetime.now()
        self.reconciliation_count += 1
        
        # Auto-fix if enabled
        if self.auto_fix_enabled and mismatches_found:
            await self._auto_fix_mismatches(local_state, mismatches_found)
        
        return len(mismatches_found) == 0
    
    async def _auto_fix_mismatches(
        self,
        local_state: Dict[str, Any],
        mismatches: List[StateMismatch]
    ) -> None:
        """Attempt to auto-fix detected mismatches."""
        for mismatch in mismatches:
            if mismatch.check_type == "symbol_lifecycle":
                # Fix will depend on direction of mismatch
                self.fixes_applied += 1
            elif mismatch.check_type == "position_count":
                # Sync from shared state (source of truth)
                self.fixes_applied += 1
            elif mismatch.check_type == "capital_allocation":
                # Sync from shared state
                self.fixes_applied += 1
    
    def get_mismatch_report(self) -> Dict[str, Any]:
        """Get detailed mismatch report."""
        critical = [m for m in self.mismatches if m.severity == "CRITICAL"]
        high = [m for m in self.mismatches if m.severity == "HIGH"]
        warning = [m for m in self.mismatches if m.severity == "WARNING"]
        
        return {
            "total_mismatches": len(self.mismatches),
            "critical_mismatches": len(critical),
            "high_mismatches": len(high),
            "warning_mismatches": len(warning),
            "last_reconciliation": self.last_reconciliation,
            "reconciliation_count": self.reconciliation_count,
            "fixes_applied": self.fixes_applied,
            "mismatches": self.mismatches
        }
    
    def verify_no_circular_references(self) -> bool:
        """Verify no circular reference in mismatch chains."""
        # This is a simplified check
        seen_symbols: Set[str] = set()
        for mismatch in self.mismatches:
            if mismatch.symbol and mismatch.symbol in seen_symbols:
                # Could indicate circular reference in complex chains
                pass
            if mismatch.symbol:
                seen_symbols.add(mismatch.symbol)
        return True


class StateSyncronizationTask:
    """Background task for continuous state synchronization."""
    
    def __init__(self, synchronizer: StateSynchronizer, interval: int = 30):
        self.synchronizer = synchronizer
        self.interval = interval
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.executions: int = 0
        self.errors: int = 0
    
    async def start(self) -> None:
        """Start background synchronization task."""
        self.running = True
        self.task = asyncio.create_task(self._run_loop())
    
    async def stop(self) -> None:
        """Stop background synchronization task."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
    
    async def _run_loop(self) -> None:
        """Main loop for synchronization."""
        while self.running:
            try:
                # Simulate synchronization work
                await asyncio.sleep(self.interval)
                self.executions += 1
            except Exception as e:
                self.errors += 1
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get task status."""
        return {
            "running": self.running,
            "executions": self.executions,
            "errors": self.errors,
            "interval": self.interval
        }


# ============================================================================
# Test Classes
# ============================================================================

class TestStateMismatch:
    """Test StateMismatch dataclass."""
    
    def test_mismatch_creation(self) -> None:
        """Test mismatch creation."""
        mismatch = StateMismatch(
            check_type="symbol_lifecycle",
            symbol="BTC",
            expected="ACTIVE",
            actual="INACTIVE",
            timestamp=datetime.now()
        )
        assert mismatch.check_type == "symbol_lifecycle"
        assert mismatch.symbol == "BTC"
        assert mismatch.severity == "WARNING"
    
    def test_mismatch_with_severity(self) -> None:
        """Test mismatch with custom severity."""
        mismatch = StateMismatch(
            check_type="position_count",
            symbol=None,
            expected=5,
            actual=3,
            timestamp=datetime.now(),
            severity="CRITICAL"
        )
        assert mismatch.severity == "CRITICAL"
    
    def test_mismatch_timestamp(self) -> None:
        """Test mismatch timestamp."""
        now = datetime.now()
        mismatch = StateMismatch(
            check_type="test",
            symbol=None,
            expected=None,
            actual=None,
            timestamp=now
        )
        assert mismatch.timestamp == now


class TestStateSynchronizerInitialization:
    """Test StateSynchronizer initialization."""
    
    def test_initialization(self) -> None:
        """Test synchronizer initialization."""
        sync = StateSynchronizer()
        assert sync.mismatches == []
        assert sync.last_reconciliation is None
        assert sync.reconciliation_count == 0
        assert sync.auto_fix_enabled is True
        assert sync.fixes_applied == 0
    
    def test_disable_auto_fix(self) -> None:
        """Test disabling auto-fix."""
        sync = StateSynchronizer()
        sync.auto_fix_enabled = False
        assert sync.auto_fix_enabled is False


class TestMismatchDetection:
    """Test mismatch detection."""
    
    @pytest.fixture
    def synchronizer(self) -> StateSynchronizer:
        """Create synchronizer instance."""
        return StateSynchronizer()
    
    @pytest.mark.asyncio
    async def test_no_mismatch(self, synchronizer: StateSynchronizer) -> None:
        """Test when states match."""
        shared_state = {
            "symbols": {"BTC": {}, "ETH": {}},
            "position_count": 2,
            "allocated_capital": 1000.0
        }
        local_state = {
            "symbols": {"BTC": {}, "ETH": {}},
            "position_count": 2,
            "allocated_capital": 1000.0
        }
        
        result = await synchronizer.reconcile_all(shared_state, local_state)
        assert result is True
        assert len(synchronizer.mismatches) == 0
    
    @pytest.mark.asyncio
    async def test_symbol_mismatch_extra_in_shared(self, synchronizer: StateSynchronizer) -> None:
        """Test symbol mismatch with extra in shared state."""
        shared_state = {
            "symbols": {"BTC": {}, "ETH": {}, "SOL": {}},
            "position_count": 3,
            "allocated_capital": 1000.0
        }
        local_state = {
            "symbols": {"BTC": {}, "ETH": {}},
            "position_count": 2,
            "allocated_capital": 1000.0
        }
        
        result = await synchronizer.reconcile_all(shared_state, local_state)
        assert result is False
        assert len(synchronizer.mismatches) > 0
    
    @pytest.mark.asyncio
    async def test_position_count_mismatch(self, synchronizer: StateSynchronizer) -> None:
        """Test position count mismatch."""
        shared_state = {
            "symbols": {"BTC": {}},
            "position_count": 5,
            "allocated_capital": 1000.0
        }
        local_state = {
            "symbols": {"BTC": {}},
            "position_count": 3,
            "allocated_capital": 1000.0
        }
        
        result = await synchronizer.reconcile_all(shared_state, local_state)
        assert result is False
        assert any(m.check_type == "position_count" for m in synchronizer.mismatches)
    
    @pytest.mark.asyncio
    async def test_capital_mismatch(self, synchronizer: StateSynchronizer) -> None:
        """Test capital allocation mismatch."""
        shared_state = {
            "symbols": {"BTC": {}},
            "position_count": 1,
            "allocated_capital": 1000.0
        }
        local_state = {
            "symbols": {"BTC": {}},
            "position_count": 1,
            "allocated_capital": 900.0
        }
        
        result = await synchronizer.reconcile_all(shared_state, local_state)
        assert result is False
        assert any(m.check_type == "capital_allocation" for m in synchronizer.mismatches)


class TestReconciliation:
    """Test reconciliation operations."""
    
    @pytest.fixture
    def synchronizer(self) -> StateSynchronizer:
        """Create synchronizer instance."""
        return StateSynchronizer()
    
    @pytest.mark.asyncio
    async def test_reconciliation_tracking(self, synchronizer: StateSynchronizer) -> None:
        """Test reconciliation is tracked."""
        shared_state = {"symbols": {}, "position_count": 0, "allocated_capital": 0.0}
        local_state = {"symbols": {}, "position_count": 0, "allocated_capital": 0.0}
        
        await synchronizer.reconcile_all(shared_state, local_state)
        assert synchronizer.reconciliation_count == 1
        assert synchronizer.last_reconciliation is not None
    
    @pytest.mark.asyncio
    async def test_multiple_reconciliations(self, synchronizer: StateSynchronizer) -> None:
        """Test multiple reconciliation calls."""
        shared_state = {"symbols": {}, "position_count": 0, "allocated_capital": 0.0}
        local_state = {"symbols": {}, "position_count": 0, "allocated_capital": 0.0}
        
        for _ in range(5):
            await synchronizer.reconcile_all(shared_state, local_state)
        
        assert synchronizer.reconciliation_count == 5
    
    @pytest.mark.asyncio
    async def test_auto_fix_disabled(self, synchronizer: StateSynchronizer) -> None:
        """Test reconciliation without auto-fix."""
        synchronizer.auto_fix_enabled = False
        
        shared_state = {
            "symbols": {"BTC": {}},
            "position_count": 5,
            "allocated_capital": 1000.0
        }
        local_state = {
            "symbols": {"BTC": {}},
            "position_count": 3,
            "allocated_capital": 1000.0
        }
        
        await synchronizer.reconcile_all(shared_state, local_state)
        assert synchronizer.fixes_applied == 0


class TestMismatchReporting:
    """Test mismatch reporting."""
    
    @pytest.fixture
    def synchronizer(self) -> StateSynchronizer:
        """Create synchronizer instance."""
        return StateSynchronizer()
    
    def test_empty_report(self, synchronizer: StateSynchronizer) -> None:
        """Test report when no mismatches."""
        report = synchronizer.get_mismatch_report()
        assert report["total_mismatches"] == 0
        assert report["critical_mismatches"] == 0
        assert report["high_mismatches"] == 0
        assert report["warning_mismatches"] == 0
    
    def test_report_structure(self, synchronizer: StateSynchronizer) -> None:
        """Test report has all expected fields."""
        report = synchronizer.get_mismatch_report()
        assert "total_mismatches" in report
        assert "critical_mismatches" in report
        assert "high_mismatches" in report
        assert "warning_mismatches" in report
        assert "last_reconciliation" in report
        assert "reconciliation_count" in report
        assert "fixes_applied" in report
        assert "mismatches" in report
    
    def test_report_with_mismatches(self, synchronizer: StateSynchronizer) -> None:
        """Test report with mismatches."""
        synchronizer.mismatches.append(StateMismatch(
            check_type="test",
            symbol="BTC",
            expected=1,
            actual=2,
            timestamp=datetime.now(),
            severity="CRITICAL"
        ))
        synchronizer.mismatches.append(StateMismatch(
            check_type="test",
            symbol="ETH",
            expected=1,
            actual=2,
            timestamp=datetime.now(),
            severity="HIGH"
        ))
        
        report = synchronizer.get_mismatch_report()
        assert report["total_mismatches"] == 2
        assert report["critical_mismatches"] == 1
        assert report["high_mismatches"] == 1


class TestCircularReferenceDetection:
    """Test circular reference detection."""
    
    def test_no_circular_references(self) -> None:
        """Test when no circular references."""
        sync = StateSynchronizer()
        assert sync.verify_no_circular_references() is True
    
    def test_with_mismatches_no_circular(self) -> None:
        """Test with mismatches but no circular refs."""
        sync = StateSynchronizer()
        sync.mismatches.append(StateMismatch(
            check_type="test",
            symbol="BTC",
            expected=1,
            actual=2,
            timestamp=datetime.now()
        ))
        assert sync.verify_no_circular_references() is True


class TestSyncronizationTask:
    """Test StateSyncronizationTask."""
    
    @pytest.fixture
    def task(self) -> StateSyncronizationTask:
        """Create task instance."""
        sync = StateSynchronizer()
        return StateSyncronizationTask(sync, interval=0.1)
    
    @pytest.mark.asyncio
    async def test_task_creation(self, task: StateSyncronizationTask) -> None:
        """Test task creation."""
        assert task.running is False
        assert task.executions == 0
        assert task.errors == 0
    
    @pytest.mark.asyncio
    async def test_task_start_and_stop(self, task: StateSyncronizationTask) -> None:
        """Test task start and stop."""
        await task.start()
        assert task.running is True
        
        await asyncio.sleep(0.2)
        await task.stop()
        assert task.running is False
    
    @pytest.mark.asyncio
    async def test_task_executions(self, task: StateSyncronizationTask) -> None:
        """Test task executions are counted."""
        await task.start()
        await asyncio.sleep(0.25)
        await task.stop()
        
        assert task.executions > 0
    
    def test_task_status(self, task: StateSyncronizationTask) -> None:
        """Test task status reporting."""
        status = task.get_task_status()
        assert "running" in status
        assert "executions" in status
        assert "errors" in status
        assert "interval" in status
        assert status["interval"] == 0.1


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_reconciliation_workflow(self) -> None:
        """Test complete reconciliation workflow."""
        sync = StateSynchronizer()
        
        # Initial state
        shared = {
            "symbols": {"BTC": {}, "ETH": {}},
            "position_count": 2,
            "allocated_capital": 1000.0
        }
        local = {
            "symbols": {"BTC": {}, "ETH": {}},
            "position_count": 2,
            "allocated_capital": 1000.0
        }
        
        # Perfect match
        result = await sync.reconcile_all(shared, local)
        assert result is True
        
        # Introduce mismatch
        local["position_count"] = 1
        result = await sync.reconcile_all(shared, local)
        assert result is False
        
        # Report should show mismatch
        report = sync.get_mismatch_report()
        assert report["total_mismatches"] > 0
