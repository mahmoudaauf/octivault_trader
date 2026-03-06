"""
Test suite for Phase 3: Dust Registry Lifecycle implementation.
Tests persistent storage and tracking of dust position lifecycle.
"""

import pytest
import asyncio
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.shared_state import SharedState, SharedStateConfig, DustRegistry, DustPosition


@pytest.fixture(autouse=True)
def cleanup_shared_dust_registry():
    """Auto-cleanup shared dust_registry.json in cwd after each test."""
    yield
    # After test completes, clean up any shared file in cwd
    shared_file = os.path.join(os.getcwd(), "dust_registry.json")
    if os.path.exists(shared_file):
        try:
            os.remove(shared_file)
        except Exception:
            pass


class TestDustRegistryBasics:
    """Test basic DustRegistry functionality."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_dust_registry_initialization(self):
        """Test that DustRegistry initializes correctly."""
        registry = DustRegistry(db_path=self.temp_dir)
        assert registry is not None
        assert registry.db_path == self.temp_dir
        assert registry.registry_file.endswith("dust_registry.json")
    
    def test_dust_registry_file_location(self):
        """Test that registry file is created in correct location."""
        registry = DustRegistry(db_path=self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "dust_registry.json")
        assert registry.registry_file == expected_file
    
    def test_dust_position_dataclass(self):
        """Test DustPosition dataclass creation and conversion."""
        pos = DustPosition(
            symbol="BTC",
            quantity=0.001,
            notional_usd=50.0,
            created_at=time.time(),
            status="NEW"
        )
        assert pos.symbol == "BTC"
        assert pos.quantity == 0.001
        assert pos.notional_usd == 50.0
        assert pos.status == "NEW"
        
        # Test to_dict
        pos_dict = pos.to_dict()
        assert pos_dict["symbol"] == "BTC"
        assert pos_dict["status"] == "NEW"
        
        # Test from_dict
        pos2 = DustPosition.from_dict(pos_dict)
        assert pos2.symbol == pos.symbol
        assert pos2.quantity == pos.quantity
    
    def test_empty_registry_initialization(self):
        """Test that empty registry has correct structure."""
        registry = DustRegistry(db_path=self.temp_dir)
        all_metrics = registry.get_all_metrics()
        assert isinstance(all_metrics, dict)
        assert "dust_positions" in all_metrics
        assert len(all_metrics["dust_positions"]) == 0


class TestDustPositionTracking:
    """Test dust position tracking functionality."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_mark_position_as_dust(self):
        """Test marking a position as dust."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        
        assert registry.is_dust_position("BTC") == True
        assert registry.get_dust_status("BTC") == "NEW"
    
    def test_mark_healing_started(self):
        """Test recording when healing starts."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        
        assert registry.get_dust_status("BTC") == "HEALING"
    
    def test_record_healing_attempt(self):
        """Test recording healing attempts."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        
        # Record multiple attempts
        registry.record_healing_attempt("BTC")
        assert registry.get_healing_attempts("BTC") == 1
        
        registry.record_healing_attempt("BTC")
        assert registry.get_healing_attempts("BTC") == 2
    
    def test_mark_healing_complete(self):
        """Test marking position as healed."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        registry.mark_healing_complete("BTC")
        
        assert registry.get_dust_status("BTC") == "HEALED"
    
    def test_dust_position_info(self):
        """Test retrieving full dust position info."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        registry.record_healing_attempt("BTC")
        
        info = registry.get_dust_position_info("BTC")
        assert info is not None
        assert info["symbol"] == "BTC"
        assert info["quantity"] == 0.001
        assert info["notional_usd"] == 50.0
        assert info["healing_attempts"] == 1
        assert info["status"] == "HEALING"
    
    def test_multiple_dust_positions(self):
        """Test tracking multiple dust positions."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_position_as_dust("ETH", 0.05, 75.0)
        registry.mark_position_as_dust("ADA", 10.0, 5.0)
        
        positions = registry.get_all_dust_positions()
        assert len(positions) == 3
        assert "BTC" in positions
        assert "ETH" in positions
        assert "ADA" in positions


class TestCircuitBreaker:
    """Test dust registry circuit breaker functionality."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trip_circuit_breaker(self):
        """Test tripping circuit breaker."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        
        assert registry.is_circuit_breaker_tripped("BTC") == False
        
        registry.trip_circuit_breaker("BTC")
        assert registry.is_circuit_breaker_tripped("BTC") == True
    
    def test_should_attempt_healing_with_tripped_breaker(self):
        """Test that healing is not attempted when circuit breaker is tripped."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        
        # Should attempt healing before breaker is tripped
        assert registry.should_attempt_healing("BTC") == True
        
        # Trip breaker
        registry.trip_circuit_breaker("BTC")
        
        # Should not attempt healing after breaker is tripped
        assert registry.should_attempt_healing("BTC") == False
    
    def test_reset_circuit_breaker(self):
        """Test resetting circuit breaker."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.trip_circuit_breaker("BTC")
        
        assert registry.is_circuit_breaker_tripped("BTC") == True
        
        registry.reset_circuit_breaker("BTC")
        assert registry.is_circuit_breaker_tripped("BTC") == False
    
    def test_should_not_attempt_healed_position(self):
        """Test that healing is not attempted for already healed positions."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        registry.mark_healing_complete("BTC")
        
        assert registry.should_attempt_healing("BTC") == False


class TestDustLifecycle:
    """Test full dust position lifecycle."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_lifecycle_new_to_healing_to_healed(self):
        """Test complete lifecycle from dust detection to healing completion."""
        registry = DustRegistry(db_path=self.temp_dir)
        
        # Phase 1: Detection (NEW)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        assert registry.get_dust_status("BTC") == "NEW"
        
        # Phase 2: Healing starts
        registry.mark_healing_started("BTC")
        assert registry.get_dust_status("BTC") == "HEALING"
        assert registry.should_attempt_healing("BTC") == True
        
        # Phase 3: Record attempts
        registry.record_healing_attempt("BTC")
        registry.record_healing_attempt("BTC")
        assert registry.get_healing_attempts("BTC") == 2
        
        # Phase 4: Healing complete
        registry.mark_healing_complete("BTC")
        assert registry.get_dust_status("BTC") == "HEALED"
        assert registry.should_attempt_healing("BTC") == False
    
    def test_lifecycle_with_circuit_breaker_trip(self):
        """Test lifecycle where circuit breaker gets tripped."""
        registry = DustRegistry(db_path=self.temp_dir)
        
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        
        # Multiple unsuccessful attempts
        for i in range(5):
            registry.record_healing_attempt("BTC")
        
        # Trip breaker due to ineffective healing
        registry.trip_circuit_breaker("BTC")
        
        assert registry.should_attempt_healing("BTC") == False
        assert registry.is_circuit_breaker_tripped("BTC") == True
        
        # Reset and try again
        registry.reset_circuit_breaker("BTC")
        assert registry.should_attempt_healing("BTC") == True
    
    def test_persistence_survives_reload(self):
        """Test that dust tracking survives reload from disk."""
        registry1 = DustRegistry(db_path=self.temp_dir)
        registry1.mark_position_as_dust("BTC", 0.001, 50.0)
        registry1.mark_healing_started("BTC")
        registry1.record_healing_attempt("BTC")
        
        # Create new instance (simulates restart)
        registry2 = DustRegistry(db_path=self.temp_dir)
        
        assert registry2.is_dust_position("BTC") == True
        assert registry2.get_dust_status("BTC") == "HEALING"
        assert registry2.get_healing_attempts("BTC") == 1


class TestDustRegistryCleanup:
    """Test dust registry cleanup functionality."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cleanup_abandoned_dust(self):
        """Test removing dust that's been healing too long."""
        registry = DustRegistry(db_path=self.temp_dir)
        
        # Create dust with old timestamp
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        
        # Manually set healing start time to 40 days ago
        positions = registry._cached_registry["dust_positions"]
        positions["BTC"]["first_healing_attempt_at"] = time.time() - (40 * 24 * 3600)
        positions["BTC"]["healing_days_elapsed"] = 40.0
        registry._write(registry._cached_registry)
        
        # Cleanup with 30 day threshold
        cleaned = registry.cleanup_abandoned_dust(days_threshold=30.0)
        
        assert "BTC" in cleaned
        assert registry.get_dust_status("BTC") == "ABANDONED"
    
    def test_get_dust_summary(self):
        """Test getting summary statistics of dust registry."""
        registry = DustRegistry(db_path=self.temp_dir)
        
        # Create various dust positions
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_position_as_dust("ETH", 0.05, 75.0)
        registry.mark_position_as_dust("ADA", 10.0, 5.0)
        
        # Change status of some
        registry.mark_healing_started("BTC")
        registry.mark_healing_complete("ADA")
        
        summary = registry.get_dust_summary()
        
        assert summary["total_dust_symbols"] == 3
        assert summary["total_dust_notional"] == 130.0  # 50 + 75 + 5
        assert summary["by_status"]["NEW"] == 1  # ETH
        assert summary["by_status"]["HEALING"] == 1  # BTC
        assert summary["by_status"]["HEALED"] == 1  # ADA
    
    def test_mark_healed_keeps_history(self):
        """Test that marking healed keeps history for analytics."""
        registry = DustRegistry(db_path=self.temp_dir)
        registry.mark_position_as_dust("BTC", 0.001, 50.0)
        registry.mark_healing_started("BTC")
        registry.mark_healing_complete("BTC")
        
        # Position should still be tracked but marked as healed
        assert registry.is_dust_position("BTC") == True
        assert registry.get_dust_status("BTC") == "HEALED"


class TestDustRegistryIntegration:
    """Test DustRegistry integration with SharedState."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_shared_state_has_dust_registry(self):
        """Test that SharedState has dust_lifecycle_registry instance."""
        config = {
            "DB_PATH": self.temp_dir,
            "COLD_BOOTSTRAP_ENABLED": False,
            "LIVE_MODE": False,
        }
        shared_state = SharedState(config=config)
        
        assert hasattr(shared_state, 'dust_lifecycle_registry')
        assert isinstance(shared_state.dust_lifecycle_registry, DustRegistry)
    
    def test_dust_registry_loads_persisted_data(self):
        """Test that DustRegistry loads persisted data on restart."""
        # First instance: create dust
        registry1 = DustRegistry(db_path=self.temp_dir)
        registry1.mark_position_as_dust("BTC", 0.001, 50.0)
        registry1.mark_healing_started("BTC")
        registry1.record_healing_attempt("BTC")
        
        # Second instance: should load data
        registry2 = DustRegistry(db_path=self.temp_dir)
        assert registry2.is_dust_position("BTC") == True
        assert registry2.get_dust_status("BTC") == "HEALING"
        assert registry2.get_healing_attempts("BTC") == 1
    
    def test_shared_state_dust_registry_persistence(self):
        """Test that SharedState's dust_lifecycle_registry persists across instances."""
        # First SharedState instance
        config1 = {
            "DB_PATH": os.path.join(self.temp_dir, "unique_test_" + str(time.time()).replace(".", "_")),
            "COLD_BOOTSTRAP_ENABLED": False,
            "LIVE_MODE": False,
        }
        os.makedirs(config1["DB_PATH"], exist_ok=True)
        shared_state1 = SharedState(config=config1)
        shared_state1.dust_lifecycle_registry.mark_position_as_dust("BTC", 0.001, 50.0)
        
        # Second SharedState instance with same DB_PATH
        config2 = {
            "DB_PATH": config1["DB_PATH"],
            "COLD_BOOTSTRAP_ENABLED": False,
            "LIVE_MODE": False,
        }
        shared_state2 = SharedState(config=config2)
        
        # Should have loaded the dust position
        assert shared_state2.dust_lifecycle_registry.is_dust_position("BTC") == True


class TestDustRegistryEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_handles_missing_registry_file(self):
        """Test that missing registry file is handled gracefully."""
        registry = DustRegistry(db_path=self.temp_dir)
        # Should initialize with empty structure, no errors
        assert registry.get_all_dust_positions() == {}
    
    def test_handles_corrupted_json(self):
        """Test that corrupted JSON is handled gracefully."""
        # Create corrupted JSON file
        registry_file = os.path.join(self.temp_dir, "dust_registry.json")
        with open(registry_file, 'w') as f:
            f.write("{ invalid json }}")
        
        # Should load empty instead of crashing
        registry = DustRegistry(db_path=self.temp_dir)
        assert registry.get_all_dust_positions() == {}
    
    def test_handles_none_db_path(self):
        """Test that None db_path defaults to cwd."""
        # Create with None path - should use cwd
        unique_dir = os.path.join(self.temp_dir, "none_path_test_" + str(time.time()).replace(".", "_"))
        os.makedirs(unique_dir, exist_ok=True)
        original_cwd = os.getcwd()
        
        try:
            os.chdir(unique_dir)
            registry = DustRegistry(db_path=None)
            # Use realpath to handle /private symlink on macOS
            assert os.path.realpath(registry.db_path) == os.path.realpath(unique_dir)
        finally:
            os.chdir(original_cwd)
    
    def test_atomic_writes_prevent_corruption(self):
        """Test that atomic writes prevent partial writes."""
        registry = DustRegistry(db_path=self.temp_dir)
        
        # Create multiple dust positions
        for i in range(5):
            symbol = f"COIN{i}"
            registry.mark_position_as_dust(symbol, 0.1 * i, 50.0 * i)
        
        # Create new instance - should have all data
        registry2 = DustRegistry(db_path=self.temp_dir)
        positions = registry2.get_all_dust_positions()
        assert len(positions) == 5
    
    def test_operations_on_nonexistent_position(self):
        """Test that operations on nonexistent positions don't crash."""
        registry = DustRegistry(db_path=self.temp_dir)
        
        # These should not crash
        assert registry.is_dust_position("NONEXISTENT") == False
        assert registry.get_dust_status("NONEXISTENT") is None
        assert registry.get_healing_attempts("NONEXISTENT") == 0
        assert registry.get_dust_position_info("NONEXISTENT") is None
        assert registry.is_circuit_breaker_tripped("NONEXISTENT") == False
        assert registry.should_attempt_healing("NONEXISTENT") == False
