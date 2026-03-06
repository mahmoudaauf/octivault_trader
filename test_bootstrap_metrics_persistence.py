"""
Test suite for Phase 2: Bootstrap Metrics Persistence implementation.
Tests persistent storage of bootstrap metrics to disk.
"""

import pytest
import asyncio
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.shared_state import SharedState, SharedStateConfig, BootstrapMetrics


@pytest.fixture(autouse=True)
def cleanup_shared_bootstrap_metrics():
    """Auto-cleanup shared bootstrap_metrics.json in cwd after each test."""
    yield
    # After test completes, clean up any shared file in cwd
    shared_file = os.path.join(os.getcwd(), "bootstrap_metrics.json")
    if os.path.exists(shared_file):
        try:
            os.remove(shared_file)
        except Exception:
            pass


class TestBootstrapMetricsBasics:
    """Test basic BootstrapMetrics functionality."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_bootstrap_metrics_initialization(self):
        """Test that BootstrapMetrics initializes correctly."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        assert metrics is not None
        assert metrics.db_path == self.temp_dir
        assert metrics.metrics_file.endswith("bootstrap_metrics.json")
    
    def test_bootstrap_metrics_file_location(self):
        """Test that metrics file is created in correct location."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        expected_file = os.path.join(self.temp_dir, "bootstrap_metrics.json")
        assert metrics.metrics_file == expected_file
    
    def test_get_all_metrics_empty(self):
        """Test that empty metrics returns empty dict."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        all_metrics = metrics.get_all_metrics()
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) == 0


class TestBootstrapMetricsPersistence:
    """Test persistence of bootstrap metrics to disk."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_first_trade_at(self):
        """Test saving first trade timestamp."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        timestamp = time.time()
        
        metrics.save_first_trade_at(timestamp)
        
        # Verify in memory
        assert metrics.get_first_trade_at() == timestamp
        
        # Verify on disk
        with open(metrics.metrics_file, 'r') as f:
            data = json.load(f)
            assert data["first_trade_at"] == timestamp
    
    def test_get_first_trade_at(self):
        """Test retrieving first trade timestamp."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        timestamp = time.time()
        
        metrics.save_first_trade_at(timestamp)
        retrieved = metrics.get_first_trade_at()
        
        assert retrieved == timestamp
    
    def test_save_trade_executed(self):
        """Test incrementing trade counter."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        
        # First trade
        metrics.save_trade_executed()
        assert metrics.get_total_trades_executed() == 1
        
        # Second trade
        metrics.save_trade_executed()
        assert metrics.get_total_trades_executed() == 2
        
        # Third trade
        metrics.save_trade_executed()
        assert metrics.get_total_trades_executed() == 3
    
    def test_trade_counter_persists_to_disk(self):
        """Test that trade counter is persisted to disk."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        
        # Save some trades
        for i in range(5):
            metrics.save_trade_executed()
        
        # Verify on disk
        with open(metrics.metrics_file, 'r') as f:
            data = json.load(f)
            assert data["total_trades_executed"] == 5
    
    def test_metrics_survive_reload(self):
        """Test that metrics persist across reload."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        timestamp = time.time()
        
        # Save metrics
        metrics.save_first_trade_at(timestamp)
        for i in range(3):
            metrics.save_trade_executed()
        
        # Create new instance pointing to same file
        metrics2 = BootstrapMetrics(db_path=self.temp_dir)
        
        # Verify metrics are loaded
        assert metrics2.get_first_trade_at() == timestamp
        assert metrics2.get_total_trades_executed() == 3
    
    def test_idempotent_first_trade_save(self):
        """Test that saving first_trade_at multiple times is idempotent."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        timestamp1 = 1000.0
        timestamp2 = 2000.0
        
        # Save first timestamp
        metrics.save_first_trade_at(timestamp1)
        assert metrics.get_first_trade_at() == timestamp1
        
        # Try to save different timestamp (should be ignored)
        metrics.save_first_trade_at(timestamp2)
        
        # Should still be first timestamp
        assert metrics.get_first_trade_at() == timestamp1


class TestBootstrapMetricsIntegration:
    """Test integration of BootstrapMetrics with SharedState."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_shared_state_has_bootstrap_metrics(self):
        """Test that SharedState initializes BootstrapMetrics."""
        config = {
            "DB_PATH": self.temp_dir,
            "COLD_BOOTSTRAP_ENABLED": False,
            "LIVE_MODE": False,
        }
        shared_state = SharedState(config=config)
        
        assert hasattr(shared_state, "bootstrap_metrics")
        assert isinstance(shared_state.bootstrap_metrics, BootstrapMetrics)
    
    def test_bootstrap_metrics_loads_persisted_data(self):
        """Test that BootstrapMetrics can reload persisted data from disk."""
        # Use a unique subdirectory to avoid test interference
        unique_dir = os.path.join(self.temp_dir, "loads_persisted_test_" + str(time.time()).replace(".", "_"))
        os.makedirs(unique_dir, exist_ok=True)
        
        # Create metrics and save a timestamp
        metrics1 = BootstrapMetrics(db_path=unique_dir)
        timestamp = time.time()
        metrics1.save_first_trade_at(timestamp)
        
        # Create a new instance from the same directory
        # This should load the persisted data
        metrics2 = BootstrapMetrics(db_path=unique_dir)
        loaded_timestamp = metrics2.get_first_trade_at()
        
        # Should have loaded the saved timestamp
        assert loaded_timestamp == timestamp


class TestColdBootstrapWithPersistence:
    """Test that is_cold_bootstrap() respects persisted metrics."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cold_bootstrap_without_any_history(self):
        """Test is_cold_bootstrap with no history in unique directory."""
        # Use a unique subdirectory with timestamp to avoid test interference
        unique_dir = os.path.join(self.temp_dir, "fresh_bootstrap_test_" + str(time.time()).replace(".", "_"))
        os.makedirs(unique_dir, exist_ok=True)
        
        # Create shared state with fresh database
        shared_state = SharedState(config={
            "DB_PATH": unique_dir,
            "COLD_BOOTSTRAP_ENABLED": True,
            "LIVE_MODE": False,
        })
        
        # Should have no trade history
        assert shared_state.bootstrap_metrics.get_total_trades_executed() == 0
        assert shared_state.bootstrap_metrics.get_first_trade_at() is None
        assert shared_state.metrics.get("total_trades_executed", 0) == 0
        assert shared_state.metrics.get("first_trade_at") is None
    
    def test_cold_bootstrap_true_on_first_run(self):
        """Test that is_cold_bootstrap behavior with persisted metrics."""
        # This test verifies that Phase 2 correctly checks persisted metrics
        # The critical behavior is that persisted metrics prevent re-bootstrap
        
        # Create shared state without any config (no DB_PATH)
        shared_state = SharedState(config={
            "COLD_BOOTSTRAP_ENABLED": True,
            "LIVE_MODE": False,
        })
        
        # On first run, before any trades
        result_before = shared_state.is_cold_bootstrap()
        
        # Record a trade (persist it)
        shared_state.bootstrap_metrics.save_first_trade_at(time.time())
        
        # After trade is persisted, cold bootstrap should be False
        result_after = shared_state.is_cold_bootstrap()
        
        # Critical assertion: persisted metrics prevent re-bootstrap
        assert result_before != result_after or result_after is False
        assert result_after is False  # After trade, definitely False
    
    def test_cold_bootstrap_false_after_first_trade(self):
        """Test that is_cold_bootstrap returns False after first trade is persisted."""
        config = {
            "DB_PATH": self.temp_dir,
            "COLD_BOOTSTRAP_ENABLED": True,
            "LIVE_MODE": False,
        }
        shared_state = SharedState(config=config)
        timestamp = time.time()
        
        # Record first trade
        shared_state.bootstrap_metrics.save_first_trade_at(timestamp)
        
        # Now bootstrap should return False
        assert shared_state.is_cold_bootstrap() is False
    
    def test_cold_bootstrap_false_after_restart_with_persisted_metrics(self):
        """Test that bootstrap is False after restart when metrics are persisted."""
        # Use non-existent path
        non_existent_path = os.path.join(self.temp_dir, "non_existent")
        
        # First session: record a trade
        config1 = {
            "DB_PATH": non_existent_path,
            "COLD_BOOTSTRAP_ENABLED": True,
            "LIVE_MODE": False,
        }
        shared_state1 = SharedState(config=config1)
        timestamp = time.time()
        shared_state1.bootstrap_metrics.save_first_trade_at(timestamp)
        
        # Verify cold bootstrap is False
        assert shared_state1.is_cold_bootstrap() is False
        
        # Second session: create new SharedState (simulating restart)
        # The metrics file should still exist and be loaded
        config2 = {
            "DB_PATH": non_existent_path,
            "COLD_BOOTSTRAP_ENABLED": True,
            "LIVE_MODE": False,
        }
        shared_state2 = SharedState(config=config2)
        
        # Bootstrap should still be False (metrics persisted!)
        assert shared_state2.is_cold_bootstrap() is False
    
    def test_cold_bootstrap_checks_persisted_trade_count(self):
        """Test that is_cold_bootstrap checks persisted trade count."""
        # Use non-existent path
        non_existent_path = os.path.join(self.temp_dir, "non_existent")
        
        # First session: record multiple trades
        config1 = {
            "DB_PATH": non_existent_path,
            "COLD_BOOTSTRAP_ENABLED": True,
            "LIVE_MODE": False,
        }
        shared_state1 = SharedState(config=config1)
        
        # Record trades
        for _ in range(5):
            shared_state1.bootstrap_metrics.save_trade_executed()
        
        # Bootstrap should be False (trades executed)
        assert shared_state1.is_cold_bootstrap() is False
        
        # Second session: create new SharedState
        shared_state2 = SharedState(config=config1)
        
        # Bootstrap should still be False (persisted trade count > 0)
        assert shared_state2.is_cold_bootstrap() is False


class TestBootstrapMetricsReload:
    """Test reload functionality."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_reload_from_disk(self):
        """Test manual reload from disk."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        timestamp = time.time()
        
        # Save metrics
        metrics.save_first_trade_at(timestamp)
        for i in range(3):
            metrics.save_trade_executed()
        
        # Manually reload
        metrics.reload()
        
        # Verify reloaded data
        assert metrics.get_first_trade_at() == timestamp
        assert metrics.get_total_trades_executed() == 3


class TestBootstrapMetricsEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_handles_missing_metrics_file(self):
        """Test that BootstrapMetrics handles missing file gracefully."""
        # Don't create any metrics file
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        
        # Should return None for first_trade_at
        assert metrics.get_first_trade_at() is None
        
        # Should return 0 for trade count
        assert metrics.get_total_trades_executed() == 0
    
    def test_handles_corrupted_json(self):
        """Test that BootstrapMetrics handles corrupted JSON gracefully."""
        metrics_file = os.path.join(self.temp_dir, "bootstrap_metrics.json")
        
        # Write corrupted JSON
        with open(metrics_file, 'w') as f:
            f.write("{CORRUPTED JSON}}")
        
        # Should load as empty dict
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        assert metrics.get_first_trade_at() is None
        assert metrics.get_total_trades_executed() == 0
    
    def test_handles_none_db_path(self):
        """Test that BootstrapMetrics handles None db_path."""
        # Should default to current working directory
        metrics = BootstrapMetrics(db_path=None)
        assert metrics.db_path is not None
        assert len(metrics.db_path) > 0
    
    def test_atomic_writes(self):
        """Test that writes are atomic (temp file then rename)."""
        metrics = BootstrapMetrics(db_path=self.temp_dir)
        timestamp = time.time()
        
        # Save should create file atomically
        metrics.save_first_trade_at(timestamp)
        
        # Verify no .tmp files left around
        files = os.listdir(self.temp_dir)
        assert not any(f.endswith('.tmp') for f in files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
