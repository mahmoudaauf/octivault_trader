"""
Test suite for Phase 4 Part 2: Cache TradeIntent Preservation.
Validates that original TradeIntent objects are preserved through the caching layer.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Import core components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stubs import TradeIntent
from core.signal_manager import SignalManager


class InMemoryCache:
    """Simple in-memory cache for testing."""
    def __init__(self, max_size=1000, default_ttl=300.0):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def set(self, key, value):
        self.cache[key] = {"value": value, "timestamp": time.time()}
    
    def get(self, key):
        if key in self.cache:
            return self.cache[key]["value"]
        return None
    
    def list_all(self):
        return [item["value"] for item in self.cache.values()]
    
    def cleanup_expired(self):
        return 0


class MockIntentManager:
    """Mock intent manager for testing."""
    def __init__(self):
        self.intents = []
    
    def add_intent(self, intent):
        self.intents.append(intent)
    
    def drain_intents(self, max_items=None):
        result = self.intents[:max_items] if max_items else self.intents
        self.intents = []
        return result


class TestCacheIntentPreservationBasic:
    """Test basic intent preservation in cache."""
    
    @pytest.fixture
    def setup(self):
        """Setup for cache tests."""
        config = Mock()
        config.signal_cache_max_size = 1000
        config.signal_cache_ttl = 300.0
        config.MIN_SIGNAL_CONF = 0.50
        config.MAX_SIGNAL_AGE_SECONDS = 60
        
        logger = Mock()
        cache = InMemoryCache()
        intent_manager = MockIntentManager()
        
        signal_manager = SignalManager(
            config=config,
            logger=logger,
            signal_cache=cache,
            intent_manager=intent_manager,
            shared_state=None
        )
        
        return {
            "signal_manager": signal_manager,
            "intent_manager": intent_manager,
            "cache": cache,
            "config": config,
            "logger": logger
        }
    
    def test_store_signal_with_intent(self, setup):
        """Test that store_signal preserves source intent."""
        sm = setup["signal_manager"]
        
        # Create a TradeIntent
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.1,
            planned_quote=45000.0,
            confidence=0.75,
            trace_id="test_trace_001",
            agent="SwingTradeHunter",
            reason="breakout signal"
        )
        
        # Create a signal dict
        signal_dict = {
            "symbol": "BTCUSDT",
            "action": "BUY",
            "confidence": 0.75,
            "agent": "SwingTradeHunter",
            "quote": 45000.0,
        }
        
        # Store with intent
        sm.store_signal("SwingTradeHunter", "BTCUSDT", signal_dict, source_intent=intent)
        
        # Retrieve from cache
        all_signals = sm.get_all_signals()
        assert len(all_signals) == 1
        
        cached_signal = all_signals[0]
        assert cached_signal["symbol"] == "BTCUSDT"
        assert cached_signal["_has_source_intent"] is True
        assert "_source_intent" in cached_signal
        
        # Verify intent is preserved
        stored_intent = cached_signal["_source_intent"]
        assert stored_intent is intent
        assert stored_intent.symbol == "BTCUSDT"
        assert stored_intent.side == "BUY"
        assert stored_intent.trace_id == "test_trace_001"
    
    def test_store_signal_without_intent(self, setup):
        """Test that store_signal works without intent (backward compatibility)."""
        sm = setup["signal_manager"]
        
        signal_dict = {
            "symbol": "ETHUSDT",
            "action": "SELL",
            "confidence": 0.60,
            "agent": "MLForecaster",
        }
        
        # Store without intent
        sm.store_signal("MLForecaster", "ETHUSDT", signal_dict, source_intent=None)
        
        # Retrieve from cache
        all_signals = sm.get_all_signals()
        assert len(all_signals) == 1
        
        cached_signal = all_signals[0]
        assert cached_signal["symbol"] == "ETHUSDT"
        assert "_has_source_intent" not in cached_signal
        assert "_source_intent" not in cached_signal
    
    def test_get_source_intent_helper(self, setup):
        """Test get_source_intent helper method."""
        sm = setup["signal_manager"]
        
        intent = TradeIntent(
            symbol="BNBUSDT",
            side="BUY",
            quantity=5.0,
            planned_quote=600.0,
            confidence=0.80,
            trace_id="test_002",
            agent="DipSniper",
            reason="dip detected"
        )
        
        signal_with_intent = {
            "symbol": "BNBUSDT",
            "_source_intent": intent,
            "_has_source_intent": True
        }
        
        signal_without_intent = {
            "symbol": "BNBUSDT",
        }
        
        # Extract from signal with intent
        extracted = sm.get_source_intent(signal_with_intent)
        assert extracted is intent
        assert extracted.symbol == "BNBUSDT"
        
        # Extract from signal without intent
        extracted = sm.get_source_intent(signal_without_intent)
        assert extracted is None


class TestCacheIntentFlushPreservation:
    """Test intent preservation during flush_intents_to_cache."""
    
    @pytest.fixture
    def setup(self):
        """Setup for flush tests."""
        config = Mock()
        config.signal_cache_max_size = 1000
        config.signal_cache_ttl = 300.0
        config.MIN_SIGNAL_CONF = 0.50
        config.MAX_SIGNAL_AGE_SECONDS = 60
        
        logger = Mock()
        cache = InMemoryCache()
        intent_manager = MockIntentManager()
        
        signal_manager = SignalManager(
            config=config,
            logger=logger,
            signal_cache=cache,
            intent_manager=intent_manager,
            shared_state=None
        )
        
        return {
            "signal_manager": signal_manager,
            "intent_manager": intent_manager,
            "cache": cache,
            "config": config,
            "logger": logger
        }
    
    def test_flush_single_intent_preserves_object(self, setup):
        """Test that flushing a single intent preserves the TradeIntent object."""
        sm = setup["signal_manager"]
        im = setup["intent_manager"]
        
        # Create a TradeIntent and add to intent manager
        intent = TradeIntent(
            symbol="ADAUSDT",
            side="BUY",
            quantity=100.0,
            planned_quote=1.0,
            confidence=0.65,
            trace_id="test_flush_001",
            agent="TrendHunter",
            reason="uptrend detected"
        )
        im.add_intent(intent)
        
        # Flush intents to cache
        now_ts = time.time()
        count = sm.flush_intents_to_cache(now_ts)
        
        assert count == 1, f"Expected 1 intent flushed, got {count}"
        
        # Retrieve signal from cache
        signals = sm.get_all_signals()
        assert len(signals) == 1
        
        cached_signal = signals[0]
        assert cached_signal["symbol"] == "ADAUSDT"
        assert cached_signal["action"] == "BUY"
        
        # Verify intent was preserved
        stored_intent = sm.get_source_intent(cached_signal)
        assert stored_intent is not None
        assert stored_intent is intent
        assert stored_intent.trace_id == "test_flush_001"
        assert stored_intent.agent == "TrendHunter"
    
    def test_flush_multiple_intents_preserves_all(self, setup):
        """Test that flushing multiple intents preserves all TradeIntent objects."""
        sm = setup["signal_manager"]
        im = setup["intent_manager"]
        
        # Create multiple intents
        intent1 = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            planned_quote=45000.0,
            confidence=0.85,
            trace_id="multi_001",
            agent="SwingTradeHunter",
            reason="bullish signal"
        )
        
        intent2 = TradeIntent(
            symbol="ETHUSDT",
            side="SELL",
            quantity=10.0,
            planned_quote=3000.0,
            confidence=0.70,
            trace_id="multi_002",
            agent="MLForecaster",
            reason="resistance hit"
        )
        
        intent3 = TradeIntent(
            symbol="BNBUSDT",
            side="BUY",
            quantity=50.0,
            planned_quote=500.0,
            confidence=0.72,
            trace_id="multi_003",
            agent="DipSniper",
            reason="dip confirmed"
        )
        
        im.add_intent(intent1)
        im.add_intent(intent2)
        im.add_intent(intent3)
        
        # Flush all intents
        now_ts = time.time()
        count = sm.flush_intents_to_cache(now_ts)
        
        assert count == 3, f"Expected 3 intents, got {count}"
        
        # Retrieve all signals
        signals = sm.get_all_signals()
        assert len(signals) == 3
        
        # Verify each intent was preserved
        intent_map = {s["symbol"]: sm.get_source_intent(s) for s in signals}
        
        assert intent_map["BTCUSDT"] is intent1
        assert intent_map["ETHUSDT"] is intent2
        assert intent_map["BNBUSDT"] is intent3
        
        # Verify trace_id propagation
        assert intent_map["BTCUSDT"].trace_id == "multi_001"
        assert intent_map["ETHUSDT"].trace_id == "multi_002"
        assert intent_map["BNBUSDT"].trace_id == "multi_003"
    
    def test_flush_with_confidence_filtering(self, setup):
        """Test that confidence filtering still works with intent preservation."""
        sm = setup["signal_manager"]
        im = setup["intent_manager"]
        
        # Create intent with low confidence (should be filtered)
        low_conf_intent = TradeIntent(
            symbol="XRPUSDT",
            side="BUY",
            quantity=100.0,
            planned_quote=2.0,
            confidence=0.40,  # Below default floor of 0.50
            trace_id="low_conf",
            agent="WeakAgent",
            reason="weak signal"
        )
        
        # Create intent with acceptable confidence
        good_intent = TradeIntent(
            symbol="LTCUSDT",
            side="BUY",
            quantity=10.0,
            planned_quote=200.0,
            confidence=0.55,  # Above floor
            trace_id="good_conf",
            agent="StrongAgent",
            reason="strong signal"
        )
        
        im.add_intent(low_conf_intent)
        im.add_intent(good_intent)
        
        # Flush - should only accept good_intent
        now_ts = time.time()
        count = sm.flush_intents_to_cache(now_ts)
        
        # Low confidence intent should be rejected
        assert count == 1, f"Expected 1 intent (low conf filtered), got {count}"
        
        signals = sm.get_all_signals()
        assert len(signals) == 1
        assert signals[0]["symbol"] == "LTCUSDT"
        
        stored_intent = sm.get_source_intent(signals[0])
        assert stored_intent is good_intent


class TestCacheIntentRoundTrip:
    """Test full round-trip: Intent → Cache → Decision."""
    
    @pytest.fixture
    def setup(self):
        """Setup for round-trip tests."""
        config = Mock()
        config.signal_cache_max_size = 1000
        config.signal_cache_ttl = 300.0
        config.MIN_SIGNAL_CONF = 0.50
        config.MAX_SIGNAL_AGE_SECONDS = 60
        
        logger = Mock()
        cache = InMemoryCache()
        intent_manager = MockIntentManager()
        
        signal_manager = SignalManager(
            config=config,
            logger=logger,
            signal_cache=cache,
            intent_manager=intent_manager,
            shared_state=None
        )
        
        return {
            "signal_manager": signal_manager,
            "intent_manager": intent_manager,
            "cache": cache,
            "config": config,
            "logger": logger
        }
    
    def test_trace_id_preserved_through_cache(self, setup):
        """Test that trace_id is preserved through caching."""
        sm = setup["signal_manager"]
        im = setup["intent_manager"]
        
        # Create intent with specific trace_id
        trace_id = "end_to_end_trace_12345"
        intent = TradeIntent(
            symbol="SOLUSDT",
            side="BUY",
            quantity=50.0,
            planned_quote=200.0,
            confidence=0.75,
            trace_id=trace_id,
            agent="IPOChaser",
            reason="new listing"
        )
        
        im.add_intent(intent)
        now_ts = time.time()
        sm.flush_intents_to_cache(now_ts)
        
        # Retrieve from cache
        signals = sm.get_signals_for_symbol("SOLUSDT")
        assert len(signals) == 1
        
        cached_signal = signals[0]
        stored_intent = sm.get_source_intent(cached_signal)
        
        # Verify trace_id is preserved
        assert stored_intent.trace_id == trace_id
        # Verify it's in the dataclass fields
        intent_dict = vars(stored_intent)
        assert "trace_id" in intent_dict
        assert intent_dict["trace_id"] == trace_id
    
    def test_all_intent_fields_accessible_from_cache(self, setup):
        """Test that all TradeIntent fields are accessible from cached signal."""
        sm = setup["signal_manager"]
        im = setup["intent_manager"]
        
        # Create intent with all fields
        intent = TradeIntent(
            symbol="DOGEUSDT",
            side="SELL",
            quantity=1000.0,
            planned_quote=0.30,
            confidence=0.68,
            trace_id="complete_field_test",
            agent="LiquidationAgent",
            reason="portfolio rebalancing"
        )
        
        im.add_intent(intent)
        now_ts = time.time()
        sm.flush_intents_to_cache(now_ts)
        
        # Get from cache
        signals = sm.get_all_signals()
        stored_intent = sm.get_source_intent(signals[0])
        
        # Verify all fields
        assert stored_intent.symbol == "DOGEUSDT"
        assert stored_intent.side == "SELL"
        assert stored_intent.quantity == 1000.0
        assert stored_intent.planned_quote == 0.30
        assert stored_intent.confidence == 0.68
        assert stored_intent.trace_id == "complete_field_test"
        assert stored_intent.agent == "LiquidationAgent"
        assert stored_intent.reason == "portfolio rebalancing"


class TestCacheIntentMigration:
    """Test migration compatibility: dict-based and intent-based signals coexist."""
    
    @pytest.fixture
    def setup(self):
        """Setup for migration tests."""
        config = Mock()
        config.signal_cache_max_size = 1000
        config.signal_cache_ttl = 300.0
        config.MIN_SIGNAL_CONF = 0.50
        config.MAX_SIGNAL_AGE_SECONDS = 60
        
        logger = Mock()
        cache = InMemoryCache()
        intent_manager = MockIntentManager()
        
        signal_manager = SignalManager(
            config=config,
            logger=logger,
            signal_cache=cache,
            intent_manager=intent_manager,
            shared_state=None
        )
        
        return {
            "signal_manager": signal_manager,
            "intent_manager": intent_manager,
            "cache": cache,
        }
    
    def test_dict_signals_still_work(self, setup):
        """Test that storing plain dict signals (without intent) still works."""
        sm = setup["signal_manager"]
        
        # Store a plain dict signal (old format)
        dict_signal = {
            "symbol": "MATICUSDT",
            "action": "BUY",
            "confidence": 0.65,
            "agent": "CustomAgent"
        }
        
        sm.store_signal("CustomAgent", "MATICUSDT", dict_signal, source_intent=None)
        
        # Retrieve and verify
        signals = sm.get_all_signals()
        assert len(signals) == 1
        assert signals[0]["symbol"] == "MATICUSDT"
        assert sm.get_source_intent(signals[0]) is None
    
    def test_mixed_intent_and_dict_signals(self, setup):
        """Test that intent-based and dict-based signals coexist in cache."""
        sm = setup["signal_manager"]
        im = setup["intent_manager"]
        
        # Add intent-based signal
        intent = TradeIntent(
            symbol="FTMUSDT",
            side="BUY",
            quantity=500.0,
            planned_quote=1.0,
            confidence=0.75,
            trace_id="with_intent",
            agent="Agent1",
            reason="signal 1"
        )
        im.add_intent(intent)
        
        # Flush intent-based
        now_ts = time.time()
        sm.flush_intents_to_cache(now_ts)
        
        # Add dict-based signal directly
        dict_signal = {
            "symbol": "AVAXUSDT",
            "action": "SELL",
            "confidence": 0.60,
            "agent": "Agent2"
        }
        sm.store_signal("Agent2", "AVAXUSDT", dict_signal, source_intent=None)
        
        # Verify both are in cache
        signals = sm.get_all_signals()
        assert len(signals) == 2
        
        signals_by_symbol = {s["symbol"]: s for s in signals}
        
        # Intent-based signal
        assert sm.get_source_intent(signals_by_symbol["FTMUSDT"]) is intent
        
        # Dict-based signal
        assert sm.get_source_intent(signals_by_symbol["AVAXUSDT"]) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
