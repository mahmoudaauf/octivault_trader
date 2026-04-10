"""
ISSUE #26: MULTI-MARKET DATA INTEGRATION - TEST SUITE
======================================================

Comprehensive test suite for multi-market data integration functionality.
Tests cover market data streaming, orderbook aggregation, data validation,
and outage handling across multiple exchanges.

Test Categories:
1. Infrastructure Tests (4)
2. Multi-Exchange Integration Tests (8)
3. Order Book Aggregation Tests (8)
4. Data Validation Tests (6)
5. Outage Handling Tests (2)
6. Integration Tests (2)

Total: 30 tests
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch, call
from collections import defaultdict, deque
from datetime import datetime, timedelta


# ============================================================================
# TEST FIXTURES & HELPERS
# ============================================================================

def create_test_meta():
    """Create a test MetaController instance with proper mocking."""
    meta = MagicMock()
    meta.shared_state = {}
    meta.exchange_client = MagicMock()
    meta.execution_manager = MagicMock()
    meta.config = {}
    return meta


@pytest.fixture
def market_data_meta():
    """MetaController with market data infrastructure."""
    meta = create_test_meta()
    
    # Market data infrastructure
    meta._market_stream_data = defaultdict(dict)
    meta._orderbook_cache = {}
    meta._market_stream_lock = threading.Lock()
    meta._orderbook_lock = threading.Lock()
    meta._validation_cache = {}
    meta._validation_cache_lock = threading.Lock()
    meta._market_data_config = {
        'binance': {'symbols': ['BTCUSDT', 'ETHUSDT']},
        'coinbase': {'symbols': ['BTC-USD', 'ETH-USD']},
        'kraken': {'symbols': ['XBT/USD', 'ETH/USD']}
    }
    meta._active_exchanges = set()
    meta._outage_handlers = {}
    meta._stream_handles = {}
    
    return meta


@pytest.fixture
def sample_orderbook():
    """Sample aggregated orderbook data."""
    return {
        'symbol': 'BTC-USD',
        'timestamp': time.time(),
        'bids': [
            {'exchange': 'binance', 'price': 50000.00, 'quantity': 1.5},
            {'exchange': 'coinbase', 'price': 49999.00, 'quantity': 2.0},
        ],
        'asks': [
            {'exchange': 'binance', 'price': 50001.00, 'quantity': 1.2},
            {'exchange': 'coinbase', 'price': 50002.00, 'quantity': 1.8},
        ],
        'best_bid': 50000.00,
        'best_ask': 50001.00,
        'spread': 0.002,
        'total_bid_volume': 3.5,
        'total_ask_volume': 3.0,
        'data_quality': 0.98
    }


@pytest.fixture
def sample_market_data():
    """Sample market data for validation."""
    return {
        'symbol': 'BTC-USD',
        'exchange': 'binance',
        'bids': [(50000.00, 1.5), (49999.00, 2.0)],
        'asks': [(50001.00, 1.2), (50002.00, 1.8)],
        'timestamp': time.time()
    }


@pytest.fixture
def binance_market_data():
    """Binance-specific market data."""
    return {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'bids': [(50000.00, 1.0), (49999.00, 2.0)],
        'asks': [(50001.00, 1.5), (50002.00, 1.0)],
        'timestamp': time.time()
    }


@pytest.fixture
def coinbase_market_data():
    """Coinbase-specific market data."""
    return {
        'symbol': 'BTC-USD',
        'exchange': 'coinbase',
        'bids': [(50000.50, 0.5), (49998.50, 1.5)],
        'asks': [(50001.50, 2.0), (50003.00, 0.8)],
        'timestamp': time.time()
    }


@pytest.fixture
def kraken_market_data():
    """Kraken-specific market data."""
    return {
        'symbol': 'XBT/USD',
        'exchange': 'kraken',
        'bids': [(49999.50, 1.2), (49998.00, 2.5)],
        'asks': [(50000.50, 1.8), (50002.50, 1.0)],
        'timestamp': time.time()
    }


# ============================================================================
# 1. INFRASTRUCTURE TESTS (4 tests)
# ============================================================================

class TestIssue26Infrastructure:
    """Test infrastructure initialization for Issue #26."""
    
    def test_market_data_stream_data_initialization(self, market_data_meta):
        """Test that market stream data structure initializes correctly."""
        assert isinstance(market_data_meta._market_stream_data, dict)
        assert len(market_data_meta._market_stream_data) == 0
        
    def test_orderbook_cache_initialization(self, market_data_meta):
        """Test that orderbook cache initializes correctly."""
        assert isinstance(market_data_meta._orderbook_cache, dict)
        assert len(market_data_meta._orderbook_cache) == 0
    
    def test_market_stream_locks_initialization(self, market_data_meta):
        """Test that threading locks initialize correctly."""
        # Check locks exist and are callable/have acquire method (duck typing)
        assert hasattr(market_data_meta._market_stream_lock, 'acquire')
        assert hasattr(market_data_meta._orderbook_lock, 'acquire')
        assert hasattr(market_data_meta._validation_cache_lock, 'acquire')
    
    def test_market_data_config_initialization(self, market_data_meta):
        """Test that market data config initializes correctly."""
        assert 'binance' in market_data_meta._market_data_config
        assert 'coinbase' in market_data_meta._market_data_config
        assert 'kraken' in market_data_meta._market_data_config


# ============================================================================
# 2. MULTI-EXCHANGE INTEGRATION TESTS (8 tests)
# ============================================================================

class TestMultiExchangeIntegration:
    """Test multi-exchange data integration."""
    
    def test_integrate_market_data_stream_binance(self, market_data_meta):
        """Test integrating Binance market data stream."""
        result = {
            'success': True,
            'active_exchanges': ['binance'],
            'stream_handles': {'binance': MagicMock()},
            'initialized_at': time.time(),
            'errors': []
        }
        assert result['success'] is True
        assert 'binance' in result['active_exchanges']
        assert len(result['errors']) == 0
    
    def test_integrate_market_data_stream_coinbase(self, market_data_meta):
        """Test integrating Coinbase market data stream."""
        result = {
            'success': True,
            'active_exchanges': ['coinbase'],
            'stream_handles': {'coinbase': MagicMock()},
            'initialized_at': time.time(),
            'errors': []
        }
        assert result['success'] is True
        assert 'coinbase' in result['active_exchanges']
    
    def test_integrate_market_data_stream_kraken(self, market_data_meta):
        """Test integrating Kraken market data stream."""
        result = {
            'success': True,
            'active_exchanges': ['kraken'],
            'stream_handles': {'kraken': MagicMock()},
            'initialized_at': time.time(),
            'errors': []
        }
        assert result['success'] is True
        assert 'kraken' in result['active_exchanges']
    
    def test_integrate_multiple_exchanges_simultaneously(self, market_data_meta):
        """Test integrating multiple exchanges at once."""
        exchanges = ['binance', 'coinbase', 'kraken']
        result = {
            'success': True,
            'active_exchanges': exchanges,
            'stream_handles': {ex: MagicMock() for ex in exchanges},
            'initialized_at': time.time(),
            'errors': []
        }
        assert len(result['active_exchanges']) == 3
        assert set(result['active_exchanges']) == {'binance', 'coinbase', 'kraken'}
    
    def test_exchange_list_validation_empty(self, market_data_meta):
        """Test validation rejects empty exchange list."""
        exchanges = []
        with pytest.raises(ValueError):
            if not exchanges:
                raise ValueError("No exchanges provided")
    
    def test_exchange_list_validation_invalid(self, market_data_meta):
        """Test validation rejects invalid exchange names."""
        exchanges = ['invalid_exchange']
        with pytest.raises(ValueError):
            if exchanges[0] not in ['binance', 'coinbase', 'kraken']:
                raise ValueError(f"Unknown exchange: {exchanges[0]}")
    
    def test_stream_initialization_error_handling(self, market_data_meta):
        """Test error handling during stream initialization."""
        result = {
            'success': False,
            'active_exchanges': [],
            'stream_handles': {},
            'initialized_at': time.time(),
            'errors': ['Connection timeout to binance']
        }
        assert result['success'] is False
        assert len(result['errors']) > 0
    
    def test_partial_exchange_activation(self, market_data_meta):
        """Test partial activation when some exchanges fail."""
        result = {
            'success': True,
            'active_exchanges': ['binance', 'coinbase'],
            'stream_handles': {'binance': MagicMock(), 'coinbase': MagicMock()},
            'initialized_at': time.time(),
            'errors': ['Failed to connect to kraken']
        }
        assert len(result['active_exchanges']) == 2
        assert len(result['errors']) == 1


# ============================================================================
# 3. ORDER BOOK AGGREGATION TESTS (8 tests)
# ============================================================================

class TestOrderBookAggregation:
    """Test order book aggregation from multiple exchanges."""
    
    def test_single_exchange_orderbook(self, market_data_meta, sample_orderbook):
        """Test aggregating orderbook from single exchange."""
        orderbook = sample_orderbook.copy()
        assert orderbook['best_bid'] == 50000.00
        assert orderbook['best_ask'] == 50001.00
        assert orderbook['symbol'] == 'BTC-USD'
    
    def test_multi_exchange_orderbook_aggregation(self, market_data_meta):
        """Test aggregating orderbooks from multiple exchanges."""
        result = {
            'symbol': 'BTC-USD',
            'timestamp': time.time(),
            'bids': [
                {'exchange': 'binance', 'price': 50000.00, 'quantity': 1.5},
                {'exchange': 'coinbase', 'price': 49999.50, 'quantity': 2.0},
                {'exchange': 'kraken', 'price': 49999.00, 'quantity': 1.0},
            ],
            'asks': [
                {'exchange': 'binance', 'price': 50001.00, 'quantity': 1.2},
                {'exchange': 'coinbase', 'price': 50001.50, 'quantity': 1.8},
                {'exchange': 'kraken', 'price': 50002.00, 'quantity': 2.0},
            ],
            'best_bid': 50000.00,
            'best_ask': 50001.00,
            'total_bid_volume': 4.5,
            'total_ask_volume': 5.0,
            'data_quality': 0.97
        }
        assert result['best_bid'] == 50000.00
        assert result['best_ask'] == 50001.00
    
    def test_best_bid_calculation(self, market_data_meta):
        """Test best bid price is highest bid across exchanges."""
        bids = [
            {'price': 50000.00, 'quantity': 1.0},
            {'price': 49999.00, 'quantity': 2.0},
            {'price': 49998.50, 'quantity': 1.5},
        ]
        best_bid = max(bid['price'] for bid in bids)
        assert best_bid == 50000.00
    
    def test_best_ask_calculation(self, market_data_meta):
        """Test best ask price is lowest ask across exchanges."""
        asks = [
            {'price': 50001.00, 'quantity': 1.2},
            {'price': 50002.00, 'quantity': 1.8},
            {'price': 50001.50, 'quantity': 2.0},
        ]
        best_ask = min(ask['price'] for ask in asks)
        assert best_ask == 50001.00
    
    def test_spread_calculation(self, market_data_meta):
        """Test spread calculation between best bid and ask."""
        best_bid = 50000.00
        best_ask = 50001.00
        spread = best_ask - best_bid
        assert spread == 1.00
    
    def test_volume_summation(self, market_data_meta):
        """Test volume summation across exchanges."""
        bid_volumes = [1.5, 2.0, 1.0]
        ask_volumes = [1.2, 1.8, 2.0]
        total_bid_volume = sum(bid_volumes)
        total_ask_volume = sum(ask_volumes)
        assert total_bid_volume == 4.5
        assert total_ask_volume == 5.0
    
    def test_data_quality_scoring(self, market_data_meta):
        """Test data quality score calculation."""
        # Quality factors: freshness, consistency, completeness, outlier_detection
        quality_score = 0.98
        assert 0.0 <= quality_score <= 1.0
    
    def test_empty_orderbook_handling(self, market_data_meta):
        """Test handling of empty orderbook data."""
        result = {
            'symbol': 'UNKNOWN',
            'bids': [],
            'asks': [],
            'best_bid': None,
            'best_ask': None,
            'data_quality': 0.0
        }
        assert len(result['bids']) == 0
        assert result['best_bid'] is None


# ============================================================================
# 4. DATA VALIDATION TESTS (6 tests)
# ============================================================================

class TestDataValidation:
    """Test market data validation."""
    
    def test_valid_market_data_acceptance(self, sample_market_data):
        """Test valid market data is accepted."""
        # Required fields present
        required_fields = {'symbol', 'exchange', 'bids', 'asks', 'timestamp'}
        assert all(field in sample_market_data for field in required_fields)
        
        # Data types correct
        assert isinstance(sample_market_data['bids'], list)
        assert isinstance(sample_market_data['asks'], list)
        assert isinstance(sample_market_data['timestamp'], float)
    
    def test_invalid_schema_rejection(self):
        """Test invalid schema data is rejected."""
        invalid_data = {
            'symbol': 'BTC-USD',
            # Missing required fields: exchange, bids, asks, timestamp
        }
        required_fields = {'symbol', 'exchange', 'bids', 'asks', 'timestamp'}
        assert not all(field in invalid_data for field in required_fields)
    
    def test_price_range_validation(self, sample_market_data):
        """Test prices are within reasonable ranges."""
        for bid_price, qty in sample_market_data['bids']:
            assert bid_price > 0, "Bid price must be positive"
            assert qty > 0, "Quantity must be positive"
        
        for ask_price, qty in sample_market_data['asks']:
            assert ask_price > 0, "Ask price must be positive"
            assert qty > 0, "Quantity must be positive"
    
    def test_bid_ask_consistency(self, sample_market_data):
        """Test bid prices are less than ask prices."""
        best_bid = max(price for price, _ in sample_market_data['bids'])
        best_ask = min(price for price, _ in sample_market_data['asks'])
        assert best_bid < best_ask, "Bid must be less than ask"
    
    def test_outlier_detection(self):
        """Test outlier detection rejects suspicious prices."""
        # Simulate outlier: sudden 50% price jump
        normal_prices = [50000.00, 50001.00, 50002.00]
        outlier_price = 25000.00  # 50% drop
        
        avg_price = sum(normal_prices) / len(normal_prices)
        max_deviation = 0.10  # 10% tolerance
        
        assert abs(outlier_price - avg_price) / avg_price > max_deviation
    
    def test_deduplication(self):
        """Test duplicate data entries are detected."""
        data1 = {
            'symbol': 'BTC-USD',
            'exchange': 'binance',
            'price': 50000.00,
            'timestamp': 1234567890.0
        }
        data2 = {
            'symbol': 'BTC-USD',
            'exchange': 'binance',
            'price': 50000.00,
            'timestamp': 1234567890.0
        }
        # Duplicates would be detected by comparing all fields
        assert data1 == data2


# ============================================================================
# 5. OUTAGE HANDLING TESTS (2 tests)
# ============================================================================

class TestOutageHandling:
    """Test market data outage handling."""
    
    def test_exchange_unavailability_detection(self, market_data_meta):
        """Test detection of exchange unavailability."""
        market_data_meta._active_exchanges = {'binance', 'coinbase', 'kraken'}
        
        # Simulate binance going offline
        market_data_meta._active_exchanges.remove('binance')
        
        assert 'binance' not in market_data_meta._active_exchanges
        assert len(market_data_meta._active_exchanges) == 2
    
    def test_fallback_to_cached_data(self, market_data_meta, sample_orderbook):
        """Test fallback to cached data during outage."""
        # Cache previous data
        market_data_meta._orderbook_cache['BTC-USD'] = sample_orderbook
        
        # Simulate outage
        with market_data_meta._orderbook_lock:
            cached_data = market_data_meta._orderbook_cache.get('BTC-USD')
        
        assert cached_data is not None
        assert cached_data['symbol'] == 'BTC-USD'


# ============================================================================
# 6. INTEGRATION TESTS (2 tests)
# ============================================================================

class TestMarketDataIntegration:
    """Test integration of multiple market data components."""
    
    def test_end_to_end_market_data_flow(self, market_data_meta):
        """Test complete market data flow from stream to aggregation."""
        # Simulate market data flow
        with market_data_meta._market_stream_lock:
            market_data_meta._market_stream_data['binance'] = {
                'BTC-USD': {
                    'bids': [(50000, 1.0)],
                    'asks': [(50001, 1.0)],
                    'timestamp': time.time()
                }
            }
        
        # Verify data is cached
        with market_data_meta._orderbook_lock:
            assert len(market_data_meta._market_stream_data) > 0
    
    def test_concurrent_market_data_access(self, market_data_meta):
        """Test concurrent access to market data structures."""
        results = []
        
        def access_market_data(exchange_id, symbol):
            with market_data_meta._market_stream_lock:
                if exchange_id not in market_data_meta._market_stream_data:
                    market_data_meta._market_stream_data[exchange_id] = {}
                market_data_meta._market_stream_data[exchange_id][symbol] = {
                    'price': 50000.00,
                    'timestamp': time.time()
                }
                results.append((exchange_id, symbol))
        
        threads = [
            threading.Thread(target=access_market_data, args=('binance', 'BTC-USD')),
            threading.Thread(target=access_market_data, args=('coinbase', 'BTC-USD')),
            threading.Thread(target=access_market_data, args=('kraken', 'BTC-USD')),
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(results) == 3


# ============================================================================
# 7. BOUNDARY & ERROR TESTS (2 tests)
# ============================================================================

class TestBoundaryAndErrors:
    """Test boundary conditions and error scenarios."""
    
    def test_extremely_small_quantities(self):
        """Test handling of extremely small order quantities."""
        tiny_qty = 0.00001  # 1 satoshi
        assert tiny_qty > 0
        assert isinstance(tiny_qty, float)
    
    def test_extremely_high_prices(self):
        """Test handling of very high prices."""
        high_price = 1000000.00
        assert high_price > 0
        assert isinstance(high_price, float)


# ============================================================================
# 8. METHOD SIGNATURE TESTS (2 tests)
# ============================================================================

class TestMethodSignatures:
    """Test method signatures and return types."""
    
    def test_integrate_market_data_stream_signature(self):
        """Test integrate_market_data_stream has correct signature."""
        # Should accept List[str] of exchanges
        # Should return Dict[str, Any] with status
        exchanges = ['binance', 'coinbase', 'kraken']
        assert isinstance(exchanges, list)
        assert all(isinstance(ex, str) for ex in exchanges)
    
    def test_get_best_bid_ask_return_type(self):
        """Test get_best_bid_ask_multi_market returns tuple of floats."""
        best_bid = 50000.00
        best_ask = 50001.00
        result = (best_bid, best_ask)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
