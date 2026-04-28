"""
Configuration Constants Module

Centralizes all hard-coded numeric values and configuration parameters.
Enables dynamic tuning without code changes and improves maintainability.

PHILOSOPHY:
  • No magic numbers scattered in code
  • Single source of truth for configuration
  • Well-documented reasoning for each constant
  • Validated ranges and constraints
  • Easy to tune for different environments
"""

from typing import Final, Dict, Any
from decimal import Decimal


class TimeoutConstants:
    """All timeout and delay values in seconds."""
    
    # Network/Exchange Timeouts
    EXCHANGE_API_TIMEOUT: Final[int] = 30
    """Timeout for exchange API calls (seconds). Default: 30s"""
    
    BOOTSTRAP_TIMEOUT: Final[int] = 10
    """Maximum time to complete bootstrap sequence (seconds). Default: 10s"""
    
    HEALTH_CHECK_TIMEOUT: Final[int] = 5
    """Timeout for health check probes (seconds). Default: 5s"""
    
    LIFECYCLE_STATE_TIMEOUT: Final[int] = 30
    """Timeout for lifecycle state transitions (seconds). Default: 30s"""
    
    RETRY_BASE_DELAY: Final[float] = 0.1
    """Base delay for exponential backoff (seconds). Default: 0.1s"""
    
    RETRY_MAX_DELAY: Final[int] = 5
    """Maximum delay between retry attempts (seconds). Default: 5s"""
    
    # Signal/Cache Timeouts
    SIGNAL_CACHE_TTL: Final[int] = 60
    """Time-to-live for cached signals (seconds). Default: 60s"""
    
    ARBITRATION_CACHE_TTL: Final[int] = 30
    """Time-to-live for arbitration results (seconds). Default: 30s"""
    
    # State Reconciliation Timeouts
    STATE_SYNC_INTERVAL: Final[int] = 5
    """Interval for state synchronization checks (seconds). Default: 5s"""
    
    STATE_SYNC_TIMEOUT: Final[int] = 10
    """Timeout for state sync operations (seconds). Default: 10s"""


class RetryConstants:
    """All retry behavior and limits."""
    
    # Retry Counts
    EXCHANGE_API_RETRIES: Final[int] = 3
    """Max retries for exchange API calls. Default: 3 attempts"""
    
    BOOTSTRAP_RETRIES: Final[int] = 5
    """Max retries for bootstrap operations. Default: 5 attempts"""
    
    STATE_SYNC_RETRIES: Final[int] = 3
    """Max retries for state synchronization. Default: 3 attempts"""
    
    HEALTH_CHECK_RETRIES: Final[int] = 2
    """Max retries for health checks. Default: 2 attempts"""
    
    # Exponential Backoff
    RETRY_BACKOFF_BASE: Final[float] = 2.0
    """Base for exponential backoff calculation. Default: 2.0 (2^attempt)"""
    
    RETRY_JITTER_FACTOR: Final[float] = 0.1
    """Jitter factor to randomize delays (0-10%). Default: 0.1"""
    
    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
    """Consecutive failures before opening circuit. Default: 5"""
    
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 30
    """Seconds before attempting half-open state. Default: 30s"""


class CapitalConstants:
    """Capital and balance thresholds."""
    
    # Minimum Capital Thresholds
    MIN_TRADING_CAPITAL: Final[float] = 10.0
    """Minimum capital required to trade (USDT). Default: $10.00"""
    
    MIN_ORDER_NOTIONAL: Final[float] = 5.0
    """Minimum order size (USDT). Default: $5.00"""
    
    MIN_ENTRY_AMOUNT: Final[float] = 1.0
    """Minimum entry allocation (USDT). Default: $1.00"""
    
    # Capital Protection
    CAPITAL_FLOOR_PERCENT: Final[float] = 0.1
    """Minimum capital floor as % of total. Default: 10%"""
    
    CAPITAL_ALLOCATION_MAX_PERCENT: Final[float] = 0.5
    """Maximum allocation per symbol as % of portfolio. Default: 50%"""
    
    # Dust Thresholds
    DUST_THRESHOLD_USDT: Final[float] = 10.0
    """Amount considered 'dust' (USDT). Default: $10.00"""
    
    DUST_THRESHOLD_PERCENT: Final[float] = 0.01
    """Dust as % of total capital. Default: 1%"""
    
    # Bootstrap Capital
    BOOTSTRAP_CAPITAL_ALLOCATION: Final[float] = 0.2
    """Capital allocation during bootstrap (% of total). Default: 20%"""


class ConfidenceConstants:
    """Signal confidence and validation thresholds."""
    
    # Confidence Thresholds
    MIN_SIGNAL_CONFIDENCE: Final[float] = 0.5
    """Minimum confidence to execute signal. Default: 0.50 (50%)"""
    
    HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.7
    """Threshold for high-confidence signals. Default: 0.70 (70%)"""
    
    CRITICAL_CONFIDENCE_THRESHOLD: Final[float] = 0.85
    """Threshold for critical signals. Default: 0.85 (85%)"""
    
    # Confidence Band Adaptive Thresholds
    CONFIDENCE_BAND_ADAPTIVE_LOW: Final[float] = 0.50
    """Lower adaptive band for confidence. Default: 0.50"""
    
    CONFIDENCE_BAND_ADAPTIVE_HIGH: Final[float] = 0.75
    """Upper adaptive band for confidence. Default: 0.75"""
    
    # Volatility Adjustment
    VOLATILITY_MULTIPLIER_MIN: Final[float] = 0.8
    """Minimum volatility adjustment multiplier. Default: 0.8x"""
    
    VOLATILITY_MULTIPLIER_MAX: Final[float] = 1.5
    """Maximum volatility adjustment multiplier. Default: 1.5x"""


class LeverageConstants:
    """Leverage and position sizing."""
    
    # Leverage Limits
    MAX_LEVERAGE: Final[float] = 3.0
    """Maximum allowed leverage. Default: 3x"""
    
    DEFAULT_LEVERAGE: Final[float] = 1.0
    """Default leverage (1 = no leverage). Default: 1.0x"""
    
    # Position Sizing
    MAX_POSITION_SIZE_PERCENT: Final[float] = 0.25
    """Maximum single position size (% of portfolio). Default: 25%"""
    
    MIN_POSITION_SIZE_PERCENT: Final[float] = 0.01
    """Minimum single position size (% of portfolio). Default: 1%"""


class RiskConstants:
    """Risk management thresholds."""
    
    # Risk Limits
    MAX_LOSS_PERCENT: Final[float] = 0.02
    """Maximum acceptable loss (% of capital). Default: 2%"""
    
    MAX_DRAWDOWN_PERCENT: Final[float] = 0.05
    """Maximum allowed drawdown (% of capital). Default: 5%"""
    
    # Stop Loss / Take Profit
    DEFAULT_STOP_LOSS_PERCENT: Final[float] = 0.05
    """Default stop loss distance (% from entry). Default: 5%"""
    
    DEFAULT_TAKE_PROFIT_PERCENT: Final[float] = 0.10
    """Default take profit distance (% from entry). Default: 10%"""
    
    # Risk Concentration
    CONCENTRATION_WARNING_THRESHOLD: Final[float] = 0.3
    """Concentration threshold for warnings. Default: 30%"""
    
    CONCENTRATION_ERROR_THRESHOLD: Final[float] = 0.5
    """Concentration threshold for blocking trades. Default: 50%"""


class BootstrapConstants:
    """Bootstrap mode and initialization."""
    
    # Bootstrap Duration
    BOOTSTRAP_DURATION_MINUTES: Final[int] = 30
    """Length of bootstrap mode (minutes). Default: 30 min"""
    
    BOOTSTRAP_INITIAL_SYMBOLS: Final[int] = 5
    """Number of symbols to initialize with. Default: 5"""
    
    # Dust Bypass
    DUST_BYPASS_BUDGET_PER_CYCLE: Final[int] = 1
    """Number of dust bypasses allowed per cycle. Default: 1"""
    
    BOOTSTRAP_DUST_HEALING_CYCLES: Final[int] = 3
    """Cycles needed to transition dust to healed. Default: 3"""


class StateTransitionConstants:
    """Lifecycle state machine timing."""
    
    # State Transition Cooldowns
    STATE_TRANSITION_COOLDOWN: Final[int] = 5
    """Cooldown between state transitions (seconds). Default: 5s"""
    
    AWAITING_SIGNAL_TIMEOUT: Final[int] = 60
    """Max time in AWAITING_SIGNAL state (seconds). Default: 60s"""
    
    DUST_HEALING_TIMEOUT: Final[int] = 300
    """Max time in DUST_HEALING state (seconds). Default: 5 min"""
    
    # State Validation
    MAX_STATE_AGE: Final[int] = 120
    """Maximum age of cached state before refresh (seconds). Default: 2 min"""


class ConcurrencyConstants:
    """Concurrency and task limits."""
    
    # Task Limits
    MAX_CONCURRENT_ORDERS: Final[int] = 5
    """Maximum concurrent order executions. Default: 5"""
    
    MAX_CONCURRENT_SYMBOL_CHECKS: Final[int] = 10
    """Maximum concurrent symbol evaluations. Default: 10"""
    
    # Queue Sizes
    SIGNAL_QUEUE_MAX_SIZE: Final[int] = 1000
    """Maximum signals in queue. Default: 1000"""
    
    EVENT_QUEUE_MAX_SIZE: Final[int] = 5000
    """Maximum events in queue. Default: 5000"""
    
    # Rate Limiting
    API_RATE_LIMIT_DELAY: Final[float] = 0.1
    """Delay between API calls (seconds). Default: 0.1s"""


class PerformanceConstants:
    """Performance tuning parameters."""
    
    # Cache Sizes
    SYMBOL_CACHE_SIZE: Final[int] = 1000
    """Maximum symbols in cache. Default: 1000"""
    
    STATE_CACHE_SIZE: Final[int] = 500
    """Maximum state entries in cache. Default: 500"""
    
    # Batch Sizes
    SIGNAL_BATCH_SIZE: Final[int] = 100
    """Process signals in batches of N. Default: 100"""
    
    STATE_SYNC_BATCH_SIZE: Final[int] = 50
    """Reconcile state in batches of N. Default: 50"""
    
    # Cleanup Intervals
    CACHE_CLEANUP_INTERVAL: Final[int] = 300
    """Interval to clean up stale cache (seconds). Default: 5 min"""
    
    STATE_CLEANUP_INTERVAL: Final[int] = 600
    """Interval to clean up old state (seconds). Default: 10 min"""


class EnvironmentConstants:
    """Environment-specific configurations."""
    
    DEBUG_MODE_ENABLED: Final[bool] = False
    """Enable debug logging. Default: False"""
    
    DEVELOPMENT_MODE: Final[bool] = False
    """Enable development-only features. Default: False"""
    
    STRICT_TYPE_CHECKING: Final[bool] = True
    """Enable strict type validation. Default: True"""


def get_all_constants() -> Dict[str, Any]:
    """
    Get all configuration constants as a dictionary.
    
    Returns:
        Dict mapping constant names to their values
    """
    constants = {}
    
    # Collect all class attributes that are constants (Final)
    for cls in [
        TimeoutConstants,
        RetryConstants,
        CapitalConstants,
        ConfidenceConstants,
        LeverageConstants,
        RiskConstants,
        BootstrapConstants,
        StateTransitionConstants,
        ConcurrencyConstants,
        PerformanceConstants,
        EnvironmentConstants,
    ]:
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if not callable(attr_value):
                    constants[f"{cls.__name__}.{attr_name}"] = attr_value
    
    return constants


def validate_constants() -> bool:
    """
    Validate all constants for reasonable values.
    
    Returns:
        True if all constants are valid, False otherwise
    """
    issues = []
    
    # Timeout validation
    if TimeoutConstants.EXCHANGE_API_TIMEOUT <= 0:
        issues.append("EXCHANGE_API_TIMEOUT must be > 0")
    
    if TimeoutConstants.RETRY_MAX_DELAY <= TimeoutConstants.RETRY_BASE_DELAY:
        issues.append("RETRY_MAX_DELAY must be > RETRY_BASE_DELAY")
    
    # Capital validation
    if CapitalConstants.MIN_TRADING_CAPITAL <= 0:
        issues.append("MIN_TRADING_CAPITAL must be > 0")
    
    if CapitalConstants.MIN_ORDER_NOTIONAL > CapitalConstants.MIN_TRADING_CAPITAL:
        issues.append("MIN_ORDER_NOTIONAL should be <= MIN_TRADING_CAPITAL")
    
    # Confidence validation
    if not (0 <= ConfidenceConstants.MIN_SIGNAL_CONFIDENCE <= 1.0):
        issues.append("MIN_SIGNAL_CONFIDENCE must be between 0 and 1")
    
    if ConfidenceConstants.HIGH_CONFIDENCE_THRESHOLD <= ConfidenceConstants.MIN_SIGNAL_CONFIDENCE:
        issues.append("HIGH_CONFIDENCE_THRESHOLD must be > MIN_SIGNAL_CONFIDENCE")
    
    if ConfidenceConstants.CRITICAL_CONFIDENCE_THRESHOLD <= ConfidenceConstants.HIGH_CONFIDENCE_THRESHOLD:
        issues.append("CRITICAL_CONFIDENCE_THRESHOLD must be > HIGH_CONFIDENCE_THRESHOLD")
    
    # Retry validation
    if RetryConstants.EXCHANGE_API_RETRIES <= 0:
        issues.append("EXCHANGE_API_RETRIES must be > 0")
    
    if RetryConstants.RETRY_BACKOFF_BASE <= 1.0:
        issues.append("RETRY_BACKOFF_BASE must be > 1.0")
    
    if issues:
        print("⚠️  Configuration validation issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    return True


if __name__ == "__main__":
    """Print all constants and validate configuration."""
    print("=" * 80)
    print("CONFIGURATION CONSTANTS - ALL VALUES")
    print("=" * 80)
    
    constants = get_all_constants()
    for name, value in sorted(constants.items()):
        print(f"  {name}: {value}")
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    if validate_constants():
        print("✅ All configuration constants are valid!")
    else:
        print("❌ Configuration has issues - see above")
