"""
Tuned hyperparameters for ML models.

This module provides optimized hyperparameters for different trading pairs
based on backtesting and historical performance analysis.
"""

import logging

logger = logging.getLogger(__name__)


def get_tuned_params(symbol=None):
    """
    Return optimized hyperparameters for symbol.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT'). If None, returns default params.
    
    Returns:
        Dict of tuned parameters with keys like:
        - learning_rate: Model learning rate
        - batch_size: Training batch size
        - epochs: Number of training epochs
        - dropout: Dropout rate for regularization
        - lstm_units: Number of LSTM units
        - lookback_window: Historical data window size
        - validation_split: Train/validation split ratio
    """
    
    # ✅ DEFAULT TUNED PARAMETERS (Baseline for all symbols)
    default_params = {
        'learning_rate': 0.001,           # Adam optimizer learning rate
        'batch_size': 32,                  # Training batch size
        'epochs': 100,                     # Maximum training epochs
        'dropout': 0.2,                    # Dropout rate (0.2 = 20%)
        'lstm_units': 64,                  # LSTM layer units
        'lookback_window': 50,             # Number of historical candles
        'validation_split': 0.2,           # 20% validation data
        'patience': 10,                    # Early stopping patience (epochs)
        'min_delta': 0.0001,               # Minimum improvement threshold
        'optimizer': 'adam',               # Optimization algorithm
        'loss': 'mse',                     # Mean squared error loss
        'metrics': ['mae', 'mse'],         # Evaluation metrics
    }
    
    # 🎯 SYMBOL-SPECIFIC TUNED PARAMETERS (Overrides default)
    # Based on volatility, trend strength, and historical performance
    symbol_params = {
        # ✅ BTC: High volatility, strong trends
        'BTCUSDT': {
            'learning_rate': 0.0005,       # Lower LR for stability
            'epochs': 150,                 # More training for complex patterns
            'lstm_units': 128,             # More capacity for complex patterns
            'lookback_window': 100,        # Longer window for macro trends
            'patience': 15,                # More patience for convergence
        },
        
        # ✅ ETH: Medium-high volatility
        'ETHUSDT': {
            'learning_rate': 0.0007,       # Slightly higher than BTC
            'epochs': 120,                 # Moderate training
            'lstm_units': 96,              # Medium capacity
            'lookback_window': 75,         # Medium window
            'patience': 12,
        },
        
        # ✅ BNB: Medium volatility
        'BNBUSDT': {
            'learning_rate': 0.0009,
            'epochs': 100,
            'lstm_units': 64,
            'lookback_window': 50,
            'patience': 10,
        },
        
        # ✅ SOL: High volatility (similar to BTC)
        'SOLUSDT': {
            'learning_rate': 0.0005,
            'epochs': 140,
            'lstm_units': 128,
            'lookback_window': 90,
            'patience': 14,
        },
        
        # ✅ XRP: Lower volatility
        'XRPUSDT': {
            'learning_rate': 0.0015,       # Higher LR for faster convergence
            'epochs': 80,                  # Less training needed
            'lstm_units': 32,              # Simpler patterns
            'lookback_window': 30,         # Shorter window
            'patience': 8,
        },
    }
    
    # 🔧 LOGIC: Merge symbol-specific with defaults
    if symbol and symbol in symbol_params:
        # Start with defaults
        result = default_params.copy()
        # Override with symbol-specific tuning
        result.update(symbol_params[symbol])
        
        logger.debug(
            "[TunedParams] Loaded params for %s: "
            "lr=%.5f epochs=%d lstm=%d lookback=%d",
            symbol,
            result['learning_rate'],
            result['epochs'],
            result['lstm_units'],
            result['lookback_window']
        )
        
        return result
    
    # Return defaults if no symbol or symbol not tuned
    logger.debug(
        "[TunedParams] Using default params: "
        "lr=%.5f epochs=%d lstm=%d lookback=%d",
        default_params['learning_rate'],
        default_params['epochs'],
        default_params['lstm_units'],
        default_params['lookback_window']
    )
    
    return default_params


def get_symbol_volatility_class(symbol):
    """
    Classify symbol by volatility level.
    
    Used to dynamically adjust hyperparameters at runtime.
    
    Args:
        symbol: Trading pair
    
    Returns:
        String: 'HIGH', 'MEDIUM', or 'LOW'
    """
    high_vol = {'BTCUSDT', 'SOLUSDT', 'ETHUSDT'}
    medium_vol = {'BNBUSDT', 'ADAUSDT'}
    low_vol = {'XRPUSDT', 'LINKUSDT'}
    
    if symbol in high_vol:
        return 'HIGH'
    elif symbol in medium_vol:
        return 'MEDIUM'
    elif symbol in low_vol:
        return 'LOW'
    
    return 'MEDIUM'  # Default to medium


def get_adaptive_learning_rate(symbol, base_volatility_score=0.5):
    """
    Calculate adaptive learning rate based on market conditions.
    
    Args:
        symbol: Trading pair
        base_volatility_score: Current market volatility (0-1, where 1=max)
    
    Returns:
        Float: Adjusted learning rate
    """
    base_lr = get_tuned_params(symbol)['learning_rate']
    
    # Scale learning rate by volatility
    # Higher volatility → lower learning rate (more stability)
    if base_volatility_score > 0.7:
        return base_lr * 0.7
    elif base_volatility_score > 0.5:
        return base_lr * 0.85
    else:
        return base_lr * 1.0  # No adjustment


# ✅ EXPORT FUNCTIONS
__all__ = [
    'get_tuned_params',
    'get_symbol_volatility_class',
    'get_adaptive_learning_rate',
]
