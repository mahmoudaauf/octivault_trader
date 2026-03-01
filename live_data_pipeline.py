"""
Live Data Pipeline

Real-time OHLCV data fetching from Binance.
Feeds into regime detection and position management.
"""

import pandas as pd
import numpy as np
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveDataFetcher:
    """Fetch real-time OHLCV data from Binance"""
    
    def __init__(self, lookback_hours: int = 240):
        """
        Args:
            lookback_hours: How many hours of history to maintain (for regime detection)
        """
        self.base_url = "https://api.binance.com/api/v3"
        self.lookback_hours = lookback_hours
        self.lookback_candles = lookback_hours  # 1H candles
        
        # Local cache of OHLCV data
        self.ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        
        self.logger = logging.getLogger(f"{__name__}.DataFetcher")
    
    def fetch_latest_ohlcv(self, symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Fetch latest OHLCV data for symbol.
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            interval: Candle interval (e.g., "1h")
        
        Returns:
            DataFrame with OHLCV data, or None if error
        """
        
        try:
            # Fetch last N candles
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': self.lookback_candles,
            }
            
            response = requests.get(
                f"{self.base_url}/klines",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse Binance response
            # [time, open, high, low, close, volume, close_time, ...]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Keep only OHLCV
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache it
            self.ohlcv_cache[symbol] = df
            self.last_update[symbol] = datetime.utcnow()
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol}, latest: {df['timestamp'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_cached_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get most recent cached OHLCV data"""
        return self.ohlcv_cache.get(symbol)
    
    def is_cache_fresh(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """Check if cache is fresh enough"""
        if symbol not in self.last_update:
            return False
        
        age = (datetime.utcnow() - self.last_update[symbol]).total_seconds() / 60
        return age < max_age_minutes
    
    def fetch_multiple(self, symbols: list, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols
            force_refresh: Force fetch even if cache fresh
        
        Returns:
            Dict of {symbol: ohlcv_dataframe}
        """
        
        result = {}
        
        for symbol in symbols:
            if force_refresh or not self.is_cache_fresh(symbol):
                df = self.fetch_latest_ohlcv(symbol)
                if df is not None:
                    result[symbol] = df
            else:
                cached = self.get_cached_ohlcv(symbol)
                if cached is not None:
                    result[symbol] = cached
                    self.logger.info(f"Using cached data for {symbol}")
        
        return result


class LivePositionManager:
    """Manage live positions and P&L tracking"""
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        self.positions: Dict[str, Dict] = {}  # symbol -> position details
        self.trades: list = []  # Trade history
        
        self.logger = logging.getLogger(f"{__name__}.PositionManager")
    
    def open_position(self, symbol: str, size: float, entry_price: float, 
                     exposure: float, timestamp: datetime):
        """
        Open a new position.
        
        Args:
            symbol: Trading pair
            size: Position size in units
            entry_price: Entry price
            exposure: Leverage multiplier
            timestamp: Entry timestamp
        """
        
        position_cost = size * entry_price
        
        self.positions[symbol] = {
            'symbol': symbol,
            'size': size,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'exposure': exposure,
            'entry_cost': position_cost,
        }
        
        self.logger.info(f"Opened {symbol}: {size} units @ ${entry_price:.2f} ({exposure}x)")
    
    def update_position(self, symbol: str, current_price: float, timestamp: datetime):
        """Update position with current price"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        pnl = (current_price - position['entry_price']) * position['size']
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        
        position['current_price'] = current_price
        position['current_pnl'] = pnl
        position['current_pnl_pct'] = pnl_pct
        position['last_update'] = timestamp
    
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime, reason: str = ""):
        """
        Close a position.
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
        """
        
        if symbol not in self.positions:
            self.logger.warning(f"No position to close for {symbol}")
            return
        
        position = self.positions[symbol]
        pnl = (exit_price - position['entry_price']) * position['size']
        pnl_pct = pnl / position['entry_cost']
        
        trade = {
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exposure': position['exposure'],
            'reason': reason,
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        self.logger.info(f"Closed {symbol}: PnL ${pnl:.2f} ({pnl_pct:.2%}) - {reason}")
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-level metrics"""
        
        total_pnl = sum(p['current_pnl'] for p in self.positions.values() if 'current_pnl' in p)
        total_pnl_pct = total_pnl / self.initial_balance
        
        num_winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        num_losing_trades = sum(1 for t in self.trades if t['pnl'] < 0)
        
        return {
            'current_balance': self.current_balance + total_pnl,
            'unrealized_pnl': total_pnl,
            'unrealized_pnl_pct': total_pnl_pct,
            'num_open_positions': len(self.positions),
            'num_closed_trades': len(self.trades),
            'winning_trades': num_winning_trades,
            'losing_trades': num_losing_trades,
            'win_rate': num_winning_trades / (num_winning_trades + num_losing_trades) if self.trades else 0,
        }
    
    def print_status(self):
        """Print current portfolio status"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PORTFOLIO STATUS")
        self.logger.info("="*80)
        
        metrics = self.get_portfolio_metrics()
        
        self.logger.info(f"Current balance: ${metrics['current_balance']:,.2f}")
        self.logger.info(f"Unrealized P&L: ${metrics['unrealized_pnl']:,.2f} ({metrics['unrealized_pnl_pct']:.2%})")
        self.logger.info(f"Open positions: {metrics['num_open_positions']}")
        self.logger.info(f"Closed trades: {metrics['num_closed_trades']}")
        
        if self.positions:
            self.logger.info("\nOpen Positions:")
            for symbol, pos in self.positions.items():
                pnl = pos.get('current_pnl', 0)
                pnl_pct = pos.get('current_pnl_pct', 0)
                self.logger.info(f"  {symbol}: {pos['size']:.2f} units @ ${pos.get('current_price', pos['entry_price']):.2f} ({pnl_pct:+.2%})")


if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("LIVE DATA PIPELINE - DEMO")
    logger.info("="*80)
    
    # Initialize
    fetcher = LiveDataFetcher(lookback_hours=240)
    position_manager = LivePositionManager(initial_balance=100000)
    
    # Fetch data
    logger.info("\nFetching live data...")
    data = fetcher.fetch_multiple(['ETHUSDT', 'BTCUSDT'])
    
    for symbol, df in data.items():
        logger.info(f"{symbol}: {len(df)} candles, latest price ${df['close'].iloc[-1]:.2f}")
    
    # Demo position
    if 'ETHUSDT' in data:
        latest_eth = data['ETHUSDT'].iloc[-1]
        position_manager.open_position(
            symbol='ETHUSDT',
            size=1.0,
            entry_price=float(latest_eth['close']),
            exposure=2.0,
            timestamp=pd.to_datetime(latest_eth['timestamp'])
        )
        
        position_manager.update_position(
            'ETHUSDT',
            current_price=float(latest_eth['close']),
            timestamp=pd.to_datetime(latest_eth['timestamp'])
        )
        
        position_manager.print_status()
    
    logger.info("\n✅ Live data pipeline ready for deployment")
