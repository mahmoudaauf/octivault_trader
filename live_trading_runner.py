"""
Integrated Live Trading Runner

Combines:
1. Real-time data fetching
2. Symbol-agnostic regime detection
3. Per-symbol exposure calculation
4. Position management
5. Risk monitoring

Deploy on ETH immediately, expand to other symbols as needed.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from live_trading_system_architecture import (
    LiveTradingOrchestrator, SymbolConfig, RegimeDetectionEngine
)
from live_data_pipeline import LiveDataFetcher, LivePositionManager

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveTradingRunner:
    """Main runner for live trading system"""
    
    def __init__(self, 
                 account_balance: float = 100000,
                 paper_trading: bool = True):
        """
        Args:
            account_balance: Initial account balance
            paper_trading: If True, simulate trades without execution
        """
        
        self.account_balance = account_balance
        self.paper_trading = paper_trading
        
        # Initialize components
        self.orchestrator = LiveTradingOrchestrator(account_balance)
        self.data_fetcher = LiveDataFetcher(lookback_hours=240)
        self.position_manager = LivePositionManager(account_balance)
        
        # Stats
        self.iteration_count = 0
        self.last_signal_time = {}
        self.signal_history = []
        
        self.logger = logging.getLogger(f"{__name__}.Runner")
    
    def initialize(self, symbols_config: dict):
        """
        Initialize trading system with symbol configurations.
        
        Args:
            symbols_config: Dict of {symbol: config_dict}
            
        Example:
            {
                'ETHUSDT': {
                    'enabled': True,
                    'base_exposure': 1.0,
                    'alpha_exposure': 2.0,
                },
                'BTCUSDT': {
                    'enabled': False,
                    'base_exposure': 1.0,
                    'alpha_exposure': 1.0,
                }
            }
        """
        
        for symbol, config_dict in symbols_config.items():
            config = SymbolConfig(symbol=symbol, **config_dict)
            self.orchestrator.initialize_symbol(config)
            self.last_signal_time[symbol] = None
        
        self.logger.info(f"Initialized {len(symbols_config)} symbols")
        self.logger.info(f"Enabled: {self.orchestrator.universe_manager.get_enabled_symbols()}")
        self.logger.info(f"Paper trading: {self.paper_trading}")
    
    def run_iteration(self):
        """
        Run one complete iteration of the trading system.
        
        Steps:
        1. Fetch live OHLCV data
        2. Update regime detection
        3. Calculate signals
        4. Size positions
        5. Execute trades
        6. Monitor risk
        """
        
        self.iteration_count += 1
        timestamp = datetime.utcnow()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"ITERATION {self.iteration_count} - {timestamp}")
        self.logger.info(f"{'='*80}")
        
        # Step 1: Fetch data
        enabled_symbols = self.orchestrator.universe_manager.get_enabled_symbols()
        if not enabled_symbols:
            self.logger.warning("No enabled symbols")
            return
        
        self.logger.info(f"Fetching data for {enabled_symbols}...")
        symbol_data = self.data_fetcher.fetch_multiple(enabled_symbols, force_refresh=True)
        
        if not symbol_data:
            self.logger.warning("No data fetched")
            return
        
        # Step 2: Update regimes
        self.logger.info("Updating regime detection...")
        self.orchestrator.update_regimes(symbol_data)
        
        # Step 3: Calculate signals
        self.logger.info("Calculating trading signals...")
        signals = self.orchestrator.calculate_signals()
        
        # Step 4: Print regimes and signals
        self.logger.info("\nRegime Summary:")
        regime_df = self.orchestrator.get_regime_summary()
        if not regime_df.empty:
            for _, row in regime_df.iterrows():
                self.logger.info(f"  {row['Symbol']}: {row['Regime']} | Macro: {row['MacroTrend']} | Price: ${row['Price']:.2f}")
        
        self.logger.info("\nTrading Signals:")
        for symbol, signal in signals.items():
            action = signal['action']
            exposure = signal['exposure']
            is_alpha = signal['is_alpha_regime']
            
            symbol_emoji = "⚡" if is_alpha else "→"
            self.logger.info(f"  {symbol_emoji} {symbol}: {action} ({exposure}x exposure)")
            
            # Record signal
            self.signal_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'regime': signal['regime'],
                'action': action,
                'exposure': exposure,
                'is_alpha': is_alpha,
            })
        
        # Step 5: Position management (demo - would connect to exchange here)
        if not self.paper_trading:
            self.logger.info("\n⚠️ PAPER TRADING MODE - Not executing real trades")
            self.logger.info("   To go live, set paper_trading=False and implement exchange API")
        
        # Step 6: Risk monitoring
        self.logger.info("\nRisk Monitoring:")
        metrics = self.position_manager.get_portfolio_metrics()
        self.logger.info(f"  Portfolio P&L: {metrics['unrealized_pnl_pct']:+.2%}")
        self.logger.info(f"  Open positions: {metrics['num_open_positions']}")
        
        if metrics['unrealized_pnl_pct'] < -0.30:
            self.logger.critical(f"⚠️ STOP LOSS TRIGGERED: {metrics['unrealized_pnl_pct']:.2%} < -30%")
        
        return signals
    
    def print_summary(self):
        """Print system summary"""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("SYSTEM SUMMARY")
        self.logger.info("="*80)
        
        self.logger.info(f"\nIterations run: {self.iteration_count}")
        self.logger.info(f"Total signals generated: {len(self.signal_history)}")
        
        # Signal distribution
        if self.signal_history:
            signal_df = pd.DataFrame(self.signal_history)
            
            self.logger.info(f"\nSignal Distribution:")
            for symbol in signal_df['symbol'].unique():
                symbol_signals = signal_df[signal_df['symbol'] == symbol]
                alpha_count = symbol_signals['is_alpha'].sum()
                total = len(symbol_signals)
                pct = alpha_count / total * 100
                
                self.logger.info(f"  {symbol}: {alpha_count}/{total} alpha regime signals ({pct:.1f}%)")
        
        # Portfolio summary
        self.logger.info(f"\nPortfolio Summary:")
        metrics = self.position_manager.get_portfolio_metrics()
        self.logger.info(f"  Current balance: ${metrics['current_balance']:,.2f}")
        self.logger.info(f"  Unrealized P&L: {metrics['unrealized_pnl_pct']:+.2%}")
        self.logger.info(f"  Closed trades: {metrics['num_closed_trades']}")
        if metrics['num_closed_trades'] > 0:
            self.logger.info(f"  Win rate: {metrics['win_rate']:.1%}")
    
    def get_signal_stats(self) -> Dict:
        """Get statistics on generated signals"""
        
        if not self.signal_history:
            return {}
        
        signal_df = pd.DataFrame(self.signal_history)
        
        stats = {}
        for symbol in signal_df['symbol'].unique():
            symbol_signals = signal_df[signal_df['symbol'] == symbol]
            alpha_count = symbol_signals['is_alpha'].sum()
            total = len(symbol_signals)
            
            stats[symbol] = {
                'total_signals': total,
                'alpha_signals': alpha_count,
                'alpha_frequency_pct': alpha_count / total * 100,
            }
        
        return stats


def main():
    """Main entry point for live trading"""
    
    logger.info("\n" + "╔" + "="*78 + "╗")
    logger.info("║" + " "*15 + "UNIVERSE-READY LIVE TRADING SYSTEM" + " "*29 + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "  Architecture: Symbol-agnostic regime detection + per-symbol exposure" + " "*5 + "║")
    logger.info("║" + "  Deploy: ETH immediately, expand via configuration" + " "*22 + "║")
    logger.info("╚" + "="*78 + "╝\n")
    
    # Initialize runner
    runner = LiveTradingRunner(
        account_balance=100000,
        paper_trading=True  # Paper trading for now
    )
    
    # Configure symbols
    symbols_config = {
        'ETHUSDT': {
            'enabled': True,
            'base_exposure': 1.0,
            'alpha_exposure': 2.0,
            'downtrend_exposure': 0.0,
            'max_position_size_pct': 0.05,
            'max_drawdown_threshold': 0.30,
        },
        'BTCUSDT': {
            'enabled': False,  # Disabled - enable when ready
            'base_exposure': 1.0,
            'alpha_exposure': 1.0,
            'downtrend_exposure': 0.0,
            'max_position_size_pct': 0.05,
            'max_drawdown_threshold': 0.30,
        },
    }
    
    runner.initialize(symbols_config)
    
    # Run iterations
    logger.info(f"\nRunning 5 iterations...\n")
    
    for i in range(5):
        try:
            runner.run_iteration()
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {e}")
        
        # Wait between iterations (in live, this would be on schedule)
        import time
        if i < 4:
            logger.info("Waiting 2 seconds before next iteration...")
            time.sleep(2)
    
    # Print summary
    runner.print_summary()
    
    logger.info("\n" + "="*80)
    logger.info("✅ DEMO COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps for production deployment:")
    logger.info("  1. Implement exchange API integration (paper_trading=False)")
    logger.info("  2. Add database for trade history and P&L tracking")
    logger.info("  3. Implement risk monitoring alerts")
    logger.info("  4. Add rotation layer to UniverseManager for multi-symbol trading")
    logger.info("  5. Deploy on live account with small initial allocation (5%)")


if __name__ == '__main__':
    main()
