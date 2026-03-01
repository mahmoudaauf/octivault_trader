#!/usr/bin/env python3
"""
Launch Regime Trading System

Simple command-line interface to run the regime trading system
in paper, backtest, or live modes.

Usage:
    python launch_regime_trading.py --mode paper
    python launch_regime_trading.py --mode backtest --symbols ETHUSDT
    python launch_regime_trading.py --mode live --symbols ETHUSDT BTCUSDT
    
Environment Variables:
    REGIME_TRADING_MODE: Override --mode (paper, backtest, live)
    PAPER_TRADING: True/False (forces paper mode)
    ENABLE_REGIME_TRADING: True/False (enable/disable system)
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Config
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.execution_manager import ExecutionManager
from core.market_data_feed import MarketDataFeed
from core.regime_trading_integration import (
    RegimeTradingAdapter,
    RegimeTradingConfig,
    create_regime_trading_adapter,
)
from live_trading_system_architecture import SymbolConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config_from_env() -> Dict:
    """Load configuration from environment variables"""
    return {
        "mode": os.getenv("REGIME_TRADING_MODE", "paper").lower(),
        "paper_trading": os.getenv("PAPER_TRADING", "true").lower() == "true",
        "enabled": os.getenv("ENABLE_REGIME_TRADING", "true").lower() == "true",
        "symbols": os.getenv("SYMBOLS", "ETHUSDT").split(","),
        "sync_interval": float(os.getenv("SYNC_INTERVAL_SECONDS", "60")),
    }


def create_symbol_configs(symbols: List[str]) -> Dict[str, SymbolConfig]:
    """Create symbol configurations from environment or defaults"""
    configs = {}
    
    for symbol in symbols:
        configs[symbol] = SymbolConfig(
            symbol=symbol,
            enabled=True,
            
            # Regime parameters (from backtest tuning)
            volatility_lookback=100,
            vol_percentile_low=0.33,
            vol_percentile_high=0.67,
            autocorr_threshold=0.1,
            sma_period=200,
            
            # Exposure configuration
            base_exposure=float(os.getenv(f"{symbol}_BASE_EXPOSURE", "1.0")),
            alpha_exposure=float(os.getenv(f"{symbol}_ALPHA_EXPOSURE", "2.0")),
            downtrend_exposure=0.0,
            
            # Risk management
            max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "0.05")),
            max_drawdown_threshold=float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.30")),
            daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "0.02")),
            
            # Validation
            min_signal_frequency=0.01,
        )
    
    return configs


# ============================================================================
# INITIALIZATION
# ============================================================================

async def initialize_components() -> tuple[SharedState, ExecutionManager, MarketDataFeed]:
    """Initialize core Octivault components"""
    try:
        logger.info("Initializing Octivault components...")
        
        # Load configuration
        config = Config()
        
        # Initialize SharedState
        shared_state = SharedState(config=config)
        await shared_state.initialize()
        
        # Initialize ExchangeClient
        exchange_client = ExchangeClient(config=config)
        shared_state.exchange_client = exchange_client
        
        # Initialize ExecutionManager
        execution_manager = ExecutionManager(
            shared_state=shared_state,
            exchange_client=exchange_client,
        )
        
        # Initialize MarketDataFeed
        market_data_feed = MarketDataFeed(
            exchange_client=exchange_client,
            shared_state=shared_state,
        )
        
        logger.info("✅ Components initialized")
        return shared_state, execution_manager, market_data_feed
        
    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}", exc_info=True)
        raise


# ============================================================================
# MAIN LOOP
# ============================================================================

async def run_paper_trading(
    adapter: RegimeTradingAdapter,
    duration_hours: Optional[int] = None,
) -> None:
    """
    Run paper trading mode (infinite loop until interrupted).
    
    Args:
        adapter: Initialized RegimeTradingAdapter
        duration_hours: Optional duration limit in hours
    """
    logger.info("📊 Starting paper trading mode...")
    logger.info("  Press Ctrl+C to stop")
    
    start_time = datetime.utcnow()
    iteration_count = 0
    
    try:
        while True:
            try:
                # Check duration limit
                if duration_hours:
                    elapsed = datetime.utcnow() - start_time
                    if elapsed.total_seconds() > duration_hours * 3600:
                        logger.info(f"Duration limit reached ({duration_hours}h)")
                        break
                
                # Run iteration
                iteration_count += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"Iteration {iteration_count} - {datetime.utcnow().isoformat()}")
                logger.info(f"{'='*80}")
                
                result = await adapter.run_iteration()
                
                # Log results
                if result["success"]:
                    logger.info(f"✅ Iteration successful")
                    logger.info(f"   Regimes detected: {len(result['regime_states'])}")
                    logger.info(f"   Trades executed: {len(result['trades_executed'])}")
                    logger.info(f"   Positions: {len(result['positions'])}")
                    
                    # Log key metrics
                    metrics = result.get("metrics", {})
                    if metrics:
                        logger.info(f"   Metrics:")
                        for key, value in metrics.items():
                            logger.info(f"      {key}: {value}")
                else:
                    logger.warning(f"⚠️  Iteration had errors:")
                    for error in result.get("errors", []):
                        logger.warning(f"    - {error}")
                
                # Wait before next iteration
                await asyncio.sleep(adapter.config.sync_interval_seconds)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Iteration error: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retry
    
    except KeyboardInterrupt:
        logger.info("\n📍 Stopping paper trading...")
    finally:
        await adapter.shutdown()
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("Paper Trading Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Total iterations: {iteration_count}")
        logger.info(f"Total trades: {len(adapter.trade_log)}")
        logger.info(f"Elapsed time: {datetime.utcnow() - start_time}")
        
        status = adapter.get_status()
        logger.info(f"\nFinal Status:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")


async def run_backtest(
    adapter: RegimeTradingAdapter,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """
    Run backtesting mode.
    
    This would use historical data instead of live data.
    Requires additional implementation of historical data fetching.
    """
    logger.info(f"📈 Starting backtest mode...")
    logger.info(f"   Period: {start_date.date()} to {end_date.date()}")
    
    # Note: Full backtest implementation would require:
    # 1. Historical data loader
    # 2. Time-travel simulation
    # 3. Performance evaluation
    
    logger.warning("Backtest mode requires additional implementation")
    logger.info("Use extended_walk_forward_validator.py for full backtests")


async def run_live_trading(
    adapter: RegimeTradingAdapter,
    dry_run: bool = True,
) -> None:
    """
    Run live trading mode (with optional dry-run).
    
    Args:
        adapter: Initialized RegimeTradingAdapter
        dry_run: If True, execute orders but in paper mode first
    """
    logger.warning("⚠️  LIVE TRADING MODE")
    logger.warning(f"   Paper trading: {adapter.config.paper_trading}")
    logger.warning(f"   Dry-run: {dry_run}")
    
    if not dry_run and not adapter.config.paper_trading:
        logger.error("❌ LIVE TRADING NOT ENABLED IN CONFIG")
        return
    
    await run_paper_trading(adapter, duration_hours=None)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Launch Regime Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_regime_trading.py --mode paper
  python launch_regime_trading.py --mode paper --symbols ETHUSDT BTCUSDT
  python launch_regime_trading.py --mode live --duration 24
  
Environment variables:
  REGIME_TRADING_MODE=paper
  PAPER_TRADING=true
  ENABLE_REGIME_TRADING=true
  SYMBOLS=ETHUSDT,BTCUSDT
  ETHUSDT_ALPHA_EXPOSURE=2.0
  MAX_POSITION_SIZE_PCT=0.05
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["paper", "backtest", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ETHUSDT"],
        help="Symbols to trade (default: ETHUSDT)",
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in hours (for paper trading, default: infinite)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode for live trading (paper orders)",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Start date for backtest (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="End date for backtest (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(args.log_level)
    
    logger.info(f"{'='*80}")
    logger.info("Regime Trading System Launcher")
    logger.info(f"{'='*80}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Duration: {args.duration} hours" if args.duration else "Duration: infinite")
    logger.info(f"{'='*80}\n")
    
    try:
        # Initialize components
        shared_state, execution_manager, market_data_feed = await initialize_components()
        
        # Create configuration
        env_config = load_config_from_env()
        symbol_configs = create_symbol_configs(args.symbols)
        
        regime_config = RegimeTradingConfig(
            enabled=env_config["enabled"],
            paper_trading=env_config["paper_trading"] or (args.mode != "live"),
            symbols=symbol_configs,
            sync_interval_seconds=env_config["sync_interval"],
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  Enabled: {regime_config.enabled}")
        logger.info(f"  Paper trading: {regime_config.paper_trading}")
        logger.info(f"  Symbols: {list(regime_config.symbols.keys())}")
        logger.info(f"  Sync interval: {regime_config.sync_interval_seconds}s\n")
        
        # Create adapter
        adapter = await create_regime_trading_adapter(
            shared_state=shared_state,
            execution_manager=execution_manager,
            market_data_feed=market_data_feed,
            config=regime_config,
        )
        
        if not adapter:
            logger.error("Failed to create adapter")
            return 1
        
        # Run requested mode
        if args.mode == "paper":
            await run_paper_trading(adapter, duration_hours=args.duration)
        elif args.mode == "backtest":
            start = datetime.strptime(args.start_date, "%Y-%m-%d")
            end = datetime.strptime(args.end_date, "%Y-%m-%d")
            await run_backtest(adapter, start_date=start, end_date=end)
        elif args.mode == "live":
            await run_live_trading(adapter, dry_run=args.dry_run)
        
        logger.info("✅ Launcher completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
