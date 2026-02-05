import asyncio
import logging
import time
import pandas as pd
from typing import List, Dict
from core.shared_state import SharedState
from core.market_simulator import MarketSimulator
from utils.backtest_data_loader import BacktestDataLoader
from core.execution_manager import ExecutionManager
from core.config import Config
from portfolio.balancer import PortfolioBalancer
from core.compounding_engine import CompoundingEngine
from utils.backtest_reporter import BacktestReporter

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestRunner")

async def run_backtest(symbols: List[str], start_idx: int = 100, end_idx: int = 1000):
    logger.info(f"🚀 Initializing Backtest for {symbols}...")
    
    # 1. Setup Infrastructure
    config = Config()
    shared_state = SharedState()
    loader = BacktestDataLoader()
    
    # Load Data
    data_map: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = loader.load_csv(s)
        if df is not None:
            data_map[s] = df
    
    if not data_map:
        logger.error("No data loaded. Aborting.")
        return

    # 2. Setup Simulator & Managers
    simulator = MarketSimulator(shared_state)
    execution_manager = ExecutionManager(config, shared_state, simulator)
    
    # Initialize Core Components
    balancer = PortfolioBalancer(shared_state, simulator, execution_manager, config)
    compounding = CompoundingEngine(shared_state, simulator, config, execution_manager)
    
    # Setup SharedState accepted symbols & Enable Balancer
    shared_state.dynamic_config["ENABLE_BALANCER"] = True
    shared_state.dynamic_config["REBALANCE_INTERVAL_SEC"] = 300
    shared_state.dynamic_config["TARGET_EXPOSURE"] = 0.8 # Deploy 80% of NAV
    await shared_state.set_accepted_symbols({s: {"score": 0.5} for s in symbols})
    
    # Seed initial positions to kickstart balancer (or use Compounding)
    # Simulator will sync these to shared_state
    simulator.balances["BTC"] = {"free": 0.0, "locked": 0.0}
    simulator.balances["ETH"] = {"free": 0.0, "locked": 0.0}
    await simulator._sync_balances()
    
    shared_state.market_data_ready_event.set()

    # 3. Simulation Loop
    reporter = BacktestReporter()
    logger.info(f"Starting simulation from index {start_idx} to {end_idx}...")
    
    for i in range(start_idx, end_idx):
        # Update Market State
        prices = {}
        for s, df in data_map.items():
            if i >= len(df): continue
            row = df.iloc[i]
            prices[s] = float(row['close'])
            
            # Update market data bar for indicators
            shared_state.market_data[(s, "5m")] = [
                {"ts": row['timestamp'], "o": row['open'], "h": row['high'], "l": row['low'], "c": row['close'], "v": row['volume']}
            ]
        
        for s, p in prices.items():
            await shared_state.update_latest_price(s, p)

        # Mock Signal Conviction: BTC is bullish, ETH is bearish for first half, then flip
        if i < (start_idx + end_idx) / 2:
            shared_state.agent_scores["BTCUSDT"] = 0.8
            shared_state.agent_scores["ETHUSDT"] = 0.4
        else:
            shared_state.agent_scores["BTCUSDT"] = 0.3
            shared_state.agent_scores["ETHUSDT"] = 0.9

        # Sync Simulator Time
        simulator.set_time_index(i)
        
        # Drive Component Ticks
        await balancer.run_once()
        
        # Record & Log Progress
        nav = shared_state.get_nav_quote()
        reporter.record_step({
            "step": i,
            "nav": nav,
            "prices": prices.copy()
        })
        
        if i % 100 == 0:
            pos = shared_state.positions
            logger.info(f"Step {i} | NAV: {nav:.2f} USDT | Positions: {list(pos.keys())}")

    # 4. Final Report
    reporter.save_csv()
    reporter.print_report()

if __name__ == "__main__":
    # Example usage:
    # Make sure you have data/historical/BTCUSDT_5m.csv etc.
    asyncio.run(run_backtest(["BTCUSDT", "ETHUSDT"]))
