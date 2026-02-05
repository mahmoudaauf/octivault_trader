
import asyncio
import logging
import os
import time
from unittest.mock import MagicMock

# Mock components to simulate the environment
class MockConfig:
    MAX_PER_TRADE_USDT = 15.0
    MIN_SIGNAL_CONF = 0.55
    TARGET_HOURLY_PNL = 10.0
    PERF_CHECK_INTERVAL = 1
    max_performance_samples = 100
    circuit_breaker_failure_threshold = 5
    circuit_breaker_timeout = 60

from core.shared_state import SharedState
from core.performance_watcher import PerformanceWatcher
from agents.trend_hunter import TrendHunter
from portfolio.balancer import PortfolioBalancer
from core.compounding_engine import CompoundingEngine

async def test_dynamic_tuning():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TestDynamicTuning")
    
    config = MockConfig()
    exchange_client = MagicMock()
    execution_manager = MagicMock()
    shared_state = SharedState({}, None, None) # config_dict is first arg now
    shared_state.config = config # Manually attach mock config for testing
    
    # 1. Initialize PerformanceWatcher
    pw = PerformanceWatcher(shared_state, config)
    
    # Mock equity calculation to simulate underperformance
    # Start equity = 200, elapsed = 1hr, current equity = 201 (Rate = 1.0/hr, Target = 10.0/hr)
    pw._get_total_equity = asyncio.iscoroutinefunction(pw._get_total_equity) # Ensure it's treated as async if needed
    pw._get_total_equity = lambda: asyncio.sleep(0, result=201.0)
    
    pw._start_equity = 200.0
    pw._start_time = time.time() - 3600.0 # 1 hour ago
    
    # 2. Initialize TrendHunter
    th = TrendHunter(shared_state, None, None, config, None, None, symbols=["BTCUSDT"])
    
    logger.info(f"Initial TrendHunter min_conf: {th.min_conf}")
    logger.info(f"Initial TrendHunter max_per_trade_usdt: {th.max_per_trade_usdt}")

    # 3. Run evaluation
    logger.info("Triggering performance evaluation (Underperforming scenario)...")
    await pw._evaluate_performance()
    
    # 4. Verify TrendHunter reflects the change
    print(f"Aggressive TrendHunter min_conf: {th.min_conf}")
    print(f"Aggressive TrendHunter max_per_trade_usdt: {th.max_per_trade_usdt}")
    
    # Check PortfolioBalancer
    pb = PortfolioBalancer(shared_state, exchange_client, execution_manager, config)
    print(f"Aggressive Balancer interval: {pb.rebalance_interval}")
    
    # Check CompoundingEngine
    ce = CompoundingEngine(shared_state, exchange_client, config, execution_manager)
    print(f"Aggressive Compounding threshold: {ce.min_compound_threshold}")

    if th.min_conf < 0.65 and pb.rebalance_interval == 120 and ce.min_compound_threshold == 5.0:
        logger.info("SUCCESS: Capital Management tuned for Aggressiveness.")
    else:
        logger.error("FAILURE: Capital Management tuning failed for Aggressiveness.")

    # 4. Simulate Drawdown (Defensive)
    logger.info("Triggering performance evaluation (Drawdown scenario)...")
    report_drawdown = {
        "pnl": -10.0,
        "target": 10.0,
        "hourly_rate": -10.0,
        "drawdown": 0.05
    }
    await pw._tune_agents(report_drawdown)

    # --- CROSS-SYSTEM LOGIC VERIFICATION (NEW) ---
    logger.info("Verifying Unified Scoring...")
    shared_state.agent_scores["BTCUSDT"] = 0.8
    score = shared_state.get_unified_score("BTCUSDT")
    print(f"Unified Score for BTCUSDT: {score}")
    if score > 0.5:
        logger.info("SUCCESS: Unified score calculated.")
    
    logger.info("Verifying Compounding/Balancer Coordination...")
    # Simulate balancer targets
    shared_state.rebalance_targets = {"BTCUSDT"}
    ce = CompoundingEngine(shared_state, exchange_client, config, execution_manager)
    # Mock symbols available and dynamic config for max_symbols
    shared_state.dynamic_config["MAX_COMPOUND_SYMBOLS"] = 10
    ce.shared_state.get_accepted_symbols_snapshot = lambda: {"BTCUSDT": {}, "ETHUSDT": {}}
    picked = ce._pick_symbols()
    print(f"Compounding picked symbols: {picked}")
    if "ETHUSDT" not in picked:
        logger.info("SUCCESS: CompoundingEngine respected Balancer targets (Filtered out ETH).")
    else:
        logger.error("FAILURE: CompoundingEngine failed to filter non-target symbols.")

    logger.info("Verifying Risk Manager Tuning...")
    if th.min_conf > 0.65 and shared_state.dynamic_config.get("MAX_DRAWDOWN_PCT") == 0.05:
         logger.info("SUCCESS: Risk Manager tuned for Defensiveness.")
    else:
         logger.error(f"FAILURE: Risk Manager tuning failed. MAX_REB={shared_state.dynamic_config.get('REBALANCE_INTERVAL_SEC')} MAX_DD={shared_state.dynamic_config.get('MAX_DRAWDOWN_PCT')}")

    if th.min_conf > 0.65 and pb.rebalance_interval == 900 and ce.min_compound_threshold == 20.0:
        logger.info("SUCCESS: Capital Management tuned for Defensiveness.")
    else:
        logger.error("FAILURE: Capital Management tuning failed for Defensiveness.")

if __name__ == "__main__":
    asyncio.run(test_dynamic_tuning())
