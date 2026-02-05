import logging
import time
from datetime import datetime, timedelta

from utils.ohlcv_cache import fetch_and_cache_ohlcv
from shared_state import SharedState
from agents import agent_factory
from execution.paper_executor import PaperExecutor
from performance.performance_monitor import PerformanceMonitor

class ABTester:
    def __init__(self, config, symbol, duration_hours=6):
        self.config = config
        self.symbol = symbol
        self.duration = timedelta(hours=duration_hours)

        self.shared_state_A = SharedState()
        self.shared_state_B = SharedState()

        self.agent_A = agent_factory.create(config["AgentA"], self.shared_state_A)
        self.agent_B = agent_factory.create(config["AgentB"], self.shared_state_B)

        self.executor_A = PaperExecutor(self.shared_state_A, config)
        self.executor_B = PaperExecutor(self.shared_state_B, config)

        self.performance_A = PerformanceMonitor(self.shared_state_A)
        self.performance_B = PerformanceMonitor(self.shared_state_B)

    def run(self):
        logging.info("🔬 Starting A/B testing...")
        end_time = datetime.utcnow()
        start_time = end_time - self.duration

        # Fetch same market data for both agents
        ohlcv = fetch_and_cache_ohlcv(self.symbol, interval="1m", limit=int(self.duration.total_seconds() / 60))

        for index, row in ohlcv.iterrows():
            candle = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "timestamp": row["timestamp"]
            }

            self.shared_state_A.update_market_data(self.symbol, [candle])
            self.shared_state_B.update_market_data(self.symbol, [candle])

            # Run agent logic
            signal_A = self.agent_A.run(self.symbol)
            signal_B = self.agent_B.run(self.symbol)

            if signal_A:
                self.executor_A.execute_trade(signal_A, self.symbol)
            if signal_B:
                self.executor_B.execute_trade(signal_B, self.symbol)

        # Collect performance data
        result_A = self.performance_A.evaluate(self.symbol)
        result_B = self.performance_B.evaluate(self.symbol)

        self.print_results(result_A, result_B)

    def print_results(self, result_A, result_B):
        print("\n🧪 A/B Test Results Comparison")
        print("--------------------------------------------------")
        print(f"Metric           | Agent A       | Agent B")
        print("--------------------------------------------------")
        print(f"PnL              | {result_A['pnl']:.2f} USDT  | {result_B['pnl']:.2f} USDT")
        print(f"Win Rate         | {result_A['win_rate']:.2%}   | {result_B['win_rate']:.2%}")
        print(f"ROI              | {result_A['roi']:.2%}   | {result_B['roi']:.2%}")
        print(f"Total Trades     | {result_A['trades']}         | {result_B['trades']}")
        print("--------------------------------------------------")

