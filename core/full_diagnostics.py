import asyncio
import logging
import time
from datetime import datetime
from utils.shared_state_tools import inject_agent_signal

logger = logging.getLogger("Diagnostics")
logger.setLevel(logging.INFO)

class DiagnosticsEngine:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.interval = 10  # seconds

    async def run(self):
        logger.info("�� Starting diagnostics loop...")
        while True:
            try:
                await self.diagnostic_loop()
            except Exception as e:
                logger.exception(f"❌ Diagnostics error: {e}")
            await asyncio.sleep(self.interval)

    async def diagnostic_loop(self):
        now = time.time()

        # 1. Market Data Health
        last_update = getattr(self.shared_state, "last_update", 0)
        if isinstance(last_update, dict):
            logger.warning("⚠️ Invalid timestamp: last_update is a dict. Resetting to current time.")
            last_update = time.time()
            self.shared_state.last_update = last_update

        age = int(now - last_update)
        logger.info(f"📊 Market data last updated {age}s ago.")

        if age > 60:
            logger.warning("🕒 Market data seems stale!")

        # 2. Active Trades Overview
        open_trades = list(self.shared_state.active_trades.keys())
        logger.info(f"📈 Open trades: {open_trades}")

        # 3. Agent Signal Sanity Check
        for symbol, signal in self.shared_state.agent_signals.items():
            logger.info(f"🧠 Signal for {symbol}: {signal}")

        # 4. Last Orders Check
        if hasattr(self.shared_state, "last_orders"):
            for symbol, order in self.shared_state.last_orders.items():
                logger.info(f"📦 Last order for {symbol}: {order}")
        else:
            logger.error("❌ SharedState missing attribute 'last_orders'")

        # 5. Inject Test Signal (Live-safe)
        test_symbol = "BTCUSDT"
        test_signal = {
            "action": "BUY",
            "confidence": 0.95,
            "reason": "Diagnostics injection",
            "timestamp": datetime.utcnow().isoformat()
        }
        await inject_agent_signal(self.shared_state, "Diagnostics", test_symbol, test_signal)
        logger.info(f"✅ Injected test signal for {test_symbol}: {test_signal}")

        # 6. Alert Review
        if hasattr(self.shared_state, "alerts"):
            for alert in self.shared_state.alerts[-5:]:
                logger.info(f"🚨 Recent Alert: {alert}")

        # 7. Portfolio Summary
        balances = self.shared_state.balances
        logger.info(f"💼 Balances: {balances}")
        equity = self.shared_state.performance_metrics.get("current_equity_usd", "N/A")
        logger.info(f"📊 Equity: ${equity}")

        # 8. Health Flags
        if self.shared_state.performance_metrics.get("max_drawdown_pct", 0.0) > 25:
            logger.warning("⚠️ High max drawdown detected!")

# Standalone diagnostic loop for minimal dependency mode
async def diagnostic_loop(shared_state, expected_agents, symbols, interval=30):
    while True:
        logger.debug(f"🔍 Running diagnostics for agents: {expected_agents}")
        for symbol in symbols:
            signal = shared_state.agent_signals.get(symbol)
            logger.info(f"🔍 Signal [{symbol}] = {signal}")
        await asyncio.sleep(interval)
