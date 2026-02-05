import asyncio
import logging
from core.app_context import AppContext

logging.basicConfig(level=logging.INFO)

async def run_dashboard():
    config = {
        "DASHBOARD_PORT": 8002,
        "UP_TO_PHASE": 8
    }
    
    ctx = AppContext(config=config)
    
    # Mock gates
    async def mock_wait(*args, **kwargs):
        return {"status": "READY", "issues": []}
    ctx._wait_until_ready = mock_wait
    
    # Seed data
    ctx._ensure_components_built()
    if ctx.shared_state:
        ctx.shared_state.accepted_symbols = ["BTCUSDT", "ETHUSDT"]
        ctx.shared_state.set_readiness_flag("accepted_symbols_ready", True)
        ctx.shared_state.balances["USDT"] = {"free": 10000.0, "locked": 0.0}
        ctx.shared_state.agent_scores["BTCUSDT"] = 0.85
        ctx.shared_state.agent_scores["ETHUSDT"] = 0.45

    try:
        await ctx.initialize_all(up_to_phase=8)
        print("Dashboard server is running at http://localhost:8002")
        # Keep alive
        while True:
            await asyncio.sleep(3600)
    finally:
        await ctx.shutdown()

if __name__ == "__main__":
    asyncio.run(run_dashboard())
