import asyncio
import logging
import os
from core.app_context import AppContext

logging.basicConfig(level=logging.INFO)

async def test_phased_dashboard():
    config = {
        "DASHBOARD_PORT": 8002,
        "UP_TO_PHASE": 8
    }
    
    ctx = AppContext(config=config)
    
    # Pre-build components so we can seed them
    ctx._ensure_components_built()

    # Mock gates to bypass hardware/API requirements
    async def mock_wait(*args, **kwargs):
        return {"status": "READY", "issues": []}
    ctx._wait_until_ready = mock_wait

    # Seed some data to make components progress
    if ctx.shared_state:
        ctx.shared_state.accepted_symbols = ["BTCUSDT", "ETHUSDT"]
        # Mark symbols ready to bypass MDF warmup wait
        ctx.shared_state.set_readiness_flag("accepted_symbols_ready", True)

    try:
        await ctx.initialize_all(up_to_phase=8)
        
        if ctx.dashboard_server:
            print("✅ Dashboard Server built")
            # Give it a second to start
            await asyncio.sleep(5)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8002/api/state") as resp:
                    if resp.status == 200:
                        print("✅ Dashboard Server responding at P8")
                        data = await resp.json()
                        print("State:", data)
                    else:
                        print(f"❌ Dashboard Server returned {resp.status}")
        else:
            print("❌ Dashboard Server not built")

    except Exception as e:
        print(f"Caught expected/unexpected error: {e}")
    finally:
        await ctx.shutdown()

if __name__ == "__main__":
    # Mock some env for MDF/Exchange if needed, or just run and see where it gates.
    # initialize_all(up_to_phase=8) will gate on P4 (MDF) if it's not ready.
    # To test only the build/start logic, we might need a more controlled test.
    asyncio.run(test_phased_dashboard())
