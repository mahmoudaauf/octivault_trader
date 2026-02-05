import asyncio
import logging
from core.shared_state import SharedState
from dashboard_server import DashboardServer

logging.basicConfig(level=logging.INFO)

async def test_dashboard():
    ss = SharedState()
    # Mock some data
    ss.balances["USDT"] = {"free": 5000.0, "locked": 0.0}
    ss.agent_scores["BTCUSDT"] = 0.75
    
    server = DashboardServer(ss, port=8001)
    
    # Run server in background
    task = asyncio.create_task(server.start())
    
    print("Dashboard server started at http://localhost:8001")
    print("Testing REST API /api/state...")
    
    await asyncio.sleep(2)
    
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8001/api/state") as resp:
            data = await resp.json()
            print("API Response:", data)
            assert data["nav"] == 5000.0
            print("✅ REST API OK")

    print("Test finished. Stopping server...")
    # Uvicorn server doesn't have a simple async stop for the server instance easily here, 
    # so we just cancel the task for this test.
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(test_dashboard())
