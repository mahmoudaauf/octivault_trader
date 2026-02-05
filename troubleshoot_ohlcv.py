import asyncio
import aiohttp
import time

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
timeframes = ["1m", "5m"]
limit = 150
timeout_per_request = 15  # seconds

async def fetch_ohlcv(session, symbol, interval):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        start = time.time()
        async with session.get(url, timeout=timeout_per_request) as response:
            if response.status != 200:
                print(f"❌ {symbol}-{interval} failed: HTTP {response.status}")
                return
            data = await response.json()
            duration = time.time() - start
            print(f"✅ {symbol}-{interval} fetched {len(data)} candles in {duration:.2f}s")
    except asyncio.TimeoutError:
        print(f"⏳ Timeout for {symbol}-{interval} after {timeout_per_request}s")
    except Exception as e:
        print(f"❌ Error fetching {symbol}-{interval}: {e}")

async def run_diagnostics():
    print("🚨 Starting OHLCV fetch diagnostic...\n")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            for tf in timeframes:
                tasks.append(fetch_ohlcv(session, symbol, tf))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
