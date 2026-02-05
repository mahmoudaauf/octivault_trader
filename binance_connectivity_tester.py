import asyncio
import time
import logging

logger = logging.getLogger("BinanceConnectivityTester")
logger.setLevel(logging.INFO)

class BinanceConnectivityTester:
    def __init__(self, exchange_client, symbols, timeframes=None, limit=150, timeout=15):
        self.exchange_client = exchange_client
        self.symbols = symbols
        self.timeframes = timeframes or ["1m", "5m"]
        self.limit = limit
        self.timeout = timeout

    async def test_symbol_tf(self, symbol, timeframe):
        start = time.time()
        try:
            candles = await asyncio.wait_for(
                self.exchange_client.get_ohlcv(symbol, timeframe, limit=self.limit),
                timeout=self.timeout
            )
            elapsed = time.time() - start
            if candles:
                logger.info(f"✅ {symbol}-{timeframe} fetched {len(candles)} candles in {elapsed:.2f}s")
            else:
                logger.warning(f"⚠️ {symbol}-{timeframe} returned no data")
        except asyncio.TimeoutError:
            logger.error(f"⏳ Timeout for {symbol}-{timeframe} after {self.timeout}s")
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}-{timeframe}: {e}")

    async def run_tests(self):
        logger.info("🚀 Starting Binance OHLCV connectivity test...\n")
        tasks = [
            self.test_symbol_tf(symbol, tf)
            for symbol in self.symbols
            for tf in self.timeframes
        ]
        await asyncio.gather(*tasks)
        logger.info("\n✅ Test complete.")

# Sample usage
if __name__ == "__main__":
    import sys
    from core.config import Config
    from core.shared_state import SharedState
    from core.exchange_client import ExchangeClient

    async def main():
        config = Config()
        shared_state = SharedState(config, None)
        exchange_client = ExchangeClient(config, shared_state)
        await exchange_client.initialize()

        # Choose a few known pairs to test
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LAYERUSDT"]

        tester = BinanceConnectivityTester(exchange_client, test_symbols)
        await tester.run_tests()
        await exchange_client.close()

    asyncio.run(main())
