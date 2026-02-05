import asyncio
from core.config import Config
from core.exchange_client import ExchangeClient
from core.symbol_manager import SymbolManager

async def test():
    config = Config()
    exchange_client = ExchangeClient(config, shared_state=None)  # ✅ Patch here
    await exchange_client.initialize()

    symbol_manager = SymbolManager(
        shared_state=None,  # Not needed for symbol validation
        config=config,
        exchange_client=exchange_client
    )

    symbols_to_test = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SHIBUSDT", "PEPEUSDT"]
    for symbol in symbols_to_test:
        is_valid, reason = await symbol_manager.is_valid_symbol(symbol)
        print(f"{symbol}: {'✅ Valid' if is_valid else '❌ Invalid'} | Reason: {reason}")

    await exchange_client.close()

asyncio.run(test())
