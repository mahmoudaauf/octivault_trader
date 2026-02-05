import asyncio
import logging
import time

from core.exchange_client import ExchangeClient
from core.config import Config
from core.shared_state import SharedState
from core.database_manager import DatabaseManager

from httpx import ConnectError, HTTPStatusError, RequestError

logging.basicConfig(level=logging.INFO)

async def debug_signed_request():
    config = Config()
    db = DatabaseManager(config=config)
    dummy_exchange_client = None  # Will assign after shared_state is created

    shared_state = SharedState(
        config=config,
        database_manager=db,
        exchange_client=dummy_exchange_client  # temporary, will patch later
    )

    # Complete circular dependency
    exchange = ExchangeClient(config=config, shared_state=shared_state)
    shared_state.exchange_client = exchange  # Patch back reference

    # Use a safe GET request that doesn't place trades
    test_endpoint = "/api/v3/account"
    test_payload = {
        "timestamp": int(time.time() * 1000)
    }

    try:
        logging.info("🚀 Sending test signed request to Binance...")
        response = await exchange._send_signed_request("GET", test_endpoint, test_payload)
        logging.info("✅ SUCCESS: Binance response:\n%s", response)

    except ConnectError as e:
        logging.error("❌ Network connection error: %s", e)

    except HTTPStatusError as e:
        logging.error("❌ HTTP error: %s | Status: %s | Content: %s", e, e.response.status_code, e.response.text)

    except RequestError as e:
        logging.error("❌ Request failed: %s", e)

    except Exception as e:
        logging.exception("🔥 Unhandled exception: %s", e)

if __name__ == "__main__":
    asyncio.run(debug_signed_request())
