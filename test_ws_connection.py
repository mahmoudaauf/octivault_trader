#!/usr/bin/env python3
"""
Test script to verify Binance WebSocket API v3 connection and authentication.
"""
import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.exchange_client import ExchangeClient
from core.config import Config
from core.shared_state import SharedState


async def test_ws_connection() -> bool:
    """
    Test WebSocket connection to Binance API v3.
    
    Returns:
        True if connection successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("WEBSOCKET API V3 CONNECTION TEST")
    logger.info("=" * 80)
    
    # Check API credentials
    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("API_SECRET")
    
    if not api_key:
        logger.error("❌ BINANCE_API_KEY not set in environment")
        return False
    
    if not api_secret:
        logger.error("❌ BINANCE_API_SECRET not set in environment")
        return False
    
    logger.info(f"✅ API Key found (length: {len(api_key)})")
    logger.info(f"✅ API Secret found (length: {len(api_secret)})")
    
    try:
        # Initialize config
        logger.info("\n[1/4] Initializing config...")
        config = Config()
        logger.info("✅ Config initialized")
        
        # Initialize shared state
        logger.info("\n[2/4] Initializing shared state...")
        shared_state = SharedState(config)
        logger.info("✅ SharedState initialized")
        
        # Initialize exchange client
        logger.info("\n[3/4] Initializing ExchangeClient...")
        exchange_client = ExchangeClient(
            config=config,
            shared_state=shared_state,
            api_key=api_key,
            api_secret=api_secret,
            testnet=False
        )
        logger.info("✅ ExchangeClient initialized")
        
        # Start exchange client
        logger.info("\n[4/4] Starting user-data WebSocket stream...")
        logger.info("⏳ Waiting 20 seconds for connection and authentication...")
        
        await exchange_client.start()
        await asyncio.sleep(20)
        
        # Check connection status
        logger.info("\n" + "=" * 80)
        logger.info("CONNECTION STATUS CHECK")
        logger.info("=" * 80)
        
        ws_health = exchange_client.get_ws_health_snapshot()
        
        logger.info(f"\n📊 WebSocket Health Snapshot:")
        logger.info(f"  - Connected: {ws_health.get('ws_connected')}")
        logger.info(f"  - User-data stream enabled: {ws_health.get('user_data_stream_enabled')}")
        logger.info(f"  - Auth mode: {ws_health.get('user_data_ws_auth_mode')}")
        logger.info(f"  - Subscription ID: {ws_health.get('user_data_subscription_id')}")
        logger.info(f"  - Reconnect count: {ws_health.get('ws_reconnect_count')}")
        logger.info(f"  - Last user data event: {ws_health.get('user_data_gap_sec'):.1f}s ago")
        
        # Determine success - either WS API v3 (with subscription) or polling mode
        is_connected = ws_health.get('ws_connected', False)
        auth_mode = ws_health.get('user_data_ws_auth_mode', 'none')
        has_subscription = ws_health.get('user_data_subscription_id') is not None
        is_polling = auth_mode == 'polling'
        gap_sec = ws_health.get('user_data_gap_sec', float('inf'))
        
        # Success criteria:
        # - Connected AND
        # - (Has subscription ID OR is in polling mode) AND
        # - Received recent data (< 10 seconds ago)
        is_receiving_data = gap_sec >= 0 and gap_sec < 10
        success = is_connected and (has_subscription or is_polling) and is_receiving_data
        
        logger.info("\n" + "=" * 80)
        if success:
            logger.info("✅ SUCCESS: User-data stream established!")
            logger.info(f"   Auth Mode: {auth_mode}")
            if has_subscription:
                logger.info(f"   Subscription ID: {ws_health.get('user_data_subscription_id')}")
            elif is_polling:
                logger.info(f"   Polling Mode: Enabled (fallback for accounts without WS support)")
            logger.info(f"   Data Gap: {gap_sec:.1f}s")
            logger.info("=" * 80)
            
            # Keep listening for a bit longer
            logger.info("\n⏳ Listening for user-data events (30 seconds)...")
            for i in range(30):
                await asyncio.sleep(1)
                ws_health = exchange_client.get_ws_health_snapshot()
                gap = ws_health.get('user_data_gap_sec', -1)
                if gap >= 0 and gap < 5:
                    logger.info(f"   ✅ Received user-data event {i}s ago")
                    break
            
            return True
        else:
            logger.error("❌ FAILED: Connection established but no active data stream")
            if not is_connected:
                logger.error("    - WebSocket not connected")
            elif not (has_subscription or is_polling):
                logger.error("    - No subscription ID and not in polling mode")
            elif not is_receiving_data:
                logger.error(f"    - No recent data received (gap: {gap_sec:.1f}s)")
            if not is_connected:
                logger.error("   - WebSocket not connected")
            if not has_subscription:
                logger.error("   - No subscription ID received")
            logger.info("=" * 80)
            return False
            
    except Exception as e:
        logger.exception(f"❌ ERROR: {e}")
        logger.info("=" * 80)
        return False
    
    finally:
        logger.info("\n[Cleanup] Stopping ExchangeClient...")
        try:
            await exchange_client.stop()
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        logger.info("✅ Cleanup complete")


async def main():
    """Main test runner."""
    try:
        success = await test_ws_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
