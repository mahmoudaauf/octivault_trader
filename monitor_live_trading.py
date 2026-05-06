#!/usr/bin/env python
"""
Real-time monitoring script for live Binance trading.

Shows:
- Current balance & NAV
- Active orders on Binance
- Detected fills
- Position updates
- P&L changes
- Cycle metrics
"""

import asyncio
import sys
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from core_engine.native.bootstrap import BootstrapConfig, build_components
from core_engine.native.app_context import build_native_app_ctx

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-6s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def monitor_live_trading():
    """Monitor live trading on real Binance account."""

    try:
        logger.info("=" * 80)
        logger.info("OCTIVAULT NATIVE STACK — LIVE TRADING MONITOR")
        logger.info("=" * 80)

        # Load config
        logger.info("📋 Loading configuration from .env...")
        cfg = BootstrapConfig.from_env()
        logger.info(f"✅ Config loaded:")
        logger.info(f"   Mode: {'TESTNET' if cfg.testnet else 'LIVE (REAL MONEY)'}")
        logger.info(f"   Symbols: {', '.join(cfg.symbols[:5])}")
        logger.info(f"   Market Poll: {cfg.market_data_poll_sec}s")
        logger.info(f"   Balance Poll: {cfg.balance_poll_sec}s")
        logger.info(f"   Fill Track Poll: {cfg.fill_tracker_poll_sec}s")

        # Build components
        logger.info("\n🔧 Building native stack components...")
        components = await build_components(cfg)
        logger.info("✅ Components built")

        # Wire app context
        logger.info("🔗 Wiring app context...")
        app_ctx, orch = build_native_app_ctx(components)
        logger.info("✅ App context wired")

        # Verify key components
        exchange_client = components.exchange_client
        shared_state = components.shared_state
        fill_tracker = components.fill_tracker
        balance_sync = components.balance_sync
        market_data = components.market_data
        logger.info(f"✅ Key components available:")
        logger.info(f"   ExchangeClient: {exchange_client is not None}")
        logger.info(f"   SharedState: {shared_state is not None}")
        logger.info(f"   FillTracker: {fill_tracker is not None}")
        logger.info(f"   BalanceSync: {balance_sync is not None}")
        logger.info(f"   MarketData: {market_data is not None}")

        # Start orchestrator
        logger.info("\n🚀 Starting orchestrator...")
        await orch.start()
        logger.info("✅ Orchestrator started")
        logger.info("   - Market data polling active")
        logger.info("   - Balance sync polling active")
        logger.info("   - Fill tracker polling active")

        # Check initial balance
        logger.info("\n💰 INITIAL STATE")
        balance = balance_sync.get_balance()
        nav = shared_state.get_nav()
        logger.info(f"   NAV: ${nav:.2f}")
        logger.info(f"   Balance: {balance}")

        # Get exchange info
        logger.info("\n📊 EXCHANGE STATUS")
        try:
            server_time = await exchange_client.get_server_time()
            logger.info(f"   Server Time: {datetime.fromtimestamp(server_time/1000)}")
            account_info = await exchange_client.get_account()
            logger.info(f"   Account: {account_info.get('accountType', 'unknown')}")
            logger.info(f"   ✅ Exchange connection verified")
        except Exception as e:
            logger.error(f"   ❌ Exchange error: {e}")

        # Run trading cycles
        logger.info("\n" + "=" * 80)
        logger.info("STARTING LIVE TRADING CYCLES")
        logger.info("=" * 80)

        cycle_count = 0
        try:
            while cycle_count < 5:  # Run 5 cycles for monitoring
                cycle_count += 1
                logger.info(f"\n{'─' * 80}")
                logger.info(f"CYCLE {cycle_count}")
                logger.info(f"{'─' * 80}")

                # Run one cycle
                metrics = await orch.run_cycle()

                # Log cycle results
                logger.info(f"\n📈 Cycle Metrics:")
                logger.info(f"   Duration: {metrics.duration_ms:.0f}ms")
                logger.info(f"   NAV: ${metrics.nav:.2f}")
                logger.info(f"   Signals: {metrics.signals_count}")
                logger.info(f"   Decisions: {metrics.decisions_count}")
                logger.info(f"   Executions: {metrics.executions_count}")
                logger.info(f"   Success: {metrics.execution_successes}/{metrics.executions_count}")
                logger.info(f"   Errors: {len(metrics.errors)}")

                if metrics.errors:
                    for err in metrics.errors:
                        logger.warning(f"      - {err}")

                # Check positions
                positions = shared_state.get_all_positions()
                if positions:
                    logger.info(f"\n📍 Positions ({len(positions)} open):")
                    for sym, pos in positions.items():
                        logger.info(f"      {sym}:")
                        logger.info(f"         Qty: {pos.qty:.8f}")
                        logger.info(f"         Entry: ${pos.entry_price:,.2f}")
                        logger.info(f"         Value: ${pos.position_value:,.2f}")
                        logger.info(f"         P&L: {pos.unrealized_pnl_pct:.2f}%")
                else:
                    logger.info(f"\n📍 No open positions")

                # Check orders
                try:
                    open_orders = await exchange_client.get_open_orders()
                    if open_orders:
                        logger.info(f"\n📋 Open Orders ({len(open_orders)}):")
                        for order in open_orders[:5]:
                            logger.info(f"      {order.get('symbol')}: {order.get('side')} {order.get('origQty')} @ ${order.get('price')}")
                    else:
                        logger.info(f"\n📋 No open orders")
                except Exception as e:
                    logger.debug(f"   (Could not fetch orders: {e})")

                # Check recent trades
                try:
                    recent_trades = []
                    for sym in cfg.symbols[:3]:  # Check first 3 symbols
                        trades = await exchange_client.get_account_trades(sym, limit=3)
                        recent_trades.extend(trades)

                    if recent_trades:
                        logger.info(f"\n🎯 Recent Fills ({len(recent_trades)} in last 3):")
                        for trade in recent_trades[:5]:
                            logger.info(f"      {trade.get('symbol')}: {trade.get('side')} {trade.get('qty')} @ ${trade.get('price')}")
                    else:
                        logger.info(f"\n🎯 No recent fills")
                except Exception as e:
                    logger.debug(f"   (Could not fetch trades: {e})")

                # Sleep before next cycle
                if cycle_count < 5:
                    logger.info(f"\n⏳ Waiting for next cycle...")
                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Interrupted by user")

        # Shutdown
        logger.info("\n" + "=" * 80)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 80)
        await orch.stop()
        logger.info("✅ Orchestrator stopped")

        # Final summary
        logger.info("\n📊 FINAL STATE")
        final_balance = balance_sync.get_balance()
        final_nav = shared_state.get_nav()
        final_positions = shared_state.get_all_positions()

        logger.info(f"   NAV: ${final_nav:.2f}")
        logger.info(f"   Balance: {final_balance}")
        logger.info(f"   Positions: {len(final_positions)} open")

        total_pnl = 0.0
        for sym, pos in final_positions.items():
            pnl = (shared_state.get_price(sym) - pos.entry_price) * pos.qty
            total_pnl += pnl

        if total_pnl != 0:
            logger.info(f"   Unrealized P&L: ${total_pnl:.2f}")

        logger.info("\n✅ MONITORING COMPLETE")
        return True

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(monitor_live_trading())
    sys.exit(0 if success else 1)
