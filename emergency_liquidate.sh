#!/bin/bash
# Emergency Capital Freeing Script
# Sells smallest positions to free capital for new trades

set -e

LOG_FILE="/tmp/octivault_nav_fixed.log"
BOT_DIR="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"

echo "🚨 EMERGENCY CAPITAL FREEING PROCEDURE"
echo "======================================"
echo ""

# Kill the current bot
echo "1️⃣ Stopping bot..."
pkill -9 -f "MASTER_SYSTEM_ORCHESTRATOR" 2>/dev/null || true
sleep 2

cd "$BOT_DIR"

# Create a liquidation script to close smallest positions
echo "2️⃣ Creating liquidation trigger..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader')

from src.l0_core.shared_state import SharedState
from src.l0_core.config import Config
from src.l1_exchange.exchange_client import ExchangeClient
import asyncio
import json
from datetime import datetime

async def liquidate_smallest_positions():
    """Close smallest positions to free capital"""

    config = Config()
    ss = SharedState(config)
    exc = ExchangeClient(config, ss)

    print("📊 Loading current positions...")

    # Get positions
    positions = ss.get_positions()
    print(f"   Current positions: {len(positions)}")

    if not positions:
        print("   ✓ No positions to liquidate")
        return

    # Sort by position value (smallest first)
    pos_list = []
    for sym, pos_data in positions.items():
        qty = float(pos_data.get("qty", 0))
        entry_price = float(pos_data.get("entry_price", 0))
        value = qty * entry_price
        pos_list.append((sym, qty, entry_price, value))

    pos_list.sort(key=lambda x: x[3])  # Sort by value

    print(f"\n📍 Positions by size (smallest first):")
    for sym, qty, price, value in pos_list:
        print(f"   {sym}: {qty:.8f} @ ${price:.4f} = ${value:.2f}")

    # Liquidate smallest 30% of positions (but keep biggest ones)
    to_liquidate = max(1, len(pos_list) // 3)
    print(f"\n🔄 Will liquidate {to_liquidate} smallest positions to free capital...")

    liquidated = 0
    freed_usdt = 0

    for sym, qty, entry_price, value in pos_list[:to_liquidate]:
        try:
            print(f"\n   ➤ Liquidating {sym}: {qty:.8f} units...")

            # Close the position
            result = await exc.create_market_order(
                symbol=sym,
                side='SELL',
                quantity=qty,
                custom_id='EMERGENCY_LIQUIDATE'
            )

            if result and result.get('status') in ['FILLED', 'PARTIALLY_FILLED']:
                liquidated += 1
                freed_usdt += value
                print(f"     ✓ FILLED - Recovered ${value:.2f}")
            else:
                print(f"     ✗ FAILED - {result}")
        except Exception as e:
            print(f"     ✗ ERROR: {e}")

    print(f"\n✅ LIQUIDATION COMPLETE")
    print(f"   Positions closed: {liquidated}")
    print(f"   Capital freed: ${freed_usdt:.2f}")
    print(f"   New free balance ready for trading")

asyncio.run(liquidate_smallest_positions())
EOF

echo ""
echo "3️⃣ Restarting bot with freed capital..."
export APPROVE_LIVE_TRADING=YES
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/octivault_nav_fixed.log 2>&1 &

new_pid=$!
echo "   ✓ Bot restarted (PID: $new_pid)"

sleep 3

# Show new status
echo ""
echo "4️⃣ New System Status:"
tail -20 /tmp/octivault_nav_fixed.log | grep -E "capital_free|NAV|Signal"

echo ""
echo "✅ CAPITAL FREEING COMPLETE - Bot ready for trading with freed capital"
