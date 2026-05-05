#!/usr/bin/env python3
"""
🔍 DIAGNOSTIC SCRIPT: Why isn't auto-liquidation working?

This script checks all the gates that block auto-liquidation.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def diagnose():
    print("\n" + "=" * 80)
    print("🔍 AUTO-LIQUIDATION DIAGNOSTIC")
    print("=" * 80)

    # Import after path is set
    from src.l0_core.config import CoreConfig
    from src.l0_core.exchange_client import ExchangeClient
    from src.l0_core.shared_state import SharedState
    from src.l3_portfolio.portfolio_buckets import PortfolioBucketState

    try:
        # Step 1: Get current portfolio
        print("\n[1/5] Fetching current portfolio...")
        client = ExchangeClient()
        balances = await client.get_spot_balances()

        usdt_free = float((balances.get("USDT") or {}).get("free", 0))
        usdt_locked = float((balances.get("USDT") or {}).get("locked", 0))

        print(f"  • USDT Free: ${usdt_free:.2f}")
        print(f"  • USDT Locked: ${usdt_locked:.2f}")
        print(
            f"  • Total positions: {len([b for b, v in balances.items() if b != 'USDT' and float(v.get('free', 0)) > 0])}"
        )

        # Step 2: Calculate total NAV
        print("\n[2/5] Calculating portfolio NAV...")
        config = CoreConfig()
        shared_state = SharedState(config=config, exchange_client=client)

        positions = {}
        total_value = 0
        dust_count = 0
        dust_value = 0

        # Populate with real data
        for symbol, balance in balances.items():
            if symbol == "USDT":
                continue
            qty = float(balance.get("free", 0))
            if qty <= 0:
                continue

            price = await shared_state.safe_price(f"{symbol}USDT", default=0.0)
            value = qty * price
            positions[symbol] = {"qty": qty, "price": price, "value": value}
            total_value += value

            if value < 25:
                dust_count += 1
                dust_value += value

        print(f"  • Total invested: ${total_value:.2f}")
        print(f"  • Dust positions (< $25): {dust_count}")
        print(f"  • Dust value: ${dust_value:.2f}")
        print(f"  • Total NAV: ${usdt_free + usdt_locked + total_value:.2f}")

        # Step 3: Check healing thresholds
        print("\n[3/5] Checking healing thresholds...")

        # Get adaptive thresholds
        total_equity = usdt_free + usdt_locked + total_value
        thresholds = PortfolioBucketState.get_adaptive_thresholds(total_equity)

        min_dead_to_heal = thresholds.get("min_dead_to_heal", 50)
        operating_cash_danger_zone = 10.0 * 1.2  # Default

        print(f"  • Total equity: ${total_equity:.2f}")
        print(f"  • Min dead to heal threshold: ${min_dead_to_heal:.2f}")
        print(f"  • Operating cash danger zone: ${operating_cash_danger_zone:.2f}")
        print(f"  • Current free USDT: ${usdt_free:.2f}")

        # Step 4: Check healing gates
        print("\n[4/5] Checking healing gate conditions...")

        gate1 = dust_value > min_dead_to_heal
        gate2 = usdt_free < operating_cash_danger_zone

        print("  GATE 1: dust_value > min_dead_to_heal?")
        print(f"          ${dust_value:.2f} > ${min_dead_to_heal:.2f}? {gate1}")
        print(f"          Status: {'✅ PASS' if gate1 else '❌ FAIL'}")

        print("\n  GATE 2: operating_cash < danger_zone?")
        print(f"          ${usdt_free:.2f} < ${operating_cash_danger_zone:.2f}? {gate2}")
        print(f"          Status: {'✅ PASS' if gate2 else '❌ FAIL'}")

        # Step 5: Recommendations
        print("\n[5/5] Recommendations...")

        if gate1 or gate2:
            print("  ✅ HEALING SHOULD TRIGGER - Auto-liquidation mechanism should be active")
        else:
            print("  ❌ HEALING IS BLOCKED - Need to lower thresholds or increase dust")
            print("\n  🔧 To enable healing immediately:")
            print("     export DEAD_CAPITAL_MIN_THRESHOLD=5.0")
            print("     export HEAL_C_WARMUP_SEC=5")
            print("     export HEAL_DUST_SWEEP_INTERVAL_SEC=60")
            print("     pkill -9 -f MASTER_SYSTEM_ORCHESTRATOR")
            print("     nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/bot.log 2>&1 &")

        # Additional insights
        print("\n" + "=" * 80)
        print("📊 PORTFOLIO SNAPSHOT")
        print("=" * 80)

        if positions:
            print(f"\n💾 Held positions ({len(positions)} total):")
            for sym, pos in sorted(positions.items(), key=lambda x: x[1]["value"], reverse=True)[
                :10
            ]:
                status = "🔴 DUST" if pos["value"] < 25 else "🟢 PRODUCTIVE"
                print(
                    f"  {status}  {sym:8s} qty={pos['qty']:.8f} price=${pos['price']:>10.2f} value=${pos['value']:>8.2f}"
                )
            if len(positions) > 10:
                print(f"  ... and {len(positions) - 10} more positions")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    result = asyncio.run(diagnose())
    sys.exit(0 if result else 1)
