#!/usr/bin/env python3
"""
Phase 1 Liquidation - Sell all 40 dust positions to free capital
"""
import os
import sys
import time

from dotenv import load_dotenv

# Load env
load_dotenv()


def main():
    try:
        from binance.client import Client
        from binance.exceptions import BinanceAPIException

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET_HMAC")

        if not api_key or not api_secret:
            print("❌ ERROR: BINANCE_API_KEY or BINANCE_API_SECRET_HMAC not set in .env")
            return False

        print("\n" + "=" * 80)
        print("PHASE 1 LIQUIDATION - Sell All Dust Positions")
        print("=" * 80)

        # Initialize Binance client
        client = Client(api_key, api_secret)

        try:
            # Get account info
            print("\n[1] Fetching account information...")
            account = client.get_account()

            balances = account.get("balances", [])

            # Find dust positions
            dust_positions = []
            for bal in balances:
                symbol = bal["asset"]
                free = float(bal["free"])

                if symbol == "USDT" or free <= 0:
                    continue

                dust_positions.append(
                    {
                        "asset": symbol,
                        "qty": free,
                    }
                )

            if not dust_positions:
                print("✅ No dust positions found - portfolio is clean!")
                return True

            print(f"\n✅ Found {len(dust_positions)} dust positions to liquidate")

            # Confirm before proceeding
            print("\n⚠️  ABOUT TO EXECUTE LIVE LIQUIDATION")
            response = input("\nType 'YES' to confirm liquidation: ").strip().upper()
            if response != "YES":
                print("❌ Liquidation cancelled.")
                return False

            # Liquidate positions
            print("\n[2] Starting liquidation...")
            print("=" * 80)

            successful = 0
            failed = 0
            total_value = 0.0
            errors = []

            for i, pos in enumerate(dust_positions, 1):
                symbol = pos["asset"]
                qty = pos["qty"]

                try:
                    print(f"\n[{i}/{len(dust_positions)}] Liquidating {symbol}...")
                    print(f"    Qty: {qty:.8f}")

                    # Get current price
                    try:
                        ticker_data = client.get_symbol_ticker(symbol=f"{symbol}USDT")
                        price = float(ticker_data["price"])
                        value = qty * price
                        print(f"    Price: ${price:.2f}")
                        print(f"    Value: ${value:.2f}")
                        total_value += value
                    except Exception as e:
                        print(f"    Warning: Could not get price ({e})")
                        # Continue anyway with 0 price for logging
                        price = 0.0
                        value = 0.0

                    # Place SELL order with minimal safeguards
                    try:
                        # Use MARKET order to guarantee execution
                        order = client.order_market_sell(
                            symbol=f"{symbol}USDT",
                            quantity=qty,
                        )
                        print(f"    ✅ Order submitted (Order ID: {order.get('orderId')})")
                        successful += 1

                        # Small delay to avoid rate limits
                        time.sleep(0.5)

                    except BinanceAPIException as api_err:
                        print(f"    ❌ API Error: {api_err.message}")
                        failed += 1
                        errors.append((symbol, api_err.message))

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    failed += 1
                    errors.append((symbol, str(e)))

            # Summary
            print("\n" + "=" * 80)
            print("LIQUIDATION SUMMARY")
            print("=" * 80)
            print(f"Total positions:   {len(dust_positions)}")
            print(f"Successful:        {successful}")
            print(f"Failed:            {failed}")
            print(f"Total value:       ${total_value:.2f}")

            if errors:
                print(f"\n⚠️  Errors ({len(errors)}):")
                for symbol, error in errors:
                    print(f"  • {symbol}: {error}")

            print("\n[3] Waiting for orders to fill (30 seconds)...")
            time.sleep(30)

            # Check updated balance
            print("\n[4] Checking updated balance...")
            account_updated = client.get_account()
            balances_updated = account_updated.get("balances", [])

            usdt_balance = next(
                (float(b["free"]) for b in balances_updated if b["asset"] == "USDT"), 0.0
            )

            print(f"    Free USDT now: ${usdt_balance:.2f}")

            # Check capital floor
            capital_floor = 10.0
            if usdt_balance >= capital_floor:
                print(
                    f"\n✅ SUCCESS: Capital floor met! (${usdt_balance:.2f} >= ${capital_floor:.2f})"
                )
                print("\n   Next steps:")
                print("   1. Restart the system: python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py")
                print("   2. System should resume trading normally")
                return True
            else:
                print(
                    f"\n⚠️  Capital floor still not met: ${usdt_balance:.2f} < ${capital_floor:.2f}"
                )
                print(f"   Shortfall: ${capital_floor - usdt_balance:.2f}")
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Setup error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
