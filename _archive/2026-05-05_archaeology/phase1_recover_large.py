#!/usr/bin/env python3
"""
Phase 1 Recover Large Positions - Sell positions with sufficient size to meet capital floor
"""
import sys
import time
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

def main():
    try:
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        from decimal import Decimal, ROUND_DOWN

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET_HMAC")

        if not api_key or not api_secret:
            print("❌ ERROR: BINANCE_API_KEY or BINANCE_API_SECRET_HMAC not set in .env")
            return False

        print("\n" + "="*80)
        print("PHASE 1 RECOVER LARGE POSITIONS - Sell Sizeable Assets")
        print("="*80)

        # Initialize Binance client
        client = Client(api_key, api_secret)

        try:
            # Get account info
            print("\n[1] Fetching account and position details...")
            account = client.get_account()
            balances = account.get('balances', [])

            # Find positions with size > $1 (large enough to sell)
            sellable_positions = []
            for bal in balances:
                symbol = bal['asset']
                free = float(bal['free'])

                if symbol == 'USDT' or free <= 0:
                    continue

                # Get current price and minimum notional
                try:
                    # Get symbol info to find step size and minimum notional
                    exchange_info = client.get_symbol_info(f"{symbol}USDT")
                    if not exchange_info:
                        continue

                    # Get price
                    ticker_data = client.get_symbol_ticker(symbol=f"{symbol}USDT")
                    price = float(ticker_data['price'])
                    value = free * price

                    # Find lot size filter
                    lot_size_filter = next(
                        (f for f in exchange_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'),
                        None
                    )
                    step_size = float(lot_size_filter['stepSize']) if lot_size_filter else 1e-8

                    # Check if we can sell this
                    if value >= 1.0:  # At least $1
                        # Calculate valid quantity (respecting step size)
                        decimal_qty = Decimal(str(free))
                        decimal_step = Decimal(str(step_size))

                        # Round down to nearest step size
                        steps = int(decimal_qty / decimal_step)
                        valid_qty = float(steps * decimal_step)

                        if valid_qty > 0:
                            sellable_positions.append({
                                'asset': symbol,
                                'qty': free,
                                'valid_qty': valid_qty,
                                'price': price,
                                'value': value,
                                'step_size': step_size,
                            })

                except Exception as e:
                    # Skip this symbol
                    pass

            # Get current USDT
            usdt_balance = next(
                (float(b['free']) for b in balances if b['asset'] == 'USDT'),
                0.0
            )
            capital_floor = 10.0
            shortfall = max(0, capital_floor - usdt_balance)

            print(f"\n✅ Current USDT: ${usdt_balance:.2f}")
            print(f"   Capital floor: ${capital_floor:.2f}")
            print(f"   Shortfall: ${shortfall:.2f}")

            if shortfall == 0:
                print("\n✅ Capital floor already met!")
                return True

            if not sellable_positions:
                print(f"\n❌ No positions large enough to sell (need > $1)")
                print(f"   Found {len(balances)} total positions, all too small")
                return False

            # Sort by value descending
            sellable_positions.sort(key=lambda x: x['value'], reverse=True)

            print(f"\n✅ Found {len(sellable_positions)} positions sellable (> $1):")
            needed = shortfall + 2.0  # Get a bit more than floor to be safe
            accumulated = 0.0

            for i, pos in enumerate(sellable_positions, 1):
                needed_for_this = needed - accumulated
                print(f"\n   [{i}] {pos['asset']}")
                print(f"       Qty: {pos['qty']:.8f}")
                print(f"       Price: ${pos['price']:.4f}")
                print(f"       Value: ${pos['value']:.2f}")
                print(f"       Sellable qty: {pos['valid_qty']:.8f} (step: {pos['step_size']:.8f})")
                if accumulated < needed:
                    print(f"       → Needed to reach floor: ${needed_for_this:.2f} ✓")
                    accumulated += pos['value']

            # Confirm before proceeding
            print("\n⚠️  ABOUT TO EXECUTE LIVE POSITION SALES")
            print(f"\nThis will sell enough positions to reach ${capital_floor:.2f} USDT")
            response = input("\nType 'YES' to confirm: ").strip().upper()
            if response != "YES":
                print("❌ Cancelled.")
                return False

            # Execute sales
            print("\n[2] Executing sales...")
            print("="*80)

            successful = 0
            failed = 0
            total_recovered = 0.0

            for pos in sellable_positions:
                # Stop if we've recovered enough
                if usdt_balance + total_recovered >= capital_floor:
                    print(f"\n✅ Capital floor reached! Stopping sales.")
                    print(f"   Current + recovered: ${usdt_balance + total_recovered:.2f}")
                    break

                symbol = pos['asset']
                sell_qty = pos['valid_qty']

                print(f"\n[{successful + failed + 1}] Selling {symbol}...")
                print(f"    Qty: {sell_qty:.8f}")
                print(f"    Expected value: ${sell_qty * pos['price']:.2f}")

                try:
                    # Place market sell
                    order = client.order_market_sell(
                        symbol=f"{symbol}USDT",
                        quantity=sell_qty,
                    )

                    # Get actual fills
                    fills = order.get('fills', [])
                    actual_value = sum(
                        float(f['qty']) * float(f['price']) for f in fills
                    )

                    print(f"    ✅ Sold! (Order ID: {order['orderId']})")
                    print(f"    Actual proceeds: ${actual_value:.2f}")

                    successful += 1
                    total_recovered += actual_value

                    # Small delay to avoid rate limits
                    time.sleep(0.5)

                except BinanceAPIException as api_err:
                    print(f"    ❌ API Error: {api_err.message}")
                    failed += 1

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    failed += 1

            # Summary
            print("\n" + "="*80)
            print("SALES SUMMARY")
            print("="*80)
            print(f"Successful sales: {successful}")
            print(f"Failed sales:     {failed}")
            print(f"Total recovered:  ${total_recovered:.2f}")
            print(f"Previous USDT:    ${usdt_balance:.2f}")
            print(f"Projected total:  ${usdt_balance + total_recovered:.2f}")

            # Wait for fills
            if successful > 0:
                print("\n[3] Waiting for orders to settle (30 seconds)...")
                time.sleep(30)

                # Check updated balance
                print("\n[4] Verifying final balance...")
                account_updated = client.get_account()
                balances_updated = account_updated.get('balances', [])

                new_usdt = next(
                    (float(b['free']) for b in balances_updated if b['asset'] == 'USDT'),
                    0.0
                )

                print(f"    Final USDT: ${new_usdt:.2f}")

                if new_usdt >= capital_floor:
                    print(f"\n✅ SUCCESS: Capital floor met! (${new_usdt:.2f} >= ${capital_floor:.2f})")
                    print("\n   Next steps:")
                    print("   1. Restart the system: python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py")
                    print("   2. System should resume trading normally")
                    return True
                else:
                    print(f"\n⚠️  Capital floor still not met: ${new_usdt:.2f} < ${capital_floor:.2f}")
                    print(f"   Still need: ${capital_floor - new_usdt:.2f}")
                    return False
            else:
                print(f"\n❌ No positions were successfully sold")
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
