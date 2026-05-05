#!/usr/bin/env python3
"""
Phase 1 Diagnostic - Check current Binance balance and dust positions
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

def main():
    try:
        from binance.client import Client

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET_HMAC")

        if not api_key or not api_secret:
            print("❌ ERROR: BINANCE_API_KEY or BINANCE_API_SECRET_HMAC not set in .env")
            return False

        print("\n" + "="*80)
        print("PHASE 1 DIAGNOSTIC - Binance Balance Check")
        print("="*80)

        # Initialize Binance client
        client = Client(api_key, api_secret)

        try:
            # Get account info
            print("\n[1] Fetching account information...")
            account = client.get_account()

            balances = account.get('balances', [])

            # Find USDT and positions
            usdt_free = 0.0
            usdt_locked = 0.0
            dust_positions = []

            for bal in balances:
                symbol = bal['asset']
                free = float(bal['free'])
                locked = float(bal['locked'])

                if symbol == 'USDT':
                    usdt_free = free
                    usdt_locked = locked
                elif free > 0:
                    dust_positions.append({
                        'asset': symbol,
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    })

            # Report
            print(f"\n✅ Account Summary:")
            print(f"   USDT Free:     ${usdt_free:.2f}")
            print(f"   USDT Locked:   ${usdt_locked:.2f}")
            print(f"   Total USDT:    ${usdt_free + usdt_locked:.2f}")

            if dust_positions:
                print(f"\n🔴 Found {len(dust_positions)} non-USDT positions:")
                total_locked_value = 0.0

                # Get ticker prices
                print("   (Fetching current prices...)")

                for pos in dust_positions:
                    symbol = pos['asset']
                    try:
                        # Get price
                        ticker_data = client.get_symbol_ticker(symbol=f"{symbol}USDT")
                        price = float(ticker_data['price'])
                        value = (pos['free'] + pos['locked']) * price
                        total_locked_value += value

                        print(f"\n   {symbol}:")
                        print(f"      Free:      {pos['free']:.8f}")
                        print(f"      Locked:    {pos['locked']:.8f}")
                        print(f"      Price:     ${price:.2f}")
                        print(f"      Value:     ${value:.2f}")
                    except Exception as e:
                        print(f"\n   {symbol}: (error getting price: {e})")

                print(f"\n   📊 Total dust value: ${total_locked_value:.2f}")
                print(f"   Capital locked %: {(total_locked_value / (usdt_free + usdt_locked + total_locked_value) * 100):.1f}%")
            else:
                print(f"\n✅ No dust positions - portfolio is clean!")

            # Check capital floor
            capital_floor = 10.0
            if usdt_free < capital_floor:
                print(f"\n🔴 CAPITAL FLOOR VIOLATED:")
                print(f"   Free USDT: ${usdt_free:.2f}")
                print(f"   Required:  ${capital_floor:.2f}")
                print(f"   Shortfall: ${capital_floor - usdt_free:.2f}")
                print(f"\n   ⚠️  System cannot trade until free USDT >= ${capital_floor:.2f}")
            else:
                print(f"\n✅ Capital floor OK (${usdt_free:.2f} >= ${capital_floor:.2f})")

            print("\n" + "="*80)

            # Recommendation
            if dust_positions and usdt_free < capital_floor:
                print("\n💡 RECOMMENDATION:")
                print(f"   Liquidate {len(dust_positions)} dust positions to free capital")
                print(f"   Estimated recovery: ${total_locked_value:.2f}")
                if total_locked_value + usdt_free >= capital_floor:
                    print(f"   After liquidation: ${total_locked_value + usdt_free:.2f} available (✅ above floor)")
                print("\n   To proceed: python3 phase1_liquidate.py")

            return True

        except Exception as e:
            print(f"❌ Error fetching account: {e}")
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
