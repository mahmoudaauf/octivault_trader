#!/usr/bin/env python3
"""
Phase 1 Dust Conversion - Convert all dust to BNB using Binance dust conversion API
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

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET_HMAC")

        if not api_key or not api_secret:
            print("❌ ERROR: BINANCE_API_KEY or BINANCE_API_SECRET_HMAC not set in .env")
            return False

        print("\n" + "="*80)
        print("PHASE 1 DUST CONVERSION - Convert Dust to BNB")
        print("="*80)

        # Initialize Binance client
        client = Client(api_key, api_secret)

        try:
            # Step 1: Get list of dust assets
            print("\n[1] Finding dust assets...")
            try:
                dust_assets = client.get_dust_assets()
                dust_list = dust_assets.get('details', [])

                if not dust_list:
                    print("✅ No dust assets found - portfolio is clean!")
                    return True

                print(f"✅ Found {len(dust_list)} dust assets:")
                for asset in dust_list:
                    asset_name = asset['asset']
                    freetx = float(asset.get('free', 0))
                    value = float(asset.get('convertedBtc', 0))
                    print(f"   • {asset_name}: {freetx:.8f} (≈ {value:.8f} BTC)")

            except BinanceAPIException as e:
                print(f"⚠️  Could not get dust assets: {e.message}")
                # Continue anyway and try to convert all small balances
                dust_list = []

            # Step 2: Get account balances to find small holdings
            print("\n[2] Fetching account balances...")
            account = client.get_account()
            balances = account.get('balances', [])

            # Identify convertible dust
            dust_to_convert = []
            dust_threshold = 10.0  # $10 or less

            for bal in balances:
                symbol = bal['asset']
                free = float(bal['free'])

                if symbol == 'USDT' or symbol == 'BNB' or free <= 0:
                    continue

                # Get price to estimate value
                try:
                    ticker_data = client.get_symbol_ticker(symbol=f"{symbol}USDT")
                    price = float(ticker_data['price'])
                    value = free * price

                    if value < dust_threshold:
                        dust_to_convert.append({
                            'asset': symbol,
                            'qty': free,
                            'price': price,
                            'value': value,
                        })
                except Exception:
                    # If can't get price, assume it's dust
                    dust_to_convert.append({
                        'asset': symbol,
                        'qty': free,
                        'price': 0.0,
                        'value': 0.0,
                    })

            if not dust_to_convert:
                print("✅ No dust found (all positions > $10)")
                return True

            print(f"✅ Found {len(dust_to_convert)} positions < $10 to convert:")
            total_dust_value = 0.0
            for dust in dust_to_convert:
                print(f"   • {dust['asset']}: {dust['qty']:.8f} (${dust['value']:.2f})")
                total_dust_value += dust['value']

            print(f"\n   Total dust value: ${total_dust_value:.2f}")

            # Step 3: Confirm before proceeding
            print("\n⚠️  ABOUT TO EXECUTE DUST CONVERSION")
            response = input("\nType 'YES' to confirm conversion: ").strip().upper()
            if response != "YES":
                print("❌ Conversion cancelled.")
                return False

            # Step 4: Convert dust
            print("\n[3] Converting dust to BNB...")
            print("="*80)

            # Prepare asset list for conversion
            assets_to_convert = [d['asset'] for d in dust_to_convert]

            try:
                print(f"\nAttempting to convert {len(assets_to_convert)} assets to BNB...")
                result = client.transfer_dust(assetsBTC=assets_to_convert)

                print(f"✅ Dust conversion executed!")
                print(f"   Transfer ID: {result.get('transferId')}")
                print(f"   Result: {result.get('totalTransferedBtc')}")

                successful = True

            except BinanceAPIException as e:
                print(f"⚠️  Dust conversion failed: {e.message}")
                print(f"   Code: {e.code}")
                print("\n   Note: Dust conversion may fail if:")
                print("   - Assets are not actually dust (below minimum)")
                print("   - Conversion already happened recently")
                print("   - API restrictions apply")
                successful = False

            # Step 5: Wait for settlement
            if successful:
                print("\n[4] Waiting for conversion to settle (30 seconds)...")
                time.sleep(30)

            # Step 6: Check updated balance
            print("\n[5] Checking updated balance...")
            account_updated = client.get_account()
            balances_updated = account_updated.get('balances', [])

            usdt_balance = next(
                (float(b['free']) for b in balances_updated if b['asset'] == 'USDT'),
                0.0
            )
            bnb_balance = next(
                (float(b['free']) for b in balances_updated if b['asset'] == 'BNB'),
                0.0
            )

            print(f"    Free USDT:  ${usdt_balance:.2f}")
            print(f"    Free BNB:   {bnb_balance:.8f}")

            # Step 7: If we got BNB, convert to USDT
            if bnb_balance > 0:
                print("\n[6] Converting BNB to USDT...")
                try:
                    ticker_data = client.get_symbol_ticker(symbol="BNBUSDT")
                    bnb_price = float(ticker_data['price'])
                    bnb_value = bnb_balance * bnb_price

                    print(f"    BNB Price: ${bnb_price:.2f}")
                    print(f"    BNB Value: ${bnb_value:.2f}")
                    print(f"    Estimated USDT after sale: ${usdt_balance + bnb_value:.2f}")
                    print("\n    💡 To convert BNB to USDT, visit Binance web interface:")
                    print("       Trade > BNB > USDT > Market Sell")

                except Exception as e:
                    print(f"    Could not get BNB price: {e}")

            # Final check
            capital_floor = 10.0
            if usdt_balance >= capital_floor:
                print(f"\n✅ SUCCESS: Capital floor met! (${usdt_balance:.2f} >= ${capital_floor:.2f})")
                print("\n   Next steps:")
                print("   1. Restart the system: python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py")
                print("   2. System should resume trading normally")
                return True
            else:
                print(f"\n⚠️  Capital floor still not met: ${usdt_balance:.2f} < ${capital_floor:.2f}")
                if bnb_balance > 0:
                    print(f"\n   💡 TIP: You have {bnb_balance:.8f} BNB that can be converted to USDT")
                    print(f"      This would give you ${usdt_balance + (bnb_balance * bnb_price):.2f} total")
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
