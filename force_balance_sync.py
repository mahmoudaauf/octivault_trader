#!/usr/bin/env python3
"""Force balance sync and break the deadlock"""
import asyncio
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def force_balance_sync():
    """Force a complete balance reload from Binance"""
    try:
        from src.l0_core.shared_state import SharedState
        from src.l1_exchange.exchange_client import ExchangeClient
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Initialize exchange client
        exchange = ExchangeClient()
        await exchange.ensure_connection()
        
        # Get account balance from Binance
        account = await exchange.get_account()
        if account:
            print("✅ Binance Account Retrieved:")
            print(f"   Free USDT: ${account.get('free_usdt', 0):.2f}")
            print(f"   Locked USDT: ${account.get('locked_usdt', 0):.2f}")
            print(f"   Total USDT: ${account.get('total_usdt', 0):.2f}")
            
            # Now sync to SharedState
            print("\n🔄 Syncing to SharedState...")
            ss = SharedState()
            await ss.sync_authoritative_balance(force=True)
            
            # Get updated NAV
            nav = ss.get_nav() or 0.0
            print(f"✅ NAV After Sync: ${nav:.2f}")
            
            if nav > 0:
                print("\n✅ SUCCESS: Balance synced! System should be unblocked.")
                print("   You may need to restart the orchestrator or wait for next eval_cycle.")
            else:
                print("\n⚠️  WARNING: NAV still 0 after sync")
                print("   Check if there are active positions or if balance is truly empty")
        else:
            print("❌ Failed to get account info from Binance")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(force_balance_sync())
