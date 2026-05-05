#!/usr/bin/env python3
"""
Recovery Script: Restore positions from Binance wallet and populate state file
"""
import sys
import json
import asyncio

sys.path.insert(0, '/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader')

from src.l0_core.config import Config
from src.l0_core.shared_state import SharedState
from src.l1_exchange.exchange_client import ExchangeClient


async def restore_positions_from_wallet():
    """Fetch current wallet state and restore positions"""
    
    print("\n" + "="*60)
    print("🔄 POSITION RESTORATION FROM BINANCE WALLET")
    print("="*60 + "\n")
    
    config = Config()
    shared_state = SharedState(config)
    exchange_client = ExchangeClient(config, shared_state)
    
    try:
        # Fetch account balance
        print("📊 Fetching account balance from Binance...")
        account = await exchange_client.get_account_balances()
        
        if not account:
            print("❌ Failed to fetch account data")
            return
        
        # Build positions dict from non-zero balances
        positions = {}
        total_nav = 0.0
        invested = 0.0
        free_usdt = 0.0
        
        print("\n💼 Active Holdings:")
        for asset, balance in account.items():
            free_qty = float(balance.get('free', 0))
            locked_qty = float(balance.get('locked', 0))
            
            if free_qty == 0 and locked_qty == 0:
                continue
            
            total_qty = free_qty + locked_qty
            
            # Special handling for USDT (base currency)
            if asset == 'USDT':
                free_usdt = free_qty
                print(f"   💵 USDT: {free_qty:.2f} (free)")
                if locked_qty > 0:
                    print(f"         : {locked_qty:.2f} (locked in orders)")
                continue
            
            # For other assets, create position records
            symbol = f"{asset}USDT"
            
            # Try to get current price
            price = 0.0
            try:
                ticker = await exchange_client._get_ticker(symbol)
                if ticker:
                    price = float(ticker.get('lastPrice', 0))
            except:
                pass
            
            position_value = total_qty * price if price > 0 else 0.0
            total_nav += position_value
            if free_qty > 0:
                invested += position_value
            
            positions[symbol] = {
                'qty': total_qty,
                'free': free_qty,
                'locked': locked_qty,
                'entry_price': price if price > 0 else 0.0,
                'current_price': price,
                'value': position_value,
                'status': 'open'
            }
            
            print(f"   🪙 {asset}: {total_qty:.8f} units @ ${price:.4f} = ${position_value:.2f}")
            if locked_qty > 0:
                print(f"         : ({free_qty:.8f} free, {locked_qty:.8f} locked)")
        
        total_nav += free_usdt
        
        # Save to state file
        state = {
            'positions': positions,
            'nav': total_nav,
            'portfolio_nav': total_nav,
            'total_equity': total_nav,
            'total_equity_usdt': total_nav,
            'free_quote': free_usdt,
            'invested_capital': invested,
            'timestamp': asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0,
            'note': 'State restored from Binance wallet'
        }
        
        state_file = '/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/state/positions_nav.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print("\n" + "="*60)
        print(f"✅ STATE RESTORED")
        print("="*60)
        print(f"📊 Total NAV: ${total_nav:.2f}")
        print(f"💵 Free USDT: ${free_usdt:.2f}")
        print(f"💼 Invested: ${invested:.2f}")
        print(f"📍 Active Positions: {len(positions)}")
        print(f"💾 Saved to: {state_file}\n")
        
        return state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    asyncio.run(restore_positions_from_wallet())
