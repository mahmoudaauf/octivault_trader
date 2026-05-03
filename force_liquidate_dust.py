#!/usr/bin/env python3
"""
🚀 FORCE LIQUIDATE DUST - Immediate Capital Recovery

This script BYPASSES the healing thresholds and immediately liquidates
all dust positions (< $25) to free up capital for trading.

Usage:
    python3 force_liquidate_dust.py [dry-run|execute]
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def get_dust_positions(
    client, 
    shared_state,
    dust_threshold: float = 25.0,
) -> List[Tuple[str, float, float, float]]:
    """
    Find all dust positions (value < threshold).
    
    Returns:
        List of (symbol, qty, price, value) tuples
    """
    
    positions = []
    balances = await client.get_spot_balances()
    
    for symbol, balance in balances.items():
        if symbol == 'USDT':
            continue
        
        qty = float(balance.get('free', 0))
        if qty <= 0:
            continue
        
        # Get current price
        price = await shared_state.safe_price(f"{symbol}USDT", default=0.0)
        if price <= 0:
            continue
        
        value = qty * price
        
        # Check if dust
        if value < dust_threshold:
            positions.append((symbol, qty, price, value))
    
    # Sort by value (smallest first - easiest to liquidate)
    positions.sort(key=lambda x: x[3])
    
    return positions

async def liquidate_positions(
    exec_mgr,
    positions: List[Tuple[str, float, float, float]],
    dry_run: bool = True,
) -> Dict:
    """
    Liquidate dust positions.
    
    Returns:
        Summary of liquidation results
    """
    
    results = {
        'total_positions': len(positions),
        'attempted': 0,
        'successful': 0,
        'failed': 0,
        'total_recovered': 0.0,
        'errors': [],
    }
    
    for symbol, qty, price, value in positions:
        try:
            print(f"\n📤 Liquidating {symbol}:")
            print(f"   qty: {qty:.8f}")
            print(f"   price: ${price:.2f}")
            print(f"   value: ${value:.2f}")
            
            if dry_run:
                print(f"   [DRY RUN] Would submit SELL order")
                results['attempted'] += 1
                results['total_recovered'] += value
                results['successful'] += 1
            else:
                # Actually execute the liquidation
                print(f"   [EXECUTING] Submitting SELL order...")
                try:
                    plan = [{"symbol": symbol, "quantity": qty, "tag": "force_dust_cleanup"}]
                    result = await exec_mgr.execute_liquidation_plan(plan)
                    print(f"   ✅ Order submitted")
                    results['attempted'] += 1
                    results['successful'] += 1
                    results['total_recovered'] += value
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    results['attempted'] += 1
                    results['failed'] += 1
                    results['errors'].append((symbol, str(e)))
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")
            results['errors'].append((symbol, str(e)))
    
    return results

async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry-run"
    
    if mode not in ("dry-run", "execute"):
        print(f"Usage: {sys.argv[0]} [dry-run|execute]")
        print(f"  dry-run:  Show what would be liquidated (DEFAULT)")
        print(f"  execute:  Actually liquidate dust (LIVE)")
        sys.exit(1)
    
    dry_run = (mode == "dry-run")
    
    print("\n" + "="*80)
    print("🚀 FORCE LIQUIDATE DUST")
    print("="*80)
    print(f"Mode: {'DRY RUN (no orders submitted)' if dry_run else '⚠️  LIVE EXECUTION (orders will be submitted)'}")
    print("="*80)
    
    try:
        # Import after path is set
        from src.l0_core.exchange_client import ExchangeClient
        from src.l0_core.shared_state import SharedState
        from src.l0_core.config import CoreConfig
        from src.l4_execution.execution_manager import ExecutionManager
        
        # Initialize components
        print("\n[INIT] Connecting to exchange...")
        client = ExchangeClient()
        config = CoreConfig()
        shared_state = SharedState(config=config, exchange_client=client)
        exec_mgr = ExecutionManager(shared_state, client, config)
        
        print("✅ Components initialized")
        
        # Find dust positions
        print("\n[SCAN] Finding dust positions (< $25)...")
        dust = await get_dust_positions(client, shared_state, dust_threshold=25.0)
        
        if not dust:
            print("✅ No dust positions found - portfolio is clean!")
            return True
        
        print(f"🔴 Found {len(dust)} dust positions:")
        total_dust = 0.0
        for symbol, qty, price, value in dust:
            print(f"  • {symbol:8s} qty={qty:.8f} price=${price:>10.2f} value=${value:>7.2f}")
            total_dust += value
        
        print(f"\nTotal dust value: ${total_dust:.2f}")
        
        # Confirm before executing
        if not dry_run:
            print("\n⚠️  ABOUT TO EXECUTE LIVE LIQUIDATION")
            response = input("Continue? (type 'YES' to confirm): ").strip().upper()
            if response != "YES":
                print("Cancelled.")
                return False
        
        # Liquidate
        print("\n[EXECUTE] Starting liquidation...")
        results = await liquidate_positions(exec_mgr, dust, dry_run=dry_run)
        
        # Summary
        print("\n" + "="*80)
        print("📊 LIQUIDATION SUMMARY")
        print("="*80)
        print(f"Total positions:    {results['total_positions']}")
        print(f"Attempted:          {results['attempted']}")
        print(f"Successful:         {results['successful']}")
        print(f"Failed:             {results['failed']}")
        print(f"Total recovered:    ${results['total_recovered']:.2f}")
        
        if results['errors']:
            print(f"\n⚠️  Errors ({len(results['errors'])}):")
            for symbol, error in results['errors']:
                print(f"  • {symbol}: {error}")
        
        print("\n" + "="*80)
        
        if mode == "dry-run":
            print("\n💡 To actually execute liquidation, run:")
            print("   python3 force_liquidate_dust.py execute")
        else:
            if results['successful'] > 0:
                print(f"\n✅ Successfully liquidated {results['successful']} positions")
                print(f"   Recovered: ${results['total_recovered']:.2f}")
                print("\n   Waiting 30 seconds for orders to fill...")
                await asyncio.sleep(30)
                
                print("\n   📊 Checking updated balance...")
                balances = await client.get_spot_balances()
                new_free = float((balances.get('USDT') or {}).get('free', 0))
                print(f"   New free USDT: ${new_free:.2f}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Liquidation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
