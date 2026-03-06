#!/usr/bin/env python3
"""
Debug script to trace discovery agent symbol flow through the system.

Run this to see:
1. What symbols discovery agents are proposing
2. Which validation gates are rejecting them
3. What symbols actually make it to accepted_symbols
4. What the gap is (discovered vs accepted)
"""

import asyncio
import logging
from typing import Dict, Set, List
from collections import defaultdict

# Patch logging to capture rejections
rejection_log = defaultdict(list)
original_debug = logging.Logger.debug
original_warning = logging.Logger.warning

def patched_debug(self, msg, *args, **kwargs):
    if "risk filter" in str(msg).lower() or "rejected" in str(msg).lower():
        try:
            formatted_msg = msg % args if args else msg
            rejection_log["debug"].append(formatted_msg)
        except:
            pass
    return original_debug(self, msg, *args, **kwargs)

def patched_warning(self, msg, *args, **kwargs):
    if "rejected" in str(msg).lower() or "skipping" in str(msg).lower() or "no symbol" in str(msg).lower():
        try:
            formatted_msg = msg % args if args else msg
            rejection_log["warning"].append(formatted_msg)
        except:
            pass
    return original_warning(self, msg, *args, **kwargs)

logging.Logger.debug = patched_debug
logging.Logger.warning = patched_warning


async def diagnose_discovery_flow():
    """Main diagnostic routine."""
    
    print("=" * 80)
    print("🔍 DISCOVERY AGENT DATA FLOW DIAGNOSTIC")
    print("=" * 80)
    print()
    
    from core.config import Config
    from core.database_manager import DatabaseManager
    from core.shared_state import SharedState
    from core.exchange_client import ExchangeClient
    from core.symbol_manager import SymbolManager
    from agents.wallet_scanner_agent import WalletScannerAgent
    from agents.ipo_chaser import IPOChaser
    from agents.symbol_screener import SymbolScreener
    from core.tp_sl_engine import TPSLEngine
    
    # Initialize
    print("[1/4] Initializing system components...")
    cfg = Config()
    db = DatabaseManager(cfg)
    await db.connect()
    
    shared_state = SharedState(cfg, db)
    db.shared_state = shared_state
    
    exchange = ExchangeClient(cfg, shared_state)
    await exchange.initialize()
    
    symbol_mgr = SymbolManager(cfg, exchange, shared_state, db)
    shared_state.symbol_manager = symbol_mgr
    
    print("   ✓ System initialized")
    print()
    
    # Track discovered symbols
    discovered: Dict[str, Set[str]] = {
        "WalletScanner": set(),
        "SymbolScreener": set(),
        "IPOChaser": set(),
    }
    
    # Monkey-patch _propose to track discoveries
    original_propose_methods = {}
    
    async def track_propose(agent_name, original_propose, symbol, **kwargs):
        discovered[agent_name].add(symbol)
        return await original_propose(symbol, **kwargs)
    
    # Run discovery
    print("[2/4] Running discovery agents...")
    print()
    
    # WalletScanner
    print("   🔎 WalletScannerAgent...")
    wallet_agent = WalletScannerAgent(shared_state, cfg, exchange, symbol_mgr)
    await wallet_agent.run_discovery()
    print(f"      (check logs for proposals)")
    print()
    
    # SymbolScreener
    print("   🔎 SymbolScreener...")
    screener_agent = SymbolScreener(shared_state, exchange, cfg, symbol_mgr)
    await screener_agent.run_discovery()
    print(f"      (check logs for proposals)")
    print()
    
    # IPOChaser
    print("   🔎 IPOChaser...")
    ipo_agent = IPOChaser(
        shared_state,
        cfg,
        exchange,
        symbol_mgr,
        None,  # execution_manager
        TPSLEngine(cfg, shared_state, None)
    )
    await ipo_agent.run_discovery()
    print(f"      (check logs for proposals)")
    print()
    
    # Flush buffered proposals
    print("[3/4] Flushing buffered proposals...")
    if hasattr(symbol_mgr, "flush_buffered_proposals_to_shared_state"):
        await symbol_mgr.flush_buffered_proposals_to_shared_state()
    print("   ✓ Flush complete")
    print()
    
    # Report
    print("[4/4] Results")
    print("-" * 80)
    print()
    
    accepted = set(shared_state.accepted_symbols.keys()) if shared_state.accepted_symbols else set()
    
    print(f"📊 ACCEPTED SYMBOLS: {len(accepted)}")
    if accepted:
        for sym in sorted(accepted)[:20]:
            meta = shared_state.accepted_symbols.get(sym, {})
            source = meta.get("source", "unknown")
            print(f"   ✓ {sym:12} (from: {source})")
        if len(accepted) > 20:
            print(f"   ... and {len(accepted) - 20} more")
    print()
    
    # Rejection analysis
    print(f"⚠️  REJECTION LOG ANALYSIS:")
    print()
    
    if rejection_log.get("debug"):
        print(f"   Debug-level rejections ({len(rejection_log['debug'])}):")
        for msg in rejection_log["debug"][:10]:
            print(f"      • {msg}")
        if len(rejection_log["debug"]) > 10:
            print(f"      ... and {len(rejection_log['debug']) - 10} more")
        print()
    
    if rejection_log.get("warning"):
        print(f"   Warning-level rejections ({len(rejection_log['warning'])}):")
        for msg in rejection_log["warning"][:10]:
            print(f"      • {msg}")
        if len(rejection_log["warning"]) > 10:
            print(f"      ... and {len(rejection_log['warning']) - 10} more")
        print()
    
    # Risk filter analysis
    print(f"🚫 MOST COMMON REJECTION REASONS:")
    rejection_reasons = defaultdict(int)
    for msg_list in rejection_log.values():
        for msg in msg_list:
            msg_lower = str(msg).lower()
            if "risk filter" in msg_lower:
                rejection_reasons["risk_filter"] += 1
            elif "already exists" in msg_lower:
                rejection_reasons["already_exists"] += 1
            elif "cap reached" in msg_lower:
                rejection_reasons["cap_reached"] += 1
            elif "no symbol" in msg_lower or "missing" in msg_lower:
                rejection_reasons["missing_data"] += 1
            elif "status" in msg_lower:
                rejection_reasons["trading_status"] += 1
            elif "price" in msg_lower:
                rejection_reasons["price_validation"] += 1
            else:
                rejection_reasons["other"] += 1
    
    if rejection_reasons:
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"   • {reason:30} → {count:3} rejections")
    else:
        print("   (no rejections detected in logs)")
    print()
    
    # Recommendations
    print("=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    print()
    
    # Check if accepted is empty
    if not accepted:
        print("❌ NO SYMBOLS ACCEPTED - This is the core issue!")
        print()
        print("   Likely causes:")
        print("   1. Risk filters are too strict")
        print("   2. Symbol validation gates are too tight")
        print("   3. Exchange data not available")
        print("   4. Discovery disabled in config")
        print()
        print("   Action: Check symbol_manager._passes_risk_filters()")
        print("          and adjust thresholds in config")
        print()
    
    # Check if cap is hit
    cap = getattr(symbol_mgr, "_cap", None)
    if cap and len(accepted) >= cap:
        print("⚠️  SYMBOL CAP REACHED")
        print(f"   Current: {len(accepted)} / {cap}")
        print(f"   Buffered: {len(getattr(symbol_mgr, 'buffered_symbols', []))}")
        print()
        print("   Action: Increase Discovery.symbol_cap in config")
        print()
    
    # Check for risk filter rejections
    if rejection_reasons.get("risk_filter", 0) > 0:
        print("🚩 RISK FILTERS ARE REJECTING SYMBOLS")
        print(f"   Rejections due to risk: {rejection_reasons['risk_filter']}")
        print()
        print("   Action: Review and loosen risk filter thresholds:")
        print("           - Check config.RiskFilter.*")
        print("           - See symbol_manager._passes_risk_filters()")
        print()
    
    # Check trading status
    if rejection_reasons.get("trading_status", 0) > 0:
        print("🚩 SYMBOLS REJECTED FOR TRADING STATUS")
        print(f"   Rejections due to status: {rejection_reasons['trading_status']}")
        print()
        print("   Action: Ensure symbols are TRADING status on Binance")
        print("           Turn off require_trading_status if needed")
        print()
    
    # Check price availability
    if rejection_reasons.get("price_validation", 0) > 0:
        print("🚩 PRICE VALIDATION FAILURES")
        print(f"   Rejections due to price: {rejection_reasons['price_validation']}")
        print()
        print("   Action: Check exchange_client.get_ticker_price()")
        print("           Verify connection to price feed")
        print()
    
    print("=" * 80)
    print()


if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    try:
        asyncio.run(diagnose_discovery_flow())
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
