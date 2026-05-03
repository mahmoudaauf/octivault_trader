#!/usr/bin/env python3
"""
Real-time Balance Monitor with WebSocket Status
"""

import re
import time
from pathlib import Path

LOG_FILE = Path("/tmp/octivault_ws.log")

def get_latest_metrics():
    """Extract latest metrics from logs"""
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    metrics = {
        'nav_current': None,
        'nav_previous': None,
        'delta': None,
        'verdict': None,
        'slope': None,
        'trading_signals': 0,
        'trades_executed': 0,
        'websocket_status': None,
    }
    
    # Search for NAV Attribution
    for line in reversed(lines[-100:]):
        if 'NavAttribution' in line and 'NAV' in line:
            match = re.search(r'NAV \$([0-9.]+)->(?:\$)?([0-9.]+)', line)
            if match:
                metrics['nav_previous'] = float(match.group(1))
                metrics['nav_current'] = float(match.group(2))
            
            match_delta = re.search(r'Δ=([+-][0-9.]+)', line)
            if match_delta:
                metrics['delta'] = float(match_delta.group(1))
            
            match_verdict = re.search(r'verdict=(\w+)', line)
            if match_verdict:
                metrics['verdict'] = match_verdict.group(1)
            
            match_slope = re.search(r'slope=([+-][0-9.]+)/min', line)
            if match_slope:
                metrics['slope'] = float(match_slope.group(1))
            break
    
    # Count trading signals
    metrics['trading_signals'] = sum(1 for line in lines if 'Published TradeIntent' in line)
    
    # Count trades executed
    metrics['trades_executed'] = sum(1 for line in lines if 'TRADE_FILLED' in line)
    
    # Check WebSocket
    for line in reversed(lines[-200:]):
        if '[WS:Connected]' in line:
            metrics['websocket_status'] = '✅ Connected'
            break
        elif '[WS:Closed]' in line:
            metrics['websocket_status'] = '⚠️  Reconnecting'
            break
        elif '[WS:AutoSubscribe]' in line:
            metrics['websocket_status'] = '✅ Auto-subscribed'
            break
    
    if metrics['websocket_status'] is None:
        metrics['websocket_status'] = '🔄 Initializing'
    
    return metrics

def print_report():
    """Print formatted report"""
    metrics = get_latest_metrics()
    
    print("\n" + "="*70)
    print("🔍 OCTIVAULT TRADING BOT - REAL-TIME STATUS")
    print("="*70)
    
    print("\n💰 BALANCE METRICS:")
    if metrics['nav_previous']:
        print(f"  Previous NAV:      ${metrics['nav_previous']:.2f}")
    if metrics['nav_current']:
        print(f"  Current NAV:       ${metrics['nav_current']:.2f}")
    if metrics['delta']:
        print(f"  Change:            ${metrics['delta']:+.4f}")
    
    if metrics['verdict'] == 'GROWING' and metrics['slope']:
        print(f"  Status:            ✅ GROWING ({metrics['slope']:+.4f}%/min)")
    elif metrics['verdict'] == 'DECLINING' and metrics['slope']:
        print(f"  Status:            ⚠️  DECLINING ({metrics['slope']:+.4f}%/min)")
    elif metrics['verdict']:
        print(f"  Status:            → {metrics['verdict']}")
    else:
        print(f"  Status:            🔄 Calculating...")
    
    print("\n📊 TRADING ACTIVITY:")
    print(f"  Signals Generated: {metrics['trading_signals']}")
    print(f"  Trades Executed:   {metrics['trades_executed']}")
    
    print("\n🌐 WEBSOCKET STATUS:")
    print(f"  Connection:        {metrics['websocket_status']}")
    
    print("\n" + "="*70)
    
    return metrics['verdict'] == 'GROWING'

if __name__ == "__main__":
    is_growing = print_report()
    
    if is_growing:
        print("✅ SYSTEM IS PROFITABLE - BALANCE GROWING!")
    else:
        print("⚠️  Monitor balance trend")
    
    print("="*70 + "\n")
