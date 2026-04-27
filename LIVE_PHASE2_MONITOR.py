#!/usr/bin/env python3
"""
LIVE TRADING - PHASE 2 REAL-TIME MONITOR
Monitor Phase 2 indicator events as they occur in trading.log
"""

import os
import time
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print monitoring dashboard header"""
    os.system('clear')
    print(f"""
{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           🚀 LIVE TRADING - PHASE 2 REAL-TIME MONITOR 🚀                  ║
║                                                                            ║
║                     Watching: trading.log                                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")

def get_log_position(log_file):
    """Get current end position of log file"""
    try:
        return os.path.getsize(log_file)
    except:
        return 0

def tail_log_from_position(log_file, position, chunk_size=4096):
    """Read new lines from log file since position"""
    try:
        with open(log_file, 'rb') as f:
            f.seek(position)
            return f.read(chunk_size).decode('utf-8', errors='ignore')
    except:
        return ""

def parse_indicators(lines):
    """Parse Phase 2 indicators from log lines"""
    indicators = {
        'recovery_bypass': [],
        'forced_rotation': [],
        'entry_sizing': [],
        'errors': [],
        'exec_rejects': [],
        'gate_drops': [],
        'signals': [],
        'rejection_counts': {},  # symbol -> count
        'strategy_counts': {},
        'timestamp': None
    }
    
    for line in lines.split('\n'):
        if not line.strip():
            continue
            
        try:
            # Extract timestamp if present (format: 2026-04-24 10:15:25,735)
            if '[' in line and ']' in line:
                time_part = line.split('][')[0].replace('[', '')
                if len(time_part) > 8:
                    indicators['timestamp'] = time_part
            
            # Detection: Recovery Exit Bypass
            if "Bypassing min-hold" in line:
                indicators['recovery_bypass'].append(line.strip())

            # Detection: Forced Rotation Override
            if "MICRO restriction OVERRIDDEN" in line:
                indicators['forced_rotation'].append(line.strip())

            # Detection: Entry Size Enforcement
            if "ENTRY_SIZE_ENFORCEMENT" in line or "ENTRY_SIZE_ENFORCEMENT" in line.upper():
                indicators['entry_sizing'].append(line.strip())

            # Detection: Execution rejections
            if "EXEC_REJECT" in line or "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD" in line or "EXEC_REJECT]" in line:
                indicators['exec_rejects'].append(line.strip())

            # Detection: Gate drops / deadlock diagnostics
            if "NO SIGNALS PASSED FILTERS" in line or "NO SIGNALS PASSED" in line or "NO SIGNALS PASSED FILTERS" in line.upper():
                indicators['gate_drops'].append(line.strip())

            # Detection: Skip/Skipping messages with rejection counts per symbol
            # e.g. "Skipping KATUSDT BUY: rejected 77 times >= threshold 10"
            if "Skipping" in line and "rejected" in line:
                try:
                    # attempt to parse symbol and count
                    parts = line.split()
                    # find token like SYMBOL
                    symbol = None
                    count = None
                    for i, tok in enumerate(parts):
                        if tok.endswith("BUY") or tok.endswith("SELL"):
                            # previous token may be symbol
                            if i > 0:
                                symbol = parts[i-1].strip(',')
                        if tok == "rejected" and i+1 < len(parts):
                            try:
                                count = int(parts[i+1])
                            except:
                                count = None
                    if symbol:
                        indicators['rejection_counts'][symbol] = indicators['rejection_counts'].get(symbol, 0) + (count or 1)
                except Exception:
                    pass

            # Detection: Strategy signals and caching
            # e.g. "Published TradeIntent: ETHUSDT BUY" or "Received signal for ETHUSDT from SwingTradeHunter"
            if "Published TradeIntent:" in line or "Received signal for" in line:
                indicators['signals'].append(line.strip())
                # try to extract strategy/source
                try:
                    if "from" in line:
                        src = line.split("from")[-1].strip()
                        src_token = src.split()[0]
                        indicators['strategy_counts'][src_token] = indicators['strategy_counts'].get(src_token, 0) + 1
                except Exception:
                    pass

            # Detection: Errors
            if "[ERROR]" in line or "[CRITICAL]" in line:
                indicators['errors'].append(line.strip())
                
        except Exception as e:
            pass
    
    return indicators

def display_metrics(metrics):
    """Display collected metrics"""
    print(f"""
{Colors.BOLD}{Colors.GREEN}═════════════════════════════════════════════════════════════════════════════
 PHASE 2 INDICATORS - SESSION STATISTICS
═════════════════════════════════════════════════════════════════════════════{Colors.ENDC}
""")
    
    print(f"{Colors.CYAN}Recovery Exit Min-Hold Bypass:{Colors.ENDC}")
    print(f"  Count: {Colors.YELLOW}{metrics['recovery_bypasses']}{Colors.ENDC}")
    print(f"  Expected: 1-2 per hour")
    print(f"  Status: {Colors.GREEN if metrics['recovery_bypasses'] > 0 else Colors.YELLOW}{'✅ DETECTED' if metrics['recovery_bypasses'] > 0 else '⏳ Waiting...'}{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}Forced Rotation MICRO Override:{Colors.ENDC}")
    print(f"  Count: {Colors.YELLOW}{metrics['forced_rotations']}{Colors.ENDC}")
    print(f"  Expected: 0-1 per hour")
    print(f"  Status: {Colors.GREEN if metrics['forced_rotations'] > 0 else Colors.YELLOW}{'✅ DETECTED' if metrics['forced_rotations'] > 0 else '⏳ Not yet...'}{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}Entry Sizing Alignment (25 USDT):{Colors.ENDC}")
    print(f"  Count: {Colors.YELLOW}{metrics['entry_sizes']}{Colors.ENDC}")
    print(f"  Expected: 5-10 per hour")
    print(f"  Status: {Colors.GREEN if metrics['entry_sizes'] > 0 else Colors.YELLOW}{'✅ ALIGNED' if metrics['entry_sizes'] > 0 else '⏳ Waiting...'}{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}Errors/Warnings:{Colors.ENDC}")
    print(f"  Count: {Colors.RED if metrics['errors'] > 0 else Colors.GREEN}{metrics['errors']}{Colors.ENDC}")
    print(f"  Status: {Colors.GREEN if metrics['errors'] == 0 else Colors.RED}{'✅ No issues' if metrics['errors'] == 0 else '⚠️  CHECK LOGS'}{Colors.ENDC}\n")

    # Gating & filtering summary
    print(f"{Colors.CYAN}Gating & Filtering:{Colors.ENDC}")
    print(f"  Gate drops: {Colors.YELLOW}{metrics['gate_drops']}{Colors.ENDC}")
    print(f"  Exec rejects: {Colors.YELLOW}{metrics['exec_rejects']}{Colors.ENDC}")
    print(f"  Symbols with high rejection counts:{Colors.ENDC}")
    # show top 5 rejected symbols
    sorted_rej = sorted(metrics['rejection_counts'].items(), key=lambda x: x[1], reverse=True)
    for sym, cnt in sorted_rej[:5]:
        print(f"    - {sym}: {cnt} rejections")
    if not sorted_rej:
        print("    (none)")

    # Strategy signal counts
    print(f"\n{Colors.CYAN}Strategy signal counts:{Colors.ENDC}")
    for strat, cnt in metrics.get('strategy_counts', {}).items():
        print(f"  - {strat}: {cnt}")
    if not metrics.get('strategy_counts'):
        print("  (no strategy signals detected yet)")
    print()

def display_recent_events(recent_lines, limit=5):
    """Display recent Phase 2 events"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}═════════════════════════════════════════════════════════════════════════════")
    print(f" RECENT PHASE 2 EVENTS (Last {limit})")
    print(f"═════════════════════════════════════════════════════════════════════════════{Colors.ENDC}\n")
    
    if not recent_lines:
        print(f"{Colors.YELLOW}No Phase 2 events yet. Waiting for trades...{Colors.ENDC}\n")
    else:
        for i, event in enumerate(recent_lines[-limit:], 1):
            if "Bypassing min-hold" in event:
                icon = "🔓"
                color = Colors.GREEN
            elif "NO SIGNALS PASSED" in event or "NO SIGNALS PASSED FILTERS" in event:
                icon = "🚫"
                color = Colors.RED
            elif "Skipping" in event and "rejected" in event:
                icon = "⚠️"
                color = Colors.YELLOW
            elif "EXEC_REJECT" in event or "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD" in event:
                icon = "🔒"
                color = Colors.RED
            elif "MICRO restriction OVERRIDDEN" in event:
                icon = "🔄"
                color = Colors.BLUE
            elif "ENTRY_SIZE_ENFORCEMENT" in event:
                icon = "💰"
                color = Colors.YELLOW
            else:
                icon = "❓"
                color = Colors.ENDC
                
            print(f"{color}{i}. {icon} {event[:120]}...{Colors.ENDC}\n")

def monitor_live(log_file, refresh_interval=5):
    """Monitor log file for Phase 2 indicators"""
    
    metrics = {
        'recovery_bypasses': 0,
        'forced_rotations': 0,
        'entry_sizes': 0,
        'errors': 0,
        'exec_rejects': 0,
        'gate_drops': 0,
        'rejection_counts': {},
        'strategy_counts': {},
        'all_events': []
    }
    
    position = 0
    last_display = 0
    
    print(f"{Colors.YELLOW}🔍 Starting monitor... refreshing every {refresh_interval}s{Colors.ENDC}\n")
    time.sleep(2)
    
    try:
        while True:
            current_time = time.time()
            
            # Read new log lines
            new_content = tail_log_from_position(log_file, position, chunk_size=8192)
            if new_content:
                position += len(new_content.encode('utf-8'))
                indicators = parse_indicators(new_content)
                
                # Update metrics
                metrics['recovery_bypasses'] += len(indicators['recovery_bypass'])
                metrics['forced_rotations'] += len(indicators['forced_rotation'])
                metrics['entry_sizes'] += len(indicators['entry_sizing'])
                metrics['errors'] += len(indicators['errors'])
                
                # Store all events
                for event in indicators['recovery_bypass']:
                    metrics['all_events'].append(event)
                for event in indicators['forced_rotation']:
                    metrics['all_events'].append(event)
                for event in indicators['entry_sizing']:
                    metrics['all_events'].append(event)
            
            # Display metrics every refresh_interval seconds
            if current_time - last_display >= refresh_interval:
                print_header()
                display_metrics(metrics)
                display_recent_events(metrics['all_events'])
                
                print(f"{Colors.BOLD}{Colors.CYAN}═════════════════════════════════════════════════════════════════════════════{Colors.ENDC}")
                print(f"⏱️  Monitor running... (Press Ctrl+C to stop)")
                print(f"📊 Session time: ~{int((time.time() - last_display) % 3600)}s")
                print(f"🔄 Refresh interval: {refresh_interval}s\n")
                
                last_display = current_time
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitor stopped.{Colors.ENDC}")
        print(f"\n{Colors.CYAN}Final Statistics:{Colors.ENDC}")
        print(f"  Recovery Bypasses: {metrics['recovery_bypasses']}")
        print(f"  Forced Rotations: {metrics['forced_rotations']}")
        print(f"  Entry Sizes: {metrics['entry_sizes']}")
        print(f"  Errors: {metrics['errors']}")
        sys.exit(0)

if __name__ == "__main__":
    log_file = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/trading.log"
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"{Colors.RED}Error: Log file not found: {log_file}{Colors.ENDC}")
        sys.exit(1)
    
    # Start monitoring
    monitor_live(log_file, refresh_interval=5)
