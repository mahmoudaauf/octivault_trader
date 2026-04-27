#!/usr/bin/env python3
"""
REALTIME SYSTEM MONITOR - 2 HOUR SESSION
Displays live system metrics and health every 60 seconds
"""

import subprocess
import time
import sys
from datetime import datetime
import re

def get_pids():
    """Get main process PIDs"""
    result = subprocess.run(
        "ps aux | grep -E 'python.*MASTER_SYSTEM_ORCHESTRATOR|2HOUR_CHECKPOINT' | grep -v grep",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def get_recent_logs(lines=30):
    """Get recent log lines"""
    result = subprocess.run(
        f"tail -{lines} /tmp/octivault_master_orchestrator.log 2>/dev/null",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def extract_metrics(logs):
    """Extract key metrics from logs"""
    metrics = {
        "components_running": 0,
        "components_healthy": 0,
        "pnl": 0.0,
        "positions": [],
        "signals": 0,
        "last_activity": "N/A"
    }
    
    # Extract components status
    component_pattern = r'\[(\w+)\] Status: (\w+) \| Detail:'
    for match in re.finditer(component_pattern, logs):
        metrics["components_running"] += 1
        if match.group(2) == "Healthy" or match.group(2) == "Running":
            metrics["components_healthy"] += 1
    
    # Extract positions
    position_pattern = r'LINKUSDT|ETHUSDT|BTCUSDT|DOGEUSDT|SOLUSDT'
    positions = re.findall(position_pattern, logs)
    if positions:
        metrics["positions"] = list(set(positions))
    
    # Get timestamp
    time_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    times = re.findall(time_pattern, logs)
    if times:
        metrics["last_activity"] = times[-1]
    
    return metrics

def print_header():
    """Print monitoring header"""
    print("\n" + "=" * 100)
    print("🎯 REALTIME 2-HOUR SESSION MONITORING".center(100))
    print("=" * 100)
    print(f"Monitor Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Check Interval: 60 seconds")
    print("=" * 100 + "\n")

def main():
    """Main monitoring loop"""
    print_header()
    
    cycle = 0
    
    while True:
        cycle += 1
        print(f"\n{'─' * 100}")
        print(f"📊 UPDATE #{cycle} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─' * 100}")
        
        # Check processes
        pids = get_pids()
        if pids:
            print("✅ PROCESSES RUNNING:")
            for line in pids.split('\n'):
                if line.strip():
                    parts = line.split()
                    pid = parts[1] if len(parts) > 1 else "?"
                    cmd = ' '.join(parts[10:]) if len(parts) > 10 else "?"
                    print(f"   └─ PID {pid}: {cmd[:60]}...")
        else:
            print("❌ NO PROCESSES RUNNING")
        
        # Get recent logs and extract metrics
        logs = get_recent_logs(50)
        metrics = extract_metrics(logs)
        
        print("\n📈 SYSTEM METRICS:")
        print(f"   ├─ Components Running: {metrics['components_running']}")
        print(f"   ├─ Components Healthy: {metrics['components_healthy']}")
        print(f"   ├─ Active Positions: {', '.join(metrics['positions']) if metrics['positions'] else 'None'}")
        print(f"   ├─ Last Activity: {metrics['last_activity']}")
        
        # Show recent activity
        print("\n📋 RECENT ACTIVITY (last 15 lines):")
        recent_logs = logs.split('\n')[-15:]
        for log_line in recent_logs:
            if log_line.strip():
                # Color code important messages
                if "Running" in log_line or "healthy" in log_line.lower():
                    print(f"   ✅ {log_line[:85]}")
                elif "ERROR" in log_line or "FAILED" in log_line:
                    print(f"   ❌ {log_line[:85]}")
                elif "WARNING" in log_line:
                    print(f"   ⚠️  {log_line[:85]}")
                else:
                    print(f"   📝 {log_line[:85]}")
        
        print(f"\n{'─' * 100}")
        print("⏳ Waiting 60 seconds for next update (Ctrl+C to stop)...")
        
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped by user")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")
        sys.exit(1)
