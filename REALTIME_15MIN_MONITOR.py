#!/usr/bin/env python3
"""
🔍 REAL-TIME SYSTEM MONITOR
============================

Monitors the running orchestrator and provides status every 2 minutes for 15 minutes.

USAGE:
    python3 REALTIME_15MIN_MONITOR.py

This script watches /tmp/octivault_master_orchestrator.log and provides:
- Status checks every 2 minutes
- Trade counts
- Error tracking
- System health metrics
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
import re

def get_log_stats(log_file: str, since_lines: int = 0):
    """Extract stats from recent log lines"""
    try:
        result = subprocess.run(
            f"tail -n 500 {log_file}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=2
        )
        lines = result.stdout.strip().split('\n')
        
        stats = {
            'total_lines': len(lines),
            'buy_orders': len([l for l in lines if 'BUY' in l and 'Published' in l]),
            'sell_orders': len([l for l in lines if 'SELL' in l and 'Published' in l]),
            'errors': len([l for l in lines if 'ERROR' in l or '❌' in l]),
            'warnings': len([l for l in lines if 'WARNING' in l or '⚠️' in l]),
            'models_trained': len([l for l in lines if 'Model saved to' in l or 'model saved' in l]),
            'executions': len([l for l in lines if '[EM:BOOTSTRAP]' in l or 'EXEC' in l]),
        }
        
        return stats, lines
    except Exception as e:
        return {}, []

def get_process_status():
    """Get orchestrator process status"""
    try:
        result = subprocess.run(
            "ps aux | grep 'MASTER_SYSTEM_ORCHESTRATOR.py' | grep -v grep",
            shell=True,
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.stdout.strip():
            parts = result.stdout.split()
            if len(parts) >= 11:
                return {
                    'pid': parts[1],
                    'cpu': parts[2],
                    'mem': parts[3],
                    'status': 'RUNNING ✅'
                }
        return {'status': 'STOPPED ❌'}
    except:
        return {'status': 'ERROR'}

def print_status_check(check_num: int, elapsed_sec: int, stats: dict, proc_status: dict, lines: list):
    """Print a formatted status check"""
    print("\n" + "="*90)
    print(f"📊 STATUS CHECK #{check_num} - {datetime.now().strftime('%H:%M:%S')} | "
          f"Elapsed: {elapsed_sec}s / 900s | Remaining: {900-elapsed_sec}s")
    print("="*90)
    
    # Process status
    print(f"\n🔄 PROCESS STATUS:")
    print(f"   Status: {proc_status.get('status', 'UNKNOWN')}")
    if 'pid' in proc_status:
        print(f"   PID: {proc_status['pid']} | CPU: {proc_status.get('cpu', '?')}% | "
              f"Memory: {proc_status.get('mem', '?')}%")
    
    # Trading stats
    print(f"\n📈 TRADING ACTIVITY (last 500 lines):")
    print(f"   BUY Intents:  {stats.get('buy_orders', 0)}")
    print(f"   SELL Intents: {stats.get('sell_orders', 0)}")
    print(f"   Executions:   {stats.get('executions', 0)}")
    print(f"   Models Saved: {stats.get('models_trained', 0)}")
    
    # Health indicators
    print(f"\n⚠️  ALERTS:")
    print(f"   Errors:   {stats.get('errors', 0)}")
    print(f"   Warnings: {stats.get('warnings', 0)}")
    
    # Recent activity
    print(f"\n📝 RECENT LOG ENTRIES (last 3):")
    for line in lines[-3:]:
        if line.strip():
            # Extract just the important part
            if '[' in line:
                timestamp_end = line.find(']', line.find('[', line.find('[') + 1) + 1) + 1
                log_content = line[timestamp_end:].strip()
            else:
                log_content = line.strip()
            
            # Truncate long lines
            if len(log_content) > 100:
                log_content = log_content[:97] + "..."
            print(f"   {log_content}")
    
    print("="*90)

def main():
    """Main monitoring loop"""
    log_file = "/tmp/octivault_master_orchestrator.log"
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        print("Make sure the orchestrator is running first!")
        return
    
    print("\n" + "="*90)
    print("🔍 OCTI AI TRADING BOT - 15 MINUTE REAL-TIME MONITOR")
    print("="*90)
    print(f"📍 Monitoring: {log_file}")
    print(f"⏱️  Duration: 15 minutes (900 seconds)")
    print(f"📊 Status updates: Every 2 minutes (120 seconds)")
    print(f"🚀 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90)
    
    start_time = time.time()
    end_time = start_time + (15 * 60)  # 15 minutes
    check_num = 1
    last_check_time = start_time
    
    try:
        while time.time() < end_time:
            current_time = time.time()
            elapsed_sec = int(current_time - start_time)
            
            # Check status every 2 minutes
            if current_time - last_check_time >= 120 or check_num == 1:
                stats, lines = get_log_stats(log_file)
                proc_status = get_process_status()
                
                if 'STOPPED' in proc_status.get('status', ''):
                    print("\n❌ PROCESS HAS STOPPED!")
                    break
                
                print_status_check(check_num, elapsed_sec, stats, proc_status, lines)
                check_num += 1
                last_check_time = current_time
            
            # Wait a bit before checking again
            time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring interrupted by user")
        return
    
    # Print final summary
    print("\n" + "="*90)
    print("🏁 15-MINUTE MONITORING SESSION COMPLETE")
    print("="*90)
    stats, lines = get_log_stats(log_file)
    proc_status = get_process_status()
    
    print(f"\nFinal Status: {proc_status.get('status', 'UNKNOWN')}")
    print(f"Total Checks: {check_num - 1}")
    print(f"\nFinal Statistics (from recent logs):")
    print(f"   BUY Intents:  {stats.get('buy_orders', 0)}")
    print(f"   SELL Intents: {stats.get('sell_orders', 0)}")
    print(f"   Executions:   {stats.get('executions', 0)}")
    print(f"   Models Saved: {stats.get('models_trained', 0)}")
    print(f"   Errors:       {stats.get('errors', 0)}")
    print(f"   Warnings:     {stats.get('warnings', 0)}")
    print(f"\nMonitoring ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
