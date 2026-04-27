#!/usr/bin/env python3
"""
Real-Time Log Monitor for 6-Hour Session
Tracks Phase 2 implementations in real-time
"""

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class Phase2LogMonitor:
    """Monitor trading logs for Phase 2 indicators"""
    
    def __init__(self, log_file: str = "6hour_session_monitored.log"):
        self.log_file = log_file
        self.last_position = 0
        self.recovery_bypasses = []
        self.forced_rotations = []
        self.entry_sizes = []
        self.errors = []
        self.warnings = []
        
    def tail_log(self, lines: int = 50) -> list:
        """Get last N lines from log file"""
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except FileNotFoundError:
            return []
    
    def analyze_new_lines(self, lines: list):
        """Analyze new log lines for Phase 2 indicators"""
        for line in lines:
            # Check for recovery bypass
            if "[Meta:SafeMinHold] Bypassing min-hold check" in line:
                match = re.search(r'for forced recovery exit: (\w+)', line)
                if match:
                    symbol = match.group(1)
                    self.recovery_bypasses.append({
                        'time': datetime.now().isoformat(),
                        'symbol': symbol,
                        'log': line.strip()
                    })
                    print(f"✅ RECOVERY BYPASS: {symbol}")
            
            # Check for forced rotation override
            if "[REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN" in line:
                match = re.search(r'for (\w+)', line)
                if match:
                    symbol = match.group(1)
                    self.forced_rotations.append({
                        'time': datetime.now().isoformat(),
                        'symbol': symbol,
                        'log': line.strip()
                    })
                    print(f"✅ FORCED ROTATION OVERRIDE: {symbol}")
            
            # Check for entry sizes
            if "Entry:" in line and "USDT" in line:
                match = re.search(r'Entry: (\w+).*@ ([\d.]+) USDT', line)
                if match:
                    symbol = match.group(1)
                    size = float(match.group(2))
                    self.entry_sizes.append({
                        'time': datetime.now().isoformat(),
                        'symbol': symbol,
                        'size': size,
                        'log': line.strip()
                    })
                    status = "✅" if abs(size - 25.0) <= 1.0 else "⚠️"
                    print(f"{status} ENTRY SIZE: {symbol} @ {size} USDT")
            
            # Check for errors
            if "ERROR" in line or "CRITICAL" in line:
                self.errors.append({
                    'time': datetime.now().isoformat(),
                    'log': line.strip()
                })
                print(f"❌ ERROR: {line.strip()}")
            
            # Check for warnings
            if "WARNING" in line:
                self.warnings.append({
                    'time': datetime.now().isoformat(),
                    'log': line.strip()
                })
                print(f"⚠️  WARNING: {line.strip()}")
    
    def get_summary(self) -> dict:
        """Get current monitoring summary"""
        entry_sizes = [e['size'] for e in self.entry_sizes]
        return {
            'recovery_bypasses': len(self.recovery_bypasses),
            'forced_rotations': len(self.forced_rotations),
            'entry_sizes_tracked': len(self.entry_sizes),
            'entry_size_avg': sum(entry_sizes) / len(entry_sizes) if entry_sizes else 0,
            'entry_size_aligned': sum(1 for s in entry_sizes if abs(s - 25.0) <= 1.0),
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'recovery_bypasses_list': self.recovery_bypasses[-5:],  # Last 5
            'forced_rotations_list': self.forced_rotations[-5:],    # Last 5
        }
    
    def print_status(self):
        """Print current status"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("📊 PHASE 2 REAL-TIME MONITOR")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        print("🔄 RECOVERY BYPASS (Min-Hold Bypass)")
        print(f"   Triggers: {summary['recovery_bypasses']}")
        for bypass in summary['recovery_bypasses_list']:
            print(f"   - {bypass['time']}: {bypass['symbol']}")
        print()
        print("🔄 FORCED ROTATION OVERRIDE (MICRO Bracket Override)")
        print(f"   Triggers: {summary['forced_rotations']}")
        for rotation in summary['forced_rotations_list']:
            print(f"   - {rotation['time']}: {rotation['symbol']}")
        print()
        print("💰 ENTRY SIZING ALIGNMENT (25 USDT Target)")
        print(f"   Total entries: {summary['entry_sizes_tracked']}")
        print(f"   Aligned (±1 USDT): {summary['entry_size_aligned']}")
        print(f"   Average: {summary['entry_size_avg']:.2f} USDT")
        print()
        print("⚠️  ALERTS")
        print(f"   Errors: {summary['errors']}")
        print(f"   Warnings: {summary['warnings']}")
        print("="*80)

async def monitor_logs_realtime(log_file: str = "6hour_session_monitored.log", 
                               refresh_interval: int = 10):
    """Monitor logs in real-time"""
    monitor = Phase2LogMonitor(log_file)
    
    print(f"🚀 Starting real-time log monitor for {log_file}")
    print(f"   Refresh interval: {refresh_interval} seconds")
    print(f"   Press Ctrl+C to stop")
    print()
    
    last_size = 0
    
    try:
        while True:
            # Check if file exists and has new content
            log_path = Path(log_file)
            if log_path.exists():
                current_size = log_path.stat().st_size
                
                if current_size > last_size:
                    # New content, read and analyze
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                        monitor.analyze_new_lines(new_lines)
                    
                    last_size = current_size
            
            # Print status every 30 seconds
            if int(datetime.now().timestamp()) % 30 == 0:
                monitor.print_status()
            
            await asyncio.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped")
        monitor.print_status()

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "6hour_session_monitored.log"
    asyncio.run(monitor_logs_realtime(log_file))
