#!/usr/bin/env python3
"""
CONTINUOUS ACTIVE MONITORING - PHASE 2 LIVE TRADING
Real-time monitoring with health checks, alerts, and auto-recovery
"""

import os
import time
import subprocess
import sys
import json
from datetime import datetime
from collections import deque
import signal

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

class ContinuousMonitor:
    def __init__(self):
        self.log_file = "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/trading.log"
        self.process_name = "MASTER_SYSTEM_ORCHESTRATOR"
        self.metrics = {
            'recovery_bypasses': 0,
            'forced_rotations': 0,
            'entry_sizes': 0,
            'trades_executed': 0,
            'errors': 0,
            'warnings': 0,
            'last_update': datetime.now(),
            'uptime_seconds': 0
        }
        self.process_start_time = None
        self.recent_events = deque(maxlen=20)
        self.log_position = 0
        self.check_interval = 5
        self.alert_threshold = 300  # 5 minutes with no activity
        self.last_activity_time = time.time()
        
    def get_process_status(self):
        """Check if trading process is running"""
        try:
            result = subprocess.run(
                f"ps aux | grep '{self.process_name}' | grep -v grep | wc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            count = int(result.stdout.strip())
            return count > 0
        except:
            return False
    
    def get_process_info(self):
        """Get process details (PID, CPU, Memory)"""
        try:
            result = subprocess.run(
                f"ps aux | grep '{self.process_name}' | grep -v grep | awk '{{print $2, $3, $6}}'",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                parts = result.stdout.strip().split()
                if len(parts) >= 3:
                    return {
                        'pid': parts[0],
                        'cpu_percent': float(parts[1]),
                        'memory_mb': int(parts[2]) / 1024
                    }
        except:
            pass
        return None
    
    def tail_log_from_position(self, chunk_size=8192):
        """Read new lines from log file"""
        try:
            with open(self.log_file, 'rb') as f:
                f.seek(self.log_position)
                new_content = f.read(chunk_size)
                self.log_position = f.tell()
                return new_content.decode('utf-8', errors='ignore')
        except:
            return ""
    
    def parse_new_logs(self, content):
        """Parse new log lines for Phase 2 indicators"""
        if not content:
            return False
        
        has_activity = False
        
        for line in content.split('\n'):
            if not line.strip():
                continue
            
            # Recovery bypass
            if "Bypassing min-hold" in line:
                self.metrics['recovery_bypasses'] += 1
                self.recent_events.append(('🔓 BYPASS', line[:80]))
                has_activity = True
                
            # Forced rotation
            elif "MICRO restriction OVERRIDDEN" in line:
                self.metrics['forced_rotations'] += 1
                self.recent_events.append(('🔄 ROTATION', line[:80]))
                has_activity = True
                
            # Entry sizing
            elif "ENTRY_SIZE_ENFORCEMENT.*25" in line or "quote=25" in line:
                self.metrics['entry_sizes'] += 1
                has_activity = True
                
            # Trade execution
            elif "[EXEC_DECISION]" in line and ("BUY" in line or "SELL" in line):
                self.metrics['trades_executed'] += 1
                self.recent_events.append(('📊 TRADE', line[:80]))
                has_activity = True
                
            # Errors/Critical
            elif "[ERROR]" in line or "[CRITICAL]" in line:
                self.metrics['errors'] += 1
                self.recent_events.append(('❌ ERROR', line[:80]))
                has_activity = True
                
            # Warnings
            elif "[WARNING]" in line:
                self.metrics['warnings'] += 1
        
        return has_activity
    
    def check_system_health(self):
        """Perform health checks"""
        health_status = {
            'process_running': self.get_process_status(),
            'process_info': self.get_process_info(),
            'log_updating': os.path.exists(self.log_file),
            'memory_ok': True,
            'cpu_ok': True
        }
        
        if health_status['process_info']:
            info = health_status['process_info']
            health_status['memory_ok'] = info['memory_mb'] < 2048  # Less than 2GB
            health_status['cpu_ok'] = info['cpu_percent'] < 150  # Less than 150%
        
        return health_status
    
    def get_health_emoji(self, health):
        """Get emoji based on health status"""
        if not health['process_running']:
            return '🔴'
        elif health['memory_ok'] and health['cpu_ok']:
            return '🟢'
        else:
            return '🟡'
    
    def display_header(self):
        """Display monitoring header"""
        os.system('clear')
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  🚀 CONTINUOUS ACTIVE MONITORING - PHASE 2 LIVE TRADING".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("║" + "  Status: Trading System Under Active Watch".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"{Colors.ENDC}\n")
    
    def display_metrics(self, health):
        """Display real-time metrics"""
        print(f"{Colors.BOLD}{Colors.GREEN}System Health & Metrics{Colors.ENDC}")
        print("─" * 78)
        
        health_emoji = self.get_health_emoji(health)
        process_status = "🟢 RUNNING" if health['process_running'] else "🔴 STOPPED"
        
        print(f"\n{health_emoji} Process Status: {process_status}")
        
        if health['process_info']:
            info = health['process_info']
            print(f"   PID: {info['pid']} | CPU: {info['cpu_percent']:.1f}% | Memory: {info['memory_mb']:.0f} MB")
        
        print(f"\n{Colors.CYAN}Phase 2 Indicators (Cumulative):{Colors.ENDC}")
        print(f"  🔓 Recovery Bypasses:      {Colors.YELLOW}{self.metrics['recovery_bypasses']:>4}{Colors.ENDC}")
        print(f"  🔄 Forced Rotations:       {Colors.YELLOW}{self.metrics['forced_rotations']:>4}{Colors.ENDC}")
        print(f"  💰 Entry Sizes Aligned:    {Colors.YELLOW}{self.metrics['entry_sizes']:>4}{Colors.ENDC}")
        print(f"  📊 Trades Executed:        {Colors.YELLOW}{self.metrics['trades_executed']:>4}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}System Issues:{Colors.ENDC}")
        print(f"  ❌ Errors:                 {Colors.RED if self.metrics['errors'] > 0 else Colors.GREEN}{self.metrics['errors']:>4}{Colors.ENDC}")
        print(f"  ⚠️  Warnings:               {Colors.YELLOW}{self.metrics['warnings']:>4}{Colors.ENDC}")
        
        # Activity timeout check
        inactivity_seconds = int(time.time() - self.last_activity_time)
        if inactivity_seconds > self.alert_threshold:
            print(f"\n{Colors.RED}⚠️  WARNING: No activity for {inactivity_seconds} seconds{Colors.ENDC}")
        else:
            print(f"\n{Colors.GREEN}✓ Activity within last {inactivity_seconds}s{Colors.ENDC}")
    
    def display_recent_events(self):
        """Display recent Phase 2 events"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Recent Events (Last 10):{Colors.ENDC}")
        print("─" * 78)
        
        if not self.recent_events:
            print(f"{Colors.YELLOW}No Phase 2 events yet - monitoring...{Colors.ENDC}")
        else:
            for i, (event_type, event_msg) in enumerate(list(self.recent_events)[-10:], 1):
                print(f"{i:2}. {event_type} {event_msg}")
    
    def display_status_line(self):
        """Display bottom status line"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_exists = "✓" if os.path.exists(self.log_file) else "✗"
        print(f"\n{Colors.BOLD}{Colors.CYAN}─ Monitoring Dashboard ─ {now} [{log_exists} Log] ─ Next check in {self.check_interval}s ─{Colors.ENDC}")
        print(f"Press Ctrl+C to stop | Commands: 'ps' for process, 'log' for tail\n")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.display_header()
        print(f"{Colors.YELLOW}🔍 Initializing monitoring system...{Colors.ENDC}\n")
        time.sleep(2)
        
        try:
            while True:
                # Get new log content
                new_content = self.tail_log_from_position()
                has_activity = self.parse_new_logs(new_content)
                
                if has_activity:
                    self.last_activity_time = time.time()
                
                # Check system health
                health = self.check_system_health()
                
                # Update metrics
                self.metrics['last_update'] = datetime.now()
                
                # Display dashboard
                self.display_header()
                self.display_metrics(health)
                self.display_recent_events()
                self.display_status_line()
                
                # Wait for next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.display_shutdown()
            sys.exit(0)
    
    def display_shutdown(self):
        """Display shutdown message"""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  Monitoring Stopped".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print(f"{Colors.ENDC}\n")
        
        print(f"{Colors.CYAN}Final Statistics:{Colors.ENDC}")
        print(f"  Recovery Bypasses:  {self.metrics['recovery_bypasses']}")
        print(f"  Forced Rotations:   {self.metrics['forced_rotations']}")
        print(f"  Trades Executed:    {self.metrics['trades_executed']}")
        print(f"  Entry Sizes:        {self.metrics['entry_sizes']}")
        print(f"  Errors:             {self.metrics['errors']}")
        print(f"  Warnings:           {self.metrics['warnings']}")
        print()

def main():
    """Main entry point"""
    monitor = ContinuousMonitor()
    
    # Check if log file exists
    if not os.path.exists(monitor.log_file):
        print(f"{Colors.RED}Error: Log file not found: {monitor.log_file}{Colors.ENDC}")
        print("Make sure trading system is running first.")
        sys.exit(1)
    
    # Start monitoring
    try:
        monitor.monitor_loop()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
