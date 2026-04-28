#!/usr/bin/env python3
"""
Phase 2 Monitoring Dashboard

Real-time monitoring of Phase 2 paper trading execution.

Usage:
    python3 phase2_monitoring.py
    
Or in a separate terminal while Phase 2 is running:
    tail -f /tmp/octivault_trader.log | grep METRICS
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import time

class Phase2Monitor:
    """Monitor Phase 2 execution in real-time"""
    
    def __init__(self, log_file: str = "/tmp/octivault_trader.log"):
        self.log_file = Path(log_file)
        self.last_line = 0
        self.metrics: Dict[str, any] = {
            "start_time": None,
            "elapsed_hours": 0,
            "orders_attempted": 0,
            "orders_succeeded": 0,
            "orders_failed": 0,
            "success_rate": 0,
            "risk_violations": 0,
            "system_errors": 0,
            "last_update": None,
        }
    
    def read_new_lines(self) -> list:
        """Read new lines from log file"""
        try:
            if not self.log_file.exists():
                return []
            
            with open(self.log_file, 'r') as f:
                current_line = 0
                lines = []
                for line_num, line in enumerate(f, 1):
                    if line_num > self.last_line:
                        lines.append(line.rstrip())
                        current_line = line_num
                
                self.last_line = current_line
                return lines
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def parse_metrics(self, line: str) -> bool:
        """Parse METRICS line from log"""
        if "METRICS |" not in line:
            return False
        
        try:
            # Extract metrics from log line
            # Format: METRICS | Elapsed: 0.1h | Orders: 5 | Success: 4 (80.0%) | Failed: 1 | Risk Violations: 0 | Errors: 0
            
            patterns = {
                "elapsed_hours": r"Elapsed: ([\d.]+)h",
                "orders_attempted": r"Orders: (\d+)",
                "orders_succeeded": r"Success: (\d+)",
                "success_rate": r"\(([\d.]+)%\)",
                "orders_failed": r"Failed: (\d+)",
                "risk_violations": r"Risk Violations: (\d+)",
                "system_errors": r"Errors: (\d+)",
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = match.group(1)
                    if key == "success_rate":
                        self.metrics[key] = float(value)
                    elif key == "elapsed_hours":
                        self.metrics[key] = float(value)
                    else:
                        self.metrics[key] = int(value)
            
            self.metrics["last_update"] = datetime.now()
            return True
        except Exception as e:
            print(f"Error parsing metrics: {e}")
            return False
    
    def check_phase_start(self, lines: list):
        """Check for Phase 2 start"""
        for line in lines:
            if "PHASE 2: PAPER TRADING VALIDATION" in line:
                self.metrics["start_time"] = datetime.now()
                print("\n" + "=" * 80)
                print("🚀 PHASE 2 PAPER TRADING STARTED")
                print("=" * 80 + "\n")
                return True
        return False
    
    def check_phase_end(self, lines: list) -> Optional[Dict]:
        """Check for Phase 2 completion and extract final report"""
        for line in lines:
            if "PHASE 2 SHUTDOWN" in line:
                print("\n" + "=" * 80)
                print("⏹️  PHASE 2 COMPLETED")
                print("=" * 80)
                return self.metrics
        return None
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        print("\033[2J\033[H")  # Clear screen
        print("=" * 80)
        print("📊 PHASE 2 MONITORING DASHBOARD")
        print("=" * 80)
        print()
        
        # Status
        status = "🟢 RUNNING" if self.metrics["last_update"] else "🔴 IDLE"
        print(f"Status: {status}")
        if self.metrics["start_time"]:
            print(f"Start Time: {self.metrics['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Performance Metrics
        print("📈 PERFORMANCE METRICS")
        print(f"  Elapsed Time: {self.metrics['elapsed_hours']:.2f} hours")
        print(f"  Orders Attempted: {self.metrics['orders_attempted']}")
        print(f"  Orders Succeeded: {self.metrics['orders_succeeded']}")
        print(f"  Orders Failed: {self.metrics['orders_failed']}")
        print(f"  Success Rate: {self.metrics['success_rate']:.2f}%")
        print()
        
        # Risk & Errors
        print("⚠️  RISK & ERRORS")
        print(f"  Risk Violations: {self.metrics['risk_violations']}")
        print(f"  System Errors: {self.metrics['system_errors']}")
        print()
        
        # Phase 2 Gate Status
        print("🎯 PHASE 2 GATE STATUS")
        checks = {
            "Success Rate ≥95%": self.metrics["success_rate"] >= 95,
            "Risk Violations = 0": self.metrics["risk_violations"] == 0,
            "System Errors < 5": self.metrics["system_errors"] < 5,
            "Orders Attempted > 0": self.metrics["orders_attempted"] > 0,
        }
        
        for check, passed in checks.items():
            status_icon = "✅" if passed else "❌"
            print(f"  {status_icon} {check}")
        print()
        
        # Last update
        if self.metrics["last_update"]:
            print(f"Last Update: {self.metrics['last_update'].strftime('%H:%M:%S')}")
        print("=" * 80)
    
    async def run_continuous(self):
        """Run continuous monitoring"""
        print("Starting Phase 2 Monitor...")
        print(f"Watching log file: {self.log_file}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                lines = self.read_new_lines()
                
                for line in lines:
                    self.check_phase_start([line])
                    self.parse_metrics(line)
                    end_report = self.check_phase_end([line])
                    
                    if end_report:
                        self.display_dashboard()
                        print("\n✅ Phase 2 execution completed!")
                        return
                
                if self.metrics["last_update"]:
                    self.display_dashboard()
                
                await asyncio.sleep(2)  # Update every 2 seconds
        
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")
            self.display_dashboard()


async def main():
    """Main entry point"""
    monitor = Phase2Monitor()
    await monitor.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
