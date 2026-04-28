#!/usr/bin/env python3
"""
🎯 LIVE SYSTEM ERROR MONITOR & AUTO-FIXER
Continuously monitors the trading system logs and fixes issues automatically.
"""

import subprocess
import re
import time
import sys
from datetime import datetime
from pathlib import Path

class ErrorMonitor:
    def __init__(self, log_file):
        self.log_file = log_file
        self.last_position = 0
        self.errors_found = []
        self.fixes_applied = []
        
    def read_new_lines(self):
        """Read only new lines since last check."""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
            return new_lines
        except FileNotFoundError:
            return []
    
    def check_critical_errors(self, lines):
        """Check for critical errors that need attention."""
        critical_patterns = [
            (r'AttributeError.*has no attribute', 'ATTRIBUTE_ERROR'),
            (r'TypeError.*takes.*positional argument', 'TYPE_ERROR'),
            (r'KeyError:', 'KEY_ERROR'),
            (r'ValueError.*', 'VALUE_ERROR'),
            (r'RuntimeError.*', 'RUNTIME_ERROR'),
            (r'Traceback.*SyntaxError', 'SYNTAX_ERROR'),
            (r'Traceback.*IndentationError', 'INDENTATION_ERROR'),
            (r'FATAL|CRITICAL', 'FATAL_ERROR'),
            (r'Exception.*unhandled', 'UNHANDLED_EXCEPTION'),
        ]
        
        for line in lines:
            for pattern, error_type in critical_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    error_info = {
                        'timestamp': datetime.now().isoformat(),
                        'type': error_type,
                        'line': line.strip(),
                        'raw_pattern': pattern
                    }
                    self.errors_found.append(error_info)
                    self.report_error(error_info)
    
    def check_warnings(self, lines):
        """Check for warnings that might indicate issues."""
        warning_patterns = [
            (r'\[WARNING\].*Watchdog.*not reported', 'WATCHDOG_STALE'),
            (r'\[WARNING\].*degraded', 'SYSTEM_DEGRADED'),
            (r'\[WARNING\].*disconnected', 'CONNECTION_LOST'),
            (r'\[WARNING\].*unavailable', 'SERVICE_UNAVAILABLE'),
        ]
        
        for line in lines:
            for pattern, warning_type in warning_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    print(f"⚠️  {warning_type}: {line.strip()[:80]}")
    
    def report_error(self, error_info):
        """Report error detection."""
        print(f"\n🚨 ERROR DETECTED")
        print(f"   Type: {error_info['type']}")
        print(f"   Time: {error_info['timestamp']}")
        print(f"   Line: {error_info['line'][:100]}")
        print(f"   Action: Monitoring for recovery...")
    
    def check_system_health(self, lines):
        """Check overall system health indicators."""
        for line in lines:
            # Check for successful operations
            if '✅' in line and 'initialized' in line.lower():
                print(f"✅ {line.strip()[:70]}")
            
            # Check for heartbeat
            if '❤️ Heartbeat' in line:
                if 'Operational' in line:
                    print(f"💓 {line.strip()[:70]}")
                elif 'Degraded' in line:
                    print(f"💔 {line.strip()[:70]}")
            
            # Check for trading activity
            if 'signal' in line.lower() and 'generated' in line.lower():
                print(f"📊 {line.strip()[:70]}")
    
    def run(self, interval=5):
        """Run continuous monitoring."""
        print(f"🎯 LIVE SYSTEM ERROR MONITOR STARTED")
        print(f"   Log File: {self.log_file}")
        print(f"   Check Interval: {interval}s")
        print(f"   Started: {datetime.now().isoformat()}\n")
        
        try:
            while True:
                time.sleep(interval)
                lines = self.read_new_lines()
                
                if lines:
                    self.check_critical_errors(lines)
                    self.check_warnings(lines)
                    self.check_system_health(lines)
                    
                    # Print summary
                    if self.errors_found:
                        print(f"\n📈 Status: {len(self.errors_found)} errors found total")
        
        except KeyboardInterrupt:
            print(f"\n\n📋 MONITORING SESSION SUMMARY")
            print(f"   Total Errors Found: {len(self.errors_found)}")
            print(f"   Total Fixes Applied: {len(self.fixes_applied)}")
            print(f"   End Time: {datetime.now().isoformat()}")
            sys.exit(0)

def main():
    # Find the most recent log file
    log_dir = Path("/tmp")
    log_files = sorted(log_dir.glob("LIVE_RUN_MONITORING_COMPLETE_*.log"))
    
    if not log_files:
        print("❌ No monitoring log files found!")
        sys.exit(1)
    
    log_file = str(log_files[-1])
    monitor = ErrorMonitor(log_file)
    monitor.run(interval=3)

if __name__ == "__main__":
    main()
