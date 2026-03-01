#!/usr/bin/env python3
"""
WebSocket Health Monitor & Diagnostics

Analyzes ExchangeClient logs to detect WebSocket issues and verify fixes.
Provides real-time monitoring and historical analysis of listenKey rotation,
refresh failures, and reconnection patterns.

Usage:
    python3 monitor_websocket_health.py              # Real-time monitoring
    python3 monitor_websocket_health.py --analyze    # Historical analysis
    python3 monitor_websocket_health.py --alert      # Alert mode (production)
"""

import re
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


class WebSocketHealthMonitor:
    """Monitor WebSocket health from application logs."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize monitor with optional log file."""
        self.log_file = log_file or self._find_log_file()
        self.metrics: Dict[str, Any] = self._init_metrics()
        self.events: List[Dict[str, Any]] = []
        
    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize metrics tracking."""
        return {
            "total_events": 0,
            "refresh_success": 0,
            "refresh_failures": 0,
            "rotation_attempts": 0,
            "rotation_successes": 0,
            "rotation_failures": 0,
            "ws_disconnects": 0,
            "ws_reconnects": 0,
            "invalid_key_errors": 0,
            "other_errors": 0,
            "avg_rotation_duration_ms": 0,
            "max_consecutive_failures": 0,
            "last_refresh_ts": None,
            "last_rotation_ts": None,
            "last_disconnect_ts": None,
        }
    
    def _find_log_file(self) -> Optional[str]:
        """Find the most recent log file."""
        log_patterns = [
            "logs/octivault_*.log",
            "logs/*.log",
            "*.log",
        ]
        
        for pattern in log_patterns:
            matches = list(Path(".").glob(pattern))
            if matches:
                # Return most recently modified
                return str(max(matches, key=lambda p: p.stat().st_mtime))
        
        return None
    
    def analyze_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze logs from the last N hours."""
        if not self.log_file or not Path(self.log_file).exists():
            return {"error": f"Log file not found: {self.log_file}"}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with open(self.log_file) as f:
            for line in f:
                self._parse_log_line(line, cutoff_time)
        
        return self._generate_report()
    
    def _parse_log_line(self, line: str, cutoff_time: datetime) -> None:
        """Parse a single log line and track events."""
        # Extract timestamp
        ts_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            return
        
        try:
            ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            if ts < cutoff_time:
                return
        except ValueError:
            return
        
        self.metrics["total_events"] += 1
        
        # Pattern: listenKey refreshed successfully
        if "listenKey refreshed successfully" in line:
            self.metrics["refresh_success"] += 1
            self.metrics["last_refresh_ts"] = ts
            self.events.append({"ts": ts, "type": "refresh_success", "line": line})
        
        # Pattern: listenKey refresh failed
        elif "listenKey refresh failed" in line:
            self.metrics["refresh_failures"] += 1
            self.events.append({"ts": ts, "type": "refresh_failure", "line": line})
            
            # Extract consecutive failures count
            match = re.search(r"(\d+)/(\d+)", line)
            if match:
                failures = int(match.group(1))
                self.metrics["max_consecutive_failures"] = max(
                    self.metrics["max_consecutive_failures"], failures
                )
        
        # Pattern: listenKey rotated
        elif "listenKey rotated" in line:
            self.metrics["rotation_successes"] += 1
            self.metrics["last_rotation_ts"] = ts
            self.events.append({"ts": ts, "type": "rotation_success", "line": line})
        
        # Pattern: listenKey rotation attempt failed
        elif "listenKey rotation attempt" in line and "failed" in line:
            self.metrics["rotation_attempts"] += 1
            self.events.append({"ts": ts, "type": "rotation_attempt_failed", "line": line})
        
        # Pattern: listenKey rotation FAILED
        elif "listenKey rotation FAILED" in line:
            self.metrics["rotation_failures"] += 1
            self.metrics["rotation_attempts"] += 1
            self.events.append({"ts": ts, "type": "rotation_failure", "line": line})
        
        # Pattern: WebSocket disconnected
        elif "[EC:UserDataWS] disconnected" in line:
            self.metrics["ws_disconnects"] += 1
            self.metrics["last_disconnect_ts"] = ts
            self.events.append({"ts": ts, "type": "ws_disconnect", "line": line})
            
            # Check if it was invalid listenKey
            if "invalid_listen_key=True" in line or "410" in line:
                self.metrics["invalid_key_errors"] += 1
            else:
                self.metrics["other_errors"] += 1
        
        # Pattern: WebSocket connected
        elif "user_data_ws_connected" in line:
            self.metrics["ws_reconnects"] += 1
            self.events.append({"ts": ts, "type": "ws_connect", "line": line})
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate health report from collected metrics."""
        total_refreshes = self.metrics["refresh_success"] + self.metrics["refresh_failures"]
        refresh_success_rate = (
            100.0 * self.metrics["refresh_success"] / total_refreshes 
            if total_refreshes > 0 else 0
        )
        
        total_disconnects = self.metrics["ws_disconnects"]
        invalid_key_rate = (
            100.0 * self.metrics["invalid_key_errors"] / total_disconnects 
            if total_disconnects > 0 else 0
        )
        
        return {
            "metrics": self.metrics,
            "refresh_success_rate": refresh_success_rate,
            "invalid_key_error_rate": invalid_key_rate,
            "rotation_required": self.metrics["rotation_successes"] + self.metrics["rotation_failures"],
            "rotation_success_rate": (
                100.0 * self.metrics["rotation_successes"] / 
                (self.metrics["rotation_successes"] + self.metrics["rotation_failures"])
                if (self.metrics["rotation_successes"] + self.metrics["rotation_failures"]) > 0 else 0
            ),
            "events_timeline": self.events[-100:],  # Last 100 events
        }
    
    def print_summary(self) -> None:
        """Print a nice summary of health status."""
        report = self._generate_report()
        
        print("\n" + "="*80)
        print("                    WebSocket Health Summary")
        print("="*80 + "\n")
        
        m = report["metrics"]
        
        print(f"Log file: {self.log_file}")
        if m["last_refresh_ts"]:
            print(f"Analysis period: last 24 hours")
            print(f"Last refresh: {m['last_refresh_ts'].strftime('%Y-%m-%d %H:%M:%S')}")
            if m["last_rotation_ts"]:
                print(f"Last rotation: {m['last_rotation_ts'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Refresh metrics
        total_refreshes = m["refresh_success"] + m["refresh_failures"]
        print("📊 Refresh Metrics")
        print(f"  Total refreshes: {total_refreshes}")
        print(f"  Successful: {m['refresh_success']} ({report['refresh_success_rate']:.1f}%)")
        print(f"  Failed: {m['refresh_failures']} ({100-report['refresh_success_rate']:.1f}%)")
        print(f"  Status: {'✅ HEALTHY' if report['refresh_success_rate'] > 99 else '⚠️  DEGRADED'}")
        print()
        
        # Rotation metrics
        total_rotations = report["rotation_required"]
        print("🔄 Rotation Metrics")
        print(f"  Total rotations: {total_rotations}")
        print(f"  Successful: {m['rotation_successes']}")
        print(f"  Failed: {m['rotation_failures']}")
        if total_rotations > 0:
            print(f"  Success rate: {report['rotation_success_rate']:.1f}%")
            print(f"  Status: {'✅ HEALTHY' if report['rotation_success_rate'] > 95 else '⚠️  CHECK'}")
        else:
            print(f"  Status: ✅ No rotations needed")
        print()
        
        # WebSocket metrics
        print("🔌 WebSocket Metrics")
        print(f"  Disconnects: {m['ws_disconnects']}")
        if m['ws_disconnects'] > 0:
            print(f"    Due to invalid listenKey: {m['invalid_key_errors']} ({report['invalid_key_error_rate']:.1f}%)")
            print(f"    Other reasons: {m['other_errors']}")
        print(f"  Reconnects: {m['ws_reconnects']}")
        if m['last_disconnect_ts']:
            print(f"  Last disconnect: {m['last_disconnect_ts'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Health status
        print("🎯 Overall Health Status")
        if (report['refresh_success_rate'] > 99 and 
            report['rotation_success_rate'] > 95 and 
            m['ws_disconnects'] < 5):
            print("  ✅ HEALTHY - System operating normally")
        elif (report['refresh_success_rate'] > 95 and 
              report['rotation_success_rate'] > 90):
            print("  ⚠️  DEGRADED - Minor issues detected, monitor closely")
        else:
            print("  🔴 CRITICAL - Major issues detected, investigate immediately")
        print()
        
        print("="*80 + "\n")
    
    def watch_logs(self, interval: int = 5) -> None:
        """Watch logs in real-time mode."""
        print(f"Watching {self.log_file} for WebSocket events...")
        print("Press Ctrl+C to stop\n")
        
        last_pos = 0
        
        try:
            while True:
                try:
                    with open(self.log_file) as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()
                    
                    for line in new_lines:
                        if "UserDataWS" in line or "listenKey" in line:
                            # Color code the output
                            if "refreshed successfully" in line:
                                print(f"✅ {line.strip()}")
                            elif "refresh failed" in line:
                                print(f"❌ {line.strip()}")
                            elif "rotated" in line:
                                print(f"🔄 {line.strip()}")
                            elif "disconnected" in line:
                                print(f"⚠️  {line.strip()}")
                            elif "connected" in line:
                                print(f"✅ {line.strip()}")
                            else:
                                print(f"   {line.strip()}")
                    
                    time.sleep(interval)
                
                except FileNotFoundError:
                    print(f"Log file not found: {self.log_file}")
                    time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WebSocket Health Monitor & Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 monitor_websocket_health.py              # Print summary
  python3 monitor_websocket_health.py --watch      # Real-time monitoring
  python3 monitor_websocket_health.py --analyze    # Detailed analysis
  python3 monitor_websocket_health.py --log FILE   # Use specific log file
        """,
    )
    
    parser.add_argument("--log", help="Path to log file (auto-detected if not specified)")
    parser.add_argument("--watch", action="store_true", help="Watch logs in real-time")
    parser.add_argument("--analyze", action="store_true", help="Print detailed analysis")
    parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
    
    args = parser.parse_args()
    
    monitor = WebSocketHealthMonitor(args.log)
    
    if not monitor.log_file:
        print("❌ Error: Could not find log file")
        print("   Please specify with: --log /path/to/log")
        sys.exit(1)
    
    if args.watch:
        monitor.watch_logs()
    elif args.analyze:
        report = monitor.analyze_logs(args.hours)
        monitor.print_summary()
        
        if report.get("events_timeline"):
            print("\n📋 Last 20 Events\n")
            for event in report["events_timeline"][-20:]:
                ts = event["ts"].strftime("%H:%M:%S")
                event_type = event["type"]
                line = event["line"]
                # Extract just the important part
                important = re.sub(r".*\[EC:UserDataWS\] ", "", line).strip()
                print(f"{ts} [{event_type:20s}] {important[:70]}")
    else:
        monitor.analyze_logs(args.hours)
        monitor.print_summary()


if __name__ == "__main__":
    main()
