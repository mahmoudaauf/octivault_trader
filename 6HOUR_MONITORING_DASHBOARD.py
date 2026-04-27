#!/usr/bin/env python3
"""
6-Hour Assessment Monitoring Dashboard
Tracks system metrics, trades, and performance every 5 minutes
"""
import asyncio
import json
import os
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
LOG_FILE = "/tmp/octivault_master_orchestrator.log"
MONITORING_INTERVAL = 300  # 5 minutes
SESSION_START = None
SESSION_DURATION = 6 * 3600  # 6 hours in seconds
METRICS_FILE = "/tmp/6hour_session_metrics.json"
CHECKPOINT_MARKERS = [1, 2, 3, 4, 5, 6]  # Hours to mark


class MonitoringDashboard:
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics_history = []
        self.checkpoint_times = {}
        self.orchestrator_pid = None
        
    def find_orchestrator_pid(self):
        """Find the orchestrator process"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "MASTER_SYSTEM_ORCHESTRATOR"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pid = int(result.stdout.strip().split('\n')[0])
                return pid
        except Exception as e:
            print(f"❌ Could not find orchestrator PID: {e}")
        return None

    def get_system_metrics(self):
        """Collect system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
        
        if not self.orchestrator_pid:
            self.orchestrator_pid = self.find_orchestrator_pid()
            
        if self.orchestrator_pid:
            try:
                proc = psutil.Process(self.orchestrator_pid)
                metrics["cpu_percent"] = proc.cpu_percent(interval=1)
                metrics["memory_mb"] = proc.memory_info().rss / (1024 * 1024)
                metrics["memory_percent"] = proc.memory_percent()
                metrics["num_threads"] = proc.num_threads()
                metrics["status"] = "running"
            except psutil.NoSuchProcess:
                metrics["status"] = "crashed"
                self.orchestrator_pid = None
            except Exception as e:
                metrics["status"] = f"error: {e}"
        else:
            metrics["status"] = "not_found"
            
        return metrics

    def extract_log_metrics(self):
        """Extract key metrics from log file"""
        metrics = {
            "trades_executed": 0,
            "dust_healed": 0,
            "recovery_activations": 0,
            "errors": 0,
            "last_trade": None,
            "last_error": None,
        }
        
        if not os.path.exists(LOG_FILE):
            return metrics
            
        try:
            # Read last 500 lines to avoid processing entire file
            result = subprocess.run(
                ["tail", "-500", LOG_FILE],
                capture_output=True,
                text=True
            )
            lines = result.stdout.split('\n')
            
            for line in reversed(lines):
                if "TRADE EXECUTED" in line or "trade_executed" in line.lower():
                    metrics["trades_executed"] += 1
                    if not metrics["last_trade"]:
                        metrics["last_trade"] = line[:100]
                        
                if "dust" in line.lower() and ("heal" in line.lower() or "consolidated" in line.lower()):
                    metrics["dust_healed"] += 1
                    
                if "recovery" in line.lower() and "active" in line.lower():
                    metrics["recovery_activations"] += 1
                    
                if "ERROR" in line or "CRITICAL" in line:
                    metrics["errors"] += 1
                    if not metrics["last_error"]:
                        metrics["last_error"] = line[:100]
        except Exception as e:
            print(f"⚠️  Could not extract log metrics: {e}")
            
        return metrics

    def display_metrics(self, system_metrics, log_metrics, elapsed_time):
        """Display metrics in dashboard format"""
        elapsed_hours = elapsed_time / 3600
        remaining_hours = 6 - elapsed_hours
        
        print("\n" + "=" * 80)
        print(f"📊 6-HOUR ASSESSMENT MONITORING DASHBOARD")
        print(f"⏱️  Session Time: {elapsed_hours:.2f}h / 6h (Remaining: {remaining_hours:.2f}h)")
        print("=" * 80)
        
        # System Status
        print(f"\n🖥️  SYSTEM METRICS:")
        print(f"   Status: {system_metrics.get('status', 'unknown')}")
        if system_metrics.get('status') == 'running':
            print(f"   CPU: {system_metrics.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {system_metrics.get('memory_mb', 0):.0f} MB ({system_metrics.get('memory_percent', 0):.1f}%)")
            print(f"   Threads: {system_metrics.get('num_threads', 0)}")
        
        # Trade Metrics
        print(f"\n💹 TRADE METRICS:")
        print(f"   Trades Executed: {log_metrics.get('trades_executed', 0)}")
        print(f"   Dust Healed: {log_metrics.get('dust_healed', 0)}")
        print(f"   Recovery Activations: {log_metrics.get('recovery_activations', 0)}")
        
        # Health
        print(f"\n⚕️  SYSTEM HEALTH:")
        print(f"   Errors: {log_metrics.get('errors', 0)}")
        if log_metrics.get('last_error'):
            print(f"   Last Error: {log_metrics['last_error']}")
        if log_metrics.get('last_trade'):
            print(f"   Last Trade: {log_metrics['last_trade'][:60]}...")
        
        print("\n" + "=" * 80)

    def check_checkpoint(self, elapsed_time):
        """Check if we've reached a checkpoint hour"""
        elapsed_hours = elapsed_time / 3600
        for hour in CHECKPOINT_MARKERS:
            if abs(elapsed_hours - hour) < 0.1 and hour not in self.checkpoint_times:
                self.checkpoint_times[hour] = datetime.now()
                print(f"\n✅ CHECKPOINT: {hour}-Hour mark reached!")
                return True
        return False

    def save_metrics(self):
        """Save metrics history to file"""
        try:
            with open(METRICS_FILE, 'w') as f:
                json.dump({
                    "start_time": self.start_time.isoformat(),
                    "monitoring_duration": (datetime.now() - self.start_time).total_seconds(),
                    "checkpoints": {str(h): t.isoformat() for h, t in self.checkpoint_times.items()},
                    "metrics_count": len(self.metrics_history),
                    "latest_metrics": self.metrics_history[-1] if self.metrics_history else None,
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save metrics: {e}")

    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        print(f"\n🚀 Starting 6-hour monitoring session...")
        print(f"📝 Metrics saved to: {METRICS_FILE}")
        print(f"📋 Log file: {LOG_FILE}")
        
        iteration = 0
        while True:
            iteration += 1
            
            # Get metrics
            system_metrics = self.get_system_metrics()
            log_metrics = self.extract_log_metrics()
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            # Combine metrics
            combined = {**system_metrics, **log_metrics}
            self.metrics_history.append(combined)
            
            # Display
            self.display_metrics(system_metrics, log_metrics, elapsed_time)
            
            # Check checkpoint
            self.check_checkpoint(elapsed_time)
            
            # Save metrics
            self.save_metrics()
            
            # Check if session complete
            if elapsed_time >= SESSION_DURATION:
                print(f"\n✅ 6-HOUR SESSION COMPLETE!")
                print(f"📊 Total metrics collected: {len(self.metrics_history)}")
                print(f"📁 Metrics saved to: {METRICS_FILE}")
                break
            
            # Wait for next interval
            await asyncio.sleep(MONITORING_INTERVAL)


async def main():
    """Main entry point"""
    dashboard = MonitoringDashboard()
    try:
        await dashboard.run_monitoring_loop()
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring stopped by user")
        dashboard.save_metrics()


if __name__ == "__main__":
    asyncio.run(main())
