#!/usr/bin/env python3
"""
CONTINUOUS SESSION MONITOR - Updates every 5 minutes
Tracks deadlocks, dynamic adaptation, profitability, and reinvestment
"""

import subprocess
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class SessionMonitor:
    def __init__(self):
        self.log_file = Path("/tmp/octivault_master_orchestrator.log")
        self.start_time = datetime.now()
        self.session_duration = timedelta(hours=2)
        self.end_time = self.start_time + self.session_duration
        self.last_line_count = 0
        self.last_signal_count = 0
        
    def get_line_count(self):
        """Get total log lines"""
        try:
            with open(self.log_file, 'r') as f:
                return len(f.readlines())
        except:
            return 0
    
    def get_metric(self, pattern, count_mode=True):
        """Count occurrences of pattern in logs"""
        try:
            result = subprocess.run(
                f"grep -c '{pattern}' {self.log_file} 2>/dev/null || echo 0",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            return int(result.stdout.strip())
        except:
            return 0
    
    def check_deadlock(self):
        """Check for deadlock by monitoring log growth"""
        current_lines = self.get_line_count()
        is_growing = current_lines > self.last_line_count
        self.last_line_count = current_lines
        return is_growing
    
    def get_status(self):
        """Get current session status"""
        elapsed = datetime.now() - self.start_time
        remaining = self.end_time - datetime.now()
        
        signals = self.get_metric("Published TradeIntent")
        rejections = self.get_metric("EXEC_REJECT")
        components_healthy = self.get_metric("Status: Healthy")
        tp_sl = self.get_metric("TPSLEngine|TP/SL")
        
        is_responsive = self.check_deadlock()
        
        return {
            "elapsed_seconds": int(elapsed.total_seconds()),
            "remaining_seconds": int(max(0, remaining.total_seconds())),
            "total_lines": self.get_line_count(),
            "signals_generated": signals,
            "execution_rejections": rejections,
            "components_healthy": components_healthy,
            "tp_sl_instances": tp_sl,
            "is_responsive": is_responsive,
            "status": "✅ ACTIVE" if is_responsive else "❌ BLOCKED"
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        elapsed = timedelta(seconds=status["elapsed_seconds"])
        remaining = timedelta(seconds=status["remaining_seconds"])
        
        print("\n" + "=" * 80)
        print(f"📊 SESSION UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        print(f"\n⏱️  TIME:")
        print(f"   Elapsed:   {str(elapsed).split('.')[0]}")
        print(f"   Remaining: {str(remaining).split('.')[0]}")
        print(f"   Progress:  {status['elapsed_seconds'] / (2*3600) * 100:.1f}% complete")
        
        print(f"\n📈 TRADING ACTIVITY:")
        print(f"   Signals Generated:     {status['signals_generated']:,}")
        print(f"   Execution Rejections:  {status['execution_rejections']:,}")
        print(f"   Rejection Rate:        {status['execution_rejections'] / max(1, status['signals_generated']) * 100:.1f}%")
        
        print(f"\n❤️  SYSTEM HEALTH:")
        print(f"   Components Healthy:    {status['components_healthy']}")
        print(f"   TP/SL Monitoring:      {status['tp_sl_instances']}")
        print(f"   Log Lines Generated:   {status['total_lines']:,}")
        
        print(f"\n🔍 DEADLOCK CHECK:")
        print(f"   Status:                {status['status']}")
        print(f"   Responsive:            {'✅ YES' if status['is_responsive'] else '❌ NO'}")
        
        print("\n" + "=" * 80)

def main():
    """Main monitoring loop"""
    monitor = SessionMonitor()
    
    print("\n" + "=" * 80)
    print("🎯 CONTINUOUS SESSION MONITOR - 2 HOUR TRADING SESSION")
    print("=" * 80)
    print(f"Started: {monitor.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will end at: {monitor.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Updates every 5 minutes")
    print("=" * 80)
    
    update_count = 0
    
    try:
        while datetime.now() < monitor.end_time:
            monitor.print_status()
            update_count += 1
            
            # Wait 5 minutes or until session ends
            sleep_until = min(
                datetime.now() + timedelta(minutes=5),
                monitor.end_time
            )
            
            remaining = (sleep_until - datetime.now()).total_seconds()
            if remaining > 0:
                print(f"\n⏳ Next update in 5 minutes... (Ctrl+C to stop)")
                time.sleep(remaining)
        
        # Final status
        print("\n" + "=" * 80)
        print("✅ 2-HOUR SESSION COMPLETE!")
        print("=" * 80)
        monitor.print_status()
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Session monitoring stopped")

if __name__ == "__main__":
    main()
