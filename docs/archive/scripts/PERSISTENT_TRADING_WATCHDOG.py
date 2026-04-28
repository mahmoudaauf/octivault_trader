#!/usr/bin/env python3
"""
🔄 PERSISTENT TRADING WATCHDOG
==============================
Monitors the trading system and auto-restarts if it crashes.
Keeps system running 24/7 with continuous profit compounding.
"""

import subprocess
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

class PersistentTradingWatchdog:
    """Monitors and maintains persistent trading"""
    
    def __init__(self):
        self.project_root = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
        self.venv_bin = self.project_root / "venv" / "bin" / "python3"
        self.main_script = self.project_root / "🚀_LIVE_ED25519_TRADING.py"
        self.logs_dir = self.project_root / "logs"
        self.log_file = self.logs_dir / "persistent_watchdog.log"
        self.pid_file = self.logs_dir / "persistent_trading.pid"
        
        self.running = True
        self.process = None
        self.cycle_count = 0
        self.start_time = datetime.utcnow()
        self.restarts = 0
        
        os.environ['APPROVE_LIVE_TRADING'] = 'YES'
        os.environ['TRADING_MODE'] = 'live'
    
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signal"""
        self.log("\n⏹️  Shutdown signal received")
        self.running = False
        if self.process:
            self.process.terminate()
    
    def start_trading_system(self):
        """Start the trading system"""
        try:
            self.log(f"▶️  Starting trading system (Restart #{self.restarts + 1})...")
            
            self.process = subprocess.Popen(
                [str(self.venv_bin), str(self.main_script)],
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.restarts += 1
            self.log(f"✅ Trading system started (PID: {self.process.pid})")
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(self.process.pid))
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error starting trading system: {str(e)}")
            return False
    
    def monitor_trading_system(self):
        """Monitor the trading system output"""
        try:
            line_count = 0
            while self.running and self.process:
                line = self.process.stdout.readline()
                
                if not line:
                    # Process ended
                    break
                
                # Log trading activity
                line = line.rstrip()
                if line:
                    # Log every line to watchdog log
                    line_count += 1
                    
                    # Print to console for critical messages
                    if any(x in line for x in ['FILLED', 'BUY', 'SELL', 'ERROR', 'WARNING', 'TRADE']):
                        print(line)
                    
                    # Log to watchdog
                    with open(self.log_file, 'a') as f:
                        f.write(line + "\n")
            
            # Process ended
            return_code = self.process.wait()
            self.log(f"⚠️  Trading system ended with code: {return_code}")
            
        except Exception as e:
            self.log(f"❌ Error monitoring trading system: {str(e)}")
    
    def run_persistent(self):
        """Run the trading system persistently"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Ensure logs directory exists
        self.logs_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("🔄 PERSISTENT TRADING WATCHDOG - CONTINUOUS MODE")
        print("="*70)
        print()
        print("📊 System Configuration:")
        print(f"  • Project: {self.project_root.name}")
        print(f"  • Python: {self.venv_bin}")
        print(f"  • Logs: {self.log_file}")
        print(f"  • Mode: LIVE (Real Money)")
        print(f"  • Started: {self.start_time.isoformat()}")
        print()
        print("🚀 Starting persistent trading with auto-restart...")
        print()
        
        restart_delay = 5  # seconds between restarts
        
        while self.running:
            self.cycle_count += 1
            
            # Start trading system
            if not self.start_trading_system():
                self.log("Failed to start trading system, retrying...")
                time.sleep(restart_delay)
                continue
            
            # Monitor it
            self.monitor_trading_system()
            
            if not self.running:
                break
            
            # System crashed, prepare for restart
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            self.log(f"📊 Statistics after restart #{self.restarts}:")
            self.log(f"   Uptime: {uptime:.0f} seconds ({uptime/3600:.1f} hours)")
            self.log(f"   Restart delay: {restart_delay} seconds")
            
            # Wait before restart
            self.log(f"⏳ Restarting in {restart_delay} seconds...")
            time.sleep(restart_delay)
        
        # Shutdown
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.log("\n🔌 Shutting down persistent watchdog...")
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        hours = uptime / 3600
        
        self.log(f"📊 Final Statistics:")
        self.log(f"   Total uptime: {uptime:.0f} seconds ({hours:.1f} hours)")
        self.log(f"   Total restarts: {self.restarts}")
        self.log(f"   Cycles completed: {self.cycle_count}")
        
        if self.process:
            self.process.terminate()
        
        self.log("✅ Watchdog shutdown complete")


def main():
    """Main entry point"""
    try:
        watchdog = PersistentTradingWatchdog()
        watchdog.run_persistent()
    except KeyboardInterrupt:
        print("\n⏹️  Watchdog interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
