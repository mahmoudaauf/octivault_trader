#!/usr/bin/env python3
"""
🤖 AUTONOMOUS SYSTEM RUNNER - PRODUCTION READY
==============================================
Runs the proven live trading system autonomously with auto-restart on errors.
"""

import os
import sys
import subprocess
import time
import signal
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
VENV_BIN = PROJECT_ROOT / "venv" / "bin" / "python3"
LOG_DIR = PROJECT_ROOT / "logs"
MAIN_SCRIPT = PROJECT_ROOT / "🚀_LIVE_ED25519_TRADING.py"

# Ensure log dir exists
LOG_DIR.mkdir(exist_ok=True)

class AutonomousRunner:
    """Run the trading system autonomously with auto-restart"""
    
    def __init__(self):
        self.running = True
        self.cycle_count = 0
        self.process = None
        self.start_time = datetime.utcnow()
        
    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        print("\n\n" + "="*80)
        print("⏹️  SHUTDOWN SIGNAL RECEIVED")
        print("="*80)
        self.running = False
        if self.process:
            self.process.terminate()
        print(f"Uptime: {(datetime.utcnow() - self.start_time).total_seconds():.0f}s")
        print(f"Cycles: {self.cycle_count}")
        sys.exit(0)
    
    async def run_forever(self):
        """Run the trading system with auto-restart"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        os.environ['APPROVE_LIVE_TRADING'] = 'YES'
        os.environ['TRADING_MODE'] = 'live'
        
        print("\n" + "="*80)
        print("🤖 OCTIVAULT AUTONOMOUS TRADING SYSTEM")
        print("="*80)
        print(f"Start Time: {self.start_time.isoformat()}")
        print(f"Script: {MAIN_SCRIPT.name}")
        print(f"Python: {VENV_BIN}")
        print(f"Logs: {LOG_DIR}")
        print("="*80 + "\n")
        
        restart_delay = 5  # seconds
        max_consecutive_errors = 5
        error_count = 0
        
        while self.running:
            self.cycle_count += 1
            cycle_start = datetime.utcnow()
            
            print(f"\n{'='*80}")
            print(f"🔄 STARTUP CYCLE #{self.cycle_count} - {cycle_start.isoformat()}")
            print(f"{'='*80}\n")
            
            try:
                # Start the process
                self.process = subprocess.Popen(
                    [str(VENV_BIN), str(MAIN_SCRIPT)],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Monitor output
                print("📊 Live Output:")
                print("-" * 80)
                
                while True:
                    if not self.running:
                        break
                    
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    
                    print(line.rstrip())
                
                # Wait for process to complete
                returncode = self.process.wait()
                error_count = 0  # Reset on successful run
                
                if not self.running:
                    break
                
                print(f"\n⏱️  Process ended with code: {returncode}")
                print(f"⏳ Restarting in {restart_delay}s...")
                time.sleep(restart_delay)
                
            except KeyboardInterrupt:
                print("\n⏹️  Interrupted by user")
                self.running = False
                break
            except Exception as e:
                error_count += 1
                print(f"\n❌ Error (#{error_count}): {str(e)}")
                
                if error_count >= max_consecutive_errors:
                    print(f"❌ Max consecutive errors ({max_consecutive_errors}) reached!")
                    break
                
                print(f"⏳ Retrying in {restart_delay}s...")
                time.sleep(restart_delay)
        
        print("\n" + "="*80)
        print("✅ AUTONOMOUS SYSTEM STOPPED")
        print("="*80)


def main():
    """Entry point"""
    import asyncio
    runner = AutonomousRunner()
    try:
        asyncio.run(runner.run_forever())
    except KeyboardInterrupt:
        print("\n⏹️  System stopped")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
