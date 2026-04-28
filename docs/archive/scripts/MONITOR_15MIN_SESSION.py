#!/usr/bin/env python3
"""
🔍 15-MINUTE SYSTEM MONITORING SESSION
========================================

Runs the trading system for 15 minutes and provides status updates every 2 minutes.

USAGE:
    export TRADING_DURATION_HOURS=0.25  (15 minutes)
    export APPROVE_LIVE_TRADING=YES      (if live trading)
    python3 MONITOR_15MIN_SESSION.py [--paper] [--live]

Flags:
    --paper     Use paper trading mode
    --live      Use live trading mode (requires APPROVE_LIVE_TRADING=YES)
    --testnet   Use testnet mode
"""

import asyncio
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import threading
from queue import Queue

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for monitor"""
    log_dir = Path("/tmp")
    log_file = log_dir / "monitor_15min_session.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# MONITORING CLASS
# ============================================================================

class SystemMonitor:
    """Monitor system health and performance"""
    
    def __init__(self, duration_minutes: int = 15, check_interval_seconds: int = 120):
        self.duration_minutes = duration_minutes
        self.check_interval_seconds = check_interval_seconds
        self.start_time = None
        self.end_time = None
        self.process = None
        self.monitoring = True
        self.status_history = []
        self.output_queue = Queue()
        self.orchestrator_pid = None
        
    def get_elapsed_time(self) -> str:
        """Get elapsed time as formatted string"""
        if self.start_time is None:
            return "Not started"
        elapsed = time.time() - self.start_time
        return f"{int(elapsed)}s / {self.duration_minutes * 60}s"
    
    def get_remaining_time(self) -> str:
        """Get remaining time as formatted string"""
        if self.start_time is None:
            return f"{self.duration_minutes * 60}s"
        elapsed = time.time() - self.start_time
        remaining = (self.duration_minutes * 60) - elapsed
        if remaining <= 0:
            return "COMPLETE"
        return f"{int(remaining)}s"
    
    def read_log_tail(self, log_file: str, lines: int = 5) -> str:
        """Read last N lines from log file"""
        try:
            result = subprocess.run(
                f"tail -n {lines} {log_file}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error reading log: {e}"
    
    def get_process_status(self) -> Dict[str, Any]:
        """Get current process status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "elapsed": self.get_elapsed_time(),
            "remaining": self.get_remaining_time(),
            "process_running": self.process is not None and self.process.poll() is None,
            "pid": self.orchestrator_pid,
        }
        
        # Try to get process info
        if self.orchestrator_pid:
            try:
                result = subprocess.run(
                    f"ps aux | grep {self.orchestrator_pid} | grep -v grep",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                status["process_info"] = result.stdout.strip()
            except:
                status["process_info"] = "Could not retrieve"
        
        # Get recent log entries
        orchestrator_log = Path("/tmp/octivault_master_orchestrator.log")
        if orchestrator_log.exists():
            status["recent_logs"] = self.read_log_tail(str(orchestrator_log), 3)
        
        return status
    
    def print_status_header(self, check_number: int):
        """Print status header"""
        print("\n" + "="*80)
        print(f"📊 STATUS CHECK #{check_number} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
    
    def print_status(self, status: Dict[str, Any], check_number: int):
        """Print formatted status"""
        self.print_status_header(check_number)
        
        print(f"⏱️  Elapsed:          {status['elapsed']}")
        print(f"⏳ Remaining:         {status['remaining']}")
        print(f"🔄 Process Running:   {'✅ YES' if status['process_running'] else '❌ NO'}")
        print(f"📌 PID:              {status['pid'] or 'N/A'}")
        
        if status.get('process_info'):
            print(f"\n📋 Process Info:")
            print(f"   {status['process_info'][:100]}")
        
        if status.get('recent_logs'):
            print(f"\n📝 Recent Logs:")
            for line in status['recent_logs'].split('\n')[-3:]:
                if line.strip():
                    print(f"   {line[:110]}")
        
        print("="*80)
    
    def reader_thread(self):
        """Read process output in background"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_queue.put(('stdout', line.rstrip()))
            for line in iter(self.process.stderr.readline, ''):
                if line:
                    self.output_queue.put(('stderr', line.rstrip()))
        except:
            pass
    
    async def run_monitoring_session(self):
        """Main monitoring session"""
        logger.info("🚀 Starting 15-minute monitoring session")
        print("\n" + "="*80)
        print("🚀 OCTI AI TRADING BOT - 15 MINUTE MONITORING SESSION")
        print("="*80)
        print(f"Duration: {self.duration_minutes} minutes")
        print(f"Check Interval: {self.check_interval_seconds} seconds")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Set up environment
        env = os.environ.copy()
        env['TRADING_DURATION_HOURS'] = "0.25"  # 15 minutes
        if '--paper' in sys.argv:
            env['PAPER_TRADING'] = 'true'
            print("📄 Mode: PAPER TRADING")
        elif '--testnet' in sys.argv:
            env['TESTNET_MODE'] = 'true'
            env['PAPER_TRADING'] = 'true'
            print("🔬 Mode: TESTNET")
        else:
            print("🔴 Mode: LIVE TRADING")
        print("="*80 + "\n")
        
        # Start orchestrator
        try:
            self.start_time = time.time()
            self.end_time = self.start_time + (self.duration_minutes * 60)
            
            orchestrator_path = Path(__file__).parent / "🎯_MASTER_SYSTEM_ORCHESTRATOR.py"
            
            self.process = subprocess.Popen(
                [sys.executable, str(orchestrator_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1
            )
            
            self.orchestrator_pid = self.process.pid
            logger.info(f"✅ Started orchestrator process with PID {self.orchestrator_pid}")
            print(f"✅ Orchestrator started (PID: {self.orchestrator_pid})\n")
            
            # Start reader thread
            reader_th = threading.Thread(target=self.reader_thread, daemon=True)
            reader_th.start()
            
        except Exception as e:
            logger.error(f"❌ Failed to start orchestrator: {e}")
            print(f"❌ Failed to start orchestrator: {e}")
            return False
        
        # Main monitoring loop
        check_number = 1
        last_check_time = time.time()
        
        try:
            while time.time() < self.end_time:
                current_time = time.time()
                
                # Check if it's time for status update
                if current_time - last_check_time >= self.check_interval_seconds:
                    status = self.get_process_status()
                    self.print_status(status, check_number)
                    self.status_history.append(status)
                    check_number += 1
                    last_check_time = current_time
                
                # Check if process has died
                if self.process.poll() is not None:
                    logger.warning("⚠️ Process terminated early")
                    print("⚠️ Process terminated early!")
                    break
                
                # Check for output messages
                try:
                    msg_type, msg = self.output_queue.get(timeout=0.5)
                    # Only print error or warning messages
                    if 'ERROR' in msg or 'WARNING' in msg or 'CRITICAL' in msg:
                        print(f"   [{msg_type}] {msg[:120]}")
                except:
                    pass
                
                await asyncio.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("⚠️ Monitoring interrupted by user")
            print("\n⚠️ Monitoring interrupted by user")
        
        # Print final status
        print("\n" + "="*80)
        print("🏁 FINAL STATUS REPORT")
        print("="*80)
        
        final_status = self.get_process_status()
        self.print_status(final_status, check_number)
        
        # Terminate process if still running
        if self.process and self.process.poll() is None:
            print("🛑 Stopping orchestrator...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("✅ Orchestrator stopped")
        
        # Summary
        print("\n" + "="*80)
        print("📈 MONITORING SESSION SUMMARY")
        print("="*80)
        print(f"Total Checks: {len(self.status_history)}")
        print(f"Duration: {self.duration_minutes} minutes")
        print(f"Session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        return True

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    monitor = SystemMonitor(duration_minutes=15, check_interval_seconds=120)
    success = await monitor.run_monitoring_session()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
