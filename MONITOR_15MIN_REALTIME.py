#!/usr/bin/env python3
"""
Real-time 15-minute monitoring script with 2-minute status updates.
Starts the trading system and provides continuous performance feedback.
"""

import asyncio
import logging
import os
import sys
import signal
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import json
import re

# Setup
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeMonitor:
    """Monitor the trading system for 15 minutes with status updates every 2 minutes"""
    
    def __init__(self, duration_minutes=15, update_interval_seconds=120):
        self.duration = duration_minutes * 60  # Convert to seconds
        self.update_interval = update_interval_seconds
        self.start_time = None
        self.end_time = None
        self.process = None
        self.log_file = None
        self.metrics = {
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': [],
            'warnings': [],
            'last_update': None,
            'system_health': 'INITIALIZING',
            'pnl': 0.0,
            'portfolio_value': 0.0,
            'positions': 0,
        }
        self.last_line_index = 0
        
    def start_orchestrator(self):
        """Start the orchestrator process"""
        logger.info("🚀 Starting trading orchestrator...")
        
        # Set environment variables
        env = os.environ.copy()
        env['PAPER_TRADING'] = 'true'  # Use paper trading for safety
        env['TRADING_DURATION_HOURS'] = '0.5'  # Run for max 30 minutes
        env['APPROVE_LIVE_TRADING'] = 'YES'  # Approval needed for orchestrator to run
        
        # Start the orchestrator
        self.log_file = project_root / "trading_session_15min_monitor.log"
        
        cmd = [
            sys.executable,
            "🎯_MASTER_SYSTEM_ORCHESTRATOR.py",
            "--paper"  # Explicitly request paper trading
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=open(self.log_file, 'w'),
                stderr=subprocess.STDOUT,
                env=env,
                cwd=project_root
            )
            logger.info(f"✅ Orchestrator started (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start orchestrator: {e}")
            return False
    
    def parse_metrics_from_log(self):
        """Parse current metrics from the log file"""
        try:
            if not self.log_file.exists():
                return
            
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Parse recent lines
            for line in lines[self.last_line_index:]:
                self.last_line_index += 1
                
                # Check for signals
                if 'SIGNAL' in line and 'BUY' in line:
                    self.metrics['signals_generated'] += 1
                
                # Check for executed trades
                if 'EXECUTED' in line or 'filled' in line.lower():
                    self.metrics['trades_executed'] += 1
                
                # Check for errors
                if '[ERROR]' in line or 'CRITICAL' in line:
                    self.metrics['errors'].append(line.strip())
                    if len(self.metrics['errors']) > 10:
                        self.metrics['errors'].pop(0)
                
                # Check for warnings
                if 'WARNING' in line:
                    self.metrics['warnings'].append(line.strip())
                    if len(self.metrics['warnings']) > 10:
                        self.metrics['warnings'].pop(0)
                
                # Check health status
                if 'healthy' in line.lower() or 'operational' in line.lower():
                    self.metrics['system_health'] = 'HEALTHY'
                elif '[ERROR]' in line:
                    self.metrics['system_health'] = 'ERROR'
                elif 'WARNING' in line:
                    self.metrics['system_health'] = 'WARNING'
                
                # Check for portfolio metrics
                if 'NAV=' in line or 'portfolio' in line.lower():
                    try:
                        # Try to extract values
                        if 'positions=' in line:
                            match = re.search(r'positions=(\d+)', line)
                            if match:
                                self.metrics['positions'] = int(match.group(1))
                        if 'NAV=' in line:
                            match = re.search(r'NAV=[\$]?([\d.]+)', line)
                            if match:
                                self.metrics['portfolio_value'] = float(match.group(1))
                    except:
                        pass
            
            self.metrics['last_update'] = datetime.now().isoformat()
        except Exception as e:
            logger.warning(f"Error parsing log: {e}")
    
    def print_status_report(self, elapsed_seconds):
        """Print a comprehensive status report"""
        elapsed_min = elapsed_seconds / 60
        remaining_min = (self.duration - elapsed_seconds) / 60
        
        print("\n" + "="*80)
        print(f"📊 TRADING SYSTEM STATUS REPORT - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        print(f"⏱️  Elapsed: {elapsed_min:.1f} min | Remaining: {remaining_min:.1f} min")
        print(f"🏥 System Health: {self.metrics['system_health']}")
        print(f"💰 Portfolio Value: ${self.metrics['portfolio_value']:.2f}")
        print(f"📍 Open Positions: {self.metrics['positions']}")
        print(f"📈 Signals Generated: {self.metrics['signals_generated']}")
        print(f"🔄 Trades Executed: {self.metrics['trades_executed']}")
        
        if self.metrics['pnl'] != 0:
            pnl_str = f"${self.metrics['pnl']:+.2f}"
            icon = "📈" if self.metrics['pnl'] > 0 else "📉"
            print(f"{icon} P&L: {pnl_str}")
        
        if self.metrics['errors']:
            print(f"\n⚠️  Recent Errors ({len(self.metrics['errors'])} total):")
            for err in self.metrics['errors'][-3:]:
                print(f"   - {err[:100]}")
        
        if self.metrics['warnings']:
            print(f"\n⚠️  Recent Warnings ({len(self.metrics['warnings'])} total):")
            for warn in self.metrics['warnings'][-3:]:
                print(f"   - {warn[:100]}")
        
        print("="*80 + "\n")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info(f"📡 Starting 15-minute monitoring session...")
        logger.info(f"📊 Status updates every {self.update_interval} seconds")
        
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration
        
        if not self.start_orchestrator():
            logger.error("Failed to start orchestrator")
            return False
        
        # Give the system time to initialize
        logger.info("⏳ Waiting for system initialization...")
        time.sleep(5)
        
        update_count = 0
        last_update_time = self.start_time
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Check if process is still running
                if self.process.poll() is not None:
                    logger.warning(f"⚠️  Orchestrator process ended (exit code: {self.process.returncode})")
                    break
                
                # Check if time is up
                if elapsed >= self.duration:
                    logger.info("✅ 15-minute session complete!")
                    break
                
                # Print status every update_interval seconds
                if current_time - last_update_time >= self.update_interval:
                    self.parse_metrics_from_log()
                    self.print_status_report(elapsed)
                    last_update_time = current_time
                    update_count += 1
                
                time.sleep(5)  # Check status every 5 seconds
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Monitoring interrupted by user")
        finally:
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("🛑 Shutting down...")
        
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("✅ Orchestrator stopped")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Force killing orchestrator...")
                self.process.kill()
        
        logger.info("\n" + "="*80)
        logger.info("📋 FINAL STATUS REPORT")
        logger.info("="*80)
        self.parse_metrics_from_log()
        self.print_status_report(time.time() - self.start_time)
        logger.info(f"📝 Full logs: {self.log_file}")

def main():
    """Main entry point"""
    # Parse command line args
    paper_mode = "--live" not in sys.argv
    
    if paper_mode:
        logger.info("📄 Running in PAPER TRADING mode (safe for testing)")
    else:
        logger.warning("⚠️  Running in LIVE TRADING mode!")
        response = input("Type 'CONFIRM' to proceed: ").strip()
        if response != "CONFIRM":
            logger.info("Cancelled.")
            return
    
    # Run monitor
    monitor = RealtimeMonitor(duration_minutes=15, update_interval_seconds=120)
    success = monitor.monitor_loop()
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
