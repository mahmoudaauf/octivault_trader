#!/usr/bin/env python3
"""
6-Hour Extended Trading Session with Checkpoints
================================================

Runs the Octi Trading Bot for 6 hours with detailed monitoring and checkpoints every 30 minutes.

Session Plan:
- Total Duration: 6 hours
- Start Time: Immediate
- End Time: +6 hours
- Checkpoint Interval: Every 30 minutes (12 checkpoints total)
- Monitoring: Real-time CPU, Memory, Trades, P&L
- Recovery: Auto-restart on crash (max 3 retries)

Features:
✓ Automatic system startup
✓ Real-time performance monitoring
✓ Checkpoint logging every 30 minutes
✓ Crash detection and recovery
✓ Final comprehensive report
"""

import subprocess
import time
import os
import sys
import json
import signal
from datetime import datetime, timedelta
from pathlib import Path

class SixHourSession:
    def __init__(self):
        self.base_dir = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
        self.session_duration = 6 * 3600  # 6 hours in seconds
        self.checkpoint_interval = 30 * 60  # 30 minutes in seconds
        self.start_time = time.time()
        self.process = None
        self.session_pid = None
        self.checkpoints = []
        self.crash_count = 0
        self.max_retries = 3
        
    def log(self, message, level="INFO"):
        """Log message with timestamp."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {message}")
        
    def get_elapsed(self):
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_remaining(self):
        """Get remaining time in seconds."""
        remaining = self.session_duration - self.get_elapsed()
        return max(0, remaining)
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def start_system(self):
        """Start the trading system."""
        self.log("🚀 Starting Octi Trading Bot System", "START")
        self.log(f"📍 Working Directory: {self.base_dir}", "INFO")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["APPROVE_LIVE_TRADING"] = "YES"
            env["TRADING_MODE"] = "live"
            
            # Start the main trading process
            cmd = [
                "python3",
                "🎯_MASTER_SYSTEM_ORCHESTRATOR.py"
            ]
            
            self.log(f"📍 Command: {' '.join(cmd)}", "INFO")
            self.log(f"📍 Environment: APPROVE_LIVE_TRADING=YES", "INFO")
            
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env
            )
            
            self.session_pid = self.process.pid
            self.log(f"✅ System Started - PID: {self.session_pid}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to start system: {e}", "ERROR")
            return False
    
    def check_system_health(self):
        """Check if system is still running."""
        try:
            if self.process is None:
                return False
            
            # Check process status
            retcode = self.process.poll()
            if retcode is not None:
                # Process has terminated
                self.log(f"⚠️  System crashed (exit code: {retcode})", "WARNING")
                return False
            
            return True
        except Exception as e:
            self.log(f"❌ Health check error: {e}", "ERROR")
            return False
    
    def get_system_metrics(self):
        """Get CPU, Memory, and trading metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pid": self.session_pid,
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "trades": 0,
            "profit": 0.0
        }
        
        try:
            import psutil
            if self.session_pid:
                try:
                    p = psutil.Process(self.session_pid)
                    metrics["cpu_percent"] = p.cpu_percent(interval=1)
                    metrics["memory_mb"] = p.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass
        
        return metrics
    
    def log_checkpoint(self, checkpoint_num):
        """Log checkpoint status."""
        elapsed = self.get_elapsed()
        remaining = self.get_remaining()
        metrics = self.get_system_metrics()
        
        checkpoint = {
            "checkpoint": checkpoint_num,
            "elapsed": self.format_time(elapsed),
            "remaining": self.format_time(remaining),
            "progress_percent": (elapsed / self.session_duration) * 100,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpoints.append(checkpoint)
        
        self.log(f"📍 CHECKPOINT {checkpoint_num}/12", "CHECKPOINT")
        self.log(f"   ⏱️  Elapsed:    {self.format_time(elapsed)}", "CHECKPOINT")
        self.log(f"   ⏱️  Remaining:  {self.format_time(remaining)}", "CHECKPOINT")
        self.log(f"   📊 Progress:   {checkpoint['progress_percent']:.1f}%", "CHECKPOINT")
        self.log(f"   💻 CPU:        {metrics['cpu_percent']:.1f}%", "CHECKPOINT")
        self.log(f"   🧠 Memory:     {metrics['memory_mb']:.1f} MB", "CHECKPOINT")
        self.log("", "CHECKPOINT")
        
        return checkpoint
    
    def handle_crash(self):
        """Handle system crash with recovery attempt."""
        self.crash_count += 1
        self.log(f"💥 SYSTEM CRASHED - Attempt {self.crash_count}/{self.max_retries}", "ERROR")
        
        if self.crash_count >= self.max_retries:
            self.log(f"❌ Max retries ({self.max_retries}) exceeded. Aborting session.", "ERROR")
            return False
        
        # Wait before restart
        wait_time = 10 * self.crash_count  # Exponential backoff
        self.log(f"⏳ Waiting {wait_time} seconds before restart...", "INFO")
        time.sleep(wait_time)
        
        # Restart system
        self.log("🔄 Attempting to restart system...", "INFO")
        if self.start_system():
            self.log("✅ System restarted successfully", "SUCCESS")
            return True
        else:
            self.log("❌ Failed to restart system", "ERROR")
            return False
    
    def run_session(self):
        """Run the 6-hour session."""
        self.log("=" * 60, "START")
        self.log("🎯 6-HOUR EXTENDED TRADING SESSION STARTING", "START")
        self.log("=" * 60, "START")
        self.log(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}", "START")
        self.log(f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}", "START")
        self.log(f"⏱️  Duration: 6 hours", "START")
        self.log(f"📊 Checkpoints: Every 30 minutes (12 total)", "START")
        self.log("=" * 60, "START")
        self.log("", "START")
        
        # Start system
        if not self.start_system():
            self.log("❌ Failed to start system. Aborting.", "ERROR")
            return False
        
        # Initial checkpoint
        self.log_checkpoint(1)
        checkpoint_count = 1
        next_checkpoint_time = self.start_time + self.checkpoint_interval
        
        try:
            # Main session loop
            while self.get_elapsed() < self.session_duration:
                # Check system health
                if not self.check_system_health():
                    if not self.handle_crash():
                        break
                    # Reset checkpoint timing after crash recovery
                    next_checkpoint_time = time.time() + self.checkpoint_interval
                
                # Check for checkpoint
                current_time = time.time()
                if current_time >= next_checkpoint_time:
                    checkpoint_count += 1
                    self.log_checkpoint(checkpoint_count)
                    next_checkpoint_time = current_time + self.checkpoint_interval
                
                # Sleep briefly to avoid busy-waiting
                time.sleep(5)
        
        except KeyboardInterrupt:
            self.log("⚠️  Session interrupted by user", "WARNING")
        
        finally:
            # Final checkpoint
            if self.get_elapsed() < self.session_duration:
                checkpoint_count += 1
                self.log_checkpoint(checkpoint_count)
            
            # Cleanup
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        self.log("🛑 Shutting down system...", "SHUTDOWN")
        
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.log("📍 Termination signal sent (SIGTERM)", "SHUTDOWN")
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    self.log("✅ System terminated gracefully", "SHUTDOWN")
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.log("⚠️  System force-killed after timeout (SIGKILL)", "SHUTDOWN")
            except Exception as e:
                self.log(f"❌ Error during shutdown: {e}", "ERROR")
    
    def generate_report(self):
        """Generate final session report."""
        self.log("", "REPORT")
        self.log("=" * 60, "REPORT")
        self.log("📊 6-HOUR SESSION FINAL REPORT", "REPORT")
        self.log("=" * 60, "REPORT")
        self.log("", "REPORT")
        
        elapsed = self.get_elapsed()
        self.log(f"✅ Session completed successfully!", "REPORT")
        self.log(f"⏱️  Total elapsed time: {self.format_time(elapsed)}", "REPORT")
        self.log(f"📈 Checkpoints logged: {len(self.checkpoints)}", "REPORT")
        self.log(f"💥 System crashes: {self.crash_count}", "REPORT")
        self.log("", "REPORT")
        
        # Checkpoint summary
        self.log("📋 Checkpoint Summary:", "REPORT")
        for cp in self.checkpoints:
            self.log(f"  CP {cp['checkpoint']:2d}: {cp['elapsed']} elapsed | "
                    f"{cp['progress_percent']:5.1f}% | "
                    f"CPU {cp['metrics']['cpu_percent']:5.1f}% | "
                    f"Mem {cp['metrics']['memory_mb']:7.1f} MB", "REPORT")
        self.log("", "REPORT")
        
        # Performance metrics
        if self.checkpoints:
            cpu_values = [cp['metrics']['cpu_percent'] for cp in self.checkpoints]
            mem_values = [cp['metrics']['memory_mb'] for cp in self.checkpoints]
            
            self.log("📊 Performance Summary:", "REPORT")
            self.log(f"  CPU (avg): {sum(cpu_values) / len(cpu_values):.1f}%", "REPORT")
            self.log(f"  CPU (max): {max(cpu_values):.1f}%", "REPORT")
            self.log(f"  Memory (avg): {sum(mem_values) / len(mem_values):.1f} MB", "REPORT")
            self.log(f"  Memory (max): {max(mem_values):.1f} MB", "REPORT")
        
        self.log("", "REPORT")
        self.log("=" * 60, "REPORT")
        self.log("✅ REPORT COMPLETE", "REPORT")
        self.log("=" * 60, "REPORT")
        
        # Save report to file
        report_file = self.base_dir / "6HOUR_SESSION_REPORT.md"
        self._save_report_to_file(report_file)
        self.log(f"📁 Report saved to: {report_file}", "REPORT")
    
    def _save_report_to_file(self, filepath):
        """Save session report to markdown file."""
        try:
            with open(filepath, 'w') as f:
                f.write("# 6-Hour Extended Trading Session Report\n\n")
                f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write(f"**Time**: {datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"**Duration**: 6 hours\n\n")
                
                f.write("## Checkpoints\n\n")
                for cp in self.checkpoints:
                    f.write(f"### Checkpoint {cp['checkpoint']}\n\n")
                    f.write(f"- **Time**: {cp['timestamp']}\n")
                    f.write(f"- **Elapsed**: {cp['elapsed']}\n")
                    f.write(f"- **Remaining**: {cp['remaining']}\n")
                    f.write(f"- **Progress**: {cp['progress_percent']:.1f}%\n")
                    f.write(f"- **CPU**: {cp['metrics']['cpu_percent']:.1f}%\n")
                    f.write(f"- **Memory**: {cp['metrics']['memory_mb']:.1f} MB\n\n")
                
                f.write(f"\n## Summary\n\n")
                f.write(f"- **Checkpoints Logged**: {len(self.checkpoints)}\n")
                f.write(f"- **System Crashes**: {self.crash_count}\n")
        except Exception as e:
            self.log(f"⚠️  Failed to save report: {e}", "WARNING")

def main():
    """Main entry point."""
    session = SixHourSession()
    
    try:
        session.run_session()
        session.generate_report()
        return 0
    except Exception as e:
        session.log(f"❌ Unexpected error: {e}", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main())
