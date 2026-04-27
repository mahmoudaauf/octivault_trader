#!/usr/bin/env python3
"""
Continuous profit accumulator monitor.
Runs until 10 USDT profit is accumulated, then stops orchestrator.
"""

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

# Configuration
PROFIT_TARGET = 10.0  # USDT
CHECK_INTERVAL = 5.0  # seconds between checks
LOG_PATTERNS = {
    "realized_pnl": r"realized_pnl[\"']?\s*:\s*([0-9\.\-]+)",
    "capital_free": r"capital_free[\"']?\s*:\s*([0-9\.\-]+)",
    "nav": r"[\"']?nav[\"']?\s*:\s*([0-9\.\-]+)",
    "portfolio_pnl": r"portfolio_pnl[\"']?\s*:\s*([0-9\.\-]+)",
}


def get_latest_log_file() -> Optional[Path]:
    """Find most recent trading_run_*.log file"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    log_files = sorted(logs_dir.glob("trading_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None


def extract_pnl_from_logs(log_file: Path, lines_to_check: int = 200) -> Tuple[float, dict]:
    """Extract PnL metrics from recent log lines"""
    try:
        # Read last N lines
        result = subprocess.run(
            ["tail", "-n", str(lines_to_check), str(log_file)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split("\n")
        
        # Search for metrics in reverse order (most recent first)
        for line in reversed(lines):
            if not line.strip():
                continue
            
            # Try to find LOOP_SUMMARY or similar diagnostic lines
            metrics = {}
            
            for key, pattern in LOG_PATTERNS.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        metrics[key] = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
            
            if metrics:
                # Return most comprehensive metric found
                if "realized_pnl" in metrics:
                    return metrics.get("realized_pnl", 0.0), metrics
                elif "portfolio_pnl" in metrics:
                    return metrics.get("portfolio_pnl", 0.0), metrics
        
        return 0.0, {}
    
    except Exception as e:
        print(f"[PnL Monitor] Error extracting PnL: {e}")
        return 0.0, {}


def get_nav_from_logs(log_file: Path, lines_to_check: int = 100) -> float:
    """Extract NAV (total portfolio value) from logs"""
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines_to_check), str(log_file)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split("\n")
        
        for line in reversed(lines):
            if "nav" in line.lower() or "portfolio_snapshot" in line.lower():
                match = re.search(r"[\"']?nav[\"']?\s*:\s*([0-9\.\-]+)", line, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except (ValueError, IndexError):
                        pass
        
        return 0.0
    
    except Exception as e:
        print(f"[PnL Monitor] Error extracting NAV: {e}")
        return 0.0


def get_orchestrator_pid() -> Optional[int]:
    """Read orchestrator PID from file"""
    try:
        pid_file = Path("orchestrator.pid")
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            # Verify process is alive
            result = subprocess.run(["kill", "-0", str(pid)], capture_output=True)
            if result.returncode == 0:
                return pid
    except Exception:
        pass
    return None


def stop_orchestrator(pid: int):
    """Gracefully stop orchestrator"""
    try:
        print(f"\n[PnL Monitor] ✅ TARGET REACHED! Stopping orchestrator (PID {pid})...")
        subprocess.run(["kill", "-TERM", str(pid)], timeout=5)
        time.sleep(2)
        # Force kill if still alive
        subprocess.run(["kill", "-9", str(pid)], timeout=5, stderr=subprocess.DEVNULL)
        print(f"[PnL Monitor] ✅ Orchestrator stopped")
    except Exception as e:
        print(f"[PnL Monitor] ⚠️ Error stopping orchestrator: {e}")


async def monitor_profit():
    """Main profit monitoring loop"""
    print("=" * 80)
    print("PROFIT ACCUMULATOR MONITOR")
    print("=" * 80)
    print(f"Target: {PROFIT_TARGET:.2f} USDT profit")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print("=" * 80)
    print()
    
    start_time = time.time()
    max_runtime = 14400  # 4 hours max
    check_count = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_runtime:
            print(f"\n[PnL Monitor] ⏰ Max runtime ({max_runtime}s) reached. Stopping.")
            pid = get_orchestrator_pid()
            if pid:
                stop_orchestrator(pid)
            break
        
        check_count += 1
        log_file = get_latest_log_file()
        
        if not log_file:
            print(f"[{check_count:04d}] ⏳ Waiting for log file...")
            await asyncio.sleep(CHECK_INTERVAL)
            continue
        
        try:
            # Extract PnL from logs
            realized_pnl, metrics = extract_pnl_from_logs(log_file)
            nav = get_nav_from_logs(log_file)
            
            # Display status
            progress = (realized_pnl / PROFIT_TARGET * 100) if PROFIT_TARGET > 0 else 0
            status_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
            
            print(
                f"[{check_count:04d}] PnL: {realized_pnl:+.2f} USDT "
                f"({progress:.1f}%) [{status_bar}] | NAV: {nav:.2f} | "
                f"Elapsed: {int(elapsed)}s"
            )
            
            # Check if target reached
            if realized_pnl >= PROFIT_TARGET:
                print(
                    f"\n🎉 SUCCESS! Reached {realized_pnl:.2f} USDT profit "
                    f"(target: {PROFIT_TARGET:.2f} USDT)"
                )
                pid = get_orchestrator_pid()
                if pid:
                    stop_orchestrator(pid)
                print("[PnL Monitor] ✅ Monitor complete")
                break
        
        except Exception as e:
            print(f"[{check_count:04d}] ⚠️ Error: {e}")
        
        await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(monitor_profit())
    except KeyboardInterrupt:
        print("\n[PnL Monitor] Interrupted by user")
        pid = get_orchestrator_pid()
        if pid:
            stop_orchestrator(pid)
