#!/usr/bin/env python3
"""
🎯 INTEGRATED TRADING + MONITORING LAUNCHER

Starts the orchestrator and runs active monitoring simultaneously.
Automatically detects issues and applies fixes.

Usage:
    python launch_with_monitor.py --duration 6 --monitor-interval 10
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_clean_state() -> None:
    """Clear old state files for fresh start."""
    state_dir = Path("state")
    files_to_remove = [
        "checkpoint.json",
        "active_trades.db",
        "portfolio_state.json",
        "nav_cache.json",
    ]

    for fname in files_to_remove:
        fpath = state_dir / fname
        if fpath.exists():
            try:
                fpath.unlink()
                print(f"  ✅ Removed {fname}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {fname}: {e}")


def start_orchestrator(
    trading_hours: float,
    approve_trading: bool = True,
) -> Optional[subprocess.Popen]:
    """Start the trading orchestrator."""
    print("\n🚀 Starting Trading Orchestrator...")
    print(f"   Duration: {trading_hours} hours")
    print(f"   Approve Live Trading: {approve_trading}")

    env = os.environ.copy()
    env["TRADING_DURATION_HOURS"] = str(trading_hours)
    env["APPROVE_LIVE_TRADING"] = "YES" if approve_trading else "NO"

    log_file = Path("/tmp/octivault_active_session.log")

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "🎯_MASTER_SYSTEM_ORCHESTRATOR.py",
            ],
            env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
        )
        print(f"   ✅ Orchestrator started (PID: {proc.pid})")
        print(f"   Log: {log_file}")
        return proc
    except Exception as e:
        print(f"   ❌ Failed to start orchestrator: {e}")
        return None


async def start_monitor(
    duration_minutes: float,
    check_interval: float = 10.0,
) -> int:
    """Start the active capital monitor."""
    print("\n🎯 Starting Active Capital Monitor...")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Check Interval: {check_interval}s")

    # Import here to avoid issues if module not found
    try:
        from monitoring.active_capital_monitor import ActiveCapitalMonitor

        monitor = ActiveCapitalMonitor(
            log_path=Path("logs/active_15m_run.log"),
            check_interval=check_interval,
        )
        return await monitor.run(duration_minutes=duration_minutes)

    except ImportError as e:
        print(f"   ❌ Failed to import monitor: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Integrated trading orchestrator + active monitoring"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Trading duration in hours (default: 6)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=10.0,
        help="Monitor check interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--no-trading",
        action="store_true",
        help="Run monitor only (don't start orchestrator)",
    )
    parser.add_argument(
        "--clean-state",
        action="store_true",
        default=True,
        help="Clear state files before starting (default: yes)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🎯 OCTIVAULT INTEGRATED TRADING + MONITORING")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Clean state if requested
    if args.clean_state:
        print("\n🧹 Cleaning state files...")
        ensure_clean_state()

    # Start orchestrator
    orchestrator_proc: Optional[subprocess.Popen] = None
    if not args.no_trading:
        orchestrator_proc = start_orchestrator(
            trading_hours=args.duration,
            approve_trading=True,
        )
        if not orchestrator_proc:
            print("\n❌ Failed to start orchestrator, aborting")
            return 1

        # Wait for orchestrator to initialize
        print("\n⏳ Waiting for orchestrator to initialize...")
        time.sleep(3)

    # Run monitor
    try:
        monitor_duration = args.duration * 60  # Convert to minutes
        exit_code = asyncio.run(
            start_monitor(
                duration_minutes=monitor_duration,
                check_interval=args.monitor_interval,
            )
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring interrupted by user")
        exit_code = 0
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")
        exit_code = 1

    # Cleanup
    print("\n🧹 Cleanup...")
    if orchestrator_proc:
        try:
            orchestrator_proc.terminate()
            orchestrator_proc.wait(timeout=5)
            print("   ✅ Orchestrator terminated gracefully")
        except subprocess.TimeoutExpired:
            orchestrator_proc.kill()
            print("   ⚠️  Orchestrator force-killed")
        except Exception as e:
            print(f"   ⚠️  Error terminating orchestrator: {e}")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
