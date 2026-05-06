#!/usr/bin/env python3
"""
⚡ QUICK STATUS CHECK - Verify monitoring setup and system health
"""

import json
import subprocess
import time
from pathlib import Path


def check_orchestrator_running() -> bool:
    """Check if orchestrator is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "MASTER_SYSTEM_ORCHESTRATOR"],
            capture_output=True,
            timeout=2,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def check_log_exists_and_fresh() -> tuple:
    """Check if log file exists and has recent content."""
    log_path = Path("logs/active_15m_run.log")
    if not log_path.exists():
        return False, 0

    try:
        stat = log_path.stat()
        age_seconds = time.time() - stat.st_mtime
        is_fresh = age_seconds < 120  # Less than 2 minutes old
        return True, age_seconds
    except Exception:
        return False, 0


def check_metrics_file() -> dict:
    """Check latest metrics from dashboard file."""
    metrics_path = Path("monitoring/dashboard_metrics.json")
    if not metrics_path.exists():
        return {}

    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return {}


def check_state_files() -> dict:
    """Check if state files exist."""
    state_dir = Path("state")
    return {
        "checkpoint.json": (state_dir / "checkpoint.json").exists(),
        "active_trades.db": (state_dir / "active_trades.db").exists(),
        "portfolio_state.json": (state_dir / "portfolio_state.json").exists(),
    }


def get_latest_log_entries(num_lines: int = 5) -> list:
    """Get latest log entries."""
    log_path = Path("logs/active_15m_run.log")
    if not log_path.exists():
        return []

    try:
        with open(log_path) as f:
            lines = f.readlines()
            return [l.strip() for l in lines[-num_lines:] if l.strip()]
    except Exception:
        return []


def print_status():
    """Print current system status."""
    print("\n" + "=" * 80)
    print("⚡ OCTIVAULT MONITORING SYSTEM STATUS".center(80))
    print("=" * 80)

    # 1. Orchestrator Status
    orch_running = check_orchestrator_running()
    status_icon = "🟢" if orch_running else "🔴"
    print("\n🤖 ORCHESTRATOR")
    print(f"   Status:  {status_icon} {'RUNNING' if orch_running else 'NOT RUNNING'}")

    # 2. Log Status
    log_exists, age_seconds = check_log_exists_and_fresh()
    if log_exists:
        is_fresh = age_seconds < 120
        status_icon = "🟢" if is_fresh else "🟡"
        print("\n📝 LOG FILE")
        print(f"   Status:  {status_icon} EXISTS")
        print(f"   Age:     {int(age_seconds)}s")
    else:
        print("\n📝 LOG FILE")
        print("   Status:  🔴 NOT FOUND")

    # 3. Metrics Status
    metrics = check_metrics_file()
    if metrics:
        print("\n📊 LATEST METRICS")
        print(f"   NAV:             ${metrics.get('nav', 0):.2f}")
        print(f"   Free Capital:    ${metrics.get('free', 0):.2f}")
        print(f"   Invested:        ${metrics.get('invested', 0):.2f}")
        print(f"   Total Return:    {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   Loop:            {metrics.get('loop', 0)}")

        health = metrics.get("health", {})
        print("\n🏥 HEALTH STATUS")
        print(f"   Balance Sync:    {health.get('balance_sync', '❓')}")
        print(f"   Execution:       {health.get('execution', '❓')}")
        print(f"   Positions:       {health.get('positions', '❓')}")
    else:
        print("\n📊 LATEST METRICS")
        print("   Status:  🔴 NO METRICS (waiting for first cycle...)")

    # 4. State Files
    state_files = check_state_files()
    print("\n💾 STATE FILES")
    for name, exists in state_files.items():
        icon = "✅" if exists else "❌"
        print(f"   {name:25} {icon}")

    # 5. Recent Log Entries
    recent_logs = get_latest_log_entries(5)
    if recent_logs:
        print("\n📋 RECENT LOG ENTRIES")
        for entry in recent_logs:
            # Truncate long entries
            if len(entry) > 70:
                entry = entry[:67] + "..."
            print(f"   {entry}")

    # 6. Quick Actions
    print("\n🔧 QUICK ACTIONS")
    print("   Start:   ./start_trading_with_monitoring.sh")
    print("   Monitor: python -m monitoring.active_capital_monitor")
    print("   Dashboard: python monitoring/real_time_dashboard.py")
    print("   Guide:   cat MONITORING_GUIDE.md")

    # Summary
    print("\n" + "=" * 80)
    if orch_running and log_exists:
        print("✅ SYSTEM IS OPERATIONAL")
    elif not orch_running:
        print("⚠️  ORCHESTRATOR NOT RUNNING - Start with: ./start_trading_with_monitoring.sh")
    else:
        print("⚠️  WAITING FOR FIRST METRICS - System starting up")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    print_status()
