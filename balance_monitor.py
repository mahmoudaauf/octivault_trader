#!/usr/bin/env python3
"""
Balance Monitor - Tracks balance growth/decay in real-time
"""
# === OCTIVAULT FREEZE BANNER ===
# STATUS:    LEGACY
# CANONICAL: src/l2_marketdata/balance_manager.py
# REASON:    Top-level monitor; superseded by OperationsEngine.get_health_report()
# POLICY:    See STEP_4_MODULE_FREEZE.md — do not import from main.py / top-level scripts.
# ===============================


import re
import sys
import time
from datetime import datetime
from pathlib import Path


def get_latest_nav():
    """Extract latest NAV from logs"""
    log_file = Path(
        "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/logs/octivault_master_orchestrator.log"
    )

    if not log_file.exists():
        return None, None

    try:
        with open(log_file) as f:
            lines = f.readlines()

        # Search for NAV in reverse (most recent first)
        for line in reversed(lines[-500:]):
            # Pattern: [NAV:...] nav=X.XX
            match = re.search(r"nav=([0-9.]+)", line)
            if match:
                nav = float(match.group(1))
                timestamp = line.split("[")[0].strip() if "[" in line else "unknown"
                return nav, timestamp

    except Exception as e:
        print(f"Error reading log: {e}")

    return None, None


def monitor_balance(duration_sec=300):
    """Monitor balance for specified duration"""
    print("=" * 80)
    print("🔍 BALANCE MONITORING SYSTEM")
    print("=" * 80)

    navs = []
    timestamps = []
    start_time = time.time()
    check_interval = 5  # seconds

    print(f"\n📊 Monitoring for {duration_sec}s (checking every {check_interval}s)...\n")
    print(f"{'Time':<25} {'NAV':<15} {'Δ from start':<15} {'Trend':<10}")
    print("-" * 80)

    while time.time() - start_time < duration_sec:
        nav, ts = get_latest_nav()

        if nav is not None:
            navs.append(nav)
            timestamps.append(ts)

            delta = nav - navs[0] if navs else 0
            change_pct = (delta / navs[0] * 100) if navs[0] > 0 else 0

            # Determine trend
            if len(navs) >= 2:
                recent_delta = nav - navs[-2]
                if recent_delta > 0.01:
                    trend = "📈 UP"
                elif recent_delta < -0.01:
                    trend = "📉 DOWN"
                else:
                    trend = "→ FLAT"
            else:
                trend = "..."

            status = "✅" if delta >= 0 else "❌"
            print(
                f"{ts:<25} ${nav:<14.2f} {status} ${delta:>+10.2f} ({change_pct:>+6.2f}%) {trend}"
            )
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<25} Waiting for data...")

        time.sleep(check_interval)

    # Summary
    print("\n" + "=" * 80)
    print("📊 MONITORING SUMMARY")
    print("=" * 80)

    if navs:
        start_nav = navs[0]
        final_nav = navs[-1]
        total_change = final_nav - start_nav
        total_change_pct = (total_change / start_nav * 100) if start_nav > 0 else 0

        print(f"\n💰 Starting Balance:    ${start_nav:.2f}")
        print(f"💰 Final Balance:       ${final_nav:.2f}")
        print(f"💰 Total Change:        ${total_change:+.2f} ({total_change_pct:+.2f}%)")

        if total_change > 0:
            print("\n✅ BALANCE GROWING - System is PROFITABLE!")
        elif total_change < -0.50:
            print("\n❌ BALANCE DECAYING - System is LOSING MONEY!")
        else:
            print("\n⚠️  BALANCE FLAT - Minimal change, monitor longer")

        # Additional metrics
        max_nav = max(navs)
        min_nav = min(navs)
        drawdown = start_nav - min_nav
        peak_to_current = final_nav - max_nav

        print(f"\n📈 Peak Balance:        ${max_nav:.2f}")
        print(f"📉 Lowest Balance:      ${min_nav:.2f}")
        print(f"📉 Max Drawdown:        ${drawdown:.2f} ({drawdown/start_nav*100:.2f}%)")
        print(f"📊 Peak to Current:     ${peak_to_current:+.2f}")

        # Trend analysis
        if len(navs) >= 3:
            recent_trend = navs[-1] - navs[0]
            direction = (
                "growing 📈"
                if recent_trend > 0
                else "declining 📉"
                if recent_trend < 0
                else "stable →"
            )
            print(f"\n🎯 Overall Trend:       Balance is {direction}")

        return total_change >= 0

    return False


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    success = monitor_balance(duration)
    sys.exit(0 if success else 1)
