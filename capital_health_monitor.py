#!/usr/bin/env python3
"""
💰 CAPITAL HEALTH MONITOR
Real-time tracking of where your money is going and why

Usage:
    python3 capital_health_monitor.py [--watch]  # Continuous monitoring
    python3 capital_health_monitor.py --csv       # Export to CSV for analysis
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def extract_pnl_events(log_file: Path, limit: int = 100) -> list[dict]:
    """Extract all PnL valuation cycles from log."""
    events = []

    try:
        with open(log_file) as f:
            for line in f:
                if '"valuation_cycle"' in line:
                    try:
                        # Extract JSON from log line
                        json_match = re.search(r"\{.*\}", line)
                        if json_match:
                            data = json.loads(json_match.group())
                            events.append(data)
                    except Exception:
                        pass
    except Exception as e:
        print(f"⚠️  Error reading log: {e}")

    return events[-limit:] if limit else events


def extract_nav_lines(log_file: Path, limit: int = 50) -> list[tuple[str, str]]:
    """Extract NAV calculation lines."""
    lines = []

    try:
        with open(log_file) as f:
            for line in f:
                if "[NAV]" in line:
                    # Extract timestamp and NAV info
                    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    nav_match = re.search(r"equity=([0-9.]+)", line)

                    if timestamp_match and nav_match:
                        timestamp = timestamp_match.group(1)
                        equity = nav_match.group(1)
                        lines.append((timestamp, equity))
    except Exception as e:
        print(f"⚠️  Error reading log: {e}")

    return lines[-limit:] if limit else lines


def format_capital_table(events: list[dict]) -> str:
    """Format PnL events as readable table."""
    if not events:
        return "No data found."

    output = []
    output.append("=" * 100)
    output.append("📊 CAPITAL HEALTH HISTORY")
    output.append("=" * 100)
    output.append(
        f"{'Timestamp':<20} {'NAV':>12} {'Realized PnL':>15} {'Unrealized':>12} "
        f"{'Total Equity':>12} {'Change':>10}"
    )
    output.append("-" * 100)

    prev_nav = None
    for event in events:
        ts = datetime.fromtimestamp(event.get("ts", 0)).strftime("%Y-%m-%d %H:%M:%S")
        nav = float(event.get("total_value", 0))
        real_pnl = float(event.get("realized_pnl", 0))
        unreal_pnl = float(event.get("unrealized_pnl", 0))
        total_eq = float(event.get("total_equity", 0))

        change = nav - prev_nav if prev_nav is not None else 0
        change_sign = "+" if change >= 0 else ""
        prev_nav = nav

        output.append(
            f"{ts:<20} ${nav:>11.2f} ${real_pnl:>14.2f} ${unreal_pnl:>11.2f} "
            f"${total_eq:>11.2f} {change_sign}${change:>8.2f}"
        )

    return "\n".join(output)


def analyze_capital_trend(events: list[dict]) -> dict:
    """Analyze capital changes and trends."""
    if not events:
        return {}

    first_event = events[0]
    last_event = events[-1]

    first_nav = float(first_event.get("total_value", 0))
    last_nav = float(last_event.get("total_value", 0))

    first_realized = float(first_event.get("realized_pnl", 0))
    last_realized = float(last_event.get("realized_pnl", 0))

    trading_losses = last_realized - first_realized
    nav_change = last_nav - first_nav

    # Find min/max NAV
    navs = [float(e.get("total_value", 0)) for e in events]
    min_nav = min(navs)
    max_nav = max(navs)

    return {
        "first_nav": first_nav,
        "last_nav": last_nav,
        "nav_change": nav_change,
        "nav_change_pct": (nav_change / first_nav * 100) if first_nav > 0 else 0,
        "trading_losses": trading_losses,
        "min_nav": min_nav,
        "max_nav": max_nav,
        "drawdown_pct": ((max_nav - min_nav) / max_nav * 100) if max_nav > 0 else 0,
        "events_count": len(events),
    }


def print_analysis(analysis: dict) -> None:
    """Print formatted analysis."""
    if not analysis:
        print("No analysis data.")
        return

    print("\n" + "=" * 100)
    print("📈 CAPITAL TREND ANALYSIS")
    print("=" * 100)

    print("\n💾 Monitoring Period:")
    print(f"   Samples:           {analysis['events_count']} valuation cycles")
    print(f"   Time span:         ~{analysis['events_count'] * 5 / 60:.0f} minutes (est.)")

    print("\n💰 Capital Changes:")
    print(f"   Starting NAV:      ${analysis['first_nav']:.2f}")
    print(f"   Ending NAV:        ${analysis['last_nav']:.2f}")
    print(
        f"   Net Change:        ${analysis['nav_change']:+.2f} ({analysis['nav_change_pct']:+.2f}%)"
    )

    print("\n📉 Trading Performance:")
    print(f"   Trading Losses:    ${analysis['trading_losses']:.2f}")
    print("   (Realized PnL)")

    print("\n📊 NAV Range:")
    print(f"   Peak NAV:          ${analysis['max_nav']:.2f}")
    print(f"   Trough NAV:        ${analysis['min_nav']:.2f}")
    print(f"   Max Drawdown:      {analysis['drawdown_pct']:.2f}%")

    # Diagnosis
    print("\n🔍 DIAGNOSIS:")
    if analysis["nav_change"] < -5:
        print(f"   ⚠️  CAPITAL DECLINING: Losing ${abs(analysis['nav_change']):.2f}")
        print(f"   Cause: Trading losses (${analysis['trading_losses']:.2f})")
        print("   Action: Review strategy, tighten filters, increase position size")
    elif analysis["nav_change"] > 5:
        print(f"   ✅ CAPITAL GROWING: Gained ${analysis['nav_change']:.2f}")
        print("   Status: Strategy is profitable")
    else:
        print(f"   ⏸️  CAPITAL STABLE: Change of ${analysis['nav_change']:.2f}")
        print("   Status: Breakeven or low activity period")

    print("\n" + "=" * 100)


def main():
    """Main entry point."""
    log_file = Path("logs/octivault_master_orchestrator.log")

    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        print("   Make sure you're running from the project root directory.")
        sys.exit(1)

    print("\n🔄 Reading capital health data...")

    # Extract data
    events = extract_pnl_events(log_file, limit=100)

    if not events:
        print("❌ No PnL events found in logs.")
        print("   The bot may not have been running long enough.")
        sys.exit(1)

    # Display table
    print(format_capital_table(events))

    # Analyze
    analysis = analyze_capital_trend(events)
    print_analysis(analysis)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if analysis["nav_change"] < -10:
        print("   1. PAUSE TRADING - Strategy is unprofitable")
        print("   2. Increase MIN_ECONOMIC_TRADE_USDT to $50 (was $25)")
        print("   3. Tighten MIN_EXPECTED_NET_PCT to 0.5% (was 0.12%)")
        print("   4. Add win-rate gate: require 55% minimum")
        print("   5. Review losing trades to find patterns")
    elif analysis["nav_change"] > 0:
        print("   1. Strategy is profitable! Keep running")
        print("   2. Monitor for consistency")
        print("   3. Gradually increase position size once stable")
    else:
        print("   1. Monitor more data (need 100+ samples)")
        print("   2. Strategy appears neutral - optimize for higher conviction")
        print("   3. Check if market conditions are favorable")


if __name__ == "__main__":
    main()
