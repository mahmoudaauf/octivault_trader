#!/usr/bin/env python3
"""
📊 REAL-TIME CAPITAL GROWTH DASHBOARD

Live dashboard showing:
- Capital growth trajectory (with sparklines)
- PnL progress
- System health indicators
- Active alerts
- Trading statistics

Updates every 30 seconds with current metrics.
"""
# === OCTIVAULT FREEZE BANNER ===
# STATUS:    LEGACY
# CANONICAL: src/l7_observability/dashboard.py
# REASON:    Pre-engine standalone dashboard; superseded by OperationsEngine
# POLICY:    See STEP_4_MODULE_FREEZE.md — do not import from main.py / top-level scripts.
# ===============================


import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional


class RealTimeDashboard:
    """Real-time dashboard for capital growth monitoring."""

    def __init__(self, log_path: Path = Path("logs/active_15m_run.log")):
        self.log_path = log_path
        self.capital_history: deque = deque(maxlen=360)  # 30 min of 5s samples
        self.last_check_time = 0
        self.start_time = None
        self.start_nav = None
        self.metrics_file = Path("monitoring/dashboard_metrics.json")

    def read_recent_logs(self, num_lines: int = 50) -> list[str]:
        """Read recent log lines."""
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path) as f:
                lines = f.readlines()
                return lines[-num_lines:] if lines else []
        except Exception:
            return []

    def extract_capital_metrics(self) -> Optional[dict]:
        """Extract current capital metrics from logs."""
        lines = self.read_recent_logs(100)

        # Look for most recent metrics line
        for line in reversed(lines):
            # Pattern: NAV: $101.70 | Free: $97.86 | Invested: $3.84 | Loop: 42
            nav_match = re.search(r"NAV[:\s]+\$?([\d.]+)", line)
            free_match = re.search(r"Free[:\s]+\$?([\d.]+)", line)
            invested_match = re.search(r"Invested[:\s]+\$?([\d.]+)", line)
            loop_match = re.search(r"Loop[:\s]+(\d+)", line)
            pnl_match = re.search(r"PnL[:\s]+\$?([\d.-]+)", line)

            if nav_match and free_match:
                return {
                    "timestamp": time.time(),
                    "nav": float(nav_match.group(1)),
                    "free": float(free_match.group(1)),
                    "invested": float(invested_match.group(1)) if invested_match else 0.0,
                    "loop": int(loop_match.group(1)) if loop_match else 0,
                    "pnl": float(pnl_match.group(1)) if pnl_match else 0.0,
                }

        return None

    def sparkline(self, values: deque, width: int = 20) -> str:
        """Generate ASCII sparkline from values."""
        if len(values) < 2:
            return "▁" * width

        values_list = list(values)
        if not values_list:
            return "▁" * width

        min_val = min(values_list)
        max_val = max(values_list)
        range_val = max_val - min_val

        if range_val == 0:
            return "▔" * width

        chars = "▁▂▃▄▅▆▇█"
        result = ""

        # Sample evenly across history
        step = max(1, len(values_list) // width)
        for i in range(0, len(values_list), step):
            if len(result) >= width:
                break

            val = values_list[i]
            normalized = (val - min_val) / range_val
            char_idx = min(len(chars) - 1, int(normalized * (len(chars) - 1)))
            result += chars[char_idx]

        # Pad if needed
        result += "▁" * (width - len(result))
        return result[:width]

    def calculate_returns(self) -> tuple[float, float, float]:
        """Calculate returns metrics."""
        if len(self.capital_history) < 2:
            return 0.0, 0.0, 0.0

        navs = [m["nav"] for m in self.capital_history]
        start_nav = navs[0]
        current_nav = navs[-1]
        peak_nav = max(navs)

        total_return = (current_nav - start_nav) / start_nav * 100.0 if start_nav else 0.0
        max_dd = (min(navs) - peak_nav) / peak_nav * 100.0 if peak_nav else 0.0

        # Calculate hourly annualized return
        elapsed_minutes = (
            self.capital_history[-1]["timestamp"] - self.capital_history[0]["timestamp"]
        ) / 60.0
        if elapsed_minutes > 0:
            hourly_return = (total_return / elapsed_minutes) * 60.0
        else:
            hourly_return = 0.0

        return total_return, hourly_return, max_dd

    def get_health_indicators(self) -> dict[str, str]:
        """Get system health indicators from logs."""
        lines = self.read_recent_logs(50)
        indicators = {
            "balance_sync": "🟢",
            "execution": "🟢",
            "positions": "🟢",
        }

        # Check for errors
        error_lines = [l for l in lines if "ERROR" in l or "error" in l]
        if error_lines:
            for error in error_lines:
                if "sync" in error.lower():
                    indicators["balance_sync"] = "🔴"
                elif "execution" in error.lower():
                    indicators["execution"] = "🔴"
                elif "position" in error.lower():
                    indicators["positions"] = "🔴"

        # Check for warnings
        warn_lines = [l for l in lines if "WARNING" in l or "WARN" in l]
        if warn_lines:
            for warn in warn_lines:
                if "sync" in warn.lower():
                    indicators["balance_sync"] = "🟡"
                elif "execution" in warn.lower():
                    indicators["execution"] = "🟡"
                elif "position" in warn.lower():
                    indicators["positions"] = "🟡"

        return indicators

    def save_metrics(self, metrics: dict) -> None:
        """Save metrics to JSON for external tools."""
        try:
            self.metrics_file.parent.mkdir(exist_ok=True)
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception:
            pass

    def print_dashboard(self) -> None:
        """Print live dashboard."""
        # Extract current metrics
        current = self.extract_capital_metrics()
        if not current:
            print("⏳ Waiting for trading data...")
            return

        # Initialize on first run
        if not self.start_time:
            self.start_time = time.time()
            self.start_nav = current["nav"]

        # Add to history
        self.capital_history.append(current)

        # Calculate returns
        total_return, hourly_return, max_dd = self.calculate_returns()

        # Get health
        indicators = self.get_health_indicators()

        # Elapsed time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        # Clear screen and print dashboard
        os.system("clear" if os.name == "posix" else "cls")

        print("\n" + "=" * 100)
        print("📊 REAL-TIME CAPITAL GROWTH DASHBOARD".center(100))
        print("=" * 100)

        # Header info
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n⏰ {now_str} | ⏱️  Elapsed: {hours}h {minutes}m {seconds}s | 📍 Loop: {current['loop']}\n"
        )

        # Capital Section
        print("💰 CAPITAL STATUS")
        print("─" * 100)
        print(f"  Current NAV:        ${current['nav']:>10.2f}  ", end="")
        if self.start_nav:
            change = current["nav"] - self.start_nav
            print(f"[{'+' if change >= 0 else ''}{change:>8.2f}]")
        else:
            print()

        print(f"  Free Capital:       ${current['free']:>10.2f}")
        print(f"  Invested:           ${current['invested']:>10.2f}")
        print(
            f"  Positions:          {len([l for l in self.read_recent_logs(5) if 'USDT' in l]):>10}"
        )

        # Returns Section
        print("\n📈 RETURNS ANALYSIS")
        print("─" * 100)
        print(f"  Total Return:       {total_return:>9.2f}%")
        print(f"  Hourly Return (Ann): {hourly_return:>8.2f}%")
        print(f"  Max Drawdown:       {max_dd:>9.2f}%")

        # Sparkline
        nav_values = deque([m["nav"] for m in self.capital_history], maxlen=40)
        sparkline = self.sparkline(nav_values, width=50)
        print(f"  History:            {sparkline}")

        # Health Section
        print("\n🏥 SYSTEM HEALTH")
        print("─" * 100)
        print(f"  Balance Sync:       {indicators['balance_sync']}")
        print(f"  Execution:          {indicators['execution']}")
        print(f"  Positions:          {indicators['positions']}")

        # Metrics Summary
        print("\n📊 METRICS SUMMARY")
        print("─" * 100)
        print(f"  History Window:     {len(self.capital_history)} samples")
        print(
            f"  NAV Range:          ${min(self.capital_history, key=lambda x: x['nav'])['nav']:.2f} - "
            f"${max(self.capital_history, key=lambda x: x['nav'])['nav']:.2f}"
        )

        # Footer
        print("\n" + "=" * 100)
        print("Press Ctrl+C to exit | Dashboard updates every 30 seconds".center(100))
        print("=" * 100 + "\n")

        # Save metrics
        metrics = {
            "timestamp": current["timestamp"],
            "nav": current["nav"],
            "free": current["free"],
            "invested": current["invested"],
            "total_return_pct": total_return,
            "hourly_return_pct": hourly_return,
            "max_drawdown_pct": max_dd,
            "loop": current["loop"],
            "health": indicators,
        }
        self.save_metrics(metrics)

    def run(self, refresh_interval: float = 30.0) -> int:
        """Run dashboard continuously."""
        print("🚀 Starting Real-Time Dashboard (Ctrl+C to exit)...")
        time.sleep(2)

        try:
            while True:
                self.print_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\n✅ Dashboard stopped")
            return 0
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")
            return 1


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time capital growth dashboard")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/active_15m_run.log"),
        help="Path to log file",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=30.0,
        help="Refresh interval in seconds (default: 30)",
    )

    args = parser.parse_args()

    dashboard = RealTimeDashboard(log_path=args.log_path)
    return dashboard.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    sys.exit(main())
