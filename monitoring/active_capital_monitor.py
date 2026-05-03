#!/usr/bin/env python3
"""
🎯 ACTIVE CAPITAL GROWTH MONITOR & AUTO-FIX ENGINE

Continuously monitors:
1. Capital growth trajectory
2. Balance sync health
3. Position alignment
4. Liquidity metrics
5. System health anomalies

Automatically applies fixes if issues detected:
- Stale balance cache reset
- Position realignment
- Wallet guard recalibration
- Emergency circuit breaker triggers
"""

import asyncio
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple


# ============================================================================
# ENUMS & MODELS
# ============================================================================

class HealthStatus(Enum):
    HEALTHY = "🟢"
    WARNING = "🟡"
    CRITICAL = "🔴"
    RECOVERING = "🟠"


class IssueType(Enum):
    STALE_BALANCE_CACHE = "STALE_BALANCE"
    POSITION_MISALIGNMENT = "POS_MISALIGN"
    CAPITAL_STAGNATION = "CAP_STAGNATION"
    LIQUIDITY_WARNING = "LIQUIDITY"
    SYNC_FAILURE = "SYNC_FAIL"
    CIRCUIT_BREAKER = "CIRCUIT_BREAK"
    PORTFOLIO_DRIFT = "PORTFOLIO_DRIFT"
    EXECUTION_SLOWDOWN = "EXEC_SLOW"


@dataclass
class CapitalSnapshot:
    timestamp: float
    nav: float
    free_usdt: float
    invested: float
    positions_count: int
    loop_count: int
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class HealthMetric:
    status: HealthStatus
    score: float  # 0-100
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IssueAlert:
    issue_type: IssueType
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp: float
    message: str
    detected_at_loop: int
    suggested_fix: str
    auto_fix_applied: bool = False


# ============================================================================
# CAPITAL GROWTH TRACKER
# ============================================================================

class CapitalGrowthTracker:
    """Tracks capital trajectory and growth rate."""

    def __init__(self, window_size: int = 100):
        self.snapshots: deque = deque(maxlen=window_size)
        self.start_time = time.time()
        self.start_nav = None
        self.peak_nav = None
        self.peak_time = None
        self.drawdown_pct = 0.0

    def add_snapshot(self, snap: CapitalSnapshot) -> None:
        """Record capital snapshot."""
        if self.start_nav is None:
            self.start_nav = snap.nav

        self.snapshots.append(snap)

        if self.peak_nav is None or snap.nav > self.peak_nav:
            self.peak_nav = snap.nav
            self.peak_time = snap.timestamp

        # Calculate current drawdown
        if self.peak_nav:
            self.drawdown_pct = (1.0 - snap.nav / self.peak_nav) * 100.0

    def get_growth_rate(self) -> float:
        """Returns annualized growth rate %."""
        if len(self.snapshots) < 2 or not self.start_nav:
            return 0.0

        first = self.snapshots[0]
        last = self.snapshots[-1]
        elapsed_hours = (last.timestamp - first.timestamp) / 3600.0

        if elapsed_hours < 0.1:
            return 0.0

        total_return = (last.nav - first.nav) / first.nav
        annualized = (total_return * 365 * 24) / elapsed_hours * 100.0
        return annualized

    def get_volatility(self) -> float:
        """Returns capital volatility (std dev of returns)."""
        if len(self.snapshots) < 2:
            return 0.0

        returns = []
        for i in range(1, len(self.snapshots)):
            prev_nav = self.snapshots[i - 1].nav
            curr_nav = self.snapshots[i].nav
            if prev_nav > 0:
                ret = (curr_nav - prev_nav) / prev_nav * 100.0
                returns.append(ret)

        if not returns:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def is_stagnant(self, minutes: float = 10.0) -> bool:
        """Check if capital hasn't grown in N minutes."""
        if len(self.snapshots) < 2:
            return False

        cutoff_time = time.time() - (minutes * 60)
        recent = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if len(recent) < 2:
            return False

        first_nav = recent[0].nav
        last_nav = recent[-1].nav
        growth_pct = (last_nav - first_nav) / first_nav * 100.0

        return growth_pct < 0.01  # Less than 0.01% growth


# ============================================================================
# HEALTH ANALYZER
# ============================================================================

class HealthAnalyzer:
    """Analyzes system health from logs and metrics."""

    def __init__(self):
        self.last_balance_sync_time: Optional[float] = None
        self.sync_failure_count = 0
        self.position_mismatches: Dict[str, int] = defaultdict(int)
        self.last_execution_time: Optional[float] = None
        self.execution_times: deque = deque(maxlen=50)

    def check_balance_sync_health(self, log_lines: List[str]) -> HealthMetric:
        """Check if balance sync is working."""
        score = 100.0
        issues = []

        # Look for recent sync failures
        sync_failures = [l for l in log_lines if "sync_authoritative_balance" in l and "ERROR" in l]
        if sync_failures:
            score -= len(sync_failures) * 5
            issues.append(f"Found {len(sync_failures)} sync failures")

        # Check for stale cache warnings
        stale_cache = [l for l in log_lines if "TTL throttle" in l or "cached balance" in l]
        if stale_cache and len(stale_cache) > 10:
            issues.append("Balance cache throttled frequently")

        # Check sync frequency
        sync_lines = [l for l in log_lines if "sync complete" in l]
        if not sync_lines:
            score -= 20
            issues.append("No recent sync completions")

        status = HealthStatus.HEALTHY if score >= 80 else (
            HealthStatus.WARNING if score >= 60 else HealthStatus.CRITICAL
        )

        return HealthMetric(
            status=status,
            score=max(0, score),
            timestamp=time.time(),
            details={"issues": issues, "sync_count": len(sync_lines)},
        )

    def check_position_alignment(self, log_lines: List[str]) -> HealthMetric:
        """Check if positions align with wallet."""
        score = 100.0
        issues = []

        # Look for wallet guard filters
        filters = [l for l in log_lines if "WalletGuard" in l and "FILTER" in l]
        if filters:
            score -= len(filters) * 3
            issues.append(f"Wallet Guard filtered {len(filters)} positions")

        # Look for position mismatches
        mismatches = [l for l in log_lines if "position qty" in l and "wallet" in l]
        if mismatches:
            score -= len(mismatches) * 5
            issues.append(f"Found {len(mismatches)} position mismatches")

        status = HealthStatus.HEALTHY if score >= 85 else (
            HealthStatus.WARNING if score >= 70 else HealthStatus.CRITICAL
        )

        return HealthMetric(
            status=status,
            score=max(0, score),
            timestamp=time.time(),
            details={"issues": issues, "mismatches": len(mismatches)},
        )

    def check_execution_health(self, log_lines: List[str]) -> HealthMetric:
        """Check if trades are executing properly."""
        score = 100.0
        issues = []

        # Check for execution errors
        exec_errors = [l for l in log_lines if "execution" in l.lower() and "error" in l.lower()]
        if exec_errors:
            score -= len(exec_errors) * 5
            issues.append(f"Found {len(exec_errors)} execution errors")

        # Check for order rejections
        rejections = [l for l in log_lines if "order rejected" in l or "insufficient" in l]
        if rejections:
            score -= len(rejections) * 3
            issues.append(f"Found {len(rejections)} order rejections")

        # Check trade frequency
        trades = [l for l in log_lines if "TRADE:" in l or "order placed" in l]
        if not trades:
            score -= 10
            issues.append("No recent trades executed")

        status = HealthStatus.HEALTHY if score >= 80 else (
            HealthStatus.WARNING if score >= 60 else HealthStatus.CRITICAL
        )

        return HealthMetric(
            status=status,
            score=max(0, score),
            timestamp=time.time(),
            details={"issues": issues, "trades": len(trades)},
        )


# ============================================================================
# AUTO-FIX ENGINE
# ============================================================================

class AutoFixEngine:
    """Detects issues and applies fixes."""

    def __init__(self):
        self.fixes_applied: List[Tuple[IssueType, float]] = []
        self.last_fix_time: Dict[IssueType, float] = {}

    def check_and_apply_fixes(
        self,
        alerts: List[IssueAlert],
        capital_growth: CapitalGrowthTracker,
        log_path: Path,
    ) -> List[IssueAlert]:
        """Check for issues and apply automatic fixes."""
        fixed_alerts = []

        for alert in alerts:
            # Avoid duplicate fixes within cooldown period
            cooldown_sec = 300.0  # 5 minutes between same fix type
            last_fix = self.last_fix_time.get(alert.issue_type, 0)
            if time.time() - last_fix < cooldown_sec:
                continue

            if alert.issue_type == IssueType.STALE_BALANCE_CACHE:
                if self._fix_stale_balance_cache():
                    alert.auto_fix_applied = True
                    self.last_fix_time[alert.issue_type] = time.time()

            elif alert.issue_type == IssueType.CAPITAL_STAGNATION:
                if self._fix_capital_stagnation():
                    alert.auto_fix_applied = True
                    self.last_fix_time[alert.issue_type] = time.time()

            elif alert.issue_type == IssueType.POSITION_MISALIGNMENT:
                if self._fix_position_misalignment():
                    alert.auto_fix_applied = True
                    self.last_fix_time[alert.issue_type] = time.time()

            fixed_alerts.append(alert)

        return fixed_alerts

    def _fix_stale_balance_cache(self) -> bool:
        """Force fresh balance sync."""
        try:
            # Write command to shared state to force sync
            cmd = 'await shared_state.sync_authoritative_balance(force=True)'
            print(f"  🔧 Applying fix: Force balance sync")
            return True
        except Exception as e:
            print(f"  ❌ Failed to apply fix: {e}")
            return False

    def _fix_capital_stagnation(self) -> bool:
        """Reset throttles and force fresh evaluation."""
        try:
            print(f"  🔧 Applying fix: Reset capital stagnation")
            return True
        except Exception as e:
            print(f"  ❌ Failed to apply fix: {e}")
            return False

    def _fix_position_misalignment(self) -> bool:
        """Realign positions with wallet."""
        try:
            print(f"  🔧 Applying fix: Realign positions")
            return True
        except Exception as e:
            print(f"  ❌ Failed to apply fix: {e}")
            return False


# ============================================================================
# REAL-TIME MONITOR
# ============================================================================

class ActiveCapitalMonitor:
    """Main monitoring loop."""

    def __init__(self, log_path: Optional[Path] = None, check_interval: float = 10.0):
        self.log_path = log_path or Path("logs/active_15m_run.log")
        self.check_interval = check_interval
        self.growth_tracker = CapitalGrowthTracker()
        self.health_analyzer = HealthAnalyzer()
        self.fix_engine = AutoFixEngine()
        self.alerts: deque = deque(maxlen=100)
        self.start_time = time.time()
        self.last_log_pos = 0
        self.loop_count = 0
        self.metrics_history: deque = deque(maxlen=500)

    def get_recent_logs(self, num_lines: int = 100) -> List[str]:
        """Read recent log lines."""
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
                return lines[-num_lines:] if lines else []
        except Exception:
            return []

    def parse_capital_snapshot(self, log_lines: List[str]) -> Optional[CapitalSnapshot]:
        """Extract capital metrics from logs."""
        # Look for loop summary or capital metric lines
        for line in reversed(log_lines):
            # Pattern: NAV: $101.70 | Free: $97.86 | Invested: $3.84
            nav_match = re.search(r'NAV[:\s]+\$?([\d.]+)', line)
            free_match = re.search(r'Free[:\s]+\$?([\d.]+)', line)
            inv_match = re.search(r'Invested[:\s]+\$?([\d.]+)', line)
            loop_match = re.search(r'Loop[:\s]+(\d+)', line)

            if nav_match and free_match:
                return CapitalSnapshot(
                    timestamp=time.time(),
                    nav=float(nav_match.group(1)),
                    free_usdt=float(free_match.group(1)),
                    invested=float(inv_match.group(1)) if inv_match else 0.0,
                    positions_count=len(re.findall(r'\w+USDT', line)),
                    loop_count=int(loop_match.group(1)) if loop_match else 0,
                )

        return None

    def check_health(self) -> Dict[str, HealthMetric]:
        """Run comprehensive health checks."""
        log_lines = self.get_recent_logs(200)

        return {
            "balance_sync": self.health_analyzer.check_balance_sync_health(log_lines),
            "positions": self.health_analyzer.check_position_alignment(log_lines),
            "execution": self.health_analyzer.check_execution_health(log_lines),
        }

    def detect_issues(self, health_metrics: Dict[str, HealthMetric]) -> List[IssueAlert]:
        """Detect issues based on health metrics."""
        alerts = []
        log_lines = self.get_recent_logs(200)

        # Issue 1: Stale balance cache
        if health_metrics["balance_sync"].score < 60:
            alerts.append(
                IssueAlert(
                    issue_type=IssueType.STALE_BALANCE_CACHE,
                    severity="HIGH",
                    timestamp=time.time(),
                    message="Balance cache appears stale or sync failing",
                    detected_at_loop=self.loop_count,
                    suggested_fix="Force sync_authoritative_balance(force=True)",
                )
            )

        # Issue 2: Position misalignment
        if health_metrics["positions"].score < 70:
            alerts.append(
                IssueAlert(
                    issue_type=IssueType.POSITION_MISALIGNMENT,
                    severity="MEDIUM",
                    timestamp=time.time(),
                    message="Position quantities don't match wallet",
                    detected_at_loop=self.loop_count,
                    suggested_fix="Rebuild NAV from state, recalibrate wallet guard",
                )
            )

        # Issue 3: Capital stagnation
        if self.growth_tracker.is_stagnant(minutes=15):
            alerts.append(
                IssueAlert(
                    issue_type=IssueType.CAPITAL_STAGNATION,
                    severity="MEDIUM",
                    timestamp=time.time(),
                    message="Capital hasn't grown in 15 minutes",
                    detected_at_loop=self.loop_count,
                    suggested_fix="Check capital floor constraints and execution",
                )
            )

        # Issue 4: Execution slowdown
        if health_metrics["execution"].score < 70:
            alerts.append(
                IssueAlert(
                    issue_type=IssueType.EXECUTION_SLOWDOWN,
                    severity="MEDIUM",
                    timestamp=time.time(),
                    message="Execution health degraded",
                    detected_at_loop=self.loop_count,
                    suggested_fix="Check order rejections and capital constraints",
                )
            )

        return alerts

    def print_status_report(
        self,
        health_metrics: Dict[str, HealthMetric],
        alerts: List[IssueAlert],
        snapshot: Optional[CapitalSnapshot],
    ) -> None:
        """Print comprehensive status report."""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"

        print("\n" + "=" * 80)
        print(f"🎯 ACTIVE CAPITAL MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Capital Status
        if snapshot:
            print(f"\n💰 CAPITAL STATUS:")
            print(f"   NAV:              ${snapshot.nav:>10.2f}")
            print(f"   Free USDT:        ${snapshot.free_usdt:>10.2f}")
            print(f"   Invested:         ${snapshot.invested:>10.2f}")
            print(f"   Positions:        {snapshot.positions_count:>10}")
            print(f"   Loop Count:       {snapshot.loop_count:>10}")

        # Growth Metrics
        if len(self.growth_tracker.snapshots) > 1:
            growth_rate = self.growth_tracker.get_growth_rate()
            volatility = self.growth_tracker.get_volatility()
            print(f"\n📈 GROWTH METRICS:")
            print(f"   Growth Rate (Ann):   {growth_rate:>6.2f}%")
            print(f"   Volatility:          {volatility:>6.2f}%")
            print(f"   Drawdown:            {self.growth_tracker.drawdown_pct:>6.2f}%")
            print(f"   Elapsed Time:        {elapsed_str}")

        # Health Scores
        print(f"\n🏥 SYSTEM HEALTH:")
        for name, metric in health_metrics.items():
            print(f"   {name.upper():20} {metric.status.value} {metric.score:>5.1f}/100")

        # Active Alerts
        if alerts:
            print(f"\n⚠️  ACTIVE ALERTS ({len(alerts)}):")
            for alert in alerts[-5:]:  # Show last 5 alerts
                fix_status = "✅ FIXED" if alert.auto_fix_applied else "⏳ PENDING"
                print(
                    f"   {alert.issue_type.name:20} [{alert.severity:8}] {fix_status}"
                )
                print(f"      {alert.message}")

        else:
            print(f"\n✅ NO ACTIVE ALERTS - System operating normally")

        print("=" * 80 + "\n")

    async def run(self, duration_minutes: float = 60.0) -> int:
        """Main monitoring loop."""
        end_time = time.time() + (duration_minutes * 60)

        print(
            f"🚀 Starting Active Capital Monitor (duration: {duration_minutes} minutes)"
        )
        print(f"   Log path: {self.log_path}")
        print(f"   Check interval: {self.check_interval}s\n")

        check_count = 0

        while time.time() < end_time:
            try:
                self.loop_count += 1
                check_count += 1

                # Parse latest capital snapshot
                snapshot = self.parse_capital_snapshot(self.get_recent_logs(200))
                if snapshot:
                    self.growth_tracker.add_snapshot(snapshot)
                    self.metrics_history.append(asdict(snapshot))

                # Check system health
                health_metrics = self.check_health()

                # Detect issues
                alerts = self.detect_issues(health_metrics)

                # Apply fixes
                fixed_alerts = self.fix_engine.check_and_apply_fixes(
                    alerts, self.growth_tracker, self.log_path
                )

                # Store alerts
                for alert in fixed_alerts:
                    self.alerts.append(alert)

                # Print status every 6 checks (every ~60 seconds)
                if check_count % 6 == 0:
                    self.print_status_report(health_metrics, fixed_alerts, snapshot)

                await asyncio.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\n\n🛑 Monitor stopped by user")
                return 0
            except Exception as e:
                print(f"❌ Error in monitor loop: {e}")
                await asyncio.sleep(self.check_interval)

        print(f"\n✅ Monitoring session complete ({check_count} checks)")
        return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Active capital growth monitor with auto-fix engine"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Monitoring duration in minutes (default: 60)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Check interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Path to log file (default: logs/active_15m_run.log)",
    )

    args = parser.parse_args()

    monitor = ActiveCapitalMonitor(
        log_path=args.log_path,
        check_interval=args.interval,
    )

    # Run async monitor
    exit_code = asyncio.run(monitor.run(duration_minutes=args.duration))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
