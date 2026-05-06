"""
Native L7: Prometheus Exporter — Metrics Export for Grafana/Prometheus

Collects and exports trading metrics in Prometheus format:
- NAV and P&L metrics
- Trade statistics (count, win rate, profit factor)
- Cycle performance (duration, errors)
- Exchange latency
- Signal metrics

Design: Lightweight collector that dumps metrics to stdout and optionally
to a file for scraping by Prometheus or Grafana.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Single snapshot of all trading metrics."""

    timestamp: float
    nav_usdt: float
    nav_peak_usdt: float
    pnl_total_usdt: float
    pnl_pct: float
    trade_count: int
    win_count: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    cycle_duration_ms: float
    cycle_count: int
    signal_count: int
    execution_count: int
    execution_errors: int


class NativePrometheusExporter:
    """Collects and exports metrics in Prometheus format."""

    def __init__(self, output_file: Optional[str] = None) -> None:
        """
        Initialize exporter.

        Args:
            output_file: Optional file path to write Prometheus metrics
        """
        self._output_file = Path(output_file) if output_file else None
        self._snapshots: list[MetricsSnapshot] = []
        self._metric_count = 0

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Record a metrics snapshot.

        Args:
            metrics: Dict with keys matching MetricsSnapshot fields

        Example:
            exporter.record_metrics({
                "timestamp": time.time(),
                "nav_usdt": 10500.0,
                "pnl_total_usdt": 500.0,
                "trade_count": 5,
                ...
            })
        """
        try:
            snapshot = MetricsSnapshot(
                timestamp=metrics.get("timestamp", time.time()),
                nav_usdt=float(metrics.get("nav_usdt", 0.0)),
                nav_peak_usdt=float(metrics.get("nav_peak_usdt", 0.0)),
                pnl_total_usdt=float(metrics.get("pnl_total_usdt", 0.0)),
                pnl_pct=float(metrics.get("pnl_pct", 0.0)),
                trade_count=int(metrics.get("trade_count", 0)),
                win_count=int(metrics.get("win_count", 0)),
                win_rate=float(metrics.get("win_rate", 0.0)),
                profit_factor=float(metrics.get("profit_factor", 0.0)),
                sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
                max_drawdown_pct=float(metrics.get("max_drawdown_pct", 0.0)),
                cycle_duration_ms=float(metrics.get("cycle_duration_ms", 0.0)),
                cycle_count=int(metrics.get("cycle_count", 0)),
                signal_count=int(metrics.get("signal_count", 0)),
                execution_count=int(metrics.get("execution_count", 0)),
                execution_errors=int(metrics.get("execution_errors", 0)),
            )
            self._snapshots.append(snapshot)
            self._metric_count += 1
        except Exception as e:
            logger.exception(f"Failed to record metrics: {e}")

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        if not self._snapshots:
            return "# No metrics recorded yet\n"

        latest = self._snapshots[-1]

        lines = [
            "# HELP trading_nav Net Asset Value in USDT",
            "# TYPE trading_nav gauge",
            f"trading_nav{{}} {latest.nav_usdt}",
            "",
            "# HELP trading_pnl Total P&L in USDT",
            "# TYPE trading_pnl gauge",
            f"trading_pnl{{}} {latest.pnl_total_usdt}",
            "",
            "# HELP trading_pnl_pct P&L percentage",
            "# TYPE trading_pnl_pct gauge",
            f"trading_pnl_pct{{}} {latest.pnl_pct}",
            "",
            "# HELP trading_trades Total trade count",
            "# TYPE trading_trades counter",
            f"trading_trades{{}} {latest.trade_count}",
            "",
            "# HELP trading_win_rate Win rate percentage",
            "# TYPE trading_win_rate gauge",
            f"trading_win_rate{{}} {latest.win_rate}",
            "",
            "# HELP trading_profit_factor Profit factor (wins/losses)",
            "# TYPE trading_profit_factor gauge",
            f"trading_profit_factor{{}} {latest.profit_factor}",
            "",
            "# HELP trading_sharpe_ratio Sharpe ratio",
            "# TYPE trading_sharpe_ratio gauge",
            f"trading_sharpe_ratio{{}} {latest.sharpe_ratio}",
            "",
            "# HELP trading_max_drawdown Maximum drawdown percentage",
            "# TYPE trading_max_drawdown gauge",
            f"trading_max_drawdown{{}} {latest.max_drawdown_pct}",
            "",
            "# HELP trading_cycle_duration_ms Average cycle duration in ms",
            "# TYPE trading_cycle_duration_ms gauge",
            f"trading_cycle_duration_ms{{}} {latest.cycle_duration_ms}",
            "",
            "# HELP trading_cycles Total cycles completed",
            "# TYPE trading_cycles counter",
            f"trading_cycles{{}} {latest.cycle_count}",
            "",
            "# HELP trading_signals Total signals generated",
            "# TYPE trading_signals counter",
            f"trading_signals{{}} {latest.signal_count}",
            "",
            "# HELP trading_executions Total executions",
            "# TYPE trading_executions counter",
            f"trading_executions{{}} {latest.execution_count}",
            "",
            "# HELP trading_execution_errors Execution errors",
            "# TYPE trading_execution_errors counter",
            f"trading_execution_errors{{}} {latest.execution_errors}",
        ]

        return "\n".join(lines) + "\n"

    def write_prometheus_file(self, path: Optional[Path] = None) -> None:
        """
        Write Prometheus metrics to file.

        Args:
            path: File path (uses self._output_file if not provided)
        """
        file_path = path or self._output_file
        if not file_path:
            logger.warning("No output file configured for Prometheus export")
            return

        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            metrics = self.export_prometheus()
            file_path.write_text(metrics)
            logger.info(f"Prometheus metrics written to {file_path}")
        except Exception as e:
            logger.exception(f"Failed to write Prometheus metrics: {e}")

    def export_json(self) -> str:
        """
        Export metrics as JSON (for API responses).

        Returns:
            JSON-formatted metrics string
        """
        if not self._snapshots:
            return "{}"

        latest = self._snapshots[-1]
        return json.dumps(asdict(latest), default=str)

    def get_latest_snapshot(self) -> Optional[MetricsSnapshot]:
        """Get the most recent metrics snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshot_count(self) -> int:
        """Get total snapshots recorded."""
        return len(self._snapshots)


__all__ = ["NativePrometheusExporter", "MetricsSnapshot"]
