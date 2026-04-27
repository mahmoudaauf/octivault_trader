"""
EXIT-FIRST Strategy — Exit Metrics Tracker
===========================================
Tracks which exit pathway fired for each closed position so you can measure
whether the system is operating within EXIT_FIRST targets:

    Target distribution: ~40% TP  |  ~30% SL  |  ~30% TIME  |  ~0% DUST

If TIME exits exceed 60%, signals are too slow to reach TP/SL within the hold window.
If DUST > 0%, capital is still getting trapped (raise MIN_ECONOMIC_TRADE_USDT).
"""

import logging
import time
from typing import Dict, Optional, List

logger = logging.getLogger("ExitMetrics")


class ExitMetricsTracker:
    """
    Track exit pathway distribution and hold-time statistics.

    Usage:
        tracker = ExitMetricsTracker()
        # After each closed position:
        tracker.record_exit("TAKE_PROFIT", entry_price=100, exit_price=102.5,
                            quantity=0.1, hold_time_sec=1200)
        tracker.print_summary()
    """

    PATHWAYS = ("TAKE_PROFIT", "STOP_LOSS", "TIME_BASED", "DUST_ROUTED")

    def __init__(self):
        self._stats: Dict[str, Dict] = {
            "TAKE_PROFIT": {"count": 0, "total_pnl": 0.0, "hold_times": []},
            "STOP_LOSS":   {"count": 0, "total_pnl": 0.0, "hold_times": []},
            "TIME_BASED":  {"count": 0, "total_pnl": 0.0, "hold_times": []},
            "DUST_ROUTED": {"count": 0, "total_pnl": 0.0, "hold_times": []},
        }
        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_exit(
        self,
        exit_pathway: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        hold_time_sec: float,
    ) -> None:
        """Record one closed position and update running statistics."""
        pathway = str(exit_pathway).upper()
        if pathway not in self._stats:
            logger.warning("[ExitMetrics] Unknown pathway '%s', falling back to DUST_ROUTED", pathway)
            pathway = "DUST_ROUTED"

        pnl = (exit_price - entry_price) * quantity
        bucket = self._stats[pathway]
        bucket["count"] += 1
        bucket["total_pnl"] += pnl
        bucket["hold_times"].append(hold_time_sec)

        logger.info(
            "[ExitMetrics] %s exit recorded: pnl=%.4f hold=%.1fs "
            "(TP=%d SL=%d TIME=%d DUST=%d)",
            pathway, pnl, hold_time_sec,
            self._stats["TAKE_PROFIT"]["count"],
            self._stats["STOP_LOSS"]["count"],
            self._stats["TIME_BASED"]["count"],
            self._stats["DUST_ROUTED"]["count"],
        )

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def total_exits(self) -> int:
        return sum(b["count"] for b in self._stats.values())

    def get_distribution(self) -> Dict[str, float]:
        """Return percentage share of each pathway (0–100)."""
        total = self.total_exits()
        if total == 0:
            return {p: 0.0 for p in self.PATHWAYS}
        return {p: self._stats[p]["count"] / total * 100.0 for p in self.PATHWAYS}

    def avg_hold_sec(self, pathway: str) -> float:
        times = self._stats.get(pathway, {}).get("hold_times", [])
        return sum(times) / len(times) if times else 0.0

    def total_pnl(self) -> float:
        return sum(b["total_pnl"] for b in self._stats.values())

    # ------------------------------------------------------------------
    # Health assessment
    # ------------------------------------------------------------------

    def health_status(self) -> str:
        """
        Returns 'GREEN', 'YELLOW', or 'RED' based on EXIT_FIRST targets.

        GREEN  — TIME < 40%, DUST == 0, total exits > 0
        YELLOW — TIME 40-60% or DUST == 0 with < 5 trades
        RED    — TIME > 60% or DUST > 0
        """
        total = self.total_exits()
        if total == 0:
            return "YELLOW"  # No data yet

        dist = self.get_distribution()
        if dist["DUST_ROUTED"] > 0:
            return "RED"
        if dist["TIME_BASED"] > 60.0:
            return "RED"
        if dist["TIME_BASED"] > 40.0 or total < 5:
            return "YELLOW"
        return "GREEN"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_dict(self) -> Dict:
        dist = self.get_distribution()
        return {
            "total_exits": self.total_exits(),
            "distribution_pct": dist,
            "avg_hold_sec": {p: self.avg_hold_sec(p) for p in self.PATHWAYS},
            "total_pnl_usdt": self.total_pnl(),
            "health": self.health_status(),
            "session_elapsed_h": (time.time() - self._session_start) / 3600.0,
        }

    def print_summary(self) -> None:
        total = self.total_exits()
        dist = self.get_distribution()
        health = self.health_status()
        elapsed_h = (time.time() - self._session_start) / 3600.0

        print(
            f"\n{'=' * 55}\n"
            f"  EXIT-FIRST METRICS SUMMARY  |  {health}  |  {elapsed_h:.1f}h\n"
            f"{'=' * 55}\n"
            f"  Total exits  : {total}\n"
            f"  Total PnL    : ${self.total_pnl():.4f}\n"
            f"\n"
            f"  Take Profit  : {dist['TAKE_PROFIT']:.1f}%  "
            f"(n={self._stats['TAKE_PROFIT']['count']}  "
            f"avg_hold={self.avg_hold_sec('TAKE_PROFIT') / 60:.1f}m  "
            f"pnl=${self._stats['TAKE_PROFIT']['total_pnl']:.4f})\n"
            f"  Stop Loss    : {dist['STOP_LOSS']:.1f}%  "
            f"(n={self._stats['STOP_LOSS']['count']}  "
            f"avg_hold={self.avg_hold_sec('STOP_LOSS') / 60:.1f}m  "
            f"pnl=${self._stats['STOP_LOSS']['total_pnl']:.4f})\n"
            f"  Time-Based   : {dist['TIME_BASED']:.1f}%  "
            f"(n={self._stats['TIME_BASED']['count']}  "
            f"avg_hold={self.avg_hold_sec('TIME_BASED') / 60:.1f}m  "
            f"pnl=${self._stats['TIME_BASED']['total_pnl']:.4f})\n"
            f"  Dust Routed  : {dist['DUST_ROUTED']:.1f}%  "
            f"(n={self._stats['DUST_ROUTED']['count']}  "
            f"pnl=${self._stats['DUST_ROUTED']['total_pnl']:.4f})\n"
            f"\n"
            f"  Targets: TP≈40%  SL≈30%  TIME≈30%  DUST=0%\n"
            f"{'=' * 55}\n"
        )

    def warn_if_unhealthy(self) -> None:
        """Log a warning if distribution drifts outside EXIT_FIRST targets."""
        status = self.health_status()
        if status == "GREEN":
            return
        dist = self.get_distribution()
        if dist["DUST_ROUTED"] > 0:
            logger.warning(
                "[ExitMetrics] 🔴 DUST exits detected (%.1f%%). "
                "Capital is still getting trapped. Check MIN_ECONOMIC_TRADE_USDT.",
                dist["DUST_ROUTED"],
            )
        if dist["TIME_BASED"] > 60.0:
            logger.warning(
                "[ExitMetrics] 🔴 TIME exits dominating (%.1f%% > 60%% threshold). "
                "Signals are too slow — TP/SL rarely triggered before 4h force-close.",
                dist["TIME_BASED"],
            )
        elif dist["TIME_BASED"] > 40.0:
            logger.warning(
                "[ExitMetrics] 🟡 TIME exits elevated (%.1f%%). "
                "TP targets may be too wide or signals may be weak.",
                dist["TIME_BASED"],
            )


# ---------------------------------------------------------------------------
# Module-level singleton — import this in any module that records exits
# ---------------------------------------------------------------------------
_tracker: Optional[ExitMetricsTracker] = None


def get_tracker() -> ExitMetricsTracker:
    """Return the module-level singleton tracker (create on first call)."""
    global _tracker
    if _tracker is None:
        _tracker = ExitMetricsTracker()
    return _tracker


def record_exit(
    exit_pathway: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    hold_time_sec: float,
) -> None:
    """Convenience wrapper — records to the singleton tracker."""
    get_tracker().record_exit(exit_pathway, entry_price, exit_price, quantity, hold_time_sec)
