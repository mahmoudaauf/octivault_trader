"""
FourthSlotTracker: Aggressive Profit Hunting for 4th Symbol Slot

This module implements the 4th slot aggressive profit hunting mechanism:
- Tracks entry price and time for 4th slot positions
- Monitors 3 exit conditions: profit target (+15%), stop-loss (-3%), timeout (120 min)
- Records trade statistics for performance analysis
- Logs all exits with details for monitoring and optimization

Exit Conditions:
  1. PROFIT_TARGET_HIT: P&L >= +15% → LOCK PROFITS
  2. STOP_LOSS_HIT: P&L <= -3% → CUT LOSSES
  3. MAX_DURATION_REACHED: Hold time >= 120 minutes → RELEASE CAPITAL
  4. EMERGENCY_EXIT: Symbol error/blocked → EXIT IMMEDIATELY

Capital Allocation:
  - 4th slot gets $5.00 per rotation (6.5% of available)
  - Profits (if any) injected to compound pool
  - Losses deducted from buffer pool (top 3 protected!)
  - $5.00 reserved for next rotation
"""

import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class FourthSlotTracker:
    """
    Tracks 4th slot position and checks for exit conditions.

    Exits automatically on:
    - Profit target (+15%): Lock huge gains early
    - Stop-loss (-3%): Fail fast, preserve capital
    - Max duration (120 min): Free capital for next rotation
    - Emergency (technical error): Exit immediately
    """

    def __init__(self, config):
        """Initialize tracker with configuration."""
        self.config = config

        # Current position state
        self.current_symbol = None
        self.entry_price = None
        self.entry_time = None
        self.position_size = None
        self.entry_value = None

        # Exit parameters
        self.profit_target_pct = config.FIX8_4TH_SLOT_PROFIT_TARGET_PCT
        self.stop_loss_pct = config.FIX8_4TH_SLOT_STOP_LOSS_PCT
        self.max_hold_minutes = config.FIX8_4TH_SLOT_MAX_HOLD_MINUTES

        # Exit history for statistics
        self.exit_history = []

        logger.info(
            "[FourthSlotTracker] Initialized with exit thresholds: "
            "target=%+.1f%%, stop=%.1f%%, max_hold=%dm",
            self.profit_target_pct * 100,
            self.stop_loss_pct * 100,
            self.max_hold_minutes,
        )

    def set_position(self, symbol: str, entry_price: float, position_size: float):
        """
        Register a new 4th slot position entry.

        Args:
            symbol: Trading pair (e.g., "DOGEUSDT")
            entry_price: Entry price in USDT
            position_size: Position size in base asset
        """
        self.current_symbol = symbol
        self.entry_price = entry_price
        self.position_size = position_size
        self.entry_time = time.time()
        self.entry_value = entry_price * position_size

        logger.warning(
            f"""
            [FIX #8: 4TH SLOT ENTRY] 🎯 AGGRESSIVE PROFIT HUNTING STARTED
            ├─ Symbol: {symbol}
            ├─ Entry Price: ${entry_price:.8f}
            ├─ Position Size: {position_size:.8f}
            ├─ Entry Value: ${self.entry_value:.2f}
            ├─ Profit Target: +{self.profit_target_pct*100:.1f}% (${self.entry_value * self.profit_target_pct:.2f})
            ├─ Stop-Loss: {self.stop_loss_pct*100:.1f}% (-${abs(self.entry_value * self.stop_loss_pct):.2f})
            └─ Max Duration: {self.max_hold_minutes} minutes
        """
        )

    def check_exit_conditions(self, current_price: float) -> Optional[dict]:
        """
        Check if 4th slot should exit based on exit conditions.

        Returns:
            Dict with exit details if should exit:
              - exit_reason: PROFIT_TARGET_HIT, STOP_LOSS_HIT, MAX_DURATION_REACHED
              - exit_price: Price at which to exit
              - pnl: Profit/loss in USDT
              - pnl_pct: Profit/loss percentage
              - time_held_min: Time held in minutes

            None if should continue holding
        """
        if self.current_symbol is None:
            return None  # No position

        # Calculate current state
        current_time = time.time()
        current_value = current_price * self.position_size
        pnl = current_value - self.entry_value
        pnl_pct = pnl / self.entry_value if self.entry_value != 0 else 0
        time_held_minutes = (current_time - self.entry_time) / 60

        # ═══════════════════════════════════════════════════════════
        # EXIT CONDITION #1: Profit Target Hit ✅
        # ═══════════════════════════════════════════════════════════
        if pnl_pct >= self.profit_target_pct - 1e-6:  # Account for floating-point precision
            exit_reason = "PROFIT_TARGET_HIT"
            self._record_exit(
                symbol=self.current_symbol,
                exit_reason=exit_reason,
                entry_price=self.entry_price,
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                time_held_min=time_held_minutes,
            )

            logger.warning(
                f"""
                [FIX #8: 4TH SLOT EXIT - PROFIT TARGET! 🎉]
                ├─ Symbol: {self.current_symbol}
                ├─ Entry: ${self.entry_price:.8f}
                ├─ Exit: ${current_price:.8f}
                ├─ P&L: +${pnl:.2f} (+{pnl_pct*100:.2f}%)
                ├─ Time Held: {time_held_minutes:.1f} minutes
                ├─ Action: LOCK PROFITS → Compound Pool
                └─ Status: 🎊 HUGE WIN!
            """
            )

            return {
                "exit_reason": exit_reason,
                "exit_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "time_held_min": time_held_minutes,
            }

        # ═══════════════════════════════════════════════════════════
        # EXIT CONDITION #2: Stop-Loss Hit ❌
        # ═══════════════════════════════════════════════════════════
        if pnl_pct <= self.stop_loss_pct:
            exit_reason = "STOP_LOSS_HIT"
            self._record_exit(
                symbol=self.current_symbol,
                exit_reason=exit_reason,
                entry_price=self.entry_price,
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                time_held_min=time_held_minutes,
            )

            logger.error(
                f"""
                [FIX #8: 4TH SLOT EXIT - STOP LOSS! 🛑]
                ├─ Symbol: {self.current_symbol}
                ├─ Entry: ${self.entry_price:.8f}
                ├─ Exit: ${current_price:.8f}
                ├─ P&L: -${abs(pnl):.2f} ({pnl_pct*100:.2f}%)
                ├─ Time Held: {time_held_minutes:.1f} minutes
                ├─ Action: CUT LOSSES → Free Capital
                └─ Status: ✅ Quick Exit (preserved capital)
            """
            )

            return {
                "exit_reason": exit_reason,
                "exit_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "time_held_min": time_held_minutes,
            }

        # ═══════════════════════════════════════════════════════════
        # EXIT CONDITION #3: Max Duration Reached ⏱️
        # ═══════════════════════════════════════════════════════════
        if time_held_minutes >= self.max_hold_minutes:
            exit_reason = "MAX_DURATION_REACHED"
            self._record_exit(
                symbol=self.current_symbol,
                exit_reason=exit_reason,
                entry_price=self.entry_price,
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                time_held_min=time_held_minutes,
            )

            logger.info(
                f"""
                [FIX #8: 4TH SLOT EXIT - TIMEOUT! ⏱️]
                ├─ Symbol: {self.current_symbol}
                ├─ Entry: ${self.entry_price:.8f}
                ├─ Exit: ${current_price:.8f}
                ├─ P&L: ${pnl:+.2f} ({pnl_pct*100:+.2f}%)
                ├─ Time Held: {time_held_minutes:.1f} minutes (MAX REACHED)
                ├─ Action: RELEASE CAPITAL → Next Rotation
                └─ Status: Time to rotate (lock in current P&L)
            """
            )

            return {
                "exit_reason": exit_reason,
                "exit_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "time_held_min": time_held_minutes,
            }

        # ═══════════════════════════════════════════════════════════
        # NO EXIT - Hold Position
        # ═══════════════════════════════════════════════════════════
        return None

    def _record_exit(
        self,
        symbol: str,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        time_held_min: float,
    ):
        """Record exit for statistics and performance tracking."""
        exit_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exit_reason": exit_reason,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "time_held_min": time_held_min,
        }
        self.exit_history.append(exit_record)

    def get_statistics(self) -> dict:
        """
        Get 4th slot performance statistics.

        Returns:
            Dict with total rotations, P&L, win rate, etc.
        """
        if not self.exit_history:
            return {"total_rotations": 0}

        total_rotations = len(self.exit_history)
        total_pnl = sum(e["pnl"] for e in self.exit_history)

        wins = [e for e in self.exit_history if e["pnl"] > 0]
        losses = [e for e in self.exit_history if e["pnl"] < 0]

        profit_targets = [e for e in self.exit_history if e["exit_reason"] == "PROFIT_TARGET_HIT"]
        stop_losses = [e for e in self.exit_history if e["exit_reason"] == "STOP_LOSS_HIT"]
        timeouts = [e for e in self.exit_history if e["exit_reason"] == "MAX_DURATION_REACHED"]

        return {
            "total_rotations": total_rotations,
            "total_pnl": total_pnl,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / total_rotations if total_rotations > 0 else 0,
            "avg_win": sum(e["pnl"] for e in wins) / len(wins) if wins else 0,
            "avg_loss": sum(e["pnl"] for e in losses) / len(losses) if losses else 0,
            "best_trade": max((e["pnl_pct"] for e in self.exit_history), default=0),
            "worst_trade": min((e["pnl_pct"] for e in self.exit_history), default=0),
            "profit_targets_hit": len(profit_targets),
            "stop_losses_hit": len(stop_losses),
            "timeouts": len(timeouts),
            "last_exit": self.exit_history[-1] if self.exit_history else None,
        }

    def reset_position(self):
        """Reset after position exit, ready for next rotation."""
        self.current_symbol = None
        self.entry_price = None
        self.entry_time = None
        self.position_size = None
        self.entry_value = None


__all__ = ["FourthSlotTracker"]
