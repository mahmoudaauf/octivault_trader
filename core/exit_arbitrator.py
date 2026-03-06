"""
Exit Arbitrator: Deterministic priority-based exit resolution.

This module provides the ExitArbitrator class, which implements the professional
pattern for resolving competing exit signals in a multi-tier hierarchy.

Instead of suppressing exits based on state, it assigns explicit priorities
and always executes the highest-priority available exit.

Priority Order (lowest number = highest priority):
  1. RISK_EXIT (capital floor, starvation, dust, liquidation)
  2. TP_SL_EXIT (take-profit, stop-loss)
  3. SIGNAL_EXIT (agent recommendations)
  4. ROTATION_EXIT (universe rotation)
  5. REBALANCE_EXIT (portfolio rebalancing)

Example:
    ```python
    arbitrator = ExitArbitrator(logger=logger)
    
    # Collect candidates
    risk_exit = await meta._evaluate_risk_exit("BTC/USDT", position)
    tp_sl_exit = await meta._evaluate_tp_sl_exit("BTC/USDT", position)
    signal_exits = [s for s in signals if s.get("action") == "SELL"]
    
    # Resolve conflict
    exit_type, exit_signal = await arbitrator.resolve_exit(
        symbol="BTC/USDT",
        position=position,
        risk_exit=risk_exit,
        tp_sl_exit=tp_sl_exit,
        signal_exits=signal_exits,
    )
    
    if exit_type:
        await meta._execute_exit(symbol, exit_signal, reason=exit_type)
    ```
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import logging


class ExitPriority(IntEnum):
    """Exit priority ranking (lower = higher priority)."""
    RISK = 1
    TP_SL = 2
    SIGNAL = 3
    ROTATION = 4
    REBALANCE = 5


@dataclass
class ExitCandidate:
    """A single exit candidate with priority and metadata."""
    exit_type: str  # "RISK", "TP_SL", "SIGNAL", "ROTATION", "REBALANCE"
    signal: Dict[str, Any]
    priority: int
    reason: str  # Human-readable reason for this exit


class ExitArbitrator:
    """
    Deterministic exit resolution using explicit priority mapping.
    
    This class implements the professional pattern for resolving conflicts
    between risk-driven, profit-aware, and signal-driven exits.
    
    Instead of using implicit if-elif chains, it collects all candidates
    and applies an explicit priority map to determine the winner.
    
    Attributes:
        logger: Logger instance for recording arbitration decisions
        priority_map: Dict mapping exit types to numeric priorities
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize ExitArbitrator.
        
        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Priority map - easily configurable
        self.priority_map = {
            "RISK": ExitPriority.RISK,
            "TP_SL": ExitPriority.TP_SL,
            "SIGNAL": ExitPriority.SIGNAL,
            "ROTATION": ExitPriority.ROTATION,
            "REBALANCE": ExitPriority.REBALANCE,
        }
    
    async def resolve_exit(
        self,
        symbol: str,
        position: Dict[str, Any],
        risk_exit: Optional[Dict[str, Any]] = None,
        tp_sl_exit: Optional[Dict[str, Any]] = None,
        signal_exits: List[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Resolve competing exit signals using priority ranking.
        
        Collects all available exit candidates and returns only the highest-priority
        one. All other candidates are suppressed and logged.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            position: Current position data dict
            risk_exit: Risk-driven exit signal (if any, highest priority)
            tp_sl_exit: TP/SL exit signal (if any, medium priority)
            signal_exits: List of agent/rotation/rebalance signals (if any, lower priority)
        
        Returns:
            Tuple of (exit_type, exit_signal) or (None, None) if no exits available.
            
            exit_type: String indicating exit category:
                - "RISK": Capital floor, starvation, dust, liquidation
                - "TP_SL": Take-profit or stop-loss
                - "SIGNAL": Agent recommendation
                - "ROTATION": Universe rotation
                - "REBALANCE": Portfolio rebalancing
            exit_signal: The dict signal to execute
        
        Raises:
            None (designed to be robust)
        """
        
        candidates = []
        signal_exits = signal_exits or []
        
        # Tier 1: Risk exits (highest priority)
        if risk_exit:
            candidates.append(
                ExitCandidate(
                    exit_type="RISK",
                    signal=risk_exit,
                    priority=self.priority_map["RISK"],
                    reason=risk_exit.get("reason", "Risk-driven exit"),
                )
            )
        
        # Tier 2: TP/SL exits
        if tp_sl_exit:
            candidates.append(
                ExitCandidate(
                    exit_type="TP_SL",
                    signal=tp_sl_exit,
                    priority=self.priority_map["TP_SL"],
                    reason=tp_sl_exit.get("reason", "TP/SL exit"),
                )
            )
        
        # Tier 3: Signal-based exits (categorized by type)
        for signal in signal_exits:
            tag = str(signal.get("tag", "")).lower()
            
            if "rotation" in tag:
                candidates.append(
                    ExitCandidate(
                        exit_type="ROTATION",
                        signal=signal,
                        priority=self.priority_map["ROTATION"],
                        reason=f"Rotation exit: {signal.get('reason', 'symbol rotation')}",
                    )
                )
            elif "rebalance" in tag:
                candidates.append(
                    ExitCandidate(
                        exit_type="REBALANCE",
                        signal=signal,
                        priority=self.priority_map["REBALANCE"],
                        reason=f"Rebalance exit: {signal.get('reason', 'portfolio rebalancing')}",
                    )
                )
            else:
                # Generic signal exit
                candidates.append(
                    ExitCandidate(
                        exit_type="SIGNAL",
                        signal=signal,
                        priority=self.priority_map["SIGNAL"],
                        reason=signal.get("reason", "Agent signal exit"),
                    )
                )
        
        # No candidates - no exit
        if not candidates:
            return None, None
        
        # Sort by priority (lower priority value = higher priority)
        candidates.sort(key=lambda c: c.priority)
        
        # Winner is the first after sort
        winner = candidates[0]
        
        # Log arbitration (if there were conflicts)
        if len(candidates) > 1:
            suppressed = [
                {
                    "type": c.exit_type,
                    "priority": c.priority,
                    "reason": c.reason,
                }
                for c in candidates[1:]
            ]
            
            self.logger.info(
                f"[ExitArbitration] Symbol={symbol} "
                f"Winner={winner.exit_type} (priority={winner.priority}) "
                f"Suppressed={len(suppressed)} "
                f"Details: {suppressed}"
            )
        else:
            self.logger.debug(
                f"[ExitArbitration] Symbol={symbol} "
                f"Exit={winner.exit_type} (priority={winner.priority}) "
                f"Reason={winner.reason}"
            )
        
        return winner.exit_type, winner.signal
    
    def set_priority(self, exit_type: str, priority: int) -> None:
        """
        Adjust exit priority at runtime.
        
        Allows dynamic priority adjustment without code changes.
        
        Args:
            exit_type: One of "RISK", "TP_SL", "SIGNAL", "ROTATION", "REBALANCE"
            priority: Numeric priority (lower = higher priority)
        
        Raises:
            ValueError: If exit_type is not recognized
        
        Example:
            ```python
            # Make rotation exits higher priority than signals
            arbitrator.set_priority("ROTATION", 2)
            arbitrator.set_priority("SIGNAL", 3)
            ```
        """
        if exit_type not in self.priority_map:
            raise ValueError(f"Unknown exit type: {exit_type}")

        # Safety guard: RISK must always remain the highest-priority exit (lowest number).
        # Demoting RISK below TP_SL would disable capital protection.
        if exit_type == "RISK" and priority > ExitPriority.TP_SL:
            raise ValueError(
                f"RISK exit priority cannot be demoted below TP_SL "
                f"(attempted {priority}, max allowed {ExitPriority.TP_SL}). "
                "Capital protection must remain the top-priority exit."
            )
        if exit_type != "RISK" and priority <= ExitPriority.RISK:
            raise ValueError(
                f"Non-RISK exit type '{exit_type}' cannot be promoted above RISK priority "
                f"(attempted {priority}, RISK is {ExitPriority.RISK}). "
                "Only RISK exits may hold the highest priority."
            )

        import traceback
        caller = "".join(traceback.format_stack(limit=3)[-2:-1]).strip()
        old_priority = self.priority_map[exit_type]
        self.priority_map[exit_type] = priority
        self.logger.warning(
            "[ExitArbitrator] Priority changed: %s %d→%d | caller: %s",
            exit_type, old_priority, priority, caller,
        )
    
    def get_priority_order(self) -> List[Tuple[str, int]]:
        """
        Get current priority order (sorted by priority).
        
        Returns:
            List of (exit_type, priority) tuples sorted by priority (ascending)
        
        Example:
            ```python
            order = arbitrator.get_priority_order()
            # [("RISK", 1), ("TP_SL", 2), ("SIGNAL", 3), ...]
            ```
        """
        items = list(self.priority_map.items())
        items.sort(key=lambda x: x[1])
        return items
    
    def reset_priorities(self) -> None:
        """Reset all priorities to default values."""
        self.priority_map = {
            "RISK": ExitPriority.RISK,
            "TP_SL": ExitPriority.TP_SL,
            "SIGNAL": ExitPriority.SIGNAL,
            "ROTATION": ExitPriority.ROTATION,
            "REBALANCE": ExitPriority.REBALANCE,
        }
        self.logger.info("[ExitArbitrator] Priorities reset to defaults")


# Module-level singleton (optional)
_default_arbitrator: Optional[ExitArbitrator] = None


def get_arbitrator(logger: logging.Logger = None) -> ExitArbitrator:
    """
    Get or create the default arbitrator instance.
    
    Args:
        logger: Optional logger for new instance
    
    Returns:
        The global ExitArbitrator instance
    """
    global _default_arbitrator
    if _default_arbitrator is None:
        _default_arbitrator = ExitArbitrator(logger=logger)
    return _default_arbitrator
