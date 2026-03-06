# ExitArbitrator: Institutional-Grade Exit Hierarchy

## Overview

The **ExitArbitrator** is a decision layer that resolves conflicts between competing exit signals using explicit priority mapping rather than fragile code ordering.

**Why This Matters:**
- Current system relies on if-then-else chains (fragile)
- No visibility into why one exit "wins" over another
- Hard to modify priority without rewriting logic
- Professional risk systems use explicit arbitration

---

## Architecture

```
MetaController.execute_trading_cycle()
    ↓
For each symbol: _collect_all_exits(symbol)
    ├─ Risk exits (starvation, dust, capital floor)
    ├─ TP/SL exits (profit-taking, stop-loss)
    ├─ Signal exits (agent, rotation, rebalance)
    └─ Collected in list (no priority yet)
    ↓
ExitArbitrator._resolve_exit(symbol, exits)
    ├─ Apply priority_map
    ├─ Sort by priority
    ├─ Select highest-priority exit
    ├─ Log suppressed alternatives
    └─ Return (exit_type, exit_signal)
    ↓
MetaController._execute_exit(symbol, signal, reason=exit_type)
    └─ Execute only the arbitrated exit
```

---

## Implementation: ExitArbitrator Class

### File: `core/exit_arbitrator.py`

```python
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
    """
    
    def __init__(self, logger: logging.Logger = None):
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
        
        Args:
            symbol: Trading pair symbol
            position: Current position data
            risk_exit: Risk-driven exit signal (if any)
            tp_sl_exit: TP/SL exit signal (if any)
            signal_exits: List of agent/rotation/rebalance signals
        
        Returns:
            (exit_type, exit_signal) or (None, None) if no exits
        
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
                print(f"Executing {exit_type} exit for BTC/USDT")
            ```
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
        
        Args:
            exit_type: "RISK", "TP_SL", "SIGNAL", "ROTATION", "REBALANCE"
            priority: Numeric priority (lower = higher priority)
        
        Example:
            ```python
            # Make rotation exits higher priority than signals
            arbitrator.set_priority("ROTATION", 2)
            arbitrator.set_priority("SIGNAL", 3)
            ```
        """
        if exit_type not in self.priority_map:
            raise ValueError(f"Unknown exit type: {exit_type}")
        
        self.priority_map[exit_type] = priority
        self.logger.info(f"[ExitArbitrator] Priority updated: {exit_type}={priority}")
    
    def get_priority_order(self) -> List[Tuple[str, int]]:
        """Get current priority order (sorted by priority)."""
        items = list(self.priority_map.items())
        items.sort(key=lambda x: x[1])
        return items


# Module-level singleton (optional)
_default_arbitrator: Optional[ExitArbitrator] = None


def get_arbitrator(logger: logging.Logger = None) -> ExitArbitrator:
    """Get or create the default arbitrator instance."""
    global _default_arbitrator
    if _default_arbitrator is None:
        _default_arbitrator = ExitArbitrator(logger=logger)
    return _default_arbitrator
```

---

## Integration: MetaController Changes

### In `core/meta_controller.py`

#### Step 1: Import

```python
from core.exit_arbitrator import ExitArbitrator
```

#### Step 2: Initialize in `__init__`

```python
def __init__(self, ...):
    # ... existing init code ...
    
    # Add exit arbitrator
    self.exit_arbitrator = ExitArbitrator(logger=self.logger)
```

#### Step 3: Create `_collect_exits()` method

```python
async def _collect_exits(
    self,
    symbol: str,
    position: Dict[str, Any],
    valid_signals: List[Dict[str, Any]],
) -> Tuple[Optional[Dict], Optional[Dict], List[Dict]]:
    """
    Collect all competing exit candidates.
    
    Returns:
        (risk_exit, tp_sl_exit, signal_exits)
    """
    
    # Risk exits
    risk_exit = await self._evaluate_risk_exit(symbol, position)
    
    # TP/SL exits
    tp_sl_exit = await self._evaluate_tp_sl_exit(symbol, position)
    
    # Signal-based exits
    signal_exits = [s for s in valid_signals if s.get("action") == "SELL"]
    
    return risk_exit, tp_sl_exit, signal_exits
```

#### Step 4: Modify `execute_trading_cycle()` exit handling

**Replace:**
```python
# OLD (fragile)
if risk_condition and risk_condition.force_exit:
    await self._execute_sell(symbol, risk_condition.signal)
elif tp_sl_signal:
    await self._execute_sell(symbol, tp_sl_signal)
elif agent_sell_signal:
    await self._execute_sell(symbol, agent_sell_signal)
```

**With:**
```python
# NEW (clean arbitration)
risk_exit, tp_sl_exit, signal_exits = await self._collect_exits(
    symbol, position, valid_signals
)

exit_type, exit_signal = await self.exit_arbitrator.resolve_exit(
    symbol=symbol,
    position=position,
    risk_exit=risk_exit,
    tp_sl_exit=tp_sl_exit,
    signal_exits=signal_exits,
)

if exit_type:
    await self._execute_sell(symbol, exit_signal, reason=exit_type)
```

---

## Observability Benefits

### Before (Current)
```
[MetaController] Position closed for BTC/USDT
(No visibility into WHY or WHAT triggered the exit)
```

### After (With Arbitrator)
```
[ExitArbitration] Symbol=BTC/USDT Winner=TP_SL (priority=2) 
Suppressed=1 Details: [
  {'type': 'SIGNAL', 'priority': 3, 'reason': 'Agent sell signal from StrategyManager'}
]

[MetaController] Executing TP exit for BTC/USDT (reason=TP_SL)
(Crystal clear: why this exit won, and what was suppressed)
```

---

## Configuration: Runtime Priority Adjustment

```python
# In your config or startup code
arbitrator = ExitArbitrator(logger=logger)

# Make rotation exits very high priority
arbitrator.set_priority("ROTATION", 1.5)

# Make signal exits lower priority
arbitrator.set_priority("SIGNAL", 3.5)

print("Current Priority Order:")
for exit_type, priority in arbitrator.get_priority_order():
    print(f"  {exit_type}: {priority}")
```

---

## Testing Strategy

```python
async def test_exit_arbitration():
    """Test that highest-priority exit always wins."""
    
    arbitrator = ExitArbitrator()
    
    # Scenario 1: Risk exit vs Signal exit
    risk_exit = {"action": "SELL", "reason": "Starvation"}
    signal_exit = {"action": "SELL", "reason": "Agent signal", "tag": "signal"}
    
    exit_type, signal = await arbitrator.resolve_exit(
        symbol="BTC/USDT",
        position={},
        risk_exit=risk_exit,
        signal_exits=[signal_exit],
    )
    
    assert exit_type == "RISK", "Risk exit should beat signal exit"
    
    # Scenario 2: TP/SL vs Signal exit
    tp_sl_exit = {"action": "SELL", "reason": "Take-profit", "tag": "tp_sl"}
    
    exit_type, signal = await arbitrator.resolve_exit(
        symbol="BTC/USDT",
        position={},
        tp_sl_exit=tp_sl_exit,
        signal_exits=[signal_exit],
    )
    
    assert exit_type == "TP_SL", "TP/SL exit should beat signal exit"
    
    # Scenario 3: No exits
    exit_type, signal = await arbitrator.resolve_exit(
        symbol="BTC/USDT",
        position={},
    )
    
    assert exit_type is None, "Should return None when no exits available"
    
    print("✅ All arbitration tests passed")
```

---

## Professional Advantages

| Aspect | Without Arbitrator | With Arbitrator |
|--------|-------------------|-----------------|
| **Readability** | Nested if-else chains | Clear method with logic flow |
| **Maintainability** | Hard to modify priority | Edit priority_map, no code changes |
| **Observability** | Implicit (no logging) | Explicit logging of all conflicts |
| **Testability** | Tightly coupled | Easy to test in isolation |
| **Extensibility** | Add new exit type = refactor | Add to priority_map + no changes |
| **Risk Clarity** | "Why did this exit?" unclear | Clear audit trail |
| **Production Readiness** | Ad-hoc | Enterprise-grade pattern |

---

## Summary

### What This Gives You:

✅ **Deterministic priority** - No ambiguity about which exit wins
✅ **Transparency** - See exactly why each decision was made
✅ **Modularity** - Exit evaluation decoupled from arbitration
✅ **Observability** - Audit trail of suppressed alternatives
✅ **Flexibility** - Change priorities without rewriting logic
✅ **Professional** - Standard pattern in institutional risk systems

### Implementation Effort:

- `exit_arbitrator.py`: ~250 lines (self-contained)
- MetaController integration: ~50 lines (minimal changes)
- Testing: ~200 lines
- **Total: 2-3 hours**

### ROI:

Once implemented, this becomes the canonical exit decision mechanism. All future exit types, modifications, and audits flow through a single, transparent interface.

---

**Ready to implement? Let's do it.** 🚀
