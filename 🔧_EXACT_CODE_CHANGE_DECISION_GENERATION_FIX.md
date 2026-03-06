## EXACT CODE CHANGE: Decision Generation Bug Fix

### Location
File: `core/meta_controller.py`
Lines: 10945-10963
Function: `_build_decisions()`
Section: Signal filtering for new BUY positions

### BEFORE (Broken):
```python
best_sig = max(buy_sigs, key=lambda s: float(s.get("confidence", 0.0)))
agent_name = best_sig.get("agent", "Meta")
agent_budget = _wallet_budget_for(agent_name)  # ❌ Gets remaining budget (unreliable!)

if agent_budget >= significant_position_usdt:
    # Entry size sufficient
    filtered_buy_symbols.append(sym)
else:
    if has_existing_position:
        # Allow scaling existing position below significant threshold
        filtered_buy_symbols.append(sym)
    else:
        self.logger.warning(
            "[Meta:Layer1] 🚫 ENTRY_TOO_SMALL_PREVENT_DUST: %s | "
            "planned=%.2f < minimum=%.2f USD (DENIED)",
            sym, agent_budget, significant_position_usdt
        )
```

### AFTER (Fixed):
```python
best_sig = max(buy_sigs, key=lambda s: float(s.get("confidence", 0.0)))
agent_name = best_sig.get("agent", "Meta")
# FIX: Check planned_quote from signal, NOT agent remaining budget
# Agent budget fluctuates during cycle; signal's planned_quote is authoritative
signal_planned_quote = float(best_sig.get("_planned_quote") or best_sig.get("planned_quote") or 0.0)
if signal_planned_quote <= 0:
    # No planned quote in signal, calculate from agent budget
    signal_planned_quote = _wallet_budget_for(agent_name)

if signal_planned_quote >= significant_position_usdt:  # ✅ Uses signal's planned amount
    # Entry size sufficient
    filtered_buy_symbols.append(sym)
else:
    if has_existing_position:
        # Allow scaling existing position below significant threshold
        filtered_buy_symbols.append(sym)
    else:
        self.logger.warning(
            "[Meta:Layer1] 🚫 ENTRY_TOO_SMALL_PREVENT_DUST: %s | "
            "planned=%.2f < minimum=%.2f USD (DENIED)",
            sym, signal_planned_quote, significant_position_usdt  # ✅ Updated variable
        )
```

### Key Changes:
1. ✅ Line 10948: Changed from `agent_budget = _wallet_budget_for(agent_name)` to extracting `signal_planned_quote` from signal
2. ✅ Lines 10951-10953: Added fallback to agent budget only if signal has no planned_quote
3. ✅ Line 10955: Updated condition to check `signal_planned_quote` instead of `agent_budget`
4. ✅ Line 10963: Updated log parameter from `agent_budget` to `signal_planned_quote`

### Problem This Fixes:
- **Before**: New BUY signals were rejected if agent's remaining budget < minimum position size
- **After**: BUY signals are evaluated based on their own planned allocation, not agent's remaining budget
- **Result**: Valid signals now convert to decisions and execute as intended

### Dependency Chain:
1. Allocator assigns capital to agents → stores in signal's `_planned_quote`
2. MetaController evaluates signals → **NOW uses signal's `_planned_quote`** (was using exhausted agent budget)
3. ExecutionManager executes the trade with the allocated quote

### Impact Verification:
Look for in logs after fix:
- ✅ `decisions_count > 0` (was `= 0`)
- ✅ `FILLED` order confirmations (was `0`)
- ✅ Execution requests converting to actual trades
