# Exit Arbitration: Quick Reference

## The Problem

**Current:** Exit decisions scattered across code
```python
if risk_condition.force:
    exit()
elif tp_sl_signal:
    exit()
elif agent_signal:
    exit()
# Where's the explicit priority? Buried in code order.
```

**Result:** Fragile, hard to audit, impossible to modify without breaking things.

---

## The Solution

**Explicit Priority Arbitration**
```python
exits = [
    ("RISK", risk_signal),
    ("TP_SL", tp_sl_signal),
    ("SIGNAL", agent_signal),
]

# Priority map
priority = {"RISK": 1, "TP_SL": 2, "SIGNAL": 3}

# Sort and execute highest priority
exits.sort(key=lambda x: priority[x[0]])
exit_type, signal = exits[0]  # Always the best one
```

**Result:** Clean, transparent, modifiable, auditable.

---

## Priority Tiers

```
┌────────┬──────────────────────────────────────┬──────┐
│ Rank   │ Exit Type                            │ When │
├────────┼──────────────────────────────────────┼──────┤
│ 1️⃣    │ RISK (starvation, dust, floor)       │ ALWAYS
│ 2️⃣    │ TP_SL (take-profit, stop-loss)       │ IF no risk
│ 3️⃣    │ SIGNAL (agent recommendations)       │ IF no risk/tp_sl
│ 4️⃣    │ ROTATION (portfolio rebalancing)     │ IF no higher
│ 5️⃣    │ REBALANCE (weight adjustments)       │ LAST resort
└────────┴──────────────────────────────────────┴──────┘

Lower number = Higher priority = Executes if available
```

---

## Why Arbitration > Suppression

### ❌ Suppression Approach
```python
# Block signal exits when risk condition exists
if risk_condition:
    block_signal_exits()
    execute_risk_exit()
```

**Problems:**
- Negative logic (what NOT to do)
- Coupling between tiers
- Hard to see suppression reasons
- Fragile to future changes

---

### ✅ Arbitration Approach
```python
# Collect ALL exits, let arbitrator pick the best
exits = [risk, tp_sl, signal]
winner = arbitrator.resolve(exits)
execute(winner)
```

**Benefits:**
- Positive logic (what TO do)
- Decoupled tiers
- Clear suppression logging
- Easy to modify priorities
- Professional pattern

---

## Implementation Checklist

### Create File
- [ ] `core/exit_arbitrator.py` (~250 lines)
  - [ ] `ExitPriority` enum
  - [ ] `ExitCandidate` dataclass
  - [ ] `ExitArbitrator` class
  - [ ] `resolve_exit()` method
  - [ ] `set_priority()` method

### Integrate into MetaController
- [ ] Add `from core.exit_arbitrator import ExitArbitrator`
- [ ] Add `self.exit_arbitrator = ExitArbitrator(logger=self.logger)` in `__init__`
- [ ] Create `_collect_exits()` method
- [ ] Modify `execute_trading_cycle()` to use arbitrator
- [ ] Replace if-then-else with `exit_arbitrator.resolve_exit()`

### Testing
- [ ] Test risk exit beats TP/SL
- [ ] Test TP/SL beats signal
- [ ] Test signal beats nothing
- [ ] Test logging of suppressed exits
- [ ] Test priority modification at runtime

### Documentation
- [ ] Update code comments
- [ ] Document exit types in config
- [ ] Add examples in docstrings

---

## Example: Day in the Life

```
⏰ MARKET UPDATE: 14:30 UTC
    BTC/USDT price: $45,230
    Position: 0.50 BTC (entry: $44,000)

📊 ANALYSIS:
    Risk Check: ✅ No starvation, no dust, capital OK
    TP/SL Check: ✅ TP triggered! ($2,000 profit target reached)
    Signal Check: ✅ Agent SELL signal (downtrend detected)

🎖️ ARBITRATION:
    Candidates: [
        ("TP_SL", tp_signal, priority=2, reason="Take-profit $2000"),
        ("SIGNAL", sell_signal, priority=3, reason="Agent downtrend signal"),
    ]
    
    Winner: TP_SL (priority=2 beats priority=3)
    Suppressed: [
        {"type": "SIGNAL", "reason": "Agent downtrend signal"}
    ]
    
    Log: "[ExitArbitration] Symbol=BTC/USDT Winner=TP_SL (priority=2) 
          Suppressed=1 Details: [{'type': 'SIGNAL', ...}]"

✅ EXECUTION:
    Action: SELL 0.50 BTC @ $45,230
    Reason: TP_SL (Take-profit)
    Profit: +$615 (after fees)
```

---

## Key Advantages

### 🎯 Clarity
```
"Why did we exit?"
→ "Because take-profit was triggered and it's priority 2"
→ Clear. Auditable. Explainable.
```

### 🔧 Maintainability
```python
# Want to make rotation exits higher priority?
# Just one line:
arbitrator.set_priority("ROTATION", 1.5)

# No need to rewrite the whole exit logic
```

### 📊 Observability
```
Every exit decision logged with:
- Winner type (RISK/TP_SL/SIGNAL/...)
- Priority value
- Reason for suppressed alternatives
- Timestamp

Perfect for post-trade analysis and risk auditing
```

### 🏛️ Professional
```
This is how institutional systems (banks, funds, 
brokers) handle competing signal priorities.

It's the standard pattern because it works.
```

---

## Real-World Scenarios

### Scenario 1: Capital Emergency
```
Position: ETH/USDT (0.50 ETH @ $2,500)
Capital: $50 (CRITICAL!)

Exits Available:
  1. RISK exit (starvation - forced liquidation)
  2. TP_SL exit (no TP/SL triggered)
  3. SIGNAL exit (agent suggests hold for uptrend)

Arbitration:
  Priority: RISK (1) > TP_SL (2) > SIGNAL (3)
  Winner: RISK
  Action: FORCED SELL immediately
  Suppressed: [TP_SL (not triggered), SIGNAL (capital is critical)]

Result: ✅ Position liquidated, capital preserved
```

### Scenario 2: Normal Trading
```
Position: BTC/USDT (0.10 BTC @ $44,000)
Current: $45,200

Exits Available:
  1. RISK exit (none)
  2. TP_SL exit (TP triggered at $45,000)
  3. SIGNAL exit (agent says sell on weakness)

Arbitration:
  Priority: RISK (none) → TP_SL (2) > SIGNAL (3)
  Winner: TP_SL
  Action: SELL for profit
  Suppressed: [SIGNAL (would have sold anyway, but TP_SL wins)]

Result: ✅ Take profit automatically, even if agent changes mind
```

### Scenario 3: Portfolio Rebalancing
```
Position: SOL/USDT (10 SOL @ $100)
Reason: Rotation engine says SOL is exiting universe

Exits Available:
  1. RISK exit (none)
  2. TP_SL exit (none)
  3. SIGNAL exit (generic signal)
  4. ROTATION exit (universe exit)

Arbitration:
  Priority: ROTATION (4) > SIGNAL (3)
  Winner: ROTATION
  Action: FORCE SELL per rotation engine
  Reason: "rotation_exit" logged

Result: ✅ Portfolio rebalanced, rotation enforced
```

---

## Configuration Example

```yaml
# config.yaml or config.json

exit_arbitration:
  enabled: true
  
  # Priority mapping (lower = higher priority)
  priorities:
    RISK: 1          # Always first
    TP_SL: 2         # Profit protection second
    SIGNAL: 3        # Agent signals third
    ROTATION: 4      # Rotation after signal
    REBALANCE: 5     # Rebalance last
  
  # Logging
  log_all_arbitrations: true
  log_suppressed_exits: true
  log_level: INFO
  
  # Options (future)
  allow_runtime_adjustment: true
  audit_trail_enabled: true
```

---

## Metrics You Can Track

Once arbitration is in place, measure:

```
1. Exit Distribution
   - How many exits of each type?
   - What % are RISK vs TP_SL vs SIGNAL?
   - Trends over time?

2. Suppression Patterns
   - When does SIGNAL lose to TP_SL?
   - When does TP_SL trigger without agent signal?
   - Correlation with profitability?

3. Priority Effectiveness
   - Did changing priorities improve returns?
   - Did it reduce risk events?
   - Trade-off analysis?

4. Decision Quality
   - Are we exiting at good prices?
   - Do risk exits preserve capital?
   - Do profit exits capture gains?
```

---

## Next Steps

1. **Create `exit_arbitrator.py`** (reference: EXIT_ARBITRATOR_BLUEPRINT.md)
2. **Integrate into MetaController** (4-5 locations, minimal changes)
3. **Test thoroughly** (unit tests + integration tests)
4. **Monitor metrics** (track exit distribution and results)
5. **Optimize priorities** (adjust based on results)

---

**Questions?** Check EXIT_ARBITRATOR_BLUEPRINT.md for full implementation details.
