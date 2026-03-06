# ⚡ CAPITAL FLOOR QUICK REFERENCE

## One-Liner
**Capital floor recalculates EVERY CYCLE using:** `max(8, NAV * 0.12, trade_size * 0.5)`

---

## Where It Happens

| Location | Method | When |
|----------|--------|------|
| MetaController | `_check_capital_floor_central()` | Cycle start (line 8701) |
| RiskManager | `validate_order()` | Every BUY order (line 624-666) |
| SharedState | `calculate_capital_floor()` | Called by both above |

---

## The Formula Breakdown

```
capital_floor = max(
    8.0,                    # Absolute minimum
    NAV * 0.12,            # 12% of current NAV
    trade_size * 0.5       # 50% of typical trade size
)
```

**Why three components?**
- **$8:** System maintenance buffer (absolute minimum)
- **NAV*0.12:** Capital preservation—scales with account size
- **trade_size*0.5:** Trade viability—ensures can execute next trade

---

## Cycle Recalculation

### Before Each Cycle (MetaController)
```python
nav = await shared_state.get_nav_quote()  # Fresh NAV!
trade_size = config.TRADE_AMOUNT_USDT      # Fresh trade size!
capital_floor = shared_state.calculate_capital_floor(nav, trade_size)
capital_ok = free_usdt >= capital_floor
```

### Before Each BUY Order (RiskManager)
```python
nav = await shared_state.get_nav_quote()  # Fresh NAV!
trade_size = config.TRADE_AMOUNT_USDT      # Fresh trade size!
capital_floor = shared_state.calculate_capital_floor(nav, trade_size)
remaining = free_usdt - order_amount
if remaining < capital_floor:
    REJECT("capital_floor_breach")
```

---

## Decision Logic

```
CYCLE START?
├─ Calculate floor = max(8, NAV*0.12, trade_size*0.5)
├─ IF free_usdt >= floor → PASS
└─ ELSE → BLOCK all BUYs

BUY ORDER?
├─ Calculate floor = max(8, NAV*0.12, trade_size*0.5)
├─ remaining = free_usdt - order_amount
├─ IF remaining >= floor → APPROVE
└─ ELSE → REJECT
```

---

## Examples (30-Min Summary)

### Growing Account
```
NAV $100  → floor = $15  → free $50  → ✓ PASS
NAV $500  → floor = $60  → free $150 → ✓ PASS
NAV $1K   → floor = $120 → free $300 → ✓ PASS
```

### Large Portfolio
```
NAV $10,000 → floor = $1,200 (NAV-based dominates)
free $5,000 → ✓ PASS
```

### Drawdown
```
NAV $10,000 → floor = $1,200
NAV $7,000  → floor = $840  (floor reduced 30% with NAV)
free $1,200 → ✓ PASS
```

### Trade Rejection
```
NAV $5,000, floor = $600
free_usdt = $2,000
Order for $1,500
remaining = 2000 - 1500 = $500
Check: $500 >= $600? NO → ✗ REJECT
```

---

## What Changed from Previous

| Aspect | Before | After |
|--------|--------|-------|
| Recalculation | At startup only | Every cycle + every trade |
| Formula | 20% of NAV | `max(8, NAV*0.12, trade_size*0.5)` |
| MetaController | Basic check | New `_check_capital_floor_central()` |
| RiskManager | No floor check | New capital floor validation |
| Logging | Minimal | Complete audit trail |

---

## Testing Results

```
✅ test_calculate_capital_floor              PASSED
✅ test_capital_floor_recalculation_on_nav_change PASSED
✅ test_capital_floor_vs_free_usdt           PASSED
✅ verify_capital_floor.py (all 5 cycles)   PASSED
```

---

## Debug Output

**Everything is logged!** Search logs for:
- `CAPITAL_FLOOR_CHECK` — Cycle-level decisions
- `capital_floor_breach` — Order rejections
- `capital_floor` — All floor calculations

---

## Configuration

**No new config needed!** Uses existing:
- `TRADE_AMOUNT_USDT` — For trade_size component
- `QUOTE_ASSET` — For balance checks (default: USDT)

---

## Files Modified

1. `core/meta_controller.py` — Lines 7586-7677
2. `core/risk_manager.py` — Lines 624-666
3. `tests/test_shared_state.py` — Added 3 new tests
4. (Created verification script) — `verify_capital_floor.py`

---

## Quick Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Floor too high | NAV increased or trade_size large | Check config TRADE_AMOUNT_USDT |
| Trades blocked | free_usdt < floor | Execute some SELLs first |
| NAV not updating | get_nav_quote() failing | Check NAV calculation in SharedState |
| Capital floor not working | RiskManager/MetaController not calling method | Verify code changes applied |

---

## TL;DR

1. **Capital floor formula:** `max(8, NAV*0.12, trade_size*0.5)`
2. **Recalculated:** Every cycle + every trade (LIVE, not static!)
3. **Enforced:** MetaController (cycle start) + RiskManager (order validation)
4. **Adapts:** Grows with NAV, shrinks with drawdowns
5. **Tested:** 4 tests passing + verification script ✅

**Result:** Optimal capital preservation at every step! 🛡️
