# 🎯 QUICK DECISION CARD - Liquidate vs Reinvest

**When faced with a dust position, this card shows the system's decision logic:**

---

## Quick Decision Matrix

```
IF Position < Min_Notional ($50)
│
├─ IF Capital < Floor ($500)        → LIQUIDATE ❌
│  Reason: Capital recovery priority
│  Action: Force-sell immediately
│  Time: Instant
│
└─ IF Capital >= Floor ($500)       → FLEXIBLE ✅
   │
   ├─ IF BUY signal arrives          → REINVEST ✅
   │  Reason: Optimal efficiency
   │  Action: Consolidation buy
   │  Time: ~30 minutes
   │
   └─ IF No BUY signal               → HOLD ⏸️
      Reason: Zero opportunity cost
      Action: Wait in inventory
      Time: Until signal arrives
```

---

## Decision Checklist

**Step 1:** Position < Min_Notional?
- [ ] YES → Position is DUST (continue to Step 2)
- [ ] NO → Normal management (no action needed)

**Step 2:** Capital < Floor?
- [ ] YES → LIQUIDATE immediately (forced exit)
- [ ] NO → Continue to Step 3

**Step 3:** BUY Signal for this symbol?
- [ ] YES → REINVEST (consolidation buy)
- [ ] NO → HOLD (wait for signal)

---

## Key Thresholds

| Metric | Threshold | Status |
|--------|-----------|--------|
| Min Notional | $50 | DUST if below |
| Capital Floor | $500 | STARVED if below |
| Consolidation Size | $5-10 | Typical healing buy |
| Dust Age | 30 min | Monitor for healing |
| Rate Limit | 2 hours | Min between heals |

---

## Real-World Examples

**Case A: Healthy Portfolio**
- DOGE dust: $0.088
- Free capital: $5,000
- Decision: HOLD (waiting for BUY)
- Action: Nothing (automatic)

**Case B: Capital Starved**
- BTC dust: $5
- Free capital: $100
- Decision: LIQUIDATE NOW
- Action: Force-sell immediately

**Case C: Signal Arrives**
- ETH dust: $2
- Free capital: $3,000
- BUY signal: YES (for ETH)
- Decision: REINVEST
- Action: Consolidation buy

---

## Control Flags

| Flag | Meaning | Status |
|------|---------|--------|
| `is_dust` | Position < min_notional | ✅ Active |
| `management_strategy` | HOLD/TRADE/LIQUIDATE | ✅ Set |
| `_capital_recovery_forced` | Force exit needed | ✅ Set if starved |
| `is_dust_healing_buy` | Consolidation buy | ✅ Set when healing |

---

## Monitoring Signals

**Watch For These Logs:**

```
[SS:Dust]              → Position marked as dust
[DEBUG:CLASSIFY]       → Classification decision made
[Meta:Capital]         → Capital status checked
[Dust:REUSE]           → Dust detected for reinvestment
[Recovery:Liquidate]   → Liquidation triggered
```

---

## Best Practice Decision

```
When in doubt, ask:

1. Is capital healthy?        → YES → REINVEST/WAIT
                              → NO  → LIQUIDATE

2. Is BUY signal present?     → YES → REINVEST NOW
                              → NO  → HOLD

3. How long has dust existed? → < 30min → WAIT
                              → > 30min → Consider healing
```

---

## Timing

| Scenario | Time | Action |
|----------|------|--------|
| Dust detected | Instant | Record in registry |
| Wait for signal | 1-30 min | Monitor for BUY |
| Consolidation buy | ~1-5 min | Execute when ready |
| Healing complete | ~30 min | Position tradeable |
| Capital recovery | Instant | If forced liquidation |

---

## Risk Indicators

🔴 **RED FLAGS:**
- Capital < Floor (need liquidation)
- Dust > 30 minutes (consider healing)
- Multiple dust positions (consolidate)
- Portfolio 95%+ locked (emergency)

🟡 **YELLOW FLAGS:**
- Capital 80-100% of floor (monitor)
- New dust position (track for healing)
- Multiple symbols in dust (track)

🟢 **GREEN:**
- Capital > 150% of floor (flexible)
- Dust < 5% of portfolio (healthy)
- Clear BUY signals (opportunities)

---

## System Summary

The system automatically:
1. **Detects** dust (< min_notional)
2. **Classifies** by capital status
3. **Decides** liquidate/reinvest/hold
4. **Executes** appropriate action
5. **Monitors** result

**All with zero manual intervention.**

---

## When to Manually Override

**Override ONLY if:**
1. Emergency capital needed
2. Position needs immediate exit
3. System detection failed (rare)

**Otherwise:** Let system handle it

---

**For detailed info, see:** `LIQUIDATE_VS_REINVEST_DECISION_LOGIC.md`
