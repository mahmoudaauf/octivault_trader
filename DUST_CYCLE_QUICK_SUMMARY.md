# ⚡ THE DUST LIQUIDATION CYCLE - QUICK EXPLANATION

## What's Happening

The system creates a **self-reinforcing dust loop**:

```
1. Propose BUY signal
         ↓
2. Order fills as dust ($14 instead of $20)
         ↓
3. System detects dust buildup (80%+ positions)
         ↓
4. After 5 minutes, SELL the dust immediately
         ↓
5. Sell at loss (slippage + fees)
         ↓
6. Capital freed but reduced ($20 → $13)
         ↓
   🔄 Back to step 1 with less capital
   
Result: Capital decays, positions get smaller, more dust
```

---

## The Real Problem

**System liquidates dust positions that are only 5 minutes old.**

Before they have time to:
- Accumulate via healing trades
- Grow from price appreciation
- Become meaningful positions

It just forces them out at a loss.

---

## Why This Is Destructive

| What Happens | Impact |
|---|---|
| Position created as dust | Value = $14 |
| Forced liquidation triggered | Pays slippage/fees |
| Position sold | Realizes = $13.80 |
| Next entry possible | Capital pool = $13.80 + remaining |
| **Result** | Lost $0.20 (1.4% loss) + capital shrunk |

**After 10 cycles:** Capital down 7-8% **with zero profits**

---

## Root Cause

```python
# The problematic code (meta_controller.py):

# Detect dust:
if dust_ratio > 60% and time_sustained > 5_minutes:
    for position in dust_positions:
        emit_sell_signal()  # ← NO AGE CHECK
        # Immediately sells FRESH dust positions

# Problem: Should check:
# - Is position >= 1 hour old? (allow time to accumulate)
# - Has it naturally healed? (value now >= $20?)
# - Has it been stuck forever? (rejection_count > 5?)
```

---

## The Fix (3 Steps)

1. **Don't propose entries if system will liquidate them**
   - During dust phase, block new sub-floor entries
   
2. **Add minimum age guard on dust positions**
   - Dust must be >= 1 hour old before liquidation
   - Exception: If it healed naturally (value grew), liquidate immediately
   - Exception: If permanently stuck (rejections > 5), give up
   
3. **Delay dust liquidation phase activation**
   - Only trigger when 80%+ dust (not 60%)
   - Wait 30 minutes (not 5 minutes)
   - Don't touch positions < 1 hour old

---

## Impact

**Without fix:** Capital decays 7-8% per 10 cycles  
**With fix:** Capital stable or grows naturally  

**Key insight:** Don't liquidate positions that haven't had time to work.

---

See `DUST_LIQUIDATION_CYCLE_ANALYSIS.md` for full technical details.
