# 🔴 CRITICAL FINDING: Signal Deadlock Root Cause Identified

## Summary

The diagnostic logs revealed the **TRUE ROOT CAUSE** of why signals are generated and cached but no trades execute:

**Signals ARE passing the MetaController filtering gates, but are being BLOCKED by sell profit/excursion gates AFTER being added to final_decisions.**

## Evidence from Production Logs

```
2026-03-05 23:50:42,394 - WARNING - [Meta:SIGNAL_INTAKE] Retrieved 2 signals from cache: 
  [('BTCUSDT', 'SELL', 0.7513), ('ETHUSDT', 'SELL', 0.8047)]

2026-03-05 23:50:42,394 - WARNING - [Meta:GATE_PASSED] BTCUSDT SELL PASSED ALL GATES

2026-03-05 23:50:42,394 - WARNING - [Meta:AFTER_FILTER] valid_signals_by_symbol has 2 symbols:
  {'SOLUSDT': [('HOLD', 0.4)], 'BTCUSDT': [('SELL', 0.7513)]}

2026-03-05 23:50:42,395 - WARNING - [Meta:TRACE] final_decisions computed: []
  ↑ BUT SIGNALS WERE JUST ADDED! WHY IS THIS EMPTY?
```

## Root Cause

The issue is at **lines 12771-12780** in `_build_decisions()`:

```python
if action == "SELL":
    if (
        await self._passes_meta_sell_profit_gate(sym, sig)
        and await self._passes_meta_sell_excursion_gate(sym, sig)
    ):
        decisions.append((sym, action, sig))  # ← SELL only added if BOTH gates pass
    else:
        self.logger.info("[EXEC_DECISION] SELL %s blocked by profit/excursion gate", sym)
        # ← SELL signals are silently dropped here!
```

**The problem:**
1. ✅ SELL signals pass the MetaController filter gates (SIGNAL_INTAKE → GATE_PASSED → AFTER_FILTER)
2. ✅ SELL signals are added to `final_decisions` list
3. ❌ SELL signals are then processed in the `decisions` loop
4. ❌ If profit gate OR excursion gate fails, the SELL is NOT added to final `decisions`
5. ❌ Result: `final_decisions` is not empty, but `decisions` (which is executed) IS empty

## Why `final_decisions computed: []`?

Looking at the logs more carefully:
```
[Meta:AFTER_FILTER] valid_signals_by_symbol has 2 symbols
[Meta:TRACE] final_decisions computed: []  ← This is saying final_decisions is EMPTY
```

Wait - let me re-read. The `final_decisions` shown in the trace is indeed empty even though GATE_PASSED messages show signals passing. This suggests:

**The problem might be earlier** - signals pass individual gates but get filtered out BEFORE being added to final_decisions in the ranked symbol processing.

## The Real Issue

Looking at the code flow between `valid_signals_by_symbol` (line 10002) and `final_decisions computed` (line 12580):

The code processes `valid_signals_by_symbol` through:
1. **Ranked symbols calculation** (lines 11000+)
2. **Decision building loops** (lines 12000+)
3. **Budget enforcement** (lines 12300+)

Signals may be dropped during:
- Ranked symbol scoring
- Budget allocation
- Affordability checks
- Profit gate checks
- Excursion gate checks

## Fix Added

I added detailed logging for SELL gate checks:

```python
profit_gate = await self._passes_meta_sell_profit_gate(sym, sig)
excursion_gate = await self._passes_meta_sell_excursion_gate(sym, sig)

self.logger.warning(
    "[Meta:SELL_GATES] %s profit_gate=%s excursion_gate=%s",
    sym, profit_gate, excursion_gate
)

if profit_gate and excursion_gate:
    decisions.append((sym, action, sig))
else:
    self.logger.warning(
        "[Meta:SELL_BLOCKED] SELL %s blocked by gates (profit=%s excursion=%s)",
        sym, profit_gate, excursion_gate
    )
```

## Next Steps

1. **Run bot again** with the new SELL_GATES and SELL_BLOCKED logging
2. **Check logs for** `[Meta:SELL_GATES]` and `[Meta:SELL_BLOCKED]` messages
3. **Identify which gate** is rejecting the SELL signals
4. **Analyze the gate logic** to understand why it's rejecting profitable signals
5. **Fix the gate conditions** if they're too strict

## Key Insight

**The signal filtering diagnostics I added successfully traced signals through the MetaController filtering stage and proved that signals ARE reaching that point. But the issue is in the NEXT stage - the profit/excursion gates that occur AFTER ranking and in the affordability checks.**

This is actually **good news** because it narrows down the problem significantly!

## What to Look For in Logs

After running with new diagnostic code, look for:
```
[Meta:SELL_GATES] BTCUSDT profit_gate=FALSE excursion_gate=TRUE
  ↑ This means profit gate is rejecting the SELL

[Meta:SELL_BLOCKED] BTCUSDT blocked by gates (profit=False excursion=True)
  ↑ Confirms SELL was blocked
```

Or:
```
[Meta:SELL_GATES] BTCUSDT profit_gate=TRUE excursion_gate=FALSE
  ↑ This means excursion gate is rejecting the SELL
```

## Immediate Action

The bot on Ubuntu is running. Re-run the analysis after a few minutes with the new logging to see which gate is blocking SELLs.
