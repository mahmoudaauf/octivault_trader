# Regime Filter Enforcement Analysis - TP/SL Engine

## Summary
**YES, there is an unenforced regime filter that should be applied before TP/SL execution.**

The system currently:
- ✅ Reads regime from `shared_state.volatility_state` in TP/SL calculations
- ✅ Uses regime to adjust TP/SL multipliers (adaptive RR, asymmetric TP bias)
- ✅ Has regime filtering implemented in **TrendHunter** (blocks BUY when 1h regime = "BEAR")
- ❌ **DOES NOT enforce a regime-based pre-execution filter in TPSLEngine.check_orders()**

---

## Current Regime Filtering (Implemented in TrendHunter)

### Location: `agents/trend_hunter.py` lines 522-550

```python
# Block BUY if 1h regime is bear
if regime_1h == "bear":
    logger.info(
        "[%s] BUY filtered for %s — 1h regime is BEAR (hands blocked by brain)",
        self.name,
        symbol,
    )
    return
```

**What it does:**
- Only applies to **BUY** signals (entry decisions)
- Uses **1h timeframe** as "brain" (regime decision)
- Uses **5m timeframe** as "hands" (execution)
- Blocks entry when 1h regime is "bear"
- Allows entry when regime is "bull" or neutral

---

## Missing: TP/SL Regime Enforcement

### Location: `core/tp_sl_engine.py` - `check_orders()` method (lines ~1420-1840)

### The Problem

The TP/SL engine **currently has NO regime-based pre-execution filter**. When a TP or SL is triggered:

```python
# Line 1768-1780 (Long positions)
if cp >= float(tp):
    reason = "TP Hit"
    if (
        self._passes_tp_distance_gate(symbol, float(entry_price), float(tp))
        and
        self._passes_net_exit_gate(pnl_pct)
        and self._passes_profit_gate(pnl_pct, reason)
        and await self._passes_excursion_gate(symbol, entry_price, cp, atr, reason)
    ):
        to_close.append((symbol, reason))
```

**Missing check:** No validation of current regime before executing the exit.

### Available Regime Data

The TP/SL engine **already retrieves regime** for calculations:

```python
# Line 453-463 (_build_volatility_profile)
ext_regime = str((getattr(self.shared_state, "volatility_state", {}) or {}).get(symbol, "") or "").lower()
if ext_regime in {"trend", "uptrend", "downtrend", "high_vol", "high", "sideways", "chop"}:
    regime = ext_regime
else:
    # Fallback to inferred regime based on ATR
    if atr_pct >= high:
        regime = "high_vol"
    elif atr_pct <= low:
        regime = "sideways"
    else:
        regime = "trend"
```

This regime is **calculated but only used for TP/SL adjustments, not for exit filtering**.

---

## Regime Classifications in the System

### From `volatility_state` (external source)
Valid regime values:
- `"trend"`, `"uptrend"`, `"downtrend"` (trending regimes)
- `"high_vol"`, `"high"` (high volatility regimes)
- `"sideways"`, `"chop"` (range/chop regimes)

### From TrendHunter filter
Regime values used for gating:
- `"bear"` (blocks BUY signals)
- `"bull"` (allows BUY signals)
- `"unknown"` or other (allows with caution)

---

## Architectural Intent

The pattern suggests a **two-layer regime check**:

1. **Brain** (1h timeframe): Strategic regime decision
   - TrendHunter uses this to block entries in bear markets
   - Should affect NEW positions

2. **Hands** (5m/current timeframe): Tactical regime context
   - TP/SL uses this to adjust exit targets dynamically
   - Should also affect EXIT decisions for consistency

---

## Recommendation: What Should Be Enforced

### Option A: Mirror TrendHunter Logic (Conservative)
Before executing any TP/SL exit, check:
```python
regime_1h = await self.shared_state.get_volatility_regime(symbol, timeframe="1h")
if regime_1h.get("regime") == "bear":
    # Defer TP exits, but allow SL exits
    # OR: Require higher profit threshold to exit in bear
    pass
```

### Option B: Use Tactical Regime (Current Volatility)
Check the `volatility_state` (current regime) before exiting:
```python
current_regime = (getattr(self.shared_state, "volatility_state", {}) or {}).get(symbol)
if current_regime in ["high_vol", "high"]:
    # More permissive with exits (positions unlocking quickly)
elif current_regime in ["sideways", "chop"]:
    # Tighter gating, higher profit thresholds
elif current_regime == "bear":
    # Allow SL exits but defer TP
```

### Option C: Hybrid (Recommended)
- **TP exits**: Respect 1h regime, defer in "bear" markets
- **SL exits**: Always execute (risk management override)
- **Time-based exits**: Respect current regime (tactical)

---

## Current Gates (Already Enforced)

The TP/SL engine **DOES enforce** these gates before execution:

1. ✅ `_passes_profit_gate()` - Min profit threshold (fees-aware)
2. ✅ `_passes_net_exit_gate()` - Risk/reward validation
3. ✅ `_passes_tp_distance_gate()` - TP too close check
4. ✅ `_passes_excursion_gate()` - Min price movement validation
5. ✅ `_passes_tp_floor_hit()` - TP floor clamping logic
6. ✅ `_pre_activation_guard()` - Entry guards
7. ✅ Debounce logic - Prevents repeated closes (per symbol)

**Missing:**
❌ Regime-based filtering before TP execution

---

## Code Locations to Review

| File | Line | What |
|------|------|------|
| `core/tp_sl_engine.py` | 453-473 | Regime detection (_build_volatility_profile) |
| `core/tp_sl_engine.py` | 1768-1785 | TP/SL trigger check (long) |
| `core/tp_sl_engine.py` | 1820-1835 | TP/SL trigger check (short) |
| `agents/trend_hunter.py` | 522-550 | Implemented BUY regime filter |
| `core/shared_state.py` | TBD | get_volatility_regime() method |
| `core/meta_controller.py` | 988+ | Regime check functions |

---

## Implementation Impact

### If Regime Filter is Added to TP/SL:
- **Risk**: Positions may be held longer in unfavorable regimes
- **Benefit**: Alignment with entry filtering (no entries in bear, so exits should respect that)
- **Config**: New parameters for regime-based gating behavior

### Configuration Flags Needed:
```python
TPSL_REGIME_FILTER_ENABLED = True
TPSL_FORBID_TP_IN_BEAR = True        # Defer TP exits in bear regime
TPSL_ALLOW_SL_IN_BEAR = True         # Always exit on SL (risk override)
TPSL_REGIME_TIMEFRAME = "1h"         # Use 1h regime (like TrendHunter)
TPSL_HIGH_VOL_EXIT_BONUS = 0.05      # +5% to exit in high vol regime
TPSL_LOW_VOL_EXIT_PENALTY = -0.10    # -10% (hold longer) in low vol
```

---

## Conclusion

**The regime filter exists in the system but is NOT enforced at TP/SL execution time.**

This creates a potential asymmetry:
- Entries are gated by regime (TrendHunter)
- Exits are NOT gated by regime (TPSLEngine)

**Recommended action:** Implement a regime-aware pre-execution check in `TPSLEngine.check_orders()` to mirror the entry-side regime filtering from TrendHunter.
