# Signal Pipeline Diagnosis - May 5, 2026

## Problem Statement
Bot is running but **generating ZERO tradable signals**. Loop shows:
- `all_signals=0`, `signals_by_sym=0`, `valid_signals_by_symbol=0`
- Portfolio: FLAT, Capital: $77.15 available
- Status: HEALTHY but completely deadlocked

---

## Investigation Results

### ✅ Components Working Correctly

1. **MLForecaster Inference**: OPERATIONAL
   - Running and predicting on symbols: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, etc.
   - **Output predictions**:
     - BNBUSDT: action=BUY, confidence=0.83
     - SOLUSDT: action=BUY, confidence=0.91
     - XRPUSDT: action=BUY, confidence=0.88
   - Models are loaded and making predictions every ~5 seconds

2. **Symbol Screener**: OPERATIONAL
   - Accepting all 10 symbols: `{'AVAXUSDT', 'ADAUSDT', 'LINKUSDT', 'DOGEUSDT', 'PEPEUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'BTCUSDT', 'ETHUSDT'}`
   - No symbols being filtered out by screener

3. **AgentManager Integration**: OPERATIONAL
   - Recognizes MLForecaster in agent registry
   - Calls `generate_signals()` method every tick
   - Passes results to `_normalize_to_intents()` and then to `receive_signal()`

4. **Signal Manager**: OPERATIONAL
   - `receive_signal()` method exists and is wired
   - Cache object initialized: `BoundedCache(max_size=1000, ttl=300s)`
   - Method can accept and cache signals

### ❌ Components NOT Working

**THE BREAK IN THE CHAIN**: Signals are not being emitted by MLForecaster to the collection buffer

```
MLForecaster (predicting BUY 0.83-0.91 confidence)
  ↓
  ✅ Predictions computed correctly
  ↓
  ❌ NOT reaching: self._collected_signals.append(signal)
  ↓
generate_signals() returns empty list
  ↓
AgentManager gets: intents=[]
  ↓
No signals forwarded to receive_signal()
  ↓
signal_cache remains empty
  ↓
MetaController sees: all_signals=0 → NO TRADES
```

---

## Root Cause Analysis

### Where Signals Should Be Emitted
**File**: `agents/ml_forecaster.py`
**Line**: 3114
```python
self._collected_signals.append(signal)
```

### Logs Searched

| Search | Result | Interpretation |
|--------|--------|-----------------|
| `[MLForecaster] SIGNAL:` | **0 matches** | Signals are NEVER reaching line 3114 |
| `MLForecaster: schema=` | ✅ Many matches | Predictions ARE happening |
| `signal_cache._cache=True, len=0` | ✅ Confirmed | Cache exists but empty |
| `Min_notional...filtering out` | 0 matches | Not being filtered by notional check |

### Possible Code Paths Leading to Signal Non-Emission

**Path 1: Early return at line 3036**
```python
if not action or action.upper() not in ("BUY", "SELL"):
    return
```
**Issue**: Action might be empty string or None?
**Likelihood**: Low (other code is executing)

**Path 2: Early return at line 3046 (SELL guard)**
```python
if action.upper() == "SELL" and not self.allow_sell_without_position:
    if not await self._has_position(symbol):
        return
```
**Issue**: But we're seeing BUY predictions, not SELL
**Likelihood**: Very low

**Path 3: Early return at line 3052 (confidence check)**
```python
if float(confidence) < float(required_conf):
    return
```
**Issue**: Confidence calculated as less than required?
**Likelihood**: MEDIUM — need to verify required_conf calculation

**Path 4: Early return at lines 3111-3112 (min notional filter)**
```python
if signal["quote"] < MIN_NOTIONAL_FLOOR * 0.8:
    return
```
**Config**: `EMIT_BUY_QUOTE=9.0`, `MIN_NOTIONAL_FLOOR=5.0 (default)`
**Check**: `9.0 < 4.0`? → FALSE, so signal should NOT be filtered
**Likelihood**: Low (math checks out)

**Path 5: Exception in shared_state.add_strategy_signal() at line 3100**
```python
if hasattr(self.shared_state, "add_strategy_signal"):
    try:
        await self.shared_state.add_strategy_signal(symbol, signal)
    except Exception as e:
        self.logger.warning(f"[{self.name}] Failed to emit strategy signal: {e}")
```
**Issue**: Signal consumed by shared_state bus instead of collected for AgentManager
**Likelihood**: HIGH — This might be the split in the pipeline

---

## Hypothesis: Signal Bus vs. Collection Buffer

MLForecaster might be implementing **two different signal paths**:

### Path A: Strategy Signal Bus
```
MLForecaster → shared_state.add_strategy_signal() → Strategy signal bus
(External consumption, not passed to AgentManager)
```

### Path B: Collection Buffer for AgentManager
```
MLForecaster → self._collected_signals.append() → AgentManager → receive_signal()
```

**Current state**: If `shared_state.add_strategy_signal()` succeeds, maybe the logic is **returning early** or **skipping append**?

### Code Flow Problem
Lines 3100-3114:
```python
# Line 3100: Try emit to signal bus
if hasattr(self.shared_state, "add_strategy_signal"):
    try:
        await self.shared_state.add_strategy_signal(symbol, signal)
    except Exception as e:
        self.logger.warning(...)
# Line 3111: Check min notional
if signal["quote"] < MIN_NOTIONAL_FLOOR * 0.8:
    return  # 🔴 EARLY RETURN!
# Line 3114: Append to collection
self._collected_signals.append(signal)
```

**Wait...** Line 3111 `return` exits **without appending to `_collected_signals`**!

---

## ⚠️ CRITICAL FINDING

### The Bug: Early Return Before Collection

```python
# Line 3100-3104: Emit to signal bus (succeeds silently)
await self.shared_state.add_strategy_signal(symbol, signal)

# Line 3111-3112: MIN NOTIONAL CHECK 🔴
if signal["quote"] < MIN_NOTIONAL_FLOOR * 0.8:
    self.logger.warning(...)
    return  # ← EXITS HERE, never reaches _collected_signals.append()

# Line 3114: NEVER REACHED if min notional fails ❌
self._collected_signals.append(signal)
```

### Verification Needed

Check if min notional filter is actually filtering:
- Log shows zero `"filtering out"` messages
- But we see zero `[MLForecaster] SIGNAL:` logs too
- This suggests either:
  1. The condition IS true (signals being filtered), OR
  2. An exception is occurring before the log at line 3115

---

## Next Steps

1. **Add Debug Logging** to trace exact code path
2. **Print MIN_NOTIONAL_FLOOR value** at runtime
3. **Check signal["quote"] value** at line 3107
4. **Verify return statements** are the blocker
5. **Instrument shared_state.add_strategy_signal()** to confirm it's being called

---

---

## 🚨 CRITICAL DISCOVERY - ROOT CAUSE IDENTIFIED

### The Real Blocker: EV Gate in process_prediction()

**Location**: `agents/ml_forecaster.py` lines 2950-2964

**The Suppression**:
```
[MLForecaster] BUY suppressed for BNBUSDT — expected_move 0.2456% < required 0.6080% (mult=1.60 round_trip=0.3800%)
[MLForecaster] BUY suppressed for SOLUSDT — expected_move 0.2801% < required 0.6080% (mult=1.60 round_trip=0.3800%)
[MLForecaster] BUY suppressed for XRPUSDT — expected_move 0.1993% < required 0.6080% (mult=1.60 round_trip=0.3800%)
```

**What's Happening**:
1. ✅ MLForecaster generates predictions: BNBUSDT (0.83), SOLUSDT (0.91), XRPUSDT (0.88)
2. ✅ ML model confidence is HIGH
3. ❌ **BUT**: The EV gate UPSTREAM of `_collect_signal()` checks: `expected_move >= 1.60 × round_trip_cost`
4. ❌ Current market volatility is TOO LOW (0.25% vs required 0.61%)
5. ❌ **RESULT**: Predictions are REJECTED before they ever reach `_collect_signal()` or the collection buffer

### The Code Path That's Blocking

```python
# Line 2951-2953: Calculate EV threshold
required_move_pct = float(round_trip_cost_ev_pct) * float(ev_mult)  # 0.38% × 1.60 = 0.6080%
ev_positive = float(expected_move_pct or 0.0) >= float(required_move_pct)

# Line 2954-2964: IF NOT ev_positive, RETURN (early exit before _collect_signal)
if not ev_positive:
    self.logger.info("[%s] BUY suppressed for %s — expected_move %.4f%% < required %.4f%% ...")
    return  # ← EXITS HERE, never reaches _collect_signal()
```

### Why This Is NOT a Bug

This is actually **CORRECT BEHAVIOR**. The EV gate is working as designed:
- **Purpose**: Prevent entry into low-volatility trades where round-trip costs eat all profit
- **Logic**: Require `expected_move >= 1.6x round_trip_cost` to have positive EV
- **Current market**: Low-vol majors (BNB, SOL, XRP) all have predicted moves of ~0.25%, below the 0.61% threshold
- **Conclusion**: Bot is CORRECTLY not entering these trades

---

## Summary

**The Signal Pipeline is NOT Broken**. The system is working correctly:

1. ✅ **MLForecaster** generates predictions correctly
2. ✅ **EV gate** correctly filters out low-EV trades (expected_move 0.25% < required 0.61%)
3. ✅ **Result**: Zero signals reach MetaController because none pass the EV gate
4. ✅ **Consequence**: Bot remains FLAT (as intended) because market conditions don't support entry

**This is not a bug—it's a feature. The bot is correctly suppressing entry into unprofitable market conditions.**

### The Real Issue

The bot was designed to trade with `expected_move >= 0.61%`, but the current market (low-vol majors) only offers `expected_move ≈ 0.25-0.28%`. This creates a **DESIGN MISMATCH**, not a code bug.

### Solutions (in order of recommendation)

**Option 1: Activate 4th-Slot Rotation** (Already implemented in this session)
- 4th slot has relaxed EV gate: `expected_move >= 0.46%` (1.2x instead of 1.6x)
- Still won't help now (0.25% < 0.46%), but prepares for when vol increases

**Option 2: Relax Core EV Multiplier** (Temporary workaround)
- Change `ev_mult=1.60` to `ev_mult=1.20` or lower
- Risk: Increases round-trip cost impact on P&L
- Trade-off: More entries but lower profit margins

**Option 3: Expand Symbol Universe**
- Current universe: only 10 symbols (all majors with low vol)
- Add altcoins/microcaps with higher volatility
- Risk: Increased slippage, liquidity concerns

**Option 4: Wait for Market Volatility**
- Current regime: "normal" (low volatility)
- When market enters "high" regime, expected_move will exceed 0.61%
- Passive; no code changes needed
