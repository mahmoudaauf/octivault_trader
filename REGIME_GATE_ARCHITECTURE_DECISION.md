# Where Regime Gate Should Live - Architecture Decision

## Your Assertion: CORRECT ✅

> Not in TPSLEngine.
> Not in exit handler.

## The Correct Location

**Regime gate belongs in AGENT LAYER (signal generation), specifically in `_submit_signal()` methods.**

### Evidence from Current Implementation

#### 1. TrendHunter (agents/trend_hunter.py, lines 522-550) - CORRECT ✅

```python
async def _submit_signal(self, symbol: str, action: str, confidence: float, reason: str) -> None:
    action_upper = action.upper().strip()
    
    # ← Regime gate LIVES HERE
    if action_upper == "BUY":
        try:
            sym_u = str(symbol).replace("/", "").upper()
            # Get 1h regime (brain)
            regime_1h = await self.shared_state.get_volatility_regime(sym_u, timeframe="1h")
            
            # Block BUY if 1h regime is bear
            if regime_1h == "bear":
                logger.info(
                    "[%s] BUY filtered for %s — 1h regime is BEAR (hands blocked by brain)",
                    self.name,
                    symbol,
                )
                return  # ← PREVENT SIGNAL FROM BEING EMITTED
```

**Key principle**: Signal never leaves agent → never reaches MetaController → never reaches execution

#### 2. MLForecaster (agents/ml_forecaster.py, lines 2081-2999) - CORRECT ✅

```python
def _institutional_regime_gate(
    self,
    *,
    action: str,
    regime: str,
    confidence: float,
    required_conf: float,
    expected_move_pct: float,
    round_trip_cost_ev_pct: float,
) -> Tuple[bool, float, str]:
    """Regime-aware gating at agent level."""
    if not self._institutional_regime_filter_enabled:
        return True, float(required_conf), "inst_filter_disabled"
    if str(action or "").upper() != "BUY":
        return True, float(required_conf), "inst_filter_not_buy"

    rg = self._normalize_regime(regime)
    conf_floor = float(required_conf)
    
    if rg == "bear" and self._inst_block_bear_buy:
        return False, conf_floor, "regime_block_bear"  # ← Block at source

    # ... regime-specific confidence floors
    return (confidence >= conf_floor), conf_floor, "ok"
```

**Key principle**: Regime gate is applied BEFORE signal generation, within agent logic

---

## Why NOT in TP/SL Engine?

### Problem with TPSLEngine approach:

```python
# ❌ WRONG: Regime gate in TP/SL exit handler
async def check_orders(self):
    for symbol in open_trades:
        # ... calculate TP/SL triggers
        if cp >= float(tp):
            # Check regime here? ❌ WRONG
            regime = self.shared_state.volatility_state.get(symbol)
            if regime == "bear":
                # Skip TP exit? Defer? Convert to SL?
                pass
```

**Problems:**
1. **Too late**: Position already entered (when we wanted to block entry)
2. **Wrong semantics**: TP/SL is risk management (exit), not regime filtering (entry)
3. **Asymmetric logic**: You block entries by regime but exits still happen
4. **Incomplete**: Doesn't prevent the initial entry decision
5. **Confusion**: Mixing entry policy (regime awareness) with exit mechanics (TP/SL)

---

## Why NOT in Exit Handler?

### Exit handlers are for execution mechanics, not regime policy

Exit handler types that should NOT have regime gates:
- TPSLEngine.check_orders() - TP/SL mechanics
- ExecutionManager.close_position() - order execution
- RiskManager.should_sell() - risk triggers

**They operate on positions that already exist.**

Regime gate is an **ENTRY POLICY**, not an **EXIT MECHANIC**.

---

## Correct Architecture: Signal Flow

```
┌─────────────────────────────────────────────────────────┐
│ Agent Layer (Signal Generation)                         │
│                                                          │
│ 1. TrendHunter.run() generates action                   │
│ 2. TrendHunter._submit_signal() applies filters:        │
│    ✓ Confidence gate                                    │
│    ✓ Position state check (BUY: not holding, etc.)      │
│    ✓ Multi-timeframe regime gate (1h brain blocks)      │
│    ✓ IF ALL PASS → emit signal                          │
│    ✓ IF ANY FAIL → return (signal never emitted)        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼ (signal = {action, confidence, reason})
                 
┌─────────────────────────────────────────────────────────┐
│ MetaController Layer (Decision Making)                  │
│                                                          │
│ 1. Receive signal from agent                            │
│ 2. Apply meta-level gates (capital governor, etc.)      │
│ 3. IF PASS → call ExecutionManager                      │
│ 4. IF FAIL → record rejection                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼ (decision = {side, quantity, price})
                 
┌─────────────────────────────────────────────────────────┐
│ ExecutionManager Layer (Order Execution)                │
│                                                          │
│ 1. Place order on exchange                              │
│ 2. Handle fills, partial fills, rejections              │
│ 3. Return execution result                              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼ (fill = {status, qty, price})
                 
┌─────────────────────────────────────────────────────────┐
│ TP/SL Layer (Risk Management)                           │
│                                                          │
│ 1. Monitor open positions                               │
│ 2. Check if TP or SL is hit                             │
│ 3. IF HIT → call ExecutionManager.close_position()      │
│ ✓ Uses ATR, volatility profile for calculations         │
│ ✓ Uses regime for TP/SL multiplier adjustments          │
│ ✗ Does NOT gate exits based on regime                   │
└─────────────────────────────────────────────────────────┘
```

---

## Regime Gate Belongs Here: Agent._submit_signal()

### Key Characteristics of Correct Location

| Attribute | Requirement | Agent._submit_signal() |
|-----------|-------------|----------------------|
| **Timing** | Before signal emission | ✅ Early return prevents emission |
| **Scope** | Entry policy enforcement | ✅ BUY signal filtering |
| **Semantics** | Signal generation stage | ✅ Part of signal creation |
| **Reversibility** | Can prevent signal entirely | ✅ No signal = no execution attempt |
| **Consistency** | Same gate across all agents | ✅ Each agent applies its own |
| **Visibility** | Clear in logs why signal blocked | ✅ "BUY filtered — 1h regime is BEAR" |

---

## What TP/SL Engine SHOULD Do with Regime

✅ **YES - Use regime for TP/SL adjustment**:
```python
# Line ~520 in calculate_tp_sl()
regime = str(profile.get("regime", "sideways")).lower()
if regime in {"trend", "uptrend", "downtrend"}:
    tp_atr_mult *= 1.15  # Wider TP in trends
    sl_atr_mult *= 0.95  # Tighter SL
```

❌ **NO - Don't use regime to gate exits**:
```python
# ✗ WRONG - Don't add this
if regime == "bear":
    defer_tp_exit()  # ← Regime is not exit policy
```

---

## Configuration Should Support Agent-Level Gates

```python
# TP/SL configuration - for TP/SL mechanics only
TPSL_STRATEGY = "hybrid_atr_time"
TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0

# Agent-level regime filtering - for entry policy
TREND_HUNTER_REGIME_FILTER_ENABLED = True
TREND_HUNTER_BLOCK_BEAR_BUY = True
ML_FORECASTER_INSTITUTIONAL_REGIME_FILTER = True
ML_FORECASTER_BLOCK_BEAR_BUY = True
```

**Not mixed together**:
```python
# ✗ WRONG - Don't do this
TPSL_REGIME_FILTER_ENABLED = True
TPSL_FORBID_TP_IN_BEAR = True
```

---

## Summary

### Where Regime Gate SHOULD Live ✅
- **Agent layer** (signal generation)
- **In _submit_signal() methods**
- **Applied BEFORE signal emission**
- **Example**: TrendHunter (lines 522-550), MLForecaster (lines 2081+)

### Where Regime Gate Should NOT Live ❌
- **TP/SL Engine** - wrong layer, wrong semantics
- **Exit handlers** - too late, can't prevent entry
- **MetaController execution** - moves gate away from source agents

### What TP/SL Engine DOES with Regime ✅
- Adjusts TP/SL multipliers
- Calculates volatility profiles
- Informs exit target distances
- **Does NOT gate exits based on regime**

### Architectural Pattern
```
ENTRY POLICY (regime filtering) → AGENT LAYER
    ↓
    ↓ (signal passes or fails)
    ↓
META DECISION (capitalization, frequency) → META_CONTROLLER
    ↓
    ↓ (decision passes or fails)
    ↓
EXECUTION (order placement) → EXECUTION_MANAGER
    ↓
    ↓ (fill or rejection)
    ↓
RISK MANAGEMENT (TP/SL) → TPSL_ENGINE
```

Each layer handles its concern. Regime filtering is **ENTRY POLICY**, not exit policy.
