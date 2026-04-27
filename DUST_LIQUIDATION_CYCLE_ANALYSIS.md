# 🔄 THE DUST CREATION CYCLE: Propose → Execute → SELL → Dust Loop

**Date:** April 27, 2026  
**Status:** SECOND PATHWAY IDENTIFIED  
**Severity:** CRITICAL - Creates endless dust recycling

---

## 🎯 EXECUTIVE SUMMARY

You've identified a **second dust creation pathway** that's even worse than the first:

1. **System proposes new BUY** for symbol (e.g., BTCUSDT)
2. **Order gets filled** below significant floor → becomes dust
3. **System decides to liquidate** the dust position
4. **SELL gets executed** at a loss
5. **Capital released** but damaged
6. **Back to step 1** → endless cycle

This creates a **self-reinforcing loop** where:
- New positions → Dust → Forced liquidation → Loss → Repeat

---

## 🔴 THE DUST LIQUIDATION CYCLE

### The Flow

```
┌─────────────────────────────────────────────────────────────┐
│ CYCLE START: Portfolio needs capital or has dust buildup    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. PROPOSE NEW SYMBOL                                       │
│    ├─ MetaController sees BUY signal for BTCUSDT            │
│    ├─ Checks gates: confidence, capital, position count     │
│    └─ ✅ Decision: "Execute BUY"                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. EXECUTE BUY FILL                                         │
│    ├─ ExecutionManager places order: 0.0003 BTC @ $44,000  │
│    ├─ Fill received: 0.0003 × $44,000 = $13.20 USDT        │
│    ├─ Position registered                                   │
│    └─ Value < $20 floor → ❌ DUST_LOCKED                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. SYSTEM DETECTS DUST BUILDUP                              │
│    ├─ Check: dust_ratio > 60%                               │
│    ├─ Check: sustained for 5+ minutes                       │
│    ├─ Decision: "Must liquidate dust"                       │
│    └─ ✅ Triggers aggressive dust liquidation               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. GENERATE SELL SIGNAL FOR DUST                            │
│    ├─ Symbol: BTCUSDT (the position we just created!)       │
│    ├─ Action: SELL 0.0003 BTC                              │
│    ├─ Confidence: 0.99 (forced liquidation)                │
│    ├─ Reason: "phase2_dust_liquidation"                     │
│    └─ Agent: MetaDustLiquidator                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. EXECUTE SELL (IMMEDIATELY)                               │
│    ├─ ExecutionManager places SELL order                    │
│    ├─ Sell: 0.0003 BTC @ $43,900 (worst case)             │
│    ├─ Proceeds: $13.17 USDT                                │
│    ├─ Loss: $0.03 USDT (slippage + fees)                   │
│    └─ Position closed, but capital is now "freed"           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. CAPITAL RELEASED (BUT DAMAGED)                           │
│    ├─ Original: $20 USDT allocated                          │
│    ├─ After buy loss: $13.20 USDT position                 │
│    ├─ After sell loss: $13.17 USDT freed                   │
│    ├─ Total loss: $0.03 + fees = real damage               │
│    └─ Capital freed: $13.17 (available for new trades)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
                      🔄 BACK TO STEP 1
               New BUY signal arrives, cycle repeats
```

---

## 📍 CODE LOCATIONS: The Dust Liquidation Cycle

### Location 1: Detect Dust Buildup & Trigger Liquidation

**File:** `core/meta_controller.py` lines 16900-16950

```python
async def _build_decisions(self, accepted_symbols_set: set):
    # PHASE 2 FIX: Check if dust ratio > 60% and sustained
    dust_ratio = dust_pos / total  # e.g., 0.80 (80% dust!)
    phase2_age = time.time() - phase2_trigger_time
    
    if dust_ratio > 0.60 and phase2_age >= 300.0:  # 5+ minutes
        # TRIGGER: Generate aggressive SELL signals for all dust
        for sym, qty, value_usdt in dust_to_liquidate:
            dust_sell_sig = {
                "symbol": sym,
                "action": "SELL",  # ← SELL the dust positions
                "confidence": 0.99,  # Force it
                "agent": "MetaDustLiquidator",
                "reason": "phase2_dust_liquidation",
                "_force_dust_liquidation": True,
            }
```

### Location 2: The Problem - Propose & Liquidate Same Position

**File:** `core/meta_controller.py` lines 16550-16600

The issue is the **timing**:

```python
# ╔════════════════════════════════════════════════════════════════╗
# ║ PROBLEM: Can happen in same cycle or adjacent cycles:        ║
# ╚════════════════════════════════════════════════════════════════╝

# Cycle N:
# ├─ 1. New BUY signal for BTCUSDT arrives
# ├─ 2. MetaController approves it (gating passes)
# ├─ 3. ExecutionManager fills it as dust
# │
# └─ Decision list includes: BUY BTCUSDT

# Cycle N+1 (5 seconds later):
# ├─ System sees dust ratio still > 60%
# ├─ Triggers dust liquidation phase
# ├─ Finds BTCUSDT in dust positions (it's only 5 seconds old!)
# ├─ Immediately generates SELL signal
# │
# └─ Decision list includes: SELL BTCUSDT ← SAME POSITION
```

---

## 🎯 WHY THIS HAPPENS

### Root Cause #1: No Minimum Age Guard on Dust

**Problem:** Dust positions get liquidated immediately, even if:
- They were created 5 seconds ago
- They haven't had time to accumulate
- They're newly proposed symbols

**Code Evidence:**

```python
# From meta_controller.py line 16910-16925
for sym, qty, value_usdt, pos_age_sec in dust_to_liquidate:
    # ⚠️ NO CHECK on pos_age_sec minimum age
    # Positions can be liquidated at ANY age
    
    if should_execute or escape_sell:
        executable_dust.append((sym, qty, value_usdt, pos_age_sec))
        
    # Generate SELL immediately for executable dust
    dust_sell_sig = {
        "symbol": sym,
        "action": "SELL",  # ← Even if created 2 minutes ago!
        "confidence": 0.99,
    }
```

### Root Cause #2: Dust Phase Activation Too Aggressive

**Config:**
```python
# From meta_controller.py - typical settings
phase2_trigger_threshold = 0.60  # Activate if dust > 60%
phase2_grace_period = 300.0  # 5 minutes

# Problem: A single sub-floor entry can trigger this
# Example: 2 positions total, 1 is dust
# Dust ratio = 1/2 = 50% (just below threshold)
# Add another dust = 2/3 = 66% (✅ THRESHOLD MET)
# Boom: Phase 2 liquidation triggered
```

### Root Cause #3: SELL Gate Bypass on Dust

**Code Evidence:**

```python
# From meta_controller.py line 16963
should_execute = await self.should_execute_sell(sym, emergency_liquidation=True)
                                                      ↑
                                         Emergency bypass enabled!

# This means SELL gates are BYPASSED for dust liquidation
# No profitability check, no timing check
# Just: "Dust detected → SELL immediately"
```

---

## 🔄 THE VICIOUS CYCLE

```
                    ┌─────────────┐
                    │   BUY DUST  │ (value < $20)
                    └──────┬──────┘
                           │
                           ↓ (dust detected after 5 min)
                    ┌─────────────┐
                    │ SELL DUST   │ (forced liquidation)
                    │   @ loss    │
                    └──────┬──────┘
                           │
                           ↓ (capital freed but damaged)
          ┌─────────────────┴──────────────────┐
          │ Available capital reduced          │
          │ $20 → $13 (after losses)          │
          └─────────────────┬──────────────────┘
                           │
                           ↓ (next signal arrives)
                    ┌─────────────┐
                    │ Propose BUY │ (new symbol)
                    │ for $13 cap │ ← Even smaller dust!
                    └──────┬──────┘
                           │
                    🔄 BACK TO START
                    (Losses accumulate)
```

---

## 💥 THE DAMAGE MECHANISM

### How Capital Gets Destroyed

```
Initial Capital: $100 USDT
├─ Position 1: BUY @ $20 → fills as $14 dust
├─ Position 2: BUY @ $20 → fills as $16 dust
└─ Portfolio: $100 - $14 - $16 = $70 left

Wait 5 minutes...

Dust ratio = 2/2 = 100% (TRIGGERS PHASE 2)
├─ SELL Position 1 @ loss: $14 → $13.80 (fees)
├─ SELL Position 2 @ loss: $16 → $15.50 (fees)
└─ Capital freed but damaged: $13.80 + $15.50 = $29.30

Available for next trade: $70 + $29.30 = $99.30 ❌
(Should be $100, but lost $0.70 to slippage/fees)
```

### Cumulative Effect

```
Cycle 1: $100 → $99.30 (lost $0.70 = 0.7% loss)
Cycle 2: $99.30 → $98.56 (lost $0.74 = 0.75% loss)
Cycle 3: $98.56 → $97.78 (lost $0.78 = 0.79% loss)
...
After 10 cycles: Capital down 7-8% with NO profits!
```

---

## 📋 THE PROBLEM CHAIN

| Problem | Where | Impact |
|---------|-------|--------|
| **Entries create dust** | `record_fill()` | Positions start as dust |
| **Dust accumulates** | System detection | Triggers after 5 min |
| **Dust immediately liquidated** | `_build_decisions()` | No maturation time |
| **Liquidation at loss** | `ExecutionManager` | Slippage + fees |
| **Capital shrinks** | Portfolio | Each cycle loses money |
| **Smaller entries happen** | Next cycle | Smaller dust created |

---

## 🔧 HOW TO FIX THIS

### Fix 1: Add Minimum Age Guard on Dust Positions (CRITICAL)

**File:** `core/meta_controller.py` - lines 16920-16930

```python
# Add before liquidation decision:
DUST_MIN_AGE_BEFORE_LIQUIDATION = 3600  # 1 hour minimum

for sym, qty, value_usdt, pos_age_sec in dust_to_liquidate:
    # NEW: Don't liquidate dust that's too fresh
    if pos_age_sec is not None and pos_age_sec < DUST_MIN_AGE_BEFORE_LIQUIDATION:
        self.logger.info(
            "[Dust:AgeGuard] Skipping %s: age %.0f sec < min %.0f sec. "
            "Give it time to accumulate.",
            sym, pos_age_sec, DUST_MIN_AGE_BEFORE_LIQUIDATION
        )
        continue  # Don't liquidate yet
    
    # Only then generate SELL
    executable_dust.append((sym, qty, value_usdt, pos_age_sec))
```

### Fix 2: Don't Propose Entry If System Will Liquidate It

**File:** `core/meta_controller.py` - lines 16250-16280

```python
# Before approving new BUY:
if await self._is_in_dust_liquidation_phase():
    current_dust_ratio = await self._get_dust_ratio()
    if current_dust_ratio > 0.50:
        # System is actively liquidating dust
        # Don't propose new entries that will become dust
        self.logger.warning(
            "[Meta:GateCheck] Dust liquidation phase active (ratio=%.1f%%). "
            "Rejecting new entries < $20 to prevent immediate liquidation.",
            current_dust_ratio * 100
        )
        # Filter out sub-floor entries
        buy_signals = [s for s in buy_signals 
                      if s.get("planned_quote", 0) >= 20.0]
```

### Fix 3: Accumulation Must Come Before Liquidation

**File:** `core/meta_controller.py` - new method

```python
async def _can_liquidate_dust(self, symbol: str, position_age_sec: float) -> bool:
    """
    Guard: Dust can only be liquidated if:
    1. Age >= 1 hour (time to accumulate/heal)
    2. OR value >= 2x min_notional (naturally healed)
    3. OR rejection_count > 5 (stuck permanently)
    """
    # Time-based: has it aged enough?
    min_age = 3600  # 1 hour
    if position_age_sec and position_age_sec < min_age:
        return False
    
    # Healing-based: did it grow enough?
    sym = self._normalize_symbol(symbol)
    pos = self.shared_state.get_position_qty(sym)
    value = (pos or 0) * await self.shared_state.safe_price(sym)
    min_notional = await self._get_min_notional(sym)
    
    if value >= min_notional * 2.0:
        return True  # It grew out of dust
    
    # Rejection-based: is it stuck?
    rejection_count = self.shared_state.get_rejection_count(sym, "SELL")
    if rejection_count > 5:
        return True  # Give up, liquidate it
    
    return False
```

### Fix 4: Delay Dust Liquidation Phase Activation

**Config Changes:**

```python
# OLD (too aggressive):
phase2_dust_threshold = 0.60  # 60% dust
phase2_grace_period = 300.0   # 5 minutes

# NEW (more lenient):
phase2_dust_threshold = 0.80  # Only if 80%+ dust
phase2_grace_period = 1800.0  # 30 minutes
phase2_min_dust_age = 3600.0  # Don't liquidate anything < 1 hour old
```

---

## 📊 EXPECTED IMPACT

### Before Fix
```
Entry: $100 USDT
├─ Dust entries: $50
├─ After 5min, liquidate dust: $48 (lost $2)
├─ Next entry: $20 from remaining $50
└─ After cycle: $98 (-2% = damage)

10 cycles: $82 total (18% damage)
```

### After Fix
```
Entry: $100 USDT
├─ No sub-floor entries (blocked)
├─ OR mature dust entries only (age >= 1 hour)
├─ Only liquidate if healed or truly stuck
└─ After cycle: $99.95 (-0.05% = minimal)

10 cycles: $99.50 total (minimal damage)
```

---

## 🚨 SUMMARY

The system creates dust through TWO pathways:

1. **Direct Entry Dust** (first analysis)
   - Entry below significant floor → classified as dust immediately
   - Fix: Validate entry value pre-execution

2. **Liquidation Cycle Dust** (this analysis)
   - Dust detected → immediately force liquidated
   - Creates losses → capital shrinks → smaller entries → more dust
   - Fix: Add minimum age guard, don't liquidate fresh positions

**Both must be fixed for clean trading.**
