# 🔧 Forced Escalation Removal — Adaptive Risk Restoration

**Status:** ✅ COMPLETE  
**Date:** 2026-02-25  
**Context:** Micro capital bootstrap (BASE_CAPITAL=400 USDT)

---

## 🎯 Problem Statement

The system was using **forced escalation** to guarantee trades during bootstrap:

```python
# BEFORE (BROKEN)
s["_planned_quote"] = 30.0               # Hard-coded, no adaptation
s["_force_min_notional"] = True          # Force exchange minimum satisfaction
s["_bootstrap"] = True                   # Mark for override logic
```

### Why This Was Wrong

1. **Mathematical Bottleneck**
   - Micro capital (400 USDT) + Institutional quote (30 USDT) + Strict EV filter (2.0 mult)
   - Result: Zero trades = No learning = No data = System stuck

2. **Violates Adaptive Principle**
   - ScalingManager computes `ADAPTIVE_MIN_TRADE_QUOTE` (24 USDT)
   - But forced escalation override it anyway
   - Defeats the entire adaptive sizing engine

3. **Bootstrap Paradox**
   - Forced escalation was meant to unlock trades
   - But it violated risk constraints
   - Trades that violate constraints ≠ valid learning signal

---

## ✅ Solution: Adaptive Risk Only

**Core Philosophy:** Let the system's native adaptive mechanisms work.

### Changes Made

#### 1. **meta_controller.py** — Remove Forced Escalation

**Location:** `_planned_quote_for()` method (line ~11830)

```python
# AFTER (CORRECT)
async def _planned_quote_for(self, symbol: str, sig: Dict[str, Any], ...) -> float:
    """Compute planned quote using ScalingManager."""
    planned_quote = await self.scaling_manager.calculate_planned_quote(
        symbol, sig, budget_override=budget_override
    )
    
    # ✅ Use adaptive quote; ExecutionManager enforces exchange min only
    if isinstance(sig, dict):
        is_bootstrap_buy = self._is_bootstrap_buy_context(sig, side="BUY")
        if is_bootstrap_buy:
            adaptive_min = float(
                self.shared_state.dynamic_config.get("ADAPTIVE_MIN_TRADE_QUOTE", 0.0) or 0.0
            )
            # ❌ REMOVED: bootstrap_floor = await self._resolve_entry_quote_floor(...)
            # ❌ REMOVED: s["_force_min_notional"] = True
            
            if self._is_bootstrap_mode():
                sig["_bootstrap"] = True
            sig["_planned_quote"] = float(planned_quote)
            
            self.logger.info(
                "[Meta:ADAPTIVE_QUOTE] %s planned=%.2f adaptive_min=%.2f "
                "(ExecutionManager enforces exchange min)",
                symbol, float(planned_quote), adaptive_min,
            )
    
    return float(planned_quote or 0.0)
```

**Effect:** Quote comes from adaptive sizing, not forced escalation.

---

#### 2. **meta_controller.py** — FLAT_PORTFOLIO Logic

**Location:** Signal processing for flat portfolio bootstrap (line ~8270)

```python
# BEFORE (ESCALATION)
s["_force_min_notional"] = True
s["_planned_quote"] = planned_quote
s["reason"] = f"FLAT_FORCED_BOOTSTRAP:{conf:.2f}:escalated"

# AFTER (ADAPTIVE)
# ❌ REMOVED: s["_force_min_notional"] = True
s["_planned_quote"] = planned_quote
s["reason"] = f"FLAT_BOOTSTRAP_ADAPTIVE:{conf:.2f}"
```

**Effect:** Bootstrap uses adaptive quote, not forced 30 USDT floor.

---

#### 3. **meta_controller.py** — Emergency Escape Hatch

**Location:** Bootstrap override escape hatch (line ~8298)

```python
# BEFORE (ESCALATION)
planned_quote = await self._resolve_entry_quote_floor(
    sym,
    proposed_quote=emergency_base_quote,
)
s["_force_min_notional"] = True
s["reason"] = f"FLAT_OVERRIDE_EMERGENCY_BOOTSTRAP:{emergency_conf:.2f}"

# AFTER (ADAPTIVE)
# Use adaptive quote only; ExecutionManager enforces exchange min
planned_quote = emergency_base_quote
# ❌ REMOVED: s["_force_min_notional"] = True
s["reason"] = f"FLAT_OVERRIDE_ADAPTIVE_BOOTSTRAP:{emergency_conf:.2f}"
```

**Effect:** Emergency bootstrap respects adaptive sizing.

---

#### 4. **.env** — Micro Capital Profile

**Changed quote defaults:**

```bash
# BEFORE (INSTITUTIONAL ONLY)
DEFAULT_PLANNED_QUOTE=30
MIN_TRADE_QUOTE=20

# AFTER (ADAPTIVE FOR MICRO)
DEFAULT_PLANNED_QUOTE=24          # ← Adaptive default (not escalated)
MIN_TRADE_QUOTE=12               # ← Lower floor for micro capital
TIER_B_MAX_QUOTE=50              # ← Ceiling for bootstrap trades
```

**Rationale:**
- 24 USDT: Balanced sweet spot
  - Exceeds typical exchange minimums (5-10 USDT)
  - Low friction for 400 USDT account
  - Sufficient margin after fees/rounding
- 12 USDT: Lower floor for micro sizing
  - Allows fine-grained position building
  - Respects budget constraints
  - Still exchange-compliant

---

#### 5. **.env** — EV Multiplier Adjustment

**New setting for micro capital:**

```bash
# EXPECTED VALUE GATING (ALPHA DISCIPLINE)
EV_MULTIPLIER=1.4                    # Micro capital profile (unlocks alpha during bootstrap)
```

**Why 1.4 instead of 2.0 or 1.65?**

| Capital | EV_MULT | Profile | Purpose |
|---------|---------|---------|---------|
| < 500   | 1.4     | **Learning** | Unlock alpha, build data, learn patterns |
| 500-1000 | 1.65    | **Growth** | Balanced discipline + frequency |
| > 1000  | 2.0     | **Institutional** | Strict discipline, proven edge |

**At 1.4:** `required_move = cost × 1.4`
- Cost = 0.025% (fees) → Requires 0.035% expected move
- Much easier to satisfy during learning phase
- Still maintains economic discipline

---

## 📊 Expected Behavior After Fix

### Before (Broken State)
```
Boot phase:
  BUY signal → forced 30 USDT → EV gate (2.0×) = high bar → REJECTED
  Result: decisions_count = 0 ❌

Why: Institutional constraints + micro capital = death spiral
```

### After (Adaptive State)
```
Boot phase:
  BUY signal → adaptive 24 USDT → EV gate (1.4×) = reachable bar → ACCEPTED ✅
  Execution: ExecutionManager enforces Binance minimum (5-10 USDT) ✅
  Trade executes → Data flows → Learning accelerates
  Result: decisions_count > 0, learning begins 🚀

Growth phase (capital > 1000):
  EV_MULTIPLIER automatically increases to 1.65+ via ADAPTIVE_CAPITAL_ENGINE
  Sizing scales up, discipline increases with capital
```

---

## 🔄 Control Flow Diagram

```
MetaController._process_flat_portfolio()
    ↓
    Get best BUY signal
    ↓
    [BRANCH 1: Executable at base quote]
    ✓ proceed with native adaptive sizing
    ↓
    [BRANCH 2: Not executable, gap exists]
    ✓ Call ScalingManager → calculate adaptive quote
    ✓ Pass to ExecutionManager
    ✓ ExecutionManager enforces exchange minimum ONLY
    ✓ No forced escalation override
    ↓
    [OUTCOME]
    ✅ Micro capital: 12-24 USDT trades execute
    ✅ Capital 500+: Scales to 30+ USDT
    ✅ All subject to EV gate (adaptive multiplier)
    ✅ Learning phase unlocked
```

---

## ⚙️ Config Summary

### Micro Capital Profile (BASE_CAPITAL < 500)

```bash
# BUY SIZING
DEFAULT_PLANNED_QUOTE=24            # Adaptive, not forced
MIN_TRADE_QUOTE=12                  # Fine-grained sizing
TIER_B_MAX_QUOTE=50                 # Bootstrap ceiling

# EXPECTED VALUE GATING
EV_MULTIPLIER=1.4                   # Learning multiplier (unlocked)

# ADAPTIVE ENGINE
ADAPTIVE_MIN_TRADE_QUOTE=24.0        # Dynamic floor
ADAPTIVE_CAPITAL_ENGINE_ENABLED=true # Growth scaling
```

### Growth Capital Profile (BASE_CAPITAL >= 1000)

```bash
# Automatic via ADAPTIVE_CAPITAL_ENGINE
EV_MULTIPLIER → 1.65 (when equity > 1000)
EV_MULTIPLIER → 2.0  (when equity > 5000 & win_rate > 55%)
MIN_TRADE_QUOTE → scales up automatically
```

---

## ✅ Verification Checklist

- [x] Removed `_force_min_notional = True` from all bootstrap paths
- [x] Removed `_resolve_entry_quote_floor()` forced escalation
- [x] Updated `.env` to adaptive defaults (24/12 USDT)
- [x] Added EV_MULTIPLIER=1.4 for micro capital
- [x] Updated core/config.py defaults to match
- [x] All changes maintain backwards compatibility
- [x] ExecutionManager still enforces exchange minimums
- [x] ScalingManager controls allocation logic
- [x] No breaking changes to signal structure

---

## 🧠 Design Principles Restored

1. **Adaptive Scaling** ✅
   - Sizing respects capital constraints
   - No artificial escalation overrides
   - System learns from realizable trades

2. **Layered Risk** ✅
   - ScalingManager: Budget allocation
   - EV Gate: Alpha discipline
   - ExecutionManager: Exchange compliance

3. **Bootstrap Grace** ✅
   - Lower EV multiplier during learning
   - Adaptive quote sizing
   - Automatic scaling as capital grows

4. **Economic Integrity** ✅
   - All trades respect cost > benefit
   - No forced losses
   - Sustainable growth path

---

## 🎯 Next Steps

1. **Deploy** this version to production
2. **Monitor** log output:
   ```bash
   grep "ADAPTIVE_QUOTE" logs/agents/meta_controller.log
   grep "decisions_count" logs/status.log
   ```
3. **Verify** trading frequency increases (target: +20-30% after fix)
4. **Watch** win rate maintenance (should stay consistent)
5. **Scale** EV_MULTIPLIER as capital grows via ADAPTIVE_CAPITAL_ENGINE

---

## 💡 Key Insight

> The system wasn't broken. It was running institutional constraints on micro capital. The solution isn't to force escalation—it's to **let adaptive mechanisms work at the right scale**.

**Forced escalation:** "Make it work anyway"  
**Adaptive approach:** "Scale the constraint to match the capital"  

This restores the integrity of the learning phase while maintaining discipline for the growth phase.

---

## 📝 Code Artifacts Changed

| File | Lines | Change |
|------|-------|--------|
| `core/meta_controller.py` | 11830-11856 | Removed forced escalation in `_planned_quote_for()` |
| `core/meta_controller.py` | 8270-8280 | Removed forced escalation in FLAT_PORTFOLIO |
| `core/meta_controller.py` | 8298-8318 | Removed forced escalation in bootstrap escape |
| `.env` | 27-36 | Updated to adaptive defaults (24/12 USDT) |
| `.env` | 144-148 | Added EV_MULTIPLIER=1.4 for micro capital |
| `core/config.py` | 50-80 | Updated defaults to match .env |

---

**Status:** ✅ Ready for deployment  
**Risk Level:** Low (restores native adaptive mechanisms)  
**Expected Outcome:** Bootstrap phase unlocked with sustainable scaling path
