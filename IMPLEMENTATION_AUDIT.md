# 🔍 CODEBASE AUDIT - Symbol Churn Fix Implementation

**Date:** May 3, 2026  
**Goal:** Identify existing code to reuse/modify instead of creating duplicates

---

## ✅ EXISTING CODE INVENTORY

### Part 1: Config.py (Lines 1-2403)

**Already Exists:**
- ✅ `MAX_ACTIVE_SYMBOLS = 5` (line ~38)
- ✅ `MIN_ACTIVE_SYMBOLS = 3` (line ~39)
- ✅ `MAX_UNIVERSE_SYMBOLS = 30` (line ~32)
- ✅ `SCREENER_MIN_VOLUME`, `SCREENER_MAX_PROPOSALS` (lines ~34-36)
- ✅ Capital allocation framework: `FIX8_COMPOUND_ALLOCATION_PCT = 0.60`, `FIX8_HEALING_ALLOCATION_PCT = 0.20`, `FIX8_BUFFER_ALLOCATION_PCT = 0.20` (lines ~70-72)
- ✅ `FIX8_4TH_SLOT_ENABLED = True` (line ~74)

**Missing (Need to Add):**
- ❌ `PROVEN_SYMBOLS` dict mapping (7 symbols with trade counts)
- ❌ `SYMBOL_CONVERGENCE_MODE` flag
- ❌ `CONVERGENCE_MIN_HISTORY_TRADES` threshold
- ❌ `CONVERGENCE_MAX_EXPERIMENTAL_SYMBOLS` limit
- ❌ `CONVERGENCE_MAX_NEW_SYMBOLS_PER_DAY` throttle
- ❌ `EXCLUDED_SYMBOLS` dict (list of bad symbols)

**Action:** Add these 6 new config params to config.py around line 75-120

---

### Part 2: SharedState (Lines 1028-3700+)

**Already Exists:**
- ✅ `accepted_symbols: Dict[str, Dict[str, Any]]` (line 1329) - stores approved symbols
- ✅ `symbol_proposals: Dict[str, Dict[str, Any]]` (line 1473) - stores proposals
- ✅ `set_accepted_symbols()` method (line 3567) - modifies accepted symbols
- ✅ `get_accepted_symbols()` method (implicit via property)

**Missing (Need to Add):**
- ❌ `is_symbol_excluded()` method
- ❌ `add_to_exclusion_list()` method  
- ❌ `new_symbol_timestamps` dict to track when symbols were added
- ❌ `symbol_convergence_state` to track convergence mode

**Action:** Add 4 new methods to SharedState class

---

### Part 3: SymbolScreener (Lines 1-628)

**Already Exists:**
- ✅ `_propose()` method (line ~13) - proposes symbols to symbol_proposals
- ✅ `_prefilter_symbol()` method (line ~39) - filters by trading status
- ✅ `_cfg()` method (line ~149) - config lookup
- ✅ Properties for thresholds: `min_volume`, `min_percent_change`, etc.

**Missing (Need to Add):**
- ❌ `_should_accept_symbol()` method - gating logic
- ❌ `_get_experimental_count()` helper - counts current experimental symbols
- ❌ `_is_proven_symbol()` method - checks if in proven list
- ❌ Integration of gating into existing `_propose()` call

**Action:** Add 3-4 new methods to SymbolScreener class, modify `_propose()` wrapper

---

### Part 4: SwingTradeHunter (Lines 1-1075)

**Already Exists:**
- ✅ Position sizing logic (line ~80+) - calculates position size
- ✅ Capital allocation framework (implicit in execution)
- ✅ Symbol loop iteration

**Missing (Need to Add):**
- ❌ `get_available_capital_for_symbol()` method - allocate 80/20 by symbol type
- ❌ Integration point to check if symbol is proven before full allocation

**Action:** Add 1 new method, integrate into position sizing logic

---

### Part 5: Capital Allocation Check

**Current State:**
- ✅ `FIX8_COMPOUND_ALLOCATION_PCT = 0.60` exists
- ✅ Healing allocation exists at 20%
- ⚠️ But: Not yet integrated with "proven symbols" concept

**Missing (Need to Do):**
- Connect allocation percentages to symbol classification (proven vs experimental)
- Modify position size calculation in SwingTradeHunter or execution layer

---

## 🔴 QUESTIONS FOR YOU

Before I implement, I need clarification on:

### Q1: Healing Liquidations
**Current Code:** `liquidation_agent.py` handles healing cycle
**Question:** Should healing agent also respect the gating? (e.g., only liquidate non-proven symbols first?)
**Status:** ⏳ NEED ANSWER

### Q2: Symbol Replacement Strategy
**Current Code:** `SYMBOL_REPLACEMENT_MULTIPLIER = 1.10` exists in config
**Question:** Should this replacement logic ONLY propose from proven symbols, or can it test new ones?
**Status:** ⏳ NEED ANSWER

### Q3: "Experimental" Symbol Definition
**Current Code:** System has `MAX_ACTIVE_SYMBOLS = 5` but no explicit "experimental" tracking
**Question:** Is an "experimental" symbol one that's in the 5 active but NOT in the proven 7?
**Status:** ⏳ NEED ANSWER - This affects counting logic

### Q4: Healing Capital Allocation
**Current Code:** Healing gets 20% of capital
**Question:** Should healing budget also vary based on # of experimental symbols?
(e.g., more healing budget if testing risky symbols)
**Status:** ⏳ NEED ANSWER

### Q5: Discovery Integration
**Current Code:** SymbolScreener writes to `symbol_proposals`, then UURE integration happens
**Question:** Should gating happen in:
- A) SymbolScreener._propose() [prevents bad symbols from entering proposals]
- B) UURE discovery integration [filters from proposals before adding to accepted]
- C) Both (belt-and-suspenders)?
**Status:** ⏳ NEED ANSWER - Best practice?

---

## 📋 RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: CONFIG (No dependencies)
- Add 6 new config params to `config.py`
- Add `PROVEN_SYMBOLS` dict with 7 proven symbols
- Add `EXCLUDED_SYMBOLS` dict with 25 bad symbols

### Phase 2: SHARED_STATE (Depends on Phase 1)
- Add `is_symbol_excluded()` method
- Add `add_to_exclusion_list()` method
- Add `new_symbol_timestamps` tracking

### Phase 3: SYMBOL_SCREENER (Depends on Phase 1-2)
- Add `_should_accept_symbol()` gating method
- Add `_is_proven_symbol()` helper
- Add `_get_experimental_count()` counter
- Modify `_propose()` to call gating

### Phase 4: SWING_TRADE_HUNTER (Depends on Phase 1)
- Add `get_available_capital_for_symbol()` method
- Integrate into position sizing logic

### Phase 5: HEALING AGENT (Depends on Phase 1-2)
- Review liquidation logic (check if gating needed)
- Possibly add early exit for experimental symbols

### Phase 6: VALIDATION & INTEGRATION (Depends on all phases)
- Create checkpoint files
- Validate all 10 fixes are in place
- Test restart behavior

---

## ⚡ QUESTIONS I'LL ANSWER BASED ON YOUR FEEDBACK

I'll wait for your responses to:
1. Q1: Healing agent gating?
2. Q2: Symbol replacement from proven only?
3. Q3: Experimental symbol definition?
4. Q4: Dynamic healing capital?
5. Q5: Gating location (screener vs UURE)?

**Please respond with:** YES/NO/MODIFIED for each question, plus any additional context.

---

## 🎯 NEXT STEP

Once you answer the 5 questions above, I'll proceed with:
1. **Config edits** (straightforward)
2. **SharedState additions** (new methods)
3. **SymbolScreener integration** (gating logic)
4. **SwingTradeHunter integration** (capital allocation)
5. **Validation** (checkpoint files)

**Estimated time after Q&A:** 30-40 minutes

**Total system downtime:** ~5 minutes for restart

---

**Status:** ⏳ AWAITING YOUR ANSWERS TO 5 QUESTIONS
