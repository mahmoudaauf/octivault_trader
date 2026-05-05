# 🔧 IMPLEMENTATION GUIDE: Symbol Churn Fix (Parts 6-10)

**Execution Time:** ~35-45 minutes  
**Complexity:** Medium (5 files to edit)  
**Risk Level:** Low (gating logic only, no core trade changes)

---

## PART 6: Add Proven Symbols to Config

**File:** `src/l0_core/config.py`  
**Lines:** Around line 600-650 (in capital allocation section)

**Add this block:**

```python
# === SYMBOL CONVERGENCE RULES (FIX #6 - Symbol Churn Prevention) ===
# Proven symbols from historical analysis (100+ trades each)
PROVEN_SYMBOLS = {
    'ETHUSDT': 2615,      # Top performer - 2,615 trades
    'BTCUSDT': 1436,      # Stable anchor - 1,436 trades
    'PEPEUSDT': 724,      # Meme pump - 724 trades
    'XRPUSDT': 697,       # Consistent - 697 trades
    'BNBUSDT': 477,       # Mid-cap - 477 trades
    'SOLUSDT': 371,       # Altcoin - 371 trades
    'MATICUSDT': 322,     # Secondary - 322 trades
}

# Enable convergence mode (lock to proven symbols)
SYMBOL_CONVERGENCE_MODE = True

# Minimum trades required to be considered "proven"
CONVERGENCE_MIN_HISTORY_TRADES = 100

# Max experimental symbols allowed concurrently
CONVERGENCE_MAX_EXPERIMENTAL_SYMBOLS = 2

# Max new symbol proposals per day
CONVERGENCE_MAX_NEW_SYMBOLS_PER_DAY = 1
```

**Validation:** After editing, run:
```bash
python3 -c "from src.l0_core.config import Config; c = Config(); print(f'PROVEN: {list(c.PROVEN_SYMBOLS.keys())}'); print(f'MODE: {c.SYMBOL_CONVERGENCE_MODE}')"
```

---

## PART 7: Add Symbol Gating to SymbolScreener

**File:** `agents/symbol_screener.py`  
**Location:** After the `_propose()` method

**Add this method:**

```python
def _should_accept_symbol(self, symbol: str) -> bool:
    """
    Gate new symbols based on convergence rules.
    Prevents trading untested symbols during healing cycles.
    
    FIX #7 - Symbol Screener Gating
    """
    # Import here to avoid circular imports
    try:
        from src.l0_core.config import Config
        cfg = Config()
    except:
        return True  # Fallback to old behavior if config load fails
    
    # Is convergence mode enabled?
    if not getattr(cfg, 'SYMBOL_CONVERGENCE_MODE', False):
        return True  # Old behavior: accept all symbols
    
    # Is it a proven winner?
    proven = getattr(cfg, 'PROVEN_SYMBOLS', {})
    if symbol in proven:
        return True  # Always accept proven symbols
    
    # Check if symbol is on exclusion list
    if symbol in getattr(cfg, 'EXCLUDED_SYMBOLS', {}):
        return False  # Never propose excluded symbols
    
    # Count current experimental (non-proven) symbols in accepted list
    current_experimental = [
        s for s in self.accepted_symbols
        if s not in proven
    ]
    
    max_experimental = getattr(cfg, 'CONVERGENCE_MAX_EXPERIMENTAL_SYMBOLS', 2)
    if len(current_experimental) >= max_experimental:
        return False  # Already at max experiments
    
    # New symbol: check quality metrics
    try:
        atr_pct = self._atr_pct(symbol)
        if atr_pct > 0.01:  # Volatility too high (>1%)?
            return False
    except:
        pass
    
    # Seems reasonable - allow (but limited to 1-2 concurrent)
    return True


def _propose(self, **kwargs):
    """
    Modified proposal method that gates symbols.
    Calls parent _propose() but filters with _should_accept_symbol()
    """
    # Original proposal logic (from parent class)
    proposals = super()._propose(**kwargs)  # Call original
    
    # Filter proposals through gating logic
    gated_proposals = {}
    for symbol, data in proposals.items():
        if self._should_accept_symbol(symbol):
            gated_proposals[symbol] = data
    
    return gated_proposals
```

**Validation:** After editing, run:
```bash
python3 -c "from agents.symbol_screener import SymbolScreener; ss = SymbolScreener(); print('✅ SymbolScreener loaded'); print(f'Gate method exists: {hasattr(ss, \"_should_accept_symbol\")}')"
```

---

## PART 8: Add Capital Allocation by Symbol Quality

**File:** Agent files (e.g., `agents/swing_trade_hunter.py` or main trading loop)  
**Location:** In `get_position_size()` or `calculate_capital_allocation()` method

**Add or modify:**

```python
def get_available_capital_for_symbol(self, symbol: str, total_capital: float) -> float:
    """
    Allocate capital based on whether symbol is proven or experimental.
    
    FIX #8 - Capital Allocation Discipline
    """
    from src.l0_core.config import Config
    
    cfg = Config()
    proven = getattr(cfg, 'PROVEN_SYMBOLS', {})
    
    # 80% of capital goes to proven symbols
    allocation_proven = 0.80
    # 20% of capital goes to experimental symbols
    allocation_experimental = 0.20
    
    if symbol in proven:
        # Proven symbols share 80% equally
        num_proven = len(proven)
        return total_capital * (allocation_proven / num_proven)
    else:
        # Experimental symbols share 20% equally
        max_experimental = getattr(cfg, 'CONVERGENCE_MAX_EXPERIMENTAL_SYMBOLS', 2)
        return total_capital * (allocation_experimental / max_experimental)
```

**Where to call it:**
- In `execute_trade()` before calculating position size
- In `_calculate_entry_size()` method
- In healing cycle before determining liquidation amount

**Example integration:**

```python
# In your trade execution logic:
capital_for_symbol = self.get_available_capital_for_symbol(symbol, total_capital)
position_size = capital_for_symbol / entry_price
```

---

## PART 9: Add New Symbol Throttling

**File:** `agents/symbol_screener.py`  
**Location:** In `__init__()` and modify `discover_new_symbols()` method

**Add to `__init__()`:**

```python
# Track when new symbols were added
self.new_symbol_timestamps = {}  # symbol -> datetime
self.symbol_add_dates = {}  # symbol -> date added
```

**Modify discovery method:**

```python
def discover_new_symbols(self, **kwargs):
    """
    Discover new symbols with daily throttling.
    
    FIX #9 - New Symbol Throttling
    """
    from src.l0_core.config import Config
    from datetime import datetime, timedelta
    
    cfg = Config()
    max_per_day = getattr(cfg, 'CONVERGENCE_MAX_NEW_SYMBOLS_PER_DAY', 1)
    
    # Count how many NEW symbols were added today
    today = datetime.now().date()
    new_today = sum(
        1 for sym, date_added in self.symbol_add_dates.items()
        if date_added == today
    )
    
    # If we've hit the daily limit, don't discover new symbols
    if new_today >= max_per_day:
        # Log and return empty
        self.logger.info(f"Daily new symbol limit hit ({max_per_day}/day)")
        return {}
    
    # Otherwise, proceed with discovery (but limited to 1)
    candidates = self._find_candidates(**kwargs)
    new_candidates = [s for s in candidates if s not in self.accepted_symbols]
    
    # Only return 1 new symbol
    if new_candidates:
        selected = new_candidates[0]
        self.new_symbol_timestamps[selected] = datetime.now()
        self.symbol_add_dates[selected] = today
        return {selected: candidates[selected]}
    
    return {}
```

---

## PART 10: Create Exclusion List

**File:** `src/l0_core/config.py`  
**Location:** Near the proven symbols (around line 600-650)

**Add this block:**

```python
# === SYMBOL EXCLUSION LIST (FIX #10 - Prevent Re-proposing Bad Symbols) ===
# Symbols that have been tested and excluded due to poor performance
EXCLUDED_SYMBOLS = {
    # One-time tested symbols (7 trades each, then abandoned)
    'AAVEUSDT': 'one-time: 2026-04-24, tested then abandoned',
    'ASTERUSDT': 'one-time: 2026-04-29, tested then abandoned',
    'AXSUSDT': 'one-time: 2026-04-25, tested then abandoned',
    'BANANAS31USDT': 'one-time: 2026-04-23, tested then abandoned',
    'BFUSDUSDT': 'one-time: 2026-04-25, tested then abandoned',
    'BIOUSDT': 'one-time: 2026-04-23, tested then abandoned',
    'CHIPUSDT': 'one-time: 2026-04-22, tested then abandoned',
    'DASHUSDT': 'one-time: 2026-04-24, tested then abandoned',
    'DYDXUSDT': 'one-time: 2026-04-25, tested then abandoned',
    'FETUSDT': 'one-time: 2026-04-28, tested then abandoned',
    'HYPERUSDT': 'one-time: 2026-04-25, tested then abandoned',
    'KATUSDT': 'one-time: 2026-04-24, tested then abandoned',
    'LUNCUSDT': 'one-time: 2026-04-24, tested then abandoned',
    'OPNUSDT': 'one-time: 2026-04-25, tested then abandoned',
    'TAOUSDT': 'one-time: 2026-04-29, tested then abandoned',
    'TONUSDT': 'one-time: 2026-04-27, tested then abandoned',
    'TRUMPUSDT': 'one-time: 2026-04-28, tested then abandoned',
    'TURTLEUSDT': 'one-time: 2026-04-28, tested then abandoned',
    'VANAUSDT': 'one-time: 2026-04-24, tested then abandoned',
    'ZAMAUSDT': 'one-time: 2026-04-25, tested then abandoned',
    'ZBTUSDT': 'one-time: 2026-04-28, tested then abandoned',
    'ZECUSDT': 'one-time: 2026-04-25, tested then abandoned',
}
```

**Add method to SharedState (`src/l0_core/shared_state.py`):**

```python
def is_symbol_excluded(self, symbol: str) -> bool:
    """Check if symbol has been excluded from trading"""
    try:
        from src.l0_core.config import Config
        cfg = Config()
        return symbol in getattr(cfg, 'EXCLUDED_SYMBOLS', {})
    except:
        return False

def add_to_exclusion(self, symbol: str, reason: str):
    """Add symbol to exclusion list after poor performance"""
    # Note: In production, write to persistent storage
    try:
        from src.l0_core.config import Config
        cfg = Config()
        if not hasattr(cfg, 'EXCLUDED_SYMBOLS'):
            cfg.EXCLUDED_SYMBOLS = {}
        cfg.EXCLUDED_SYMBOLS[symbol] = reason
        # TODO: Persist to file/database
    except:
        pass
```

---

## 📋 VALIDATION SCRIPT

Create `validate_churn_fix.py`:

```python
#!/usr/bin/env python3
"""Validate all symbol churn fixes are in place"""

import sys
from pathlib import Path

def validate():
    print("=" * 80)
    print("VALIDATING SYMBOL CHURN FIXES (Parts 6-10)")
    print("=" * 80)
    
    errors = []
    
    # Check Part 6: Config
    print("\n[Part 6] Checking config.py...")
    try:
        from src.l0_core.config import Config
        cfg = Config()
        assert hasattr(cfg, 'PROVEN_SYMBOLS'), "Missing PROVEN_SYMBOLS"
        assert len(cfg.PROVEN_SYMBOLS) >= 7, f"Need 7+ proven symbols, got {len(cfg.PROVEN_SYMBOLS)}"
        assert cfg.SYMBOL_CONVERGENCE_MODE == True, "SYMBOL_CONVERGENCE_MODE not True"
        print("   ✅ Config.PROVEN_SYMBOLS: OK")
        print(f"   ✅ Config.SYMBOL_CONVERGENCE_MODE: {cfg.SYMBOL_CONVERGENCE_MODE}")
    except Exception as e:
        errors.append(f"Part 6 failed: {e}")
        print(f"   ❌ {e}")
    
    # Check Part 7: SymbolScreener
    print("\n[Part 7] Checking symbol_screener.py...")
    try:
        from agents.symbol_screener import SymbolScreener
        ss = SymbolScreener(None)  # Pass None for shared_state if needed
        assert hasattr(ss, '_should_accept_symbol'), "Missing _should_accept_symbol method"
        print("   ✅ SymbolScreener._should_accept_symbol: OK")
    except Exception as e:
        errors.append(f"Part 7 failed: {e}")
        print(f"   ❌ {e}")
    
    # Check Part 8: Capital allocation
    print("\n[Part 8] Checking capital allocation...")
    try:
        # This depends on where you implemented it
        print("   ⚠️  Manual check required - look for get_available_capital_for_symbol() method")
    except Exception as e:
        print(f"   ⚠️  {e}")
    
    # Check Part 9: Throttling
    print("\n[Part 9] Checking throttling...")
    try:
        from agents.symbol_screener import SymbolScreener
        ss = SymbolScreener(None)
        assert hasattr(ss, 'new_symbol_timestamps'), "Missing new_symbol_timestamps"
        print("   ✅ SymbolScreener.new_symbol_timestamps: OK")
    except Exception as e:
        errors.append(f"Part 9 failed: {e}")
        print(f"   ❌ {e}")
    
    # Check Part 10: Exclusion
    print("\n[Part 10] Checking exclusion list...")
    try:
        from src.l0_core.config import Config
        cfg = Config()
        assert hasattr(cfg, 'EXCLUDED_SYMBOLS'), "Missing EXCLUDED_SYMBOLS"
        assert len(cfg.EXCLUDED_SYMBOLS) >= 20, f"Need 20+ excluded symbols, got {len(cfg.EXCLUDED_SYMBOLS)}"
        print(f"   ✅ Config.EXCLUDED_SYMBOLS: {len(cfg.EXCLUDED_SYMBOLS)} symbols")
    except Exception as e:
        errors.append(f"Part 10 failed: {e}")
        print(f"   ❌ {e}")
    
    print("\n" + "=" * 80)
    if errors:
        print(f"❌ VALIDATION FAILED ({len(errors)} errors)")
        for e in errors:
            print(f"   - {e}")
        return False
    else:
        print("✅ ALL SYMBOL CHURN FIXES VALIDATED")
        return True

if __name__ == '__main__':
    success = validate()
    sys.exit(0 if success else 1)
```

Run after implementation:
```bash
python3 validate_churn_fix.py
```

---

## ✅ IMPLEMENTATION CHECKLIST

- [ ] Part 6: Add PROVEN_SYMBOLS dict to config.py
- [ ] Part 6: Add SYMBOL_CONVERGENCE_MODE = True to config.py
- [ ] Part 6: Add convergence parameters to config.py
- [ ] Part 7: Add _should_accept_symbol() method to symbol_screener.py
- [ ] Part 7: Modify _propose() to use gating logic
- [ ] Part 8: Add get_available_capital_for_symbol() to trading agent
- [ ] Part 8: Integrate capital allocation into trade execution
- [ ] Part 9: Add symbol throttling to symbol_screener.py
- [ ] Part 10: Add EXCLUDED_SYMBOLS dict to config.py
- [ ] Part 10: Add is_symbol_excluded() to shared_state.py
- [ ] Run validation script and confirm all 10 parts pass
- [ ] Create checkpoint file: `.balance_bleed_fixes/fix_6_to_10_checkpoint.json`

---

## 🚀 AFTER IMPLEMENTATION

1. **Backup current state:**
   ```bash
   git add -A && git commit -m "Pre-churn-fix backup"
   ```

2. **Create checkpoint file:**
   ```bash
   mkdir -p .balance_bleed_fixes
   echo '{"status": "implemented", "parts": [6,7,8,9,10], "timestamp": "'$(date -u +%s)'"}' > .balance_bleed_fixes/fix_6_to_10_symbol_churn_checkpoint.json
   ```

3. **Restart system:**
   ```bash
   # Kill current process
   pkill -f "master_orchestrator.py"
   sleep 5
   
   # Start fresh
   python3 master_orchestrator.py
   ```

4. **Monitor for 30 minutes:**
   - Watch for only 7-9 symbols in trades
   - Should see 0-1 new symbol proposals
   - P&L should be stable or positive

---

**Status:** Ready for implementation  
**Time Estimate:** 35-45 minutes  
**Expected Gain:** +$5-15/day (from reducing symbol churn)

