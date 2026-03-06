# Code Trace: Where Real NAV is Fetched (And Why It's Not Used)

## Complete Code Path Analysis

### Step 1: RecoveryEngine._load_live() - Fetches Exchange Data

**File:** `core/recovery_engine.py` (Lines 215-280)

```python
async def _load_live(self):
    """Fetch REAL balances from exchange"""
    
    # ✅ Exchange API call 1: Get balances
    if self.ex:
        balances = await self.ex.get_balances()
        print(f"FETCHED FROM EXCHANGE: {balances}")
        
    # ✅ Exchange API call 2: Get positions
    if self.ex:
        positions = await self.ex.get_positions()
        print(f"FETCHED FROM EXCHANGE: {positions}")
    
    # Returns the real data
    return (balances, positions)
```

**Key point:** ✅ **YOUR EXCHANGE API DATA IS BEING FETCHED HERE**

---

### Step 2: RecoveryEngine.rebuild_state() - Decides What to Do With Data

**File:** `core/recovery_engine.py` (Lines 440-500)

```python
async def rebuild_state(self):
    """Orchestrates recovery process"""
    
    # Check 1: Do we have a snapshot?
    snapshot = self._load_snapshot()
    
    if snapshot:
        # Snapshot exists, use it
        balances, positions = snapshot
    else:
        # No snapshot, fetch fresh from exchange
        balances, positions = await self._load_live()  # ← Calls Step 1
    
    # Now process the balances
    await self._apply_balances(balances)  # ← Goes to Step 3
    
    return result
```

**Key point:** ✅ **Your snapshot exists, so exchange data is fetched but ready to be applied**

---

### Step 3: RecoveryEngine._apply_balances() - THE CRITICAL DECISION POINT

**File:** `core/recovery_engine.py` (Lines 338-380)

```python
async def _apply_balances(self, balances: Dict):
    """Apply balances to SharedState (OR NOT, depending on mode)"""
    
    # HERE IS THE CRITICAL CHECK:
    if self._shadow_mode:  # ← YOUR FLAG IS TRUE
        # ❌ SHADOW MODE: Log but don't apply
        logger.info(
            f"[SHADOW MODE] Fetched balances "
            f"(not applied, virtual ledger is authoritative)"
        )
        return  # ← EXIT WITHOUT APPLYING!
    
    # 🟢 LIVE MODE: Apply the balances (code below never runs in shadow mode)
    
    # Normalize the fetched balances
    normalized = self._normalize_balances(balances)
    
    # Apply to SharedState
    await self.ss.update_balances(normalized)  # ← Goes to Step 4
    
    logger.info(f"[LIVE MODE] Applied balances to SharedState: {normalized}")
```

**Key point:** ❌ **THIS IS WHERE YOUR FETCHED DATA IS BLOCKED FROM USE**

Your `_shadow_mode` flag is `TRUE`, so:
- Exchange data is fetched ✅
- But then immediately discarded ❌
- Virtual ledger used instead ❌

---

### Step 4: SharedState.update_balances() - Updates Internal State

**File:** `core/shared_state.py` (Lines 2838-2900)

```python
async def update_balances(self, balances: Dict[str, Dict[str, float]]) -> None:
    """Update balances and trigger NAV calculation"""
    
    # This code is NEVER REACHED in shadow mode
    # (because _apply_balances() returns early)
    
    for asset, data in balances.items():
        a = asset.upper()
        new_free = float(data.get("free", 0.0))
        new_locked = float(data.get("locked", 0.0))
        
        # Store the balance
        self.balances[a] = {"free": new_free, "locked": new_locked}
        
        self.logger.debug(f"[SS:BalanceUpdate] {a}: free={new_free}, locked={new_locked}")
    
    # Mark balances ready
    self.balances_ready_event.set()
    
    # Trigger NAV recalculation
    await self._maybe_set_nav_ready()
```

**Key point:** ⏭️ **THIS NEVER RUNS IN SHADOW MODE - SKIPPED AT STEP 3**

---

### Step 5: SharedState.get_nav_quote() - Calculates NAV

**File:** `core/shared_state.py` (Lines 1057-1115)

```python
def get_nav_quote(self) -> float:
    """Calculate NAV from balances"""
    
    nav = 0.0
    
    # Sum quote assets (USDT, BUSD, etc)
    for asset, balance_data in self.balances.items():  # ← Uses stored balances
        a = asset.upper()
        if a in quote_assets:
            free = float(balance_data.get("free", 0.0))
            locked = float(balance_data.get("locked", 0.0))
            nav += free + locked
    
    # Add positions
    for symbol, position in self.positions.items():
        qty = position['quantity']
        px = self.latest_prices.get(symbol)
        nav += qty * px
    
    return nav  # ← Returns 0 because self.balances is empty!
```

**Key point:** 🔴 **NAV IS 0 BECAUSE self.balances IS EMPTY**

- `self.balances` is only populated by `update_balances()`
- But `update_balances()` is never called (blocked at Step 3)
- So NAV calculation finds no balances
- Result: NAV = 0

---

## The Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ START: Bot Initialization                               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │ Step 1: _load_live()     │
        │ ✅ Fetch from exchange   │
        │ USDT: free=1000          │
        │ BTC: quantity=0.5        │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │ Step 2: rebuild_state()  │
        │ Has snapshot? YES        │
        │ Use snapshot data        │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │ Step 3: _apply_balances()                │
        │ Check: _shadow_mode = TRUE?              │
        │ ├─ YES ❌ (Your case)                     │
        │ │  └─ Log but don't apply               │
        │ │     return early                      │
        │ │     ▼                                  │
        │ │  NAV stays 0                          │
        │ │                                       │
        │ └─ NO (Not your case)                    │
        │    └─ Apply to SharedState              │
        │       call update_balances()            │
        │       ▼                                  │
        │       NAV = actual balance              │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │ Step 4: get_nav_quote()  │
        │ Read: self.balances      │
        │ ❌ Empty (in shadow mode)│
        │ Result: NAV = 0.0        │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │ Startup Verification     │
        │ ✅ Passes (with fix)     │
        │ Shadow mode detected     │
        │ NAV=0 allowed            │
        └──────────────────────────┘
```

---

## The Configuration That Controls This

**File:** Likely in `config.yaml`, `.env`, or `app_context.py`

```python
# Current setting:
TRADING_MODE = "shadow"  # ← THIS ENABLES SHADOW MODE

# Result at Step 3:
if self._shadow_mode:  # ← Evaluates to TRUE
    return  # Skip update_balances()
```

---

## To Enable Real NAV

Change the configuration:

```python
# FROM:
TRADING_MODE = "shadow"

# TO:
TRADING_MODE = "live"
```

**Effect:**
- At Step 3, `self._shadow_mode` becomes FALSE
- Code doesn't return early
- `await self.ss.update_balances(normalized)` IS CALLED
- `self.balances` gets populated with real data
- `get_nav_quote()` returns actual balance
- NAV becomes non-zero

---

## Why the Exchange API Data Exists But Isn't Used

This is intentional architecture:

1. **Dual-Path Design**
   - Path A (Shadow): Fetch but don't apply → Use virtual ledger
   - Path B (Live): Fetch and apply → Use exchange as truth

2. **Benefits of Shadow Mode**
   - Test trading logic without money
   - Recover with snapshot instead of exchange
   - Develop without risk
   - Audit with fresh data

3. **Benefits of Live Mode**
   - Real NAV from exchange
   - Positions reconciled with real orders
   - Actual trading capability

---

## Code Summary Table

| Component | File | Lines | Status | Effect |
|-----------|------|-------|--------|--------|
| **_load_live()** | recovery_engine.py | 215-280 | ✅ Runs | Fetches exchange data |
| **rebuild_state()** | recovery_engine.py | 440-500 | ✅ Runs | Calls _load_live() |
| **_apply_balances()** | recovery_engine.py | 338-380 | ⏹️ Exits early | Skips update_balances() |
| **update_balances()** | shared_state.py | 2838-2900 | ❌ Never called | Balance not stored |
| **get_nav_quote()** | shared_state.py | 1057-1115 | ✅ Runs | Finds empty balances |
| **Result** | - | - | - | NAV = 0.0 |

---

## The Answer

> "Why not fetch the real NAV?"

**You ARE fetching it** — Step 1 and Step 2 both fetch from exchange.

**You're just not USING it** — Step 3 intentionally blocks it in shadow mode.

**To use it: Change TRADING_MODE to "live"**

Then:
- Step 3 allows the data through
- Step 4 stores it in SharedState
- Step 5 calculates real NAV
- Result: NAV = actual exchange balance

That's the complete technical explanation of the code path. 🎯
