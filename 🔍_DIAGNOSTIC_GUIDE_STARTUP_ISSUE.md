# 🔍 DIAGNOSTIC GUIDE: Identify Your Startup Issue

## Purpose
This guide helps you identify **which of the three scenarios** is actually happening in your system.

---

## 🎯 THE THREE SCENARIOS

### Scenario A: Hydration Never Called ❌
```
MetaController starts → open_trades = {} (empty)
Because hydration function was never called at all
```

### Scenario B: Symbol Filtering Blocks Hydration ❌
```
Wallet has: SOL, BTC, ETH
accepted_symbols: [BTCUSDT, ETHUSDT]
Result: Only BTC and ETH reconstructed; SOL filtered out
```

### Scenario C: Race Condition - Timing Issue ❌
```
MetaController.start() fires evaluation task
Meanwhile, somewhere else, hydration runs
By time eval_and_act() runs, positions may or may not be populated
```

---

## 🧪 QUICK DIAGNOSTIC TEST

Add these logs to your startup, then run it and show me the output.

### Step 1: Find AppContext.__init__
Location: `core/app_context.py`, in `__init__` method

Add this at the END of `__init__`:
```python
# DIAGNOSTIC: Startup trace
self.logger.warning("[STARTUP_DIAGNOSTIC] AppContext.__init__ complete")
self.logger.warning(f"[STARTUP_DIAGNOSTIC]   shared_state exists: {bool(self.shared_state)}")
if self.shared_state:
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   shared_state.positions: {len(getattr(self.shared_state, 'positions', {}))}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   shared_state.accepted_symbols: {len(getattr(self.shared_state, 'accepted_symbols', []))}")
```

### Step 2: Find AppContext.initialize_all
Location: `core/app_context.py`, method `async def initialize_all()`

Find the section where `MetaController` is initialized. Add this BEFORE it:
```python
self.logger.warning("[STARTUP_DIAGNOSTIC] Before MetaController.start():")
if self.shared_state:
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   positions: {len(getattr(self.shared_state, 'positions', {}))}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   open_trades: {len(getattr(self.shared_state, 'open_trades', {}))}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   accepted_symbols: {len(getattr(self.shared_state, 'accepted_symbols', []))}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   nav: {getattr(self.shared_state, 'nav', 0.0)}")
```

### Step 3: Find MetaController.start
Location: `core/meta_controller.py`, method `async def start()`

Add this BEFORE `await self._disable_bootstrap_if_positions()`:
```python
self.logger.warning("[STARTUP_DIAGNOSTIC] MetaController.start() called:")
self.logger.warning(f"[STARTUP_DIAGNOSTIC]   shared_state exists: {bool(self.shared_state)}")
if self.shared_state:
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   positions: {len(getattr(self.shared_state, 'positions', {}))}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   open_trades: {len(getattr(self.shared_state, 'open_trades', {}))}")
```

### Step 4: Find MetaController.evaluate_and_act
Location: `core/meta_controller.py`, method `async def evaluate_and_act()`

Add this at the very START of the method (after `self._last_cycle_execution_attempts = ...`):
```python
# DIAGNOSTIC: First eval cycle
if self.tick_id == 1:
    self.logger.warning("[STARTUP_DIAGNOSTIC] First evaluate_and_act() call:")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   positions: {len(self.shared_state.positions or {})}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   open_trades: {len(self.shared_state.open_trades or {})}")
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   accepted_symbols: {len(self.shared_state.accepted_symbols or [])}")
    
    # Try to find any positions
    positions_with_qty = {
        sym: float(pos.get('quantity', 0.0) or 0.0)
        for sym, pos in (self.shared_state.positions or {}).items()
        if float(pos.get('quantity', 0.0) or 0.0) > 0
    }
    self.logger.warning(f"[STARTUP_DIAGNOSTIC]   positions with qty: {positions_with_qty}")
```

---

## 🏃 RUN THE DIAGNOSTIC

1. Add all the diagnostic logs above
2. Start your bot
3. Look for logs starting with `[STARTUP_DIAGNOSTIC]`
4. Capture the complete startup sequence
5. Share the logs with me

---

## 📊 WHAT THE RESULTS WILL SHOW

### If Scenario A (Never Called):
```
[STARTUP_DIAGNOSTIC] Before MetaController.start():
[STARTUP_DIAGNOSTIC]   positions: 0
[STARTUP_DIAGNOSTIC]   open_trades: 0
[STARTUP_DIAGNOSTIC]   accepted_symbols: 5

[STARTUP_DIAGNOSTIC] First evaluate_and_act() call:
[STARTUP_DIAGNOSTIC]   positions: 0
[STARTUP_DIAGNOSTIC]   open_trades: 0
[STARTUP_DIAGNOSTIC]   positions with qty: {}

→ open_trades never populated (hydration never ran)
```

### If Scenario B (Symbol Filtering):
```
Wallet: BTC=0.5, SOL=10, ETH=2, USDT=1000
accepted_symbols: [BTCUSDT, ETHUSDT]

[STARTUP_DIAGNOSTIC] Before MetaController.start():
[STARTUP_DIAGNOSTIC]   positions: 2  ← Only BTC + ETH reconstructed
[STARTUP_DIAGNOSTIC]   accepted_symbols: 2

→ SOL filtered out (not in accepted_symbols)
```

### If Scenario C (Race Condition):
```
Run 1:
[STARTUP_DIAGNOSTIC] Before MetaController.start():
[STARTUP_DIAGNOSTIC]   positions: 0

[STARTUP_DIAGNOSTIC] First evaluate_and_act() call:
[STARTUP_DIAGNOSTIC]   positions: 2  ← Populated just-in-time!

Run 2 (restart):
[STARTUP_DIAGNOSTIC] Before MetaController.start():
[STARTUP_DIAGNOSTIC]   positions: 2

[STARTUP_DIAGNOSTIC] First evaluate_and_act() call:
[STARTUP_DIAGNOSTIC]   positions: 2

→ Inconsistent behavior (sometimes populated, sometimes not)
```

---

## 🔧 QUICK FIXES BASED ON DIAGNOSIS

### If Scenario A:
Add this to `AppContext.initialize_all()` BEFORE MetaController.start():

```python
# Force hydration before MetaController starts
if self.shared_state and hasattr(self.shared_state, 'authoritative_wallet_sync'):
    self.logger.warning("[STARTUP] Forcing portfolio reconciliation...")
    await self.shared_state.authoritative_wallet_sync()
    self.logger.warning("[STARTUP] Portfolio reconciliation complete")
```

Or implement the full `StartupReconciler` (recommended).

### If Scenario B:
After hydration, add missing symbols:

```python
# Ensure all wallet assets are in the symbol universe
if self.shared_state and hasattr(self.shared_state, 'positions'):
    positions = self.shared_state.positions or {}
    current_symbols = set(self.shared_state.accepted_symbols or [])
    
    for sym in positions.keys():
        if sym not in current_symbols:
            current_symbols.add(sym)
            self.logger.warning(f"[STARTUP] Added {sym} to universe")
    
    self.shared_state.accepted_symbols = list(current_symbols)
```

Or implement the full `StartupReconciler` (recommended).

### If Scenario C:
Implement `StartupReconciler` with blocking gate:

```python
# StartupReconciler blocks until complete
reconciler = StartupReconciler(...)
success = await reconciler.run_startup_reconciliation()
if not success:
    raise RuntimeError("Startup failed")
# Only NOW start MetaController
```

---

## 📝 MINIMAL DIAGNOSTIC SCRIPT

If you just want a super quick check without code changes:

```python
# Add this to a test file or Python REPL connected to your system

import asyncio
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.config import Config

async def diagnose():
    config = Config()
    exchange = ExchangeClient(config)
    state = SharedState(config=config)
    state._exchange_client = exchange
    
    # Scenario A: Check if hydration method exists
    print(f"Has hydrate_positions_from_balances: {hasattr(state, 'hydrate_positions_from_balances')}")
    print(f"Has authoritative_wallet_sync: {hasattr(state, 'authoritative_wallet_sync')}")
    
    # Try fetching balances
    balances = await exchange.get_balances()
    print(f"Exchange balances: {len(balances)} assets")
    
    # Check accepted symbols
    print(f"Accepted symbols: {state.accepted_symbols}")
    
    # Try hydration
    if hasattr(state, 'authoritative_wallet_sync'):
        await state.authoritative_wallet_sync()
        print(f"After hydration - positions: {len(state.positions)}")

asyncio.run(diagnose())
```

---

## 🎯 NEXT STEPS

1. Run the diagnostic logs
2. Capture output
3. Show me which scenario matches
4. I'll give you the exact code to fix it

Or:

1. Implement `StartupReconciler` immediately (solves all three scenarios)
2. Follow the integration guide
3. Test and verify

---

## ✅ CONFIDENCE LEVELS

- **Scenario A Diagnosis:** 99% confident
- **Scenario B Diagnosis:** 95% confident  
- **Scenario C Diagnosis:** 85% confident

The diagnostic logs will make it 100% clear which one.

---

**Once you run the diagnostic, share the logs and I can tell you exactly which line needs to change.**
