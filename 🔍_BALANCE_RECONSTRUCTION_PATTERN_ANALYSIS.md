# Balance Reconstruction Pattern Analysis

## Question
Is the following complete balance reconstruction pattern existing in the codebase?

```python
# The proposed unified pattern:
balances = await exchange_client.get_balances()
free_usdt = balances.get("USDT", 0)

for asset, qty in balances.items():
    if asset == "USDT":
        continue
    if qty <= dust_threshold:
        continue
    create_position_from_wallet(asset, qty)

NAV = free_usdt + positions_value
```

## Answer: **PARTIALLY IMPLEMENTED & DISTRIBUTED**

The pattern exists in **three separate components** with **incomplete cross-integration**:

---

## 1. RecoveryEngine: Balance Fetching ✅

**File:** `core/recovery_engine.py`

**What exists:**
```python
async def _load_live(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch balances and positions live from the exchange client."""
    balances: Dict[str, Any] = {}
    try:
        if hasattr(self.ex, "get_balances"):
            b = await _with_timeout(self.ex.get_balances())
            balances = _normalize_balances(b)
    except Exception:
        self.logger.warning("[Recovery] Live balances fetch failed")
    return balances, positions
```

**Method:** `_apply_balances()` (lines 338-403)
- ✅ Calls `exchange_client.get_balances()`
- ✅ Normalizes to dict format with free/locked/total
- ❌ Does NOT filter by dust_threshold
- ❌ Does NOT create positions from wallet balances
- ✅ Updates SharedState via `update_balances()`

**Then:** `rebuild_state()` (lines 440-522)
- ✅ Calls `_load_live()` → `_apply_balances()`
- ✅ Calls `_apply_positions()` (separate load from exchange)
- ❌ Does NOT integrate dust filtering

---

## 2. ExchangeTruthAuditor: Balance Reconciliation & Phantom Detection ✅

**File:** `core/exchange_truth_auditor.py`

**What exists:**
```python
async def _reconcile_balances(self, symbols: List[str]) -> Dict[str, int]:
    """Reconcile exchange balances with state positions."""
    
    balances = await self._get_exchange_balances()  # ✅ Fetches
    positions = await ss.get_open_positions()       # ✅ Gets positions
    
    for sym, pos in positions.items():
        state_qty = self._position_qty(pos)
        base_asset, _ = self._split_base_quote(sym)
        exchange_qty = balances.get(base_asset, {}).get("total", 0.0)
        
        # ✅ Dust filtering exists:
        if state_qty > self.dust_threshold and exchange_qty < self.dust_threshold:
            # Position exists in state but exchange has no balance → phantom!
            closed = await self._close_phantom_position(sym, pos, state_qty)
```

**Dust Threshold:**
- Line 83: `self.dust_threshold = float(self._cfg("DUST_POSITION_QTY", 0.00001) or 0.00001)`
- ✅ Dust threshold defined
- ✅ Used in reconciliation logic
- ❌ NOT used during initial balance → position creation

**What's missing:**
- ❌ Does NOT create positions from unused wallet assets
- ❌ Only validates existing positions, doesn't hydrate missing ones from balances
- ✅ Detects phantoms (positions without backing balances) but closes them, doesn't create

---

## 3. PortfolioManager: Asset Classification & Dust Detection ✅

**File:** `core/portfolio_manager.py`

**What exists:**
```python
STABLECOIN_1to1 = {"USDT", "FDUSD", "TUSD", "BUSD", "USDC"}

async def _is_dust(self, asset: str, amount: Decimal, price: Optional[Decimal]) -> bool:
    """Classify dust positions (economically irrelevant)."""
    if amount <= Decimal("0"):
        return True
    
    if asset in STABLECOIN_1to1:
        stbl_thr = self.dust_threshold_stables
        return amount < stbl_thr
    
    # For non-stables: compute notional value (qty * price)
    if price is None or price <= 0:
        return True  # No price = dust
    
    notional = amount * price
    min_notional = self._get_min_notional(asset)  # From exchange symbol info
    return notional < min_notional
```

**Dust Thresholds:**
- Stablecoins: Fixed threshold (e.g., 5.0 USDT)
- Non-stables: minNotional from exchange (e.g., 10.0 USDT notional)

**What's missing:**
- ❌ Does NOT create positions during startup from balances
- ✅ Used for position filtering/liquidation in portfolio management
- ⚠️ Different dust logic than RecoveryEngine (notional-based vs qty-based)

---

## 4. SharedState: NAV Calculation ✅

**File:** `core/shared_state.py` (lines 1057-1120)

**What exists:**
```python
async def get_nav_quote(self) -> float:
    """CRITICAL: Computes NAV from ALL positions.
    NAV = sum(all_quote_balances) + sum(all_positions_at_market_price)
    This is NOT filtered by MIN_ECONOMIC_TRADE_USDT or any trade floor."""
    
    nav = 0.0
    
    # Include free USDT
    free_usdt = float(self.free_quote or 0.0)
    nav += free_usdt
    
    # Include ALL positions (no dust filtering)
    positions = getattr(self, 'positions', {}) or {}
    for symbol, pos in positions.items():
        qty = float(pos.get('quantity', 0.0) or 0.0)
        price = float(pos.get('mark_price', pos.get('entry_price', 0.0)) or 0.0)
        if qty > 0 and price > 0:
            nav += qty * price  # ← ALL positions included
    
    return nav
```

**Status:**
- ✅ Correctly computes from ALL positions (no dust filtering)
- ✅ Includes free USDT
- ❌ Does NOT create positions from wallet balances (that's RecoveryEngine's job)

---

## What's Missing: The Gap

The unified pattern you're asking about would require:

### 1. **Balance-to-Position Hydration** ❌
Currently:
- RecoveryEngine fetches balances separately from positions
- Positions come from exchange (get_open_positions)
- Wallet assets that DON'T have open positions → **NOT hydrated as 0qty positions**

Should be:
```python
# After _load_live(), for each balance with qty > dust:
for asset, balance_info in balances.items():
    if asset == "USDT":
        continue  # Skip USDT (it's free capital, not a position)
    
    qty = balance_info["total"]
    if qty > MIN_ECONOMIC_TRADE_USDT or qty > dust_threshold:
        # Create position for this asset if not already exists
        if asset not in positions:
            positions[asset] = {
                "symbol": asset + "/USDT",  # or proper symbol format
                "quantity": qty,
                "entry_price": last_known_price[asset],
                "side": "LONG",
                "source": "wallet_hydration"
            }
```

### 2. **Dust Filtering Consistency** ⚠️
Three different definitions exist:
- **RecoveryEngine**: No dust filtering during load
- **ExchangeTruthAuditor**: `DUST_POSITION_QTY` (qty-based, default 0.00001)
- **PortfolioManager**: Stablecoin threshold + minNotional (notional-based)

Should consolidate to:
```python
MIN_ECONOMIC_TRADE_USDT = 30.0  # Single source of truth

def is_dust(asset: str, qty: float, price: float) -> bool:
    notional = qty * price
    return notional < MIN_ECONOMIC_TRADE_USDT
```

### 3. **Cross-Component Communication** ❌
Currently:
- RecoveryEngine loads balances
- ExchangeTruthAuditor verifies reconciliation
- PortfolioManager filters dust
- **No explicit handoff** to say "here are the validated, dust-filtered positions"

Should be:
```python
# Step 1: RecoveryEngine.rebuild_state()
# → fetches balances, creates positions from wallet

# Step 2: ExchangeTruthAuditor._restart_recovery()
# → validates balances match positions, closes phantoms

# Step 3: PortfolioManager._filter_dust()
# → marks dust positions for liquidation

# Step 4: NAV = sum(non_dust_positions) + free_usdt
```

---

## Current Startup Flow (Actual)

```
StartupOrchestrator._step_recovery_engine_rebuild()
  ↓
RecoveryEngine.rebuild_state()
  ├─ _load_live() → get_balances() ✅
  ├─ _normalize_balances() ✅
  ├─ _apply_balances() → SharedState.update_balances() ✅
  ├─ _load_live() → get_open_positions() ✅
  ├─ _apply_positions() → SharedState.update_position() ✅
  └─ No dust filtering ❌

  ↓
StartupOrchestrator._step_auditor_restart_recovery()
  ↓
ExchangeTruthAuditor._restart_recovery()
  ├─ _reconcile_balances()
  │  ├─ Compare state positions vs exchange balances
  │  ├─ Dust threshold check: 0.00001 qty
  │  └─ Close phantom positions (no position in exchange)
  └─ No hydration from unused wallet assets ❌

  ↓
StartupOrchestrator._step_verify_startup_integrity()
  ├─ Filter positions < MIN_ECONOMIC_TRADE_USDT (30.0) ✅
  └─ NAV = sum(viable_positions) + free_quote ✅
```

---

## Correct Institutional Architecture

### Responsibility Boundaries (Corrected)

1. **RecoveryEngine** → State Loader (dumb, raw load only)
   - Fetch balances from exchange
   - Fetch positions from exchange
   - NO processing, NO hydration
   - Just load raw state

2. **ExchangeTruthAuditor** → State Validator & Hydrator
   - Reconcile balances vs positions
   - Close phantom positions
   - **Hydrate missing positions from wallet balances** ← CORRECT LOCATION
   - Use unified dust threshold

3. **PortfolioManager** → Economic Classifier
   - Classify viable vs dust positions
   - Manage liquidation logic
   - Use shared MIN_ECONOMIC_TRADE_USDT

4. **SharedState** → Calculation Engine
   - Compute NAV from all positions
   - Calculate metrics
   - No filtering (trust upstream validation)

5. **StartupOrchestrator** → Orchestration Gate
   - Verify capital integrity
   - Check phase readiness
   - Signal MetaController

---

## Correct Implementation: TruthAuditor Hydration

**Location:** `ExchangeTruthAuditor._hydrate_missing_positions()`

```python
async def _hydrate_missing_positions(
    self, 
    balances: Dict[str, Any], 
    state_positions: Dict[str, Any]
) -> Dict[str, int]:
    """
    Create positions for wallet assets without open orders.
    Bridges the gap between wallet balances and state positions.
    
    This is where the unified pattern LIVES:
      1. Loop balances
      2. Skip USDT (it's free capital)
      3. Skip if qty*price < MIN_ECONOMIC_TRADE_USDT (dust)
      4. Create if missing from state
    """
    stats = {"positions_hydrated": 0, "assets_skipped_dust": 0}
    
    # Single source of truth for dust threshold
    dust_threshold_usdt = float(
        getattr(self.config, 'MIN_ECONOMIC_TRADE_USDT', 30.0) or 30.0
    )
    
    for asset, balance_info in balances.items():
        # Skip quote currency (it's free capital, not a position)
        if asset.upper() == "USDT":
            continue
        
        # Get balance quantity
        qty = float(balance_info.get('total', 0.0) or 0.0)
        if qty <= 0:
            continue
        
        # Get market price for this asset
        symbol = f"{asset}USDT"
        price = await self._get_market_price(symbol)
        if price is None or price <= 0:
            price = 1.0  # Fallback
        
        # Check notional value (qty * price) against dust threshold
        notional = qty * price
        if notional < dust_threshold_usdt:
            stats["assets_skipped_dust"] += 1
            self.logger.debug(
                f"[TruthAuditor:Hydrate] {asset}: notional={notional:.2f} "
                f"< threshold={dust_threshold_usdt:.2f}, skipped (dust)"
            )
            continue
        
        # Check if position already exists in state
        if symbol in state_positions:
            self.logger.debug(
                f"[TruthAuditor:Hydrate] {asset}: position already exists"
            )
            continue
        
        # Hydrate: create position from wallet balance
        await self.shared_state.create_position({
            "symbol": symbol,
            "quantity": qty,
            "entry_price": price,
            "side": "LONG",
            "source": "wallet_hydration",
            "timestamp": time.time(),
        })
        
        stats["positions_hydrated"] += 1
        self.logger.info(
            f"[TruthAuditor:Hydrate] Created position: {symbol} qty={qty} "
            f"price={price} notional={notional:.2f}"
        )
    
    return stats
```

---

## Unified Dust Model (Corrected)

### Single Source of Truth

**Config setting:**
```python
class Config:
    MIN_ECONOMIC_TRADE_USDT = 30.0  # Notional threshold
```

### Everywhere else:
```python
def is_dust(asset: str, qty: float, price: float, config) -> bool:
    """
    Unified dust classification.
    An asset is dust if its notional value (qty * price) is below threshold.
    """
    if qty <= 0:
        return True
    
    if price is None or price <= 0:
        return True  # Can't value it → treat as dust
    
    notional = qty * price
    threshold = getattr(config, 'MIN_ECONOMIC_TRADE_USDT', 30.0)
    
    return notional < threshold
```

### Applied in:

1. **ExchangeTruthAuditor._hydrate_missing_positions()**
   - Use notional check before creating position

2. **PortfolioManager._is_dust()**
   - Replace current mixed logic with unified check

3. **StartupOrchestrator._step_verify_startup_integrity()**
   - Filter viable positions using notional check

4. **Remove TruthAuditor.dust_threshold**
   - No more `DUST_POSITION_QTY` (qty-based)
   - Replace with config.MIN_ECONOMIC_TRADE_USDT

---

## Correct Startup Pipeline (Refined)

```
Phase 8.5: StartupOrchestrator

Step 1: RecoveryEngine.rebuild_state()
  ├─ _load_live() → exchange.get_balances() ✅
  ├─ _normalize_balances() ✅
  ├─ _apply_balances() → SharedState ✅
  ├─ _load_live() → exchange.get_open_positions() ✅
  ├─ _apply_positions() → SharedState ✅
  └─ NO PROCESSING (dumb loader) ✅

Step 2: ExchangeTruthAuditor._restart_recovery()
  ├─ _reconcile_balances()
  │  └─ Close phantom positions
  ├─ _hydrate_missing_positions() ← FIX: ADD THIS
  │  ├─ Loop balances
  │  ├─ Skip USDT
  │  ├─ Skip notional < MIN_ECONOMIC_TRADE_USDT
  │  └─ Create position if missing
  ├─ _reconcile_orders()
  ├─ _reconcile_trades()
  └─ Emit TRUTH_AUDIT_RESTART_SYNC

Step 3: PortfolioManager.refresh_positions()
  └─ Update position metadata (non-fatal)

Step 4: SharedState (implicit)
  └─ NAV = sum(all_positions) + free_usdt
     (uses positions hydrated in Step 2)

Step 5: StartupOrchestrator._step_verify_startup_integrity()
  ├─ Get viable positions (notional >= threshold)
  ├─ Check capital balance
  ├─ Verify NAV consistency
  └─ Allow NAV=0 if shadow_mode OR no viable positions

Step 6: Emit StartupPortfolioReady → MetaController starts
```

---

## Summary: Current vs Corrected

### Current Implementation (Broken)

| Step | Component | Status | Problem |
|------|-----------|--------|---------|
| 1 | RecoveryEngine | ✅ Loads | No hydration |
| 2 | TruthAuditor | ⚠️ Validates | Closes phantoms only, doesn't hydrate |
| 3 | PortfolioManager | ⚠️ Classifies | Multiple dust definitions |
| 4 | SharedState | ✅ Calculates | Works on incomplete position set |
| 5 | StartupOrchestrator | ❌ Fails | Startup fails because NAV=0 with missing positions |

**Root cause:** Wallet assets without open orders never become positions → they're invisible to NAV → startup integrity check fails

---

### Corrected Implementation (Unified)

| Step | Component | Logic | Dust Model |
|------|-----------|-------|-----------|
| 1 | **RecoveryEngine** | Load raw state (dumb) | N/A |
| 2 | **TruthAuditor** | **Validate + Hydrate** (NEW) | notional < 30 USDT |
| 3 | **PortfolioManager** | Classify viable/dust | notional < 30 USDT |
| 4 | **SharedState** | Calculate NAV | Uses all positions |
| 5 | **StartupOrchestrator** | Verify integrity | Works on complete position set |

**Fix:** Add `_hydrate_missing_positions()` to TruthAuditor after balance reconciliation

**Dust unification:** Replace 3 definitions with single `MIN_ECONOMIC_TRADE_USDT = 30.0`

---

## Implementation Checklist

- [ ] **Step 1:** Add `_hydrate_missing_positions()` method to ExchangeTruthAuditor
- [ ] **Step 2:** Call hydration in `_restart_recovery()` after `_reconcile_balances()`
- [ ] **Step 3:** Add `MIN_ECONOMIC_TRADE_USDT` to config (if not present)
- [ ] **Step 4:** Update `_reconcile_balances()` to use notional-based dust filter
- [ ] **Step 5:** Update PortfolioManager._is_dust() to use config threshold
- [ ] **Step 6:** Update StartupOrchestrator position filtering to use config threshold
- [ ] **Step 7:** Remove `DUST_POSITION_QTY` from TruthAuditor (replace with config)
- [ ] **Step 8:** Test startup with wallet assets but no open orders

