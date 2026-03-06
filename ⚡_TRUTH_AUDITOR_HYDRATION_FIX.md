# TruthAuditor Hydration Fix: Institutional Architecture

## Problem Statement

**Current startup failure root cause:**
1. RecoveryEngine loads wallet balances: `{BTC: 0.5, ETH: 2.0, USDT: 1000}`
2. RecoveryEngine loads positions: `{}` (no open orders)
3. SharedState has balances but NO positions for BTC/ETH
4. NAV calculation: only free USDT + empty positions = 0 (WRONG!)
5. StartupOrchestrator: "NAV=0 but wallet has assets!" → FAIL

**Missing link:** Wallet assets → Positions

**Correct location:** ExchangeTruthAuditor (state validator), not RecoveryEngine (state loader)

---

## Architecture: Clean Responsibility Boundaries

```
RecoveryEngine
  └─ Job: Load raw state from exchange (dumb)
     ├─ Fetch balances
     ├─ Fetch positions
     └─ NO processing

TruthAuditor ← FIX HERE
  └─ Job: Validate & hydrate state
     ├─ Reconcile: balances vs positions
     ├─ Close: phantom positions
     ├─ Hydrate: missing positions from balances ← ADD THIS
     └─ Use: unified dust threshold

PortfolioManager
  └─ Job: Classify & manage positions
     ├─ Classify: viable vs dust
     ├─ Liquidate: dust positions
     └─ Rebalance: portfolio

SharedState
  └─ Job: Calculate metrics (read-only)
     ├─ NAV = Σ(positions) + free_capital
     ├─ Metrics = PnL, leverage, etc.
     └─ No filtering (trust upstream)

StartupOrchestrator
  └─ Job: Orchestrate & verify gates
     ├─ Call: RecoveryEngine
     ├─ Call: TruthAuditor
     ├─ Call: PortfolioManager
     ├─ Verify: capital integrity
     └─ Signal: MetaController (ready)
```

---

## Implementation: ExchangeTruthAuditor._hydrate_missing_positions()

### Location
File: `core/exchange_truth_auditor.py`

After `_reconcile_balances()` method, add this new method:

```python
async def _hydrate_missing_positions(
    self,
    balances: Dict[str, Any],
    state_positions: Dict[str, Any]
) -> Dict[str, int]:
    """
    Hydrate missing positions from wallet balances.
    
    This closes the gap between exchange balances and state positions:
      - Exchange has asset in wallet: BTC 0.5
      - State has NO position for BTC
      - Solution: Create position from wallet balance
    
    Uses unified dust threshold: MIN_ECONOMIC_TRADE_USDT
    
    Returns:
        stats dict with counts of hydrated positions and skipped dust
    """
    stats = {
        "positions_hydrated": 0,
        "assets_skipped_dust": 0,
        "assets_skipped_already_exists": 0,
    }
    
    ss = self.shared_state
    if ss is None:
        return stats
    
    # === STEP 1: Get unified dust threshold from config ===
    dust_threshold_usdt = float(
        getattr(self.config, 'MIN_ECONOMIC_TRADE_USDT', 30.0) or 30.0
    )
    
    # === STEP 2: Loop through wallet balances ===
    for asset, balance_info in (balances or {}).items():
        asset = str(asset).upper()
        
        # SKIP 1: Quote currency (USDT is free capital, not a position)
        if asset == "USDT":
            continue
        
        # SKIP 2: Zero or negative balance
        qty = float(balance_info.get('total', 0.0) or 0.0)
        if qty <= 0:
            continue
        
        # === STEP 3: Fetch market price ===
        symbol = f"{asset}USDT"
        price = None
        
        try:
            # Try TruthAuditor's price cache first
            if hasattr(self, 'latest_prices') and self.latest_prices:
                price = float(self.latest_prices.get(symbol, 0.0) or 0.0)
        except Exception:
            pass
        
        if not price or price <= 0:
            try:
                # Fallback: Query exchange
                if self.exchange_client and hasattr(self.exchange_client, 'get_current_price'):
                    price = await self._maybe_await(
                        self.exchange_client.get_current_price(symbol)
                    )
                    price = float(price or 0.0)
            except Exception:
                pass
        
        # Fallback: Use 1.0 (neutral) if no price available
        if not price or price <= 0:
            price = 1.0
        
        # === STEP 4: Check notional value (qty * price) ===
        notional = qty * price
        
        # SKIP 3: Dust positions (notional < threshold)
        if notional < dust_threshold_usdt:
            stats["assets_skipped_dust"] += 1
            self.logger.info(
                f"[TruthAuditor:Hydrate] Skipped {asset}: "
                f"notional={notional:.2f} USDT < "
                f"threshold={dust_threshold_usdt:.2f} USDT (dust)"
            )
            continue
        
        # === STEP 5: Check if position already exists ===
        if symbol in (state_positions or {}):
            stats["assets_skipped_already_exists"] += 1
            self.logger.debug(
                f"[TruthAuditor:Hydrate] Skipped {asset}: "
                f"position {symbol} already exists in state"
            )
            continue
        
        # === STEP 6: Create position from wallet balance ===
        try:
            position_data = {
                "symbol": symbol,
                "quantity": qty,
                "entry_price": price,
                "side": "LONG",
                "source": "wallet_hydration",
                "timestamp": time.time(),
                "mark_price": price,
            }
            
            # Use SharedState's position creation method if available
            if hasattr(ss, 'create_position'):
                await self._maybe_await(ss.create_position(symbol, position_data))
            elif hasattr(ss, 'update_position'):
                await self._maybe_await(ss.update_position(symbol, position_data))
            else:
                # Fallback: Direct dict update
                positions = getattr(ss, 'positions', None) or {}
                if isinstance(positions, dict):
                    positions[symbol] = position_data
            
            stats["positions_hydrated"] += 1
            self.logger.info(
                f"[TruthAuditor:Hydrate] ✅ Hydrated: {symbol} "
                f"qty={qty:.8f} price={price:.2f} USDT "
                f"notional={notional:.2f} USDT"
            )
            
            # Emit event for monitoring
            await self._emit_event("POSITION_HYDRATED", {
                "symbol": symbol,
                "quantity": qty,
                "entry_price": price,
                "notional": notional,
                "source": "wallet_hydration",
                "timestamp": time.time(),
            })
        
        except Exception as e:
            self.logger.warning(
                f"[TruthAuditor:Hydrate] Failed to hydrate {asset}: {e}",
                exc_info=True
            )
    
    # === STEP 7: Log summary ===
    self.logger.info(
        f"[TruthAuditor:Hydrate] Summary: "
        f"hydrated={stats['positions_hydrated']} "
        f"dust_skipped={stats['assets_skipped_dust']} "
        f"already_exist={stats['assets_skipped_already_exists']}"
    )
    
    return stats
```

### Integration: Call hydration in _restart_recovery()

**Location:** `core/exchange_truth_auditor.py` → `_restart_recovery()` method

**Current code (lines ~565-595):**
```python
async def _restart_recovery(self) -> None:
    symbols = await self._collect_symbols()
    if len(symbols) > 0 and self.max_symbols_per_cycle > 0:
        self._cursor = 0

    fills = await self._reconcile_orders(symbols=symbols, startup=True)
    trades = await self._reconcile_trades(symbols=symbols, startup=True)
    balances = await self._reconcile_balances(symbols=symbols)  ← Returns only stats
    orders = await self._reconcile_open_orders(symbols=symbols)
    sell_map = await self._validate_sell_finalize_mapping(startup=True)
```

**Problem:** `_reconcile_balances()` reconciles but doesn't return balances dict → can't pass to hydration

**Fix:** Modify `_reconcile_balances()` to ALSO return balances dict

---

### Modification A: Update _reconcile_balances() signature

**Current (lines 937-985):**
```python
async def _reconcile_balances(self, symbols: List[str]) -> Dict[str, int]:
    stats = {"phantoms_closed": 0, "mismatches": 0}
    ss = self.shared_state
    if ss is None:
        return stats

    balances = await self._get_exchange_balances()  # ← Fetches balances
    # ... reconciliation logic ...
    return stats  # ← Only returns stats, loses balances!
```

**Fixed:**
```python
async def _reconcile_balances(self, symbols: List[str]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Reconcile exchange balances with state positions.
    
    Returns:
        (stats_dict, balances_dict) - stats for logging, balances for hydration
    """
    stats = {"phantoms_closed": 0, "mismatches": 0}
    ss = self.shared_state
    if ss is None:
        return stats, {}

    balances = await self._get_exchange_balances()
    if not balances:
        return stats, {}

    # ... existing reconciliation logic (unchanged) ...
    
    return stats, balances  # ← Return both!
```

### Modification B: Update _restart_recovery() to use hydration

**Add after balance reconciliation:**
```python
async def _restart_recovery(self) -> None:
    symbols = await self._collect_symbols()
    if len(symbols) > 0 and self.max_symbols_per_cycle > 0:
        self._cursor = 0

    fills = await self._reconcile_orders(symbols=symbols, startup=True)
    trades = await self._reconcile_trades(symbols=symbols, startup=True)
    
    # MODIFIED: Capture balances returned from reconciliation
    balance_stats, balances = await self._reconcile_balances(symbols=symbols)
    
    # NEW: Hydrate missing positions from wallet balances
    state_positions = await self._get_state_positions() if hasattr(self, '_get_state_positions') else {}
    hydrate_stats = await self._hydrate_missing_positions(balances, state_positions)
    
    orders = await self._reconcile_open_orders(symbols=symbols)
    sell_map = await self._validate_sell_finalize_mapping(startup=True)

    # Update event with hydration results
    await self._emit_event(
        "TRUTH_AUDIT_RESTART_SYNC",
        {
            "symbols": len(symbols),
            "fills_recovered": fills.get("fills_recovered", 0),
            "trades_recovered": trades.get("trades_recovered", 0),
            "trades_sell_finalized": trades.get("trades_sell_finalized", 0),
            "phantoms_closed": balance_stats.get("phantoms_closed", 0),
            "positions_hydrated": hydrate_stats.get("positions_hydrated", 0),  # ← NEW
            "assets_skipped_dust": hydrate_stats.get("assets_skipped_dust", 0),  # ← NEW
            "open_order_mismatch": orders.get("open_order_mismatch", 0),
            "sell_missing_canonical": fills.get("sell_missing_canonical", 0),
            "sell_finalize_fills_seen": sell_map.get("sell_finalize_fills_seen", 0),
            "sell_finalize_finalized": sell_map.get("sell_finalize_finalized", 0),
            "sell_finalize_pending": sell_map.get("sell_finalize_pending", 0),
            "sell_finalize_gap": sell_map.get("sell_finalize_gap", 0),
            "ts": time.time(),
        },
    )
```

### Helper: Get state positions

**Add this helper if not present:**
```python
async def _get_state_positions(self) -> Dict[str, Any]:
    """Get current positions from SharedState."""
    if self.shared_state is None:
        return {}
    
    try:
        if hasattr(self.shared_state, 'get_open_positions'):
            positions = self.shared_state.get_open_positions()
            if hasattr(positions, '__await__'):
                positions = await positions
            return positions or {}
    except Exception:
        pass
    
    # Fallback
    return getattr(self.shared_state, 'positions', {}) or {}
```

---

## Unified Dust Threshold: Config Updates

### Step 1: Add to config (if not present)

**File:** `config.py` or wherever config is defined

```python
class Config:
    # ... existing settings ...
    
    # === Unified dust threshold (new) ===
    # An asset is "dust" if its notional value (qty * price) 
    # is below this USDT threshold
    MIN_ECONOMIC_TRADE_USDT = 30.0  # Notional value in USDT
```

### Step 2: Update StartupOrchestrator to use config

**File:** `core/startup_orchestrator.py` → `_step_verify_startup_integrity()`

**Current (lines 425-475):**
```python
min_economic_trade = float(
    getattr(self.shared_state.config, 'MIN_ECONOMIC_TRADE_USDT', 30.0)
    if hasattr(self.shared_state, 'config')
    else 30.0
)
```

**This is already correct!** ✅ No change needed.

### Step 3: Update TruthAuditor to use config

**File:** `core/exchange_truth_auditor.py` → `__init__()`

**Remove:**
```python
# OLD: qty-based dust threshold
self.dust_threshold = float(self._cfg("DUST_POSITION_QTY", 0.00001) or 0.00001)
```

**Replace with:**
```python
# NEW: notional-based unified threshold
self.min_economic_trade_usdt = float(
    getattr(self.config, 'MIN_ECONOMIC_TRADE_USDT', 30.0) or 30.0
)
```

**Update all references:**
```python
# OLD: if state_qty > self.dust_threshold
# NEW: if notional > self.min_economic_trade_usdt
```

### Step 4: Update PortfolioManager to use config

**File:** `core/portfolio_manager.py` → `_is_dust()`

**Current logic:** Mixed stablecoin + notional

**Simplify to:**
```python
async def _is_dust(self, asset: str, amount: Decimal, price: Optional[Decimal]) -> bool:
    """
    Unified dust classification.
    An asset is dust if its notional value < MIN_ECONOMIC_TRADE_USDT.
    """
    if amount is None or amount <= Decimal("0"):
        return True
    
    if price is None or price <= Decimal("0"):
        return True  # Can't value → dust
    
    notional = amount * price
    threshold = Decimal(str(
        getattr(self.config, 'MIN_ECONOMIC_TRADE_USDT', 30.0)
    ))
    
    return notional < threshold
```

---

## Verification Checklist

- [ ] **TruthAuditor: Add method** `_hydrate_missing_positions()`
- [ ] **TruthAuditor: Update return** `_reconcile_balances()` to include balances dict
- [ ] **TruthAuditor: Call hydration** in `_restart_recovery()`
- [ ] **TruthAuditor: Remove** `self.dust_threshold` (DUST_POSITION_QTY)
- [ ] **TruthAuditor: Add** `self.min_economic_trade_usdt`
- [ ] **Config: Add** `MIN_ECONOMIC_TRADE_USDT = 30.0`
- [ ] **StartupOrchestrator: Verify** already uses config (no change)
- [ ] **PortfolioManager: Update** `_is_dust()` to use notional
- [ ] **Test startup:** with wallet assets but no open orders
  - Verify positions are hydrated
  - Verify NAV is non-zero
  - Verify startup completes

---

## Expected Behavior After Fix

### Startup with wallet assets:

**Exchange state:**
```
Balances: {BTC: 0.5, ETH: 2.0, USDT: 1000}
Open positions: {} (no open orders)
```

**RecoveryEngine._load_live():**
```
Loads: balances={BTC: 0.5, ETH: 2.0, USDT: 1000}
       positions={}
```

**TruthAuditor._hydrate_missing_positions() (NEW):**
```
BTC: notional = 0.5 * 65000 = 32500 USDT → CREATE POSITION ✅
ETH: notional = 2.0 * 3500 = 7000 USDT → CREATE POSITION ✅
(Assuming MIN_ECONOMIC_TRADE_USDT = 30.0)

Result:
positions = {
    BTCUSDT: {qty: 0.5, entry_price: 65000, source: "wallet_hydration"},
    ETHUSDT: {qty: 2.0, entry_price: 3500, source: "wallet_hydration"},
}
```

**SharedState.get_nav_quote():**
```
NAV = free_usdt + Σ(positions)
    = 1000 + (0.5 * 65000) + (2.0 * 3500)
    = 1000 + 32500 + 7000
    = 40500 USDT ✅ (Non-zero!)
```

**StartupOrchestrator._step_verify_startup_integrity():**
```
Checks: NAV=40500 (non-zero) ✅
        free=1000 >= 0 ✅
        positions are viable ✅
Result: PASS → MetaController starts ✅
```

---

## Institutional Benefits

1. **Clean separation of concerns**
   - RecoveryEngine: Load only
   - TruthAuditor: Validate + hydrate
   - PortfolioManager: Classify
   - SharedState: Calculate

2. **Single source of truth for dust**
   - One config setting: `MIN_ECONOMIC_TRADE_USDT`
   - Used everywhere: TruthAuditor, PortfolioManager, StartupOrchestrator

3. **Correct startup flow**
   - RecoveryEngine loads raw state
   - TruthAuditor hydrates from wallet → NAV becomes non-zero
   - StartupOrchestrator verifies on complete state → passes

4. **Auditable & debuggable**
   - Each component has clear input/output
   - Hydration events logged for troubleshooting
   - Clean error handling at each stage
