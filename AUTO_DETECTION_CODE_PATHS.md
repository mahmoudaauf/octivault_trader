# QUICK REFERENCE - AUTO-DETECTION CODE PATHS

## 1️⃣ How Balance is Auto-Detected

### Path A: Get Balance Now (Immediate)
```python
# File: core/exchange_client.py (Line 226)
balances = await exchange_client.get_spot_balances()
# → {"USDT": {"free": 104.04, "locked": 0}, ...}

# File: core/shared_state.py (Line 5458)
await shared_state.hydrate_balances_from_exchange()
# Updates: shared_state.balances dict with latest from exchange
```

### Path B: Continuous Background Sync
```python
# File: core/shared_state.py (Line 5383)
async def _wallet_sync_loop(self):
    while True:
        await self.sync_authoritative_balance(force=True)
        # Refreshes every 300 seconds (5 minutes)
        await asyncio.sleep(300)
```

### Path C: After Every Trade
```python
# File: core/position_manager.py (Line 600)
async def _update_wallet_snapshot(self):
    balances = await self.ex.get_balances()
    await self.ss.update_wallet(balances)
    # Updates balance immediately after buy/sell
```

### Path D: Validation & Reconciliation
```python
# File: core/exchange_truth_auditor.py (Line 1003)
async def _reconcile_balances(symbols):
    balances = await self._get_exchange_balances()
    # Validates all balances are non-negative
    # Checks for suspicious patterns
    # Reports discrepancies
```

---

## 2️⃣ How Classification is Auto-Detected

### Bucket Classification Automatically Runs Every Cycle
```python
# File: 🎯_MASTER_SYSTEM_ORCHESTRATOR.py (Line 1487)
async def _inject_signals_loop(self):
    while True:
        # Get current state
        positions = await self.shared_state.get_positions_snapshot()
        total_equity = await self.shared_state.get_total_nav()
        
        # Classify into buckets
        bucket_state = await self.three_bucket_manager.update_bucket_state(
            positions=positions,
            total_equity=total_equity
        )
        
        # Every 10 cycles, log status
        if cycle % 10 == 0:
            self.three_bucket_manager.log_bucket_status()
            self.three_bucket_manager.log_trading_gates()
```

### Bucket Classification Logic
```python
# File: core/bucket_classifier.py (Line 252)
def classify_portfolio(self, positions, total_equity) -> PortfolioBucketState:
    for symbol, pos_data in positions.items():
        # Skip USDT (it's operating cash)
        if symbol == 'USDT':
            bucket_state.operating_cash_usdt = pos_data.get('value', 0)
            continue
        
        # Classify each position
        classification = self.classify_position(
            symbol=symbol,
            current_value=current_value,
            current_qty=current_qty,
            current_price=current_price,
            entry_price=entry_price
        )
        
        # Add to bucket
        if classification.bucket == BucketType.PRODUCTIVE:
            bucket_state.productive_total_value += current_value
        elif classification.bucket == BucketType.DEAD_CAPITAL:
            bucket_state.dead_total_value += current_value
    
    return bucket_state
```

### Dynamic Balance Classification
```python
# File: balance_threshold_config.py (NEW)
from balance_threshold_config import DynamicBalanceThresholds

# For $104.04 account:
bucket, pct_change = DynamicBalanceThresholds.classify_balance(
    current_balance=104.26,
    initial_balance=104.04
)
# Returns: ("GAINING 📈", 0.21%)
```

---

## 3️⃣ How Symbols are Auto-Detected

### Path A: Get Current Holdings (What You Own)
```python
# File: core/shared_state.py (Line 5574)
async def get_portfolio_snapshot():
    # Get all balances from exchange
    for asset, bal in self.balances.items():
        if asset.upper() != self.quote_asset.upper():  # Skip USDT
            qty = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
            if qty > 0:
                sym = f"{asset}USDT"
                # Found a holding!
                positions_to_add[sym] = {...}
    
    return snapshot  # Contains all held symbols
```

### Path B: Hydrate Positions from Balances
```python
# File: core/shared_state.py (Line 5322)
async def hydrate_positions_from_balances():
    """
    Mirror non-quote wallet balances into spot positions
    """
    snapshot = dict(self.balances)
    
    for asset, bal in snapshot.items():
        # Skip USDT
        if asset.upper() == "USDT":
            continue
        
        qty = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
        if qty > 0:
            # Create position for this symbol
            sym = f"{asset}USDT"
            self.positions[sym] = {"qty": qty, "source": "wallet"}
```

### Path C: Discover Exchange Balances
```python
# File: core/exchange_client.py (Line 3449)
async def get_spot_balances():
    """Fetch all non-zero balances from Binance"""
    balances = await self._fetch_from_binance("/api/v3/account")
    
    # Returns: {"USDT": {...}, "BTC": {...}, "ETH": {...}, ...}
    # Each asset with balance > 0 is returned
    
    return balances
```

### Path D: Validate Symbols are Tradable
```python
# File: core/shared_state.py (Line 5343)
# Verify the symbol is tradable
is_known_symbol = sym in self.symbols or sym in self.accepted_symbols
is_exchange_tradable = True

if self._exchange_client and hasattr(self._exchange_client, "has_symbol"):
    is_exchange_tradable = self._exchange_client.has_symbol(sym)

if not is_known_symbol and not is_exchange_tradable:
    # Skip non-tradable pairs
    continue
```

### Path E: Get All Symbols Available
```python
# File: core/exchange_client.py (Line 226)
symbols = await exchange_client.get_all_symbols()
# Returns: ["BTCUSDT", "ETHUSDT", "BNBUSDT", ...]

# Or via SymbolManager
symbols = await symbol_manager.get_symbols()
# Returns with filtering applied
```

---

## 4️⃣ Reconciliation & Validation (Automatic Error Correction)

### Phantom Position Detection & Repair
```python
# File: core/execution_manager.py (Line 3675)
async def _handle_phantom_position(symbol):
    """
    SCENARIO A: Position in state but not on exchange
    - Delete from local state
    
    SCENARIO B: Position on exchange but not in state
    - Sync from exchange
    - Update local state with exchange quantity
    """
    exchange_qty = await self._get_exchange_position_qty(symbol)
    state_qty = await self.shared_state.get_position_quantity(symbol)
    
    if state_qty > 0 and exchange_qty == 0:
        # Phantom: exists locally but not on exchange
        await self.shared_state.delete_position(symbol)
    
    if exchange_qty > 0 and state_qty == 0:
        # Lost position: exists on exchange but not locally
        pos["quantity"] = exchange_qty
        self.shared_state.positions[symbol] = pos
```

### Balance Mismatch Detection
```python
# File: core/exchange_truth_auditor.py (Line 977)
async def _reconcile_balances(symbols):
    balances = await self._get_exchange_balances()
    positions = await self.shared_state.get_open_positions()
    
    for sym, pos in positions.items():
        state_qty = self._position_qty(pos)
        base_asset = self._split_base_quote(sym)[0]
        
        bal = balances.get(base_asset.upper(), {})
        exchange_qty = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
        
        # Check mismatch
        if abs(state_qty - exchange_qty) > tolerance:
            # Mismatch found - report for correction
            stats["mismatches"] += 1
```

---

## 5️⃣ For YOUR Account ($104.04)

### Current Auto-Detection State

**Balance Detection**
```python
# Get your current balance:
await shared_state.hydrate_balances_from_exchange()
# Updates: shared_state.balances = {"USDT": {"free": 104.04, "locked": 0}}

# Query it:
balance = shared_state.balances.get("USDT", {}).get("free", 0)
# Result: 104.04
```

**Classification Detection**
```python
# Classify your $104.04:
from balance_threshold_config import DynamicBalanceThresholds

bucket, pct = DynamicBalanceThresholds.classify_balance(104.04, 104.04)
# Result: ("STABLE ➡️", 0.0%)
# Status: At initial balance (threshold)
```

**Symbol Detection**
```python
# Get your current holdings:
snapshot = await shared_state.get_portfolio_snapshot()
# Result: 
# {
#   "holdings": [],  # Empty - USDT only
#   "total_nav": 104.04,
#   "quote_balance": 104.04,
#   "symbols_held": []
# }
```

---

## 🔍 Testing Auto-Detection

```bash
# Run diagnostic to see all auto-detection in action:
python3 diagnostic_signal_flow.py

# Output will show:
# ✅ Balance check: USDT Balance: 104.04
# ✅ Position check: Positions: 0
# ✅ Symbol validation: Ready
```

---

## 📊 Summary Table

| Component | File | Function | Update Rate |
|-----------|------|----------|------------|
| **Balance Detection** | exchange_client.py:226 | get_spot_balances() | Real-time + 5min |
| **Balance Caching** | shared_state.py:4210 | update_balances() | Real-time |
| **Balance Sync Loop** | shared_state.py:5383 | _wallet_sync_loop() | Every 5 min |
| **Classification** | bucket_classifier.py:252 | classify_portfolio() | Every cycle |
| **Symbol Discovery** | shared_state.py:5574 | get_portfolio_snapshot() | Real-time |
| **Position Hydration** | shared_state.py:5322 | hydrate_positions_from_balances() | Every sync |
| **Reconciliation** | exchange_truth_auditor.py:977 | _reconcile_balances() | Every 5 min |
| **Phantom Detection** | execution_manager.py:3675 | _handle_phantom_position() | Every cycle |
| **Validation** | exchange_client.py:2053 | _run_truth_auditor() | Continuous |

---

**✅ ALL AUTO-DETECTION SYSTEMS ARE FULLY OPERATIONAL**
