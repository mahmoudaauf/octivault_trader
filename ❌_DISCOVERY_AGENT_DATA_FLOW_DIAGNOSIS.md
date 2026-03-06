# ❌ Discovery Agent Data Flow Diagnosis: Why Better Symbols Aren't Being Selected

## The Problem: You Are Correct! ✓

Yes, your architecture includes three discovery agents that **DISCOVER** great symbols, but there's a **critical gap in the data flow** that prevents those discoveries from being integrated into the active trading set.

---

## Architecture Overview

```
Discovery Agents (Find symbols)
    ├─ WalletScannerAgent     (finds assets you own → convert to symbols)
    ├─ SymbolScreener          (finds high-volatility liquid symbols)
    └─ IPOChaser               (finds newly listed symbols)
           ↓
        ⚠️ PROPOSE SYMBOLS
           ↓
    SymbolManager (validate & accept)
           ↓
    SharedState.accepted_symbols (CANONICAL STORE)
           ↓
    MetaController (reads these for trading)
```

---

## The Data Flow Breakdown

### Phase 1: Discovery Agents Discover Symbols ✓ (WORKING)

**WalletScannerAgent (`wallet_scanner_agent.py` lines 329-396):**
```python
# Finds symbols from wallet balances
candidates = []
for base, _bal in filtered.items():
    symbol = f"{base}{quote}"
    if await self._is_tradable_symbol(symbol):
        candidates.append(symbol)

# PROPOSES BATCH:
if candidates and self.symbol_manager:
    accepted_list = await self.symbol_manager.propose_symbols(
        candidates, 
        source=self.name
    )
    # ✓ Proposals made
```

**SymbolScreener (`symbol_screener.py` lines 304-388):**
```python
# Finds top liquid symbols by 24h volume
tickers = await self.exchange_client.get_24hr_tickers()
candidates = [
    {
        "symbol": "ETHUSDT",
        "quote_volume": 500000000,
        "atr_pct": 3.5,
        ...
    },
    ...
]

# PROPOSES INDIVIDUALLY:
for item in candidates:
    accepted_flag = await self._propose(
        symbol,
        source=self.name,
        metadata={...}
    )
```

**IPOChaser (`ipo_chaser.py` lines 140-165):**
```python
# Finds newly listed symbols
listings = await self.exchange_client.get_new_listings()
usdt_pairs = [s for s in listings if s.endswith("USDT")]

# PROPOSES INDIVIDUALLY:
for sym in usdt_pairs:
    res = await self.symbol_manager.propose_symbol(
        sym, 
        source=self.name,
        metadata={"reason": "IPO candidate"}
    )
```

**Status:** ✓ All agents are **FINDING** symbols correctly.

---

### Phase 2: Proposals → SymbolManager ⚠️ (PARTIALLY WORKING)

**SymbolManager.propose_symbol() (`symbol_manager.py` lines 510-535):**

```python
async def propose_symbol(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str]]:
    # 1. CHECK if accepting new symbols
    if not getattr(self, "_accept_new", True):
        logger.info("Discovery.accept_new_symbols is False; rejecting proposal %s", symbol)
        return False, "discovery disabled"  # ⚠️ REJECTION POINT 1
    
    # 2. CHECK if symbol already exists
    snap = await self._get_symbols_snapshot()
    if symbol in snap:
        logger.info("⚠️ %s already exists; skipping proposal.", symbol)
        return False, "already exists"  # ⚠️ REJECTION POINT 2
    
    # 3. CHECK if we've hit the symbol cap
    if self._cap and len(snap) >= self._cap:
        # Try prioritized add; if fails, buffer for later
        ok, reason = await self.add_symbol(symbol, source=source, **kwargs)
        if ok:
            return True, None  # ✓ ACCEPTED
        # ❌ IF ADD FAILS: just buffer, not integrated
        self.buffered_symbols.append(_meta_from_kwargs(symbol, source, **kwargs))
        logger.info("⏳ Queued %s (cap reached).", symbol)
        return False, reason or "cap reached"  # ⚠️ REJECTION POINT 3
    
    # 4. TRY to add
    return await self.add_symbol(symbol, source=source, **kwargs)
```

**Key Rejection Points:**
1. **Discovery disabled** → `config.Discovery.accept_new_symbols = False`
2. **Already exists** → Symbol already in accepted_symbols
3. **Cap reached** → Buffered but **NOT auto-integrated**

---

### Phase 3: The Critical Gap ❌ (THE PROBLEM)

**SymbolManager.add_symbol() → SymbolManager._is_symbol_valid() (`symbol_manager.py` lines 488-510)**

```python
async def _is_symbol_valid(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str], float]:
    # Validation gates (in order):
    
    # 1. Blacklist check
    if s in self._blacklist:
        return False, "blacklisted", 0.0
    
    # 2. Format check
    ok, reason = self.validate_symbol_format(s)
    if not ok:
        return False, reason, 0.0
    
    # 3. Exchange validity check
    ok, reason = await self.is_valid_symbol(s)
    if not ok:
        return False, reason, 0.0
    
    # 4. RISK FILTER check ⚠️
    ok, reason = await self._passes_risk_filters(s, source, **kwargs)
    if not ok:
        self.logger.debug("risk filter failed for %s: %s", s, reason)
        return False, reason, 0.0  # ❌ REJECTED HERE
    
    # 5. Price validation
    price = float(kwargs.get("price", 0.0))
    if price <= 0:
        try:
            price = await self.exchange_client.get_ticker_price(s)
        except Exception:
            price = 0.0
    
    if price <= 0:
        return False, "market price unavailable", 0.0  # ❌ REJECTED HERE
    
    return True, None, price
```

**The Problem: `_passes_risk_filters()` is likely rejecting discovered symbols!**

This means:
- SymbolScreener finds ETHUSDT (high volatility, high volume)
- SymbolScreener proposes ETHUSDT to SymbolManager
- SymbolManager._is_symbol_valid() calls `_passes_risk_filters()`
- Risk filters **REJECT** ETHUSDT
- Discovery agent never knows why
- Symbol is NOT added to accepted_symbols
- **MetaController never sees the symbol**

---

## Current Symbol Selection Flow (What MetaController Actually Uses)

```python
# In MetaController:
cycle_symbols = self.shared_state.get_symbols()  # ← Reads from accepted_symbols

for symbol in cycle_symbols:
    # Only these symbols are evaluated
    signal = await agent.evaluate(symbol)
```

**The Issue:** `accepted_symbols` only contains symbols that:
1. Passed discovery
2. Passed SymbolManager validation
3. Passed risk filters
4. Have valid prices

If risk filters are too strict, even high-quality discovered symbols get filtered out!

---

## Risk Filter Likelihood Issues

Search for `_passes_risk_filters()` in symbol_manager.py to understand what's filtering symbols.

Potential culprits:
- **Minimum volatility threshold** → Rejecting stable symbols
- **Maximum volatility threshold** → Rejecting too-volatile symbols
- **Volume requirement** → Rejecting low-liquidity pairs
- **Price range filter** → Rejecting very high/low prices
- **Blacklist** → Explicit exclusions
- **Slippage estimate** → Rejecting hard-to-trade pairs

---

## How to Diagnose

### Step 1: Check Discovery Agent Logs

```bash
# After running discovery agents:
grep "✅ Accepted\|❌ Rejected" logs/*.log

# Should see something like:
# ✅ Accepted ETHUSDT from SymbolScreener
# ✅ Accepted BNBUSDT from WalletScannerAgent
```

### Step 2: Check SymbolManager Proposal Logs

```bash
grep "Proposal\|risk filter\|already exists" logs/*.log

# Should see proposals being tracked or rejected
```

### Step 3: Compare Discovered vs Accepted

```python
# In Python:
discovered = [
    # from discovery agent logs
]
accepted = shared_state.accepted_symbols.keys()

missing = set(discovered) - set(accepted)
print(f"Discovered but not accepted: {missing}")

# If missing is non-empty, we have a filtering issue
```

---

## Solutions

### Solution 1: Loosen Risk Filters
Check `config.RiskFilter` settings and relax thresholds that are too strict.

### Solution 2: Bypass Risk Filters for Discovered Symbols
Add a bypass flag for symbols that passed rigorous discovery criteria:

```python
async def _is_symbol_valid(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str], float]:
    # ... existing checks ...
    
    # NEW: Trust discovery agents on good-faith proposals
    if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
        # These agents do their own pre-filtering
        # Skip our risk filters if their metadata is good
        if kwargs.get("atr_pct", 0) > 1.0 and kwargs.get("quote_volume", 0) > 1_000_000:
            self.logger.info(f"Trusted discovery source {source} for {symbol}")
            # Jump to price validation
            return await self._validate_price_only(symbol, **kwargs)
    
    # ... normal risk filter path ...
```

### Solution 3: Enable Buffering for Cap Overflow
Make sure buffered symbols (queued when cap is hit) are being flushed:

```python
# In MetaController or startup:
if hasattr(self.symbol_manager, "flush_buffered_proposals_to_shared_state"):
    await self.symbol_manager.flush_buffered_proposals_to_shared_state()
```

### Solution 4: Increase Symbol Cap
If cap is reached early:

```python
# In config:
Discovery.symbol_cap = 100  # Increase from current value
```

---

## Verification Checklist

- [ ] Discovery agents are running and logging proposals (grep for "Proposed" or "Found")
- [ ] Check SymbolManager logs for rejection reasons
- [ ] Verify `accepted_symbols` is being populated
- [ ] Check if risk filters are too strict
- [ ] Verify symbol cap isn't being hit
- [ ] Confirm `Discovery.accept_new_symbols = True` in config
- [ ] Check if buffered symbols are being flushed
- [ ] Verify MetaController is actually reading `accepted_symbols`

---

## Summary

**Your diagnosis is correct:** Discovery agents are finding symbols, but they're probably not getting into `accepted_symbols` due to:

1. **Risk filters rejecting them** (most likely)
2. **Symbol cap being reached** without proper buffering
3. **Discovery being disabled** in config
4. **Validation gates** (price, format, exchange status)

The fix is to **trace the rejection path** → **find which filter is too strict** → **adjust thresholds or add bypass logic**.

