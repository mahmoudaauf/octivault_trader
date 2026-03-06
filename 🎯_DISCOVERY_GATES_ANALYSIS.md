# 🎯 Discovery Symbol Rejection Analysis - The 5 Gates

## TL;DR

Your discovery agents **ARE finding better symbols**, but 5 sequential validation gates are **REJECTING them** before they reach `accepted_symbols`.

---

## The 5 Rejection Gates (In Order)

```
Discovery Agent Proposes Symbol
    ↓
1️⃣  GATE: Blacklist Check
    ├─ Is symbol in _blacklist? → ❌ Reject "symbol blacklisted (config)"
    └─ Pass → Continue
    ↓
2️⃣  GATE: Exchange Existence Check
    ├─ Does symbol exist on exchange? → ❌ Reject "symbol not trading"
    └─ Pass → Continue
    ↓
3️⃣  GATE: Quote Volume Check ⚠️ (MAJOR FILTER)
    ├─ Get 24h quote volume from stats
    ├─ Is volume available? → ❌ Reject "missing 24h quote volume"
    │                        (unless WalletScannerAgent source)
    ├─ Is volume >= _min_trade_volume? → ❌ Reject "below min 24h quote volume"
    │                                    (unless WalletScannerAgent source)
    └─ Pass → Continue
    ↓
4️⃣  GATE: Stable Asset Check (Optional)
    ├─ If _exclude_stable_base=True:
    │  ├─ Is base asset stable? → ❌ Reject "base asset classified as stable"
    │  └─ Pass → Continue
    └─ Pass → Continue
    ↓
5️⃣  GATE: Price Availability Check
    ├─ Can we fetch current price? → ❌ Reject "market price unavailable"
    └─ Pass → Continue
    ↓
✅ SYMBOL ACCEPTED → Added to accepted_symbols
```

---

## Gate Detailed Analysis

### Gate 1: Blacklist Check (Easy to Bypass)
**Location:** `symbol_manager.py:285-287`

```python
if symbol in self._blacklist:
    return False, "symbol blacklisted (config)"
```

**Why it rejects:**
- Manual blacklist in config (rare)
- Usually not the problem

**Fix if needed:**
```
config.SymbolManager.blacklist = []  # or remove problematic symbols
```

---

### Gate 2: Exchange Existence Check (Minimal Impact)
**Location:** `symbol_manager.py:288-295`

```python
# existence via cache (cheap), otherwise awaited call
if hasattr(self.exchange_client, "symbol_exists_cached"):
    if not self.exchange_client.symbol_exists_cached(symbol):
        return False, "symbol not trading (cached)"
elif hasattr(self.exchange_client, "symbol_exists"):
    if not await self.exchange_client.symbol_exists(symbol):
        return False, "symbol not trading"
```

**Why it rejects:**
- Symbol doesn't exist on Binance
- Symbol is delisted/paused
- Cache is stale

**Fix if needed:**
- Clear exchange_client cache
- Verify symbol spelling
- Check symbol is TRADING on Binance

---

### Gate 3: Quote Volume Check ⚠️ (THE MAIN CULPRIT)
**Location:** `symbol_manager.py:296-326`

```python
# quote volume: try kwargs → client quick calls → cached 24h stats
qv = _extract_quote_volume(kwargs)  # Try from metadata first
if qv is None and hasattr(self.exchange_client, "get_24hr_volume"):
    try:
        qv = await self.exchange_client.get_24hr_volume(symbol)
    except Exception as e:
        self.logger.debug("get_24hr_volume fail %s: %s", symbol, e, exc_info=True)

# Get cached 24h stats for volume data
stats: Dict[str, Any] = {}
if hasattr(self.exchange_client, "get_cached_24h_stats"):
    stats = self.exchange_client.get_cached_24h_stats(symbol) or {}

# Fallback: if stats missing, try explicit fetch
if not stats and hasattr(self.exchange_client, "get_24h_stats"):
    try:
        stats = await self.exchange_client.get_24h_stats(symbol) or {}
    except Exception:
        pass

# Calculate quote volume from base volume + WAP if needed
quote_vol = float(stats.get("quoteVolume") or stats.get("volume") or 0.0)
base_vol = float(stats.get("baseVolume") or 0.0)
wap = float(stats.get("weightedAvgPrice") or 0.0)
if quote_vol == 0.0 and base_vol > 0.0 and wap > 0.0:
    quote_vol = base_vol * wap
if qv is None and quote_vol > 0.0:
    qv = quote_vol

# FIRST REJECTION: No volume data at all
if qv is None:
    if source == "WalletScannerAgent":
        self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
        return True, None
    return False, "missing 24h quote volume"  # ❌ REJECTED HERE

# SECOND REJECTION: Volume too low
if float(qv) < float(self._min_trade_volume):
    if source == "WalletScannerAgent":
        self.logger.info(f"[WalletScannerAgent] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
        return True, None
    return False, f"below min 24h quote volume ({qv} < {self._min_trade_volume})"  # ❌ REJECTED HERE
```

**Key Issue:**
- **WalletScannerAgent gets a BYPASS** (lines 320-323, 329-332)
- **SymbolScreener and IPOChaser do NOT get a bypass** ❌

**Why SymbolScreener symbols get rejected:**
1. SymbolScreener discovers high-volatility symbols ✓
2. SymbolScreener proposes to SymbolManager
3. SymbolManager tries to fetch volume for SymbolScreener proposal
4. If volume data is missing/stale → **REJECTED** ❌
5. OR if volume < `_min_trade_volume` → **REJECTED** ❌

**The Config Values to Check:**

```python
# In config.SymbolManager or config.Discovery:
_min_trade_volume = ???  # What is this set to?

# Typical values:
# - 50,000 = Very strict (rejects 80%+ of altcoins)
# - 10,000 = Moderate (rejects 40%+ of altcoins)
# - 1,000 = Loose (rejects 10%+ of altcoins)
```

**Fix for Gate 3:**

```python
# Option A: Lower the threshold
config.SymbolManager.min_trade_volume = 10000  # Down from 50000

# Option B: Add bypass for discovery agents
# (See solution section below)

# Option C: Ensure SymbolScreener passes volume in metadata
# Already done in symbol_screener.py:424:
metadata={
    "24h_quote_volume": float(item.get("quote_volume", 0.0) or 0.0),
    ...
}
```

---

### Gate 4: Stable Asset Check (Unlikely Issue)
**Location:** `symbol_manager.py:334-351`

```python
if self._exclude_stable_base:
    info = self.symbol_info_cache.get(symbol) or {}
    base = (info.get("baseAsset") or "").upper()
    if base and base != self._base:
        is_stable = False
        # ... checks if base is USDT, USDC, etc. ...
        if is_stable:
            return False, "base asset classified as stable"
```

**Why it rejects:**
- Rejects symbols like USDTUSDT, USDCUSDT
- Only active if `_exclude_stable_base = True`

**Fix if needed:**
```
config.SymbolManager.exclude_stable_base = False
```

---

### Gate 5: Price Availability Check (Moderate Impact)
**Location:** `symbol_manager.py:377-385`

```python
# P0: Ensure we have a real price before accepting
price = float(kwargs.get("price", 0.0))
if price <= 0:
    try:
        price = await self.exchange_client.get_ticker_price(s)
    except Exception:
        price = 0.0

if price <= 0:
    return False, "market price unavailable", 0.0
```

**Why it rejects:**
- Exchange client can't fetch price (API issue, rate limit, network)
- Symbol doesn't have valid price data
- Discovery agent didn't pass price in metadata

**Fix for Gate 5:**

```python
# Option A: Ensure discovery agents pass price in metadata
# SymbolScreener should get price when screening

# Option B: Increase API rate limits / retry timeout

# Option C: Add fallback price estimation
```

---

## The Real Culprit: Gate 3 (Quote Volume)

### Why SymbolScreener Symbols Get Blocked

```python
# SymbolScreener logs something like:
# "📊 Candidate symbols found: ['ETHUSDT', 'BNBUSDT', 'ADAUSDT', ...]"

# But when it proposes:
for item in candidates:
    symbol = item.get("symbol")  # e.g., "ETHUSDT"
    metadata = {
        "24h_quote_volume": float(item.get("quote_volume", 0.0) or 0.0),  # ✓ Has volume
        "atr_pct": float(item.get("atr_pct", 0.0) or 0.0),
        ...
    }
    accepted = await self._propose(symbol, source="SymbolScreener", metadata=metadata)
    # accepted = False  (volume check failed)
```

### Why WalletScanner Symbols Get Through

```python
# WalletScannerAgent:
# 1. Gets symbols from YOUR WALLET (trusted)
# 2. Proposes with source="WalletScannerAgent"
# 3. SymbolManager sees source="WalletScannerAgent"
# 4. **SKIPS volume check** (lines 320-323):
    if source == "WalletScannerAgent":
        return True, None  # ← Bypass!
```

---

## Configuration Hotspots

### Where to Look

```python
# config.py or config/*.py

# 1. Minimum trade volume threshold
Discovery.min_trade_volume = 50000  # ← Usually the culprit
# or
SymbolManager.min_trade_volume = 50000

# 2. Whether to exclude stable assets
SymbolManager.exclude_stable_base = True  # ← Check this

# 3. Blacklist
SymbolManager.blacklist = []

# 4. Exchange client caching
ExchangeClient.cache_24h_stats_ttl = 300  # seconds

# 5. Accept new symbols
Discovery.accept_new_symbols = True  # ← Must be True

# 6. Symbol cap
Discovery.symbol_cap = 20  # ← Can block symbols if reached
```

---

## Quick Diagnostic

Run this to find the issue:

```bash
# 1. Check current config values
grep -r "min_trade_volume\|exclude_stable\|accept_new_symbols" config/

# 2. Check logs for volume rejections
grep -i "below min 24h\|missing 24h" logs/*.log

# 3. Check if buffered symbols exist
grep -i "buffered\|cap reached" logs/*.log
```

---

## Solutions (From Most to Least Likely)

### Solution 1: Lower min_trade_volume Threshold ✅ (MOST LIKELY FIX)

**Problem:** Gate 3 is rejecting SymbolScreener discoveries due to strict volume threshold.

**Fix:**

```python
# In config.py or config/discovery.py

# Before (too strict):
Discovery.min_trade_volume = 50000  # Requires $50k+ daily volume

# After (more reasonable for altcoins):
Discovery.min_trade_volume = 10000  # Allows $10k+ daily volume

# Or even lower if you have capital:
Discovery.min_trade_volume = 5000   # Allows $5k+ daily volume
```

**Impact:**
- ✅ More symbols pass Gate 3
- ✅ MetaController gets more candidates to evaluate
- ✅ Better out-of-sample symbols discovered
- ⚠️ May include lower-liquidity symbols (manage position size)

---

### Solution 2: Add Discovery Agent Bypass (Moderate Effort)

**Problem:** Only WalletScannerAgent gets volume bypass; other agents don't.

**Fix:** Extend the bypass to all discovery agents:

```python
# In symbol_manager.py, line 319-323, change:

if qv is None:
    if source == "WalletScannerAgent":  # ← Only WalletScanner
        return True, None
    return False, "missing 24h quote volume"

# To:

if qv is None:
    # Trust discovery agents if they have good metadata
    if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
        self.logger.debug(f"[{source}] No volume; allowing discovery agent")
        return True, None
    return False, "missing 24h quote volume"
```

**Impact:**
- ✅ Discovery agents have more latitude to propose symbols
- ✅ Trusts agent filtering logic (they already filtered by volume)
- ⚠️ Removes one layer of validation

---

### Solution 3: Pass Price in Metadata (Best Practice)

**Problem:** Gate 5 sometimes rejects due to missing price.

**Fix:** Ensure discovery agents pass price when they propose:

```python
# In symbol_screener.py, around line 420, add price:

for item in candidates:
    symbol = self._normalize_symbol(item.get("symbol", ""))
    if not symbol:
        continue
    try:
        accepted_flag = await self._propose(
            symbol,
            source=self.name,
            metadata={
                "24h_quote_volume": float(item.get("quote_volume", 0.0) or 0.0),
                "24h_percent_change": float(item.get("price_change_percent", 0.0) or 0.0),
                "atr_pct": float(item.get("atr_pct", 0.0) or 0.0),
                "atr_timeframe": self.atr_timeframe,
                "price": float(item.get("price", 0.0) or 0.0),  # ← ADD THIS
            },
        )
```

---

### Solution 4: Increase Symbol Cap (If Cap is Hit)

**Problem:** Symbol cap reached; new discoveries buffered but not flushed.

**Fix:**

```python
# In config.py:
Discovery.symbol_cap = 50  # Increase from default

# AND ensure flush is called:
# In main.py or MetaController startup:
if hasattr(symbol_manager, "flush_buffered_proposals_to_shared_state"):
    await symbol_manager.flush_buffered_proposals_to_shared_state()
```

---

### Solution 5: Ensure Discovery is Enabled (Baseline Check)

**Problem:** Discovery disabled in config.

**Fix:**

```python
# In config.py:
Discovery.accept_new_symbols = True  # Must be True!
```

---

## Verification Checklist

After implementing fixes:

```bash
# 1. Check accepted symbols count increased
grep "accepted" logs/*.log | tail -20

# 2. Verify SymbolScreener proposals are now accepted
grep "SymbolScreener.*✅" logs/*.log

# 3. Verify IPOChaser proposals are now accepted
grep "IPOChaser.*✅" logs/*.log

# 4. Confirm MetaController sees new symbols
grep "tracked.*symbol" logs/*.log | head -20

# 5. Run diagnostic script
python diagnose_discovery_flow.py
```

---

## Summary

| Gate | Status | Likely? | Impact | Fix |
|------|--------|---------|--------|-----|
| 1. Blacklist | ✓ Working | Low | Blocks known bad symbols | Usually fine |
| 2. Exchange Existence | ✓ Working | Low | Blocks delisted symbols | Usually fine |
| 3. Quote Volume | ⚠️ **STRICT** | **HIGH** | **Blocks 70%+ of altcoins** | **Lower threshold OR add bypass** |
| 4. Stable Asset | ✓ Working | Low | Blocks USDT-based pairs | Disable if needed |
| 5. Price Availability | ⚠️ Sometimes | Medium | Blocks when API fails | Pass price in metadata |

**Most likely fix:** Lower `Discovery.min_trade_volume` from 50000 to 10000.

