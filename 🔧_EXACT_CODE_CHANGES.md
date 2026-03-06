# 🔧 Exact Code Changes to Fix Discovery Flow

## Summary

Three simple fixes, in order of recommendation:

1. **EASIEST** - Lower the volume threshold
2. **BETTER** - Add source bypass
3. **BEST** - Do both

---

## Fix #1: Lower Volume Threshold (EASIEST - 1 line change)

### What to do:
Find and lower `min_trade_volume` in your config.

### Step 1: Find the config value

```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
grep -rn "min_trade_volume" config/ core/ | head -20
```

You should see something like:
```
config/discovery.py:45: min_trade_volume = 50000
```

### Step 2: Edit the value

**Current:**
```python
min_trade_volume = 50000  # or whatever it currently is
```

**Change to:**
```python
min_trade_volume = 10000  # Reduced from 50k to 10k
```

### Step 3: Restart and verify

```bash
# Restart your bot
# Check logs for more ✅ acceptances
grep "Accepted.*from Symbol\|Accepted.*from IPO" logs/*.log | wc -l
```

---

## Fix #2: Add Source Bypass (BETTER - 4 line changes)

### What to do:
Modify `core/symbol_manager.py` to trust discovery agents on volume thresholds.

### File: `core/symbol_manager.py`

**Location 1:** Lines 319-323

#### BEFORE:
```python
        if qv is None:
            # P9 Guard: If the source is authoritative (WalletScanner), we skip volume check
            # if we can't find volume, as it's better to keep the symbol than to lose it.
            if source == "WalletScannerAgent":
                self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
                return True, None
            return False, "missing 24h quote volume"
```

#### AFTER:
```python
        if qv is None:
            # P9 Guard: If the source is authoritative (discovery agent), we skip volume check
            # as it's better to keep the symbol than to lose it.
            if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
                self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
                return True, None
            return False, "missing 24h quote volume"
```

**Change Summary:**
- Line 323: Change `if source == "WalletScannerAgent":` 
- To: `if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):`

---

**Location 2:** Lines 329-332

#### BEFORE:
```python
        if float(qv) < float(self._min_trade_volume):
            if source == "WalletScannerAgent":
                 self.logger.info(f"[WalletScannerAgent] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
                 return True, None
            return False, f"below min 24h quote volume ({qv} < {self._min_trade_volume})"
```

#### AFTER:
```python
        if float(qv) < float(self._min_trade_volume):
            if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
                self.logger.info(f"[{source}] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
                return True, None
            return False, f"below min 24h quote volume ({qv} < {self._min_trade_volume})"
```

**Change Summary:**
- Line 330: Change `if source == "WalletScannerAgent":` 
- To: `if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):`
- Line 331: Change `[WalletScannerAgent]` to `[{source}]` for better logging

---

### Step 3: Restart and verify

```bash
# Restart your bot
# Check logs for SymbolScreener and IPOChaser acceptances
grep "Accepted.*from SymbolScreener\|Accepted.*from IPOChaser" logs/*.log | head -20
```

---

## Fix #3: Do Both (BEST - 5 total line changes)

Apply **both** Fix #1 AND Fix #2 for maximum effect.

This gives you:
- ✅ A reasonable baseline threshold (10k instead of 50k)
- ✅ Additional trust for discovery agents
- ✅ Safety net in both directions

### Combined Changes:

#### Change 1: Lower threshold in config
```python
# config/discovery.py
min_trade_volume = 10000  # Down from 50000
```

#### Change 2: Add source bypass at line 323
```python
# core/symbol_manager.py line 323
if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
```

#### Change 3: Add source bypass at line 330
```python
# core/symbol_manager.py line 330
if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
```

---

## Verification Steps

### Step 1: Confirm config change

```bash
grep "min_trade_volume" config/*.py
```

Expected output:
```
config/discovery.py:45: min_trade_volume = 10000
```

### Step 2: Confirm code changes

```bash
grep -A 2 "if source in" core/symbol_manager.py | head -10
```

Expected output:
```
if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
    self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
    return True, None
--
if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
    self.logger.info(f"[{source}] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
```

### Step 3: Restart and check logs

```bash
# Kill current process
pkill -f "python.*main.py"

# Restart
python main.py &

# Wait 30 seconds
sleep 30

# Check for acceptances
tail -100 logs/symbol_manager.log | grep -E "Accepted|bypassed|authoritative"
```

Expected to see:
```
[INFO] SymbolManager: ✅ Accepted ETHUSDT from SymbolScreener.
[INFO] SymbolManager: ✅ Accepted SOLUSDT from SymbolScreener.
[INFO] SymbolManager: [SymbolScreener] No volume info for AVAXUSDT; allowing as authoritative
[INFO] SymbolManager: [IPOChaser] Symbol NEWCOINUSDT volume 5000 < 10000, but allowing as authoritative
```

### Step 4: Count improvements

```bash
# Count accepted symbols before fix
echo "Before fix: $(grep '✅ Accepted' logs/symbol_manager.log.old 2>/dev/null | wc -l) symbols"

# Count accepted symbols after fix
echo "After fix:  $(grep '✅ Accepted' logs/symbol_manager.log | wc -l) symbols"

# Should show significant increase
```

### Step 5: Verify MetaController sees them

```bash
# Check that MetaController is evaluating more symbols
grep -o "Evaluating [0-9]* symbols" logs/meta_controller.log | tail -5
```

Expected to increase from:
```
Evaluating 5 symbols
```

To:
```
Evaluating 28 symbols
```

---

## Detailed Line-by-Line Changes

### File: `core/symbol_manager.py`

#### Change A (Line 323):

**Before:**
```python
319:    async def _passes_risk_filters(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str]]:
320:        if symbol in self._blacklist:
321:            return False, "symbol blacklisted (config)"
322:        if not self.exchange_client:
323:            return False, "exchange client unavailable"
324:
325:        # existence via cache (cheap), otherwise awaited call
326:        if hasattr(self.exchange_client, "symbol_exists_cached"):
327:            if not self.exchange_client.symbol_exists_cached(symbol):
328:                return False, "symbol not trading (cached)"
...
318:        if qv is None:
319:            # P9 Guard: If the source is authoritative (WalletScanner), we skip volume check
320:            # if we can't find volume, as it's better to keep the symbol than to lose it.
321:            if source == "WalletScannerAgent":
322:                self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
323:                return True, None
324:            return False, "missing 24h quote volume"
```

**After (Change line 321):**
```python
318:        if qv is None:
319:            # P9 Guard: If the source is authoritative (discovery agent), we skip volume check
320:            # if we can't find volume, as it's better to keep the symbol than to lose it.
321:            if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
322:                self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
323:                return True, None
324:            return False, "missing 24h quote volume"
```

#### Change B (Line 330):

**Before:**
```python
329:        if float(qv) < float(self._min_trade_volume):
330:            if source == "WalletScannerAgent":
331:                 self.logger.info(f"[WalletScannerAgent] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
332:                 return True, None
333:            return False, f"below min 24h quote volume ({qv} < {self._min_trade_volume})"
```

**After (Change lines 330-331):**
```python
329:        if float(qv) < float(self._min_trade_volume):
330:            if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):
331:                self.logger.info(f"[{source}] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
332:                 return True, None
333:            return False, f"below min 24h quote volume ({qv} < {self._min_trade_volume})"
```

---

## Alternative: Automated Patch Script

If you want to apply changes automatically:

```bash
#!/bin/bash
# apply_discovery_fix.sh

set -e

echo "🔧 Applying discovery agent fix..."

# Backup original
cp core/symbol_manager.py core/symbol_manager.py.backup

# Apply changes using sed
echo "  ✓ Changing line 321..."
sed -i.bak '321s/if source == "WalletScannerAgent":/if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):/' core/symbol_manager.py

echo "  ✓ Changing line 330..."
sed -i.bak '330s/if source == "WalletScannerAgent":/if source in ("WalletScannerAgent", "SymbolScreener", "IPOChaser"):/' core/symbol_manager.py

echo "  ✓ Changing line 331..."
sed -i.bak '331s/\[WalletScannerAgent\]/[{source}]/' core/symbol_manager.py

# Verify
echo ""
echo "✅ Changes applied. Verification:"
grep -n "if source in" core/symbol_manager.py | head -2

echo ""
echo "✅ Fix complete! Changes made:"
echo "   • Line 321: Extended source check to include SymbolScreener, IPOChaser"
echo "   • Line 330: Extended source check to include SymbolScreener, IPOChaser"
echo "   • Line 331: Improved logging with {source} placeholder"
echo ""
echo "Next steps:"
echo "   1. Update config: min_trade_volume = 10000"
echo "   2. Restart bot"
echo "   3. Verify: grep 'Accepted.*SymbolScreener' logs/*.log"
```

---

## Rollback (If Needed)

If something goes wrong:

```bash
# Restore from backup
cp core/symbol_manager.py.backup core/symbol_manager.py

# Or restore from git
git checkout core/symbol_manager.py
```

---

## Testing Changes

After applying fixes, test with this script:

```python
#!/usr/bin/env python3
"""Test discovery fix."""

import asyncio
from core.config import Config
from core.database_manager import DatabaseManager
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient
from core.symbol_manager import SymbolManager
from agents.symbol_screener import SymbolScreener

async def test():
    cfg = Config()
    db = DatabaseManager(cfg)
    await db.connect()
    
    ss = SharedState(cfg, db)
    ec = ExchangeClient(cfg, ss)
    await ec.initialize()
    
    sm = SymbolManager(cfg, ec, ss, db)
    ss.symbol_manager = sm
    
    # Test proposal with low volume
    print("Testing SymbolScreener proposal with volume edge case...")
    
    ok, reason, price = await sm._is_symbol_valid(
        "TESTUSDT",
        source="SymbolScreener",
        quote_volume=5000  # Below 10k threshold
    )
    
    if ok:
        print("✅ PASSED: SymbolScreener proposal accepted despite low volume")
    else:
        print(f"❌ FAILED: {reason}")
        
    await db.disconnect()

asyncio.run(test())
```

---

## Summary Checklist

- [ ] Found config value: `min_trade_volume = ???`
- [ ] Lowered threshold to 10,000 (or 5,000 if aggressive)
- [ ] Modified line 321 in `core/symbol_manager.py`
- [ ] Modified line 330 in `core/symbol_manager.py`
- [ ] Modified line 331 in `core/symbol_manager.py`
- [ ] Backup created
- [ ] Changes verified with grep
- [ ] Bot restarted
- [ ] Logs show more acceptances
- [ ] MetaController evaluates more symbols
- [ ] Accepted count increased significantly

---

## FAQ

**Q: Will lowering the threshold cause bad symbol selection?**
A: Discovery agents already filter by volatility and volume. The SymbolScreener does its own ATR filtering. You're just lowering the gate threshold, not removing quality filters.

**Q: Should I do both Fix #1 and Fix #2?**
A: Yes. Together they're more robust. Fix #1 is the floor threshold, Fix #2 is the trust override.

**Q: What if I set it too low?**
A: Set position size as a % of capital, not fixed amount. Lower-volume symbols just means smaller max position.

**Q: Can I revert easily?**
A: Yes, restore from backup: `cp core/symbol_manager.py.backup core/symbol_manager.py`

---

Done! Apply these changes and your discovery agents will feed properly into accepted_symbols. 🎉

