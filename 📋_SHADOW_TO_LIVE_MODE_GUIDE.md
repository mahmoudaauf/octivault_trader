# Complete Guide: Switch from Shadow Mode to Live Trading Mode

## Summary

Your bot is currently running in **SHADOW MODE** (virtual ledger simulation). To switch to **LIVE MODE** with real exchange NAV, you need to change the `TRADING_MODE` configuration.

---

## Quick Change

Change this configuration setting:

```python
# FROM (current):
TRADING_MODE = "shadow"

# TO (real mode):
TRADING_MODE = "live"
```

---

## Where to Find TRADING_MODE

### Option 1: Configuration File (Most Likely)

Search for `TRADING_MODE` in your config files:

```bash
# In terminal, search for the setting:
grep -r "TRADING_MODE" . --include="*.py" --include="*.yaml" --include="*.json"
```

Common locations:
- `config/config.yaml` or `config/settings.yaml`
- `config/secrets.env`
- `.env` file
- `core/app_context.py` (initialization code)
- Environment variables

### Option 2: Direct Source Code Change

If TRADING_MODE is hardcoded in Python, find it in:

1. **`core/shared_state.py` (line 531)**
   ```python
   self.trading_mode: str = str(getattr(self.config, 'trading_mode', 'live') or 'live')
   ```
   This reads from `self.config.trading_mode`

2. **Find where config.trading_mode is SET**
   ```bash
   grep -r "config.trading_mode\s*=" . --include="*.py"
   grep -r "\.trading_mode.*=" . --include="*.py"
   ```

3. **Look in `core/app_context.py`**
   Configuration is typically initialized here

---

## Complete Implementation Example

If you need to set it programmatically:

```python
# In your main.py or app_context.py initialization:

# BEFORE (Shadow Mode):
config = Config()
config.trading_mode = "shadow"  # ← Virtual ledger mode
shared_state = SharedState(config)

# AFTER (Live Mode):
config = Config()
config.trading_mode = "live"    # ← Real exchange NAV
shared_state = SharedState(config)
```

---

## What Changes When You Switch

### Current State (SHADOW MODE):

```
Configuration: TRADING_MODE = "shadow"
SharedState: trading_mode = "shadow"
RecoveryEngine: Fetches balances but doesn't apply them
NAV Calculation: Uses virtual ledger (NAV = 0)
Positions: From internal snapshot/virtual ledger
Use Case: Testing, simulation, recovery mode
```

**Logs Show:**
```
[SHADOW MODE - balances not updated, virtual ledger is authoritative]
NAV is 0.0 (SHADOW MODE - this is expected)
Positions: 3 (from virtual ledger)
```

### After Switch (LIVE MODE):

```
Configuration: TRADING_MODE = "live"
SharedState: trading_mode = "live"
RecoveryEngine: Fetches and applies balances
NAV Calculation: Uses real exchange balances
Positions: Reconciled with real orders
Use Case: Production trading, real exchange sync
```

**Logs Will Show:**
```
[SS:BalanceUpdate] USDT: free=1234.56, locked=0.0
[NAV] Total: 1234.56 | Quotes: {'USDT': {...}}
NAV is 1234.56 (from real exchange)
Positions: 3 (reconciled with orders)
```

---

## Step-by-Step Instructions

### Step 1: Locate the Configuration

```bash
# Search for TRADING_MODE setting
grep -r "trading_mode" . --include="*.py" | head -20
grep -r "TRADING_MODE" . --include="*.yaml" --include="*.env"
```

### Step 2: Identify the File

Note the file path where `trading_mode` is set. Examples:
- `config.yaml`
- `.env`
- `core/app_context.py`
- `main.py`

### Step 3: Make the Change

**If in YAML:**
```yaml
# config.yaml
trading_mode: "live"  # Changed from "shadow"
```

**If in .env:**
```bash
# .env
TRADING_MODE=live  # Changed from shadow
```

**If in Python code:**
```python
# core/app_context.py or similar
config.trading_mode = "live"  # Changed from "shadow"
```

### Step 4: Restart the Bot

```bash
# Stop current bot
# Restart with new configuration

python main.py
# or
docker-compose restart octivault_trader
# or your deployment method
```

### Step 5: Verify the Change

Check the startup logs for:

```
✅ [shared_state] trading_mode = live
✅ [SS:BalanceUpdate] USDT: free=X.XX, locked=Y.YY
✅ [NAV] Total: Z.ZZ
✅ NAV Ready (value > 0)
```

---

## Verification Checklist

After switching, verify these logs appear:

- [ ] `trading_mode = "live"` (not "shadow")
- [ ] `[SS:BalanceUpdate]` messages with real balances
- [ ] `[NAV]` messages showing actual balance amount
- [ ] NAV is greater than 0
- [ ] No "[SHADOW MODE]" messages
- [ ] Positions reconciled with real orders
- [ ] Step 5 verification passes (NAV > 0)

---

## Reverting Back to Shadow Mode

If you need to return to testing:

```python
config.trading_mode = "shadow"
```

Then restart the bot. It will return to virtual ledger mode.

---

## What Happens in Each Mode

### SHADOW MODE
- **When to use:** Testing, development, recovery drills
- **NAV:** 0 (virtual ledger, not real exchange)
- **Balances:** Fetched but not synced
- **Positions:** From internal snapshot
- **Orders:** Virtual (not sent to exchange)
- **Risk:** NONE (paper trading only)

### LIVE MODE
- **When to use:** Production trading
- **NAV:** Real exchange wallet balance
- **Balances:** Fetched and synced
- **Positions:** Reconciled with real orders
- **Orders:** Sent to real exchange
- **Risk:** REAL (actual trades will execute)

---

## Troubleshooting

### If NAV is still 0 after switching to LIVE:

1. **Check configuration was applied:**
   ```bash
   # Verify in logs:
   # Should show: trading_mode = live
   ```

2. **Check exchange connection:**
   ```
   Logs should show: [SS:BalanceUpdate] USDT: ...
   If missing: Exchange API connection failed
   ```

3. **Check bot restarted:**
   ```bash
   # Make sure you restarted after changing config
   # Changes don't apply to running process
   ```

4. **Check account has balance:**
   ```
   If all logs correct but NAV=0: 
   Your real exchange account is empty
   (This is not a bug, just reality)
   ```

### If you see error after changing mode:

- **NAV validation fails:** Means exchange API working but mode wasn't set correctly
- **Exchange connection fails:** Network issue or API key problem
- **Positions mismatch:** Recovery may need a clean sync

---

## Advanced: Environment Variable Override

If TRADING_MODE is hardcoded, you may override via environment:

```bash
# At bot startup:
export TRADING_MODE=live
python main.py

# Or inline:
TRADING_MODE=live python main.py
```

This only works if your code checks environment variables.

---

## Final Answer to Your Question

> "but why not to fetch the real nac?"

**It IS fetching the real NAV**, but:
1. It's set to SHADOW MODE intentionally
2. In shadow mode, fetched balances aren't applied
3. Virtual ledger is authoritative instead
4. NAV=0 is correct for this mode

**To get real NAV:**
1. Change `TRADING_MODE` from "shadow" to "live"
2. Restart the bot
3. Real exchange balances will be applied
4. NAV will show actual wallet balance

That's it! 🎯
