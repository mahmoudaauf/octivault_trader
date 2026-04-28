# 🔍 DIAGNOSTIC COMMANDS - RUN THESE TO UNDERSTAND YOUR SYSTEM STATE

**Goal**: Get concrete data about the phantom ETHUSDT position before we fix anything

---

## Command 1: Check Current Position State (Local Cache)

```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Check if there are any position state files
find . -name "*position*" -name "*.json" -o -name "*portfolio*" -o -name "*state*" | grep -v ".pyc" | head -20
```

**What to look for**: Any JSON files with recent timestamps that might contain positions

---

## Command 2: Search Logs for ETHUSDT State Info

```bash
# Find all mentions of ETHUSDT and what the system knew about it
grep -i "ETHUSDT\|ETH/USDT" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/monitor.log | grep -i "qty\|quantity\|position\|balance" | head -50
```

**Expected output**: Will show attempts to get qty, what qty values were seen

---

## Command 3: Extract Rejection Pattern

```bash
# Show the rejection messages for ETHUSDT
grep "ETHUSDT.*SELL.*amount.*positive" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/MONITOR_SUMMARY.log | wc -l
```

**Expected output**: A number showing how many times this error occurred

---

## Command 4: Find When ETHUSDT Became Phantom

```bash
# Look for ETHUSDT in older log files to find when it got stuck
grep -i "ETHUSDT.*qty\|ETHUSDT.*quantity" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/monitor.log | head -20
```

**Expected output**: Timeline of ETHUSDT qty changes, last one should show qty becoming 0 or very small

---

## Command 5: Check Most Recent System Startup

```bash
# See what the system loaded on most recent restart
tail -500 /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/system_restart.log | grep -i "position\|ethusdt\|qty\|quantity"
```

**Expected output**: Whether system loaded positions from state or exchange

---

## Command 6: Check Execution Manager for ETHUSDT Attempts

```bash
# Find every place system tried to process ETHUSDT
grep -n "ETHUSDT" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/core/execution_manager.py | head -10
```

**Expected output**: Should show ETHUSDT is being handled (or not, depending on how symbols are selected)

---

## Command 7: Manual Account Check (Python)

If you want to verify from Binance directly:

```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Create a quick Python script to check your account
cat > check_account.py << 'EOF'
import os
from dotenv import load_dotenv
from binance.spot import Spot

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Spot(key=api_key, secret=api_secret, base_url="https://api.binance.com")

# Get all open positions
account_info = client.account()
balances = account_info.get('balances', [])

print("Current Holdings:")
for b in balances:
    asset = b['asset']
    free = float(b['free'])
    locked = float(b['locked'])
    total = free + locked
    
    if total > 0.00001:  # Only show non-dust
        print(f"{asset}: {total} (free: {free}, locked: {locked})")

# Check specifically for ETH
eth_balance = next((b for b in balances if b['asset'] == 'ETH'), None)
if eth_balance:
    print(f"\nETH DETAILED: {eth_balance}")
else:
    print("\nNo ETH found in account")

print("\nUSDAT Balance:", next((b for b in balances if b['asset'] == 'USDT'), {}).get('free', '0'))
EOF

python check_account.py
```

---

## Command 8: Check If Previous Partial Exit is Still There

```bash
# Look for SELL orders with small amounts that might have left remainder
grep -i "SELL.*ETH\|ETHUSDT.*SELL" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/MONITOR_SUMMARY.log | tail -30
```

**What to look for**: Any SELL orders with small fill amounts

---

## Command 9: Verify My Dust Fix Code is Actually Running

```bash
# Check if my new dust prevention code exists and is integrated
grep -A 5 "notional_residual_is_dust" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/core/execution_manager.py | head -20
```

**Expected output**: Should show my 3-layer dust detection code (confirming changes were applied)

---

## Command 10: Check Loop State at Cycle 1195

```bash
# Find exactly what the system was doing when it froze
grep "Loop.*1195" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/monitor.log | tail -5
```

**Expected output**: Show what symbol was being processed when loop froze

---

## Quick Summary Command (Run ALL Together)

```bash
#!/bin/bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

echo "=== ETHUSDT REJECTION COUNT ==="
grep "ETHUSDT.*amount.*positive" MONITOR_SUMMARY.log 2>/dev/null | wc -l || echo "File not found"

echo -e "\n=== LAST 20 ETHUSDT REFERENCES IN LOGS ==="
grep -i "ETHUSDT" monitor.log 2>/dev/null | tail -20 || echo "No logs found"

echo -e "\n=== MY DUST FIX CODE VERIFICATION ==="
if grep -q "notional_residual_is_dust" core/execution_manager.py; then
    echo "✅ Dust prevention code IS in place"
else
    echo "❌ Dust prevention code NOT found"
fi

echo -e "\n=== CURRENT FREE BALANCE ==="
tail -50 system_restart.log 2>/dev/null | grep -i "capital\|balance\|free" | tail -3 || echo "No restart log found"
```

---

## What Each Command Tells Us

| Command | Purpose | Answer Needed |
|---------|---------|---------------|
| 1 | Find position state files | Where is phantom stored? |
| 2 | Search logs for ETHUSDT qty | What qty values did system see? |
| 3 | Count rejections | How many times stuck? |
| 4 | Find when it happened | Exact timestamp of freeze? |
| 5 | Check startup | Did system load phantom on restart? |
| 6 | Verify My Fixes | Are my changes actually there? |
| 7 | Direct Binance Check | Does real ETHUSDT exist or is it gone? |
| 8 | Find partial exit | Was there an incomplete SELL? |
| 9 | Verify dust code | Is my new logic integrated? |
| 10 | Loop state | Which symbol broke the loop? |

---

## Run These IN ORDER and Share Results

1. **Start with Commands 1-3** (fast, read logs only)
2. **Then Command 7** (if you want real account state from Binance)
3. **Share the output** and I'll tell you exact fix needed

**Most Important Answer**: 
- Does Binance actually have an ETH position right now?
- OR was it already sold/deleted and only phantom in your local state?

---

## Expected Scenarios & What They Mean

### Scenario A: ETHUSDT exists on Binance with qty > 0
- **Diagnosis**: State sync issue - local qty showing 0.0 but exchange has real qty
- **Fix**: Force sync from exchange, resume trading

### Scenario B: ETHUSDT deleted from Binance but persists locally  
- **Diagnosis**: Stale position cache - position was already closed
- **Fix**: Delete phantom from local state, resume trading

### Scenario C: ETHUSDT exists on Binance but qty < min_qty
- **Diagnosis**: Phantom remainder - too small to trade
- **Fix**: Force liquidate as zero-value position

### Scenario D: ETHUSDT exists on Binance with qty = 0.0
- **Diagnosis**: Exactly phantom problem - exchange created zero position somehow  
- **Fix**: Delete from exchange, delete from local state

All scenarios have straightforward fixes. We just need data first.

