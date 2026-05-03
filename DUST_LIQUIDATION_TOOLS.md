# 🔧 EXISTING DUST LIQUIDATION & HEALING TOOLS

**You're correct - there are already well-established tools for this!**

---

## Available Tools

### 1️⃣ **force_liquidate_dust.py** (Fast & Direct)
```bash
# Quick overview:
python3 force_liquidate_dust.py dry-run   # See what will be liquidated
python3 force_liquidate_dust.py execute   # Execute liquidation
```

**What it does:**
- Finds all dust positions (< $25 value by default)
- Liquidates them immediately (bypasses healing thresholds)
- Returns capital to USDT balance
- **Time:** 2-5 minutes

**Use this when:** You need IMMEDIATE capital recovery

---

### 2️⃣ **emergency_liquidate.sh** (Full Recovery)
```bash
bash emergency_liquidate.sh
```

**What it does:**
- Stops the bot cleanly
- Closes smallest positions first
- Frees capital for new trades
- Restarts bot

**Use this when:** System is stuck and you need emergency recovery

---

### 3️⃣ **diagnose_healing.py** (Diagnostic)
```bash
python3 diagnose_healing.py
```

**What it does:**
- Checks why auto-liquidation is blocked
- Shows all gates preventing healing
- Identifies bottlenecks
- Suggests next steps

**Use this when:** You want to understand what's blocking consolidation

---

### 4️⃣ **LiquidationAgent** (Automatic)
- Built into the trading system
- Runs continuously as background task
- Automatically triggers when:
  - Capital shortage detected
  - ROI/Loss thresholds hit
  - Dust accumulation excessive
  - Rebalancing needed

**Status:** Already integrated, but currently disabled in MICRO_SNIPER regime

---

### 5️⃣ **dead_capital_healer.py** (Regime-based)
- Located in: `src/l3_portfolio/dead_capital_healer.py`
- Monitors capital efficiency
- Triggers consolidation when capital is "dead" (not generating returns)

---

## Recommended Action Path

### Option A: Fast (5 minutes)
```bash
# 1. Check what needs liquidating
python3 force_liquidate_dust.py dry-run

# 2. Execute liquidation
python3 force_liquidate_dust.py execute

# 3. Restart bot
bash START_TRADING.sh
```

### Option B: Diagnostic + Action (10 minutes)
```bash
# 1. Understand the blocks
python3 diagnose_healing.py

# 2. Use emergency liquidation if needed
bash emergency_liquidate.sh

# 3. Bot restarts automatically
```

### Option C: Let System Handle It
- Switch trading regime from MICRO_SNIPER → NORMAL
- LiquidationAgent will automatically consolidate positions
- **Time:** 10-15 minutes

---

## Current Status

**Portfolio:** 28 active positions (all dust)  
**NAV:** $41.70  
**Blockage:** Kill-switch (waiting for consolidation)

**Which tool to use?**
- If you want **immediate** results → `force_liquidate_dust.py execute`
- If you want to **understand** the issue → `diagnose_healing.py`
- If you want **full recovery** → `bash emergency_liquidate.sh`
- If you want **system to handle it** → Switch regime + wait

---

## Next Step

**My recommendation:** Run the diagnostic first, then decide:

```bash
python3 diagnose_healing.py
```

This will show:
- Why consolidation is blocked
- What capital can be freed
- Suggested next actions

Then use the appropriate tool based on output.

---

## Git Status

After choosing action, commit with:
```bash
git add -A && git commit -m "Liquidate dust positions + resume trading

Used: [TOOL_NAME]
Result: Capital freed from X positions
Effect: Kill-switch should auto-disable
Status: Ready to resume trading"
```

**All 4 PRETRADE fixes are still in place - just need consolidation first!**
