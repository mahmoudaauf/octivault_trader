# 🧹 ALL DUST SCENARIOS - COMPLETE TAXONOMY

**Purpose:** Catalog every possible dust position scenario  
**Date:** April 27, 2026  
**Scope:** Entry, detection, handling, recovery, liquidation

---

## 📋 SCENARIO CATEGORIES

1. **Entry Scenarios** (How dust is created)
2. **Detection Scenarios** (When/how dust is identified)
3. **Accumulation Scenarios** (How dust builds up)
4. **Liquidation Scenarios** (How dust is removed)
5. **Recovery Scenarios** (How dust can recover)
6. **Edge Case Scenarios** (Unexpected situations)
7. **System State Scenarios** (Portfolio conditions)

---

## 1️⃣ ENTRY SCENARIOS (Dust Creation)

### Entry Scenario 1.1: Normal Market Slippage Entry Dust

**Conditions:**
- Entry planned at $20.00
- Market moves 2% during order placement
- Position fills at lower price

**Sequence:**
```
Entry Decision: BUY $20 ETHUSDT
    ↓
Place order: 0.00035 BTC
    ↓
Market moves down 2%
    ↓
Fill: 0.00035 × $41,300 = $14.45
    ↓
Result: DUST ($14.45 < $20)
```

**Probability:** HIGH (very common)  
**Recovery:** Possible if market recovers 38%+  
**Liquidation:** Forced at 20%+ loss if triggered

---

### Entry Scenario 1.2: Entry During Capital Shortage Entry Dust

**Conditions:**
- Capital depleted by previous losses
- Entry planned at $20, but only $15 available
- Must enter anyway

**Sequence:**
```
Available capital: $15.00 (was $50)
    ↓
Entry decision: BUY $20 → overrides to $15
    ↓
Fill: 0.00026 BTC @ $42,500 = $11.05
    ↓
Result: DUST ($11.05 < $20)
```

**Probability:** MEDIUM (after losses)  
**Recovery:** Difficult, capital too small  
**Liquidation:** Likely triggered immediately

---

### Entry Scenario 1.3: Lot-Step Rounding Entry Dust

**Conditions:**
- Exchange requires specific lot sizes
- Planned quantity gets rounded down
- Resulting value below dust threshold

**Sequence:**
```
Planned: 0.00035 BTC worth $20
    ↓
Exchange min lot: 0.0001 BTC per step
    ↓
Rounded: 0.0003 BTC (down from 0.00035)
    ↓
Fill: 0.0003 × $42,500 = $12.75
    ↓
Result: DUST ($12.75 < $20)
```

**Probability:** LOW (but systematic)  
**Recovery:** Depends on price recovery  
**Liquidation:** Potential if dust accumulates

---

### Entry Scenario 1.4: Fee-Induced Entry Dust

**Conditions:**
- Entry fills at exactly $20
- Trading fee (0.1%) applied
- Net value below threshold

**Sequence:**
```
Fill: 0.00035 × $42,500 = $14.875
    ↓
Trading fee: -$0.015 (0.1%)
    ↓
Net: $14.875 - $0.015 = $14.86
    ↓
Result: DUST ($14.86 < $20)
```

**Probability:** VERY HIGH (affects all trades)  
**Recovery:** Needs 34% price recovery  
**Liquidation:** Common trigger

---

### Entry Scenario 1.5: Flash Crash Entry Dust

**Conditions:**
- Sudden price spike/drop during entry
- 5-10% adverse price movement
- Entry catches the bottom/top

**Sequence:**
```
Market price: $42,500
    ↓
Flash crash: -8%
    ↓
Entry fills: $39,100
    ↓
Fill amount: 0.00051 BTC (more qty, lower price)
    ↓
Value: $20.00 initially
    ↓
Market recovers: +8%
    ↓
Result: Position now worth $21.60 (NOT dust!)
```

**Probability:** LOW (but recovers quickly)  
**Recovery:** YES (market usually recovers)  
**Liquidation:** Avoided if market bounces

---

### Entry Scenario 1.6: Cascade Entry Dust (Multiple Dust Entries)

**Conditions:**
- First entry creates dust
- Capital shrinks
- Second entry with less capital
- Creates worse dust

**Sequence:**
```
Entry 1: BUY $20 → fills $14 dust
    ↓
Capital: $50 → $50-$0.20 loss = $49.80
    ↓
Entry 2: BUY $20 → only $19.50 available
    ↓
Entry 2 fill: 0.00045 × $42,000 = $18.90
    ↓
Result: Entry 2 is DUST ($18.90 < $20)
    ↓
Entry 3: Capital even less, dust even worse
    ↓
Repeat: More dust each cycle
```

**Probability:** VERY HIGH (after losses)  
**Recovery:** Difficult, spiral downward  
**Liquidation:** Triggered when accumulates

---

### Entry Scenario 1.7: Dust Entry During Liquidation Phase

**Conditions:**
- Portfolio already 80%+ dust
- System in PHASE 2 liquidation
- New entry still executed
- Creates additional dust

**Sequence:**
```
Dust ratio: 85%
    ↓
Phase 2: Active liquidation
    ↓
New signal: BUY SOLANA
    ↓
Entry: BUY $20 → fills $17 dust
    ↓
Result: New dust added to existing 85%
    ↓
Dust ratio: 87%
    ↓
More liquidation triggered
```

**Probability:** MEDIUM (gate should prevent this)  
**Recovery:** No, will be liquidated  
**Liquidation:** Immediate (5+ minutes)

---

### Entry Scenario 1.8: Gap Open Entry Dust

**Conditions:**
- Market gaps overnight
- Significant price movement
- Entry fills at gap price (much worse)

**Sequence:**
```
Previous close: $42,500 BTC
    ↓
Overnight gap: DOWN 3%
    ↓
Gap open: $41,225
    ↓
Entry at gap: BUY $20 @ $41,225
    ↓
Fill: 0.00048 × $41,225 = $19.79
    ↓
Result: DUST ($19.79 < $20) - barely!
```

**Probability:** LOW but impactful  
**Recovery:** Depends on gap direction  
**Liquidation:** Triggered if accumulated

---

### Entry Scenario 1.9: Partial Fill Entry Dust

**Conditions:**
- Order placed for full amount
- Only partial fill received
- Must calculate based on partial qty

**Sequence:**
```
Order: BUY 0.00035 BTC @ $42,500
    ↓
Market: Only 0.00028 BTC available
    ↓
Fill: 0.00028 × $42,500 = $11.90
    ↓
Result: DUST ($11.90 < $20)
```

**Probability:** LOW (not common on Binance)  
**Recovery:** Depends on additional fills  
**Liquidation:** Likely if not completed

---

### Entry Scenario 1.10: Stale Price Entry Dust

**Conditions:**
- Price data lagging
- Entry made with old price
- Actual market much different
- Recompute with new price → dust

**Sequence:**
```
Price cached: $42,500 (5 min old)
    ↓
Entry amount: $20 / $42,500 = 0.00047
    ↓
Actual market: $41,800
    ↓
Real fill: 0.00047 × $41,800 = $19.65
    ↓
Result: DUST ($19.65 < $20)
```

**Probability:** MEDIUM (if data lag exists)  
**Recovery:** Possible with price recovery  
**Liquidation:** If detected during dust phase

---

## 2️⃣ DETECTION SCENARIOS (Dust Identification)

### Detection Scenario 2.1: Immediate Post-Fill Detection

**Conditions:**
- Position fills
- record_fill() called immediately
- Value calculated at fill time
- Dust detected on same cycle

**Sequence:**
```
Time: 13:05:30
├─ Fill received: 0.00035 BTC @ $42,500 = $14.875
├─ record_fill() called
├─ Position value calculated: $14.875
├─ Significant floor check: $14.875 < $20? YES
├─ Status set: DUST_LOCKED
└─ Dust registry updated
```

**Detection Time:** Immediate  
**State:** DUST_LOCKED (correct state)  
**Action:** Wait for recovery or liquidation phase

---

### Detection Scenario 2.2: Delayed Detection (After Portfolio Update)

**Conditions:**
- Position fills
- record_fill() doesn't run immediately
- Later portfolio scan detects it
- Classified as dust in next cycle

**Sequence:**
```
Time: 13:05:30 - Fill
├─ Position created: $14.875
├─ Status: UNKNOWN
│
Time: 13:10:00 - Portfolio scan
├─ Detect position value: $14.875
├─ Significant floor check: $14.875 < $20? YES
├─ Status updated: DUST_LOCKED
└─ Dust registry updated
```

**Detection Time:** 4+ minutes delayed  
**State:** Briefly unclassified  
**Action:** Eventually liquidated in next phase

---

### Detection Scenario 2.3: Recovered Dust Detection (Not Dust Anymore)

**Conditions:**
- Position marked dust
- Market price recovered significantly
- Price now > $20
- Dust detection checks and finds position recovered

**Sequence:**
```
Time: 13:05:00
├─ Fill: 0.00035 BTC @ $42,500 = $14.875 → DUST
│
Time: 13:15:00
├─ Market now: $43,500 (+2.3%)
├─ Position value: 0.00035 × $43,500 = $15.225 → Still DUST
│
Time: 13:25:00
├─ Market now: $60,000 (+41%)
├─ Position value: 0.00035 × $60,000 = $21.00 → NOT DUST! ✓
├─ Status updated: ACTIVE
└─ Removed from dust registry
```

**Detection Time:** At recovery price > $20  
**State:** Changed from DUST_LOCKED to ACTIVE  
**Action:** Can now be traded normally

---

### Detection Scenario 2.4: Zombie Dust Detection

**Conditions:**
- Position marked dust years ago
- Still in system
- Unlikely to ever recover
- System detects it as "permanent dust"

**Sequence:**
```
Position: ETHUSDT dust from 6 months ago
├─ Value: $8.50
├─ Entry price: $2,000
├─ Current price: $60,000
├─ Expected recovery: Would need price = $5,714,286 (impossible)
│
System check: This will NEVER recover
├─ Mark as: PERMANENT_DUST
├─ Retire: Exclude from all operations
└─ Close: Write down as loss
```

**Detection Time:** Periodic review  
**State:** PERMANENT_DUST (irreversible)  
**Action:** Close/liquidate immediately

---

### Detection Scenario 2.5: Dust During Circuit Breaker

**Conditions:**
- Exchange enters circuit breaker mode
- Trading halted
- Positions can't be modified
- But dust status needs detection

**Sequence:**
```
Exchange halt: 5% down circuit breaker triggered
    ↓
Trading suspended: No buys/sells allowed
    ↓
Position registered as dust: 0.00035 BTC @ $14.875
    ↓
But can't liquidate (market closed)
    ↓
Detection: YES, dust detected
    ↓
Action: HOLD (can't liquidate during halt)
```

**Detection Time:** Immediate  
**State:** DUST_LOCKED but FROZEN  
**Action:** Liquidate when market resumes

---

## 3️⃣ ACCUMULATION SCENARIOS (Dust Buildup)

### Accumulation Scenario 3.1: Single Dust Accumulation

**Conditions:**
- One dust position created
- Portfolio: 1 total, 1 dust = 100% dust
- Immediately triggers Phase 2

**Sequence:**
```
Portfolio before: 0 positions
    ↓
Entry: BUY → fills as dust
    ↓
Portfolio after: 1 total, 1 dust
    ↓
Dust ratio: 1/1 = 100%
    ↓
Phase 2 check: YES (100% > 60%)
    ↓
Age check: Wait 5 minutes
    ↓
Action: Liquidate at 5+ minutes
```

**Duration:** 5+ minutes  
**Ratio:** 100%  
**Trigger:** IMMEDIATE (next cycle after 5 min)

---

### Accumulation Scenario 3.2: Gradual Dust Accumulation (Slow Build)

**Conditions:**
- Multiple entries over time
- Some succeed, some become dust
- Dust ratio slowly increases
- Accumulates to critical level

**Sequence:**
```
Hour 1:
├─ Entry 1: SUCCESS (active position)
├─ Entry 2: DUST (fills $15)
└─ Dust ratio: 1/2 = 50%

Hour 2:
├─ Entry 3: DUST (fills $17)
├─ Dust ratio: 2/3 = 67%
└─ Phase 2: TRIGGERED (>60%)

Hour 2:05
├─ Liquidation: DUST positions sold
├─ Entry 1: Still active
└─ Dust ratio: 1/1 = 100% (if liquidation failed)
```

**Duration:** 1-2 hours  
**Ratio:** 50% → 67% → 100%  
**Trigger:** When exceeds 60%

---

### Accumulation Scenario 3.3: Rapid Dust Accumulation (Fast Build)

**Conditions:**
- Multiple entries in quick succession
- All become dust
- Ratio exceeds threshold in seconds
- Cascade effect

**Sequence:**
```
Second 0: Entry 1 fills as dust
Second 5: Entry 2 fills as dust → ratio 100%
Second 10: Entry 3 fills as dust → ratio 100%
Second 15: Dust ratio check triggered
...
Second 300: Phase 2 liquidation starts
```

**Duration:** 5+ minutes (must wait 300s)  
**Ratio:** 50% → 100% (instantly)  
**Trigger:** After 5 minute gate passes

---

### Accumulation Scenario 3.4: Partial Recovery During Accumulation

**Conditions:**
- Dust accumulating
- Price recovers partially
- Some dust positions recover above $20
- Dust ratio changes

**Sequence:**
```
Time 0:
├─ Position 1: DUST ($14)
├─ Position 2: DUST ($16)
├─ Dust ratio: 100%
│
Time 5 min:
├─ Market +5%
├─ Position 1: $14.70 (still DUST)
├─ Position 2: $16.80 (still DUST)
├─ Dust ratio: Still 100%
│
Time 10 min:
├─ Market +15%
├─ Position 1: $16.10 (still DUST)
├─ Position 2: $18.40 (still DUST)
├─ Dust ratio: Still 100%
│
Time 15 min:
├─ Market +40%
├─ Position 1: $19.60 (CLOSE but DUST)
├─ Position 2: $22.40 (NOT DUST) ✓
├─ Dust ratio: 1/2 = 50%
└─ Phase 2: Can deactivate
```

**Duration:** 15+ minutes  
**Ratio:** 100% → 50% (after recovery)  
**Recovery:** YES if price moves 40%+

---

### Accumulation Scenario 3.5: New Entry During Liquidation Phase

**Conditions:**
- Phase 2 liquidation active
- New signal arrives
- Entry executed anyway
- Adds to dust count being liquidated

**Sequence:**
```
State: Phase 2 active, liquidating 5 dust positions
    ↓
Time: 10:05:00 - New BUY signal arrives
    ↓
Gate check: Passes confidence/capital gates
    ↓
Execution: Entry placed
    ↓
Fill: Position fills as DUST ($16)
    ↓
Dust count: 5 → 6
    ↓
Liquidation: New dust also liquidated
```

**Duration:** 5+ minutes until liquidation  
**Ratio:** Increases by 1 position  
**Result:** New dust immediately scheduled for liquidation

---

## 4️⃣ LIQUIDATION SCENARIOS (Dust Removal)

### Liquidation Scenario 4.1: Forced Immediate Liquidation

**Conditions:**
- Dust ratio > 60%
- Age check passed (5+ minutes)
- Phase 2 activated
- All dust sells immediately

**Sequence:**
```
Dust detected: 3 positions, 100% dust ratio
    ↓
Time: 10:05:00 - Phase 2 triggered
    ↓
Generate SELL signals: 3 dust SELL orders
    ↓
Execution: All 3 sell immediately
    ↓
Results:
├─ Dust 1: $14 → $13.80 (loss $0.20)
├─ Dust 2: $16 → $15.75 (loss $0.25)
└─ Dust 3: $15 → $14.70 (loss $0.30)
```

**Loss per position:** 1-3%  
**Total capital lost:** $0.75 (1.5%)  
**Time:** Immediate after 5 min wait

---

### Liquidation Scenario 4.2: Partial Liquidation (Age-Based)

**Conditions:**
- Dust positions of varying ages
- Only liquidate positions > 1 hour old (with age guard)
- Keep fresh dust

**Sequence:**
```
Dust positions:
├─ Position 1: 45 minutes old, value $14 → KEEP (fresh)
├─ Position 2: 2 hours old, value $15 → LIQUIDATE (mature)
└─ Position 3: 1.5 hours old, value $16 → LIQUIDATE (mature)

Action:
├─ Position 1: Skip (not old enough)
├─ Position 2: SELL → $14.85 (recovers capital)
└─ Position 3: SELL → $15.85 (recovers capital)

Result: Capital recovered, fresh dust kept for recovery
```

**Loss:** Only on mature dust  
**Capital recovered:** ~$30.70 (from 2 positions)  
**Fresh dust:** Kept for potential recovery

---

### Liquidation Scenario 4.3: Natural Liquidation (Price-Based Exit)

**Conditions:**
- Dust position recovers to > $20
- Position no longer dust
- Normal TP/SL can exit it
- No forced liquidation needed

**Sequence:**
```
Position: DUST at $14 (0.00035 BTC @ $42,500)
    ↓
Market: +40%
    ↓
Current value: 0.00035 × $59,500 = $20.825
    ↓
Status: NO LONGER DUST ✓
    ↓
Normal trading: Can set TP at $21, SL at $19
    ↓
Market hits TP: Sell at $21 profit
    ↓
Result: Position exited profitably!
```

**Exit price:** $21  
**Profit:** $7 (33% gain)  
**Result:** Converts dust to profit

---

### Liquidation Scenario 4.4: Defensive Liquidation (Reducing Risk)

**Conditions:**
- Market crashes
- Dust position value drops further
- Must liquidate to preserve capital
- Accept small loss vs bigger loss

**Sequence:**
```
Dust position: $14 (down from $20 planned)
    ↓
Market crashes: -5%
    ↓
Position now: $13.30
    ↓
Signal: Market going lower
    ↓
Decision: Liquidate now ($13.30) vs wait ($12.50+)
    ↓
Action: SELL at $13.30
    ↓
Result: Preserve $13.30 vs potential $12 loss
```

**Liquidation price:** $13.30 (vs could be $12)  
**Loss prevented:** ~$1.30  
**Rationale:** Accept small loss to avoid bigger loss

---

### Liquidation Scenario 4.5: Opportunistic Liquidation (Price Bounce)

**Conditions:**
- Dust position
- Price bounces intraday
- Brief recovery to near $20
- Liquidate on the bounce

**Sequence:**
```
Dust: $14 value
    ↓
Market: Brief rally
    ↓
Bounce: +6%
    ↓
Position: $14 × 1.06 = $14.84
    ↓
Still dust: $14.84 < $20
    ↓
But: Good opportunity to reduce loss
    ↓
Action: SELL on bounce at $14.84
    ↓
Result: Recover capital faster, reduce exposure
```

**Liquidation price:** $14.84 (vs $14)  
**Capital recovered:** +$0.84 better  
**Timing:** Requires active monitoring

---

### Liquidation Scenario 4.6: Forced Portfolio Liquidation

**Conditions:**
- Multiple dust positions
- Capital needed urgently
- System in distress
- Liquidate ALL dust at market

**Sequence:**
```
Portfolio state: 8 dust positions, 0 active
    ↓
Capital level: $5 (critical low)
    ↓
Need: Capital for margin/emergency
    ↓
Action: Emergency liquidation
    ↓
Liquidate: All 8 dust positions at market
    ↓
Proceeds: ~$112 (including slippage)
    ↓
Result: Recover capital, accept losses
```

**Liquidation:** All dust at once  
**Capital recovered:** ~$112 from $120 dust  
**Loss:** -$8 (6.7%)

---

### Liquidation Scenario 4.7: Cascade Liquidation (Chain Reaction)

**Conditions:**
- Liquidate first dust position
- Capital freed
- New entry with capital
- Creates dust again
- Liquidates again (repeat)

**Sequence:**
```
Cycle 1:
├─ Dust 1: $14 liquidated → capital freed
├─ Entry: BUY $20 → fills $15 dust
├─ Dust count: Still 1
│
Cycle 2:
├─ Dust 2: $15 liquidated
├─ Entry: BUY $20 → fills $14 dust
├─ Dust count: Still 1
│
Repeat: Capital shrinking each cycle ($0.50 loss per cycle)
```

**Pattern:** Liquidate → Enter → Dust → Repeat  
**Capital decay:** -$0.50 per cycle  
**Duration:** Continuous

---

## 5️⃣ RECOVERY SCENARIOS (Dust Healing)

### Recovery Scenario 5.1: Natural Price Recovery (Market Bounce)

**Conditions:**
- Dust position created
- Market reverses
- Price recovers above $20
- Position exits dust status

**Sequence:**
```
Entry: $14 dust (0.00035 BTC @ $42,500)
    ↓
Market crashes: -5% → $40,375
    ↓
Position: $14.13 (worse, still dust)
    ↓
Market bounces: +50% → $60,375
    ↓
Position: $21.13 (recovered above $20!)
    ↓
Status: ACTIVE (no longer dust)
    ↓
Result: Can now trade normally
```

**Recovery price needed:** > $20 / 0.00035 = $57,143  
**Price movement needed:** +34% from original  
**Duration:** Could take days/weeks

---

### Recovery Scenario 5.2: Healing with Averaging Down

**Conditions:**
- Dust position at $14
- Add capital
- Buy more at lower price
- Average up position value
- Reach above dust threshold

**Sequence:**
```
Position 1: 0.00035 BTC @ $42,500 = $14.875 (DUST)
    ↓
Market: $40,000 (down 6%)
    ↓
Add capital: BUY $20 more @ $40,000
    ↓
Position 2: 0.0005 BTC @ $40,000 = $20.00
    ↓
Combined: 0.00085 BTC worth = $34.00
    ↓
Average price: $34/0.00085 = $40
    ↓
Status: ACTIVE (total > $20)
```

**Capital added:** $20  
**Total position:** $34  
**Result:** Rescue from dust by adding capital

---

### Recovery Scenario 5.3: Healing with Rebalancing

**Conditions:**
- Dust position exists
- Liquidate different position
- Use capital to consolidate dust
- Merge into larger position

**Sequence:**
```
Portfolio:
├─ ETHUSDT dust: $14
├─ SOLANA: $25 (active)

Rebalance:
├─ SELL SOLANA: Get $25 capital
├─ BUY more ETHUSDT: $25 + $14 = $39 total
├─ New ETHUSDT: $39 (no longer dust)
└─ SOLANA: Closed

Result: Consolidated dust into active position
```

**Capital: Moved:** $25 from SOLANA to ETHUSDT  
**Dust status:** Resolved  
**Portfolio:** 1 active instead of 1 dust + 1 active

---

### Recovery Scenario 5.4: Dust Healing Without Entry

**Conditions:**
- Dust position exists
- Market gradually recovers
- No new entries needed
- Dust naturally heals

**Sequence:**
```
Week 1: Dust created at $14
    ├─ Market: Stable, no movement
    └─ Value: $14 (dust remains)

Week 2: Market +3%
    ├─ Position: $14.42 (still dust)
    └─ Value: $14.42

Week 3: Market +8% total
    ├─ Position: $15.12 (still dust)
    └─ Value: $15.12

Week 4: Market +30% total
    ├─ Position: $18.20 (still dust, but close)
    └─ Value: $18.20

Week 5: Market +40% total
    ├─ Position: $19.60 (almost recovered!)
    └─ Value: $19.60

Week 6: Market +45% total
    ├─ Position: $20.30 (RECOVERED!)
    └─ Status: ACTIVE ✓
```

**Time needed:** 6 weeks  
**Price movement needed:** +45%  
**Capital required:** None (passive recovery)

---

### Recovery Scenario 5.5: Dust Recovery via TP/SL Exit

**Conditions:**
- Dust position
- Set TP slightly above dust level
- Wait for market to hit TP
- Exit at minimal gain

**Sequence:**
```
Position: DUST at $14
    ↓
Set: TP at $14.50 (3.6% gain)
    ↓
Set: SL at $13.50 (3.6% loss)
    ↓
Wait: For market to hit either TP or SL
    ↓
Market: Recovers to $14.50
    ↓
Action: TP triggered → Position closes
    ↓
Recover capital: $14.50 (lose $0.375 from dust)
```

**Exit price:** $14.50  
**Gain:** +$0.50 (+3.6%)  
**Duration:** Could be hours/days

---

### Recovery Scenario 5.6: Dust Conversion to Stablecoin

**Conditions:**
- Dust position in volatile asset
- Convert to stablecoin
- Reduce volatility
- Wait for stable recovery

**Sequence:**
```
Position: ETHUSDT dust at $14
    ↓
Swap: ETHUSDT → USDT stablecoin
    ↓
Capital: $14.00 USDT
    ↓
Wait: Market recovers
    ↓
When: Market +40%
    ↓
Swap back: $14.00 USDT → 0.00035 ETHUSDT
    ↓
Position: $19.60 worth (still dust but better)
```

**Capital preserved:** $14 exactly  
**Volatility reduced:** Converted to stablecoin  
**Timing:** Can optimize entry back

---

## 6️⃣ EDGE CASE SCENARIOS (Unusual Situations)

### Edge Case Scenario 6.1: Zombie Dust (Forever Dust)

**Conditions:**
- Dust position created years ago
- Exchange delisted the token
- Position can't be liquidated
- Trapped permanently

**Sequence:**
```
2023: Position created DUST ($14)
    ↓
2024: Token still trading, dust remains
    ↓
2025: Token delisted by exchange
    ↓
2026: Position locked, can't trade
    ↓
Status: PERMANENT_DUST (irreversible)
    ↓
Result: Write down as permanent loss
```

**Capital lost:** $14 (permanent)  
**Recovery:** Impossible  
**Action:** Close as dead capital

---

### Edge Case Scenario 6.2: Phantom Dust (No Quantity)

**Conditions:**
- Dust position created
- Quantity becomes 0 somehow
- Position shows dust but no asset
- Status mismatch

**Sequence:**
```
Position: DUST (0.00035 BTC)
    ↓
Market issue: Position qty reset to 0
    ↓
System check: qty = 0, status = DUST
    ↓
Contradiction: Can't be dust with 0 qty
    ↓
Result: PHANTOM DUST (invalid state)
    ↓
Action: Clean up, mark as CLOSED
```

**Capital: Lost?:** $14 unrecovered (unclear state)  
**Status:** Requires manual intervention  
**Recovery:** Possible via account history

---

### Edge Case Scenario 6.3: Negative Dust (Shorting Dust)

**Conditions:**
- Short position created
- Fills above short price
- Short becomes dust (loss instead of gain)
- Negative position value

**Sequence:**
```
Short entry: SELL 0.00035 BTC @ $42,500
    ↓
Market moves: UP 2%
    ↓
Fill: 0.00035 × $43,500 = $15.225 (loss)
    ↓
Status: SHORT DUST (losing, not gaining)
    ↓
Value: -$0.35 (negative)
    ↓
Result: Dust in reverse
```

**Capital at risk:** Unlimited (short)  
**Loss:** -$0.35 (and could get worse)  
**Recovery:** Must cover short

---

### Edge Case Scenario 6.4: Dust in Forced Liquidation

**Conditions:**
- Margin call triggered
- Exchange force liquidates positions
- Dust position included
- Liquidated at market price

**Sequence:**
```
Margin level: 2.0x (critical)
    ↓
Exchange margin call: Liquidate immediately
    ↓
Force liquidate: ALL positions including dust
    ↓
Dust position: $14 sold at market
    ↓
Slippage: -2% due to forced sale
    ↓
Proceeds: $13.72
    ↓
Result: Position liquidated, capital recovered
```

**Loss:** -$0.28 (2%)  
**Recovery:** Capital received ($13.72)  
**Duration:** Instant (forced)

---

### Edge Case Scenario 6.5: Dust During Trading Halt

**Conditions:**
- Dust position created
- Exchange halts trading (maintenance)
- Can't liquidate or trade
- Position frozen in dust state

**Sequence:**
```
Dust created: $14
    ↓
Exchange maintenance: Trading halted 1 hour
    ↓
Can't sell, can't modify
    ↓
Position frozen: DUST_LOCKED but can't act
    ↓
Market moves: +5% during halt
    ↓
When resumed: Position worth $14.70
    ↓
Result: Dust could have recovered but frozen
```

**Capital trapped:** $14  
**Opportunity lost:** Market moved +5%  
**Action:** Liquidate when market resumes

---

### Edge Case Scenario 6.6: Dust with Extreme Leverage

**Conditions:**
- Dust position created on margin
- Leverage 5x or higher
- Position value at risk
- Liquidation cascade possible

**Sequence:**
```
Dust position: $14 (5x leverage)
    ↓
Margin required: $14/5 = $2.80
    ↓
Market drops: -3%
    ↓
Position: $13.58
    ↓
Margin loss: -$0.42
    ↓
Account margin: Falls below 5% threshold
    ↓
Result: Forced liquidation (cascade)
```

**Initial loss:** -$0.42  
**Risk:** Cascade liquidation of other positions  
**Outcome:** Amplified losses

---

### Edge Case Scenario 6.7: Dust Recovery During Flash Crash

**Conditions:**
- Dust position
- Flash crash occurs (sudden spike)
- Price recovers instantly
- Position briefly above dust

**Sequence:**
```
Dust: $14 (0.00035 BTC @ $42,500)
    ↓
Flash: Market UP 50% to $63,750
    ↓
Position: 0.00035 × $63,750 = $22.3125 (NOT DUST!)
    ↓
Duration: 2 seconds
    ↓
Market recovers: Back to $42,500
    ↓
Position: Back to $14.875 (DUST again)
    ↓
Result: Brief moment of recovery, missed opportunity
```

**Recovery seen:** $22.31 (3 seconds)  
**Permanent recovery:** No  
**Opportunity:** Extremely difficult to catch

---

### Edge Case Scenario 6.8: Dust Split Across Multiple Accounts

**Conditions:**
- Sub-account dust transfer
- Dust position split
- Each portion still dust
- Recombining possible

**Sequence:**
```
Original dust: 0.00035 BTC worth $14

Split:
├─ Account A: 0.00017 BTC worth $7 (DUST)
└─ Account B: 0.00018 BTC worth $7.20 (DUST)

Recombine:
├─ Transfer B to A: 0.00035 BTC
├─ Total: $14.20 (still DUST, but consolidated)
└─ Result: Single dust position

Better strategy: Combine into $20+ position
```

**After split:** 2 dust positions  
**After recombine:** 1 dust position  
**Recovery:** Possible if combined > $20

---

## 7️⃣ SYSTEM STATE SCENARIOS (Portfolio Conditions)

### System State Scenario 7.1: Clean Portfolio (No Dust)

**Conditions:**
- All positions > $20
- No dust positions
- Capital efficiently used
- Normal trading operations

**Characteristics:**
```
Portfolio:
├─ ETHUSDT: $50 (ACTIVE)
├─ BNBUSDT: $35 (ACTIVE)
└─ SOLANA: $42 (ACTIVE)

Metrics:
├─ Total capital: $127
├─ Dust count: 0
├─ Dust ratio: 0%
├─ Available capital: Good
└─ Status: HEALTHY ✓
```

**Dust ratio:** 0%  
**Liquidation risk:** None  
**Trading ability:** Full

---

### System State Scenario 7.2: Dusty Portfolio (5-20% Dust)

**Conditions:**
- 1-2 dust positions
- Majority still active
- Some capital tied up in dust
- Liquidation not required yet

**Characteristics:**
```
Portfolio:
├─ ETHUSDT: $50 (ACTIVE)
├─ BNBUSDT: $35 (ACTIVE)
├─ DUST_1: $14 (DUST)
└─ DUST_2: $16 (DUST)

Metrics:
├─ Total capital: $115
├─ Dust: $30
├─ Dust ratio: 26%
├─ Available capital: $35
└─ Status: CAUTION ⚠️
```

**Dust ratio:** 26%  
**Liquidation risk:** Low  
**Trading ability:** Reduced capital

---

### System State Scenario 7.3: Heavily Dusty Portfolio (60-80% Dust)

**Conditions:**
- Multiple dust positions
- Most capital in dust
- Phase 2 likely triggered
- Liquidation imminent

**Characteristics:**
```
Portfolio:
├─ ACTIVE: $25 (ACTIVE)
├─ DUST_1: $14 (DUST)
├─ DUST_2: $16 (DUST)
├─ DUST_3: $15 (DUST)
└─ DUST_4: $13 (DUST)

Metrics:
├─ Total capital: $83
├─ Dust: $58
├─ Dust ratio: 70%
├─ Available capital: Low
└─ Status: CRITICAL ⛔
```

**Dust ratio:** 70%  
**Liquidation risk:** HIGH (Phase 2 active)  
**Trading ability:** Severely limited

---

### System State Scenario 7.4: All Dust Portfolio (100% Dust)

**Conditions:**
- All positions are dust
- No active positions
- All capital tied in dust
- Immediate liquidation needed

**Characteristics:**
```
Portfolio:
├─ DUST_1: $14
├─ DUST_2: $16
├─ DUST_3: $15
└─ DUST_4: $13

Metrics:
├─ Total capital: $58
├─ Dust: $58
├─ Dust ratio: 100%
├─ Available capital: $0
└─ Status: EMERGENCY 🔥
```

**Dust ratio:** 100%  
**Liquidation risk:** IMMEDIATE  
**Trading ability:** Frozen (can only liquidate)

---

### System State Scenario 7.5: Recovery Portfolio (Dust Healing)

**Conditions:**
- Mostly dust initially
- Market recovered 30%+
- Dust positions recovering
- Transitioning to active

**Characteristics:**
```
Portfolio:
├─ DUST_1: $14 → $19 (near recovery)
├─ DUST_2: $16 → $21.50 (recovered!)
├─ DUST_3: $15 → $20.25 (recovered!)
└─ DUST_4: $13 → $18.20 (near recovery)

Metrics:
├─ Total capital: $78.95
├─ Dust: $37.20 (DUST_1, DUST_4)
├─ Active: $41.75 (DUST_2, DUST_3)
├─ Dust ratio: 47%
└─ Status: IMPROVING ✓
```

**Dust ratio:** 100% → 47% (recovering)  
**Liquidation risk:** Decreasing  
**Trading ability:** Improving

---

### System State Scenario 7.6: Liquidation Portfolio (Phase 2 Active)

**Conditions:**
- Phase 2 dust liquidation running
- Dust being force-sold
- Capital being recovered
- Transitional state

**Characteristics:**
```
Portfolio (before Phase 2):
├─ DUST_1: $14
├─ DUST_2: $16
├─ DUST_3: $15

Portfolio (during Phase 2):
├─ DUST_1: (selling → $13.80)
├─ DUST_2: (selling → $15.85)
├─ DUST_3: (selling → $14.70)

Portfolio (after Phase 2):
├─ Capital recovered: $44.35
├─ Loss: -$0.65
└─ Status: RESET
```

**Dust ratio:** 100% → 0%  
**Capital recovered:** $44.35  
**Duration:** Seconds to minutes

---

### System State Scenario 7.7: Bootstrap Portfolio (Small Capital)

**Conditions:**
- Very small account (<$50)
- Each position small
- Dust threshold relative
- Most positions dust-like

**Characteristics:**
```
Portfolio:
├─ ETHUSDT: $8 (< dust threshold)
├─ BNBUSDT: $7 (< dust threshold)
├─ SOLANA: $9 (< dust threshold)
├─ RIPPLE: $6 (< dust threshold)

Metrics:
├─ Total capital: $30
├─ Effective dust: $30 (all below $20)
├─ Available capital: $0
└─ Status: FRAGMENTED 🧩
```

**Dust ratio:** 100%+ (all positions small)  
**Liquidation risk:** Continuous  
**Trading ability:** Can't enter $20+ positions

---

## 📊 SCENARIO FREQUENCY & IMPACT TABLE

| Scenario | Frequency | Impact | Recovery | Liquidation |
|----------|-----------|--------|----------|-------------|
| Normal slippage dust | Very High | -0.5% | Moderate | Yes |
| Capital shortage dust | High | -1-2% | Difficult | Yes |
| Lot-step rounding dust | High | -2-5% | Depends | Yes |
| Fee-induced dust | Very High | -0.1% | Moderate | Yes |
| Flash crash dust | Low | -5% | Yes (quick) | Maybe |
| Cascade dust | High | Spiral | Very Hard | Yes |
| Zombie dust | Low | Permanent | Never | Never |
| Phantom dust | Very Low | Unclear | Depends | Manual |
| Dust during halt | Medium | Trapped | Delayed | On resume |
| Recovery phase | Medium | +20-40% | Yes! | No |

---

## 🎯 KEY INSIGHTS

### Most Common Scenarios
1. **Normal slippage dust** - Happens on nearly every trade
2. **Fee-induced dust** - Affects all trades 
3. **Cascade dust** - After losses, capital shrinks
4. **Liquidation phase** - Triggered regularly when dust > 60%

### Most Damaging Scenarios
1. **All-dust portfolio** - Capital locked, no trading
2. **Death spiral** - Dust accumulates, capital decays
3. **Forced liquidation** - Loses 2-3% per cycle
4. **Zombie dust** - Permanent capital loss

### Most Recoverable Scenarios
1. **Natural price recovery** - Market moves restore positions
2. **Averaging down** - Add capital to recover
3. **Recovery phase** - Market helps dust heal
4. **TP/SL exits** - Planned exits work

### Rarest but Critical Scenarios
1. **Circuit breaker halt** - Trading frozen
2. **Flash crash recovery** - Huge brief recovery
3. **Exchange delisting** - Permanent loss
4. **Force margin liquidation** - Cascade effects

---

## 🛑 CONCLUSION

The system can encounter **dozens of dust scenarios** ranging from:
- **Simple** (normal slippage) to **Complex** (cascade spirals)
- **Common** (every session) to **Rare** (emergency situations)
- **Recoverable** (market correction) to **Permanent** (delisting)
- **Preventable** (better validation) to **Unavoidable** (market conditions)

Most scenarios are **fixable** with:
1. Pre-execution validation (prevent entry dust)
2. Age guards on liquidation (allow recovery time)
3. Raising liquidation triggers (80% instead of 60%)
4. Portfolio management (consolidation, healing)

Without fixes: System trapped in **dust death spiral** (capital decays each cycle)

With fixes: System becomes **resilient to dust** (capital grows, dust recovers or liquidates profitably)
