# 🎯 OCTI AI TRADING BOT - TECHNICAL DECISION FLOWS

**Reference Document** | **For:** Developers & Advanced Operators  
**Date:** 2026-02-14

---

## MAIN DECISION FLOW (Per-Cycle)

```
START CYCLE
    │
    ├─→ [1] TICK INCREMENT
    │   └─ Increment cycle counter
    │   └─ Record cycle start time
    │
    ├─→ [2] DRAIN MARKET EVENTS
    │   ├─ Process price updates
    │   ├─ Process fill notifications
    │   ├─ Update position snapshots
    │   └─ Recalculate metrics
    │
    ├─→ [3] GUARD EVALUATION (All Must Pass)
    │   │
    │   ├─ Guard 1: Market Data Ready?
    │   │   ├─ ✓ Have price for each symbol
    │   │   └─ ✓ Price age < 5 seconds
    │   │   ├─ ✗ SKIP CYCLE → [END]
    │   │
    │   ├─ Guard 2: Balances Available?
    │   │   ├─ ✓ USDT balance > 0
    │   │   └─ ✓ Free balance > MIN_CAPITAL
    │   │   ├─ ✗ TRIGGER RECOVERY → [END]
    │   │
    │   ├─ Guard 3: Ops Plane Ready?
    │   │   ├─ ✓ Exchange client functional
    │   │   └─ ✓ Order placement succeeding
    │   │   ├─ ✗ HALT TRADING → [END]
    │   │
    │   ├─ Guard 4: Trading Hours Valid?
    │   │   ├─ ✓ Within allowed hours (24/7 for crypto)
    │   │   └─ ✓ Not in maintenance window
    │   │   ├─ ✗ SKIP CYCLE → [END]
    │   │
    │   ├─ Guard 5: Position Constraints Met?
    │   │   ├─ ✓ Total open < max for regime
    │   │   └─ ✓ Concentration ratio < limit
    │   │   ├─ ✗ SKIP CYCLE → [END]
    │   │
    │   └─ Guard 6: Capital Adequacy?
    │       ├─ ✓ Free capital >= MIN_CAPITAL
    │       └─ ✓ No forced recovery in progress
    │       ├─ ✗ TRIGGER RECOVERY → [END]
    │
    ├─→ [4] SIGNAL INTAKE & FILTERING
    │   ├─ Fetch new signals from agents
    │   ├─ Apply confidence floor: conf >= 0.50
    │   ├─ Age filter: age < 60 seconds
    │   ├─ Deduplicate: 1 BUY + 1 SELL per symbol (highest conf)
    │   └─ Output: Processed signal batch
    │
    ├─→ [5] BATCH COLLECTION & SORTING
    │   ├─ Collect up to 50 signals
    │   ├─ Sort by confidence (highest first)
    │   └─ Ready for arbitration
    │
    ├─→ [6] FOR EACH SIGNAL IN BATCH
    │   │
    │   ├─ Signal: {symbol, side, confidence, expected_move, agent, ...}
    │   │
    │   ├─ [ARBITRATION] ─────────────────────────────────────────
    │   │ │
    │   │ ├─ GATE 1: Lifecycle State Check
    │   │ │   ├─ Get symbol_lifecycle[symbol]
    │   │ │   ├─ If DUST_HEALING and side==SELL
    │   │ │   │   └─ Check age >= 5 min → allow or skip
    │   │ │   ├─ If ROTATION_PENDING and side==DUST_HEALING
    │   │ │   │   └─ SKIP (state blocks authority)
    │   │ │   └─ Continue if NO state conflict
    │   │ │
    │   │ ├─ GATE 2: Portfolio Health Check
    │   │ │   ├─ Count bot-managed positions
    │   │ │   ├─ Classify as SIGNIFICANT vs DUST
    │   │ │   ├─ Check position count < regime_max
    │   │ │   └─ Check dust ratio < threshold
    │   │ │   └─ SKIP if portfolio full
    │   │ │
    │   │ ├─ GATE 3: Capital Availability Check
    │   │ │   ├─ Calculate free quote available
    │   │ │   │   └─ free = (balance - reserve) - allocated
    │   │ │   ├─ Resolve entry_quote via ScalingManager
    │   │ │   ├─ Check quote >= MIN_ENTRY_QUOTE
    │   │ │   └─ SKIP if insufficient capital
    │   │ │
    │   │ ├─ GATE 4: Economic Gate (Anti-Churn)
    │   │ │   ├─ Calculate round-trip cost
    │   │ │   │   └─ RT = (2 * taker_fee) + (2 * slippage)
    │   │ │   │       = (2 * 0.1%) + (2 * 0.15%) = 0.50%
    │   │ │   ├─ Calculate min profitable move
    │   │ │   │   └─ min_move = RT + safety_buffer
    │   │ │   │       = 0.50% + 0.05% = 0.55%
    │   │ │   ├─ Compare expected_alpha to min_move
    │   │ │   └─ SKIP if alpha < min_profitable
    │   │ │
    │   │ ├─ GATE 5: Signal Confidence Gate
    │   │ │   ├─ Get regime minimum confidence
    │   │ │   │   ├─ MICRO_SNIPER: 0.50
    │   │ │   │   ├─ STANDARD: 0.55
    │   │ │   │   └─ MULTI_AGENT: 0.60
    │   │ │   ├─ Check confidence >= regime_min
    │   │ │   └─ SKIP if below minimum
    │   │ │
    │   │ ├─ GATE 6: Regime Gating
    │   │ │   ├─ If MICRO_SNIPER (NAV < 1000)
    │   │ │   │   ├─ Max 1 position (if side==BUY)
    │   │ │   │   ├─ Max 1 symbol
    │   │ │   │   ├─ Disable rotation (no SELL unless SL/TP)
    │   │ │   │   └─ Disable dust healing
    │   │ │   ├─ If STANDARD (NAV 1000-5000)
    │   │ │   │   ├─ Max 2 positions
    │   │ │   │   ├─ Max 3 symbols
    │   │ │   │   ├─ Rotation enabled (60s cooldown)
    │   │ │   │   └─ Dust healing enabled
    │   │ │   ├─ If MULTI_AGENT (NAV >= 5000)
    │   │ │   │   ├─ Max 3+ positions
    │   │ │   │   ├─ Max 5+ symbols
    │   │ │   │   ├─ Rotation enabled (30s cooldown)
    │   │ │   │   └─ Dust healing enabled
    │   │ │   └─ SKIP if regime limits exceeded
    │   │ │
    │   │ └─ ALL GATES PASSED? → DECISION APPROVED ✓
    │   │    → SKIP? → Continue to next signal
    │   │
    │   ├─ [DECISION] ────────────────────────────────────────
    │   │ ├─ Determine action: BUY or SELL
    │   │ ├─ Calculate position size (confidence scaling)
    │   │ ├─ Create TradeIntent object
    │   │ └─ Queue for execution
    │   │
    │   └─ Loop to next signal
    │
    ├─→ [7] EXECUTION DISPATCH
    │   ├─ For each approved TradeIntent
    │   ├─ Route through Execution Logic
    │   ├─ Place order with ExecutionManager
    │   ├─ Wait for fill (non-blocking)
    │   └─ Update SharedState on completion
    │
    ├─→ [8] POST-EXECUTION BOOKKEEPING
    │   ├─ Record executed trades in history
    │   ├─ Update metrics (NAV, PnL, etc.)
    │   ├─ Update TP/SL levels for new positions
    │   ├─ Check for forced rotations
    │   └─ Reset per-cycle counters
    │
    ├─→ [9] CYCLE END
    │   ├─ Calculate cycle duration
    │   ├─ Log cycle summary
    │   └─ Record performance metrics
    │
    └─→ [END] → LOOP TO NEXT CYCLE
```

---

## BUY ORDER DECISION FLOW

```
Signal Arrives: BUY BTCUSDT conf=0.72 expected_move=0.35%
    │
    ├─→ [1] POSITION CHECK
    │   ├─ Does position already exist?
    │   │   ├─ YES: Check if accumulation allowed
    │   │   │   ├─ Single-position-per-symbol rule?
    │   │   │   └─ If YES → BLOCK (can't add more)
    │   │   └─ NO: Continue
    │   │
    │   └─ Reserve symbol (atomic lock)
    │       └─ Prevent concurrent BUY orders
    │
    ├─→ [2] CALCULATE POSITION SIZE
    │   │
    │   ├─ Base Quote: 25 USDT (from config)
    │   │
    │   ├─ Confidence Scaling:
    │   │   ├─ if conf <= 0.50 → size = 25 (minimum)
    │   │   ├─ if conf == 0.65 → size = 25 * 1.4 = 35
    │   │   ├─ if conf == 0.80 → size = 25 * 2.0 = 50 (capped)
    │   │   └─ if conf >= 1.00 → size = 50 (max cap)
    │   │
    │   ├─ Policy Nudge (from PolicyManager):
    │   │   ├─ NORMAL mode: multiplier = 1.0x → size = 50 USDT
    │   │   ├─ AGGRESSIVE mode: multiplier = 1.5x → size = 75 USDT (capped at max)
    │   │   └─ RECOVERY mode: multiplier = 0.5x → size = 25 USDT
    │   │
    │   └─ Final Quote: 50 USDT (with caps applied)
    │
    ├─→ [3] RESOLVE TO QUANTITY
    │   │
    │   ├─ Current Price: 42,500 USDT/BTC
    │   │
    │   ├─ Raw Quantity: 50 / 42,500 = 0.001176 BTC
    │   │
    │   ├─ Step Size Rounding: 0.0001 BTC
    │   │   ├─ Rounded down: 0.001 BTC = 42,500 USDT
    │   │   └─ Rounded up: 0.0012 BTC = 51,000 USDT (hits min notional?)
    │   │
    │   ├─ Min Notional Check: 10 USDT (exchange minimum)
    │   │   ├─ 0.0001 BTC @ 42,500 = 4.25 USDT < 10 → TOO SMALL
    │   │   ├─ 0.0006 BTC @ 42,500 = 25.5 USDT >= 10 → OK
    │   │   └─ Adjust qty UP to meet minimum
    │   │
    │   └─ Final Quantity: 0.0006 BTC (notional: 25.5 USDT)
    │
    ├─→ [4] CREATE TRADE INTENT
    │   │
    │   ├─ symbol: BTCUSDT
    │   ├─ side: BUY
    │   ├─ quantity: 0.0006
    │   ├─ planned_quote: 50 (original)
    │   ├─ confidence: 0.72
    │   ├─ agent: TrendHunter
    │   ├─ tag: meta/TrendHunter
    │   ├─ trace_id: cycle_12345_btcusdt_buy
    │   └─ policy_context: {mode: NORMAL, multiplier: 1.0}
    │
    ├─→ [5] SUBMIT ORDER
    │   │
    │   ├─ Call ExecutionManager.place_order()
    │   │   ├─ Submit to Binance API
    │   │   ├─ Order type: MARKET (immediate execution)
    │   │   └─ Receive order_id: 123456789
    │   │
    │   └─ Status: PLACED (waiting for fill)
    │
    ├─→ [6] MONITOR FILL
    │   │
    │   ├─ Poll order status every 100ms
    │   │   ├─ Status: PARTIALLY_FILLED (0.0005 BTC filled)
    │   │   ├─ Status: FILLED (0.0006 BTC filled)
    │   │   └─ Filled price: 42,510 USDT/BTC (slight slippage)
    │   │
    │   └─ When FILLED → Continue to bookkeeping
    │
    ├─→ [7] CALCULATE AVERAGE ENTRY
    │   │
    │   ├─ Total filled: 0.0006 BTC
    │   ├─ Total paid: 25.5 USDT
    │   └─ Avg entry price: 25.5 / 0.0006 = 42,500 USDT/BTC
    │
    ├─→ [8] SET TP/SL LEVELS
    │   │
    │   ├─ Entry price: 42,500 USDT/BTC
    │   │
    │   ├─ TP Level (2% profit):
    │   │   └─ 42,500 * 1.02 = 43,350 USDT/BTC
    │   │
    │   ├─ SL Level (-1% loss):
    │   │   └─ 42,500 * 0.99 = 42,075 USDT/BTC
    │   │
    │   └─ TP/SL Engine monitors these prices
    │
    ├─→ [9] UPDATE POSITION TRACKING
    │   │
    │   ├─ shared_state.positions[BTCUSDT]:
    │   │   ├─ qty: 0.0006 BTC
    │   │   ├─ entry_price: 42,500
    │   │   ├─ avg_price: 42,500
    │   │   ├─ status: OPEN
    │   │   ├─ tag: meta/TrendHunter
    │   │   └─ opened_at: 2026-02-14T12:34:56Z
    │   │
    │   └─ shared_state.open_trades[BTCUSDT]:
    │       ├─ symbol: BTCUSDT
    │       ├─ side: BUY
    │       ├─ quantity: 0.0006
    │       ├─ entry_price: 42,500
    │       ├─ opened_at: 2026-02-14T12:34:56Z
    │       └─ tag: meta/TrendHunter
    │
    ├─→ [10] UPDATE METRICS
    │   │
    │   ├─ NAV: 10,000 → 9,974.50 (paid 25.50 fees)
    │   ├─ Open positions: 0 → 1
    │   ├─ Position value: +25.50 USDT
    │   ├─ Unrealized PnL: 0 (just filled)
    │   └─ Trades executed: 0 → 1
    │
    ├─→ [11] RECORD IN HISTORY
    │   │
    │   ├─ trade_history.append({
    │   │     symbol: BTCUSDT,
    │   │     side: BUY,
    │   │     qty: 0.0006,
    │   │     entry_price: 42,500,
    │   │     filled_price: 42,510,
    │   │     timestamp: 2026-02-14T12:34:56Z,
    │   │     order_id: 123456789,
    │   │     trace_id: cycle_12345_btcusdt_buy,
    │   │     profit: 0,
    │   │   })
    │   │
    │   └─ Emit trade event to observers
    │
    ├─→ [12] RELEASE SYMBOL LOCK
    │   │
    │   └─ Allow concurrent operations on other symbols
    │
    └─→ [DONE] BUY ORDER COMPLETE ✓
```

---

## SELL ORDER DECISION FLOW

```
Signal Arrives: SELL BTCUSDT reason=TP_HIT tag=tp_sl
    │
    ├─→ [1] POSITION VALIDATION
    │   │
    │   ├─ Does position exist?
    │   │   ├─ YES, qty > 0 → Continue
    │   │   └─ NO → BLOCK (can't sell what we don't have)
    │   │
    │   ├─ Is position significant? (> $0.50 USDT value)
    │   │   ├─ YES → Continue
    │   │   └─ NO → Dust position, may skip
    │   │
    │   └─ Reserve symbol (atomic lock)
    │       └─ Prevent concurrent SELL orders
    │
    ├─→ [2] LIFECYCLE GATING
    │   │
    │   ├─ Get lifecycle state for BTCUSDT
    │   │   ├─ State: DUST_HEALING
    │   │   │   ├─ Check accumulated time:
    │   │   │   │   ├─ If < 5 min → SKIP (wait for recovery)
    │   │   │   │   └─ If >= 5 min → ALLOW (timeout expired)
    │   │   │   │
    │   │   ├─ State: ROTATION_PENDING
    │   │   │   ├─ Check age:
    │   │   │   │   ├─ If < 900s → SKIP (cooldown active)
    │   │   │   │   └─ If >= 900s → ALLOW (cooldown expired)
    │   │   │   │
    │   │   ├─ State: STRATEGY_OWNED
    │   │   │   └─ ALLOW (normal trading)
    │   │   │
    │   │   └─ No state (None)
    │   │       └─ ALLOW (no restrictions)
    │   │
    │   └─ If SKIP at this stage → Early exit
    │
    ├─→ [3] PROFIT GATE (meta_exit only, NOT for TP/SL)
    │   │
    │   ├─ Is this a forced exit (TP/SL/Recovery/Liquidation)?
    │   │   ├─ YES → Bypass profit gate
    │   │   └─ NO → Check profit gate below
    │   │
    │   ├─ Calculate entry price: 42,500 USDT/BTC
    │   ├─ Calculate current price: 42,750 USDT/BTC
    │   │
    │   ├─ Calculate P&L %:
    │   │   ├─ P&L = (42,750 - 42,500) / 42,500 = 0.588% profit
    │   │
    │   ├─ Calculate round-trip cost:
    │   │   ├─ Fee (entry + exit): 2 * 0.1% = 0.2%
    │   │   ├─ Slippage (entry + exit): 2 * 0.15% = 0.3%
    │   │   └─ Total RT cost: 0.5%
    │   │
    │   ├─ Min profit gate (covers costs):
    │   │   ├─ Min profit = RT cost * fee_multiplier
    │   │   ├─ Min profit = 0.5% * 2.0 = 1.0%
    │   │   ├─ Actual P&L (0.588%) < Min profit (1.0%)?
    │   │   │   └─ YES → BLOCK (not profitable enough)
    │   │   │   └─ NO → Continue
    │   │   └─ In this case: 0.588% < 1.0% → BLOCK ✗
    │   │
    │   └─ If profit gate blocks → SKIP (wait for better exit)
    │
    ├─→ [4] EXCURSION GATE (minimum price movement)
    │   │
    │   ├─ Calculate minimum required price excursion
    │   │   ├─ Tick size: 0.01 USDT
    │   │   ├─ Min tick move: 0.01 * 2 = 0.02 USDT (2 ticks)
    │   │   ├─ ATR (14-period): 150 USDT
    │   │   ├─ Min ATR move: 150 * 0.35 = 52.5 USDT
    │   │   ├─ Bid-ask spread: 0.5 USDT
    │   │   ├─ Min spread move: 0.5 * 3.0 = 1.5 USDT
    │   │   └─ Required threshold: max(0.02, 52.5, 1.5) = 52.5 USDT
    │   │
    │   ├─ Calculate actual excursion:
    │   │   ├─ Current price: 42,750 USDT/BTC
    │   │   ├─ Entry price: 42,500 USDT/BTC
    │   │   └─ Excursion: |42,750 - 42,500| = 250 USDT
    │   │
    │   ├─ Check if excursion >= threshold:
    │   │   ├─ 250 USDT >= 52.5 USDT? YES → Continue
    │   │   └─ 250 USDT < 52.5 USDT? NO → BLOCK
    │   │   └─ In this case: 250 >= 52.5 → PASS ✓
    │   │
    │   └─ If excursion gate passes → Continue to exit
    │
    ├─→ [5] POSITION CONSOLIDATION
    │   │
    │   ├─ Is this a full exit? (qty >= total_position)
    │   │   ├─ YES → Use close_position() method
    │   │   └─ NO → Use execute_quantity_sell() method
    │   │
    │   ├─ Get total position qty: 0.0006 BTC
    │   ├─ Signal qty: 0.0006 BTC (full exit)
    │   │
    │   └─ Full exit detected → Use close_position()
    │
    ├─→ [6] CREATE SELL TRADE INTENT
    │   │
    │   ├─ symbol: BTCUSDT
    │   ├─ side: SELL
    │   ├─ quantity: 0.0006 BTC (total position)
    │   ├─ tag: tp_sl (from signal)
    │   ├─ reason: TP_HIT (from signal)
    │   ├─ trace_id: cycle_12345_btcusdt_sell
    │   └─ is_liquidation: false
    │
    ├─→ [7] SUBMIT SELL ORDER
    │   │
    │   ├─ Call ExecutionManager.close_position()
    │   │   ├─ Submit market SELL order
    │   │   ├─ Target qty: 0.0006 BTC
    │   │   └─ Receive order_id: 987654321
    │   │
    │   └─ Status: PLACED (waiting for fill)
    │
    ├─→ [8] MONITOR FILL
    │   │
    │   ├─ Poll order status every 100ms
    │   │   ├─ Status: FILLED (0.0006 BTC sold)
    │   │   └─ Filled price: 42,740 USDT/BTC (slight slippage)
    │   │
    │   └─ When FILLED → Continue to bookkeeping
    │
    ├─→ [9] CALCULATE REALIZED PROFIT
    │   │
    │   ├─ Entry price: 42,500 USDT/BTC
    │   ├─ Exit price: 42,740 USDT/BTC
    │   ├─ Profit per BTC: 42,740 - 42,500 = 240 USDT
    │   ├─ Total profit: 240 * 0.0006 = 0.144 USDT
    │   ├─ Fees (round-trip): 25.5 * 0.002 = 0.051 USDT
    │   └─ Net realized P&L: 0.144 - 0.051 = +0.093 USDT ✓
    │
    ├─→ [10] UPDATE LIFECYCLE STATE
    │   │
    │   ├─ Clear position: qty = 0
    │   ├─ Set state: ROTATION_PENDING (cooldown 900s)
    │   ├─ Freeze dust healing: cooldown until +600s
    │   └─ Record exit time
    │
    ├─→ [11] UPDATE METRICS
    │   │
    │   ├─ NAV: 9,974.50 → 10,074.59 (realized profit)
    │   ├─ Realized P&L: 0 → +0.093 USDT
    │   ├─ Open positions: 1 → 0
    │   ├─ Win trades: 0 → 1
    │   ├─ Closed trades: 0 → 1
    │   └─ Win rate: N/A → 100% (1 win)
    │
    ├─→ [12] RECORD IN HISTORY
    │   │
    │   ├─ Update trade_history with exit:
    │   │   ├─ symbol: BTCUSDT
    │   │   ├─ side: SELL
    │   │   ├─ qty: 0.0006
    │   │   ├─ exit_price: 42,740
    │   │   ├─ profit: +0.093 USDT
    │   │   ├─ reason: TP_HIT
    │   │   ├─ hold_time: 120 seconds
    │   │   └─ exit_time: 2026-02-14T12:36:56Z
    │   │
    │   └─ Emit exit event to observers
    │
    ├─→ [13] SETUP REENTRY LOCK
    │   │
    │   ├─ Exit reason: TP_HIT
    │   ├─ Reentry lock duration:
    │   │   ├─ TP exit: +300s (30% of normal 900s)
    │   │   ├─ SL exit: +900s (full cooldown)
    │   │   └─ Rotation exit: 0s (immediate reentry allowed)
    │   │
    │   └─ Symbol locked for reentry until 2026-02-14T12:41:56Z
    │
    ├─→ [14] RELEASE SYMBOL LOCK
    │   │
    │   └─ Allow concurrent operations on other symbols
    │
    └─→ [DONE] SELL ORDER COMPLETE ✓
```

---

## REGIME DETERMINATION FLOW

```
System starts or metrics updated
    │
    ├─→ [1] FETCH CURRENT NAV
    │   │
    │   ├─ Get total portfolio value
    │   │   └─ NAV = sum(all_positions) + free_balance
    │   │
    │   └─ Example: NAV = $8,500
    │
    ├─→ [2] DETERMINE REGIME
    │   │
    │   ├─ if NAV < 1,000
    │   │   │
    │   │   ├─ Regime: MICRO_SNIPER (Capital-starved mode)
    │   │   │
    │   │   ├─ Limits:
    │   │   │   ├─ Max positions: 1
    │   │   │   ├─ Max symbols: 1
    │   │   │   ├─ Position size limit: 30% of NAV
    │   │   │   └─ Min confidence: 0.50
    │   │   │
    │   │   ├─ Restrictions:
    │   │   │   ├─ Rotation: DISABLED (only market exits)
    │   │   │   ├─ Dust healing: DISABLED (preserve capital)
    │   │   │   ├─ Scaling: 50% of normal (preserve capital)
    │   │   │   └─ Trade frequency: Reduced
    │   │   │
    │   │   └─ Goal: Preserve capital until NAV > 1000
    │   │
    │   ├─ elif NAV >= 1,000 AND NAV < 5,000
    │   │   │
    │   │   ├─ Regime: STANDARD (Normal operation)
    │   │   │
    │   │   ├─ Limits:
    │   │   │   ├─ Max positions: 2
    │   │   │   ├─ Max symbols: 3
    │   │   │   ├─ Position size limit: 25% of NAV
    │   │   │   └─ Min confidence: 0.55
    │   │   │
    │   │   ├─ Features:
    │   │   │   ├─ Rotation: ENABLED (60s cooldown)
    │   │   │   ├─ Dust healing: ENABLED
    │   │   │   ├─ Scaling: Normal (1.0x)
    │   │   │   └─ Trade frequency: Normal (up to 12/hour)
    │   │   │
    │   │   └─ Goal: Steady growth
    │   │
    │   ├─ elif NAV >= 5,000 AND NAV < 20,000
    │   │   │
    │   │   ├─ Regime: MULTI_AGENT (Aggressive growth)
    │   │   │
    │   │   ├─ Limits:
    │   │   │   ├─ Max positions: 3-5
    │   │   │   ├─ Max symbols: 5-10
    │   │   │   ├─ Position size limit: 20% of NAV
    │   │   │   └─ Min confidence: 0.60
    │   │   │
    │   │   ├─ Features:
    │   │   │   ├─ Rotation: ENABLED (30s cooldown)
    │   │   │   ├─ Dust healing: ENABLED
    │   │   │   ├─ Scaling: Aggressive (1.0-1.5x)
    │   │   │   └─ Trade frequency: High (up to 24/hour)
    │   │   │
    │   │   └─ Goal: Maximize capital efficiency
    │   │
    │   └─ elif NAV >= 20,000
    │       │
    │       ├─ Regime: INSTITUTIONAL (Scalable operations)
    │       │
    │       ├─ Limits:
    │       │   ├─ Max positions: 5-20 (diversified)
    │       │   ├─ Max symbols: 10-50
    │       │   ├─ Position size limit: 10-15% of NAV
    │       │   └─ Min confidence: 0.65
    │       │
    │       └─ Goal: Institutional-grade edge capture
    │
    ├─→ [3] LOG REGIME CHANGE (if different)
    │   │
    │   ├─ Previous: STANDARD
    │   ├─ Current: MICRO_SNIPER (drawdown triggered)
    │   ├─ Event: Regime change detected
    │   └─ Update configuration limits
    │
    └─→ [REGIME ACTIVE] Continue with regime limits
```

---

## POLICY NUDGE CALCULATION

```
Policy Manager receives system metrics
    │
    ├─→ [1] EVALUATE SYSTEM STATE
    │   │
    │   ├─ Velocity: realized_pnl_per_hour = $45/hour
    │   ├─ Drawdown: current_drawdown_pct = -8%
    │   ├─ Volatility: 30-day realized_vol = 2.1%
    │   ├─ Capital: free_capital_pct = 35% deployed
    │   └─ Signal: average_signal_confidence = 0.62
    │
    ├─→ [2] DETERMINE MODE
    │   │
    │   ├─ Is drawdown > 5%?
    │   │   ├─ YES (-8% > 5%) → SAFE mode
    │   │   └─ Velocity is positive? NO, use SAFE
    │   │
    │   ├─ Apply policy weights for SAFE mode:
    │   │   ├─ velocity weight: 0.0 (disabled)
    │   │   ├─ drawdown weight: 1.0 (full)
    │   │   ├─ volatility weight: 1.0 (full)
    │   │   ├─ capital weight: 0.0 (disabled)
    │   │   └─ signal weight: 0.0 (disabled)
    │
    ├─→ [3] APPLY POLICIES WITH WEIGHTS
    │   │
    │   ├─ VELOCITY POLICY (weight: 0.0x)
    │   │   ├─ Calculated nudge: confidence +0.1, size 1.2x
    │   │   ├─ Applied nudge: +0.1 * 0.0 = +0.0 ✗ disabled
    │   │   └─ Result: no impact on SAFE mode
    │   │
    │   ├─ DRAWDOWN POLICY (weight: 1.0x)
    │   │   ├─ Drawdown: 8%
    │   │   ├─ Calculated nudge: confidence +0.05, size 0.5x
    │   │   ├─ Applied nudge: full weight
    │   │   └─ Result: reduce position size to 50%, no confidence change
    │   │
    │   ├─ VOLATILITY POLICY (weight: 1.0x)
    │   │   ├─ Volatility: 2.1%
    │   │   ├─ Calculated nudge: confidence +0.1, size 0.8x
    │   │   ├─ Applied nudge: full weight
    │   │   └─ Result: require higher confidence, reduce size
    │   │
    │   ├─ CAPITAL POLICY (weight: 0.0x)
    │   │   ├─ Calculated nudge: confidence -0.05, size 0.9x
    │   │   ├─ Applied nudge: +0.0 * 0.0 = +0.0 ✗ disabled
    │   │   └─ Result: no impact in SAFE mode
    │   │
    │   └─ SIGNAL POLICY (weight: 0.0x)
    │       ├─ Calculated nudge: confidence +0.15, size 1.0x
    │       ├─ Applied nudge: +0.0 * 0.0 = +0.0 ✗ disabled
    │       └─ Result: no signal flexibility in SAFE mode
    │
    ├─→ [4] MERGE WEIGHTED NUDGES
    │   │
    │   ├─ confidence_nudge = 0.0 + 0.05 + 0.1 + 0.0 + 0.0 = +0.15
    │   ├─ position_size_mult = 1.0 * (1 + (0.5-1)*1) * (1 + (0.8-1)*1)
    │   │                     = 1.0 * 0.5 * 0.8 = 0.4x
    │   ├─ cooldown_nudge = 0 + 60 + 0 + 0 + 0 = +60 seconds
    │   └─ max_positions_nudge = 0 + 0 + 0 + 0 + 0 = 0 (no change)
    │
    ├─→ [5] APPLY NUDGES TO DECISIONS
    │   │
    │   ├─ Incoming signal: confidence=0.65
    │   ├─ Add nudge: 0.65 + 0.15 = 0.80 (higher bar in SAFE)
    │   │
    │   ├─ Position size: 25 USDT (baseline)
    │   ├─ Apply multiplier: 25 * 0.4x = 10 USDT
    │   │   └─ Effectively reduces position size for recovery
    │   │
    │   ├─ Between-trade cooldown: 300s (baseline)
    │   ├─ Add nudge: 300s + 60s = 360s (wait longer between trades)
    │   │   └─ Slows down trading frequency
    │   │
    │   └─ Max open positions: 2 (regime limit)
    │       └─ No change (nudge was 0)
    │
    └─→ [NUDGES APPLIED] Continue with adjusted parameters
```

---

## ERROR HANDLING & RECOVERY FLOW

```
Error Detected in Cycle
    │
    ├─→ [1] CLASSIFY ERROR
    │   │
    │   ├─ Type: ExchangeError vs ExecutionError vs StateError
    │   ├─ Severity: CRITICAL vs MAJOR vs MINOR
    │   ├─ Recoverable: YES/NO
    │   └─ Symbol: (if applicable)
    │
    ├─→ [2] ROUTE TO HANDLER
    │   │
    │   ├─ ExchangeError (connection lost, API timeout)
    │   │   ├─ Action: Retry with backoff (exp, up to 3x)
    │   │   ├─ If still fails: Skip cycle, continue monitoring
    │   │   └─ Alert: "Exchange connectivity issue"
    │   │
    │   ├─ ExecutionError (insufficient balance, min notional)
    │   │   ├─ Action: Log & classify
    │   │   ├─ If MIN_NOTIONAL: Increase quote size
    │   │   ├─ If INSUFFICIENT_BALANCE: Trigger recovery
    │   │   └─ Block this symbol temporarily
    │   │
    │   ├─ StateError (corrupt position data)
    │   │   ├─ Action: Reconcile with exchange
    │   │   ├─ Reset to authoritative exchange state
    │   │   ├─ Rebuild position snapshot
    │   │   └─ Resume trading after reconciliation
    │   │
    │   └─ Unexpected Error (unknown)
    │       ├─ Action: Pause trading immediately
    │       ├─ Alert: Critical error requires investigation
    │       └─ Wait for manual intervention
    │
    ├─→ [3] ATTEMPT RECOVERY
    │   │
    │   ├─ For MINOR errors (1-2 retry)
    │   │   ├─ Retry same operation
    │   │   └─ If succeeds → Resume normally
    │   │
    │   ├─ For MAJOR errors (symbol-level)
    │   │   ├─ Block symbol for N seconds
    │   │   ├─ Quarantine from trading
    │   │   └─ Continue with other symbols
    │   │
    │   ├─ For CRITICAL errors (system-level)
    │   │   ├─ Halt all trading immediately
    │   │   ├─ Set mode to PAUSED
    │   │   ├─ Alert operator
    │   │   └─ Investigate before resuming
    │   │
    │   └─ Recovery timeout: 30 seconds
    │       ├─ If recovered → Resume normally
    │       └─ If not recovered → Escalate
    │
    └─→ [END] Continue to next cycle or PAUSED state
```

---

**Document Version:** 1.0  
**Created:** 2026-02-14  
**For:** Technical Reference
