# Capital Freeing Implementation — Completed (2026-05-07)

## Overview

Implemented autonomous capital freeing in `core_engine/native/decisions.py` to liquidate dust holdings (AVAX, DOGE, SOL, etc.) when:
1. **Balance insufficient**: `balance_usdt < min_order_usdt` (can't allocate standard position)
2. **Strong BUY signals present**: `len(buy_sigs) > 0` (trading opportunities waiting)
3. **Weak signal on holding**: Asset has SELL/HOLD signal (safe to exit), not BUY

This enables autonomous capital recycling on micro-accounts (e.g., $0.58) by selling old/dust positions instead of waiting for manual deposits.

---

## Implementation Details

### Location
**File**: `core_engine/native/decisions.py` lines 184-242

### Algorithm

**Phase 1: Candidate Selection**
- Iterate through `portfolio.balance` (all non-USDT holdings)
- For each asset, evaluate signal quality:
  - **SELL signal** → `priority_score = 0.0 + signal_score` (best opportunity)
  - **HOLD signal** → `priority_score = 0.5 + signal_score` (safe to exit)
  - **BUY signal** → `priority_score = 2.0` (never sell)

- Consider asset size:
  - **Dust**: `qty < 0.001 OR qty < 2% of NAV` (preferred)
  - **Large**: Not considered unless signal is SELL

- **Filter**: Only consider if:
  - `priority_score < 2.0` (not a strong BUY)
  - AND (`is_dust` OR `signal_score < 0.5`) (weak conviction)

**Phase 2: Best Candidate Selection**
- Track lowest `priority_score` across all candidates
- Select asset with best (lowest) score

**Phase 3: Execution**
- Create CLOSE decision with:
  - `action = Action.CLOSE`
  - `quantity = full_holding_qty`
  - `reason = f"capital_freeing:{direction}:{score:.2f}"`
  - `risk_score = 0.2` (very low risk)
- Log message: `"💰 CAPITAL FREEING: {asset_symbol} qty={qty} (signal={direction} score={score}) → frees capital for BUY opportunity"`

### Example Flow

**Micro-account ($0.58 USDT):**
```
Holdings:  AVAX=4.6, DOGE=373.77, SOL=0.08
Signals:   AVAXUSDT=HOLD(0.3), DOGEUSDT=HOLD(0.2), SOLUSDT=BUY(0.8)
BUY_sigs:  BTCUSDT, ETHUSDT (both high conviction)

Phase 1: Evaluate candidates
  AVAX: priority=0.5+0.3=0.8, is_dust=True → INCLUDE
  DOGE: priority=0.5+0.2=0.7, is_dust=True → INCLUDE (BEST)
  SOL:  priority=2.0 (BUY) → SKIP

Phase 2: Select best
  Best = DOGE (priority=0.7 < AVAX priority=0.8)

Phase 3: Execute
  Create CLOSE(DOGEUSDT, qty=373.77, reason="capital_freeing:hold:0.20")
  Log: "💰 CAPITAL FREEING: DOGEUSDT qty=373.77 (signal=HOLD score=0.20) → frees capital for BUY opportunity"
```

**Next cycle:**
- DOGE SELL executes → ~$0.05-0.10 freed (depending on price)
- USDT balance now $0.63-0.68
- Allocator can now open small BTCUSDT or ETHUSDT position
- Capital recycled!

---

## Code Changes

**Lines 184-187**: Capital freeing trigger
```python
# Only sell when: (1) we need capital for strong BUY signals, (2) asset has weak signal
if balance_usdt < self.min_order_usdt and len(buy_sigs) > 0:
```

**Lines 188-241**: Candidate evaluation + execution
- Build priority scores based on signal direction
- Filter by dust size
- Select best candidate
- Append CLOSE decision with capital_freeing reason

---

## Commit

**Hash**: `3eeac83` (phase-3/wiring branch)
```
feat: Add capital freeing via dust liquidation in decision engine

When balance is insufficient for standard allocation AND strong BUY signals
exist, liquidate dust holdings with weak signals to free capital. Prioritizes:
1. SELL signals (best exit opportunity)
2. HOLD signals (safe to exit)
3. Prefers dust (<0.001 qty or <2% of NAV)

This enables autonomous capital recycling on micro-accounts ($0.58) by selling
old positions (AVAX, DOGE) when good trading opportunities appear, instead of
waiting for manual deposits.
```

---

## Testing Status

### Current Blocker: IP Ban (418 Rate Limit)
- Binance banned IP until Unix timestamp 1778131911234 (~May 8, 2026)
- Root cause: Aggressive REST polling during bootstrap (balance_sync every 5s, market_data every 2s)
- Will retry after ban expires

### Expected Behavior (Once Ban Expires)
When running `python3 run_and_monitor.py 100`:

**Cycle 1-5**: Initial trades placed
```
✅ Cycle  1 | NAV=$0.58 | Sig=8 Dec=1 Exe=1
           Positions: BTCUSDT
```

**Cycle 5-10**: Balance depleted, capital freeing triggered
```
✅ Cycle  6 | NAV=$0.58 (+0.00) | Sig=10 Dec=3 Exe=2
💰 CAPITAL FREEING: DOGEUSDT qty=373.77 (signal=HOLD score=0.20) → frees capital for BUY opportunity
```

**Cycle 10-15**: DOGE sell executes, new BUY placed
```
✅ Cycle  9 | NAV=$0.63 (+0.05) | Sig=12 Dec=4 Exe=4
           Positions: BTCUSDT, ETHUSDT (DOGE sold, freed capital deployed)
```

**Success Indicators**:
- 🎉 First SELL detected (from DOGE liquidation)
- 🔄 Symbol interchange (DOGE closes, BUY opens)
- 📈 NAV increasing (profits compounding)

---

## Architecture Integration

### Signal Flow
```
NativeSignalEngine (L2)
  ↓ generates signals per symbol

NativeDecisionEngine (L4)
  ├─ SELL signals → close winning positions
  ├─ BUY signals → open new positions
  └─ CAPITAL FREEING → liquidate dust when balance low & opportunity good

Executor (L4)
  ↓ executes CLOSE/OPEN/HOLD decisions

FillTracker (L4)
  ├─ tracks fills
  ├─ calculates realized PnL
  └─ updates SharedState.metrics

Next Cycle (P3)
  ↓ freed USDT available for new allocations
```

### State Tracking
- **SharedState.positions**: Current holdings (AVAX, DOGE, BTC, ETH)
- **SharedState.balance**: Coins per asset (AVAX=4.6, DOGE=373.77, etc.)
- **SharedState.metrics**: Win rate, realized_pnl, avg_fee_bps
- **CapitalAllocator**: Uses freed USDT from prior CLOSE decisions

---

## Behavioral Notes

1. **Signal-Aware Exit**: Only sells dust with weak signals (never liquidates winning positions)
2. **Opportunity-Driven**: Only activates when there's a reason (strong BUY signals waiting)
3. **Low-Risk**: Risk score=0.2 ensures CLOSE decisions get lower priority if needed
4. **Composable**: Works with existing profit-gating (only SELL decisions with realized_pnl > 0 execute)

---

## Next Steps

1. **Wait for IP ban to expire** (May 8, 2026)
2. **Run live monitor**: `python3 run_and_monitor.py 100`
3. **Verify capital freeing triggers** (look for 💰 CAPITAL FREEING logs)
4. **Verify symbol interchange** (DOGE closes, new position opens)
5. **Measure NAV growth**: $0.58 → $0.70+ via dust liquidation

---

## Related Documentation

- [CAPITAL_GROWTH_MECHANICS.md](CAPITAL_GROWTH_MECHANICS.md) — Multi-symbol trading flow
- [START_HERE_CAPITAL_GROWTH.md](START_HERE_CAPITAL_GROWTH.md) — System run guide
- [core_engine/native/decisions.py](core_engine/native/decisions.py) — Implementation
