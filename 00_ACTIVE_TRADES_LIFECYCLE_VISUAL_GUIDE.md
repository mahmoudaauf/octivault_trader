# ACTIVE TRADES LIFECYCLE: VISUAL ARCHITECTURE

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     ExecutionManager                            │
│                  _handle_post_fill() Flow                       │
└─────────────────────────────────────────────────────────────────┘

                          ┌──────────────┐
                          │ Order Filled │
                          │  on Exchange │
                          └────────┬─────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            ┌───────▼─────────┐         ┌────────▼──────────┐
            │   BUY Fill      │         │   SELL Fill       │
            │ qty=1.0         │         │ qty=1.0           │
            │ price=67000     │         │ price=68000       │
            └────────┬────────┘         └────────┬──────────┘
                     │                           │
     ┌───────────────┴────────────┐             │
     │                            │             │
     ▼                            ▼             ▼
[1] Update Position      [2] Create Active Trade    [1] Update Position
    positions[BTC]           (NEW LIFECYCLE)            positions[BTC]
    .qty += 1.0                                         .qty -= 1.0
                           active_trades[BTC] = {
                             "entry_price": 67000,     ▼
                             "qty": 1.0,          [2] Manage Active Trade
                             "side": "BUY",           (NEW LIFECYCLE)
                             "opened_at": now,    remaining = 1.0 - 1.0 = 0
                             ...
                           }                      ├─ if remaining > 0:
                                                  │  active_trades[BTC].qty = remaining
     [3] Record Trade (Existing)                 │
         SharedState.record_trade()              └─ if remaining <= 0:
                                                     ├─ Delete active_trades[BTC]
     [4] Arm TP/SL                                 ├─ PnL = (68000-67000)*1.0-fees
         (Now sees open_trades > 0 ✅)            ├─ increment_realized_pnl(pnl)
                                                  └─ emit RealizedPnlUpdated
                           Log:
                           [LIFECYCLE_BUY_OPEN]      Log:
                                                     [LIFECYCLE_SELL_CLOSE]
```

---

## State Transition Diagram

```
        ┌─────────────────────────────────────┐
        │  Initial State (No Positions)       │
        │  active_trades = {}                 │
        │  realized_pnl = 0                   │
        └────────────────┬────────────────────┘
                         │
                    BUY 1.0 BTC
                   @ 67,000 USDT
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  After BUY (Position Open)          │
        ├─────────────────────────────────────┤
        │ active_trades = {                   │
        │   "BTCUSDT": {                      │
        │     entry_price: 67000,             │
        │     qty: 1.0,                       │
        │     side: "BUY",                    │
        │     opened_at: t1                   │
        │   }                                 │
        │ }                                   │
        │ realized_pnl = 0                    │
        │                                     │
        │ TP/SL: ARMED ✅                    │
        └────────────────┬────────────────────┘
                         │
                    SELL 0.5 BTC
                   @ 68,000 USDT
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  After Partial SELL (Reduce Qty)    │
        ├─────────────────────────────────────┤
        │ active_trades = {                   │
        │   "BTCUSDT": {                      │
        │     entry_price: 67000,             │
        │     qty: 0.5,    ← REDUCED          │
        │     side: "BUY",                    │
        │     opened_at: t1                   │
        │   }                                 │
        │ }                                   │
        │ realized_pnl = 0                    │
        │                                     │
        │ TP/SL: STILL ARMED ✅              │
        └────────────────┬────────────────────┘
                         │
                    SELL 0.5 BTC
                   @ 68,500 USDT
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  After Full SELL (Position Closed)  │
        ├─────────────────────────────────────┤
        │ active_trades = {}     ← CLEARED    │
        │                                     │
        │ realized_pnl = 1480    ← UPDATED    │
        │ (= (68000+68500)/2 - 67000 * 1.0    │
        │    - fees)                          │
        │                                     │
        │ TP/SL: DISARMED ✅                 │
        │ Event: RealizedPnlUpdated emitted ✅│
        └─────────────────────────────────────┘
```

---

## Lifecycle State Machine

```
                    ┌─────────────────────┐
                    │  NO POSITION        │
                    │  (initial state)    │
                    └──────────┬──────────┘
                               │
                           BUY FILL
                               │
                               ▼
                    ┌─────────────────────┐
                    │ POSITION OPEN       │
                    │ qty = exec_qty      │
                    │ entry = entry_price │
                    │ TP/SL: ARMED ✅     │
                    └──────┬───────┬──────┘
                           │       │
              PARTIAL SELL │       │ FULL SELL
                           │       │
                           ▼       ▼
                    ┌──────────────────────────┐
              ┌─────┤ QTY REDUCED              │
              │     │ (still open)             │
              │     │ TP/SL: STILL ARMED ✅   │
              │     └──────┬──────┬────────────┘
              │            │      │
              │ PARTIAL    │      │ FINAL SELL
              │ SELL       │      │
              │            ▼      ▼
              │     ┌──────────────────────────┐
              └────▶│ POSITION CLOSED          │
                    │ qty = 0                  │
                    │ PnL: REALIZED            │
                    │ TP/SL: DISARMED ✅       │
                    │ Event: EMITTED ✅        │
                    └──────────────────────────┘
```

---

## Data Structure: active_trades[symbol]

```
shared_state.active_trades = {
  "BTCUSDT": {
    ┌─────────────────────────────────────────┐
    │ symbol          : "BTCUSDT"             │
    │ entry_price     : 67000.0       ← KEY! │
    │ qty             : 1.0            ← KEY! │
    │ side            : "BUY"                 │
    │ opened_at       : 1709472000.123        │
    │ order_id        : "123456789"           │
    │ client_order_id : "cli-123"             │
    │ fee_quote       : 10.0                  │
    └─────────────────────────────────────────┘
  },
  "ETHUSDT": {
    ┌─────────────────────────────────────────┐
    │ symbol          : "ETHUSDT"             │
    │ entry_price     : 3800.0                │
    │ qty             : 10.0                  │
    │ side            : "BUY"                 │
    │ opened_at       : 1709472100.456        │
    │ ...                                     │
    └─────────────────────────────────────────┘
  }
}
```

---

## Processing Flow: BUY Side

```
BUY Order Filled
(executedQty=1.0, price=67000)
        │
        ▼
┌─────────────────────────────────┐
│ _handle_post_fill() Entry       │
│ side_u = "BUY"                  │
│ exec_qty = 1.0                  │
│ price = 67000                   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Update Position (existing)      │
│ positions[BTCUSDT].qty += 1.0   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ ⸻ NEW LIFECYCLE LOGIC ⸻         │
│ if side == "BUY":               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Create/Initialize active_trades │
│ if not hasattr(ss, "active_"... │
│   ss.active_trades = {}         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Record Active Trade Entry       │
│ active_trades[BTCUSDT] = {      │
│   entry_price: 67000,           │
│   qty: 1.0,                     │
│   ...metadata                   │
│ }                               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Log Event                       │
│ [LIFECYCLE_BUY_OPEN]            │
│  BTCUSDT opened                 │
│  entry_price=67000 qty=1.0      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Record Trade (existing)         │
│ SharedState.record_trade()      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Arm TP/SL (existing)            │
│ Now sees: open_trades > 0 ✅    │
└─────────────────────────────────┘
```

---

## Processing Flow: SELL Side

```
SELL Order Filled
(executedQty=1.0, price=68000)
        │
        ▼
┌──────────────────────────────────┐
│ _handle_post_fill() Entry        │
│ side_u = "SELL"                  │
│ exec_qty = 1.0                   │
│ price = 68000                    │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Update Position (existing)       │
│ positions[BTCUSDT].qty -= 1.0    │
│ (now qty = 0)                    │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ ⸻ NEW LIFECYCLE LOGIC ⸻          │
│ elif side == "SELL":             │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Get Active Trade                 │
│ trade = active_trades[BTCUSDT]   │
│ current_qty = 1.0                │
│ remaining = 1.0 - 1.0 = 0        │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Check if Fully Closed            │
│ if remaining_qty <= 0:           │
│   ✅ YES, fully closed           │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Delete Active Trade              │
│ del active_trades[BTCUSDT]       │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Calculate Realized PnL           │
│ entry_price = 67000              │
│ pnl = (68000-67000)*1.0 - 10.0   │
│     = 990 USDT                   │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Update Realized PnL (atomic)     │
│ await increment_realized_pnl(990)│
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Emit RealizedPnlUpdated Event    │
│ {                                │
│   pnl_delta: 990,                │
│   symbol: "BTCUSDT",             │
│   timestamp: now,                │
│   nav_quote: 101000              │
│ }                                │
└─────────┬────────────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│ Log Event                        │
│ [LIFECYCLE_SELL_CLOSE]           │
│  BTCUSDT closed                  │
│  realized_qty=1.0                │
└──────────────────────────────────┘
```

---

## Before vs After: Visual Comparison

### BEFORE (Broken)
```
┌─────────────────────────────────────────────────┐
│ BUY Fill → _handle_post_fill()                 │
├─────────────────────────────────────────────────┤
│ ✅ Update positions[BTCUSDT].qty = 1.0         │
│ ❌ active_trades[BTCUSDT] = ? (MISSING!)       │
│ ✅ Record trade                                 │
│ ❌ TP/SL checks: open_trades = 0, DON'T ARM    │
│                                                 │
│ TruthAuditor sees:                             │
│   positions[BTCUSDT].qty = 1.0  ← YES          │
│   BUT: active_trades = {}       ← NO           │
│   Patches: active_trades[BTCUSDT] = {...}      │
│   ⚠️  Fragile, masking real problem            │
└─────────────────────────────────────────────────┘
```

### AFTER (Fixed)
```
┌─────────────────────────────────────────────────┐
│ BUY Fill → _handle_post_fill()                 │
├─────────────────────────────────────────────────┤
│ ✅ Update positions[BTCUSDT].qty = 1.0         │
│ ✅ ⸻NEW⸻ active_trades[BTCUSDT] = {...}       │
│ ✅ Record trade                                 │
│ ✅ TP/SL checks: open_trades > 0, ARM! ✅     │
│                                                 │
│ State is Coherent:                             │
│   positions[BTCUSDT].qty = 1.0  ✅             │
│   active_trades[BTCUSDT].qty = 1.0  ✅         │
│   These match → System is coherent             │
│                                                 │
│ TruthAuditor:                                  │
│   "Everything is in sync, nothing to patch"    │
│   ✅ Isolated, as intended                     │
└─────────────────────────────────────────────────┘
```

---

## Summary: Visual Architecture

The fix ensures:
1. **Full Lifecycle**: BUY → CREATE / SELL → REDUCE or DELETE
2. **State Coherence**: positions ≠ active_trades (they match now)
3. **TPSL Visibility**: Can count and arm on actual trades
4. **PnL Accuracy**: Computed from stored entry_price
5. **Clean Architecture**: No TruthAuditor patches needed

**Result**: ✅ **Professional, coherent, production-grade system**

