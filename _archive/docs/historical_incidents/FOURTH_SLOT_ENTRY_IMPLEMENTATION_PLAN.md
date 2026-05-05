# 4th-Slot Entry Implementation Plan

**Date:** 2026-05-05
**Author:** GitHub Copilot (Plan B deliverable)
**Status:** AWAITING USER APPROVAL — no code changes yet
**Scope:** Implement the missing entry-side path that activates `FourthSlotTracker`, so the bot resumes rotation, compounding, and re-investment of idle USDT.

---

## 1. Problem Recap

| Layer | Status |
|---|---|
| `FourthSlotTracker.set_position()` (entry hook) | **never called** anywhere in the codebase |
| `FourthSlotTracker.check_exit_conditions()` (exit hook) | wired in `meta_controller.py:10520` |
| Forced-exit injection into `_build_decisions` | wired at `meta_controller.py:13768` |
| Trade journal | zero records tagged `4th_slot` |

**Effect:** the high-volatility rotating slot — the only mechanism designed to surface candidates that can clear the 0.61 % EV gate — has been dormant since FIX8 was merged. With idle USDT at $77.15, the bot is sitting on capital it has no path to deploy.

---

## 2. Design Contract (verified by code read)

### 2.1 FourthSlotTracker (`src/l8_lifecycle/fourth_slot_tracker.py`)
- `set_position(symbol, entry_price, position_size)` — registers the slot. Idempotent: refuses if `current_symbol` is already set.
- `check_exit_conditions(current_price) → Optional[dict]` — returns `{exit_reason, exit_price, pnl, pnl_pct, time_held_min}` when TP (+15 %) / SL (−3 %) / max-hold (120 min) hit, else `None`.
- `reset_position()` — clears state, called from the existing exit watcher after forced SELL is queued.

### 2.2 Config knobs (`src/l0_core/config.py`)
| Key | Value |
|---|---|
| `FIX8_4TH_SLOT_ENABLED` | `True` |
| `FIX8_4TH_SLOT_CAPITAL_ALLOCATION_USD` | `5.0` |
| `FIX8_4TH_SLOT_CAPITAL_PCT` | `0.065` |
| `FIX8_4TH_SLOT_MIN_COOLDOWN_SECONDS` | `30` |
| `FIX8_4TH_SLOT_CANDIDATES_TO_CONSIDER` | `20` |
| `FIX8_4TH_SLOT_PROFIT_TARGET_PCT` | `0.15` |
| `FIX8_4TH_SLOT_STOP_LOSS_PCT` | `-0.03` |
| `FIX8_4TH_SLOT_MAX_HOLD_MINUTES` | `120` |

### 2.3 Existing exit-side wiring (reference)
- `meta_controller.py:1990` initializes `self.fourth_slot_tracker`.
- `meta_controller.py:10520` polls every loop, builds a `forced_exit` tuple, stashes it on `self._forced_exit_intent`, and calls `reset_position()`.
- `meta_controller.py:13768` consumes `_forced_exit_intent` inside `_build_decisions` and routes it into the SELL pipeline.

### 2.4 Execution surface
- Canonical entry: `await self.execution_manager.execute_trade(intent: TradeIntent)` (`src/l4_execution/execution_manager.py:7083`).
- Canonical contract: `src.l0_core.contracts.TradeIntent` (`quote=` for USDT-sized BUYs, `quantity=` for SELLs).
- Returns `{ok, status, executedQty, avgPrice, cummulativeQuoteQty, orderId, reason, ...}`.

### 2.5 Candidate source
- `shared_state.accepted_symbols` is a `Dict[symbol → metadata]` already populated by `SymbolScreener` with volume, % change, and ATR% fields. **No new screener work needed.**
- `regime_proposal_analyzer` already keys on `source == "symbol_screener" and atr_pct …` — same metadata is reusable.

---

## 3. Implementation Plan

### 3.1 New method: `_attempt_fourth_slot_entry()` on `MetaController`

Add a single async method (≈80 lines) in `src/l8_lifecycle/meta_controller.py`, placed immediately **above** the existing exit-watcher block at line 10515 (so entry and exit logic sit together and the order is: try-exit → try-entry on the same loop).

#### 3.1.1 Pre-flight gates (fail fast, return)
1. `self.fourth_slot_tracker` is not None.
2. `self.fourth_slot_tracker.current_symbol is None` (slot is empty).
3. `time.time() - self._fourth_slot_last_rotation_ts >= FIX8_4TH_SLOT_MIN_COOLDOWN_SECONDS` (30 s).
4. `self._forced_exit_intent is None` (don’t race a pending forced SELL).
5. Free quote (USDT) ≥ `_alloc_quote()` + a 5 % safety buffer.
6. Portfolio is below `capital_governor.max_concurrent_positions` **OR** the 4th-slot is the configured overflow (decision: treat 4th slot as `+1` over the bracket cap, gated by NAV ≥ $50; document this as the only exception to the bracket).

#### 3.1.2 Sizing
```python
nav = self.shared_state.get_nav_quote() or 0.0
alloc_pct = nav * float(self.config.FIX8_4TH_SLOT_CAPITAL_PCT)
alloc_usd = float(self.config.FIX8_4TH_SLOT_CAPITAL_ALLOCATION_USD)
quote_amount = round(min(alloc_usd, alloc_pct), 2)   # cap at $5 even if 6.5 % > $5
if quote_amount < 5.0:                               # Binance MIN_NOTIONAL safety
    return
```

#### 3.1.3 Candidate selection
```python
N = int(self.config.FIX8_4TH_SLOT_CANDIDATES_TO_CONSIDER)  # 20
universe = self.shared_state.accepted_symbols                # {sym: meta}

# Exclude symbols already held / blocked / on cooldown / in dust-recovery
held = set(self.shared_state.positions.keys())
blocked = self._collect_blocked_symbols()                    # reuse helper
candidates = [
    (s, m) for s, m in universe.items()
    if s not in held and s not in blocked and s.endswith("USDT")
]

# Rank by ATR% desc (volatility — what clears the EV gate)
candidates.sort(key=lambda kv: float(kv[1].get("atr_pct", 0.0)), reverse=True)
top = candidates[:N]
```

For each `(sym, meta)` in `top`:
1. Pull MLForecaster prediction (already cached per loop): `pred = self.shared_state.get_latest_prediction(sym)`.
2. Compute expected round-trip cost: `rt = self._round_trip_cost_pct(sym)` (existing helper, ≈0.38 %).
3. Compute expected move from `pred.expected_move_pct` (or fallback `meta["atr_pct"] * 0.5` if no ML signal yet).
4. **EV gate (relaxed for 4th slot):** require `expected_move >= rt * 1.2` (vs 1.6× for core). Rationale: the slot is risk-budgeted at $5 with a hard −3 % SL = −$0.15 max loss; we trade tighter EV for higher rotation throughput.
5. Require `pred.action == "BUY"` and `pred.confidence >= 0.55`.
6. First candidate that passes → selected.

If nothing passes → log `[4thSlot] no qualifying candidate (top=%s, best_em=%.3f%%, gate=%.3f%%)` and return.

#### 3.1.4 Order placement
```python
trace_id = self._make_trace_id("4TH_SLOT_ENTRY", symbol)
intent = TradeIntent(
    symbol=symbol,
    side="buy",
    quote=quote_amount,                        # $5 USDT
    confidence=float(pred.confidence),
    reason="4TH_SLOT_ROTATION",
    agent="FourthSlotEntry",
    tag="4th_slot/entry",
    trace_id=trace_id,
    tier="ROTATION",                            # ExecutionManager will reconcile via SOP if needed
    policy_context={
        "fourth_slot": True,
        "expected_move_pct": expected_move,
        "atr_pct": float(meta.get("atr_pct", 0.0)),
        "ev_ratio": expected_move / max(rt, 1e-6),
    },
)
result = await self.execution_manager.execute_trade(intent)
```

#### 3.1.5 Post-fill bookkeeping
```python
if result.get("ok") and float(result.get("executedQty", 0.0)) > 0:
    avg_px = float(result["avgPrice"])
    qty    = float(result["executedQty"])
    self.fourth_slot_tracker.set_position(symbol, avg_px, qty)
    self._fourth_slot_last_rotation_ts = time.time()
    self._record_journal_event("4TH_SLOT_ENTRY", {
        "symbol": symbol, "qty": qty, "avg_px": avg_px,
        "quote": quote_amount, "trace_id": trace_id,
        "expected_move_pct": expected_move,
    })
    self.logger.info("[4thSlot] ENTRY %s qty=%.6f @ %.6f ($%.2f) tp=+15%% sl=-3%%",
                     symbol, qty, avg_px, quote_amount)
else:
    # Apply a longer cooldown on failure to avoid hammering the same bad symbol
    self._fourth_slot_last_rotation_ts = time.time()
    self.logger.info("[4thSlot] entry failed for %s: %s", symbol, result.get("reason"))
```

### 3.2 New instance attributes (init in `__init__`, near line 1991)

```python
self._fourth_slot_last_rotation_ts: float = 0.0
```

### 3.3 Loop integration

Single call site, just before line 10515:

```python
# FIX#2A-ENTRY: 4th-slot rotation entry (paired with exit watcher below)
if self.fourth_slot_tracker is not None:
    try:
        await self._attempt_fourth_slot_entry()
    except Exception:
        self.logger.exception("[4thSlot] entry attempt raised; continuing")
```

### 3.4 No changes required elsewhere

- `_build_decisions` already routes forced exits → unchanged.
- `capital_governor` bracket logic stays as-is; the 4th slot is treated as a documented +1 overflow over the MICRO `max_concurrent_positions=2` cap (already implied by FIX8 docs).
- `SymbolScreener` already populates `accepted_symbols` with the metadata we need.
- `MLForecaster` already runs each loop on the universe.

---

## 4. Risk Controls

| Risk | Mitigation |
|---|---|
| Slot keeps re-entering the same loser | 30 s cooldown after any attempt (success or fail) + per-symbol cooldown map keyed by recent SL exits |
| Position sizing exceeds free USDT | Hard pre-flight check `free_usdt >= quote * 1.05` |
| Concurrent entry + exit race | Pre-flight `_forced_exit_intent is None` and `current_symbol is None` |
| Phantom re-entry after restart | `set_position()` only on `result['ok'] and executedQty>0`; tracker is in-memory only — restart clears it cleanly (acceptable for $5 max risk) |
| Entry on a stale price | `execute_trade` already validates against live ticker; no extra work |
| Sub-$5 fill rounding | Reject result if `executedQty * avgPrice < 4.50`; immediately submit a DUST_LIQUIDATION SELL (reuse existing path) |

---

## 5. Validation Steps (before live activation)

1. `python3 -m py_compile src/l8_lifecycle/meta_controller.py`
2. `grep -n "fourth_slot_tracker.set_position" src/` → must show exactly one new call site.
3. **Dry run:** export `LIVE_MODE=false`, run for ≥3 loops, confirm log line `[4thSlot] would-buy SYMBOL …` (add an explicit dry-run branch).
4. **Single-shot live:** allow one live entry, watch log for `[4thSlot] ENTRY`, then confirm in Binance UI that the order filled and the position appears.
5. **Exit verification:** after entry, observe one of: TP hit, SL hit, or 120-min max-hold. Confirm forced SELL fires via the existing exit-side path and tracker resets.
6. **Journal audit:** `grep "4TH_SLOT_ENTRY\|4TH_SLOT_EXIT" trade_journal.jsonl | tail`.

---

## 6. Rollback

Single env flag: `FIX8_4TH_SLOT_ENABLED=false` disables both entry and exit paths (existing behaviour). No DB migration, no state file. Safe to disable mid-flight: in-flight position will be exited by the standard SOP exit pipeline, not the tracker.

---

## 7. Files Touched (proposed)

| File | Change |
|---|---|
| `src/l8_lifecycle/meta_controller.py` | + ≈90 lines: new `_attempt_fourth_slot_entry()`, init of `_fourth_slot_last_rotation_ts`, one-line call in main loop |
| **(no other files)** | candidate source, ML forecaster, execute_trade, tracker, governor — all already present |

---

## 8. Out-of-Scope (explicit)

- **Compound pool wiring** (60 % of profitable 4th-slot exits → core sizing). Tracked separately; needs UURE-side change. We can add this **after** the entry path is proven in production.
- **Auto-reconciliation watchdog** (prevents the phantom-state bug we hit earlier). Separate task.
- **EV gate retuning for core symbols.** Out of scope; the 4th-slot path itself addresses the “no trades” symptom by introducing high-ATR candidates that satisfy the existing core gate too.

---

## 9. Decision Requested

Please confirm:
1. ✅ / ❌ — Proceed with **§3.1 — §3.3** as written.
2. ✅ / ❌ — Treat the 4th slot as **+1 overflow** above MICRO `max_concurrent_positions=2` (vs replacing the rotating slot).
3. ✅ / ❌ — Use the **relaxed EV gate (1.2× round-trip)** for the 4th slot specifically.
4. ✅ / ❌ — Accept that the tracker is in-memory only (no persistence across restarts; max risk per slot is $0.15 SL, so acceptable).

On 4×✅ I will implement and validate per §5.
