# Step 3 — Façade Contract: `main.py` Talks Only to 5 Engines

**Date**: May 6, 2026
**Status**: ✅ IMPLEMENTED
**File**: `main.py` (replaces direct module access)
**Legacy**: `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` retained for backward compat

---

## The Contract

```
   BEFORE                         AFTER (this step)
   ──────                         ─────────────────

   main                           main.py
    ├─→ exchange_client            └─→ MarketAccountEngine    ─┐
    ├─→ market_data_feed                                       │
    ├─→ balance_manager            └─→ SituationEngine        ─┤
    ├─→ portfolio_manager                                      │
    ├─→ regime_detector            └─→ DecisionEngine         ─┤  (façade)
    ├─→ signal_manager                                         │
    ├─→ signal_fusion              └─→ SafeExecutionEngine    ─┤
    ├─→ arbitration_engine                                     │
    ├─→ mode_manager               └─→ OperationsEngine       ─┘
    ├─→ risk_manager                       │
    ├─→ execution_manager                  ▼
    ├─→ tp_sl_engine                  L0–L8 (145+ modules)
    ├─→ health_monitor                  encapsulated
    ├─→ recovery_engine
    ├─→ watchdog
    ├─→ … 130 more
```

---

## Allowed Imports in `main.py`

| Source | Allowed? | Why |
|---|---|---|
| Python stdlib (`asyncio`, `argparse`, `logging`, `signal`, `sys`, `time`) | ✅ | Entry-point plumbing |
| `core_engine.MarketAccountEngine` | ✅ | Façade |
| `core_engine.SituationEngine` | ✅ | Façade |
| `core_engine.DecisionEngine` | ✅ | Façade |
| `core_engine.SafeExecutionEngine` | ✅ | Façade |
| `core_engine.OperationsEngine` | ✅ | Façade |
| `core_engine.integration.setup_core_engines` | ✅ | Wires app_ctx |
| `src.l1_exchange.*` | ❌ | Bypasses MarketAccountEngine |
| `src.l3_portfolio.*` | ❌ | Bypasses SituationEngine |
| `src.l4_execution.*` | ❌ | Bypasses SafeExecutionEngine |
| `src.l5_strategy.*` | ❌ | Bypasses DecisionEngine |
| `src.l7_observability.*` | ❌ | Bypasses OperationsEngine |
| Anything else under `src/` | ❌ | Violates façade contract |

---

## The Cycle (Pure Façade Calls)

```python
# 1. READ
account = await engines.market.get_account_state()
prices  = await engines.market.get_market_prices()

# 2. UNDERSTAND
portfolio = await engines.situation.get_portfolio_snapshot()
regime    = await engines.situation.get_market_regime()
signals   = await engines.situation.get_all_signals()

# 3. DECIDE
decision = await engines.decision.make_buy_decision(symbol, edge)

# 4. EXECUTE
result = await engines.execution.place_buy_order(symbol, qty, price)

# 5. RECOVER / OBSERVE
health = await engines.operations.get_health_report()
await  engines.operations.log_event("cycle_complete", {...})
```

**Zero direct imports of L0–L8 components.** All 145+ modules are reached
exclusively through the 5 engine façades.

---

## Verification Command

```bash
# Static check: main.py must not import any src.* module
grep -nE "^(from|import) (src\.|.*l[0-8]_)" main.py
# expected: no output

# Smoke test: run 5 cycles in dry-run
python3 main.py --mode=dry-run --cycles=5 --interval=0.1
```

---

## CLI

```bash
# Paper trading for 24h (default)
python3 main.py --mode=paper-trade --duration=24h

# Live trading with $1,000 capital
python3 main.py --mode=live --capital=1000

# Quick dry-run smoke test
python3 main.py --mode=dry-run --cycles=10
```

---

## Why Two Entry Points During Transition?

| File | Purpose | Status |
|---|---|---|
| `main.py` (NEW) | Façade entry point — talks to 5 engines only | ✅ Step 3 deliverable |
| `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` (legacy) | Existing 3,290-line orchestrator | ⏳ Keep until Phase 7 cutover validates `main.py` end-to-end |

Once `main.py` runs through paper trading + live $1k validation, the legacy
orchestrator can be deprecated and moved to `_archive/`.

---

## Acceptance Criteria

- [x] `main.py` exists and is < 300 lines
- [x] `main.py` imports only stdlib + `core_engine`
- [x] All 5 engines instantiated from `core_engine`
- [x] Trading cycle uses only façade methods (verified line-by-line)
- [x] Graceful SIGINT/SIGTERM handling
- [x] CLI parity with legacy orchestrator (`--mode`, `--duration`, `--capital`)
- [x] Auto-recovery delegated to `OperationsEngine`

✅ **Step 3 complete.**
