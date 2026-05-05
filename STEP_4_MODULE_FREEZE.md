# Step 4 — Module Freeze Manifest

**Date**: May 6, 2026
**Status**: ✅ FROZEN (no deletions yet)
**Machine-readable**: `MODULE_FREEZE_MANIFEST.json`
**Enforcement**: Step 5 will add `scripts/check_frozen_modules.py` to CI

---

## Status Labels

| Label | Meaning | Editable? | Importable? |
|---|---|---|---|
| 🟢 **ACTIVE** | Canonical implementation. Used directly by `core_engine` façades or top-level entry. | ✅ Yes | ✅ Yes (canonical path) |
| 🔵 **WRAPPED** | Reachable only through a `core_engine` façade. | ✅ Yes (inside its layer) | ⚠️  Only via `core_engine.*` — never from `main.py` or top-level scripts |
| 🟡 **LEGACY** | Superseded. Kept for reference until Phase 7 cutover validates `main.py`. | ❌ No new edits | ❌ Do not import in new code |
| 🔴 **QUARANTINED** | Broken / abandoned / unknown lineage. | ❌ No | ❌ No — moves to `_archive/` in Step 5 |

---

## Headline Counts

| Family             | ACTIVE | WRAPPED | LEGACY | QUARANTINED |
|--------------------|:------:|:-------:|:------:|:-----------:|
| Dashboards         |   1    |    1    |   3    |      0      |
| WebSocket          |   0    |    1    |   0    |      0      |
| Balance Sync       |   0    |    4    |   2    |      0      |
| Runners/Launchers  |   3    |    0    |   4    |      6      |
| Recovery           |   0    |    3    |   1    |      0      |
| Capital Engines    |   0    |    6    |   3    |      0      |
| Watchdogs          |   0    |    1    |   0    |      2      |
| Regime Detectors   |   0    |    5    |   0    |      0      |
| **Totals**         | **4**  | **21**  | **13** |    **8**    |

---

## Dashboards (`5 modules`)

| File | Status | Note |
|---|---|---|
| `src/l7_observability/dashboard.py` | 🟢 ACTIVE | Canonical — wrapped by `OperationsEngine.get_health_report()` |
| `src/l7_observability/monitors/balance_dashboard.py` | 🔵 WRAPPED | Balance view inside OperationsEngine |
| `monitoring/real_time_dashboard.py` | 🟡 LEGACY | Pre-engine standalone |
| `monitoring/capital_dashboard.py` | 🟡 LEGACY | Duplicate of `capital_growth_dashboard.py` |
| `monitoring/capital_growth_dashboard.py` | 🟡 LEGACY | Standalone — data flows through OperationsEngine |

## WebSocket (`1 module`)

| File | Status | Note |
|---|---|---|
| `src/l1_exchange/market_data_websocket.py` | 🔵 WRAPPED | Sole WS client — via `MarketAccountEngine.get_market_prices()` |

## Balance Sync (`6 modules`)

| File | Status | Note |
|---|---|---|
| `src/l2_marketdata/balance_manager.py` | 🔵 WRAPPED | Canonical — via `MarketAccountEngine.get_account_state()` |
| `src/l2_marketdata/balance_sync.py` | 🔵 WRAPPED | Sync coordinator inside balance_manager |
| `src/l2_marketdata/balance_cache_updater.py` | 🔵 WRAPPED | Cache writer |
| `src/l1_exchange/balance_sync_backoff.py` | 🔵 WRAPPED | L1 sync backoff helper |
| `balance_monitor.py` | 🟡 LEGACY | Top-level — superseded by OperationsEngine |
| `balance_threshold_config.py` | 🟡 LEGACY | Config now in `core_engine.integration` |

## Runners / Launchers (`13 modules`)

| File | Status | Note |
|---|---|---|
| `main.py` | 🟢 ACTIVE | Step 3 façade-only entry point |
| `START_TRADING.sh` | 🟢 ACTIVE | Canonical launcher wrapper |
| `LAUNCH_BOT.command` | 🟢 ACTIVE | macOS double-click launcher |
| `LAUNCH_MONITOR.sh` | 🟢 ACTIVE | Monitoring sidecar launcher |
| `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | 🟡 LEGACY | Pre-Step-3 entry — retained until paper-trade + $1k validation |
| `start_trading_with_monitoring.sh` | 🟡 LEGACY | Duplicate combo |
| `run_bot_resilient.sh` | 🟡 LEGACY | Auto-restart now in `OperationsEngine.recover()` |
| `restart_with_optimization.sh` | 🟡 LEGACY | One-shot shim |
| `launch_run6.sh … launch_run11.sh` (×6) | 🔴 QUARANTINED | Numbered debug runners — abandoned |

## Recovery (`4 modules`)

| File | Status | Note |
|---|---|---|
| `src/l4_execution/recovery_engine.py` | 🔵 WRAPPED | Via `SafeExecutionEngine` + `OperationsEngine.recover()` |
| `src/l8_lifecycle/runners/auto_recovery.py` | 🔵 WRAPPED | Lifecycle runner used by OperationsEngine |
| `src/l8_lifecycle/runners/apply_recovery_to_live.py` | 🔵 WRAPPED | Live-mode applier |
| `auto_recovery.py` | 🟡 LEGACY | Top-level shim duplicate |

## Capital Engines (`9 modules`)

| File | Status | Note |
|---|---|---|
| `src/l6_governance/capital_governor.py` | 🔵 WRAPPED | Canonical — via DecisionEngine + OperationsEngine |
| `src/l6_governance/capital_allocator.py` | 🔵 WRAPPED | Allocation policy |
| `src/l6_governance/capital_symbol_governor.py` | 🔵 WRAPPED | Per-symbol guardrails |
| `src/l6_governance/adaptive_capital_engine.py` | 🔵 WRAPPED | Adaptive sizing |
| `src/l3_portfolio/dead_capital_healer.py` | 🔵 WRAPPED | Idle-position healer in SituationEngine |
| `src/l5_strategy/capital_velocity_optimizer.py` | 🔵 WRAPPED | Velocity scoring |
| `capital_health_monitor.py` | 🟡 LEGACY | Superseded by OperationsEngine |
| `monitoring/active_capital_monitor.py` | 🟡 LEGACY | Standalone monitor |
| `monitoring/capital_growth_monitor.py` | 🟡 LEGACY | Duplicate |

## Watchdogs (`3 modules`)

| File | Status | Note |
|---|---|---|
| `src/l8_lifecycle/watchdog.py` | 🔵 WRAPPED | Canonical — via OperationsEngine |
| `docs/archive/scripts/GATING_WATCHDOG.py` | 🔴 QUARANTINED | Already archived — retain location |
| `docs/archive/scripts/PERSISTENT_TRADING_WATCHDOG.py` | 🔴 QUARANTINED | Already archived — retain location |

## Regime Detectors (`5 modules`)

| File | Status | Note |
|---|---|---|
| `src/l2_marketdata/market_regime_detector.py` | 🔵 WRAPPED | Canonical — via `SituationEngine.get_market_regime()` |
| `src/l2_marketdata/market_regime_integration.py` | 🔵 WRAPPED | Integration adapter |
| `src/l2_marketdata/regime_proposal_analyzer.py` | 🔵 WRAPPED | Proposal analyzer for DecisionEngine |
| `src/l2_marketdata/volatility_regime.py` | 🔵 WRAPPED | Volatility-axis classifier |
| `src/l2_marketdata/nav_regime.py` | 🔵 WRAPPED | NAV-state classifier |

---

## Rules

1. **`main.py` may import only `core_engine.*`** (already enforced — Step 3).
2. **WRAPPED modules may not be imported by `main.py` or top-level scripts.**
3. **LEGACY modules accept no new edits** — bug-fix only with explicit waiver.
4. **QUARANTINED modules accept zero edits** — they move to `_archive/` in Step 5.
5. **New code imports the canonical (ACTIVE) path** listed in each family's header.

## Promotion / Demotion Path

```
QUARANTINED  ─────►  _archive/        (Step 5)
LEGACY       ─────►  _archive/        (after Phase 7 cutover validates main.py)
WRAPPED      ─◄────  ACTIVE           (only when a façade engine is split)
```

A LEGACY module may **not** be promoted back to ACTIVE without first deleting
the canonical path it duplicates and running the full test suite.

---

## Verification

```bash
# JSON manifest is valid
python3 -c "import json; json.load(open('MODULE_FREEZE_MANIFEST.json'))" && echo "✓ valid"

# Count by status
python3 -c "
import json
m = json.load(open('MODULE_FREEZE_MANIFEST.json'))
from collections import Counter
c = Counter()
for fam in m['families'].values():
    for mod in fam['modules']:
        c[mod['status']] += 1
for k,v in sorted(c.items()): print(f'{k:14s} {v}')
"
```

✅ **Step 4 complete.** No files moved or deleted; 46 modules across 8 families now have explicit lifecycle status.
