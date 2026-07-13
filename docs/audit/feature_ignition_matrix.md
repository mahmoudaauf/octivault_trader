# Feature Ignition Matrix

"Connected" = a wiring path exists from trigger to component. "Observed during run" is
left as **TBD (Phase 2)** for anything that requires a live process — Phase 1 is
static/wiring analysis only; this column will be filled in after the controlled runtime
observation session.

| Feature | Intended trigger | Actual trigger | Connected? | Observed during run | Result (static analysis) |
|---|---|---|---|---|---|
| Symbol discovery | App startup / periodic rescan | `orchestrator` startup + `symbol_discovery_interval_sec` (900s) timer | Yes | TBD (Phase 2) | Wired; gated by `symbol_discovery_enabled` (default True) |
| Symbol rotation | Per trading cycle | `orchestrator.py:379,415` → `symbol_rotator.maybe_rotate()` | Yes | TBD (Phase 2) | Wired; non-fatal on internal failure |
| Market data poll (REST) | App startup | `NativeOrchestrator.start()` → `market_data.start()` (`orchestrator.py:150`) | Yes | TBD (Phase 2) | Wired |
| Market data (WebSocket) | App startup, if configured | `NativeOrchestrator.start()` → `market_data_ws.start()` (`orchestrator.py:151-152`), conditional on non-None | Yes | TBD (Phase 2) | Wired |
| Gated polling (orders/balance/positions/fills) | App startup, default config | `NativeOrchestrator.start()` → `polling_coordinator.start()` (`orchestrator.py:155-158`), preferred over `balance_sync` | Yes | TBD (Phase 2) | Wired; active-trades gate may reduce actual poll frequency when no open trades |
| Legacy balance sync / fill tracker | App startup, `polling_enabled=False` only | Same orchestrator start block, `elif`/conditional branch | Yes, but **not the default path** | TBD (Phase 2) | Correctly idle under default config — not a bug |
| Signal generation (ML forecaster via bridge) | Every decision cycle | `SignalManagerBridge.get_all_signals()` called from `implementations.py::SituationEngineImpl.get_all_signals` | Yes | TBD (Phase 2) | Wired, primary live signal path (bridge checked before raw legacy `signal_manager` fallback) |
| Signal generation (paper synthetic) | `paper_mode` or `synthetic_live_signals_enabled` | `bootstrap.py:650`, `enable_paper_signals = cfg.paper_mode or cfg.synthetic_live_signals_enabled` | Yes, conditional | TBD (Phase 2) | Off by default (`paper_mode=False`, `synthetic_live_signals_enabled=False`) |
| Decision engine | Every cycle | `orchestrator.py:598` → `decision_engine.decide(...)` | Yes | TBD (Phase 2) | Wired |
| Concentration guard | Every BUY decision | `decisions.py:127` | Yes | TBD (Phase 2) | Wired but **untested** |
| Regime gate | Every BUY decision (decisions.py) and arbitration path | `decisions.py:131`, `arbitration_engine.py:45` | Yes (decisions.py path); arbitration path unreachable | TBD (Phase 2) | Reachable only via `decisions.py` since `arbitration_engine` itself is never called |
| **Arbitration engine** | Meta-arbitration across signals, every BUY/SELL decision | **CORRECTED (Phase 4):** `implementations.py::DecisionEngineImpl.make_buy_decision`/`make_sell_decision` → `evaluate_signal()` → `arbitration_engine.evaluate()`, on the real production path (`main.py` → façade `DecisionEngine`) | **Yes** | Not observed in the Phase 2 session — but only because zero signals reached decision-making that session (upstream gating), not because arbitration is unreachable | Live and wired; the original Phase 1/2 "dead" finding was a research gap (missed `implementations.py`) combined with a session where the funnel never got that far upstream |
| Capital allocation | Every BUY decision | Called from executor/decision path via `allocate_for_buy(symbol)` | Yes | TBD (Phase 2) | Wired |
| Adaptive capital engine | Every `allocate_for_buy`, NAV≥$100 | `capital_allocator.py:264` | Yes, conditional on NAV and `adaptive_capital_engine_enabled` (default True) | TBD (Phase 2) | Wired; may appear "idle" on small accounts (<$100 NAV) — correctly idle, not broken |
| Daily compounding | Every `allocate_for_buy` | `capital_allocator.py:151` → `DailyCompoundingPolicy.sizing_nav()` | Yes | TBD (Phase 2) | Wired |
| Objective Feedback Controller | Heartbeat timer (≥60s) | `orchestrator.py:227-228` → `ofc.start()`; internal `_run_loop` | Yes, conditional on `ofc_enabled` (default True) | TBD (Phase 2) | Wired |
| Order execution | Per decision batch | `orchestrator.py:603` → `executor.execute(decisions)` | Yes | TBD (Phase 2) | Wired |
| TP/SL triggers | Per position per cycle | `orchestrator.py:531` → `tp_sl_engine.check_triggers()` | Yes | TBD (Phase 2) | Wired |
| TP/SL aged-position recalculation | Every 300s | `orchestrator.py:614` → `recalculate_aged_positions()`, inside `_phase_recover()` | Yes, but only if `run_cycle()`/`run_loop()` actually drives the cycle | TBD (Phase 2) — **need to confirm whether main.py's own `trading_cycle()` drives this or if it's orchestrator-loop-only** | Potential gap: `main.py` has its own `trading_cycle()` (line 131) using facade engines directly via `CadenceScheduler`, while `orchestrator.run_loop()` is noted as "effectively unused in production, only exercised by tests." Need Phase 2 confirmation of whether `_phase_recover()` fires in the real process |
| Symbol performance tracking | Consulted by arbitration engine | `arbitration_engine.py:53`, live via the `evaluate()` path above | **CORRECTED:** Yes, reachable | Not observed this session for the same upstream-gating reason | Live — no longer classified as unignited |
| Fear & Greed fetch | App startup | `bootstrap.py:883` → `fear_greed_fetcher.start()` (started directly in `build_components`, not via orchestrator) | Yes | TBD (Phase 2) | Wired, untested |
| NAV protection/attribution | Unclear — intended per-cycle NAV check | **No confirmed instantiation site** outside its own module | **No (unconfirmed)** | TBD (Phase 2) — grep runtime logs for any `NAVProtectionEngine`/`NAVAttributionEngine` activity | Candidate unwired feature; needs explicit runtime confirmation since static grep found no external instantiation |
| Model management / auto-cleanup | App startup (model load) | `bootstrap.py:675` (inline import), plus `MODEL_AUTO_CLEANUP_INCOMPATIBLE` flag | Yes | TBD (Phase 2) | Wired |
| Background model retraining | Drift detection / startup, inside live process | `agents/ml_forecaster.py` → `ModelTrainer` calls at 3 points | Yes (indirect, via legacy bridge) | TBD (Phase 2) | Wired but runs inside legacy `MLForecaster`, not native code |
| Weekly offline retraining | Cron (external, Sunday) | `retrain_weekly.py`, run manually/externally — not part of supervised runtime | N/A (by design) | N/A — out of scope for `supervisor.sh`-driven observation | Correctly out-of-band |
| Health monitoring | Continuous | `bootstrap.py:927` instantiation confirmed; loop mechanics not traced this pass | Partially confirmed | TBD (Phase 2) | Needs Phase 2 to confirm actual health-check cadence |
| Watchdog | Continuous | Instantiated in bootstrap; loop mechanics not traced this pass | Partially confirmed | TBD (Phase 2) | Needs Phase 2 confirmation |
| Prometheus export | Continuous, if `PROMETHEUS_EXPORT_PATH` set | Instantiated in bootstrap; gated by env var | Yes, conditional | TBD (Phase 2) | Off unless env var set — need to confirm current `.env` state before declaring idle-by-config vs idle-by-omission |
| Telemetry export | Continuous, if `TELEMETRY_EXPORT_PATH` set | `telemetry_exporter.start()` in `build_components` | Yes, conditional | TBD (Phase 2) | Same caveat as Prometheus export |
| Runtime state snapshot | Every 10s | `runtime_state_exporter.start()` in `build_components` | Yes | TBD (Phase 2) | Wired — `runtime_state_snapshot.json` is a modified/dirty file in this working tree, consistent with recent activity |
| Strategy validation | Manual CLI invocation (intended) | None — no caller found | No | N/A | Standalone offline tool, correctly idle (not yet integrated), not a runtime bug |
| Post-reset strategy overrides (`STRATEGY_OPTIMIZATION_v2.py`) | Unclear — appears to have been a manual one-off patch | None found | No | N/A | Orphaned config file — candidate for cleanup or explicit documentation of intended re-use |

## Highest-confidence findings carried into Phase 2

1. ~~Arbitration engine is fully built but never invoked.~~ **CORRECTED in Phase 4: it IS
   invoked, via `implementations.py`, on the real production decision path — see
   `current_state_assessment.md` for the full correction.**
2. **TP/SL aged-position recalculation** depends on which cycle driver is actually live —
   `main.py`'s own `trading_cycle()` vs `NativeOrchestrator.run_loop()`. Needs runtime
   confirmation of which one governs `_phase_recover()`.
3. **NAV protection/attribution** has tests but no confirmed instantiation site — needs
   explicit runtime check for whether it's ever constructed.
4. Several Tier B features (concentration guard, symbol performance tracker, fear/greed
   fetcher) are wired but have zero test coverage — not a wiring gap, but a monitoring/quality gap.
