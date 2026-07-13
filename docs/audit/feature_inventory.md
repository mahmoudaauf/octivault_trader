# Feature Inventory

Features detected via static/wiring analysis in Phase 1. "Undocumented" features are
marked — most native/ modules carry no external architecture doc beyond code comments.

| Feature | Owning component(s) | Documented elsewhere? | Notes |
|---|---|---|---|
| Wallet-based symbol auto-discovery | `symbol_discovery.py` | Partially (`.LEGACY_TO_NATIVE_MAP.md`) | Gated by `symbol_discovery_enabled` (default True); rescans every 900s |
| Symbol rotation (ATR floor + preferred-symbol bonus) | `symbol_rotator.py` | Recent commit message only | Per-cycle, non-fatal on failure |
| Gated staggered polling (orders/balance/positions/fills) | `polling_coordinator.py` | No | Default poller substitute for legacy `balance_sync`/`fill_tracker`; active-trades gate skips polling when no open trades |
| Legacy balance/fill polling (fallback path) | `balance_sync.py`, `fill_tracker.py` | No | Only active when `polling_enabled=False` (non-default) |
| Market data (REST + WebSocket) | `market_data.py`, `market_data_websocket.py` | No | Both constructed in bootstrap, started by `NativeOrchestrator.start()` |
| Market regime detection | `market_regime_detector.py` | No | Per-symbol trend classification (MA5 vs MA20) |
| Signal generation — ML forecaster (legacy) bridged to native | `agents/ml_forecaster.py` via `legacy_signal_adapter.py` → `signal_manager_bridge.py` | Partially (`LEGACY_SIGNAL_INTEGRATION.md`) | Actively drives live signal generation; MLForecaster itself imports `src.l5_strategy.*` |
| Signal generation — paper mode synthetic signals | `paper_signal_generator.py` | No | Gated by `paper_mode or synthetic_live_signals_enabled` |
| Native signal cross-check (RSI/MACD/MA/Momentum) | `signals.py` (`NativeSignalEngine`) | No | Nudges ML confidence ±0.05 based on indicator agreement, inside `signal_manager_bridge.py` |
| Decision engine (BUY/SELL/HOLD) | `decisions.py` | No | Consumes concentration guard, regime gate |
| Concentration/cluster exposure guard | `concentration_guard.py` | No | **Untested** — no test file found |
| Regime gate (orderbook imbalance veto) | `regime_gate.py` | No | Used by both `decisions.py` and `arbitration_engine.py` |
| Arbitration engine (meta-arbitration across signals) | `arbitration_engine.py` | No | **Instantiated but never called at runtime** — see ignition matrix |
| Capital allocation (Kelly-based sizing) | `capital_allocator.py` | No | Core sizing entry point for every BUY |
| Adaptive capital engine (feedback-driven risk fraction) | `adaptive_capital_engine.py` | No | Only engages above $100 NAV |
| Daily compounding policy | `daily_compounding.py` (new) | No | Caps sizing NAV using persisted daily state; fully wired via `capital_allocator.py` |
| Objective Feedback Controller (auto-calibration) | `objective_feedback_controller.py` | No | Heartbeat loop (≥60s), gated by `ofc_enabled` (default True) |
| Order execution | `executor.py`, `order_execution.py` | No | Duplicates fill/position bookkeeping to compensate for `fill_tracker=None` default |
| TP/SL engine | `tp_sl_engine.py` | No | Armed on start, periodic aged-position recalculation every 300s |
| Symbol performance tracking / throttling | `symbol_performance_tracker.py` | No | **Untested** — no test file found; used by `arbitration_engine.py` (which is itself unignited) |
| Fear & Greed index fetch | `fear_greed.py` | No | **Untested** — no test file found |
| NAV protection / attribution | `nav_protection.py` | Has tests | **No confirmed instantiation site** outside its own module — candidate unwired feature |
| Model management (TF Keras load/cache/quarantine) | `model_manager.py` | No | Auto-cleans incompatible artifacts if `MODEL_AUTO_CLEANUP_INCOMPATIBLE` set |
| Weekly offline model retraining | `retrain_weekly.py` + `src/l5_strategy/model_trainer.py` | Yes (docstring: cron-driven) | Standalone, not part of live runtime; writes models that `ml_forecaster.py` later loads |
| Background/startup model retraining | `agents/ml_forecaster.py` → `ModelTrainer` | No | Triggered inline by MLForecaster on drift/startup, inside the live process |
| Health monitoring | `health_monitor.py` | No | — |
| Mode management (operating-mode state machine) | `mode_manager.py` | No | — |
| Watchdog / heartbeat | `watchdog.py` | No | — |
| Prometheus metrics export | `prometheus_exporter.py` | No | Gated by `PROMETHEUS_EXPORT_PATH` (empty = disabled) |
| Telemetry export | `telemetry_export.py` | No | Gated by `TELEMETRY_EXPORT_PATH` (empty = disabled) |
| Runtime state snapshotting | `runtime_state.py` | No | Writes `runtime_state_snapshot.json` every 10s by default |
| Trade journal (crash-safe) | `trade_journal.py` | No | — |
| Recovery engine | `recovery_engine.py` | No | — |
| Portfolio recovery / dust disposal | `portfolio_recovery.py` | No | Not independently deep-dived this pass |
| Position hydration on startup | `position_hydration_engine.py` | No | Gates BUY until hydration completes, via `startup_state_machine.py` |
| Safety order (OCO) management | `safety_order_manager.py` | No | — |
| Bounded cache (idempotency) | `bounded_cache.py` | No | — |
| Compat stub layer | `compat.py` | No | Provides null stubs for legacy app_ctx keys; call-site wiring unconfirmed |
| Strategy validation (offline KPI check) | `strategy_validation.py` (new, untracked) | No | **Not imported anywhere** except its own test — unwired offline tool |
| Post-reset strategy overrides | `config/STRATEGY_OPTIMIZATION_v2.py` | No | **Not imported by any active code** — dead/orphaned patch file |
| Canonical EV alignment config | `config/EV_ALIGNMENT_CONFIG.py` | Yes (docstring) | Documents intended usage but **not imported anywhere** in active code — aspirational/orphaned |
