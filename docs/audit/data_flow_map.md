# Data Flow Map

Traced from Phase 1 wiring analysis and confirmed against the Phase 2 dry-run session log.

## Primary cycle data flow (confirmed live, this session)

```
market_data_websocket / market_data (REST)
        │  prices, klines
        ▼
NativeSharedState (shared, in-process)
        │
        ▼
symbol_discovery (wallet scan, per cycle)  ──► updates traded-symbol universe
        │                                         │
        │                                         ▼
        │                              market_data_websocket universe update
        │                              (triggers WS disconnect/reconnect)
        ▼
SignalManagerBridge.get_all_signals()
   ├── LegacySignalAdapter → MLForecaster.generate_signals()
   │        │  per-symbol: schema, probs, action, confidence
   │        ▼
   │     PERSIST_GATE (2-bar confirmation streak)  ──► holds until streak=2/2
   │        │
   │        ▼
   │     ConfFloor check (required=0.9500 this session, backtest-derived)
   │        │  only signals ≥ floor pass through
   │        ▼
   ├── NativeSignalEngine cross-check (RSI/MACD/MA/Momentum) — nudges confidence ±0.05
   └── PaperModeSignalGenerator (idle this session — paper_mode=False)
        │
        ▼  (0 signals passed through in this session)
SituationEngineImpl.get_all_signals()  [core_engine/implementations.py]
        │
        ▼
DecisionEngineImpl.make_buy_decision / make_sell_decision  [core_engine/implementations.py — the actual live path, NOT NativeDecisionEngine.decide()]
   ├── evaluate_signal() → arbitration_engine.evaluate() — CORRECTED: this IS invoked live, not dead (see current_state_assessment.md)
   └── consumes: ConcentrationGuard, RegimeGate, MarketRegimeDetector (via NativeDecisionEngine.decide(), only reachable through the separate, inert NativeOrchestrator.run_loop() path)
        │  0 decisions this session (no signal reached decision-making, not because arbitration is unreachable)
        ▼
NativeExecutor.execute(decisions)  [orchestrator.py:603]  — GATED OUT in dry-run mode
        │  (main.py:362 `if mode != "dry-run"`: skipped entirely)
        ▼
[would flow to: NativeCapitalAllocator.allocate_for_buy → AdaptiveCapitalEngine /
 DailyCompoundingPolicy → order_execution.py → exchange_client.py → Binance API]
```

## Producer → transport → consumer detail

| Flow | Producer | Transport | Consumer | Validation/error handling | Observed this session |
|---|---|---|---|---|---|
| Price/kline data | `market_data.py`, `market_data_websocket.py` | In-process shared state (`NativeSharedState`), direct method calls | `symbol_rotator.py`, `MLForecaster`, `NativeDecisionEngine`, `tp_sl_engine.py` | Staleness threshold (`stale_threshold_sec`, 30s default) | Yes — 64 symbols priced at startup, kline pre-fetch for 10 symbols completed in 8s |
| Wallet holdings → symbol universe | `symbol_discovery.py` (REST wallet scan) | Shared state update, triggers `market_data_websocket` re-subscribe | `market_data_websocket.py`, `MLForecaster` | None observed (empty result silently accepted, not retried faster) | Yes — 0 symbols initially, 5 by cycle 2; each change triggers a full WS reconnect |
| ML signal generation | `agents/ml_forecaster.py` (legacy, invoked via `LegacySignalAdapter`) | Direct async call `ml_forecaster.generate_signals()` | `SignalManagerBridge.get_all_signals()` | `signal_adapter_timeout_sec` (20s default); PERSIST_GATE streak requirement; ConfFloor threshold | Yes — every cycle, BNBUSDT conf=0.83 candidate consistently held by PERSIST_GATE and would additionally fail the 0.95 ConfFloor |
| Signal cross-check | `signals.py` (`NativeSignalEngine`) | In-process call from `SignalManagerBridge` | Adjusts ML signal confidence ±0.05 | RSI/MACD/MA/Momentum agreement heuristic | Not independently confirmed in log output (no per-indicator log lines observed at INFO level) |
| Signals → decisions | `SignalManagerBridge` (via `SituationEngineImpl.get_all_signals`) | Direct call from `main.py::trading_cycle()` | `NativeDecisionEngine.decide()` | `ConcentrationGuard`, `RegimeGate` (orderbook imbalance veto) | Yes — 0 signals reached this stage in all 3 cycles (gated upstream) |
| Decisions → arbitration gate feedback | `main.py` (post-execution, lines ~368-393) | Direct calls: `arb_engine.record_buy()`, `.record_sl_exit()`, `.record_win()`, etc. | `NativeArbitrationEngine` internal counters (loss streaks, cooldowns) | Only fires when `mode != "dry-run"` AND a decision executed successfully | **No — not exercised in dry-run** (execution phase skipped entirely). **CORRECTED:** this outbound feedback path is in addition to, not instead of, the inbound `evaluate_signal()` → `arbitration_engine.evaluate()` call in `implementations.py::make_buy_decision`/`make_sell_decision`, which IS the live gating path (see `current_state_assessment.md`). The earlier claim that arbitration has "no producer feeding it" was incorrect. |
| Decisions → execution | `NativeDecisionEngine` | Direct call, `orchestrator.py:603` / `main.py` execute phase | `NativeExecutor` → `SafeExecutionEngineImpl.place_buy_order/place_sell_order` → `exchange_client.py` | `mode != "dry-run"` gate; order validation (`validate_order`); FIX #2 idempotent SELL guard via `bounded_cache` | Not exercised — dry-run skips this phase by design |
| Fills → position state | Either `NativeFillTracker` (idle by default) or `NativePollingCoordinator`'s fills-reconciliation loop, **or** `NativeExecutor`'s own post-fill bookkeeping (compensating for `fill_tracker=None`) | In-process shared-state writes | `NativePositionManager`, `tp_sl_engine.py` | — | Not exercised — 0 executions this session |
| Runtime state → disk | `NativeSharedState` | `NativeRuntimeStateExporter`, periodic snapshot (10s default) | `runtime_state_snapshot.json` (read by `trade_monitor.py` and other external monitoring scripts) | Atomic write (tmpfile + `os.replace`) | Confirmed restored at startup ("Restored runtime state from runtime_state_snapshot.json"); periodic re-write not directly observed in stdout (writes to file, not log) |
| Trade journal | `NativeTradeJournal` | File writes to `logs/` | Read back by `position_hydration_engine` on next startup (`local journal recovery`) | — | Attempted this session ("Attempting local journal recovery...") — outcome not distinctly logged before falling through to the exchange-fills path that errored |
| Position recovery on restart | `position_hydration_engine.py` | Two paths: local journal, then exchange trade history via `exchange_client.get_all_orders()` | `NativeSharedState` (hydrated positions) | **Broken**: `NativeExchangeClient` has no `get_all_orders` method — `AttributeError` caught, logged as ERROR, falls back to "assuming fresh account" | **Yes — confirmed broken this session** (see `runtime_timeline.md` Finding #3). Harmless this run only because the account genuinely holds 0 positions |

## Format/contract mismatches identified

1. **`position_hydration_engine.py` expects `NativeExchangeClient.get_all_orders()` — the method doesn't exist on the class.** This is a producer-consumer contract break within native code itself (not even a native/legacy mismatch) — confirmed by runtime `AttributeError`. Needs a source read of `exchange_client.py` to determine the correct method name (likely renamed or never implemented) and a fix in Phase 4 remediation.
2. **Arbitration engine has no live producer feeding its `.evaluate()` path** — signals and decisions never call into it for a verdict; only `main.py`'s post-execution outcome-recording calls (`record_buy`, `record_win`, etc.) touch it, and even those never fire in dry-run/no-execution paths. Any config flags gating its behavior (`DOWNTREND_MARGIN`, `SYMBOL_DOWNTREND_VETO_ENABLED`, etc. — see `configuration_map.md`) are consequently inert.
3. **Two independent config systems** (`BootstrapConfig` vs `config_loader.py`) mean a data-flow consumer reading, say, `TAKE_PROFIT_PCT` via `config_loader.py` may see a different effective value than a consumer reading `TP_PCT` via `BootstrapConfig` — a latent state-inconsistency risk flagged in `configuration_map.md`, not exercised/proven in this session (would require instrumenting both config objects side-by-side, out of scope for this pass).
