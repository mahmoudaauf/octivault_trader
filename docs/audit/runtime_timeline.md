# Runtime Timeline — Controlled Observation

## Session parameters

- **Command:** `.venv/bin/python3 main.py --mode=dry-run --cycles=3 --interval=5 --duration=90s`
- **Working tree:** current dirty `phase-3/wiring` tree (not stashed), per user decision
- **Environment:** real (non-testnet) Binance credentials from `.env`, `PAPER_MODE=False`, `TRADING_MODE=live`
- **Safety constraint honored:** `--mode=dry-run` — execution phase (`main.py:362`, `if mode != "dry-run"`) never entered. Confirmed: `exe=0` on every cycle, and no `place_buy_order`/`place_sell_order` log lines appear anywhere in the capture.
- **Log source:** `logs/run_latest.log` lines 38410-38722 (this run's slice within the shared rotating log), cross-checked against stdout capture
- **Exit code:** 0 (clean shutdown), but see Finding #1 below — actual process lifetime exceeded the requested budget

## Timeline

| Time | Component | Event | Result | Evidence |
|---|---|---|---|---|
| T+0.0s (00:32:06) | Process | `main.py` started, `Mode=dry-run duration=90s capital=1000.0 cycles=3 native=True compat=True` | Started | log:38412 |
| T+0.0s | Bootstrap | Runtime state restored from `runtime_state_snapshot.json` | Success | log:38414 |
| T+0.0s | PollingCoordinator | Initialized (orders=25s, balance=40s, positions=25s, price_refresh=60s, gate=enabled) | Success | log:38415-38416 |
| T+0.0s | SymbolRotator | Instantiated (TOP_N=8, interval=2h) | Success | log:38418 |
| T+0.0s | Bootstrap | `native bootstrap: testnet=False symbols=0 (cycle-dynamic) polling=enabled` | Success | log:38419 |
| T+0.0s | market_data_websocket | Initialized for 10 symbols, pre-fetching 3000 klines via REST | Started | log:38420-38421 |
| T+8s (00:32:14) | market_data_websocket | Kline pre-fetch complete: 10 fetched, 0 failed | Success | log:38422 |
| T+8s | Bootstrap | `paper_mode=False, symbols=0` | Confirmed | log:38423 |
| T+16s (00:32:22) | ModelManager | Initialized | Success | log:38424 — **8s gap between T+8s and T+16s with no intervening log lines; likely TensorFlow/Keras import cost, not investigated further this pass** |
| T+16s | MLForecaster | Initialized (model_manager=✓, market_data_feed=✓) | Success | log:38425 |
| T+16s | SymbolScreener | Initialized (interval=3600s) | Success | log:38427-38434 |
| T+16s | LegacySignalAdapter / SignalManagerBridge | Initialized (`legacy=no, paper_mode=no, forecaster=yes, screener=yes`) | Success | log:38437-38438 |
| T+16s | ObjectiveFeedbackController | Restored knobs from artefact (`conf_floor=0.50 size_mult=0.50`), initialised | Success | log:38439-38440 |
| T+16s | Bootstrap | Fetched symbol filters (0 symbols, since discovery hasn't run yet) | Success | log:38441-38442 |
| T+17s | symbol_performance_tracker | Restored state for 18 symbols (persisted from prior sessions) | Success | log:38443 |
| T+17s (00:32:23) | fear_greed | Fetched: 28 (Fear), refreshes hourly | Success | log:38444-38445 |
| T+17s | integration | Native app_ctx built: 26 keys, testnet=False, compat=True | Success | log:38446 |
| T+17s | integration | Engines wired to L0-L8 components, including `DecisionEngine → arbitration_engine (L5)` | Success (wired) | log:38452 — **wiring confirmed but see Finding #2: never invoked during the run** |
| T+17s | integration | `⚠️ startup_orchestrator not found in app_ctx` | Warning (expected — legacy fallback absent, native path used instead) | log:38457 |
| T+17s | main | All 5 engines online | Success | log:38459 |
| T+18s (00:32:23) | market_data_websocket | Starting WebSocket (10 symbols, 1 timeframe) | Started | log:38460 |
| T+18s | PollingCoordinator | All polling loops started | Started | log:38461-38462, 38472-38476 |
| T+18s | orchestrator | Initial data ready (prices=64 symbols, balance=57.85 USDT) | Success | log:38463 |
| T+18s | orchestrator → startup_state_machine | Startup sequence: BOOTING → HYDRATING → RECONCILING → VALIDATING → READY | Success, 0.9s total | log:38464-38489 |
| T+18s | position_hydration_engine | Balance fetch: $57.85 (free=$57.85, locked=$0.00) | Success | log:38478 |
| T+18s | position_hydration_engine | Local journal recovery attempted, then exchange trade history recovery attempted | Attempted | log:38480-38481 |
| T+18s | position_hydration_engine | **`Failed to fetch exchange fills: 'NativeExchangeClient' object has no attribute 'get_all_orders'`** | **ERROR** | log:38482 — **Finding #3, real bug, confirmed at runtime** |
| T+18s | position_hydration_engine | Falls back: "No fills found in journal or exchange; assuming fresh account" | Degraded (masked by error) | log:38483 |
| T+18s | position_hydration_engine | Hydration complete: 0 positions, $0.00 value | Success (vacuously, since account genuinely has 0 positions right now) | log:38487 |
| T+19s (00:32:24) | orchestrator | Startup complete; trading ready | Success | log — orchestrator |
| T+19s | market_data_websocket | WebSocket connected, receiving messages | Success | — |
| T+35s (00:32:41) | orchestrator | Session anchor NAV = 57.87 USDT (peak reset) | Success | — |
| T+35s | TPSLEngine | Starting (Tier 2: fee-aware + time + trailing) | Started | — |
| T+35s | ObjectiveFeedbackController | Started, heartbeat every 900s | Started | — |
| T+35s | implementations | "Native orchestrator started (market_data + balance_sync)" | Success — **log message text says "balance_sync" though PollingCoordinator (not legacy balance_sync) is the active poller; a stale log string, not a functional bug** | — |
| T+35s | 5 façade engines | MarketAccountEngine, SituationEngine, DecisionEngine, SafeExecutionEngine, OperationsEngine all initialize | Success | — |
| T+36s (00:32:42) | symbol_discovery | Wallet scan: **0 symbols discovered** from holdings | Empty result | — |
| T+36s | market_data_websocket | WS universe updated to 2 symbols (BTCUSDT, ETHUSDT fallback), reconnects | Success, but causes disconnect/reconnect churn | — |
| T+36s | MLForecaster | `run_once` starts; `accepted_symbols empty`, falls back to 10 DEFAULT_SYMBOLS | Warning, fallback engaged | — |
| T+36-41s | MLForecaster / ModelManager | Loads 10 `.keras` models (BTC, ETH, BNB, SOL, XRP, ADA, DOGE, LINK, AVAX, MATIC), runs inference on each | Success | — |
| T+40s (00:32:46) | MLForecaster | BNBUSDT: `action=buy, conf=0.83` — highest-confidence signal this cycle | Signal generated internally | — |
| T+40s | MLForecaster | `PERSIST_GATE BNBUSDT BUY held: streak=1/2` — requires 2 consecutive confirming bars before emitting | **Held, not emitted** | — |
| T+42s (00:32:48) | MLForecaster | ConfBacktest: `required=0.9500 break_even=0.9503`; historical confidence buckets 0.65-0.85 show **0-12.2% win rate, negative EV** | Confirms an extremely high confidence floor is intentional given current model calibration | — |
| T+43s (00:32:49) | implementations | `QUANT_LOOP_SUMMARY`: `market_regime=DOWNTREND portfolio_state=CASH_HEAVY allowed=False execution_result=NONE` | Cycle 1 result | — |
| T+43s | main | **cycle 00001**: 7987.4ms, nav=57.85, sigs=0, dec=0, exe=0, [RUDEO], OK | Complete | — |
| T+48s (00:32:54) | main | Symbols discovered: `[] → ['ADAUSDT','BNBUSDT','LUNCUSDT','PEPEUSDT','SEIUSDT']` — discovery now finds 5 symbols | Delayed discovery (took ~2 full cycles to populate) | — |
| T+48s | market_data_websocket | WS universe updated to 7 symbols; disconnects and reconnects again | Success, more churn | — |
| T+49s | MLForecaster | Second run_once; BNBUSDT again `conf=0.83`, `PERSIST_GATE ... streak=1/2` — **streak did not advance from cycle 1** | Held again | — |
| T+49s | main | **cycle 00002**: 1366.2ms, nav=57.85, sigs=0, dec=0, exe=0, [RUDEO], OK | Complete | — |
| T+54s (00:33:00) | MLForecaster | Third run_once; BNBUSDT again `conf=0.83`, `streak=1/2` — still not advancing | Held again | — |
| T+54s | main | **cycle 00003**: 190.2ms, nav=57.85, sigs=0, dec=0, exe=0, [RUDEO], OK | Complete | — |
| T+59s (00:33:05) | main | `Cycle budget exhausted — exiting loop` | Correct behavior — `--cycles=3` honored | — |
| T+59s | 5 façade engines | Shutdown sequence, all report clean shutdown | Success | — |
| T+59-60s (00:33:05-06) | native bootstrap | "Native bootstrap shut down"; `Total cycles: 3` logged | Nominally complete | — |
| **T+80s (00:33:26)** | **ModelTrainer_BTCUSDT** | **Training progress epoch=5/15 — logged AFTER the "shut down" line above** | **Orphaned background task, see Finding #1** | — |
| T+111s (00:33:57) | ModelTrainer_BTCUSDT | epoch=10/15 | Still running | — |
| T+142s (00:34:28) | ModelTrainer_BTCUSDT | epoch=15/15; "Skipping model persistence for BTCUSDT (save_model_artifact=False)" | Training completes, process finally exits | — |

## Findings from this session (runtime-confirmed)

1. **A background model-training task survives the "clean shutdown" and keeps the process alive well past the requested budget.** `--duration=90s --cycles=3` requested a bounded ~90s run; the process actually ran ~142s (00:32:06 to 00:34:28) because a `ModelTrainer_BTCUSDT` background training task — queued during MLForecaster's startup full-tier refresh at T+40s — was neither awaited before shutdown nor cancelled by it. The main loop logged "Clean shutdown complete" and "Native bootstrap shut down" at T+59-60s while this task kept running for another ~82 seconds. This matches the original audit spec's explicit concern: "tasks that survive shutdown," "tasks whose exceptions are never inspected." In production under `supervisor.sh`, this likely means graceful shutdown is not actually graceful/prompt when a training cycle is in flight.

2. **`NativeArbitrationEngine` is confirmed silent at runtime**, not just statically. Across the entire captured session — bootstrap, 3 full cycles, shutdown — the only log line mentioning "arbitration" is the one-time wiring confirmation (`✓ Wired DecisionEngine → arbitration_engine (L5)`). No `.evaluate()`/`.evaluate_gates()` activity, no arbitration-related decision output. This corroborates the Phase 1 static finding.

3. **Position hydration error is real, not theoretical.** `'NativeExchangeClient' object has no attribute 'get_all_orders'` fires on every startup, is caught, and silently degrades to "assuming fresh account." In this session that's harmless (account genuinely holds 0 positions — 100% cash, $57.85). But the code path that's supposed to recover positions from exchange history on restart is broken, which would matter if the bot restarted while holding open positions.

4. **Confidence floor never came close to being crossed.** Every candidate signal in this session topped out around conf=0.83-0.85 (BNBUSDT most consistently), against a `required=0.9500` confidence floor derived from live confidence-bucket backtesting (buckets 0.65-0.85 show 0-12.2% win rate / negative EV in the current calibration). Combined with the `PERSIST_GATE` 2-bar confirmation requirement (which never advanced past streak=1/2 in 3 cycles spanning ~54 seconds, consistent with the 5-minute signal timeframe not having produced a new closed bar), zero signals passed through to the decision engine in this session. `sigs=0 dec=0 exe=0` on every cycle is explained, not anomalous.

5. **Symbol discovery is slow to populate on cold start.** Wallet scan returned 0 symbols on the very first attempt (T+36s) and only found 5 real holdings by the second cycle (T+48s, roughly one polling interval later). Each time the discovered symbol set changes, `market_data_websocket` fully disconnects and reconnects — 2 reconnects observed in a 54-second, 3-cycle window. This is inefficient but not broken; worth flagging as a monitoring item (WS churn under fast-changing symbol sets), not this audit's priority finding.

## Not observed / needs Phase 3 or environment-specific follow-up

- Prometheus/telemetry export: `PROMETHEUS_EXPORT_PATH`/`TELEMETRY_EXPORT_PATH` were not evaluated in this pass — no export-related log lines appeared, consistent with them being unset (idle-by-config), but not independently confirmed from `.env` contents.
- `NAVProtectionEngine`/`NAVAttributionEngine`: no log lines observed referencing NAV protection during this session — consistent with the Phase 1 finding that no instantiation site was found; this run does not contradict that.
- Health monitor / watchdog cadence: instantiated per Phase 1, but no periodic health-check log lines were observed in this ~142s window — likely because their check interval exceeds the observation window; not confirmed broken, just not observed.
