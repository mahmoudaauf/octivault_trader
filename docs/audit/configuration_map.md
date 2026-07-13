# Configuration Map

## Top finding: two parallel, independently-parsed config systems

`core_engine/native/bootstrap.py::BootstrapConfig` (the composition root's config, ~50
fields) and `core_engine/native/config_loader.py::ConfigLoader`/`ConfigGroup` (a second
L0 config loader, used by `config/__init__.py` and `core_engine/native/__init__.py`)
read overlapping env vars **under different names**, into two separate config objects.
Neither is a strict superset of the other. This is a structural risk: changing one
value in one system silently does not change the other.

### Confirmed naming collisions (same concept, different env var name)

| Concept | `BootstrapConfig` (bootstrap.py) | `config_loader.py` |
|---|---|---|
| Exit slippage | `EXIT_SLIPPAGE_BPS` (fallback `CR_PRICE_SLIPPAGE_BPS`) | `EXIT_SLIP_BPS` |
| Daily compounding toggle | `DAILY_COMPOUNDING_ENABLED` | `COMPOUNDING_ENABLED` |
| Take profit / stop loss | `TP_PCT` / `SL_PCT` | `TAKE_PROFIT_PCT` / `STOP_LOSS_PCT` |
| Max position size | `MAX_POSITION_PCT` (same name, two independent parsers — duplicate-definition risk, not a spelling mismatch) | `MAX_POSITION_PCT` |
| Max drawdown | `MAX_DRAWDOWN_PCT` (same name, two independent parsers) | `MAX_DRAWDOWN_PCT` |

`tp_sl_engine.py` also reads `CR_PRICE_SLIPPAGE_BPS` directly, independent of both
config objects above — a third read site for the same concept.

## BootstrapConfig fields (`bootstrap.py:88-281`)

| Field | Default | Env var | Purpose |
|---|---|---|---|
| api_key | required | `BINANCE_API_KEY` | Exchange API key |
| api_secret | required | `BINANCE_API_SECRET` | Exchange API secret |
| testnet | False | `BINANCE_TESTNET` | Use Binance testnet endpoints |
| paper_mode | False | `PAPER_MODE` | Simulated $1000 USDT paper trading |
| synthetic_live_signals_enabled | False | `SYNTHETIC_LIVE_SIGNALS_ENABLED` | Opt-in synthetic signal generator on live (non-paper) runs |
| symbols | [] | `SYMBOLS` | Trading universe override (else auto wallet-scan) |
| market_data_poll_sec | 2.0 | `MARKET_DATA_POLL_SEC` | Market data poll cadence |
| symbol_discovery_enabled | True | `SYMBOL_DISCOVERY_ENABLED` | Auto-discover symbols from wallet |
| symbol_discovery_interval_sec | 900.0 | `SYMBOL_DISCOVERY_INTERVAL_SEC` | Rescan interval |
| symbol_discovery_empty_retry_sec | 300.0 | `SYMBOL_DISCOVERY_EMPTY_RETRY_SEC` | Retry interval when discovery yields nothing |
| klines_cache_size | 64 | `KLINES_CACHE_SIZE` | Kline cache depth |
| stale_threshold_sec | 30.0 | `STALE_THRESHOLD_SEC` | Market data staleness cutoff |
| signal_adapter_timeout_sec | 20.0 | `SIGNAL_ADAPTER_TIMEOUT_SEC` | Signal adapter call timeout |
| polling_enabled | True | `POLLING_ENABLED` | Use staggered gated polling (vs legacy aggressive REST) |
| polling_open_orders_interval_sec | 25.0 | `POLLING_OPEN_ORDERS_INTERVAL_SEC` | Open-orders poll cadence |
| polling_balance_interval_sec | 40.0 | `POLLING_BALANCE_INTERVAL_SEC` | Balance poll cadence |
| polling_position_interval_sec | 25.0 | `POLLING_POSITION_INTERVAL_SEC` | Position poll cadence |
| polling_enable_active_trades_gate | True | `POLLING_ENABLE_ACTIVE_TRADES_GATE` | Skip polling when no open trades |
| balance_poll_sec | 5.0 | `BALANCE_POLL_SEC` | Legacy balance sync interval (ignored if polling_enabled) |
| balance_min_refresh_interval_sec | 60.0 | `BALANCE_MIN_REFRESH_INTERVAL_SEC` | Min gap between forced balance refreshes |
| fill_tracker_poll_sec | 5.0 | `FILL_TRACKER_POLL_SEC` | Legacy fill tracker interval (ignored if polling_enabled) |
| kelly_fraction | 0.25 | `KELLY_FRACTION` | Kelly sizing fraction |
| max_position_size_pct | 5.0 | `MAX_POSITION_PCT` | Max position size as % equity |
| max_concurrent_positions | 10 | `MAX_CONCURRENT_POSITIONS` | Position count cap |
| min_order_usdt | 1.0 | `MIN_ORDER_USDT` | Minimum order notional |
| max_drawdown_pct | 10.0 | `MAX_DRAWDOWN_PCT` | Drawdown circuit breaker |
| daily_loss_limit_pct | 2.0 | `DAILY_LOSS_LIMIT_PCT` | Daily loss circuit breaker |
| risk_per_symbol_pct | 20.0 | `RISK_PER_SYMBOL_PCT` | Per-symbol risk allocation |
| capital_allocation_pct | 5.0 | `CAPITAL_ALLOCATION_PCT` | Capital allocation sizing |
| default_planned_quote | 12.0 | `DEFAULT_PLANNED_QUOTE` | Fixed quote-per-trade for small accounts |
| daily_compounding_enabled | True | `DAILY_COMPOUNDING_ENABLED` | Enables daily equity compounding into sizing |
| daily_compounding_state_path | logs/daily_compounding_state.json | `DAILY_COMPOUNDING_STATE_PATH` | Persisted compounding state file |
| quote_reserve_ratio | 0.10 | `QUOTE_RESERVE_RATIO` | Reserve % of quote currency held back |
| quote_min_reserve_usdt | 0.0 | `QUOTE_MIN_RESERVE_USDT` | Absolute quote reserve floor |
| max_total_exposure_pct | 60.0 | `MAX_TOTAL_EXPOSURE_PCT` | Total exposure cap |
| confidence_floor | 0.50 | `CONFIDENCE_FLOOR` | Minimum signal confidence to trade |
| max_cluster_exposure_pct | 40.0 | `MAX_CLUSTER_EXPOSURE_PCT` | Correlated-asset cluster exposure cap |
| taker_fee_bps | 10.0 | `TAKER_FEE_BPS` | Taker fee assumption |
| maker_fee_bps | 10.0 | `MAKER_FEE_BPS` | Maker fee assumption |
| exit_slippage_bps | 10.0 | `EXIT_SLIPPAGE_BPS` (fallback `CR_PRICE_SLIPPAGE_BPS`) | Exit slippage assumption |
| tp_pct | 0.03 | `TP_PCT` | Take-profit % |
| sl_pct | 0.02 | `SL_PCT` | Stop-loss % |
| signal_cooldown_sec | 0.0 | `SIGNAL_COOLDOWN_SEC` | Cooldown between signals per symbol |
| telemetry_capacity | 1024 | `TELEMETRY_CAPACITY` | Telemetry ring buffer size |
| telemetry_export_path | "" | `TELEMETRY_EXPORT_PATH` | Telemetry export file (empty=disabled) |
| telemetry_export_interval_sec | 10.0 | `TELEMETRY_EXPORT_INTERVAL_SEC` | Telemetry export cadence |
| runtime_state_path | runtime_state_snapshot.json | `RUNTIME_STATE_PATH` | Runtime state snapshot file |
| runtime_state_interval_sec | 10.0 | `RUNTIME_STATE_INTERVAL_SEC` | Snapshot write cadence |
| duration_sec | 3600.0 | `DURATION_SEC` | Session run duration |
| request_timeout_sec | 10.0 | `REQUEST_TIMEOUT_SEC` | HTTP request timeout |
| trade_journal_dir | logs | `TRADE_JOURNAL_DIR` | Trade journal output directory |
| prometheus_export_path | "" | `PROMETHEUS_EXPORT_PATH` | Prometheus metrics file (empty=disabled) |
| prometheus_export_interval_sec | 10.0 | `PROMETHEUS_EXPORT_INTERVAL_SEC` | Prometheus export cadence |
| adaptive_capital_engine_enabled | True | `ADAPTIVE_CAPITAL_ENGINE_ENABLED` | Enables adaptive capital feedback engine |
| ofc_enabled | True | `OFC_ENABLED` | Enables Objective Feedback Controller |
| ofc_heartbeat_sec | 900.0 | `OFC_HEARTBEAT_SEC` | OFC heartbeat interval |
| adaptive_risk_fraction_min | 0.05 | `ADAPTIVE_RISK_FRACTION_MIN` | Adaptive risk floor |
| adaptive_risk_fraction_max | 0.35 | `ADAPTIVE_RISK_FRACTION_MAX` | Adaptive risk ceiling |

## Feature flags — confirmed gate targets

| Flag | Default | Confirmed to gate |
|---|---|---|
| `adaptive_capital_engine_enabled` | True | `bootstrap.py:799` — instantiates `NativeAdaptiveCapitalEngine` or `None` |
| `ofc_enabled` | True | `bootstrap.py:805` — wires Objective Feedback Controller |
| `daily_compounding_enabled` | True | `bootstrap.py:828` → `capital_allocator.py`'s internal compounding toggle |
| `polling_enabled` | True | `bootstrap.py:496,578,585,637` — selects gated-polling path vs legacy fallback |
| `symbol_discovery_enabled` | True | `bootstrap.py:542` — triggers wallet auto-scan |
| `polling_enable_active_trades_gate` | True | `bootstrap.py:501,513` — skips polling when no open trades |
| `synthetic_live_signals_enabled` | False | `bootstrap.py:650` — enables synthetic signals even outside paper mode |
| `paper_mode` | False | `bootstrap.py:293-294,650` — sentinel API keys + simulated trading/signals |

## Other config files

| File | Purpose | Wired? |
|---|---|---|
| `config/EV_ALIGNMENT_CONFIG.py` | `CanonicalEVConfig` — intended single source of truth for round-trip cost (fee+slippage+buffer) so universe scanner and execution manager agree on EV thresholds | **Not imported by any active file** except a layer-boundary registration in `scripts/check_layer_imports.py`. Aspirational/orphaned; its `EXIT_SLIPPAGE_BPS=10.0` duplicates the bootstrap value without being wired to it |
| `config/STRATEGY_OPTIMIZATION_v2.py` | Standalone dated (2026-05-01) "post-reset" patch — hardcoded overrides for min trade size, entry filter thresholds, win-rate gate; reads `TRADING_ENABLED` | **Not imported anywhere** — dead/orphaned patch file |
| `config/sandbox.yaml` | Phase-4 48-hour sandbox monitoring fixture (sample portfolio, env=sandbox) | Referenced only by `monitoring/sandbox_monitor.py` (hardcoded fallback) and an archived script — not part of the live trading path |
| `config/tuned_params.json` | ML forecaster hyperparameters (window_size, n_layers, n_neurons) | **Live/wired** — used by `utils/tuned_params.py::get_tuned_params` and `agents/ml_forecaster.py` |

## Stray env vars read outside `bootstrap.py` (active code only)

- **`core_engine/native/config_loader.py`** (the second config system — largest cluster): `MIN_NOTIONAL_USDT`, `SYMBOLS_LIMIT`, `MAX_POSITION_PCT`, `MIN_RESERVE_USDT`, `CAPITAL_RESERVE_PCT`, `COMPOUNDING_ENABLED`, `TRADING_MODE`, `DURATION_SEC`, `INTERVAL_SEC`, `LIQUIDATION_ENABLED`, `REENTRY_OVERRIDE`, `MAX_DRAWDOWN_PCT`, `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `VOLATILITY_REGIME`, `EXIT_FEE_BPS`, `EXIT_SLIP_BPS`, `TP_BUF_BPS`, `MIN_NET`, `TP_MIN`, `TP_MAX`, `RETRY_MAX_ATTEMPTS`, `RETRY_BASE_DELAY_SEC`. Used by `config/__init__.py`, `core_engine/native/__init__.py`.
- `tp_sl_engine.py`: `CR_PRICE_SLIPPAGE_BPS`, `TPSL_FORCE_EXIT_H`
- `executor.py`: `MAKER_ENTRY_ENABLED`, `MAKER_GRACE_S`, `MAKER_OFFSET_BPS`, `PRICE_MAX_AGE_S`, `PRICE_MAX_DEVIATION`
- `arbitration_engine.py`: `DOWNTREND_MARGIN`, `DOWNTREND_MA_BARS`, `DOWNTREND_SLOPE_LAG`, `NO_AVGDOWN_REGIMES`, `SYMBOL_DOWNTREND_TF`, `SYMBOL_DOWNTREND_VETO_ENABLED`, `REBUY_BLOCK_NOTIONAL` — **note: this file's flags are unreachable in practice since the engine itself is never invoked (see feature_ignition_matrix.md)**
- `regime_gate.py`: `ORDERBOOK_IMBALANCE_BUY_MIN`, `ORDERBOOK_IMBALANCE_VETO_ENABLED`, `ORDERBOOK_MAX_AGE_S`, `ORDERBOOK_SPREAD_MAX_PCT`
- `symbol_rotator.py`: `SYMBOL_ROTATION_*`
- `symbol_performance_tracker.py`: `SYMPERF_LOSS_STREAK_BLOCK`
- `model_manager.py`: `MODEL_AUTO_CLEANUP_INCOMPATIBLE`
- `objective_feedback_controller.py`: `OBJ_ARTEFACT_PATH`
- `market_data_websocket.py`: `WS_ENABLE_BOOKTICKER`
- `core_engine/implementations.py`: `EXIT_SLIPPAGE_BPS`
- `automation/rule_overrides.py`: `APPLY_PROPOSED_RULES`, `MIN_REQUIRED_CONF_FLOOR`
- `agents/ml_forecaster.py`: large `ML_*` training block (legacy)
- `src/l5_strategy/model_trainer.py`: `ML_TRAIN_*`, `ML_TRIPLE_BARRIER_*` block (legacy)
- `config/STRATEGY_OPTIMIZATION_v2.py`: `TRADING_ENABLED` (orphaned, see above)
- Standalone tools, each with their own env sets (not part of supervised runtime): `carry_paper_trader.py`/`funding_carry_backtest.py` (`CARRY_*`, `FUNDING_*`), `statarb_discover.py` (`SA_*`), `retrain_weekly.py` (`RETRAIN_*`, `SYMBOLS`, `BINANCE_API_KEY/SECRET`), `trade_monitor.py`, `testnet_validate*.py`, `backtest_edge.py`, `scripts/native_smoke.py`

## Follow-ups for Phase 2/3

1. Confirm at runtime whether `config_loader.py`'s `ConfigGroup` values actually diverge
   from `BootstrapConfig` in the current `.env` (i.e., is this collision live or latent).
2. Confirm current `.env` state for `PROMETHEUS_EXPORT_PATH` / `TELEMETRY_EXPORT_PATH` —
   these gate whether two observability features are idle-by-config or idle-by-omission.
3. Decide whether `EV_ALIGNMENT_CONFIG.py` and `STRATEGY_OPTIMIZATION_v2.py` should be
   wired in, updated, or removed — both are currently dead weight.
