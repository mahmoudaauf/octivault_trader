# Legacy System Archive — 2026-06-19

Archived by: GitHub Copilot  
Date: June 19, 2026  
Branch at archive: phase-3/wiring

## What Was Archived

### `agents/` — Legacy Signal Agents (NOT wired into live pipeline)
| File | Reason Archived |
|------|----------------|
| `swing_trade_hunter.py` | Not called by native orchestrator; EMA/MACD logic duplicated by MLForecaster |
| `dip_sniper.py` | Not called by native orchestrator; Bollinger/ATR logic not in live path |
| `trend_hunter.py` | Not called; `generate_signals()` returned empty list |
| `ipo_chaser.py` | Not called by native orchestrator |
| `liquidation_agent.py` | Not called in live path |
| `wallet_scanner_agent.py` | Not called in live path |
| `edge_calculator.py` | Only used by swing_trade_hunter (also archived) |

### `src/` (L0–L8 layer) — Legacy Architecture
The full `src/` L0–L8 layer was the original modular architecture (Phase 1–7).  
It has been replaced entirely by `core_engine/native/` (Phase 8+).  
All imports from `integration.py` were already commented out confirming it was dead.

## What Remains Active (DO NOT ARCHIVE)

```
agents/
  ml_forecaster.py      ← ACTIVE: Only live signal source (Keras ML model)
  symbol_screener.py    ← ACTIVE: Symbol discovery/filtering

src/                    ← MINIMAL STUBS only (support ml_forecaster imports)
  _lazy.py, _layer_index.py
  l0_core/component_status_logger.py, stubs.py
  l5_strategy/agent_optimizer.py, model_trainer.py, model_manager.py
  l3_portfolio/bootstrap_symbols.py

core_engine/native/     ← ACTIVE: Full live trading engine
  orchestrator.py, decisions.py, regime_gate.py, signals.py ...
```

## Live Pipeline (post-archive)
```
MLForecaster → SignalManagerBridge → NativeArbitrationEngine (7 gates) → Execute
```
