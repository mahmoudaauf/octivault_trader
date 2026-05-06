# Legacy launchers (archived 2026-05-06)

These shell scripts launched, monitored, killed, or restarted the legacy
`🎯_MASTER_SYSTEM_ORCHESTRATOR.py` (deleted in commit `55213e6`,
Phase 8.2.8 step 7). They are kept here for historical reference and
forensics — **not** for execution.

## Contents

| File | Purpose |
|---|---|
| `launch_run6.sh` … `launch_run11.sh` | Session-specific nohup launchers for individual paper-trade runs |
| `START_TRADING.sh` | Generic foreground launcher with tee pipeline |
| `launch_growth_mode.sh` | nohup wrapper for "growth mode" sessions |
| `run_bot_resilient.sh` | Resilient (auto-restart) wrapper |
| `emergency_liquidate.sh` | Kill orchestrator, run emergency liquidation script, restart |
| `restart_with_optimization.sh` | PID-aware restart wrapper |
| `start_trading_with_monitoring.sh` | Co-launch orchestrator + monitor scripts |
| `run_orchestrator_for_4h.sh` | Bounded 4-hour run with logfile rotation |
| `QUICK_REFERENCE.sh` | Operator runbook (pgrep/pkill recipes) |

## Modern equivalent

```sh
# Foreground run (default: native L0-L8 + compat stubs)
python main.py --mode=paper-trade --duration=30min

# Mock mode (no creds / smoke)
python main.py --mode=dry-run --cycles=10 --no-native

# End-to-end native smoke (offline, ~5s)
python scripts/native_smoke.py --offline --duration 5

# End-to-end native smoke (live testnet, needs creds)
BINANCE_API_KEY=… BINANCE_API_SECRET=… BINANCE_TESTNET=true \
    python scripts/native_smoke.py --live --duration 60
```

The native bootstrap (`core_engine/native/bootstrap.py`) replaced the
legacy orchestrator entirely; there is no more PID-juggling, no more
nohup wrappers, no more emoji-prefix import shimming.
