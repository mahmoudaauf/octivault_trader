# OctiVault — Quick Reference

> Phase 8.2.8 closeout: the legacy `🎯_MASTER_SYSTEM_ORCHESTRATOR.py`
> and all its shell wrappers were retired. Everything below is the
> single supported path now.

## Run the bot

```sh
# Default: native L0-L8 + compat stubs, paper trade, indefinite
python main.py --mode=paper-trade

# Bounded run
python main.py --mode=paper-trade --duration=30min --cycles=0

# Live (needs real BINANCE_API_KEY / BINANCE_API_SECRET in env)
python main.py --mode=live --duration=2h

# Mock mode (zero network, structural smoke)
python main.py --mode=dry-run --cycles=5 --no-native
```

## Smoke the native stack directly

```sh
# Offline (stub exchange client, no network)
python scripts/native_smoke.py --offline --duration 5

# Live testnet
BINANCE_API_KEY=… BINANCE_API_SECRET=… BINANCE_TESTNET=true \
    python scripts/native_smoke.py --live --duration 60
```

## Tests

```sh
# Native L0-L8 suite (~2.5s, 204 tests)
pytest tests/test_native_*.py tests/test_integration_native_wiring.py -q

# Full suite
pytest -q
```

## CLI flags reference

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `paper-trade` | one of `dry-run`, `paper-trade`, `live` |
| `--duration` | unlimited | wall-clock budget, e.g. `30min`, `2h` |
| `--cycles` | `0` (unlimited) | cycle count budget |
| `--interval` | `1.0` | seconds between cycles |
| `--capital` | `1000.0` | initial USDT |
| `--no-native` | off | skip native bootstrap → empty mock app_ctx |
| `--no-compat` | off | when native is on, skip the 6 compat null-stubs |

## Process hygiene

```sh
# Find a running bot
pgrep -f "main.py.*paper-trade"

# Stop one
pkill -f "main.py.*paper-trade"

# Force-stop everything bot-related
pkill -9 -f "main.py" || true
```

## Reference docs

- `PHASE_8_2_NATIVE_MIGRATION_ROADMAP.md` — the L0-L8 migration plan
- `PHASE_8_2_8_PREP.md` — final-step (bridge deletion) doc
- `PHASE_8_2_8_TRIAGE.md` — per-key triage of unmigrated façade keys
- `core_engine/native/` — the native L0-L8 implementation
- `_archive/2026-05-06_legacy_launchers/` — retired shell wrappers
