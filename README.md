# Octivault Trader AI Wealth Engine

This project implements a self-compounding, multi-agent AI trading bot that generates profits autonomously.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system architecture, component relationships, and design decisions
- **Update Requirement**: When making architectural changes, please update ARCHITECTURE.md to maintain current documentation
- **Maintenance Script**: Run `python scripts/check_architecture_updates.py` to check if documentation needs updating

## Structure
- `core/` – system brains (allocation, control, execution)
- `agents/` – individual trading agents
- `models/` – ML/RL model storage
- `logs/` – performance and trade logs

## TPSL Volatility-Adaptive Profiles
Set `TPSL_PROFILE` in your environment to pick a default TP/SL behavior:
- `scalp` for tighter exits and faster turnover
- `balanced` for general purpose (default)
- `swing` for wider targets and trend-following holds

Example:
```bash
TPSL_PROFILE=balanced
```

Any explicit TPSL env var (for example `TARGET_RR_RATIO`, `TP_ATR_MULT`, `TRAILING_ATR_MULT`) overrides the profile value.

<!-- GIT-SYNC-TEST -->
Last sync test from VS Code: 2026-02-05 15:1000
Last sync test from VS Code: 2026-02-08 15:100
Last sync test from VS Code: 2026-02-09 15:100////


