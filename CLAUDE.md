# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation trust levels (read this first)

This repo has accumulated a large amount of stale, aspirational documentation from earlier phases of the project — dozens of root-level `PHASE_*.md`, `DEPLOYMENT_STATUS_*.md`, `ARCHITECTURE_*.md`, `SESSION_SUMMARY_*.md` files, plus most of `docs/` (`docs/README.md`, `docs/architecture/`, `docs/operations/`, `docs/strategy/`). Several describe directories, files, or wiring states that no longer exist (e.g. `README_PHASE_3.md` links to `PHASE_3_QUICK_START.md` and four other files that were never created, and describes 16 placeholder engine methods that are long since implemented for real). **Do not trust a claim in these docs without checking it against the actual code.**

The one doc set that stays current and is actively cited *from inside the code itself* is `docs/audit/` (`remediation_plan.md`, `native_scope.md`, `baseline_test_report.md`) — `config_loader.py` and `tests/conftest.py` both reference it directly. Trust that one first.

## Commands

Setup:
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # runtime deps
pip install -r requirements-dev.txt      # ruff, mypy — not installed in venv by default
```

Run the bot:
```
python3 main.py --mode {dry-run|paper-trade|live} [--duration N] [--cycles N] [--interval SECS] [--capital N] [--no-native] [--no-compat]
```
`dry-run` runs the full cycle but skips `execute_decision()`. A stale PID lock at `/tmp/octivault_trader.pid` will SIGTERM/SIGKILL a previous instance on startup.

Tests:
```
python3 -m pytest tests/                          # full suite (pytest.ini: testpaths = tests)
python3 -m pytest tests/test_foo.py::test_name    # single test
python3 -m pytest -m unit                         # by marker: unit | integration | slow | exchange | market_data | database
```
`run-local.sh` is stale — it shells out to test files and a `core/` directory that no longer exist. Use plain `pytest` invocations instead of that script.

Lint/type-check: enforced via pre-commit, not routine CLI use — `ruff`/`mypy` aren't in `venv` unless you installed `requirements-dev.txt`.
```
pre-commit run --all-files                        # ruff check + ruff format, on commit-scoped paths
pre-commit run --hook-stage manual mypy            # mypy only runs at this manual stage, never on a normal commit
```
Ruff/ruff-format are scoped to `^(core_engine/|main\.py$|tests/|scripts/|deployment/.*\.py$)` only — `src/`, `utils/`, `agents/`, and root-level scripts are excluded (marked "slated for cleanup"). Note `mypy.ini` (`python_version = 3.9`) and `pyproject.toml`'s `[tool.mypy]` block (`python_version = "3.10"`) disagree — `mypy.ini` is the one mypy actually consults; this inconsistency is unresolved, not a typo to silently fix.

## Architecture

**Façade contract.** `main.py` is the single entry point and by its own docstring may only import from `core_engine/` and stdlib — never reach into lower-level components directly. This is a real, enforced convention, verified true in the current code.

**Startup sequence** (`main.py`): `load_dotenv()` → `parse_args()` → acquire PID lock → `setup_core_engines()` builds `app_ctx` → `Engines(app_ctx)` constructs 5 façade engines → `engines.initialize()` calls `OperationsEngine.startup_system()`, which **aborts the process if it returns False** (deliberate: prevents trading on a falsely-hydrated NAV=0 after a failed native-orchestrator startup).

**Main loop** (`run()` → `trading_cycle()`): runs every `--interval` seconds as a 5-phase cycle — DISCOVER → READ → UNDERSTAND → DECIDE → EXECUTE → RECOVER — with expensive calls (account state, portfolio snapshot, health) throttled independently via a `CadenceScheduler`. Every 50 cycles it runs `system_invariants.check_live()` and can auto-heal orphaned holdings via a forced balance reconcile. Shutdown calls `os._exit(rc)` rather than a normal exit — deliberate, because a non-daemon Keras training thread otherwise hangs shutdown for minutes.

**Live code path**: `core_engine/` is the entire running system. The 5 façade classes (`market_account_engine.py`, `situation_engine.py`, `decision_engine.py`, `safe_execution_engine.py`, `operations_engine.py`) delegate to `core_engine/implementations.py` (real logic, not stubs), which delegates into `core_engine/native/` (~68 files: exchange client, market data, orchestrator, arbitration/decision/capital-allocation engines, TP/SL, NAV protection, carry subsystem), wired together by `core_engine/native/bootstrap.py`. `utils/` (logging setup, tuned params, indicators) and `agents/` (`ml_forecaster.py`, `symbol_screener.py` only — imported directly by `bootstrap.py`) are live.

**Dead/abandoned — not on any live import path.** Don't build on these without checking imports first: `src/` (referenced only in comments and one dead import inside an equally-dead `ai_office/`), `ai_office/`, `automation/`, `deployment/` (just an empty `__init__.py`), `monitoring/`, `stream/`, `tools/`, `dashboard/` (no `.py` files at all), `portfolio/`.

**Config**: the real config path is `core_engine/native/bootstrap.py`'s `BootstrapConfig.from_env()`, reading `.env` directly (`BINANCE_API_KEY/SECRET`, `BINANCE_TESTNET`, `PAPER_MODE`, `ADAPTIVE_*`, `CAPITAL_ALLOCATION_PCT`, `BINANCE_FUTURES_TESTNET_KEY/SECRET`, etc.). `core_engine/native/config_loader.py`'s `ConfigLoader` is dead — its own docstring says it has zero production callers and that some of its env var names collide with different-meaning `BootstrapConfig` keys with no live effect. The `config/` directory is a mix of an old test fixture (`sandbox.yaml`) and one file actually read by `utils/tuned_params.py` (`tuned_params.json`) — check per-file, don't assume the directory is load-bearing as a whole. Trading mode is controlled by two independent switches: `BINANCE_TESTNET`/`PAPER_MODE` env vars, and `main.py --mode` — the latter gates whether `execute_decision()` is actually called.

**Testing fixtures**: `tests/conftest.py` provides mock fixtures (`mock_exchange_client`, `mock_market_data`, `mock_database`, `mock_cache`, `mock_websocket`), `app_context`/`sync_app_context` (falls back to a `MockAppContext` since the legacy `src.l8_lifecycle.app_context.AppContext` it originally wrapped no longer exists), and `shared_state`/`position_manager`/`portfolio_manager`/`risk_manager`/`temp_config`. It also hard-excludes about a dozen legacy-namespace test files via `collect_ignore` because they fail at import — see `docs/audit/baseline_test_report.md` for why before trying to re-enable one.

**Standalone strategy/backtest scripts**: the root-level `*_backtest.py` / `*_discover.py` files (`backtest_edge.py`, `funding_carry_backtest.py`, `orderbook_imbalance_backtest.py`, `cross_exchange_edge_discover.py`, `statarb_discover.py`, `news_sentiment_backtest.py`, etc.) are independent one-off scripts, not a shared framework — unified only by convention (`load_dotenv()` + direct imports from `core_engine.native.*`, intentionally bypassing the façade contract since these are offline analysis tools, not the live app). The paper-trader daemons (`carry_paper_trader.py`, `delisting_exit_paper_trader.py`, `negative_carry_paper_trader.py`) are a different, heavier shape: long-running processes with their own JSON state + JSONL ledger under `logs/`, a `report` CLI subcommand, and env-var-driven parameters documented in their own docstrings. `supervisor.sh` (wraps `main.py`) has real crash-loop backoff, log rotation, and a stall watchdog keyed on cycle-count progress in the log, not just process liveness; the strategy-specific `*_supervisor.sh` scripts are simpler restart-loop wrappers around their respective paper traders.

**Current live state**: `runtime_state_snapshot.json` reflects a near-zero-capital, dormant account (`nav_usdt` ~57.87, ~20 positions flagged `MICRO_DUST`/`IGNORE`, `connectivity_halted: true`). Don't assume the bot is actively trading live capital — check this file's current contents before making that assumption.
