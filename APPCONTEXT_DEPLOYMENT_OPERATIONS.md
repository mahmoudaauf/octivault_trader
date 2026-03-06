# AppContext Deployment & Operations Guide

## Quick Start Checklist

### Pre-Deployment

- [ ] Python 3.9+ available
- [ ] All required dependencies installed (see imports in appcontext.py)
- [ ] BINANCE_API_KEY set in environment
- [ ] BINANCE_API_SECRET set in environment
- [ ] config.SYMBOLS or config.SEED_SYMBOLS defined (initial universe)
- [ ] LOG_LEVEL set (default: INFO)

### Configuration Setup

```python
# config.py (example)
class AppConfig:
    # Timeframes
    VOLATILITY_REGIME_TIMEFRAME = "1h"  # Brain: slow regime detection
    ohlcv_timeframes = ["5m", "1h"]     # Hands: execution, Brain: regime
    
    # Startup Policy
    LIVE_MODE = False                   # If True: always pure reconciliation
    BOOTSTRAP_SEED_ENABLED = True       # Allow bootstrap seed (cold-start only)
    COLD_BOOTSTRAP_ENABLED = True       # Enable cold bootstrap mode
    
    # Phases
    START_TIMEOUT_SEC = 30.0            # Component start timeout
    P4_MARKET_DATA_START_TIMEOUT_SEC = 90.0  # MDF warmup timeout
    P4_MARKET_DATA_READY_TIMEOUT_SEC = 180.0 # MDF readiness gate timeout
    WAIT_READY_SECS = 0                 # P9 blocking timeout (0 = non-blocking)
    GATE_READY_ON = ""                  # P9 gate list (comma-separated)
    
    # Universe
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Initial seed symbols
    
    # Liquidity
    LIQUIDITY_ORCHESTRATION_MODE = "agent"  # One of: cash_router, orchestrator, agent, event_bus
    LIQ_ORCH_ENABLE = True
    AFFORD_SCOUT_ENABLE = True
    AFFORD_SCOUT_INTERVAL_SEC = 15
    UURE_ENABLE = True
    UURE_INTERVAL_SEC = 300            # 5 minutes
    
    # Safety
    STRICT_SHARED_STATE_IDENTITY = False  # Raise on SharedState mismatch (strict mode)
    
    # Logging
    LOG_FILE = "/tmp/octivault_trader.log"  # Optional file logging
    QUIET_TF_LOGS = True                # Suppress TensorFlow verbose logs
```

### Environment Variables

```bash
# REQUIRED
export BINANCE_API_KEY="your_key_here"
export BINANCE_API_SECRET="your_secret_here"

# OPTIONAL
export APP_LOG_FILE="/var/log/octivault_trader.log"
export TF_CPP_MIN_LOG_LEVEL="2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
export OMP_NUM_THREADS="1"       # BLAS threading
export MKL_NUM_THREADS="1"       # Intel MKL threading
```

---

## Deployment Process

### 1. Create AppContext Instance

```python
import asyncio
import logging
from core.app_context import AppContext
from config import AppConfig

# Create logger
logger = logging.getLogger("Octivault")
logger.setLevel(logging.INFO)

# Create config and AppContext
config = AppConfig()
app = AppContext(config=config, logger=logger)

# Run initialization
asyncio.run(app.initialize_all(up_to_phase=9))
```

### 2. Monitor Startup

**Watch these logs:**
```
[AppContext] SUMMARY INIT_START
[AppContext] SUMMARY PHASE_START phase=P3_exchange_gate
[AppContext] SUMMARY PHASE_START phase=P4_market_data
... (more phases)
[AppContext] SUMMARY PHASE_DONE phase=P9_finalization
[AppContext] SUMMARY INIT_COMPLETE ready=True
```

**If phase fails:**
```
[AppContext] SUMMARY PHASE_TIMEOUT phase=P4_market_data timeout_sec=90.0
[AppContext] SUMMARY PHASE_ERROR phase=P5_execution error="..."
→ Check logs for specific component errors
→ Verify prerequisites (balances, universe, exchange connectivity)
```

### 3. Check Readiness

```python
# After initialize_all() completes
snapshot = await app._ops_plane_snapshot()
print(snapshot['ready'])      # True if all gates clear
print(snapshot['issues'])     # List of blockers
print(snapshot['detail'])     # Detailed metrics
```

**Example output:**
```python
{
    'ready': True,
    'issues': [],
    'detail': {
        'FiltersCoveragePct': 100.0,
        'FreeQuoteFloor': 27.50,
        'PlannedQuoteUsed': 25.0,
        'MinNotionalFloor': 25.0,
        'AffordabilityProbe': {
            'symbol': 'BTCUSDT',
            'ok': True,
            'amount': 0.0,
            'code': 'OK',
            'planned_quote': 25.0
        },
        'Liquidity': {},      # No gaps
        'Dust': {
            'registry_size': 0,
            'origin_breakdown': {},
            'strategy_pct': 0.0,
            'external_pct': 0.0,
            'has_external_trash': False
        }
    }
}
```

---

## Operating the System

### Health Monitoring

**1. Subscribe to Summary Events**
```python
# AppContext emits events to SharedState
# You can subscribe to observe startup progress
async def on_summary(event_dict):
    print(f"[{event_dict.get('event')}] {event_dict}")

# Wire subscription (implementation varies by SharedState)
await app.shared_state.subscribe("events.summary", on_summary)
```

**2. Health Status Events**
```
event="INIT_COMPLETE" ready=True        → Startup success
event="READINESS_TICK" ready=True       → Periodic health check (30s)
event="READINESS_TICK" issues=[...]     → Capital or infrastructure issue
```

**3. Logs to Watch**
```
[AppContext] SUMMARY STARTUP_POLICY mode=RECONCILIATION_ONLY is_restart=True
→ Restart detected; no forced entries

[AppContext] SUMMARY STARTUP_POLICY mode=COLD_START is_restart=False
→ Cold-start; bootstrap seed allowed

[AppContext] SUMMARY LIQUIDITY_NEEDED symbol=ETHUSDT gap_usdt=150.00
→ Liquidity shortfall; orchestrator may trigger

[AffordScout] started (interval≈15s, jitter≈10%, only_usdt=True)
→ Background affordability scout running

[UURE] background loop started (immediate + periodic every 300s)
→ Universe rotation engine running
```

### Background Task Management

**1. Affordability Scout**
- Runs every 15s (configurable: AFFORD_SCOUT_INTERVAL_SEC)
- Round-robin over symbols, calls ExecutionManager.can_afford_market_buy()
- Emits LIQUIDITY_NEEDED if gap detected
- Can be disabled: AFFORD_SCOUT_ENABLE=False

**2. Universe Rotation Engine (UURE)**
- Runs immediately once at P9 startup (critical for universe population)
- Then periodic every 5 minutes (configurable: UURE_INTERVAL_SEC)
- Calls universe_rotation_engine.compute_and_apply_universe()
- Emits UNIVERSE_ROTATION summary
- Can be disabled: UURE_ENABLE=False

**3. Periodic Readiness Logger**
- Runs every 30 seconds
- Emits READINESS_TICK summary with health status
- Shows current issues, liquidity gaps, dust metrics

**4. Adaptive Capital Engine Monitor**
- Runs every 5 minutes (after P5)
- Evaluates adaptive sizing decisions
- Logs: risk_fraction, min_trade_quote, win_rate, avg_r_multiple
- Can be disabled: Ensure AdaptiveCapitalEngine module unavailable

---

## Troubleshooting

### Phase P3 Issues

**"ExchangeClientNotReady"**
- Check: BINANCE_API_KEY and BINANCE_API_SECRET in environment
- Check: Exchange client module imports properly
- Solution: Restart with valid API credentials

**"SymbolsUniverseEmpty"**
- Check: config.SYMBOLS or config.SEED_SYMBOLS defined
- Check: WalletScanner ran (P3.7) but found no balances
- Solution: Set SYMBOLS in config or ensure account has balances

**"NAVNotReady" / "BalancesNotReady"**
- Check: API key has read permissions (spot balances)
- Check: Exchange connectivity (ping binance.com)
- Solution: Verify API key permissions; check network

### Phase P4 Issues

**"MarketDataNotReady"**
- Check: MarketDataFeed start() completed within 90s
- Check: WebSocket connections stable
- Solution: Increase P4_MARKET_DATA_START_TIMEOUT_SEC or check network

**P4 Gate Timeout (blocks P5)**
- Logs: "[AppContext] P4 gate failed — aborting P5+ startup"
- Solution: Check MarketDataFeed logs; may need longer timeout or network fix

### Phase P5 Issues

**"Execution Manager construction failed"**
- Logs show missing dependencies
- Check: ExecutionManager module imports properly
- Check: SharedState and ExchangeClient both available
- Solution: Fix import paths; restart

### Phase P6 Issues

**"MetaController construction failed"**
- Check: MetaController module imports properly
- Check: SharedState, ExchangeClient, ExecutionManager all ready
- Solution: Fix import paths; restart

**"MetaController failed to enter running state"**
- Check: MetaController._running flag after start()
- Logs: "MetaController running state declared INVALID"
- Solution: Check MetaController.start() implementation; may need debugging

**"Authoritative wallet sync failed"**
- Check: SharedState.authoritative_wallet_sync() or hard_reset_capital_state()
- Logs: "[BOOT] Authoritative wallet sync complete"
- Solution: Verify SharedState has these methods; fallback to hard_reset

### Capital / Liquidity Issues

**"FreeQuoteBelowFloor"**
- Account has insufficient free USDT for minimum order
- Check: Account balance on exchange
- Solution: Deposit USDT or wait for trades to close

**"LIQUIDITY_NEEDED" events flooding**
- Affordability scout detecting persistent gaps
- Check: Throttle AFFORD_SCOUT_INTERVAL_SEC
- Solution: Monitor scanner; if gaps persist, deposit capital

**"LIQUIDITY_ORCH_MISSING"**
- Liquidity orchestrator not available but gap persists
- Check: LiquidationOrchestrator module available
- Check: LIQUIDITY_ORCHESTRATION_MODE configured
- Solution: Ensure LIQUIDITY_ORCHESTRATION_MODE = "agent" or load orchestrator module

---

## Scaling to Production

### 1. Enable Strict Mode

```python
config.STRICT_SHARED_STATE_IDENTITY = True  # Fail fast on identity mismatch
```

### 2. Enable Live Mode (Production)

```python
config.LIVE_MODE = True  # Always pure reconciliation (no bootstrap seed)
```

### 3. Set Readiness Gating

```python
config.WAIT_READY_SECS = 120  # Wait up to 2 minutes for all gates
config.GATE_READY_ON = "market_data,execution,capital,exchange,startup_sanity"
```

### 4. Enable Persistent Logging

```python
config.LOG_FILE = "/var/log/octivault_trader.log"
# Or via environment: export APP_LOG_FILE="/var/log/octivault_trader.log"
```

### 5. Increase Component Timeouts (Optional)

```python
config.P4_MARKET_DATA_START_TIMEOUT_SEC = 120  # 2 minutes for MDF warmup
config.P4_MARKET_DATA_READY_TIMEOUT_SEC = 240  # 4 minutes for market data ready gate
```

### 6. Configure Liquidity Mode

```python
config.LIQUIDITY_ORCHESTRATION_MODE = "orchestrator"  # or "agent", "cash_router"
config.LIQ_ORCH_ENABLE = True
```

### 7. Monitor Continuously

```python
# Spawn periodic monitoring (outside AppContext)
async def monitor_health():
    while True:
        snap = await app._ops_plane_snapshot()
        if not snap['ready']:
            logger.warning("Health degraded: %s", snap['issues'])
        await asyncio.sleep(30)

asyncio.create_task(monitor_health())
```

---

## Graceful Shutdown

```python
# Stop the system cleanly
await app.shutdown(save_snapshot=True)
# → Cancels background tasks
# → Stops all components in reverse order
# → Saves state snapshot (if available)
# → Emits SHUTDOWN_DONE summary
```

**On Restart:**
- AppContext detects prior state via _is_restart flag
- Startup policy switches to RECONCILIATION_ONLY
- No forced entries; existing positions observed and managed
- System resumes trading per current strategy

---

## Performance Tuning

### 1. Reduce Polling Intervals (More Responsive)

```python
config.AFFORD_SCOUT_INTERVAL_SEC = 10      # More frequent (was 15)
config.UURE_INTERVAL_SEC = 180             # More frequent (was 300)
```

### 2. Increase Polling Intervals (Less CPU)

```python
config.AFFORD_SCOUT_INTERVAL_SEC = 30      # Less frequent (was 15)
config.UURE_INTERVAL_SEC = 600             # Less frequent (was 300)
```

### 3. Disable Optional Components

```python
config.DASHBOARD_ENABLED = False           # Skip web UI
config.CAPITAL_ALLOCATOR = {"ENABLED": False}  # Skip P9 allocator
```

### 4. Disable Optional Loops

```python
config.AFFORD_SCOUT_ENABLE = False  # Skip affordability scout
config.UURE_ENABLE = False          # Skip universe rotation
```

---

## Monitoring Checklist

Daily/Weekly:
- [ ] Check app startup logs for INIT_COMPLETE and ready=True
- [ ] Verify READINESS_TICK events (every 30s)
- [ ] Monitor LIQUIDITY_NEEDED frequency (should be rare)
- [ ] Check dust_registry_size (should be stable or declining)
- [ ] Review P&L and win rate (via PerformanceEvaluator)

On Alerts:
- [ ] READINESS_TICK with issues → Check _ops_plane_snapshot() for details
- [ ] LIQUIDITY_NEEDED → Verify account balance and min notional floors
- [ ] PHASE_ERROR during startup → Check logs for component-specific errors
- [ ] SHARED_STATE_IDENTITY mismatch → Verify all components using canonical instance

---

## Common Deployment Patterns

### Development (Testing)

```python
# Local, non-blocking startup
config.LIVE_MODE = False
config.BOOTSTRAP_SEED_ENABLED = True
config.WAIT_READY_SECS = 0  # Non-blocking
await app.initialize_all(up_to_phase=9)
```

### Staging (Pre-Production)

```python
# Blocking startup with gating
config.LIVE_MODE = False
config.WAIT_READY_SECS = 60  # 1-minute gate wait
config.GATE_READY_ON = "market_data,execution,capital"
config.STRICT_SHARED_STATE_IDENTITY = False  # Warn, don't fail
await app.initialize_all(up_to_phase=9)
```

### Production (Live Trading)

```python
# Pure reconciliation, strict mode, long timeouts
config.LIVE_MODE = True
config.WAIT_READY_SECS = 120  # 2-minute gate wait
config.GATE_READY_ON = "market_data,execution,capital,exchange,startup_sanity"
config.STRICT_SHARED_STATE_IDENTITY = True  # Fail fast
config.P4_MARKET_DATA_START_TIMEOUT_SEC = 120  # 2-minute MDF warmup
await app.initialize_all(up_to_phase=9)
```

---

## Post-Deployment Validation

1. **Check Logs**
   ```bash
   tail -100 /var/log/octivault_trader.log | grep "INIT_COMPLETE\|READINESS_TICK"
   ```

2. **Verify Components**
   ```python
   print(f"MetaController: {bool(app.meta_controller)}")
   print(f"ExecutionManager: {bool(app.execution_manager)}")
   print(f"SharedState: {bool(app.shared_state)}")
   ```

3. **Test Execution Path**
   ```python
   snapshot = await app._ops_plane_snapshot()
   print(f"Ready: {snapshot['ready']}, Issues: {snapshot['issues']}")
   ```

4. **Check Background Tasks**
   ```python
   print(f"Scout task: {app._scout_task.done() if app._scout_task else 'None'}")
   print(f"UURE task: {app._uure_task.done() if app._uure_task else 'None'}")
   ```

---

**Deployment Guide Complete** ✅
