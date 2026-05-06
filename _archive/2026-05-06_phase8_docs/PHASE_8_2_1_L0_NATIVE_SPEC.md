# Phase 8.2.1: L0 Native Implementation Spec

**Layer:** L0 (Core Utilities)  
**Scope:** SharedState, TimeUtils, ConfigConstants, RetryManager  
**Timeline:** 1-2 weeks  
**Target Completion:** 2026-05-20  
**Expected Cycle Time Improvement:** -20ms (300ms → 280ms)

---

## Component Inventory

### Current Legacy Components (in L0)

| Component | File | Lines | Role | Criticality |
|-----------|------|-------|------|------------|
| `SharedState` | `src/l0_utilities/shared_state.py` | ~1200 | Central state hub (positions, balances, nav) | 🔴 Critical |
| `TimeUtils` | `src/l0_utilities/time_utils.py` | ~150 | Unix timestamps, timezone handling | 🟡 High |
| `ConfigConstants` | `src/l0_utilities/config_constants.py` | ~300 | ~65 config groups | 🟡 High |
| `RetryManager` | `src/l0_utilities/retry_manager.py` | ~100 | Exponential backoff for API calls | 🟢 Medium |
| `ChaosMonkey` | `src/l0_utilities/chaos_monkey.py` | ~50 | Testing utility | 🔵 Low |

**Total Legacy L0 Lines:** ~1,800

### Native Target Structure

```
core_engine/
├── native/
│   ├── __init__.py
│   ├── shared_state.py        (NativeSharedState, ~300 lines)
│   ├── time_utils.py          (NativeTimeUtils, ~80 lines)
│   ├── config_loader.py       (ConfigLoader, ~150 lines)
│   └── retry_manager.py       (NativeRetryManager, ~100 lines)
├── integration.py             (add --native-l0 flag dispatch)
└── ...
```

**Total Native L0 Lines:** ~630 (65% reduction!)

---

## 1. NativeSharedState

### Current Legacy Complexity (1,200 lines)

- Position invariant checking (complex business logic)
- Multi-format balance sync (legacy position objects)
- NAV rebuilding with fallback chains
- Symbol convergence tracking
- Dust position detection
- Quota reservations
- Event emission (Position closed, NAV updated, etc)

### Native Simplification (target ~300 lines)

**Core State:**
```python
class NativeSharedState:
    """Minimal in-memory state (replaces 1,200-line legacy SharedState)"""
    
    def __init__(self):
        # Essential data structures
        self.nav_usdt: float = 0.0
        self.free_balance_usdt: float = 0.0
        self.invested_capital_usdt: float = 0.0
        
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.open_orders: dict[str, Order] = {}  # order_id -> Order
        self.price_cache: dict[str, float] = {}  # symbol -> latest_price
        
        self.accepted_symbols: set[str] = set()  # actively trading symbols
        self.dust_symbols: set[str] = set()  # below-threshold symbols
        
        # Hydration state
        self._ready_event = asyncio.Event()
    
    async def wait_ready(self):
        """Wait until positions hydrated"""
        await self._ready_event.wait()
    
    def mark_ready(self):
        """Signal positions are ready"""
        self._ready_event.set()
    
    def update_position(self, symbol: str, qty: float, entry: float, current: float):
        """Update position (simple, no invariant checking)"""
        if qty > 0:
            self.positions[symbol] = Position(
                symbol=symbol, qty=qty, entry_price=entry, mark_price=current
            )
        else:
            self.positions.pop(symbol, None)
    
    def update_nav(self, nav: float):
        """Update NAV (single source of truth)"""
        self.nav_usdt = nav
    
    def get_nav(self) -> float:
        """Get current NAV"""
        return self.nav_usdt
```

**Trade-offs:**
- ✅ No position invariant checking (let execution layer validate)
- ✅ No event system (use callback functions instead)
- ✅ No quota reservations (managed in execution layer)
- ✅ ~75% less code

---

## 2. NativeTimeUtils

### Current Legacy (150 lines)
- Unix timestamp conversions
- Timezone handling (UTC, local)
- Candle time alignment
- Interval parsing

### Native Implementation (target ~80 lines)

```python
import time
from datetime import datetime, timezone
from typing import Optional

class NativeTimeUtils:
    """Lightweight time utilities"""
    
    @staticmethod
    def unix_now_ms() -> int:
        """Current Unix timestamp (milliseconds)"""
        return int(time.time() * 1000)
    
    @staticmethod
    def unix_now_s() -> float:
        """Current Unix timestamp (seconds)"""
        return time.time()
    
    @staticmethod
    def iso_now() -> str:
        """ISO8601 timestamp (UTC)"""
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def align_candle_time(unix_ms: int, interval_sec: int) -> int:
        """Align timestamp to candle boundary"""
        interval_ms = interval_sec * 1000
        return (unix_ms // interval_ms) * interval_ms
    
    @staticmethod
    def seconds_until_next_candle(interval_sec: int) -> float:
        """Seconds until next candle opens"""
        now_ms = NativeTimeUtils.unix_now_ms()
        next_candle = NativeTimeUtils.align_candle_time(now_ms, interval_sec) + (interval_sec * 1000)
        return (next_candle - now_ms) / 1000.0
```

**Usage Example:**
```python
# Legacy
time_utils = TimeUtils()
time_utils.get_current_unix_ms()

# Native
nav = NativeTimeUtils.unix_now_ms()  # Static, no instantiation
```

---

## 3. ConfigLoader

### Current Legacy (300 lines across multiple files)
- 65 config groups loaded from disk
- Environment variable overrides
- Validation checks
- Hardcoded constants

### Native Implementation (target ~150 lines)

```python
from dataclasses import dataclass
from typing import Any
import os
import json

@dataclass
class ConfigGroup:
    """Single config group"""
    name: str
    values: dict[str, Any]

class ConfigLoader:
    """Load config once at startup"""
    
    def __init__(self):
        self._config: dict[str, ConfigGroup] = {}
        self._load()
    
    def _load(self):
        """Load from .env or defaults"""
        # SYMBOLS
        symbols = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT")
        self._config["SYMBOLS"] = ConfigGroup("SYMBOLS", {
            "symbols": symbols.split(","),
            "limit": 10
        })
        
        # CAPITAL
        self._config["CAPITAL"] = ConfigGroup("CAPITAL", {
            "reserve_pct": float(os.getenv("CAPITAL_RESERVE_PCT", "0.10")),
            "min_reserve_usdt": float(os.getenv("MIN_RESERVE_USDT", "10.00"))
        })
        
        # ... etc for other 63 groups
    
    def get(self, group: str, key: str, default: Any = None) -> Any:
        """Get config value"""
        return self._config.get(group, ConfigGroup(group, {})).values.get(key, default)
    
    def get_group(self, group: str) -> dict[str, Any]:
        """Get entire config group"""
        return self._config.get(group, ConfigGroup(group, {})).values
```

---

## 4. NativeRetryManager

### Current Legacy (100 lines)
- Exponential backoff
- Jitter
- Max retry logic

### Native Implementation (target ~100 lines, same complexity)

```python
import asyncio
import random
from typing import Callable, Any

class NativeRetryManager:
    """Simple async retry with exponential backoff"""
    
    def __init__(
        self, 
        max_attempts: int = 3,
        base_delay_sec: float = 0.1,
        max_delay_sec: float = 10.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self.jitter = jitter
    
    async def call(
        self, 
        coro_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with retries"""
        last_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_attempts:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, ...
                    delay = min(
                        self.base_delay_sec * (2 ** (attempt - 1)),
                        self.max_delay_sec
                    )
                    
                    # Add jitter
                    if self.jitter:
                        delay *= (0.5 + random.random())
                    
                    await asyncio.sleep(delay)
        
        raise last_error
```

---

## Integration Checklist

### 1. Create Native L0 Module
```bash
mkdir -p core_engine/native
touch core_engine/native/__init__.py
touch core_engine/native/shared_state.py
touch core_engine/native/time_utils.py
touch core_engine/native/config_loader.py
touch core_engine/native/retry_manager.py
```

### 2. Update integration.py

```python
# In integration.py, add:

async def create_app_context(production: bool = False, native_l0: bool = False):
    """Build app context with L0 choice"""
    
    if not production:
        # Mock mode: always empty dict
        return {}
    
    if native_l0:
        # NEW: Use native L0
        from core_engine.native import NativeSharedState, NativeTimeUtils, ConfigLoader
        
        app_ctx = {
            "shared_state": NativeSharedState(),
            "time_utils": NativeTimeUtils,  # Static class
            "config": ConfigLoader(),
            "retry_manager": RetryManager(),  # TODO: also native
        }
        # ... load legacy L1-L8 components as before
        return app_ctx
    else:
        # Legacy: Use production bridge
        return await build_production_app_ctx()
```

### 3. Update main.py

```python
# Add to argparse:
parser.add_argument(
    "--native-l0",
    action="store_true",
    help="Use native L0 (SharedState, TimeUtils) instead of legacy"
)

# In run():
production = getattr(args, "production", False)
native_l0 = getattr(args, "native_l0", False)
await setup_core_engines(production=production, native_l0=native_l0)
```

---

## Testing Strategy

### Unit Tests (new file: `tests/test_native_l0.py`)

```python
import pytest
from core_engine.native import NativeSharedState, NativeTimeUtils

def test_native_shared_state_nav():
    """Test NAV update"""
    state = NativeSharedState()
    state.update_nav(86.99)
    assert state.get_nav() == 86.99

def test_native_time_utils_unix():
    """Test Unix timestamp generation"""
    now_ms = NativeTimeUtils.unix_now_ms()
    assert isinstance(now_ms, int)
    assert now_ms > 0

def test_native_time_utils_candle_align():
    """Test candle time alignment"""
    unix_ms = 1000000  # arbitrary timestamp
    aligned = NativeTimeUtils.align_candle_time(unix_ms, 60)  # 1-minute candle
    assert aligned % 60000 == 0  # Should align to minute boundary
```

### Integration Tests (update: `tests/test_production_bridge.py`)

Add new test:
```python
async def test_production_with_native_l0():
    """Test production mode with native L0"""
    from core_engine.integration import create_app_context
    
    app_ctx = await create_app_context(production=True, native_l0=True)
    
    assert "shared_state" in app_ctx
    assert isinstance(app_ctx["shared_state"], NativeSharedState)
    assert app_ctx["shared_state"].get_nav() == 0.0  # Initially empty
```

### Equivalence Test (CLI)

```bash
# Test 1: Legacy baseline
python3 main.py --mode=paper-trade --duration=30s --interval=2 --production
# Output: nav=86.99, cycle ~300ms, signals=0

# Test 2: Native L0
python3 main.py --mode=paper-trade --duration=30s --interval=2 --production --native-l0
# Output: nav=86.99, cycle ~280ms, signals=0 (should match ±0.1%)
```

---

## Success Criteria

| Criterion | Target | Validation |
|-----------|--------|-----------|
| Cycle time | -20ms (300 → 280ms) | `time.perf_counter()` measurements in logs |
| NAV equivalence | ±0.1% | Baseline $86.99 vs native $86.98-87.00 |
| Signal equivalence | ±5% | sigs=0 for both (warmup period) |
| Code reduction | 65% | ~1,800 lines → ~630 lines |
| Test coverage | 100% (L0 only) | All NativeSharedState, TimeUtils methods tested |
| Startup time | <50ms for L0 init | Should be instant (no Binance I/O) |
| Error logs | 0 in 30s run | Clean execution, no warnings/errors |

---

## Known Limitations (L0 Native)

1. **No position invariant checking** → Moved to execution layer validation
2. **No event system** → Use callback functions instead
3. **No quota reservations** → Managed by ExecutionManager (L3)
4. **Simple NAV model** → No fallback chain (handled by bridge for now)

These are acceptable because L1-L8 still use bridge until Phase 8.2.2.

---

## Rollback Plan

If equivalence test fails:
```bash
# Rollback to legacy
git revert COMMIT_SHA_OF_L0_NATIVE
git push origin phase-3/wiring
python3 main.py --production  # Auto-reverts to bridge
```

---

## Post-Implementation Checklist

- [ ] All native L0 files created
- [ ] integration.py updated with native_l0 dispatch
- [ ] main.py CLI flag added
- [ ] Unit tests passing (6/6)
- [ ] Integration test added and passing
- [ ] Equivalence test run (30s, both versions)
- [ ] NAV matches ±0.1%
- [ ] Cycle time -20ms confirmed
- [ ] Documentation updated
- [ ] Code committed with clear message
- [ ] Code review/sign-off

---

## Next Phase (8.2.2)

Once L0 native validated:
- Begin L1 (ExchangeClient) native implementation
- Keep L0 native, bridge handles L1-L8
- Expected cycle time: 300 → 260ms (-40ms cumulative)

---

**Status:** Ready for implementation  
**Owner:** @mauf  
**Start Date:** 2026-05-07  
**Target Completion:** 2026-05-20  
**Est. Effort:** 40-60 hours
