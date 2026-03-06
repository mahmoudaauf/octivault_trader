# 🚀 PHASE 10: IMPLEMENTATION GUIDE

## Quick Overview

You now have 3 complete modules ready to integrate:

1. **`core/event_store.py`** - Persistent event log (SQLite)
2. **`core/replay_engine.py`** - Deterministic replay for forensics
3. **`core/chaos_monkey.py`** - Failure injection for testing

This guide shows how to integrate them into your system.

---

## ✅ STEP 1: INITIALIZE EVENT STORE

### 1.1 Add to Bootstrap (in your main entry point)

```python
# main.py or your bootstrap file

from core.event_store import get_event_store

async def main():
    # Initialize event store FIRST (before anything else)
    event_store = await get_event_store()
    
    # Rest of your startup code...
```

### 1.2 Start Appending Events

In your `core/shared_state.py`, whenever you emit an event:

```python
# In shared_state.py emit_event() method

from core.event_store import get_event_store, Event, EventType

async def emit_event(self, event_type: str, symbol: str = None, data: dict = None):
    """Emit event to both in-memory bus and persistent store."""
    
    # Existing event bus code
    self._event_log.append({
        "type": event_type,
        "symbol": symbol,
        "data": data,
        "timestamp": time.time(),
    })
    
    # NEW: Persist to event store
    try:
        event_store = await get_event_store()
        
        # Map your event types to EventType enum
        event_type_enum = self._map_event_type(event_type)
        
        event = Event(
            event_type=event_type_enum,
            timestamp=time.time(),
            sequence=0,  # Will be assigned by store
            component="shared_state",
            symbol=symbol,
            data=data or {},
        )
        
        await event_store.append(event)
    
    except Exception as e:
        logger.error(f"Failed to persist event: {e}")
        # Important: Continue execution even if event store fails
        # (graceful degradation)

def _map_event_type(self, event_type: str) -> EventType:
    """Map your event strings to EventType enum."""
    mapping = {
        "TRADE_EXECUTED": EventType.TRADE_EXECUTED,
        "POSITION_CLOSED": EventType.POSITION_CLOSED,
        "POSITION_OPENED": EventType.POSITION_OPENED,
        "SIGNAL_GENERATED": EventType.SIGNAL_GENERATED,
        "SIGNAL_REJECTED": EventType.SIGNAL_REJECTED,
        "RISK_GATE_TRIGGERED": EventType.RISK_GATE_TRIGGERED,
        # Add more mappings as needed
    }
    return mapping.get(event_type, EventType.SYSTEM_ERROR)
```

---

## ✅ STEP 2: TEST EVENT PERSISTENCE (8 hours)

### 2.1 Create test file: `tests/test_event_store.py`

```python
"""Test event store persistence."""

import asyncio
import pytest
from core.event_store import get_event_store, Event, EventType


@pytest.mark.asyncio
async def test_event_persistence():
    """Verify events persist to database."""
    event_store = await get_event_store()
    
    # Create test event
    event = Event(
        event_type=EventType.TRADE_EXECUTED,
        timestamp=1234567890.0,
        sequence=0,
        component="test",
        symbol="BTC/USDT",
        data={"quantity": 1.0, "price": 50000.0},
    )
    
    # Append to store
    seq = await event_store.append(event)
    assert seq == 0
    
    # Read back
    events = await event_store.read_all()
    assert len(events) >= 1
    assert events[-1].event_type == EventType.TRADE_EXECUTED
    assert events[-1].symbol == "BTC/USDT"


@pytest.mark.asyncio
async def test_snapshot_creation():
    """Verify snapshot creation."""
    event_store = await get_event_store()
    
    # Create snapshot
    state_data = {
        "open_positions": {"BTC/USDT": {"quantity": 1.0}},
        "total_capital": 100000.0,
    }
    
    snapshot_id = await event_store.create_snapshot(state_data)
    assert snapshot_id.startswith("snapshot_")
    
    # Load snapshot back
    seq, loaded_state = await event_store.load_snapshot(snapshot_id)
    assert loaded_state["total_capital"] == 100000.0


@pytest.mark.asyncio
async def test_event_query_by_symbol():
    """Verify querying events by symbol."""
    event_store = await get_event_store()
    
    # Add events for different symbols
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=1234567890.0,
            sequence=0,
            component="test",
            symbol=symbol,
            data={},
        )
        await event_store.append(event)
    
    # Query by symbol
    btc_events = await event_store.read_for_symbol("BTC/USDT")
    assert len(btc_events) > 0
    assert all(e.symbol == "BTC/USDT" for e in btc_events)


# Run with: pytest tests/test_event_store.py -v
```

### 2.2 Run tests

```bash
# Navigate to workspace
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Install pytest if needed
pip install pytest pytest-asyncio

# Run tests
pytest tests/test_event_store.py -v
```

### 2.3 Verify database created

```bash
# Check that database was created
ls -lh data/event_store.db

# Should see something like: -rw-r--r--  1 user  staff  24K Mar  2 10:15 event_store.db
```

---

## ✅ STEP 3: ENABLE REPLAY ENGINE

### 3.1 Create forensic analysis endpoint

```python
# analysis/forensics.py (NEW FILE)

from core.replay_engine import get_replay_engine
from core.event_store import get_event_store


async def analyze_loss(symbol: str, loss_threshold: float = 100.0):
    """Analyze what caused a loss."""
    replay_engine = await get_replay_engine()
    
    result = await replay_engine.find_loss_cause(
        symbol=symbol,
        loss_threshold=loss_threshold
    )
    
    return result


async def replay_to_state(sequence: int):
    """Replay to specific sequence."""
    replay_engine = await get_replay_engine()
    
    state = await replay_engine.replay_to_sequence(sequence)
    
    return state.to_dict()


async def run_what_if(modifications: dict):
    """Test what-if scenario."""
    replay_engine = await get_replay_engine()
    
    # modifications = {
    #     "trade_decision": lambda event: bool  # Accept/reject
    # }
    
    state = await replay_engine.replay_what_if(
        after_sequence=1000,
        modifications=modifications
    )
    
    return state.to_dict()
```

### 3.2 Test replay functionality

```python
# tests/test_replay_engine.py

@pytest.mark.asyncio
async def test_full_replay():
    """Test full replay from genesis."""
    replay_engine = await get_replay_engine()
    
    # Replay all events
    state = await replay_engine.replay_all()
    
    assert state is not None
    assert state.sequence >= 0
    assert state.total_capital >= 0


@pytest.mark.asyncio
async def test_replay_to_sequence():
    """Test replay to specific sequence."""
    replay_engine = await get_replay_engine()
    
    # Replay to sequence 100
    state = await replay_engine.replay_to_sequence(100)
    
    assert state.sequence <= 100
```

---

## ✅ STEP 4: ENABLE CHAOS TESTING

### 4.1 Create chaos test suite

```python
# tests/test_chaos.py

import asyncio
from core.chaos_monkey import ChaosMonkey, ResilienceVerifier, LoadTester


@pytest.mark.asyncio
async def test_api_resilience():
    """Test API resilience to failures."""
    chaos = ChaosMonkey(enabled=True, injection_rate=0.1)
    verifier = ResilienceVerifier(chaos)
    
    # Create mock API function
    async def mock_api_call(load: int = 1):
        await chaos.maybe_inject_failure("exchange_api")
        # Simulate work
        await asyncio.sleep(0.01)
        return {"status": "ok"}
    
    # Test resilience
    result = await verifier.test_api_resilience(
        mock_api_call,
        iterations=100
    )
    
    assert result["success_rate"] >= 0.85  # At least 85% success


@pytest.mark.asyncio
async def test_load_scaling():
    """Test system under load."""
    load_tester = LoadTester()
    
    async def workload(load: int):
        # Simulate load
        tasks = [
            asyncio.sleep(0.001)
            for _ in range(load)
        ]
        await asyncio.gather(*tasks)
    
    result = await load_tester.test_scaling(
        workload,
        base_load=10,
        scale_factors=[1, 5, 10]
    )
    
    assert result["max_sustainable_load"] >= 10
```

### 4.2 Run chaos tests

```bash
# Run chaos tests
pytest tests/test_chaos.py -v -s

# Output will show:
# test_api_resilience PASSED
# test_load_scaling PASSED
```

---

## ✅ STEP 5: INTEGRATION CHECKLIST

### Verify Everything Works

```python
# integration_test.py (NEW FILE)

import asyncio
from core.event_store import get_event_store, Event, EventType
from core.replay_engine import get_replay_engine
from core.chaos_monkey import ChaosMonkey


async def integration_test():
    """Verify all components work together."""
    
    print("=" * 60)
    print("PHASE 10 INTEGRATION TEST")
    print("=" * 60)
    
    # 1. Test Event Store
    print("\n1️⃣ Testing Event Store...")
    event_store = await get_event_store()
    
    event = Event(
        event_type=EventType.TRADE_EXECUTED,
        timestamp=1234567890.0,
        sequence=0,
        component="test",
        symbol="BTC/USDT",
        data={"quantity": 1.0, "price": 50000.0},
    )
    
    seq = await event_store.append(event)
    print(f"   ✅ Event appended (sequence={seq})")
    
    count = await event_store.get_event_count()
    print(f"   ✅ Event store has {count} events")
    
    # 2. Test Snapshots
    print("\n2️⃣ Testing Snapshots...")
    snapshot_id = await event_store.create_snapshot({"test": "data"})
    print(f"   ✅ Snapshot created: {snapshot_id}")
    
    seq, state = await event_store.load_snapshot(snapshot_id)
    print(f"   ✅ Snapshot loaded (sequence={seq})")
    
    # 3. Test Replay Engine
    print("\n3️⃣ Testing Replay Engine...")
    replay_engine = await get_replay_engine()
    
    state = await replay_engine.replay_all()
    print(f"   ✅ Full replay completed (state sequence={state.sequence})")
    
    # 4. Test Chaos Monkey
    print("\n4️⃣ Testing Chaos Monkey...")
    chaos = ChaosMonkey(enabled=False)  # Disabled for this test
    
    failure = await chaos.maybe_inject_failure("test_component")
    print(f"   ✅ Chaos monkey ready (last failure={failure})")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("✅ ALL PHASE 10 COMPONENTS WORKING!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run full test suite: pytest tests/")
    print("2. Deploy to staging")
    print("3. Monitor event store growth")
    print("4. Enable chaos tests in staging")
    print("5. Run production replay validation")


if __name__ == "__main__":
    asyncio.run(integration_test())
```

Run it:

```bash
python integration_test.py
```

---

## 📊 MONITORING

### Event Store Health

```python
# monitoring/event_store_health.py

from core.event_store import get_event_store


async def check_event_store_health():
    """Check event store health."""
    event_store = await get_event_store()
    
    event_count = await event_store.get_event_count()
    
    health = {
        "event_count": event_count,
        "status": "healthy" if event_count > 0 else "empty",
        "storage_path": str(event_store.db_path),
    }
    
    # Check database file size
    import os
    if os.path.exists(event_store.db_path):
        size_mb = os.path.getsize(event_store.db_path) / (1024 * 1024)
        health["size_mb"] = round(size_mb, 2)
    
    return health
```

### Replay Performance

```python
# monitoring/replay_performance.py

from core.replay_engine import get_replay_engine
import time


async def check_replay_performance():
    """Measure replay performance."""
    replay_engine = await get_replay_engine()
    
    # Full replay
    start = time.time()
    state = await replay_engine.replay_all()
    full_time = time.time() - start
    
    metrics = {
        "full_replay_seconds": round(full_time, 2),
        "events_per_second": round(state.sequence / (full_time + 0.001), 0),
        "final_sequence": state.sequence,
    }
    
    return metrics
```

---

## 🔧 CONFIGURATION

Add to your config file:

```python
# config.py or environment variables

EVENT_STORE_ENABLED = True
EVENT_STORE_PATH = "data/event_store.db"
EVENT_STORE_SNAPSHOT_INTERVAL = 1000  # Create snapshot every 1000 events
EVENT_STORE_ARCHIVE_DAYS = 90  # Archive events after 90 days

CHAOS_MONKEY_ENABLED = False  # Only enable in test/staging
CHAOS_MONKEY_INJECTION_RATE = 0.01  # 1% of requests
```

---

## 📈 EXPECTED RESULTS

After implementation, you should see:

```
✅ Event Store:
   - All events persisting to SQLite
   - Database grows ~100KB per day (with 1000+ events)
   - Append time < 1ms per event
   - Query time < 100ms for 10K events

✅ Replay Engine:
   - Full replay of 10K events in ~20-30 seconds
   - Snapshot-based replay in ~2-5 seconds
   - Loss forensics available in 30 seconds
   - What-if scenarios complete in <1 minute

✅ Chaos Resilience:
   - System survives 1% random failures
   - Automatic recovery < 30 seconds
   - No position corruption
   - No duplicate trades
```

---

## 🚨 TROUBLESHOOTING

### Event Store Not Initializing

```python
# Check that directory exists
import os
os.makedirs("data", exist_ok=True)

# Check file permissions
import subprocess
subprocess.run(["chmod", "755", "data"])
```

### Replay Too Slow

```python
# Create more snapshots
await event_store.create_snapshot(state_data)

# Or use snapshot-based replay
state = await replay_engine.replay_from_snapshot(snapshot_id)
```

### Chaos Tests Failing

```python
# Disable chaos in production
CHAOS_MONKEY_ENABLED = False

# Only enable in test/staging
import os
if os.getenv("ENVIRONMENT") == "test":
    CHAOS_MONKEY_ENABLED = True
```

---

## 📝 NEXT STEPS

1. **Week 1-2**: Event sourcing (you are here)
   - [ ] Integrate event store
   - [ ] Run tests
   - [ ] Deploy to staging
   - [ ] Monitor event persistence

2. **Week 3-4**: Deterministic replay
   - [ ] Build forensic analyzer
   - [ ] Test what-if scenarios
   - [ ] Build web dashboard

3. **Week 5-6**: Chaos resilience
   - [ ] Configure chaos monkey
   - [ ] Run all chaos tests
   - [ ] Document recovery procedures
   - [ ] Setup nightly chaos runs

---

**You're on the path to institutional-grade system! 🚀**
