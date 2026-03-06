# 🎯 PHASE 10 QUICK START - First 8 Hours

## What You're Doing

Implementing **Persistent Event Sourcing** - the foundation that unblocks replay and chaos testing.

**Goal**: By end of today, every trade will be persisted to disk.

---

## Step 1: Create EventStore Core (2 hours)

Create file: `core/event_store.py`

```python
# core/event_store.py
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import asyncio

@dataclass
class Event:
    """Immutable event record"""
    sequence_number: int
    event_type: str
    timestamp: float
    symbol: Optional[str]
    data: Dict[str, Any]
    actor: Optional[str]
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def checksum(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


class EventStore:
    """Persistent event log backed by SQLite"""
    
    def __init__(self, db_path: str = "data/event_store.db", logger=None):
        self.db_path = db_path
        self.logger = logger
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if missing"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Events table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                sequence_number INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                symbol TEXT,
                data TEXT NOT NULL,
                actor TEXT,
                version INTEGER DEFAULT 1,
                checksum TEXT NOT NULL
            )
        """)
        
        # Snapshots table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                sequence_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                state_json TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
        """)
        
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        
        conn.commit()
        conn.close()
    
    async def append(self, event_type: str, payload: Dict[str, Any], 
                     symbol: Optional[str] = None, 
                     actor: Optional[str] = None) -> int:
        """Append event to store. Returns sequence number."""
        timestamp = datetime.utcnow().timestamp()
        data_json = json.dumps(payload, default=str)
        
        # Create and checksum
        event = Event(
            sequence_number=0,
            event_type=event_type,
            timestamp=timestamp,
            symbol=symbol,
            data=payload,
            actor=actor
        )
        checksum = event.checksum()
        
        # Persist
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO events 
                (event_type, timestamp, symbol, data, actor, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_type, timestamp, symbol, data_json, actor, checksum))
            
            sequence = cur.lastrowid
            conn.commit()
            
            if self.logger:
                self.logger.debug(f"Event stored: seq={sequence}, type={event_type}")
            
            return sequence
        finally:
            conn.close()
    
    async def read_all(self, limit: Optional[int] = None) -> List[Event]:
        """Read all events in sequence order"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            query = "SELECT * FROM events ORDER BY sequence_number"
            if limit:
                query += f" LIMIT {limit}"
            
            cur.execute(query)
            rows = cur.fetchall()
            
            events = []
            for row in rows:
                seq, event_type, ts, symbol, data_json, actor, version, checksum = row
                events.append(Event(
                    sequence_number=seq,
                    event_type=event_type,
                    timestamp=ts,
                    symbol=symbol,
                    data=json.loads(data_json),
                    actor=actor,
                    version=version
                ))
            
            return events
        finally:
            conn.close()
    
    async def read_from(self, after_sequence: int) -> List[Event]:
        """Read events after given sequence number"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT * FROM events 
                WHERE sequence_number > ? 
                ORDER BY sequence_number
            """, (after_sequence,))
            
            events = []
            for row in cur.fetchall():
                seq, event_type, ts, symbol, data_json, actor, version, checksum = row
                events.append(Event(
                    sequence_number=seq,
                    event_type=event_type,
                    timestamp=ts,
                    symbol=symbol,
                    data=json.loads(data_json),
                    actor=actor,
                    version=version
                ))
            
            return events
        finally:
            conn.close()
    
    async def read_for_symbol(self, symbol: str) -> List[Event]:
        """Read all events for specific symbol"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT * FROM events 
                WHERE symbol = ? 
                ORDER BY sequence_number
            """, (symbol,))
            
            events = []
            for row in cur.fetchall():
                seq, event_type, ts, symbol_col, data_json, actor, version, checksum = row
                events.append(Event(
                    sequence_number=seq,
                    event_type=event_type,
                    timestamp=ts,
                    symbol=symbol_col,
                    data=json.loads(data_json),
                    actor=actor,
                    version=version
                ))
            
            return events
        finally:
            conn.close()
    
    async def create_snapshot(self, sequence_number: int, 
                             state_dict: Dict[str, Any]) -> str:
        """Create point-in-time snapshot. Returns snapshot ID."""
        import time as time_module
        snapshot_id = f"snap_{sequence_number}_{int(time_module.time())}"
        state_json = json.dumps(state_dict, default=str)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()
        timestamp = datetime.utcnow().timestamp()
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO snapshots 
                (snapshot_id, sequence_number, timestamp, state_json, checksum)
                VALUES (?, ?, ?, ?, ?)
            """, (snapshot_id, sequence_number, timestamp, state_json, checksum))
            
            conn.commit()
            return snapshot_id
        finally:
            conn.close()
    
    async def load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Load state from snapshot"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT state_json, sequence_number FROM snapshots 
                WHERE snapshot_id = ?
            """, (snapshot_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            state_json, sequence = row
            return {
                "state": json.loads(state_json),
                "sequence_number": sequence,
                "snapshot_id": snapshot_id
            }
        finally:
            conn.close()
```

**What to do**:
1. Create file `core/event_store.py` and paste the code above
2. Run: `mkdir -p data`

---

## Step 2: Integrate with SharedState (2 hours)

Edit: `core/shared_state.py`

Find the `__init__` method and add:

```python
# In SharedState.__init__(), after other initializations:
from core.event_store import EventStore
self.event_store = EventStore(
    db_path=getattr(config, 'EVENT_STORE_PATH', 'data/event_store.db'),
    logger=self.logger
)
```

Find the `emit_event` method (around line 4302) and modify:

```python
async def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
    """Structured event emission path; persists in-memory and notifies subscribers."""
    # Existing code...
    if event_name == "AllocationPlan":
        self._latest_allocation_plan = dict(event_data or {})
        self.logger.info("[SS] Captured latest AllocationPlan: pool=%.2f", event_data.get("pool_quote", 0))

    ts = event_data.get("ts") or event_data.get("timestamp") or time.time()
    ev_obj = {"name": event_name, "data": event_data, "timestamp": ts}
    self._event_log.append(ev_obj)
    
    # NEW: Persist to event store
    if hasattr(self, 'event_store') and self.event_store:
        symbol = event_data.get("symbol")
        actor = event_data.get("actor") or event_data.get("component")
        try:
            await self.event_store.append(
                event_type=event_name,
                payload=event_data,
                symbol=symbol,
                actor=actor
            )
        except Exception as e:
            self.logger.warning(f"Failed to persist event {event_name}: {e}")
    
    # Original subscriber notification
    await self.publish_event(event_name, event_data)
```

---

## Step 3: Create Quick Test (1 hour)

Create file: `tests/test_event_store_quick.py`

```python
# tests/test_event_store_quick.py
import pytest
import asyncio
import tempfile
import os
from core.event_store import EventStore

@pytest.mark.asyncio
async def test_append_and_read():
    """Quick sanity check"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(db_path=os.path.join(tmpdir, "test.db"))
        
        # Append event
        seq = await store.append(
            event_type="TRADE_EXECUTED",
            payload={"side": "BUY", "qty": 1.0},
            symbol="BTC/USDT",
            actor="ExecutionManager"
        )
        
        assert seq == 1, "First event should have sequence 1"
        
        # Read back
        events = await store.read_all()
        assert len(events) == 1
        assert events[0].event_type == "TRADE_EXECUTED"
        assert events[0].symbol == "BTC/USDT"
        
        print("✅ EventStore test passed!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

Run it:
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python -m pytest tests/test_event_store_quick.py -v -s
```

---

## Step 4: Configuration Updates (0.5 hours)

Edit: `config/config.py`

Add at the end:

```python
# Event sourcing configuration
EVENT_STORE_ENABLED = os.getenv("EVENT_STORE_ENABLED", "true").lower() == "true"
EVENT_STORE_PATH = os.getenv("EVENT_STORE_PATH", "data/event_store.db")
EVENT_STORE_SNAPSHOT_INTERVAL = int(os.getenv("EVENT_STORE_SNAPSHOT_INTERVAL", "3600"))  # seconds
```

---

## Step 5: Verify Integration (1 hour)

Run a quick live test:

```python
# test_live_event_persistence.py
import asyncio
from core.shared_state import SharedState
from core.event_store import EventStore

async def test():
    # Initialize SharedState
    from config import config
    shared_state = SharedState(config)
    
    # Emit an event
    await shared_state.emit_event(
        "TRADE_EXECUTED",
        {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "qty": 1.0,
            "price": 45000,
            "actor": "ExecutionManager"
        }
    )
    
    # Read it back
    events = await shared_state.event_store.read_all()
    print(f"✅ Event persisted! Total events: {len(events)}")
    
    if events:
        print(f"First event: {events[0].to_dict()}")

asyncio.run(test())
```

---

## What You've Built (So Far)

✅ **EventStore**: SQLite-backed persistent event log  
✅ **SharedState integration**: All events now saved to disk  
✅ **Event querying**: Read by symbol, sequence, type  
✅ **Snapshots**: Fast recovery capability  

---

## What's Next (Tomorrow)

- **Step 6**: Replay engine (reads events, replays state)
- **Step 7**: Forensic analyzer (finds what went wrong)
- **Step 8**: What-if simulator (tests decisions)

---

## Key Metrics to Check

After implementation:

1. **Event persistence**: Check `data/event_store.db` grows (should be ~1KB per 100 events)
2. **Performance**: Event append < 5ms
3. **Query speed**: Read 1000 events < 100ms
4. **Storage**: ~1MB per 10,000 events

---

## Troubleshooting

**Problem**: `sqlite3.OperationalError: database is locked`  
**Solution**: Close other connections, or increase timeout: `sqlite3.connect(db, timeout=10)`

**Problem**: `AttributeError: 'SharedState' has no 'event_store'`  
**Solution**: Make sure EventStore is initialized in `__init__` BEFORE other code uses it

**Problem**: Events not appearing in DB  
**Solution**: Check that `await` is used: `await store.append(...)`

---

## Success Indicators

After Part 1 (8 hours):
- [ ] `data/event_store.db` exists and grows
- [ ] Test `test_event_store_quick.py` passes
- [ ] All trades emit events to DB
- [ ] Can query events by symbol and sequence
- [ ] Snapshots create without error

🎉 **Then you can move to Part 2: Deterministic Replay!**
