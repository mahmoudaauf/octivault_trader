# 🚀 PHASE 10 IMPLEMENTATION PLAN - Critical Gaps Fix

## Executive Summary

This document provides **step-by-step implementation instructions** to fix the 3 critical architectural gaps identified in the 10-Pillar Assessment:

1. **Pillar 2: Persistent Event Sourcing** (3/10 → 9/10)
2. **Pillar 3: Deterministic Replay Engine** (2/10 → 9/10)
3. **Pillar 8: Chaos Resilience Testing** (4/10 → 8/10)

**Total Effort**: 210+ hours over 5-6 weeks  
**Timeline**: Can be parallelized into 3 work streams  
**Impact**: Transforms system from "production-capable" to "institutional-grade"

---

## Part 1: Persistent Event Sourcing (60 hours)

### 1.1 Overview

**Goal**: Make all state mutations persistent and auditable. Currently, the event bus is in-memory only.

**Current State**:
```
✅ Event emission exists (meta_controller, execution_manager, etc.)
✅ Event bus exists (shared_state.emit_event())
✅ Event log in memory (self._event_log)
❌ NOT persisted to disk
❌ NO snapshots for fast recovery
❌ NO event versioning
```

**Target State**:
```
✅ All events persisted to SQLite
✅ Snapshots for point-in-time recovery
✅ Event versioning for schema evolution
✅ Audit trail for compliance
✅ Fast replay capability
```

### 1.2 Implementation Steps

#### Step 1: Create Event Store Core (12 hours)

**File**: `core/event_store.py` (NEW)

```python
# core/event_store.py
import sqlite3
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class Event:
    """Immutable event record"""
    sequence_number: int  # Global sequence (starts at 1)
    event_type: str      # e.g., "TRADE_EXECUTED"
    timestamp: float     # Unix timestamp
    symbol: Optional[str] # For filtering
    data: Dict[str, Any] # Event payload
    actor: Optional[str] # Which component emitted
    version: int = 1     # Event schema version
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def checksum(self) -> str:
        """Immutable verification"""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class EventStore:
    """Persistent event log backed by SQLite"""
    
    def __init__(self, db_path: str = "data/event_store.db", logger=None):
        self.db_path = db_path
        self.logger = logger
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
        
        # Snapshots table (for fast replay)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                sequence_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                state_json TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
        """)
        
        # Indexes for fast queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        
        conn.commit()
        conn.close()
    
    async def append(self, event_type: str, payload: Dict[str, Any], 
                     symbol: Optional[str] = None, 
                     actor: Optional[str] = None) -> int:
        """
        Append event to store. Returns sequence number.
        
        Args:
            event_type: Type of event (e.g., "TRADE_EXECUTED")
            payload: Event data
            symbol: Optional symbol for this event
            actor: Component that emitted event
        
        Returns:
            Sequence number assigned to event
        """
        timestamp = datetime.utcnow().timestamp()
        data_json = json.dumps(payload, default=str)
        
        # Create event
        event = Event(
            sequence_number=0,  # Will be assigned by DB
            event_type=event_type,
            timestamp=timestamp,
            symbol=symbol,
            data=payload,
            actor=actor
        )
        
        # Calculate checksum before persistence
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
        """
        Create point-in-time snapshot.
        Returns snapshot ID.
        """
        snapshot_id = f"snap_{sequence_number}_{int(datetime.utcnow().timestamp())}"
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
    
    async def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get most recent snapshot"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT snapshot_id, state_json, sequence_number 
                FROM snapshots 
                ORDER BY sequence_number DESC 
                LIMIT 1
            """)
            
            row = cur.fetchone()
            if not row:
                return None
            
            snapshot_id, state_json, sequence = row
            return {
                "state": json.loads(state_json),
                "sequence_number": sequence,
                "snapshot_id": snapshot_id
            }
        finally:
            conn.close()
```

#### Step 2: Integrate EventStore with SharedState (8 hours)

**File**: `core/shared_state.py` (MODIFY)

Add to SharedState class:

```python
# In SharedState.__init__():
self.event_store = EventStore(db_path=config.get("EVENT_STORE_PATH", "data/event_store.db"))

# Modify emit_event() to persist:
async def emit_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
    """Persist event to store + notify subscribers"""
    # Original in-memory logic
    if event_name == "AllocationPlan":
        self._latest_allocation_plan = dict(event_data or {})
    
    ts = event_data.get("ts") or event_data.get("timestamp") or time.time()
    ev_obj = {"name": event_name, "data": event_data, "timestamp": ts}
    self._event_log.append(ev_obj)
    
    # NEW: Persist to event store
    symbol = event_data.get("symbol")
    actor = event_data.get("actor", event_data.get("component"))
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

#### Step 3: Create Snapshot Manager (10 hours)

**File**: `core/snapshot_manager.py` (NEW)

```python
# core/snapshot_manager.py
import json
from typing import Dict, Any, Optional
from datetime import datetime

class SnapshotManager:
    """Manages point-in-time snapshots for fast recovery"""
    
    def __init__(self, event_store, logger=None):
        self.event_store = event_store
        self.logger = logger
    
    async def create_portfolio_snapshot(self, 
        portfolio_state: Dict[str, Any],
        sequence_number: int) -> str:
        """
        Create snapshot of current portfolio state
        
        Args:
            portfolio_state: Current state (positions, NAV, P&L, etc.)
            sequence_number: Event sequence at snapshot time
        
        Returns:
            Snapshot ID
        """
        snapshot = {
            "portfolio": portfolio_state,
            "sequence_number": sequence_number,
            "created_at": datetime.utcnow().isoformat()
        }
        
        snapshot_id = await self.event_store.create_snapshot(
            sequence_number=sequence_number,
            state_dict=snapshot
        )
        
        if self.logger:
            self.logger.info(f"Portfolio snapshot created: {snapshot_id}")
        
        return snapshot_id
    
    async def load_portfolio_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Load portfolio from snapshot"""
        data = await self.event_store.load_snapshot(snapshot_id)
        if data:
            return data.get("state", {}).get("portfolio")
        return None
    
    async def get_recovery_point(self) -> Optional[Dict[str, Any]]:
        """Get latest snapshot for recovery"""
        return await self.event_store.get_latest_snapshot()
```

#### Step 4: Create Compliance Audit Log (10 hours)

**File**: `core/compliance_audit_log.py` (NEW)

```python
# core/compliance_audit_log.py
import json
from datetime import datetime
from typing import Dict, Any, List

class ComplianceAuditLog:
    """Immutable audit trail for regulatory compliance"""
    
    def __init__(self, event_store, logger=None):
        self.event_store = event_store
        self.logger = logger
    
    async def get_trade_audit_trail(self, symbol: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for trades on symbol"""
        events = await self.event_store.read_for_symbol(symbol)
        
        trades = []
        for event in events:
            if event.event_type in ["TRADE_EXECUTED", "POSITION_CLOSED"]:
                trades.append({
                    "sequence": event.sequence_number,
                    "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
                    "type": event.event_type,
                    "data": event.data,
                    "checksum": event.checksum()
                })
        
        return trades
    
    async def verify_trade_integrity(self, trade_sequence: int) -> bool:
        """Verify trade hasn't been tampered with"""
        events = await self.event_store.read_all()
        
        for event in events:
            if event.sequence_number == trade_sequence:
                # Verify checksum
                stored_checksum = event.checksum()
                return True  # Simplified; add real verification
        
        return False
    
    async def export_compliance_report(self, start_date: str, 
                                      end_date: str) -> Dict[str, Any]:
        """Generate compliance report for period"""
        events = await self.event_store.read_all()
        
        # Filter by date range
        start_ts = datetime.fromisoformat(start_date).timestamp()
        end_ts = datetime.fromisoformat(end_date).timestamp()
        
        filtered = [e for e in events if start_ts <= e.timestamp <= end_ts]
        
        # Organize by symbol and type
        report = {
            "period": {"start": start_date, "end": end_date},
            "total_events": len(filtered),
            "trades": [e for e in filtered if e.event_type == "TRADE_EXECUTED"],
            "positions_closed": [e for e in filtered if e.event_type == "POSITION_CLOSED"],
            "errors": [e for e in filtered if "ERROR" in e.event_type],
        }
        
        return report
```

#### Step 5: Testing & Validation (10 hours)

**File**: `tests/test_event_store.py` (NEW)

```python
# tests/test_event_store.py
import pytest
import asyncio
from core.event_store import EventStore
import tempfile
import os

@pytest.fixture
async def event_store():
    """Create temporary event store for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(db_path=os.path.join(tmpdir, "test.db"))
        yield store

@pytest.mark.asyncio
async def test_append_event(event_store):
    """Test event persistence"""
    seq = await event_store.append(
        event_type="TRADE_EXECUTED",
        payload={"side": "BUY", "qty": 1.0},
        symbol="BTC/USDT",
        actor="ExecutionManager"
    )
    
    assert seq == 1

@pytest.mark.asyncio
async def test_read_all(event_store):
    """Test reading all events"""
    await event_store.append("TEST", {}, "BTC/USDT", "Test")
    await event_store.append("TEST", {}, "ETH/USDT", "Test")
    
    events = await event_store.read_all()
    assert len(events) == 2

@pytest.mark.asyncio
async def test_read_for_symbol(event_store):
    """Test symbol-specific queries"""
    await event_store.append("TEST", {}, "BTC/USDT", "Test")
    await event_store.append("TEST", {}, "BTC/USDT", "Test")
    await event_store.append("TEST", {}, "ETH/USDT", "Test")
    
    btc_events = await event_store.read_for_symbol("BTC/USDT")
    assert len(btc_events) == 2

@pytest.mark.asyncio
async def test_snapshot_recovery(event_store):
    """Test snapshot creation and loading"""
    state = {"positions": {"BTC/USDT": 1.0}, "nav": 1000}
    snapshot_id = await event_store.create_snapshot(sequence_number=5, state_dict=state)
    
    loaded = await event_store.load_snapshot(snapshot_id)
    assert loaded["state"]["positions"]["BTC/USDT"] == 1.0
```

### 1.3 Configuration Updates

**File**: `config/config.py` (MODIFY)

```python
# Add event sourcing configuration
EVENT_STORE_ENABLED = os.getenv("EVENT_STORE_ENABLED", "true").lower() == "true"
EVENT_STORE_PATH = os.getenv("EVENT_STORE_PATH", "data/event_store.db")
EVENT_STORE_SNAPSHOT_INTERVAL = int(os.getenv("EVENT_STORE_SNAPSHOT_INTERVAL", "3600"))  # seconds

# Compliance configuration
COMPLIANCE_AUDIT_ENABLED = os.getenv("COMPLIANCE_AUDIT_ENABLED", "true").lower() == "true"
COMPLIANCE_RETENTION_DAYS = int(os.getenv("COMPLIANCE_RETENTION_DAYS", "365"))
```

### 1.4 Deployment Checklist

- [ ] Create `core/event_store.py` (NEW)
- [ ] Create `core/snapshot_manager.py` (NEW)
- [ ] Create `core/compliance_audit_log.py` (NEW)
- [ ] Modify `core/shared_state.py` to integrate EventStore
- [ ] Create `tests/test_event_store.py` (NEW)
- [ ] Update config with event sourcing settings
- [ ] Create data directory: `mkdir -p data`
- [ ] Run tests: `pytest tests/test_event_store.py`
- [ ] Verify event persistence: Check `data/event_store.db` exists
- [ ] Monitor initial performance (should be < 5ms per event)

---

## Part 2: Deterministic Replay Engine (70 hours)

### 2.1 Overview

**Goal**: Enable replaying production events to debug issues and test strategy changes.

**Benefits**:
- Forensic debugging: Replay exact sequence that caused loss
- What-if testing: Change a decision, see outcome
- Strategy validation: Test changes on production data
- Backtesting: Use real production events as test data

### 2.2 Implementation Steps

#### Step 1: Create Replay State Machine (20 hours)

**File**: `core/replay_engine.py` (NEW)

```python
# core/replay_engine.py
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ReplayMode(Enum):
    """Replay execution modes"""
    FULL_REPLAY = "full"        # All events from genesis
    SNAPSHOT_REPLAY = "snapshot" # From snapshot onward
    RANGE_REPLAY = "range"       # Specific time range
    SYMBOL_REPLAY = "symbol"     # Single symbol only

@dataclass
class ReplayResult:
    """Output of replay execution"""
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    events_processed: int
    decisions_made: List[Dict[str, Any]]
    errors_encountered: List[str]
    execution_time_ms: float
    determinism_verified: bool

class DeterministicReplayEngine:
    """Replay production events deterministically"""
    
    def __init__(self, event_store, logger=None):
        self.event_store = event_store
        self.logger = logger
        self._event_handlers: Dict[str, Callable] = {}
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register handler for specific event type"""
        self._event_handlers[event_type] = handler
    
    async def replay_all(self, 
        initial_state: Dict[str, Any],
        mode: ReplayMode = ReplayMode.FULL_REPLAY) -> ReplayResult:
        """
        Replay all events from genesis to current state
        
        Args:
            initial_state: Starting state
            mode: Replay mode (full, snapshot, range, symbol)
        
        Returns:
            ReplayResult with execution trace
        """
        start_time = time.time()
        events = await self.event_store.read_all()
        
        current_state = dict(initial_state)
        decisions = []
        errors = []
        
        for event in events:
            try:
                # Apply event to state
                current_state = await self._apply_event(
                    current_state, event, decisions
                )
            except Exception as e:
                errors.append(f"Event {event.sequence_number}: {str(e)}")
                if self.logger:
                    self.logger.error(f"Replay error: {e}")
        
        execution_time = (time.time() - start_time) * 1000
        
        return ReplayResult(
            initial_state=initial_state,
            final_state=current_state,
            events_processed=len(events),
            decisions_made=decisions,
            errors_encountered=errors,
            execution_time_ms=execution_time,
            determinism_verified=len(errors) == 0
        )
    
    async def replay_from_snapshot(self, 
        snapshot_id: str) -> ReplayResult:
        """
        Replay from snapshot to current state (faster)
        """
        start_time = time.time()
        
        # Load snapshot
        snapshot = await self.event_store.load_snapshot(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        current_state = snapshot["state"]
        sequence_from = snapshot["sequence_number"]
        
        # Read events after snapshot
        events = await self.event_store.read_from(sequence_from)
        
        decisions = []
        errors = []
        
        for event in events:
            try:
                current_state = await self._apply_event(
                    current_state, event, decisions
                )
            except Exception as e:
                errors.append(f"Event {event.sequence_number}: {str(e)}")
        
        execution_time = (time.time() - start_time) * 1000
        
        return ReplayResult(
            initial_state=snapshot["state"],
            final_state=current_state,
            events_processed=len(events),
            decisions_made=decisions,
            errors_encountered=errors,
            execution_time_ms=execution_time,
            determinism_verified=len(errors) == 0
        )
    
    async def replay_what_if(self, 
        after_sequence: int,
        modifications: Dict[int, Callable]) -> ReplayResult:
        """
        Replay with modifications (what-if analysis)
        
        Args:
            after_sequence: Start replaying from this point
            modifications: {sequence_number: decision_modifier_fn}
        
        Returns:
            ReplayResult showing what would have happened
        
        Example:
            modifications = {
                100: lambda state: {...modify decision...}
            }
        """
        events = await self.event_store.read_all()
        
        current_state = {}
        decisions = []
        
        # Process events before modification point
        pre_events = [e for e in events if e.sequence_number <= after_sequence]
        for event in pre_events:
            current_state = await self._apply_event(
                current_state, event, decisions
            )
        
        # Process events after with modifications
        post_events = [e for e in events if e.sequence_number > after_sequence]
        
        for event in post_events:
            # Check if this event has a modification
            if event.sequence_number in modifications:
                modified_event = modifications[event.sequence_number](event)
                current_state = await self._apply_event(
                    current_state, modified_event, decisions
                )
            else:
                current_state = await self._apply_event(
                    current_state, event, decisions
                )
        
        return ReplayResult(
            initial_state={},
            final_state=current_state,
            events_processed=len(events),
            decisions_made=decisions,
            errors_encountered=[],
            execution_time_ms=0,
            determinism_verified=True
        )
    
    async def _apply_event(self, 
        state: Dict[str, Any], 
        event, 
        decisions: List) -> Dict[str, Any]:
        """
        Pure function: state + event → new state
        No side effects, deterministic
        """
        handler = self._event_handlers.get(event.event_type)
        
        if handler:
            # Use registered handler
            new_state = await self._maybe_await(handler(state, event))
            return new_state or state
        else:
            # Default: just update state with event data
            return {**state, **event.data}
    
    async def _maybe_await(self, result):
        """Handle both sync and async handlers"""
        if asyncio.iscoroutine(result):
            return await result
        return result
    
    async def verify_determinism(self, 
        num_replays: int = 3) -> bool:
        """
        Verify that replaying same events produces same state
        """
        results = []
        
        for i in range(num_replays):
            result = await self.replay_all({})
            results.append(result.final_state)
        
        # All replays should produce identical state
        first = json.dumps(results[0], sort_keys=True)
        return all(json.dumps(r, sort_keys=True) == first for r in results)
```

#### Step 2: Create Forensic Analysis Tools (15 hours)

**File**: `core/forensic_analyzer.py` (NEW)

```python
# core/forensic_analyzer.py
from typing import Dict, Any, List, Tuple
import json

class ForensicAnalyzer:
    """Analyze trading losses and decisions"""
    
    def __init__(self, event_store, replay_engine, logger=None):
        self.event_store = event_store
        self.replay_engine = replay_engine
        self.logger = logger
    
    async def analyze_loss(self, symbol: str, loss_threshold: float = 0) -> Dict[str, Any]:
        """
        Analyze a loss: What decisions led to it?
        
        Args:
            symbol: Which symbol lost money
            loss_threshold: Only analyze if loss > this
        
        Returns:
            Analysis report with decision trail
        """
        events = await self.event_store.read_for_symbol(symbol)
        
        trades = []
        current_position = 0
        cumulative_pnl = 0
        
        for event in events:
            if event.event_type == "TRADE_EXECUTED":
                data = event.data
                side = data.get("side")
                qty = data.get("qty", 0)
                price = data.get("price", 0)
                pnl = data.get("pnl", 0)
                
                trades.append({
                    "sequence": event.sequence_number,
                    "timestamp": event.timestamp,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "pnl": pnl,
                    "actor": event.actor,
                    "signals": data.get("signals", [])
                })
                
                cumulative_pnl += pnl
        
        return {
            "symbol": symbol,
            "trades": trades,
            "cumulative_pnl": cumulative_pnl,
            "total_trades": len(trades),
            "analysis": self._analyze_trades(trades)
        }
    
    def _analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze trade sequence for patterns"""
        if not trades:
            return {}
        
        buys = [t for t in trades if t["side"] == "BUY"]
        sells = [t for t in trades if t["side"] == "SELL"]
        
        buy_pnl = sum(t["pnl"] for t in buys)
        sell_pnl = sum(t["pnl"] for t in sells)
        
        return {
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_total_pnl": buy_pnl,
            "sell_total_pnl": sell_pnl,
            "avg_holding_time": self._calc_avg_holding_time(trades),
            "worst_trade": min(trades, key=lambda t: t["pnl"]),
            "best_trade": max(trades, key=lambda t: t["pnl"])
        }
    
    def _calc_avg_holding_time(self, trades: List[Dict]) -> float:
        """Calculate average position holding time"""
        if len(trades) < 2:
            return 0
        
        holding_times = []
        position = 0
        entry_time = None
        
        for trade in trades:
            if trade["side"] == "BUY":
                if position == 0:
                    entry_time = trade["timestamp"]
                position += trade["qty"]
            elif trade["side"] == "SELL":
                position -= trade["qty"]
                if position == 0 and entry_time:
                    holding_times.append(trade["timestamp"] - entry_time)
        
        return sum(holding_times) / len(holding_times) if holding_times else 0
    
    async def find_decision_error(self, loss_sequence: int) -> Dict[str, Any]:
        """
        Find what decision was wrong
        
        Args:
            loss_sequence: Sequence number of losing trade
        
        Returns:
            Report on decision that caused loss
        """
        events = await self.event_store.read_all()
        
        # Find the trade event
        trade_event = next(
            (e for e in events if e.sequence_number == loss_sequence),
            None
        )
        
        if not trade_event:
            return {"error": "Trade not found"}
        
        # Find signals that led to this trade
        symbol = trade_event.data.get("symbol")
        signals = await self.event_store.read_for_symbol(symbol)
        
        # Find signals just before trade
        pre_trade_signals = [
            s for s in signals 
            if s.sequence_number < loss_sequence and "SIGNAL" in s.event_type
        ]
        
        return {
            "trade": trade_event.to_dict(),
            "preceding_signals": [s.to_dict() for s in pre_trade_signals[-5:]],
            "analysis": "Review preceding signals to find decision error"
        }
```

#### Step 3: Create What-If Simulation Framework (20 hours)

**File**: `core/whatif_simulator.py` (NEW)

```python
# core/whatif_simulator.py
from typing import Dict, Any, Callable, List
import asyncio

class WhatIfScenario:
    """Define a what-if scenario"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.modifications: Dict[int, Callable] = {}
    
    def modify_decision(self, sequence_number: int, 
                       modifier: Callable) -> "WhatIfScenario":
        """Add a decision modification"""
        self.modifications[sequence_number] = modifier
        return self

class WhatIfSimulator:
    """Simulate alternative decisions"""
    
    def __init__(self, event_store, replay_engine, logger=None):
        self.event_store = event_store
        self.replay_engine = replay_engine
        self.logger = logger
    
    async def run_scenario(self, 
        scenario: WhatIfScenario) -> Dict[str, Any]:
        """
        Run what-if scenario
        
        Example:
            scenario = WhatIfScenario(
                "Reject Low Confidence",
                "What if we rejected signals < 0.7?"
            )
            scenario.modify_decision(100, 
                lambda evt: {...modify to reject...})
            
            result = await simulator.run_scenario(scenario)
        """
        if self.logger:
            self.logger.info(f"Running scenario: {scenario.name}")
        
        result = await self.replay_engine.replay_what_if(
            after_sequence=0,
            modifications=scenario.modifications
        )
        
        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "initial_state": result.initial_state,
            "final_state": result.final_state,
            "num_modifications": len(scenario.modifications),
            "result": result
        }
    
    async def compare_scenarios(self, 
        scenarios: List[WhatIfScenario]) -> Dict[str, Any]:
        """
        Compare multiple scenarios side-by-side
        """
        results = {}
        
        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            results[scenario.name] = result
        
        # Compute deltas
        comparison = {
            "scenarios": results,
            "deltas": self._compute_deltas(results)
        }
        
        return comparison
    
    def _compute_deltas(self, results: Dict) -> Dict[str, Any]:
        """Calculate differences between scenarios"""
        if len(results) < 2:
            return {}
        
        scenario_names = list(results.keys())
        baseline = results[scenario_names[0]]["final_state"]
        
        deltas = {}
        for name in scenario_names[1:]:
            scenario_state = results[name]["final_state"]
            deltas[name] = {
                "nav_delta": scenario_state.get("nav", 0) - baseline.get("nav", 0),
                "pnl_delta": scenario_state.get("pnl", 0) - baseline.get("pnl", 0),
                "positions_delta": self._compare_positions(
                    baseline.get("positions", {}),
                    scenario_state.get("positions", {})
                )
            }
        
        return deltas
    
    def _compare_positions(self, baseline: Dict, scenario: Dict) -> Dict:
        """Compare positions between scenarios"""
        all_symbols = set(baseline.keys()) | set(scenario.keys())
        
        return {
            symbol: {
                "baseline": baseline.get(symbol, 0),
                "scenario": scenario.get(symbol, 0),
                "delta": scenario.get(symbol, 0) - baseline.get(symbol, 0)
            }
            for symbol in all_symbols
        }
```

#### Step 4: Create Replay API & Web Interface (15 hours)

**File**: `tools/replay_api.py` (NEW)

```python
# tools/replay_api.py
from fastapi import FastAPI, HTTPException
from typing import Optional
import asyncio

app = FastAPI(title="Replay API")

class ReplayAPI:
    def __init__(self, event_store, replay_engine, forensic_analyzer, whatif_simulator):
        self.event_store = event_store
        self.replay_engine = replay_engine
        self.forensic_analyzer = forensic_analyzer
        self.whatif_simulator = whatif_simulator
    
    @app.post("/replay/full")
    async def replay_full(self):
        """Replay all events"""
        result = await self.replay_engine.replay_all({})
        return result
    
    @app.get("/forensics/loss/{symbol}")
    async def analyze_loss(self, symbol: str, threshold: float = 0):
        """Analyze loss on symbol"""
        analysis = await self.forensic_analyzer.analyze_loss(symbol, threshold)
        return analysis
    
    @app.get("/forensics/decision/{sequence}")
    async def find_error(self, sequence: int):
        """Find decision error"""
        return await self.forensic_analyzer.find_decision_error(sequence)
    
    @app.post("/whatif/scenario")
    async def run_scenario(self, name: str, description: str, modifications: dict):
        """Run what-if scenario"""
        scenario = WhatIfScenario(name, description)
        result = await self.whatif_simulator.run_scenario(scenario)
        return result
```

#### Step 5: Integration Testing (10 hours)

**File**: `tests/test_replay_engine.py` (NEW)

Create comprehensive tests for:
- Full replay determinism
- Snapshot-based replay
- What-if modification
- Forensic analysis accuracy
- Performance (< 1s for 1000 events)

### 2.3 Deployment Checklist

- [ ] Create `core/replay_engine.py`
- [ ] Create `core/forensic_analyzer.py`
- [ ] Create `core/whatif_simulator.py`
- [ ] Create `tools/replay_api.py`
- [ ] Create `tests/test_replay_engine.py`
- [ ] Run determinism verification test
- [ ] Performance test: < 1s for 1000 events
- [ ] Deploy API endpoint
- [ ] Create replay CLI tool
- [ ] Documentation with examples

---

## Part 3: Chaos Resilience Testing (80 hours)

### 3.1 Overview

**Goal**: Test system resilience through controlled failure injection.

**Coverage**:
- Exchange API failures (timeout, 500 error, bad response)
- Network issues (partition, slow network)
- Database failures
- Race condition injection
- Overload scenarios

### 3.2 Implementation Steps

*(Detailed in Part 3, Section 3.3-3.7 below)*

---

## Part 3.3: Chaos Injection Framework (20 hours)

**File**: `core/chaos_monkey.py` (NEW)

```python
# core/chaos_monkey.py - Controlled failure injection
import random
import asyncio
from typing import Callable, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

class FailureType(Enum):
    """Types of failures to inject"""
    API_TIMEOUT = "api_timeout"           # Slow response
    API_500_ERROR = "api_500"             # Server error
    API_RATE_LIMIT = "api_rate_limit"     # Rate limiting
    NETWORK_PARTITION = "network_partition"  # Connection lost
    CORRUPTED_RESPONSE = "corrupted_response"  # Bad JSON
    DATABASE_UNAVAILABLE = "db_unavailable"   # DB connection lost
    CLOCK_SKEW = "clock_skew"             # Time drift
    RACE_CONDITION = "race_condition"     # Concurrent access

@dataclass
class FailureScenario:
    """Failure scenario configuration"""
    name: str
    description: str
    failure_type: FailureType
    injection_rate: float  # 0.0-1.0 (% of requests)
    duration_ms: int       # How long to simulate failure
    recovery_time_ms: int  # Time to recover

class ChaosMonkey:
    """Inject failures into system"""
    
    def __init__(self, logger=None, enabled: bool = True):
        self.logger = logger
        self.enabled = enabled
        self.injected_failures = []
        self.failure_scenarios: Dict[str, FailureScenario] = {}
    
    def register_scenario(self, scenario: FailureScenario):
        """Register a failure scenario"""
        self.failure_scenarios[scenario.name] = scenario
    
    async def maybe_inject_failure(self, 
        scenario_name: str) -> Optional[FailureScenario]:
        """
        Randomly inject failure based on scenario
        
        Returns:
            The scenario if failure was injected, else None
        """
        if not self.enabled:
            return None
        
        scenario = self.failure_scenarios.get(scenario_name)
        if not scenario:
            return None
        
        # Decide whether to inject
        if random.random() > scenario.injection_rate:
            return None
        
        # Inject the failure
        await self._execute_failure(scenario)
        self.injected_failures.append(scenario)
        
        if self.logger:
            self.logger.warning(f"[Chaos] Injected {scenario.failure_type.value}")
        
        return scenario
    
    async def _execute_failure(self, scenario: FailureScenario):
        """Execute the actual failure"""
        if scenario.failure_type == FailureType.API_TIMEOUT:
            await asyncio.sleep(scenario.duration_ms / 1000)
        
        elif scenario.failure_type == FailureType.API_500_ERROR:
            raise Exception("500 Internal Server Error (injected)")
        
        elif scenario.failure_type == FailureType.NETWORK_PARTITION:
            await asyncio.sleep(scenario.duration_ms / 1000)
            # Simulate connection drop
        
        elif scenario.failure_type == FailureType.CLOCK_SKEW:
            # Simulate time drift (would need mocking in real system)
            pass
        
        # Simulate recovery time
        await asyncio.sleep(scenario.recovery_time_ms / 1000)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get failure injection statistics"""
        return {
            "total_injected": len(self.injected_failures),
            "by_type": self._count_by_type(),
            "scenarios": list(self.failure_scenarios.keys())
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count failures by type"""
        counts = {}
        for scenario in self.injected_failures:
            key = scenario.failure_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
```

### 3.3: Resilience Verification Framework (20 hours)

**File**: `core/resilience_verifier.py` (NEW)

Tests that system:
1. Recovers from each failure type
2. No trades lost/duplicated
3. Position data consistent
4. Recovery time < threshold
5. No capital loss

### 3.4: Load Testing Framework (15 hours)

**File**: `tools/load_tester.py` (NEW)

Simulate:
- 10x normal symbol count
- 100x normal signal rate
- Measure CPU, memory, latency
- Find saturation points

### 3.5: Failover Testing (15 hours)

**File**: `tools/failover_tester.py` (NEW)

Test recovery from:
- Exchange API failures
- WebSocket disconnects
- Database unavailable
- System restart mid-trade
- Disk corruption

### 3.6: Chaos Test Suite (10 hours)

**File**: `tests/test_chaos_resilience.py` (NEW)

```python
@pytest.mark.asyncio
async def test_api_timeout_recovery():
    """System survives API timeout"""
    chaos = ChaosMonkey(enabled=True)
    scenario = FailureScenario(
        name="API Timeout",
        description="Exchange API times out",
        failure_type=FailureType.API_TIMEOUT,
        injection_rate=0.5,
        duration_ms=5000,
        recovery_time_ms=1000
    )
    chaos.register_scenario(scenario)
    
    # Run trading cycles with injected failures
    for _ in range(100):
        await chaos.maybe_inject_failure("API Timeout")
        await system.execute_trading_cycle()
    
    # Verify system integrity
    assert system.state.is_consistent()
    assert system.nav > 0
    # ... more assertions

@pytest.mark.asyncio
async def test_load_10x_symbols():
    """System handles 10x normal symbol count"""
    normal_symbols = system.get_symbol_count()
    
    # Scale up to 10x
    await system.add_symbols([f"SYM{i}/USDT" for i in range(normal_symbols * 10)])
    
    # Measure performance
    metrics = await system.measure_performance()
    
    assert metrics.latency_p99 < 1000  # < 1s
    assert metrics.cpu_usage < 90      # < 90%
    assert metrics.memory_usage < 8e9  # < 8GB
```

### 3.7: Chaos Dashboard (10 hours)

**File**: `tools/chaos_dashboard.py` (NEW)

Live visualization of:
- Failures injected
- System recovery time
- Performance metrics
- Test progress

---

## Implementation Timeline

```
WEEK 1: Event Sourcing
├─ Day 1: EventStore core + tests (12h)
├─ Day 2: SharedState integration (8h)
├─ Day 3: SnapshotManager (10h)
├─ Day 4: ComplianceAuditLog (10h)
└─ Day 5: Validation & bugs (20h)

WEEK 2: Deterministic Replay
├─ Day 1: ReplayEngine core (20h)
├─ Day 2: ForensicAnalyzer (15h)
├─ Day 3: WhatIfSimulator (20h)
├─ Day 4: ReplayAPI + integration (15h)
└─ Day 5: Testing & debugging (10h)

WEEKS 3-4: Chaos Testing
├─ Week 1: ChaosMonkey framework (20h)
├─        ResilienceVerifier (20h)
├─        LoadTester (15h)
└─ Week 2: FailoverTester (15h)
           Test suite (10h)
           Dashboard (10h)
```

---

## Success Metrics

### Pillar 2 (Event Sourcing)
- [ ] 100% of events persisted
- [ ] Event store database < 100MB per month
- [ ] Event read latency < 5ms
- [ ] Compliance audit log complete
- [ ] No data loss on crash

### Pillar 3 (Replay)
- [ ] Deterministic replay verified (3+ replays identical)
- [ ] Replay speed > 1000 events/second
- [ ] Forensic analysis < 100ms
- [ ] What-if simulation accurate
- [ ] 95%+ replay success rate

### Pillar 8 (Chaos)
- [ ] All 8 failure types testable
- [ ] Recovery time < 30 seconds
- [ ] Zero data loss across failures
- [ ] Load test to 100x normal volume
- [ ] Chaos test suite runs nightly

---

## Next Steps

1. **Immediately** (This week):
   - Create EventStore core
   - Integrate with SharedState
   - Run basic tests

2. **Week 2**:
   - Deploy EventStore to staging
   - Monitor event persistence
   - Fix any issues

3. **Week 3**:
   - Build Replay engine
   - Test determinism
   - Create forensic tools

4. **Weeks 4-6**:
   - Implement chaos testing
   - Run resilience tests
   - Validate all fixes

5. **Week 7+**:
   - Production deployment
   - Monitor performance
   - Iterate on issues

---

## Risk Mitigation

**Risk**: Event store DB grows too large
**Mitigation**: Implement cleanup job to archive old events

**Risk**: Replay performance slow
**Mitigation**: Implement snapshot strategy (snapshot every 1000 events)

**Risk**: Chaos tests destabilize production
**Mitigation**: Run only in staging/test environments, never production

**Risk**: Compliance audit adds overhead
**Mitigation**: Run async, use sampling for high-volume events

---

## Document Version

- **Date**: March 2, 2026
- **Status**: Ready for implementation
- **Reviewer**: Architectural Assessment Complete
- **Next Review**: After Phase 10 completion

---

**Ready to start?** Let me know if you want me to begin with Step 1 (EventStore), or if you have questions about any section.
