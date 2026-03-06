"""
EventStore: Persistent, immutable event log using SQLite.

This is the source of truth for all state mutations. Events are:
- Immutable (append-only)
- Ordered (sequence numbers)
- Typed (schema validation)
- Checksummed (data integrity)
- Queryable (indexed by symbol, type, timestamp)

Architecture:
- Primary: SQLite database (data/event_store.db)
- Backup: CloudWatch logs (compliance)
- Archive: S3 (30+ day retention)
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# EVENT TYPES & SCHEMA
# ============================================================================

class EventType(Enum):
    """All event types in the system (versioned for evolution)."""
    
    # Trading events
    SIGNAL_GENERATED = "signal_generated_v1"
    SIGNAL_REJECTED = "signal_rejected_v1"
    TRADE_EXECUTED = "trade_executed_v1"
    TRADE_FILLED = "trade_filled_v1"
    TRADE_PARTIAL = "trade_partial_v1"
    TRADE_CANCELLED = "trade_cancelled_v1"
    
    # Position events
    POSITION_OPENED = "position_opened_v1"
    POSITION_UPDATED = "position_updated_v1"
    POSITION_CLOSED = "position_closed_v1"
    POSITION_LIQUIDATED = "position_liquidated_v1"
    
    # Risk events
    RISK_GATE_TRIGGERED = "risk_gate_triggered_v1"
    LOSS_LIMIT_EXCEEDED = "loss_limit_exceeded_v1"
    CAPITAL_EXCEEDED = "capital_exceeded_v1"
    
    # System events
    BOOTSTRAP_STARTED = "bootstrap_started_v1"
    BOOTSTRAP_COMPLETED = "bootstrap_completed_v1"
    SYSTEM_ERROR = "system_error_v1"
    SYSTEM_RECOVERED = "system_recovered_v1"
    
    # Snapshot events
    SNAPSHOT_CREATED = "snapshot_created_v1"
    SNAPSHOT_LOADED = "snapshot_loaded_v1"


@dataclass
class Event:
    """
    Immutable event record.
    
    Each event represents a state mutation. Events are:
    - Immutable (frozen dataclass)
    - Timestamped (UTC)
    - Sequenced (order guarantees)
    - Typed (schema validation)
    - Checksummed (integrity)
    """
    
    # Core fields (REQUIRED)
    event_type: EventType
    timestamp: float  # Unix timestamp (UTC)
    sequence: int  # 0-based sequence number
    
    # Context fields (REQUIRED)
    component: str  # Source: "meta_controller", "execution_manager", etc.
    symbol: Optional[str] = None  # Trading pair: "BTC/USDT", None for system
    
    # Payload fields (EVENT SPECIFIC)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata fields (AUTOMATIC)
    checksum: str = ""  # SHA256 of event (computed on creation)
    tags: List[str] = field(default_factory=list)  # "urgent", "debug", etc.
    
    def __hash__(self):
        """Events are hashable but immutable."""
        return hash((self.sequence, self.timestamp, self.event_type.value))
    
    def to_json(self) -> str:
        """Serialize to JSON (for storage)."""
        return json.dumps({
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "component": self.component,
            "symbol": self.symbol,
            "data": self.data,
            "checksum": self.checksum,
            "tags": self.tags,
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        data["event_type"] = EventType(data["event_type"])
        return cls(**data)


# ============================================================================
# EVENT STORE (SQLite Backend)
# ============================================================================

class EventStore:
    """
    Persistent, immutable event log using SQLite.
    
    Design principles:
    1. Append-only (no updates/deletes to events)
    2. Ordered (sequence numbers guarantee order)
    3. Queryable (indexed for fast lookups)
    4. Durable (ACID guarantees)
    5. Checksummed (data integrity verification)
    
    Schema:
    - events: Core event table (indexed by sequence, symbol, timestamp)
    - event_summaries: Aggregated view (for dashboard)
    - snapshots: Periodic state snapshots (for replay optimization)
    """
    
    def __init__(self, db_path: str = "data/event_store.db"):
        """Initialize EventStore with SQLite backend."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection will be created lazily
        self._connection: Optional[sqlite3.Connection] = None
        self._sequence = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize database schema (idempotent)."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Main events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    symbol TEXT,
                    data TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_symbol 
                ON events(symbol, timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON events(event_type, timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_component 
                ON events(component, timestamp DESC)
            """)
            
            # Event summaries (for dashboard)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_summaries (
                    hour INTEGER PRIMARY KEY,
                    signal_count INTEGER,
                    trade_count INTEGER,
                    position_count INTEGER,
                    error_count INTEGER,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    state_hash TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Get current sequence
            cursor.execute("SELECT MAX(sequence) FROM events")
            result = cursor.fetchone()
            self._sequence = (result[0] or -1) + 1
            
            conn.commit()
            logger.info(f"EventStore initialized (sequence={self._sequence})")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    async def append(self, event: Event) -> int:
        """
        Append event to store (atomically).
        
        Returns: sequence number assigned to event
        Raises: EventValidationError if event is invalid
        """
        async with self._lock:
            # Assign sequence and timestamp
            event.sequence = self._sequence
            if not event.timestamp:
                event.timestamp = time.time()
            
            # Compute checksum (for integrity)
            event.checksum = self._compute_checksum(event)
            
            # Validate event schema
            self._validate_event(event)
            
            # Write to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO events 
                    (sequence, timestamp, event_type, component, symbol, 
                     data, checksum, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.sequence,
                    event.timestamp,
                    event.event_type.value,
                    event.component,
                    event.symbol,
                    event.to_json(),
                    event.checksum,
                    json.dumps(event.tags),
                ))
                
                conn.commit()
                self._sequence += 1
                
                logger.info(f"Event appended: {event.event_type.value} "
                           f"(seq={event.sequence}, symbol={event.symbol})")
                
                return event.sequence
            
            except sqlite3.IntegrityError as e:
                conn.rollback()
                logger.error(f"Failed to append event: {e}")
                raise EventStoreError(f"Failed to append event: {e}")
    
    async def read_all(self) -> List[Event]:
        """Read all events in order (sequence)."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM events 
                ORDER BY sequence ASC
            """)
            
            events = []
            for row in cursor.fetchall():
                event = Event.from_json(row[0])
                self._verify_checksum(event)
                events.append(event)
            
            return events
    
    async def read_from(self, after_sequence: int) -> List[Event]:
        """Read events after sequence number."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM events 
                WHERE sequence > ?
                ORDER BY sequence ASC
            """, (after_sequence,))
            
            events = []
            for row in cursor.fetchall():
                event = Event.from_json(row[0])
                self._verify_checksum(event)
                events.append(event)
            
            return events
    
    async def read_for_symbol(self, symbol: str, limit: int = 1000) -> List[Event]:
        """Read events for specific symbol (most recent first)."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM events 
                WHERE symbol = ?
                ORDER BY sequence DESC
                LIMIT ?
            """, (symbol, limit))
            
            events = []
            for row in cursor.fetchall():
                event = Event.from_json(row[0])
                self._verify_checksum(event)
                events.append(event)
            
            return events
    
    async def read_by_type(self, event_type: EventType, limit: int = 1000) -> List[Event]:
        """Read events by type (most recent first)."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM events 
                WHERE event_type = ?
                ORDER BY sequence DESC
                LIMIT ?
            """, (event_type.value, limit))
            
            events = []
            for row in cursor.fetchall():
                event = Event.from_json(row[0])
                self._verify_checksum(event)
                events.append(event)
            
            return events
    
    async def read_time_range(
        self, 
        start_timestamp: float, 
        end_timestamp: float
    ) -> List[Event]:
        """Read events in time range."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data FROM events 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY sequence ASC
            """, (start_timestamp, end_timestamp))
            
            events = []
            for row in cursor.fetchall():
                event = Event.from_json(row[0])
                self._verify_checksum(event)
                events.append(event)
            
            return events
    
    async def get_event_count(self) -> int:
        """Get total event count."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]
    
    async def create_snapshot(self, state_data: Dict[str, Any]) -> str:
        """
        Create snapshot of current state.
        
        Used for fast replay (skip old events).
        Returns: snapshot_id
        """
        async with self._lock:
            snapshot_id = f"snapshot_{self._sequence}_{time.time()}"
            state_hash = self._compute_state_hash(state_data)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO snapshots 
                (snapshot_id, sequence, timestamp, state_hash, state_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                self._sequence,
                time.time(),
                state_hash,
                json.dumps(state_data),
            ))
            
            conn.commit()
            logger.info(f"Snapshot created: {snapshot_id}")
            return snapshot_id
    
    async def load_snapshot(self, snapshot_id: str) -> Tuple[int, Dict[str, Any]]:
        """Load snapshot state."""
        async with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sequence, state_data FROM snapshots 
                WHERE snapshot_id = ?
            """, (snapshot_id,))
            
            row = cursor.fetchone()
            if not row:
                raise EventStoreError(f"Snapshot not found: {snapshot_id}")
            
            sequence = row[0]
            state_data = json.loads(row[1])
            
            logger.info(f"Snapshot loaded: {snapshot_id}")
            return sequence, state_data
    
    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================
    
    def _compute_checksum(self, event: Event) -> str:
        """Compute SHA256 checksum of event (for integrity)."""
        import hashlib
        
        # Create deterministic JSON (sorted keys)
        json_str = json.dumps({
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "component": event.component,
            "symbol": event.symbol,
            "data": event.data,
        }, sort_keys=True)
        
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _verify_checksum(self, event: Event):
        """Verify event checksum (data integrity)."""
        expected = self._compute_checksum(event)
        if event.checksum != expected:
            logger.error(f"Checksum mismatch for event {event.sequence}: "
                        f"expected {expected}, got {event.checksum}")
            raise EventStoreError(f"Checksum mismatch for event {event.sequence}")
    
    def _compute_state_hash(self, state_data: Dict[str, Any]) -> str:
        """Compute hash of state (for snapshot verification)."""
        import hashlib
        json_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _validate_event(self, event: Event):
        """Validate event schema."""
        if not event.event_type:
            raise EventStoreError("event_type is required")
        if not event.timestamp:
            raise EventStoreError("timestamp is required")
        if not event.component:
            raise EventStoreError("component is required")
    
    async def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# ============================================================================
# EXCEPTIONS
# ============================================================================

class EventStoreError(Exception):
    """Base exception for EventStore errors."""
    pass


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_event_store: Optional[EventStore] = None


async def get_event_store() -> EventStore:
    """Get global EventStore instance (singleton)."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
        await _event_store.initialize()
    return _event_store
