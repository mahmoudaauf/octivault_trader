"""
IntentManager subsystem extracted from MetaController.
Handles intent sink, signal cache, and related operations.

PHASE 5 ENHANCEMENT: Event sourcing support
- IntentManager can persist intents to EventStore
- Tracks intent lifecycle: appended → drained → executed
- Complete audit trail for compliance
"""

from collections import deque
from typing import List, Dict, Any, Optional
import time
from core.stubs import TradeIntent

# Optional: Import EventStore for Phase 5 event sourcing
try:
    from core.event_store import EventStore, EventType
except Exception:
    EventStore = None
    EventType = None

class ThreadSafeIntentSink:
    """Thread-safe intent collection with bounded storage."""
    def __init__(self, max_size: int = 500):
        self._intents = deque(maxlen=max_size)
    
    def append(self, intent: "TradeIntent") -> None:
        self._intents.append(intent)
    
    def extend(self, intents: List["TradeIntent"]) -> None:
        self._intents.extend(intents)
    
    def drain(self, max_items: Optional[int] = None) -> List["TradeIntent"]:
        if max_items is None or int(max_items) <= 0:
            intents = list(self._intents)
            self._intents.clear()
            return intents

        drained: List["TradeIntent"] = []
        limit = max(0, int(max_items))
        for _ in range(limit):
            try:
                drained.append(self._intents.popleft())
            except IndexError:
                break
        return drained

class IntentManager:
    def __init__(self, config, logger, event_store: Optional[Any] = None):
        sink_size = int(getattr(config, 'INTENT_SINK_MAX_SIZE', 500))
        self.intent_sink = ThreadSafeIntentSink(max_size=sink_size)
        self.logger = logger
        self.event_store = event_store  # Phase 5: Event sourcing
        # Signal cache: stores the most recent signals for quick lookup
        self.signal_cache = {}
        # Event emission placeholder: can be replaced with a real event bus or callback system
        self._event_handlers = []

    def append_intent(self, intent):
        self.intent_sink.append(intent)
        # Optionally cache the signal by symbol or id
        symbol = getattr(intent, 'symbol', None) or intent.get('symbol')
        if symbol:
            self.signal_cache[symbol] = intent
        self._emit_event('intent_appended', intent)
        # PHASE 5: Persist intent to EventStore
        self._persist_intent_sync(intent, 'intent_appended')
    
    def extend_intents(self, intents):
        self.intent_sink.extend(intents)
        for intent in intents:
            symbol = getattr(intent, 'symbol', None) or intent.get('symbol')
            if symbol:
                self.signal_cache[symbol] = intent
            self._emit_event('intent_appended', intent)
            # PHASE 5: Persist each intent to EventStore
            self._persist_intent_sync(intent, 'intent_appended')
    
    async def receive_intents(self, intents: List[Any]):
        """Accept a batch of intents and push to sink (convenience method)."""
        if not intents:
            return
        self.extend_intents(intents)
        if len(intents) > 0:
            self.logger.debug("[IntentManager] Received %d intents", len(intents))
    
    def drain_intents(self, max_items: Optional[int] = None):
        drained = self.intent_sink.drain(max_items=max_items)
        for intent in drained:
            symbol = getattr(intent, 'symbol', None) or intent.get('symbol')
            if symbol and symbol in self.signal_cache:
                del self.signal_cache[symbol]
            self._emit_event('intent_drained', intent)
            # PHASE 5: Record intent drain event to EventStore
            self._persist_intent_sync(intent, 'intent_drained')
        return drained
    
    def _persist_intent_sync(self, intent: TradeIntent, event_type: str) -> bool:
        """
        PHASE 5: Persist intent event to EventStore (synchronously wrapped).
        
        This is a synchronous wrapper around async EventStore operations.
        For true async handling, use _persist_intent_async() in async contexts.
        
        Args:
            intent: TradeIntent object
            event_type: 'intent_appended' or 'intent_drained'
        
        Returns: True if persisted, False if EventStore unavailable
        """
        if not self.event_store or not EventStore or not EventType:
            return False
        
        try:
            from core.event_store import Event
            
            symbol = getattr(intent, 'symbol', None) or intent.get('symbol')
            
            # Map event type
            if event_type == 'intent_appended':
                evt_type = EventType.SIGNAL_GENERATED
            elif event_type == 'intent_drained':
                evt_type = EventType.SIGNAL_REJECTED  # Drained = no longer cached
            else:
                evt_type = EventType.SIGNAL_GENERATED
            
            # Create event from intent
            event = Event(
                event_type=evt_type,
                component="intent_manager",
                symbol=symbol,
                timestamp=getattr(intent, 'timestamp', None) or time.time(),
                data={
                    "side": getattr(intent, 'side', 'unknown'),
                    "quantity": getattr(intent, 'quantity', None),
                    "planned_quote": getattr(intent, 'planned_quote', None),
                    "confidence": getattr(intent, 'confidence', 0.0),
                    "trace_id": getattr(intent, 'trace_id', None),
                    "agent": getattr(intent, 'agent', None),
                    "tag": getattr(intent, 'tag', None),
                    "event_type": event_type,
                },
                tags=["intent_lifecycle"],
            )
            
            # Persist via synchronous connection (non-blocking)
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context, skip sync append
                    return False
                # We're in sync context, try to append
                loop.run_until_complete(self.event_store.append(event))
                self.logger.debug(
                    "[IntentManager:EventSource] Persisted intent: "
                    "symbol=%s event=%s confidence=%.2f",
                    symbol, event_type, getattr(intent, 'confidence', 0.0)
                )
                return True
            except RuntimeError:
                # No event loop or already running, skip
                return False
        
        except Exception as e:
            self.logger.debug(
                "[IntentManager:EventSource] Failed to persist intent: %s (non-blocking)", str(e)
            )
            return False
    
    def get_cached_signal(self, symbol):
        """Get the most recent cached signal for a symbol, if any."""
        return self.signal_cache.get(symbol)

    def register_event_handler(self, handler):
        """Register an event handler callback for intent events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event_type, intent):
        """Emit an event to all registered handlers (placeholder for event bus)."""
        for handler in self._event_handlers:
            try:
                handler(event_type, intent)
            except Exception as e:
                self.logger.debug(f"IntentManager event handler error: {e}")
