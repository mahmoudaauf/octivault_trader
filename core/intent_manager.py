"""
IntentManager subsystem extracted from MetaController.
Handles intent sink, signal cache, and related operations.
"""

from collections import deque
from typing import List, Dict, Any
from core.stubs import TradeIntent

class ThreadSafeIntentSink:
    """Thread-safe intent collection with bounded storage."""
    def __init__(self, max_size: int = 500):
        self._intents = deque(maxlen=max_size)
    
    def append(self, intent: "TradeIntent") -> None:
        self._intents.append(intent)
    
    def extend(self, intents: List["TradeIntent"]) -> None:
        self._intents.extend(intents)
    
    def drain(self) -> List["TradeIntent"]:
        intents = list(self._intents)
        self._intents.clear()
        return intents

class IntentManager:
    def __init__(self, config, logger):
        sink_size = int(getattr(config, 'INTENT_SINK_MAX_SIZE', 500))
        self.intent_sink = ThreadSafeIntentSink(max_size=sink_size)
        self.logger = logger
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
    
    def extend_intents(self, intents):
        self.intent_sink.extend(intents)
        for intent in intents:
            symbol = getattr(intent, 'symbol', None) or intent.get('symbol')
            if symbol:
                self.signal_cache[symbol] = intent
            self._emit_event('intent_appended', intent)
    
    async def receive_intents(self, intents: List[Any]):
        """Accept a batch of intents and push to sink (convenience method)."""
        if not intents:
            return
        self.extend_intents(intents)
        if len(intents) > 0:
            self.logger.debug("[IntentManager] Received %d intents", len(intents))
    
    def drain_intents(self):
        drained = self.intent_sink.drain()
        for intent in drained:
            symbol = getattr(intent, 'symbol', None) or intent.get('symbol')
            if symbol and symbol in self.signal_cache:
                del self.signal_cache[symbol]
            self._emit_event('intent_drained', intent)
        return drained
    
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
