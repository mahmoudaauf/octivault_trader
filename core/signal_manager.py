"""
SignalManager subsystem extracted from MetaController.
Handles signal cache, signal intake, validation, deduplication, and queue management.
"""
from typing import Dict, Any, List, Optional, Tuple
import time

# Import utilities needed for intent processing
try:
    from core.meta_controller import parse_timestamp as _parse_ts
except ImportError:
    # Fallback timestamp parsing
    def _parse_ts(ts_val, now_ts):
        if isinstance(ts_val, (int, float)):
            return float(ts_val)
        return now_ts

class SignalManager:
    def __init__(self, config, logger, signal_cache=None, intent_manager=None):
        self.config = config
        self.logger = logger
        self.intent_manager = intent_manager
        
        # Use provided cache or create new one
        if signal_cache is not None:
            self.signal_cache = signal_cache
            self.logger.info("[SignalManager] Using provided signal cache")
        else:
            # Try to use BoundedCache from meta_controller.cache, fallback to InlineBoundedCache
            try:
                from core.meta_controller import BoundedCache
                cache_size = int(getattr(config, 'signal_cache_max_size', 1000))
                cache_ttl = float(getattr(config, 'signal_cache_ttl', 300.0))
                self.signal_cache = BoundedCache(max_size=cache_size, default_ttl=cache_ttl)
                self.logger.info(f"[SignalManager] Signal cache initialized: max_size={cache_size}, ttl={cache_ttl}s")
            except ImportError:
                self.logger.warning("[SignalManager] BoundedCache unavailable, using InlineBoundedCache fallback")
                self.signal_cache = InlineBoundedCache(max_size=1000, default_ttl=300)

        # Configuration for signal validation
        self._min_conf_ingest = float(getattr(config, 'MIN_SIGNAL_CONF', 0.50))
        self._max_age_sec = float(getattr(config, 'MAX_SIGNAL_AGE_SECONDS', 60))
        self._known_quotes = {"USDT", "FDUSD", "USDC", "BUSD", "TUSD", "DAI"}

    def receive_signal(self, agent_name: str, symbol: str, signal: Dict[str, Any]) -> bool:
        """
        Accept and cache signals with validation and deduplication.

        Args:
            agent_name: Name of the agent sending the signal
            symbol: Trading symbol
            signal: Signal dictionary

        Returns:
            True if signal was accepted and cached, False otherwise
        """
        if not symbol or not isinstance(signal, dict):
            self.logger.warning("[SignalManager] Invalid signal received: symbol=%s signal=%s", symbol, signal)
            return False

        # Normalize symbol
        sym = self._normalize_symbol(symbol)
        if not sym or len(sym) < 6:
            self.logger.debug("[SignalManager] Rejected suspicious symbol: %r", sym)
            return False

        # Check if base is a known quote token
        base, quote = self._split_base_quote(sym)
        if base.upper() in self._known_quotes:
            self.logger.debug("[SignalManager] %s rejected: base is a known quote token.", sym)
            return False

        # Check if quote is a known quote token
        if quote.upper() not in self._known_quotes:
            self.logger.debug("[SignalManager] %s rejected: quote %s is not a known quote token.", sym, quote)
            return False

        # Early confidence check
        conf_raw = float(signal.get("confidence", 0.0))
        if conf_raw < self._min_conf_ingest:
            self.logger.debug("[SignalManager] %s conf %.2f < ingest floor %.2f", sym, conf_raw, self._min_conf_ingest)
            return False

        if conf_raw > 1.0:
            self.logger.warning("[SignalManager] Confidence inflation detected from %s for %s: %.2f. Clamping to 1.0", agent_name, sym, conf_raw)

        # Prepare signal for caching
        s = dict(signal)
        s["agent"] = agent_name
        s["symbol"] = sym
        s["timestamp"] = time.time()
        s["confidence"] = max(0.0, min(1.0, conf_raw))

        # Set default quote if not provided
        if "quote" not in s or float(s.get("quote") or 0) <= 0:
            s["quote"] = float(getattr(self.config, 'DEFAULT_PLANNED_QUOTE', 10.0))

        # Store in cache with deduplication by symbol:agent
        cache_key = f"{sym}:{agent_name}"
        self.signal_cache.set(cache_key, s)
        return True

    def store_signal(self, agent_name: str, symbol: str, signal: Dict[str, Any]) -> None:
        """
        Store a processed signal in the cache.

        Args:
            agent_name: Name of the agent
            symbol: Trading symbol
            signal: Processed signal dictionary
        """
        cache_key = f"{symbol}:{agent_name}"
        self.signal_cache.set(cache_key, signal)
        self.logger.debug("[SignalManager] Signal stored for %s from %s", symbol, agent_name)

    def get_all_signals(self) -> List[Dict[str, Any]]:
        """
        Get all non-expired signals from the cache.

        Returns:
            List of all cached signals
        """
        return self.signal_cache.list_all()

    def get_signals_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all signals for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of signals for the symbol
        """
        all_signals = self.get_all_signals()
        return [s for s in all_signals if s.get("symbol") == symbol]

    def cleanup_expired_signals(self) -> int:
        """
        Clean up expired signals from the cache.

        Returns:
            Number of expired signals removed
        """
        return self.signal_cache.cleanup_expired()

    def flush_intents_to_cache(self, now_ts: float) -> int:
        """
        Process intents from intent_manager into signal cache.

        Args:
            now_ts: Current timestamp

        Returns:
            Number of intents processed into signals
        """
        if not self.intent_manager:
            return 0

        intents = self.intent_manager.drain_intents()
        if not intents:
            return 0

        self.logger.debug("[SignalManager:Flush] Draining %d intents from sink...", len(intents))

        accepted = 0
        for it in intents:
            try:
                if hasattr(it, "to_dict"):
                    d = it.to_dict()
                elif isinstance(it, dict):
                    d = dict(it)
                else:
                    continue

                symbol = self._normalize_symbol(d.get("symbol") or "")
                # Flexible action/side lookup
                action_raw = d.get("action") or d.get("side") or ""
                action = str(action_raw).upper()
                conf = float(d.get("confidence", 0.0))
                agent = d.get("agent", "Agent")

                # Validate symbol format (same as receive_signal)
                if not symbol or len(symbol) < 6:
                    self.logger.debug("[SignalManager:Flush] Rejected suspicious symbol: %r", symbol)
                    continue

                # Check if base is a known quote token
                base, quote = self._split_base_quote(symbol)
                if base.upper() in self._known_quotes:
                    self.logger.debug("[SignalManager:Flush] %s rejected: base is a known quote token.", symbol)
                    continue

                # Check if quote is a known quote token
                if quote.upper() not in self._known_quotes:
                    self.logger.debug("[SignalManager:Flush] %s rejected: quote %s is not a known quote token.", symbol, quote)
                    continue

                if not symbol or action not in ("BUY", "SELL"):
                    if not symbol:
                        self.logger.debug("[SignalManager:Flush] Rejected intent: missing symbol.")
                    elif action not in ("BUY", "SELL"):
                        self.logger.debug("[SignalManager:Flush] Rejected intent for %s: invalid action '%s'", symbol, action)
                    continue

                # Confidence check
                if conf < self._min_conf_ingest:
                    self.logger.debug("[SignalManager:Flush] %s conf %.2f < ingest floor %.2f", symbol, conf, self._min_conf_ingest)
                    continue

                # TTL check
                ts_val = d.get("timestamp") or d.get("ts") or now_ts
                ts = _parse_ts(ts_val, now_ts)
                ttl_sec = float(d.get("ttl_sec", 0.0) or 0.0)
                if ttl_sec > 0 and (now_ts - ts) > ttl_sec:
                    continue

                # Budget requirement check (BUY requires budget, SELL/HOLD don't)
                budget_required = action == "BUY"

                # Use dictionary update to preserve all metadata from intent
                sig = dict(d)
                sig.update({
                    "symbol": symbol,
                    "action": action,
                    "confidence": conf,
                    "agent": agent,
                    "timestamp": now_ts,
                    "budget_required": budget_required,
                })
                
                if "planned_quote" in d or "quote" in d:
                    sig["quote"] = float(d.get("planned_quote", d.get("quote", float(getattr(self.config, 'DEFAULT_PLANNED_QUOTE', 10.0)))))
                if "planned_qty" in d or "quantity" in d:
                    sig["quantity"] = float(d.get("planned_qty", d.get("quantity", 0.0)))

                # Store the signal directly (this is synchronous, unlike MetaController.receive_signal which is async)
                self.store_signal(agent, symbol, sig)
                accepted += 1
            except Exception:
                self.logger.debug("intent->signal failed: %r", it, exc_info=True)

        if accepted > 0:
            self.logger.info("[SignalManager:Flush] Successfully ingested %d signals into cache.", accepted)
        return accepted

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to uppercase."""
        return symbol.upper().strip() if symbol else ""

    def _split_base_quote(self, symbol: str) -> Tuple[str, str]:
        """Split symbol into base and quote currencies."""
        # Simple split - assumes format like BTCUSDT
        if len(symbol) < 6:
            return "", ""
        # Try to find quote in known quotes, starting with longest quotes first
        sorted_quotes = sorted(self._known_quotes, key=len, reverse=True)
        for quote in sorted_quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        # Fallback: try common quote lengths (4, 3)
        for quote_len in [4, 3]:
            if len(symbol) > quote_len:
                potential_quote = symbol[-quote_len:]
                if potential_quote.upper() in self._known_quotes:
                    return symbol[:-quote_len], potential_quote
        # Final fallback: assume last 4 chars are quote (for backward compatibility)
        return symbol[:-4], symbol[-4:]

class InlineBoundedCache:
    def __init__(self, max_size=1000, default_ttl=300):
        from collections import deque
        self._cache = {}
        self._timestamps = {}
        self._access_order = deque(maxlen=max_size)
        self.max_size = max_size
        self.default_ttl = default_ttl
    def get(self, key, default=None):
        import time
        now = time.time()
        if key in self._cache:
            if now - self._timestamps.get(key, 0) < self.default_ttl:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                return self._cache[key]
            else:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)
        return default
    def set(self, key, value, ttl=None):
        import time
        now = time.time()
        if key not in self._cache and len(self._cache) >= self.max_size:
            if self._access_order:
                oldest = self._access_order.popleft()
                self._cache.pop(oldest, None)
                self._timestamps.pop(oldest, None)
        self._cache[key] = value
        self._timestamps[key] = now
        self._access_order.append(key)
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()
        self._access_order.clear()
    def cleanup_expired(self):
        import time
        now = time.time()
        expired = [k for k, t in self._timestamps.items() if now - t >= self.default_ttl]
        for k in expired:
            self._cache.pop(k, None)
            self._timestamps.pop(k, None)
            try:
                self._access_order.remove(k)
            except ValueError:
                pass
        return len(expired)

# Provide a simple list_all implementation so the SignalManager can
# call `signal_cache.list_all()` on the inline fallback.
    def list_all(self) -> List[Dict[str, Any]]:
        """Return a list of cached values (non-expired)."""
        now = time.time()
        out = []
        for k, v in list(self._cache.items()):
            ts = self._timestamps.get(k, 0)
            if now - ts < self.default_ttl:
                out.append(v)
            else:
                # lazily clean expired
                self._cache.pop(k, None)
                self._timestamps.pop(k, None)
                try:
                    self._access_order.remove(k)
                except ValueError:
                    pass
        return out

# If the external BoundedCache import fails, use the inline fallback
# by aliasing the expected name.
BoundedCache = InlineBoundedCache
