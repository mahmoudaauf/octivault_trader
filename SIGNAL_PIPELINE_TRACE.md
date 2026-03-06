# Signal Pipeline End-to-End Trace

## Overview
This document traces the complete signal flow from Strategy Agent signal generation through to execution attempt, with detailed code locations and data transformations.

---

## 1. STRATEGY AGENT → Signal Generation

### Entry Point
**File:** `core/agent_manager.py`  
**Method:** `collect_and_forward_signals()` (line 374)

```
Strategy Agent (external)
    ↓
generate_signals() [called once per tick]
    ↓
Returns raw signal(s) - dict | list | single object
```

### Code Flow
```python
# agent_manager.py:374-460
async def collect_and_forward_signals(self):
    """Single signal collection point - calls generate_signals() once per tick."""
    
    batch = []
    strategy_agents = [
        (name, agent) for name, agent in list(self.agents.items())
        if getattr(agent, "agent_type", None) != "discovery" 
        and hasattr(agent, "generate_signals")
    ]
    
    # For each strategy agent:
    for name, agent in strategy_agents:
        try:
            fn = getattr(agent, "generate_signals")
            res = fn()
            if inspect.isawaitable(res):
                res = await res
            
            # Normalize raw signals to TradeIntent objects
            intents = self._normalize_to_intents(name, res)
            if intents:
                batch.extend(intents)
                self.logger.info("[%s] Normalized %d intents", name, len(intents))
```

### Key Characteristics
- **Timing:** Called once per Agent tick
- **Output:** List of `TradeIntent` objects (dicts with standardized schema)
- **Normalization:** Raw signals converted to canonical TradeIntent format via `_normalize_to_intents()`
- **Debug Logging:** `[AgentManager:BATCH] Submitted batch of %d intents`

---

## 2. AGENT MANAGER → submit_trade_intents()

### Entry Point
**File:** `core/agent_manager.py`  
**Method:** `submit_trade_intents()` (line 255)

### Code Flow
```python
# agent_manager.py:255-290
async def submit_trade_intents(self, intents: List[Dict[str, Any]]):
    """
    Bind Agent→Meta pipe through the event bus.
    Publishes on: events.trade.intent
    """
    event_bus = getattr(self.shared_state, "event_bus", None)
    publish = getattr(event_bus, "publish", None)
    
    if not callable(publish):
        self.logger.warning("submit_trade_intents: event_bus.publish unavailable")
        return
    
    published = 0
    for raw in intents:
        ti = self._coerce_trade_intent(raw)
        if ti is None:
            continue
        
        try:
            await publish("events.trade.intent", ti)
            published += 1
        except Exception as e:
            self.logger.warning("Failed to publish trade intent for %s", symbol)
    
    self.logger.info("[AgentManager] Published %d trade intent events", published)
```

### Key Characteristics
- **Event Topic:** `"events.trade.intent"`
- **Payload:** Canonical `TradeIntent` object (coerced via `_coerce_trade_intent()`)
- **Transport:** Async publish to event bus (non-blocking)
- **Timing:** Immediately after batch collection

---

## 3. EVENT BUS → Publish/Subscribe

### Entry Point
**File:** `core/shared_state.py`  
**Method:** Event Bus publish mechanism

### Architecture
- **Bus Type:** Internal async queue-based event bus
- **Topic:** `"events.trade.intent"`
- **Subscriber:** Meta._trade_intent_subscriber_name
- **Queue:** `self._trade_intent_event_queue` (asyncio.Queue)

### Code Flow (Event Publishing)
```python
# shared_state.py:623
self.event_bus = _SharedStateEventBus(self)

# Publishing:
await event_bus.publish("events.trade.intent", trade_intent_dict)
# ↓ Queued to all subscribers listening on this topic
```

### Subscription Setup
**File:** `core/meta_controller.py`  
**Method:** `_ensure_trade_intent_subscription()` (line 4913)

```python
# meta_controller.py:4913-4950
async def _ensure_trade_intent_subscription(self) -> bool:
    """Subscribe MetaController to trade intent events."""
    
    if self._trade_intent_event_queue is not None:
        return True
    
    event_bus = getattr(self.shared_state, "event_bus", None)
    subscribe = getattr(event_bus, "subscribe", None)
    
    if not callable(subscribe):
        self.logger.warning("Event bus subscribe unavailable")
        return False
    
    try:
        q = await subscribe(self._trade_intent_subscriber_name, "events.trade.intent")
        self._trade_intent_event_queue = q
        self.logger.info("[Meta:EventBus] Subscribed to events.trade.intent")
        return True
    except Exception as e:
        self.logger.warning("[Meta:EventBus] Subscription failed: %s", e)
        return False
```

### Key Characteristics
- **Queue Type:** `asyncio.Queue` (non-blocking, FIFO)
- **Subscriber ID:** `meta_controller_trade_intent_subscriber`
- **Event Format:** `{"name": "events.trade.intent", "timestamp": ts, "data": trade_intent_dict}`

---

## 4. META → _drain_trade_intent_events()

### Entry Point
**File:** `core/meta_controller.py`  
**Method:** `_drain_trade_intent_events()` (line 4992)

### Code Flow
```python
# meta_controller.py:4992-6035
async def _drain_trade_intent_events(self, max_items: int = 500) -> int:
    """Drain `events.trade.intent` messages from the bus into IntentManager."""
    
    if not await self._ensure_trade_intent_subscription():
        return 0
    
    q = self._trade_intent_event_queue
    if q is None:
        return 0
    
    accepted: List[Dict[str, Any]] = []
    max_items = max(1, int(max_items or 1))
    
    # Non-blocking bulk drain
    for _ in range(max_items):
        try:
            ev = q.get_nowait()  # ← Non-blocking dequeue
        except _asyncio.QueueEmpty:
            break
        
        try:
            name = str((ev or {}).get("name") or "")
            if name != "events.trade.intent":
                continue
            
            event_ts = float((ev or {}).get("timestamp") or time.time())
            
            # Normalize event payload
            norm = self._normalize_trade_intent_event((ev or {}).get("data"), event_ts)
            if norm is not None:
                accepted.append(norm)
        finally:
            try:
                q.task_done()
            except Exception:
                pass
    
    # Forward to Intent Manager (caches in signal_cache)
    if accepted:
        await self.intent_manager.receive_intents(accepted)
        self.logger.debug("[Meta:EventBus] Drained %d trade intents", len(accepted))
    
    return len(accepted)
```

### Key Characteristics
- **Drain Pattern:** Non-blocking bulk dequeue (get_nowait)
- **Max Items:** Configurable (default 500 per cycle)
- **Timing:** Called at start of `evaluate_and_act()` lifecycle
- **Debug Logging:** `[Meta:DRAIN] ⚠️ DRAINED %d events from event_bus`

### Normalization
**Method:** `_normalize_trade_intent_event()` (internal)

Converts raw event payload to canonical schema:
```python
{
    "symbol": "BTCUSDT",
    "action": "BUY" | "SELL",
    "confidence": 0.0-1.0,
    "agent": "AgentName",
    "timestamp": epoch_seconds,
    "reason": "signal_type",
    ...other_fields...
}
```

---

## 5. INTENT MANAGER → Signal Cache

### Entry Point
**File:** `core/meta_controller.py`  
**Method:** `intent_manager.receive_intents()` (called from `_drain_trade_intent_events`)

### Code Flow
The drained events are cached in the **Signal Cache** by intent_manager:

```python
# From _drain_trade_intent_events:
await self.intent_manager.receive_intents(accepted)

# Intent Manager stores in signal_cache (per-symbol)
# Signal structure:
{
    "symbol": "BTCUSDT",
    "action": "BUY" | "SELL",
    "confidence": float,
    "agent": "AgentName",
    "timestamp": epoch_seconds,
    "reason": str,
    "_planned_quote": float,  # ← Budget override if provided
    "execution_tag": str,      # ← Tracing ID
    ...
}
```

### Signal Cache Organization
**File:** `core/meta_controller.py`

```python
self.signal_manager  # Manages per-symbol signal cache
self.signal_manager.get_signals_for_symbol(symbol)
    ↓
Returns: List[Dict] of cached signals for symbol
```

---

## 6. META → _build_decisions()

### Entry Point
**File:** `core/meta_controller.py`  
**Method:** `_build_decisions()` (line 8221)

### Full Lifecycle (evaluate_and_act → _build_decisions)

```python
# meta_controller.py:5800-5900 (evaluate_and_act)
async def evaluate_and_act(self):
    """P9 Lifecycle: Ingest signals, build decisions, execute."""
    
    # STEP 1: Signal Ingestion (Event Bus Drain)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    drained = await self._drain_trade_intent_events(
        max_items=int(self._cfg("TRADE_INTENT_EVENT_DRAIN_MAX", 1000) or 1000)
    )
    self.logger.warning("[Meta:DRAIN] ⚠️ DRAINED %d events from event_bus", drained)
    
    # Flush ingested intents to signal cache
    await self._flush_intents_to_cache(now_ts=now_epoch)
    await self._ingest_strategy_bus(now_ts=now_epoch)
    await self._ingest_liquidation_signals(now_ts=now_epoch)
    
    # STEP 2: Symbol Universe Synchronization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    accepted_symbols_set = set(self.shared_state.get_analysis_symbols())
    
    # STEP 3: Build Decision Context (Arbitration)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    decisions = await self._build_decisions(accepted_symbols_set)
    # ← Returns: List[Tuple[symbol, side, signal_dict]]
```

### _build_decisions Implementation

```python
# meta_controller.py:8221-8955
async def _build_decisions(self, accepted_symbols_set: set):
    """
    Portfolio arbitration pipeline.
    
    INPUT:  accepted_symbols_set (set of tradable symbols)
    OUTPUT: List[(symbol, side, decision_signal_dict)]
    
    FLOW:
    1. Check portfolio flat/active state
    2. Apply governance decisions (mode enforcement)
    3. For each accepted symbol:
       a. Query signal_cache for latest signals
       b. Evaluate signal confidence & timing
       c. Apply position rules (max_pos, dust handling)
       d. Apply capital rules (budget available)
       e. Arbitrate conflicting signals (multiple agents)
       f. Build decision tuple
    4. Rank by priority/confidence
    5. Return top decisions
    """
    
    self.logger.warning("[Meta:TRACE] Enter _build_decisions accepted_symbols=%s", 
                        accepted_symbols_set)
    
    # STEP 0: Governance (Mode enforcement, blockers)
    is_flat = await self._check_portfolio_flat()
    gov_decision = self._get_governance_decision(is_flat, bootstrap_override=False)
    self._emit_governance_decision(gov_decision)
    
    if gov_decision["mode"] == "PAUSED":
        self.logger.info("[Meta:PAUSED] Blocking ALL trading activity")
        return []
    
    allowed_actions = gov_decision["allowed_actions"]  # BUY/SELL/HOLD/etc
    
    # STEP 1: Portfolio Analysis
    total_pos, sig_pos, dust_pos = await self._count_significant_positions()
    
    # STEP 2: Capital Floor Check (early gate)
    capital_ok = await self._check_capital_floor_central()
    capital_block = not capital_ok
    
    if not capital_ok:
        self.logger.warning("[Meta] CAPITAL_FLOOR_VIOLATION - BUYs blocked")
        # → SELLs still allowed for position cleanup
    
    # STEP 3: Build Decision List (per accepted symbol)
    decisions = []
    
    for symbol in accepted_symbols_set:
        # A. Query cached signals for this symbol
        cached_signals = self.signal_manager.get_signals_for_symbol(symbol)
        
        if not cached_signals:
            continue  # No signals for this symbol
        
        # B. Evaluate freshest signal
        sig = cached_signals[0]  # Most recent first
        action = sig.get("action", "HOLD")
        confidence = sig.get("confidence", 0.0)
        agent = sig.get("agent", "Unknown")
        reason = sig.get("reason", "unknown")
        
        # C. Apply position constraints
        if action == "BUY":
            # Check max open positions constraint
            if sig_pos >= self._get_max_positions():
                continue  # Position cap reached
            
            # Check capital constraint
            if capital_block:
                continue  # BUYs blocked (capital floor)
        
        elif action == "SELL":
            # Validate position exists
            if symbol not in current_positions:
                continue  # Nothing to sell
        
        # D. Arbitrate conflicting signals
        # (If multiple agents signal same symbol, apply tie-breaking rules)
        arbitrated_action, arbitrated_conf = await self._arbitrate_signals(
            symbol, cached_signals, allowed_actions
        )
        
        # E. Apply confidence gate
        min_confidence = self._get_min_confidence_for_action(arbitrated_action)
        if arbitrated_conf < min_confidence:
            continue  # Confidence too low
        
        # F. Build decision tuple
        decision_sig = {
            "symbol": symbol,
            "action": arbitrated_action,
            "confidence": arbitrated_conf,
            "agent": agent,
            "reason": reason,
            "_planned_quote": sig.get("_planned_quote", 0.0),
            "trace_id": sig.get("trace_id", str(uuid.uuid4())),
            ...
        }
        
        decisions.append((symbol, arbitrated_action, decision_sig))
    
    # STEP 4: Rank Decisions
    decisions.sort(
        key=lambda d: (
            # Higher confidence first
            -d[2].get("confidence", 0.0),
            # Prefer urgency signals (liquidation, forced exit)
            -int(d[2].get("_forced_exit", False)),
            # FIFO for same confidence
            d[2].get("timestamp", float('inf'))
        )
    )
    
    self.logger.warning("[Meta:POST_BUILD] decisions_count=%d", len(decisions))
    
    return decisions
```

### Key Data Structures

**Input:** `accepted_symbols_set`
```python
{"BTCUSDT", "ETHUSDT", "BNBUSDT", ...}  # Tradable symbols from universe
```

**Output:** `List[Tuple[str, str, Dict]]`
```python
[
    ("BTCUSDT", "BUY", {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "confidence": 0.85,
        "agent": "VolatilityAgent",
        "reason": "volatility_spike",
        "_planned_quote": 1000.0,
        "trace_id": "dec_abc123",
        ...
    }),
    ("ETHUSDT", "SELL", {...}),
    ...
]
```

---

## 7. META → Execution Attempt

### Entry Point
**File:** `core/meta_controller.py`  
**Method:** `evaluate_and_act()` continuation (line 5900+)

### Execution Flow

```python
# meta_controller.py:5900-6200
async def evaluate_and_act(self):
    # ... (previous steps: drain, build_decisions) ...
    
    decisions = await self._build_decisions(accepted_symbols_set)
    
    # STEP 4: Readiness Gating
    snap = await self._readiness_snapshot()
    gated = not snap.get("ops_plane_ready", True)
    
    # STEP 5: Execution Pipeline
    if not decisions:
        self.logger.debug("[Meta] No decisions to execute")
        return
    
    # Process decisions in priority order
    execution_count = 0
    for symbol, side, decision_sig in decisions:
        try:
            # Execute single decision
            res = await self._execute_decision(
                symbol,
                side,
                decision_sig,
                accepted_symbols_set
            )
            
            # Evaluate result
            status = "FAILED"
            if isinstance(res, dict):
                status = str(res.get("status", "FAILED")).upper()
            elif res is True:
                status = "FILLED"
            
            if status in ("FILLED", "PLACED", "EXECUTED"):
                execution_count += 1
                self.logger.info("[Meta:EXEC] ✓ %s %s executed", symbol, side)
            else:
                self.logger.warning("[Meta:EXEC] ✗ %s %s failed: %s", symbol, side, status)
        
        except Exception as e:
            self.logger.error("[Meta:EXEC] Execution error for %s: %s", symbol, e)
    
    # STEP 6: Emit Summary
    self._emit_loop_summary()
```

### _execute_decision Method

**File:** `core/meta_controller.py`  
**Method:** `_execute_decision()` (line ~13000)

```python
async def _execute_decision(self, symbol: str, side: str, signal: dict, 
                            accepted_symbols_set: set) -> Union[bool, Dict]:
    """
    Execute a single decision.
    
    FLOW:
    1. Validate decision preconditions
    2. Compute order parameters (price, quantity, quote)
    3. Check pre-execution safety gates
    4. Call execution_manager.execute(...)
    5. Track result + update state
    """
    
    # Validate
    if not symbol or not side:
        return False
    
    # Get order parameters
    price = await self._get_reference_price(symbol)
    planned_quote = signal.get("_planned_quote", 0.0)
    
    # Compute quantity from quote + price
    quantity = planned_quote / price if price > 0 else 0.0
    
    # Safety gates (min notional, balance, etc)
    if not await self._check_pre_execution_safety(symbol, side, quantity, planned_quote):
        self.logger.warning("[Meta:EXEC] Safety gate rejected %s %s", symbol, side)
        return False
    
    # Emit intent event
    if hasattr(self.shared_state, "emit_event"):
        await self.shared_state.emit_event("TradeIntent", {
            "symbol": symbol,
            "action": side,
            "quantity": quantity,
            "price": price,
            "quote": planned_quote,
            "confidence": signal.get("confidence", 0.0),
            "agent": signal.get("agent"),
            "reason": signal.get("reason"),
            "trace_id": signal.get("trace_id"),
            "ts": time.time()
        })
    
    # Execute via ExecutionManager
    try:
        result = await self.execution_manager.execute(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type="LIMIT",  # or MARKET depending on config
            metadata={
                "agent": signal.get("agent"),
                "reason": signal.get("reason"),
                "trace_id": signal.get("trace_id"),
                "confidence": signal.get("confidence"),
            }
        )
        
        return result
    
    except ExecutionError as e:
        self.logger.error("[Meta:EXEC] Execution error: %s", e)
        return {"status": "FAILED", "error": str(e)}
    except Exception as e:
        self.logger.error("[Meta:EXEC] Unexpected error: %s", e)
        return {"status": "FAILED", "error": str(e)}
```

---

## Complete Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ STRATEGY AGENT (External)                                            │
│ └─ generate_signals() → raw_signal                                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ AGENT MANAGER                                                        │
│ ├─ collect_and_forward_signals()                                    │
│ │  └─ _normalize_to_intents(raw_signal)                             │
│ │     → [TradeIntent, TradeIntent, ...]                             │
│ └─ submit_trade_intents(intents)                                    │
│    └─ event_bus.publish("events.trade.intent", intent)              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ EVENT BUS (SharedState)                                              │
│ ├─ Topic: "events.trade.intent"                                     │
│ └─ Queue: _trade_intent_event_queue (asyncio.Queue)                 │
│    ├─ Subscribers: [MetaController, ...]                            │
│    └─ Events: {name, timestamp, data: TradeIntent}                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ META CONTROLLER                                                      │
│ ├─ evaluate_and_act() [lifecycle loop]                              │
│ │  ├─ _drain_trade_intent_events()                                  │
│ │  │  └─ q.get_nowait() → event                                     │
│ │  │     └─ intent_manager.receive_intents([...])                   │
│ │  │        └─ [SIGNAL CACHE] signal_manager.store(symbol, signal)  │
│ │  │                                                                │
│ │  ├─ _build_decisions(accepted_symbols_set)                        │
│ │  │  ├─ For each symbol in accepted_symbols_set:                  │
│ │  │  │  ├─ signal_manager.get_signals_for_symbol(symbol)          │
│ │  │  │  │  └─ [SIGNAL CACHE] cached_signals                       │
│ │  │  │  ├─ Evaluate signal confidence, timing, constraints        │
│ │  │  │  ├─ Arbitrate conflicting signals (multi-agent)            │
│ │  │  │  ├─ Apply governance rules (mode, allowed_actions)         │
│ │  │  │  ├─ Apply capital rules (budget, floor)                    │
│ │  │  │  ├─ Apply position rules (max_pos, dust)                   │
│ │  │  │  └─ Build decision tuple: (symbol, side, decision_sig)     │
│ │  │  └─ Return: [(symbol, side, signal), ...]                     │
│ │  │                                                                │
│ │  ├─ _execute_decision(symbol, side, signal)                      │
│ │  │  ├─ Compute order params (price, qty, quote)                  │
│ │  │  ├─ Run pre-execution safety checks                           │
│ │  │  ├─ execution_manager.execute(...)                            │
│ │  │  │  ├─ Order validation                                       │
│ │  │  │  ├─ Exchange API call (async)                              │
│ │  │  │  ├─ Order tracking                                         │
│ │  │  │  └─ Return: execution result                               │
│ │  │  └─ Return: {status, ...}                                     │
│ │  │                                                                │
│ │  └─ _emit_loop_summary()                                          │
│ │     └─ Log: [LOOP_SUMMARY] tick metrics                           │
│ │                                                                   │
│ └─ run() [starts evaluate_and_act in loop]                          │
│    └─ await asyncio.sleep(interval_sec)                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Signal Gen** | Market data, Config | `generate_signals()` | Raw signal dict |
| **Normalization** | Raw signal | `_normalize_to_intents()` | Canonical `TradeIntent` |
| **Event Bus** | `TradeIntent` | `publish()` → Queue | Event in `_trade_intent_event_queue` |
| **Drain** | Queue | `get_nowait()` loop | List of normalized intents |
| **Signal Cache** | Intents | `intent_manager.receive_intents()` | Per-symbol cached signal list |
| **Arbitration** | Cached signals, Universe | `_build_decisions()` | Ranked decision list |
| **Execution** | Decision tuple | `_execute_decision()` | Order result / status |

---

## Key Timing & Lifecycle

```
┌─ start() → starts run() loop
│
├─ run() loop [every interval_sec]
│  └─ evaluate_and_act()
│     ├─ _drain_trade_intent_events()  [SIGNAL INTAKE]
│     ├─ _build_decisions()             [ARBITRATION]
│     ├─ _execute_decision()            [EXECUTION]
│     └─ _emit_loop_summary()           [TELEMETRY]
│
└─ tick_id increments every cycle
```

**Typical Interval:** 2.0 seconds (configurable via `META_TICK_INTERVAL_SEC`)

---

## Error Handling & Robustness

### Event Bus Failures
```python
# If event_bus not available:
- Log warning
- Skip drain → no intents received
- Signals from signal_manager cache used if available

# If subscribe fails:
- Log warning
- Return False from _ensure_trade_intent_subscription()
- Subsequent drains return 0
```

### Decision Failures
```python
# Per-symbol failures:
- Log warning for each rejection (capital block, position cap, etc)
- Continue with next symbol

# Execution failures:
- Log error
- Return failure dict
- Emit [EXEC_REJECT] event for monitoring
- Capital/position reserved, not committed
```

### Signal Ingestion Failures
```python
# Event malformed:
- q.task_done() called regardless
- Event skipped, queue advances
- No halt of entire pipeline

# Normalization fails:
- Skip intent
- Log warning
- Continue with next event
```

---

## Debugging Hooks

### Critical Log Lines (Searchable)

| Log Pattern | Location | Meaning |
|---|---|---|
| `[Meta:DRAIN] ⚠️ DRAINED %d events` | evaluate_and_act | Events ingested from bus |
| `[Meta:TRACE] Enter _build_decisions` | _build_decisions | Arbitration starting |
| `[Meta:POST_BUILD] decisions_count=%d` | evaluate_and_act | Decisions ready for execution |
| `[Meta:EXEC] ✓ %s %s executed` | _execute_decision | Trade successfully placed |
| `[Meta:EXEC] ✗ %s %s failed` | _execute_decision | Trade rejected |
| `[LOOP_SUMMARY]` | _emit_loop_summary | End-of-tick metrics |

### Tracing IDs

Each decision carries unique trace IDs:
```python
"trace_id": signal.get("trace_id", str(uuid.uuid4()))
```

Use `trace_id` to correlate:
- Signal → Decision → Execution → Order Status

---

## Configuration Parameters

| Parameter | File | Default | Purpose |
|---|---|---|---|
| `META_TICK_INTERVAL_SEC` | config | 2.0 | Cycle interval |
| `TRADE_INTENT_EVENT_DRAIN_MAX` | config | 1000 | Max events per drain |
| `CAPITAL_FLOOR_PCT` | config | 0.20 | Min capital as % of NAV |
| `ABSOLUTE_MIN_FLOOR` | config | 10.0 | Absolute min capital floor |
| `MIN_CONFIDENCE` | signal_manager | varies | Min signal confidence to act |

---

## Summary

The signal pipeline is a clean, event-driven architecture:

1. **Generation:** Strategy agents generate raw signals (external)
2. **Normalization:** Converted to canonical `TradeIntent` format
3. **Transport:** Published to event bus topic `events.trade.intent`
4. **Intake:** MetaController drains queue in bulk (non-blocking)
5. **Caching:** Signals stored per-symbol in `signal_manager`
6. **Arbitration:** Multi-symbol decision-building with governance/capital/position constraints
7. **Execution:** Ranked decisions executed in priority order via ExecutionManager

Each stage includes comprehensive error handling, logging, and trace IDs for debuggability.
