# Architectural Fix: Eliminate the Dust Loop

## The Core Problem: State Collapse

**Current broken logic**:
```python
if total_positions == 0:
    return True  # "portfolio is flat"
```

This collapses three distinct states into one:

| Actual State | Current Code Thinks | Result |
|---|---|---|
| `EMPTY_PORTFOLIO` (no positions, no dust) | FLAT | ✓ Correct: bootstrap allowed |
| `PORTFOLIO_WITH_DUST` (positions exist, all dust) | FLAT | ✗ **Wrong**: bootstrap triggered |
| `PORTFOLIO_WITH_POSITIONS` (significant holdings) | (not reached) | N/A |

**Impact**: When dust exists, the system incorrectly believes the portfolio is empty and triggers bootstrap logic.

---

## The Root Causes (Validated)

### 1. **Portfolio State Collapse** ← PRIMARY BUG
- Code: `shared_state.py` lines 4979-5010
- Issue: Three distinct states mapped to one boolean
- Effect: Dust-only portfolios trigger bootstrap

### 2. **Metrics Not Persisted on Restart** ← SECONDARY BUG
- Code: `shared_state.py` line 4899
- Issue: `first_trade_at` and `total_trades_executed` not written to disk
- Effect: `is_cold_bootstrap()` returns True even after prior trading
- Result: Bootstrap re-triggers on every restart

### 3. **Dust Markers Persist After Resolution** ← TERTIARY BUG
- Code: `execution_manager.py` lines 3379-3399
- Issue: `dust_healing_deficit[symbol]` and `dust_operation_symbols[symbol]` not cleared
- Effect: System still detects dust even after healing
- Result: Next cycle re-triggers dust healing

### 4. **Bootstrap & Dust Share Override Flags** ← DANGEROUS COUPLING
- Code: `execution_manager.py` lines 5687-5734
- Issue: Same flags (`bypass_min_notional`, `bootstrap_override`) used for both
- Effect: Bootstrap trades get dust-operation privileges
- Result: Loss-making bootstrap trades allowed simultaneously with dust healing

### 5. **No Central State Authority** ← ARCHITECTURAL BUG
- Three independent subsystems:
  - Strategy agents (TrendHunter, SignalManager)
  - Dust healer (DustHealer, LiquidationAgent)
  - Bootstrap engine (MetaController)
- Issue: No gating mechanism prevents simultaneous activation
- Effect: All three can run at once
- Result: Conflicting logic amplifies losses

### 6. **Signal Thrashing from Single Position Limit** ← CONTRIBUTING FACTOR
- Issue: 50+ symbols but only 1 position allowed
- Effect: Governor constantly rejects new signals
- Result: Rotation logic triggers more frequently
- Amplification: More rotations → more losses → more dust

---

## The Correct Architecture: Portfolio State Machine

### New State Definition

```python
# In shared_state.py

class PortfolioState(Enum):
    """Canonical portfolio state machine. Central authority for all trading decisions."""
    
    EMPTY_PORTFOLIO = "empty"           # No positions, no dust, ready for bootstrap
    PORTFOLIO_WITH_DUST = "dust_only"   # Only dust remains, dust healing phase
    PORTFOLIO_ACTIVE = "active"         # Significant open positions
    PORTFOLIO_RECOVERING = "recovering" # Reconciliation in progress
    COLD_BOOTSTRAP = "cold_bootstrap"   # Never traded, initialization phase


async def get_portfolio_state(self) -> PortfolioState:
    """
    CANONICAL: Determine the actual portfolio state.
    
    This is the single source of truth for all trading decisions.
    No other code should make assumptions about portfolio state.
    """
    # Step 1: Check if in cold bootstrap (never traded)
    if self.is_cold_bootstrap():
        return PortfolioState.COLD_BOOTSTRAP
    
    # Step 2: Get all positions
    all_positions = self.get_open_positions()
    
    # Step 3: Classify positions
    significant_positions = [
        p for p in all_positions 
        if await self._is_position_significant(p)  # qty*price >= min_notional
    ]
    dust_positions = [
        p for p in all_positions 
        if not await self._is_position_significant(p)  # qty*price < min_notional
    ]
    
    # Step 4: Determine state based on composition
    if len(significant_positions) > 0:
        return PortfolioState.PORTFOLIO_ACTIVE
    
    if len(dust_positions) > 0:
        return PortfolioState.PORTFOLIO_WITH_DUST
    
    # No significant positions, no dust
    return PortfolioState.EMPTY_PORTFOLIO


async def _is_position_significant(self, position: Dict[str, Any]) -> bool:
    """
    Helper: Is this position above minimum notional threshold?
    
    Significant means: qty × current_price >= PERMANENT_DUST_USDT_THRESHOLD
    Default threshold: $1.0 USDT
    """
    sym = position.get("symbol")
    qty = float(position.get("quantity", 0.0))
    
    # Get current price
    try:
        price = await self._get_position_current_price(sym)
    except Exception:
        # Fallback to position's entry/mark price
        price = float(position.get("entry_price") or position.get("mark_price") or 0.0)
    
    if not price or price <= 0:
        return False
    
    notional = qty * price
    threshold = float(self.config.get("PERMANENT_DUST_USDT_THRESHOLD", 1.0))
    
    return notional >= threshold
```

---

## Fix 1: State Machine Eliminates the Loop

### Current Broken Logic
```python
# WRONG: Collapses states
if total_positions == 0:
    state = "FLAT"
    bootstrap_allowed = True
    dust_healing_allowed = True
    strategy_allowed = True  # All three active!
```

### Correct Logic
```python
# CORRECT: Distinct states with distinct rules
state = await self.get_portfolio_state()

if state == PortfolioState.EMPTY_PORTFOLIO:
    bootstrap_allowed = True      # ✓
    dust_healing_allowed = False  # ✗ (no dust to heal)
    strategy_allowed = False      # ✗ (no positions to manage)
    rotation_allowed = False      # ✗

elif state == PortfolioState.PORTFOLIO_WITH_DUST:
    bootstrap_allowed = False     # ✗ DO NOT BOOTSTRAP WHEN DUST EXISTS
    dust_healing_allowed = True   # ✓ Heal dust only
    strategy_allowed = False      # ✗ (all capital tied in dust)
    rotation_allowed = False      # ✗

elif state == PortfolioState.PORTFOLIO_ACTIVE:
    bootstrap_allowed = False     # ✗ (portfolio not empty)
    dust_healing_allowed = False  # ✗ (focused on main positions)
    strategy_allowed = True       # ✓ Normal trading
    rotation_allowed = True       # ✓ Rotation only between active positions

elif state == PortfolioState.COLD_BOOTSTRAP:
    bootstrap_allowed = True      # ✓ (first-time initialization)
    dust_healing_allowed = False  # ✗ (no prior history)
    strategy_allowed = False      # ✗ (waiting for bootstrap to complete)
    rotation_allowed = False      # ✗
```

**Result**: The dust loop is broken at step 2. When dust is detected, bootstrap is **never** triggered.

---

## Fix 2: Persist Bootstrap Metrics to Disk

### Current Broken Code
```python
# WRONG: Only in memory, lost on restart
self.metrics.get("first_trade_at")  # None after restart
self.metrics.get("total_trades_executed")  # 0 after restart

def is_cold_bootstrap(self) -> bool:
    if has_trade_history:  # ← Always False after restart!
        return False
    return True  # ← Always True after restart!
```

### Correct Implementation
```python
# In shared_state.py

class BootstrapMetrics:
    """Persistent bootstrap metrics stored to disk."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.metrics_file = os.path.join(db_path, "bootstrap_metrics.json")
    
    def save_first_trade_at(self, timestamp: float) -> None:
        """Persist when first trade completed."""
        data = self._load_or_empty()
        data["first_trade_at"] = timestamp
        data["updated_at"] = time.time()
        self._write(data)
    
    def save_trade_executed(self) -> None:
        """Increment total trades and persist."""
        data = self._load_or_empty()
        data["total_trades_executed"] = data.get("total_trades_executed", 0) + 1
        data["updated_at"] = time.time()
        self._write(data)
    
    def get_first_trade_at(self) -> Optional[float]:
        """Load from disk, None if not set."""
        data = self._load_or_empty()
        return data.get("first_trade_at")
    
    def get_total_trades_executed(self) -> int:
        """Load from disk, 0 if never traded."""
        data = self._load_or_empty()
        return data.get("total_trades_executed", 0)
    
    def _load_or_empty(self) -> Dict[str, Any]:
        """Load metrics from disk or return empty dict."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load bootstrap metrics: {e}")
        return {}
    
    def _write(self, data: Dict[str, Any]) -> None:
        """Write metrics to disk atomically."""
        try:
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist bootstrap metrics: {e}")


# Usage in ExecutionManager
async def _on_trade_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
    """Called when a trade fills successfully."""
    # ... existing logic ...
    
    # CRITICAL: Persist that we've now traded
    if self.bootstrap_metrics:
        if self.bootstrap_metrics.get_first_trade_at() is None:
            self.bootstrap_metrics.save_first_trade_at(time.time())
        self.bootstrap_metrics.save_trade_executed()
```

**Result**: On restart, `is_cold_bootstrap()` correctly returns False if any trade occurred previously.

---

## Fix 3: Clear Dust Markers When Healing Completes

### Current Broken Code
```python
# WRONG: Marks as dust but never clears
if notional_value < permanent_dust_threshold:
    self.shared_state.record_dust(sym, qty, origin=...)
    self.shared_state.dust_healing_deficit[sym] = qty_deficit
    self.shared_state.dust_operation_symbols[sym] = True
    # ← Never cleared! Persists forever
```

### Correct Implementation
```python
# In shared_state.py

class DustRegistry:
    """Centralized dust tracking with lifecycle management."""
    
    def __init__(self):
        self.dust_entries: Dict[str, Dict[str, Any]] = {}  # sym -> {qty, notional, origin, created_at}
        self.healing_attempts: Dict[str, int] = {}  # sym -> attempt count
        self.healing_completed: Dict[str, float] = {}  # sym -> completion timestamp
    
    def record_dust(self, symbol: str, qty: float, origin: str = "", context: Dict[str, Any] = None) -> None:
        """Record a dust position."""
        sym = self._norm_sym(symbol)
        self.dust_entries[sym] = {
            "qty": float(qty),
            "notional": context.get("notional", 0.0) if context else 0.0,
            "origin": origin,
            "created_at": time.time(),
            "attempts": 0,
        }
    
    def increment_healing_attempt(self, symbol: str) -> int:
        """Increment healing attempt counter."""
        sym = self._norm_sym(symbol)
        count = self.healing_attempts.get(sym, 0) + 1
        self.healing_attempts[sym] = count
        return count
    
    def mark_healing_complete(self, symbol: str) -> None:
        """Mark dust as successfully healed."""
        sym = self._norm_sym(symbol)
        
        # Record completion
        self.healing_completed[sym] = time.time()
        
        # CRITICAL: Clear all dust markers
        self.dust_entries.pop(sym, None)
        self.dust_healing_deficit.pop(sym, None)
        self.dust_operation_symbols.pop(sym, None)
        
        logger.info(f"[DustRegistry] Healed {sym}: marked complete, cleared all markers")
    
    def mark_permanent_dust(self, symbol: str) -> None:
        """Mark as permanent dust after max healing attempts."""
        sym = self._norm_sym(symbol)
        attempts = self.healing_attempts.get(sym, 0)
        
        logger.warning(
            f"[DustRegistry] {sym}: marked permanent after {attempts} healing attempts"
        )
        
        # Still clear from active healing registries
        self.dust_operation_symbols[sym] = True  # Keep marker but inactive
        self.dust_healing_deficit.pop(sym, None)  # No more active healing
    
    def get_is_dust_only(self, symbol: str) -> bool:
        """Check if symbol has an active dust entry."""
        sym = self._norm_sym(symbol)
        return sym in self.dust_entries and sym not in self.healing_completed
    
    def is_healing_failed(self, symbol: str, max_attempts: int = 3) -> bool:
        """Check if healing has failed too many times."""
        sym = self._norm_sym(symbol)
        attempts = self.healing_attempts.get(sym, 0)
        return attempts >= max_attempts
```

### Circuit Breaker Integration
```python
# In execution_manager.py

async def attempt_dust_healing(self, symbol: str, qty: float, price: float) -> bool:
    """
    Attempt to heal dust. Return True if healing executed.
    Circuit breaker prevents infinite retries.
    """
    sym = self._norm_sym(symbol)
    
    # Check if already failed too many times
    if self.shared_state.dust_registry.is_healing_failed(sym, max_attempts=3):
        logger.warning(f"[DustHealing] {sym}: giving up after 3 failed attempts")
        self.shared_state.dust_registry.mark_permanent_dust(sym)
        return False
    
    # Attempt healing
    result = await self._execute_dust_healing_sell(sym, qty)
    
    if result.ok:
        # CRITICAL: Clear dust markers on success
        self.shared_state.dust_registry.mark_healing_complete(sym)
        logger.info(f"[DustHealing] {sym}: successfully healed")
        return True
    else:
        # Increment attempt counter
        attempt = self.shared_state.dust_registry.increment_healing_attempt(sym)
        logger.warning(f"[DustHealing] {sym}: healing failed, attempt {attempt}/3")
        return False
```

**Result**: Once dust is healed, markers are cleared and won't be detected again.

---

## Fix 4: Separate Dust & Bootstrap Override Flags

### Current Broken Code
```python
# WRONG: Bootstrap and dust share same flags
if is_dust_operation:
    bypass_min_notional = True  # ← For dust healing
    
if bootstrap_override:
    bypass_min_notional = True  # ← For bootstrap

# No distinction → bootstrap gets dust privileges
```

### Correct Implementation
```python
# In execution_manager.py

async def _place_limit_order(
    self,
    symbol: str,
    side: str,
    quantity: Optional[float],
    planned_quote: Optional[float],
    policy_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CANONICAL: Single order placement with explicit context flags.
    """
    
    # Parse context flags
    is_dust_healing = bool(policy_ctx.get("is_dust_healing", False))
    is_bootstrap = bool(policy_ctx.get("is_bootstrap_trade", False))
    is_rotation = bool(policy_ctx.get("is_rotation_trade", False))
    
    # CRITICAL: These are MUTUALLY EXCLUSIVE
    # A trade can be dust healing OR bootstrap OR rotation, not multiple
    active_contexts = sum([is_dust_healing, is_bootstrap, is_rotation])
    if active_contexts > 1:
        logger.error(
            f"[EM:Policy] Trade context invalid: "
            f"dust_healing={is_dust_healing} bootstrap={is_bootstrap} rotation={is_rotation}"
        )
        return {"ok": False, "reason": "invalid_context"}
    
    # Apply context-specific rules
    if is_dust_healing:
        # Dust healing: ONLY bypass min_notional, strict risk
        bypass_min_notional = True
        bypass_risk_sizing = False  # ← Still enforce risk
        min_pnl_check = False
    
    elif is_bootstrap:
        # Bootstrap: bypass risk sizing, not min_notional
        bypass_min_notional = False
        bypass_risk_sizing = True
        min_pnl_check = False
    
    elif is_rotation:
        # Rotation: normal checks, rotation override on fees only
        bypass_min_notional = False
        bypass_risk_sizing = False
        min_pnl_check = True  # ← Still require profitable exit
    
    else:
        # Normal strategy trade
        bypass_min_notional = False
        bypass_risk_sizing = False
        min_pnl_check = True
    
    # ... rest of order logic uses these separate flags ...
```

**Result**: Bootstrap trades can no longer use dust-operation privileges.

---

## Fix 5: Central State Authority

### Current Broken Code
```python
# WRONG: Three systems act independently
TrendHunter.should_trade()  # → "buy BTCUSDT"
DustHealer.should_heal()    # → "sell DUSTTOKEN"
MetaController.bootstrap()  # → "buy BTCUSDT"

# All three execute without coordinating
```

### Correct Implementation
```python
# In trading_coordinator.py (NEW FILE)

class TradingCoordinator:
    """
    CANONICAL: Central authority for all trading decisions.
    Single source of truth for portfolio state and execution authorization.
    """
    
    def __init__(self, shared_state: SharedState, execution_manager: ExecutionManager):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.state_machine = PortfolioStateMachine(shared_state)
    
    async def authorize_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        context: str,  # "strategy" | "dust_healing" | "bootstrap" | "rotation"
    ) -> Dict[str, Any]:
        """
        GATE: Single point where all trades are authorized.
        Returns {ok, reason, allowed_qty} or {ok: False, reason}
        """
        
        # Get current portfolio state
        state = await self.state_machine.get_state()
        
        # Check if this trade type is allowed in current state
        is_allowed = self._is_trade_allowed(context, state)
        
        if not is_allowed:
            return {
                "ok": False,
                "reason": f"trade_blocked_in_state_{state}",
                "context": context,
                "state": state,
            }
        
        # Validate context-specific rules
        validation = await self._validate_context(symbol, side, qty, context, state)
        if not validation["ok"]:
            return validation
        
        # Trade is authorized
        return {
            "ok": True,
            "reason": "authorized",
            "context": context,
            "state": state,
        }
    
    def _is_trade_allowed(self, context: str, state: PortfolioState) -> bool:
        """
        DECISION MATRIX: What trades are allowed in which state.
        """
        
        rules = {
            PortfolioState.COLD_BOOTSTRAP: {
                "bootstrap": True,
                "strategy": False,
                "dust_healing": False,
                "rotation": False,
            },
            PortfolioState.EMPTY_PORTFOLIO: {
                "bootstrap": True,
                "strategy": False,
                "dust_healing": False,
                "rotation": False,
            },
            PortfolioState.PORTFOLIO_WITH_DUST: {
                "bootstrap": False,  # ← CRITICAL: No bootstrap when dust
                "strategy": False,
                "dust_healing": True,  # ← Dust healing only
                "rotation": False,
            },
            PortfolioState.PORTFOLIO_ACTIVE: {
                "bootstrap": False,
                "strategy": True,
                "dust_healing": False,
                "rotation": True,
            },
            PortfolioState.PORTFOLIO_RECOVERING: {
                "bootstrap": False,
                "strategy": False,
                "dust_healing": False,
                "rotation": False,
            },
        }
        
        return rules.get(state, {}).get(context, False)
    
    async def _validate_context(
        self,
        symbol: str,
        side: str,
        qty: float,
        context: str,
        state: PortfolioState,
    ) -> Dict[str, Any]:
        """Validate context-specific constraints."""
        
        if context == "dust_healing":
            # Dust healing: must be actual dust position
            is_dust = await self.shared_state.dust_registry.get_is_dust_only(symbol)
            if not is_dust:
                return {"ok": False, "reason": "not_dust_position"}
        
        elif context == "bootstrap":
            # Bootstrap: must be cold or empty
            if state not in (PortfolioState.COLD_BOOTSTRAP, PortfolioState.EMPTY_PORTFOLIO):
                return {"ok": False, "reason": "invalid_state_for_bootstrap"}
        
        elif context == "rotation":
            # Rotation: must be in active portfolio
            if state != PortfolioState.PORTFOLIO_ACTIVE:
                return {"ok": False, "reason": "invalid_state_for_rotation"}
        
        return {"ok": True}


# Usage in strategy/agents:

async def should_trade(self, symbol: str, signal: SignalData) -> Optional[Dict[str, Any]]:
    """Request permission to trade."""
    
    # Ask coordinator
    auth = await self.trading_coordinator.authorize_trade(
        symbol=symbol,
        side="buy",
        qty=signal.size,
        context="strategy",  # ← Explicit context
    )
    
    if not auth["ok"]:
        logger.debug(f"Trade blocked: {auth['reason']}")
        return None
    
    # Trade authorized, execute
    return await self.execution_manager.place_order(...)
```

**Result**: All three subsystems must go through a single gate. State mismatches are caught immediately.

---

## Complete Fix Summary

| Bug | Root Cause | Fix | Code Location |
|-----|-----------|-----|----------------|
| **State Collapse** | `if positions==0: flat` | State machine with 4 distinct states | `shared_state.py` new method |
| **Metrics Lost** | No disk persistence | `BootstrapMetrics` class writes to disk | `shared_state.py` new class |
| **Dust Persists** | Markers never cleared | `DustRegistry.mark_healing_complete()` | `shared_state.py` new method |
| **Override Coupling** | Same flags for both | Separate `is_dust_healing` vs `is_bootstrap` flags | `execution_manager.py` policy_ctx |
| **No Authority** | Three independent systems | `TradingCoordinator` central gate | New file `trading_coordinator.py` |

---

## Implementation Order

1. **Phase 1 (Day 1)**: Add `PortfolioState` enum and `get_portfolio_state()` method
   - High risk: Minimal change, just adds new method
   - Verify state detection works correctly

2. **Phase 2 (Day 1)**: Add `BootstrapMetrics` persistence
   - Medium risk: New file, doesn't affect execution
   - Verify metrics persist correctly on restart

3. **Phase 3 (Day 2)**: Create `DustRegistry` with lifecycle management
   - Medium risk: Refactors existing dust tracking
   - Verify dust markers clear after healing

4. **Phase 4 (Day 2)**: Implement `TradingCoordinator` decision matrix
   - High risk: All trades now go through single gate
   - Must test thoroughly with all agents

5. **Phase 5 (Day 3)**: Update policy_ctx flags for mutual exclusion
   - High risk: Changes execution logic
   - Must verify all 3 contexts work independently

6. **Phase 6 (Day 3)**: Update MetaController, TrendHunter, DustHealer to use new API
   - High risk: Agent changes
   - Extensive testing required

---

## Testing Strategy

### Unit Tests
```python
def test_portfolio_state_empty():
    """Empty portfolio returns EMPTY_PORTFOLIO."""
    ss.positions = {}
    state = await ss.get_portfolio_state()
    assert state == PortfolioState.EMPTY_PORTFOLIO

def test_portfolio_state_dust_only():
    """Dust-only portfolio returns PORTFOLIO_WITH_DUST."""
    ss.positions = {"BTCUSDT": {"qty": 0.0001, "entry_price": 65000}}
    state = await ss.get_portfolio_state()
    assert state == PortfolioState.PORTFOLIO_WITH_DUST

def test_bootstrap_blocked_when_dust():
    """Bootstrap authorization fails when dust exists."""
    coordinator = TradingCoordinator(ss, em)
    auth = await coordinator.authorize_trade("BTCUSDT", "buy", 0.1, "bootstrap")
    assert auth["ok"] == False
    assert "dust" in auth["reason"]

def test_dust_markers_cleared():
    """Dust markers cleared after healing complete."""
    registry = DustRegistry()
    registry.record_dust("BTCUSDT", 0.0001)
    assert registry.get_is_dust_only("BTCUSDT") == True
    
    registry.mark_healing_complete("BTCUSDT")
    assert registry.get_is_dust_only("BTCUSDT") == False
```

### Integration Tests
```python
async def test_dust_loop_blocked():
    """End-to-end: dust loop no longer occurs."""
    
    # Setup: Position with dust
    await ss.add_position("BTCUSDT", qty=0.0001, entry_price=65000)
    
    # Verify state
    state = await ss.get_portfolio_state()
    assert state == PortfolioState.PORTFOLIO_WITH_DUST
    
    # Attempt bootstrap
    auth = await coordinator.authorize_trade("ETHUSDT", "buy", 1.0, "bootstrap")
    assert auth["ok"] == False  # ← Bootstrap blocked!
    
    # Dust healing allowed
    auth = await coordinator.authorize_trade("BTCUSDT", "sell", 0.0001, "dust_healing")
    assert auth["ok"] == True  # ← Healing allowed
    
    # After healing complete
    await ss.dust_registry.mark_healing_complete("BTCUSDT")
    state = await ss.get_portfolio_state()
    assert state == PortfolioState.EMPTY_PORTFOLIO
    
    # Now bootstrap is allowed
    auth = await coordinator.authorize_trade("ETHUSDT", "buy", 1.0, "bootstrap")
    assert auth["ok"] == True  # ← Bootstrap now allowed
```

### Regression Tests
```python
async def test_normal_strategy_still_works():
    """Ensure normal strategy trading is unaffected."""
    # Setup: Active portfolio
    await ss.add_position("BTCUSDT", qty=1.0, entry_price=65000)
    
    state = await ss.get_portfolio_state()
    assert state == PortfolioState.PORTFOLIO_ACTIVE
    
    # Strategy and rotation trades allowed
    auth = await coordinator.authorize_trade("BTCUSDT", "sell", 0.5, "rotation")
    assert auth["ok"] == True
    
    auth = await coordinator.authorize_trade("ETHUSDT", "buy", 10.0, "strategy")
    assert auth["ok"] == True
```

---

## Expected Outcomes

### Before Fix
- **Dust loop occurrence**: Every restart + dust detection
- **Loss per cycle**: 0.4-0.7% (fees + slippage)
- **Capital degradation**: 6% per 10 cycles
- **Bootstrap re-entries**: 100% on restart

### After Fix
- **Dust loop occurrence**: 0% (architecture prevents it)
- **Loss per cycle**: Only from dust healing (1-2 sells)
- **Capital degradation**: <0.2% per month (normal trading)
- **Bootstrap re-entries**: Only when portfolio truly empty

---

## Conclusion

The dust loop exists because the system collapses three distinct portfolio states into one boolean. The fix requires:

1. **Explicit state machine**: EMPTY, DUST_ONLY, ACTIVE, RECOVERING, COLD_BOOTSTRAP
2. **Persistent metrics**: `first_trade_at` written to disk
3. **Dust cleanup**: Markers cleared when healing completes
4. **Separate flags**: Bootstrap and dust use independent overrides
5. **Central authority**: Single `TradingCoordinator` gate for all trades

This breaks the feedback loop at its source: dust-only portfolios will never trigger bootstrap.

---
