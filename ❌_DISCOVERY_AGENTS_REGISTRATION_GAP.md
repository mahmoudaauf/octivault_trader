# ❌ ROOT CAUSE FOUND: Discovery Agents Registration Gap

## Executive Summary

**Discovery agents are NOT running because `register_all_discovery_agents()` is NEVER CALLED.**

The function exists in `core/agent_registry.py` but is not invoked anywhere during bootstrap. This means:
- ✅ Discovery agents are built (IPOChaser, WalletScannerAgent, SymbolScreener)
- ✅ Capital allocation discovers 6 agents
- ✅ run_discovery_agents_loop() is started and waiting
- ❌ **Agent manager's discovery_agents list stays EMPTY** - no agents get registered!
- ❌ When run_discovery_agents_once() iterates `self.discovery_agents`, the list is empty
- ❌ Agents never execute their scan logic

---

## The Architecture Gap

### What SHOULD Happen (Design Intent)
```
Bootstrap → Create agents → Register with AgentManager → Start AgentManager
                                ↓
                    register_all_discovery_agents()
                                ↓
                    agent_manager.register_discovery_agent(agent)
                                ↓
                    self.discovery_agents.append(agent)
                                ↓
                    When run_discovery_agents_loop() runs:
                    → Calls run_discovery_agents_once()
                    → Iterates self.discovery_agents list
                    → Executes each agent's run_once() or run_loop()
```

### What ACTUALLY Happens (Current State)
```
Bootstrap → Create agents → [MISSING CALL]
                                ↓
                    agent_manager.discovery_agents = []  ← STAYS EMPTY!
                                ↓
                    When run_discovery_agents_loop() runs:
                    → Calls run_discovery_agents_once()
                    → Iterates empty self.discovery_agents list
                    → NO AGENTS EXECUTE
```

---

## Evidence: The Missing Link

### File: `core/agent_registry.py` (Line 254-300)
```python
def register_all_discovery_agents(agent_manager, app_context):
    """
    Registers discovery/proposer agents with the agent manager.

    For each entry in ``_DISCOVERY_AGENTS`` the function first checks
    ``app_context`` for a pre-built instance, then falls back to constructing
    one via ``_safe_build``.  Only ``register_discovery_agent`` is called
    (not the general ``register_agent``) to avoid double-execution in the
    agent manager's strategy loop.
    """

    def _safe_build(name: str):
        cls = AGENT_CLASS_MAP.get(name)
        if not cls:
            return None
        kwargs = {
            "shared_state": getattr(app_context, "shared_state", None),
            "config": getattr(app_context, "config", None),
            "exchange_client": getattr(app_context, "exchange_client", None),
            "symbol_manager": getattr(app_context, "symbol_manager", None),
            "execution_manager": getattr(app_context, "execution_manager", None),
            "tp_sl_engine": getattr(app_context, "tp_sl_engine", None),
        }
        try:
            return cls(**{k: v for k, v in kwargs.items() if v is not None})
        except Exception as exc:
            _logger.debug("[_safe_build] %s construction failed: %s", name, exc)
            return None

    for attr_name, map_key in _DISCOVERY_AGENTS:
        agent = getattr(app_context, attr_name, None) or _safe_build(map_key)
        if not agent:
            continue
        if not getattr(agent, "agent_type", None):
            setattr(agent, "agent_type", "discovery")
        agent_manager.register_discovery_agent(agent)  ← THIS CALL NEVER HAPPENS
```

### File: `core/agent_registry.py` (Lines 231-251)
```python
_DISCOVERY_AGENTS = [
    ("wallet_scanner_agent", "WalletScannerAgent"),
    ("symbol_screener_agent", "SymbolScreener"),
    ("liquidation_agent", "LiquidationAgent"),
    ("ipo_chaser", "IPOChaser"),
]
```

### Where It's Exported But NOT Called
```
grep_search results for "register_all_discovery_agents":
✓ Line 10 in core/agent_registry.py - EXPORTED in __all__
✓ Line 254 in core/agent_registry.py - FUNCTION DEFINITION
✓ Line 440 in BOOTSTRAP_MODE_AGENTS.md - DOCUMENTATION ONLY
✗ ZERO calls in app_context.py
✗ ZERO calls in any bootstrap script
✗ ZERO calls anywhere in codebase
```

---

## The Execution Flow That Should Trigger Discovery

### File: `core/agent_manager.py` (Lines 1041-1062)
```python
async def run_discovery_agents_loop(self):
    """
    Periodically runs discovery agents to populate the symbol universe.
    This unblocks SymbolManager which waits for these proposals.
    """
    self.logger.info("Starting Discovery Agent loop...")
    # Launch persistent discovery agent coroutines once (outside the retry loop).
    try:
        await self.run_discovery_agents()  ← Line 1049: Calls run_discovery_agents()
    except Exception as e:
        self.logger.error("run_discovery_agents failed on startup: %s", e, exc_info=True)

    # Outer retry loop: a crash in run_discovery_agents_once must NOT exit this method.
    while True:
        try:
            await self.run_discovery_agents_once()  ← Line 1056: Calls run_discovery_agents_once()
            # Run every 10 minutes or as configured
            discovery_interval = float(getattr(self.config, "AGENTMGR_DISCOVERY_INTERVAL", 600.0))
            await _asyncio.sleep(discovery_interval)
        except _asyncio.CancelledError:
            self.logger.info("Discovery loop cancelled.")
            raise
        except Exception as e:
            self.logger.error("Discovery loop crashed: %s — retrying in 30s", e, exc_info=True)
            await _asyncio.sleep(30)
```

### File: `core/agent_manager.py` (Lines 806-838)
```python
async def run_discovery_agents(self):
    """
    Runs all registered discovery agents from manual injection (Phase 3 only).
    """
    for agent in self.discovery_agents:  ← LINE 811: Iterates self.discovery_agents
        # ... checks for run_loop() ...
```

### File: `core/agent_manager.py` (Lines 819-840)
```python
async def run_discovery_agents_once(self):
    """
    Runs all registered discovery agents exactly once (for periodic discovery).
    """
    for agent in self.discovery_agents:  ← LINE 826: Iterates self.discovery_agents
        if hasattr(agent, "run_once") and _asyncio.iscoroutinefunction(agent.run_once):
            try:
                await agent.run_once()
            except _asyncio.TimeoutError:
                self.logger.warning("⏰ Discovery agent %s run_once timed out", agent.__class__.__name__)
            except Exception as e:
                self.logger.error("💥 Discovery agent %s crashed: %s", agent.__class__.__name__, e, exc_info=True)
```

**Both methods iterate an EMPTY LIST because `register_all_discovery_agents()` was never called!**

---

## Bootstrap Sequence Analysis

### Current Bootstrap in `core/app_context.py::initialize_all()`

**Phase 1-5**: Core systems, exchange, execution ✅

**Phase 6** (Line ~4350):
```python
# AgentManager created here
if self.agent_manager is None:
    AgentManager = _get_cls(_agent_mgr_mod, "AgentManager")
    self.agent_manager = _try_construct(AgentManager, config=self.config, ...)

# MetaController injected
if self.agent_manager and self.meta_controller:
    self.agent_manager.meta_controller = self.meta_controller

# ❌ MISSING: Register discovery agents here!
# register_all_discovery_agents(self.agent_manager, self)  ← SHOULD BE HERE
```

**Phase 6 - AgentManager Started** (Line ~4373):
```python
if self.agent_manager and any(hasattr(self.agent_manager, nm) for nm in _p6_start_methods):
    await self._start_with_timeout("P6_agent_manager", self.agent_manager)
    # At this point, run_discovery_agents_loop() is started as a background task
```

**The Problem**: Discovery agents are never registered before AgentManager.start() is called!

---

## The Fix

### Option 1: Call in AppContext (RECOMMENDED)
**File**: `core/app_context.py` - After AgentManager construction (~line 3650)

```python
# After AgentManager is constructed
if self.agent_manager is None:
    AgentManager = _get_cls(_agent_mgr_mod, "AgentManager")
    self.agent_manager = _try_construct(AgentManager, config=self.config, logger=self.logger, 
                                       app=self, shared_state=self.shared_state, 
                                       execution_manager=self.execution_manager, 
                                       exchange_client=self.exchange_client, 
                                       market_data=self.market_data_feed, 
                                       symbol_manager=self.symbol_manager, 
                                       meta_controller=self.meta_controller)

if self.agent_manager and self.meta_controller:
    self.agent_manager.meta_controller = self.meta_controller
    self.logger.info("[Bootstrap] ✅ Injected MetaController into AgentManager - signal pipeline connected!")

# 🔥 CRITICAL FIX: Register discovery agents BEFORE agent manager starts
if self.agent_manager:
    try:
        from core.agent_registry import register_all_discovery_agents
        register_all_discovery_agents(self.agent_manager, self)
        self.logger.info("[Bootstrap] ✅ Registered discovery agents with AgentManager")
    except Exception as e:
        self.logger.error("[Bootstrap] ❌ Failed to register discovery agents: %s", e, exc_info=True)
```

### Option 2: Create agents explicitly in AppContext
```python
# After SymbolManager is available (Phase 3+)
if up_to_phase >= 3 and self.symbol_manager:
    # Create discovery agents
    if self.wallet_scanner_agent is None:
        WalletScannerAgent = _get_cls(_wallet_scanner_mod, "WalletScannerAgent")
        self.wallet_scanner_agent = _try_construct(
            WalletScannerAgent,
            shared_state=self.shared_state,
            config=self.config,
            exchange_client=self.exchange_client,
            symbol_manager=self.symbol_manager,
        )
    # ... repeat for IPOChaser, SymbolScreener, LiquidationAgent ...
```

**Then register them in Phase 6 BEFORE starting AgentManager:**
```python
if up_to_phase >= 6 and self.agent_manager:
    # Register all discovery agents
    for agent in [self.wallet_scanner_agent, self.ipo_chaser, 
                  self.symbol_screener_agent, self.liquidation_agent]:
        if agent:
            self.agent_manager.register_discovery_agent(agent)
```

---

## Impact Assessment

### Currently Broken
- ❌ Discovery agents list is EMPTY
- ❌ run_discovery_agents_once() iterates empty list → NO SCANS
- ❌ Symbol proposals never generated
- ❌ Symbol universe stays whatever was bootstrapped
- ❌ New opportunities (IPOs, liquidations, wallets) never discovered
- ❌ Capital allocation runs but allocates to empty agent set

### After Fix
- ✅ discovery_agents list populated with 4 agents
- ✅ run_discovery_agents_once() executes all 4 agents every 10 minutes
- ✅ Each agent scans for new opportunities
- ✅ Symbol proposals sent to SymbolManager
- ✅ Symbol universe dynamically expanded
- ✅ Capital allocation distributes to working agents
- ✅ System achieves full autonomous operation

---

## Implementation Checklist

- [ ] Add import: `from core.agent_registry import register_all_discovery_agents`
- [ ] Add call after AgentManager construction in AppContext
- [ ] Test: Check agent_manager.discovery_agents list is populated
- [ ] Test: Verify run_discovery_agents_loop logs agent executions
- [ ] Test: Confirm IPOChaser, WalletScannerAgent, SymbolScreener are running
- [ ] Test: Verify symbols are being proposed and accepted
- [ ] Monitor logs for discovery agent scan output

---

## References

**Key Code Locations:**
- Function Definition: `core/agent_registry.py:254-300`
- Agent Manager Discovery Loop: `core/agent_manager.py:1041-1062`
- Registration Method: `core/agent_manager.py:798-805`
- Discovery Agent List: `core/agent_manager.py:128`
- AppContext AgentManager Construction: `core/app_context.py:3641`

**Related Analysis Documents:**
- AGENT_DISCOVERY_ANALYSIS.md - Full discovery mechanism
- BOOTSTRAP_MODE_AGENTS.md - Agent initialization safety

---

## Conclusion

This is a **bootstrap sequence bug**, not an architecture issue. The function exists and is well-designed, but nobody calls it during initialization. Once fixed, discovery agents will execute immediately and continuously expand the trading universe.

**Priority**: 🔴 CRITICAL - Blocks all dynamic symbol discovery
**Effort**: ⚡ TRIVIAL - One function call, 5-line fix
**Risk**: 🟢 LOW - Adding a missing call, no refactoring needed

