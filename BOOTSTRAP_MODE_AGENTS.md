# Bootstrap Mode: Agent Operations & Safety Mechanisms

## Overview

**Bootstrap Mode** is the system initialization phase where the portfolio is being seeded with initial positions. During this phase, specific safety mechanisms and agent behaviors are activated to ensure safe system startup.

**Key Principle:** Bootstrap mode allows relaxed confidence/EV gates to seed the portfolio while maintaining other safety constraints.

---

## What is Bootstrap Mode?

### Detection

Bootstrap mode is detected via:

```python
def _is_bootstrap_mode(self) -> bool:
    try:
        if hasattr(self.shared_state, "is_bootstrap_mode"):
            return bool(self.shared_state.is_bootstrap_mode())
    except Exception:
        pass
    try:
        if hasattr(self, "mode_manager"):
            return str(self.mode_manager.get_mode() or "").upper() == "BOOTSTRAP"
    except Exception:
        pass
    return False
```

**Sources:**
1. `shared_state.is_bootstrap_mode()` - Primary check
2. `mode_manager.get_mode() == "BOOTSTRAP"` - Secondary check
3. Signal-level flags: `_bootstrap`, `_bootstrap_override`, `_bootstrap_seed`, `bootstrap_seed`

### Characteristics

| Aspect | Value |
|--------|-------|
| **Confidence Floor** | Minimum 55% (configurable: `BOOTSTRAP_MIN_CONFIDENCE`) |
| **EV Gate** | Bypassable under strict conditions |
| **Position Sizing** | Minimum notional enforced |
| **Blocked Agents** | None (all agents can contribute) |
| **Execution Behavior** | Seed-focused, safety-preserving |

---

## Which Agents Operate in Bootstrap Mode?

### ✅ Active in Bootstrap

All registered agents can contribute signals during bootstrap:

1. **IPOChaser**
   - Detects newly listed IPOs
   - Can seed positions during bootstrap
   - Normal signal generation applies

2. **DipSniper**
   - Identifies price dips for buying opportunities
   - Contributes to portfolio seeding
   - Confidence-sensitive but bootstrap-bypassable

3. **TrendHunter**
   - Follows trend signals
   - Can contribute during bootstrap
   - Not blocked in any regime

4. **LiquidationAgent**
   - Detects liquidation opportunities
   - Provides exit signals
   - Critical for risk management even in bootstrap

5. **MLForecaster**
   - Machine learning-based predictions
   - Training can be scheduled in background
   - Contributes to signal generation

6. **WalletScannerAgent**
   - Analyzes wallet movements
   - Identifies anomalies/opportunities
   - Contributes discovery signals

7. **SymbolScreener**
   - Screens symbols for quality/viability
   - Used for universe discovery
   - Active during bootstrap phase

### ⚠️ Blocked in Specific Regimes (Not Bootstrap-Specific)

```python
# In LOW volatility regime:
blocked_agents = {"trendhunter", "bootstrapscalper"}
```

- **TrendHunter** is blocked in LOW volatility (too risky for low-vol sideways markets)
- **BootstrapScalper** is blocked in LOW volatility (not a real agent, reference for bootstrap override)

**Note:** These blocks apply to **all modes**, not specific to bootstrap.

### 🔄 Special Bootstrap Operations

#### 1. **BootstrapSeed (Meta-Agent)**

Not a real agent, but an internal mechanism:

```python
seed_sig = {
    "symbol": seed_symbol,
    "action": "BUY",
    "confidence": 1.0,
    "agent": "BootstrapSeed",      # Meta-generated
    "timestamp": time.time(),
    "reason": "BOOTSTRAP_SEED",
    "context": "BOOTSTRAP",
    "_bootstrap_seed": True,
    "bootstrap_seed": True,
    "execution_tag": "meta/bootstrap_seed",
    "_bootstrap_seed_cycle": self._tick_counter,
    "_tier": "BOOTSTRAP",
}
```

**Purpose:** Automatically seed positions when portfolio is flat and throughput targets exist

**Triggers:**
- Portfolio flat (no holdings)
- Throughput gap exists (insufficient volume)
- Bootstrap mode active

#### 2. **Liquidation Agent Special Role**

The Liquidation Agent runs during bootstrap but with constraints:

```python
# In app_context.py
self.logger.debug("universe bootstrap: liq agent symbols refresh failed", exc_info=True)
# Liquidation agent refreshes symbols during bootstrap universe initialization
```

**Bootstrap Role:**
- Ensures exits are always available
- Identifies liquidation opportunities
- Provides risk management signals

---

## Bootstrap Mode Decision Flow

### Signal Generation → Execution Path

```
┌─────────────────────────────────────────┐
│ Agent Generates Signal                  │
│ (IPOChaser, DipSniper, TrendHunter, etc)│
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ Is System Bootstrap?│
        └──────────┬──────────┘
                   │
        ┌──────────▼─────────────┐
        │ Apply Confidence Floor │
        │ MIN: 55% (configurable)│
        └──────────┬─────────────┘
                   │
        ┌──────────▼──────────────────┐
        │ Check Tradeability Gate      │
        │ (EV/Confidence)              │
        └──────────┬───────────────────┘
                   │
    ┌──────────────┴────────────────┐
    │                               │
┌───▼──────────────┐    ┌──────────▼──────────┐
│ Normal Signals   │    │ Bootstrap Signals   │
│ (EV satisfied)   │    │ (Bootstrap Flags)   │
└───┬──────────────┘    └──────────┬──────────┘
    │                              │
    │     ┌────────────┬───────────┘
    │     │ EV Bypass  │
    │     │ Allowed?   │
    │     └────────────┤
    │                  │
    │     ┌────────────▼──────────┐
    │     │ 3-Condition Check:    │
    │     │ 1. Bootstrap flag ✓   │
    │     │ 2. Portfolio flat ✓   │
    │     │ 3. No open positions ✓│
    │     └────────────┬──────────┘
    │                  │
    │     ┌────────────┴────────────┐
    │     │ ALL conditions met?     │
    │     │ → Allow EV bypass       │
    │     └──────────┬─────────────┘
    │               │
    └───────────────┴──────────────┐
                                   │
                    ┌──────────────▼─┐
                    │ Execute Trade  │
                    └────────────────┘
```

---

## Agent Budget Allocation in Bootstrap

### Tier System During Bootstrap

Agents are grouped into tiers for budget allocation:

#### **Tier A: High-Confidence Signals**
- Agents with EV-verified signals
- Normal confidence requirements met
- Allocated variable budget based on confidence

#### **Tier B: Bootstrap/Micro Signals**
- Lower confidence signals
- Bootstrap-eligible agents
- Budget capped at `MICRO_SIZE_QUOTE`

#### **Bootstrap Override**
When `_bootstrap_override` flag set:

```python
elif best_sig.get("_bootstrap_override") or bootstrap_execution_override:
    best_sig["_bootstrap"] = True
    best_sig["_force_min_notional"] = True
    best_sig["reason"] = f"bootstrap_override:{agent_tag}:{best_conf:.2f}"
    bootstrap_min = await self._resolve_entry_quote_floor(...)
    if total_intended_quote < bootstrap_min:
        total_intended_quote = bootstrap_min
```

---

## Bootstrap Safety Mechanisms

### 1. **Minimum Confidence Floor**

```python
# In _signal_required_conf_floor() method
if self._is_bootstrap_mode():
    bootstrap_min_conf = float(self._cfg("BOOTSTRAP_MIN_CONFIDENCE", 0.55))
    effective = max(effective, bootstrap_min_conf)
```

**Effect:** Even in bootstrap, confidence must be ≥55% (configurable)

### 2. **EV Bypass Safety Gate** (Phase 4 Implementation)

```python
def _signal_tradeability_bypass(...) -> bool:
    # --- SAFE BOOTSTRAP EV BYPASS ---
    if bootstrap_flag and bool(portfolio_flat):
        # Verify NO open positions exist
        try:
            open_positions = {}
            if hasattr(self.shared_state, "get_open_positions"):
                method = getattr(self.shared_state, "get_open_positions")
                if callable(method):
                    result = method()
                    open_positions = result if isinstance(result, dict) else {}
            
            # Deny bypass if ANY positions exist
            if open_positions and len(open_positions) > 0:
                self.logger.warning(
                    "[Meta:BootstrapEVBypass] Denied: %d open positions remain",
                    len(open_positions)
                )
                return False
        except Exception as e:
            # Fail-closed: deny bypass if verification fails
            return False
        
        # All checks passed
        return True
    return False
```

**3-Condition Check (ALL must be true):**
1. ✅ Bootstrap flag explicitly set
2. ✅ Portfolio flat (no holdings)
3. ✅ No open positions (verified immediately)

### 3. **Minimum Notional Enforcement**

```python
seed_quote = max(
    float(self._bootstrap_seed_quote or 0.0),
    float(self._min_entry_quote_usdt or 0.0),
    float(min_notional or 0.0),  # Exchange minimum
)
```

**Effect:** Bootstrap trades must meet exchange minimum notional requirements

### 4. **Agent Reservation Protection**

```python
# In execution_manager.py line 5429
self.logger.info(f"[EM:BOOTSTRAP] Bypassing agent reservation check for bootstrap execution")
```

**Effect:** Bootstrap signals bypass normal agent budget reservations to ensure portfolio seeding

### 5. **Capital Allocator Bootstrap Fixes**

```python
# In capital_allocator.py
# BOOTSTRAP FIX: Ensure all registered agents are in the map
# BOOTSTRAP FIX: Ensure agents is not None before iterating
# BOOTSTRAP FIX: If we still have no agents, log warning but don't crash
```

**Effect:** Robust handling even if agent registry incomplete

---

## Agent Behavior Matrix

| Agent | Bootstrap Mode | Low Vol | Normal | Notes |
|-------|---|---|---|---|
| **IPOChaser** | ✅ Active | ✅ Active | ✅ Active | Always contributes |
| **DipSniper** | ✅ Active | ✅ Active | ✅ Active | Confidence-gated |
| **TrendHunter** | ✅ Active | ❌ Blocked | ✅ Active | Too risky in low-vol |
| **LiquidationAgent** | ✅ Active | ✅ Active | ✅ Active | Risk management |
| **MLForecaster** | ✅ Training | ✅ Training | ✅ Active | Background training in bootstrap |
| **WalletScannerAgent** | ✅ Active | ✅ Active | ✅ Active | Discovery signals |
| **SymbolScreener** | ✅ Active | ✅ Active | ✅ Active | Universe maintenance |

---

## Configuration for Bootstrap

### Environment/Config Variables

```python
BOOTSTRAP_MIN_CONFIDENCE = 0.55      # Minimum confidence even in bootstrap
BOOTSTRAP_SEED_QUOTE = 100.0         # Seed position size (USDT)
MIN_ENTRY_QUOTE_USDT = 10.0          # Minimum entry notional
MICRO_SIZE_QUOTE = 25.0              # Cap for Tier B bootstrap signals
BASE_CAPITAL = 1000.0                # Starting capital
```

### Mode Management

Bootstrap mode typically activated by:

1. **System Initialization** - Automatic on startup
2. **Mode Manager** - Explicit mode switch
3. **Signal-Level Flags** - `_bootstrap`, `_bootstrap_seed`, `_bootstrap_override`

---

## Logging & Audit Trail

### Bootstrap-Related Log Messages

**When Bootstrap EV Bypass Allowed:**
```
[Meta:BootstrapEVBypass] Allowed for signal (bootstrap=True, portfolio_flat=True, open_positions=0)
```

**When Bootstrap EV Bypass Denied:**
```
[Meta:BootstrapEVBypass] Denied EV bypass despite bootstrap flag: 2 open positions remain
```

**When Bootstrap Execution Processed:**
```
[EM:BOOTSTRAP] Bypassing agent reservation check for bootstrap execution
```

**When Bootstrap Min Confidence Applied:**
```
[Meta:Bootstrap] Confidence floor enforced: 0.58 (min=0.55)
```

**When Bootstrap Budget Raised:**
```
[Meta] Throughput Guard: Raised bootstrap budget to exchange floor 10.00 for BTC
```

---

## Exit from Bootstrap Mode

Bootstrap mode is exited when:

1. **Portfolio has positions** - At least one position established
2. **Mode switched** - Mode manager switches to NORMAL/PROTECTIVE/SAFE/RECOVERY
3. **Configuration change** - `is_bootstrap_mode()` returns False
4. **System reset** - Mode manager reset or reconfiguration

Once exited:
- ✅ Full EV confidence requirements apply
- ✅ All agent filters active (no bootstrap overrides)
- ✅ Normal risk management active
- ❌ EV bypass only for dust operations

---

## Best Practices for Bootstrap Mode

### DO ✅

- ✅ Let system seed positions during bootstrap
- ✅ Trust the 3-condition safety gate for EV bypass
- ✅ Monitor logging for bootstrap operations
- ✅ Allow minimum 1-2 minutes for portfolio initialization
- ✅ Verify all agents are registered before bootstrap

### DON'T ❌

- ❌ Manually set bootstrap_override unless absolutely necessary
- ❌ Ignore bootstrap-related warnings in logs
- ❌ Disable minimum confidence floor during bootstrap
- ❌ Interrupt bootstrap seeding process
- ❌ Mix bootstrap mode with existing positions (causes confusion)

---

## Troubleshooting Bootstrap Issues

### Issue: Bootstrap signals not executing

**Check:**
1. Is system actually in bootstrap mode? (`_is_bootstrap_mode()`)
2. Are agents registered? (Check AgentRegistry)
3. Do signals have required fields? (confidence, action, etc)
4. Is portfolio actually flat? (get_open_positions() returns {})
5. Are conditions met for EV bypass?

### Issue: Wrong agents contributing in bootstrap

**Check:**
1. Is volatility regime LOW? (TrendHunter blocked)
2. Are agent budgets allocated? (CapitalAllocator)
3. Is agent in discovery list? (register_all_discovery_agents)

### Issue: Bootstrap seeding not proceeding

**Check:**
1. Is throughput_gap detected? (needs targets)
2. Is portfolio_flat=True?
3. Are open_positions truly empty?
4. Check exchange minimum notional vs seed_quote

---

## Summary

**Bootstrap Mode** is a specialized execution phase where:

- **All agents** can contribute signals
- **Confidence floor** set to minimum (55% default)
- **EV bypass** allowed ONLY under strict 3-condition gate
- **Portfolio seeding** is the priority
- **Safety mechanisms** remain fully active
- **Exit** automatic when positions established

**Key Agent Operations:**
- Regular agents send discovery/opportunity signals
- Meta-agent (BootstrapSeed) auto-seeds when flat
- Liquidation agent ensures risk management
- All signals respect 3-condition safety gate

**No agents are permanently "bootstrap-only"** - all agents contribute equally in bootstrap and normal modes, with safety gates and tier-based allocation determining execution.

