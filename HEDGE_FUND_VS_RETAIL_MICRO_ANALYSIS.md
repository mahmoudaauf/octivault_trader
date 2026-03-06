# System Architecture Analysis: Model Switching Capability

## Question: Can the system switch between Hedge Fund Model and Retail Micro Model?

**Short Answer:** ✅ **YES, but with an important caveat** - The system has **automatic, dynamic regime switching** based on **Net Asset Value (NAV)**, NOT a traditional "Hedge Fund vs Retail Micro" binary choice.

---

## Current Architecture: NAV-Based Regime Switching

Instead of static models, the system uses **three dynamic operating regimes** that automatically switch based on live account capital:

### The Three Regimes

#### 1️⃣ **MICRO_SNIPER Mode** (NAV < $1,000) 🎯
**This is the "Retail Micro Model"**

**Configuration:**
```
Max Open Positions: 1 (single position only)
Max Active Symbols: 1 (one asset at a time)
Min Expected Move: 1.0% (hardened gate)
Min Confidence: 0.70 (70% minimum)
Min Hold Time: 600 sec (10 minutes)
Max Trades/Day: 3 (hourly discipline)
Position Size: 30% of NAV
```

**Disabled Features:**
- ❌ Symbol rotation (no multi-symbol trading)
- ❌ Dust healing (no cleanup trades)
- ❌ Capital reservations (bypass to max)
- ❌ Expansion (stay focused on best asset)

**Use Case:** Starting capital, learning phase, small accounts

---

#### 2️⃣ **STANDARD Mode** (NAV $1,000 - $5,000) ⚙️
**This is the "Bridge Model"**

**Configuration:**
```
Max Open Positions: 2 (limited diversification)
Max Active Symbols: 3 (conservative universe)
Min Expected Move: 0.50% (relaxed vs MICRO)
Min Confidence: 0.65 (65% minimum)
Min Hold Time: 300 sec (5 minutes)
Max Trades/Day: 6 (moderate frequency)
Position Size: 25% of NAV
```

**Enabled Features:**
- ✅ Symbol rotation (multi-symbol trading)
- ✅ Dust healing (cleanup trades)
- ✅ Capital reservations (normal allocations)
- ✅ Controlled expansion

**Use Case:** Growth phase, transitional accounts, skill building

---

#### 3️⃣ **MULTI_AGENT Mode** (NAV ≥ $5,000) 🚀
**This is the "Hedge Fund / Professional Model"**

**Configuration:**
```
Max Open Positions: 3+ (full portfolio)
Max Active Symbols: 5+ (aggressive diversification)
Min Expected Move: 0.30% (full sensitivity)
Min Confidence: 0.60 (standard threshold)
Min Hold Time: 180 sec (normal scaling)
Max Trades/Day: 20+ (unrestricted)
Position Size: Standard ScalingManager logic
```

**Enabled Features:**
- ✅ All advanced features enabled
- ✅ Full multi-agent system active
- ✅ Advanced portfolio management
- ✅ Maximum optimization

**Use Case:** Professional operations, large capital, institutional-grade trading

---

## How Switching Works

### Automatic Regime Switching

The system **automatically switches regimes** at the start of each trading cycle:

```python
# In MetaController.run_once():
nav = self.signal_manager.get_current_nav()  # Get fresh NAV
regime = nav_regime.get_nav_regime(nav)      # Determine regime
self.regime_manager.update_regime(nav)       # Switch if needed

# Decision-making uses the active regime rules
```

### Zero-Downtime Switching

✅ **Seamless**: No restart required when crossing NAV thresholds
✅ **Reversible**: Automatically goes back to MICRO if NAV drops
✅ **Transparent**: Logged for audit and monitoring

---

## Architecture Components

### 1. NAV Regime Module (`core/nav_regime.py`)
Defines the three regimes and their configuration:
- Determines regime based on NAV
- Provides hard rules for each regime
- Manages transitions automatically

### 2. Regime Manager (`core/regime_manager.py`)
Manages regime state and transitions:
- Updates current regime based on NAV
- Tracks regime changes
- Provides regime info to decision systems

### 3. Capital Governor (`core/capital_governor.py`)
Enforces capital-aware position limits:
- Gets position limits based on NAV
- Calculates position sizing
- Enforces bracket constraints

### 4. Mode Manager (`core/mode_manager.py`)
Manages operational modes (separate from regimes):
- BOOTSTRAP: Initial capital bootstrap
- NORMAL: Normal trading
- AGGRESSIVE: Higher risk tolerance
- PROTECTIVE: Defense mode
- RECOVERY: After drawdown
- SAFE: Maximum caution
- PAUSED: Manual halt
- SIGNAL_ONLY: Monitoring mode

---

## Regime Enforcement Points

### Signal Validation
```python
# Signals rejected if they don't meet regime requirements
if signal.confidence < regime_config.MIN_CONFIDENCE:
    reject_signal("Confidence below regime floor")

if expected_move_pct < regime_config.MIN_EXPECTED_MOVE_PCT:
    reject_signal("Expected move below regime minimum")
```

### Position Limits
```python
# Position counts limited by regime
max_positions = regime_config.MAX_OPEN_POSITIONS
if current_positions >= max_positions:
    reject_buy("Position limit reached for regime")
```

### Feature Availability
```python
# Components check regime before executing
if not regime.ROTATION_ENABLED:
    skip_rotation()

if not regime.DUST_HEALING_ENABLED:
    skip_dust_healing()
```

---

## Real-World Examples

### Example 1: Growing from $500 to $10,000

**Day 1: NAV = $500 (MICRO_SNIPER)**
```
Account Status: Learning phase
Active Regime: MICRO_SNIPER
- Max 1 position, 1 asset
- Need 1.0% move minimum
- Max 3 trades/day
- Result: Safe, focused testing
```

**Week 2: NAV = $1,200 (STANDARD)**
```
Account Status: Growing
Active Regime: Switches automatically to STANDARD
- Max 2 positions, 3 assets
- Need 0.50% move minimum
- Max 6 trades/day
- Result: Controlled expansion
```

**Month 3: NAV = $5,500 (MULTI_AGENT)**
```
Account Status: Professional
Active Regime: Switches automatically to MULTI_AGENT
- Max 3+ positions, 5+ assets
- Need 0.30% move minimum
- Max 20+ trades/day
- Result: Full system activated
```

---

### Example 2: Drawdown Recovery

**Normal:** NAV = $6,000 (MULTI_AGENT)
```
System operates with full features
Max 3 positions, 5 assets
```

**Drawdown:** NAV = $800 (MICRO_SNIPER)
```
System automatically switches to MICRO_SNIPER
Max 1 position, 1 asset
Tighter gates (1.0% move minimum)
More conservative pace (3 trades/day max)
→ Capital preservation mode
```

**Recovery:** NAV = $1,200 (STANDARD)
```
System switches back to STANDARD
Controlled expansion allowed
Max 2 positions, 3 assets
```

---

## Key Differences: Micro Model vs Hedge Fund Model

| Aspect | Micro Model (NAV < $1K) | Hedge Fund Model (NAV ≥ $5K) |
|--------|------------------------|-----------------------------|
| **Max Positions** | 1 | 3+ |
| **Max Assets** | 1 | 5+ |
| **Min Move Required** | 1.0% | 0.30% |
| **Confidence Floor** | 70% | 60% |
| **Trade Frequency** | 3/day max | 20+/day |
| **Position Size** | 30% NAV | Standard scaling |
| **Rotation** | Disabled | Enabled |
| **Dust Healing** | Disabled | Enabled |
| **Capital Reserve** | Bypass | Enforced |
| **Complexity** | Minimal | Full suite |
| **Risk Level** | Very Low | Managed High |

---

## System Advantages

### ✅ Automatic Scaling
- No manual configuration needed
- Grows with account
- Shrinks on drawdown
- Always appropriate for account size

### ✅ Risk Management
- Micro model protects small accounts
- Graduated progression reduces catastrophic loss
- Automatic defensive scaling on drops

### ✅ Seamless Transitions
- No reboot required
- No order cancellations
- Smooth behavioral changes
- Transparent logging

### ✅ Capital Preservation
- Starts conservative
- Unlocks features gradually
- Reverts quickly on drawdown
- Prevents over-leverage at any stage

---

## Monitoring & Control

### View Current Regime
```python
nav = signal_manager.get_current_nav()
regime = nav_regime.get_nav_regime(nav)
print(f"Current NAV: ${nav:.2f}")
print(f"Active Regime: {regime}")  # MICRO_SNIPER, STANDARD, or MULTI_AGENT
```

### Logs Show Regime Switches
```
[2026-03-02 14:23:45] NAV crossed $1000 threshold
[2026-03-02 14:23:46] Regime switched: MICRO_SNIPER → STANDARD
[2026-03-02 14:23:47] Updating position limits: 1 → 2
[2026-03-02 14:23:48] Updating symbol limit: 1 → 3
```

### Monitor Position Limits
```python
limits = capital_governor.get_position_limits(nav)
print(f"Max concurrent positions: {limits['max_concurrent_positions']}")
print(f"Max active symbols: {limits['max_active_symbols']}")
print(f"Quote per position: ${limits['quote_per_position']:.2f}")
```

---

## Configuration

### In `core/config.py`
```python
# NAV thresholds for regime switching
NAV_MICRO_THRESHOLD = 1000.0   # Switch to STANDARD above this
NAV_STANDARD_THRESHOLD = 5000.0  # Switch to MULTI_AGENT above this

# These are automatic - no manual setting needed
```

### Regime-specific parameters are hardcoded in:
- `MicroSniperConfig` (lines 100-125)
- `StandardConfig` (lines 128-155)
- `MultiAgentConfig` (lines 158-185)

---

## Answer to Your Question

### Can the system switch between models?

**YES** ✅

The system has **automatic, continuous switching** between three models:

1. **Retail Micro Model** (< $1K NAV) - Ultra-conservative, single-position focus
2. **Standard Model** ($1K-$5K NAV) - Balanced growth phase
3. **Hedge Fund Model** (≥ $5K NAV) - Full professional trading system

**This is not a binary choice but a graduated progression.**

### How does it work?

- 🔄 **Automatic**: Checks NAV at start of each cycle
- 📊 **Dynamic**: Switches regime based on live account value
- 🔐 **Seamless**: No restart or configuration changes needed
- 📈 **Reversible**: Automatically scales back down if capital drops
- 📋 **Transparent**: All changes logged for audit

### Example:
```
Start: $500 → MICRO_SNIPER mode (1 position max, 1.0% move min)
        ↓
Grow:  $1,200 → STANDARD mode (2 positions max, 0.5% move min)
        ↓
Scale: $5,500 → MULTI_AGENT mode (3+ positions, 0.3% move min)
        ↓
Drop:  $800 → Back to MICRO_SNIPER mode (automatic protection)
```

---

## Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| **Regime Detection** | ✅ Implemented | `core/nav_regime.py` - fully functional |
| **Regime Switching** | ✅ Implemented | Automatic at each cycle |
| **Micro Model** | ✅ Implemented | MICRO_SNIPER configuration ready |
| **Standard Model** | ✅ Implemented | STANDARD configuration ready |
| **Hedge Fund Model** | ✅ Implemented | MULTI_AGENT configuration ready |
| **Enforcement** | ✅ Implemented | Position limits, signal gates, features |
| **Monitoring** | ✅ Implemented | Logging, metrics, reporting |
| **Testing** | ✅ Implemented | Multiple test scenarios |

---

## Conclusion

The Octivault Trader system is **sophisticated in its approach to model switching** - rather than a manual toggle between "Hedge Fund" and "Retail Micro" modes, it implements **automatic, continuous regime adjustment** based on live account NAV.

This approach is:
- ✅ **More robust** - Always calibrated to account size
- ✅ **More safe** - Conservative by default, grows gradually  
- ✅ **More efficient** - No manual reconfiguration needed
- ✅ **More professional** - Used by real prop trading firms

It's a **graduated progression system** that ensures appropriate risk management at every capital level while seamlessly unlocking advanced features as the account grows.
