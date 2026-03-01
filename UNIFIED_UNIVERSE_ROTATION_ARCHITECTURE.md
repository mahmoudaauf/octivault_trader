# UNIFIED UNIVERSE ROTATION ARCHITECTURE

## The Problem You Identified

**3 competing authorities, no canonical source of truth:**

```
Discovery Agent
  ├─ Finds 50 symbols
  └─ Feeds arbitrary list to SharedState

Capital Governor
  ├─ Trims by COUNT only (first N, not best N)
  └─ Can't re-rank (insertion order lock-in)

PortfolioBalancer
  ├─ Picks top-N from CURRENT positions only
  ├─ Can't evaluate new candidates
  └─ Weak symbols persist forever

SharedState
  ├─ May merge or shrink incorrectly
  ├─ Defensive logging after damage
  └─ No determinism
```

**Result:** Non-deterministic universe, race conditions, 2 → 24 symbols

---

## The Solution: Unified Universe Rotation Engine (UURE)

**One canonical pipeline with total ordering:**

```
Discovery (wide net: 50-200 candidates)
  ↓ [All candidates collected]
Unified Scoring (score ALL)
  ↓ [Every symbol gets a score]
Global Ranking (sort descending)
  ↓ [Best symbols at the top]
Governor Cap (dynamic cap)
  ↓ [Top-N where N = smart cap]
Hard Replace Universe (exact top-N)
  ↓ [Committed to canonical store]
Rotation Cleanup (liquidate removed symbols)
```

---

## Architecture Principles

### 1. **Canonical Authority**
- **Single source of truth:** UniverseRotationEngine
- **All symbol decisions** flow through this engine
- **No alternate paths** to modify universe

### 2. **Score-Based Selection**
- **NOT insertion order** (wrong: first 2)
- **NOT current positions** (wrong: stuck symbols)
- **Best scoring symbols** (correct: optimal universe)

### 3. **Deterministic Rotation**
- Same candidates + same scores = same universe
- No race conditions
- No merge behavior

### 4. **Capital-Aware Cap**
```python
# Smart cap (replaces simple count cap)
deployable_capital = NAV * max_exposure
dynamic_cap = floor(deployable_capital / min_entry_quote)
final_cap = min(dynamic_cap, MAX_SYMBOL_LIMIT, governor_cap)
```

### 5. **Rotation as Feature**
- Weak symbols are automatically liquidated
- Strong symbols are added
- No manual intervention needed

---

## How UURE Replaces Old Logic

### OLD (Broken)
```
Discovery
  ├─ Sends arbitrary 50 symbols to SymbolManager
  └─ SymbolManager calls: set_accepted_symbols({50 symbols})

Governor
  ├─ Computes cap = 2
  └─ Trims symbols[:2]  ← Wrong! Takes FIRST 2, not BEST 2

PortfolioBalancer
  ├─ Scores CURRENT positions only
  └─ Can't add new candidates

Result:
  • Wrong symbols selected (first, not best)
  • No rotation (weak symbols stuck)
  • Accumulation possible (merge mode)
  • Non-deterministic
```

### NEW (Correct)
```
Discovery
  ├─ Sends 50 candidates to UURE
  └─ UURE collects ALL (positions + accepted + new)

UURE Core Logic
  ├─ Scores ALL candidates: get_unified_score()
  ├─ Ranks by score: sorted(desc)
  ├─ Applies smart cap: min(dynamic, governor, limit)
  ├─ Hard replaces: set_accepted_symbols(top-N, allow_shrink=True)
  └─ Liquidates removed: triggers sell intents

Result:
  ✅ Best symbols selected (score-based ranking)
  ✅ Rotation enabled (weak symbols liquidated)
  ✅ No accumulation (hard replace, deterministic)
  ✅ Deterministic universe
```

---

## Key Changes to Architecture

### Change 1: Governor Role

**OLD:**
```python
# Governor just counts
cap = compute_symbol_cap()  # Returns: 2
symbols = symbols[:cap]  # Keeps first 2
```

**NEW:**
```python
# Governor is one input to UURE's smart cap
governor_cap = await governor.compute_symbol_cap()  # 2

# UURE combines with capital metrics
dynamic_cap = floor((NAV * exposure) / min_entry_quote)

# Final cap is minimum of all constraints
final_cap = min(dynamic_cap, governor_cap, MAX_SYMBOL_LIMIT)
```

**Why better:**
- Governor is ONE constraint, not THE constraint
- Respects both capital limits AND governor safety rules
- Allows intelligent growth when capital increases

### Change 2: Portfolio Balancer Role

**OLD:**
```python
async def _select_topN(self, tradable, atr):
    # Only looks at CURRENT positions
    scored = []
    for s in tradable:  # tradable = current positions only!
        score = self.ss.get_unified_score(s)
        scored.append((score, s))
    # Can't add new symbols from discovery
    # Can't remove weak symbols
```

**NEW:**
```python
# PortfolioBalancer no longer decides universe
# It only sizes positions within accepted symbols

async def _select_topN(self, tradable, atr):
    # Get accepted symbols (decided by UURE)
    accepted = await self.ss.get_accepted_symbols()
    
    # Size positions within accepted only
    # (UURE already removed weak symbols)
    for s in accepted:
        score = self.ss.get_unified_score(s)
        # ... size based on score ...
```

**Why better:**
- Separation of concerns (universe vs. sizing)
- Rotation happens via UURE (weak symbols exit)
- Balancer focuses on capital allocation

### Change 3: Discovery Role

**OLD:**
```python
# Discovery feeds arbitrary list
discovered = screener.find_symbols()  # 50 symbols
accepted = await symbol_manager.propose_symbols(discovered)
```

**NEW:**
```python
# Discovery is ONE input to UURE
discovered = screener.find_symbols()  # 50 symbols

# Add to candidates (UURE will score/rank)
await universe_rotation_engine.compute_and_apply_universe()
# UURE automatically includes discovered symbols
# Scores them against current universe
# Keeps best N
```

**Why better:**
- Discovery is a data source, not a decision authority
- UURE makes the universe decision
- Automatic evaluation against current positions

---

## Smart Cap Calculation

### Formula
```python
deployable_capital = NAV * max_exposure_ratio
min_position_size = MIN_ENTRY_QUOTE_USDT  # e.g., $20

dynamic_cap = floor(deployable_capital / min_position_size)

# Example: $172 account
NAV = $172
max_exposure = 0.8
min_entry = $20
deployable = 172 * 0.8 = $137.60
dynamic_cap = floor(137.60 / 20) = 6

# But limited by governor and max
governor_cap = 2  # (Capital Floor rule)
max_limit = 30

final_cap = min(6, 2, 30) = 2  # Governor wins
```

### Bootstrap Case
```
$172 Account
  ├─ Deployable: $172 × 0.8 = $137.60
  ├─ Min position: $20
  ├─ Dynamic cap: 6 symbols possible
  ├─ Governor cap: 2 symbols (Capital Floor)
  └─ Final cap: 2 ✓ (Governor constraint binds)

$500 Account
  ├─ Deployable: $500 × 0.8 = $400
  ├─ Min position: $20
  ├─ Dynamic cap: 20 symbols possible
  ├─ Governor cap: 4 symbols (Capital Floor)
  └─ Final cap: 4 ✓ (Governor constraint binds)

$10K Account
  ├─ Deployable: $10K × 0.8 = $8000
  ├─ Min position: $20
  ├─ Dynamic cap: 400 symbols possible
  ├─ Governor cap: 12 symbols (Capital Floor)
  ├─ Max limit: 30 symbols
  └─ Final cap: 12 ✓ (Governor constraint binds)
```

---

## Integration Points

### 1. AppContext Initialization
```python
# In app initialization
self.universe_rotation_engine = UniverseRotationEngine(
    shared_state=self.shared_state,
    capital_governor=self.capital_symbol_governor,
    config=self.config,
    execution_manager=self.execution_manager,
    meta_controller=self.meta_controller,
)
```

### 2. Periodic Rotation (New Background Task)
```python
# Run every 5-10 minutes
async def universe_rotation_loop():
    while running:
        try:
            result = await app.universe_rotation_engine.compute_and_apply_universe()
            if result["rotation"]["added"] or result["rotation"]["removed"]:
                logger.info(f"Universe rotated: {result['rotation']}")
        except Exception as e:
            logger.error(f"Universe rotation failed: {e}")
        
        await asyncio.sleep(300)  # 5 minutes
```

### 3. Discovery Integration (Modified)
```python
# OLD: SymbolScreener sends raw list to SymbolManager
# NEW: SymbolScreener adds to candidates, UURE evaluates

# In SymbolScreener.run_once():
discovered = self._screen_symbols()

# Instead of:
# await symbol_manager.propose_symbols(discovered)

# Now just publish as candidates:
await shared_state.emit_event("CandidateSymbolsDiscovered", {
    "candidates": discovered,
    "source": "SymbolScreener"
})

# UURE will see them in next rotation cycle
```

### 4. PortfolioBalancer Simplification
```python
# OLD: _select_topN() tried to be universe authority
# NEW: Just size positions

async def _select_topN(self, tradable, atr):
    # UURE already removed weak symbols
    # Just apply sizing within accepted universe
    accepted = await self.ss.get_accepted_symbols()
    return list(accepted.keys())[:self.max_positions]
```

---

## Execution Flow Example (Bootstrap, $172)

```
┌─────────────────────────────────────────────────────────────┐
│                  HOUR 1: DISCOVERY                          │
└─────────────────────────────────────────────────────────────┘

SymbolDiscoverer runs
  └─ Finds 50 candidates: [BTC, ETH, ADA, ...]
  └─ Emits: CandidateSymbolsDiscovered

PortfolioBalancer runs
  └─ Sees no accepted symbols yet
  └─ Waits for UURE

UURE Rotation Cycle Fires
  ├─ Step 1: Collect candidates
  │   └─ Union: (discovery: 50) + (positions: 0) = 50
  │
  ├─ Step 2: Score ALL
  │   ├─ BTCUSDT: 0.85 (strong)
  │   ├─ ETHUSDT: 0.78 (strong)
  │   ├─ ADAUSDT: 0.45 (weak)
  │   └─ ... 47 more
  │
  ├─ Step 3: Rank by score
  │   └─ [0.85: BTC, 0.78: ETH, 0.45: ADA, ...]
  │
  ├─ Step 4: Compute smart cap
  │   ├─ NAV = $172, exposure = 0.8
  │   ├─ deployable = $137.60
  │   ├─ min_entry = $20
  │   ├─ dynamic = 6
  │   ├─ governor = 2
  │   └─ final_cap = 2
  │
  ├─ Step 5: Apply cap
  │   └─ Take top-2: [BTC (0.85), ETH (0.78)]
  │
  ├─ Step 6: Hard replace universe
  │   └─ set_accepted_symbols({BTC, ETH})
  │
  └─ Step 7: Liquidate removed
      └─ None (starting from 0)

Result:
  accepted_symbols = {BTCUSDT, ETHUSDT}

┌─────────────────────────────────────────────────────────────┐
│              HOUR 2: REBALANCING                            │
└─────────────────────────────────────────────────────────────┘

PortfolioBalancer runs
  ├─ Gets accepted: {BTC, ETH}
  ├─ Scores: BTC=0.85, ETH=0.78
  ├─ Targets: BTC=$86, ETH=$86
  ├─ Buys both
  └─ Result: 2 positions opened

MetaController executes trades
  └─ BTC position: $86 (0.0019 BTC @ $45K)
  └─ ETH position: $86 (0.047 ETH @ $1,830)

┌─────────────────────────────────────────────────────────────┐
│           HOUR 3: SECOND DISCOVERY (DIFFERENT RANKING)      │
└─────────────────────────────────────────────────────────────┘

SymbolDiscoverer runs again
  └─ Finds 50 NEW candidates (market changed)
  └─ Top by volatility: [ETH, ADA, SOL, ...]

UURE Rotation Cycle Fires
  ├─ Step 1: Collect candidates
  │   └─ Union: (discovery: 50) + (positions: 2) = 50+
  │
  ├─ Step 2: Score ALL (including positions)
  │   ├─ ETHUSDT: 0.82 (held, still strong)
  │   ├─ BTCUSDT: 0.65 (held, now weaker)
  │   ├─ ADAUSDT: 0.75 (new, strong)
  │   ├─ SOLUSDT: 0.72 (new)
  │   └─ ... 46 more
  │
  ├─ Step 3: Rank by score
  │   └─ [0.82: ETH, 0.75: ADA, 0.72: SOL, 0.65: BTC, ...]
  │
  ├─ Step 4-5: Apply cap
  │   └─ Take top-2: [ETH (0.82), ADA (0.75)]
  │
  ├─ Step 6: Hard replace universe
  │   ├─ Old: {BTC, ETH}
  │   ├─ New: {ETH, ADA}
  │   └─ set_accepted_symbols({ETH, ADA})
  │
  └─ Step 7: Liquidate removed
      └─ BTC (0.65 < threshold)
         ├─ Create sell intent
         └─ Liquidate $86 worth

Result:
  ├─ BTC position liquidated (rotation exit)
  ├─ ADA position added (rotation entry)
  ├─ ETH position remains
  ├─ Portfolio rotated to better symbols
  └─ Universe still exactly 2 symbols

┌─────────────────────────────────────────────────────────────┐
│            HOUR 4: CONTINUED MONITORING                     │
└─────────────────────────────────────────────────────────────┘

Every 5 minutes:
  UURE rotation cycle → Evaluates universe → Rotation if needed
  
Result:
  ✅ Always 2 symbols (governor cap enforced)
  ✅ Always best symbols (score-ranked)
  ✅ Automatic rotation (weak → strong)
  ✅ Deterministic universe
  ✅ No accumulation
  ✅ No race conditions
```

---

## Comparison: Old vs. New

| Aspect | OLD (Broken) | NEW (Correct) |
|--------|--------------|---------------|
| **Authority** | Scattered (discovery, governor, balancer) | Canonical (UURE) |
| **Selection** | First-N (insertion order) | Best-N (by score) |
| **Cap Logic** | Count only | Smart (capital + count) |
| **Rotation** | Manual/never | Automatic (UURE-driven) |
| **Weak Symbols** | Persist forever | Auto-liquidated |
| **New Candidates** | Ignored after init | Evaluated every cycle |
| **Determinism** | Race conditions | Deterministic |
| **Code Complexity** | High (scattered) | Medium (centralized) |

---

## Implementation Checklist

### Phase 1: Create Engine
- [x] UniverseRotationEngine class (294 lines)
- [ ] Integration with AppContext

### Phase 2: Wire Into AppContext
- [ ] Add to __init__
- [ ] Create background task
- [ ] Hook discovery events

### Phase 3: Simplify Old Components
- [ ] Remove universe logic from SymbolManager
- [ ] Simplify PortfolioBalancer._select_topN()
- [ ] Update Discovery integration

### Phase 4: Testing
- [ ] Unit tests (UURE logic)
- [ ] Integration tests (with governor)
- [ ] Bootstrap scenario tests

### Phase 5: Documentation
- [ ] Architecture guide (this doc)
- [ ] Integration guide
- [ ] Troubleshooting guide

---

## Professional Benefits

1. **Enterprise-Grade Architecture**
   - Single source of truth (UURE)
   - Deterministic behavior
   - Professional hedge-fund pattern

2. **Automatic Rotation**
   - Weak symbols liquidated
   - Strong symbols added
   - No manual intervention

3. **Capital Efficiency**
   - Smart cap respects capital
   - No over-concentration
   - Proper position sizing

4. **Scalability**
   - Same logic from $172 → $1M
   - Just change capital/config
   - No code changes needed

5. **Maintainability**
   - All universe logic in ONE place
   - Easy to modify rules
   - Clear data flow

---

## Summary

**You were absolutely right:**

The old design had 3 competing authorities.

**The new design has 1:**

- **Discovery** = data source
- **Governor** = one constraint
- **PortfolioBalancer** = sizing only
- **UURE** = canonical authority

The 2-symbol bootstrap is now:
- ✅ Score-based (best symbols)
- ✅ Capital-aware (smart cap)
- ✅ Rotation-enabled (weak → strong)
- ✅ Deterministic (same inputs → same universe)
- ✅ Professionally architected
