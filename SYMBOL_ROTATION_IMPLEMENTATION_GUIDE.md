# Symbol Rotation Phases — Implementation Guide

**Status**: ❌ **NOT YET IMPLEMENTED (0% Complete)**  
**Recommendation**: Implement all 3 phases  
**Estimated Effort**: 2-3 weeks (prioritize Phase 1 first)  

---

## Executive Summary

Your requirements ask for a 3-phase symbol rotation upgrade:

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1** | Safe Upgrade (soft lock, screener proposals, 3-5 universe) | ❌ Missing |
| **Phase 2** | Professional Mode (performance-weighted scoring) | ❌ Missing |
| **Phase 3** | Advanced (dynamic universe by volatility) | ❌ Missing |

### Current State
- ✅ Bootstrap lock EXISTS but is hard
- ✅ Volatility regime detection EXISTS
- ✅ Symbol discovery framework EXISTS
- ❌ Everything else is missing

---

## What Needs to Be Done

### Phase 1: Safe Upgrade (Priority: HIGH)
**Effort**: 2-3 days | **Impact**: Medium (better testing, safer scaling)

```python
# 1. Soft Bootstrap Lock (currently hard-locked)
   - Remove: bootstrap_hard_lock always True
   - Add: soft lock with duration (1 hour)
   - Effect: Allows rotation after period without full reset

# 2. Symbol Screener (20-30 proposals)
   - Create: core/symbol_screener.py
   - Methods: get_proposed_symbols()
   - Effect: Feed rotation with candidate pool

# 3. Universe Cap (3-5 symbols)
   - Add: MAX_ACTIVE_SYMBOLS = 5
   - Add: MIN_ACTIVE_SYMBOLS = 3
   - Effect: Prevent over-diversification

# 4. Replacement Multiplier (1.10)
   - Change: 1.25 → 1.10 (make tunable)
   - Effect: Easier to rotate underperformers
```

### Phase 2: Professional Mode (Priority: MEDIUM)
**Effort**: 3-4 days | **Impact**: High (better signal quality)

```python
# Create: core/symbol_scorer_professional.py
# Scoring Algorithm:
#   final_score = expected_edge × 0.40
#               + realized_pnl × 0.25
#               + confidence × 0.20
#               - correlation_penalty × 0.10
#               - drawdown_penalty × 0.05

# Components:
#   1. Expected Edge (from ML model)
#   2. Realized PnL (recent win rate, 30 days)
#   3. Model Confidence (how sure is prediction)
#   4. Correlation Penalty (diversification)
#   5. Drawdown Penalty (recent losses)

# Integration: Replace simple scoring with professional scorer
```

### Phase 3: Advanced (Priority: LOW)
**Effort**: 2-3 days | **Impact**: High (adaptive to market conditions)

```python
# Dynamic Universe Sizing:
#   if volatility == HIGH:     → 5-7 symbols
#   if volatility == NORMAL:   → 3-5 symbols
#   if volatility == LOW:      → 2-3 symbols
#   if volatility == EXTREME:  → 1-2 symbols

# Integration: Adjust caps based on real-time regime
```

---

## Quick Start: Phase 1 Implementation

### Step 1: Add Configuration
**File**: `core/config.py` or `.env`

```python
# Symbol Rotation (Phase 1)
BOOTSTRAP_SOFT_LOCK_ENABLED = True
BOOTSTRAP_SOFT_LOCK_DURATION_SEC = 3600  # 1 hour
SYMBOL_REPLACEMENT_MULTIPLIER = 1.10     # Was 1.25
MAX_ACTIVE_SYMBOLS = 5
MIN_ACTIVE_SYMBOLS = 3
```

### Step 2: Create Soft Bootstrap Lock
**File**: `core/symbol_rotation.py` (NEW)

```python
import time
from typing import Optional

class SymbolRotationManager:
    """Manages soft bootstrap lock and symbol rotation eligibility"""
    
    def __init__(self, config):
        self.soft_lock_enabled = getattr(config, 'BOOTSTRAP_SOFT_LOCK_ENABLED', True)
        self.soft_lock_duration = getattr(config, 'BOOTSTRAP_SOFT_LOCK_DURATION_SEC', 3600)
        self.replacement_multiplier = getattr(config, 'SYMBOL_REPLACEMENT_MULTIPLIER', 1.10)
        self.last_rotation_ts = 0.0
    
    def is_locked(self) -> bool:
        """Check if bootstrap soft lock is still active"""
        if not self.soft_lock_enabled:
            return False  # Soft lock disabled, allow immediate rotation
        
        elapsed = time.time() - self.last_rotation_ts
        return elapsed < self.soft_lock_duration
    
    def lock(self):
        """Activate soft lock after a trade"""
        self.last_rotation_ts = time.time()
    
    def can_rotate_to_score(self, current_score: float, candidate_score: float) -> bool:
        """Check if candidate exceeds replacement threshold"""
        return candidate_score > (current_score * self.replacement_multiplier)
```

### Step 3: Add to MetaController
**File**: `core/meta_controller.py`

```python
# In __init__:
from core.symbol_rotation import SymbolRotationManager
self.rotation_manager = SymbolRotationManager(config)

# In rotation decision logic:
def should_rotate_symbol(self, current_symbol, candidate_symbol):
    # Check if locked
    if self.rotation_manager.is_locked():
        return False
    
    # Check replacement threshold
    current_score = self.score_symbol(current_symbol)
    candidate_score = self.score_symbol(candidate_symbol)
    
    return self.rotation_manager.can_rotate_to_score(current_score, candidate_score)

# After executing trade:
def after_trade_executed(self):
    self.rotation_manager.lock()  # Engage soft lock
```

### Step 4: Implement Screener
**File**: `core/symbol_screener.py` (NEW)

```python
from typing import List, Tuple

class SymbolScreener:
    """Proposes 20-30 symbol candidates for rotation"""
    
    def __init__(self, config, exchange_client):
        self.exchange_client = exchange_client
        self.min_proposals = 20
        self.max_proposals = 30
    
    async def get_proposed_symbols(self) -> List[str]:
        """
        Get 20-30 symbol proposals based on:
        - Volume > $1M
        - Price > $0.01
        - Recent volatility
        """
        try:
            # Get all USDT pairs
            all_pairs = await self.exchange_client.get_all_symbols()
            
            # Score and filter
            scored = []
            for symbol in all_pairs:
                if 'USDT' not in symbol:
                    continue
                
                score = await self._score_symbol(symbol)
                if score > 0:
                    scored.append((symbol, score))
            
            # Sort by score
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Return 20-30
            result = [s for s, _ in scored[:self.max_proposals]]
            return result[max(0, self.min_proposals - len(result)):]
        
        except Exception as e:
            print(f"Screener error: {e}")
            return []
    
    async def _score_symbol(self, symbol: str) -> float:
        """Basic scoring (can be enhanced)"""
        try:
            ticker = await self.exchange_client.get_ticker(symbol)
            volume = float(ticker.get('quoteAssetVolume', 0))
            price = float(ticker.get('lastPrice', 0))
            
            score = 0.0
            if volume > 1000000:  # $1M+
                score += 1.0
            if price > 0.01:
                score += 1.0
            
            return score
        except:
            return 0.0
```

### Step 5: Universe Enforcer
**File**: `core/meta_controller.py` (add method)

```python
def enforce_universe_size(self):
    """Enforce MIN/MAX active symbols"""
    config = self.config
    max_symbols = getattr(config, 'MAX_ACTIVE_SYMBOLS', 5)
    min_symbols = getattr(config, 'MIN_ACTIVE_SYMBOLS', 3)
    
    active = self.get_active_symbols()
    
    # Too many?
    if len(active) > max_symbols:
        # Remove worst performers
        # ... implementation depends on your scoring ...
        pass
    
    # Too few?
    elif len(active) < min_symbols:
        # Add best candidates from screener
        # ... implementation depends on screener ...
        pass
```

---

## Testing Phase 1

```python
# Test 1: Soft lock works
def test_soft_bootstrap_lock():
    mgr = SymbolRotationManager(config)
    
    mgr.lock()
    assert mgr.is_locked() == True
    
    # Simulate 31 minutes passing
    mgr.last_rotation_ts = time.time() - 1860
    assert mgr.is_locked() == False  ✅
    
# Test 2: Replacement multiplier works
def test_replacement_multiplier():
    mgr = SymbolRotationManager(config)
    mgr.replacement_multiplier = 1.10
    
    can_rotate = mgr.can_rotate_to_score(100, 115)  # 115 > 110 ✅
    assert can_rotate == True
    
    can_rotate = mgr.can_rotate_to_score(100, 105)  # 105 < 110
    assert can_rotate == False  ✅
    
# Test 3: Screener returns 20-30
def test_screener():
    screener = SymbolScreener(config, exchange_client)
    symbols = asyncio.run(screener.get_proposed_symbols())
    
    assert 20 <= len(symbols) <= 30  ✅
    assert all('USDT' in s for s in symbols)  ✅
```

---

## Next: Phase 2 (Professional Scoring)

Once Phase 1 is working, implement Phase 2:

```python
# File: core/symbol_scorer_professional.py

class ProfessionalScorer:
    def score(self, symbol: str) -> float:
        edge = self._expected_edge(symbol) * 0.40        # 40% weight
        pnl = self._realized_pnl(symbol) * 0.25          # 25% weight
        conf = self._confidence(symbol) * 0.20           # 20% weight
        corr = self._correlation_penalty(symbol) * 0.10  # -10% weight
        draw = self._drawdown_penalty(symbol) * 0.05     # -5% weight
        
        return edge + pnl + conf - corr - draw
```

---

## Priority Recommendation

### 🔴 Do This First (Phase 1)
- Removes bootstrap hard lock (frees up rotation)
- Enables screener (gives rotation candidates)
- Enforces universe size (prevents over-diversification)
- Configurable multiplier (easier tuning)

**Time**: 2-3 days  
**Impact**: Medium (better testing, foundation for Phase 2)

### 🟡 Then This (Phase 2)
- Professional scoring (better signal quality)
- Uses realized PnL, expected edge, confidence
- Adds correlation & drawdown penalties

**Time**: 3-4 days  
**Impact**: High (significantly better rotations)

### 🟢 Finally (Phase 3)
- Dynamic universe sizing (market-aware)
- Adapts to volatility regime
- 5-7 in high vol, 2-3 in low vol

**Time**: 2-3 days  
**Impact**: High (adaptive to market conditions)

---

## Summary

| Item | Status | Notes |
|------|--------|-------|
| Soft Bootstrap Lock | ❌ Missing | Easy, ~4 hours |
| Screener (20-30) | ❌ Missing | Easy, ~4 hours |
| Universe Cap (3-5) | ❌ Missing | Easy, ~2 hours |
| Replacement Multiplier | ❌ Missing | Easy, ~1 hour |
| Professional Scoring | ❌ Missing | Medium, ~1-2 days |
| Dynamic Universe | ❌ Missing | Medium, ~1 day |

**Total Effort**: 2-3 weeks (start Phase 1, then Phase 2-3)  
**Total Impact**: High (much better symbol rotation)  

**Start with Phase 1. It's the foundation for everything else.**

