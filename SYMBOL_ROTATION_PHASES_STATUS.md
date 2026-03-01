# Symbol Rotation Phased Implementation — Current Status & Roadmap

**Date**: March 1, 2026  
**Status**: ❌ **NOT YET IMPLEMENTED**  
**Recommendation**: Need to implement all 3 phases  

---

## Current State

### What We Have ✅
- ✅ Bootstrap lock mechanism exists (`meta_controller.py:8293, 8837`)
- ✅ Volatility regime detection (`volatility_regime.py`)
- ✅ Symbol discovery framework exists
- ✅ Capital allocator infrastructure (`capital_allocator.py`)
- ✅ Multi-symbol rotation mentioned in documentation

### What We DON'T Have ❌
- ❌ Phase 1: Safe Upgrade (hard bootstrap removal, screener proposals)
- ❌ Phase 2: Professional Mode (performance-weighted scoring)
- ❌ Phase 3: Advanced (dynamic universe sizing based on volatility)
- ❌ Replacement multiplier configuration (1.25 → 1.10)
- ❌ Expected edge weighted scoring
- ❌ Realized PnL weighting
- ❌ Correlation penalty calculation
- ❌ Drawdown penalty calculation

---

## Phase 1: Safe Upgrade (Conservative but Adaptive)

### Objective
Allow gradual removal of hard bootstrap locking while maintaining stability.

### Requirements
```
✅ Remove hard bootstrap locking
   └─ Currently: Bootstrap lock prevents symbol rotation
   └─ Target: Soft lock with gradual enable

✅ Allow screener to propose 20–30 pairs
   └─ Currently: Unknown if screener exists
   └─ Target: Symbol discovery proposes candidates

✅ Keep active universe at 3–5 symbols
   └─ Currently: Unknown if enforced
   └─ Target: Hard cap on active positions

✅ Lower replacement multiplier 1.25 → 1.10
   └─ Currently: Not configurable
   └─ Target: Make multiplier tunable
```

### Implementation Plan

#### 1.1 Make Bootstrap Lock Soft
**File**: `core/meta_controller.py` or new `core/symbol_rotation.py`

```python
class SymbolRotationManager:
    def __init__(self, config):
        self.bootstrap_hard_lock = False  # Was: always True
        self.bootstrap_soft_lock_duration_sec = 3600  # 1 hour
        self.bootstrap_soft_lock_until_ts = 0.0
        
    def is_bootstrap_locked(self) -> bool:
        """
        Hard lock: completely disabled
        Soft lock: enable after duration expires
        """
        if self.bootstrap_hard_lock:
            return True  # Hard lock active
        
        now = time.time()
        if now < self.bootstrap_soft_lock_until_ts:
            return True  # Soft lock still active
        
        return False  # Lock expired, rotation allowed
    
    def reset_soft_lock(self):
        """Called after each trade to restart soft lock"""
        self.bootstrap_soft_lock_until_ts = time.time() + self.bootstrap_soft_lock_duration_sec
```

#### 1.2 Screener Proposal System
**File**: New `core/symbol_screener.py`

```python
class SymbolScreener:
    def __init__(self, config, exchange_client):
        self.exchange_client = exchange_client
        self.max_proposals = 30  # Can propose up to 30
        self.min_proposals = 20  # At least 20
        
    async def get_proposed_symbols(self) -> List[str]:
        """
        Screener proposes 20-30 symbols based on:
        - Volume > threshold
        - Price > min_price
        - Recent performance
        - Volatility profile
        """
        all_symbols = await self.exchange_client.get_trading_symbols()
        
        scored = []
        for symbol in all_symbols:
            score = await self._score_symbol(symbol)
            if score > 0:
                scored.append((symbol, score))
        
        # Sort by score, return top 20-30
        scored.sort(key=lambda x: x[1], reverse=True)
        proposed = [s for s, _ in scored[:30]]
        
        return proposed[:max(20, min(len(proposed), 30))]
    
    async def _score_symbol(self, symbol: str) -> float:
        """Score based on basic criteria"""
        try:
            ticker = await self.exchange_client.get_ticker(symbol)
            volume = float(ticker.get('volume', 0))
            price = float(ticker.get('lastPrice', 0))
            
            # Basic scoring
            score = 0.0
            if volume > 1000000:  # $1M+ volume
                score += 1.0
            if price > 0.01:  # Not dust
                score += 1.0
            
            return score
        except:
            return 0.0
```

#### 1.3 Universe Size Cap (3-5)
**File**: `core/meta_controller.py` or `core/symbol_rotation.py`

```python
class UniverseManager:
    def __init__(self, config):
        self.max_active_symbols = 5  # Hard cap
        self.min_active_symbols = 3  # Hard floor
        self.active_symbols: List[str] = []
    
    def get_active_universe(self) -> List[str]:
        """Return current active symbols (3-5)"""
        return self.active_symbols[:self.max_active_symbols]
    
    def add_symbol(self, symbol: str) -> bool:
        """Add symbol if under cap"""
        if len(self.active_symbols) < self.max_active_symbols:
            self.active_symbols.append(symbol)
            return True
        return False  # At capacity
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol, maintain minimum"""
        if len(self.active_symbols) > self.min_active_symbols:
            self.active_symbols.remove(symbol)
            return True
        return False  # At minimum
```

#### 1.4 Replacement Multiplier (1.25 → 1.10)
**File**: `core/config.py` or environment

```python
# Configuration
SYMBOL_REPLACEMENT_MULTIPLIER = 1.10  # Was: 1.25
# This means: replace if new_score > 1.10 * current_score
```

---

## Phase 2: Professional Mode (Performance-Weighted Rotation)

### Objective
Add sophisticated scoring based on expected value, realized PnL, confidence, and risk metrics.

### Algorithm

```
final_score = 
    expected_edge_weighted * 0.40
  + recent_realized_pnl_weight * 0.25
  + confidence_weight * 0.20
  - correlation_penalty * 0.10
  - drawdown_penalty * 0.05
```

### Implementation

**File**: New `core/symbol_scorer_professional.py`

```python
class ProfessionalSymbolScorer:
    """
    Ranks symbols by comprehensive score:
    - Expected value (edge)
    - Realized PnL (recent performance)
    - Confidence (model confidence)
    - Correlation (diversification penalty)
    - Drawdown (risk penalty)
    """
    
    def __init__(self, config, ml_model, position_manager):
        self.ml_model = ml_model
        self.position_manager = position_manager
        self.history_lookback_days = 30
    
    def score_symbol(self, symbol: str) -> float:
        """
        Calculate final_score for symbol ranking
        Higher = better candidate for portfolio
        """
        scores = {
            'expected_edge': self._calc_expected_edge(symbol),
            'realized_pnl': self._calc_realized_pnl(symbol),
            'confidence': self._calc_confidence(symbol),
            'correlation': self._calc_correlation_penalty(symbol),
            'drawdown': self._calc_drawdown_penalty(symbol),
        }
        
        # Weighted sum
        final_score = (
            scores['expected_edge'] * 0.40
            + scores['realized_pnl'] * 0.25
            + scores['confidence'] * 0.20
            - scores['correlation'] * 0.10
            - scores['drawdown'] * 0.05
        )
        
        return final_score
    
    def _calc_expected_edge(self, symbol: str) -> float:
        """
        Expected edge from ML model
        Range: 0.0 to 1.0
        """
        try:
            pred = self.ml_model.predict(symbol)
            edge = pred.get('expected_move', 0.0)
            confidence = pred.get('confidence', 0.0)
            
            # Weight by confidence
            weighted_edge = edge * confidence
            return min(1.0, weighted_edge)
        except:
            return 0.0
    
    def _calc_realized_pnl(self, symbol: str) -> float:
        """
        Recent realized PnL (last 30 days)
        Normalize to 0.0-1.0
        """
        try:
            trades = self.position_manager.get_trades(symbol, days=30)
            if not trades:
                return 0.5  # Neutral (no history)
            
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
            
            # Normalize
            realized_pnl_score = min(1.0, max(0.0, win_rate))
            return realized_pnl_score
        except:
            return 0.5
    
    def _calc_confidence(self, symbol: str) -> float:
        """
        Model confidence in its prediction
        Range: 0.0 to 1.0
        """
        try:
            pred = self.ml_model.predict(symbol)
            confidence = pred.get('confidence', 0.0)
            return min(1.0, max(0.0, confidence))
        except:
            return 0.5
    
    def _calc_correlation_penalty(self, symbol: str) -> float:
        """
        Penalty for correlation with existing holdings
        If correlated, reduce score
        Range: 0.0 (no correlation) to 1.0 (highly correlated)
        """
        try:
            active_symbols = self.position_manager.get_active_symbols()
            correlations = []
            
            for existing_symbol in active_symbols:
                corr = self._calc_symbol_correlation(symbol, existing_symbol)
                correlations.append(corr)
            
            if not correlations:
                return 0.0  # No existing positions, no penalty
            
            # Average correlation
            avg_corr = sum(correlations) / len(correlations)
            penalty = max(0.0, avg_corr - 0.3)  # Start penalty above 0.3
            return min(1.0, penalty)
        except:
            return 0.0
    
    def _calc_drawdown_penalty(self, symbol: str) -> float:
        """
        Penalty for recent drawdown
        Higher recent loss = higher penalty
        Range: 0.0 (no drawdown) to 1.0 (severe drawdown)
        """
        try:
            trades = self.position_manager.get_trades(symbol, days=30)
            if not trades:
                return 0.0
            
            worst_loss = min(t['pnl'] for t in trades)
            loss_rate = abs(worst_loss) / sum(t['quantity'] * t['entry_price'] for t in trades)
            
            # Penalize if loss > 5%
            penalty = max(0.0, loss_rate - 0.05)
            return min(1.0, penalty * 2)  # Scale up
        except:
            return 0.0
    
    def _calc_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols (0.0 to 1.0)"""
        try:
            # Get recent returns for both symbols
            returns1 = self._get_symbol_returns(symbol1, days=30)
            returns2 = self._get_symbol_returns(symbol2, days=30)
            
            if not returns1 or not returns2:
                return 0.0
            
            # Calculate Pearson correlation
            import numpy as np
            corr = np.corrcoef(returns1, returns2)[0, 1]
            
            # Normalize to 0-1
            return max(0.0, min(1.0, (corr + 1) / 2))
        except:
            return 0.5
    
    def _get_symbol_returns(self, symbol: str, days: int) -> List[float]:
        """Get daily returns for symbol"""
        try:
            # Fetch OHLCV data
            candles = self.position_manager.get_candles(symbol, days=days)
            
            closes = [float(c['close']) for c in candles]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            
            return returns
        except:
            return []
    
    def rank_symbols(self, symbols: List[str]) -> List[Tuple[str, float]]:
        """
        Rank symbols by final_score
        Returns: [(symbol, score), ...] sorted descending
        """
        scores = [(symbol, self.score_symbol(symbol)) for symbol in symbols]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
```

---

## Phase 3: Advanced (Dynamic Universe Based on Volatility)

### Objective
Adapt universe size dynamically based on market volatility regime.

### Algorithm

```
if volatility_regime == HIGH:
    universe_cap = 5–7 symbols  (more opportunities in volatility)
elif volatility_regime == NORMAL:
    universe_cap = 3–5 symbols
elif volatility_regime == LOW:
    universe_cap = 2–3 symbols  (consolidate into best setups)
else:  # EXTREME
    universe_cap = 1–2 symbols  (only core positions)
```

### Implementation

**File**: `core/symbol_rotation.py` or `core/meta_controller.py`

```python
class DynamicUniverseManager:
    """
    Adjust universe size based on volatility regime
    """
    
    def __init__(self, config, volatility_detector):
        self.volatility_detector = volatility_detector
        self.universe_caps = {
            'extreme': (1, 2),    # 1-2 symbols
            'high': (5, 7),       # 5-7 symbols
            'normal': (3, 5),     # 3-5 symbols
            'low': (2, 3),        # 2-3 symbols
        }
    
    def get_universe_cap(self) -> Tuple[int, int]:
        """
        Get (min, max) for current volatility regime
        """
        regime = self.volatility_detector.get_regime()
        
        return self.universe_caps.get(regime, (3, 5))
    
    def get_max_positions(self) -> int:
        """Get maximum active positions for current regime"""
        min_cap, max_cap = self.get_universe_cap()
        return max_cap
    
    def get_min_positions(self) -> int:
        """Get minimum active positions for current regime"""
        min_cap, max_cap = self.get_universe_cap()
        return min_cap
    
    def should_add_position(self, current_count: int) -> bool:
        """Should we add a new position?"""
        max_cap = self.get_max_positions()
        return current_count < max_cap
    
    def should_remove_position(self, current_count: int) -> bool:
        """Should we remove a position?"""
        min_cap = self.get_min_positions()
        return current_count > min_cap
    
    def adjust_universe(self, current_symbols: List[str], candidates: List[Tuple[str, float]]) -> List[str]:
        """
        Adjust active universe to match volatility regime
        
        Logic:
        1. If current_count < min_cap: add best candidates
        2. If current_count > max_cap: remove worst performers
        3. Otherwise: rotate if replacement_multiplier satisfied
        """
        min_cap, max_cap = self.get_universe_cap()
        current_count = len(current_symbols)
        
        # Phase 1: Enforce minimum
        if current_count < min_cap:
            # Add best candidates
            needed = min_cap - current_count
            candidates_not_active = [c for c in candidates if c[0] not in current_symbols]
            to_add = candidates_not_active[:needed]
            return current_symbols + [s for s, _ in to_add]
        
        # Phase 2: Enforce maximum
        elif current_count > max_cap:
            # Remove worst performers
            # (implementation depends on performance metrics)
            return current_symbols[:max_cap]
        
        # Phase 3: Maintain current
        else:
            return current_symbols

# Integration into meta_controller:
def _should_rotate_symbol(self, current_symbol: str, candidate_symbol: str) -> bool:
    """
    Decide if we should rotate out current_symbol for candidate
    
    Criteria:
    1. Universe allows it (not at minimum)
    2. Replacement multiplier satisfied (new_score > 1.10 * old_score)
    3. Not bootstrap locked
    """
    # Check bootstrap lock
    if self.rotation_manager.is_bootstrap_locked():
        return False
    
    # Check universe cap
    if not self.universe_manager.should_add_position(len(self.active_symbols)):
        return False  # At capacity, must remove current first
    
    # Check replacement multiplier
    current_score = self.symbol_scorer.score_symbol(current_symbol)
    candidate_score = self.symbol_scorer.score_symbol(candidate_symbol)
    
    replacement_multiplier = 1.10
    return candidate_score > (current_score * replacement_multiplier)
```

---

## Implementation Roadmap

### Week 1: Phase 1 (Safe Upgrade)
```
Day 1-2: Implement soft bootstrap lock
Day 3-4: Implement screener (20-30 proposals)
Day 5: Add universe cap (3-5 symbols)
Day 6-7: Make replacement multiplier configurable
```

### Week 2: Phase 2 (Professional Mode)
```
Day 1-3: Implement expected edge calculation
Day 4: Implement realized PnL weighting
Day 5: Implement correlation penalty
Day 6: Implement drawdown penalty
Day 7: Integrate scoring system
```

### Week 3: Phase 3 (Advanced)
```
Day 1-2: Wire volatility regime detection
Day 3-4: Implement dynamic universe sizing
Day 5-6: Test integration with all phases
Day 7: Performance validation
```

---

## Current Files to Modify

1. **`core/meta_controller.py`** (8293, 8837)
   - Replace bootstrap hard lock with soft lock

2. **`core/exchange_client.py`**
   - Add symbol discovery/screener methods

3. **`core/config.py`**
   - Add configuration for:
     - `SYMBOL_REPLACEMENT_MULTIPLIER = 1.10`
     - `BOOTSTRAP_SOFT_LOCK_DURATION_SEC = 3600`
     - Universe caps per regime

4. **NEW: `core/symbol_rotation.py`**
   - SymbolRotationManager
   - UniverseManager
   - DynamicUniverseManager

5. **NEW: `core/symbol_screener.py`**
   - SymbolScreener (20-30 proposals)

6. **NEW: `core/symbol_scorer_professional.py`**
   - ProfessionalSymbolScorer
   - Expected edge, PnL, confidence, correlation, drawdown

---

## Status Summary

| Phase | Status | Effort | Timeline |
|-------|--------|--------|----------|
| Phase 1 | ❌ Not Implemented | ~2-3 days | Week 1 |
| Phase 2 | ❌ Not Implemented | ~3-4 days | Week 2 |
| Phase 3 | ❌ Not Implemented | ~2-3 days | Week 3 |

---

## Recommendation

**Start with Phase 1** (Safe Upgrade) first:
1. Remove bootstrap hard lock (better for testing)
2. Enable screener proposals (20-30 pairs available)
3. Enforce universe cap (3-5 active)
4. Configure replacement multiplier (1.10)

Then proceed to Phase 2 (Professional Mode) for better signal.

Finally Phase 3 (Advanced) for volatility-aware sizing.

