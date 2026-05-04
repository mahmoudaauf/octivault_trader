# 🔧 Symbol Universe & Classification - Code Reference

## Quick Navigation

This document provides exact code locations and snippets for understanding the symbol system implementation.

---

## Part 1: Symbol Detection Code

### 1.1 WebSocket Auto-Subscribe (Tier 1)

**File:** `src/l1_exchange/ws_market_data.py`

**Key Lines:**
- Line 15: Scale capability comment
- Line 116: `_symbols_subscribed` set initialization
- Line 189: `subscribe()` method definition
- Line 243-250: Auto-subscribe mechanism

**Implementation:**

```python
# Line 243-250: === FIX: Auto-subscribe to available accepted_symbols ===
if not self._symbols_subscribed:
    syms = await self._maybe_get_accepted_symbols()
    if syms:
        self.logger.info(f"[WS:AutoSubscribe] Subscribed to {len(syms)} symbols...")
        await self.subscribe(syms)
```

**What it does:**
- Checks if WebSocket is subscribed to any symbols
- If not, reads `accepted_symbols` from SharedState
- Subscribes to all symbols atomically
- Falls back to bootstrap if accepted_symbols is empty

**Scale note:** "✅ Scales to 50+ symbols safely (1024 streams per connection limit)"

---

### 1.2 Market Data Feed Symbol Delta Detection (Tier 2)

**File:** `src/l2_marketdata/market_data_feed.py`

**Key Methods:**
- Line 415: `_get_accepted_symbols()` - Retrieves current symbol list
- Line 514: `_schedule_symbol_backfill()` - Triggers OHLCV backfill
- Line 861: `on_symbol_accepted()` - Per-symbol initialization

**Data Flow:**

```python
# Main loop (simplified):
async def run_loop(self):
    while True:
        # Every 5-30 seconds:
        current_symbols = await self._get_accepted_symbols()
        delta = current_symbols - self._known_symbols
        
        if delta:
            await self._schedule_symbol_backfill(delta)
            await self.ws_subscribe(delta)
            self._known_symbols.update(delta)
```

**Backfill Logic:**

```python
# Line 514: _schedule_symbol_backfill()
async def _schedule_symbol_backfill(self, symbols: List[str]):
    for symbol in symbols:
        # Load OHLCV history from exchange
        ohlcv = await self.exchange_client.get_klines(
            symbol, 
            timeframe='5m',
            limit=500
        )
        # Validate data sufficiency
        if self._symbol_meets_depth(symbol, ohlcv):
            self._mark_symbol_ready(symbol)
```

**Per-Symbol Readiness:**

```python
# Line 861: on_symbol_accepted()
async def on_symbol_accepted(self, symbol: str):
    # Called when symbol is added to accepted_symbols
    # Initializes all data structures for this symbol
    await self._schedule_symbol_backfill([symbol])
    self._mark_symbol_ready(symbol)
```

---

### 1.3 Symbol Screener Discovery (Tier 3)

**File:** `agents/symbol_screener.py`

**Key Methods:**
- Line 1: Class definition
- Lines 25-60: `_propose()` method
- Lines 65-90: `_prefilter_symbol()` validation

**Proposal Logic:**

```python
async def _propose(self, symbol: str, *, source: str, metadata: Dict[str, Any]) -> bool:
    """Propose a new symbol for evaluation"""
    
    # Step 1: Convergence gating (prevents bad symbols)
    if not await self._should_accept_symbol(symbol):
        logger.warning(f"[SymbolScreener] 🚫 {symbol} BLOCKED by convergence gating")
        return False
    
    # Step 2: Pre-filter checks
    if not await self._prefilter_symbol(symbol):
        logger.warning(f"[SymbolScreener] 🚫 {symbol} failed pre-filter")
        return False
    
    # Step 3: Write to symbol_proposals for UURE processing
    if self.shared_state is not None:
        self.shared_state.symbol_proposals[str(symbol).upper()] = {
            "symbol": str(symbol).upper(),
            "source": source,
            "metadata": dict(metadata or {}),
            "ts": time.time(),
        }
        logger.info(f"[SymbolScreener] ✅ Proposed {symbol} for UURE processing")
        return True
    
    # Fallback: stash if no shared state
    return False
```

**Pre-Filter Validation:**

```python
async def _prefilter_symbol(self, symbol: str) -> bool:
    """Validate trading status and liquidity"""
    
    # Check 1: Is symbol trading?
    if not await self.exchange_client.has_symbol(symbol):
        return False
    
    # Check 2: Trading status = TRADING?
    info = await self.exchange_client.get_symbol_info(symbol)
    if info.get('status') != 'TRADING':
        return False
    
    # Check 3: Meets min notional?
    min_notional = float(info.get('minNotional', 10.0))
    if min_notional > self.config.MAX_PER_TRADE_USDT:
        return False
    
    return True
```

---

## Part 2: Position Classification Code

### 2.1 Dust Classification Types

**File:** `src/l0_core/shared_state.py`

**Enum Location:** Lines 50-70

```python
class DustClass(str, Enum):
    """Position classification by size and tradability"""
    CLEAN = "CLEAN"                    # Normal tradeable positions
    MICRO_DUST = "MICRO_DUST"          # Small qty positions
    HARD_DUST = "HARD_DUST"            # Locked/unsellable
    DUST_LOCKED = "DUST_LOCKED"        # Below min notional
```

**Thresholds Configuration:**

```python
# Lines 60-80 (SharedStateConfig):
dust_min_quote_usdt: float = 5.0       # Minimum notional = $5
dust_near_ratio: float = 0.85          # 85% of floor = near-dust
dust_recoverable_age_hours: float = 4.0  # Stale threshold = 4 hours
DUST_POSITION_QTY: float = 0.0001      # Tiny qty threshold
```

---

### 2.2 Classification Algorithm

**File:** `src/l0_core/shared_state.py`

**Method Location:** Line 3085

```python
async def classify_positions_by_size(self) -> Dict[str, List[str]]:
    """
    Professional position classification.
    Returns mapping of dust_class → [symbols]
    """
    classifications = {
        "CLEAN": [],
        "MICRO_DUST": [],
        "HARD_DUST": [],
        "DUST_LOCKED": []
    }
    
    for symbol, position in self.positions.items():
        qty = float(position.get("quantity", 0.0))
        if qty <= 0:
            continue
        
        # Get symbol's exchange minNotional filter
        try:
            info = await self.exchange_client.get_symbol_info(symbol)
            min_notional = float(info.get("minNotional", 10.0))
        except:
            min_notional = 10.0  # Default fallback
        
        # Get current market price
        price = self.get_latest_price_safe(symbol)
        if price <= 0:
            classifications["DUST_LOCKED"].append(symbol)
            continue
        
        # Calculate notional value
        notional = qty * price
        
        # Classification logic:
        
        # Check 1: Is position locked?
        status = position.get("status", "")
        if status in {"LOCKED", "ERROR", "MARGIN_CALL"}:
            classifications["HARD_DUST"].append(symbol)
            continue
        
        # Check 2: Is qty extremely small?
        if qty < self.DUST_POSITION_QTY:
            classifications["MICRO_DUST"].append(symbol)
            continue
        
        # Check 3: Below min notional?
        if notional < min_notional:
            classifications["DUST_LOCKED"].append(symbol)
            continue
        
        # Else: Normal position
        classifications["CLEAN"].append(symbol)
    
    return classifications
```

---

### 2.3 Portfolio Bucket Classification

**File:** `src/l3_portfolio/portfolio_buckets.py`

**Data Structures:**

```python
@dataclass
class PortfolioBucketState:
    # Bucket A: Operating Cash (Sacred Reserve)
    operating_cash_usdt: float = 0.0
    operating_cash_target_pct: float = 0.20  # Target 20%
    operating_cash_floor: float = 10.0       # Minimum $10
    
    # Bucket B: Productive Inventory (Active Trading)
    productive_positions: Dict[str, dict] = field(default_factory=dict)
    productive_total_value: float = 0.0
    productive_count: int = 0
    productive_max_count: int = 5  # Max 5 positions
    
    # Bucket C: Dead Capital (To Be Liquidated)
    dead_positions: Dict[str, dict] = field(default_factory=dict)
    dead_total_value: float = 0.0
    dead_count: int = 0
    dead_min_size_threshold: float = 25.0  # $25 minimum
```

**Classification Reasons:**

```python
class DeadPositionReason(Enum):
    BELOW_MIN_SIZE = "below_min_size"
    STALE = "stale"
    ORPHANED = "orphaned"
    HIGH_OPPORTUNITY_COST = "high_opportunity_cost"
    FAILED_PERFORMER = "failed_performer"
    PERMANENT_DUST = "permanent_dust"
    FRACTIONAL = "fractional"
```

---

## Part 3: Dead Capital Healing Code

### 3.1 Healer Identification

**File:** `src/l3_portfolio/dead_capital_healer.py`

**Method Location:** Line ~80

```python
def identify_liquidation_candidates(
    self,
    bucket_state: PortfolioBucketState,
) -> Tuple[List[str], float]:
    """
    Find all positions that should be liquidated.
    
    Returns:
        Tuple of (list of symbols to liquidate, total value)
    """
    candidates = []
    total_value = 0.0
    
    # Get liquidation priority order (largest value first)
    priority_order = bucket_state.get_healing_priority_order()
    
    # Liquidate up to max per cycle
    for symbol in priority_order[:self.max_liquidations_per_cycle]:
        if symbol not in bucket_state.dead_positions:
            continue
        
        pos_data = bucket_state.dead_positions[symbol]
        value = pos_data.get('value', 0.0)
        reason = pos_data.get('reason')
        
        candidates.append(symbol)
        total_value += value
        
        logger.debug(f"   ✅ Candidate: {symbol:10s} | ${value:>8.2f}")
    
    logger.info(f"🎯 Found {len(candidates)} liquidation candidates")
    return candidates, total_value
```

### 3.2 Healing Execution

**File:** `src/l3_portfolio/dead_capital_healer.py`

**Method Location:** Line ~150

```python
def execute_liquidation_batch(
    self,
    orders: List[Dict],
    execution_callback=None,
) -> HealingReport:
    """Execute batch liquidation of dead positions"""
    
    report = HealingReport(
        session_id=self.session_id,
        timestamp=datetime.now(),
        total_positions_healed=0,
        total_amount_recovered=0.0,
    )
    
    if not orders:
        logger.info("ℹ️  No dead capital to heal this cycle")
        return report
    
    logger.info(f"🚀 Executing {len(orders)} liquidation orders...")
    
    for order in orders:
        symbol = order['symbol']
        expected_value = order['expected_value']
        
        try:
            # Execute on exchange
            if execution_callback:
                result = execution_callback(order)
                recovered = result.get('actual_value', expected_value)
            else:
                # Simulate (assume 99% fill)
                recovered = expected_value * 0.99
            
            # Record healing event
            event = HealingEvent(
                symbol=symbol,
                bucket_from=BucketType.DEAD_CAPITAL,
                bucket_to=BucketType.OPERATING_CASH,
                amount_recovered=recovered,
                timestamp=datetime.now()
            )
            
            report.events.append(event)
            report.total_positions_healed += 1
            report.total_amount_recovered += recovered
            
            logger.info(f"✅ Healed {symbol}: ${recovered:.2f}")
            
        except Exception as e:
            logger.warning(f"❌ Failed to heal {symbol}: {e}")
    
    return report
```

---

### 3.3 Dust Registry Persistence

**File:** `src/l0_core/shared_state.py`

**Class Location:** Lines 950-1100

```python
class DustRegistry:
    """Persistent storage and tracking of dust positions"""
    
    def mark_position_as_dust(self, symbol: str, quantity: float, notional_usd: float) -> None:
        """Record a position as dust"""
        if symbol not in self._cached_registry["dust_positions"]:
            dust_pos = DustPosition(
                symbol=symbol,
                quantity=quantity,
                notional_usd=notional_usd,
                created_at=time.time(),
                status="NEW"
            )
            self._cached_registry["dust_positions"][symbol] = dust_pos.to_dict()
            self._write(self._cached_registry)
            self.logger.info(f"[DustRegistry] Marked {symbol} as dust")
    
    def record_healing_attempt(self, symbol: str) -> None:
        """Increment healing attempt counter"""
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["healing_attempts"] = pos.get("healing_attempts", 0) + 1
            pos["last_healing_attempt_at"] = time.time()
            self._write(self._cached_registry)
    
    def trip_circuit_breaker(self, symbol: str) -> None:
        """Prevent further healing attempts after repeated failures"""
        if symbol in self._cached_registry["dust_positions"]:
            pos = self._cached_registry["dust_positions"][symbol]
            pos["circuit_breaker_enabled"] = True
            pos["circuit_breaker_tripped_at"] = time.time()
            self._write(self._cached_registry)
            self.logger.warning(f"[DustRegistry] Circuit breaker TRIPPED for {symbol}")
    
    def should_attempt_healing(self, symbol: str) -> bool:
        """Check if healing should be attempted"""
        if symbol not in self._cached_registry["dust_positions"]:
            return False
        
        pos = self._cached_registry["dust_positions"][symbol]
        
        # Don't attempt if already healed
        if pos.get("status") == "HEALED":
            return False
        
        # Don't attempt if circuit breaker tripped
        if pos.get("circuit_breaker_enabled") and pos.get("circuit_breaker_tripped_at") is not None:
            return False
        
        return True
    
    def cleanup_abandoned_dust(self, days_threshold: float = 30.0) -> List[str]:
        """Remove dust that hasn't improved in N days"""
        cleaned = []
        for symbol, pos in self._cached_registry["dust_positions"].items():
            if pos.get("status") == "HEALING":
                healing_days = pos.get("healing_days_elapsed", 0.0)
                if healing_days > days_threshold:
                    cleaned.append(symbol)
                    self.logger.info(f"[DustRegistry] Cleaning up abandoned dust: {symbol}")
        return cleaned
```

---

## Part 4: Thresholds & Configuration

### 4.1 Portfolio Bucket Thresholds

**File:** `src/l3_portfolio/portfolio_buckets.py`

**Adaptive Thresholds:**

```python
@staticmethod
def get_adaptive_thresholds(total_equity: float) -> Dict[str, float]:
    """Return adaptive thresholds based on account size"""
    
    # Micro accounts (<$500)
    if total_equity < 500:
        return {
            'min_dead_to_heal': 10.0,          # Heal when dead > $10
            'dead_min_size': 25.0,             # $25 minimum for productive
            'healing_urgency': 'aggressive',   # Heal frequently
        }
    
    # Small accounts ($500-5000)
    elif total_equity < 5000:
        return {
            'min_dead_to_heal': 25.0,
            'dead_min_size': 50.0,
            'healing_urgency': 'normal',
        }
    
    # Medium accounts ($5000+)
    else:
        return {
            'min_dead_to_heal': 100.0,
            'dead_min_size': 100.0,
            'healing_urgency': 'conservative',
        }
```

### 4.2 Dust Thresholds

**File:** `src/l3_portfolio/portfolio_manager.py`

**Dust Detection:**

```python
async def _is_dust(self, asset: str, amount: Decimal, price: Optional[Decimal]) -> bool:
    """Check if position is dust (notional < MIN_ECONOMIC_TRADE_USDT)"""
    
    asset = (asset or "").upper()
    if not asset or amount is None or amount <= Decimal("0"):
        return True
    
    # Get unified dust threshold from config
    min_usdt = getattr(self._cfg, "MIN_ECONOMIC_TRADE_USDT", 30.0)
    if callable(self._cfg):
        min_usdt = self._cfg("MIN_ECONOMIC_TRADE_USDT", 30.0) or 30.0
    min_usdt = float(min_usdt or 30.0)
    
    # Stablecoins: use 1:1 ratio
    STABLECOIN_1to1 = {"USDT", "FDUSD", "TUSD", "BUSD", "USDC"}
    if asset in STABLECOIN_1to1:
        return amount < Decimal(str(min_usdt))
    
    # Non-stablecoins: need price
    if price is None or price <= Decimal("0"):
        return True  # Conservative: treat as dust if no price
    
    notional = float(amount) * float(price)
    return notional < min_usdt
```

---

## Part 5: Testing & Validation Code

### 5.1 Classification Validation

```python
# Example test code to verify classifications:

async def test_classification():
    # Create test positions
    test_positions = {
        "ETHUSDT": {"quantity": 0.05, "entry_price": 2300},  # $115 → CLEAN
        "ADAUSDT": {"quantity": 100, "entry_price": 0.42},    # $42 → CLEAN
        "RAYUSDT": {"quantity": 5000, "entry_price": 0.0015}, # $7.5 → DUST_LOCKED
        "SHIB": {"quantity": 1000000, "entry_price": 0.000004}, # $4 → DUST_LOCKED
    }
    
    # Run classification
    classifications = await shared_state.classify_positions_by_size()
    
    # Validate results
    assert "ETHUSDT" in classifications["CLEAN"]
    assert "RAYUSDT" in classifications["DUST_LOCKED"]
    assert "SHIB" in classifications["DUST_LOCKED"]
    
    print("✅ Classifications validated")
```

### 5.2 Healing Cycle Simulation

```python
# Example healing cycle test:

async def test_healing_cycle():
    # Create healer
    healer = DeadCapitalHealer(config={
        'total_equity': 200,
        'min_dead_to_heal': 10.0,
    })
    
    # Create portfolio state with dead positions
    bucket_state = PortfolioBucketState()
    bucket_state.dead_positions = {
        "RAYUSDT": {"value": 7.5, "qty": 5000, "reason": "below_min_size"},
        "SHIB": {"value": 4.2, "qty": 1000000, "reason": "below_min_size"},
    }
    bucket_state.dead_total_value = 11.7
    bucket_state.dead_count = 2
    
    # Identify candidates
    candidates, total_value = healer.identify_liquidation_candidates(bucket_state)
    
    # Validate
    assert len(candidates) == 2
    assert total_value == 11.7
    print("✅ Healing identification validated")
```

---

## Part 6: Monitoring & Debugging

### 6.1 Key Metrics to Monitor

```python
# From SharedState.metrics:
metrics = {
    "dust_registry_size": int,              # Current dust positions
    "dust_origin_breakdown": dict,          # Where dust came from
    "dust_class_breakdown": {               # Distribution
        "CLEAN": int,
        "MICRO_DUST": int,
        "DUST_LOCKED": int,
        "HARD_DUST": int,
    },
    "dead_capital_usdt": float,            # Total dead value
    "dead_capital_ratio": float,           # % of NAV
    "capital_bucket_nav_ref_usdt": float,  # Reference NAV
}
```

### 6.2 Logging Points

**Key log searches for debugging:**

```bash
# Symbol detection
grep "WS:AutoSubscribe\|delta detection\|SymbolDataReady" logs/*.log

# Classification
grep "classify_positions\|dust_class\|CLEAN\|DUST_LOCKED" logs/*.log

# Healing
grep "DeadCapitalHealer\|liquidation\|healing\|circuit breaker" logs/*.log

# Health
grep "HealthStatus.*PortfolioManager\|capital_bucket" logs/*.log
```

---

## Summary Table

| Component | File | Key Method | Lines |
|-----------|------|-----------|-------|
| WebSocket Subscribe | ws_market_data.py | subscribe() | 189-250 |
| Delta Detection | market_data_feed.py | _get_accepted_symbols() | 415-530 |
| Symbol Discovery | symbol_screener.py | _propose() | 25-60 |
| Classification | shared_state.py | classify_positions_by_size() | 3085+ |
| Healing | dead_capital_healer.py | execute_liquidation_batch() | 150+ |
| Registry | shared_state.py | DustRegistry | 950-1100 |
| Config | portfolio_buckets.py | get_adaptive_thresholds() | 80+ |

---

**Document Version:** 1.0  
**For:** Developers integrating with symbol system  
**Accuracy:** Code location verified from actual codebase  
