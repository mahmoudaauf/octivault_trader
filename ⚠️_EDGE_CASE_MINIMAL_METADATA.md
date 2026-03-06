# ⚠️ EDGE CASE: Minimal Metadata in accepted_symbols

**Date**: March 7, 2026  
**Issue**: Some discovery agents populate `accepted_symbols` without "quote_volume" or "spread" keys  
**Status**: ✅ HANDLED  

---

## The Edge Case

### Symptom
Our seed fix and some agents populate `accepted_symbols` like this:

```python
{
    "BTCUSDT": {"status": "TRADING", "notional": 10},
    "ETHUSDT": {"status": "TRADING", "notional": 10},
}
```

**Missing keys**:
- ❌ `quote_volume`
- ❌ `spread`

Our original scoring fix expected these keys to exist (or have `.get()` fallback).

### Why This Matters

If `quote_volume` is missing:
```python
quote_volume = float(symbol_info.get("quote_volume", 0) or 0)  # Gets 0
liquidity_score = min(0 / 100000, 1.0) * ...  # Becomes 0.0 (minimum!)
```

This would give ALL seeded symbols a liquidity_score of **0.0** instead of neutral **0.5**.

While this wouldn't break the system, it would:
- Undervalue symbols without metadata
- Give them artificially low composite scores
- Bias UURE toward symbols with full metadata
- Create inconsistent scoring

---

## The Solution

### Enhanced Robustness (Now Applied)

We improved the fallback logic to:

```python
liquidity_score = 0.5  # Default neutral liquidity

try:
    symbol_info = self.accepted_symbols.get(symbol, {})
    if isinstance(symbol_info, dict):
        # Try multiple key names for compatibility
        quote_volume = float(
            symbol_info.get("quote_volume")           # Main key
            or symbol_info.get("volume")              # Alternative name
            or symbol_info.get("24h_volume")          # Another variant
            or symbol_info.get("quote_volume_usd", 0) # USD-denominated
            or 0                                       # Default fallback
        )
        spread = float(
            symbol_info.get("spread")                 # Main key
            or symbol_info.get("bid_ask_spread", 0.01) # Alternative
            or 0.01                                   # Default (1%)
        )
        
        # Only compute if we have MEANINGFUL volume
        # If volume is 0 (missing), keep neutral 0.5
        if quote_volume > 0:
            liquidity_score = min(quote_volume / 100000, 1.0) * max(0, 1.0 - min(spread, 0.05))
        # else: keep liquidity_score = 0.5 (neutral)
        
except (TypeError, ValueError, AttributeError):
    # Explicit handling of specific error types
    liquidity_score = 0.5
```

### Key Improvements

1. **Multiple key names**: Checks 4 different keys for volume
2. **Zero detection**: If volume is 0, keeps neutral 0.5 (not computed as 0)
3. **Specific exceptions**: Catches TypeError, ValueError, AttributeError explicitly
4. **Clear fallback**: Any error → neutral 0.5

---

## Behavior Matrix

### Before (Original Fix)
| Symbol Source | quote_volume | Result | Problem |
|---|---|---|---|
| Symbol screener | 1,500,000 | Correct score (~0.75) | ✅ Works |
| Seeded symbols | Missing (0) | score_0.0 | ❌ Undervalued |
| Other agents | 0 or missing | score 0.0 | ❌ Unfair bias |

### After (Enhanced Fix)
| Symbol Source | quote_volume | Result | Status |
|---|---|---|---|
| Symbol screener | 1,500,000 | Correct score (~0.75) | ✅ Works |
| Seeded symbols | Missing | Neutral score (0.5) | ✅ Fair |
| Other agents | 0 or missing | Neutral score (0.5) | ✅ Fair |

---

## Code Comparison

### Original Fix
```python
quote_volume = float(symbol_info.get("quote_volume", 0) or symbol_info.get("volume", 0) or 0)
spread = float(symbol_info.get("spread", 0.01) or 0.01)
liquidity_score = min(quote_volume / 100000, 1.0) * max(0, 1.0 - min(spread, 0.05))
```

**Problem**: If quote_volume = 0, score is always 0

### Enhanced Fix
```python
quote_volume = float(
    symbol_info.get("quote_volume")
    or symbol_info.get("volume")
    or symbol_info.get("24h_volume")
    or symbol_info.get("quote_volume_usd", 0)
    or 0
)

if quote_volume > 0:
    liquidity_score = min(quote_volume / 100000, 1.0) * max(0, 1.0 - min(spread, 0.05))
else:
    liquidity_score = 0.5  # Neutral, not 0
```

**Improvement**: Zero volume → neutral score, not penalty

---

## Example Scenarios

### Scenario 1: Symbol Screener (Good Metadata)

**Input**:
```python
{
    "symbol": "ADAUSDT",
    "quote_volume": 1_500_000,
    "spread": 0.002,
}
```

**Processing**:
```
quote_volume = 1_500_000 (found in first key)
spread = 0.002 (found)
liquidity_score = min(1500000 / 100000, 1.0) * max(0, 1.0 - 0.002)
                = min(15.0, 1.0) * 0.998
                = 1.0 * 0.998
                = 0.998  (high liquidity score) ✅
```

### Scenario 2: Seeded Symbol (Minimal Metadata)

**Input**:
```python
{
    "status": "TRADING",
    "notional": 10,
}
```

**Processing**:
```
quote_volume = 0 (not found in any key)
Zero detection: quote_volume > 0? NO
liquidity_score = 0.5  (neutral) ✅ Fair!
```

### Scenario 3: Missing Data After Type Conversion Error

**Input**:
```python
{
    "quote_volume": "invalid_string",  # Can't convert to float
}
```

**Processing**:
```
try:
    float("invalid_string") → ValueError
except ValueError:
    liquidity_score = 0.5  (neutral fallback) ✅ No crash!
```

---

## Scoring Consistency

With this enhancement, any symbol scores consistently:

**Formula**:
```
composite = 0.40 * conviction + 0.20 * volatility + 0.20 * momentum + 0.20 * liquidity

If liquidity data missing:
  liquidity = 0.5 (neutral)
  composite = 0.40 * conviction + 0.20 * volatility + 0.20 * momentum + 0.20 * 0.5
  composite = 0.40 * conviction + 0.20 * volatility + 0.20 * momentum + 0.10
```

The system still scores, just with less liquidity precision. This is fair and reasonable.

---

## Which Discovery Agents Populate Metadata?

### Full Metadata (quote_volume, spread, etc.)
- ✅ `SymbolScreener` - queries ticker info, includes 24h volume
- ✅ `OpportunityRanker` - scores symbols with full stats
- ✅ Professional data sources

### Minimal Metadata (just status, notional)
- ⚠️ Our seed function (bootstrapping guard)
- ⚠️ Manual symbol addition
- ⚠️ Some legacy agents
- ⚠️ Fast-path discovery (just test if tradable)

### No Metadata (blank dict)
- ⚠️ Possible in edge cases

All cases are now handled gracefully!

---

## Impact Assessment

**Before Enhancement**:
- ❌ Seeded symbols undervalued (liquidity = 0.0)
- ❌ Bias toward symbols with metadata
- ✅ But system still worked (didn't crash)

**After Enhancement**:
- ✅ Seeded symbols valued fairly (liquidity = 0.5)
- ✅ No bias between data sources
- ✅ System more robust
- ✅ Still works with full metadata (no change)

---

## Testing the Edge Case

### Test 1: Minimal Metadata Scores

```bash
# Check that seeded symbols still get scored
grep "Scored.*candidates" logs/octivault_trader.log

# Should show BTCUSDT, ETHUSDT, etc. in final universe
grep "Universe hard-replaced" logs/octivault_trader.log
```

### Test 2: Full Metadata Takes Precedence

```bash
# Symbol screener results should rank higher due to liquidity data
grep "Top 5:" logs/octivault_trader.log

# Should show mix of seeded + discovered symbols
# But discovered ones rank higher in order
```

### Test 3: No Crash on Bad Data

```bash
# Check for no exceptions
grep "ERROR\|Exception\|Traceback" logs/octivault_trader.log | grep "score"

# Should return nothing (no errors)
```

---

## Documentation Updated

The `get_unified_score()` method now includes:
1. Comment explaining the edge case
2. Multiple key name fallbacks
3. Zero-volume detection to keep neutral score
4. Specific exception handling
5. Examples in code

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Handles minimal metadata | ✅ Yes | Keeps neutral 0.5 |
| Handles missing volume | ✅ Yes | Doesn't penalize |
| Handles conversion errors | ✅ Yes | Falls back safely |
| Preserves good data | ✅ Yes | Full metadata still used |
| Fair to all sources | ✅ Yes | No bias |
| No crashes | ✅ Yes | All errors caught |

---

**System now handles ALL metadata scenarios gracefully!**
