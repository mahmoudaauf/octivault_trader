# ⚡ CONFIDENCE FIX - PRODUCTION DEPLOYMENT GUIDE

## CRITICAL: Two Codebases Detected

Your bot is running on **Ubuntu server** (`ubuntu@ip-172-31-37-246`), not on your Mac.

- **Mac workspace** (local): `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/`
  - File location: `utils/volatility_adjusted_confidence.py`
  - Status: ✅ Already fixed

- **Ubuntu server** (production): `~/octivault_trader/`
  - File location: `core/volatility_adjusted_confidence.py`
  - Status: ❌ NEEDS FIXES APPLIED

---

## Production Server Fixes Required

Connect to Ubuntu and apply these 5 fixes to `core/volatility_adjusted_confidence.py`:

### Fix #1: Make talib Import Optional (Lines 12-21)
```bash
# SSH to Ubuntu server
ssh ubuntu@ip-172-31-37-246

cd ~/octivault_trader

# View current import
sed -n '1,30p' core/volatility_adjusted_confidence.py | head -20
```

**Replace** hardcoded import:
```python
import talib
```

**With** optional import:
```python
try:
    import talib
    _HAS_TALIB = True
except (ImportError, ModuleNotFoundError):
    _HAS_TALIB = False
```

---

### Fix #2: Add Near-Zero Magnitude Guard (Lines ~87-101)
**Replace** this section in `compute_histogram_magnitude()`:
```python
max_hist = np.max(recent)

magnitude = latest_mag / max_hist
return np.clip(magnitude, 0.0, 1.0)
```

**With**:
```python
max_hist = np.max(recent)

# CRITICAL FIX: If all histogram values are near zero (e.g., chop/sideways),
# don't divide by near-zero. Instead, return raw magnitude bounded to [0, 1]
if max_hist < 1e-6:  # Near-zero threshold
    magnitude = np.clip(latest_mag * 1000, 0.0, 0.3)
    logger.debug(
        "[VolumeAdjConf] Chop-mode magnitude: max_hist=%.8f latest_mag=%.8f "
        "→ chop_magnitude=%.3f (signals too weak to normalize)",
        max_hist,
        latest_mag,
        magnitude,
    )
    return magnitude

magnitude = latest_mag / max_hist
return np.clip(magnitude, 0.0, 1.0)
```

---

### Fix #3: Update `get_signal_quality_metrics()` Signature (Line 352)
**Current**:
```python
def get_signal_quality_metrics(
    hist_values: np.ndarray,
    regime: str = "normal",
) -> Dict[str, float]:
```

**Change to**:
```python
def get_signal_quality_metrics(
    hist_values: np.ndarray,
    regime: str = "normal",
    closes: np.ndarray = None,
) -> Dict[str, float]:
```

---

### Fix #4: Update Function Body (Line 371)
**Current** (inside `get_signal_quality_metrics()`):
```python
magnitude = compute_histogram_magnitude(hist_values)
```

**Change to**:
```python
magnitude = compute_histogram_magnitude(hist_values, closes=closes)
```

---

### Fix #5: Update Call Site in `trend_hunter.py` (Lines ~893-897)
**Current** (in `agents/trend_hunter.py` around line 892):
```python
metrics = get_signal_quality_metrics(
    hist_values=np.asarray(hist[-50:], dtype=float) if len(hist) >= 50 else np.asarray(hist, dtype=float),
    regime=regime,
)
```

**Change to**:
```python
metrics = get_signal_quality_metrics(
    hist_values=np.asarray(hist[-50:], dtype=float) if len(hist) >= 50 else np.asarray(hist, dtype=float),
    regime=regime,
    closes=closes[-50:] if len(closes) >= 50 else closes,
)
```

---

## Quickstart: Using sed to Apply Fixes

```bash
cd ~/octivault_trader

# Backup files
cp core/volatility_adjusted_confidence.py core/volatility_adjusted_confidence.py.backup
cp agents/trend_hunter.py agents/trend_hunter.py.backup

# Fix 1: Make talib optional
# (Use your editor of choice or the patch below)

# Verify fixes applied
python3 << 'EOFPY'
import inspect
from core.volatility_adjusted_confidence import get_signal_quality_metrics
sig = inspect.signature(get_signal_quality_metrics)
print(f"Function signature: {sig}")
print(f"Parameters: {list(sig.parameters.keys())}")
EOFPY

# Should output:
# Function signature: (hist_values: numpy.ndarray, regime: str = 'normal', closes: numpy.ndarray = None) -> Dict[str, float]
# Parameters: ['hist_values', 'regime', 'closes']
```

---

## After Fixes: Restart Bot

```bash
# Kill bot
pkill -f "python.*run_bot\|python.*octivault_trader"
sleep 3

# Restart
python3 run_bot.py --debug

# Monitor logs
tail -f logs/bot.log | grep "heuristic for"
```

---

## Expected Behavior After Fix

**Before**:
```
mag=0.0000 accel=0.0000 raw=0.000 → adj=0.000 (floor=0.00) → final=0.500
```

**After**:
```
mag=0.3456 accel=0.1200 raw=0.595 → adj=0.595 (floor=0.55) → final=0.595
```

---

## Files to Modify on Ubuntu

1. `core/volatility_adjusted_confidence.py` - 4 changes
2. `agents/trend_hunter.py` - 1 change

---

## Verification

After restart, check:
```bash
# 1. Confidence varies (not always 0.500)
tail -50 logs/bot.log | grep "final=" | sort | uniq

# 2. Floor shows correct value (0.55, 0.65, etc., not 0.00)
tail -50 logs/bot.log | grep "floor=" | sort | uniq
```

---

## Do NOT Use the Mac Workspace Version

The fixes are already in the Mac `utils/` folder but won't affect the Ubuntu server until applied there.

**Next Step**: SSH to Ubuntu and apply these 5 fixes, then restart the bot.

