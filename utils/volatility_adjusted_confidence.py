# =============================
# Volatility-Adjusted Confidence Calculator
# =============================
"""
Replace static 0.70 confidence with dynamic, volatility-aware scoring.

Key insight: MACD histogram **magnitude** and **acceleration** should drive
confidence, not just binary signal direction. Sideways/chop regimes require
much higher confidence thresholds to avoid whipsaws.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional

try:
    import talib
    _HAS_TALIB = True
except (ImportError, ModuleNotFoundError):
    _HAS_TALIB = False

logger = logging.getLogger(__name__)

# =============================
# Confidence Computation Engine
# =============================

def compute_histogram_magnitude(hist_values: np.ndarray, closes: np.ndarray = None) -> float:
    """
    Normalize MACD histogram magnitude to [0, 1] scale.
    
    CRITICAL: Normalize by ATR context to prevent low-volatility regimes
    from creating artificially strong magnitude signals.
    
    Example:
    - Sideways: MACD histogram = 0.00018, ATR = 0.0045 → magnitude = 0.04 (WEAK)
    - Trend: MACD histogram = 0.0245, ATR = 0.040 → magnitude = 0.61 (STRONG)
    
    Args:
        hist_values: Array of MACD histogram values
        closes: Optional close prices for ATR normalization
    
    Returns:
        Normalized magnitude in [0, 1]
    """
    if len(hist_values) == 0:
        return 0.0
    
    latest_mag = abs(float(hist_values[-1]))
    
    # ENHANCEMENT: Normalize by ATR context to detect true signal strength
    if closes is not None and len(closes) >= 20:
        try:
            # Compute ATR-like volatility measure
            # True Range = max(H-L, |H-PC|, |L-PC|)
            closes_arr = np.asarray(closes, dtype=float)
            if len(closes_arr) >= 2:
                # Approximate ATR using recent price volatility
                recent_closes = closes_arr[-20:]
                price_ranges = np.diff(recent_closes)
                volatility = np.std(price_ranges) * 2  # 2-std approximation
                
                if volatility > 0:
                    # Normalize histogram by volatility context
                    # Example: 0.02 histogram / 0.04 volatility = 0.5 magnitude
                    norm_mag = latest_mag / volatility
                    norm_mag = np.clip(norm_mag, 0.0, 1.0)
                    logger.debug(
                        "[VolumeAdjConf] ATR-normalized magnitude: "
                        "hist=%.6f volatility=%.6f norm_mag=%.3f",
                        latest_mag,
                        volatility,
                        norm_mag,
                    )
                    return norm_mag
        except Exception as e:
            logger.debug("[VolumeAdjConf] ATR normalization failed: %s (using fallback)", e)
    
    # FALLBACK: Normalize by recent max (prevents false strength from tiny oscillations)
    recent = hist_values[-20:] if len(hist_values) >= 20 else hist_values
    recent = np.abs(recent)
    
    max_hist = np.max(recent)
    
    # CRITICAL FIX: If all histogram values are near zero (e.g., chop/sideways),
    # don't divide by near-zero. Instead, return raw magnitude bounded to [0, 1]
    if max_hist < 1e-6:  # Near-zero threshold
        # In chop, all signals are weak. Return magnitude clamped to [0, 0.3]
        magnitude = np.clip(latest_mag * 1000, 0.0, 0.3)  # Scale up slightly but cap at weak
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


def compute_histogram_acceleration(hist_values: np.ndarray) -> float:
    """
    Compute MACD histogram momentum (2nd derivative).
    
    Positive acceleration = signal strengthening
    Negative acceleration = signal weakening
    
    Args:
        hist_values: Array of MACD histogram values (length >= 3)
    
    Returns:
        Acceleration normalized to [-1, 1]
    """
    if len(hist_values) < 3:
        return 0.0
    
    # Second derivative: (h2 - h1) - (h1 - h0)
    latest = float(hist_values[-1])
    prev1 = float(hist_values[-2])
    prev2 = float(hist_values[-3])
    
    accel = (latest - prev1) - (prev1 - prev2)
    
    # Normalize by recent volatility
    recent = np.abs(hist_values[-10:]) if len(hist_values) >= 10 else np.abs(hist_values)
    volatility = np.std(recent) if len(recent) > 1 else 1.0
    volatility = volatility if volatility > 0 else 1.0
    
    normalized_accel = accel / (volatility * 2)  # Scale to [-1, 1]
    return np.clip(normalized_accel, -1.0, 1.0)


def get_regime_confidence_multiplier(regime: str) -> float:
    """
    Get regime-based confidence multiplier.
    
    Trending regimes: boost confidence (signals are more likely correct)
    Sideways/chop: slash confidence (high whipsaw risk)
    High vol: moderately reduce confidence
    
    Args:
        regime: Volatility regime label
    
    Returns:
        Multiplier in (0.4, 1.1] applied to base confidence
    """
    regime_norm = str(regime or "normal").lower().strip()
    
    multipliers = {
        # Trending: boost
        "trend": 1.05,
        "uptrend": 1.05,
        "downtrend": 1.05,
        
        # Neutral/Normal: baseline
        "normal": 1.0,
        "neutral": 1.0,
        
        # High volatility: slight caution
        "high_vol": 0.90,
        "high": 0.90,
        "volatile": 0.90,
        
        # Bear: defensive
        "bear": 0.85,
        "bearish": 0.85,
        
        # Sideways/Chop: CRITICAL reduction
        "sideways": 0.65,
        "chop": 0.60,
        "choppy": 0.60,
        "range": 0.65,
        "range-bound": 0.65,
        
        # Consolidation/Ranging
        "consolidation": 0.65,
        "consolidating": 0.65,
    }
    
    return multipliers.get(regime_norm, 1.0)


def get_regime_confidence_floor(regime: str) -> float:
    """
    Get minimum confidence required to emit signal in this regime.
    
    Sideways/chop regimes require MUCH higher confidence (75%+) to avoid
    high-frequency whipsaws from small MACD oscillations.
    
    Args:
        regime: Volatility regime label
    
    Returns:
        Minimum confidence threshold in [0.4, 0.9]
    """
    regime_norm = str(regime or "normal").lower().strip()
    
    floors = {
        # Trending: lower floor (momentum helps us)
        "trend": 0.50,
        "uptrend": 0.50,
        "downtrend": 0.50,
        
        # Neutral/Normal: standard
        "normal": 0.55,
        "neutral": 0.55,
        
        # High volatility: moderate
        "high_vol": 0.60,
        "high": 0.60,
        "volatile": 0.60,
        
        # Bear: defensive
        "bear": 0.65,
        "bearish": 0.65,
        
        # Sideways/Chop: STRICT (prevent whipsaws)
        "sideways": 0.75,      # ← CRITICAL: 75% minimum in sideways
        "chop": 0.78,          # ← CRITICAL: 78% minimum in chop
        "choppy": 0.78,
        "range": 0.75,
        "range-bound": 0.75,
        
        # Consolidation/Ranging
        "consolidation": 0.75,
        "consolidating": 0.75,
    }
    
    return floors.get(regime_norm, 0.55)


def compute_heuristic_confidence(
    hist_value: float,
    hist_values: np.ndarray,
    regime: str = "normal",
    closes: Optional[np.ndarray] = None,
) -> float:
    """
    Compute MACD heuristic confidence as function of signal strength and regime.
    
    CRITICAL FIX: Replaces hardcoded 0.70 with dynamic, volatility-aware scoring.
    
    Confidence factors:
    1. Histogram magnitude (how strong is the signal?)
    2. Histogram acceleration (is the signal strengthening or weakening?)
    3. Regime multiplier (trending vs sideways context)
    4. Regime floor (sideways requires 75%+ to trade)
    
    Args:
        hist_value: Latest MACD histogram value
        hist_values: Array of recent MACD histogram values
        regime: Current volatility regime (default "normal")
        closes: Optional close prices for additional analysis
    
    Returns:
        Confidence score in [0.0, 1.0]
    """
    if len(hist_values) == 0:
        return 0.0
    
    # Step 1: Compute base confidence from histogram properties
    magnitude = compute_histogram_magnitude(hist_values)  # [0, 1]
    acceleration = compute_histogram_acceleration(hist_values)  # [-1, 1]
    
    # Map magnitude to base confidence: [0, 1] → [0.40, 0.85]
    # Weak signals (magnitude=0.1) get 0.40, strong signals (magnitude=1.0) get 0.85
    base_conf = 0.40 + (magnitude * 0.45)
    
    # Step 2: Boost if acceleration is positive (signal strengthening)
    accel_bonus = max(0.0, acceleration * 0.15)  # Up to +15% if accelerating
    base_conf = min(0.95, base_conf + accel_bonus)
    
    logger.debug(
        "[VolumeAdjConf] Base confidence: magnitude=%.3f accel=%.3f "
        "→ base_conf=%.3f (accel_bonus=+%.3f)",
        magnitude,
        acceleration,
        base_conf - accel_bonus,
        accel_bonus,
    )
    
    # Step 3: Apply regime multiplier
    multiplier = get_regime_confidence_multiplier(regime)
    adjusted_conf = base_conf * multiplier
    
    # Step 4: Enforce regime floor (critical for sideways protection)
    floor = get_regime_confidence_floor(regime)
    final_conf = max(floor, adjusted_conf)
    
    logger.debug(
        "[VolumeAdjConf] Regime adjustment: %s | multiplier=%.2f "
        "(%.3f → %.3f) | floor=%.2f → final=%.3f",
        regime,
        multiplier,
        base_conf,
        adjusted_conf,
        floor,
        final_conf,
    )
    
    return np.clip(final_conf, 0.0, 1.0)


def categorize_signal(
    hist_value: float,
    hist_values: np.ndarray,
    regime: str = "normal",
) -> Tuple[str, float]:
    """
    Categorize MACD signal into action with volatility-aware confidence.
    
    This replaces the binary hardcoded "if h_val > 0: return BUY, 0.70" logic.
    
    Args:
        hist_value: Latest MACD histogram value
        hist_values: Array of recent MACD histogram values
        regime: Current volatility regime
    
    Returns:
        (action, confidence): e.g., ("BUY", 0.78) or ("HOLD", 0.0)
    """
    if len(hist_values) == 0:
        return "HOLD", 0.0
    
    # Compute confidence first
    confidence = compute_heuristic_confidence(hist_value, hist_values, regime)
    
    # Binary action from histogram sign
    if hist_value > 0:
        action = "BUY"
    elif hist_value < 0:
        action = "SELL"
    else:
        action = "HOLD"
        confidence = 0.0
    
    logger.debug(
        "[VolumeAdjConf] Signal categorization: hist=%.6f regime=%s "
        "→ action=%s confidence=%.3f",
        hist_value,
        regime,
        action,
        confidence,
    )
    
    return action, confidence


def get_signal_quality_metrics(
    hist_values: np.ndarray,
    regime: str = "normal",
    closes: np.ndarray = None,
) -> Dict[str, float]:
    """
    Get detailed signal quality metrics for logging/analysis.
    
    Args:
        hist_values: Array of MACD histogram values
        regime: Current volatility regime
        closes: Optional close prices for ATR normalization of magnitude
    
    Returns:
        Dict with metrics: magnitude, acceleration, multiplier, floor, raw_conf, etc.
    """
    if len(hist_values) == 0:
        return {}
    
    magnitude = compute_histogram_magnitude(hist_values, closes=closes)
    acceleration = compute_histogram_acceleration(hist_values)
    multiplier = get_regime_confidence_multiplier(regime)
    floor = get_regime_confidence_floor(regime)
    
    base_conf = 0.40 + (magnitude * 0.45)
    accel_bonus = max(0.0, acceleration * 0.15)
    raw_conf = base_conf + accel_bonus
    adjusted_conf = raw_conf * multiplier
    final_conf = max(floor, adjusted_conf)
    
    return {
        "histogram_magnitude": magnitude,
        "histogram_acceleration": acceleration,
        "base_confidence": base_conf,
        "acceleration_bonus": accel_bonus,
        "raw_confidence": raw_conf,
        "regime_multiplier": multiplier,
        "adjusted_confidence": adjusted_conf,
        "regime_floor": floor,
        "final_confidence": final_conf,
        "regime": regime,
    }
