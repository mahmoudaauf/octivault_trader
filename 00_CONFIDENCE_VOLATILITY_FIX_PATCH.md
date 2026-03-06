# =============================
# Integration Patch for trend_hunter.py
# =============================
"""
Apply volatility-adjusted confidence to replace hardcoded 0.70.

Changes:
1. Import volatility_adjusted_confidence module
2. Replace heuristic signal generation (lines 800-806)
3. Update _generate_signal() to use regime during confidence computation
4. Add new method _get_regime_aware_confidence()
"""

# =============================
# PATCH: Add import at top (after line 37)
# =============================

# ADD THIS:
from utils.volatility_adjusted_confidence import (
    compute_heuristic_confidence,
    categorize_signal,
    get_signal_quality_metrics,
)

# =============================
# PATCH 1: New method in TrendHunter class
# =============================
# INSERT after _get_regime_scaling_factors() method (around line 570)

async def _get_regime_aware_confidence(self, symbol: str, timeframe: str = None) -> str:
    """
    Get volatility regime for signal confidence calculation.
    
    Preferentially uses high timeframe (1h) brain regime for entry decisions,
    falls back to symbol/global regime on the current timeframe.
    
    Returns:
        Regime string: "trend", "sideways", "high_vol", "bear", "normal", etc.
    """
    if timeframe is None:
        timeframe = self.timeframe
    
    try:
        sym_u = str(symbol).replace("/", "").upper()
        
        # Try 1h regime first (longer-term context)
        try:
            reginfo_1h = await self.shared_state.get_volatility_regime(sym_u, timeframe="1h")
            if not reginfo_1h:
                reginfo_1h = await self.shared_state.get_volatility_regime("GLOBAL", timeframe="1h")
            regime = (reginfo_1h or {}).get("regime", "").lower() if reginfo_1h else None
            if regime:
                return regime
        except Exception:
            pass
        
        # Fallback to symbol regime on current timeframe
        try:
            reginfo = await self.shared_state.get_volatility_regime(sym_u, timeframe=timeframe)
            if not reginfo:
                reginfo = await self.shared_state.get_volatility_regime("GLOBAL", timeframe=timeframe)
            regime = (reginfo or {}).get("regime", "").lower() if reginfo else None
            if regime:
                return regime
        except Exception:
            pass
        
        # Fallback to normal if no regime available
        return "normal"
        
    except Exception as e:
        logger.debug("[%s] Failed to get regime for %s: %s", self.name, symbol, e)
        return "normal"


# =============================
# PATCH 2: Replace hardcoded 0.70 in _generate_signal()
# =============================
# REPLACE lines 800-806:

# OLD:
#        if h_val > 0:
#            # P9: Heuristic signals have lower confidence than ML but must pass floors
#            h_conf = float(self._cfg("HEURISTIC_CONFIDENCE", 0.70))
#            return "BUY", h_conf, f"Heuristic MACD Bullish (hist={h_val:.6f})"
#        if h_val < 0:
#            h_conf = float(self._cfg("HEURISTIC_CONFIDENCE", 0.70))
#            return "SELL", h_conf, f"Heuristic MACD Bearish (hist={h_val:.6f})"

# NEW:
        # 2b) Volatility-Adjusted Confidence (CRITICAL FIX for sideways protection)
        regime = await self._get_regime_aware_confidence(symbol)
        
        # Use new volatility-adjusted confidence instead of hardcoded 0.70
        h_conf = compute_heuristic_confidence(
            hist_value=h_val,
            hist_values=np.asarray(np.asarray(macd_line), dtype=float)[-50:],  # Recent history
            regime=regime,
            closes=closes[-50:] if len(closes) >= 50 else closes,
        )
        
        # Get signal metrics for logging
        metrics = get_signal_quality_metrics(
            hist_values=np.asarray(np.asarray(macd_line), dtype=float)[-50:],
            regime=regime,
        )
        
        if h_val > 0:
            action = "BUY"
        elif h_val < 0:
            action = "SELL"
        else:
            return "HOLD", 0.0, "No clear heuristic signal"
        
        # Log detailed confidence breakdown for debugging
        logger.info(
            "[%s] Heuristic signal for %s (regime=%s) | "
            "magnitude=%.3f accel=%.3f raw=%.3f → adjusted=%.3f (floor=%.2f) → final=%.3f",
            self.name,
            symbol,
            regime,
            metrics.get("histogram_magnitude", 0),
            metrics.get("histogram_acceleration", 0),
            metrics.get("raw_confidence", 0),
            metrics.get("adjusted_confidence", 0),
            metrics.get("regime_floor", 0),
            h_conf,
        )
        
        return action, h_conf, f"Heuristic MACD {action.title()} (hist={h_val:.6f}, conf={h_conf:.3f}, regime={regime})"


# =============================
# PATCH 3: Update _generate_signal method signature
# =============================
# Update the method to be properly async (already is)
# Ensure it properly awaits the regime resolution

async def _generate_signal(self, symbol: str, is_ml_capable: bool = False) -> Tuple[str, float, str]:
    """
    Generate a trading signal. If is_ml_capable is True and a model exists,
    use the model for prediction. Otherwise, fallback to MACD/EMA heuristic
    with VOLATILITY-ADJUSTED CONFIDENCE.
    """
    # ... [existing code up to line 795] ...
    
    # 2) Fallback to MACD/EMA Heuristic with Volatility-Adjusted Confidence
    s_val = float(np.asarray(ema_short)[-1])
    l_val = float(np.asarray(ema_long)[-1])
    h_val = float(np.asarray(hist)[-1])
    
    logger.debug("[%s] Heuristic check for %s: EMA_S=%.2f EMA_L=%.2f HIST=%.6f", 
                 self.name, symbol, s_val, l_val, h_val)
    
    # Get regime for confidence calculation
    regime = await self._get_regime_aware_confidence(symbol)
    
    # Use NEW volatility-adjusted confidence computation
    h_conf = compute_heuristic_confidence(
        hist_value=h_val,
        hist_values=np.asarray(hist[-50:], dtype=float),  # Last 50 bars of history
        regime=regime,
        closes=closes[-50:] if len(closes) >= 50 else closes,
    )
    
    # Get metrics for detailed logging
    metrics = get_signal_quality_metrics(
        hist_values=np.asarray(hist[-50:], dtype=float),
        regime=regime,
    )
    
    # Determine action
    if h_val > 0:
        action = "BUY"
    elif h_val < 0:
        action = "SELL"
    else:
        return "HOLD", 0.0, "No clear heuristic signal"
    
    # Log confidence computation details
    logger.info(
        "[%s] %s signal for %s (regime=%s) | hist_mag=%.4f hist_accel=%.4f "
        "raw_conf=%.3f → adjusted=%.3f (floor=%.2f) → final=%.3f",
        self.name,
        action,
        symbol,
        regime,
        metrics.get("histogram_magnitude", 0),
        metrics.get("histogram_acceleration", 0),
        metrics.get("raw_confidence", 0),
        metrics.get("adjusted_confidence", 0),
        metrics.get("regime_floor", 0),
        h_conf,
    )
    
    return action, h_conf, (
        f"Heuristic MACD {action.title()} | hist={h_val:.6f} conf={h_conf:.3f} "
        f"regime={regime} (mag={metrics.get('histogram_magnitude', 0):.3f})"
    )


# =============================
# CONFIGURATION ADDITIONS
# =============================
# Add to config (e.g., conf/config.yaml):

config_additions = """
# Volatility-Adjusted Confidence Settings
# ==========================================

# Disable static HEURISTIC_CONFIDENCE (now computed dynamically)
# HEURISTIC_CONFIDENCE: 0.70  # ← DEPRECATED, now auto-computed per regime

# Override regime-specific confidence multipliers if needed
# (Default values from utils/volatility_adjusted_confidence.py will be used)
REGIME_CONFIDENCE_MULTIPLIER_OVERRIDE:
  sideways: 0.65      # Reduce sideways signals by 35%
  chop: 0.60          # Reduce chop signals by 40%
  trend: 1.05         # Boost trending signals by 5%

# Override regime-specific confidence floors
REGIME_CONFIDENCE_FLOOR_OVERRIDE:
  sideways: 0.75      # Require 75%+ confidence in sideways
  chop: 0.78          # Require 78%+ confidence in chop
  trend: 0.50         # Allow 50%+ confidence in trend

# Use high timeframe (1h) regime for entry decisions
ENTRY_REGIME_TIMEFRAME: "1h"  # Brain uses 1h context

# Include historical close data for additional analysis
INCLUDE_CLOSES_IN_CONFIDENCE: true
"""
