"""
CANONICAL EV ALIGNMENT CONFIGURATION
=====================================

Phase 1: Unified Expected Value (EV) formula across UURE and ExecutionManager

Purpose:
  • Ensure UURE and ExecutionManager use identical EV calculations
  • Eliminate discrepancies in symbol admission vs execution rejection
  • Enable 10x+ capital scaling without threshold conflicts

Key Principle:
  UURE (Universe Scanner) and EM (Trade Executor) must agree on edge calculations.
  If UURE says "this trade has positive edge", EM must not reject it with EV gate.

Configuration Options:
  - Round-trip cost formula (aligned across both systems)
  - EV multipliers by market regime
  - Spot trading mode for lower thresholds
  - Minimum edge floor (absolute, not relative)

Usage:
  In config/__init__.py or app_context, reference:
    from config.EV_ALIGNMENT_CONFIG import CANONICAL_EV_CONFIG
"""

from decimal import Decimal
from typing import Tuple, Dict, Any


class CanonicalEVConfig:
    """
    Canonical EV configuration enforced across UURE and ExecutionManager.
    
    Both systems MUST use this exact formula for expected value calculations.
    """
    
    # ==================
    # ROUND-TRIP COST
    # ==================
    # Total cost for opening and closing a position (entry + exit fees/slippage)
    
    TAKER_FEE_PCT = 0.001  # 0.1% (Binance standard)
    ROUND_TRIP_TAKER_FEES_PCT = TAKER_FEE_PCT * 2.0  # 0.2% both ways
    
    # Slippage on exit (when market absorbs our full position)
    EXIT_SLIPPAGE_BPS = 15.0  # 0.15% average slippage on exit
    EXIT_SLIPPAGE_PCT = EXIT_SLIPPAGE_BPS / 10000.0
    
    # Safety buffer for TP/SL precision
    TP_MIN_BUFFER_BPS = 0.0  # 0% (adjust if needed for limit order safety)
    TP_MIN_BUFFER_PCT = TP_MIN_BUFFER_BPS / 10000.0
    
    # CANONICAL FORMULA: Total round-trip cost
    @classmethod
    def calculate_round_trip_cost_pct(cls) -> float:
        """
        Calculate round-trip cost as decimal (e.g., 0.0055 = 0.55%).
        
        Formula:
          round_trip_cost = (2 × taker_fee) + slippage + buffer
        
        Example for SPOT trading:
          = (2 × 0.1%) + 0.15% + 0%
          = 0.35%
        """
        return float(
            cls.ROUND_TRIP_TAKER_FEES_PCT +
            cls.EXIT_SLIPPAGE_PCT +
            cls.TP_MIN_BUFFER_PCT
        )
    
    # ==================
    # EV MULTIPLIERS
    # ==================
    # Controls required edge threshold: required_edge = round_trip_cost × multiplier
    # Lower multiplier = lower barrier to entry (more aggressive)
    # Higher multiplier = higher barrier to entry (more conservative)
    
    # NORMAL regime (steady market, 20-30 day VOL)
    EV_MULT_NORMAL = 1.3
    
    # BULL regime (trending up, confirmed upside bias)
    EV_MULT_BULL = 1.8
    
    # OTHER regimes (low volatility, high volatility, sideways, etc.)
    EV_MULT_OTHER = 2.0
    
    # ==================
    # SPOT TRADING MODE
    # ==================
    # For spot trading with lower edge requirements (vs futures)
    
    # If enabled, use relaxed multipliers for all regimes
    SPOT_MODE_ENABLED = False  # Change to True for aggressive spot trading
    
    # Lower multipliers for spot (more opportunities)
    EV_MULT_SPOT_NORMAL = 0.7   # vs 1.3 (46% reduction)
    EV_MULT_SPOT_BULL = 1.0     # vs 1.8 (44% reduction)
    EV_MULT_SPOT_OTHER = 1.4    # vs 2.0 (30% reduction)
    
    # ==================
    # MINIMUM EDGE FLOOR
    # ==================
    # Alternative to multiplier: absolute minimum edge required
    # More permissive than multiplier-based approach
    
    # Absolute minimum edge required (e.g., 0.001 = 0.1%)
    MINIMUM_EDGE_PCT = None  # None = use multiplier mode; set to float for absolute floor
    
    # If set, prefer minimum edge over superiority factor for rotation
    PREFER_MINIMUM_EDGE = False
    
    # ==================
    # OVERRIDE MECHANISMS
    # ==================
    # Allow per-system overrides while maintaining alignment
    
    # UURE-specific round-trip cost override (e.g., 0.003 for spot tuning)
    UURE_ROUND_TRIP_COST_OVERRIDE = None
    
    # ExecutionManager EV multiplier override (e.g., 1.0 for aggressive spot)
    EM_EV_MULTIPLIER_OVERRIDE = None
    
    # ==================
    # VALIDATION & TESTING
    # ==================
    
    @classmethod
    def validate_alignment(cls, uure_round_trip: float, em_round_trip: float) -> Tuple[bool, str]:
        """
        Validate that UURE and EM round-trip costs are aligned.
        
        Args:
            uure_round_trip: Round-trip cost % from UURE
            em_round_trip: Round-trip cost % from ExecutionManager
        
        Returns:
            (is_aligned, message)
        """
        threshold = 0.00001  # Allow 0.001% variance due to rounding
        difference = abs(uure_round_trip - em_round_trip)
        
        if difference <= threshold:
            return True, f"✓ Aligned (diff={difference:.6f}%)"
        else:
            return False, f"✗ Misaligned (UURE={uure_round_trip:.6f}% EM={em_round_trip:.6f}% diff={difference:.6f}%)"
    
    @classmethod
    def validate_multiplier_consistency(cls, regime: str, uure_mult: float, em_mult: float) -> Tuple[bool, str]:
        """
        Validate that UURE and EM use same multiplier for a given regime.
        
        Args:
            regime: Market regime ('normal', 'bull', 'other', 'low', 'high', 'extreme')
            uure_mult: EV multiplier from UURE
            em_mult: EV multiplier from ExecutionManager
        
        Returns:
            (is_consistent, message)
        """
        threshold = 0.01  # Allow 1% variance
        difference = abs(uure_mult - em_mult)
        
        if difference <= threshold:
            return True, f"✓ {regime}: Consistent (UURE={uure_mult:.2f} EM={em_mult:.2f} diff={difference:.4f})"
        else:
            return False, f"✗ {regime}: Inconsistent (UURE={uure_mult:.2f} EM={em_mult:.2f} diff={difference:.4f})"
    
    @classmethod
    def get_multiplier_for_regime(cls, regime: str) -> float:
        """
        Get canonical EV multiplier for a given regime.
        
        Args:
            regime: Market regime ('normal', 'bull', 'other', 'low', 'high', 'extreme')
        
        Returns:
            EV multiplier
        """
        regime_lower = str(regime or "").strip().lower()
        
        # Check for spot mode
        if cls.SPOT_MODE_ENABLED:
            if regime_lower == "normal":
                return cls.EV_MULT_SPOT_NORMAL
            elif regime_lower == "bull":
                return cls.EV_MULT_SPOT_BULL
            else:
                return cls.EV_MULT_SPOT_OTHER
        
        # Standard regime multipliers
        if regime_lower == "normal":
            return cls.EV_MULT_NORMAL
        elif regime_lower == "bull":
            return cls.EV_MULT_BULL
        else:
            return cls.EV_MULT_OTHER
    
    @classmethod
    def get_required_edge_for_regime(cls, regime: str) -> float:
        """
        Get required minimum edge for a given regime.
        
        Args:
            regime: Market regime
        
        Returns:
            Required edge as decimal (e.g., 0.0055 = 0.55%)
        """
        round_trip = cls.calculate_round_trip_cost_pct()
        multiplier = cls.get_multiplier_for_regime(regime)
        return round_trip * multiplier
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary for logging/validation."""
        return {
            "round_trip_cost_pct": cls.calculate_round_trip_cost_pct(),
            "taker_fee_pct": cls.TAKER_FEE_PCT,
            "round_trip_taker_fees_pct": cls.ROUND_TRIP_TAKER_FEES_PCT,
            "exit_slippage_pct": cls.EXIT_SLIPPAGE_PCT,
            "tp_min_buffer_pct": cls.TP_MIN_BUFFER_PCT,
            "ev_mult_normal": cls.EV_MULT_NORMAL,
            "ev_mult_bull": cls.EV_MULT_BULL,
            "ev_mult_other": cls.EV_MULT_OTHER,
            "spot_mode_enabled": cls.SPOT_MODE_ENABLED,
            "minimum_edge_pct": cls.MINIMUM_EDGE_PCT,
            "prefer_minimum_edge": cls.PREFER_MINIMUM_EDGE,
        }


# ==================
# SINGLETON INSTANCE
# ==================
CANONICAL_EV_CONFIG = CanonicalEVConfig


# ==================
# QUICK REFERENCE
# ==================
"""
SPOT TRADING (Low Requirements):
  Spot mode reduces thresholds by 40-50%
  
  With spot mode ON:
    • NORMAL: 0.35% × 0.7 = 0.245% required edge (+20-30% more opportunities)
    • BULL: 0.35% × 1.0 = 0.35% required edge (+30-40% more opportunities)
    • OTHER: 0.35% × 1.4 = 0.49% required edge (+15-25% more opportunities)

FUTURES TRADING (Standard Requirements):
  With spot mode OFF (default):
    • NORMAL: 0.35% × 1.3 = 0.455% required edge
    • BULL: 0.35% × 1.8 = 0.63% required edge
    • OTHER: 0.35% × 2.0 = 0.70% required edge

MINIMUM EDGE MODE (Most Permissive):
  Alternative to multiplier-based approach
  
  With MINIMUM_EDGE_PCT = 0.001 (0.1%):
    • All regimes: required edge = 0.1% (hard floor)
    • Ideal for consistent edge requirements across regimes


CONFIGURATION PRIORITIES (Top to Bottom):
  1. UURE_ROUND_TRIP_COST_OVERRIDE (if set, used by UURE only)
  2. Spot mode enabled (affects multiplier selection)
  3. Minimum edge mode (if MINIMUM_EDGE_PCT set)
  4. Standard multiplier-based (default)
  5. EM_EV_MULTIPLIER_OVERRIDE (if set, used by EM only)
"""


# ==================
# IMPORT GUARD
# ==================
if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("CANONICAL EV ALIGNMENT CONFIG")
    print("=" * 60)
    print(f"\nRound-trip cost: {CANONICAL_EV_CONFIG.calculate_round_trip_cost_pct()*100:.4f}%")
    print(f"EV Multipliers: normal={CANONICAL_EV_CONFIG.EV_MULT_NORMAL} bull={CANONICAL_EV_CONFIG.EV_MULT_BULL} other={CANONICAL_EV_CONFIG.EV_MULT_OTHER}")
    print(f"\nRequired edges by regime:")
    for regime in ["normal", "bull", "other", "low", "high"]:
        edge = CANONICAL_EV_CONFIG.get_required_edge_for_regime(regime)
        print(f"  {regime:10s}: {edge*100:.4f}%")
    print("\n" + "=" * 60)
