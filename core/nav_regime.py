# -*- coding: utf-8 -*-
"""
NAV Regime Engine (MICRO_SNIPER Mode for Small Capital)

Dynamically switches system behavior based on live NAV from SharedState.

REGIME DEFINITIONS:
- MICRO_SNIPER (NAV < 1000 USDT): Simplified sniper mode for micro accounts
- STANDARD (1000 <= NAV < 5000): Normal multi-agent with basic constraints
- MULTI_AGENT (NAV >= 5000): Full architecture with all features enabled

INTEGRATION POINTS:
- MetaController queries this module at start of each evaluation cycle
- SharedState.get_nav_quote() or SharedState.nav provides live NAV
- Each regime has hard rules for signal filtering and execution

NAV REGIME RULES:

[MICRO_SNIPER MODE] NAV < 1000
├─ Max positions: 1 (no concurrent diversification)
├─ Max assets: 1 (single symbol focus)
├─ Min expected_move: 1.0% (hard gate)
├─ Min confidence: 0.70 (quality filter)
├─ Min hold time: 600 sec (10 minutes)
├─ Max trades/day: 3 (hourly execution limit)
├─ Position sizing: min(30% NAV, available USDT)
├─ DISABLED:
│  ├─ RotationAuthority (no multi-symbol rotation)
│  ├─ DustHealing (no consolidation trading)
│  ├─ CapitalAllocator reservations (bypass to max available)
│  └─ Symbol expansion (stay on best asset)
└─ Regime switches back to STANDARD when NAV >= 1000

[STANDARD MODE] 1000 <= NAV < 5000
├─ Max positions: 2 (limited diversification)
├─ Max assets: 2-3 (conservative universe)
├─ Min expected_move: 0.50% (relaxed vs MICRO)
├─ Min confidence: 0.65
├─ Min hold time: 300 sec (5 minutes)
├─ Max trades/day: 6 (less aggressive than MULTI_AGENT)
├─ Position sizing: min(25% NAV, available USDT)
└─ Regime switches to MICRO_SNIPER if NAV < 1000

[MULTI_AGENT MODE] NAV >= 5000
├─ Max positions: 3+ (full portfolio)
├─ Max assets: 5+ (aggressive diversification)
├─ Min expected_move: 0.30% (full sensitivity)
├─ Min confidence: 0.60 (standard threshold)
├─ Min hold time: 180 sec (normal scaling)
├─ Max trades/day: 20+ (unrestricted)
├─ Position sizing: Standard ScalingManager logic
└─ Regime switches to STANDARD if NAV < 5000

ECONOMIC GATE (All Regimes):
- reject if expected_move_percent < (2 * maker_fee_pct + 0.3%)
- Log edge: "edge=${edge_pct:.3f}% (move=${move_pct}% - fees=${fee_pct}%)"
- Minimum profitable move = 0.55% (0.1% × 2 + 0.3% slippage + margin)

HOLDING DISCIPLINE (All Regimes):
- No SELL unless TP or SL triggered
- No counter-trend micro-scalping (flag re-entry blocks if flat, no signal change)
- Minimum hold time prevents flip-flopping

REGIME SWITCHING BEHAVIOR:
- Automatic at each cycle start via get_nav_regime(nav)
- Applies hard rules to signal filtering
- Disabled components return early with NO_ACTION
- SEAMLESS: No restart required on NAV threshold crossing
- REVERSIBLE: System resumes full architecture when NAV >= 1000

FILE DEPENDENCIES:
- MetaController queries this module via: regime = nav_regime.get_nav_regime(nav)
- RotationAuthority checks regime before proceeding
- DustHealing checks regime before triggering
- CapitalAllocator checks regime before applying reservations
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Set
from enum import Enum


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Safely parse bool-like values from config/env."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _coerce_int(value: Any, default: int, minimum: int = 1) -> int:
    """Safely parse int-like values with lower bound."""
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(int(minimum), parsed)


class NAVRegime:
    """NAV-based regime constants."""
    MICRO_SNIPER = "MICRO_SNIPER"
    STANDARD = "STANDARD"
    MULTI_AGENT = "MULTI_AGENT"


class MicroSniperConfig:
    """Configuration for MICRO_SNIPER mode (NAV < 1000)."""
    
    # Position limits
    MAX_OPEN_POSITIONS = 1  # Only 1 concurrent trade
    MAX_ACTIVE_SYMBOLS = 1  # Only 1 asset at a time
    
    # Quality gates
    MIN_EXPECTED_MOVE_PCT = 1.0  # 1.0% minimum move required
    MIN_CONFIDENCE = 0.50  # 🔧 FIX: Lowered from 0.70 to 0.50 to allow high-quality signals (0.75+) to execute
    MIN_HOLD_TIME_SEC = 600  # 10 minutes minimum holding period
    MAX_TRADES_PER_DAY = 3  # Maximum 3 trades per calendar day
    
    # Position sizing
    POSITION_SIZE_PCT_NAV = 0.30  # Use up to 30% of NAV per position
    
    # Disabled features
    ROTATION_ENABLED = False  # Disable symbol rotation
    DUST_HEALING_ENABLED = False  # Disable dust healing trades
    CAPITAL_RESERVATIONS_ENABLED = False  # Bypass reservation logic
    
    # Economic gate for small accounts
    # For MICRO accounts: target TP ≈ 1.8% – 2.5% to overcome fees
    # Fee structure: 0.2% taker × 2 (entry + exit) + ~0.3% slippage = 0.7% friction minimum
    # Therefore minimum profitable move should be ~2.0% to ensure positive EV
    MIN_PROFITABLE_MOVE_PCT = 2.0  # Increased from 0.55% to account for fees dominating on small accounts


class StandardConfig:
    """Configuration for STANDARD mode (1000 <= NAV < 5000)."""
    
    # Position limits
    MAX_OPEN_POSITIONS = 2  # Conservative: 2 concurrent trades
    MAX_ACTIVE_SYMBOLS = 3  # Up to 3 symbols
    
    # Quality gates
    MIN_EXPECTED_MOVE_PCT = 0.50  # Relaxed vs MICRO
    MIN_CONFIDENCE = 0.55  # 🔧 FIX: Lowered from 0.65 to 0.55
    MIN_HOLD_TIME_SEC = 300  # 5 minutes
    MAX_TRADES_PER_DAY = 6  # Moderate trade frequency
    
    # Position sizing
    POSITION_SIZE_PCT_NAV = 0.25
    
    # Enabled features (defaults, controlled elsewhere)
    ROTATION_ENABLED = True
    DUST_HEALING_ENABLED = True
    CAPITAL_RESERVATIONS_ENABLED = True
    
    # Economic gate for mid-size accounts
    # STANDARD accounts: higher min profitable move than MULTI_AGENT
    # Still need to overcome ~0.7% friction, target 1.2% – 1.5%
    MIN_PROFITABLE_MOVE_PCT = 1.2  # Increased from 0.55% to ensure profitability on modest accounts


class MultiAgentConfig:
    """Configuration for MULTI_AGENT mode (NAV >= 5000)."""
    
    # Position limits
    MAX_OPEN_POSITIONS = 3  # Full multi-asset
    MAX_ACTIVE_SYMBOLS = 5  # Aggressive universe
    
    # Quality gates
    MIN_EXPECTED_MOVE_PCT = 0.30  # Full sensitivity
    MIN_CONFIDENCE = 0.45  # 🔧 FIX: Lowered from 0.60 to 0.45
    MIN_HOLD_TIME_SEC = 180  # Normal scaling
    MAX_TRADES_PER_DAY = 20  # Unrestricted
    
    # Position sizing
    POSITION_SIZE_PCT_NAV = 0.20  # Standard allocation
    
    # Enabled features (full architecture)
    ROTATION_ENABLED = True
    DUST_HEALING_ENABLED = True
    CAPITAL_RESERVATIONS_ENABLED = True
    
    # Economic gate for large accounts
    # MULTI_AGENT accounts: sufficient NAV to absorb friction more effectively
    # Can operate with lower minimum profitable move
    MIN_PROFITABLE_MOVE_PCT = 0.8  # Increased from 0.55% but lower than smaller account tiers


def get_nav_regime(
    nav: float,
    *,
    micro_threshold: float = 1000.0,
    multi_agent_threshold: float = 5000.0,
) -> str:
    """
    Determine regime based on live NAV.
    
    Args:
        nav: Net Asset Value in USDT
        
    Returns:
        str: NAVRegime constant (MICRO_SNIPER, STANDARD, or MULTI_AGENT)
    """
    micro_threshold = float(micro_threshold or 1000.0)
    multi_agent_threshold = float(multi_agent_threshold or 5000.0)
    if multi_agent_threshold <= micro_threshold:
        multi_agent_threshold = micro_threshold * 2.0

    if nav < micro_threshold:
        return NAVRegime.MICRO_SNIPER
    elif nav < multi_agent_threshold:
        return NAVRegime.STANDARD
    else:
        return NAVRegime.MULTI_AGENT


def get_regime_config(regime: str) -> Dict[str, Any]:
    """
    Get regime configuration as dict.
    
    Args:
        regime: Regime constant from NAVRegime
        
    Returns:
        dict: Configuration for regime with all rules
    """
    if regime == NAVRegime.MICRO_SNIPER:
        config_class = MicroSniperConfig
    elif regime == NAVRegime.STANDARD:
        config_class = StandardConfig
    else:  # MULTI_AGENT
        config_class = MultiAgentConfig
    
    return {
        "regime": regime,
        "max_open_positions": config_class.MAX_OPEN_POSITIONS,
        "max_active_symbols": config_class.MAX_ACTIVE_SYMBOLS,
        "min_expected_move_pct": config_class.MIN_EXPECTED_MOVE_PCT,
        "min_confidence": config_class.MIN_CONFIDENCE,
        "min_hold_time_sec": config_class.MIN_HOLD_TIME_SEC,
        "max_trades_per_day": config_class.MAX_TRADES_PER_DAY,
        "position_size_pct_nav": config_class.POSITION_SIZE_PCT_NAV,
        "rotation_enabled": config_class.ROTATION_ENABLED,
        "dust_healing_enabled": config_class.DUST_HEALING_ENABLED,
        "capital_reservations_enabled": config_class.CAPITAL_RESERVATIONS_ENABLED,
        "min_profitable_move_pct": config_class.MIN_PROFITABLE_MOVE_PCT,
    }


class RegimeManager:
    """
    Tracks regime state and provides logging.
    
    Integrates with MetaController to:
    1. Query current regime at cycle start
    2. Detect regime switches
    3. Log regime state changes
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Any] = None):
        self.logger = logger or logging.getLogger("NAVRegime")
        self.config = config
        self.micro_threshold = float(
            getattr(config, "NAV_REGIME_MICRO_THRESHOLD", 0.0)
            or getattr(config, "CAPITAL_MICRO_THRESHOLD", 0.0)
            or os.getenv("NAV_REGIME_MICRO_THRESHOLD", os.getenv("CAPITAL_MICRO_THRESHOLD", "500"))
            or 500.0
        )
        self.multi_agent_threshold = float(
            getattr(config, "NAV_REGIME_MULTI_AGENT_THRESHOLD", 0.0)
            or getattr(config, "CAPITAL_MEDIUM_THRESHOLD", 0.0)
            or os.getenv(
                "NAV_REGIME_MULTI_AGENT_THRESHOLD",
                os.getenv("CAPITAL_MEDIUM_THRESHOLD", "10000"),
            )
            or 10000.0
        )
        if self.multi_agent_threshold <= self.micro_threshold:
            self.multi_agent_threshold = self.micro_threshold * 2.0

        self.current_regime = NAVRegime.MULTI_AGENT  # Default assume large account
        self.current_config = self._apply_runtime_overrides(get_regime_config(NAVRegime.MULTI_AGENT))
        self.last_regime_switch_ts = time.time()
        self.regime_switch_count = 0
        self._trades_executed_today = 0
        self._last_trade_day_utc = None
        self.logger.info(
            "[NAVRegime] thresholds loaded: micro<%.2f, standard<%.2f, multi_agent>=%.2f",
            self.micro_threshold,
            self.multi_agent_threshold,
            self.multi_agent_threshold,
        )
    
    def update_regime(self, nav: float) -> bool:
        """
        Update regime based on live NAV. Returns True if regime switched.
        
        Args:
            nav: Current NAV from SharedState
            
        Returns:
            bool: True if regime changed, False if same regime
        """
        new_regime = get_nav_regime(
            nav,
            micro_threshold=self.micro_threshold,
            multi_agent_threshold=self.multi_agent_threshold,
        )
        regime_switched = (new_regime != self.current_regime)
        
        if regime_switched:
            old_regime = self.current_regime
            self.current_regime = new_regime
            self.current_config = self._apply_runtime_overrides(get_regime_config(new_regime))
            self.last_regime_switch_ts = time.time()
            self.regime_switch_count += 1
            
            self.logger.info(
                "[REGIME_SWITCH] NAV=%.2f USD: %s → %s (switch_count=%d)",
                nav, old_regime, new_regime, self.regime_switch_count
            )
        
        return regime_switched

    def _apply_runtime_overrides(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply runtime micro-capital overrides so NAV regime gates stay aligned with
        adaptive capital-governor settings.
        """
        cfg = dict(base_config or {})
        regime = str(cfg.get("regime") or "")
        if regime != NAVRegime.MICRO_SNIPER:
            return cfg

        default_max_pos = int(cfg.get("max_open_positions", 1) or 1)
        default_max_symbols = int(cfg.get("max_active_symbols", default_max_pos) or default_max_pos)
        default_rotation = bool(cfg.get("rotation_enabled", False))

        raw_micro_max = getattr(self.config, "CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS", None)
        if raw_micro_max is None:
            raw_micro_max = os.getenv("CAPITAL_MICRO_ADAPTIVE_MAX_CONCURRENT_POSITIONS")
        micro_max_pos = _coerce_int(raw_micro_max, default=default_max_pos, minimum=1)

        raw_rotation = getattr(self.config, "CAPITAL_MICRO_ALLOW_ROTATION", None)
        if raw_rotation is None:
            raw_rotation = os.getenv("CAPITAL_MICRO_ALLOW_ROTATION")
        rotation_enabled = _coerce_bool(raw_rotation, default=default_rotation)

        raw_rot_slots = getattr(self.config, "CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS", None)
        if raw_rot_slots is None:
            raw_rot_slots = os.getenv("CAPITAL_MICRO_ADAPTIVE_MAX_ROTATING_SLOTS")
        rotating_slots = _coerce_int(raw_rot_slots, default=1, minimum=0)

        target_symbols = micro_max_pos + (rotating_slots if rotation_enabled else 0)
        cfg["max_open_positions"] = micro_max_pos
        cfg["rotation_enabled"] = rotation_enabled
        cfg["max_active_symbols"] = max(default_max_symbols, target_symbols, 1)

        if (
            cfg["max_open_positions"] != default_max_pos
            or cfg["rotation_enabled"] != default_rotation
            or cfg["max_active_symbols"] != default_max_symbols
        ):
            self.logger.info(
                "[NAVRegime:Override] MICRO_SNIPER overrides applied: max_open_positions=%d max_active_symbols=%d rotation_enabled=%s",
                cfg["max_open_positions"],
                cfg["max_active_symbols"],
                cfg["rotation_enabled"],
            )

        return cfg
    
    def get_regime(self) -> str:
        """Get current regime."""
        return self.current_regime
    
    def get_config(self) -> Dict[str, Any]:
        """Get current regime configuration."""
        return self.current_config
    
    def get_max_positions(self) -> int:
        """Get max open positions for current regime."""
        return self.current_config["max_open_positions"]
    
    def get_max_symbols(self) -> int:
        """Get max active symbols for current regime."""
        return self.current_config["max_active_symbols"]
    
    def get_min_move(self) -> float:
        """Get minimum expected_move_pct for current regime."""
        return self.current_config["min_expected_move_pct"]
    
    def get_min_confidence(self) -> float:
        """Get minimum confidence for current regime."""
        return self.current_config["min_confidence"]
    
    def get_min_hold_time(self) -> float:
        """Get minimum hold time in seconds for current regime."""
        return self.current_config["min_hold_time_sec"]
    
    def get_max_trades_per_day(self) -> int:
        """Get max trades per day for current regime."""
        return self.current_config["max_trades_per_day"]
    
    def is_micro_sniper(self) -> bool:
        """Check if in MICRO_SNIPER regime."""
        return self.current_regime == NAVRegime.MICRO_SNIPER
    
    def is_standard(self) -> bool:
        """Check if in STANDARD regime."""
        return self.current_regime == NAVRegime.STANDARD
    
    def is_multi_agent(self) -> bool:
        """Check if in MULTI_AGENT regime."""
        return self.current_regime == NAVRegime.MULTI_AGENT
    
    def is_rotation_enabled(self) -> bool:
        """Check if rotation is enabled in current regime."""
        return self.current_config["rotation_enabled"]
    
    def is_dust_healing_enabled(self) -> bool:
        """Check if dust healing is enabled in current regime."""
        return self.current_config["dust_healing_enabled"]
    
    def is_capital_reservations_enabled(self) -> bool:
        """Check if capital reservations are enabled in current regime."""
        return self.current_config["capital_reservations_enabled"]
    
    def increment_daily_trade_count(self) -> None:
        """Increment daily trade counter (call when trade executed)."""
        import datetime
        today_utc = datetime.datetime.utcnow().date()
        
        # Reset counter if date changed
        if self._last_trade_day_utc != today_utc:
            self._trades_executed_today = 0
            self._last_trade_day_utc = today_utc
        
        self._trades_executed_today += 1
    
    def get_daily_trade_count(self) -> int:
        """Get trades executed today (UTC)."""
        import datetime
        today_utc = datetime.datetime.utcnow().date()
        
        # Reset if date changed
        if self._last_trade_day_utc != today_utc:
            self._trades_executed_today = 0
            self._last_trade_day_utc = today_utc
        
        return self._trades_executed_today
    
    def can_execute_trade_today(self) -> bool:
        """Check if daily trade limit reached."""
        return self.get_daily_trade_count() < self.get_max_trades_per_day()
    
    def reset_daily_counter(self) -> None:
        """Manually reset daily trade counter (e.g., at system restart)."""
        self._trades_executed_today = 0
        self._last_trade_day_utc = None
