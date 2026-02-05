"""
Rotation & Exit Authority (REA) - P9 Canonical Design
Provides capital velocity governance by authorizing forced exits for rotation.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple

class RotationExitAuthority:
    def __init__(self, logger: logging.Logger, config: Any, shared_state: Any):
        self.logger = logger
        self.config = config
        self.ss = shared_state
        
        # Configuration
        self.base_alpha_gap = float(getattr(config, "ROTATION_BASE_ALPHA_GAP", 0.005)) # 0.5% alpha gap
        self.winner_protection_threshold = float(getattr(config, "ROTATION_WINNER_PROTECTION_PNL", 0.002)) # 0.2% PnL
        self.winner_extra_alpha = float(getattr(config, "ROTATION_WINNER_EXTRA_ALPHA", 0.03)) # 3% extra alpha to kick winner
        
        # Mode-based thresholds
        self.mode_thresholds = {
            "BOOTSTRAP": 0.3,   # Aggressive rotation
            "NORMAL": 0.6,      # Balanced
            "RECOVERY": 0.8,    # Conservative
            "AGGRESSIVE": 0.4,  # High velocity
            "SAFE": 1.0,        # Disabled
            "PROTECTIVE": 0.9   # Minimal
        }

    def calculate_rotation_score(self, position: Dict[str, Any], best_opp_score: float) -> float:
        """
        R2/R3: Score a held position for rotation eligibility.
        rotation_score = f(time_decay, pnl_efficiency, opportunity_cost)
        
        Higher score = STRONGER candidate for EXIT.
        """
        symbol = position.get("symbol", "unknown")
        
        # 1. Time Factor (Normalized age)
        entry_ts = float(position.get("entry_time") or position.get("opened_at") or time.time())
        max_hold_sec = float(getattr(self.config, "MAX_HOLD_SEC", 1800)) # 30 mins default
        age_sec = time.time() - entry_ts
        time_factor = min(age_sec / max_hold_sec, 1.0)
        
        # 2. PnL Efficiency (Clamped)
        pnl_pct = float(position.get("unrealized_pnl_pct", 0.0) or 0.0)
        target_pnl = 0.03 # 3% target
        pnl_factor = max(-1.0, min(1.0, pnl_pct / target_pnl))
        
        # We want to keep winners (high pnl) and cycle losers (low/neg pnl)
        # So efficiency_score is lower for winners
        efficiency_score = 1.0 - pnl_factor
        
        # 3. Opportunity Cost
        # (How much better is the candidate compared to this position?)
        # best_opp_score is expected ROI * confidence
        held_score = pnl_pct * 0.5 # Basic proxy for held performance expectancy
        opportunity_cost = max(0.0, best_opp_score - held_score)
        
        # Weighting
        w_time = 0.3
        w_eff = 0.3
        w_opp = 0.4
        
        rotation_score = (w_time * time_factor) + (w_eff * efficiency_score) + (w_opp * opportunity_cost)
        
        self.logger.debug(
            "[REA:Score] %s rotation_score=%.4f (time=%.2f, eff=%.2f, opp=%.2f)",
            symbol, rotation_score, time_factor, efficiency_score, opportunity_cost
        )
        return rotation_score

    async def authorize_rotation(
        self, 
        sig_pos: int, 
        max_pos: int, 
        owned_positions: Dict[str, Any], 
        best_opp: Dict[str, Any],
        current_mode: str,
        is_starved: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        R4: Authorize a FORCED_EXIT if rotation criteria met.
        Returns the signal for the position to exit, or None.
        
        OVERRIDE AUTHORITY: This can override standard TPSL and mode constraints
        if the alpha gap (opportunity cost) is sufficient.
        """
        # Condition Check: Either we are full OR we are out of capital
        if sig_pos < max_pos and not is_starved:
            return None # Sufficient slots and capital; no rotation needed
            
        if not best_opp:
            return None # Don't churn into nothing
            
        best_opp_sym = best_opp.get("symbol")
        opp_score = float(best_opp.get("_opp_score", 0.0))
        
        # Find candidates
        candidates = []
        now = time.time()
        cooldown = float(getattr(self.config, "ROTATION_COOLDOWN_SEC", 600))
        
        # Bootstrap bypass: ignore cooldown if we need velocity
        if current_mode == "BOOTSTRAP":
            cooldown = 60.0 # 1 min during bootstrap
            
        for sym, pos in owned_positions.items():
            if sym == best_opp_sym:
                continue
            
            # Filter check (e.g. TPSL protect)
            if pos.get("state") == "EXITING":
                continue
                
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or 0.0)
            if (now - entry_ts) < cooldown:
                continue
                
            r_score = self.calculate_rotation_score(pos, opp_score)
            candidates.append((sym, r_score, pos))
            
        if not candidates:
            return None
            
        # Find best exit candidate (highest rotation score)
        worst_sym, highest_r_score, worst_pos = max(candidates, key=lambda x: x[1])
        
        # Threshold logic
        threshold = self.mode_thresholds.get(current_mode, 0.6)
        
        # Starvation override: Be more aggressive if starved
        if is_starved:
            threshold *= 0.8 # 20% easier to authorize if capital is zero
        
        max_hold_sec = float(getattr(self.config, "MAX_HOLD_SEC", 1800))
        stagnation_mult = float(getattr(self.config, "STAGNATION_HOLD_MULT", 4.0))
        stagnation_time = max_hold_sec * stagnation_mult
        stagnation_band = float(getattr(self.config, "STAGNATION_PNL_BAND", 0.001))
        stagnation_override_enabled = bool(getattr(self.config, "STAGNATION_OVERRIDE_ENABLED", True))

        entry_ts = float(worst_pos.get("entry_time") or worst_pos.get("opened_at") or 0.0)
        age_sec = now - entry_ts
        worst_pnl_pct = float(worst_pos.get("unrealized_pnl_pct", 0.0) or 0.0)
        stagnation_override = (
            stagnation_override_enabled
            and age_sec >= stagnation_time
            and abs(worst_pnl_pct) <= stagnation_band
        )

        if highest_r_score >= threshold or stagnation_override:
            reason = "ROTATION_EXIT_VELOCITY_OVERRIDE"
            if stagnation_override and highest_r_score < threshold:
                reason = "ROTATION_STAGNATION_OVERRIDE"
            self.logger.info(
                "[REA:Authorized] 🔄 Rotation GRANTED: %s (score=%.2f) -> %s (opp=%.2f) [Mode: %s, Starved: %s]",
                worst_sym, highest_r_score, best_opp_sym, opp_score, current_mode, is_starved
            )
            
            return {
                "symbol": worst_sym,
                "action": "SELL",
                "confidence": 1.0,
                "agent": "RotationExitAuthority",
                "timestamp": now,
                "reason": reason,
                "priority": "HIGH",
                "replaced_by": best_opp_sym,
                "_is_rotation": True,
                "_forced_exit": True,
                "_rotation_score": highest_r_score,
                "_stagnation_override": stagnation_override,
                "allow_partial": True,
                "target_fraction": 0.5
            }
            
        return None

    def authorize_stagnation_exit(
        self,
        owned_positions: Dict[str, Any],
        current_mode: str
    ) -> Optional[Dict[str, Any]]:
        """
        STAGNATION AUTHORITY: Identify and purge positions that are dragging down 
        capital velocity, even if no immediate replacement is waiting.
        
        This prevents the "zombie portfolio" where all capital is stuck in flat trades.
        """
        candidates = []
        now = time.time()
        
        # Stagnation Thresholds
        # If held for N x MAX_HOLD_SEC and PnL < 0.1%, it's a zombie.
        max_hold = float(getattr(self.config, "MAX_HOLD_SEC", 1800))
        stagnation_mult = float(getattr(self.config, "STAGNATION_HOLD_MULT", 4.0))
        stagnation_time = max_hold * stagnation_mult
        
        for sym, pos in owned_positions.items():
            if pos.get("state") == "EXITING":
                continue
                
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or 0.0)
            age = now - entry_ts
            
            if age < stagnation_time:
                continue
                
            pnl_pct = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
            
            # Stagnant if very old and low/neg profit
            # (Basically it's taking up a slot and not performing)
            if pnl_pct < 0.001: # less than 0.1% profit
                # Calculate a stagnation score [0-1]
                # age/max_hold clamped, weighted by lack of performance
                score = min(age / (stagnation_time * 2), 1.0) * (1.1 - pnl_pct)
                candidates.append((sym, score))
                
        if not candidates:
            return None
            
        # Best stagnation candidate
        worst_sym, highest_score = max(candidates, key=lambda x: x[1])
        
        # High threshold for non-replacement exit to avoid churn
        # But RECOVERY mode might want it higher, BOOTSTRAP might want it lower.
        stagnation_threshold = self.mode_thresholds.get(current_mode, 0.6) + 0.2
        
        if highest_score >= stagnation_threshold:
            self.logger.warning(
                "[REA:Stagnation] 🔥 AUTHORIZING PURGE: %s (age=%.1fh, score=%.2f) - Clearing for future velocity.",
                worst_sym, (now - float(owned_positions[worst_sym]['entry_time']))/3600, highest_score
            )
            
            return {
                "symbol": worst_sym,
                "action": "SELL",
                "confidence": 1.0,
                "agent": "ExitAuthority",
                "timestamp": now,
                "reason": "STAGNATION_PURGE_OVERRIDE",
                "priority": "HIGH",
                "_forced_exit": True,
                "_is_stagnation": True
            }
            
        return None

    def authorize_concentration_exit(
        self,
        owned_positions: Dict[str, Any],
        nav: float
    ) -> Optional[Dict[str, Any]]:
        """
        Layer 2: Identify if a single symbol is consuming too much portfolio bandwidth.
        Different from Layer 3 rebalancing, this is about ALPHA-weighted concentration.
        """
        if nav <= 0: return None
        
        # Max bandwidth for any single signal (e.g. 40%)
        max_bandwidth = float(getattr(self.config, "MAX_REA_CONCENTRATION_PCT", 0.4))
        
        for sym, pos in owned_positions.items():
            val = float(pos.get("value_usdt", 0.0))
            if (val / nav) > max_bandwidth:
                self.logger.warning(
                    "[REA:Concentration] 🛡️ BANDWIDTH LIMIT: %s consuming over %.1f%% of NAV. Triggering tactical exit.",
                    sym, max_bandwidth * 100
                )
                return {
                    "symbol": sym,
                    "action": "SELL",
                    "confidence": 1.0,
                    "agent": "RotationExitAuthority",
                    "reason": "CONCENTRATION_LIMIT",
                    "_forced_exit": True,
                    "allow_partial": True,
                    "target_fraction": 0.5 
                }
        return None
    def authorize_starvation_efficiency_exit(
        self,
        owned_positions: Dict[str, Any],
        nav: float,
        free_usdt: float
    ) -> Optional[Dict[str, Any]]:
        """
        Rule 4: Capital Starvation Exit
        If free_usdt < capital_floor, EXIT lowest efficiency position.
        Efficiency = unrealized_pnl_pct / hold_time_hours
        """
        if not owned_positions: return None
        
        capital_floor = float(getattr(self.config, "ABSOLUTE_MIN_FLOOR", 10.0))
        if free_usdt >= capital_floor:
            return None # Capital is healthy
            
        now = time.time()
        candidates = []
        
        for sym, pos in owned_positions.items():
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or now)
            age_hours = (now - entry_ts) / 3600.0
            pnl_pct = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
            
            # Efficiency: profit generated per hour held
            # We use age_hours + 0.1 to avoid division by zero and dampen very new positions
            efficiency = pnl_pct / (age_hours + 0.1)
            candidates.append((sym, efficiency, pos))
            
        if not candidates:
            return None
            
        # Exit the one with the LOWEST efficiency
        worst_sym, lowest_eff, worst_pos = min(candidates, key=lambda x: x[1])
        
        self.logger.warning(
            "[REA:Starvation] 🚨 CAPITAL STARVED (free=%.2f < floor=%.2f). "
            "Exiting lowest efficiency position: %s (eff=%.4f/hr)",
            free_usdt, capital_floor, worst_sym, lowest_eff
        )
        
        return {
            "symbol": worst_sym,
            "action": "SELL",
            "confidence": 1.0,
            "agent": "RotationExitAuthority",
            "reason": "STARVATION_EFFICIENCY_EXIT",
            "_forced_exit": True,
            "allow_partial": False # Full exit to maximize recovery
        }
