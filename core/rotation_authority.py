"""
Rotation & Exit Authority (REA) - P9 Canonical Design
Provides capital velocity governance by authorizing forced exits for rotation.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from utils.shared_state_tools import fee_bps


def _dynamic_exposure_cap(nav: float) -> float:
    """Return max single-symbol exposure fraction based on NAV bracket.

    Micro accounts are allowed to be concentrated (only one position possible).
    As NAV grows, diversification rules progressively tighten.
    """
    if nav < 200.0:
        return 0.90
    elif nav < 500.0:
        return 0.70
    elif nav < 2000.0:
        return 0.50
    else:
        return 0.30

class RotationExitAuthority:
    def __init__(self, logger: logging.Logger, config: Any, shared_state: Any, capital_governor=None):
        self.logger = logger
        self.config = config
        self.ss = shared_state
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE C: Capital Governor Integration
        # Enforce bracket-based rotation restrictions
        # ═══════════════════════════════════════════════════════════════════
        self.capital_governor = capital_governor
        if self.capital_governor is None:
            # Try to import if not provided (fallback)
            try:
                from core.capital_governor import CapitalGovernor
                self.capital_governor = CapitalGovernor(config)
                self.logger.info("[REA:Init] Capital Governor initialized for rotation enforcement (PHASE C)")
            except ImportError:
                self.logger.warning("[REA:Init] Capital Governor not available, rotation will not be restricted by bracket")
                self.capital_governor = None
        
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

        # Stagnation-based forced rotation controls
        self.stagnation_force_enabled = bool(
            getattr(config, "STAGNATION_FORCE_ROTATION_ENABLED", True)
        )
        self.stagnation_force_consec_cycles = int(
            getattr(
                config,
                "STAGNATION_STREAK_LIMIT",
                getattr(config, "STAGNATION_FORCE_ROTATION_CONSEC_CYCLES", 3),
            )
            or 3
        )
        self.stagnation_force_age_sec = float(
            getattr(config, "STAGNATION_AGE_SEC", 0.0) or 0.0
        )
        self.stagnation_force_age_mult = float(
            getattr(config, "STAGNATION_FORCE_ROTATION_MIN_AGE_MULT", 2.5) or 2.5
        )
        self.stagnation_force_pnl_band = float(
            getattr(
                config,
                "STAGNATION_PNL_THRESHOLD",
                getattr(config, "STAGNATION_FORCE_ROTATION_PNL_BAND", 0.0025),
            )
            or 0.0025
        )
        self.stagnation_force_sell_fraction = float(
            getattr(config, "STAGNATION_FORCE_ROTATION_SELL_FRACTION", 0.50) or 0.50
        )
        self.stagnation_continuation_min_score = float(
            getattr(config, "STAGNATION_CONTINUATION_MIN_SCORE", 0.65) or 0.65
        )
        self._stagnation_streaks: Dict[str, int] = {}
        # Track latest seen entry timestamp per symbol to detect new buys / re-entries
        self._stagnation_entry_ts: Dict[str, float] = {}
    # (legacy) we track latest seen entry timestamp per symbol in
    # _stagnation_entry_ts; no separate last_known or purge accumulators needed
        # Grace period after opening a position during which stagnation purge is disabled
        self.hold_grace_seconds = float(getattr(self.config, "HOLD_GRACE_SECONDS", 180.0) or 180.0)

    def _is_cold_bootstrap_active(self) -> bool:
        """Forced rotations must never fire during cold bootstrap."""
        checker = getattr(self.ss, "is_cold_bootstrap", None)
        if callable(checker):
            try:
                res = checker()
                if hasattr(res, "__await__"):
                    # Coroutine returned — close it to avoid RuntimeWarning,
                    # then fall through to attribute check.
                    res.close()
                    self.logger.warning("[REA] is_cold_bootstrap is async; falling back to attribute check")
                else:
                    return bool(res)
            except Exception:
                return False
        return bool(getattr(self.ss, "cold_bootstrap", False))

    def should_restrict_rotation(self, symbol: str) -> Tuple[bool, str]:
        """
        PHASE C: Check if rotation should be restricted for this symbol.
        
        Uses Capital Governor to enforce bracket-based rotation rules:
        - MICRO: ✅ Restrict (no rotation allowed - focused learning)
        - SMALL+: ❌ Allow (rotation permitted within tier limits)
        
        Args:
            symbol: Symbol being considered for rotation
            
        Returns:
            Tuple[bool, str]: (should_restrict, reason)
            - (True, "micro_bracket_restriction") if rotation should be blocked
            - (False, "") if rotation is allowed
        """
        if not self.capital_governor:
            # No Governor available, allow rotation
            return False, ""
        
        try:
            # CRITICAL: Sync authoritative balance to get fresh NAV
            if hasattr(self.ss, "sync_authoritative_balance"):
                try:
                    import asyncio
                    # Check if we're already in an async context
                    try:
                        loop = asyncio.get_running_loop()
                        # Already in async context - can't use run_until_complete
                        self.logger.debug(
                            "[REA:RotationRestriction] In async context, skipping sync"
                        )
                    except RuntimeError:
                        # No running loop - we're in sync context, safe to use run_until_complete
                        asyncio.get_event_loop().run_until_complete(
                            self.ss.sync_authoritative_balance(force=True)
                        )
                except Exception as e:
                    self.logger.debug(
                        "[REA:RotationRestriction] Balance sync unavailable: %s", e
                    )
            
            # Get current NAV from SharedState after sync
            nav = float(getattr(self.ss, "nav", 0.0) or 
                       getattr(self.ss, "total_value", 0.0) or 0.0)
            
            if nav <= 0:
                # Default to allowing if NAV unavailable
                self.logger.debug(
                    "[REA:RotationRestriction] NAV unavailable (%.2f), allowing rotation", nav
                )
                return False, ""
            
            # Check if rotation should be restricted (MICRO bracket only)
            should_restrict = self.capital_governor.should_restrict_rotation(nav)
            
            if should_restrict:
                self.logger.warning(
                    "[REA:RotationRestriction] Rotation blocked for %s: "
                    "MICRO bracket (NAV=$%.2f) - focused learning phase",
                    symbol, nav
                )
                return True, "micro_bracket_restriction"
            else:
                # Rotation allowed for this bracket
                bracket = self.capital_governor.get_bracket(nav).value
                self.logger.debug(
                    "[REA:RotationRestriction] Rotation allowed for %s: "
                    "%s bracket (NAV=$%.2f)",
                    symbol, bracket, nav
                )
                return False, ""
            
        except Exception as e:
            self.logger.error("[REA:RotationRestriction] Check failed: %s", e)
            # Graceful fallback: allow rotation on error
            return False, ""

    def _continuation_score(self, pos: Dict[str, Any]) -> float:
        """Best-effort continuation strength used to avoid rotating strong trend holds."""
        if not isinstance(pos, dict):
            return 0.0
        for key in (
            "continuation_score",
            "continuation_strength",
            "signal_strength",
            "trend_strength",
            "_continuation_confidence",
        ):
            try:
                if key in pos and pos.get(key) is not None:
                    score = float(pos.get(key) or 0.0)
                    return max(0.0, min(1.0, score))
            except Exception:
                continue
        return 0.0

    def _round_trip_fee_pct(self) -> float:
        """Round-trip fee cost as a ratio (e.g. 0.002 = 0.2%)."""
        try:
            taker_bps = float(fee_bps(self.ss, "taker") or 10.0)
            slippage_bps = float(
                getattr(self.config, "EXIT_SLIPPAGE_BPS",
                        getattr(self.config, "CR_PRICE_SLIPPAGE_BPS", 0.0)) or 0.0
            )
            return ((taker_bps * 2.0) + slippage_bps) / 10000.0
        except Exception:
            return 0.002  # conservative fallback: 20bps round-trip

    def _is_permanent_dust_position(self, symbol: str, pos: Dict[str, Any]) -> bool:
        """Permanent dust is invisible to rotation governance."""
        sym = str(symbol or "").upper()
        try:
            if hasattr(self.ss, "is_permanent_dust") and self.ss.is_permanent_dust(sym):
                return True
        except Exception:
            pass
        threshold = float(getattr(self.config, "PERMANENT_DUST_USDT_THRESHOLD", 1.0) or 1.0)
        try:
            value = float((pos or {}).get("value_usdt", 0.0) or 0.0)
            if value <= 0:
                qty = float((pos or {}).get("quantity", 0.0) or (pos or {}).get("qty", 0.0) or 0.0)
                px = 0.0
                with_ohlc = getattr(self.ss, "latest_prices", {}) or {}
                px = float(with_ohlc.get(sym, 0.0) or 0.0) if isinstance(with_ohlc, dict) else 0.0
                if qty > 0 and px > 0:
                    value = qty * px
            return bool(value > 0 and value < threshold)
        except Exception:
            return False

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
        
        # 2. PnL Efficiency (Clamped) — use net PnL after round-trip fees
        gross_pnl_pct = float(position.get("unrealized_pnl_pct", 0.0) or 0.0)
        rt_fee = self._round_trip_fee_pct()
        pnl_pct = gross_pnl_pct - rt_fee  # net of fees
        target_pnl = 0.03 # 3% target
        pnl_factor = max(-1.0, min(1.0, pnl_pct / target_pnl))

        # We want to keep winners (high pnl) and cycle losers (low/neg pnl)
        # So efficiency_score is lower for winners
        efficiency_score = 1.0 - pnl_factor

        # 3. Opportunity Cost
        # (How much better is the candidate compared to this position?)
        # best_opp_score is expected ROI * confidence; compare against net held performance
        held_score = pnl_pct * 0.5 # Basic proxy for held performance expectancy (net of fees)
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
        # ─────────────────────────────────────────────────────────────────
        # PHASE C: Capital Governor Rotation Restriction Check
        # Block rotation in MICRO bracket for focused learning
        # ─────────────────────────────────────────────────────────────────
        if owned_positions:
            first_symbol = next(iter(owned_positions.keys()), None)
            if first_symbol:
                should_restrict, reason = self.should_restrict_rotation(first_symbol)
                if should_restrict:
                    self.logger.warning(
                        "[REA:authorize_rotation] PHASE_C_BLOCK: Rotation denied for %s: %s",
                        first_symbol, reason
                    )
                    return None  # Block rotation
        
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
            if self._is_permanent_dust_position(sym, pos):
                continue
            if sym == best_opp_sym:
                continue
            
            # Filter check (e.g. TPSL protect)
            if pos.get("state") == "EXITING":
                continue
                
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or 0.0)
            if (now - entry_ts) < cooldown:
                continue

            # Winner protection: positions above threshold need extra alpha gap to rotate out
            pos_pnl = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
            if pos_pnl >= self.winner_protection_threshold:
                alpha_gap = opp_score - pos_pnl
                if alpha_gap < self.winner_extra_alpha:
                    self.logger.debug(
                        "[REA:WinnerProtect] %s pnl=%.2f%% protected (alpha_gap=%.2f%% < required=%.2f%%)",
                        sym, pos_pnl * 100, alpha_gap * 100, self.winner_extra_alpha * 100,
                    )
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
        if self._is_cold_bootstrap_active():
            self._stagnation_streaks.clear()
            self.logger.debug("[REA:Stagnation] Cold bootstrap active; stagnation rotation disabled.")
            return None

        # ─────────────────────────────────────────────────────────────────
        # PHASE C: Capital Governor Rotation Restriction Check
        # Block stagnation-based rotation in MICRO bracket
        # ─────────────────────────────────────────────────────────────────
        if owned_positions:
            first_symbol = next(iter(owned_positions.keys()), None)
            if first_symbol:
                should_restrict, reason = self.should_restrict_rotation(first_symbol)
                if should_restrict:
                    self.logger.warning(
                        "[REA:authorize_stagnation_exit] PHASE_C_BLOCK: "
                        "Stagnation-based rotation denied for %s: %s",
                        first_symbol, reason
                    )
                    return None

        # Startup grace: avoid purge during initial warm-up period after process start
        try:
            grace_min = float(getattr(self.config, "STARTUP_STAGNATION_GRACE_MINUTES", 30.0) or 30.0)
            start_ts = None
            if hasattr(self.ss, "_start_time_unix"):
                start_ts = float(getattr(self.ss, "_start_time_unix", 0.0) or 0.0)
            else:
                # fallback to common metric keys
                start_ts = float((getattr(self.ss, "metrics", {}) or {}).get("startup_time", 0.0) or 0.0)
            if start_ts and ((time.time() - start_ts) < (float(grace_min) * 60.0)):
                self.logger.debug("[REA:Stagnation] Startup grace active (%.1fmin) — skipping stagnation purge.", grace_min)
                return None
        except Exception:
            pass

        candidates = []
        forced_candidates = []
        now = time.time()
        
        # Stagnation Thresholds
        # If held for N x MAX_HOLD_SEC and PnL < 0.1%, it's a zombie.
        max_hold = float(getattr(self.config, "MAX_HOLD_SEC", 1800))
        stagnation_mult = float(getattr(self.config, "STAGNATION_HOLD_MULT", 4.0))
        stagnation_time = max_hold * stagnation_mult
        force_age = (
            float(self.stagnation_force_age_sec)
            if float(self.stagnation_force_age_sec) > 0
            else max_hold * self.stagnation_force_age_mult
        )
        
        rt_fee = self._round_trip_fee_pct()
        active_symbols = set()
        for sym, pos in owned_positions.items():
            active_symbols.add(sym)
            if self._is_permanent_dust_position(sym, pos):
                self._stagnation_streaks.pop(sym, None)
                continue
            if pos.get("state") == "EXITING":
                continue
                
            # Age is strictly computed from the open position's entry timestamp (NOT symbol activity)
            entry_ts = float(pos.get("entry_time") or pos.get("opened_at") or 0.0)
            age = now - entry_ts

            # Reset stagnation streak when the entry timestamp changes (new BUY / re-entry)
            prev_entry_ts = self._stagnation_entry_ts.get(sym)
            try:
                if prev_entry_ts is None or abs((prev_entry_ts or 0.0) - entry_ts) > 1e-6:
                    # New position detected → reset streak
                    self._stagnation_streaks[sym] = 0
                    self._stagnation_entry_ts[sym] = entry_ts
            except Exception:
                # tolerant fallback: if anything goes wrong, don't raise
                pass

            # Note: we used to separately track a "last known" entry timestamp and
            # an accumulator for purge scoring. That complexity caused inconsistencies
            # and isn't necessary — the _stagnation_entry_ts above already detects
            # entry changes and resets streaks when a new buy/re-entry occurs.

            # Hold grace: do not consider positions younger than configured hold_grace_seconds
            try:
                if entry_ts and age < float(getattr(self.config, "HOLD_GRACE_SECONDS", self.hold_grace_seconds) or self.hold_grace_seconds):
                    # Reset any transient stagnation state for fresh positions
                    self._stagnation_streaks.pop(sym, None)
                    continue
            except Exception:
                pass
            
            pnl_pct = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
            net_pnl_pct = pnl_pct - rt_fee
            continuation_score = self._continuation_score(pos)
            continuation_strong = continuation_score >= self.stagnation_continuation_min_score

            # Reset and skip if this position has become dust / zero quantity
            try:
                qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
            except Exception:
                qty = 0.0
            if qty <= 0:
                self._stagnation_streaks.pop(sym, None)
                self._stagnation_entry_ts.pop(sym, None)
                continue

            # Use net PnL (after fees) for stagnation: position is stagnant if net profit < 0
            qualifies_soft = age >= stagnation_time and net_pnl_pct < 0.0 and not continuation_strong
            qualifies_force = (
                self.stagnation_force_enabled
                and age >= force_age
                and abs(pnl_pct) <= self.stagnation_force_pnl_band
                and not continuation_strong
            )

            prev_streak = int(self._stagnation_streaks.get(sym, 0) or 0)
            if qualifies_force:
                streak = prev_streak + 1
            else:
                streak = 0
            self._stagnation_streaks[sym] = streak

            # Stagnant if very old and low/neg profit
            # (Basically it's taking up a slot and not performing)
            if qualifies_soft:
                # Calculate a stagnation score [0-1]
                # age/max_hold clamped, weighted by lack of performance
                score = min(age / (stagnation_time * 2), 1.0) * (1.1 - pnl_pct)
                candidates.append((sym, score, age, pnl_pct))

            # Forced rotation branch: explicit multi-cycle stagnation deadlock breaker
            if qualifies_force and streak >= self.stagnation_force_consec_cycles:
                force_score = min(age / max(force_age, 1.0), 2.0) + max(
                    0.0,
                    (self.stagnation_force_consec_cycles / 10.0),
                )
                forced_candidates.append((sym, force_score, age, pnl_pct, streak, continuation_score))

        # Prune symbols that no longer exist to keep streak map bounded.
        stale = [s for s in list(self._stagnation_streaks.keys()) if s not in active_symbols]
        for s in stale:
            self._stagnation_streaks.pop(s, None)
            self._stagnation_entry_ts.pop(s, None)

        if forced_candidates:
            worst_sym, force_score, age, pnl_pct, streak, continuation_score = max(forced_candidates, key=lambda x: x[1])
            self.logger.warning(
                "[REA:StagnationForce] 🚨 FORCED_ROTATION: %s age=%.1fh pnl=%.3f%% streak=%d score=%.2f continuation=%.2f",
                worst_sym,
                age / 3600.0,
                pnl_pct * 100.0,
                streak,
                force_score,
                continuation_score,
            )
            return {
                "symbol": worst_sym,
                "action": "SELL",
                "confidence": 1.0,
                "agent": "RotationExitAuthority",
                "timestamp": now,
                "reason": "FORCED_ROTATION_STAGNATION",
                "tag": "meta-rotation_authority",
                "priority": "HIGH",
                "_forced": True,
                "_is_rotation": True,
                "_forced_exit": True,
                "_is_stagnation": True,
                "_stagnation_force": True,
                "_stagnation_override": True,
                "_stagnation_streak": int(streak),
                "_continuation_score": float(continuation_score),
                "_rotation_stage": "forced",
                "allow_partial": True,
                "target_fraction": max(0.10, min(1.0, self.stagnation_force_sell_fraction)),
            }
                
        if not candidates:
            return None
            
        # Best stagnation candidate
        worst_sym, highest_score, worst_age, worst_pnl = max(candidates, key=lambda x: x[1])
        
        # High threshold for non-replacement exit to avoid churn
        # But RECOVERY mode might want it higher, BOOTSTRAP might want it lower.
        stagnation_threshold = self.mode_thresholds.get(current_mode, 0.6) + 0.2
        
        if highest_score >= stagnation_threshold:
            self.logger.warning(
                "[REA:Stagnation] 🔥 AUTHORIZING PURGE: %s (age=%.1fh, score=%.2f) - Clearing for future velocity.",
                worst_sym, (worst_age / 3600.0), highest_score
            )
            
            return {
                "symbol": worst_sym,
                "action": "SELL",
                "confidence": 1.0,
                "agent": "RotationExitAuthority",
                "timestamp": now,
                "reason": "STAGNATION_ROTATION_NOMINATION",
                "tag": "meta-rotation_authority",
                "priority": "HIGH",
                "_forced": False,
                "_is_rotation": True,
                "_forced_exit": True,
                "_is_stagnation": True,
                "_stagnation_pnl_pct": float(worst_pnl),
                "_rotation_stage": "nomination",
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

        # Dynamic cap: config override takes precedence; fallback to NAV-bracket table.
        config_cap = float(getattr(self.config, "MAX_REA_CONCENTRATION_PCT", 0.0) or 0.0)
        cap = config_cap if config_cap > 0 else _dynamic_exposure_cap(nav)
        self.logger.debug("[REA:Concentration] DynamicExposure NAV=%.2f → cap=%.0f%%", nav, cap * 100)

        for sym, pos in owned_positions.items():
            if self._is_permanent_dust_position(sym, pos):
                continue
            val = float(pos.get("value_usdt", 0.0))
            if (val / nav) > cap:
                self.logger.warning(
                    "[REA:Concentration] 🛡️ BANDWIDTH LIMIT: %s consuming over %.0f%% of NAV (NAV=%.2f). Triggering tactical exit.",
                    sym, cap * 100, nav
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
            if self._is_permanent_dust_position(sym, pos):
                continue
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
