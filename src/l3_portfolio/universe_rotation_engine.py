# core/universe_rotation_engine.py
"""
Unified Universe Rotation Engine (UURE)

THE CANONICAL SYMBOL AUTHORITY

Problem Being Solved:
  - Governor only trims by count (wrong)
  - PortfolioBalancer trims by score (too late)
  - Discovery feeds arbitrary lists
  - Result: Non-deterministic universe, race conditions

Solution:
  Single pipeline from discovery → ranking → cap → commit

Architecture:
  Discovery (wide net: 50-200 candidates)
    ↓
  Unified Scoring (score ALL candidates)
    ↓
  Global Ranking (sort descending by score)
    ↓
  Governor Cap (compute dynamic cap)
    ↓
  Hard Replace Universe (exact top-N)
    ↓
  Rotation Cleanup (liquidate weak symbols)

Key Properties:
  ✅ Deterministic: Same inputs → same universe
  ✅ Score-based: Keeps best symbols, not first symbols
  ✅ Governor-aware: Respects capital constraints
  ✅ Rotation-ready: Can swap weak ↔ strong
  ✅ Canonical: Single source of truth for universe

Usage:
  engine = UniverseRotationEngine(shared_state, capital_governor, config)
  await engine.compute_and_apply_universe()
  
This replaces:
  - Scattered governor calls
  - PortfolioBalancer deciding universe
  - Discovery-driven arbitrary lists
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from types import SimpleNamespace

from utils.shared_state_tools import spread_bps as ss_spread_bps, min_notional as ss_min_notional


logger = logging.getLogger("UniverseRotationEngine")


class UniverseRotationEngine:
    """
    Canonical authority for symbol universe.
    
    Responsibilities:
    1. Collect all candidate symbols from discovery
    2. Score each candidate (unified scoring)
    3. Rank by score (descending)
    4. Apply governor cap
    5. Identify rotation (symbols to add/remove)
    6. Commit hard-replace to SharedState
    7. Trigger liquidation of removed symbols
    """

    def __init__(
        self,
        shared_state: Any,
        capital_governor: Optional[Any] = None,
        config: Optional[Any] = None,
        execution_manager: Optional[Any] = None,
        meta_controller: Optional[Any] = None,
        logger: Optional[Any] = None,
        capital_symbol_governor: Optional[Any] = None,
        **_: Any,
    ):
        """
        Args:
            shared_state: SharedState for symbol universe & pricing
            capital_governor: CapitalSymbolGovernor for dynamic cap
            config: Configuration (MIN_ENTRY_QUOTE_USDT, MAX_SYMBOL_LIMIT, etc.)
            execution_manager: For liquidation of removed symbols
            meta_controller: For trade intents
        """
        self.ss = shared_state
        # Accept both naming variants for compatibility with AppContext wiring.
        self.governor = capital_governor or capital_symbol_governor
        self.config = config or SimpleNamespace()
        self.exec = execution_manager
        self.mc = meta_controller

        self.logger = logger or logging.getLogger("UniverseRotationEngine")

        # Config extraction
        self.min_entry_quote = float(self._cfg("MIN_ENTRY_QUOTE_USDT", 20.0))
        self.max_symbol_limit = int(self._cfg("MAX_SYMBOL_LIMIT", 30))
        # Use MAX_TOTAL_EXPOSURE_PCT from Config (default 0.6), not hardcoded 0.8
        # This ensures universe cap respects actual portfolio allocation settings
        self.max_exposure = float(self._cfg("MAX_TOTAL_EXPOSURE_PCT", 0.6))
        # Cycle-local snapshot of proposal metadata (captured before proposal stores are cleared).
        self._latest_proposal_snapshot: Dict[str, Dict[str, Any]] = {}

    def wire_runtime_dependencies(
        self,
        *,
        capital_governor: Optional[Any] = None,
        execution_manager: Optional[Any] = None,
        meta_controller: Optional[Any] = None,
    ) -> None:
        """Hot-wire dependencies that may be constructed after UURE."""
        if capital_governor is not None:
            self.governor = capital_governor
        if execution_manager is not None:
            self.exec = execution_manager
        if meta_controller is not None:
            self.mc = meta_controller

    # ============================================================================
    # PHASE 1: EV ALIGNMENT METHODS
    # ============================================================================
    # Expose EV calculation methods for validation and alignment with ExecutionManager
    
    def get_round_trip_cost_pct(self) -> float:
        """
        PHASE 1: PUBLIC API for round-trip cost calculation.
        
        This method should return EXACTLY the same value as ExecutionManager.
        Used for EV alignment validation.
        
        Returns:
            Round-trip cost as decimal (e.g., 0.0035 = 0.35%)
        """
        return self._round_trip_cost_pct()
    
    def get_ev_multiplier_for_regime(self, regime: str) -> float:
        """
        PHASE 1: PUBLIC API for EV multiplier by regime.
        
        This method should return the same multiplier as ExecutionManager would use
        for the same regime. Used for alignment validation.
        
        Args:
            regime: Market volatility regime ('normal', 'bull', 'other', etc.)
        
        Returns:
            EV multiplier (e.g., 1.3 means required_edge = 0.35% × 1.3 = 0.455%)
        """
        return self._ev_multiplier_for_regime(regime)
    
    def get_required_edge_for_regime(self, regime: str) -> float:
        """
        PHASE 1: PUBLIC API for required minimum edge by regime.
        
        This is the minimum edge (expressed as decimal) that must be present
        for a symbol to be admitted to the universe. Same formula as ExecutionManager uses.
        
        Args:
            regime: Market regime
        
        Returns:
            Required edge as decimal (e.g., 0.00455 = 0.455%)
        """
        round_trip = self.get_round_trip_cost_pct()
        multiplier = self.get_ev_multiplier_for_regime(regime)
        return round_trip * multiplier
    
    def get_ev_config_summary(self) -> Dict[str, Any]:
        """
        PHASE 1: Export EV configuration for alignment audit.
        
        Returns a dict summarizing current EV settings used by UURE.
        Compare this with ExecutionManager's settings to verify alignment.
        
        Returns:
            Dictionary with EV configuration details
        """
        spot_mode = bool(self._cfg("UURE_SPOT_MODE", False))
        return {
            "round_trip_cost_pct": self.get_round_trip_cost_pct(),
            "ev_mult_normal": self.get_ev_multiplier_for_regime("normal"),
            "ev_mult_bull": self.get_ev_multiplier_for_regime("bull"),
            "ev_mult_other": self.get_ev_multiplier_for_regime("other"),
            "spot_mode_enabled": spot_mode,
            "required_edge_normal": self.get_required_edge_for_regime("normal"),
            "required_edge_bull": self.get_required_edge_for_regime("bull"),
            "required_edge_other": self.get_required_edge_for_regime("other"),
            "source": "UniverseRotationEngine (PHASE 1 EV ALIGNMENT)",
        }

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Config getter that supports dict or attribute configs."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
            return default
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _looks_like_leveraged_symbol(symbol: str, base: str) -> bool:
        base_u = str(base or "").upper()
        sym_u = str(symbol or "").upper()
        leveraged_suffixes = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")
        return any(base_u.endswith(sfx) or sym_u.endswith(f"{sfx}USDT") for sfx in leveraged_suffixes)

    def _reject_discovery_symbol(self, symbol: str, proposal: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        if not self._as_bool(self._cfg("DISCOVERY_FILTER_LOW_UTILITY_SYMBOLS", True), True):
            return False, ""
        sym = str(symbol or "").replace("/", "").strip().upper()
        if not sym:
            return True, "empty_symbol"

        quote = str(self._cfg("QUOTE_ASSET", "USDT") or "USDT").upper()
        base = sym
        if quote and sym.endswith(quote):
            base = sym[: -len(quote)]

        if not base:
            return True, "invalid_symbol"

        stable_csv = str(
            self._cfg(
                "DISCOVERY_STABLE_ASSETS",
                "USDT,USDC,FDUSD,BUSD,TUSD,USDP,DAI,USDE,USD1,USDS,USD0",
            )
            or ""
        )
        stable_assets = {token.strip().upper() for token in stable_csv.split(",") if token.strip()}
        if self._as_bool(self._cfg("DISCOVERY_REJECT_STABLE_BASE_PAIRS", True), True):
            if base in stable_assets:
                return True, "stablecoin_base_pair"

        if self._as_bool(self._cfg("DISCOVERY_REJECT_LEVERAGED_TOKENS", True), True):
            if self._looks_like_leveraged_symbol(sym, base):
                return True, "leveraged_token"

        min_vol = float(self._cfg("DISCOVERY_MIN_PROPOSAL_VOLUME_USDT", 0.0) or 0.0)
        if min_vol > 0.0 and isinstance(proposal, dict):
            try:
                vol = float(
                    proposal.get("24h_volume")
                    or proposal.get("volume_24h")
                    or proposal.get("quote_volume")
                    or 0.0
                )
                if 0.0 < vol < min_vol:
                    return True, "volume_below_floor"
            except Exception:
                pass
        return False, ""

    async def _safe_price(self, symbol: str) -> float:
        sym = str(symbol or "").replace("/", "").upper()
        try:
            lp = getattr(self.ss, "latest_prices", {}) or {}
            if isinstance(lp, dict):
                px = float(lp.get(sym, 0.0) or 0.0)
                if px > 0:
                    return px
        except Exception:
            pass
        try:
            safe_price = getattr(self.ss, "safe_price", None)
            if callable(safe_price):
                res = safe_price(sym, default=0.0)
                res = await res if asyncio.iscoroutine(res) else res
                px = float(res or 0.0)
                if px > 0:
                    return px
        except Exception:
            pass
        return 0.0

    async def _maybe_await(self, value: Any) -> Any:
        """Compatibility helper: accept sync values or awaitables."""
        if asyncio.iscoroutine(value):
            return await value
        return value

    def _round_trip_cost_pct(self) -> float:
        """
        Best-effort round-trip cost as a ratio (e.g. 0.0025 = 0.25%).
        Mirrors ExecutionManager EV hard-gate cost composition: fees (round-trip) + slippage + buffer.
        
        Can be overridden by UURE_ROUND_TRIP_COST_PCT for spot trading tuning.
        """
        # Check for explicit override (useful for spot trading tuning)
        override = self._cfg("UURE_ROUND_TRIP_COST_PCT", None)
        if override is not None:
            try:
                return max(0.0, float(override))
            except Exception:
                pass
        
        fee_pct = 0.0
        try:
            if self.exec is not None and getattr(self.exec, "trade_fee_pct", None) is not None:
                fee_pct = float(getattr(self.exec, "trade_fee_pct") or 0.0)
        except Exception:
            fee_pct = 0.0
        if fee_pct <= 0:
            fee_bps = float(self._cfg("EXIT_FEE_BPS", self._cfg("CR_FEE_BPS", 10.0)) or 10.0)
            fee_pct = fee_bps / 10000.0

        slippage_bps = float(self._cfg("EXIT_SLIPPAGE_BPS", self._cfg("CR_PRICE_SLIPPAGE_BPS", 15.0)) or 0.0)
        buffer_bps = float(self._cfg("TP_MIN_BUFFER_BPS", 0.0) or 0.0)
        return (fee_pct * 2.0) + ((slippage_bps + buffer_bps) / 10000.0)

    async def _volatility_regime_1h(self, symbol: str) -> str:
        sym = str(symbol or "").replace("/", "").upper()
        tf = str(self._cfg("VOLATILITY_REGIME_TIMEFRAME", "1h") or "1h")
        try:
            get_regime = getattr(self.ss, "get_volatility_regime", None)
            if callable(get_regime):
                rr = await get_regime(sym, timeframe=tf, max_age_seconds=600)
                if not rr:
                    rr = await get_regime("GLOBAL", timeframe=tf, max_age_seconds=600)
                if isinstance(rr, dict):
                    return str(rr.get("regime") or "").strip().lower()
        except Exception:
            pass
        try:
            return str((getattr(self.ss, "metrics", {}) or {}).get("volatility_regime") or "").strip().lower()
        except Exception:
            return ""

    async def _expected_move_pct_1h(self, symbol: str, price: float) -> float:
        """Expected move proxy for a 1h horizon using ATR% (1h preferred, 5m fallback scaled)."""
        if price <= 0:
            return 0.0
        sym = str(symbol or "").replace("/", "").upper()
        period = int(self._cfg("UURE_EXPECTED_MOVE_ATR_PERIOD", self._cfg("VOLATILITY_REGIME_ATR_PERIOD", 14)) or 14)
        atr_tf = str(self._cfg("UURE_EXPECTED_MOVE_TIMEFRAME", "1h") or "1h")
        atr_val = 0.0
        try:
            if hasattr(self.ss, "calc_atr"):
                atr_val = float(await self.ss.calc_atr(sym, atr_tf, period) or 0.0)
        except Exception:
            atr_val = 0.0
        if atr_val <= 0:
            # Fallback: use 5m ATR scaled to 1h horizon (~sqrt(12) clamped ≈ 3).
            try:
                if hasattr(self.ss, "calc_atr"):
                    atr_5m = float(await self.ss.calc_atr(sym, "5m", period) or 0.0)
                    if atr_5m > 0:
                        scale = float(self._cfg("UURE_EXPECTED_MOVE_5M_TO_1H_SCALE", 3.0) or 3.0)
                        atr_val = atr_5m * max(1.0, float(scale))
            except Exception:
                atr_val = 0.0
        if atr_val <= 0:
            return 0.0
        return float(atr_val) / max(float(price), 1e-9)

    def _disable_sideways_in_uure(self) -> bool:
        return bool(
            self._cfg(
                "UURE_DISABLE_SIDEWAYS_REGIME_TRADING",
                self._cfg("DISABLE_SIDEWAYS_REGIME_TRADING", False),
            )
        )

    def _estimate_volatility_regime(self) -> str:
        """
        Estimate current market volatility regime using multiple signals.
        
        Returns one of: "low", "normal", "high", "extreme"
        """
        try:
            # Signal 1: VIX-like indicator from metrics
            vix_like = 20.0  # default
            try:
                metrics = getattr(self.ss, "metrics", {}) or {}
                vix_like = float(metrics.get("vix_like", 20.0) or 20.0)
            except Exception:
                pass
            
            # Signal 2: Average ATR ratio from active symbols
            atr_ratio_avg = 2.0  # default
            try:
                universe = getattr(self.ss, "universe", []) or []
                atr_ratios = []
                for sym in list(universe)[:10]:  # Sample first 10 symbols
                    try:
                        hist = getattr(self.ss, f"ohlcv_{sym}", None)
                        if hist and len(hist) > 14:
                            closes = [float(h.get("close", 0)) for h in hist[-14:]]
                            highs = [float(h.get("high", 0)) for h in hist[-14:]]
                            lows = [float(h.get("low", 0)) for h in hist[-14:]]
                            if closes and min(closes) > 0:
                                tr = [max(h - l, abs(h - closes[i-1]) if i > 0 else 0, abs(l - closes[i-1]) if i > 0 else 0)
                                      for i, (h, l) in enumerate(zip(highs, lows))]
                                atr = sum(tr) / len(tr)
                                atr_ratio = atr / sum(closes) * len(closes)
                                atr_ratios.append(atr_ratio)
                    except Exception:
                        pass
                if atr_ratios:
                    atr_ratio_avg = sum(atr_ratios) / len(atr_ratios)
            except Exception:
                pass
            
            # Signal 3: Bid-ask spread sampling
            spread_avg = 0.1  # default 0.1%
            try:
                universe = getattr(self.ss, "universe", []) or []
                spreads = []
                for sym in list(universe)[:5]:
                    try:
                        orderbook = getattr(self.ss, f"orderbook_{sym}", None)
                        if orderbook and "bids" in orderbook and "asks" in orderbook:
                            bids = orderbook.get("bids", [])
                            asks = orderbook.get("asks", [])
                            if bids and asks:
                                best_bid = float(bids[0][0]) if bids[0] else 0
                                best_ask = float(asks[0][0]) if asks[0] else 0
                                if best_bid > 0 and best_ask > best_bid:
                                    spread_pct = (best_ask - best_bid) / best_bid * 100
                                    spreads.append(spread_pct)
                    except Exception:
                        pass
                if spreads:
                    spread_avg = sum(spreads) / len(spreads)
            except Exception:
                pass
            
            # Combine signals to classify regime
            # Classification thresholds:
            # Low: vix < 15, atr < 1.5, spread < 0.05%
            # Normal: vix 15-25, atr 1.5-2.5, spread 0.05-0.15%
            # High: vix 25-40, atr 2.5-4.0, spread 0.15-0.3%
            # Extreme: vix > 40, atr > 4.0, spread > 0.3%
            
            signal_count = 0
            high_count = 0
            extreme_count = 0
            
            if vix_like > 40 or vix_like < 0:
                extreme_count += 1
            elif vix_like > 25:
                high_count += 1
            signal_count += 1
            
            if atr_ratio_avg > 4.0:
                extreme_count += 1
            elif atr_ratio_avg > 2.5:
                high_count += 1
            signal_count += 1
            
            if spread_avg > 0.3:
                extreme_count += 1
            elif spread_avg > 0.15:
                high_count += 1
            signal_count += 1
            
            if extreme_count >= signal_count * 0.5:
                return "extreme"
            elif high_count >= signal_count * 0.5:
                return "high"
            elif vix_like < 15 and atr_ratio_avg < 1.5 and spread_avg < 0.05:
                return "low"
            else:
                return "normal"
        except Exception:
            return "normal"

    def _get_dynamic_profile(self) -> Optional[Dict[str, Any]]:
        """
        Get dynamically adjusted parameter profile based on current market state.
        
        Returns dict with:
          - regime: volatility regime classification
          - regime_strength: confidence in regime (0.0-1.0)
          - ev_mult_normal: adjusted EV multiplier for normal regime
          - ev_mult_bull: adjusted EV multiplier for bull regime
          - ev_mult_other: adjusted EV multiplier for other regimes
          - vix_like, atr_ratio_avg, spread_avg: signal values
          - timestamp: profile creation time
        """
        # Check if we have a cached profile that's still valid (30 second TTL)
        if not hasattr(self, "_profile_cache"):
            self._profile_cache = {"profile": None, "timestamp": 0}
        
        import time
        current_time = time.time()
        cache_validity = float(self._cfg("UURE_PROFILE_CACHE_VALIDITY_SEC", 30.0) or 30.0)
        
        if self._profile_cache["profile"] is not None and (current_time - self._profile_cache["timestamp"]) < cache_validity:
            return self._profile_cache["profile"]
        
        try:
            # Estimate volatility regime
            regime = self._estimate_volatility_regime()
            
            # Get base multipliers
            base_mult_normal = 1.3
            base_mult_bull = 1.8
            base_mult_other = 2.0
            
            # Adjust based on regime
            if regime == "extreme":
                regime_strength = 0.9
                mult_adjustment = 1.5  # Tighten filters significantly
            elif regime == "high":
                regime_strength = 0.8
                mult_adjustment = 1.2  # Tighten filters
            elif regime == "low":
                regime_strength = 0.7
                mult_adjustment = 0.9  # Relax filters slightly
            else:  # normal
                regime_strength = 0.6
                mult_adjustment = 1.0
            
            # Get recent performance metrics to fine-tune
            try:
                win_rate = 0.5
                try:
                    trades = getattr(self.ss, "recent_trades", []) or []
                    if trades:
                        wins = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
                        win_rate = wins / len(trades) if trades else 0.5
                except Exception:
                    pass
                
                # If performance is poor, tighten filters
                if win_rate < 0.45:
                    mult_adjustment *= 1.1
                elif win_rate > 0.55:
                    mult_adjustment *= 0.95
            except Exception:
                pass
            
            profile = {
                "regime": regime,
                "regime_strength": regime_strength,
                "ev_mult_normal": max(0.5, base_mult_normal * mult_adjustment),
                "ev_mult_bull": max(0.5, base_mult_bull * mult_adjustment),
                "ev_mult_other": max(0.5, base_mult_other * mult_adjustment),
                "timestamp": current_time,
            }
            
            # Cache the profile
            self._profile_cache = {"profile": profile, "timestamp": current_time}
            
            self.logger.debug(
                "[UURE] Dynamic profile: regime=%s, strength=%.2f, "
                "ev_mults=(normal=%.2f, bull=%.2f, other=%.2f)",
                regime,
                regime_strength,
                profile["ev_mult_normal"],
                profile["ev_mult_bull"],
                profile["ev_mult_other"],
            )
            
            return profile
        except Exception as e:
            self.logger.warning("[UURE] Error computing dynamic profile: %s", str(e))
            return None

    def _ev_multiplier_for_regime(self, regime: str) -> float:
        """
        Get EV multiplier for a given regime.
        
        Controls required edge threshold:
          required_edge = round_trip_cost × multiplier
        
        Lower multiplier = lower barrier to entry (good for spot trading).
        
        Configs:
          UURE_SOFT_EV_MULTIPLIER: Override all regimes (e.g., 0.8 for loose spot)
          UURE_EV_MULT_NORMAL: Normal regime (default 1.3, for spot try 0.9-1.0)
          UURE_EV_MULT_BULL: Bull regime (default 1.8, for spot try 1.2-1.3)
          UURE_EV_MULT_OTHER: Other regimes (default 2.0, for spot try 1.5-1.8)
          UURE_SPOT_MODE: If True, use relaxed spot-trading multipliers (0.7, 1.0, 1.4)
        """
        rg = str(regime or "").strip().lower()
        
        # Check for global override
        override = self._cfg("UURE_SOFT_EV_MULTIPLIER", None)
        if override is not None:
            try:
                return max(0.5, float(override))
            except Exception:
                pass
        
        # Check for spot mode (all regimes use relaxed thresholds)
        spot_mode = bool(self._cfg("UURE_SPOT_MODE", False))
        if spot_mode:
            if rg == "normal":
                return max(0.5, float(self._cfg("UURE_EV_MULT_SPOT_NORMAL", 0.7)))
            elif rg == "bull":
                return max(0.5, float(self._cfg("UURE_EV_MULT_SPOT_BULL", 1.0)))
            else:
                return max(0.5, float(self._cfg("UURE_EV_MULT_SPOT_OTHER", 1.4)))
        
        # Use dynamic profile if available
        profile = self._get_dynamic_profile()
        if profile is not None:
            if rg == "normal":
                return max(0.5, float(profile.get("ev_mult_normal", 1.3)))
            elif rg == "bull":
                return max(0.5, float(profile.get("ev_mult_bull", 1.8)))
            else:
                return max(0.5, float(profile.get("ev_mult_other", 2.0)))
        
        # Fallback to regime-specific config
        if rg == "normal":
            default = 1.3
            key = "UURE_EV_MULT_NORMAL"
        elif rg == "bull":
            default = 1.8
            key = "UURE_EV_MULT_BULL"
        else:
            default = 2.0
            key = "UURE_EV_MULT_OTHER"
        try:
            return max(0.5, float(self._cfg(key, default) or default))
        except Exception:
            return default

    async def compute_and_apply_universe(self) -> Dict[str, Any]:
        """
        Main entry point: Compute new universe and apply it.
        
        Returns dict with:
          - new_universe: List of symbols to keep
          - score_info: Scoring details
          - rotation: {added, removed, kept} symbols
          - execution: What trades were triggered
        """
        result = {
            "new_universe": [],
            "score_info": {},
            "rotation": {"added": [], "removed": [], "kept": []},
            "execution": [],
            "error": None,
        }

        try:
            # Step 1: Collect all candidate symbols
            self.logger.info("[UURE] Starting universe rotation cycle")
            all_candidates = await self._collect_candidates()
            if not all_candidates:
                self.logger.warning("[UURE] No candidates found")
                return result

            # Step 2: Score all candidates
            scored = await self._score_all(all_candidates)
            if not scored:
                self.logger.warning("[UURE] Scoring failed")
                return result

            # Step 3: Rank by score
            ranked = self._rank_by_score(scored)
            self.logger.info(
                f"[UURE] Ranked {len(ranked)} candidates. Top 5: {ranked[:5]}"
            )

            # Step 4: Apply governor cap
            capped = await self._apply_governor_cap(ranked)
            self.logger.info(
                f"[UURE] Governor cap applied: {len(ranked)} → {len(capped)}"
            )

            # Step 4.5: Apply profitability filter (ExecutionManager EV logic)
            profitable = await self._apply_profitability_filter(capped)
            self.logger.info(
                f"[UURE] Profitability filter applied: {len(capped)} → {len(profitable)}"
            )

            # Step 4.6: Apply relative replacement rule (only rotate if superior edge)
            current_universe = self.ss.get_accepted_symbol_list()
            final_universe = await self._apply_relative_replacement_rule(
                profitable, current_universe
            )
            self.logger.info(
                f"[UURE] Relative replacement rule applied: {len(profitable)} → {len(final_universe)}"
            )

            # Step 5: Identify candidate rotation
            candidate_rotation = await self._identify_rotation(final_universe)
            self.logger.info(
                f"[UURE] Candidate rotation: +{len(candidate_rotation['added'])} "
                f"-{len(candidate_rotation['removed'])} ={len(candidate_rotation['kept'])}"
            )

            # Step 6: Hard replace universe
            applied_universe = await self._hard_replace_universe(final_universe)
            rotation = await self._identify_rotation(applied_universe)
            self.logger.info(
                f"[UURE] Universe applied: {len(applied_universe)} symbols "
                f"(effective rotation +{len(rotation['added'])} -{len(rotation['removed'])} ={len(rotation['kept'])})"
            )

            # Step 7: Trigger liquidation of effectively removed symbols only.
            if rotation["removed"]:
                await self._trigger_liquidation(rotation["removed"])

            result["new_universe"] = applied_universe
            result["score_info"] = {sym: scored[sym] for sym in applied_universe if sym in scored}
            result["rotation"] = rotation
            result["candidate_rotation"] = candidate_rotation
            result["execution"] = rotation["removed"]  # Liquidated symbols

            return result

        except Exception as e:
            self.logger.error(f"[UURE] Error in compute_and_apply_universe: {e}")
            result["error"] = str(e)
            return result

    async def _is_dust_position(self, symbol: str) -> bool:
        """
        Check if a position is dust (value below minimum tradable notional).
        
        Returns True if position value is:
        - Less than dust_min_quote_usdt, OR
        - Less than MIN_POSITION_VALUE_USDT, OR
        - Less than symbol's exchange minNotional
        
        This prevents scoring and rotating dust positions.
        """
        try:
            # Get position from shared_state
            positions = await self._maybe_await(self.ss.get_positions_snapshot())
            if symbol not in positions:
                return False
            
            pos = positions[symbol]
            qty = float(pos.get("quantity", 0.0) or 0.0)
            
            # Zero qty = dust (closed position)
            if qty <= 0:
                return True
            
            # Get price
            price = await self._safe_price(symbol)
            if price <= 0:
                return True  # No price = can't trade = dust
            
            # Calculate notional value
            notional = qty * price
            
            # Get thresholds
            dust_floor = float(self._cfg("dust_min_quote_usdt", 5.0))
            min_position_value = float(self._cfg("MIN_POSITION_VALUE_USDT", 10.0))
            
            # Try to get exchange minNotional
            try:
                filters = await self.ss.get_symbol_filters_cached(symbol) if hasattr(self.ss, "get_symbol_filters_cached") else None
                min_notional = float(filters.get("minNotional", 10.0) or 10.0) if filters else 10.0
            except Exception:
                min_notional = 10.0
            
            # Use highest threshold as the barrier
            threshold = max(dust_floor, min_position_value, min_notional)
            is_dust = notional < threshold
            
            if is_dust:
                self.logger.debug(
                    f"[UURE:DustFilter] {symbol}: notional=${notional:.2f} < threshold=${threshold:.2f} "
                    f"(dust_floor=${dust_floor:.2f}, min_pos=${min_position_value:.2f}, min_notional=${min_notional:.2f})"
                )
            
            return is_dust
        
        except Exception as e:
            self.logger.debug(f"[UURE:DustFilter] Error checking dust for {symbol}: {e}")
            return False  # Treat unknown as non-dust (safe default)

    async def _collect_candidates(self) -> List[str]:
        """Step 1: Collect all candidate symbols from discovery & current positions.

        ARCHITECTURE (Professional Standard):
          Trading Universe != Wallet Assets.

          Three distinct sources are combined here:
            1. accepted_syms  — symbols currently in the accepted universe
            2. position_syms  — positions the BOT placed (NOT wallet-only mirrors)
            3. discovery_syms — proposals from discovery agents (SymbolScreener, etc.)

          Wallet-hydrated positions (_mirrored=True) are deliberately excluded.
          The wallet is a STATE RECONCILIATION source, not a discovery source.
          Symbols enter the universe only through proposers -> filters -> UURE.
        """
        try:
            # Get symbols from accepted set
            accepted = await self._maybe_await(self.ss.get_accepted_symbols())
            accepted_syms = set(accepted.keys())

            # Get BOT-MANAGED positions only — exclude wallet-mirror positions.
            # hydrate_positions_from_balances() tags wallet-reconstructed positions
            # with _mirrored=True. We must NOT feed those back into the universe:
            # that would let arbitrary wallet assets (GAS, NEO, TRX...) contaminate
            # the trading universe. The wallet is for reconciliation, not discovery.
            positions = await self._maybe_await(self.ss.get_positions_snapshot())
            position_syms = set(
                sym
                for sym, pos in positions.items()
                if not pos.get("_mirrored", False)
            )
            wallet_only_count = len(positions) - len(position_syms)

            # ✅ PHASE 2c: DISCOVERY PROPOSALS - Wire discovery proposals into UURE
            # This is the KEY CHANGE that enables 10x candidate expansion
            discovery_syms = await self._collect_discovery_proposals_weighted()

            # Union of all sources (accepted + bot-managed positions + discovery)
            all_syms = accepted_syms | position_syms | discovery_syms

            # ✅ DUST FILTER: Exclude dust positions from rotation cycle
            non_dust_syms = []
            dust_count = 0
            for sym in all_syms:
                if await self._is_dust_position(sym):
                    dust_count += 1
                else:
                    non_dust_syms.append(sym)

            self.logger.info(
                f"[UURE] Candidates: {len(accepted_syms)} accepted, "
                f"{len(position_syms)} bot-positions "
                f"({wallet_only_count} wallet-only excluded), "
                f"{len(discovery_syms)} discovery, "
                f"{len(all_syms)} total, filtered {dust_count} dust → {len(non_dust_syms)} viable"
            )
            return non_dust_syms

        except Exception as e:
            self.logger.error(f"[UURE] Error collecting candidates: {e}")
            return []

    async def _collect_discovery_proposals(self) -> Set[str]:
        """
        ✅ PHASE 2c: Collect discovery proposals from DiscoveryCoordinator.

        This is the KEY INTEGRATION POINT that expands UURE's candidate pool
        from 2-5 symbols to 20-30+ symbols by pulling from discovery agents.

        Reads from TWO stores (union):
          1. ss.discovery_proposals  — written by DiscoveryCoordinator (if wired)
          2. ss.symbol_proposals     — written directly by SymbolScreener, IPOChaser,
                                       TrendHunter, WalletScannerAgent as fallback

        Returns:
            Set of symbol strings proposed by discovery agents
        """
        try:
            # Primary store (DiscoveryCoordinator output)
            proposals = getattr(self.ss, "discovery_proposals", None) or {}
            # Fallback store (direct agent writes — always present even without DiscoveryCoordinator)
            symbol_proposals = getattr(self.ss, "symbol_proposals", None) or {}

            if not proposals and not symbol_proposals:
                self.logger.debug("[UURE] No discovery proposals available this cycle")
                self._latest_proposal_snapshot = {}
                return set()

            # Extract symbols from both stores
            discovery_syms: Set[str] = set()
            filtered_counts: Dict[str, int] = {}
            proposal_snapshot: Dict[str, Dict[str, Any]] = {}

            for source_store in (proposals, symbol_proposals):
                for symbol, prop in source_store.items():
                    if isinstance(prop, dict):
                        sym = str(prop.get("symbol", symbol)).upper()
                    else:
                        sym = str(symbol).upper()
                    if sym:
                        reject, reason = self._reject_discovery_symbol(
                            sym,
                            prop if isinstance(prop, dict) else None,
                        )
                        if reject:
                            filtered_counts[reason] = filtered_counts.get(reason, 0) + 1
                            continue
                        discovery_syms.add(sym)
                        if isinstance(prop, dict):
                            existing = proposal_snapshot.get(sym, {}) or {}
                            merged = dict(existing)
                            merged.update(prop)
                            proposal_snapshot[sym] = merged

            # Keep cycle-local snapshot for quality scoring even after stores are cleared.
            self._latest_proposal_snapshot = proposal_snapshot

            if filtered_counts:
                rejected = sum(filtered_counts.values())
                self.logger.info(
                    "[UURE] Discovery quality filter rejected %d proposal(s): %s",
                    rejected,
                    filtered_counts,
                )

            self.logger.info(
                f"[UURE] Collected {len(discovery_syms)} discovery proposals "
                f"({len(proposals)} from discovery_proposals, "
                f"{len(symbol_proposals)} from symbol_proposals): "
                f"{list(discovery_syms)[:8]}..."
            )
            
            # ✅ FIX #8 PART 2: Clear proposals after collection to prevent accumulation
            # Each UURE cycle should see fresh discoveries from the last agent run
            self.ss.discovery_proposals.clear()
            self.ss.symbol_proposals.clear()
            self.logger.debug("[UURE] Cleared discovery_proposals and symbol_proposals for next cycle")
            
            return discovery_syms

        except Exception as e:
            self.logger.error(f"[UURE] Error collecting discovery proposals: {e}")
            self._latest_proposal_snapshot = {}
            return set()

    async def _collect_discovery_proposals_weighted(self) -> Set[str]:
        """
        ✅ PHASE 3: Collect discovery proposals with regime-based weighting.
        
        Enhanced version that prefers regime-aligned proposals for better signal quality.
        
        Returns:
            Set of symbol strings, ordered by weighted_score (descending)
        """
        try:
            use_weighting = bool(self._cfg("DISCOVERY_USE_REGIME_WEIGHTING", False))
            
            if not use_weighting:
                # Fall back to Phase 2c behavior (no weighting)
                return await self._collect_discovery_proposals()
            
            # Get weighted proposals (if available)
            weighted_props = getattr(self.ss, "discovery_proposals_weighted", None) or {}
            
            if not weighted_props:
                self.logger.debug("[UURE] No weighted discovery proposals available")
                return await self._collect_discovery_proposals()
            
            # Extract symbols sorted by weighted_score
            discovery_syms = []
            proposal_snapshot: Dict[str, Dict[str, Any]] = {}
            for symbol, prop in sorted(
                weighted_props.items(),
                key=lambda x: x[1].get("weighted_score", 0.0) if isinstance(x[1], dict) else 0.0,
                reverse=True
            ):
                if isinstance(prop, dict):
                    symbol = str(prop.get("symbol", symbol)).upper()
                    weighted_score = float(prop.get("weighted_score", 0.0))
                    proposal_snapshot[symbol] = dict(prop)
                else:
                    symbol = str(symbol).upper()
                    weighted_score = 0.0
                
                if symbol:
                    discovery_syms.append((symbol, weighted_score))
            
            # Extract just symbols
            syms = set(s[0] for s in discovery_syms)
            self._latest_proposal_snapshot = proposal_snapshot
            
            self.logger.debug(
                f"[UURE] Collected {len(syms)} weighted discovery proposals, "
                f"top scores: {discovery_syms[:3]}"
            )
            
            return syms
        
        except Exception as e:
            self.logger.error(f"[UURE] Error collecting weighted proposals: {e}")
            # Fall back to unweighted
            return await self._collect_discovery_proposals()

    @staticmethod
    def _extract_first_float(payload: Optional[Dict[str, Any]], keys: Tuple[str, ...], default: float = 0.0) -> float:
        if not isinstance(payload, dict):
            return float(default)
        for key in keys:
            try:
                val = payload.get(key)
                if val is not None:
                    return float(val)
            except Exception:
                continue
        return float(default)

    async def _quality_multiplier(self, symbol: str) -> float:
        """
        Quality multiplier to break flat-score ties using already-available execution quality signals.
        Reuses:
          - discovery proposal metadata (volume/spread hints),
          - shared_state symbol filters (spread/minNotional),
          - live price availability.
        """
        sym = str(symbol or "").replace("/", "").upper()
        if not sym:
            return 1.0

        score = 1.0
        proposal = self._latest_proposal_snapshot.get(sym, {}) or {}

        # Spread quality (prefer tighter markets).
        spread_bps_val = ss_spread_bps(self.ss, sym)
        if spread_bps_val is None:
            spread_bps_val = self._extract_first_float(
                proposal,
                ("spread_bps", "spreadBasisPoints", "spread"),
                0.0,
            )
            if 0.0 < spread_bps_val < 1.0:
                spread_bps_val *= 10000.0
        if spread_bps_val is not None and spread_bps_val > 0.0:
            max_spread = float(self._cfg("UURE_QUALITY_MAX_SPREAD_BPS", 30.0) or 30.0)
            ratio = float(spread_bps_val) / max(max_spread, 1e-6)
            if ratio <= 1.0:
                score += 0.06 * (1.0 - ratio)
            else:
                score -= min(0.25, 0.12 * (ratio - 1.0))

        # Notional executability quality (penalize symbols where default entry is below minNotional).
        min_notional_val = ss_min_notional(self.ss, sym)
        if min_notional_val is None:
            min_notional_val = self._extract_first_float(
                proposal,
                ("minNotional", "min_notional", "min_notional_usdt"),
                0.0,
            )
        if min_notional_val and min_notional_val > 0.0:
            entry_quote = float(
                self._cfg(
                    "MIN_ENTRY_QUOTE_USDT",
                    self._cfg("MIN_ENTRY_USDT", self._cfg("DEFAULT_PLANNED_QUOTE", 0.0)),
                )
                or 0.0
            )
            ratio = entry_quote / max(float(min_notional_val), 1e-6)
            if ratio < 1.0:
                score -= min(0.22, 0.18 * (1.0 - ratio))
            elif ratio >= 1.5:
                score += min(0.06, 0.03 * (ratio - 1.0))

        # Proposal liquidity quality (prefer higher-volume discovery candidates when available).
        volume_24h = self._extract_first_float(
            proposal,
            ("24h_volume", "volume_24h", "quote_volume", "quote_volume_usdt"),
            0.0,
        )
        if volume_24h > 0.0:
            min_vol_cfg = float(self._cfg("DISCOVERY_MIN_PROPOSAL_VOLUME_USDT", 0.0) or 0.0)
            ref_vol = float(self._cfg("UURE_QUALITY_REF_VOLUME_USDT", max(min_vol_cfg, 500000.0)) or max(min_vol_cfg, 500000.0))
            vol_ratio = volume_24h / max(ref_vol, 1e-6)
            if vol_ratio >= 1.0:
                score += min(0.12, 0.08 * (vol_ratio - 1.0))
            else:
                score -= min(0.12, 0.10 * (1.0 - vol_ratio))

        # Price readiness quality (small penalty if no current price is available yet).
        px = float(await self._safe_price(sym) or 0.0)
        if px <= 0.0:
            score -= 0.10
        else:
            score += 0.02

        min_mult = float(self._cfg("UURE_QUALITY_MIN_MULT", 0.65) or 0.65)
        max_mult = float(self._cfg("UURE_QUALITY_MAX_MULT", 1.35) or 1.35)
        return max(min_mult, min(max_mult, float(score)))

    async def _score_all(
        self, candidates: List[str]
    ) -> Dict[str, float]:
        """Step 2: Unified score for all candidates (dust already pre-filtered)."""
        try:
            scores = {}
            skipped = []
            for candidate in candidates:
                # FIX #2: Safe handling of mixed candidate types
                # Handle string candidates, dict candidates, and float/invalid candidates
                if isinstance(candidate, str):
                    symbol = candidate
                elif isinstance(candidate, dict):
                    # If candidate is a dict, extract symbol
                    symbol = candidate.get("symbol")
                    if not symbol:
                        self.logger.debug(f"[UURE] Skipping candidate dict without symbol: {candidate}")
                        continue
                    symbol = str(symbol).upper()
                else:
                    # Skip invalid types (float, int, None, etc.)
                    self.logger.debug(f"[UURE] Skipping non-string/non-dict candidate: {type(candidate).__name__} = {candidate}")
                    continue
                
                # Ensure symbol is string and uppercase
                symbol = str(symbol).upper()
                
                try:
                    base_score = float(self.ss.get_unified_score(symbol))
                    quality_mult = float(await self._quality_multiplier(symbol))
                    score = float(base_score * quality_mult)
                    scores[symbol] = score
                    if abs(quality_mult - 1.0) >= 0.05:
                        self.logger.debug(
                            "[UURE:Quality] %s base=%.4f mult=%.3f final=%.4f",
                            symbol,
                            base_score,
                            quality_mult,
                            score,
                        )
                except Exception as score_err:
                    self.logger.debug(f"[UURE] Failed to score {symbol}: {score_err}")
                    skipped.append(symbol)
                    continue

            if scores:
                self.logger.info(
                    f"[UURE] Scored {len(scores)} candidates. "
                    f"Mean: {sum(scores.values())/len(scores):.3f}, "
                    f"skipped: {len(skipped)}"
                )
            else:
                self.logger.warning(f"[UURE] No candidates scored (processed {len(candidates)} inputs)")
            return scores

        except Exception as e:
            self.logger.error(f"[UURE] Error scoring candidates: {e}", exc_info=True)
            return {}

    def _rank_by_score(
        self, scores: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Step 3: Sort candidates by score descending."""
        ranked = sorted(
            ((sym, score) for sym, score in scores.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    async def _apply_governor_cap(
        self, ranked: List[Tuple[str, float]]
    ) -> List[str]:
        """Step 4: Apply governor cap using SMART cap logic."""
        if not ranked:
            return []

        # Compute smart cap (based on capital, not just count)
        cap = await self._compute_smart_cap()

        # Take top-N by score
        capped = [sym for sym, _ in ranked[:cap]]

        self.logger.info(
            f"[UURE] Applied smart cap: {cap} symbols "
            f"(top by score: {capped[:3]}...)"
        )
        return capped

    async def _apply_profitability_filter(
        self, candidates: List[str]
    ) -> List[str]:
        """
        Step 4.5: Soft profitability filter for candidate admission.

        NEW STRATEGY (2024-03-07): Top-N ranking instead of hard threshold
        
        Problem with hard threshold:
          - Too aggressive: filters 30 → 1 symbols
          - No diversification
          - Fewer trading signals
        
        Solution: Keep top N candidates ranked by profitability score
        
        Algorithm:
          1. Score each candidate by: expected_move_pct - (required_move_pct × edge_penalty)
          2. Rank by score (descending)
          3. Keep top N (configurable, default: 20)
          4. Also apply regime filter if enabled (low/sideways rejection)
        
        Config:
          UURE_KEEP_TOP_PROFITABLE: How many top candidates to keep (default: 20)
          UURE_DISABLE_SIDEWAYS_REGIME_TRADING: Reject low/sideways (default: False)
        """
        try:
            if not candidates:
                return []

            round_trip_cost_pct = float(self._round_trip_cost_pct())
            disable_sideways = self._disable_sideways_in_uure()
            keep_top = int(self._cfg("UURE_KEEP_TOP_PROFITABLE", 20) or 20)

            # Score each candidate
            scored_candidates: List[Tuple[str, float]] = []
            filtered_out: List[str] = []

            for sym in candidates:
                sym_u = str(sym or "").replace("/", "").upper()
                try:
                    regime = await self._volatility_regime_1h(sym_u)
                    regime = regime or "normal"
                    
                    # Reject low/sideways if configured
                    if disable_sideways:
                        if regime in ("low", "sideways"):
                            filtered_out.append(sym_u)
                            self.logger.debug(
                                "[UURE] %s rejected (sideways regime disabled)", sym_u
                            )
                            continue

                    price = float(await self._safe_price(sym_u) or 0.0)
                    if price <= 0:
                        # No price yet: assign a neutral score so the symbol is ranked
                        # but not favoured over priced symbols. Do NOT hard-reject —
                        # prices may arrive before the next candle.
                        multiplier = float(self._ev_multiplier_for_regime(regime))
                        required_move_pct = float(round_trip_cost_pct) * float(multiplier)
                        edge_pct = -required_move_pct  # neutral penalty
                        scored_candidates.append((sym_u, edge_pct))
                        self.logger.debug("[UURE] %s has no price yet; using neutral score %.6f", sym_u, edge_pct)
                        continue

                    # Calculate expected move and required threshold
                    expected_move_pct = float(await self._expected_move_pct_1h(sym_u, price) or 0.0)
                    multiplier = float(self._ev_multiplier_for_regime(regime))
                    required_move_pct = float(round_trip_cost_pct) * float(multiplier)

                    # Profitability score = expected_move - required_move (can be negative)
                    # This allows exploration of symbols with slightly negative edge
                    edge_pct = expected_move_pct - required_move_pct

                    scored_candidates.append((sym_u, edge_pct))
                    
                    self.logger.debug(
                        "[UURE] %s scored: expected=%.6f required=%.6f edge=%.6f regime=%s",
                        sym_u,
                        expected_move_pct,
                        required_move_pct,
                        edge_pct,
                        regime,
                    )
                except Exception as e:
                    # Safe default: keep symbol on unexpected errors
                    self.logger.debug("[UURE] %s error during scoring: %s", sym, str(e))
                    scored_candidates.append((str(sym or "").replace("/", "").upper(), 0.0))
            
            # Sort by edge (descending) and keep top N
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            profitable = [sym for sym, _ in scored_candidates[:max(1, keep_top)]]
            
            # If we have fewer candidates than keep_top, return all
            if len(scored_candidates) <= keep_top:
                profitable = [sym for sym, _ in scored_candidates]
            
            # Sanity check: if somehow we have no profitable candidates
            if not profitable:
                current = self.ss.get_accepted_symbol_list()
                self.logger.warning(
                    f"[UURE] No candidates passed basic checks. "
                    f"Keeping current universe: {current}"
                )
                return current
            
            self.logger.info(
                f"[UURE] Profitability filter (keep_top={keep_top}): {len(candidates)} → {len(profitable)} symbols "
                f"({len(filtered_out)} regime-filtered, top edge=+{scored_candidates[0][1]:.6f}%)"
            )
            return profitable
        
        except Exception as e:
            self.logger.error(f"[UURE] Error in profitability filter: {e}")
            return candidates

    async def _apply_relative_replacement_rule(
        self, new_candidates: List[str], current_universe: List[str]
    ) -> List[str]:
        """
        Step 4.6: Relative Replacement Rule with Adaptive Thresholds
        
        Controls whether incoming candidates can rotate OUT existing symbols.
        Supports both:
          A) Relative mode: incoming > weakest × factor (conservative, prevents downside rotation)
          B) Absolute minimum mode: incoming > min_edge_pct (spot-friendly, simpler threshold)
        
        Rule (default):
          incoming_edge > weakest_active_edge × ROTATION_SUPERIORITY_FACTOR
          if weakest_active_edge <= 0, allow free rotation
        
        Spot trading mode (UURE_MINIMUM_EDGE_PCT set):
          incoming_edge > UURE_MINIMUM_EDGE_PCT
          Simpler, more permissive threshold ideal for 0.05%-0.15% edge strategies
        
        Configs:
          ROTATION_SUPERIORITY_FACTOR: Relative multiplier (default 1.25, try 1.1-1.2 for spot)
          UURE_MINIMUM_EDGE_PCT: Absolute edge floor as decimal (e.g., 0.001 = 0.1%)
          UURE_SPOT_MODE: If True, prefer MINIMUM_EDGE_PCT over superiority factor
          UURE_RELATIVE_REPLACE_DISABLED: If True, disable rule entirely (discovery-first mode)
        """
        try:
            # ✅ FIX #7B: Discovery-first mode - accept all candidates without relative gating
            disable_relative = bool(self._cfg("UURE_RELATIVE_REPLACE_DISABLED", True))
            if disable_relative:
                self.logger.info(
                    "[UURE] Relative rule DISABLED (discovery-first mode) — accepting all %d candidates",
                    len(new_candidates)
                )
                return new_candidates
            
            # Get config options
            superiority_factor = float(
                self._cfg("ROTATION_SUPERIORITY_FACTOR", 1.25)
            )
            minimum_edge_pct = self._cfg("UURE_MINIMUM_EDGE_PCT", None)
            spot_mode = bool(self._cfg("UURE_SPOT_MODE", False))
            
            # No current universe = fresh start, allow all
            if not current_universe:
                self.logger.info(
                    "[UURE] No current universe, relative rule skipped (fresh rotation)"
                )
                return new_candidates

            cap_target = max(1, len(new_candidates))
            round_trip_cost_pct = float(self._round_trip_cost_pct())
            disable_sideways = self._disable_sideways_in_uure()

            # Compute weakest incumbent net-edge (expected_move - required_move).
            incumbent_edges: Dict[str, float] = {}
            for sym in current_universe:
                sym_u = str(sym or "").replace("/", "").upper()
                try:
                    regime = await self._volatility_regime_1h(sym_u)
                    regime = regime or "normal"
                    if disable_sideways:
                        if regime in ("low", "sideways"):
                            # Sideways incumbents shouldn't be used as the baseline for rotation.
                            continue
                    price = float(await self._safe_price(sym_u) or 0.0)
                    exp_move = float(await self._expected_move_pct_1h(sym_u, price) or 0.0) if price > 0 else 0.0
                    required_move_pct = float(round_trip_cost_pct) * float(self._ev_multiplier_for_regime(regime))
                    incumbent_edges[sym_u] = float(exp_move - required_move_pct)
                except Exception:
                    continue
            
            weakest_edge = min(incumbent_edges.values()) if incumbent_edges else 0.0
            
            # Determine threshold strategy
            use_minimum_mode = (
                spot_mode and minimum_edge_pct is not None
            ) or (
                minimum_edge_pct is not None and 
                self._cfg("UURE_PREFER_MINIMUM_EDGE", False)
            )
            
            if use_minimum_mode:
                # Absolute minimum edge mode (spot-friendly)
                try:
                    threshold = max(0.0, float(minimum_edge_pct))
                except Exception:
                    threshold = 0.001  # Fallback: 0.1%
                
                self.logger.info(
                    "[UURE] Relative rule (MINIMUM_EDGE mode): threshold=%.6f (%.4f%%) weakest_active=%.6f",
                    float(threshold),
                    float(threshold) * 100.0,
                    float(weakest_edge),
                )
            else:
                # Relative superiority mode (conservative)
                free_rotation_mode = float(weakest_edge) <= 0.0
                threshold = (
                    0.0 if free_rotation_mode else float(weakest_edge) * float(superiority_factor)
                )
                
                self.logger.info(
                    "[UURE] Relative rule (SUPERIORITY mode): weakest_active_net_edge=%.6f threshold=%.6f (factor=%.2f free_mode=%s)",
                    float(weakest_edge),
                    float(threshold),
                    float(superiority_factor),
                    str(free_rotation_mode),
                )

            # Accept new candidates only if they clear the threshold.
            current_set = {str(s or "").replace("/", "").upper() for s in current_universe}
            accepted: List[str] = []
            rejected: List[str] = []
            for sym in new_candidates:
                sym_u = str(sym or "").replace("/", "").upper()
                if sym_u in current_set:
                    # Incumbents stay by default
                    accepted.append(sym_u)
                    continue
                try:
                    regime = await self._volatility_regime_1h(sym_u)
                    regime = regime or "normal"
                    if disable_sideways:
                        if regime in ("low", "sideways"):
                            rejected.append(sym_u)
                            continue
                    
                    price = float(await self._safe_price(sym_u) or 0.0)
                    exp_move = float(await self._expected_move_pct_1h(sym_u, price) or 0.0) if price > 0 else 0.0
                    required_move_pct = float(round_trip_cost_pct) * float(self._ev_multiplier_for_regime(regime))
                    net_edge = float(exp_move - required_move_pct)
                    
                    if net_edge > float(threshold):
                        accepted.append(sym_u)
                    else:
                        rejected.append(sym_u)
                except Exception:
                    # On error: keep the candidate (avoid blocking rotation loop).
                    accepted.append(sym_u)

            # If rotation is fully blocked, keep current universe unchanged.
            if not accepted:
                self.logger.warning(
                    "[UURE] Relative rule: no candidates accepted; keeping current universe."
                )
                return [str(s or "").replace("/", "").upper() for s in current_universe if s]

            # Fill back rejected slots with incumbents (avoid shrinking purely due to relative rule).
            final: List[str] = list(accepted)
            if len(final) < cap_target:
                fill = [s for s in current_universe if str(s or "").replace("/", "").upper() not in set(final)]
                fill.sort(key=lambda s: float(self.ss.get_unified_score(str(s))), reverse=True)
                for s in fill:
                    if len(final) >= cap_target:
                        break
                    final.append(str(s or "").replace("/", "").upper())

            if rejected:
                self.logger.info(
                    "[UURE] Relative rule blocked %d incoming candidates (kept incumbents instead).",
                    len(rejected),
                )

            # Ensure deterministic cap safety.
            return final[:cap_target]
        
        except Exception as e:
            self.logger.error(f"[UURE] Error in relative replacement rule: {e}")
            return new_candidates

    async def _compute_smart_cap(self) -> int:
        """
        Compute dynamic cap based on:
          • Governor calculation
          • Available capital (total equity)
          • Min entry size
          • Max symbol limit
        
        Formula:
          equity = await shared_state.get_total_equity()  # Professional standard NAV
          deployable = equity * exposure
          dynamic_cap = floor(deployable / min_entry_quote)
          cap = min(dynamic_cap, MAX_SYMBOL_LIMIT)
          cap = max(cap, 1)  # Always at least 1
        """
        try:
            # Get governor's base cap
            governor_cap = None
            if self.governor is not None and hasattr(self.governor, "compute_symbol_cap"):
                governor_cap = await self.governor.compute_symbol_cap()
            if governor_cap is None:
                governor_cap = self.max_symbol_limit

            # Get total equity using professional standard calculation
            # This ensures NAV is calculated from shared_state, not raw wallet positions
            equity = await self.ss.get_total_equity()
            if equity is None or equity <= 0:
                self.logger.warning(
                    "[UURE] Total equity unavailable, using governor cap"
                )
                return governor_cap

            # Compute dynamic cap
            deployable = equity * self.max_exposure
            dynamic_cap = int(deployable / self.min_entry_quote)

            # Apply limits
            final_cap = min(dynamic_cap, self.max_symbol_limit)
            final_cap = max(final_cap, 1)

            # NOTE: Governor cap is informational but not limiting for smart cap
            # The dynamic formula already respects capital constraints via exposure + min_entry
            # Applying governor_cap here would double-constrain and prevent universe growth
            # Keep for reference but don't apply as hard ceiling
            if governor_cap is not None:
                self.logger.debug(
                    f"[UURE] Governor cap available: {governor_cap} (not applied to smart cap)"
                )

            self.logger.info(
                f"[UURE] Smart cap: equity={equity:.2f}, exposure={self.max_exposure}, "
                f"dynamic={dynamic_cap}, final={final_cap}"
            )
            return final_cap

        except Exception as e:
            self.logger.error(f"[UURE] Error computing smart cap: {e}")
            return self.max_symbol_limit  # Safe fallback — don't collapse universe to 2

    async def _identify_rotation(
        self, new_universe: List[str]
    ) -> Dict[str, List[str]]:
        """Step 5: Identify rotation (symbols to add/remove/keep)."""
        try:
            current = set(self.ss.get_accepted_symbol_list())
            wanted = set(new_universe)

            added = list(wanted - current)
            removed = list(current - wanted)
            kept = list(wanted & current)

            self.logger.info(
                f"[UURE] Rotation: +{added} -{removed} ={kept}"
            )

            return {
                "added": added,
                "removed": removed,
                "kept": kept,
            }

        except Exception as e:
            self.logger.error(f"[UURE] Error identifying rotation: {e}")
            return {"added": [], "removed": [], "kept": []}

    async def _hard_replace_universe(self, new_universe: List[str]) -> List[str]:
        """Step 6: Merge new universe with existing accepted symbols (union, not replace).
        
        CRITICAL FIX: Use union instead of hard replace.
        
        Why this matters:
          - Hard replace kills symbols that discovery just added
          - Union preserves discovery additions while enabling rotation
          - Discovery flow: SymbolScreener proposes → UURE ranks → union preserves all
          
        Architecture:
          new_universe = top-ranked candidates for this cycle
          current = symbols already accepted (from previous cycles + discovery)
          result = current ∪ new_universe (keeps all, rotation by score)
        """
        try:
            # Get current accepted symbols
            current_accepted = set()
            try:
                current = await self._maybe_await(self.ss.get_accepted_symbols())
                if isinstance(current, dict):
                    current_accepted = set(current.keys())
                elif isinstance(current, (list, set)):
                    current_accepted = set(current)
            except Exception:
                pass

            # Union: keep current symbols + add new ranked symbols
            # This prevents killing discovery proposals
            merged_symbols = current_accepted.union(set(new_universe))
            
            # Build metadata for merged universe
            symbols_with_meta = {}
            for sym in merged_symbols:
                symbols_with_meta[sym] = {
                    "symbol": sym,
                    "source": "UniverseRotationEngine",
                    "rotation_cycle": self._get_cycle_timestamp(),
                }

            # Update symbols using union (NOT hard replace with allow_shrink=True)
            await self.ss.set_accepted_symbols(
                symbols_with_meta,
                allow_shrink=False,  # ✅ FIXED: Use union instead of destructive replace
                source="UniverseRotationEngine"
            )

            self.logger.info(
                f"[UURE] Merged universe: {len(current_accepted)} existing + "
                f"{len(new_universe)} new ranked = {len(merged_symbols)} total"
            )
            return sorted(str(s).replace("/", "").upper() for s in merged_symbols if s)

        except Exception as e:
            self.logger.error(f"[UURE] Error merging universe: {e}")
            return [str(s).replace("/", "").upper() for s in (new_universe or []) if s]

    async def _trigger_liquidation(self, symbols_to_remove: List[str]) -> None:
        """Step 7: Trigger liquidation of removed symbols."""
        try:
            positions = await self._maybe_await(self.ss.get_positions_snapshot())

            for sym in symbols_to_remove:
                if sym not in positions:
                    continue  # No position to liquidate

                pos = positions[sym]
                current_qty = float(
                    pos.get("quantity", 0.0)
                    or pos.get("qty", 0.0)
                    or pos.get("current_qty", 0.0)
                    or 0.0
                )

                if current_qty <= 0:
                    continue  # Nothing to sell

                # Create liquidation intent
                intent = {
                    "symbol": sym,
                    "action": "SELL",
                    "confidence": 1.0,
                    "planned_qty": current_qty,
                    "agent": "UniverseRotationEngine",
                    "tag": "liquidation",
                    "execution_tag": "rotation_liquidation",
                    "reason": f"Removed from universe (rotation cycle)",
                    "_is_liquidation": True,
                    "_is_rotation": True,
                    "_forced_exit": True,
                    "ttl_sec": 300,
                }

                # Submit to MetaController
                if self.mc and hasattr(self.mc, "receive_intents"):
                    await self.mc.receive_intents([intent])
                    self.logger.info(
                        f"[UURE] Liquidation intent submitted: {sym} ({current_qty} qty)"
                    )
                else:
                    self.logger.warning(
                        f"[UURE] No MetaController available for liquidation"
                    )

        except Exception as e:
            self.logger.error(f"[UURE] Error triggering liquidation: {e}")

    def _get_cycle_timestamp(self) -> str:
        """Get rotation cycle timestamp."""
        import datetime
        return datetime.datetime.now(datetime.timezone.utc).isoformat()


# Convenience function for integration
async def run_universe_rotation(
    shared_state: Any,
    capital_governor: Any,
    config: Optional[Any] = None,
    execution_manager: Optional[Any] = None,
    meta_controller: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run a single universe rotation cycle.
    
    Called periodically (e.g., every 5-10 minutes) to:
    1. Reassess symbol universe
    2. Identify rotation opportunities
    3. Liquidate weak symbols
    4. Update accepted symbols
    
    Returns rotation result dictionary.
    """
    engine = UniverseRotationEngine(
        shared_state,
        capital_governor,
        config,
        execution_manager,
        meta_controller,
    )
    return await engine.compute_and_apply_universe()
