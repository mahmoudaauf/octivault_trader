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
        self.max_exposure = float(self._cfg("MAX_EXPOSURE_RATIO", 0.8))

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

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Config getter that supports dict or attribute configs."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

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
        """
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
        rg = str(regime or "").strip().lower()
        override = self._cfg("UURE_SOFT_EV_MULTIPLIER", None)
        if override is not None:
            try:
                return max(0.5, float(override))
            except Exception:
                pass
        
        # Use dynamic profile if available
        profile = self._get_dynamic_profile()
        if profile is not None:
            if rg == "normal":
                return max(0.5, float(profile.get("ev_mult_normal", 1.3)))
            elif rg == "bull":
                return max(0.5, float(profile.get("ev_mult_bull", 1.8)))
            else:
                return max(0.5, float(profile.get("ev_mult_other", 2.0)))
        
        # Fallback to legacy config
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

            # Step 5: Identify rotation
            rotation = await self._identify_rotation(final_universe)
            self.logger.info(
                f"[UURE] Rotation: +{len(rotation['added'])} -{len(rotation['removed'])} ={len(rotation['kept'])}"
            )

            # Step 6: Hard replace universe
            await self._hard_replace_universe(final_universe)
            self.logger.info(
                f"[UURE] Universe hard-replaced: {len(final_universe)} symbols"
            )

            # Step 7: Trigger liquidation of removed symbols
            if rotation["removed"]:
                await self._trigger_liquidation(rotation["removed"])

            result["new_universe"] = final_universe
            result["score_info"] = {sym: scored[sym] for sym in final_universe}
            result["rotation"] = rotation
            result["execution"] = rotation["removed"]  # Liquidated symbols

            return result

        except Exception as e:
            self.logger.error(f"[UURE] Error in compute_and_apply_universe: {e}")
            result["error"] = str(e)
            return result

    async def _collect_candidates(self) -> List[str]:
        """Step 1: Collect all candidate symbols from discovery & current positions."""
        try:
            # Get symbols from accepted set
            accepted = await self._maybe_await(self.ss.get_accepted_symbols())
            accepted_syms = set(accepted.keys())

            # Get symbols from positions
            positions = await self._maybe_await(self.ss.get_positions_snapshot())
            position_syms = set(positions.keys())

            # Union of both (all candidates)
            all_syms = accepted_syms | position_syms

            self.logger.debug(
                f"[UURE] Candidates: {len(accepted_syms)} accepted, "
                f"{len(position_syms)} positions, {len(all_syms)} total"
            )
            return list(all_syms)

        except Exception as e:
            self.logger.error(f"[UURE] Error collecting candidates: {e}")
            return []

    async def _score_all(
        self, candidates: List[str]
    ) -> Dict[str, float]:
        """Step 2: Unified score for all candidates."""
        try:
            scores = {}
            for sym in candidates:
                score = self.ss.get_unified_score(sym)
                scores[sym] = score

            self.logger.debug(
                f"[UURE] Scored {len(scores)} candidates. "
                f"Mean: {sum(scores.values())/len(scores):.3f}"
            )
            return scores

        except Exception as e:
            self.logger.error(f"[UURE] Error scoring candidates: {e}")
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

        Admission rule:
          expected_move >= round_trip_cost × regime_multiplier

        Default regime multipliers:
          - normal: 1.3
          - bull:   1.8
          - other:  2.0
        """
        try:
            if not candidates:
                return []

            round_trip_cost_pct = float(self._round_trip_cost_pct())
            disable_sideways = self._disable_sideways_in_uure()

            profitable: List[str] = []
            filtered_out: List[str] = []

            for sym in candidates:
                sym_u = str(sym or "").replace("/", "").upper()
                try:
                    regime = await self._volatility_regime_1h(sym_u)
                    regime = regime or "normal"
                    if disable_sideways:
                        if regime in ("low", "sideways"):
                            filtered_out.append(sym_u)
                            continue

                    price = float(await self._safe_price(sym_u) or 0.0)
                    if price <= 0:
                        filtered_out.append(sym_u)
                        continue

                    expected_move_pct = float(await self._expected_move_pct_1h(sym_u, price) or 0.0)
                    multiplier = float(self._ev_multiplier_for_regime(regime))
                    required_move_pct = float(round_trip_cost_pct) * float(multiplier)
                    if expected_move_pct >= required_move_pct:
                        profitable.append(sym_u)
                    else:
                        filtered_out.append(sym_u)
                        self.logger.debug(
                            "[UURE] %s filtered: expected=%.6f required=%.6f regime=%s mult=%.2f",
                            sym_u,
                            expected_move_pct,
                            required_move_pct,
                            regime,
                            multiplier,
                        )
                except Exception:
                    # Safe default: keep symbol on unexpected errors (avoid freezing rotation loop).
                    profitable.append(sym_u)
            
            # If no symbols pass filter, keep current universe (rotation blocked)
            if not profitable:
                current = self.ss.get_accepted_symbol_list()
                self.logger.warning(
                    f"[UURE] No candidates met profitability threshold. "
                    f"Keeping current universe: {current}"
                )
                return current
            
            self.logger.info(
                f"[UURE] Profitability filter: {len(candidates)} → {len(profitable)} symbols "
                f"({len(filtered_out)} filtered)"
            )
            return profitable
        
        except Exception as e:
            self.logger.error(f"[UURE] Error in profitability filter: {e}")
            return candidates

    async def _apply_relative_replacement_rule(
        self, new_candidates: List[str], current_universe: List[str]
    ) -> List[str]:
        """
        Step 4.6: Relative Replacement Rule
        
        Only allows rotation OUT of a symbol if incoming candidates
        have superior edge vs the weakest active symbol.
        
        Rule: incoming_edge > weakest_active_edge × ROTATION_SUPERIORITY_FACTOR
              if weakest_active_edge <= 0, allow free rotation
        
        Default ROTATION_SUPERIORITY_FACTOR = 1.25 (25% edge premium required)
        
        This prevents rotating out of proven winners into marginal candidates.
        """
        try:
            # Get config
            superiority_factor = float(
                self._cfg("ROTATION_SUPERIORITY_FACTOR", 1.25)
            )
            
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
            free_rotation_mode = float(weakest_edge) <= 0.0
            required_edge = (
                0.0 if free_rotation_mode else float(weakest_edge) * float(superiority_factor)
            )

            self.logger.info(
                "[UURE] Relative replacement rule: weakest_active_net_edge=%.6f required_net_edge=%.6f (factor=%.2f free_mode=%s)",
                float(weakest_edge),
                float(required_edge),
                float(superiority_factor),
                str(free_rotation_mode),
            )

            # Accept new candidates only if they clear the superiority threshold.
            current_set = {str(s or "").replace("/", "").upper() for s in current_universe}
            accepted: List[str] = []
            rejected: List[str] = []
            for sym in new_candidates:
                sym_u = str(sym or "").replace("/", "").upper()
                if sym_u in current_set:
                    accepted.append(sym_u)
                    continue
                try:
                    regime = await self._volatility_regime_1h(sym_u)
                    regime = regime or "normal"
                    if disable_sideways:
                        if regime in ("low", "sideways"):
                            rejected.append(sym_u)
                            continue
                    if free_rotation_mode:
                        accepted.append(sym_u)
                        continue
                    price = float(await self._safe_price(sym_u) or 0.0)
                    exp_move = float(await self._expected_move_pct_1h(sym_u, price) or 0.0) if price > 0 else 0.0
                    required_move_pct = float(round_trip_cost_pct) * float(self._ev_multiplier_for_regime(regime))
                    net_edge = float(exp_move - required_move_pct)
                    if net_edge > float(required_edge):
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
          • Available capital
          • Min entry size
          • Max symbol limit
        
        Formula:
          dynamic_cap = floor((NAV * exposure) / min_entry_quote)
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

            # Get capital metrics
            nav = self.ss.get_nav_quote()
            if nav is None or nav <= 0:
                self.logger.warning(
                    "[UURE] NAV unavailable, using governor cap"
                )
                return governor_cap

            # Compute dynamic cap
            deployable = nav * self.max_exposure
            dynamic_cap = int(deployable / self.min_entry_quote)

            # Apply limits
            final_cap = min(dynamic_cap, self.max_symbol_limit)
            final_cap = max(final_cap, 1)

            # But don't exceed governor cap
            final_cap = min(final_cap, governor_cap)

            self.logger.info(
                f"[UURE] Smart cap: NAV={nav:.2f}, exposure={self.max_exposure}, "
                f"dynamic={dynamic_cap}, final={final_cap}"
            )
            return final_cap

        except Exception as e:
            self.logger.error(f"[UURE] Error computing smart cap: {e}")
            return 2  # Default to 2 if error

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

    async def _hard_replace_universe(self, new_universe: List[str]) -> None:
        """Step 6: Hard replace accepted symbols with new universe."""
        try:
            # Build metadata for new universe
            symbols_with_meta = {}
            for sym in new_universe:
                symbols_with_meta[sym] = {
                    "symbol": sym,
                    "source": "UniverseRotationEngine",
                    "rotation_cycle": self._get_cycle_timestamp(),
                }

            # Hard replace (allow_shrink=True since we control the replacement)
            await self.ss.set_accepted_symbols(
                symbols_with_meta,
                allow_shrink=True,
                source="UniverseRotationEngine"
            )

            self.logger.info(
                f"[UURE] Hard-replaced universe: {len(new_universe)} symbols"
            )

        except Exception as e:
            self.logger.error(f"[UURE] Error hard-replacing universe: {e}")

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
