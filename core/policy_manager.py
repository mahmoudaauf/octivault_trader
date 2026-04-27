from .core_utils import _safe_await
"""
PolicyManager subsystem extracted from MetaController.
Handles policy evaluation, decision logic, and orchestration of the main evaluation loop.
"""
from typing import Any, Dict, Optional, List, Tuple
import time
import os

class PolicyManager:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        # Policy cache: stores the most recent policy decisions for quick lookup
        self.policy_cache = {}
        # Event emission placeholder: can be replaced with a real event bus or callback system
        self._event_handlers = []
        
        # Initialize policy configurations
        self._economic_guard = {
            "enabled": getattr(config, "ECONOMIC_GUARD_ENABLED", True),  # Enabled by default (config can disable)
            "min_edge_bps": float(getattr(config, "MIN_EXPECTED_EDGE_BPS", 10.0)),
            "fallback_edge_bps": float(getattr(config, "FALLBACK_EDGE_BPS", 80.0)),  # INCREASED: 80 bps fallback (2.7x fees)
            # Use actual system trading costs from config.py defaults
            "fee_bps": float(os.getenv("CR_FEE_BPS", "10")),  # 0.1% trading fees (system default)
            "slippage_bps": float(os.getenv("CR_PRICE_SLIPPAGE_BPS", "15")),  # 0.15% slippage (system default)  
            "buffer_bps": float(getattr(config, "SAFETY_BUFFER_BPS", 5.0)),  # 0.05% safety buffer
            "min_positive_edge_bps": float(getattr(config, "MIN_POSITIVE_EDGE_BPS", 35.0)),  # INCREASED: 35 bps minimum (covers 30 bps costs + 5 bps profit)
            "fee_expectancy_multiplier": float(os.getenv("FEE_EXPECTANCY_MULTIPLIER", "2.5")), # INCREASED: Profit >= 2.5x fees (was 1.6x)
            "fee_floor_usdt": float(os.getenv("FEE_FLOOR_USDT", "0.15")), # 0.15 USDT fee floor
        }
        
        # Initialize Phase 2 guard configuration for dust liquidation
        self._phase2_guard = {
            "dust_ratio_trigger": float(getattr(config, 'PHASE2_DUST_RATIO_TRIGGER', 0.8)),  # 80% dust ratio trigger
            "active_since": None,  # Timestamp when phase2 was activated
            "activation_age_sec": float(getattr(config, 'PHASE2_ACTIVATION_AGE_SEC', 300.0)),  # 5 minutes grace period
            "position_grace_sec": float(getattr(config, 'PHASE2_POSITION_GRACE_SEC', 600.0)),  # 10 minutes position grace
        }
        
        self.logger.info(f"[PolicyManager:Init] Economic guard initialized: {self._economic_guard}")
        self.logger.info(f"[PolicyManager:Init] Phase 2 guard initialized: {self._phase2_guard}")

        # Mode x Policy Activation Matrix (Weights: 1.0=Full, 0.5=Limited, 0.0=Disabled)
        self.policy_activation_matrix = {
            "SAFE":       {"velocity": 0.0, "drawdown": 1.0, "volatility": 1.0, "capital": 0.0, "signal": 0.0},
            "PROTECTIVE": {"velocity": 0.0, "drawdown": 1.0, "volatility": 1.0, "capital": 0.0, "signal": 0.5},
            "NORMAL":     {"velocity": 0.5, "drawdown": 1.0, "volatility": 1.0, "capital": 0.5, "signal": 1.0},
            "AGGRESSIVE": {"velocity": 1.0, "drawdown": 0.5, "volatility": 0.5, "capital": 1.0, "signal": 1.0},
            "RECOVERY":   {"velocity": 0.0, "drawdown": 1.0, "volatility": 0.0, "capital": 0.0, "signal": 0.0},
            "BOOTSTRAP":  {"velocity": 0.5, "drawdown": 1.0, "volatility": 1.0, "capital": 1.0, "signal": 1.0},
            "PAUSED":     {"velocity": 0.0, "drawdown": 0.0, "volatility": 0.0, "capital": 0.0, "signal": 0.0},
            "SIGNAL_ONLY": {"velocity": 1.0, "drawdown": 0.0, "volatility": 0.0, "capital": 0.0, "signal": 1.0},
        }

    def calculate_policy_nudges(self, metrics: Dict[str, Any], mode_envelope: Dict[str, Any], mode: str = "NORMAL") -> Dict[str, float]:
        """
        Calculate policy nudges (Soft Controllers) based on system metrics and Mode x Policy Matrix.
        
        Outputs:
        - confidence_nudge (additive)
        - cooldown_nudge (additive seconds)
        - trade_size_multiplier (multiplier)
        - max_positions_nudge (additive)
        """
        # Get weights for current mode
        weights = self.policy_activation_matrix.get(mode, self.policy_activation_matrix["NORMAL"])
        
        final_nudges = {
            "confidence_nudge": 0.0,
            "cooldown_nudge": 0.0,
            "trade_size_multiplier": 1.0,
            "max_positions_nudge": 0,
        }

        # Helper to merge weighted nudges
        def merge(target_dict, source_dict, weight):
            if weight <= 0.01: return # Optimization for disabled policies
            
            for k, v in source_dict.items():
                if k == "trade_size_multiplier":
                    # Multiplier merging: 1.0 + (raw - 1.0) * weight
                    # e.g. raw=0.8, w=0.5 -> 1.0 + (-0.2)*0.5 -> 0.9
                    delta = v - 1.0
                    target_dict[k] *= (1.0 + (delta * weight)) 
                    # Note: consecutive multipliers multiply. 
                    # If we simply added delta, we might get negative multipliers.
                    # Ideally we should accumulate log changes but this linear approx is fine for small nudges.
                    # Better: Apply (1 + delta * weight) to the running product.
                elif k == "max_positions_nudge":
                    # Integer additive
                    target_dict[k] += int(v * weight)
                else:
                    # Float additive
                    target_dict[k] += (v * weight)

        # 1. VelocityPolicy
        n_vel = self._new_nudge_dict()
        self._apply_velocity_policy(metrics, n_vel)
        merge(final_nudges, n_vel, weights["velocity"])

        # 2. DrawdownPolicy
        n_dd = self._new_nudge_dict()
        self._apply_drawdown_policy(metrics, n_dd)
        merge(final_nudges, n_dd, weights["drawdown"])

        # 3. VolatilityPolicy
        n_vol = self._new_nudge_dict()
        self._apply_volatility_policy(metrics, n_vol)
        merge(final_nudges, n_vol, weights["volatility"])

        # 4. CapitalUtilizationPolicy
        n_cap = self._new_nudge_dict()
        self._apply_capital_utilization_policy(metrics, mode_envelope, n_cap)
        merge(final_nudges, n_cap, weights["capital"])

        # 5. SignalQualityPolicy
        n_sig = self._new_nudge_dict()
        self._apply_signal_quality_policy(metrics, n_sig)
        merge(final_nudges, n_sig, weights["signal"])
        
        return final_nudges

    def calculate_min_viable_trade_size(self, nav: float, taker_bps: float) -> float:
        """
        Policy Rule: No trade is allowed unless it's significantly larger than fees.
        min_trade_notional = max(exchange_min_notional, fee_floor_usdt / round_trip_fee_rate)
        """
        fee_floor = self._economic_guard.get("fee_floor_usdt", 0.15)
        rt_fee_rate = (taker_bps * 2.0) / 10000.0
        
        # MVT based on fee floor (e.g., 0.15 / 0.002 = 75 USDT)
        mvt_fees = fee_floor / rt_fee_rate if rt_fee_rate > 0 else 10.0
        
        # Scale MVT slightly by NAV if capital is high (> $1000)
        # For small capital ($250), we stick to the 75-100 range.
        if nav > 1000:
             mvt_fees = max(mvt_fees, nav * 0.05) # 5% minimum
             
        return max(10.0, mvt_fees)

    def check_entry_profitability(self, planned_quote: float, expected_alpha: float, taker_bps: float) -> Tuple[bool, str]:
        """
        Rule: Expected gross profit >= 3 * total fees.
        """
        if planned_quote <= 0: return False, "Zero quote"
        
        rt_fee_pct = (taker_bps * 2.0) / 10000.0
        total_fees = planned_quote * rt_fee_pct
        
        # expected_alpha is usually something like 0.008 (0.8%)
        expected_profit = planned_quote * expected_alpha
        
        multiplier = self._economic_guard.get("fee_expectancy_multiplier", 1.6)
        
        if expected_profit < (total_fees * multiplier):
             return False, f"Low expectancy: profit {expected_profit:.2f} < {multiplier}x fees {total_fees:.2f}"
             
        return True, "Profitability OK"
        

    def _new_nudge_dict(self):
        return {
            "confidence_nudge": 0.0,
            "cooldown_nudge": 0.0,
            "trade_size_multiplier": 1.0,
            "max_positions_nudge": 0,
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            num = float(value)
            if num != num:
                return float(default)
            return float(num)
        except Exception:
            return float(default)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    def _normalize_market_regime(self, volatility: str, mode: str) -> str:
        vol_u = str(volatility or "").strip().upper()
        mode_u = str(mode or "").strip().upper()
        if vol_u in {"EXTREME", "HIGH"}:
            return "volatile"
        if vol_u in {"LOW"}:
            return "sideways"
        if mode_u in {"PROTECTIVE", "RECOVERY", "SAFE", "PAUSED"}:
            return "volatile"
        if mode_u in {"BOOTSTRAP", "NORMAL", "AGGRESSIVE"}:
            return "trend"
        return "unknown"

    def _compute_conf_policy_bias(
        self,
        metrics: Dict[str, Any],
        mode: str,
        nudges: Dict[str, float],
    ) -> Dict[str, Any]:
        run_rate = self._safe_float(metrics.get("run_rate", 0.0), 0.0)
        target_rr = max(1e-6, self._safe_float(metrics.get("target_run_rate", 20.0), 20.0))
        rr_ratio = run_rate / target_rr
        dd_pct = max(0.0, self._safe_float(metrics.get("drawdown_pct", 0.0), 0.0))
        idle_sec = max(0.0, self._safe_float(metrics.get("idle_time_sec", 0.0), 0.0))
        win_rate = self._clamp(self._safe_float(metrics.get("win_rate", 0.5), 0.5), 0.0, 1.0)
        health_ok = bool(metrics.get("health_ok", True))
        repeated_failures = bool(metrics.get("repeated_failures", False))
        volatility = str(metrics.get("volatility", "NORMAL") or "NORMAL")

        bias = 0.0
        # Risk-on tightening branch.
        if dd_pct >= 4.0:
            bias += 0.08
        elif dd_pct >= 2.0:
            bias += 0.04

        if not health_ok:
            bias += 0.05
        if repeated_failures:
            bias += 0.04

        vol_u = volatility.upper()
        if vol_u == "EXTREME":
            bias += 0.06
        elif vol_u == "HIGH":
            bias += 0.03

        if win_rate <= 0.38:
            bias += 0.02
        elif win_rate >= 0.66:
            bias -= 0.01

        # Throughput recovery branch (only when risk is stable).
        if rr_ratio < 0.45 and idle_sec > 600 and dd_pct < 1.5 and health_ok and not repeated_failures:
            bias -= 0.02
        if rr_ratio < 0.25 and idle_sec > 1800 and dd_pct < 1.0 and health_ok and vol_u not in {"HIGH", "EXTREME"}:
            bias -= 0.02

        # Keep policy bias directionally aligned with nudges.
        conf_nudge = self._safe_float((nudges or {}).get("confidence_nudge", 0.0), 0.0)
        bias += self._clamp(-0.35 * conf_nudge, -0.02, 0.02)

        raw_bias = self._clamp(bias, -0.10, 0.15)
        regime_key = self._normalize_market_regime(volatility, mode)
        regime_multiplier = {
            "volatile": 1.15,
            "trend": 0.85,
            "sideways": 0.90,
            "unknown": 1.00,
        }.get(regime_key, 1.0)
        regime_bias = self._clamp(raw_bias * regime_multiplier, -0.12, 0.18)

        return {
            "raw_bias": float(raw_bias),
            "regime_key": regime_key,
            "raw_regime_bias": float(regime_bias),
            "diagnostics": {
                "run_rate_ratio": float(rr_ratio),
                "drawdown_pct": float(dd_pct),
                "idle_time_sec": float(idle_sec),
                "win_rate": float(win_rate),
                "volatility": vol_u,
                "health_ok": bool(health_ok),
                "repeated_failures": bool(repeated_failures),
            },
        }

    async def _apply_dynamic_tradeability_policy(
        self,
        meta_controller,
        metrics: Dict[str, Any],
        mode: str,
        nudges: Dict[str, float],
    ) -> None:
        shared_state = getattr(meta_controller, "shared_state", None)
        if shared_state is None:
            return

        try:
            dyn_cfg = getattr(shared_state, "dynamic_config", {}) or {}
            if not isinstance(dyn_cfg, dict):
                dyn_cfg = {}
        except Exception:
            dyn_cfg = {}

        computed = self._compute_conf_policy_bias(metrics, mode=mode, nudges=nudges)
        raw_bias = self._safe_float(computed.get("raw_bias", 0.0), 0.0)
        regime_key = str(computed.get("regime_key", "unknown") or "unknown")
        raw_regime_bias = self._safe_float(computed.get("raw_regime_bias", raw_bias), raw_bias)

        alpha = self._clamp(self._safe_float(getattr(self.config, "POLICY_CONF_BIAS_EMA_ALPHA", 0.25), 0.25), 0.05, 1.0)
        regime_alpha = self._clamp(
            self._safe_float(getattr(self.config, "POLICY_CONF_BIAS_REGIME_EMA_ALPHA", 0.35), 0.35),
            0.05,
            1.0,
        )

        prev_global = self._safe_float(dyn_cfg.get("ML_DYNAMIC_REQUIRED_CONF_POLICY_BIAS", 0.0), 0.0)
        smoothed_global = prev_global + (alpha * (raw_bias - prev_global))
        smoothed_global = self._clamp(smoothed_global, -0.15, 0.20)

        prev_map = dyn_cfg.get("ML_DYNAMIC_REQUIRED_CONF_POLICY_BIAS_BY_REGIME", {}) or {}
        if not isinstance(prev_map, dict):
            prev_map = {}
        next_map = dict(prev_map)
        prev_regime = self._safe_float(next_map.get(regime_key, smoothed_global), smoothed_global)
        smoothed_regime = prev_regime + (regime_alpha * (raw_regime_bias - prev_regime))
        smoothed_regime = self._clamp(smoothed_regime, -0.18, 0.22)
        next_map[regime_key] = float(smoothed_regime)

        payload = {
            "ML_DYNAMIC_REQUIRED_CONF_POLICY_BIAS": float(smoothed_global),
            "ML_DYNAMIC_REQUIRED_CONF_POLICY_BIAS_BY_REGIME": next_map,
            "ML_DYNAMIC_REQUIRED_CONF_POLICY_BIAS_TS": float(time.time()),
            "ML_DYNAMIC_REQUIRED_CONF_POLICY_MODE": str(mode or "UNKNOWN").upper(),
            "ML_DYNAMIC_REQUIRED_CONF_POLICY_REGIME": regime_key,
        }
        if raw_bias != smoothed_global or raw_regime_bias != smoothed_regime:
            payload["ML_DYNAMIC_REQUIRED_CONF_POLICY_LAST_RAW_BIAS"] = float(raw_bias)
            payload["ML_DYNAMIC_REQUIRED_CONF_POLICY_LAST_RAW_REGIME_BIAS"] = float(raw_regime_bias)

        update_dynamic = getattr(shared_state, "update_dynamic_config", None)
        if callable(update_dynamic):
            await _safe_await(update_dynamic(payload))
        else:
            if not hasattr(shared_state, "dynamic_config") or getattr(shared_state, "dynamic_config", None) is None:
                shared_state.dynamic_config = {}
            shared_state.dynamic_config.update(payload)

        if abs(smoothed_global) >= 0.005 or abs(smoothed_regime) >= 0.005:
            meta_controller.logger.info(
                "[PolicyManager:DynamicConf] mode=%s regime=%s bias=%.4f regime_bias=%.4f rr=%.2f dd=%.2f idle=%ds win=%.2f",
                str(mode).upper(),
                regime_key,
                smoothed_global,
                smoothed_regime,
                float(computed.get("diagnostics", {}).get("run_rate_ratio", 0.0)),
                float(computed.get("diagnostics", {}).get("drawdown_pct", 0.0)),
                int(float(computed.get("diagnostics", {}).get("idle_time_sec", 0.0))),
                float(computed.get("diagnostics", {}).get("win_rate", 0.5)),
            )

    def _apply_velocity_policy(self, metrics: Dict[str, Any], nudges: Dict[str, Any]):
        """
        VelocityPolicy: Run-rate < Target -> Loosen constraints to encourage trading.
        """
        run_rate = float(metrics.get("run_rate", 0.0))
        target_rr = float(metrics.get("target_run_rate", 20.0))
        idle_time = float(metrics.get("idle_time_sec", 0.0))
        
        # If run-rate is low (< 50% target), loosen confidence and cooldown
        if run_rate < (target_rr * 0.5):
            # Loosen confidence (e.g. -0.05)
            nudges["confidence_nudge"] -= 0.05
            # Reduce cooldown (e.g. -30s)
            nudges["cooldown_nudge"] -= 30.0
            
        # If system is very idle (> 1 hour no trades), incentivize action
        if idle_time > 3600:
            nudges["confidence_nudge"] -= 0.02
            
    def _apply_drawdown_policy(self, metrics: Dict[str, Any], nudges: Dict[str, Any]):
        """
        DrawdownPolicy: Drawdown > Threshold -> Tighten sizers and concurrency.
        """
        dd_pct = float(metrics.get("drawdown_pct", 0.0))
        
        # If drawdown is creeping up (e.g. > 2%), reduce trade size
        if dd_pct > 2.0:
            nudges["trade_size_multiplier"] *= 0.8  # 20% reduction
            
        # If drawdown is significant (> 4%), reduce concurrency (if possible)
        if dd_pct > 4.0:
            nudges["max_positions_nudge"] -= 1

    def _apply_volatility_policy(self, metrics: Dict[str, Any], nudges: Dict[str, Any]):
        """
        VolatilityPolicy: High Volatility -> Reduce sizing.
        """
        volatility = str(metrics.get("volatility", "NORMAL")).upper()
        
        if volatility == "HIGH":
            nudges["trade_size_multiplier"] *= 0.75  # 25% reduction
            nudges["confidence_nudge"] += 0.05       # Require higher confidence
        elif volatility == "EXTREME":
            nudges["trade_size_multiplier"] *= 0.5   # 50% reduction
            nudges["confidence_nudge"] += 0.10

    def _apply_capital_utilization_policy(self, metrics: Dict[str, Any], envelope: Dict[str, Any], nudges: Dict[str, Any]):
        """
        CapitalUtilizationPolicy: High idle capital -> Allow more positions (up to mode limit).
        """
        # Note: This policy encourages using available slots if we have cash, 
        # but MetaController enforces the hard mode limit.
        # We can "nudge" if the mode allows a range, but typically mode sets a hard cap.
        # Here we merely ensure we don't artificially restrict below the mode limit if we have cash.
        
        # If we have lots of cash (> 80% free) and low exposure, ensure we aren't restricting count
        free_pct = float(metrics.get("capital_free_pct", 0.0))
        if free_pct > 0.80:
             # If we have massive idle capital, slightly loosen entry confidence 
             # to encourage capital deployment (within mode limits)
             nudges["confidence_nudge"] -= 0.02
             pass

    def _apply_signal_quality_policy(self, metrics: Dict[str, Any], nudges: Dict[str, Any]):
        """
        SignalQualityPolicy: High win rate -> Slight loosen.
        """
        win_rate = float(metrics.get("win_rate", 0.5))
        
        if win_rate > 0.70:
            # High win rate, slightly lower confidence requirement
            nudges["confidence_nudge"] -= 0.03

    def _is_budget_required(self, action: str) -> bool:
        """P9: Canonical budget requirement check. SELL/HOLD reduce risk and don't require budget."""
        return str(action).upper() == "BUY"

    async def _maybe_build_rotation_escape_decisions(self, meta_controller) -> list:
        """
        Delegate rotation escape decision building to MetaController.
        This method is too complex and MetaController-specific to extract.
        """
        return await meta_controller._maybe_build_rotation_escape_decisions()

    async def _build_decisions(self, meta_controller, accepted_symbols_set: set) -> list:
        """
        Delegate decision building to MetaController.
        This method is too complex and MetaController-specific to extract.
        """
        return await meta_controller._build_decisions(accepted_symbols_set)

    async def _execute_decision(self, meta_controller, symbol: str, side: str, signal: dict, accepted_symbols_set: set):
        # Example execution logic: call execution_logic for actual trade
        try:
            result = await meta_controller.execution_logic._execute_decision(symbol, side, signal, accepted_symbols_set)
            meta_controller.logger.info(f"[PolicyManager] Executed {side} for {symbol}: {result}")
            return result
        except Exception as e:
            meta_controller.logger.error(f"[PolicyManager] Error executing {side} for {symbol}: {e}", exc_info=True)
            return {"ok": False, "status": "error", "reason": str(e)}
    async def evaluate_policies(self, meta_controller, loop_id):
        # 1. Evaluate Soft Policies (Nudges)
        try:
            # Gather mode metrics
            metrics = await meta_controller._gather_mode_metrics()
            
            # Augment with Policy-specific metrics
            # Idle time — guard against last_exec=0 producing a Unix-epoch-sized
            # idle value (~1.77 billion seconds) that permanently triggers PROTECTIVE mode.
            last_exec = 0.0
            if hasattr(meta_controller.state_manager, "_last_execution_ts"):
                ts_map = meta_controller.state_manager._last_execution_ts
                if isinstance(ts_map, dict) and ts_map:
                    # Only consider values that look like real Unix timestamps (post-year-2001)
                    _valid_ts = [v for v in ts_map.values()
                                 if isinstance(v, (int, float)) and float(v) > 1_000_000_000]
                    last_exec = max(_valid_ts) if _valid_ts else 0.0
                elif isinstance(ts_map, (int, float)) and float(ts_map) > 1_000_000_000:
                    last_exec = float(ts_map)
            _pm_now = time.time()
            _pm_bot_start = float(getattr(meta_controller, "_start_time", 0) or 0)
            if _pm_bot_start < 1_000_000_000:
                _pm_bot_start = _pm_now  # unknown start → treat bot as just started
            if last_exec < _pm_bot_start:
                # No trade yet this session → idle since bot start
                metrics["idle_time_sec"] = _pm_now - _pm_bot_start
            else:
                metrics["idle_time_sec"] = _pm_now - last_exec
            
            # Win rate (calculated from metrics)
            win_rate = 0.5
            try:
                from core.metrics import get_metrics
                pnl_hist = get_metrics().get_metric("realized_pnl")
                if pnl_hist and len(pnl_hist) > 0:
                     wins = sum(1 for item in pnl_hist if item.get("value", 0) > 0)
                     win_rate = wins / len(pnl_hist)
            except Exception:
                 pass
            metrics["win_rate"] = win_rate
            
            # Capital Free %
            if hasattr(meta_controller.shared_state, "get_portfolio_status"):
                 p_status = await _safe_await(meta_controller.shared_state.get_portfolio_status())
                 total = float(p_status.get("total_capital", 1.0))
                 free = float(p_status.get("free_usdt", 0.0))
                 metrics["capital_free_pct"] = free / total if total > 0 else 0.0

            # Calculate Nudges
            env = meta_controller.mode_manager.get_envelope()
            mode = meta_controller.mode_manager.get_mode()
            nudges = self.calculate_policy_nudges(metrics, env, mode=mode)
            
            # Apply to MetaController
            if hasattr(meta_controller, "set_active_policy_nudges"):
                meta_controller.set_active_policy_nudges(nudges)

            # Feed dynamic confidence control into shared dynamic config.
            await self._apply_dynamic_tradeability_policy(meta_controller, metrics, mode=mode, nudges=nudges)
                
            # Log significant nudges
            if any(abs(v) > 0.001 if isinstance(v, (int, float)) else False for v in nudges.values()):
                 meta_controller.logger.info(f"[PolicyManager] Active Nudges: {nudges}")

        except Exception as e:
            meta_controller.logger.warning(f"[PolicyManager] Policy evaluation failed: {e}")

    def cache_policy_decision(self, symbol, decision):
        """Cache the most recent policy decision for a symbol."""
        self.policy_cache[symbol] = decision
        self._emit_event('policy_decision_cached', {'symbol': symbol, 'decision': decision})

    def get_cached_policy(self, symbol):
        """Get the most recent cached policy decision for a symbol, if any."""
        return self.policy_cache.get(symbol)

    def register_event_handler(self, handler):
        """Register an event handler callback for policy events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event_type, payload):
        """Emit an event to all registered handlers (placeholder for event bus)."""
        for handler in self._event_handlers:
            try:
                handler(event_type, payload)
            except Exception as e:
                self.logger.debug(f"PolicyManager event handler error: {e}")

    def _build_policy_context(self, symbol: str, side: str, policies: Optional[List[str]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build policy context for MetaController authority."""
        # Base policy context for MetaController authority
        ctx = {
            "authority": "metacontroller",
            "policy_authority": "metacontroller",
            "symbol": symbol.upper().strip(),
            "side": side.upper(),
            "validated_at": time.time(),
            # Unified economic thresholds from config
            "min_notional": float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0)),
            "min_entry_quote": float(getattr(self.config, "MIN_ENTRY_QUOTE_USDT", 80.0)),
            "scout_min_notional": float(getattr(self.config, "SCOUT_MIN_NOTIONAL", 6.0)),
            "base_min_notional": float(getattr(self.config, "BASE_MIN_NOTIONAL", 5.0)),
            "dust_threshold": float(getattr(self.config, "DUST_EXIT_THRESHOLD", 0.60)),
            "min_economic_trade": float(getattr(self.config, "MIN_ECONOMIC_TRADE_USDT", 40.0)),
        }
        if policies:
            ctx["validated_policies"] = sorted({p for p in policies if p})
        if extra:
            ctx.update(extra)
        return ctx

    def _extract_expected_edge_bps(self, signal: Optional[Dict[str, Any]]) -> Optional[float]:
        """Extract expected edge in basis points from signal."""
        if not signal:
            return None
        candidate_keys = [
            "edge_bps",
            "expected_edge_bps",
            "expectedEdgeBps",
            "edge",
            "expected_edge",
        ]
        for key in candidate_keys:
            if key in signal and signal[key] is not None:
                try:
                    return float(signal[key])
                except (TypeError, ValueError):
                    continue
        # ROI expressed as fraction (e.g., 0.03 => 3%)
        if signal.get("expected_roi") is not None:
            try:
                return float(signal["expected_roi"]) * 10000.0
            except (TypeError, ValueError):
                pass
        if signal.get("expected_roi_pct") is not None:
            try:
                return float(signal["expected_roi_pct"]) * 100.0
            except (TypeError, ValueError):
                pass
        if signal.get("expected_return") is not None:
            try:
                return float(signal["expected_return"]) * 10000.0
            except (TypeError, ValueError):
                pass
        # MLForecaster emits _expected_move_pct (percentage move expected)
        # Convert: 0.05 (5% move) => 500 basis points
        if signal.get("_expected_move_pct") is not None:
            try:
                move_pct = float(signal["_expected_move_pct"])
                return move_pct * 10000.0  # Convert percentage to basis points
            except (TypeError, ValueError):
                pass
        return None

    def _check_economic_profitability(self, symbol: str, signal: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check economic profitability of a signal."""
        cfg = self._economic_guard
        if not cfg.get("enabled", True):
            return True, "disabled", {"net_edge_bps": None, "edge_source": "disabled"}

        edge_bps = self._extract_expected_edge_bps(signal)
        source = "signal" if edge_bps is not None else "fallback"
        if edge_bps is None:
            edge_bps = cfg["fallback_edge_bps"]

        total_cost_bps = cfg["fee_bps"] + cfg["slippage_bps"] + cfg["buffer_bps"]
        net_edge_bps = edge_bps - total_cost_bps
        ok = net_edge_bps >= cfg["min_positive_edge_bps"]
        reason = "economic_ok" if ok else f"net_edge_bps<{cfg['min_positive_edge_bps']}"
        metrics = {
            "edge_bps": edge_bps,
            "edge_source": source,
            "total_cost_bps": total_cost_bps,
            "net_edge_bps": net_edge_bps,
            "min_positive_edge_bps": cfg["min_positive_edge_bps"],
        }
        if not ok:
            metrics["blocked_reason"] = reason
        return ok, reason, metrics

    def _update_phase2_guard(self, dust_ratio: float) -> Tuple[bool, float]:
        """Update phase 2 guard state based on dust ratio."""
        cfg = self._phase2_guard
        now = time.time()
        if dust_ratio >= cfg["dust_ratio_trigger"]:
            if not cfg["active_since"]:
                cfg["active_since"] = now
            phase2_age = now - cfg["active_since"]
            allow = phase2_age >= cfg["activation_age_sec"]
            return allow, phase2_age
        cfg["active_since"] = None
        return False, 0.0
    
    def get_fee_bps(self, fee_type: str = "taker") -> float:
        """Get trading fee in bps from authoritative config."""
        return self._economic_guard.get("fee_bps", 10.0)

    async def dust_accumulation_guard(self, symbol: str, position_qty: float, 
                                      position_mark_price: float, 
                                      symbol_min_notional: Optional[float] = None,
                                      exchange_client=None, shared_state=None) -> bool:
        """
        HARD INVARIANT — Canonical Dust Accumulation Guard
        
        A dust position MUST NEVER generate a TradeIntent until its executable 
        notional ≥ minNotional.
        
        Formally:
            IF position.qty > 0 AND position.notional < symbol.min_notional
            THEN
              • DO NOT emit TradeIntent
              • DO NOT call ExecutionManager
              • DO NOT retry
              • DO NOT escalate
              • DO NOT bypass checks
              • ONLY record + accumulate
        
        This invariant cannot be bypassed by mode, bootstrap, or agent.
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            position_qty: Quantity held
            position_mark_price: Current mark price
            symbol_min_notional: Exchange minNotional (fetched if not provided)
            exchange_client: Exchange client for fetching symbol info
            shared_state: Shared state for recording accumulation
        
        Returns:
            True: ALLOW - position.notional >= minNotional, safe to emit TradeIntent
            False: BLOCK_ACCUMULATE - position.notional < minNotional, must accumulate
        """
        try:
            symbol = (symbol or "").upper()
            if not symbol:
                return False
            
            # Step 1: Position validation
            if position_qty <= 0:
                # Nothing to do - no position to trade
                return True  # ALLOW (no position to block)
            
            # Step 2: Get minNotional threshold
            min_notional = symbol_min_notional
            if min_notional is None:
                try:
                    if exchange_client and hasattr(exchange_client, "get_symbol_info"):
                        info = await exchange_client.get_symbol_info(symbol)
                        if info and isinstance(info, dict):
                            min_notional = float(info.get("minNotional", 10.0))
                except Exception as e:
                    self.logger.debug(f"[INVARIANT:DustGuard] Failed to get minNotional for {symbol}: {e}")
                
                # Fallback to config default
                if min_notional is None:
                    min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))
            
            # Step 3: Calculate notional value
            notional = float(position_qty) * float(position_mark_price)
            
            # Step 4: Check invariant
            if notional < min_notional:
                # DUST DETECTED: Below minNotional
                # → DO NOT emit TradeIntent
                # → Record accumulating position
                
                self.logger.warning(
                    "[INVARIANT:DustGuard:BLOCK_ACCUMULATE] %s position below minNotional. "
                    "qty=%.8f price=%.2f notional=%.2f < min_notional=%.2f. "
                    "Recording as accumulating, NO TradeIntent will be emitted.",
                    symbol, position_qty, position_mark_price, notional, min_notional
                )
                
                # Record accumulating position in shared state
                if shared_state and hasattr(shared_state, "mark_accumulating"):
                    try:
                        await shared_state.mark_accumulating(symbol, position_qty, notional)
                    except Exception as e:
                        self.logger.debug(f"[INVARIANT:DustGuard] Failed to record accumulating: {e}")

                # Dead-loop breaker:
                # Ultra-small residuals (below permanent dust threshold) cannot
                # economically recover through repeated liquidation attempts.
                # Retire them once so they stop cycling in SELL suppression loops.
                try:
                    auto_retire_raw = getattr(
                        self.config, "DUST_GUARD_AUTO_RETIRE_BELOW_PERMANENT", True
                    )
                    auto_retire = (
                        auto_retire_raw.strip().lower() in {"1", "true", "yes", "on"}
                        if isinstance(auto_retire_raw, str)
                        else bool(auto_retire_raw)
                    )
                    permanent_floor = float(
                        getattr(self.config, "PERMANENT_DUST_USDT_THRESHOLD", 1.0) or 1.0
                    )
                    if (
                        auto_retire
                        and shared_state is not None
                        and 0.0 < notional <= permanent_floor
                        and hasattr(shared_state, "mark_as_permanent_dust")
                    ):
                        shared_state.mark_as_permanent_dust(symbol)
                        if hasattr(shared_state, "dust_unhealable"):
                            shared_state.dust_unhealable = getattr(shared_state, "dust_unhealable", {})
                            shared_state.dust_unhealable[str(symbol).upper()] = "UNHEALABLE_LT_MIN_NOTIONAL"
                        self.logger.info(
                            "[INVARIANT:DustGuard:RETIRE] %s notional=%.4f <= permanent_floor=%.4f "
                            "→ marked as PERMANENT_DUST to stop repeat retries.",
                            symbol,
                            notional,
                            permanent_floor,
                        )
                except Exception:
                    self.logger.debug("[INVARIANT:DustGuard] Auto-retire path failed for %s", symbol, exc_info=True)
                
                return False  # ❌ BLOCK: Cannot emit TradeIntent
            
            # Step 5: Position is above minNotional - safe to emit TradeIntent
            self.logger.debug(
                "[INVARIANT:DustGuard:ALLOW] %s position is tradable. "
                "qty=%.8f price=%.2f notional=%.2f >= min_notional=%.2f",
                symbol, position_qty, position_mark_price, notional, min_notional
            )
            
            return True  # ✅ ALLOW: Safe to emit TradeIntent
            
        except Exception as e:
            self.logger.warning(
                "[INVARIANT:DustGuard] Error evaluating dust invariant for %s: %s. "
                "Failing safely (BLOCK).",
                symbol, e
            )
            # Fail-safe: block if uncertain (prefer safety over execution)
            return False
