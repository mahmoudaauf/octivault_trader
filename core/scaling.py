"""
ScalingManager - Position Scaling & Compounding Logic
======================================================

Extracted from MetaController to handle:
1. Scale-in opportunity detection (compounding winners)
2. Account-size-based quote scaling
3. Position scaling validation
4. Adaptive risk per trade based on account size

This module is responsible for all scaling-related decisions
and calculations, keeping MetaController focused on orchestration.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING, Tuple
import logging
from decimal import Decimal
import time
import asyncio
import inspect
from core.adaptive_capital_engine import AdaptiveCapitalEngine

if TYPE_CHECKING:
    from .shared_state import SharedState
    from .execution_manager import ExecutionManager
    from .config import Config
    from .mode_manager import ModeManager


class ScalingManager:
    """
    Manages position scaling, compounding, and adaptive sizing logic.
    
    Responsibilities:
    - Detect scale-in opportunities for winning positions
    - Scale trade quotes based on account size
    - Validate scaling eligibility
    - Generate compounding signals
    """
    
    def __init__(self, shared_state: "SharedState", execution_manager: "ExecutionManager", config: "Config", logger: logging.Logger, mode_manager: Optional["ModeManager"] = None):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.config = config
        self.logger = logger
        self.mode_manager = mode_manager
        
        # Configuration
        # Phase‑9 wealth engine tuning - lower friction to allow safe compounding but
        # prevent rapid startup flip. Defaults: 3 minutes and 0.2% pnl.
        self._scale_in_min_age_min = float(getattr(config, 'SCALE_IN_MIN_AGE_MIN', 3.0))
        # NOTE: this value is in percent units (e.g. 0.2 == 0.2%) because pnl_pct is computed *100
        self._scale_in_min_pnl_pct = float(getattr(config, 'SCALE_IN_MIN_PNL_PCT', 0.2))
        self._default_planned_quote = float(getattr(config, 'DEFAULT_PLANNED_QUOTE', 20.0))
        
        # Account size thresholds for adaptive scaling
        self._large_account_threshold = 1000.0  # $1000+
        self._medium_account_threshold = 300.0  # $300+
        
        # Risk percentages by account size
        self._large_account_risk_pct = 0.01  # 1% per trade
        self._medium_account_risk_pct = 0.02  # 2% per trade
        self._small_account_risk_pct = 0.03  # 3% per trade

        # Compounding growth curve (Phase 1 -> Phase 4)
        self._compounding_growth_enabled = bool(
            getattr(
                config,
                "COMPOUNDING_ENABLED",
                getattr(config, "COMPOUNDING_GROWTH_CURVE_ENABLED", True),
            )
        )
        configured_growth_phases = list(
            getattr(config, "COMPOUNDING_GROWTH_PHASES", []) or []
        )
        if not configured_growth_phases:
            configured_growth_phases = [
                {
                    "name": "PHASE_1_SEED",
                    "min_equity": 0.0,
                    "max_equity": 249.99,
                    "min_momentum": -9999.0,
                    "quote_mult": 0.90,
                    "risk_mult": 0.90,
                    "min_quote": 40.0,
                    "max_quote": 100.0,
                    "tp_asym_mult": 1.00,
                    "sl_asym_mult": 1.00,
                    "rr_bonus": 0.00,
                },
                {
                    "name": "PHASE_2_TRACTION",
                    "min_equity": 250.0,
                    "max_equity": 399.99,
                    "min_momentum": 0.0,
                    "quote_mult": 1.00,
                    "risk_mult": 1.00,
                    "min_quote": 55.0,
                    "max_quote": 130.0,
                    "tp_asym_mult": 1.06,
                    "sl_asym_mult": 0.97,
                    "rr_bonus": 0.08,
                },
                {
                    "name": "PHASE_3_ACCELERATE",
                    "min_equity": 400.0,
                    "max_equity": 699.99,
                    "min_momentum": 0.0,
                    "quote_mult": 1.12,
                    "risk_mult": 1.08,
                    "min_quote": 70.0,
                    "max_quote": 180.0,
                    "tp_asym_mult": 1.14,
                    "sl_asym_mult": 0.94,
                    "rr_bonus": 0.16,
                },
                {
                    "name": "PHASE_4_SNOWBALL",
                    "min_equity": 700.0,
                    "max_equity": None,
                    "min_momentum": 0.0,
                    "quote_mult": 1.25,
                    "risk_mult": 1.16,
                    "min_quote": 90.0,
                    "max_quote": 250.0,
                    "tp_asym_mult": 1.22,
                    "sl_asym_mult": 0.90,
                    "rr_bonus": 0.28,
                },
            ]
        self._compounding_growth_phases = self._normalize_growth_phases(configured_growth_phases)
        configured_growth_thresholds = list(
            getattr(config, "GROWTH_PHASE_THRESHOLDS", []) or []
        )
        if not configured_growth_thresholds:
            configured_growth_thresholds = [
                {"name": "PHASE_1_SEED", "min_ratio": 0.0, "max_ratio": 1.25},
                {"name": "PHASE_2_TRACTION", "min_ratio": 1.25, "max_ratio": 1.75},
                {"name": "PHASE_3_ACCELERATE", "min_ratio": 1.75, "max_ratio": 2.50},
                {"name": "PHASE_4_SNOWBALL", "min_ratio": 2.50, "max_ratio": None},
            ]
        self._growth_phase_thresholds = self._normalize_growth_thresholds(configured_growth_thresholds)
        self._phase_size_multipliers = dict(
            getattr(config, "PHASE_SIZE_MULTIPLIERS", {}) or {}
        )
        self._phase_max_trade_cap = dict(
            getattr(config, "PHASE_MAX_TRADE_CAP", {}) or {}
        )
        self._compounding_drawdown_guard_pct = float(
            getattr(config, "COMPOUNDING_MAX_DRAWDOWN_PCT", 2.5) or 2.5
        )
        self._compounding_min_positive_streak = int(
            getattr(config, "COMPOUNDING_MIN_POSITIVE_STREAK", 0) or 0
        )

        # Dynamic position sizing formula coefficients
        self._dynamic_position_sizing_enabled = bool(
            getattr(config, "DYNAMIC_POSITION_SIZING_ENABLED", True)
        )
        self._dyn_conf_floor_mult = float(
            getattr(config, "DYNAMIC_SIZE_CONF_FLOOR_MULT", 0.85) or 0.85
        )
        self._dyn_conf_ceil_mult = float(
            getattr(config, "DYNAMIC_SIZE_CONF_CEIL_MULT", 1.25) or 1.25
        )
        self._dyn_vol_floor_mult = float(
            getattr(config, "DYNAMIC_SIZE_VOL_FLOOR_MULT", 0.70) or 0.70
        )
        self._dyn_vol_ceil_mult = float(
            getattr(config, "DYNAMIC_SIZE_VOL_CEIL_MULT", 1.35) or 1.35
        )
        self._dyn_momentum_mult = float(
            getattr(config, "DYNAMIC_SIZE_MOMENTUM_MULT", 0.45) or 0.45
        )
        self._dyn_blend_weight = float(
            getattr(config, "DYNAMIC_SIZE_BLEND_WEIGHT", 0.65) or 0.65
        )
        self._dyn_upside_cap_mult = float(
            getattr(config, "DYNAMIC_SIZE_UPSIDE_CAP_MULT", 2.25) or 2.25
        )
        self._dyn_momentum_lookback = int(
            getattr(config, "DYNAMIC_SIZE_MOMENTUM_LOOKBACK_TRADES", 20) or 20
        )
        self._adaptive_engine = AdaptiveCapitalEngine(config, self.logger)
        self._adaptive_min_trade_quote_by_symbol: Dict[str, float] = {}
        self._adaptive_min_trade_quote_logged: Dict[str, float] = {}

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback."""
        return getattr(self.config, key, default)

    def _get_realized_equity(self) -> float:
        """Compute realized-only equity for tiering."""
        try:
            base = float(getattr(self.config, "BASE_CAPITAL", 0.0) or 0.0)
            realized = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
            return base + realized
        except Exception:
            return 0.0

    def _get_realized_pnl_last_n(self, n: int = 20) -> float:
        """Sum realized PnL deltas for the last N trades."""
        try:
            history = list(getattr(self.shared_state, "trade_history", []) or [])
            if not history:
                return 0.0
            recent = history[-int(n):]
            return float(sum(float(t.get("realized_delta", 0.0) or 0.0) for t in recent))
        except Exception:
            return 0.0

    def _normalize_growth_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, phase in enumerate(phases):
            if not isinstance(phase, dict):
                continue
            p = dict(phase)
            p.setdefault("name", f"PHASE_{idx + 1}")
            p["min_equity"] = float(p.get("min_equity", 0.0) or 0.0)
            p["max_equity"] = None if p.get("max_equity", None) is None else float(p.get("max_equity"))
            p["min_momentum"] = float(p.get("min_momentum", -9999.0) or -9999.0)
            p["quote_mult"] = float(p.get("quote_mult", 1.0) or 1.0)
            p["risk_mult"] = float(p.get("risk_mult", 1.0) or 1.0)
            p["min_quote"] = float(p.get("min_quote", 0.0) or 0.0)
            p["max_quote"] = float(p.get("max_quote", 0.0) or 0.0)
            p["tp_asym_mult"] = float(p.get("tp_asym_mult", 1.0) or 1.0)
            p["sl_asym_mult"] = float(p.get("sl_asym_mult", 1.0) or 1.0)
            p["rr_bonus"] = float(p.get("rr_bonus", 0.0) or 0.0)
            out.append(p)
        out.sort(key=lambda x: float(x.get("min_equity", 0.0) or 0.0))
        return out

    def _normalize_growth_thresholds(self, thresholds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, row in enumerate(thresholds):
            if not isinstance(row, dict):
                continue
            entry = dict(row)
            entry["name"] = str(entry.get("name", f"PHASE_{idx + 1}") or f"PHASE_{idx + 1}")
            entry["min_ratio"] = float(entry.get("min_ratio", 0.0) or 0.0)
            max_ratio = entry.get("max_ratio", None)
            entry["max_ratio"] = None if max_ratio is None else float(max_ratio)
            out.append(entry)
        out.sort(key=lambda x: float(x.get("min_ratio", 0.0) or 0.0))
        return out

    def _phase_index_by_name(self, phase_name: str) -> int:
        name = str(phase_name or "").upper()
        for idx, th in enumerate(self._growth_phase_thresholds):
            if str(th.get("name", "")).upper() == name:
                return idx
        return -1

    def _phase_name_by_index(self, idx: int) -> str:
        if 0 <= idx < len(self._growth_phase_thresholds):
            return str(self._growth_phase_thresholds[idx].get("name", "PHASE_1_SEED"))
        if self._growth_phase_thresholds:
            return str(self._growth_phase_thresholds[0].get("name", "PHASE_1_SEED"))
        return "PHASE_1_SEED"

    def _phase_from_name(self, phase_name: str) -> Dict[str, Any]:
        p_name = str(phase_name or "PHASE_1_SEED")
        phase = next(
            (
                p
                for p in self._compounding_growth_phases
                if str(p.get("name", "")).upper() == p_name.upper()
            ),
            None,
        )
        if phase is None:
            phase = {"name": p_name, "quote_mult": 1.0, "risk_mult": 1.0, "min_quote": 0.0, "max_quote": 0.0}
        phase_cfg = dict(phase)
        phase_cfg["name"] = p_name
        # Config map overrides are authoritative for sizing/caps.
        size_mult = self._phase_size_multipliers.get(p_name)
        if size_mult is None:
            size_mult = self._phase_size_multipliers.get(p_name.upper())
        if size_mult is not None:
            try:
                phase_cfg["quote_mult"] = float(size_mult)
                phase_cfg["risk_mult"] = float(size_mult)
            except Exception:
                pass
        cap = self._phase_max_trade_cap.get(p_name)
        if cap is None:
            cap = self._phase_max_trade_cap.get(p_name.upper())
        if cap is not None:
            try:
                phase_cfg["max_quote"] = float(cap)
            except Exception:
                pass
        return phase_cfg

    def _pick_compounding_phase_by_ratio(self, growth_ratio: float) -> Dict[str, Any]:
        if not self._growth_phase_thresholds:
            return self._phase_from_name("PHASE_1_SEED")

        selected_name = str(self._growth_phase_thresholds[0].get("name", "PHASE_1_SEED"))
        for threshold in self._growth_phase_thresholds:
            min_ratio = float(threshold.get("min_ratio", 0.0) or 0.0)
            max_ratio = threshold.get("max_ratio", None)
            max_val = None if max_ratio is None else float(max_ratio)
            if growth_ratio < min_ratio:
                continue
            if max_val is not None and growth_ratio >= max_val:
                continue
            selected_name = str(threshold.get("name", selected_name))
        return self._phase_from_name(selected_name)

    async def _get_total_equity(self) -> float:
        try:
            total_equity = float(getattr(self.shared_state, "total_equity", 0.0) or 0.0)
            if total_equity > 0:
                return total_equity
        except Exception:
            pass
        try:
            from core.core_utils import _safe_await
            if hasattr(self.shared_state, "get_nav_quote"):
                return float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
        except Exception:
            pass
        return float(self._get_realized_equity() or 0.0)

    def _get_bootstrap_equity(self, equity_now: float) -> float:
        base = 0.0
        try:
            base = float(getattr(self.shared_state, "bootstrap_equity", 0.0) or 0.0)
        except Exception:
            base = 0.0
        if base <= 0:
            try:
                dyn = getattr(self.shared_state, "dynamic_config", {}) or {}
                base = float(dyn.get("compounding_equity_base", 0.0) or 0.0)
            except Exception:
                base = 0.0
        if base <= 0:
            base = float(getattr(self.config, "BASE_CAPITAL", 0.0) or 0.0)
        if base <= 0 and equity_now > 0:
            base = float(equity_now)
        if base > 0:
            try:
                self.shared_state.bootstrap_equity = float(base)
            except Exception:
                pass
        return float(base)

    def _positive_realized_streak(self, n: int = 20) -> int:
        try:
            history = list(getattr(self.shared_state, "trade_history", []) or [])
            if not history:
                return 0
            streak = 0
            for trade in reversed(history[-int(n):]):
                delta = float(trade.get("realized_delta", 0.0) or 0.0)
                if delta > 0:
                    streak += 1
                else:
                    break
            return int(streak)
        except Exception:
            return 0

    def _sync_compounding_phase_state(
        self,
        phase: Optional[Dict[str, Any]],
        equity_base: float,
        equity_now: float,
        growth_ratio: float,
        momentum: float,
        positive_streak: int,
        drawdown_pct: float,
        active: bool,
    ) -> None:
        try:
            state = getattr(self.shared_state, "dynamic_config", None)
            if state is None:
                self.shared_state.dynamic_config = {}
                state = self.shared_state.dynamic_config
            previous_phase = str(
                state.get("compounding_phase")
                or state.get("COMPOUNDING_PHASE")
                or "PHASE_1_SEED"
            )
            current_phase = str((phase or {}).get("name", "PHASE_1_SEED"))
            state["COMPOUNDING_GROWTH_ACTIVE"] = bool(active)
            state["compounding_phase"] = current_phase
            state["COMPOUNDING_PHASE"] = current_phase
            state["COMPOUNDING_PHASE_RATIO"] = float(growth_ratio)
            state["COMPOUNDING_PHASE_QUOTE_MULT"] = float((phase or {}).get("quote_mult", 1.0) or 1.0)
            state["COMPOUNDING_PHASE_RISK_MULT"] = float((phase or {}).get("risk_mult", 1.0) or 1.0)
            state["COMPOUNDING_PHASE_TP_ASYM_MULT"] = float((phase or {}).get("tp_asym_mult", 1.0) or 1.0)
            state["COMPOUNDING_PHASE_SL_ASYM_MULT"] = float((phase or {}).get("sl_asym_mult", 1.0) or 1.0)
            state["COMPOUNDING_PHASE_RR_BONUS"] = float((phase or {}).get("rr_bonus", 0.0) or 0.0)
            state["compounding_equity_base"] = float(equity_base)
            state["COMPOUNDING_EQUITY_NOW"] = float(equity_now)
            state["COMPOUNDING_REALIZED_EQUITY"] = float(equity_now)
            state["COMPOUNDING_REALIZED_MOMENTUM"] = float(momentum)
            state["COMPOUNDING_POSITIVE_STREAK"] = int(positive_streak)
            state["COMPOUNDING_DRAWDOWN_PCT"] = float(drawdown_pct)
            state["COMPOUNDING_LAST_UPDATE_TS"] = float(time.time())
            self.shared_state.dynamic_config = state

            if previous_phase != current_phase:
                payload = {
                    "old_phase": previous_phase,
                    "new_phase": current_phase,
                    "equity_base": float(equity_base),
                    "equity_now": float(equity_now),
                    "growth_ratio": float(growth_ratio),
                    "drawdown_pct": float(drawdown_pct),
                    "ts": float(time.time()),
                }
                emitter = getattr(self.shared_state, "emit", None)
                if not callable(emitter):
                    emitter = getattr(self.shared_state, "emit_event", None)
                if callable(emitter):
                    try:
                        res = emitter("CompoundingPhaseChanged", payload)
                        if inspect.isawaitable(res):
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(res)
                            except RuntimeError:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    async def _get_active_compounding_phase(self) -> Optional[Dict[str, Any]]:
        equity_now = float(await self._get_total_equity() or 0.0)
        equity_base = float(self._get_bootstrap_equity(equity_now) or 0.0)
        growth_ratio = (equity_now / max(equity_base, 1e-9)) if equity_base > 0 else 1.0

        base_phase = self._pick_compounding_phase_by_ratio(growth_ratio)
        drawdown_pct = float(getattr(self.shared_state, "metrics", {}).get("drawdown_pct", 0.0) or 0.0)
        momentum = float(self._get_realized_pnl_last_n(self._dyn_momentum_lookback) or 0.0)
        positive_streak = int(self._positive_realized_streak(self._dyn_momentum_lookback))

        # Optional momentum/drawdown modifiers to avoid phase overshoot.
        adjusted_phase = dict(base_phase or {})
        idx = self._phase_index_by_name(str(adjusted_phase.get("name", "")))
        if drawdown_pct > float(self._compounding_drawdown_guard_pct):
            idx = 0
            adjusted_phase = self._phase_from_name(self._phase_name_by_index(idx))
        elif self._compounding_min_positive_streak > 0 and positive_streak < self._compounding_min_positive_streak and idx > 0:
            adjusted_phase = self._phase_from_name(self._phase_name_by_index(idx - 1))

        mode = ""
        try:
            if self.mode_manager and hasattr(self.mode_manager, "get_mode"):
                mode = str(self.mode_manager.get_mode() or "").upper()
        except Exception:
            mode = ""
        active = bool(self._compounding_growth_enabled and mode not in {"SAFE", "PROTECTIVE"})
        self._sync_compounding_phase_state(
            adjusted_phase,
            equity_base=equity_base,
            equity_now=equity_now,
            growth_ratio=growth_ratio,
            momentum=momentum,
            positive_streak=positive_streak,
            drawdown_pct=drawdown_pct,
            active=active,
        )
        return adjusted_phase if active else None

    def _signal_confidence(self, sig: Dict[str, Any]) -> float:
        try:
            return max(0.0, min(1.0, float(sig.get("confidence", 0.0) or 0.0)))
        except Exception:
            return 0.0

    async def _estimate_sl_distance_pct(self, symbol: str) -> float:
        sl_pct_min = float(getattr(self.config, "SL_PCT_MIN", 0.003) or 0.003)
        sl_pct_max = float(getattr(self.config, "SL_PCT_MAX", 0.010) or 0.010)
        fallback = float(getattr(self.config, "TPSL_FALLBACK_ATR_PCT", sl_pct_min) or sl_pct_min)
        try:
            md = getattr(self.shared_state, "market_data", {}) or {}
            sym_md = md.get(symbol, {}) if isinstance(md, dict) else {}
            tf5 = sym_md.get("5m", {}) if isinstance(sym_md, dict) else {}
            atr = float(sym_md.get("atr") or tf5.get("atr") or 0.0)
            px = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)
            if atr > 0 and px > 0:
                sl_atr_mult = float(getattr(self.config, "SL_ATR_MULT", 1.0) or 1.0)
                atr_sl = (atr * sl_atr_mult) / px
                return max(sl_pct_min, min(sl_pct_max, float(atr_sl)))
        except Exception:
            pass
        return max(sl_pct_min, min(sl_pct_max, float(fallback)))

    async def _get_available_capital(self) -> float:
        """Best-effort spendable quote capital used by dynamic sizing."""
        try:
            from core.core_utils import _safe_await
            quote = str(getattr(self.config, "QUOTE_ASSET", "USDT") or "USDT").upper()
            getter = getattr(self.shared_state, "get_spendable_balance", None)
            if callable(getter):
                spendable = float(await _safe_await(getter(quote)) or 0.0)
                if spendable > 0:
                    return spendable
            getter_usdt = getattr(self.shared_state, "get_spendable_usdt", None)
            if callable(getter_usdt):
                spendable = float(await _safe_await(getter_usdt()) or 0.0)
                if spendable > 0:
                    return spendable
        except Exception:
            pass
        try:
            from core.core_utils import _safe_await
            if hasattr(self.shared_state, "get_nav_quote"):
                return float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
        except Exception:
            pass
        return float(getattr(self.shared_state, "nav", 0.0) or 0.0)

    async def _get_nav_quote(self) -> float:
        """Best-effort NAV in quote asset."""
        try:
            from core.core_utils import _safe_await
            if hasattr(self.shared_state, "get_nav_quote"):
                nav = float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
                if nav > 0:
                    return nav
        except Exception:
            pass
        return float(getattr(self.shared_state, "nav", 0.0) or 0.0)

    async def _compute_dynamic_position_quote(
        self,
        symbol: str,
        base_quote: float,
        sig: Dict[str, Any],
        phase: Optional[Dict[str, Any]] = None,
        capital_override: Optional[float] = None,
    ) -> float:
        if not self._dynamic_position_sizing_enabled:
            return base_quote

        if bool(sig.get("is_dust_healing") or sig.get("_dust_healing")):
            return base_quote

        use_pool_fraction = capital_override is not None and float(capital_override or 0.0) > 0.0
        if use_pool_fraction:
            available_capital = float(capital_override or 0.0)
        else:
            available_capital = float(await self._get_available_capital() or 0.0)
        if available_capital <= 0:
            return base_quote

        tier = str(sig.get("_tier", "") or "").upper()
        if tier == "B":
            base_risk_pct = float(
                getattr(
                    self.config,
                    "DYNAMIC_RISK_BUDGET_PCT_TIER_B",
                    getattr(self.config, "TIER_B_RISK_PCT", 0.005),
                )
                or 0.005
            )
        else:
            base_risk_pct = float(
                getattr(
                    self.config,
                    "DYNAMIC_RISK_BUDGET_PCT",
                    getattr(self.config, "RISK_PCT_PER_TRADE", 0.01),
                )
                or 0.01
            )

        adaptive_min_quote = 0.0
        nav_quote = float(await self._get_nav_quote() or 0.0)
        adaptive_decision = None
        try:
            if getattr(self, "_adaptive_engine", None) and self._adaptive_engine.enabled:
                metrics = getattr(self.shared_state, "metrics", {}) or {}
                drawdown_pct = float(metrics.get("drawdown_pct", 0.0) or 0.0)
                utilization_pct = float(metrics.get("capital_utilization_pct", 0.0) or 0.0)
                throughput_per_hour = float(metrics.get("usdt_per_hour", 0.0) or 0.0)
                ratio_target = float(getattr(self.config, "TARGET_PROFIT_RATIO_PER_HOUR", 0.0008) or 0.0008)
                base_target = float(getattr(self.config, "BASE_TARGET_PER_HOUR", 0.0) or 0.0)
                target_throughput = max(base_target, max(0.0, nav_quote) * ratio_target)
                vol_pct = float(await self._estimate_sl_distance_pct(symbol) or 0.0)
                fee_bps = float(getattr(self.config, "EXIT_FEE_BPS", 10.0) or 10.0)
                slippage_bps = float(getattr(self.config, "EXIT_SLIPPAGE_BPS", 0.0) or 0.0)
                min_notional = float(await self._get_exchange_min_notional(symbol) or 0.0)
                if min_notional <= 0:
                    min_notional = float(getattr(self.config, "MIN_ORDER_USDT", 10.0) or 10.0)

                open_positions = 0
                try:
                    if hasattr(self.shared_state, "open_positions_count"):
                        open_positions = int(self.shared_state.open_positions_count() or 0)
                    elif hasattr(self.shared_state, "get_open_positions"):
                        positions = self.shared_state.get_open_positions() or {}
                        if isinstance(positions, dict):
                            open_positions = len(positions)
                except Exception:
                    open_positions = 0
                max_slots = int(getattr(self.config, "MAX_POSITIONS_TOTAL", 1) or 1)
                slot_utilization = float(open_positions) / float(max(1, max_slots))

                adaptive_decision = self._adaptive_engine.evaluate(
                    symbol=symbol,
                    nav=nav_quote,
                    free_capital=available_capital,
                    base_risk_fraction=base_risk_pct,
                    volatility_pct=vol_pct,
                    drawdown_pct=drawdown_pct,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    min_notional=min_notional,
                    slot_utilization=slot_utilization,
                    throughput_per_hour=throughput_per_hour,
                    target_throughput_per_hour=target_throughput,
                    trade_history=list(getattr(self.shared_state, "trade_history", []) or []),
                )
                base_risk_pct = float(adaptive_decision.risk_fraction or base_risk_pct)
                adaptive_min_quote = float(adaptive_decision.min_trade_quote or 0.0)
                self._adaptive_min_trade_quote_by_symbol[symbol] = adaptive_min_quote
                prev_logged = float(self._adaptive_min_trade_quote_logged.get(symbol, 0.0) or 0.0)
                if abs(adaptive_min_quote - prev_logged) >= 0.25:
                    self.logger.info(
                        "[Scaling:Adaptive] %s dynamic_min_trade_quote=%.2f risk_fraction=%.4f cooldown_mult=%.2f tp_bias_mult=%.2f",
                        symbol,
                        adaptive_min_quote,
                        float(adaptive_decision.risk_fraction or base_risk_pct),
                        float(adaptive_decision.cooldown_mult or 1.0),
                        float(adaptive_decision.tp_bias_mult or 1.0),
                    )
                    self._adaptive_min_trade_quote_logged[symbol] = adaptive_min_quote

                if isinstance(sig, dict):
                    sig["_adaptive_risk_fraction"] = float(base_risk_pct)
                    sig["_adaptive_min_trade_quote"] = float(adaptive_min_quote)
                    sig["_adaptive_tp_bias_mult"] = float(adaptive_decision.tp_bias_mult)

                if not hasattr(self.shared_state, "dynamic_config") or getattr(self.shared_state, "dynamic_config", None) is None:
                    self.shared_state.dynamic_config = {}
                self.shared_state.dynamic_config["ADAPTIVE_RISK_FRACTION"] = float(base_risk_pct)
                self.shared_state.dynamic_config["ADAPTIVE_MIN_TRADE_QUOTE"] = float(adaptive_min_quote)
                self.shared_state.dynamic_config["ADAPTIVE_COOLDOWN_MULT"] = float(adaptive_decision.cooldown_mult)
                self.shared_state.dynamic_config["ADAPTIVE_TP_BIAS_MULT"] = float(adaptive_decision.tp_bias_mult)
        except Exception as e:
            self.logger.debug("[Scaling:Adaptive] evaluation failed for %s: %s", symbol, e)

        stable_boost_applied = False
        try:
            stable_enabled = bool(getattr(self.config, "STABLE_RISK_BUDGET_ENABLED", True))
            if stable_enabled:
                mode = ""
                if self.mode_manager and hasattr(self.mode_manager, "get_mode"):
                    mode = str(self.mode_manager.get_mode() or "").upper()
                metrics = getattr(self.shared_state, "metrics", {}) or {}
                drawdown_pct = float(metrics.get("drawdown_pct", 0.0) or 0.0)
                positive_streak = int(self._positive_realized_streak(self._dyn_momentum_lookback))
                stable_min_streak = int(getattr(self.config, "STABLE_RISK_MIN_POSITIVE_STREAK", 3) or 3)
                stable_max_dd = float(getattr(self.config, "STABLE_RISK_MAX_DRAWDOWN_PCT", 1.5) or 1.5)
                if mode == "NORMAL" and positive_streak >= stable_min_streak and drawdown_pct <= stable_max_dd:
                    stable_mult = float(getattr(self.config, "STABLE_RISK_BUDGET_MULT", 1.0) or 1.0)
                    if stable_mult > 0:
                        base_risk_pct *= stable_mult
                        stable_boost_applied = stable_mult > 1.0
        except Exception:
            pass
        base_risk_pct = max(0.0, min(0.95, float(base_risk_pct)))

        phase_multiplier = float((phase or {}).get("quote_mult", 1.0) or 1.0)
        confidence_raw = self._signal_confidence(sig)
        conf_min = float(getattr(self.config, "DYNAMIC_CONFIDENCE_MIN", 0.65) or 0.65)
        conf_max = float(getattr(self.config, "DYNAMIC_CONFIDENCE_MAX", 0.90) or 0.90)
        if conf_min > conf_max:
            conf_min, conf_max = conf_max, conf_min
        confidence_weight = max(conf_min, min(conf_max, confidence_raw))
        atr_pct = float(await self._estimate_sl_distance_pct(symbol))
        volatility_adjust = 1.0 / (1.0 + max(0.0, float(atr_pct)))

        base_risk_budget = available_capital * base_risk_pct
        if use_pool_fraction:
            # Shared-wallet mode: direct risk-fraction sizing from allocator usable pool.
            planned_quote = base_risk_budget
        else:
            planned_quote = base_risk_budget * confidence_weight * volatility_adjust * phase_multiplier
        if planned_quote <= 0:
            return base_quote

        # Structural clamp from adaptive engine:
        # quote = clamp(nav * risk_fraction, min_trade_quote * buffer, nav * max_position_pct)
        if not use_pool_fraction and adaptive_decision is not None and nav_quote > 0 and tier != "B":
            floor_buffer = float(getattr(self.config, "ADAPTIVE_MIN_QUOTE_BUFFER_MULT", 1.20) or 1.20)
            max_position_pct = float(
                getattr(self.config, "MAX_POSITION_EXPOSURE_PERCENTAGE", 0.20) or 0.20
            )
            nav_target_quote = nav_quote * float(base_risk_pct)
            nav_cap_quote = nav_quote * max(0.01, max_position_pct)
            floor_quote = float(adaptive_min_quote) * max(1.0, floor_buffer)
            clamped_quote = max(nav_target_quote, floor_quote)
            if nav_cap_quote > 0:
                clamped_quote = min(clamped_quote, nav_cap_quote)
            planned_quote = max(planned_quote, clamped_quote)

        phase_cap_usdt = float((phase or {}).get("max_quote", 0.0) or 0.0)
        if phase_cap_usdt <= 0:
            try:
                phase_cap_map = getattr(self.config, "PHASE_MAX_TRADE_CAP", {}) or {}
                if isinstance(phase_cap_map, dict):
                    phase_cap_usdt = float(
                        phase_cap_map.get(str((phase or {}).get("name", "")), 0.0) or 0.0
                    )
            except Exception:
                phase_cap_usdt = 0.0
        if phase_cap_usdt > 0:
            planned_quote = min(planned_quote, phase_cap_usdt)

        # Keep dynamic growth bounded relative to base sizing.
        if use_pool_fraction:
            capped_quote = float(planned_quote)
        else:
            capped_quote = min(
                planned_quote,
                float(base_quote) * max(1.0, self._dyn_upside_cap_mult),
            )

        # Do not enforce static MIN_TRADE_QUOTE here; exchange floor clamp is applied later.
        min_q = float(adaptive_min_quote or 0.0)
        max_q = float(getattr(self.config, "MAX_TRADE_QUOTE", 0.0) or 0.0)
        if max_q > 0:
            capped_quote = min(capped_quote, max_q)
        if min_q > 0:
            capped_quote = max(capped_quote, min_q)

        if not use_pool_fraction:
            phase_min_q = float((phase or {}).get("min_quote", 0.0) or 0.0)
            phase_max_q = float((phase or {}).get("max_quote", 0.0) or 0.0)
            if phase_min_q > 0:
                capped_quote = max(capped_quote, phase_min_q)
            if phase_max_q > 0:
                capped_quote = min(capped_quote, phase_max_q)

        self.logger.info(
            "[Scaling:Dynamic] %s mode=%s base=%.2f dyn=%.2f nav=%.2f risk_pct=%.3f conf=%.2f atr%%=%.3f vol_adj=%.3f phase=%s",
            symbol,
            "pool_risk_fraction" if use_pool_fraction else "signal_weighted",
            base_quote,
            capped_quote,
            available_capital,
            base_risk_pct,
            confidence_weight,
            atr_pct * 100.0,
            volatility_adjust,
            str((phase or {}).get("name", "PHASE_1_SEED")),
        )
        if stable_boost_applied:
            self.logger.info("[Scaling:Dynamic] %s stable risk budget boost applied.", symbol)
        return float(capped_quote)

    def _get_equity_tiers(self) -> List[Dict[str, Any]]:
        tiers = list(getattr(self.config, "SCALING_EQUITY_TIERS", []) or [])
        try:
            tiers.sort(key=lambda t: float(t.get("min", 0.0) or 0.0))
        except Exception:
            pass
        return tiers

    def _pick_tier_by_equity(self, tiers: List[Dict[str, Any]], equity: float) -> Optional[Dict[str, Any]]:
        for tier in tiers:
            min_e = float(tier.get("min", 0.0) or 0.0)
            max_e = tier.get("max", None)
            max_v = float(max_e) if max_e is not None else None
            if equity >= min_e and (max_v is None or equity <= max_v):
                return tier
        return None

    def _tier_name(self, tier: Optional[Dict[str, Any]]) -> str:
        if not tier:
            return ""
        return str(tier.get("name") or "")

    async def _apply_equity_tier_overrides(self) -> Optional[Dict[str, Any]]:
        tiers = self._get_equity_tiers()
        if not tiers:
            return None

        now_ts = time.time()
        equity = float(self._get_realized_equity() or 0.0)
        desired = self._pick_tier_by_equity(tiers, equity)
        if not desired:
            return None

        state = getattr(self.shared_state, "dynamic_config", {}) or {}
        current_name = str(state.get("EQUITY_TIER_CURRENT", ""))
        candidate_name = str(state.get("EQUITY_TIER_CANDIDATE", ""))
        candidate_since = float(state.get("EQUITY_TIER_CANDIDATE_SINCE", 0.0) or 0.0)

        def _tier_index(name: str) -> int:
            for i, t in enumerate(tiers):
                if self._tier_name(t) == name:
                    return i
            return -1

        current_idx = _tier_index(current_name)
        if current_idx < 0:
            current_idx = _tier_index(self._tier_name(desired))
            current_name = self._tier_name(desired)

        realized_pnl = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
        realized_pnl_last_20 = float(self._get_realized_pnl_last_n(20))
        recovery_active = bool((getattr(self.shared_state, "capital_recovery_mode", {}) or {}).get("recovery_active", False))
        active_liquidations = bool(getattr(self.shared_state, "active_liquidations", set()) or False)

        open_positions = 0
        try:
            if hasattr(self.shared_state, "open_positions_count"):
                open_positions = int(self.shared_state.open_positions_count())
            else:
                open_positions = len(getattr(self.shared_state, "get_open_positions", lambda: {})() or {})
        except Exception:
            open_positions = len(getattr(self.shared_state, "positions", {}) or {})

        block_on_recovery = bool(getattr(self.config, "EQUITY_TIER_BLOCK_ON_RECOVERY", True))
        block_on_liq = bool(getattr(self.config, "EQUITY_TIER_BLOCK_ON_LIQUIDATION", True))
        block_on_open = bool(getattr(self.config, "EQUITY_TIER_BLOCK_ON_OPEN_POSITIONS", True))

        # Demotion rule: immediate on recovery_active
        if recovery_active:
            current_idx = 0
            current_name = self._tier_name(tiers[0]) if tiers else ""
            candidate_name = ""
            candidate_since = 0.0
        else:
            # Hard guard: no tier changes unless portfolio is flat and healthy
            if (block_on_liq and active_liquidations) or (block_on_open and open_positions > 0):
                candidate_name = ""
                candidate_since = 0.0
            else:
                next_idx = min(len(tiers) - 1, current_idx + 1)
                next_tier = tiers[next_idx] if next_idx != current_idx else None
                current_tier = tiers[current_idx] if 0 <= current_idx < len(tiers) else desired
                current_max = current_tier.get("max", None)

                promote_ready = False
                if next_tier is not None and current_max is not None:
                    promote_ready = (
                        equity >= float(current_max)
                        and realized_pnl_last_20 > 0
                        and not (block_on_recovery and recovery_active)
                        and not (block_on_liq and active_liquidations)
                        and open_positions == 0
                    )

                if promote_ready and next_tier is not None:
                    hold_min = float(current_tier.get("hold_minutes", 30.0) or 30.0) * 60.0
                    desired_name = self._tier_name(next_tier)
                    if candidate_name != desired_name:
                        candidate_name = desired_name
                        candidate_since = now_ts
                    elif (now_ts - candidate_since) >= hold_min:
                        current_idx = next_idx
                        current_name = desired_name
                        candidate_name = ""
                        candidate_since = 0.0
                else:
                    candidate_name = ""
                    candidate_since = 0.0

        effective = tiers[current_idx] if 0 <= current_idx < len(tiers) else desired

        # Apply non-sizing overrides to config (sizing stays per-trade in calculate_planned_quote()).
        planned_quote = float(effective.get("planned_quote", self._default_planned_quote) or self._default_planned_quote)
        max_positions = int(effective.get("max_positions", getattr(self.config, "MAX_POSITIONS_TOTAL", 1)) or 1)
        risk_mode = str(effective.get("risk_mode", ""))
        max_daily_trades = int(effective.get("max_daily_trades", getattr(self.config, "MAX_TRADES_PER_DAY", 0)) or 0)
        tp_target_pct = float(effective.get("tp_target_pct", getattr(self.config, "TP_TARGET_PCT", 0.0)) or 0.0)
        sl_cap_pct = float(effective.get("sl_cap_pct", getattr(self.config, "SL_CAP_PCT", 0.0)) or 0.0)

        if int(getattr(self.config, "MAX_POSITIONS_TOTAL", 0) or 0) != max_positions:
            self.config.MAX_POSITIONS_TOTAL = max_positions
        if max_daily_trades > 0:
            setattr(self.config, "MAX_TRADES_PER_DAY", max_daily_trades)
        setattr(self.config, "RISK_MODE", risk_mode)
        if tp_target_pct > 0:
            setattr(self.config, "TP_TARGET_PCT", tp_target_pct)
            # Do not let tier TP caps collapse RR feasibility.
            # Keep TP_PCT_MAX at least as large as:
            # - current configured TP_PCT_MAX
            # - tier target tp_target_pct
            # - SL_PCT_MIN * TP_SL_MIN_RR (RR floor feasibility)
            try:
                current_tp_max = float(getattr(self.config, "TP_PCT_MAX", 0.0) or 0.0)
            except Exception:
                current_tp_max = 0.0
            try:
                sl_pct_min = float(getattr(self.config, "SL_PCT_MIN", 0.0) or 0.0)
            except Exception:
                sl_pct_min = 0.0
            try:
                min_rr = float(getattr(self.config, "TP_SL_MIN_RR", 1.0) or 1.0)
            except Exception:
                min_rr = 1.0
            rr_floor_tp_max = max(0.0, sl_pct_min * max(min_rr, 0.0))
            safe_tp_max = max(current_tp_max, float(tp_target_pct), rr_floor_tp_max)
            setattr(self.config, "TP_PCT_MAX", safe_tp_max)
        if sl_cap_pct != 0:
            setattr(self.config, "SL_CAP_PCT", sl_cap_pct)
            setattr(self.config, "SL_PCT_MAX", abs(sl_cap_pct))

        state.update({
            "EQUITY_TIER_CURRENT": self._tier_name(effective),
            "EQUITY_TIER_CANDIDATE": candidate_name,
            "EQUITY_TIER_CANDIDATE_SINCE": candidate_since,
            "EQUITY_TIER_EQUITY": equity,
            "EQUITY_TIER_REALIZED_PNL": realized_pnl,
            "EQUITY_TIER_REALIZED_PNL_LAST_20": realized_pnl_last_20,
            "EQUITY_TIER_PLANNED_QUOTE": planned_quote,
            "EQUITY_TIER_MAX_POSITIONS": max_positions,
            "EQUITY_TIER_RISK_MODE": risk_mode,
            "EQUITY_TIER_MAX_DAILY_TRADES": max_daily_trades,
            "EQUITY_TIER_TP_TARGET_PCT": tp_target_pct,
            "EQUITY_TIER_SL_CAP_PCT": sl_cap_pct,
            "EQUITY_TIER_OPEN_POSITIONS": open_positions,
            "EQUITY_TIER_ACTIVE_LIQUIDATIONS": active_liquidations,
            "EQUITY_TIER_LAST_TS": now_ts,
        })
        try:
            self.shared_state.dynamic_config = state
        except Exception:
            pass

        return effective
        
    async def check_scale_in_opportunity(
        self, 
        owned_positions: Dict[str, Any], 
        now_ts: float,
        focus_mode_active: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect scale-in opportunities for compounding winners.
        
        Policy:
        - Only in focus mode
        - Position age >= 60 minutes (configurable)
        - PnL > 1.0% (winning position)
        - Generate BUY signal with COMPOUNDING_ADD tag
        
        Args:
            owned_positions: Current positions {symbol: position_data}
            now_ts: Current timestamp
            focus_mode_active: Whether focus mode is active
            
        Returns:
            List of scale-in signals to inject
        """
        if not focus_mode_active:
            return []
            
        scale_signals = []
        open_trades = getattr(self.shared_state, "open_trades", {}) or {}
        
        for sym, pos in owned_positions.items():
            ot = open_trades.get(sym, {}) if isinstance(open_trades, dict) else {}
            opened_at = float(ot.get("opened_at", pos.get("opened_at", 0.0)) or 0.0)
            
            if opened_at <= 0:
                continue
                
            age_min = (now_ts - opened_at) / 60.0
            
            # Policy: Age >= configured minimum for scaling in
            if age_min < self._scale_in_min_age_min:
                continue
                
            # Get entry price
            entry = float(ot.get("entry_price", pos.get("avg_price", 0.0)) or 0.0)
            if entry <= 0:
                continue
            
            # Get current price
            price = await self._get_current_price(sym)
            if price <= 0:
                continue
                
            # Calculate PnL
            pnl_pct = ((price - entry) / entry) * 100.0
            
            # Policy: pnl > configured minimum (Compounding winners)
            if pnl_pct > self._scale_in_min_pnl_pct:
                self.logger.warning(
                    "[SCALE_IN] Winner detected: %s | age=%.1fm | pnl=%.3f%% | Scaling in for compounding",
                    sym, age_min, pnl_pct
                )
                
                # Generate scale-in signal
                sig = {
                    "symbol": sym,
                    "action": "BUY",
                    "confidence": 0.80,
                    "agent": "MetaCompounding",
                    "timestamp": now_ts,
                    "reason": f"SCALE_IN_COMPOUND age={age_min:.1f}m pnl={pnl_pct:.2f}% (winning)",
                    "tag": "COMPOUNDING_ADD",
                    "_planned_quote": self._default_planned_quote,
                    "_allow_reentry": True,
                    "_is_compounding": True,
                    "_tier": "A"
                }
                scale_signals.append(sig)
                
        return scale_signals
        
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        price = 0.0
        try:
            # Try safe_price method first
            if hasattr(self.shared_state, "safe_price"):
                from core.core_utils import _safe_await
                price = float(await _safe_await(self.shared_state.safe_price(symbol)) or 0.0)
        except Exception:
            pass
            
        # Fallback to latest_prices
        if price <= 0:
            price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0) or 0.0)
            
        return price
        
    async def scale_quote_by_account_size(self, base_q: float, agent_name: str = "Meta", context: Optional[Dict[str, Any]] = None) -> float:
        """Scale quote adaptively based on account size/NAV."""
        # Guard for dust healing: do not resize
        if context and context.get("is_dust_healing", False):
            return base_q
            
        try:
            if getattr(self.config, "SCALING_EQUITY_TIERS", None):
                return base_q
            from core.core_utils import _safe_await
            nav = 0.0
            if hasattr(self.shared_state, "get_nav_quote"):
                nav = float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
            else:
                nav = float(getattr(self.shared_state, "nav", 0.0) or 0.0)

            if nav <= 0:
                return base_q

            # Select risk multiplier based on NAV tiers
            if nav >= self._large_account_threshold:
                # Large account: use risk-based sizing
                target_q = nav * self._large_account_risk_pct
                return max(base_q, target_q)
            elif nav >= self._medium_account_threshold:
                target_q = nav * self._medium_account_risk_pct
                return max(base_q, target_q)
            else:
                # Small account: standard base sizing
                return base_q
        except Exception as e:
            self.logger.debug("[Scaling] Account scaling failed: %s", e)
            return base_q

    async def calculate_planned_quote(
        self, 
        symbol: str, 
        sig: Dict[str, Any], 
        budget_override: Optional[float] = None
    ) -> float:
        """
        Comprehensive planned quote calculation.
        Migrated from MetaController._planned_quote_for.
        
        Args:
            symbol: Trading symbol
            sig: Signal dictionary
            budget_override: Optional override for base allocation
            
        Returns:
            Calculated quote amount (USDT/QuoteAsset)
        """
        try:
            from core.core_utils import _safe_await
        except Exception:
            async def _safe_await(x):
                return x

        agent_name = sig.get("agent", "Meta")
        q = 0.0
        shared_wallet_mode = bool(getattr(self.config, "CAPITAL_ALLOCATOR_SHARED_WALLET", True))
        pool_capital_override: Optional[float] = None
        if shared_wallet_mode and budget_override is not None:
            try:
                bo = float(budget_override)
                if bo > 0:
                    pool_capital_override = bo
            except Exception:
                pool_capital_override = None

        # Dust healing is a surgical recovery action: use exact healing amount.
        if bool(sig.get("is_dust_healing") or sig.get("_dust_healing")):
            heal_q = float(
                sig.get("_healing_amount")
                or sig.get("amount_usdt")
                or sig.get("planned_quote")
                or 0.0
            )
            if heal_q > 0:
                return round(heal_q, 4)

        # 0. Apply equity-tier overrides (planned quote + max positions)
        tier = await self._apply_equity_tier_overrides()
        
        # 1. Base allocation from budget or signal
        if budget_override is not None and pool_capital_override is None:
            q = float(budget_override)
        else:
            try:
                if hasattr(self.shared_state, "get_authoritative_reservation"):
                    from core.core_utils import _safe_await
                    reserved = await _safe_await(self.shared_state.get_authoritative_reservation(agent_name))
                    if reserved and reserved > 0:
                        q = float(reserved)
            except Exception:
                pass
        
        if q <= 0:
            if tier and tier.get("planned_quote"):
                q = float(tier.get("planned_quote"))
            else:
                q = self._default_planned_quote
        base_quote_anchor = float(max(q, 0.0))
        if pool_capital_override is not None:
            # Shared wallet: budget_override is usable_pool input for risk sizing, not a hard per-trade floor.
            base_quote_anchor = 0.0
            if isinstance(sig, dict):
                sig["_wallet_usable_pool"] = float(pool_capital_override)

        # 2. Scale quote by account size (Adaptive risk)
        q = await self.scale_quote_by_account_size(q, agent_name=agent_name, context=sig)

        # 2.5 Apply compounding growth phase curve (Phase 1 -> Phase 4)
        growth_phase = None
        if self._compounding_growth_enabled:
            growth_phase = await self._get_active_compounding_phase()
            if growth_phase:
                growth_phase = dict(growth_phase)
                quote_mult = float(growth_phase.get("quote_mult", 1.0) or 1.0)
                envelope_max_positions = 0
                try:
                    if self.mode_manager and hasattr(self.mode_manager, "get_envelope"):
                        envelope = self.mode_manager.get_envelope() or {}
                        envelope_max_positions = int(envelope.get("max_positions", 0) or 0)
                except Exception:
                    envelope_max_positions = 0
                # Safeguard: compounding multipliers cannot escalate in low-capacity mode envelopes.
                if envelope_max_positions > 0 and envelope_max_positions <= 1 and quote_mult > 1.0:
                    quote_mult = 1.0
                    growth_phase["risk_mult"] = min(
                        1.0,
                        float(growth_phase.get("risk_mult", 1.0) or 1.0),
                    )
                growth_phase["quote_mult"] = quote_mult

                q *= quote_mult
                phase_min = float(growth_phase.get("min_quote", 0.0) or 0.0)
                phase_max = float(growth_phase.get("max_quote", 0.0) or 0.0)
                if envelope_max_positions > 0 and envelope_max_positions <= 1:
                    phase_min = min(phase_min, float(q))
                if phase_min > 0:
                    q = max(q, phase_min)
                if phase_max > 0:
                    q = min(q, phase_max)

        # 2.6 Dynamic position sizing formula (risk budget + confidence + volatility)
        q = await self._compute_dynamic_position_quote(
            symbol,
            q,
            sig,
            phase=growth_phase,
            capital_override=pool_capital_override,
        )

        # 3. Risk-based cap (ATR and NAV based)
        q = await self._apply_risk_based_cap(symbol, q, context=sig)

        # 4. Exchange-only floor clamp (allocation remains risk-driven per trade).
        exchange_floor = float(await self._get_exchange_min_notional(symbol) or 0.0)
        if exchange_floor <= 0:
            exchange_floor = float(getattr(self.config, "MIN_ORDER_USDT", 0.0) or 0.0)
        adaptive_symbol_floor = float(self._adaptive_min_trade_quote_by_symbol.get(symbol, 0.0) or 0.0)
        q = max(float(q), float(base_quote_anchor), float(adaptive_symbol_floor), float(exchange_floor))

        # 5. Global cap & Reservation Cap & SOP Mode Envelope Cap
        max_spend = float(getattr(self.config, "MAX_SPEND_PER_TRADE_USDT", 50.0))
        
        if self.mode_manager:
            envelope = self.mode_manager.get_envelope()
            mode_max_trade = envelope.get("max_trade_usdt", 0.0)
            if mode_max_trade > 0:
                # Mode envelope is authoritative for the max spend limit
                max_spend = mode_max_trade
                self.logger.debug("[Scaling:Envelope] Setting max_spend to mode limit: %.2f", max_spend)

        if max_spend > 0 and q > max_spend:
            q = max_spend
            
        # Reservation Cap Alignment
        if budget_override is None:
             try:
                 auth_res = float(await _safe_await(self.shared_state.get_authoritative_reservation(agent_name)))
                 if auth_res > 0 and q > auth_res:
                     self.logger.info("[Scaling] Capping quote %.2f -> %.2f (Agent Budget)", q, auth_res)
                     q = auth_res
             except Exception: 
                 pass

        # Exchange notional hard-cap safeguard.
        try:
            _, max_notional = await self._get_exchange_notional_bounds(symbol)
            if max_notional > 0 and q > max_notional:
                self.logger.info(
                    "[Scaling] Capping quote %.2f -> %.2f (Exchange max_notional)",
                    q,
                    max_notional,
                )
                q = max_notional
        except Exception:
            pass

        return round(float(q), 4)

    async def _apply_risk_based_cap(self, symbol: str, current_q: float, context: Optional[Dict[str, Any]] = None) -> float:
        """Apply ATR-based risk sizing cap."""
        # Guard for dust healing: skip risk-based sizing
        if context and context.get("is_dust_healing", False):
            return current_q
            
        q = current_q
        try:
            # FIRST: Check if TPSLEngine calculated risk-based sizing
            risk_based_quote = getattr(self.shared_state, "risk_based_quote", {}).get(symbol)
            if risk_based_quote and risk_based_quote > 0:
                self.logger.info("[Scaling] Using TPSLEngine risk-based sizing %s: %.2f USDT", symbol, risk_based_quote)
                return float(risk_based_quote)

            # FALLBACK: Original ATR-based risk sizing
            from core.core_utils import _safe_await
            # Use authoritative getter for NAV
            if hasattr(self.shared_state, "get_nav_quote"):
                nav = float(await _safe_await(self.shared_state.get_nav_quote()) or 0.0)
            else:
                nav = float(getattr(self.shared_state, "nav", 0.0) or 0.0)

            max_risk_per_trade = float(getattr(self.config, "MAX_RISK_PER_TRADE", 0.02))
            sl_atr_mult = float(getattr(self.config, "SL_ATR_MULT", 2.0))

            if nav > 0:
                target_risk = nav * max_risk_per_trade

                # Get ATR (5m timeframe baseline)
                md = getattr(self.shared_state, "market_data", {}) or {}
                atr = 0.0
                sym_data = md.get(symbol, {}).get("5m")
                if isinstance(sym_data, dict):
                    atr = float(sym_data.get("atr", 0.0))

                if atr > 0:
                    price = float(getattr(self.shared_state, "latest_prices", {}).get(symbol, 0.0))
                    if price > 0:
                         sl_dist = atr * sl_atr_mult
                         sl_pct = sl_dist / price
                         # Cap at 20% SL width for sanity
                         risk_sized_q = target_risk / max(0.005, min(0.20, sl_pct))
                         if risk_sized_q < q:
                              self.logger.info("[Scaling] Fallback risk sizing reducing %s: %.2f -> %.2f (RiskLimit: %.2f USDT)",
                                               symbol, q, risk_sized_q, target_risk)
                              q = risk_sized_q
        except Exception as e:
            self.logger.debug("[Scaling] Risk sizing failed: %s", e)
        return q

    async def _get_exchange_notional_bounds(self, symbol: str) -> Tuple[float, float]:
        """Fetch exchange min/max notional from filters."""
        min_notional = 0.0
        max_notional = 0.0
        try:
            from core.core_utils import _safe_await
            if hasattr(self.execution_manager, "get_symbol_filters_cached"):
                f = await _safe_await(self.execution_manager.get_symbol_filters_cached(symbol))
                if f:
                    min_block = f.get("MIN_NOTIONAL") or {}
                    notional_block = f.get("NOTIONAL") or {}
                    v_min = min_block.get("minNotional")
                    if v_min is None:
                        v_min = notional_block.get("minNotional")
                    v_max = notional_block.get("maxNotional")
                    if v_min is not None:
                        min_notional = float(v_min)
                    if v_max is not None:
                        max_notional = float(v_max)
        except Exception:
            pass
        return float(min_notional), float(max_notional)

    async def _get_exchange_min_notional(self, symbol: str) -> float:
        """Fetch exchange minimum notional from filters."""
        mn, _ = await self._get_exchange_notional_bounds(symbol)
        return float(mn)

    def get_scaling_config(self) -> Dict[str, Any]:
        """
        Get current scaling configuration.
        
        Returns:
            Dictionary of scaling configuration parameters
        """
        return {
            'scale_in_min_age_min': self._scale_in_min_age_min,
            'scale_in_min_pnl_pct': self._scale_in_min_pnl_pct,
            'default_planned_quote': self._default_planned_quote,
            'large_account_threshold': self._large_account_threshold,
            'medium_account_threshold': self._medium_account_threshold,
            'large_account_risk_pct': self._large_account_risk_pct,
            'medium_account_risk_pct': self._medium_account_risk_pct,
            'small_account_risk_pct': self._small_account_risk_pct,
        }
