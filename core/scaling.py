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

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging
from decimal import Decimal
import time

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
        self._scale_in_min_age_min = float(getattr(config, 'SCALE_IN_MIN_AGE_MIN', 60.0))
        self._scale_in_min_pnl_pct = float(getattr(config, 'SCALE_IN_MIN_PNL_PCT', 1.0))
        self._default_planned_quote = float(getattr(config, 'DEFAULT_PLANNED_QUOTE', 20.0))
        
        # Account size thresholds for adaptive scaling
        self._large_account_threshold = 1000.0  # $1000+
        self._medium_account_threshold = 300.0  # $300+
        
        # Risk percentages by account size
        self._large_account_risk_pct = 0.01  # 1% per trade
        self._medium_account_risk_pct = 0.02  # 2% per trade
        self._small_account_risk_pct = 0.03  # 3% per trade
        
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

        # Apply overrides to config (authoritative for downstream consumers)
        planned_quote = float(effective.get("planned_quote", self._default_planned_quote) or self._default_planned_quote)
        max_positions = int(effective.get("max_positions", getattr(self.config, "MAX_POSITIONS_TOTAL", 1)) or 1)
        risk_mode = str(effective.get("risk_mode", ""))
        max_daily_trades = int(effective.get("max_daily_trades", getattr(self.config, "MAX_TRADES_PER_DAY", 0)) or 0)
        tp_target_pct = float(effective.get("tp_target_pct", getattr(self.config, "TP_TARGET_PCT", 0.0)) or 0.0)
        sl_cap_pct = float(effective.get("sl_cap_pct", getattr(self.config, "SL_CAP_PCT", 0.0)) or 0.0)

        if float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 0.0) or 0.0) != planned_quote:
            self.config.DEFAULT_PLANNED_QUOTE = planned_quote
        if int(getattr(self.config, "MAX_POSITIONS_TOTAL", 0) or 0) != max_positions:
            self.config.MAX_POSITIONS_TOTAL = max_positions
        if hasattr(self.config, "MAX_SPEND_PER_TRADE_USDT"):
            self.config.MAX_SPEND_PER_TRADE_USDT = planned_quote
        if max_daily_trades > 0:
            setattr(self.config, "MAX_TRADES_PER_DAY", max_daily_trades)
        setattr(self.config, "RISK_MODE", risk_mode)
        if tp_target_pct > 0:
            setattr(self.config, "TP_TARGET_PCT", tp_target_pct)
            setattr(self.config, "TP_PCT_MAX", tp_target_pct)
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
        
    async def scale_quote_by_account_size(self, base_q: float, agent_name: str = "Meta") -> float:
        """Scale quote adaptively based on account size/NAV."""
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
        agent_name = sig.get("agent", "Meta")
        q = 0.0

        # 0. Apply equity-tier overrides (planned quote + max positions)
        tier = await self._apply_equity_tier_overrides()
        
        # 1. Base allocation from budget or signal
        if budget_override is not None:
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

        # 2. Scale quote by account size (Adaptive risk)
        q = await self.scale_quote_by_account_size(q, agent_name=agent_name)

        # 3. Risk-based cap (ATR and NAV based)
        q = await self._apply_risk_based_cap(symbol, q)

        # 4. Exchange Minimums & Tier-aware Floor enforcement
        mn = await self._get_exchange_min_notional(symbol)
        exit_floor = mn
        try:
            if hasattr(self.shared_state, "compute_min_entry_quote"):
                exit_floor = await self.shared_state.compute_min_entry_quote(
                    symbol,
                    default_quote=self._default_planned_quote,
                )
        except Exception:
            exit_floor = mn
        
        # Phase A: Minimum notional floor (Tier-aware)
        if sig.get("_bootstrap") or sig.get("_force_min_notional"):
            min_notional_floor = float(self._cfg("MIN_NOTIONAL_FLOOR", 10.0))
            scout_min = float(getattr(self.config, "SCOUT_MIN_NOTIONAL", 5.0))
            # Force quote to be at least min_notional + buffer to guarantee execution
            floor_val = max(min_notional_floor, exit_floor, scout_min)
            if q < floor_val:
                self.logger.info("[Scaling] Boosting bootstrap/forced quote %.2f -> %.2f", q, floor_val)
                q = floor_val
        
        is_micro = (sig.get("_tier") == "B")
        if is_micro:
            # Tier B: Micro-sizing
            floor = max(exit_floor, 10.0) if exit_floor > 0 else 10.0
            if q < floor:
                q = floor
        else:
            # Tier A: Normal floor enforcement
            base_min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))
            buy_headroom = float(getattr(self.config, "BUY_HEADROOM_FACTOR", 1.05))
            scout_min = float(getattr(self.config, "SCOUT_MIN_NOTIONAL", 5.0))
            
            base_floor = max(exit_floor, base_min_notional)
            
            # Scout logic
            trade_count = int(getattr(self.shared_state, "trade_count", 0))
            if trade_count < 2:
                base_floor = max(base_floor, scout_min)

            if q < base_floor:
                q = base_floor * max(1.0, float(buy_headroom or 1.0))

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

        return round(float(q), 4)

    async def _apply_risk_based_cap(self, symbol: str, current_q: float) -> float:
        """Apply ATR-based risk sizing cap."""
        q = current_q
        try:
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
                              self.logger.info("[Scaling] Risk sizing reducing %s: %.2f -> %.2f (RiskLimit: %.2f USDT)", 
                                               symbol, q, risk_sized_q, target_risk)
                              q = risk_sized_q
        except Exception as e:
            self.logger.debug("[Scaling] Risk sizing failed: %s", e)
        return q

    async def _get_exchange_min_notional(self, symbol: str) -> float:
        """Fetch exchange minimum notional from filters."""
        mn = 0.0
        try:
            from core.core_utils import _safe_await
            if hasattr(self.execution_manager, "get_symbol_filters_cached"):
                f = await _safe_await(self.execution_manager.get_symbol_filters_cached(symbol))
                if f:
                    block = f.get("MIN_NOTIONAL") or f.get("NOTIONAL") or {}
                    v = block.get("minNotional")
                    if v is not None: 
                        mn = float(v)
        except Exception:
            pass
        return mn

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
