"""
Native arbitration adapter.

Wraps the native decision/risk stack behind the legacy L5 arbitration-style
interface expected by the façade engine.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from .capital_policy import compute_spendable_quote
from .decisions import PortfolioSnapshot
from .regime_gate import NativeRegimeGate
from .symbol_performance_tracker import SymbolPerformanceTracker

_log = logging.getLogger(__name__)

# How long (seconds) to block re-entry on a symbol after a BUY — forces diversification
_SYMBOL_REENTRY_COOLDOWN_SECS = 900  # 15 minutes (normal — faster recycling)
_SYMBOL_LOSS_STREAK_COOLDOWN_SECS = 14400  # 4 hours (after 3 consecutive losses — was 24h)
_SYMBOL_LOSS_STREAK_THRESHOLD = 3  # losses before extended cooldown
_SYMBOL_LOSS_STREAK_RESET_SECS = 86400  # 24 hours without a trade resets streak


class NativeArbitrationEngine:
    def __init__(
        self,
        *,
        shared_state: Any,
        decision_engine: Any,
        signal_fusion: Any | None = None,
        mode_manager: Any | None = None,
    ) -> None:
        self._shared_state = shared_state
        self._decision_engine = decision_engine
        self._signal_fusion = signal_fusion
        self._mode_manager = mode_manager
        self._regime_gate = NativeRegimeGate(shared_state=shared_state)
        self._last_buy_ts: dict[str, float] = {}  # symbol → epoch of last BUY
        self._loss_streak: dict[str, int] = {}  # symbol → consecutive loss count
        self._last_trade_ts: dict[str, float] = {}  # symbol → epoch of last closed trade
        self._last_sl_ts: dict[str, float] = {}  # symbol → epoch of last SL exit
        self._global_buy_history: list[float] = []  # epoch of every BUY (pruned to 1h)
        self._global_sl_history: list[float] = []  # epoch of every SL exit (pruned to 1h)
        self._arb_state_path = os.path.join("logs", "arb_state.json")
        self._perf_tracker = SymbolPerformanceTracker()
        self._load_streak_state()

    def record_trade_outcome(self, symbol: str, pnl: float) -> None:
        """Call after every confirmed closed trade. Updates both gate_7 streak and gate_8 performance."""
        self._perf_tracker.record_trade(symbol, pnl)
        if pnl <= 0:
            self.record_loss(symbol)
        else:
            self.record_win(symbol)

    def get_symbol_size_mult(self, symbol: str) -> float:
        """Returns gate_8 size multiplier for this symbol (0.0-1.25)."""
        return self._perf_tracker.get_size_multiplier(symbol)

    def get_perf_summary(self) -> dict:
        return self._perf_tracker.summary()

    def record_buy(self, symbol: str) -> None:
        """Call this after a BUY order fills to start the reentry cooldown."""
        now = time.time()
        self._last_buy_ts[symbol] = now
        self._global_buy_history.append(now)
        _log.info(f"[gate_7] {symbol} reentry cooldown started ({_SYMBOL_REENTRY_COOLDOWN_SECS}s)")

    def record_loss(self, symbol: str) -> None:
        """Call after a closed trade that resulted in a net loss."""
        now = time.time()
        # Reset streak if it's been 48h since last trade on this symbol
        last = self._last_trade_ts.get(symbol, 0.0)
        if now - last > _SYMBOL_LOSS_STREAK_RESET_SECS:
            self._loss_streak[symbol] = 0
        self._loss_streak[symbol] = self._loss_streak.get(symbol, 0) + 1
        self._last_trade_ts[symbol] = now
        streak = self._loss_streak[symbol]
        if streak >= _SYMBOL_LOSS_STREAK_THRESHOLD:
            _log.warning(
                f"[gate_7] {symbol} loss streak={streak} — extended cooldown "
                f"{_SYMBOL_LOSS_STREAK_COOLDOWN_SECS/3600:.0f}h applied"
            )
        else:
            _log.info(
                f"[gate_7] {symbol} loss recorded (streak={streak}/{_SYMBOL_LOSS_STREAK_THRESHOLD})"
            )
        self._save_streak_state()

    def record_win(self, symbol: str) -> None:
        """Call after a closed trade that resulted in a net profit — resets loss streak."""
        prev = self._loss_streak.get(symbol, 0)
        self._loss_streak[symbol] = 0
        self._last_trade_ts[symbol] = time.time()
        if prev > 0:
            _log.info(f"[gate_7] {symbol} win — loss streak reset (was {prev})")
        self._save_streak_state()

    async def evaluate(self, symbol: str, signal_type: str, edge_score: float) -> dict[str, Any]:
        signal_type = str(signal_type or "").upper()
        gates_status: dict[str, bool] = {}
        blocking_gates: list[str] = []

        gates_status["gate_1_symbol_format"] = self.gate_1_symbol_format(symbol)
        if not gates_status["gate_1_symbol_format"]:
            blocking_gates.append("gate_1_symbol_format")

        mode_name = str(getattr(self._shared_state, "current_mode", "") or "").upper()
        gates_status["gate_2_confidence"] = self.gate_2_confidence(edge_score, mode_name)
        if not gates_status["gate_2_confidence"]:
            blocking_gates.append("gate_2_confidence")

        fused_signal = await self._fuse_signal(symbol, signal_type, edge_score)
        _regime_decision = self._regime_gate.evaluate(fused_signal)
        gates_status["gate_3_regime"] = _regime_decision.allowed
        regime_floor_bump = float(getattr(_regime_decision, "confidence_floor_bump", 0.0) or 0.0)
        if not gates_status["gate_3_regime"]:
            blocking_gates.append("gate_3_regime")

        gates_status["gate_4_position_limit"] = self.gate_4_position_limit(symbol)
        if not gates_status["gate_4_position_limit"]:
            blocking_gates.append("gate_4_position_limit")

        gates_status["gate_5_capital"] = self.gate_5_capital(symbol)
        if not gates_status["gate_5_capital"]:
            blocking_gates.append("gate_5_capital")

        # For SELL: skip exposure check in gate_6 — high exposure is exactly when we need to sell
        gates_status["gate_6_risk_manager"] = self.gate_6_risk_manager(
            check_exposure=(signal_type != "SELL")
        )
        if not gates_status["gate_6_risk_manager"]:
            blocking_gates.append("gate_6_risk_manager")

        if signal_type == "BUY":
            gates_status["gate_7_reentry_cooldown"] = self.gate_7_reentry_cooldown(symbol)
            if not gates_status["gate_7_reentry_cooldown"]:
                blocking_gates.append("gate_7_reentry_cooldown")

            tradeable, perf_reason = self._perf_tracker.is_tradeable(symbol)
            gates_status["gate_8_symbol_performance"] = tradeable
            if not tradeable:
                blocking_gates.append("gate_8_symbol_performance")
                _log.info("[gate_8] %s blocked by performance tracker: %s", symbol, perf_reason)
            else:
                _log.debug("[gate_8] %s allowed: %s", symbol, perf_reason)

            gates_status["gate_9_global_pace"] = self.gate_9_global_pace()
            if not gates_status["gate_9_global_pace"]:
                blocking_gates.append("gate_9_global_pace")

        passed = all(gates_status.values())
        if signal_type == "SELL":
            passed = gates_status["gate_1_symbol_format"] and gates_status["gate_6_risk_manager"]
            blocking_gates = [
                g for g in blocking_gates if g in {"gate_1_symbol_format", "gate_6_risk_manager"}
            ]

        reason = "passed" if passed else ",".join(blocking_gates)
        return {
            "passed": passed,
            "gates_status": gates_status,
            "blocking_gates": blocking_gates,
            "reason": reason,
            "mode": mode_name or "BOOTSTRAP",
            "regime_floor_bump": regime_floor_bump,
            "symbol_size_mult": self._perf_tracker.get_size_multiplier(symbol),
        }

    async def evaluate_gates(self, symbol: str, signal_type: str, edge: float) -> dict[str, Any]:
        return await self.evaluate(symbol, signal_type, edge)

    def gate_1_symbol_format(self, symbol: str) -> bool:
        sym = str(symbol or "").upper()
        return bool(sym) and sym.endswith("USDT") and sym.isalnum()

    def gate_2_confidence(self, edge: float, mode: str) -> bool:
        confidence = max(0.0, min(1.0, float(edge or 0.0)))
        return confidence >= self._confidence_floor(mode)

    def gate_3_regime(self, signal: dict[str, Any]) -> bool:
        # evaluate() inlines this call to capture regime_floor_bump; this method remains for tests.
        return self._regime_gate.evaluate(signal).allowed

    def gate_4_position_limit(self, symbol: str) -> bool:
        snapshot = self._portfolio_snapshot()
        positions = dict(getattr(self._shared_state, "positions", {}) or {})
        balance = dict(getattr(self._shared_state, "balance", {}) or {})
        prices = dict(getattr(self._shared_state, "prices", {}) or {})
        # Use min_notional ($10) for both the per-symbol re-buy block and the slot counter so
        # they are consistent: a dust position below $10 neither blocks re-entry nor counts as a slot.
        count_threshold = float(getattr(self._decision_engine, "min_notional_usdt", 10.0) or 10.0)
        rebuy_threshold = count_threshold  # was $1; raised to $10 to match decision engine logic

        def _pos_qty(pos: object) -> float:
            return float(
                getattr(pos, "qty", None) or (pos.get("qty", 0) if isinstance(pos, dict) else 0)
            )

        def _bal_notional(asset: str, sym: str) -> float:
            qty = float(balance.get(asset, 0.0) or 0.0)
            if qty <= 0.0:
                return 0.0
            price = float(prices.get(sym, 0.0) or 0.0)
            # Unknown price: only block if qty is large enough to plausibly be > min_notional.
            # Use a conservative $1/unit estimate so tiny dust quantities don't claim slots.
            fallback_price = 1.0 if price <= 0.0 else price
            return qty * fallback_price

        def _pos_notional(sym: str, pos: object) -> float:
            qty = _pos_qty(pos)
            if qty <= 1e-8:
                return 0.0
            price = float(prices.get(sym, 0.0) or 0.0)
            return qty * price if price > 0.0 else qty * 1e6

        # Block if this symbol is already held in positions or balance with >= $1 notional
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            pos_n = _pos_notional(symbol, positions[symbol]) if symbol in positions else 0.0
            bal_n = _bal_notional(base, symbol)
            if pos_n >= rebuy_threshold:
                _log.info(f"[gate_4] {symbol} BLOCK: held in positions notional=${pos_n:.2f}")
                return False
            if bal_n >= rebuy_threshold:
                _log.info(f"[gate_4] {symbol} BLOCK: held in balance notional=${bal_n:.2f}")
                return False

        # Also block via recovery classification
        if self._decision_engine._is_slot_blocking_position(symbol, snapshot):
            _log.info(f"[gate_4] {symbol} BLOCK: _is_slot_blocking_position=True")
            return False

        mode = self._decision_engine._resolve_mode(snapshot)
        max_positions = min(
            int(getattr(self._decision_engine, "max_concurrent_positions", mode["max_positions"])),
            int(mode["max_positions"]),
        )

        # BNB excluded from slot count only when below min_notional (pure fee dust).
        # When BNB is actively traded (>= count_threshold), it occupies a slot like any symbol.
        fee_reserve_assets = frozenset()
        pos_count = sum(
            1 for sym, pos in positions.items() if _pos_notional(sym, pos) >= count_threshold
        )
        bal_count = sum(
            1
            for asset, qty in balance.items()
            if asset != "USDT" and _bal_notional(asset, f"{asset}USDT") >= count_threshold
        )
        tradable_count = self._decision_engine._count_active_tradable_positions(snapshot)
        active_count = max(tradable_count, pos_count, bal_count)
        _log.info(
            f"[gate_4] {symbol} counts: tradable={tradable_count} pos={pos_count} bal={bal_count} "
            f"active={active_count} max={max_positions} → {'PASS' if active_count < max_positions else 'BLOCK'}"
        )
        return active_count < max_positions

    def gate_5_capital(self, symbol: str) -> bool:
        del symbol
        free_usdt = float(getattr(self._shared_state, "free_balance_usdt", 0.0) or 0.0)
        if free_usdt <= 0.0:
            free_usdt = float(getattr(self._shared_state, "balance", {}).get("USDT", 0.0) or 0.0)
        reserved_quote = 0.0
        if hasattr(self._shared_state, "reserved_quote_total"):
            reserved_quote = float(self._shared_state.reserved_quote_total("USDT") or 0.0)
        spendable = compute_spendable_quote(
            free_usdt,
            reserve_ratio=float(getattr(self._decision_engine, "quote_reserve_ratio", 0.0) or 0.0),
            min_reserve=float(getattr(self._decision_engine, "quote_min_reserve_usdt", 0.0) or 0.0),
            reserved_quote=reserved_quote,
        )
        # Use the higher of min_order_usdt and min_notional_usdt so gate_5 aligns with
        # the allocator's actual floor — prevents false passes that execution will reject.
        min_order = float(getattr(self._decision_engine, "min_order_usdt", 0.0) or 0.0)
        min_notional = float(getattr(self._decision_engine, "min_notional_usdt", 10.0) or 10.0)
        return spendable >= max(0.0, min_order, min_notional)

    def gate_6_risk_manager(self, check_exposure: bool = True) -> bool:
        # trading_halted blocks new BUYs only — SELL/exit signals must always pass
        if check_exposure and bool(getattr(self._shared_state, "trading_halted", False)):
            return False
        snapshot = self._portfolio_snapshot()
        if self._decision_engine._check_drawdown_exceeded(snapshot):
            return False
        if self._decision_engine._check_daily_loss_exceeded(snapshot):
            return False
        if not check_exposure:
            return True
        spendable = compute_spendable_quote(
            float(getattr(self._shared_state, "free_balance_usdt", 0.0) or 0.0),
            reserve_ratio=float(getattr(self._decision_engine, "quote_reserve_ratio", 0.0) or 0.0),
            min_reserve=float(getattr(self._decision_engine, "quote_min_reserve_usdt", 0.0) or 0.0),
        )
        return not self._decision_engine._check_total_exposure_exceeded(snapshot, spendable)

    def gate_7_reentry_cooldown(self, symbol: str) -> bool:
        now = time.time()

        # Post-SL block: if this symbol was stopped out recently, require 2h before re-entry
        last_sl = self._last_sl_ts.get(symbol, 0.0)
        if last_sl > 0:
            elapsed_since_sl = now - last_sl
            if elapsed_since_sl < 7200:
                remaining = int(7200 - elapsed_since_sl)
                _log.info(
                    "[gate_7] %s SL-exit cooldown — %dm%ds remaining before re-entry",
                    symbol,
                    remaining // 60,
                    remaining % 60,
                )
                return False

        # Extended cooldown if symbol has hit the loss streak threshold
        streak = self._loss_streak.get(symbol, 0)
        if streak >= _SYMBOL_LOSS_STREAK_THRESHOLD:
            last_trade = self._last_trade_ts.get(symbol, 0.0)
            elapsed_since_loss = now - last_trade if last_trade > 0 else 0.0
            if elapsed_since_loss < _SYMBOL_LOSS_STREAK_COOLDOWN_SECS:
                remaining = int(_SYMBOL_LOSS_STREAK_COOLDOWN_SECS - elapsed_since_loss)
                _log.info(
                    f"[gate_7] {symbol} LOSS_STREAK={streak} — extended cooldown, "
                    f"{remaining//3600}h{(remaining%3600)//60}m remaining"
                )
                return False
            else:
                # Cooldown expired — reset streak so symbol gets a fresh start
                self._loss_streak[symbol] = 0
                _log.info(f"[gate_7] {symbol} loss-streak cooldown expired — streak reset")

        # Normal post-buy cooldown
        last_buy = self._last_buy_ts.get(symbol, 0.0)
        if last_buy <= 0.0:
            return True
        elapsed = now - last_buy
        if elapsed < _SYMBOL_REENTRY_COOLDOWN_SECS:
            remaining = int(_SYMBOL_REENTRY_COOLDOWN_SECS - elapsed)
            _log.info(f"[gate_7] {symbol} reentry blocked — {remaining}s cooldown remaining")
            return False
        return True

    async def _fuse_signal(
        self, symbol: str, signal_type: str, edge_score: float
    ) -> dict[str, Any]:
        if self._signal_fusion and hasattr(self._signal_fusion, "fuse_signal"):
            fused = await self._signal_fusion.fuse_signal(symbol)
            if fused:
                return fused
        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": signal_type,
            "score": float(edge_score or 0.0),
            "confidence": float(edge_score or 0.0),
        }

    def _portfolio_snapshot(self) -> PortfolioSnapshot:
        nav = float(getattr(self._shared_state, "nav_usdt", 0.0) or 0.0)
        free_usdt = float(getattr(self._shared_state, "free_balance_usdt", 0.0) or 0.0)
        price_cache = dict(getattr(self._shared_state, "price_cache", {}) or {})
        positions_raw = getattr(self._shared_state, "positions", {}) or {}
        positions: dict[str, float] = {}
        for sym, pos in positions_raw.items():
            qty = getattr(pos, "qty", None)
            if qty is None and isinstance(pos, dict):
                qty = pos.get("qty", 0.0)
            positions[sym] = float(qty or 0.0)
        balances = dict(getattr(self._shared_state, "balance", {}) or {})
        if nav <= 0.0:
            nav = free_usdt + sum(
                qty * float(price_cache.get(sym, 0.0) or 0.0) for sym, qty in positions.items()
            )
        nav_peak = float(getattr(self._shared_state, "metrics", {}).get("peak_nav", 0.0) or 0.0)
        if nav_peak <= 0.0:
            nav_peak = max(nav, 1.0)
        session_anchor = float(getattr(self._shared_state, "session_anchor_nav", 0.0) or 0.0)
        realized_pnl = float(
            getattr(self._shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0
        )
        daily_pnl_pct = 0.0
        if session_anchor > 0.0:
            daily_pnl_pct = (realized_pnl / session_anchor) * 100.0
        return PortfolioSnapshot(
            nav=nav,
            nav_peak=nav_peak,
            balance=balances,
            positions=positions,
            open_orders=dict(getattr(self._shared_state, "open_orders", {}) or {}),
            daily_pnl_pct=daily_pnl_pct,
            mode_name=str(getattr(self._shared_state, "current_mode", "") or ""),
        )

    def record_sl_exit(self, symbol: str) -> None:
        """Call after a confirmed SL exit. Extends per-symbol cooldown and tracks global SL rate."""
        now = time.time()
        self._last_sl_ts[symbol] = now
        self._global_sl_history.append(now)
        _log.info("[gate_7] SL exit recorded for %s — 2h re-entry block applied", symbol)
        self._save_streak_state()

    def gate_9_global_pace(self) -> bool:
        """
        Dynamic global pace gate — prevents opening too many positions in a short window.
        Adapts cooldown windows to current win rate (losing streak → longer windows).
        Also acts as a circuit breaker when multiple SLs fire in quick succession.
        """
        now = time.time()
        # Prune to rolling 2h window
        self._global_buy_history = [t for t in self._global_buy_history if now - t < 7200]
        self._global_sl_history = [t for t in self._global_sl_history if now - t < 7200]

        metrics = dict(getattr(self._shared_state, "metrics", {}) or {})
        # Prefer TP/SL-only win rate — isolates signal quality from recovery/rotation noise.
        # Falls back to blended rate when tpsl metric not yet populated.
        win_rate = float(metrics.get("win_rate_tpsl", metrics.get("win_rate_window", 0.5)) or 0.5)

        # Circuit breaker: 2+ SLs in the last hour → block for 1h after the most recent SL
        recent_sls_1h = [t for t in self._global_sl_history if now - t < 3600]
        if len(recent_sls_1h) >= 2:
            most_recent_sl = max(recent_sls_1h)
            remaining = int(3600 - (now - most_recent_sl))
            if remaining > 0:
                _log.info(
                    "[gate_9] SL circuit breaker — %d SLs in 1h, blocking new BUYs for %dm%ds",
                    len(recent_sls_1h),
                    remaining // 60,
                    remaining % 60,
                )
                return False

        # Adaptive pace: BUY limit and window both scale with win rate.
        # Higher win rate → more BUYs allowed in a shorter window.
        if win_rate >= 0.70:
            pace_window_sec = 900  # 15 min — strong edge, deploy aggressively
            max_buys_in_window = 4
        elif win_rate >= 0.60:
            pace_window_sec = 900  # 15 min — edge confirmed
            max_buys_in_window = 3
        elif win_rate >= 0.50:
            pace_window_sec = 1800  # 30 min — slight edge, moderate pace
            max_buys_in_window = 3
        elif win_rate >= 0.40:
            pace_window_sec = 2700  # 45 min — neutral, cautious
            max_buys_in_window = 3
        else:
            pace_window_sec = 3600  # 60 min — losing streak, slow down
            max_buys_in_window = 2

        recent_buys = [t for t in self._global_buy_history if now - t < pace_window_sec]
        if len(recent_buys) >= max_buys_in_window:
            oldest = min(recent_buys)
            remaining = int(pace_window_sec - (now - oldest))
            if remaining > 0:
                _log.info(
                    "[gate_9] Global pace gate — %d/%d BUYs in %dm window (win_rate=%.0f%%), blocking for %dm%ds",
                    len(recent_buys),
                    max_buys_in_window,
                    pace_window_sec // 60,
                    win_rate * 100,
                    remaining // 60,
                    remaining % 60,
                )
                return False
        return True

    def get_dynamic_floor_delta(self, symbol: str) -> float:
        """
        Returns a confidence floor adjustment based on live market conditions.
        Additive on top of playbook floor + OFC floor — does NOT replace them.
        Range: [-0.08, +0.35]
        """
        now = time.time()
        metrics = dict(getattr(self._shared_state, "metrics", {}) or {})
        win_rate = float(metrics.get("win_rate_tpsl", metrics.get("win_rate_window", 0.5)) or 0.5)
        trend_regime = str(metrics.get("trend_regime", "UNKNOWN") or "UNKNOWN").upper()

        delta = 0.0

        # Win rate: raise bar when losing, relax slightly when winning
        if win_rate < 0.40:
            delta += 0.08
        elif win_rate < 0.50:
            delta += 0.04
        elif win_rate > 0.65:
            delta -= 0.03

        # Trend regime awareness (separate from regime_gate bump which can be suppressed)
        if trend_regime == "DOWNTREND":
            delta += 0.08
        elif trend_regime in {"CHOPPY", "VOLATILE"}:
            delta += 0.04
        elif trend_regime == "UPTREND":
            delta -= 0.03

        # Per-symbol SL penalty: symbol recently stopped out → require higher conviction
        last_sl = self._last_sl_ts.get(symbol, 0.0)
        if last_sl > 0 and now - last_sl < 7200:  # within 2h
            delta += 0.05

        # Global SL rate penalty
        recent_sls = sum(1 for t in self._global_sl_history if now - t < 3600)
        if recent_sls >= 3:
            delta += 0.08
        elif recent_sls >= 1:
            delta += 0.04

        return max(-0.05, min(0.18, delta))

    def _confidence_floor(self, mode_name: str) -> float:
        # Get base floor — from mode_manager if available, else from decision engine
        if self._mode_manager and hasattr(self._mode_manager, "get_constraints"):
            constraints = self._mode_manager.get_constraints(mode_name)
            mm_floor = constraints.get("confidence_floor")
            if mm_floor is not None:
                base_floor = max(0.0, min(1.0, float(mm_floor)))
            else:
                snapshot = self._portfolio_snapshot()
                base_floor = float(
                    self._decision_engine._resolve_mode(snapshot)["confidence_floor"]
                )
        else:
            snapshot = self._portfolio_snapshot()
            base_floor = float(self._decision_engine._resolve_mode(snapshot)["confidence_floor"])
        # Adapt floor to market regime: raise bar in hostile/noisy conditions.
        regime = str(
            (getattr(self._shared_state, "metrics", {}) or {}).get("market_regime", "") or ""
        ).upper()
        if regime in {"CHOPPY", "RANGING"}:
            # Fee drag consumes the entire typical move — require high conviction only.
            base_floor = min(0.90, base_floor + 0.25)
        elif regime == "UNKNOWN":
            # No regime data — apply mild guard; don't over-restrict
            base_floor = min(0.75, base_floor + 0.05)
        elif regime == "TRENDING":
            # Clear trend — slight relaxation to capture momentum
            base_floor = max(0.45, base_floor - 0.05)
        elif regime in {"VOLATILE", "CRISIS"}:
            base_floor = min(0.85, base_floor + 0.15)
        return max(0.0, min(1.0, base_floor))

    def _load_streak_state(self) -> None:
        """Restore loss streaks and trade timestamps from disk so 24h cooldowns survive restarts."""
        try:
            if os.path.exists(self._arb_state_path):
                with open(self._arb_state_path) as _f:
                    data = json.loads(_f.read())
                self._loss_streak = {k: int(v) for k, v in data.get("loss_streak", {}).items()}
                self._last_trade_ts = {
                    k: float(v) for k, v in data.get("last_trade_ts", {}).items()
                }
                self._last_sl_ts = {k: float(v) for k, v in data.get("last_sl_ts", {}).items()}
                # Restore global SL history (only keep last 2h)
                now = time.time()
                self._global_sl_history = [
                    float(t) for t in data.get("global_sl_history", []) if now - float(t) < 7200
                ]
                # Prune entries older than the reset window — no point keeping them
                stale = [
                    s
                    for s, ts in self._last_trade_ts.items()
                    if now - ts > _SYMBOL_LOSS_STREAK_RESET_SECS
                ]
                for s in stale:
                    self._loss_streak.pop(s, None)
                    self._last_trade_ts.pop(s, None)
                # Prune stale SL timestamps (older than 2h)
                stale_sl = [s for s, ts in self._last_sl_ts.items() if now - ts > 7200]
                for s in stale_sl:
                    self._last_sl_ts.pop(s, None)
                active = {
                    s: self._loss_streak[s] for s in self._loss_streak if self._loss_streak[s] > 0
                }
                if active:
                    _log.info("[gate_7] Restored loss streaks from disk: %s", active)
        except Exception as exc:
            _log.debug("[gate_7] _load_streak_state failed: %s", exc)

    def _save_streak_state(self) -> None:
        """Persist loss streaks and trade timestamps so cooldowns survive restarts."""
        try:
            os.makedirs(os.path.dirname(self._arb_state_path), exist_ok=True)
            now = time.time()
            payload = {
                "loss_streak": dict(self._loss_streak),
                "last_trade_ts": dict(self._last_trade_ts),
                "last_sl_ts": dict(self._last_sl_ts),
                "global_sl_history": [t for t in self._global_sl_history if now - t < 7200],
                "saved_at": now,
            }
            tmp = self._arb_state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self._arb_state_path)
        except Exception as exc:
            _log.debug("[gate_7] _save_streak_state failed: %s", exc)


__all__ = ["NativeArbitrationEngine"]
