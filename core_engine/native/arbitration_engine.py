"""
Native arbitration adapter.

Wraps the native decision/risk stack behind the legacy L5 arbitration-style
interface expected by the façade engine.
"""

from __future__ import annotations

from typing import Any

from .capital_policy import compute_spendable_quote
from .decisions import PortfolioSnapshot
from .regime_gate import NativeRegimeGate


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
        self._regime_gate = NativeRegimeGate()

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
        gates_status["gate_3_regime"] = self.gate_3_regime(fused_signal)
        if not gates_status["gate_3_regime"]:
            blocking_gates.append("gate_3_regime")

        gates_status["gate_4_position_limit"] = self.gate_4_position_limit(symbol)
        if not gates_status["gate_4_position_limit"]:
            blocking_gates.append("gate_4_position_limit")

        gates_status["gate_5_capital"] = self.gate_5_capital(symbol)
        if not gates_status["gate_5_capital"]:
            blocking_gates.append("gate_5_capital")

        gates_status["gate_6_risk_manager"] = self.gate_6_risk_manager()
        if not gates_status["gate_6_risk_manager"]:
            blocking_gates.append("gate_6_risk_manager")

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
        }

    async def evaluate_gates(self, symbol: str, signal_type: str, edge: float) -> dict[str, Any]:
        return await self.evaluate(symbol, signal_type, edge)

    def gate_1_symbol_format(self, symbol: str) -> bool:
        sym = str(symbol or "").upper()
        return bool(sym) and sym.endswith("USDT") and sym.isalnum()

    def gate_2_confidence(self, edge: float, mode: str) -> bool:
        confidence = max(0.0, min(1.0, float(edge or 0.0)))
        floor = self._confidence_floor(mode)
        return confidence >= floor

    def gate_3_regime(self, signal: dict[str, Any]) -> bool:
        return self._regime_gate.evaluate(signal).allowed

    def gate_4_position_limit(self, symbol: str) -> bool:
        snapshot = self._portfolio_snapshot()
        if symbol in snapshot.positions:
            return True
        mode = self._decision_engine._resolve_mode(snapshot)
        max_positions = min(
            int(getattr(self._decision_engine, "max_concurrent_positions", mode["max_positions"])),
            int(mode["max_positions"]),
        )
        return len(snapshot.positions) < max_positions

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
        min_order = float(getattr(self._decision_engine, "min_order_usdt", 0.0) or 0.0)
        return spendable >= max(0.0, min_order)

    def gate_6_risk_manager(self) -> bool:
        if bool(getattr(self._shared_state, "trading_halted", False)):
            return False
        snapshot = self._portfolio_snapshot()
        if self._decision_engine._check_drawdown_exceeded(snapshot):
            return False
        if self._decision_engine._check_daily_loss_exceeded(snapshot):
            return False
        spendable = compute_spendable_quote(
            float(getattr(self._shared_state, "free_balance_usdt", 0.0) or 0.0),
            reserve_ratio=float(getattr(self._decision_engine, "quote_reserve_ratio", 0.0) or 0.0),
            min_reserve=float(getattr(self._decision_engine, "quote_min_reserve_usdt", 0.0) or 0.0),
        )
        return not self._decision_engine._check_total_exposure_exceeded(snapshot, spendable)

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
        positions = {
            sym: float(getattr(pos, "qty", pos) or 0.0) for sym, pos in positions_raw.items()
        }
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

    def _confidence_floor(self, mode_name: str) -> float:
        if self._mode_manager and hasattr(self._mode_manager, "get_constraints"):
            constraints = self._mode_manager.get_constraints(mode_name)
            floor = constraints.get("confidence_floor")
            if floor is not None:
                return max(0.0, min(1.0, float(floor)))
        snapshot = self._portfolio_snapshot()
        return float(self._decision_engine._resolve_mode(snapshot)["confidence_floor"])


__all__ = ["NativeArbitrationEngine"]
