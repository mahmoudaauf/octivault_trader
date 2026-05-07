"""
Native signal-fusion adapter.

Exposes a lightweight L5-style interface over the native signal engine so the
legacy façade can consume fused signals without pretending this layer is absent.
"""

from __future__ import annotations

from typing import Any

from .signals import AggregatedSignal


class NativeSignalFusion:
    def __init__(
        self,
        *,
        signal_engine: Any,
        market_data: Any | None = None,
        shared_state: Any | None = None,
        mode_manager: Any | None = None,
    ) -> None:
        self._signal_engine = signal_engine
        self._market_data = market_data
        self._shared_state = shared_state
        self._mode_manager = mode_manager

    async def fuse_signal(self, symbol: str) -> dict[str, Any] | None:
        agg = self._evaluate_symbol(symbol)
        if agg is None:
            return None
        return self._to_payload(agg)

    def apply_weights(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        if not signals:
            return {
                "direction": "HOLD",
                "edge_score": 0.0,
                "confidence": 0.0,
                "threshold": self.get_threshold(),
            }

        total_weight = 0.0
        net_score = 0.0
        for sig in signals:
            weight = float(sig.get("weight", 1.0) or 1.0)
            edge_score = float(sig.get("edge_score", sig.get("score", 0.0)) or 0.0)
            signal_type = str(
                sig.get("signal_type", sig.get("direction", "HOLD")) or "HOLD"
            ).upper()
            total_weight += weight
            if signal_type == "BUY":
                net_score += weight * edge_score
            elif signal_type == "SELL":
                net_score -= weight * edge_score

        if total_weight <= 0.0:
            return {
                "direction": "HOLD",
                "edge_score": 0.0,
                "confidence": 0.0,
                "threshold": self.get_threshold(),
            }

        normalized = net_score / total_weight
        direction = "HOLD"
        if normalized > 1e-9:
            direction = "BUY"
        elif normalized < -1e-9:
            direction = "SELL"
        edge_score = min(1.0, abs(float(normalized)))
        return {
            "direction": direction,
            "edge_score": edge_score,
            "confidence": edge_score,
            "threshold": self.get_threshold(),
        }

    def get_threshold(self) -> float:
        overrides = (
            getattr(self._shared_state, "runtime_overrides", {}) if self._shared_state else {}
        )
        override_floor = overrides.get("CONFIDENCE_FLOOR")
        if override_floor is not None:
            try:
                return max(0.0, min(1.0, float(override_floor)))
            except (TypeError, ValueError):
                pass

        mode_name = str(getattr(self._shared_state, "current_mode", "") or "").upper()
        if self._mode_manager and hasattr(self._mode_manager, "get_constraints"):
            constraints = self._mode_manager.get_constraints(mode_name)
            floor = constraints.get("confidence_floor")
            if floor is not None:
                return max(0.0, min(1.0, float(floor)))

        return 0.5

    def _evaluate_symbol(self, symbol: str) -> AggregatedSignal | None:
        if not self._signal_engine:
            return None
        klines = self._get_best_klines(symbol)
        if not klines or not hasattr(self._signal_engine, "evaluate"):
            return None
        return self._signal_engine.evaluate(symbol, klines)

    def _get_best_klines(self, symbol: str) -> list[list[Any]] | None:
        klines_cache = getattr(self._market_data, "_klines", {}) if self._market_data else {}
        if not klines_cache:
            return None

        best_data: list[list[Any]] | None = None
        best_size = 0
        for key, (_ts, data) in klines_cache.items():
            if isinstance(key, tuple) and key and key[0] == symbol:
                data_size = len(data) if isinstance(data, (list, tuple)) else 0
                if data_size > best_size:
                    best_data = list(data)
                    best_size = data_size
        return best_data

    def _to_payload(self, agg: AggregatedSignal) -> dict[str, Any]:
        payload = {
            "symbol": agg.symbol,
            "signal_type": agg.direction,
            "direction": agg.direction,
            "edge_score": float(agg.score),
            "score": float(agg.score),
            "confidence": float(agg.score),
            "threshold": self.get_threshold(),
            "timestamp": agg.ts,
            "contributions": [
                {
                    "strategy": s.strategy,
                    "direction": s.direction,
                    "score": float(s.score),
                }
                for s in agg.contributions
            ],
        }
        payload.update(dict(agg.meta or {}))
        return payload


__all__ = ["NativeSignalFusion"]
