"""
Native portfolio concentration guard.

Lightweight port of the legacy concentration/correlation ideas. Instead of
computing full rolling correlations, the native path uses pragmatic cluster
rules so new BUY decisions can be blocked when they would over-concentrate the
portfolio in one theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConcentrationCheck:
    allowed: bool
    reason: str = ""
    cluster: str = ""
    cluster_exposure_pct: float = 0.0


class NativeConcentrationGuard:
    """
    Cluster-based concentration gate for small portfolios.

    Exposure is approximated using existing spot positions and the proposed new
    allocation. This keeps the check synchronous and dependency-light while
    still preventing obvious "all one beta" behavior.
    """

    def __init__(
        self,
        *,
        max_cluster_exposure_pct: float = 40.0,
        cluster_map: dict[str, str] | None = None,
    ) -> None:
        self.max_cluster_exposure_pct = max(0.0, float(max_cluster_exposure_pct))
        self.cluster_map = {str(k).upper(): str(v).upper() for k, v in (cluster_map or {}).items()}

    def check_new_position(
        self,
        *,
        symbol: str,
        proposed_quote: float,
        portfolio: Any,
        price_map: dict[str, float] | None = None,
    ) -> ConcentrationCheck:
        if self.max_cluster_exposure_pct <= 0:
            return ConcentrationCheck(True)

        nav = max(float(getattr(portfolio, "nav", 0.0) or 0.0), 1.0)
        cluster = self.classify_symbol(symbol)
        current_cluster_quote = self._cluster_exposure_quote(
            cluster=cluster,
            portfolio=portfolio,
            price_map=price_map or {},
        )
        cluster_exposure_pct = (
            (current_cluster_quote + max(0.0, float(proposed_quote or 0.0))) / nav
        ) * 100.0
        if cluster_exposure_pct > self.max_cluster_exposure_pct:
            return ConcentrationCheck(
                False,
                reason="cluster_exposure_limit",
                cluster=cluster,
                cluster_exposure_pct=cluster_exposure_pct,
            )
        return ConcentrationCheck(True, cluster=cluster, cluster_exposure_pct=cluster_exposure_pct)

    def classify_symbol(self, symbol: str) -> str:
        sym = str(symbol or "").upper()
        if sym in self.cluster_map:
            return self.cluster_map[sym]

        if sym.endswith("USDT"):
            base = sym[:-4]
        else:
            base = sym

        majors = {"BTC", "ETH"}
        layer1 = {"BNB", "SOL", "ADA", "AVAX", "ATOM", "DOT", "NEAR"}
        memes = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK"}
        exchange = {"BNB"}

        if base in majors:
            return "MAJORS"
        if base in memes:
            return "MEME"
        if base in exchange:
            return "EXCHANGE"
        if base in layer1:
            return "L1"
        return f"ASSET:{base}"

    def _cluster_exposure_quote(
        self,
        *,
        cluster: str,
        portfolio: Any,
        price_map: dict[str, float],
    ) -> float:
        positions = getattr(portfolio, "positions", {}) or {}
        total = 0.0
        for sym, qty in positions.items():
            if self.classify_symbol(sym) != cluster:
                continue
            price = float(price_map.get(sym, 0.0) or 0.0)
            if price <= 0:
                continue
            total += float(qty or 0.0) * price
        return total
