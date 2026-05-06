"""
Native L0-L8 Bootstrap (Phase 8.2.8)

Constructs ``NativeComponents`` from configuration. This is the only
place in the native stack that owns I/O setup (exchange client,
credentials). Everything downstream (``build_native_app_ctx``) is pure
assembly.

Why a separate module?
----------------------
* Keeps ``app_context.build_native_app_ctx`` purely functional
  (zero I/O, zero credentials) and trivially testable.
* Concentrates env-var / credential handling so paper / live / backtest
  all use the same code path with different ``BootstrapConfig`` inputs.
* Provides an injection seam for tests (``exchange_client_factory``)
  so we can exercise the bootstrap without touching the network.

Usage::

    from core_engine.native.bootstrap import (
        BootstrapConfig, build_components,
    )
    from core_engine.native.app_context import build_native_app_ctx

    cfg = BootstrapConfig.from_env()
    components = await build_components(cfg)
    app_ctx, orch = build_native_app_ctx(components)
    try:
        await orch.run_loop(duration_sec=cfg.duration_sec)
    finally:
        await shutdown_components(components)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .app_context import NativeComponents
from .balance_sync import NativeBalanceSync
from .decisions import NativeDecisionEngine
from .exchange_client import NativeExchangeClient
from .executor import NativeExecutor
from .market_data import NativeMarketData
from .observability import NativeTelemetry
from .order_execution import NativeOrderExecution
from .shared_state import NativeSharedState
from .signals import NativeSignalEngine

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapConfig:
    """
    All inputs needed to construct the native L0-L6 stack.

    Values mirror what ``ConfigLoader`` reads from the environment.
    Fields are explicit (no ``**kwargs``) so the surface is auditable.
    """

    # --- credentials ---
    api_key: str
    api_secret: str
    testnet: bool = False

    # --- market data ---
    symbols: list[str] = field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
        ]
    )
    market_data_poll_sec: float = 2.0
    klines_cache_size: int = 64
    stale_threshold_sec: float = 30.0

    # --- balance sync ---
    balance_poll_sec: float = 5.0

    # --- decisions / risk ---
    kelly_fraction: float = 0.25
    max_position_size_pct: float = 5.0
    max_concurrent_positions: int = 10
    min_order_usdt: float = 10.0
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 5.0
    risk_per_symbol_pct: float = 2.0

    # --- signals ---
    signal_cooldown_sec: float = 0.0

    # --- telemetry / runtime ---
    telemetry_capacity: int = 1024
    duration_sec: float = 3600.0
    request_timeout_sec: float = 10.0

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> BootstrapConfig:
        """
        Construct from environment variables (or an explicit mapping).

        Required: ``BINANCE_API_KEY``, ``BINANCE_API_SECRET``.
        Raises ``ValueError`` if either is missing/empty.
        """
        e = env if env is not None else os.environ

        api_key = (e.get("BINANCE_API_KEY") or "").strip()
        api_secret = (e.get("BINANCE_API_SECRET") or "").strip()
        if not api_key or not api_secret:
            raise ValueError(
                "BootstrapConfig.from_env: BINANCE_API_KEY and " "BINANCE_API_SECRET must be set."
            )

        symbols_raw = e.get(
            "SYMBOLS",
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT",
        )
        symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

        return cls(
            api_key=api_key,
            api_secret=api_secret,
            testnet=_bool(e.get("BINANCE_TESTNET"), default=False),
            symbols=symbols,
            market_data_poll_sec=_float(e.get("MARKET_DATA_POLL_SEC"), 2.0),
            klines_cache_size=_int(e.get("KLINES_CACHE_SIZE"), 64),
            stale_threshold_sec=_float(e.get("STALE_THRESHOLD_SEC"), 30.0),
            balance_poll_sec=_float(e.get("BALANCE_POLL_SEC"), 5.0),
            kelly_fraction=_float(e.get("KELLY_FRACTION"), 0.25),
            max_position_size_pct=_float(e.get("MAX_POSITION_PCT"), 5.0),
            max_concurrent_positions=_int(e.get("MAX_CONCURRENT_POSITIONS"), 10),
            min_order_usdt=_float(e.get("MIN_ORDER_USDT"), 10.0),
            max_drawdown_pct=_float(e.get("MAX_DRAWDOWN_PCT"), 10.0),
            daily_loss_limit_pct=_float(e.get("DAILY_LOSS_LIMIT_PCT"), 5.0),
            risk_per_symbol_pct=_float(e.get("RISK_PER_SYMBOL_PCT"), 2.0),
            signal_cooldown_sec=_float(e.get("SIGNAL_COOLDOWN_SEC"), 0.0),
            telemetry_capacity=_int(e.get("TELEMETRY_CAPACITY"), 1024),
            duration_sec=_float(e.get("DURATION_SEC"), 3600.0),
            request_timeout_sec=_float(e.get("REQUEST_TIMEOUT_SEC"), 10.0),
        )


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------
ExchangeClientFactory = Callable[[BootstrapConfig], NativeExchangeClient]
"""Optional injection seam for tests; defaults to real ``NativeExchangeClient``."""


def _default_exchange_factory(cfg: BootstrapConfig) -> NativeExchangeClient:
    return NativeExchangeClient(
        api_key=cfg.api_key,
        api_secret=cfg.api_secret,
        testnet=cfg.testnet,
        request_timeout_sec=cfg.request_timeout_sec,
    )


# ----------------------------------------------------------------------
# Portfolio snapshot helper
# ----------------------------------------------------------------------
@dataclass
class _PortfolioSnapshot:
    """
    Minimal duck-typed snapshot consumed by ``NativeDecisionEngine.decide``.

    Attributes match exactly what ``decisions.py`` reads:
    ``nav`` (float), ``nav_peak`` (float), ``balance`` (dict asset->qty),
    ``positions`` (dict symbol->qty).
    """

    nav: float
    nav_peak: float
    balance: dict[str, float]
    positions: dict[str, float]


def _make_portfolio_accessor(
    shared_state: NativeSharedState,
    balance_sync: NativeBalanceSync,
) -> Callable[[], _PortfolioSnapshot]:
    """
    Build a zero-arg callable returning a fresh ``_PortfolioSnapshot``.

    NAV is sourced from ``shared_state.nav_usdt`` (canonical field).
    ``nav_peak`` tracks the running maximum so the drawdown gate in
    ``NativeDecisionEngine`` has a meaningful denominator from cycle 1.
    Balances come from the L1 poller; positions from L0 shared state
    (qty extracted from the ``Position`` record).
    """
    peak_holder: dict[str, float] = {"peak": 0.0}

    def _accessor() -> _PortfolioSnapshot:
        nav = float(getattr(shared_state, "nav_usdt", 0.0))
        balance = balance_sync.get_balance()

        # Positions: shared_state.positions is dict[str, Position]; the
        # decision engine only needs symbol -> qty.
        positions_raw = getattr(shared_state, "positions", {}) or {}
        positions: dict[str, float] = {}
        for sym, pos in positions_raw.items():
            qty = getattr(pos, "qty", None)
            if qty is None and isinstance(pos, (int, float)):
                qty = float(pos)  # already a scalar
            positions[sym] = float(qty or 0.0)

        # Fallback: if shared_state hasn't been hydrated with a NAV
        # (e.g. very early cycles before NAV writer wires up), derive
        # an approximate NAV from the USDT balance + open positions
        # marked at the latest cached price. This keeps the drawdown
        # gate in NativeDecisionEngine from spuriously firing 100%.
        if nav <= 0.0:
            usdt = float(balance.get("USDT", 0.0))
            price_cache = getattr(shared_state, "price_cache", {}) or {}
            position_value = sum(
                qty * float(price_cache.get(sym, 0.0)) for sym, qty in positions.items()
            )
            nav = usdt + position_value

        if nav > peak_holder["peak"]:
            peak_holder["peak"] = nav
        peak = peak_holder["peak"] if peak_holder["peak"] > 0.0 else max(nav, 1.0)

        return _PortfolioSnapshot(
            nav=nav,
            nav_peak=peak,
            balance=balance,
            positions=positions,
        )

    return _accessor


async def build_components(
    cfg: BootstrapConfig,
    *,
    exchange_client_factory: ExchangeClientFactory | None = None,
) -> NativeComponents:
    """
    Build all L0-L6 native components from ``cfg``.

    The returned ``NativeComponents`` is ready to be passed to
    ``core_engine.native.app_context.build_native_app_ctx``.

    This function does **not** call ``start()`` on background-polling
    components (market data, balance sync). The orchestrator's
    ``start()`` does that. The caller is responsible for shutdown via
    ``shutdown_components()``.

    Parameters
    ----------
    cfg
        Frozen ``BootstrapConfig``.
    exchange_client_factory
        Optional callable to construct the exchange client. Tests can
        inject a stub here to avoid network setup.
    """
    factory = exchange_client_factory or _default_exchange_factory

    logger.info(
        "native bootstrap: testnet=%s symbols=%d md_poll=%.1fs balance_poll=%.1fs",
        cfg.testnet,
        len(cfg.symbols),
        cfg.market_data_poll_sec,
        cfg.balance_poll_sec,
    )

    # L0
    shared_state = NativeSharedState()

    # L1
    exchange_client = factory(cfg)
    order_execution = NativeOrderExecution(exchange_client)
    balance_sync = NativeBalanceSync(
        exchange_client,
        poll_interval_sec=cfg.balance_poll_sec,
    )

    # L2
    market_data = NativeMarketData(
        exchange_client,
        poll_interval_sec=cfg.market_data_poll_sec,
        symbols=list(cfg.symbols),
        stale_threshold_sec=cfg.stale_threshold_sec,
        klines_cache_size=cfg.klines_cache_size,
    )

    # L3
    signal_engine = NativeSignalEngine(cooldown_sec=cfg.signal_cooldown_sec)

    # L4
    decision_engine = NativeDecisionEngine(
        kelly_fraction=cfg.kelly_fraction,
        max_position_size_pct=cfg.max_position_size_pct,
        max_concurrent_positions=cfg.max_concurrent_positions,
        min_order_usdt=cfg.min_order_usdt,
        max_drawdown_pct=cfg.max_drawdown_pct,
        daily_loss_limit_pct=cfg.daily_loss_limit_pct,
        risk_per_symbol_pct=cfg.risk_per_symbol_pct,
    )

    # L5
    executor = NativeExecutor(order_execution)

    # L6
    telemetry = NativeTelemetry(capacity=cfg.telemetry_capacity)

    # L8 portfolio accessor: minimal snapshot built from L0/L1/L2 state.
    # Returns a duck-typed object with the four attributes
    # NativeDecisionEngine.decide() reads: nav, nav_peak, balance,
    # positions. State is captured by reference so each cycle sees the
    # latest poller results.
    portfolio_accessor = _make_portfolio_accessor(shared_state, balance_sync)

    return NativeComponents(
        shared_state=shared_state,
        market_data=market_data,
        signal_engine=signal_engine,
        decision_engine=decision_engine,
        executor=executor,
        balance_sync=balance_sync,
        telemetry=telemetry,
        exchange_client=exchange_client,
        portfolio_accessor=portfolio_accessor,
    )


async def shutdown_components(components: NativeComponents) -> None:
    """
    Best-effort lifecycle teardown for components built by ``build_components``.

    Stops background pollers and closes the underlying HTTP session if
    one was created. Safe to call multiple times.
    """
    # market data + balance sync own background tasks
    for comp in (components.market_data, components.balance_sync):
        try:
            await comp.stop()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("native shutdown: %s.stop() raised: %r", type(comp).__name__, e)

    # close exchange HTTP session if reachable
    client = components.exchange_client or getattr(components.balance_sync, "_client", None)
    if client is not None and hasattr(client, "close"):
        try:
            await client.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("native shutdown: exchange_client.close() raised: %r", e)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _bool(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    return default


def _float(raw: Any, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int(raw: Any, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


__all__ = [
    "BootstrapConfig",
    "ExchangeClientFactory",
    "build_components",
    "shutdown_components",
]
