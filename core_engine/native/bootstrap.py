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
from pathlib import Path
from typing import Any

from .adaptive_capital_engine import NativeAdaptiveCapitalEngine
from .app_context import NativeComponents
from .balance_sync import NativeBalanceSync
from .capital_allocator import NativeCapitalAllocator
from .decisions import NativeDecisionEngine
from .exchange_client import NativeExchangeClient
from .executor import NativeExecutor
from .fill_tracker import NativeFillTracker
from .market_data import NativeMarketData
from .objective_feedback_controller import NativeObjectiveFeedbackController
from .observability import NativeTelemetry
from .order_execution import NativeOrderExecution
from .polling_coordinator import NativePollingConfig, NativePollingCoordinator
from .portfolio_manager import NativePortfolioManager
from .position_manager import NativePositionManager
from .prometheus_exporter import NativePrometheusExporter
from .recovery_engine import NativeRecoveryEngine
from .safety_order_manager import NativeSafetyOrderManager
from .shared_state import NativeSharedState
from .signals import NativeSignalEngine
from .telemetry_export import NativeTelemetryExporter
from .tp_sl_engine import NativeTPSLEngine
from .trade_journal import NativeTradeJournal
from .watchdog import NativeWatchdog

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
        default_factory=list
    )  # Auto-populated via wallet scan; can be overridden via SYMBOLS env
    market_data_poll_sec: float = 2.0
    symbol_discovery_enabled: bool = True  # Auto-discover from wallet holdings
    klines_cache_size: int = 64
    stale_threshold_sec: float = 30.0

    # --- polling (legacy-style staggered with active-trades gate, not aggressive REST polling) ---
    polling_enabled: bool = True
    polling_open_orders_interval_sec: float = 25.0  # vs aggressive 5s REST polling
    polling_balance_interval_sec: float = 40.0  # vs aggressive 5s balance sync
    polling_position_interval_sec: float = 25.0
    polling_enable_active_trades_gate: bool = True  # Skip polling when no trades (huge savings)

    # --- legacy: balance_sync & fill_tracker disabled when polling_enabled=True ---
    balance_poll_sec: float = 5.0  # Ignored if polling_enabled=True
    fill_tracker_poll_sec: float = 5.0  # Ignored if polling_enabled=True

    # --- decisions / risk ---
    kelly_fraction: float = 0.25
    max_position_size_pct: float = 5.0
    max_concurrent_positions: int = 10
    min_order_usdt: float = 1.0  # Lowered from 10.0 to allow micro trades on small account
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 5.0
    risk_per_symbol_pct: float = 20.0  # Increased from 2.0: micro accounts need higher per-symbol allocation to generate trades
    capital_allocation_pct: float = 5.0
    default_planned_quote: float = 12.0  # Fixed quote per trade for small accounts (like legacy system). Autonomously scales with equity growth.

    # --- exit logic (TP/SL) ---
    tp_pct: float = 0.03  # +3% take profit
    sl_pct: float = 0.02  # -2% stop loss

    # --- signals ---
    signal_cooldown_sec: float = 0.0

    # --- telemetry / runtime ---
    telemetry_capacity: int = 1024
    telemetry_export_path: str = ""  # empty = disabled
    telemetry_export_interval_sec: float = 10.0
    duration_sec: float = 3600.0
    request_timeout_sec: float = 10.0

    # --- logging / observability ---
    trade_journal_dir: str = "logs"
    prometheus_export_path: str = ""  # empty = disabled
    prometheus_export_interval_sec: float = 10.0

    # --- feedback loop (adaptive capital engine + objective feedback controller) ---
    adaptive_capital_engine_enabled: bool = True
    ofc_enabled: bool = True
    ofc_heartbeat_sec: float = 900.0  # 15 minutes
    adaptive_risk_fraction_min: float = 0.05
    adaptive_risk_fraction_max: float = 0.35

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

        # Symbols: prefer auto-discovery unless explicitly overridden in .env
        symbols_raw = e.get("SYMBOLS", "").strip()
        symbols = (
            [s.strip().upper() for s in symbols_raw.split(",") if s.strip()] if symbols_raw else []
        )
        symbol_discovery_enabled = _bool(e.get("SYMBOL_DISCOVERY_ENABLED"), default=True)

        return cls(
            api_key=api_key,
            api_secret=api_secret,
            testnet=_bool(e.get("BINANCE_TESTNET"), default=False),
            symbols=symbols,  # Empty list → auto-scan wallet during bootstrap
            symbol_discovery_enabled=symbol_discovery_enabled,
            market_data_poll_sec=_float(e.get("MARKET_DATA_POLL_SEC"), 2.0),
            klines_cache_size=_int(e.get("KLINES_CACHE_SIZE"), 64),
            stale_threshold_sec=_float(e.get("STALE_THRESHOLD_SEC"), 30.0),
            balance_poll_sec=_float(e.get("BALANCE_POLL_SEC"), 5.0),
            fill_tracker_poll_sec=_float(e.get("FILL_TRACKER_POLL_SEC"), 5.0),
            polling_enabled=_bool(e.get("POLLING_ENABLED"), default=True),
            polling_open_orders_interval_sec=_float(
                e.get("POLLING_OPEN_ORDERS_INTERVAL_SEC"), 25.0
            ),
            polling_balance_interval_sec=_float(e.get("POLLING_BALANCE_INTERVAL_SEC"), 40.0),
            polling_position_interval_sec=_float(e.get("POLLING_POSITION_INTERVAL_SEC"), 25.0),
            polling_enable_active_trades_gate=_bool(
                e.get("POLLING_ENABLE_ACTIVE_TRADES_GATE"), default=True
            ),
            kelly_fraction=_float(e.get("KELLY_FRACTION"), 0.25),
            max_position_size_pct=_float(e.get("MAX_POSITION_PCT"), 5.0),
            max_concurrent_positions=_int(e.get("MAX_CONCURRENT_POSITIONS"), 10),
            min_order_usdt=_float(e.get("MIN_ORDER_USDT"), 1.0),
            max_drawdown_pct=_float(e.get("MAX_DRAWDOWN_PCT"), 10.0),
            daily_loss_limit_pct=_float(e.get("DAILY_LOSS_LIMIT_PCT"), 5.0),
            risk_per_symbol_pct=_float(e.get("RISK_PER_SYMBOL_PCT"), 20.0),
            capital_allocation_pct=_float(e.get("CAPITAL_ALLOCATION_PCT"), 5.0),
            default_planned_quote=_float(e.get("DEFAULT_PLANNED_QUOTE"), 12.0),
            tp_pct=_float(e.get("TP_PCT"), 0.03),
            sl_pct=_float(e.get("SL_PCT"), 0.02),
            signal_cooldown_sec=_float(e.get("SIGNAL_COOLDOWN_SEC"), 0.0),
            telemetry_capacity=_int(e.get("TELEMETRY_CAPACITY"), 1024),
            telemetry_export_path=(e.get("TELEMETRY_EXPORT_PATH") or "").strip(),
            telemetry_export_interval_sec=_float(e.get("TELEMETRY_EXPORT_INTERVAL_SEC"), 10.0),
            duration_sec=_float(e.get("DURATION_SEC"), 3600.0),
            request_timeout_sec=_float(e.get("REQUEST_TIMEOUT_SEC"), 10.0),
            trade_journal_dir=(e.get("TRADE_JOURNAL_DIR") or "logs").strip(),
            prometheus_export_path=(e.get("PROMETHEUS_EXPORT_PATH") or "").strip(),
            prometheus_export_interval_sec=_float(e.get("PROMETHEUS_EXPORT_INTERVAL_SEC"), 10.0),
            adaptive_capital_engine_enabled=_bool(
                e.get("ADAPTIVE_CAPITAL_ENGINE_ENABLED"), default=True
            ),
            ofc_enabled=_bool(e.get("OFC_ENABLED"), default=True),
            ofc_heartbeat_sec=_float(e.get("OFC_HEARTBEAT_SEC"), 900.0),
            adaptive_risk_fraction_min=_float(e.get("ADAPTIVE_RISK_FRACTION_MIN"), 0.05),
            adaptive_risk_fraction_max=_float(e.get("ADAPTIVE_RISK_FRACTION_MAX"), 0.35),
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


async def _fetch_symbol_filters_batch(
    exchange_client: NativeExchangeClient,
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Fetch and cache Binance symbol filters for a batch of symbols.
    Returns dict[symbol] -> {min_notional, step_size}.
    Errors are logged but don't block; returns partial results.
    """
    filters_cache: dict[str, dict[str, Any]] = {}
    if not symbols:
        return filters_cache

    try:
        exchange_info = await exchange_client.get_exchange_info()
        symbol_data = {s["symbol"]: s for s in exchange_info.get("symbols", [])}

        for symbol in symbols:
            try:
                sym_info = symbol_data.get(symbol, {})
                filters = sym_info.get("filters", [])

                min_notional = 10.0
                step_size = 0.00000001

                for f in filters:
                    if f.get("filterType") == "MIN_NOTIONAL":
                        min_notional = float(f.get("minNotional", 10.0))
                    elif f.get("filterType") == "LOT_SIZE":
                        step_size = float(f.get("stepSize", 0.00000001))

                filters_cache[symbol] = {
                    "min_notional": min_notional,
                    "step_size": step_size,
                }
                logger.debug(
                    "📊 %s filters: min_notional=%.2f step_size=%.8f",
                    symbol,
                    min_notional,
                    step_size,
                )
            except Exception as e:
                logger.warning("Failed to fetch filters for %s: %s", symbol, e)
                # Use defaults
                filters_cache[symbol] = {
                    "min_notional": 10.0,
                    "step_size": 0.00000001,
                }
    except Exception as e:
        logger.warning("Failed to fetch exchange info: %s (will use defaults)", e)

    return filters_cache


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

    # L0
    shared_state = NativeSharedState()

    # L1
    exchange_client = factory(cfg)
    order_execution = NativeOrderExecution(exchange_client)

    # L1: Polling coordinator (legacy-style staggered polling with active-trades gate)
    # Replaces aggressive REST polling (2s market data, 5s balance, 5s fills)
    # Reduces API weight from ~1800/min to ~200/min via:
    #   1. Wider intervals (25-40s vs 2-5s)
    #   2. Active-trades gate (skip polling when no trades)
    polling_coordinator = None
    balance_sync = None
    fill_tracker = None

    if cfg.polling_enabled:
        polling_config = NativePollingConfig(
            open_orders_interval_sec=cfg.polling_open_orders_interval_sec,
            balance_interval_sec=cfg.polling_balance_interval_sec,
            position_interval_sec=cfg.polling_position_interval_sec,
            enable_active_trades_gate=cfg.polling_enable_active_trades_gate,
        )
        polling_coordinator = NativePollingCoordinator(
            shared_state=shared_state,
            exchange_client=exchange_client,
            config=polling_config,
        )
        logger.info(
            "✅ Polling coordinator enabled (orders=%.0fs, balance=%.0fs, positions=%.0fs, gate=%s)",
            cfg.polling_open_orders_interval_sec,
            cfg.polling_balance_interval_sec,
            cfg.polling_position_interval_sec,
            "on" if cfg.polling_enable_active_trades_gate else "off",
        )
    else:
        # Legacy: fall back to aggressive polling (will cause 418 rate limits on real account)
        balance_sync = NativeBalanceSync(
            exchange_client,
            poll_interval_sec=cfg.balance_poll_sec,
        )
        fill_tracker = NativeFillTracker(
            exchange_client=exchange_client,
            shared_state=shared_state,
            poll_interval_sec=cfg.fill_tracker_poll_sec,
        )
        logger.warning("⚠️ Polling coordinator disabled; using legacy aggressive REST polling")

    # Symbol discovery: defer to per-cycle scanning unless explicitly overridden
    symbols = list(cfg.symbols) if cfg.symbols else []
    symbol_discoverer = None
    if symbols:
        logger.info("Using %d symbols from config (explicit override)", len(symbols))
    elif cfg.symbol_discovery_enabled:
        logger.info("📱 Symbol discovery: will scan wallet each cycle (not at bootstrap)")
        from .symbol_discovery import NativeSymbolDiscovery

        symbol_discoverer = NativeSymbolDiscovery(exchange_client, base_currency="USDT")
        symbols = []  # Start empty; orchestrator will populate from balance
    else:
        logger.error("No symbols configured and discovery disabled; nothing to trade")
        symbols = []

    logger.info(
        "native bootstrap: testnet=%s symbols=%d (cycle-dynamic) polling=%s",
        cfg.testnet,
        len(symbols),
        "enabled" if cfg.polling_enabled else "disabled",
    )

    # L2: Market data with WebSocket primary (zero rate limits)
    market_data = NativeMarketData(
        exchange_client,
        poll_interval_sec=cfg.market_data_poll_sec
        if not cfg.polling_enabled
        else 999.0,  # Disabled when polling is on
        symbols=symbols,  # May be empty; will be updated per-cycle
        stale_threshold_sec=cfg.stale_threshold_sec,
        klines_cache_size=cfg.klines_cache_size,
        shared_state=shared_state,  # Read prices from WebSocket if available
    )

    # L2: Optional WebSocket for real-time prices + klines (bypasses API rate limits)
    # When available, WebSocket feeds market_data.prices and market_data.market_data
    # When unavailable, REST polling serves as fallback
    market_data_ws = None
    if cfg.testnet is False:  # Only for live (not testnet)
        # Start with current symbols, or popular defaults for symbol discovery
        ws_symbols = (
            symbols
            if symbols
            else [
                "BTCUSDT",
                "ETHUSDT",
                "BNBUSDT",
                "SOLUSDT",
                "XRPUSDT",
                "ADAUSDT",
                "DOGEUSDT",
                "AVAXUSDT",
                "LUNCUSDT",
                "PEPEUSDT",
            ]
        )
        try:
            from core_engine.native.market_data_websocket import NativeMarketDataWebSocket

            market_data_ws = NativeMarketDataWebSocket(
                exchange_client=exchange_client,
                shared_state=shared_state,
                symbols=ws_symbols,
                timeframes=["1m"],
            )
            logger.info(f"✅ WebSocket market data initialized ({len(ws_symbols)} symbols)")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket initialization failed: {e} (will use REST fallback)")
            market_data_ws = None

    # L3 fill tracker: only created if polling_enabled=False (legacy fallback)
    if not cfg.polling_enabled and fill_tracker is None:
        fill_tracker = NativeFillTracker(
            exchange_client=exchange_client,
            shared_state=shared_state,
            poll_interval_sec=cfg.fill_tracker_poll_sec,
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
        min_notional_usdt=10.0,  # Layer 1: minimum notional filter for Binance LOT_SIZE compliance
    )

    # L5
    executor = NativeExecutor(
        order_execution, market_data=market_data, exchange_client=exchange_client
    )

    # L6
    telemetry = NativeTelemetry(capacity=cfg.telemetry_capacity)

    # L6 exporter (optional — only if TELEMETRY_EXPORT_PATH is configured)
    telemetry_exporter: NativeTelemetryExporter | None = None
    if cfg.telemetry_export_path:
        telemetry_exporter = NativeTelemetryExporter(
            telemetry=telemetry,
            output_path=Path(cfg.telemetry_export_path),
            interval_sec=cfg.telemetry_export_interval_sec,
        )
        await telemetry_exporter.start()

    # L8 portfolio accessor: minimal snapshot built from L0/L1/L2 state.
    # Returns a duck-typed object with the four attributes
    # NativeDecisionEngine.decide() reads: nav, nav_peak, balance,
    # positions. State is captured by reference so each cycle sees the
    # latest poller results.
    portfolio_accessor = _make_portfolio_accessor(shared_state, balance_sync)

    # L3 portfolio manager: read-only aggregator over shared_state +
    # balance_sync. Replaces the compat null-stub for the
    # ``portfolio_manager`` app_ctx key consumed by SituationEngine.
    portfolio_manager = NativePortfolioManager(
        shared_state=shared_state,
        balance_sync=balance_sync,
        min_order_usdt=cfg.min_order_usdt,
    )

    # L6 adaptive capital engine: dynamic position sizing based on performance
    ace = NativeAdaptiveCapitalEngine(config=cfg) if cfg.adaptive_capital_engine_enabled else None

    # L2 objective feedback controller: PI control on runtime knobs (confidence_floor,
    # size_multiplier, target_throughput_per_hour) every 15 minutes to track NAV target
    ofc = (
        NativeObjectiveFeedbackController(config=cfg, shared_state=shared_state)
        if cfg.ofc_enabled
        else None
    )

    # Pre-fetch symbol filters from Binance and inject into capital allocator
    # This ensures allocator has real step-sizes for each symbol, not just conservative defaults
    logger.info("📊 Fetching symbol filters from Binance (up to %d symbols)...", len(symbols))
    symbol_filters = await _fetch_symbol_filters_batch(exchange_client, symbols)
    logger.info("✅ Fetched filters for %d symbols", len(symbol_filters))

    # L6 capital allocator: allocates trading capital per buy signal
    # Hybrid strategy: fixed quote for small accounts (<$100), % for larger
    # This matches legacy system's autonomous scaling behavior
    capital_allocator = NativeCapitalAllocator(
        portfolio_manager=portfolio_manager,
        market_data=market_data,
        allocation_pct=cfg.capital_allocation_pct,
        adaptive_engine=ace,
        shared_state=shared_state,
        exchange_client=exchange_client,
        default_planned_quote=cfg.default_planned_quote,
    )
    # Inject pre-fetched filters so allocator can use real step-sizes immediately
    capital_allocator._symbol_filters_cache = symbol_filters

    # L3 position manager: read-only per-symbol accessor over
    # shared_state. Replaces the compat null-stub for the
    # ``position_manager`` app_ctx key consumed by SituationEngine
    # and DecisionEngine.
    position_manager = NativePositionManager(
        shared_state=shared_state,
        min_order_usdt=cfg.min_order_usdt,
    )

    # L4 TP/SL engine: per-symbol exit-target store. Replaces the
    # compat null-stub for the ``tp_sl_engine`` app_ctx key consumed
    # by DecisionEngine.evaluate_exit_signals.
    tp_sl_engine_native = NativeTPSLEngine(
        shared_state=shared_state,
        tp_pct=cfg.tp_pct,
        sl_pct=cfg.sl_pct,
    )

    # L4 safety order manager: per-symbol OCO intent store with
    # best-effort exchange placement. Replaces the compat null-stub
    # for the ``safety_order_manager`` app_ctx key consumed by
    # SafeExecutionEngine.place_safety_order.
    safety_order_manager = NativeSafetyOrderManager(
        exchange_client=exchange_client,
        min_order_usdt=cfg.min_order_usdt,
    )

    # L4 recovery engine: in-process self-diagnosis and recovery
    # planner. Replaces the compat null-stub for the
    # ``recovery_engine`` app_ctx key consumed by
    # OperationsEngine.recover_state / apply_recovery.
    recovery_engine = NativeRecoveryEngine(
        shared_state=shared_state,
        safety_order_manager=safety_order_manager,
    )

    # L7 watchdog: liveness + anomaly detection. Final native impl
    # for the ``watchdog`` app_ctx key consumed by
    # OperationsEngine.check_liveness / detect_anomalies. With this
    # in place the ``core_engine.native.compat`` module was retired
    # (G5 cleanup, Phase 8.3.12).
    watchdog = NativeWatchdog(
        shared_state=shared_state,
        balance_sync=balance_sync,
        market_data=market_data,
        exchange_client=exchange_client,
    )

    # L0-L7 observability enhancements (legacy features ported)
    # Trade journal: crash-safe, immutable JSONL audit trail
    trade_journal = NativeTradeJournal(log_dir=cfg.trade_journal_dir)

    # Prometheus exporter: metrics export for Grafana
    prometheus_exporter: NativePrometheusExporter | None = None
    if cfg.prometheus_export_path:
        prometheus_exporter = NativePrometheusExporter(
            output_file=cfg.prometheus_export_path,
        )

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
        telemetry_exporter=telemetry_exporter,
        portfolio_manager=portfolio_manager,
        capital_allocator=capital_allocator,
        position_manager=position_manager,
        tp_sl_engine=tp_sl_engine_native,
        safety_order_manager=safety_order_manager,
        recovery_engine=recovery_engine,
        watchdog=watchdog,
        trade_journal=trade_journal,
        prometheus_exporter=prometheus_exporter,
        fill_tracker=fill_tracker,
        adaptive_capital_engine=ace,
        polling_coordinator=polling_coordinator,
        objective_feedback_controller=ofc,
        symbol_discovery=symbol_discoverer,
        market_data_ws=market_data_ws,
    )


async def shutdown_components(components: NativeComponents) -> None:
    """
    Best-effort lifecycle teardown for components built by ``build_components``.

    Stops background pollers and closes the underlying HTTP session if
    one was created. Safe to call multiple times.
    """
    # telemetry exporter (also a background task) — stop first so its
    # final snapshot reflects an idle, drained state.
    exporter = components.telemetry_exporter
    if exporter is not None:
        try:
            await exporter.stop()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("native shutdown: telemetry_exporter.stop() raised: %r", e)

    # market data + market_data_ws + balance sync + fill tracker own background tasks
    for comp in (
        components.market_data,
        components.market_data_ws,
        components.balance_sync,
        components.fill_tracker,
    ):
        if comp is None:
            continue
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
