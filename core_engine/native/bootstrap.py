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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adaptive_capital_engine import NativeAdaptiveCapitalEngine
from .app_context import NativeComponents
from .arbitration_engine import NativeArbitrationEngine
from .balance_sync import NativeBalanceSync
from .balance_validator import NativeBalanceValidator
from .bounded_cache import NativeBoundedCache
from .capital_allocator import NativeCapitalAllocator
from .decisions import NativeDecisionEngine
from .exchange_client import NativeExchangeClient
from .executor import NativeExecutor
from .fill_tracker import NativeFillTracker
from .health_monitor import NativeHealthMonitor
from .market_data import NativeMarketData
from .market_regime_detector import NativeMarketRegimeDetector
from .mode_manager import NativeModeManager
from .objective_feedback_controller import NativeObjectiveFeedbackController
from .observability import NativeTelemetry
from .order_execution import NativeOrderExecution
from .paper_signal_generator import PaperModeSignalGenerator
from .polling_coordinator import NativePollingConfig, NativePollingCoordinator
from .portfolio_manager import NativePortfolioManager
from .position_hydration_engine import NativePositionHydrationEngine
from .position_manager import NativePositionManager
from .prometheus_exporter import NativePrometheusExporter
from .recovery_engine import NativeRecoveryEngine
from .runtime_state import NativeRuntimeStateExporter, load_runtime_state
from .safety_order_manager import NativeSafetyOrderManager
from .shared_state import NativeSharedState
from .signal_fusion import NativeSignalFusion
from .signal_manager_bridge import SignalManagerBridge
from .signals import NativeSignalEngine
from .startup_state_machine import NativeStartupStateMachine
from .telemetry_export import NativeTelemetryExporter
from .tp_sl_engine import NativeTPSLEngine
from .trade_journal import NativeTradeJournal
from .watchdog import NativeWatchdog
from .fear_greed import FearGreedFetcher

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
    paper_mode: bool = False  # Paper mode: use simulated $1000 USDT balance
    synthetic_live_signals_enabled: bool = False  # Explicit opt-in for live synthetic signals

    # --- market data ---
    symbols: list[str] = field(
        default_factory=list
    )  # Auto-populated via wallet scan; can be overridden via SYMBOLS env
    market_data_poll_sec: float = 2.0
    symbol_discovery_enabled: bool = True  # Auto-discover from wallet holdings
    symbol_discovery_interval_sec: float = 900.0
    symbol_discovery_empty_retry_sec: float = 300.0
    klines_cache_size: int = 64
    stale_threshold_sec: float = 30.0
    signal_adapter_timeout_sec: float = 20.0

    # --- polling (legacy-style staggered with active-trades gate, not aggressive REST polling) ---
    polling_enabled: bool = True
    polling_open_orders_interval_sec: float = 25.0  # vs aggressive 5s REST polling
    polling_balance_interval_sec: float = 40.0  # vs aggressive 5s balance sync
    polling_position_interval_sec: float = 25.0
    polling_enable_active_trades_gate: bool = True  # Skip polling when no trades (huge savings)

    # --- legacy: balance_sync & fill_tracker disabled when polling_enabled=True ---
    balance_poll_sec: float = 5.0  # Ignored if polling_enabled=True
    balance_min_refresh_interval_sec: float = 60.0
    fill_tracker_poll_sec: float = 5.0  # Ignored if polling_enabled=True

    # --- decisions / risk ---
    kelly_fraction: float = 0.25
    max_position_size_pct: float = 5.0
    max_concurrent_positions: int = 10
    min_order_usdt: float = 1.0  # Lowered from 10.0 to allow micro trades on small account
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 2.0
    risk_per_symbol_pct: float = 20.0  # Increased from 2.0: micro accounts need higher per-symbol allocation to generate trades
    capital_allocation_pct: float = 5.0
    default_planned_quote: float = 12.0  # Fixed quote per trade for small accounts (like legacy system). Autonomously scales with equity growth.
    daily_compounding_enabled: bool = True
    daily_compounding_state_path: str = "logs/daily_compounding_state.json"
    quote_reserve_ratio: float = 0.10
    quote_min_reserve_usdt: float = 0.0
    max_total_exposure_pct: float = 60.0
    confidence_floor: float = 0.50
    max_cluster_exposure_pct: float = 40.0
    taker_fee_bps: float = 10.0
    maker_fee_bps: float = 10.0
    exit_slippage_bps: float = 10.0

    # --- exit logic (TP/SL) ---
    tp_pct: float = 0.03  # +3% take profit
    sl_pct: float = 0.02  # -2% stop loss

    # --- signals ---
    signal_cooldown_sec: float = 0.0

    # --- telemetry / runtime ---
    telemetry_capacity: int = 1024
    telemetry_export_path: str = ""  # empty = disabled
    telemetry_export_interval_sec: float = 10.0
    runtime_state_path: str = "runtime_state_snapshot.json"
    runtime_state_interval_sec: float = 10.0
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
            paper_mode=_bool(e.get("PAPER_MODE"), default=False),
            synthetic_live_signals_enabled=_bool(
                e.get("SYNTHETIC_LIVE_SIGNALS_ENABLED"), default=False
            ),
            symbols=symbols,  # Empty list → auto-scan wallet during bootstrap
            symbol_discovery_enabled=symbol_discovery_enabled,
            symbol_discovery_interval_sec=_float(e.get("SYMBOL_DISCOVERY_INTERVAL_SEC"), 900.0),
            symbol_discovery_empty_retry_sec=_float(
                e.get("SYMBOL_DISCOVERY_EMPTY_RETRY_SEC"), 300.0
            ),
            market_data_poll_sec=_float(e.get("MARKET_DATA_POLL_SEC"), 2.0),
            klines_cache_size=_int(e.get("KLINES_CACHE_SIZE"), 64),
            stale_threshold_sec=_float(e.get("STALE_THRESHOLD_SEC"), 30.0),
            signal_adapter_timeout_sec=_float(e.get("SIGNAL_ADAPTER_TIMEOUT_SEC"), 20.0),
            balance_poll_sec=_float(e.get("BALANCE_POLL_SEC"), 5.0),
            balance_min_refresh_interval_sec=_float(
                e.get("BALANCE_MIN_REFRESH_INTERVAL_SEC"), 60.0
            ),
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
            daily_loss_limit_pct=_float(e.get("DAILY_LOSS_LIMIT_PCT"), 2.0),
            risk_per_symbol_pct=_float(e.get("RISK_PER_SYMBOL_PCT"), 20.0),
            capital_allocation_pct=_float(e.get("CAPITAL_ALLOCATION_PCT"), 5.0),
            default_planned_quote=_float(e.get("DEFAULT_PLANNED_QUOTE"), 12.0),
            daily_compounding_enabled=_bool(e.get("DAILY_COMPOUNDING_ENABLED"), default=True),
            daily_compounding_state_path=(
                e.get("DAILY_COMPOUNDING_STATE_PATH") or "logs/daily_compounding_state.json"
            ).strip(),
            quote_reserve_ratio=_float(e.get("QUOTE_RESERVE_RATIO"), 0.10),
            quote_min_reserve_usdt=_float(e.get("QUOTE_MIN_RESERVE_USDT"), 0.0),
            max_total_exposure_pct=_float(e.get("MAX_TOTAL_EXPOSURE_PCT"), 60.0),
            confidence_floor=_float(e.get("CONFIDENCE_FLOOR"), 0.50),
            max_cluster_exposure_pct=_float(e.get("MAX_CLUSTER_EXPOSURE_PCT"), 40.0),
            taker_fee_bps=_float(e.get("TAKER_FEE_BPS"), 10.0),
            maker_fee_bps=_float(e.get("MAKER_FEE_BPS"), 10.0),
            exit_slippage_bps=_float(
                e.get("EXIT_SLIPPAGE_BPS", e.get("CR_PRICE_SLIPPAGE_BPS")), 10.0
            ),
            tp_pct=_float(e.get("TP_PCT"), 0.03),
            sl_pct=_float(e.get("SL_PCT"), 0.02),
            signal_cooldown_sec=_float(e.get("SIGNAL_COOLDOWN_SEC"), 0.0),
            telemetry_capacity=_int(e.get("TELEMETRY_CAPACITY"), 1024),
            telemetry_export_path=(e.get("TELEMETRY_EXPORT_PATH") or "").strip(),
            telemetry_export_interval_sec=_float(e.get("TELEMETRY_EXPORT_INTERVAL_SEC"), 10.0),
            runtime_state_path=(
                e.get("RUNTIME_STATE_PATH") or "runtime_state_snapshot.json"
            ).strip(),
            runtime_state_interval_sec=_float(e.get("RUNTIME_STATE_INTERVAL_SEC"), 10.0),
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
    # In paper mode, use sentinel keys to enable simulated trading
    api_key = "paper_key" if cfg.paper_mode else cfg.api_key
    api_secret = "paper_secret" if cfg.paper_mode else cfg.api_secret
    return NativeExchangeClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=cfg.testnet,
        request_timeout_sec=cfg.request_timeout_sec,
        signed_request_cooldown_sec=0.0,  # Disable local cooldown; rely on Binance 429 throttling only
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
    mode_name: str = ""


def _make_portfolio_accessor(
    shared_state: NativeSharedState,
    balance_sync: NativeBalanceSync | None,
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
        if balance_sync is not None and hasattr(balance_sync, "get_balance"):
            balance = balance_sync.get_balance()
        else:
            balance = dict(getattr(shared_state, "balance", {}) or {})

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
            mode_name=str(getattr(shared_state, "current_mode", "") or ""),
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
    # Canonical cost source used by signal, sizing, TP/SL, and exit gates.
    shared_state.config = cfg
    if cfg.runtime_state_path:
        restored = load_runtime_state(shared_state, Path(cfg.runtime_state_path))
        if restored:
            logger.info("✅ Restored runtime state from %s", cfg.runtime_state_path)
            # Clear expired throttle states (throttle window may have passed since last run)
            throttle_ts = float(getattr(shared_state, "exchange_throttle_until_ts", 0.0) or 0.0)
            if throttle_ts > 0 and throttle_ts <= time.time():
                logger.info("🟢 Throttle window expired; clearing throttle state")
                shared_state.set_exchange_throttle(False, reason="", until_ts=0.0)

    # L1
    exchange_client = factory(cfg)
    if hasattr(exchange_client, "restore_throttle_state"):
        exchange_client.restore_throttle_state(
            until_ts=float(getattr(shared_state, "exchange_throttle_until_ts", 0.0) or 0.0),
            reason=str(getattr(shared_state, "exchange_throttle_reason", "") or ""),
        )
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
        def update_shared_state_balance(balances: dict[str, float]) -> None:
            """Callback: update shared_state.free_balance_usdt when balances change."""
            usdt_balance = float(balances.get("USDT", 0.0))
            if usdt_balance > 0:
                shared_state.free_balance_usdt = usdt_balance

        balance_sync = NativeBalanceSync(
            exchange_client,
            poll_interval_sec=cfg.balance_poll_sec,
            min_refresh_interval_sec=cfg.balance_min_refresh_interval_sec,
            on_update=update_shared_state_balance,
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
        shared_state.set_accepted_symbols(symbols)
    elif cfg.symbol_discovery_enabled:
        logger.info("📱 Symbol discovery: will scan wallet each cycle (not at bootstrap)")
        from .symbol_discovery import NativeSymbolDiscovery

        symbol_discoverer = NativeSymbolDiscovery(
            exchange_client,
            base_currency="USDT",
            shared_state=shared_state,
            min_scan_interval_sec=cfg.symbol_discovery_interval_sec,
            empty_scan_retry_sec=cfg.symbol_discovery_empty_retry_sec,
        )
        symbols = []  # Start empty; orchestrator will populate from balance
    else:
        logger.error("No symbols configured and discovery disabled; nothing to trade")
        symbols = []

    # Symbol Rotator: scores all trained models by regime/momentum/win_rate every 2h
    symbol_rotator = None
    try:
        from .symbol_rotator import SymbolRotator

        symbol_rotator = SymbolRotator(
            shared_state=shared_state,
            regime_detector=None,  # wired below after market_regime_detector is built
            market_data=None,  # wired below after market_data is built
            perf_tracker=None,
            fallback_symbols=symbols or [],
        )
        logger.info("✅ SymbolRotator instantiated (TOP_N=%d, interval=2h)", 8)
    except Exception as _e:
        logger.warning("SymbolRotator init failed (non-fatal): %r", _e)

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
        prime_on_start=cfg.testnet,
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
            # Pre-fetch 3000 klines per symbol so ML models can train immediately on startup.
            # MLForecaster trains on 5m candles (self.timeframe="5m"), so pre-fetch 5m.
            try:
                await market_data_ws.prefetch_klines_history(limit=3000, timeframe="5m")
            except Exception as _pf_err:
                logger.warning("Kline pre-fetch failed (non-fatal): %s", _pf_err)
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

    # L3: Paper mode signal generator (for synthetic signals when explicitly allowed)
    paper_signal_gen: PaperModeSignalGenerator | None = None
    logger.info(f"[Bootstrap] paper_mode={cfg.paper_mode}, symbols={len(cfg.symbols)}")
    enable_paper_signals = cfg.paper_mode or cfg.synthetic_live_signals_enabled
    if enable_paper_signals and cfg.symbols:
        paper_signal_gen = PaperModeSignalGenerator(
            config=cfg,
            symbols=cfg.symbols,
            enabled=True,
        )
        logger.info(
            "📊 Synthetic signal generator enabled (%d symbols, paper_mode=%s)",
            len(cfg.symbols),
            cfg.paper_mode,
        )
    elif cfg.symbols:
        logger.info("📊 Synthetic signal generator disabled for this run")

    # L3: Legacy signal agents (MLForecaster, SymbolScreener)
    ml_forecaster = None
    symbol_screener = None
    try:
        # Try to import legacy agents
        from agents.ml_forecaster import MLForecaster
        from agents.symbol_screener import SymbolScreener
        from core_engine.native.model_manager import ModelManager

        # Initialize ModelManager for MLForecaster
        model_manager = ModelManager(config=cfg)

        # Instantiate MLForecaster — always initialize; symbols synced dynamically from SharedState
        try:
            ml_forecaster = MLForecaster(
                shared_state=shared_state,
                execution_manager=None,  # Not used by forecaster
                config=cfg,
                symbols=list(cfg.symbols) if cfg.symbols else [],
                timeframe="5m",
                market_data_feed=market_data,
                exchange_client=exchange_client,
                model_manager=model_manager,
            )
            logger.info(
                "✅ MLForecaster initialized (%d symbols at boot, will sync dynamically)",
                len(cfg.symbols),
            )
        except Exception as e:
            logger.warning("Failed to initialize MLForecaster: %s", e)

        # Instantiate SymbolScreener
        try:
            symbol_screener = SymbolScreener(
                shared_state=shared_state,
                exchange_client=exchange_client,
                config=cfg,
            )
            logger.info("✅ SymbolScreener initialized")
        except Exception as e:
            logger.warning("Failed to initialize SymbolScreener: %s", e)

    except ImportError as e:
        logger.warning("Legacy agents not available: %s (using paper signals only)", e)

    # L3: Signal manager bridge (integrates legacy + paper mode signals)
    signal_manager_bridge = SignalManagerBridge(
        config=cfg,
        legacy_signal_manager=None,
        paper_generator=paper_signal_gen,
        shared_state=shared_state,
        ml_forecaster=ml_forecaster,
        symbol_screener=symbol_screener,
        timeout_sec=cfg.signal_adapter_timeout_sec,
        native_signal_engine=signal_engine,  # gap2: NativeSignalEngine vote layer
    )

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
        quote_reserve_ratio=cfg.quote_reserve_ratio,
        quote_min_reserve_usdt=cfg.quote_min_reserve_usdt,
        max_total_exposure_pct=cfg.max_total_exposure_pct,
        confidence_floor=cfg.confidence_floor,
        max_cluster_exposure_pct=cfg.max_cluster_exposure_pct,
    )
    mode_manager = NativeModeManager(
        bootstrap_trade_usdt=max(cfg.min_order_usdt, 20.0),
        recovery_trade_usdt=max(cfg.min_order_usdt, 50.0),
        normal_trade_usdt=max(cfg.min_order_usdt, 150.0),
    )

    # Trade journal — created early so executor can write fills immediately
    trade_journal = NativeTradeJournal(log_dir=cfg.trade_journal_dir)

    # L5
    executor = NativeExecutor(
        order_execution,
        market_data=market_data,
        exchange_client=exchange_client,
        shared_state=shared_state,
        balance_validator=NativeBalanceValidator(),
        trade_journal=trade_journal,
    )
    # tp_sl_engine injected below after it is created (line ~843)

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

    runtime_state_exporter: NativeRuntimeStateExporter | None = None
    if cfg.runtime_state_path:
        runtime_state_exporter = NativeRuntimeStateExporter(
            shared_state=shared_state,
            output_path=Path(cfg.runtime_state_path),
            interval_sec=cfg.runtime_state_interval_sec,
        )
        await runtime_state_exporter.start()

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
        quote_reserve_ratio=cfg.quote_reserve_ratio,
        quote_min_reserve_usdt=cfg.quote_min_reserve_usdt,
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
        quote_reserve_ratio=cfg.quote_reserve_ratio,
        quote_min_reserve_usdt=cfg.quote_min_reserve_usdt,
        daily_compounding_enabled=cfg.daily_compounding_enabled,
        daily_compounding_state_path=cfg.daily_compounding_state_path,
    )
    # Inject pre-fetched filters so allocator can use real step-sizes immediately
    capital_allocator._symbol_filters_cache = symbol_filters

    signal_fusion = NativeSignalFusion(
        signal_engine=signal_engine,
        market_data=market_data,
        shared_state=shared_state,
        mode_manager=mode_manager,
    )

    arbitration_engine = NativeArbitrationEngine(
        shared_state=shared_state,
        decision_engine=decision_engine,
        signal_fusion=signal_fusion,
        mode_manager=mode_manager,
        ml_forecaster=ml_forecaster,  # gap8: drift-retrain callback
    )
    market_regime_detector = NativeMarketRegimeDetector(
        market_data=market_data,
        shared_state=shared_state,
        signal_engine=signal_engine,
    )

    # Wire regime_detector and market_data into SymbolRotator now that both are available
    if symbol_rotator is not None:
        symbol_rotator._regime_detector = market_regime_detector
        symbol_rotator._market_data = market_data

    # L3 position manager: read-only per-symbol accessor over
    # shared_state. Replaces the compat null-stub for the
    # ``position_manager`` app_ctx key consumed by SituationEngine
    # and DecisionEngine.
    position_manager = NativePositionManager(
        shared_state=shared_state,
        min_order_usdt=cfg.min_order_usdt,
    )

    # L4 TP/SL engine: Volatility-adaptive TP/SL with risk-based sizing
    # Tier 1 features: ATR-based adaptation, risk-based position sizing, auto-arm
    tp_sl_engine_native = NativeTPSLEngine(
        shared_state=shared_state,
        config=cfg,
    )

    # Wire TP/SL engine into fill tracker and executor (late injection — created after executor)
    if fill_tracker is not None and hasattr(fill_tracker, "set_tp_sl_engine"):
        fill_tracker.set_tp_sl_engine(tp_sl_engine_native)
    executor._tp_sl_engine = tp_sl_engine_native

    # Fear & Greed fetcher — fetch once on boot then refresh every hour.
    # Writes shared_state.fear_greed_score (int 0-100) and fear_greed_label (str).
    # All downstream consumers (model trainer, inference filter) read from shared_state.
    fear_greed_fetcher = FearGreedFetcher(
        shared_state=shared_state, exchange_client=exchange_client
    )
    try:
        await fear_greed_fetcher.start()
        # Store ref on shared_state so inference filter can call _btc_confirmed_reversal()
        shared_state._fear_greed_fetcher = fear_greed_fetcher
        logger.info(
            "✅ Fear & Greed Index: %d (%s) — refreshes every 1h",
            fear_greed_fetcher.score or 0,
            fear_greed_fetcher.classification or "unknown",
        )
    except Exception as _fg_err:
        logger.warning("FearGreedFetcher failed to start (non-fatal): %s", _fg_err)

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
    health_monitor = NativeHealthMonitor(
        shared_state=shared_state,
        exchange_client=exchange_client,
        market_data=market_data,
        market_data_ws=market_data_ws,
        watchdog=watchdog,
        mode_manager=mode_manager,
    )
    bounded_cache = NativeBoundedCache()

    # L0-L7 observability enhancements (legacy features ported)
    # (trade_journal already created above, before executor)

    # Prometheus exporter: metrics export for Grafana
    prometheus_exporter: NativePrometheusExporter | None = None
    if cfg.prometheus_export_path:
        prometheus_exporter = NativePrometheusExporter(
            output_file=cfg.prometheus_export_path,
        )

    # L0 Position hydration engine: Reconstructs portfolio state from trade journal on restart
    # Reads all fills, calculates entry prices, restores TP/SL, computes realized/unrealized PnL
    position_hydration_engine = NativePositionHydrationEngine(
        shared_state=shared_state,
        trade_journal=trade_journal,
        exchange_client=exchange_client,
        journal_dir=cfg.trade_journal_dir,
        allow_exchange_fallback=True,
        stale_position_age_sec=3600.0,
        dust_threshold_usdt=1.0,
    )

    # L0 Startup state machine: Enforces strict startup progression
    # BOOTING → HYDRATING → RECONCILING → VALIDATING → READY
    # Blocks BUY decisions until READY state
    startup_state_machine = NativeStartupStateMachine(
        decision_engine=decision_engine,
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
        runtime_state_exporter=runtime_state_exporter,
        portfolio_manager=portfolio_manager,
        capital_allocator=capital_allocator,
        signal_fusion=signal_fusion,
        arbitration_engine=arbitration_engine,
        market_regime_detector=market_regime_detector,
        health_monitor=health_monitor,
        bounded_cache=bounded_cache,
        position_manager=position_manager,
        tp_sl_engine=tp_sl_engine_native,
        safety_order_manager=safety_order_manager,
        recovery_engine=recovery_engine,
        watchdog=watchdog,
        trade_journal=trade_journal,
        prometheus_exporter=prometheus_exporter,
        fill_tracker=fill_tracker,
        position_hydration_engine=position_hydration_engine,
        startup_state_machine=startup_state_machine,
        adaptive_capital_engine=ace,
        polling_coordinator=polling_coordinator,
        objective_feedback_controller=ofc,
        mode_manager=mode_manager,
        symbol_discovery=symbol_discoverer,
        market_data_ws=market_data_ws,
        signal_manager_bridge=signal_manager_bridge,
        ml_forecaster=ml_forecaster,
        symbol_screener=symbol_screener,
        symbol_rotator=symbol_rotator,
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
    runtime_exporter = components.runtime_state_exporter
    if runtime_exporter is not None:
        try:
            await runtime_exporter.stop()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("native shutdown: runtime_state_exporter.stop() raised: %r", e)

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
