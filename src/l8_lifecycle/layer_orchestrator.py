# core/layer_orchestrator.py
# Three-Layer Architecture Orchestrator
# Coordinates Wallet Layer → Portfolio Layer → Strategy Layer cycle

import asyncio
import logging
import time
from typing import Any, Dict, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger("LayerOrchestrator")


class LayerSyncStatus(Enum):
    """Status of layer synchronization."""
    INITIALIZING = "INITIALIZING"
    WALLET_SYNCED = "WALLET_SYNCED"
    PORTFOLIO_UPDATED = "PORTFOLIO_UPDATED"
    STRATEGY_EXECUTED = "STRATEGY_EXECUTED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class LayerOrchestrator:
    """
    Orchestrates the professional three-layer capital accounting cycle.
    
    Architecture:
      1. WALLET LAYER (Layer 1): Sync exchange balances & positions
         → Classify assets (EXTERNAL_POSITION, BOT_POSITION, STABLE, DUST)
         → Output: classified_assets, verified_positions
      
      2. PORTFOLIO LAYER (Layer 2): Ingest wallet output, compute portfolio
         → Update position registry with classifications
         → Compute NAV (Net Asset Value)
         → Detect dust, calculate exposure, concentration
         → Output: portfolio, nav, capital_available, risk_metrics
      
      3. STRATEGY LAYER (Layer 3): Execute trading strategy
         → Consume portfolio state
         → Generate & execute trades (only on BOT_POSITION)
         → Never touch EXTERNAL_POSITION (user holdings)
         → Emit trade results, PnL updates
         → Output: trades_executed, pnl, audit_log
    
    Benefits:
      - Clear separation of concerns (wallet sync / portfolio accounting / trading)
      - Professional-grade contracts between layers
      - Safety validation at each stage
      - Audit trail for compliance
      - Scalable: can parallelize within layers
    
    Cycle Time:
      - Default: ~2 seconds (wallet_sync: 0.3s, portfolio: 0.5s, strategy: 1.2s)
      - Configurable interval per layer
      - Adaptive: scales with market volatility
    """
    
    def __init__(
        self,
        shared_state: Any,
        config: Any,
        wallet_scanner: Optional[Any] = None,
        portfolio_manager: Optional[Any] = None,
        strategy_executor: Optional[Any] = None,
    ):
        self.ss = shared_state
        self.config = config
        self.wallet_scanner = wallet_scanner
        self.portfolio_manager = portfolio_manager
        self.strategy_executor = strategy_executor
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Layer intervals (configurable)
        self.wallet_interval_s = float(
            getattr(config, "LAYER_WALLET_INTERVAL_S", 2.0)
        )
        self.portfolio_interval_s = float(
            getattr(config, "LAYER_PORTFOLIO_INTERVAL_S", 1.0)
        )
        self.strategy_interval_s = float(
            getattr(config, "LAYER_STRATEGY_INTERVAL_S", 2.0)
        )
        
        # Overall cycle metrics
        self.cycle_count = 0
        self.last_sync_status = LayerSyncStatus.INITIALIZING
        self.last_sync_timestamp = 0.0
        self.cycle_times: Dict[str, List[float]] = {
            "wallet": [],
            "portfolio": [],
            "strategy": [],
            "total": [],
        }
        self.max_cycle_history = 100
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Last layer outputs (for debugging)
        self.last_wallet_output: Optional[Dict[str, Any]] = None
        self.last_portfolio_output: Optional[Dict[str, Any]] = None
        self.last_strategy_output: Optional[Dict[str, Any]] = None
        
        self.logger.info(
            f"LayerOrchestrator initialized ("
            f"wallet_interval={self.wallet_interval_s}s, "
            f"portfolio_interval={self.portfolio_interval_s}s, "
            f"strategy_interval={self.strategy_interval_s}s)"
        )
    
    # ===== PUBLIC LIFECYCLE =====
    
    async def start(self):
        """Start the orchestration loop."""
        if self._running:
            self.logger.warning("LayerOrchestrator already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._orchestration_loop())
        self.logger.info("LayerOrchestrator started")
    
    async def stop(self):
        """Stop the orchestration loop gracefully."""
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            except asyncio.CancelledError:
                pass
        self.logger.info("LayerOrchestrator stopped")
    
    # ===== ORCHESTRATION LOOP =====
    
    async def _orchestration_loop(self):
        """Main orchestration loop: cycles through all three layers."""
        self.logger.info("🎯 Starting three-layer orchestration cycle")
        
        while self._running:
            try:
                async with self._lock:
                    await self._run_orchestration_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(
                    f"Orchestration cycle error: {e}",
                    exc_info=True
                )
                self.last_sync_status = LayerSyncStatus.ERROR
            
            await asyncio.sleep(0.1)  # Prevent busy loop
        
        self.logger.info("Orchestration loop stopped")
    
    async def _run_orchestration_cycle(self):
        """
        Execute one complete three-layer cycle.
        
        Sequence:
          1. Wallet Layer: Sync exchange balances & positions
          2. Portfolio Layer: Update position registry, compute NAV
          3. Strategy Layer: Execute trades on BOT_POSITION assets
        """
        cycle_start = time.time()
        self.cycle_count += 1
        
        try:
            # ===== LAYER 1: WALLET SYNCHRONIZATION =====
            self.logger.debug(f"[CYCLE {self.cycle_count}] LAYER 1: Wallet sync starting")
            wallet_start = time.time()
            
            wallet_ok = await self._sync_wallet_layer()
            
            wallet_time = time.time() - wallet_start
            self.cycle_times["wallet"].append(wallet_time)
            if len(self.cycle_times["wallet"]) > self.max_cycle_history:
                self.cycle_times["wallet"].pop(0)
            
            if wallet_ok:
                self.logger.debug(
                    f"[CYCLE {self.cycle_count}] LAYER 1: OK ({wallet_time:.3f}s)"
                )
            else:
                self.logger.warning(
                    f"[CYCLE {self.cycle_count}] LAYER 1: FAILED ({wallet_time:.3f}s)"
                )
                # Continue to portfolio despite wallet failure (use last known state)
            
            # ===== LAYER 2: PORTFOLIO UPDATE =====
            self.logger.debug(f"[CYCLE {self.cycle_count}] LAYER 2: Portfolio update starting")
            portfolio_start = time.time()
            
            portfolio_ok = await self._update_portfolio_layer()
            
            portfolio_time = time.time() - portfolio_start
            self.cycle_times["portfolio"].append(portfolio_time)
            if len(self.cycle_times["portfolio"]) > self.max_cycle_history:
                self.cycle_times["portfolio"].pop(0)
            
            if portfolio_ok:
                self.logger.debug(
                    f"[CYCLE {self.cycle_count}] LAYER 2: OK ({portfolio_time:.3f}s)"
                )
            else:
                self.logger.warning(
                    f"[CYCLE {self.cycle_count}] LAYER 2: FAILED ({portfolio_time:.3f}s)"
                )
            
            # ===== LAYER 3: STRATEGY EXECUTION =====
            self.logger.debug(f"[CYCLE {self.cycle_count}] LAYER 3: Strategy execution starting")
            strategy_start = time.time()
            
            strategy_ok = await self._execute_strategy_layer()
            
            strategy_time = time.time() - strategy_start
            self.cycle_times["strategy"].append(strategy_time)
            if len(self.cycle_times["strategy"]) > self.max_cycle_history:
                self.cycle_times["strategy"].pop(0)
            
            if strategy_ok:
                self.logger.debug(
                    f"[CYCLE {self.cycle_count}] LAYER 3: OK ({strategy_time:.3f}s)"
                )
            else:
                self.logger.debug(
                    f"[CYCLE {self.cycle_count}] LAYER 3: SKIPPED ({strategy_time:.3f}s)"
                )
            
            # ===== CYCLE COMPLETE =====
            cycle_time = time.time() - cycle_start
            self.cycle_times["total"].append(cycle_time)
            if len(self.cycle_times["total"]) > self.max_cycle_history:
                self.cycle_times["total"].pop(0)
            
            avg_cycle_time = sum(self.cycle_times["total"]) / len(self.cycle_times["total"])
            
            self.logger.info(
                f"[CYCLE {self.cycle_count}] COMPLETE "
                f"(wallet={wallet_time:.3f}s, portfolio={portfolio_time:.3f}s, "
                f"strategy={strategy_time:.3f}s, total={cycle_time:.3f}s, "
                f"avg={avg_cycle_time:.3f}s)"
            )
            
            self.last_sync_timestamp = time.time()
            self.last_sync_status = LayerSyncStatus.COMPLETE
        
        except Exception as e:
            self.logger.error(f"[CYCLE {self.cycle_count}] ERROR: {e}", exc_info=True)
            self.last_sync_status = LayerSyncStatus.ERROR
    
    # ===== LAYER IMPLEMENTATIONS =====
    
    async def _sync_wallet_layer(self) -> bool:
        """
        LAYER 1: Synchronize exchange wallet and positions.
        
        Returns: True if wallet sync completed successfully
        """
        try:
            # Trigger wallet scanner if available
            if self.wallet_scanner and hasattr(self.wallet_scanner, "run_once"):
                try:
                    res = self.wallet_scanner.run_once()
                    if asyncio.iscoroutine(res):
                        await asyncio.wait_for(res, timeout=30.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Wallet sync timed out (30s)")
                    return False
                except Exception as e:
                    self.logger.warning(f"Wallet sync failed: {e}")
                    return False
            
            # Mark wallet layer as synced
            if hasattr(self.ss, "balances_ready_event"):
                self.ss.balances_ready_event.set()
            
            self.logger.debug("Wallet layer synchronized")
            return True
        
        except Exception as e:
            self.logger.error(f"Wallet layer sync error: {e}", exc_info=True)
            return False
    
    async def _update_portfolio_layer(self) -> bool:
        """
        LAYER 2: Update portfolio, compute NAV, classify positions.
        
        Returns: True if portfolio update completed successfully
        """
        try:
            # Trigger portfolio manager if available
            if self.portfolio_manager and hasattr(self.portfolio_manager, "compute_nav"):
                try:
                    res = self.portfolio_manager.compute_nav()
                    if asyncio.iscoroutine(res):
                        nav = await asyncio.wait_for(res, timeout=10.0)
                    else:
                        nav = res
                    
                    self.logger.debug(f"Portfolio NAV computed: {nav}")
                except asyncio.TimeoutError:
                    self.logger.warning("Portfolio update timed out (10s)")
                    return False
                except Exception as e:
                    self.logger.debug(f"Portfolio update: {e}")
                    return False
            
            # Mark portfolio layer as ready
            if hasattr(self.ss, "nav_ready_event"):
                self.ss.nav_ready_event.set()
            
            self.logger.debug("Portfolio layer updated")
            return True
        
        except Exception as e:
            self.logger.error(f"Portfolio layer update error: {e}", exc_info=True)
            return False
    
    async def _execute_strategy_layer(self) -> bool:
        """
        LAYER 3: Execute trading strategy.
        
        Returns: True if strategy execution was allowed (doesn't mean trades were made)
        """
        try:
            # Strategy execution is typically handled by agent loop
            # This is a placeholder for explicit orchestration trigger
            
            # Verify NAV is ready before strategy can execute
            if hasattr(self.ss, "nav_ready_event"):
                if not self.ss.nav_ready_event.is_set():
                    self.logger.debug("Skipping strategy: NAV not ready yet")
                    return False
            
            # Mark strategy layer as ready
            if hasattr(self.ss, "ops_plane_ready_event"):
                self.ss.ops_plane_ready_event.set()
            
            self.logger.debug("Strategy layer ready (agents will execute)")
            return True
        
        except Exception as e:
            self.logger.error(f"Strategy layer error: {e}", exc_info=True)
            return False
    
    # ===== DIAGNOSTICS =====
    
    def get_cycle_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for current cycle."""
        return {
            "cycle_count": self.cycle_count,
            "last_sync_status": self.last_sync_status.value,
            "last_sync_timestamp": self.last_sync_timestamp,
            "avg_wallet_time": (
                sum(self.cycle_times["wallet"]) / len(self.cycle_times["wallet"])
                if self.cycle_times["wallet"] else 0.0
            ),
            "avg_portfolio_time": (
                sum(self.cycle_times["portfolio"]) / len(self.cycle_times["portfolio"])
                if self.cycle_times["portfolio"] else 0.0
            ),
            "avg_strategy_time": (
                sum(self.cycle_times["strategy"]) / len(self.cycle_times["strategy"])
                if self.cycle_times["strategy"] else 0.0
            ),
            "avg_total_time": (
                sum(self.cycle_times["total"]) / len(self.cycle_times["total"])
                if self.cycle_times["total"] else 0.0
            ),
        }
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        metrics = self.get_cycle_metrics()
        return (
            f"LayerOrchestrator: "
            f"cycles={metrics['cycle_count']}, "
            f"status={metrics['last_sync_status']}, "
            f"avg_total={metrics['avg_total_time']:.3f}s"
        )


# ============================================================================
# 8-LAYER DETERMINISTIC BOOT SEQUENCER
# (LOGICAL_LAYERED_ARCHITECTURE.md §11.4)
# ============================================================================

from dataclasses import dataclass, field          # noqa: E402
from typing import Callable, Awaitable             # noqa: E402

try:
    from src.l0_core.layer_contracts import (             # noqa: E402
        LayerName,
        L8LifecycleContract,
    )
except Exception:  # pragma: no cover - fallback if import order shifts
    LayerName = None                                # type: ignore
    L8LifecycleContract = None                      # type: ignore


@dataclass
class LayerBootSpec:
    """One step in the deterministic boot sequence."""
    layer: Any                          # core.layer_contracts.LayerName
    name: str
    start: Optional[Callable[[], Awaitable[bool]]] = None
    stop: Optional[Callable[[], Awaitable[None]]] = None
    health: Optional[Callable[[], bool]] = None
    required: bool = True               # if True, boot aborts on failure
    started: bool = field(default=False, init=False)
    last_error: Optional[str] = field(default=None, init=False)


class LayerHealth(Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


class BootSequencer:
    """
    Deterministic 8-layer boot supervisor (L8 component).

    Boot order is fixed by L8LifecycleContract.BOOT_ORDER:
        L0 → L1 → L2 → L3 → L4 → L6 → L5 → L7
    (L6 before L5 so the policy gate exists before any intent is produced.)

    Each layer registers a `LayerBootSpec` with start/stop/health callables.
    BootSequencer:
      - boots layers in order, aborting on first required-layer failure;
      - exposes per-layer restart for the watchdog (L8);
      - performs graceful shutdown in reverse order.

    This is additive: it does NOT replace the runtime LayerOrchestrator above,
    which continues to drive the wallet→portfolio→strategy cycle each tick.
    BootSequencer runs ONCE at process start and on watchdog-triggered restarts.
    """

    # Authoritative order (matches L8LifecycleContract.BOOT_ORDER)
    DEFAULT_ORDER: List[str] = [
        "L0_CROSS_CUTTING",
        "L1_EXCHANGE_IO",
        "L2_WALLET_MARKETDATA",
        "L3_PORTFOLIO_STATE",
        "L4_EXECUTION",
        "L6_GOVERNANCE",
        "L5_STRATEGY",
        "L7_OBSERVABILITY",
    ]

    def __init__(self) -> None:
        self.logger = logging.getLogger("BootSequencer")
        self._specs: Dict[str, LayerBootSpec] = {}
        self._boot_started_at: float = 0.0
        self._boot_completed_at: float = 0.0

    # ----- registration -----

    def register(self, spec: LayerBootSpec) -> None:
        key = spec.layer.value if hasattr(spec.layer, "value") else str(spec.layer)
        if key not in self.DEFAULT_ORDER:
            self.logger.warning(
                f"BootSequencer.register: unknown layer {key}; ignoring"
            )
            return
        self._specs[key] = spec
        self.logger.debug(f"BootSequencer registered {key} ({spec.name})")

    # ----- lifecycle -----

    async def boot(self) -> bool:
        """Run the deterministic boot. Returns True iff all required layers started."""
        self._boot_started_at = time.time()
        self.logger.info("🚀 BootSequencer: starting 8-layer deterministic boot")

        for key in self.DEFAULT_ORDER:
            spec = self._specs.get(key)
            if spec is None:
                self.logger.debug(f"  · {key}: no spec registered (skipped)")
                continue

            t0 = time.time()
            try:
                ok = True
                if spec.start is not None:
                    res = spec.start()
                    if asyncio.iscoroutine(res):
                        ok = bool(await asyncio.wait_for(res, timeout=60.0))
                    else:
                        ok = bool(res)
                spec.started = ok
                dt = time.time() - t0
                if ok:
                    self.logger.info(f"  ✅ {key} ({spec.name}) up in {dt:.2f}s")
                else:
                    spec.last_error = "start() returned False"
                    self.logger.error(f"  ❌ {key} ({spec.name}) failed in {dt:.2f}s")
                    if spec.required:
                        self.logger.error(
                            f"BootSequencer aborting: required layer {key} failed"
                        )
                        return False
            except Exception as e:
                spec.started = False
                spec.last_error = repr(e)
                self.logger.error(
                    f"  ❌ {key} ({spec.name}) raised: {e}", exc_info=True
                )
                if spec.required:
                    return False

        self._boot_completed_at = time.time()
        elapsed = self._boot_completed_at - self._boot_started_at
        self.logger.info(
            f"🎉 BootSequencer: 8-layer boot complete in {elapsed:.2f}s"
        )
        return True

    async def shutdown(self) -> None:
        """Stop layers in reverse order."""
        self.logger.info("🛑 BootSequencer: graceful shutdown")
        for key in reversed(self.DEFAULT_ORDER):
            spec = self._specs.get(key)
            if spec is None or not spec.started or spec.stop is None:
                continue
            try:
                res = spec.stop()
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, timeout=15.0)
                spec.started = False
                self.logger.info(f"  · {key} stopped")
            except Exception as e:
                self.logger.warning(f"  · {key} stop() raised: {e}")

    async def restart_layer(self, layer_key: str) -> bool:
        """Watchdog hook: restart a single layer in place."""
        spec = self._specs.get(layer_key)
        if spec is None:
            self.logger.error(f"restart_layer: unknown layer {layer_key}")
            return False
        self.logger.warning(f"♻️  Restarting layer {layer_key}")
        if spec.stop is not None and spec.started:
            try:
                res = spec.stop()
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, timeout=15.0)
            except Exception as e:
                self.logger.warning(f"restart_layer: stop() raised: {e}")
            spec.started = False
        if spec.start is None:
            return False
        try:
            res = spec.start()
            ok = bool(await asyncio.wait_for(res, timeout=60.0)) \
                if asyncio.iscoroutine(res) else bool(res)
            spec.started = ok
            return ok
        except Exception as e:
            spec.last_error = repr(e)
            self.logger.error(f"restart_layer: start() raised: {e}", exc_info=True)
            return False

    # ----- introspection -----

    def system_health(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key in self.DEFAULT_ORDER:
            spec = self._specs.get(key)
            if spec is None:
                out[key] = LayerHealth.DOWN.value
                continue
            if not spec.started:
                out[key] = LayerHealth.DOWN.value
            elif spec.health is not None:
                try:
                    healthy = bool(spec.health())
                    out[key] = LayerHealth.OK.value if healthy else LayerHealth.DEGRADED.value
                except Exception:
                    out[key] = LayerHealth.DEGRADED.value
            else:
                out[key] = LayerHealth.OK.value
        return out

    def boot_report(self) -> Dict[str, Any]:
        return {
            "boot_order": list(self.DEFAULT_ORDER),
            "started_at": self._boot_started_at,
            "completed_at": self._boot_completed_at,
            "uptime_s": (
                (time.time() - self._boot_completed_at)
                if self._boot_completed_at else 0.0
            ),
            "layer_health": self.system_health(),
            "errors": {
                k: s.last_error for k, s in self._specs.items() if s.last_error
            },
        }

