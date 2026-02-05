import asyncio
import logging
from datetime import datetime, timedelta
import time as _t
from typing import Dict, List, Optional, Any, Set

from core.component_status_logger import log_component_status, ComponentStatusLogger as CSL
from core.stubs import TradeIntent, ExecOrder

AGENT_NAME = "LiquidationAgent"

class LiquidationAgent:
    """
    P9 Canonical LiquidationAgent: The 'Risk Desk + Treasury Desk'.
    
    Mission: Free capital when needed, safely and intelligently.
    Identity: Infrastructure agent (discovery type), background task-driven.
    
    Triggers:
    A) Capital Shortage (Requested via Orchestrator -> propose_liquidations)
    B) Performance-based exit (ROI/Loss thresholds)
    C) Dust/MinNotional Cleanup
    D) Rebalancing/Maintenance
    """
    
    agent_type = "discovery" # Infrastructure agent, background task-driven.

    def __init__(
        self,
        config,
        shared_state,
        market_data_feed=None,
        execution_manager=None,
        tp_sl_engine=None,
        meta_controller=None,
        exchange_client=None,
        name: str = "LiquidationAgent",
        **kwargs,
    ):
        self.name = name
        self.shared_state = shared_state
        self.market_data_feed = market_data_feed
        self.execution_manager = execution_manager
        self.config = config
        self.tp_sl_engine = tp_sl_engine
        self.meta_controller = meta_controller
        self.exchange_client = exchange_client
        
        self.logger = logging.getLogger(self.name)
        self.logger.propagate = False
        
        # Proper status logging: initialize instance with agent's logger
        self.csl = CSL(logger=self.logger)
        
        self.symbols: List[str] = []
        self._last_rebalance_ts = 0.0
        self.cooldown_tracker: Dict[str, float] = {}
        self.intent_sink: List[Any] = []
        self.active_liquidations: Set[str] = set()
        self._bg_tasks: Set[asyncio.Task] = set()
        
        self.csl.log_status(self.name, "Operational", detail="Initialized as Treasury/Risk Desk")

    # -----------------------------
    # Configuration Accessors
    # -----------------------------
    def _cfg(self, key: str, default=None):
        if hasattr(self.config, key):
            return getattr(self.config, key)
        return default

    @property
    def target_free_usdt(self) -> float: return float(self._cfg("CAPITAL_TARGET_FREE_USDT", 10.0))
    @property
    def buffer_mult(self) -> float: return float(self._cfg("LIQUIDATION_BUFFER_MULT", 1.05))
    @property
    def min_edge_bps(self) -> float: return float(self._cfg("LIQ_MIN_EDGE_BPS", 8.0))
    @property
    def est_liq_cost_bps(self) -> float: return float(self._cfg("EST_LIQ_COST_BPS", 12.0))
    @property
    def evaluation_window_hours(self) -> float: return float(self._cfg("LIQ_EVAL_WINDOW_HOURS", 6.0))
    @property
    def loss_threshold(self) -> int: return int(self._cfg("LIQ_LOSS_THRESHOLD", 3))
    @property
    def roi_threshold(self) -> float: return float(self._cfg("LIQ_ROI_THRESHOLD", -0.05))
    @property
    def min_notional_dust_factor(self) -> float: return float(self._cfg("LIQ_DUST_FACTOR", 1.2))

    # -----------------------------
    # Lifecycle
    # -----------------------------
    async def start(self):
        """P9 Start Hook: Launch background loops."""
        self._spawn(self.scheduler(), "liq_main_loop")
        self.logger.info("[%s] Risk Desk active. Loops started.", self.name)

    async def stop(self):
        for t in list(self._bg_tasks):
            t.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()

    def _spawn(self, coro, label: str):
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(lambda t: self._bg_tasks.discard(t))
        return task

    # -----------------------------
    # Symbol Universe
    # -----------------------------
    async def _refresh_universe(self):
        """Ensures the agent sees ALL relevant symbols from SharedState analysis universe."""
        if hasattr(self.shared_state, "get_analysis_symbols"):
            self.symbols = self.shared_state.get_analysis_symbols()
        else:
            self.symbols = list(getattr(self.shared_state, "symbols", {}).keys())
        
    # -----------------------------
    # Core Scheduler
    # -----------------------------
    async def scheduler(self):
        """Main loop managing internal hygiene (Triggers B, C)."""
        interval = float(self._cfg("LIQ_SCHED_INTERVAL_SEC", 30))
        while True:
            try:
                await self._refresh_universe()
                # Trigger B & C: Internal Hygiene (Performance & Dust)
                await self._process_internal_hygiene()
                self.csl.set_status(self.name, "Operational", f"Risk desk heartbeating | Universe: {len(self.symbols)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Liquidation scheduler error: %s", e, exc_info=True)
            await asyncio.sleep(interval)

    async def _process_internal_hygiene(self):
        """Trigger B (Performance) & Trigger C (Dust)."""
        positions = self.shared_state.get_positions_snapshot() or {}
        for sym, pos in positions.items():
            if sym in self.active_liquidations: continue
            qty = float(pos.get("quantity", 0.0))
            if qty <= 0: continue
            price = float(await self.shared_state.safe_price(sym, default=0.0))
            if price <= 0: continue
            
            # TRIGGER C: Dust Cleanup
            is_dust = await self._check_dust(sym, qty, price)
            if is_dust:
                await self._liquidate_symbol(sym, reason="Trigger C: Dust Cleanup", tag="liquidation_dust")
                continue

            # TRIGGER B: Performance
            is_bad = await self._check_performance(sym, pos)
            if is_bad:
                await self._liquidate_symbol(sym, reason="Trigger B: Poor Performance", tag="liquidation_performance")

    async def _check_dust(self, symbol: str, qty: float, price: float) -> bool:
        filters = await self.shared_state.get_symbol_filters_cached(symbol) or {}
        min_notional = float(filters.get("minNotional") or filters.get("min_notional") or 10.0)
        notional = qty * price
        return 0 < notional < (min_notional * self.min_notional_dust_factor)

    async def _check_performance(self, symbol: str, pos: Dict) -> bool:
        agent_scores = getattr(self.shared_state, "agent_scores", {}).get(symbol, {})
        for agent, metrics in agent_scores.items():
            losses = metrics.get("consecutive_losses", 0)
            roi = metrics.get("roi", 0.0)
            if losses >= self.loss_threshold or roi < self.roi_threshold:
                return True
        return False

    # -----------------------------
    # Planning API (Decision Bridge)
    # -----------------------------
    async def build_plan(self, target_symbol: str = None, needed_quote: float = 0.0, opp_meta: Dict = None, force: bool = False) -> Dict:
        """
        The decision engine called by LiquidationOrchestrator.
        Either plans a specific symbol exit or finding assets to meet 'needed_quote'.
        """
        self.logger.info("Building liquidation plan (target=%s, needed=%.2f, force=%s)", target_symbol, needed_quote, force)
        
        intents = []
        if target_symbol:
            qty = await self.shared_state.get_position_quantity(target_symbol)
            if qty > 0:
                intents.append(self._create_intent(target_symbol, qty, reason=opp_meta.get("reason", "manual"), force=force))
        
        elif needed_quote > 0:
            freed = 0.0
            positions = self.shared_state.get_positions_snapshot() or {}
            candidates = []
            for sym, pos in positions.items():
                qty = float(pos.get("quantity", 0.0))
                if qty <= 0: continue
                price = float(await self.shared_state.safe_price(sym, default=0.0))
                if price <= 0: continue
                roi = float(pos.get("roi", 0.0))
                candidates.append({"symbol": sym, "qty": qty, "value": qty * price, "roi": roi})
            
            candidates.sort(key=lambda x: x["roi"]) # Worst ROI first
            for cand in candidates:
                if freed >= needed_quote: break
                intents.append(self._create_intent(cand["symbol"], cand["qty"], reason=f"Free capital: {opp_meta.get('reason','gap')}", force=force))
                freed += cand["value"]

        if intents:
            self.intent_sink.extend(intents)
            return {"status": "APPROVED", "count": len(intents)}
        return {"status": "NOOP", "count": 0}

    async def propose_liquidations(self, gap_usdt: float, reason: str, force: bool = False) -> List[Dict]:
        """Back-compat API for Orchestrator. Returns serialized intents."""
        res = await self.build_plan(needed_quote=gap_usdt, opp_meta={"reason": reason}, force=force)
        if res["status"] == "APPROVED":
            # Return current sink for Orchestrator to emit
            out = [it.__dict__ if hasattr(it, "__dict__") else it for it in self.intent_sink]
            self.intent_sink.clear()
            return out
        return []

    async def produce_orders(self) -> List[Dict]:
        """Shim for Orchestrator drain."""
        out = []
        for it in self.intent_sink:
            # Convert intent to order-like dict for legacy Orchestrator drain
            out.append({
                "symbol": it.symbol,
                "side": it.action,
                "quantity": it.qty_hint,
                "tag": it.tag
            })
        self.intent_sink.clear()
        return out

    # -----------------------------
    # Execution Logic
    # -----------------------------
    def _create_intent(self, symbol: str, qty: float, reason: str, force: bool = False) -> TradeIntent:
        return TradeIntent(
            symbol=symbol,
            side="SELL",
            confidence=0.99 if force else 0.85,
            agent=self.name,
            qty_hint=qty,
            ttl_sec=300,
            tag="liquidation_force" if force else "liquidation",
            rationale=reason
        )

    async def request_liquidation(self, symbol: str, side: str = "SELL", qty: float = 0.0, reason: str = "External request", **kwargs):
        """Standardized interface for external agents (like WalletScanner) to request an exit."""
        self.logger.info("External Liquidation Request: %s | Reason: %s", symbol, reason)
        await self._liquidate_symbol(symbol, reason=reason)

    async def _liquidate_symbol(self, symbol: str, reason: str, tag: str = "liquidation"):
        """Internal hygiene exit: emits directly."""
        try:
            self.active_liquidations.add(symbol)
            qty = await self.shared_state.get_position_quantity(symbol)
            if qty <= 0: return

            # P9 STABILITY FIX: Prevent loop on "Hard Dust" (Value < MinNotional)
            try:
                price = float(await self.shared_state.safe_price(symbol, default=0.0))
                filters = await self.shared_state.get_symbol_filters_cached(symbol) or {}
                min_notional = float(filters.get("minNotional") or filters.get("min_notional") or 10.0)
                
                # If value is too small to trade, do NOT emit a signal (it will just fail and loop)
                val = qty * price
                if val < min_notional:
                    # Log once every N times or debug only to avoid noise
                    self.logger.debug(f"[{self.name}] Ignoring Hard Dust {symbol}: Val {val:.2f} < Min {min_notional}")
                    return
            except Exception as e:
                self.logger.warning(f"[{self.name}] Error checking dust executability: {e}")


            # Mandatory P9 Signal Contract: Emit to Signal Bus
            if hasattr(self.shared_state, "add_agent_signal"):
                try:
                    await self.shared_state.add_agent_signal(
                        symbol=symbol,
                        agent=self.name,
                        side="SELL",
                        confidence=0.95,  # High confidence for liquidations
                        ttl_sec=300,
                        tier="A",
                        rationale=reason
                    )
                except Exception as e:
                    self.logger.warning(f"[{self.name}] Failed to emit to signal bus: {e}")

            # Backward compatibility / specific intent flow
            intent = self._create_intent(symbol, qty, reason, force=False)
            if hasattr(self.shared_state, "emit_event"):
                await self.shared_state.emit_event("TradeIntent", intent.__dict__)
            self.logger.info("HYGIENE PROPOSED EXIT: %s | Reason: %s", symbol, reason)
        finally:
            self.active_liquidations.discard(symbol)

    async def run(self, symbol): pass
    def health(self): return {"status": "Operational", "universe_size": len(self.symbols)}
