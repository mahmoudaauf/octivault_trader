from __future__ import annotations

"""
RecoveryEngine (P9‑compliant, Institutional Architecture)

Purpose
-------
Self‑heal boot path that rebuilds in‑memory state after a crash/restart and re‑establishes
phase readiness before runtime loops start. It DOES NOT place orders.

Institutional Architecture (WebSocket-First)
---------------------------------------------
Uses 3-strategy cascade to minimize API calls and maximize reliability:
  1. SharedState Memory (WebSocket-updated, <1ms latency, zero API calls)
  2. Database Snapshot (persisted, no API call)
  3. REST API (only if RECOVERY_ALLOW_REST=True configured, minimizes calls)

This ensures:
  ✅ Zero polling (event-driven updates from WebSocket)
  ✅ Minimal REST usage (only for order placement, rare sync checks)
  ✅ Instant data access (memory-resident prices/balances/positions)
  ✅ Startup recovery without API calls (use DB snapshot)

Conformance (Spec 2025‑08‑20)
-----------------------------
- Methods: rebuild_state(), verify_integrity()
- Emits HealthStatus events for start/success/failure
- Restores balances, positions, (optionally) filters and prices snapshot
- Recomputes unrealized PnL/NAV (pnl_calculator if available, else lightweight calc)
- Sets phase gates: AcceptedSymbolsReady and MarketDataReady (when conditions allow)
- Honors RECOVERY.VERIFY_INTEGRITY flag
- Uses 3-strategy cascade: SharedState → Database → REST (configured only)
- Never places orders (ExecutionManager remains the single order path)

Configuration
--------------
RECOVERY_ALLOW_REST: bool (default: False)
  - False: Institutional mode (use memory + DB only, zero REST calls)
  - True: Allow REST as fallback (for setup/debugging)

Integrations
------------
- shared_state: authoritative memory (balances, positions, accepted_symbols, metrics, events)
- sstools: event emitter & helpers (emit_event, nav_quote)
- exchange_client: read‑only price/balance/position queries (REST fallback only)
- database_manager: load_last_snapshot() for persisted recovery (no API call)
- pnl_calculator (optional): for unrealized PnL computation
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

# ------------------------------- timeouts -----------------------------------
async def _with_timeout(coro, sec: float = 8.0):
    try:
        return await asyncio.wait_for(_maybe_await(coro), timeout=sec)
    except asyncio.TimeoutError:
        return None

# ------------------------------ normalizers ---------------------------------
def _normalize_balances(raw) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if raw is None:
        return out
    if isinstance(raw, dict):
        for a, v in raw.items():
            if isinstance(v, dict):
                free = float(v.get("free", 0) or 0)
                locked = float(v.get("locked", 0) or 0)
            else:
                free = float(v or 0)
                locked = 0.0
            out[str(a).upper()] = {"free": free, "locked": locked, "total": free + locked}
    elif isinstance(raw, list):
        for e in raw:
            a = str(e.get("asset") or e.get("currency") or e.get("symbol") or "").upper()
            if not a:
                continue
            free = float(e.get("free", 0) or e.get("available", 0) or 0)
            locked = float(e.get("locked", 0) or e.get("hold", 0) or 0)
            out[a] = {"free": free, "locked": locked, "total": free + locked}
    return out

def _normalize_positions(raw) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {
            str(k).upper().replace("/", ""): v
            for k, v in raw.items()
        }
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for p in raw:
            sym = (p.get("symbol") or p.get("asset") or "").upper().replace("/", "")
            if not sym:
                continue
            qty = float(p.get("current_qty") or p.get("qty") or p.get("positionAmt") or p.get("size") or 0)
            side = (p.get("side") or ("SHORT" if qty < 0 else "LONG")).upper()
            entry = float(p.get("entry_price") or p.get("entryPrice") or 0)
            mult  = float(p.get("contract_multiplier") or p.get("multiplier") or 1.0)
            out[sym] = {"side": side, "current_qty": qty, "entry_price": entry, "contract_multiplier": mult}
    return out


# --------------------------- dataclasses & helpers ---------------------------
@dataclass
class RecoveryConfig:
    verify_integrity: bool = True
    # how many recent symbols/prices we consider sufficient to mark MarketDataReady
    min_symbols_for_md_ready: int = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------- main engine --------------------------------
class RecoveryEngine:

    async def load_snapshot(self):
        """Load SharedState from database snapshot."""
        if not hasattr(self, "db") or not hasattr(self, "ss"):
            raise RuntimeError("RecoveryEngine missing required dependencies")
        snapshot = await self.db.load_shared_state_snapshot()
        await self.ss.rebuild_from_snapshot(snapshot)
        self.logger.info("✅ RecoveryEngine: SharedState successfully rehydrated from snapshot.")

    component_name = "RecoveryEngine"

    def __init__(
        self,
        config: Any,
        shared_state: Any,
        exchange_client: Any,
        database_manager: Any,
        sstools: Optional[Any] = None,
        pnl_calculator: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        *,
        tools: Optional[Any] = None,
        **kwargs,
    ) -> None:
        # Accept both legacy (sstools=...) and new (tools=...) call styles; ignore extra kwargs
        self.config = config
        self.ss = shared_state
        self.ex = exchange_client
        self.db = database_manager
        self.sstools = sstools or tools
        self.tools = tools or sstools
        self.pnl_calc = pnl_calculator
        self.logger = logger or logging.getLogger(self.component_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

        self.rcfg = RecoveryConfig(
            verify_integrity=bool(config.get("VERIFY_INTEGRITY", True) if hasattr(config, 'get') else getattr(config, 'VERIFY_INTEGRITY', True)),
            min_symbols_for_md_ready=int(config.get("MIN_SYMBOLS_FOR_MD_READY", 1) if hasattr(config, 'get') else getattr(config, 'MIN_SYMBOLS_FOR_MD_READY', 1)),
        )

    async def run(self, symbols: Optional[List[str]] = None):
        """
        Main entry point for background recovery/monitoring.
        For now, runs one rebuild pass and exits, unless a loop is desired.
        """
        try:
             self.logger.info("RecoveryEngine run() called.")
             if hasattr(self, "rebuild_state"):
                 await self.rebuild_state()
             else:
                 self.logger.warning("rebuild_state method missing in RecoveryEngine.")
        except Exception as e:
             self.logger.error(f"Recovery run failed: {e}", exc_info=True)

    # --------------------------- telemetry / events ---------------------------
    async def _emit_health(self, status: str, message: str) -> None:
        try:
            payload = {
                "component": self.component_name,
                "status": status,
                "message": message,
                "timestamp": _now_iso(),
            }
            if self.sstools and hasattr(self.sstools, "emit_event"):
                await _maybe_await(self.sstools.emit_event("HealthStatus", payload))
            elif hasattr(self.ss, "emit_event"):
                await _maybe_await(self.ss.emit_event("HealthStatus", payload))
        except Exception:
            # Don’t break recovery due to telemetry issues
            self.logger.debug("HealthStatus emit failed", exc_info=True)

    async def _emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            if self.sstools and hasattr(self.sstools, "emit_event"):
                await _maybe_await(self.sstools.emit_event(name, payload or {"timestamp": _now_iso()}))
            elif hasattr(self.ss, "emit_event"):
                await _maybe_await(self.ss.emit_event(name, payload or {"timestamp": _now_iso()}))
        except Exception:
            self.logger.debug("Event emit failed: %s", name, exc_info=True)

    async def _set_readiness(self, key: str, value: bool) -> None:
        """Update readiness flags; set both PascalCase and snake_case for compatibility."""
        try:
            keys = {key}
            if "_" in key:
                pascal = "".join(part.capitalize() for part in key.split("_"))
                keys.add(pascal)
            else:
                snake = "".join(["_"+c.lower() if c.isupper() else c for c in key]).lstrip("_")
                keys.add(snake)
            for k in keys:
                if hasattr(self.ss, "set_readiness_flag"):
                    await _maybe_await(self.ss.set_readiness_flag(k, bool(value)))
                elif hasattr(self.ss, "update_readiness"):
                    await _maybe_await(self.ss.update_readiness({k: bool(value)}))
        except Exception:
            self.logger.debug(f"_set_readiness({key}) failed", exc_info=True)

    # ------------------------------- loaders ---------------------------------
    def _load_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            if self.db and hasattr(self.db, "load_last_snapshot"):
                snap = self.db.load_last_snapshot()
                if snap:
                    self.logger.info("[Recovery] Loaded snapshot from DB")
                    return snap
        except Exception:
            self.logger.warning("[Recovery] Failed loading snapshot from DB", exc_info=True)
        return None

    async def _load_live(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch balances and positions using 3-strategy cascade (institutional architecture).
        
        Strategy 1: Try SharedState memory (WebSocket-updated, zero latency)
        Strategy 2: Fall back to DB snapshot (persisted, no API call)
        Strategy 3: REST API only if explicitly configured (minimize API calls)
        
        Returns (balances, positions_by_symbol) with light normalization.
        """
        balances: Dict[str, Any] = {}
        positions: Dict[str, Any] = {}
        
        # ========== STRATEGY 1: SharedState Memory (WebSocket-updated) ==========
        try:
            # Check if SharedState has recent balances from WebSocket
            if hasattr(self.ss, "balances") and self.ss.balances:
                memory_balances = self.ss.balances
                if isinstance(memory_balances, dict) and memory_balances:
                    balances = _normalize_balances(memory_balances)
                    self.logger.info("[Recovery] ✅ Strategy 1: Balances loaded from SharedState memory (WebSocket-updated)")
        except Exception:
            self.logger.debug("[Recovery] Strategy 1 (SharedState balances) failed", exc_info=True)
        
        # Check if SharedState has recent positions from WebSocket
        if not positions:
            try:
                if hasattr(self.ss, "positions") and self.ss.positions:
                    memory_positions = self.ss.positions
                    if isinstance(memory_positions, dict) and memory_positions:
                        positions = _normalize_positions(memory_positions)
                        self.logger.info("[Recovery] ✅ Strategy 1: Positions loaded from SharedState memory (WebSocket-updated)")
            except Exception:
                self.logger.debug("[Recovery] Strategy 1 (SharedState positions) failed", exc_info=True)
        
        # ========== STRATEGY 2: Database Snapshot (persisted, no API call) ==========
        if not balances:
            try:
                snapshot = self._load_snapshot()
                if snapshot:
                    snapshot_balances = snapshot.get("balances")
                    if snapshot_balances:
                        balances = _normalize_balances(snapshot_balances)
                        self.logger.info("[Recovery] ✅ Strategy 2: Balances loaded from DB snapshot")
            except Exception:
                self.logger.debug("[Recovery] Strategy 2 (snapshot balances) failed", exc_info=True)
        
        if not positions:
            try:
                snapshot = self._load_snapshot()
                if snapshot:
                    snapshot_positions = snapshot.get("positions")
                    if snapshot_positions:
                        positions = _normalize_positions(snapshot_positions)
                        self.logger.info("[Recovery] ✅ Strategy 2: Positions loaded from DB snapshot")
            except Exception:
                self.logger.debug("[Recovery] Strategy 2 (snapshot positions) failed", exc_info=True)
        
        # ========== STRATEGY 3: REST API (only if configured, minimizes API calls) ==========
        allow_rest = getattr(self.config, "RECOVERY_ALLOW_REST", False)
        if allow_rest:
            # Balances via REST (only if configured)
            if not balances:
                try:
                    if hasattr(self.ex, "get_balances"):
                        b = await _with_timeout(self.ex.get_balances())
                        balances = _normalize_balances(b)
                        self.logger.info("[Recovery] ✅ Strategy 3: Balances loaded via REST API (configured)")
                except Exception:
                    self.logger.warning("[Recovery] Strategy 3 (REST balances) failed", exc_info=True)
            
            # Positions via REST (only if configured)
            if not positions:
                try:
                    getter = None
                    if hasattr(self.ex, "get_open_positions"):
                        getter = self.ex.get_open_positions
                    elif hasattr(self.ex, "get_positions"):
                        getter = self.ex.get_positions
                    if getter:
                        p = await _with_timeout(getter())
                        positions = _normalize_positions(p)
                        self.logger.info("[Recovery] ✅ Strategy 3: Positions loaded via REST API (configured)")
                except Exception:
                    self.logger.warning("[Recovery] Strategy 3 (REST positions) failed", exc_info=True)
        else:
            # REST not configured - log that we're not using it
            if not balances or not positions:
                self.logger.info("[Recovery] ⚠️  RECOVERY_ALLOW_REST=False (institutional mode): Skipping REST fallback, using memory+DB only")
        
        return balances, positions

    # ------------------------------ calculators ------------------------------
    async def _recompute_unrealized(self, positions: Dict[str, Any]) -> float:
        """Compute unrealized PnL using provided pnl_calculator or a simple fallback.
        Returns total unrealized PnL in quote currency (float).
        """
        try:
            if self.pnl_calc and hasattr(self.pnl_calc, "compute_unrealized"):
                # expect: compute_unrealized(positions, prices) -> float
                prices = await self._latest_prices(list(positions.keys()))
                return float(await _maybe_await(self.pnl_calc.compute_unrealized(positions, prices)))
        except Exception:
            self.logger.debug("[Recovery] pnl_calc path failed; using fallback", exc_info=True)

        # fallback: side-aware computation with contract multiplier
        unreal = 0.0
        try:
            prices = await self._latest_prices(list(positions.keys()))
            for sym, pos in positions.items():
                qty = float(pos.get("current_qty") or pos.get("qty") or 0.0)
                if not qty:
                    continue
                entry = float(pos.get("entry_price") or 0.0)
                px = float(prices.get(sym, 0.0))
                if px <= 0 or entry <= 0:
                    continue
                side = (pos.get("side") or ("SHORT" if qty < 0 else "LONG")).upper()
                eff_qty = abs(qty)
                mult = float(pos.get("contract_multiplier") or 1.0)
                if side == "LONG":
                    unreal += (px - entry) * eff_qty * mult
                else:  # SHORT
                    unreal += (entry - px) * eff_qty * mult
        except Exception:
            self.logger.debug("[Recovery] fallback unrealized failed", exc_info=True)
        return float(unreal)

    async def _latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch latest prices using 3-strategy cascade (institutional architecture).
        
        Strategy 1: SharedState memory (WebSocket prices, zero latency, zero API calls)
        Strategy 2: Database snapshot (persisted prices, no API call)
        Strategy 3: REST API (only if explicitly configured, minimizes API calls)
        
        Returns {symbol: price, ...} dict with prices for requested symbols.
        """
        out: Dict[str, float] = {}
        if not symbols:
            return out
        
        async def fetch_sym_cascading(s: str) -> Tuple[str, Optional[float]]:
            """Fetch price for symbol using 3-strategy cascade."""
            try:
                px = None
                
                # ===== STRATEGY 1: SharedState memory (WebSocket-updated prices) =====
                if hasattr(self.ss, "prices") and self.ss.prices:
                    px = self.ss.prices.get(s.upper())
                    if px:
                        return s, float(px)
                
                # ===== STRATEGY 2: sstools safe_price (may use snapshot) =====
                if self.sstools and hasattr(self.sstools, "safe_price"):
                    px = await _maybe_await(self.sstools.safe_price(s))
                    if px:
                        return s, float(px)
                
                # ===== STRATEGY 3: Database snapshot (persisted prices) =====
                if not px:
                    try:
                        snapshot = self._load_snapshot()
                        if snapshot and "prices" in snapshot:
                            snapshot_prices = snapshot.get("prices", {})
                            px = snapshot_prices.get(s.upper())
                            if px:
                                return s, float(px)
                    except Exception:
                        pass
                
                # ===== STRATEGY 4: REST API (only if explicitly configured) =====
                allow_rest = getattr(self.config, "RECOVERY_ALLOW_REST", False)
                if allow_rest and not px:
                    if hasattr(self.ex, "get_current_price"):
                        px = await _with_timeout(self.ex.get_current_price(s))
                        if px:
                            return s, float(px)
                
                return s, None
            except Exception as e:
                self.logger.debug(f"[Recovery] Price fetch for {s} failed: {e}")
                return s, None
        
        # Fetch all prices in parallel
        pairs = await asyncio.gather(*[fetch_sym_cascading(s) for s in symbols], return_exceptions=False)
        for s, px in pairs:
            if px:
                out[s] = px
        
        # Log which symbols we couldn't find
        missing = [s for s in symbols if s not in out]
        if missing:
            self.logger.debug(f"[Recovery] Could not get prices for: {missing}")
        
        return out

    # ------------------------------- integrity -------------------------------
    async def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Lightweight integrity verification. Returns (ok, problems)."""
        problems: List[str] = []
        try:
            # accepted symbols
            symbols = getattr(self.ss, "accepted_symbols", None) or {}
            if not symbols:
                problems.append("accepted_symbols empty")

            # balances
            bal = None
            if hasattr(self.ss, "get_balance_snapshot"):
                bal = await _maybe_await(self.ss.get_balance_snapshot())
            else:
                bal = getattr(self.ss, "balances", None)
            if not isinstance(bal, dict) or not bal:
                problems.append("balances missing")

            # NAV sanity
            nav = None
            if hasattr(self.ss, "get_nav_quote"):
                try:
                    nav = float(await _maybe_await(self.ss.get_nav_quote()))
                except Exception:
                    nav = None
            if nav is None and self.sstools and hasattr(self.sstools, "nav_quote"):
                try:
                    nav = float(await _maybe_await(self.sstools.nav_quote()))
                except Exception:
                    nav = None
            if nav is None or nav <= 0:
                problems.append("NAV not computed or non‑positive")
        except Exception:
            self.logger.debug("[Recovery] verify_integrity crashed", exc_info=True)
            problems.append("integrity check exception")

        return (len(problems) == 0, problems)

    # ------------------------------- applying --------------------------------
    async def _apply_balances(self, balances: Dict[str, Any]) -> None:
        try:
            normalized = _normalize_balances(balances)
            if normalized and hasattr(self.ss, "update_balances"):
                await _maybe_await(self.ss.update_balances(normalized))
            elif normalized is not None:
                setattr(self.ss, "balances", normalized)

            nonzero = normalized and any(float(v.get("total", 0) or v.get("free", 0) or 0) > 0 for v in normalized.values())
            if nonzero:
                await self._set_readiness("BalancesReady", True)
                await self._emit_event("BalancesReady", {"timestamp": _now_iso(), "source": "recovery"})
            else:
                self.logger.info("[Recovery] Balances empty/zero; not marking BalancesReady")

            try:
                quote   = getattr(self.ss, "quote_asset", "USDT").upper()
                free_usdt = float(normalized.get(quote, {}).get("free", 0.0))
                target_free_usdt = 0.0
                cap = getattr(self.config, "capital", None) or getattr(self.config, "CAPITAL", None)
                if cap:
                    target_free_usdt = float(getattr(cap, "target_free_usdt", getattr(cap, "TARGET_FREE_USDT", 0.0)) or 0.0)
                if not target_free_usdt:
                    ss_cfg = getattr(self.ss, "config", None)
                    if ss_cfg and hasattr(ss_cfg, "target_free_usdt"):
                        target_free_usdt = float(getattr(ss_cfg, "target_free_usdt") or 0.0)
                rec = getattr(self.config, "RECOVERY", None)
                if not target_free_usdt and rec and "TARGET_FREE_USDT" in rec:
                    target_free_usdt = float(rec["TARGET_FREE_USDT"] or 0.0)

                ready = free_usdt >= target_free_usdt
                await self._set_readiness("FreeUSDTReady", ready)
                await self._emit_event("FreeUSDTReady", {
                    "timestamp": _now_iso(),
                    "source": "recovery",
                    "free_usdt": free_usdt,
                    "target_free_usdt": target_free_usdt,
                    "ready": ready
                })
            except Exception:
                self.logger.debug("[Recovery] FreeUSDTReady check failed", exc_info=True)
        except Exception:
            self.logger.warning("[Recovery] Failed applying balances", exc_info=True)

    async def _apply_positions(self, positions: Dict[str, Any]) -> None:
        if not positions:
            return
        try:
            if hasattr(self.ss, "update_position"):
                for sym, pos in positions.items():
                    await _maybe_await(self.ss.update_position(sym, pos))
            else:
                # best‑effort legacy path
                sspos = getattr(self.ss, "positions", None)
                if isinstance(sspos, dict):
                    sspos.update(positions)
                else:
                    setattr(self.ss, "positions", positions)
        except Exception:
            self.logger.warning("[Recovery] Failed applying positions", exc_info=True)

    async def _apply_symbols_if_missing(self, candidate_symbols: List[str]) -> None:
        try:
            current = getattr(self.ss, "accepted_symbols", None)
            if not current and candidate_symbols:
                m = {s.upper().replace("/", ""): {"source": "recovery"} for s in candidate_symbols if s}
                if hasattr(self.ss, "set_accepted_symbols"):
                    await _maybe_await(self.ss.set_accepted_symbols(m))
                else:
                    setattr(self.ss, "accepted_symbols", m)
                await self._emit_event("AcceptedSymbolsReady", {"count": len(m), "timestamp": _now_iso(), "source": "recovery"})
                await self._set_readiness("AcceptedSymbolsReady", True)
        except Exception:
            self.logger.debug("[Recovery] apply_symbols_if_missing failed", exc_info=True)

    async def _maybe_mark_market_data_ready(self) -> None:
        """
        Mark MarketDataReady only if we have prices for at least N accepted symbols.
        Uses institutional 3-strategy cascade: memory → DB → REST (configured).
        Logs why if not ready.
        """
        try:
            symbols_map = getattr(self.ss, "accepted_symbols", None) or {}
            syms = list(symbols_map.keys())
            if not syms:
                self.logger.info("[Recovery] No accepted symbols available for MarketDataReady check")
                return
            want = max(1, self.rcfg.min_symbols_for_md_ready)
            have = 0
            for s in syms:
                px = None
                
                # ===== STRATEGY 1: SharedState memory (WebSocket-updated) =====
                if hasattr(self.ss, "prices") and self.ss.prices:
                    px = self.ss.prices.get(s.upper())
                
                # ===== STRATEGY 2: sstools safe_price =====
                if not px and self.sstools and hasattr(self.sstools, "safe_price"):
                    px = await _maybe_await(self.sstools.safe_price(s))
                
                # ===== STRATEGY 3: REST API (only if configured) =====
                if not px:
                    allow_rest = getattr(self.config, "RECOVERY_ALLOW_REST", False)
                    if allow_rest and hasattr(self.ex, "get_current_price"):
                        px = await _with_timeout(self.ex.get_current_price(s))
                
                if px:
                    have += 1
                if have >= want:
                    await self._emit_event("MarketDataReady", {"timestamp": _now_iso(), "source": "recovery", "count": have})
                    await self._set_readiness("MarketDataReady", True)
                    return
            self.logger.info("[Recovery] MarketDataReady not set (have_prices=%d, need=%d, total_symbols=%d)", have, want, len(syms))
        except Exception:
            self.logger.debug("[Recovery] mark MarketDataReady failed", exc_info=True)

    # ------------------------------- public API -------------------------------
    async def rebuild_state(self) -> Dict[str, Any]:
        """Rebuild state from DB snapshot or live queries; recompute NAV/unrealized; set phase gates."""
        await self._emit_health("Starting", "Recovery begin")
        try:
            if hasattr(self.ss, "update_timestamp"):
                await _maybe_await(self.ss.update_timestamp(self.component_name))
        except Exception:
            pass

        # P9 Phase 4: Hydrate pending position accumulation intents
        if hasattr(self.ss, "load_pending_intents_from_db"):
            await self.ss.load_pending_intents_from_db()

        snapshot = self._load_snapshot()
        balances: Dict[str, Any] = {}
        positions: Dict[str, Any] = {}
        accepted_symbols: List[str] = []

        if snapshot:
            balances = snapshot.get("balances") or {}
            positions = snapshot.get("positions") or {}
            accepted_symbols = list((snapshot.get("accepted_symbols") or {}).keys())
            await self._apply_balances(balances)
            await self._apply_positions(positions)
            await self._apply_symbols_if_missing(accepted_symbols)
        else:
            live_bal, live_pos = await self._load_live()
            balances = live_bal or {}
            positions = live_pos or {}
            accepted_symbols = list(positions.keys()) if positions else []
            await self._apply_balances(balances)
            await self._apply_positions(positions)
            await self._apply_symbols_if_missing(accepted_symbols)

        # Recompute unrealized & NAV best‑effort
        try:
            unreal = await self._recompute_unrealized(positions)
            # store into SharedState metrics if structure exists
            metrics = getattr(self.ss, "metrics", None)
            if isinstance(metrics, dict):
                metrics["unrealized_pnl_quote"] = float(unreal)
        except Exception:
            self.logger.debug("[Recovery] unrealized calculation failed", exc_info=True)

        # Try to mark MarketDataReady if reasonable
        await self._maybe_mark_market_data_ready()

        # Integrity verification (optional)
        if self.rcfg.verify_integrity:
            ok, problems = await self.verify_integrity()
            if not ok:
                msg = f"Integrity issues: {problems}"
                await self._emit_health("Error", msg)
                # don't raise to keep boot progressing; caller may decide policy
                self.logger.warning("[Recovery] %s", msg)
            else:
                await self._emit_health("Running", "Recovery completed; integrity OK")
                try:
                    if hasattr(self.ss, "update_timestamp"):
                        await _maybe_await(self.ss.update_timestamp(self.component_name))
                except Exception:
                    pass
        else:
            await self._emit_health("Running", "Recovery completed (integrity skipped)")
            try:
                if hasattr(self.ss, "update_timestamp"):
                    await _maybe_await(self.ss.update_timestamp(self.component_name))
            except Exception:
                pass

        return {
            "balances": balances,
            "positions": positions,
            "accepted_symbols": accepted_symbols,
        }


# ------------------------------- async helper -------------------------------
async def _maybe_await(maybe):
    if asyncio.iscoroutine(maybe):
        return await maybe
    return maybe
