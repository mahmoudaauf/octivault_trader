from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryPositionRecord:
    symbol: str
    asset: str
    qty: float
    current_price: float
    notional_usdt: float
    avg_entry_price: float = 0.0
    entry_price_confidence: str = "UNKNOWN"
    entry_source: str = "UNKNOWN"
    entry_time: float = 0.0
    unrealized_pnl_usdt: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    position_age_sec: Optional[float] = None
    status: str = "ENTRY_UNKNOWN"
    suggested_action: str = "WAIT"
    reason: str = ""
    sellable: bool = False
    needs_top_up: bool = False
    top_up_qty: float = 0.0


class PortfolioRecoveryEngine:
    def __init__(
        self,
        *,
        shared_state: Any,
        exchange_client: Any | None = None,
        trade_journal_dir: str = "logs",
        reserve_floor_ratio: float = 0.20,
        critical_free_ratio: float = 0.10,
        dust_ratio_threshold: float = 0.03,
        stale_position_hours: float = 48.0,
        min_recovery_notional_usdt: float = 10.0,
        allow_unknown_entry_recovery_sell: bool = True,
        max_recovery_sells_per_cycle: int = 1,
        recovery_decision_interval_sec: float = 60.0,
        buy_block_when_recovery_active: bool = True,
        micro_nav_threshold_usdt: float = 100.0,
        micro_reserve_floor_ratio: float = 0.30,
        micro_max_positions: int = 2,
        micro_min_trade_notional_usdt: float = 10.0,
    ) -> None:
        self._state = shared_state
        self._exchange = exchange_client
        self._journal_dir = Path(trade_journal_dir)
        self._reserve_floor_ratio = float(reserve_floor_ratio)
        self._critical_free_ratio = float(critical_free_ratio)
        self._dust_ratio_threshold = float(dust_ratio_threshold)
        self._stale_position_sec = float(stale_position_hours) * 3600.0
        self._min_recovery_notional_usdt = float(min_recovery_notional_usdt)
        self._allow_unknown_entry_recovery_sell = bool(allow_unknown_entry_recovery_sell)
        self._max_recovery_sells_per_cycle = max(1, int(max_recovery_sells_per_cycle))
        self._recovery_decision_interval_sec = float(recovery_decision_interval_sec)
        self._buy_block_when_recovery_active = bool(buy_block_when_recovery_active)
        self._micro_nav_threshold_usdt = float(micro_nav_threshold_usdt)
        self._micro_reserve_floor_ratio = float(micro_reserve_floor_ratio)
        self._micro_max_positions = max(1, int(micro_max_positions))
        self._micro_min_trade_notional_usdt = float(micro_min_trade_notional_usdt)

    async def refresh(self, *, force: bool = False) -> dict[str, Any]:
        if self._state is None:
            return {}
        state = getattr(self._state, "recovery_state", {}) or {}
        now = time.time()
        if (
            not force
            and float(state.get("last_eval_ts", 0.0) or 0.0) > 0.0
            and (now - float(state.get("last_eval_ts", 0.0) or 0.0)) < 5.0
        ):
            return state

        throttled = bool(getattr(self._state, "exchange_throttled", False))
        hydrated = bool(getattr(self._state, "is_ready", lambda: False)())
        positions = await self._hydrate_positions_from_wallet(throttled=throttled)
        nav = self._compute_nav(positions)
        free_usdt = float(getattr(self._state, "free_balance_usdt", 0.0) or 0.0)
        if free_usdt <= 0.0:
            free_usdt = float((getattr(self._state, "balance", {}) or {}).get("USDT", 0.0) or 0.0)
        reserve_floor_ratio = (
            self._micro_reserve_floor_ratio if 0.0 < nav <= self._micro_nav_threshold_usdt else self._reserve_floor_ratio
        )
        min_trade_usdt = max(self._micro_min_trade_notional_usdt, self._min_recovery_notional_usdt)
        free_ratio = (free_usdt / nav) if nav > 0 else 0.0
        # Exclude MICRO_DUST from all counts — they're permanently unsellable
        active_positions = {sym: p for sym, p in positions.items() if p.status != "MICRO_DUST"}
        material_positions = [p for p in active_positions.values() if p.notional_usdt >= self._min_recovery_notional_usdt]
        unknown_entry_count = sum(
            1 for p in material_positions if p.entry_price_confidence == "UNKNOWN"
        )
        dust_count = sum(1 for p in active_positions.values() if p.status == "DUST")
        stale_count = sum(1 for p in active_positions.values() if p.status == "STALE")
        capital_locked_ratio = max(0.0, 1.0 - free_ratio) if nav > 0 else 0.0
        dust_ratio = (
            sum(p.notional_usdt for p in active_positions.values() if p.status == "DUST") / nav if nav > 0 else 0.0
        )
        reasons: list[str] = []
        advisory_reasons: list[str] = []
        if not hydrated and not throttled:
            reasons.append("POSITIONS_NOT_HYDRATED")
        if free_ratio < reserve_floor_ratio:
            reasons.append("LOW_FREE_USDT")
        if free_usdt < min_trade_usdt:
            reasons.append("MIN_TRADE_NOT_MET")
        if unknown_entry_count > 0:
            reasons.append("UNKNOWN_ENTRY_PRESENT")
        if dust_ratio > self._dust_ratio_threshold:
            reasons.append("DUST_RATIO_HIGH")
        if stale_count > 0:
            advisory_reasons.append("STALE_POSITIONS")
        if capital_locked_ratio > (1.0 - reserve_floor_ratio):
            reasons.append("CAPITAL_LOCKED")
        recovery_mode_active = bool(reasons)
        ranked = self.rank_capital_unlock_candidates(
            positions=active_positions,
            free_usdt=free_usdt,
            nav=nav,
        )
        # Never recommend SELL_PROFIT on a position younger than 5 minutes.
        # A fresh position has no reliable entry price yet (journal write is async)
        # so P&L calculation may be based on stale data from a prior session.
        _MIN_PROFIT_SELL_AGE_SEC = 300.0
        actionable_ranked = [
            p for p in ranked
            if not (
                p.suggested_action == "SELL_PROFIT"
                and (p.position_age_sec or 0.0) < _MIN_PROFIT_SELL_AGE_SEC
            )
        ]
        candidate_symbols = [p.symbol for p in actionable_ranked]
        selected = actionable_ranked[0] if actionable_ranked else None
        # When capital is healthy, only allow SELL_PROFIT/SELL_STALE — never SELL_RECOVERY.
        # SELL_RECOVERY must only fire when recovery_mode_active=True (capital actually constrained).
        if selected is not None and not recovery_mode_active and selected.suggested_action == "SELL_RECOVERY":
            selected = None
        selected_action = selected.suggested_action if selected else "WAIT"
        selected_symbol = selected.symbol if selected else ""
        selected_reason = selected.reason if selected else "NO_CANDIDATE"
        effective_reasons = list(reasons)
        if stale_count > 0 and not recovery_mode_active and selected is not None:
            effective_reasons.extend(advisory_reasons)
        buy_blocked = (not hydrated and not throttled) or (
            self._buy_block_when_recovery_active and recovery_mode_active
        )

        next_state = {
            "ready": hydrated or throttled,
            "wallet_hydrated": hydrated,
            "hydration_deferred": throttled and not hydrated,
            "recovery_mode_active": recovery_mode_active,
            "buy_blocked": buy_blocked,
            "reason": ",".join(effective_reasons) if effective_reasons else "CAPITAL_HEALTHY",
            "hard_reasons": reasons,
            "advisory_reasons": advisory_reasons,
            "candidate_symbols": candidate_symbols,
            "selected_symbol": selected_symbol,
            "selected_action": selected_action,
            "selected_reason": selected_reason,
            "last_eval_ts": now,
            "nav_usdt": nav,
            "free_usdt": free_usdt,
            "free_ratio": free_ratio,
            "reserve_floor_ratio": reserve_floor_ratio,
            "positions_count": len(active_positions),
            "unknown_entry_count": unknown_entry_count,
            "dust_count": dust_count,
            "stale_count": stale_count,
            "capital_locked_ratio": capital_locked_ratio,
            "positions": {sym: asdict(pos) for sym, pos in positions.items()},
            "last_recovery_decision_ts": float(state.get("last_recovery_decision_ts", 0.0) or 0.0),
            "max_recovery_sells_per_cycle": self._max_recovery_sells_per_cycle,
            "recovery_decision_interval_sec": self._recovery_decision_interval_sec,
            "min_trade_usdt": min_trade_usdt,
        }
        self._state.position_recovery = next_state["positions"]
        self._state.recovery_state = next_state
        self._log_event(
            "PORTFOLIO_RECOVERY_EVAL",
            {
                "nav_usdt": nav,
                "free_usdt": free_usdt,
                "free_ratio": free_ratio,
                "reserve_floor_ratio": reserve_floor_ratio,
                "positions_count": len(active_positions),
                "unknown_entry_count": unknown_entry_count,
                "dust_count": dust_count,
                "stale_count": stale_count,
                "capital_locked_ratio": capital_locked_ratio,
                "recovery_mode_active": recovery_mode_active,
                "candidate_symbols": candidate_symbols,
                "selected_action": selected_action,
                "selected_symbol": selected_symbol,
                "reason": next_state["reason"],
            },
        )
        return next_state

    def rank_capital_unlock_candidates(
        self,
        *,
        positions: dict[str, RecoveryPositionRecord],
        free_usdt: float,
        nav: float,
    ) -> list[RecoveryPositionRecord]:
        critical = (free_usdt / nav) if nav > 0 else 0.0

        def rank_tuple(pos: RecoveryPositionRecord) -> tuple[int, float]:
            if pos.status == "DUST" and pos.sellable:
                return (0, -pos.notional_usdt)
            if pos.status == "PROFITABLE":
                return (1, -(pos.unrealized_pnl_pct or 0.0))
            if pos.status == "STALE":
                return (2, -pos.notional_usdt)
            if pos.status == "WEAK":
                return (3, -pos.notional_usdt)
            if (
                pos.status == "ENTRY_UNKNOWN"
                and self._allow_unknown_entry_recovery_sell
                and critical <= self._critical_free_ratio
                and pos.notional_usdt >= self._min_recovery_notional_usdt
            ):
                return (4, -pos.notional_usdt)
            return (9, 0.0)

        ranked = sorted(positions.values(), key=rank_tuple)
        return [p for p in ranked if rank_tuple(p)[0] < 9]

    def should_block_buy(self) -> tuple[bool, str]:
        state = getattr(self._state, "recovery_state", {}) or {}
        blocked = bool(state.get("buy_blocked", True))
        reason = str(state.get("reason", "BOOTSTRAP") or "BOOTSTRAP")
        return blocked, reason

    def can_emit_recovery_sell(self) -> bool:
        state = getattr(self._state, "recovery_state", {}) or {}
        last_ts = float(state.get("last_recovery_decision_ts", 0.0) or 0.0)
        return (time.time() - last_ts) >= self._recovery_decision_interval_sec

    def mark_recovery_sell_emitted(self) -> None:
        state = getattr(self._state, "recovery_state", {}) or {}
        state["last_recovery_decision_ts"] = time.time()
        self._state.recovery_state = state

    async def _hydrate_positions_from_wallet(
        self, *, throttled: bool
    ) -> dict[str, RecoveryPositionRecord]:
        balances = dict(getattr(self._state, "balance", {}) or {})
        prices = dict(getattr(self._state, "prices", {}) or {})
        out: dict[str, RecoveryPositionRecord] = {}
        _FEE_RESERVE_ASSETS: frozenset[str] = frozenset({"BNB"})
        for asset, qty_raw in balances.items():
            asset_u = str(asset or "").upper()
            qty = float(qty_raw or 0.0)
            if asset_u == "USDT" or asset_u in _FEE_RESERVE_ASSETS or qty <= 0.0:
                continue
            symbol = f"{asset_u}USDT"
            current_price = float(prices.get(symbol, 0.0) or 0.0)
            notional = qty * current_price
            if notional <= 0.0 and qty <= 1e-8:
                continue
            entry_price, confidence, source, entry_time = await self._recover_entry_price(
                symbol=symbol,
                qty=qty,
                current_price=current_price,
                throttled=throttled,
            )
            rec = RecoveryPositionRecord(
                symbol=symbol,
                asset=asset_u,
                qty=qty,
                current_price=current_price,
                notional_usdt=notional,
                avg_entry_price=entry_price,
                entry_price_confidence=confidence,
                entry_source=source,
                entry_time=entry_time,
            )
            self._classify_position(rec)
            out[symbol] = rec
        return out

    async def _recover_entry_price(
        self,
        *,
        symbol: str,
        qty: float,
        current_price: float,
        throttled: bool,
    ) -> tuple[float, str, str, float]:
        # Priority 1: SharedState — always has the live fill data from the current session.
        # Check this FIRST so a fresh BUY (entry_time = seconds ago) is never overridden
        # by a stale journal entry from a previous session (days ago).
        shared_pos = (getattr(self._state, "position_recovery", {}) or {}).get(symbol) or (
            getattr(self._state, "positions", {}) or {}
        ).get(symbol)
        if shared_pos:
            entry = float(getattr(shared_pos, "entry_price", 0.0) or shared_pos.get("entry_price", 0.0) or 0.0)
            entry_time = float(getattr(shared_pos, "entry_time", 0.0) or shared_pos.get("entry_time", 0.0) or 0.0)
            if entry > 0 and entry_time > 0:
                return entry, "HIGH", "SHARED_STATE", entry_time

        # Priority 2: Local journal (most recent BUY timestamp — see _recover_from_local_journal).
        journal = self._recover_from_local_journal(symbol)
        if journal is not None:
            # If SharedState had no entry_time but has a price, prefer SharedState price
            # with journal timestamp to avoid cross-symbol price contamination.
            if shared_pos:
                entry = float(getattr(shared_pos, "entry_price", 0.0) or shared_pos.get("entry_price", 0.0) or 0.0)
                if entry > 0:
                    return entry, "HIGH", "SHARED_STATE_J", journal[3]
            return journal

        # Priority 3: Exchange trade history.
        if not throttled and self._exchange is not None and hasattr(self._exchange, "get_my_trades"):
            try:
                trades = await self._exchange.get_my_trades(symbol, limit=50)
                exchange_rec = self._recover_from_exchange_trades(symbol, trades, qty)
                if exchange_rec is not None:
                    return exchange_rec
            except Exception as exc:
                logger.debug("exchange trade recovery failed for %s: %s", symbol, exc)

        # Priority 4: SharedState price with no timestamp (entry_time unknown).
        if shared_pos:
            entry = float(getattr(shared_pos, "entry_price", 0.0) or shared_pos.get("entry_price", 0.0) or 0.0)
            if entry > 0:
                return entry, "MEDIUM", "SHARED_STATE", 0.0

        if current_price > 0:
            return current_price, "LOW", "ESTIMATED", 0.0
        return 0.0, "UNKNOWN", "UNKNOWN", 0.0

    def _recover_from_local_journal(self, symbol: str) -> Optional[tuple[float, str, str, float]]:
        fills: list[dict[str, Any]] = []
        try:
            for jf in sorted(self._journal_dir.glob("trade_journal_*.jsonl"), reverse=True)[:10]:
                with open(jf) as fh:
                    for line in fh:
                        rec = json.loads(line)
                        if rec.get("symbol") == symbol and str(rec.get("side", "")).upper() == "BUY":
                            fills.append(rec)
        except Exception as exc:
            logger.debug("journal recovery failed for %s: %s", symbol, exc)
        if not fills:
            return None
        # Sort by epoch descending — most recent buy first.
        # We use the latest BUY as entry_time so that a rebuy of a previously
        # held symbol doesn't inherit a stale entry_time from months ago.
        fills.sort(key=lambda r: float(r.get("epoch", 0.0) or 0.0), reverse=True)
        qty_total = 0.0
        cost_total = 0.0
        latest_ts = 0.0
        for fill in fills:
            qty = float(fill.get("qty", fill.get("executedQty", 0.0)) or 0.0)
            price = float(fill.get("price", fill.get("avg_price", 0.0)) or 0.0)
            if qty <= 0.0 or price <= 0.0:
                continue
            qty_total += qty
            cost_total += qty * price
            epoch = float(fill.get("epoch", 0.0) or 0.0)
            if epoch > latest_ts:
                latest_ts = epoch
        if qty_total <= 0.0 or cost_total <= 0.0:
            return None
        # Cap journal timestamp at 7 days old. If the most recent BUY in the journal
        # is older than 7 days, treat entry_time as unknown (0) so the position is
        # classified ENTRY_UNKNOWN rather than immediately STALE.
        _MAX_JOURNAL_AGE_SEC = 7 * 86400.0
        if latest_ts > 0.0 and (time.time() - latest_ts) > _MAX_JOURNAL_AGE_SEC:
            latest_ts = 0.0
        return cost_total / qty_total, "HIGH", "LOCAL_JOURNAL", latest_ts

    def _recover_from_exchange_trades(
        self, symbol: str, trades: list[dict[str, Any]], qty: float
    ) -> Optional[tuple[float, str, str, float]]:
        del symbol
        buys = [t for t in trades if str(t.get("isBuyer", False)).lower() in ("true", "1") or str(t.get("side", "")).upper() == "BUY"]
        if not buys:
            return None
        # Sort most-recent first so entry_time reflects the latest buy, not a stale one.
        buys.sort(key=lambda t: float(t.get("time", 0.0) or 0.0), reverse=True)
        qty_total = 0.0
        cost_total = 0.0
        latest_ts = 0.0
        for trade in buys:
            t_qty = float(trade.get("qty", trade.get("executedQty", 0.0)) or 0.0)
            t_price = float(trade.get("price", 0.0) or 0.0)
            if t_qty <= 0.0 or t_price <= 0.0:
                continue
            qty_total += t_qty
            cost_total += t_qty * t_price
            ts_ms = float(trade.get("time", 0.0) or 0.0)
            ts = ts_ms / 1000.0 if ts_ms > 1e12 else ts_ms
            if ts > latest_ts:
                latest_ts = ts
            if qty_total >= qty:
                break
        if qty_total <= 0.0 or cost_total <= 0.0:
            return None
        return cost_total / qty_total, "MEDIUM", "EXCHANGE_TRADES", latest_ts

    def _classify_position(self, pos: RecoveryPositionRecord) -> None:
        if pos.avg_entry_price > 0.0 and pos.current_price > 0.0:
            # Fee-adjusted P&L: 0.1% buy + 0.1% sell = 0.2% round-trip.
            # Without this, positions show positive P&L before breaking even on fees
            # and get incorrectly flagged as SELL_PROFIT.
            _fee = 0.001  # per side
            cost_basis = pos.avg_entry_price * pos.qty * (1.0 + _fee)
            sell_proceeds = pos.current_price * pos.qty * (1.0 - _fee)
            pos.unrealized_pnl_usdt = sell_proceeds - cost_basis
            pos.unrealized_pnl_pct = (pos.unrealized_pnl_usdt / cost_basis) * 100.0
        if pos.entry_time > 0.0:
            pos.position_age_sec = max(0.0, time.time() - pos.entry_time)
        pos.sellable = pos.notional_usdt >= self._min_recovery_notional_usdt
        # Micro-dust: notional < $0.50 — practically worthless, Binance won't
        # accept sell orders, and top-up is not worth it. Mark as IGNORED so
        # slot-counting and recovery ranking skip these permanently.
        if pos.notional_usdt < 0.50:
            pos.status = "MICRO_DUST"
            pos.suggested_action = "IGNORE"
            pos.reason = "micro_dust_below_50_cents"
            pos.sellable = False
            return
        if pos.notional_usdt < self._min_recovery_notional_usdt:
            pos.status = "DUST"
            pos.suggested_action = "WAIT"
            pos.reason = "below_min_recovery_notional"
            # Compute sellability from exchange filters regardless of age —
            # sellable=True means the exchange accepts the order; age only delays action.
            pos.sellable = self._is_dust_sellable(pos)
            # Dust positions must age 45 min before any sell or top-up attempt.
            # Without this gate the system creates a buy→sell loop: recovery emits
            # dust_promotion BUY + SELL, the BUY creates new dust, recovery fires again.
            _DUST_MIN_AGE_SEC = 2700.0  # 45 min — same as normal minimum hold
            _dust_age = pos.position_age_sec or 0.0
            if _dust_age < _DUST_MIN_AGE_SEC:
                pos.reason = f"dust_too_new(age={int(_dust_age/60)}m)"
                return
            if pos.sellable:
                pos.suggested_action = "SELL_DUST"
            else:
                self._compute_dust_top_up_need(pos)
            return
        if pos.entry_price_confidence == "UNKNOWN":
            # Only attempt recovery sell if position is old enough — new positions
            # may not have journal entries yet (entry written async after fill).
            # Wait at least 15 minutes regardless of whether entry_time is set,
            # because entry_price_confidence may resolve to UNKNOWN briefly after
            # a fresh BUY while the journal write is still in flight.
            _min_unknown_age_sec = 900.0
            if (pos.position_age_sec or 0.0) < _min_unknown_age_sec:
                pos.status = "RECOVERABLE"
                pos.suggested_action = "HOLD"
                pos.reason = "entry_unknown_too_new"
                return
            pos.status = "ENTRY_UNKNOWN"
            # Only force-sell unknown-entry positions if deeply in loss (< -5%).
            # If PnL is flat or small loss, hold and let TP/SL engine manage the exit —
            # hydration gives a valid entry price for TP/SL even when confidence is LOW.
            _unknown_pnl = float(pos.unrealized_pnl_pct or 0.0)
            if self._allow_unknown_entry_recovery_sell and _unknown_pnl < -5.0:
                pos.suggested_action = "SELL_RECOVERY"
            else:
                pos.suggested_action = "HOLD"
            pos.reason = f"entry_price_unknown(pnl={_unknown_pnl:.1f}%)"
            return
        # Hard minimum: never flag a position as stale under 30 minutes.
        # Protects freshly bought positions whose journal entry_time may be stale
        # from a previous session holding the same symbol.
        _MIN_STALE_AGE_SEC = 1800.0
        age = pos.position_age_sec or 0.0
        if age >= self._stale_position_sec and age >= _MIN_STALE_AGE_SEC:
            # Only mark STALE if the position is losing money (< -5% unrealized PnL) OR
            # entry price confidence is unknown. Profitable old positions should keep running —
            # they just need the trailing stop to manage exit, not a forced stale-sell.
            _pnl_pct = float(pos.unrealized_pnl_pct or 0.0)
            # Only force-sell stale positions at a deep loss (≤ -8%).
            # Flat (0% to -8%) and unknown-entry positions should be held — data shows
            # 99.97% of recovery sells were flat positions that would have recovered.
            # The ML signal exit (100% win rate) is the right exit mechanism for those.
            if _pnl_pct <= -8.0:
                # Guard: a position below Binance's min_notional cannot be sold.
                if not self._is_dust_sellable(pos):
                    pos.status = "DUST"
                    pos.suggested_action = "WAIT"
                    pos.reason = f"stale_but_unsellable(age={age/3600:.1f}h notional={pos.notional_usdt:.2f})"
                    self._compute_dust_top_up_need(pos)
                    return
                pos.status = "STALE"
                pos.suggested_action = "SELL_STALE"
                pos.reason = f"stale_deep_loss(age={age/3600:.1f}h pnl={_pnl_pct:.1f}%)"
                return
            # Old but profitable or flat — hold and let ML signal drive the exit.
            if _pnl_pct > 0.0:
                pos.status = "PROFITABLE"
                pos.suggested_action = "HOLD"
                pos.reason = f"old_but_profitable(age={age/3600:.1f}h pnl={_pnl_pct:.1f}%)"
            else:
                pos.status = "WEAK"
                pos.suggested_action = "HOLD"
                pos.reason = f"stale_flat_holding(age={age/3600:.1f}h pnl={_pnl_pct:.1f}%)"
            return
        # Minimum hold before any recovery/profit sell.
        # Raised to 45 min to stop the TONUSDT-style churn (buy → immediate loss → sell → repeat).
        # Emergency override: allow early exit at 15 min if PnL is already ≤ −3% — that is
        # a real signal the entry was wrong, not just noise. This is tighter than the post-hold
        # WEAK threshold (−2%) so normal volatility never triggers the emergency path.
        _MIN_HOLD_FOR_ANY_SELL_SEC = 7200.0   # 2 hours — matches direct-sell gate
        _EMERGENCY_LOSS_PCT = -3.0            # early exit allowed above this loss
        _EMERGENCY_MIN_AGE_SEC = 900.0        # but only after 15 min (journal write lag)
        pnl_pct = float(pos.unrealized_pnl_pct or 0.0)
        age = pos.position_age_sec or 0.0
        if age < _MIN_HOLD_FOR_ANY_SELL_SEC:
            if pnl_pct <= _EMERGENCY_LOSS_PCT and age >= _EMERGENCY_MIN_AGE_SEC:
                # Position is clearly wrong — allow emergency exit to stop further bleeding
                pos.status = "WEAK"
                pos.suggested_action = "SELL_RECOVERY"
                pos.reason = f"emergency_loss(age={age/60:.0f}m pnl={pnl_pct:.1f}%)"
                return
            pos.status = "RECOVERABLE"
            pos.suggested_action = "HOLD"
            pos.reason = "min_hold_not_reached"
            return
        # Require ≥0.5% PnL before selling as "profitable" — covers Binance 0.2% round-trip
        # fee plus generates real profit. Exits at +0.1% technically above zero but net
        # negative after fees, causing the fee-drain churn observed in 39 recovery trades.
        _MIN_PROFIT_PCT = float(
            __import__("os").environ.get("RECOVERY_MIN_PROFIT_PCT", "0.5")
        )
        if pnl_pct >= _MIN_PROFIT_PCT:
            pos.status = "PROFITABLE"
            pos.suggested_action = "SELL_PROFIT"
            pos.reason = f"profitable_position(pnl={pnl_pct:.2f}%>={_MIN_PROFIT_PCT:.1f}%)"
            return
        if pnl_pct <= -2.0:
            pos.status = "WEAK"
            pos.suggested_action = "SELL_RECOVERY"
            pos.reason = "negative_momentum"
            return
        pos.status = "RECOVERABLE"
        pos.suggested_action = "HOLD"
        pos.reason = "recoverable_hold"

    def _is_dust_sellable(self, pos: RecoveryPositionRecord) -> bool:
        """Check if a dust position meets exchange filter requirements for sale.

        Dust is only sellable if qty >= min_qty AND notional >= min_notional per Binance.
        """
        if pos.qty <= 0.0 or pos.current_price <= 0.0:
            return False

        symbol = pos.symbol
        if not symbol:
            return False

        filters = self._get_symbol_filters(symbol)
        min_qty = filters.get("min_qty", 0.0)
        min_notional = filters.get("min_notional", 10.0)

        if pos.qty < min_qty:
            return False
        if pos.notional_usdt < min_notional:
            return False
        return True

    def _compute_dust_top_up_need(self, pos: RecoveryPositionRecord) -> None:
        """Check if dust can be promoted to sellable via small BUY top-up.

        If qty < min_qty (but close), flag for promotion. Only promote if the cost is small.
        """
        if not pos.symbol or pos.current_price <= 0.0:
            return

        filters = self._get_symbol_filters(pos.symbol)
        min_qty = filters.get("min_qty", 0.0)
        min_notional = filters.get("min_notional", 10.0)

        promotion_cost_limit = 2.0

        if pos.qty < min_qty:
            qty_needed = min_qty - pos.qty
            cost_to_promote = qty_needed * pos.current_price
            if cost_to_promote <= promotion_cost_limit:
                pos.needs_top_up = True
                pos.top_up_qty = qty_needed
                logger.debug(
                    "Dust %s: qty=%.8f < min=%.8f; can promote for $%.2f",
                    pos.symbol,
                    pos.qty,
                    min_qty,
                    cost_to_promote,
                )
                return

        if pos.notional_usdt < min_notional:
            qty_needed = (min_notional - pos.notional_usdt) / pos.current_price
            cost_to_promote = qty_needed * pos.current_price
            if cost_to_promote <= promotion_cost_limit:
                pos.needs_top_up = True
                pos.top_up_qty = qty_needed
                logger.debug(
                    "Dust %s: notional=$%.2f < min=$%.2f; can promote for $%.2f",
                    pos.symbol,
                    pos.notional_usdt,
                    min_notional,
                    cost_to_promote,
                )
                return

        logger.debug(
            "Dust %s: qty=%.8f notional=$%.2f; too expensive to promote (would cost $%.2f)",
            pos.symbol,
            pos.qty,
            pos.notional_usdt,
            max(
                (min_qty - pos.qty) * pos.current_price,
                (min_notional - pos.notional_usdt),
            ),
        )

    def _get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        """Fetch symbol filters (LOT_SIZE, MIN_NOTIONAL) from Binance.

        Defaults to conservative minimums if fetch fails.
        """
        if not symbol or not self._exchange:
            return {"min_qty": 0.0, "min_notional": 10.0, "step_size": 0.00000001}

        try:
            # Prefer cached filters; fall back to synchronous exchange call (works in tests and
            # sync contexts where get_exchange_info is not async).
            info = None
            cached = getattr(self._state, "symbol_filters", {}) or {}
            if isinstance(cached, dict) and symbol in cached:
                info = {"symbols": [cached[symbol]]}
            if not info and hasattr(self._exchange, "get_exchange_info"):
                try:
                    result = self._exchange.get_exchange_info()
                    if isinstance(result, dict) and "symbols" in result:
                        info = result
                except Exception:
                    pass
            if not info or "symbols" not in info or not info["symbols"]:
                return {"min_qty": 0.0, "min_notional": 10.0, "step_size": 0.00000001}
            # Find the matching symbol entry
            matched = next((s for s in info["symbols"] if s.get("symbol") == symbol), None)
            if matched:
                info = {"symbols": [matched]}

            symbol_info = info["symbols"][0]
            filters = {}
            for f in symbol_info.get("filters", []):
                ftype = f.get("filterType", "")
                if ftype == "LOT_SIZE":
                    filters["min_qty"] = float(f.get("minQty", "0.0") or 0.0)
                    filters["step_size"] = float(f.get("stepSize", "0.00000001") or 0.00000001)
                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    filters["min_notional"] = float(f.get("minNotional", "10.0") or 10.0)

            if "min_qty" not in filters:
                filters["min_qty"] = 0.0
            if "min_notional" not in filters:
                filters["min_notional"] = 10.0
            if "step_size" not in filters:
                filters["step_size"] = 0.00000001

            return filters
        except Exception as e:
            logger.debug("Failed to fetch filters for %s: %s; using defaults", symbol, e)
            return {"min_qty": 0.0, "min_notional": 10.0, "step_size": 0.00000001}

    def _compute_nav(self, positions: dict[str, RecoveryPositionRecord]) -> float:
        nav = float(getattr(self._state, "nav_usdt", 0.0) or 0.0)
        if nav > 0.0:
            return nav
        free_usdt = float((getattr(self._state, "balance", {}) or {}).get("USDT", 0.0) or 0.0)
        return free_usdt + sum(pos.notional_usdt for pos in positions.values())

    def _log_event(self, event: str, payload: dict[str, Any]) -> None:
        logger.info("%s %s", event, json.dumps(payload, sort_keys=True, default=str))
