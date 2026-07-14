from __future__ import annotations

import pytest

from core_engine.implementations import SituationEngineImpl
from core_engine.native.decisions import Action, NativeDecisionEngine, PortfolioSnapshot
from core_engine.native.nav_protection import (
    NAVAttributionEngine,
    NAVProtectionEngine,
    evaluate_nav_protection,
)
from core_engine.native.position_hydration_engine import HydrationState, HydratedPosition, NativePositionHydrationEngine
from core_engine.native.shared_state import NativeSharedState


def _portfolio(*, nav: float, positions: dict[str, float], mode_name: str = "") -> PortfolioSnapshot:
    return PortfolioSnapshot(
        nav=nav,
        nav_peak=max(nav, 1.0),
        balance={"USDT": nav},
        positions=positions,
        open_orders={},
        mode_name=mode_name,
    )


def test_realized_profit_attribution_enters_profit_lock() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.session_anchor_nav = 80.0
    state.nav_usdt = 80.0
    state.update_nav(90.0)
    state.metrics["realized_pnl"] = 10.0
    state.metrics["unrealized_pnl"] = 0.0
    state.last_nav_attribution = {
        "realized_pnl_total_usdt": 0.0,
        "unrealized_pnl_total_usdt": 0.0,
        "free_usdt": 80.0,
    }

    attribution, protection = evaluate_nav_protection(state)

    assert attribution.attribution_type == "REALIZED_PROFIT"
    assert protection.protection_mode == "PROFIT_LOCK"
    assert protection.locked_profit_usdt > 0.0


def test_unrealized_market_beta_attribution_enters_floating_gain_protection() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.session_anchor_nav = 80.0
    state.nav_usdt = 80.0
    state.update_nav(90.0)
    state.free_balance_usdt = 10.0
    state.positions = {
        "BTCUSDT": {"qty": 1.0, "entry_price": 10.0, "mark_price": 15.0},
        "ETHUSDT": {"qty": 1.0, "entry_price": 10.0, "mark_price": 15.0},
    }
    state.metrics["realized_pnl"] = 0.0
    state.metrics["unrealized_pnl"] = 10.0
    state.last_nav_attribution = {
        "realized_pnl_total_usdt": 0.0,
        "unrealized_pnl_total_usdt": 0.0,
        "free_usdt": 10.0,
    }

    attribution, protection = evaluate_nav_protection(state)

    assert attribution.attribution_type == "UNREALIZED_MARKET_BETA"
    assert protection.protection_mode == "FLOATING_GAIN_PROTECTION"
    assert protection.suggested_size_multiplier == 0.5
    assert "TIGHTEN_TP_SL" in protection.suggested_actions


def test_first_eval_can_infer_floating_gain_without_prior_attribution_snapshot() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.nav_usdt = 83.25
    state.update_nav(84.57)
    state.free_balance_usdt = 83.25
    state.positions = {
        "BNBUSDT": {"qty": 0.0018, "entry_price": 630.0, "mark_price": 636.0},
        "AVAXUSDT": {"qty": 0.0081, "entry_price": 9.40, "mark_price": 9.52},
    }
    state.metrics["realized_pnl"] = 0.0
    state.metrics["unrealized_pnl"] = 1.32
    state.last_nav_attribution = {}

    attribution, protection = evaluate_nav_protection(state)

    assert attribution.unrealized_pnl_delta_usdt > 0.0
    assert attribution.attribution_type in {"UNREALIZED_ALPHA", "UNREALIZED_MARKET_BETA"}
    assert protection.protection_mode in {"NORMAL", "FLOATING_GAIN_PROTECTION"}


def test_nav_decay_enters_defensive_mode() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    # session_anchor must be close to peak so cross-session cap doesn't trigger
    state.session_anchor_nav = 98.0
    state.peak_nav_usdt = 100.0
    state.protection_floor_usdt = 95.0
    state.nav_usdt = 98.0
    state.update_nav(96.0)
    state.last_nav_attribution = {
        "realized_pnl_total_usdt": 0.0,
        "unrealized_pnl_total_usdt": 2.0,
        "free_usdt": 30.0,
    }
    state.metrics["unrealized_pnl"] = 0.0

    _, protection = evaluate_nav_protection(state)

    assert protection.protection_mode == "DEFENSIVE"
    assert protection.allow_buy is True
    assert protection.suggested_size_multiplier == 0.5


def test_freeze_buy_blocks_new_entries() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    # session_anchor close to peak so cap logic doesn't suppress it
    state.session_anchor_nav = 97.0
    state.peak_nav_usdt = 100.0
    state.nav_usdt = 97.0
    # drawdown_freeze_buy_pct is 4% (see nav_protection.py "gap6" threshold,
    # deliberately lowered from 5%->4% in commit 66fd382c); 97 -> 93 is a
    # ~4.12% drawdown from the session anchor, comfortably past the FREEZE_BUY
    # threshold. The original 97 -> 94 scenario here was only a ~3.09% drawdown
    # and never actually crossed either the old or current threshold.
    state.update_nav(93.0)
    state.metrics["unrealized_pnl"] = -4.0
    state.last_nav_attribution = {
        "realized_pnl_total_usdt": 0.0,
        "unrealized_pnl_total_usdt": 0.0,
        "free_usdt": 93.0,
    }
    _, protection = evaluate_nav_protection(state)
    assert protection.protection_mode == "FREEZE_BUY"
    assert protection.allow_buy is False


def test_unknown_attribution_healthy_capital_stays_normal() -> None:
    # Unknown attribution with healthy capital should not penalise aggression.
    # Previously this triggered DEFENSIVE (false positive); now it stays NORMAL.
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.previous_nav_usdt = 0.0
    state.update_nav(90.0)
    attribution, protection = evaluate_nav_protection(state)

    assert attribution.confidence in {"LOW", "UNKNOWN"}
    assert protection.protection_mode == "NORMAL"
    assert protection.suggested_size_multiplier == 1.0


def test_unknown_flat_healthy_nav_stays_normal() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.nav_usdt = 84.5752
    state.peak_nav_usdt = 84.7519
    state.free_balance_usdt = 83.25268
    state.update_nav(84.5752)
    state.last_nav_attribution = {
        "realized_pnl_total_usdt": 0.0,
        "unrealized_pnl_total_usdt": 0.0,
        "free_usdt": 83.25268,
    }

    attribution, protection = evaluate_nav_protection(state)

    assert attribution.attribution_type == "UNKNOWN"
    assert protection.protection_mode == "NORMAL"
    assert protection.suggested_size_multiplier == 1.0
    assert protection.suggested_confidence_floor_delta == 0.0


def test_concentration_gain_blocks_adding_more_to_same_symbol() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.nav_usdt = 80.0
    state.update_nav(90.0)
    state.positions = {
        "SOLUSDT": {"qty": 5.0, "entry_price": 10.0, "mark_price": 12.0},
        "ADAUSDT": {"qty": 1.0, "entry_price": 10.0, "mark_price": 10.0},
    }
    state.metrics["realized_pnl"] = 0.0
    state.metrics["unrealized_pnl"] = 10.0
    state.last_nav_attribution = {
        "realized_pnl_total_usdt": 0.0,
        "unrealized_pnl_total_usdt": 0.0,
        "free_usdt": 20.0,
    }
    attribution, protection = evaluate_nav_protection(state)

    assert attribution.attribution_type == "CONCENTRATION_GAIN"
    assert protection.protection_mode == "FLOATING_GAIN_PROTECTION"
    assert "BLOCK_ADD_TO_CONCENTRATED_SYMBOL" in protection.suggested_actions
    assert protection.suggested_size_multiplier == 0.5


@pytest.mark.asyncio
async def test_situation_state_exposes_nav_protection_metrics() -> None:
    state = NativeSharedState()
    state.recovery_state["recovery_mode_active"] = False
    state.update_nav(90.0)
    state.free_balance_usdt = 40.0
    state.invested_capital_usdt = 50.0
    state.nav_protection_state = {
        "protection_mode": "FLOATING_GAIN_PROTECTION",
        "allow_buy": True,
        "suggested_size_multiplier": 0.5,
        "suggested_confidence_floor_delta": 0.1,
        "protection_floor_usdt": 85.0,
    }
    state.last_nav_attribution = {
        "attribution_type": "UNREALIZED_MARKET_BETA",
        "confidence": "MEDIUM",
    }

    situation = await SituationEngineImpl.get_situation_state({"shared_state": state})

    assert situation["metrics"]["nav_protection_mode"] == "FLOATING_GAIN_PROTECTION"
    assert situation["metrics"]["nav_attribution_type"] == "UNREALIZED_MARKET_BETA"
    assert situation["risk_state"] == "DEFENSIVE"


def test_nav_protection_respects_explicit_recovery_mode() -> None:
    state = NativeSharedState()
    state.previous_nav_usdt = 90.0
    state.update_nav(89.0)
    state.recovery_state["recovery_mode_active"] = True

    _, protection = evaluate_nav_protection(
        state,
        attribution_engine=NAVAttributionEngine(),
        protection_engine=NAVProtectionEngine(),
    )

    assert protection.protection_mode == "RECOVERY"
    assert protection.allow_buy is False


@pytest.mark.asyncio
async def test_hydration_resets_nav_attribution_baseline() -> None:
    state = NativeSharedState()
    state.balance = {"USDT": 83.25}
    hydration = NativePositionHydrationEngine(shared_state=state)
    hydrated = HydrationState(
        success=True,
        message="ok",
        positions={
            "BNBUSDT": HydratedPosition(
                symbol="BNBUSDT",
                qty=0.0018,
                avg_entry_price=636.0,
                current_price=636.0,
            )
        },
        total_balance_usdt=83.25,
        free_balance_usdt=83.25,
        portfolio_value=1.1448,
        total_realized_pnl=0.0,
        total_unrealized_pnl=0.0,
        positions_count=1,
    )

    await hydration.apply_to_shared_state(hydrated)

    assert state.previous_nav_usdt == pytest.approx(state.nav_usdt)
    assert state.session_anchor_nav == pytest.approx(state.nav_usdt)
    assert state.last_nav_attribution["reason"] == "STARTUP_HYDRATION_RECONCILIATION"


# ── Regression tests: 2026-07-14 main.py previous_nav_usdt ordering bug ────
# and shared_state.update_nav_protection() dropping the attribution arg.

def test_update_nav_protection_stores_attribution() -> None:
    """update_nav_protection() must actually store the attribution dict it's
    given -- it previously silently dropped it, freezing last_nav_attribution
    at whatever position_hydration_engine set at startup forever, so
    NAVAttributionEngine's realized/unrealized delta baseline never advanced."""
    state = NativeSharedState()
    attribution = {
        "realized_pnl_total_usdt": 12.5,
        "unrealized_pnl_total_usdt": 3.0,
        "free_usdt": 50.0,
        "evaluated_at": 123456.0,
    }
    state.update_nav_protection(attribution=attribution, protection_state={"peak_nav_usdt": 0.0})
    assert state.last_nav_attribution == attribution


def test_evaluate_nav_protection_advances_attribution_baseline_across_calls() -> None:
    """Two consecutive evaluate_nav_protection() calls (simulating two 60s
    cycles) must see the SECOND call's prev_realized reflect the FIRST call's
    realized total, not a value frozen at whatever it was before either call."""
    state = NativeSharedState()
    state.session_anchor_nav = 100.0
    state.nav_usdt = 100.0
    state.metrics["realized_pnl"] = 5.0

    attr1, _ = evaluate_nav_protection(state)
    # First-ever call has no prior snapshot: NAVAttributionEngine defensively
    # defaults prev_realized to current_realized to avoid a false first-call
    # spike, so delta is 0 here -- this is correct, existing behavior.
    assert attr1.realized_pnl_delta_usdt == pytest.approx(0.0)

    # Simulate more realized profit accruing between cycles.
    state.metrics["realized_pnl"] = 8.0
    attr2, _ = evaluate_nav_protection(state)
    # Must reflect the delta since the LAST call's snapshot (8 - 5 = 3), not
    # since some frozen startup baseline (which would wrongly show 8 - 0 = 8).
    assert attr2.realized_pnl_delta_usdt == pytest.approx(3.0)


def test_nav_delta_reflects_real_change_when_previous_nav_maintained_correctly() -> None:
    """Simulates the FIXED main.py flow: previous_nav_usdt is maintained by
    shared_state.update_nav() (called every trading cycle via
    get_portfolio_snapshot), NOT stomped by the NAV-protection block itself
    right before evaluate_nav_protection() reads it. nav_delta must reflect
    the real change, not collapse to ~0."""
    state = NativeSharedState()
    state.session_anchor_nav = 100.0
    state.update_nav(100.0)  # first cycle: previous_nav_usdt stays 0 (guard: nav_usdt was 0)
    state.update_nav(110.0)  # a later cycle: previous_nav_usdt correctly becomes 100.0

    attribution, _ = evaluate_nav_protection(state)

    assert state.previous_nav_usdt == pytest.approx(100.0)
    assert attribution.nav_delta_usdt == pytest.approx(10.0)
    assert attribution.nav_delta_pct == pytest.approx(0.10)


# ── Regression test: 2026-07-14 NAV peak-capping discarded genuine peaks ────

def test_genuine_intra_session_peak_above_anchor_is_preserved_on_pullback() -> None:
    """The exact scenario found by the audit: anchor=$1000, NAV climbs to a
    genuine session peak of $1500 (set THIS session, so peak_ts >=
    _session_start_ts), then retraces to $1400. The next evaluation must NOT
    collapse the tracked peak down to current_nav -- the true $1500 peak,
    since it's fresh this session, must be preserved.

    Uses realistic time.time()-relative timestamps throughout -- using small
    fixed epoch values (e.g. 1_000_000.0) would make _age_sec (real now minus
    that value) enormous, spuriously triggering the unrelated 7-day peak-decay
    logic and masking whether THIS fix actually works.
    """
    import time as _time
    now = _time.time()
    state = NativeSharedState()
    state._session_start_ts = now - 3600  # session started 1h ago
    state.session_anchor_nav = 1000.0
    state.nav_usdt = 1500.0
    state.peak_nav_usdt = 1500.0
    state.metrics["peak_nav"] = 1500.0
    state.metrics["peak_nav_ts"] = now - 600  # peak set 10min ago -- fresh this session

    # Now NAV retraces to $1400.
    state.update_nav(1400.0)

    _, protection = evaluate_nav_protection(state)

    assert protection.peak_nav_usdt == pytest.approx(1500.0)


def test_stale_pre_session_peak_is_still_capped_to_anchor() -> None:
    """A peak recorded BEFORE this session started (e.g. restored from a
    prior session's disk state) must still be capped -- this fix narrows the
    guard, it does not remove the original cross-session protection."""
    import time as _time
    now = _time.time()
    state = NativeSharedState()
    state._session_start_ts = now - 60  # session started 1min ago
    state.session_anchor_nav = 100.0
    state.nav_usdt = 100.0
    state.peak_nav_usdt = 500.0  # a stale, much-larger prior-session peak
    state.metrics["peak_nav"] = 500.0
    state.metrics["peak_nav_ts"] = now - 120  # set 2min ago -- BEFORE session start, stale

    _, protection = evaluate_nav_protection(state)

    # Capped to session_anchor (or current_nav, whichever is larger) -- the
    # stale $500 prior-session peak must not dominate this session.
    assert protection.peak_nav_usdt == pytest.approx(100.0)


def test_missing_session_start_ts_falls_back_to_conservative_capping() -> None:
    """If _session_start_ts is unavailable (e.g. some test harness or an
    unusual startup path), the fix must fall back to the original
    conservative behavior (apply the cap) rather than silently disabling
    cross-session protection."""
    import time as _time
    now = _time.time()
    state = NativeSharedState()
    state.session_anchor_nav = 100.0
    state.nav_usdt = 100.0
    state.peak_nav_usdt = 500.0
    state.metrics["peak_nav"] = 500.0
    state.metrics["peak_nav_ts"] = now - 10  # fresh, but no _session_start_ts to compare against

    _, protection = evaluate_nav_protection(state)

    assert protection.peak_nav_usdt == pytest.approx(100.0)
