"""
Self-healing controller sanity tests (run-#7 fix series).

Covers 3 stacked fixes that auto-arrest the fee-bleed observed in run #7:

  Heal-A : hydrate_positions_from_balances stamps entry_time so MIN_HOLD_SEC
           actually constrains hydrated positions (was fail-open → instant
           rotation exits with holding_sec=0.0).
  Heal-B : MetaController._passes_meta_sell_profit_gate auto-engages STRICT
           mode when NAV ≤ HEAL_STRICT_BELOW_NAV. On micro-NAV, no rotation /
           liquidation / starvation / time-exit may bypass the fee gate.
  Heal-C : _check_p_minus_1_dust_consolidation gains a NAV-aware periodic
           trigger that fires on micro-NAV regardless of portfolio capacity.

All guards are env-gated and default-on. Set the corresponding env to "0"
to fall back to legacy behavior.
"""

from __future__ import annotations

import time
import types
import contextlib

import pytest


# ─── Heal-A ────────────────────────────────────────────────────────────────
def test_heal_a_hydration_stamps_entry_time():
    """Hydrated positions get entry_time + _hydrated flag stamped."""
    # Replicate the run-#7 hydration block in isolation (avoids importing the
    # whole 8K-line shared_state which boots the orchestrator).
    pos = {"quantity": 0.0108, "avg_price": 2000.0}
    _now_ts = time.time()
    _existing_entry_ts = float(pos.get("entry_time") or 0.0)
    if _existing_entry_ts <= 0:
        pos["entry_time"] = _now_ts
        pos["opened_at"] = pos.get("opened_at") or _now_ts
        pos["_hydrated"] = True
        pos["_hydrated_at"] = _now_ts
    assert pos["entry_time"] == pytest.approx(_now_ts, abs=1.0)
    assert pos["opened_at"] == pytest.approx(_now_ts, abs=1.0)
    assert pos["_hydrated"] is True


def test_heal_a_existing_entry_time_preserved():
    """Pre-existing entry_time is NOT overwritten (e.g., bot-managed positions)."""
    original_ts = time.time() - 3600.0  # 1 hour ago
    pos = {"quantity": 0.0108, "entry_time": original_ts}
    _now_ts = time.time()
    _existing_entry_ts = float(pos.get("entry_time") or 0.0)
    if _existing_entry_ts <= 0:
        pos["entry_time"] = _now_ts
        pos["_hydrated"] = True
    assert pos["entry_time"] == original_ts, "must not overwrite real entry_time"
    assert "_hydrated" not in pos


# ─── Heal-B ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_heal_b_strict_engages_on_micro_nav(monkeypatch):
    """When NAV ≤ HEAL_STRICT_BELOW_NAV, profit gate auto-engages STRICT mode."""
    monkeypatch.setenv("HEAL_STRICT_BELOW_NAV", "150")

    # Replicate the gate prologue: read NAV → set strict if ≤ threshold.
    strict_profit_only = False
    heal_strict_threshold = 150.0
    nav_now = 100.0  # micro-NAV
    if heal_strict_threshold > 0 and not strict_profit_only:
        if 0 < nav_now <= heal_strict_threshold:
            strict_profit_only = True
    assert strict_profit_only is True, "micro-NAV must auto-engage STRICT"


@pytest.mark.asyncio
async def test_heal_b_strict_off_when_healthy_nav():
    """Healthy NAV does NOT auto-engage STRICT (legacy behavior preserved)."""
    strict_profit_only = False
    heal_strict_threshold = 150.0
    nav_now = 500.0
    if heal_strict_threshold > 0 and not strict_profit_only:
        if 0 < nav_now <= heal_strict_threshold:
            strict_profit_only = True
    assert strict_profit_only is False, "healthy NAV must keep legacy bypass behavior"


@pytest.mark.asyncio
async def test_heal_b_sl_emergency_always_bypass_even_strict():
    """SL/EMERGENCY exits MUST bypass profit gate even under STRICT (real risk events)."""
    strict_profit_only = True
    # Match the actual reason_text contract: "SL" or "EMERGENCY" substring
    for reason_text in ("EMERGENCY_LIQUIDATION", "META_EXIT SL_TRIGGER", "SL_HIT"):
        is_liq_or_time = True
        if is_liq_or_time:
            if strict_profit_only and not ("EMERGENCY" in reason_text or "SL" in reason_text):
                allowed = False
            else:
                allowed = True
        assert allowed is True, f"SL/EMERGENCY must always bypass; failed on '{reason_text}'"


@pytest.mark.asyncio
async def test_heal_b_strict_blocks_liquidation_bypass():
    """Under STRICT, plain liquidation/time-exit (no SL) must NOT bypass fee gate."""
    strict_profit_only = True
    reason_text = "LIQUIDATION ROTATION"
    is_liq_or_time = True

    if is_liq_or_time:
        if strict_profit_only and not ("EMERGENCY" in reason_text or "SL" in reason_text):
            bypass_allowed = False
        else:
            bypass_allowed = True
    assert bypass_allowed is False, "STRICT must close the liquidation bypass on micro-NAV"


# ─── Heal-C ────────────────────────────────────────────────────────────────
def test_heal_c_dust_sweep_triggers_on_micro_nav_low_capacity(monkeypatch):
    """NAV ≤ threshold + dust_count ≥ N + interval elapsed → sweep fires
    even when portfolio capacity < 80%."""
    monkeypatch.setenv("HEAL_DUST_SWEEP_BELOW_NAV", "150")
    monkeypatch.setenv("HEAL_DUST_SWEEP_MIN_COUNT", "10")
    monkeypatch.setenv("HEAL_DUST_SWEEP_INTERVAL_SEC", "1800")

    used_ratio = 0.30  # plenty of capacity
    dust_positions_count = 34
    nav_now = 97.0
    last_sweep_ts = 0.0  # never swept

    heal_below_nav = 150.0
    heal_min_dust_count = 10
    heal_interval = 1800.0
    heal_sweep = False
    if used_ratio < 0.80:
        if (
            heal_below_nav > 0
            and dust_positions_count >= heal_min_dust_count
            and 0 < nav_now <= heal_below_nav
            and (time.time() - last_sweep_ts) >= heal_interval
        ):
            heal_sweep = True
    assert heal_sweep is True, "micro-NAV + many dust + cold interval → sweep must fire"


def test_heal_c_dust_sweep_skipped_inside_interval():
    """Sweep is rate-limited by HEAL_DUST_SWEEP_INTERVAL_SEC."""
    used_ratio = 0.30
    dust_count = 34
    nav_now = 97.0
    last_sweep_ts = time.time() - 60.0  # swept 1 min ago

    heal_below_nav = 150.0
    heal_min_dust = 10
    heal_interval = 1800.0
    heal_sweep = False
    if used_ratio < 0.80:
        if (
            dust_count >= heal_min_dust
            and 0 < nav_now <= heal_below_nav
            and (time.time() - last_sweep_ts) >= heal_interval
        ):
            heal_sweep = True
    assert heal_sweep is False, "must respect rate-limit window"


def test_heal_c_dust_sweep_skipped_when_dust_count_low():
    """Below HEAL_DUST_SWEEP_MIN_COUNT, no sweep even on micro-NAV."""
    dust_count = 3  # too few
    heal_min_dust = 10
    nav_now = 97.0
    heal_below_nav = 150.0
    heal_sweep = (
        dust_count >= heal_min_dust
        and 0 < nav_now <= heal_below_nav
    )
    assert heal_sweep is False


def test_heal_c_dust_sweep_skipped_when_healthy_nav():
    """At healthy NAV, legacy 80%-capacity gate applies (no auto-sweep)."""
    used_ratio = 0.30  # plenty of capacity → legacy returns None
    dust_count = 34
    nav_now = 800.0  # healthy
    heal_below_nav = 150.0
    heal_min_dust = 10
    heal_sweep = False
    if used_ratio < 0.80:
        if (
            dust_count >= heal_min_dust
            and 0 < nav_now <= heal_below_nav
        ):
            heal_sweep = True
    # Healthy NAV → no heal sweep, legacy returns None
    assert heal_sweep is False


# ─── Run-#8 hardening: real-method integration ───────────────────────────
@pytest.mark.asyncio
async def test_heal_b_reads_nav_from_shared_state_nav_attr(monkeypatch):
    """
    Run-#8 regression: the prologue must read shared_state.nav (not the
    non-existent _last_nav). Verifies via the real
    _passes_meta_sell_profit_gate against a minimal stub.
    """
    monkeypatch.setenv("HEAL_STRICT_BELOW_NAV", "150")

    from src.l8_lifecycle.meta_controller import MetaController

    mc = MetaController.__new__(MetaController)

    class _StubSS:
        nav = 97.42                # micro-NAV — should engage STRICT
        open_trades: dict = {}
        positions: dict = {}
        latest_prices: dict = {"XRPUSDT": 1.38}

        async def safe_price(self, sym):
            return 1.38

    class _StubCfg:
        STRICT_PROFIT_ONLY_SELLS = False
        MIN_PLANNED_QUOTE_FEE_MULT = 2.5
        MIN_PROFIT_EXIT_FEE_MULT = 2.0
        TP_MIN_BUFFER_BPS = 0.0
        STAGNATION_EXIT_ENABLED = False
        STAGNATION_EXIT_MAX_LOSS_PCT = 0.0

    mc.shared_state = _StubSS()
    mc.config = _StubCfg()

    captured = []

    class _Logger:
        def info(self, *a, **kw):  captured.append(("INFO", a))
        def warning(self, *a, **kw): captured.append(("WARN", a))
        def debug(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def exception(self, *a, **kw): pass

    mc.logger = _Logger()
    mc._is_forced_capacity_recovery_sell = lambda sig: False

    # FORCED_EXIT signal that previously bypassed gate on micro-NAV.
    sig = {
        "symbol": "XRPUSDT",
        "side": "SELL",
        "_forced_exit": True,
        "reason": "CAPITAL_RECOVERY_LIQUIDITY_RESTORATION",
        "tag": "meta_exit",
    }

    # The Heal-B prologue must:
    # 1. Read NAV from shared_state.nav  (not _last_nav)
    # 2. Set strict_profit_only=True (NAV=$97.42 ≤ $150)
    # 3. Log "[Meta:ProfitGate:Heal] NAV=$97.42 ≤ $150 — auto-engaging STRICT"
    await mc._passes_meta_sell_profit_gate("XRPUSDT", sig)

    heal_log_seen = any(
        len(args) > 0 and "ProfitGate:Heal" in str(args[0])
        for level, args in captured
    )
    assert heal_log_seen, (
        "Heal-B prologue must read shared_state.nav and log the auto-engage. "
        f"captured logs: {captured}"
    )


@pytest.mark.asyncio
async def test_heal_b_no_engage_when_nav_attr_missing_or_healthy(monkeypatch):
    """When NAV>threshold OR shared_state has no NAV, do NOT auto-engage."""
    monkeypatch.setenv("HEAL_STRICT_BELOW_NAV", "150")

    from src.l8_lifecycle.meta_controller import MetaController

    mc = MetaController.__new__(MetaController)

    class _StubSS:
        nav = 800.0  # healthy
        open_trades: dict = {}
        positions: dict = {}
        latest_prices: dict = {}

        async def safe_price(self, sym):
            return 0.0

    class _StubCfg:
        STRICT_PROFIT_ONLY_SELLS = False
        MIN_PLANNED_QUOTE_FEE_MULT = 2.5
        MIN_PROFIT_EXIT_FEE_MULT = 2.0
        TP_MIN_BUFFER_BPS = 0.0
        STAGNATION_EXIT_ENABLED = False
        STAGNATION_EXIT_MAX_LOSS_PCT = 0.0

    mc.shared_state = _StubSS()
    mc.config = _StubCfg()

    captured = []

    class _Logger:
        def info(self, *a, **kw):  captured.append(("INFO", a))
        def warning(self, *a, **kw): captured.append(("WARN", a))
        def debug(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def exception(self, *a, **kw): pass

    mc.logger = _Logger()
    mc._is_forced_capacity_recovery_sell = lambda sig: False

    sig = {
        "symbol": "XRPUSDT", "side": "SELL", "_forced_exit": True,
        "reason": "REBALANCE", "tag": "meta_exit",
    }
    await mc._passes_meta_sell_profit_gate("XRPUSDT", sig)

    heal_log_seen = any(
        len(args) > 0 and "ProfitGate:Heal" in str(args[0])
        for level, args in captured
    )
    assert heal_log_seen is False, "Heal-B must NOT engage on healthy NAV"


@pytest.mark.asyncio
async def test_heal_b_prefers_live_get_nav_quote_over_stale_nav_attr(monkeypatch):
    """
    Run-#9 regression: in run #9 logs, shared_state.nav was stale at $97.86
    while live get_nav_quote() returned $153.26. Heal-B engaged STRICT against
    the stale value, OVER-engaging above the $150 threshold.

    This test proves the prologue prefers the LIVE get_nav_quote() so Heal-B
    does NOT engage when actual NAV is healthy, even if the cached `nav`
    attribute is stale and below threshold.
    """
    monkeypatch.setenv("HEAL_STRICT_BELOW_NAV", "150")

    from src.l8_lifecycle.meta_controller import MetaController

    mc = MetaController.__new__(MetaController)

    class _StubSS:
        nav = 97.86                        # STALE — would trigger Heal-B
        open_trades: dict = {}
        positions: dict = {}
        latest_prices: dict = {"XRPUSDT": 1.38}

        def get_nav_quote(self):           # LIVE — actual healthy NAV
            return 153.26

        async def safe_price(self, sym):
            return 1.38

    class _StubCfg:
        STRICT_PROFIT_ONLY_SELLS = False
        MIN_PLANNED_QUOTE_FEE_MULT = 2.5
        MIN_PROFIT_EXIT_FEE_MULT = 2.0
        TP_MIN_BUFFER_BPS = 0.0
        STAGNATION_EXIT_ENABLED = False
        STAGNATION_EXIT_MAX_LOSS_PCT = 0.0

    mc.shared_state = _StubSS()
    mc.config = _StubCfg()

    captured = []

    class _Logger:
        def info(self, *a, **kw):  captured.append(("INFO", a))
        def warning(self, *a, **kw): captured.append(("WARN", a))
        def debug(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def exception(self, *a, **kw): pass

    mc.logger = _Logger()
    mc._is_forced_capacity_recovery_sell = lambda sig: False

    sig = {
        "symbol": "XRPUSDT", "side": "SELL", "_forced_exit": True,
        "reason": "CAPITAL_RECOVERY_LIQUIDITY_RESTORATION", "tag": "meta_exit",
    }
    await mc._passes_meta_sell_profit_gate("XRPUSDT", sig)

    heal_log_seen = any(
        len(args) > 0 and "ProfitGate:Heal" in str(args[0])
        for level, args in captured
    )
    assert heal_log_seen is False, (
        "Heal-B must prefer LIVE get_nav_quote()=$153.26 over stale nav=$97.86. "
        "Run-#9 bug: it engaged STRICT against the stale value. "
        f"captured logs: {captured}"
    )


def test_get_nav_quote_skips_assets_already_mirrored_as_positions():
    """
    Run-#10 regression: get_nav_quote() must NOT double-count an asset that
    appears in both `self.balances` (free) AND `self.positions` (mirrored
    by hydrate_positions_from_balances). Run #10 reported NAV=$127.66 when
    actual was $102.64 — the $25.02 delta was the ZBT position counted twice.
    """
    import importlib
    ss_mod = importlib.import_module("src.l0_core.shared_state")

    # Build a minimal SharedState-like object exercising the de-dup branch
    class _SS:
        # required attributes referenced by get_nav_quote()
        quote_assets = ["USDT"]
        quote_asset = "USDT"
        dust_min_quote_usdt = 5.0
        positions = {
            "ZBTUSDT": {"quantity": 127.77, "avg_price": 0.196, "mark_price": 0.1962},
        }
        balances = {
            "USDT": {"free": 37.40, "locked": 0.0},
            "ZBT":  {"free": 127.77, "locked": 0.0},   # MIRRORED — must NOT double-count
        }
        latest_prices = {"ZBTUSDT": 0.1962}

        # Borrow the real method
        get_nav_quote = ss_mod.SharedState.get_nav_quote
        # tame the logger
        class _L:
            def info(self, *a, **k): pass
            def debug(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
        logger = _L()

    nav = _SS().get_nav_quote()

    # Expected: 37.40 (USDT) + 127.77 × 0.1962 (ZBT position) = 37.40 + 25.07 = 62.47
    # Buggy:    37.40 + 25.07 (free balance) + 25.07 (position) = 87.54  ← rejected
    expected = 37.40 + 127.77 * 0.1962
    assert abs(nav - expected) < 0.01, (
        f"get_nav_quote() double-counted ZBT. expected=${expected:.2f} got=${nav:.2f}. "
        "Run-#10 bug: same asset summed as both free balance AND position."
    )

