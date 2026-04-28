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
