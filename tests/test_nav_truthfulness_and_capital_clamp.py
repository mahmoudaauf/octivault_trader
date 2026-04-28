"""
Sanity tests for commit f211249:
  Fix #1 — execution_manager.py: capital-aware downscale (no more EXEC_QUOTE_MISMATCH
           when downscaled quote still meets exchange floor)
  Fix #2 — shared_state.py: NAV counts mirrored/EXTERNAL positions (no more flicker)
  Fix #3 — shared_state.py: invested_capital still purely bot-managed

These are pure-logic tests — no live network, no full app boot.
"""

import asyncio
import pytest
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────
# Fix #1 — Source-level assertion that the new capital-downscale branch
# exists in execution_manager.py
# ─────────────────────────────────────────────────────────────────────

def test_fix1_capital_downscale_branch_present():
    src = (ROOT / "src/l4_execution/execution_manager.py").read_text()
    assert "_legitimate_capital_downscale" in src, (
        "Fix #1 missing: capital-downscale exemption not found in executor"
    )
    assert "EM:CAPITAL_DOWNSCALE" in src, (
        "Fix #1 missing: diagnostic log marker not present"
    )
    # Must guard with min_notional check, otherwise we'd let through dust trades
    pattern = re.compile(
        r"_legitimate_capital_downscale\s*=\s*\(\s*\n\s*reason\s*==\s*\"OK_DOWNSCALED\""
        r"\s*\n\s*and\s+execute_quote\s*>=\s*_exch_floor",
        re.MULTILINE,
    )
    assert pattern.search(src), "Fix #1 guard does not enforce ≥ min_notional"


# ─────────────────────────────────────────────────────────────────────
# Fix #2/#3 — Real NAV-rebuild test using SharedState
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fix2_nav_includes_mirrored_positions():
    """
    Scenario reproducing run #4's flicker:
      - $14.60 free USDT
      - 1 bot-managed DOGE position (qty=252, px=0.10) → value $25.20
      - 1 mirrored ETH position (qty=0.034, px=2300) → value $78.20
    Expected:
      nav             = 14.60 + 25.20 + 78.20 = 118.00
      invested_capital= 25.20  (bot-managed only)
    Pre-fix bug:
      nav would be 14.60 + 25.20 = 39.80 (mirrored ETH dropped)
      → flicker when classifier toggled _mirrored
    """
    from src.l0_core.shared_state import SharedState, SharedStateConfig

    cfg = SharedStateConfig()
    ss = SharedState(config=cfg)

    # Direct-poke balances + positions (bypass network)
    ss.balances = {"USDT": {"free": 14.60, "locked": 0.0}}
    ss.latest_prices = {"DOGEUSDT": 0.10, "ETHUSDT": 2300.0}
    ss.positions = {
        "DOGEUSDT": {
            "quantity": 252.0,
            "avg_price": 0.099,
            "entry_price": 0.099,
            "mark_price": 0.10,
            "_mirrored": False,        # bot-managed
            "classification": "BOT_POSITION",
        },
        "ETHUSDT": {
            "quantity": 0.034,
            "avg_price": 2295.0,
            "entry_price": 2295.0,
            "mark_price": 2300.0,
            "_mirrored": True,         # wallet-inherited
            "classification": "EXTERNAL_POSITION",
        },
    }

    result = await ss.rebuild_nav_from_state(source="unit_test")

    nav = float(result["nav"])
    invested = float(result["invested_capital"])
    unreal = float(result["unrealized_pnl"])

    # ── NAV must include BOTH positions ──
    expected_nav = 14.60 + 25.20 + 78.20
    assert abs(nav - expected_nav) < 0.01, (
        f"Fix #2 broken: NAV {nav:.2f} != {expected_nav:.2f} (mirrored ETH not counted)"
    )

    # ── invested_capital must EXCLUDE mirrored ──
    assert abs(invested - 25.20) < 0.01, (
        f"Fix #3 broken: invested_capital {invested:.2f} != 25.20 "
        f"(mirrored ETH leaked into invested)"
    )

    # ── unrealized_pnl must come ONLY from bot position ──
    # bot DOGE: (0.10 - 0.099) * 252 = 0.252
    expected_unreal = (0.10 - 0.099) * 252.0
    assert abs(unreal - expected_unreal) < 0.01, (
        f"Fix #3 broken: unrealized_pnl {unreal:.4f} != {expected_unreal:.4f} "
        f"(mirrored ETH leaked into unrealized PnL)"
    )


@pytest.mark.asyncio
async def test_fix2_nav_stable_when_mirrored_flag_toggles():
    """
    Reproduces the exact $103↔$24 flicker:
    Toggle _mirrored on a position and verify NAV does not jump.
    Pre-fix: flipping _mirrored True/False would change NAV by $78.
    Post-fix: NAV is invariant under _mirrored toggle (only invested_capital shifts).
    """
    from src.l0_core.shared_state import SharedState, SharedStateConfig

    cfg = SharedStateConfig()
    ss = SharedState(config=cfg)
    ss.balances = {"USDT": {"free": 24.60, "locked": 0.0}}
    ss.latest_prices = {"BTCUSDT": 65000.0}
    base_pos = {
        "quantity": 0.0012,           # value = 78.0
        "avg_price": 64000.0,
        "entry_price": 64000.0,
        "mark_price": 65000.0,
    }

    # State A: _mirrored = True
    ss.positions = {"BTCUSDT": {**base_pos, "_mirrored": True, "classification": "EXTERNAL_POSITION"}}
    nav_a = float((await ss.rebuild_nav_from_state(source="toggle_a"))["nav"])

    # State B: _mirrored = False (same value, just classification toggled)
    ss.positions = {"BTCUSDT": {**base_pos, "_mirrored": False, "classification": "BOT_POSITION"}}
    nav_b = float((await ss.rebuild_nav_from_state(source="toggle_b"))["nav"])

    assert abs(nav_a - nav_b) < 0.01, (
        f"NAV FLICKER STILL PRESENT: mirrored=True nav={nav_a:.2f} vs "
        f"mirrored=False nav={nav_b:.2f} (delta={nav_b-nav_a:+.2f})"
    )
    expected = 24.60 + 0.0012 * 65000.0
    assert abs(nav_a - expected) < 0.01, f"NAV total wrong: {nav_a:.2f} != {expected:.2f}"


if __name__ == "__main__":
    # Allow direct execution: `python tests/test_nav_truthfulness_and_capital_clamp.py`
    test_fix1_capital_downscale_branch_present()
    print("✅ Fix #1 source-check passed")
    asyncio.run(test_fix2_nav_includes_mirrored_positions())
    print("✅ Fix #2/#3 mirrored-inclusion passed")
    asyncio.run(test_fix2_nav_stable_when_mirrored_flag_toggles())
    print("✅ Fix #2 flicker-invariance passed")
    print("\n🎯 All 3 sanity tests passed")
