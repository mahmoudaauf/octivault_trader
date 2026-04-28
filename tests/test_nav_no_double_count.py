"""
Run-#10 regression: get_nav_quote() must NOT double-count assets that
appear in BOTH self.balances (non-quote) AND self.positions.

Hydration sets position.quantity from the wallet balance. The original
code added asset_value from `balances` AND pos_value from `positions`
for the same coins → systematically over-reported NAV.

User-reported truth: actual NAV = $102.64.
System (pre-fix) reported $127.66 (+$25) ≈ one ZBT holding double-counted.
"""
import sys
import types
import asyncio
import pytest


def _build_state(balances, positions, prices, quote="USDT"):
    """Construct a minimal SharedState-like object exercising get_nav_quote()."""
    from src.l0_core.shared_state import SharedState

    ss = SharedState.__new__(SharedState)
    ss.balances = balances
    ss.positions = positions
    ss.latest_prices = prices
    ss.quote_asset = quote
    ss.quote_assets = [quote]
    ss._shadow_mode = False
    ss.dust_min_quote_usdt = 5.0
    ss.metrics = {}

    import logging
    ss.logger = logging.getLogger("nav_test")

    return ss


def test_get_nav_quote_no_double_count_when_balance_and_position_overlap():
    """Hydrated wallet → asset appears in both balances and positions; must count ONCE."""
    state = _build_state(
        balances={
            "USDT": {"free": 47.40, "locked": 0.0},
            "ZBT":  {"free": 127.77, "locked": 0.0},   # 127.77 × 0.1958 ≈ $25.02
            "XRP":  {"free": 18.0,   "locked": 0.0},   # 18 × 2.50 = $45
        },
        positions={
            "ZBTUSDT": {"quantity": 127.77, "avg_price": 0.1961, "entry_price": 0.1961},
            "XRPUSDT": {"quantity": 18.0,   "avg_price": 2.50,    "entry_price": 2.50},
        },
        prices={"ZBTUSDT": 0.1958, "XRPUSDT": 2.50},
    )

    nav = state.get_nav_quote()

    # Truthful NAV: $47.40 (USDT) + $25.02 (ZBT pos) + $45.00 (XRP pos) = $117.42
    # Buggy (double-count): would be $117.42 + $25.02 + $45.00 = $187.44
    assert 117.0 <= nav <= 117.5, (
        f"NAV must NOT double-count hydrated assets. Got nav=${nav:.2f}; "
        f"expected ~$117.42. Buggy code would return ~$187.44."
    )


def test_get_nav_quote_includes_balance_only_assets():
    """Non-quote balance with NO position record must still count (it's a free holding)."""
    state = _build_state(
        balances={
            "USDT": {"free": 50.0, "locked": 0.0},
            "BTC":  {"free": 0.001, "locked": 0.0},   # 0.001 × $90000 = $90
        },
        positions={},  # no hydrated position
        prices={"BTCUSDT": 90000.0},
    )

    nav = state.get_nav_quote()
    assert 139.5 <= nav <= 140.5, f"Free BTC must contribute. Got ${nav:.2f}, expected ~$140.00"


def test_get_nav_quote_position_only_counts_when_no_balance_entry():
    """Position record with no corresponding balance entry must still count."""
    state = _build_state(
        balances={"USDT": {"free": 30.0, "locked": 0.0}},
        positions={"ETHUSDT": {"quantity": 0.1, "avg_price": 3000.0, "entry_price": 3000.0}},
        prices={"ETHUSDT": 3000.0},
    )

    nav = state.get_nav_quote()
    assert 329.5 <= nav <= 330.5, f"Position-only entry must count. Got ${nav:.2f}"
