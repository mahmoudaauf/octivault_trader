"""
Test dust-remnant disposal policy:
  1. Classify weak dust as disposal candidates
  2. Optionally promote the smallest amount needed to make them sellable
  3. Exit them through the normal native path
"""

import pytest
from core_engine.native.portfolio_recovery import (
    PortfolioRecoveryEngine,
    RecoveryPositionRecord,
)
from unittest.mock import MagicMock, AsyncMock


class TestDustDisposalClassification:
    """Test dust classification and sellability detection."""

    def test_dust_with_sufficient_qty_and_notional_is_sellable(self):
        """Dust position with qty >= min_qty and notional >= min_notional is sellable."""
        ss = MagicMock()
        ss.balance = {"USDT": 1000.0}

        exchange = MagicMock()
        exchange.get_exchange_info = MagicMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "0.0001", "stepSize": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "2.0"},
                        ],
                    }
                ]
            }
        )

        engine = PortfolioRecoveryEngine(
            shared_state=ss,
            exchange_client=exchange,
            min_recovery_notional_usdt=10.0,
        )

        pos = RecoveryPositionRecord(
            symbol="BTCUSDT",
            asset="BTC",
            qty=0.0001,
            current_price=45000.0,
            notional_usdt=4.5,
            avg_entry_price=45000.0,
            position_age_sec=3600.0,  # 1 hour — past the 45-min age gate
        )

        engine._classify_position(pos)

        assert pos.status == "DUST"
        assert pos.notional_usdt < engine._min_recovery_notional_usdt
        assert pos.sellable is True
        assert pos.suggested_action == "SELL_DUST"

    def test_dust_below_exchange_min_qty_is_not_sellable(self):
        """Dust position with qty < exchange min_qty cannot be sold."""
        ss = MagicMock()
        ss.balance = {"USDT": 1000.0}

        exchange = MagicMock()
        exchange.get_exchange_info = MagicMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "SMALLUSDT",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "10.0", "stepSize": "1.0"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5.0"},
                        ],
                    }
                ]
            }
        )

        engine = PortfolioRecoveryEngine(
            shared_state=ss,
            exchange_client=exchange,
            min_recovery_notional_usdt=10.0,
        )

        pos = RecoveryPositionRecord(
            symbol="SMALLUSDT",
            asset="SMALL",
            qty=2.0,
            current_price=1.0,
            notional_usdt=2.0,
            avg_entry_price=1.0,
        )

        engine._classify_position(pos)

        assert pos.status == "DUST"
        assert pos.sellable is False
        assert pos.suggested_action == "WAIT"

    def test_dust_promotion_flagged_if_cost_is_small(self):
        """Dust that needs qty promotion gets flagged if promotion cost is small."""
        ss = MagicMock()
        ss.balance = {"USDT": 1000.0}

        exchange = MagicMock()
        exchange.get_exchange_info = MagicMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "ADAUSDT",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "3.5", "stepSize": "0.1"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5.0"},
                        ],
                    }
                ]
            }
        )

        engine = PortfolioRecoveryEngine(
            shared_state=ss,
            exchange_client=exchange,
            min_recovery_notional_usdt=10.0,
        )

        pos = RecoveryPositionRecord(
            symbol="ADAUSDT",
            asset="ADA",
            qty=3.0,
            current_price=0.9,
            notional_usdt=2.7,
            avg_entry_price=0.9,
            position_age_sec=3600.0,  # 1 hour — past the 45-min age gate
        )

        engine._classify_position(pos)

        assert pos.status == "DUST"
        assert pos.sellable is False
        assert pos.needs_top_up is True
        assert pos.top_up_qty > 0.0
        assert pos.top_up_qty * pos.current_price <= 2.0

    def test_dust_promotion_not_flagged_if_cost_is_high(self):
        """Dust that needs expensive promotion is left unsellable and not flagged."""
        ss = MagicMock()
        ss.balance = {"USDT": 100.0}

        exchange = MagicMock()
        exchange.get_exchange_info = MagicMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "0.01", "stepSize": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "500.0"},
                        ],
                    }
                ]
            }
        )

        engine = PortfolioRecoveryEngine(
            shared_state=ss,
            exchange_client=exchange,
            min_recovery_notional_usdt=10.0,
        )

        pos = RecoveryPositionRecord(
            symbol="BTCUSDT",
            asset="BTC",
            qty=0.0001,
            current_price=45000.0,
            notional_usdt=4.5,
            avg_entry_price=45000.0,
        )

        engine._classify_position(pos)

        assert pos.status == "DUST"
        assert pos.sellable is False
        assert pos.needs_top_up is False


class TestDustRanking:
    """Test that sellable dust is ranked for capital unlock."""

    def test_sellable_dust_ranks_tier_0_for_capital_unlock(self):
        """Sellable DUST positions rank highest (tier 0) for capital unlock."""
        ss = MagicMock()
        ss.balance = {"USDT": 100.0}

        engine = PortfolioRecoveryEngine(
            shared_state=ss,
            exchange_client=None,
            min_recovery_notional_usdt=10.0,
        )

        sellable_dust = RecoveryPositionRecord(
            symbol="BTCUSDT",
            asset="BTC",
            qty=0.0001,
            current_price=45000.0,
            notional_usdt=4.5,
            avg_entry_price=45000.0,
            status="DUST",
            sellable=True,
        )

        stale_pos = RecoveryPositionRecord(
            symbol="ETHUSDT",
            asset="ETH",
            qty=1.0,
            current_price=2500.0,
            notional_usdt=2500.0,
            avg_entry_price=2500.0,
            status="STALE",
            sellable=True,
        )

        ranked = engine.rank_capital_unlock_candidates(
            positions={
                "BTCUSDT": sellable_dust,
                "ETHUSDT": stale_pos,
            },
            free_usdt=50.0,
            nav=2600.0,
        )

        assert len(ranked) == 2
        assert ranked[0].symbol == "BTCUSDT"
        assert ranked[0].status == "DUST"
        assert ranked[1].symbol == "ETHUSDT"
        assert ranked[1].status == "STALE"

    def test_unsellable_dust_does_not_rank(self):
        """Unsellable DUST positions do not rank for capital unlock."""
        ss = MagicMock()
        ss.balance = {"USDT": 100.0}

        engine = PortfolioRecoveryEngine(
            shared_state=ss,
            exchange_client=None,
            min_recovery_notional_usdt=10.0,
        )

        unsellable_dust = RecoveryPositionRecord(
            symbol="BTCUSDT",
            asset="BTC",
            qty=0.0001,
            current_price=45000.0,
            notional_usdt=4.5,
            avg_entry_price=45000.0,
            status="DUST",
            sellable=False,
        )

        stale_pos = RecoveryPositionRecord(
            symbol="ETHUSDT",
            asset="ETH",
            qty=1.0,
            current_price=2500.0,
            notional_usdt=2500.0,
            avg_entry_price=2500.0,
            status="STALE",
            sellable=True,
        )

        ranked = engine.rank_capital_unlock_candidates(
            positions={
                "BTCUSDT": unsellable_dust,
                "ETHUSDT": stale_pos,
            },
            free_usdt=50.0,
            nav=2600.0,
        )

        assert len(ranked) == 1
        assert ranked[0].symbol == "ETHUSDT"
        assert ranked[0].status == "STALE"
