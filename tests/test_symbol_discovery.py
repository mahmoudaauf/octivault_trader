"""
tests/test_symbol_discovery.py
Tests for SymbolDiscovererAgent and SymbolScreener.

Coverage targets:
  SymbolDiscovererAgent:
    - run_once: skips when discovery complete / accepted symbols present / disabled / cap hit
    - run_once: deduplicates, takes top 10, batches proposals
    - run_once: disables discovery after cap reached
    - discover_symbols: SYMBOL_UNIVERSE path
    - discover_symbols: trending path (filter logic, scoring, limit)
    - discover_symbols: fallback to get_exchange_info
    - discover_symbols: no exchange client
    - run_loop: stops on stop_flag / is_discovery_complete

  SymbolScreener:
    - _is_leveraged_symbol: detects UP/DOWN/BULL/BEAR etc., passes normal symbols
    - _prefilter_symbol: status != TRADING rejected; MIN_NOTIONAL > cap rejected; no client passes
    - _build_exclude_set: merges config exclude_list, accepted_symbols, wallet balances
    - _atr_pct: delegates to shared_state.calc_atr; returns 0 on missing price/error
    - _passes_regime_filter: disabled flag passes all; low-confidence low regime rejected
    - _evaluate_candidate: prefilter failure → None; ATR below threshold skipped in _perform_scan
    - _perform_scan: no exchange client; no tickers; no liquid symbols; ATR fallback path; happy path
    - _process_and_add_symbols: proposes via _propose; empty list is safe no-op
    - _propose: writes to symbol_proposals; returns True; returns False when shared_state is None
    - run_once / run_loop: lifecycle (start/stop idempotent; stop cancels task)
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ticker(symbol, quote_volume=2_000_000, pct=3.0, last=1.0, status="TRADING"):
    return {
        "symbol": symbol,
        "quoteVolume": quote_volume,
        "priceChangePercent": pct,
        "lastPrice": last,
        "status": status,
    }


def _make_ss(accepted=None, *, atr=0.05):
    """Minimal SharedState-like mock."""
    ss = MagicMock()
    ss.accepted_symbols = dict(accepted or {})
    ss.dynamic_config = {}
    ss.symbol_proposals = {}
    ss.symbol_discovery_enabled = True
    ss.symbols = {}

    async def _get_accepted():
        return ss.accepted_symbols

    ss.get_accepted_symbols = _get_accepted

    async def _calc_atr(sym, tf, period):
        return atr

    ss.calc_atr = _calc_atr
    ss.get_volatility_regime = AsyncMock(return_value=None)
    return ss


def _make_exchange(tickers=None, exchange_info=None, balances=None, symbol_info_map=None):
    ec = MagicMock()
    ec.get_24hr_tickers = AsyncMock(return_value=tickers or [])
    ec.get_exchange_info = AsyncMock(return_value=exchange_info or {"symbols": []})
    ec.get_account_balances = AsyncMock(return_value=balances or {})

    async def _symbol_info(sym):
        return (symbol_info_map or {}).get(sym)

    ec.symbol_info = _symbol_info
    return ec


def _make_config(**kwargs):
    from types import SimpleNamespace
    defaults = dict(
        SCREENER_MIN_VOLUME=50_000,
        SCREENER_MAX_PROPOSALS=10,
        SYMBOL_TOP_N=10,
        MAX_UNIVERSE_SYMBOLS=30,
        SYMBOL_MIN_ATR_PCT=0.003,
        SYMBOL_ATR_TIMEFRAME="1h",
        SYMBOL_ATR_PERIOD=14,
        SYMBOL_ATR_CONCURRENCY=8,
        SYMBOL_EXCLUDE_LIST=[],
        BASE_CURRENCY="USDT",
        SCREENER_INTERVAL_SECONDS=1800,
        SYMBOL_SCREENER_INTERVAL=3600,
        MAX_PER_TRADE_USDT=100.0,
        REQUIRE_TRADING_STATUS=True,
        SYMBOL_ALLOW_ATR_FALLBACK=True,
        SYMBOL_SCREENER_REGIME_FILTER=False,
        STOP_JOIN_TIMEOUT_S=5.0,
        TRENDING_DISCOVERY_ENABLED=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# SymbolDiscovererAgent imports
# ──────────────────────────────────────────────────────────────────────────────

from agents.symbol_discoverer_agent import SymbolDiscovererAgent


# ══════════════════════════════════════════════════════════════════════════════
# SymbolDiscovererAgent tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSymbolDiscovererAgentInit:
    def test_defaults(self):
        ss = MagicMock()
        ss.exchange_client = None
        ss.symbol_manager = None
        cfg = MagicMock(spec=[])
        agent = SymbolDiscovererAgent(ss, cfg)
        assert agent.name == "SymbolDiscovererAgent"
        assert agent.is_discovery_agent is True
        assert agent.stop_flag is False
        assert agent.interval == 3600

    def test_late_binding_from_shared_state(self):
        ec = MagicMock()
        sm = MagicMock()
        ss = MagicMock()
        ss.exchange_client = ec
        ss.symbol_manager = sm
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]))
        assert agent.exchange_client is ec
        assert agent.symbol_manager is sm

    def test_explicit_args_take_priority_over_late_binding(self):
        explicit_ec = MagicMock()
        ss_ec = MagicMock()
        ss = MagicMock()
        ss.exchange_client = ss_ec
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), exchange_client=explicit_ec)
        assert agent.exchange_client is explicit_ec


@pytest.mark.asyncio
class TestSymbolDiscovererAgentRunOnce:

    async def test_skips_when_discovery_complete(self):
        ss = MagicMock()
        ss.is_discovery_complete = AsyncMock(return_value=True)
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]))
        agent.discover_symbols = AsyncMock()
        await agent.run_once()
        agent.discover_symbols.assert_not_called()

    async def test_skips_when_accepted_symbols_not_empty(self):
        ss = _make_ss(accepted={"BTCUSDT": {}})
        ss.is_discovery_complete = AsyncMock(return_value=False)
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]))
        agent.discover_symbols = AsyncMock()
        await agent.run_once()
        agent.discover_symbols.assert_not_called()

    async def test_skips_when_discovery_disabled(self):
        ss = _make_ss()
        ss.symbol_discovery_enabled = False
        ss.is_discovery_complete = AsyncMock(return_value=False)
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]))
        agent.discover_symbols = AsyncMock()
        await agent.run_once()
        agent.discover_symbols.assert_not_called()

    async def test_skips_when_cap_hit(self):
        ss = _make_ss()
        ss.symbols = {f"SYM{i}USDT": {} for i in range(5)}
        ss.get_symbol_count = AsyncMock(return_value=5)
        ss.is_discovery_complete = AsyncMock(return_value=False)
        cfg = MagicMock(spec=["MAX_DISCOVERED_SYMBOLS"])
        cfg.MAX_DISCOVERED_SYMBOLS = 5
        agent = SymbolDiscovererAgent(ss, cfg)
        agent.discover_symbols = AsyncMock()
        await agent.run_once()
        agent.discover_symbols.assert_not_called()

    async def test_returns_early_when_no_symbols_discovered(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        sm = MagicMock()
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[])
        await agent.run_once()
        sm.propose_symbol.assert_not_called()

    async def test_returns_early_when_no_symbol_manager(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=None)
        agent.discover_symbols = AsyncMock(return_value=[{"symbol": "BTCUSDT", "score": 1.0}])
        # Should not raise; no symbol manager means early return
        await agent.run_once()

    async def test_proposes_all_new_symbols_when_accepted_empty(self):
        """With no accepted symbols, all discovered symbols get proposed."""
        ss = _make_ss()  # accepted={}
        ss.is_discovery_complete = AsyncMock(return_value=False)
        captured = []
        sm = MagicMock()

        async def _propose_many(symbols, source):
            captured.extend(symbols)
            return symbols

        sm.propose_symbols = _propose_many
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[
            {"symbol": "BTCUSDT", "score": 10.0},
            {"symbol": "ETHUSDT", "score": 5.0},
        ])
        await agent.run_once()
        assert "BTCUSDT" in captured
        assert "ETHUSDT" in captured

    async def test_selects_top_10_by_score(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        captured = []

        sm = MagicMock()
        async def _propose_many(symbols, source):
            captured.extend(symbols)
            return symbols
        sm.propose_symbols = _propose_many

        # 15 symbols, each with a score
        discovered = [{"symbol": f"SYM{i:02d}USDT", "score": float(i)} for i in range(15)]
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=discovered)
        await agent.run_once()
        # Should only propose the top 10 by score (indices 14..5)
        assert len(captured) == 10
        assert "SYM14USDT" in captured
        assert "SYM00USDT" not in captured

    async def test_disables_discovery_after_cap_reached(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        # First call (cap check at top) returns 4 → below cap, proceed.
        # Second call (toggle-off check) returns 5 → at cap, disable.
        ss.get_symbol_count = AsyncMock(side_effect=[4, 5])
        ss.set_symbol_discovery_enabled = AsyncMock()
        sm = MagicMock()
        sm.propose_symbols = AsyncMock(return_value=["NEWUSDT"])
        cfg = MagicMock(spec=["MAX_DISCOVERED_SYMBOLS"])
        cfg.MAX_DISCOVERED_SYMBOLS = 5
        agent = SymbolDiscovererAgent(ss, cfg, symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[{"symbol": "NEWUSDT", "score": 1.0}])
        await agent.run_once()
        ss.set_symbol_discovery_enabled.assert_awaited_once_with(False)

    async def test_disables_discovery_via_setattr_when_no_setter(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        # Use None so callable() check fails → setattr fallback is taken
        ss.set_symbol_discovery_enabled = None
        # First call: 4 (below cap, proceed); second call: 5 (at cap, disable)
        ss.get_symbol_count = AsyncMock(side_effect=[4, 5])
        sm = MagicMock()
        sm.propose_symbols = AsyncMock(return_value=["NEWUSDT"])
        cfg = MagicMock(spec=["MAX_DISCOVERED_SYMBOLS"])
        cfg.MAX_DISCOVERED_SYMBOLS = 5
        agent = SymbolDiscovererAgent(ss, cfg, symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[{"symbol": "NEWUSDT", "score": 1.0}])
        await agent.run_once()
        assert ss.symbol_discovery_enabled is False

    async def test_batches_proposals_in_groups_of_5(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        call_batches = []
        sm = MagicMock()
        async def _propose_many(symbols, source):
            call_batches.append(list(symbols))
            return symbols
        sm.propose_symbols = _propose_many

        # 10 unique symbols
        discovered = [{"symbol": f"SYM{i:02d}USDT", "score": float(i)} for i in range(10)]
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=discovered)
        await agent.run_once()
        # Should have been called twice (batches of 5)
        assert len(call_batches) == 2
        assert all(len(b) <= 5 for b in call_batches)

    async def test_handles_proposal_timeout_gracefully(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        sm = MagicMock()

        async def _slow(*a, **kw):
            await asyncio.sleep(100)

        sm.propose_symbols = _slow
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[{"symbol": "BTCUSDT", "score": 1.0}])
        # Should not raise
        await agent.run_once()

    async def test_handles_proposal_exception_gracefully(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        sm = MagicMock()
        sm.propose_symbols = AsyncMock(side_effect=RuntimeError("DB error"))
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[{"symbol": "BTCUSDT", "score": 1.0}])
        await agent.run_once()  # Must not raise

    async def test_fallback_to_per_symbol_propose(self):
        ss = _make_ss()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        sm = MagicMock(spec=["propose_symbol"])  # no propose_symbols method
        sm.propose_symbol = AsyncMock(return_value=True)
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), symbol_manager=sm)
        agent.discover_symbols = AsyncMock(return_value=[
            {"symbol": "BTCUSDT", "score": 2.0},
            {"symbol": "ETHUSDT", "score": 1.0},
        ])
        await agent.run_once()
        assert sm.propose_symbol.await_count == 2


@pytest.mark.asyncio
class TestSymbolDiscovererAgentDiscoverSymbols:

    async def test_returns_empty_when_no_exchange_client(self):
        ss = _make_ss()
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]))
        agent.exchange_client = None
        result = await agent.discover_symbols()
        assert result == []

    async def test_uses_symbol_universe_from_config(self):
        ss = _make_ss()
        cfg = MagicMock(spec=["SYMBOL_UNIVERSE"])
        cfg.SYMBOL_UNIVERSE = ["BTCUSDT", "ETHUSDT"]
        ec = _make_exchange()
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        assert [r["symbol"] for r in result] == ["BTCUSDT", "ETHUSDT"]
        assert all(r["source"] == "config" for r in result)

    async def test_trending_discovery_applies_volume_filter(self):
        tickers = [
            {"symbol": "BTCUSDT", "volume": "500", "lastPrice": "1.0", "priceChangePercent": "5.0", "status": "TRADING"},
            {"symbol": "ETHUSDT", "volume": "1500", "lastPrice": "1.0", "priceChangePercent": "3.0", "status": "TRADING"},
        ]
        ss = _make_ss()
        cfg = MagicMock(spec=["TRENDING_DISCOVERY_ENABLED", "TRENDING_MIN_VOLUME",
                               "TRENDING_MIN_PRICE", "TRENDING_MIN_PRICE_CHANGE_PERCENT",
                               "TRENDING_LIMIT"])
        cfg.TRENDING_DISCOVERY_ENABLED = True
        cfg.TRENDING_MIN_VOLUME = 1000
        cfg.TRENDING_MIN_PRICE = 0
        cfg.TRENDING_MIN_PRICE_CHANGE_PERCENT = 0
        cfg.TRENDING_LIMIT = 50
        cfg.SYMBOL_UNIVERSE = None
        ec = _make_exchange(tickers=tickers)
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        symbols = [r["symbol"] for r in result]
        assert "BTCUSDT" not in symbols   # volume 500 < 1000
        assert "ETHUSDT" in symbols

    async def test_trending_discovery_filters_non_usdt(self):
        tickers = [
            {"symbol": "BTCBUSD", "volume": "5000", "lastPrice": "1.0", "priceChangePercent": "5.0", "status": "TRADING"},
            {"symbol": "BTCUSDT", "volume": "5000", "lastPrice": "1.0", "priceChangePercent": "5.0", "status": "TRADING"},
        ]
        ss = _make_ss()
        cfg = MagicMock(spec=["TRENDING_DISCOVERY_ENABLED", "TRENDING_MIN_VOLUME",
                               "TRENDING_MIN_PRICE", "TRENDING_MIN_PRICE_CHANGE_PERCENT",
                               "TRENDING_LIMIT"])
        cfg.TRENDING_DISCOVERY_ENABLED = True
        cfg.TRENDING_MIN_VOLUME = 0
        cfg.TRENDING_MIN_PRICE = 0
        cfg.TRENDING_MIN_PRICE_CHANGE_PERCENT = 0
        cfg.TRENDING_LIMIT = 50
        cfg.SYMBOL_UNIVERSE = None
        ec = _make_exchange(tickers=tickers)
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        symbols = [r["symbol"] for r in result]
        assert "BTCBUSD" not in symbols
        assert "BTCUSDT" in symbols

    async def test_trending_discovery_scores_and_sorts(self):
        """Higher log(volume)*abs(pct_change) should rank first."""
        from math import log as mlog
        tickers = [
            {"symbol": "LOWUSDT", "volume": "100", "lastPrice": "1.0", "priceChangePercent": "1.0", "status": "TRADING"},
            {"symbol": "HIGHUSDT", "volume": "100000", "lastPrice": "1.0", "priceChangePercent": "10.0", "status": "TRADING"},
        ]
        ss = _make_ss()
        cfg = MagicMock(spec=["TRENDING_DISCOVERY_ENABLED", "TRENDING_MIN_VOLUME",
                               "TRENDING_MIN_PRICE", "TRENDING_MIN_PRICE_CHANGE_PERCENT",
                               "TRENDING_LIMIT"])
        cfg.TRENDING_DISCOVERY_ENABLED = True
        cfg.TRENDING_MIN_VOLUME = 0
        cfg.TRENDING_MIN_PRICE = 0
        cfg.TRENDING_MIN_PRICE_CHANGE_PERCENT = 0
        cfg.TRENDING_LIMIT = 50
        cfg.SYMBOL_UNIVERSE = None
        ec = _make_exchange(tickers=tickers)
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        assert result[0]["symbol"] == "HIGHUSDT"
        high_score = result[0]["score"]
        low_score = next(r for r in result if r["symbol"] == "LOWUSDT")["score"]
        assert high_score > low_score

    async def test_trending_discovery_handles_bad_ticker_data(self):
        tickers = [
            {"symbol": "BADUSDT", "volume": "NaN", "lastPrice": "x", "priceChangePercent": "", "status": "TRADING"},
            {"symbol": "GOODUSDT", "volume": "5000", "lastPrice": "1.0", "priceChangePercent": "3.0", "status": "TRADING"},
        ]
        ss = _make_ss()
        cfg = MagicMock(spec=["TRENDING_DISCOVERY_ENABLED", "TRENDING_MIN_VOLUME",
                               "TRENDING_MIN_PRICE", "TRENDING_MIN_PRICE_CHANGE_PERCENT",
                               "TRENDING_LIMIT"])
        cfg.TRENDING_DISCOVERY_ENABLED = True
        cfg.TRENDING_MIN_VOLUME = 0
        cfg.TRENDING_MIN_PRICE = 0
        cfg.TRENDING_MIN_PRICE_CHANGE_PERCENT = 0
        cfg.TRENDING_LIMIT = 50
        cfg.SYMBOL_UNIVERSE = None
        ec = _make_exchange(tickers=tickers)
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        symbols = [r["symbol"] for r in result]
        assert "BADUSDT" not in symbols
        assert "GOODUSDT" in symbols

    async def test_fallback_uses_get_exchange_info(self):
        exchange_info = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},
                {"symbol": "ETHUSDT", "status": "BREAK"},     # not TRADING
                {"symbol": "BNBBUSD", "status": "TRADING"},   # ends with BUSD, skipped
                {"symbol": "ADAUSDT", "status": "TRADING"},
            ]
        }
        ss = _make_ss()
        cfg = MagicMock(spec=[])  # no SYMBOL_UNIVERSE, no TRENDING_DISCOVERY_ENABLED
        cfg.SYMBOL_UNIVERSE = None
        cfg.TRENDING_DISCOVERY_ENABLED = False
        ec = _make_exchange(exchange_info=exchange_info)
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        symbols = [r["symbol"] for r in result]
        assert "BTCUSDT" in symbols
        assert "ADAUSDT" in symbols
        assert "ETHUSDT" not in symbols
        assert "BNBBUSD" not in symbols

    async def test_fallback_returns_empty_on_exchange_error(self):
        ss = _make_ss()
        cfg = MagicMock(spec=[])
        cfg.SYMBOL_UNIVERSE = None
        cfg.TRENDING_DISCOVERY_ENABLED = False
        ec = MagicMock()
        ec.get_exchange_info = AsyncMock(side_effect=RuntimeError("network error"))
        agent = SymbolDiscovererAgent(ss, cfg, exchange_client=ec)
        result = await agent.discover_symbols()
        assert result == []


@pytest.mark.asyncio
class TestSymbolDiscovererAgentRunLoop:

    async def test_run_loop_stops_on_stop_flag(self):
        ss = MagicMock()
        ss.is_discovery_complete = AsyncMock(return_value=False)
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), interval=0)
        agent.stop_flag = True  # Already flagged before loop starts

        # Patch asyncio.sleep to prevent blocking
        with patch("agents.symbol_discoverer_agent.asyncio.sleep", new=AsyncMock()):
            # The loop should see stop_flag=True immediately after first sleep(10)
            # We'll also make is_discovery_complete raise after first check to break loop
            call_count = [0]
            orig = ss.is_discovery_complete.side_effect

            async def _complete():
                call_count[0] += 1
                if call_count[0] >= 1:
                    return True
                return False

            ss.is_discovery_complete = _complete
            agent.stop_flag = False
            agent.run_once = AsyncMock()
            # Just verify run_loop terminates without hanging
            await asyncio.wait_for(agent.run_loop(), timeout=2.0)

    async def test_run_loop_handles_run_once_exception(self):
        ss = MagicMock()
        call_count = [0]

        async def _complete():
            call_count[0] += 1
            return call_count[0] >= 2

        ss.is_discovery_complete = _complete
        agent = SymbolDiscovererAgent(ss, MagicMock(spec=[]), interval=0)
        agent.run_once = AsyncMock(side_effect=RuntimeError("run_once failed"))
        with patch("agents.symbol_discoverer_agent.asyncio.sleep", new=AsyncMock()):
            await asyncio.wait_for(agent.run_loop(), timeout=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# SymbolScreener tests
# ══════════════════════════════════════════════════════════════════════════════

from agents.symbol_screener import SymbolScreener


def _make_screener(ss=None, ec=None, cfg=None, **kwargs):
    ss = ss or _make_ss()
    ec = ec or _make_exchange()
    cfg = cfg or _make_config()
    return SymbolScreener(ss, exchange_client=ec, config=cfg, **kwargs)


class TestSymbolScreenerIsLeveraged:
    def test_detects_up_token(self):
        s = _make_screener()
        assert s._is_leveraged_symbol("BTCUPUSDT") is True

    def test_detects_down_token(self):
        s = _make_screener()
        assert s._is_leveraged_symbol("ETHDOWNUSDT") is True

    def test_detects_bull_bear(self):
        s = _make_screener()
        assert s._is_leveraged_symbol("BNBBULLUSDT") is True
        assert s._is_leveraged_symbol("BNBBEARUSDT") is True

    def test_detects_3l_3s(self):
        s = _make_screener()
        assert s._is_leveraged_symbol("ETH3LUSDT") is True
        assert s._is_leveraged_symbol("ETH3SUSDT") is True

    def test_passes_normal_symbols(self):
        s = _make_screener()
        assert s._is_leveraged_symbol("BTCUSDT") is False
        assert s._is_leveraged_symbol("ETHUSDT") is False
        assert s._is_leveraged_symbol("ADAUSDT") is False

    def test_passes_empty_string(self):
        s = _make_screener()
        assert s._is_leveraged_symbol("") is False

    def test_does_not_match_when_base_too_short(self):
        # "UPUSDT" — base="UP", but len(base) must be > len(suffix)+1
        s = _make_screener()
        # "UPUSDT" would have base="UP" and suffix="UP", len("UP")=2 == len("UP")+1? No: needs > len(suf)+1=3
        # So should return False (too short)
        assert s._is_leveraged_symbol("UPUSDT") is False


@pytest.mark.asyncio
class TestSymbolScreenerPrefilter:

    async def test_passes_when_no_exchange_client(self):
        s = _make_screener()
        s.exchange_client = None
        assert await s._prefilter_symbol("BTCUSDT") is True

    async def test_rejects_non_trading_status(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {"status": "BREAK", "filters": {}}
        })
        s = _make_screener(ec=ec)
        assert await s._prefilter_symbol("BTCUSDT") is False

    async def test_rejects_when_symbol_info_returns_none(self):
        ec = _make_exchange(symbol_info_map={})  # returns None for all
        s = _make_screener(ec=ec)
        assert await s._prefilter_symbol("UNKNOWNUSDT") is False

    async def test_rejects_when_min_notional_exceeds_cap(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {
                "status": "TRADING",
                "filters": [
                    {"filterType": "MIN_NOTIONAL", "minNotional": "500"}
                ]
            }
        })
        cfg = _make_config(MAX_PER_TRADE_USDT=100.0)
        s = _make_screener(ec=ec, cfg=cfg)
        assert await s._prefilter_symbol("BTCUSDT") is False

    async def test_passes_when_min_notional_within_cap(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {
                "status": "TRADING",
                "filters": [
                    {"filterType": "MIN_NOTIONAL", "minNotional": "10"}
                ]
            }
        })
        cfg = _make_config(MAX_PER_TRADE_USDT=100.0)
        s = _make_screener(ec=ec, cfg=cfg)
        assert await s._prefilter_symbol("BTCUSDT") is True

    async def test_passes_when_no_min_notional_filter(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {"status": "TRADING", "filters": []}
        })
        s = _make_screener(ec=ec)
        assert await s._prefilter_symbol("BTCUSDT") is True

    async def test_handles_dict_format_filters(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {
                "status": "TRADING",
                "filters": {"MIN_NOTIONAL": "50"}
            }
        })
        cfg = _make_config(MAX_PER_TRADE_USDT=100.0)
        s = _make_screener(ec=ec, cfg=cfg)
        assert await s._prefilter_symbol("BTCUSDT") is True

    async def test_passes_when_require_trading_status_false(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {"status": "BREAK", "filters": {}}
        })
        cfg = _make_config(REQUIRE_TRADING_STATUS=False)
        s = _make_screener(ec=ec, cfg=cfg)
        assert await s._prefilter_symbol("BTCUSDT") is True


@pytest.mark.asyncio
class TestSymbolScreenerBuildExcludeSet:

    async def test_includes_config_exclude_list(self):
        cfg = _make_config(SYMBOL_EXCLUDE_LIST=["DOGEUSDT", "SHIBUSDT"])
        s = _make_screener(cfg=cfg)
        s.exchange_client = None
        result = await s._build_exclude_set()
        assert "DOGEUSDT" in result
        assert "SHIBUSDT" in result

    async def test_includes_accepted_symbols(self):
        ss = _make_ss(accepted={"BTCUSDT": {}, "ETHUSDT": {}})
        s = _make_screener(ss=ss)
        s.exchange_client = None
        result = await s._build_exclude_set()
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result

    async def test_includes_wallet_balances(self):
        ec = _make_exchange(balances={"BTC": {"free": 0.5, "locked": 0.0}})
        s = _make_screener(ec=ec)
        result = await s._build_exclude_set()
        assert "BTCUSDT" in result

    async def test_excludes_zero_balance_assets(self):
        ec = _make_exchange(balances={"BTC": {"free": 0.0, "locked": 0.0}})
        s = _make_screener(ec=ec)
        result = await s._build_exclude_set()
        assert "BTCUSDT" not in result

    async def test_excludes_base_currency_itself(self):
        ec = _make_exchange(balances={"USDT": {"free": 100.0, "locked": 0.0}})
        s = _make_screener(ec=ec)
        result = await s._build_exclude_set()
        # USDT itself should never appear as USDTUSDT
        assert "USDTUSDT" not in result

    async def test_handles_exchange_error_gracefully(self):
        ec = MagicMock()
        ec.get_account_balances = AsyncMock(side_effect=RuntimeError("timeout"))
        s = _make_screener(ec=ec)
        result = await s._build_exclude_set()
        assert isinstance(result, set)  # should return something, not raise


@pytest.mark.asyncio
class TestSymbolScreenerAtrPct:

    async def test_returns_atr_over_price(self):
        ss = _make_ss(atr=1.0)
        s = _make_screener(ss=ss)
        result = await s._atr_pct("BTCUSDT", 100.0)
        assert abs(result - 0.01) < 1e-9

    async def test_returns_zero_when_price_zero(self):
        s = _make_screener()
        result = await s._atr_pct("BTCUSDT", 0.0)
        assert result == 0.0

    async def test_returns_zero_when_atr_unavailable(self):
        ss = _make_ss(atr=0.0)
        s = _make_screener(ss=ss)
        result = await s._atr_pct("BTCUSDT", 100.0)
        assert result == 0.0

    async def test_returns_zero_on_exception(self):
        ss = _make_ss()
        ss.calc_atr = MagicMock(side_effect=RuntimeError("boom"))
        s = _make_screener(ss=ss)
        result = await s._atr_pct("BTCUSDT", 100.0)
        assert result == 0.0


@pytest.mark.asyncio
class TestSymbolScreenerRegimeFilter:

    async def test_passes_when_regime_filter_disabled(self):
        cfg = _make_config(SYMBOL_SCREENER_REGIME_FILTER=False)
        s = _make_screener(cfg=cfg)
        assert await s._passes_regime_filter("BTCUSDT") is True

    async def test_passes_normal_regime(self):
        cfg = _make_config(SYMBOL_SCREENER_REGIME_FILTER=True)
        ss = _make_ss()
        ss.get_volatility_regime = AsyncMock(return_value={"regime": "normal", "confidence": 0.9})
        s = _make_screener(ss=ss, cfg=cfg)
        assert await s._passes_regime_filter("BTCUSDT") is True

    async def test_rejects_low_confidence_sideways_regime(self):
        cfg = _make_config(SYMBOL_SCREENER_REGIME_FILTER=True)
        ss = _make_ss()
        ss.get_volatility_regime = AsyncMock(return_value={"regime": "sideways", "confidence": 0.5})
        s = _make_screener(ss=ss, cfg=cfg)
        assert await s._passes_regime_filter("BTCUSDT") is False

    async def test_passes_high_confidence_low_regime(self):
        cfg = _make_config(SYMBOL_SCREENER_REGIME_FILTER=True)
        ss = _make_ss()
        ss.get_volatility_regime = AsyncMock(return_value={"regime": "low", "confidence": 0.85})
        s = _make_screener(ss=ss, cfg=cfg)
        assert await s._passes_regime_filter("BTCUSDT") is True

    async def test_passes_when_no_regime_data(self):
        cfg = _make_config(SYMBOL_SCREENER_REGIME_FILTER=True)
        ss = _make_ss()
        ss.get_volatility_regime = AsyncMock(return_value=None)
        s = _make_screener(ss=ss, cfg=cfg)
        assert await s._passes_regime_filter("BTCUSDT") is True

    async def test_passes_on_exception(self):
        cfg = _make_config(SYMBOL_SCREENER_REGIME_FILTER=True)
        ss = _make_ss()
        ss.get_volatility_regime = AsyncMock(side_effect=RuntimeError("boom"))
        s = _make_screener(ss=ss, cfg=cfg)
        assert await s._passes_regime_filter("BTCUSDT") is True


@pytest.mark.asyncio
class TestSymbolScreenerPerformScan:

    async def test_returns_empty_when_no_exchange_client(self):
        s = _make_screener()
        s.exchange_client = None
        result = await s._perform_scan()
        assert result == []

    async def test_returns_empty_when_tickers_empty(self):
        ec = _make_exchange(tickers=[])
        s = _make_screener(ec=ec)
        result = await s._perform_scan()
        assert result == []

    async def test_filters_non_usdt_symbols(self):
        tickers = [
            _make_ticker("BTCBUSD", quote_volume=5_000_000),
            _make_ticker("BTCUSDT", quote_volume=5_000_000),
        ]
        ss = _make_ss(atr=0.01)  # 1% ATR
        ec = _make_exchange(tickers=tickers)
        # Exchange says TRADING for all
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        s = _make_screener(ss=ss, ec=ec)
        result = await s._perform_scan()
        symbols = [r["symbol"] for r in result]
        assert "BTCBUSD" not in symbols

    async def test_filters_leveraged_tokens(self):
        tickers = [
            _make_ticker("BTCUPUSDT", quote_volume=5_000_000),
            _make_ticker("BTCUSDT", quote_volume=5_000_000),
        ]
        ss = _make_ss(atr=0.01)
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        s = _make_screener(ss=ss, ec=ec)
        result = await s._perform_scan()
        symbols = [r["symbol"] for r in result]
        assert "BTCUPUSDT" not in symbols

    async def test_filters_below_min_volume(self):
        tickers = [
            _make_ticker("LOWUSDT", quote_volume=10_000),
            _make_ticker("HIGHUSDT", quote_volume=5_000_000),
        ]
        cfg = _make_config(SCREENER_MIN_VOLUME=50_000)
        ss = _make_ss(atr=0.01)
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        s = _make_screener(ss=ss, ec=ec, cfg=cfg)
        result = await s._perform_scan()
        symbols = [r["symbol"] for r in result]
        assert "LOWUSDT" not in symbols
        assert "HIGHUSDT" in symbols

    async def test_uses_atr_fallback_when_no_candidates_pass_threshold(self):
        tickers = [_make_ticker("BTCUSDT", quote_volume=5_000_000)]
        ss = _make_ss(atr=0.0)  # 0 ATR — will fail threshold
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        cfg = _make_config(SYMBOL_MIN_ATR_PCT=0.003, SYMBOL_ALLOW_ATR_FALLBACK=True)
        s = _make_screener(ss=ss, ec=ec, cfg=cfg)
        result = await s._perform_scan()
        # Fallback allows it through
        assert len(result) >= 1
        assert result[0]["symbol"] == "BTCUSDT"

    async def test_returns_empty_when_atr_fallback_disabled(self):
        tickers = [_make_ticker("BTCUSDT", quote_volume=5_000_000)]
        ss = _make_ss(atr=0.0)
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        cfg = _make_config(SYMBOL_MIN_ATR_PCT=0.003, SYMBOL_ALLOW_ATR_FALLBACK=False)
        s = _make_screener(ss=ss, ec=ec, cfg=cfg)
        result = await s._perform_scan()
        assert result == []

    async def test_respects_candidate_pool_size(self):
        tickers = [_make_ticker(f"SYM{i:02d}USDT", quote_volume=5_000_000 - i) for i in range(20)]
        ss = _make_ss(atr=0.05)
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        cfg = _make_config(SCREENER_MAX_PROPOSALS=5)
        s = _make_screener(ss=ss, ec=ec, cfg=cfg)
        result = await s._perform_scan()
        assert len(result) <= 5

    async def test_sorts_by_atr_desc_then_volume(self):
        tickers = [
            _make_ticker("HIGHUSDT", quote_volume=1_000_000),
            _make_ticker("LOWUSDT",  quote_volume=5_000_000),
        ]
        call_count = {"HIGHUSDT": 0, "LOWUSDT": 0}
        async def _atr(sym, tf, period):
            if sym == "HIGHUSDT":
                return 2.0  # higher ATR → should rank first
            return 0.5

        ss = _make_ss()
        ss.calc_atr = _atr
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        s = _make_screener(ss=ss, ec=ec)
        result = await s._perform_scan()
        assert result[0]["symbol"] == "HIGHUSDT"

    async def test_excludes_excluded_symbols(self):
        tickers = [
            _make_ticker("DOGEUSDT", quote_volume=5_000_000),
            _make_ticker("BTCUSDT", quote_volume=5_000_000),
        ]
        cfg = _make_config(SYMBOL_EXCLUDE_LIST=["DOGEUSDT"])
        ss = _make_ss(atr=0.05)
        ec = _make_exchange(tickers=tickers)
        ec.symbol_info = AsyncMock(return_value={"status": "TRADING", "filters": []})
        s = _make_screener(ss=ss, ec=ec, cfg=cfg)
        result = await s._perform_scan()
        symbols = [r["symbol"] for r in result]
        assert "DOGEUSDT" not in symbols


@pytest.mark.asyncio
class TestSymbolScreenerPropose:

    async def test_writes_to_symbol_proposals(self):
        ss = _make_ss()
        s = _make_screener(ss=ss)
        accepted = await s._propose("BTCUSDT", source="SymbolScreener", metadata={"volume": 1_000_000})
        assert accepted is True
        assert "BTCUSDT" in ss.symbol_proposals
        assert ss.symbol_proposals["BTCUSDT"]["source"] == "SymbolScreener"
        assert ss.symbol_proposals["BTCUSDT"]["metadata"]["volume"] == 1_000_000

    async def test_normalizes_symbol_to_uppercase(self):
        ss = _make_ss()
        s = _make_screener(ss=ss)
        await s._propose("btcusdt", source="test", metadata={})
        assert "BTCUSDT" in ss.symbol_proposals

    async def test_returns_false_when_no_shared_state(self):
        s = _make_screener()
        s.shared_state = None
        result = await s._propose("BTCUSDT", source="test", metadata={})
        assert result is False

    async def test_proposal_has_timestamp(self):
        ss = _make_ss()
        s = _make_screener(ss=ss)
        before = time.time()
        await s._propose("BTCUSDT", source="test", metadata={})
        after = time.time()
        ts = ss.symbol_proposals["BTCUSDT"]["ts"]
        assert before <= ts <= after


@pytest.mark.asyncio
class TestSymbolScreenerProcessAndAddSymbols:

    async def test_proposes_each_valid_candidate(self):
        ss = _make_ss()
        s = _make_screener(ss=ss)
        # prefilter passes by having no exchange client filters to check
        s.exchange_client = None
        candidates = [
            {"symbol": "BTCUSDT", "quote_volume": 1e6, "price_change_percent": 3.0, "atr_pct": 0.01},
            {"symbol": "ETHUSDT", "quote_volume": 500_000, "price_change_percent": 2.0, "atr_pct": 0.008},
        ]
        await s._process_and_add_symbols(candidates)
        assert "BTCUSDT" in ss.symbol_proposals
        assert "ETHUSDT" in ss.symbol_proposals

    async def test_handles_empty_candidate_list(self):
        ss = _make_ss()
        s = _make_screener(ss=ss)
        await s._process_and_add_symbols([])  # must not raise

    async def test_skips_candidates_failing_prefilter(self):
        ec = _make_exchange(symbol_info_map={
            "BTCUSDT": {"status": "BREAK", "filters": {}}
        })
        ss = _make_ss()
        s = _make_screener(ss=ss, ec=ec)
        candidates = [{"symbol": "BTCUSDT", "quote_volume": 1e6,
                       "price_change_percent": 3.0, "atr_pct": 0.01}]
        await s._process_and_add_symbols(candidates)
        assert "BTCUSDT" not in ss.symbol_proposals


@pytest.mark.asyncio
class TestSymbolScreenerLifecycle:

    async def test_start_creates_background_task(self):
        s = _make_screener()
        s.run_loop = AsyncMock()
        await s.start()
        assert s._task is not None
        assert not s._task.done()
        s._task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await s._task

    async def test_start_is_idempotent(self):
        s = _make_screener()
        s.run_loop = AsyncMock()
        await s.start()
        first_task = s._task
        await s.start()  # second call should not replace running task
        assert s._task is first_task
        s._task.cancel()

    async def test_stop_cancels_task(self):
        s = _make_screener()
        event = asyncio.Event()

        async def _long_loop():
            await event.wait()

        s.run_loop = _long_loop
        await s.start()
        assert s._task and not s._task.done()
        await s.stop()
        assert s._task is None

    async def test_run_once_is_reentrant_safe(self):
        """Concurrent run_once calls must not overlap."""
        s = _make_screener()
        # Keep exchange_client non-None so _perform_scan is reached

        started = []
        finished = []

        orig = s._perform_scan

        async def _slow_scan():
            started.append(1)
            await asyncio.sleep(0.1)
            finished.append(1)
            return []

        s._perform_scan = _slow_scan
        # Launch two concurrent run_once calls
        await asyncio.gather(s.run_once(), s.run_once())
        # Due to the lock, only one should have run _perform_scan
        assert len(started) == 1

    async def test_run_loop_respects_stop_event(self):
        s = _make_screener()
        s.run_once = AsyncMock()
        s._stop_event.set()
        with patch("agents.symbol_screener.asyncio.sleep", new=AsyncMock()):
            await asyncio.wait_for(s.run_loop(), timeout=2.0)
        # run_once may have been called once (first iteration checks stop event after run_once)
        # but the loop must terminate

    async def test_run_loop_sleeps_between_iterations(self):
        s = _make_screener()
        s.run_once = AsyncMock()
        sleep_calls = []

        async def _fake_sleep(sec):
            sleep_calls.append(sec)
            s._stop_event.set()  # stop after first sleep so loop exits

        with patch("agents.symbol_screener.asyncio.sleep", new=_fake_sleep):
            await asyncio.wait_for(s.run_loop(), timeout=2.0)

        assert len(sleep_calls) >= 1
