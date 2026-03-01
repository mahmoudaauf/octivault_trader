import asyncio

from core.universe_rotation_engine import UniverseRotationEngine


class _SharedStateCandidatesStub:
    async def get_accepted_symbols(self):
        return {"ETHUSDT": {"source": "test"}}

    def get_positions_snapshot(self):
        # Sync method on purpose: mirrors current SharedState interface.
        return {"BTCUSDT": {"quantity": 0.25}}


class _SharedStateLiquidationStub:
    def __init__(self):
        self.positions = {
            "BTCUSDT": {"quantity": 0.42},
            "ETHUSDT": {"qty": 0.15},
            "XRPUSDT": {"current_qty": 0.0},
        }

    def get_positions_snapshot(self):
        return dict(self.positions)


class _MetaControllerStub:
    def __init__(self):
        self.received = []

    async def receive_intents(self, intents):
        self.received.extend(list(intents or []))


def test_collect_candidates_supports_sync_positions_snapshot():
    engine = UniverseRotationEngine(
        shared_state=_SharedStateCandidatesStub(),
        capital_governor=None,
        config={},
    )

    candidates = asyncio.run(engine._collect_candidates())
    assert set(candidates) == {"ETHUSDT", "BTCUSDT"}


def test_trigger_liquidation_uses_quantity_and_marks_liquidation_intents():
    meta = _MetaControllerStub()
    engine = UniverseRotationEngine(
        shared_state=_SharedStateLiquidationStub(),
        capital_governor=None,
        config={},
        meta_controller=meta,
    )

    asyncio.run(engine._trigger_liquidation(["BTCUSDT", "ETHUSDT", "XRPUSDT"]))

    # XRPUSDT has 0 qty, so only BTC/ETH should be emitted.
    assert len(meta.received) == 2
    syms = {i["symbol"] for i in meta.received}
    assert syms == {"BTCUSDT", "ETHUSDT"}
    for intent in meta.received:
        assert intent["action"] == "SELL"
        assert intent["tag"] == "liquidation"
        assert intent["execution_tag"] == "rotation_liquidation"
        assert intent["_is_liquidation"] is True
        assert intent["_is_rotation"] is True
        assert intent["_forced_exit"] is True
        assert float(intent["planned_qty"]) > 0.0
