"""L8 — Lifecycle: deterministic boot, single-layer restart, abort-on-fail."""
import asyncio
import pytest

from src.l0_core.layer_contracts import LayerName, L8LifecycleContract
from src.l8_lifecycle.layer_orchestrator import BootSequencer, LayerBootSpec


def _spec(layer, name, *, ok=True, started_log=None, stopped_log=None,
          health=lambda: True):
    async def start():
        if started_log is not None:
            started_log.append(name)
        return ok
    async def stop():
        if stopped_log is not None:
            stopped_log.append(name)
    return LayerBootSpec(layer=layer, name=name, start=start, stop=stop,
                         health=health)


def _all_layers(started_log=None, stopped_log=None):
    return [
        _spec(LayerName.L0_CROSS_CUTTING,      "l0", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L1_EXCHANGE_IO,        "l1", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L2_WALLET_MARKETDATA,  "l2", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L3_PORTFOLIO_STATE,    "l3", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L4_EXECUTION,          "l4", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L5_STRATEGY,           "l5", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L6_GOVERNANCE,         "l6", started_log=started_log, stopped_log=stopped_log),
        _spec(LayerName.L7_OBSERVABILITY,      "l7", started_log=started_log, stopped_log=stopped_log),
    ]


def test_l8_boot_order_matches_spec():
    started = []
    seq = BootSequencer()
    for s in _all_layers(started_log=started):
        seq.register(s)
    asyncio.get_event_loop().run_until_complete(seq.boot())
    # Spec: L0→L1→L2→L3→L4→L6→L5→L7
    assert started == ["l0", "l1", "l2", "l3", "l4", "l6", "l5", "l7"]


def test_l8_required_failure_aborts_boot():
    seq = BootSequencer()
    seq.register(_spec(LayerName.L0_CROSS_CUTTING, "l0"))
    seq.register(_spec(LayerName.L1_EXCHANGE_IO,   "l1", ok=False))
    seq.register(_spec(LayerName.L2_WALLET_MARKETDATA, "l2"))
    ok = asyncio.get_event_loop().run_until_complete(seq.boot())
    assert ok is False
    assert seq.system_health()["L1_EXCHANGE_IO"] == "DOWN"


def test_l8_single_layer_restart_isolated():
    started = []
    stopped = []
    seq = BootSequencer()
    for s in _all_layers(started_log=started, stopped_log=stopped):
        seq.register(s)
    asyncio.get_event_loop().run_until_complete(seq.boot())

    started.clear(); stopped.clear()
    ok = asyncio.get_event_loop().run_until_complete(
        seq.restart_layer("L4_EXECUTION")
    )
    assert ok
    # Only L4 was touched
    assert stopped == ["l4"]
    assert started == ["l4"]


def test_l8_shutdown_is_reverse_order():
    started = []
    stopped = []
    seq = BootSequencer()
    for s in _all_layers(started_log=started, stopped_log=stopped):
        seq.register(s)
    asyncio.get_event_loop().run_until_complete(seq.boot())
    asyncio.get_event_loop().run_until_complete(seq.shutdown())
    assert stopped == list(reversed(started))


def test_l8_health_degrades_when_probe_fails():
    seq = BootSequencer()
    seq.register(_spec(LayerName.L0_CROSS_CUTTING, "l0"))
    seq.register(_spec(LayerName.L1_EXCHANGE_IO,   "l1",
                       health=lambda: False))     # degraded probe
    asyncio.get_event_loop().run_until_complete(seq.boot())
    h = seq.system_health()
    assert h["L0_CROSS_CUTTING"] == "OK"
    assert h["L1_EXCHANGE_IO"] == "DEGRADED"
