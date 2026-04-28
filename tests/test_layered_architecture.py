"""
tests/test_layered_architecture.py
==================================

Smoke tests for the 8-layer logical architecture.
Verifies:
  1. All 8 layer contracts exist and validate sensibly.
  2. Allowed-call-graph is internally consistent.
  3. Boot order is deterministic and matches the specification.
  4. CI guard runs cleanly (exits 0) against the current workspace.
"""
from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_eight_layer_contracts_importable():
    mod = importlib.import_module("src.l0_core.layer_contracts")
    for name in (
        "L1ExchangeContract", "L2WalletContract", "L3PortfolioContract",
        "L4ExecutionContract", "L5StrategyContract", "L6PolicyContract",
        "L7ObservabilityContract", "L8LifecycleContract",
        "EightLayerContractManager", "ALLOWED_DEPENDENCIES",
    ):
        assert hasattr(mod, name), f"missing {name}"


def test_boot_order_is_deterministic():
    from src.l0_core.layer_contracts import L8LifecycleContract, LayerName
    expected = [
        LayerName.L0_CROSS_CUTTING,
        LayerName.L1_EXCHANGE_IO,
        LayerName.L2_WALLET_MARKETDATA,
        LayerName.L3_PORTFOLIO_STATE,
        LayerName.L4_EXECUTION,
        LayerName.L6_GOVERNANCE,        # before L5: gate exists before intents
        LayerName.L5_STRATEGY,
        LayerName.L7_OBSERVABILITY,
    ]
    assert L8LifecycleContract.BOOT_ORDER == expected


def test_allowed_dependencies_are_acyclic_and_downward():
    from src.l0_core.layer_contracts import ALLOWED_DEPENDENCIES
    order = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
    rank = {l: i for i, l in enumerate(order)}
    # L7 reads everything below it; L8 boots everything; both fine.
    # For all other layers, every allowed dep must be at a strictly lower rank.
    for caller in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
        for callee in ALLOWED_DEPENDENCIES.get(caller, set()):
            assert rank[callee] < rank[caller], (
                f"{caller} -> {callee} is not downward"
            )


def test_validators_reject_missing_fields():
    from src.l0_core.layer_contracts import EightLayerContractManager, LayerName
    m = EightLayerContractManager()
    assert m.validate(LayerName.L4_EXECUTION,
                      {"tickets": [], "fills": [], "cancels": [], "timestamp": 0.0})
    assert not m.validate(LayerName.L4_EXECUTION, {"tickets": []})
    assert m.validate(LayerName.L1_EXCHANGE_IO, {
        "balances": {}, "open_positions": [],
        "exchange_time_ms": 0, "rate_limit_remaining": 1000,
    })
    assert not m.validate(LayerName.L1_EXCHANGE_IO, {"balances": {}})


def test_call_graph_helper():
    from src.l0_core.layer_contracts import EightLayerContractManager
    chk = EightLayerContractManager.is_call_allowed
    assert chk("L4", "L3")           # execution reads portfolio
    assert chk("L6", "L3")
    assert not chk("L3", "L4")       # portfolio must not call execution
    assert not chk("L5", "L1")       # strategy must not bypass to exchange
    assert not chk("L0", "L1")       # cross-cutting is pure


def test_boot_sequencer_runs_in_order():
    from src.l0_core.layer_contracts import LayerName
    from src.l8_lifecycle.layer_orchestrator import BootSequencer, LayerBootSpec

    seq = BootSequencer()
    started: list[str] = []

    def make(layer: LayerName, name: str):
        async def _start() -> bool:
            started.append(name); return True
        async def _stop() -> None: pass
        return LayerBootSpec(layer=layer, name=name, start=_start, stop=_stop,
                             health=lambda: True)

    for layer, nm in [
        (LayerName.L0_CROSS_CUTTING,       "l0"),
        (LayerName.L1_EXCHANGE_IO,         "l1"),
        (LayerName.L2_WALLET_MARKETDATA,   "l2"),
        (LayerName.L3_PORTFOLIO_STATE,     "l3"),
        (LayerName.L4_EXECUTION,           "l4"),
        (LayerName.L5_STRATEGY,            "l5"),
        (LayerName.L6_GOVERNANCE,          "l6"),
        (LayerName.L7_OBSERVABILITY,       "l7"),
    ]:
        seq.register(make(layer, nm))

    ok = asyncio.get_event_loop().run_until_complete(seq.boot())
    assert ok
    # L6 must come before L5 (governance gate before strategy)
    assert started == ["l0", "l1", "l2", "l3", "l4", "l6", "l5", "l7"]
    health = seq.system_health()
    assert all(v == "OK" for v in health.values())


def test_boot_sequencer_aborts_on_required_failure():
    from src.l0_core.layer_contracts import LayerName
    from src.l8_lifecycle.layer_orchestrator import BootSequencer, LayerBootSpec

    seq = BootSequencer()

    async def _ok() -> bool:   return True
    async def _fail() -> bool: return False

    seq.register(LayerBootSpec(layer=LayerName.L0_CROSS_CUTTING, name="ok",
                               start=_ok, required=True))
    seq.register(LayerBootSpec(layer=LayerName.L1_EXCHANGE_IO, name="fail",
                               start=_fail, required=True))
    seq.register(LayerBootSpec(layer=LayerName.L2_WALLET_MARKETDATA, name="late",
                               start=_ok, required=True))

    ok = asyncio.get_event_loop().run_until_complete(seq.boot())
    assert ok is False
    health = seq.system_health()
    assert health["L1_EXCHANGE_IO"] == "DOWN"


def test_ci_guard_passes():
    """The CI guard must exit 0 against the current workspace baseline."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_layer_imports.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert res.returncode == 0, (
        f"layer-import guard failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
