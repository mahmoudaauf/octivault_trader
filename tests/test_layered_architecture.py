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

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_eight_layer_contracts_importable():
    mod = importlib.import_module("src.l0_core.layer_contracts")
    for name in (
        "L1ExchangeContract",
        "L2WalletContract",
        "L3PortfolioContract",
        "L4ExecutionContract",
        "L5StrategyContract",
        "L6PolicyContract",
        "L7ObservabilityContract",
        "L8LifecycleContract",
        "EightLayerContractManager",
        "ALLOWED_DEPENDENCIES",
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
        LayerName.L6_GOVERNANCE,  # before L5: gate exists before intents
        LayerName.L5_STRATEGY,
        LayerName.L7_OBSERVABILITY,
    ]
    assert expected == L8LifecycleContract.BOOT_ORDER


def test_allowed_dependencies_are_acyclic_and_downward():
    from src.l0_core.layer_contracts import ALLOWED_DEPENDENCIES

    order = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
    rank = {layer_name: i for i, layer_name in enumerate(order)}
    # L7 reads everything below it; L8 boots everything; both fine.
    # For all other layers, every allowed dep must be at a strictly lower rank.
    for caller in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
        for callee in ALLOWED_DEPENDENCIES.get(caller, set()):
            assert rank[callee] < rank[caller], f"{caller} -> {callee} is not downward"


def test_validators_reject_missing_fields():
    from src.l0_core.layer_contracts import EightLayerContractManager, LayerName

    m = EightLayerContractManager()
    assert m.validate(
        LayerName.L4_EXECUTION, {"tickets": [], "fills": [], "cancels": [], "timestamp": 0.0}
    )
    assert not m.validate(LayerName.L4_EXECUTION, {"tickets": []})
    assert m.validate(
        LayerName.L1_EXCHANGE_IO,
        {
            "balances": {},
            "open_positions": [],
            "exchange_time_ms": 0,
            "rate_limit_remaining": 1000,
        },
    )
    assert not m.validate(LayerName.L1_EXCHANGE_IO, {"balances": {}})


def test_call_graph_helper():
    from src.l0_core.layer_contracts import EightLayerContractManager

    chk = EightLayerContractManager.is_call_allowed
    assert chk("L4", "L3")  # execution reads portfolio
    assert chk("L6", "L3")
    assert not chk("L3", "L4")  # portfolio must not call execution
    assert not chk("L5", "L1")  # strategy must not bypass to exchange
    assert not chk("L0", "L1")  # cross-cutting is pure


def test_ci_guard_passes():
    """The CI guard must exit 0 against the current workspace baseline."""
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_layer_imports.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert (
        res.returncode == 0
    ), f"layer-import guard failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
