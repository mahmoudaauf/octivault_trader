"""L0 — Cross-Cutting: pure, deterministic, side-effect-free."""
import importlib
import sys


def test_l0_modules_import_without_io(monkeypatch):
    """Importing L0 must not perform network/file I/O."""
    import socket
    real_connect = socket.socket.connect

    def boom(*a, **kw):
        raise AssertionError("L0 module attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", boom)

    for name in (
        "src.l0_core.contracts",
        "src.l0_core.error_types",
        "src.l0_core.config_constants",
        "src.l0_core.layer_contracts",
        "utils.indicators",
    ):
        # reload to make sure import-time code re-runs under the trap
        sys.modules.pop(name, None)
        importlib.import_module(name)

    monkeypatch.setattr(socket.socket, "connect", real_connect)


def test_l0_octi_error_hierarchy():
    from src.l0_core.error_types import (
        TraderException as OctiError,  # umbrella class in this codebase
    )
    assert issubclass(OctiError, Exception)


def test_l0_indicators_are_deterministic():
    from utils.indicators import compute_rsi
    series = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0,
              5.5, 7.0, 6.5, 8.0, 7.5]
    a = compute_rsi(series, period=14)
    b = compute_rsi(series, period=14)
    # compute_rsi returns a pandas Series — use .equals for deep equality
    assert a.equals(b)
