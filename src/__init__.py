"""Octivault Trader — layer-aligned namespace root.

Phase A of the directory-vs-layering alignment: this `src/` package mirrors
the 8-layer logical architecture (L0 → L8) without moving any files.

Each `src.lN_*` subpackage re-exports the modules assigned to layer N by
`scripts/check_layer_imports.py::FILE_LAYER_MAP`, using lazy import
(`__getattr__`) so there is **zero overhead and zero shim files** on disk.

Usage
-----
    # NEW (layer-aligned, recommended for new code):
    from src.l1_exchange import exchange_client
    from src.l3_portfolio import portfolio_manager
    from src.l4_execution import execution_manager

    # OLD (still works, will be deprecated in Phase D):
    from src.l1_exchange.exchange_client import ...
    from src.l3_portfolio.portfolio_manager import ...

Both paths resolve to the **same** module object — no duplication, no
state divergence. This is verified by `tests/test_layer_namespace.py`.
"""

from src._layer_index import LAYER_MODULES, layer_of  # noqa: F401

__all__ = ["LAYER_MODULES", "layer_of"]
