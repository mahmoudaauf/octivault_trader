"""Lazy re-export factory for `src.lN_*` namespace packages.

Each layer's `__init__.py` calls `build_layer_namespace(__name__, "lN_short")`
to install a module-level `__getattr__` that resolves attribute access to
the canonical module path declared in `src._layer_index.LAYER_MODULES`.

This avoids duplicate shim files while letting consumers write::

    from src.l1_exchange import exchange_client
    exchange_client.SomeClient(...)

The returned module object is identical (`is`) to the one obtained via the
legacy path (`import core.exchange_client`), so there is no state divergence.
"""

from __future__ import annotations
import importlib
import sys
from types import ModuleType
from typing import List

from src._layer_index import LAYER_MODULES


def build_layer_namespace(package_name: str, layer_key: str) -> List[str]:
    """Wire `__getattr__` + `__dir__` on the calling package module.

    Parameters
    ----------
    package_name : str
        The `__name__` of the calling `__init__.py` (e.g. ``"src.l1_exchange"``).
    layer_key : str
        Key into `LAYER_MODULES` (e.g. ``"l1_exchange"``).

    Returns
    -------
    list[str]
        The exposed short names — assign to `__all__` in the caller.
    """
    if layer_key not in LAYER_MODULES:
        raise KeyError(
            f"Unknown layer '{layer_key}'. Known: {sorted(LAYER_MODULES)}"
        )
    mapping = LAYER_MODULES[layer_key]
    pkg: ModuleType = sys.modules[package_name]

    def __getattr__(name: str) -> ModuleType:
        try:
            target = mapping[name]
        except KeyError as exc:
            raise AttributeError(
                f"module '{package_name}' has no attribute '{name}'. "
                f"Known: {sorted(mapping)}"
            ) from exc
        mod = importlib.import_module(target)
        # Cache on the package so repeated access is O(1) and `is` stable.
        setattr(pkg, name, mod)
        return mod

    def __dir__() -> List[str]:
        return sorted(set(list(mapping)) | set(vars(pkg)))

    pkg.__getattr__ = __getattr__       # type: ignore[attr-defined]
    pkg.__dir__ = __dir__                # type: ignore[attr-defined]
    return sorted(mapping)


__all__ = ["build_layer_namespace"]
