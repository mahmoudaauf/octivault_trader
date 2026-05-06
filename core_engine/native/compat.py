"""
Native compat stubs (Phase 8.2.8)

Provides null-object stubs for the six legacy ``app_ctx`` keys that
the façade engines reference but the native stack has not yet ported.

Background
----------
Every reference site in the 5 façade engines today follows this shape::

    xxx = self.app_ctx.get("xxx")
    if xxx:
        # await xxx.some_method(...)
        pass

That is — the keys act as **feature flags**. The bodies are commented
out pending the relevant native layer landing (Phase 8.3+). Therefore:

* Whether these stubs are *present* or *absent* in ``app_ctx`` has
  **no behavioral effect today**.
* Their value is **forward-compat**: when a façade uncomments its
  ``await xxx.method(...)`` line, the stub catches the call and logs
  it instead of crashing with AttributeError.

Stubbed keys
------------
``portfolio_manager``, ``position_manager``, ``tp_sl_engine``,
``safety_order_manager``, ``recovery_engine``, ``watchdog``.

The five "drop" decisions from ``PHASE_8_2_8_TRIAGE.md``
(``risk_manager``, ``signal_fusion``, ``arbitration_engine``,
``startup_orchestrator``, ``performance_monitor``) are deliberately
**not** registered — their behavior is already in native or genuinely
not needed.

Usage
-----
Opt-in via the ``compat=True`` argument to ``build_native_app_ctx``::

    app_ctx, orch = build_native_app_ctx(components, compat=True)

Tests of the pure factory should leave it ``False`` (default).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Generic null-stub
# ----------------------------------------------------------------------
class _NullStub:
    """
    Forward-compat null object.

    * Truthy (so ``if xxx:`` checks pass).
    * Any attribute access returns an async no-op that logs at DEBUG.
    * Repeated access returns the *same* method object so identity
      checks in tests are stable.
    """

    __slots__ = ("_name", "_method_cache")

    def __init__(self, name: str) -> None:
        self._name = name
        self._method_cache: dict[str, Any] = {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<NullStub {self._name!r}>"

    def __bool__(self) -> bool:
        return True

    def __getattr__(self, attr: str) -> Any:
        # __getattr__ only fires for missing attributes. Both __slots__
        # entries are set in __init__, so plain attribute access for
        # _name / _method_cache below never re-enters this method.
        # Use object.__getattribute__ to be defensive against subclassing.
        cache: dict[str, Any] = object.__getattribute__(self, "_method_cache")
        if attr in cache:
            return cache[attr]

        name: str = object.__getattribute__(self, "_name")

        async def _noop(*args: Any, **kwargs: Any) -> None:
            logger.debug(
                "compat-stub %s.%s called (args=%d, kwargs=%s) - returning None",
                name,
                attr,
                len(args),
                sorted(kwargs.keys()) if kwargs else [],
            )
            return None

        _noop.__name__ = f"{name}.{attr}"
        cache[attr] = _noop
        return _noop


# ----------------------------------------------------------------------
# Public registration helper
# ----------------------------------------------------------------------
COMPAT_KEYS: tuple[str, ...] = (
    # "portfolio_manager",   # Phase 8.3.7  — replaced by NativePortfolioManager
    # "position_manager",    # Phase 8.3.8  — replaced by NativePositionManager
    # "tp_sl_engine",        # Phase 8.3.9  — replaced by NativeTPSLEngine
    # "safety_order_manager",# Phase 8.3.10 — replaced by NativeSafetyOrderManager
    # "recovery_engine",     # Phase 8.3.11 — replaced by NativeRecoveryEngine
    "watchdog",
)


def make_compat_stubs() -> dict[str, _NullStub]:
    """Construct one ``_NullStub`` per compat key."""
    return {key: _NullStub(key) for key in COMPAT_KEYS}


def register_compat_stubs(app_ctx: dict[str, Any]) -> None:
    """
    Install compat stubs into ``app_ctx`` for any compat key not
    already populated. Pre-existing entries are left untouched.
    """
    for key, stub in make_compat_stubs().items():
        app_ctx.setdefault(key, stub)


__all__ = [
    "COMPAT_KEYS",
    "make_compat_stubs",
    "register_compat_stubs",
]
