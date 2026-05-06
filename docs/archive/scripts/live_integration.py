"""Backward-compatibility shim — DEPRECATED.

This module was moved to `src.l8_lifecycle.runners.live_integration` in Phase B
of the directory-vs-layering alignment. Please update imports::

    # OLD (works via this shim, will be removed in Phase D)
    import live_integration

    # NEW (recommended)
    from src.l8_lifecycle.runners import live_integration

The shim re-exports the *same* module object, so no state divergence.
"""
import warnings as _warnings

from src.l8_lifecycle.runners import live_integration as _real

_warnings.warn(
    "Importing 'live_integration' from project root is deprecated; use "
    "'src.l8_lifecycle.runners.live_integration' instead.",
    DeprecationWarning,
    stacklevel=2,
)

globals().update({k: v for k, v in vars(_real).items() if not k.startswith("_")})
