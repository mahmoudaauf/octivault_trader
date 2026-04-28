"""Backward-compatibility shim — DEPRECATED.

This module was moved to `src.l8_lifecycle.runners.auto_recovery` in Phase B
of the directory-vs-layering alignment. Please update imports::

    # OLD (works via this shim, will be removed in Phase D)
    import auto_recovery

    # NEW (recommended)
    from src.l8_lifecycle.runners import auto_recovery

The shim re-exports the *same* module object, so no state divergence.
"""
import warnings as _warnings
from src.l8_lifecycle.runners import auto_recovery as _real

_warnings.warn(
    "Importing 'auto_recovery' from project root is deprecated; use "
    "'src.l8_lifecycle.runners.auto_recovery' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export every public attribute
globals().update({k: v for k, v in vars(_real).items() if not k.startswith("_")})
