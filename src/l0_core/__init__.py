"""L0 — Cross-cutting: contracts, config, errors, indicators, shared infra.

This namespace re-exports modules that physically still live in `core/`,
`utils/`, and root. See `src/__init__.py` for design rationale.
"""
from src._lazy import build_layer_namespace
__all__ = build_layer_namespace(__name__, "l0_core")
