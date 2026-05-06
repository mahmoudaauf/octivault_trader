"""
core_engine/production_bridge.py — Phase 8.1 Production Wiring Bridge
══════════════════════════════════════════════════════════════════════

Purpose
───────
Reuse the legacy `MasterSystemOrchestrator` (3,306 lines, ~50 components
already correctly constructed) to populate `app_ctx` for the 5 façade
engines. Avoids duplicating the construction logic.

Strategy
────────
1. Import legacy orchestrator class (deferred — not at module load).
2. Run its `check_prerequisites()` and `initialize_components()`.
3. Map its `self.<component>` attributes → `app_ctx[<key>]`.
4. Return the populated app_ctx + a handle to the orchestrator (for
   later shutdown).

⚠️  IMPORTANT
This bridge intentionally violates the façade-import rule because its
sole responsibility is *building* the app_ctx. The 5 engines and main.py
still talk only to the façade. This file is the seam where legacy
construction meets the new architecture.

Phase 8.2 will gradually replace this bridge with native construction
in `core_engine/integration.py`, layer by layer.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Phase 8.2.8: deprecation flag — emit warning at most once per process.
_DEPRECATION_EMITTED = False


def _emit_deprecation_warning() -> None:
    """Emit a DeprecationWarning at most once per process."""
    global _DEPRECATION_EMITTED
    if _DEPRECATION_EMITTED:
        return
    _DEPRECATION_EMITTED = True
    warnings.warn(
        "core_engine.production_bridge is deprecated (Phase 8.2.8). "
        "It loads the legacy '🎯_MASTER_SYSTEM_ORCHESTRATOR.py' to populate "
        "app_ctx and will be removed once the native production bootstrapper "
        "(core_engine.native.app_context.build_native_app_ctx) covers the "
        "full L0-L8 surface. See PHASE_8_2_8_PREP.md.",
        DeprecationWarning,
        stacklevel=3,
    )


# Legacy orchestrator filename has emoji + spaces — must import via spec
_LEGACY_FILE = "🎯_MASTER_SYSTEM_ORCHESTRATOR.py"


def _load_legacy_module() -> Any:
    """
    Load the legacy orchestrator module from its emoji-prefixed file.

    importlib can't handle the emoji filename via dotted import, so we
    use spec_from_file_location.
    """
    project_root = Path(__file__).resolve().parent.parent
    legacy_path = project_root / _LEGACY_FILE

    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Legacy orchestrator not found at {legacy_path}. "
            "Production bridge requires the legacy file to be present."
        )

    # Ensure project root is on sys.path so `src.l*_*` imports resolve
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    spec = importlib.util.spec_from_file_location(  # type: ignore[attr-defined]
        "octivault_legacy_orchestrator",
        str(legacy_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {legacy_path}")

    module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
    sys.modules["octivault_legacy_orchestrator"] = module
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────
# Component Mapping (legacy attr → app_ctx key)
# ─────────────────────────────────────────────────────────────────────
# Engines look up by these keys via app_ctx.get(<key>). Missing keys
# trigger graceful degradation in `core_engine/implementations.py`.
ATTR_TO_CTX_KEY: dict[str, str] = {
    # Layer 0 — shared infrastructure
    "shared_state": "shared_state",
    # Layer 1 — exchange I/O
    "exchange_client": "exchange_client",
    "order_cache_manager": "order_cache_manager",
    # Layer 2 — market data & wallet
    "market_data_feed": "market_data_feed",
    "balance_sync": "balance_manager",  # legacy uses balance_sync
    "balance_cache_updater": "balance_cache_updater",
    "market_regime_detector": "market_regime_detector",
    "volatility_regime": "volatility_regime",
    "heartbeat": "heartbeat",
    # Layer 3 — portfolio
    "portfolio_manager": "portfolio_manager",
    "position_manager": "position_manager",
    "symbol_manager": "symbol_manager",
    "three_bucket_manager": "three_bucket_manager",
    # Layer 4 — execution
    "execution_manager": "execution_manager",
    "tp_sl_engine": "tp_sl_engine",
    "safety_order_manager": "safety_order_manager",
    "recovery_engine": "recovery_engine",
    # Layer 5 — strategy
    "signal_manager": "signal_manager",
    "agent_manager": "signal_fusion",  # closest analog in legacy
    "meta_controller": "arbitration_engine",  # legacy controller acts as gate
    # Layer 6 — governance
    "risk_manager": "risk_manager",
    # Layer 7 — observability
    "health_monitor": "health_monitor",
    "performance_monitor": "performance_monitor",
    "alert_system": "alert_system",
    # Layer 8 — lifecycle
    "startup_orchestrator": "startup_orchestrator",
    "watchdog": "watchdog",
}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
async def build_production_app_ctx() -> tuple[dict[str, Any], Any]:
    """
    Build a fully wired app_ctx by reusing the legacy orchestrator.

    Returns
    -------
    (app_ctx, legacy_orchestrator)
        - app_ctx: dict ready for the 5 façade engines
        - legacy_orchestrator: handle for later shutdown / introspection

    Raises
    ------
    Any exception from the legacy orchestrator's `initialize_components()`.
    Caller should fall back to mock mode if this fails.

    Deprecated
    ----------
    Phase 8.2.8: this function will be removed once a native production
    bootstrapper exists. See ``core_engine.native.app_context.build_native_app_ctx``.
    """
    _emit_deprecation_warning()
    logger.info("🌉 Phase 8.1 Production Bridge — initializing…")
    legacy_mod = _load_legacy_module()

    Orchestrator = getattr(legacy_mod, "MasterSystemOrchestrator", None)
    if Orchestrator is None:
        raise ImportError("MasterSystemOrchestrator not found in legacy module")

    orch = Orchestrator()

    # Prerequisite check (config validity, API keys, dirs, live-trading approval)
    if not orch.check_prerequisites():
        raise RuntimeError("Legacy prerequisite check failed — see log above for details.")

    # Heavy init: ~9 layers, real Binance auth, balance sync, WS connect
    ok = await orch.initialize_components()
    if not ok:
        raise RuntimeError("Legacy initialize_components() returned False.")

    # Map legacy attributes to app_ctx keys
    app_ctx: dict[str, Any] = {}
    mapped, missing = 0, 0
    for legacy_attr, ctx_key in ATTR_TO_CTX_KEY.items():
        component = getattr(orch, legacy_attr, None)
        if component is not None:
            app_ctx[ctx_key] = component
            mapped += 1
        else:
            missing += 1
            logger.debug(f"  · legacy attr `{legacy_attr}` is None — `{ctx_key}` will be missing")

    # Always expose the orchestrator itself for last-resort access
    app_ctx["_legacy_orchestrator"] = orch
    app_ctx["_production_mode"] = True

    logger.info(
        f"🌉 Production Bridge wired: {mapped} components mapped, "
        f"{missing} missing (graceful degradation)"
    )
    return app_ctx, orch


async def shutdown_production_bridge(legacy_orchestrator: Any) -> None:
    """Best-effort shutdown of the legacy orchestrator's resources."""
    if legacy_orchestrator is None:
        return
    try:
        if hasattr(legacy_orchestrator, "shutdown"):
            await legacy_orchestrator.shutdown()
        elif hasattr(legacy_orchestrator, "stop"):
            await legacy_orchestrator.stop()
        else:
            # Try common cleanup hooks
            for attr in ("exchange_client", "market_data_feed", "balance_sync"):
                comp = getattr(legacy_orchestrator, attr, None)
                if comp and hasattr(comp, "close"):
                    with contextlib.suppress(Exception):
                        await comp.close()
        logger.info("🌉 Production Bridge shutdown complete")
    except Exception as e:
        logger.warning(f"🌉 Production Bridge shutdown error (non-fatal): {e}")
