# core/bootstrap_symbols.py
"""Bootstrap symbols for agents when discovery fails."""

import os

DEFAULT_SYMBOLS = {
    "BTCUSDT": {
        "symbol": "BTCUSDT", 
        "base": "BTC",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 2},
        "limits": {"amount": {"min": 0.001}, "cost": {"min": 10}},
    },
    "ETHUSDT": {
        "symbol": "ETHUSDT",
        "base": "ETH", 
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 2},
        "limits": {"amount": {"min": 0.01}, "cost": {"min": 10}},
    },
    "BNBUSDT": {
        "symbol": "BNBUSDT",
        "base": "BNB",
        "quote": "USDT", 
        "active": True,
        "precision": {"amount": 8, "price": 2},
        "limits": {"amount": {"min": 0.01}, "cost": {"min": 10}},
    },
    "SOLUSDT": {
        "symbol": "SOLUSDT",
        "base": "SOL",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 4},
        "limits": {"amount": {"min": 0.1}, "cost": {"min": 10}},
    },
    "XRPUSDT": {
        "symbol": "XRPUSDT",
        "base": "XRP",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 4},
        "limits": {"amount": {"min": 1}, "cost": {"min": 10}},
    },
    "ADAUSDT": {
        "symbol": "ADAUSDT",
        "base": "ADA",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 4},
        "limits": {"amount": {"min": 1}, "cost": {"min": 10}},
    },
    "DOGEUSDT": {
        "symbol": "DOGEUSDT",
        "base": "DOGE",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 4},
        "limits": {"amount": {"min": 1}, "cost": {"min": 10}},
    },
    "LINKUSDT": {
        "symbol": "LINKUSDT",
        "base": "LINK",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 2},
        "limits": {"amount": {"min": 0.1}, "cost": {"min": 10}},
    },
    "MATICUSDT": {
        "symbol": "MATICUSDT",
        "base": "MATIC",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 4},
        "limits": {"amount": {"min": 1}, "cost": {"min": 10}},
    },
    "AVAXUSDT": {
        "symbol": "AVAXUSDT",
        "base": "AVAX",
        "quote": "USDT",
        "active": True,
        "precision": {"amount": 8, "price": 2},
        "limits": {"amount": {"min": 0.01}, "cost": {"min": 10}},
    },
}

def _build_seed_symbols(shared_state, logger):
    """
    Resolve bootstrap symbol universe.

    Priority:
    1) Explicit config.SYMBOLS list (env-driven, operator-controlled)
    2) Static DEFAULT_SYMBOLS fallback
    """
    cfg = getattr(shared_state, "config", None)
    raw_syms = []
    if cfg is not None:
        # Support both object-style config (Config/SimpleNamespace) and dict-style config.
        if isinstance(cfg, dict):
            raw_syms = cfg.get("SYMBOLS", []) or []
        else:
            raw_syms = getattr(cfg, "SYMBOLS", []) or []

    # Fallback to shared_state _cfg accessor if available.
    if not raw_syms:
        try:
            if hasattr(shared_state, "_cfg"):
                raw_syms = shared_state._cfg("SYMBOLS", []) or []
        except Exception:
            raw_syms = []

    # Final fallback: direct environment variable (works even before config is attached).
    if not raw_syms:
        raw_syms = os.getenv("SYMBOLS", "") or ""

    # Normalize strings and iterables to canonical uppercase symbol list.
    if isinstance(raw_syms, str):
        cfg_syms = [s.strip().upper() for s in raw_syms.split(",") if s.strip()]
    else:
        cfg_syms = [str(s).strip().upper() for s in raw_syms if str(s).strip()]
    if not cfg_syms:
        return dict(DEFAULT_SYMBOLS)

    seed = {}
    for sym in cfg_syms:
        if sym in DEFAULT_SYMBOLS:
            seed[sym] = dict(DEFAULT_SYMBOLS[sym])
            continue

        # Fallback template for custom symbols not pre-baked in DEFAULT_SYMBOLS.
        base = sym[:-4] if sym.endswith("USDT") else sym
        seed[sym] = {
            "symbol": sym,
            "base": base,
            "quote": "USDT",
            "active": True,
            "precision": {"amount": 8, "price": 4},
            "limits": {"amount": {"min": 0.0001}, "cost": {"min": 10}},
        }

    logger.warning(
        "[Bootstrap] Using config.SYMBOLS seed universe (%d symbols): %s",
        len(seed),
        list(seed.keys()),
    )
    return seed


async def bootstrap_default_symbols(shared_state, logger):
    """Seed accepted_symbols with defaults if empty."""
    try:
        logger.warning("[Bootstrap] STARTING AGGRESSIVE BOOTSTRAP...")
        logger.warning("[Bootstrap] Checking current state:")
        logger.warning("[Bootstrap]   - shared_state type: %s", type(shared_state).__name__)
        logger.warning("[Bootstrap]   - has accepted_symbols attr: %s", hasattr(shared_state, 'accepted_symbols'))
        
        current = shared_state.accepted_symbols or {}
        logger.warning("[Bootstrap]   - current accepted_symbols count: %d", len(current))
        logger.warning("[Bootstrap]   - current keys: %s", list(current.keys()))
        
        if len(current) == 0:
            seed_symbols = _build_seed_symbols(shared_state, logger)
            logger.warning(
                "[Bootstrap] ⚠️  accepted_symbols is EMPTY! Seeding with %d symbols...",
                len(seed_symbols),
            )
            
            # FORCE: Directly set symbols on shared_state
            logger.warning("[Bootstrap] FORCE-SETTING symbols directly...")
            for sym, meta in seed_symbols.items():
                shared_state.accepted_symbols[sym] = meta
                logger.warning("[Bootstrap]   ✓ Direct set: %s", sym)
            
            logger.warning("[Bootstrap] Direct verification after force-set: %d symbols", len(shared_state.accepted_symbols))
            
            # ALSO call set_accepted_symbols for consistency
            logger.warning("[Bootstrap] Calling set_accepted_symbols (merge_mode=True)...")
            await shared_state.set_accepted_symbols(
                seed_symbols,
                source="bootstrap_default_symbols",
                merge_mode=True
            )
            logger.info(
                "[Bootstrap] ✅ Seeded %d symbols (merge_mode=True)",
                len(seed_symbols),
            )
            
            # Verify again
            final_count = len(shared_state.accepted_symbols)
            logger.warning("[Bootstrap] FINAL verification: %d symbols in accepted_symbols", final_count)
            if final_count == 0:
                logger.error("[Bootstrap] ❌ BOOTSTRAP FAILED! Symbols still empty after force-set!")
                return False
            
            # Mark symbol universe as ready
            try:
                if hasattr(shared_state, "accepted_symbols_ready_event"):
                    shared_state.accepted_symbols_ready_event.set()
                    logger.info("[Bootstrap] ✅ Marked accepted_symbols_ready_event")
            except Exception as e:
                logger.warning("[Bootstrap] Failed to set ready event: %s", e)
            
            return True
        else:
            logger.info("[Bootstrap] accepted_symbols already has %d symbols", len(current))
            
            # Ensure ready event is set
            try:
                if hasattr(shared_state, "accepted_symbols_ready_event"):
                    shared_state.accepted_symbols_ready_event.set()
            except Exception:
                pass
            
            return True
    except Exception as e:
        logger.error("[Bootstrap] Failed to seed symbols: %s", e, exc_info=True)
        return False
