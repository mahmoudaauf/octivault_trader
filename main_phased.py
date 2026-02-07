
# -*- coding: utf-8 -*-
# main_phased.py - P9-aligned minimal runner

import os
import sys
import logging
import asyncio
import signal
import argparse
from typing import Any, Dict, Optional

# Optional: uvloop for better perf on Linux; safe no-op elsewhere
try:
    import uvloop  # type: ignore
    uvloop.install()
except Exception:
    pass

from core.app_context import AppContext, log_structured_error
from core.config import Config

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="octivault-main-phased", description="P9-aligned runner")
    parser.add_argument("--phase", type=int, default=None, help="Initialize up to this phase (default from env UP_TO_PHASE or 9).")
    parser.add_argument("--no-recovery", action="store_true", help="Disable recovery regardless of config or env.")
    return parser.parse_args()

def get_up_to_phase(ns: argparse.Namespace) -> int:
    # precedence: CLI --phase > env UP_TO_PHASE > default 9
    env_val = os.getenv("UP_TO_PHASE")
    try:
        env_phase = int(env_val) if env_val is not None else 9
    except Exception:
        env_phase = 9
    return ns.phase if ns.phase is not None else env_phase

def to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

def load_config() -> Config:
    # .env اختياري
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception as e:
        logger = logging.getLogger("Main")
        logger.warning("dotenv load failed: %s (Tip: pip install python-dotenv and ensure .env exists)", e)

    cfg = Config()
    cfg.env = os.getenv("ENV", "prod")
    cfg.recovery = {
        "enabled": to_bool(os.getenv("RECOVERY_ENABLED"), True),
        "snapshot_dir": os.getenv("RECOVERY_SNAPSHOT_DIR", "./snapshots"),
    }
    cfg.logging = {"level": os.getenv("LOG_LEVEL", "INFO")}
    return cfg

def _configure_logging() -> logging.Logger:
    # Configure root once; module loggers inherit handlers/level by default
    root_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root_level = getattr(logging, root_level_name, logging.INFO)
    if root_level_name not in logging._nameToLevel:
        # Early warn to stderr if invalid
        sys.stderr.write(f"[main] Unknown LOG_LEVEL={root_level_name}, defaulting to INFO\n")

    # Idempotent basicConfig (only applies if no handlers on root)
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    logger = logging.getLogger("Main")
    # Ensure “Main” doesn’t double-log if others attach root handlers later
    logger.propagate = True  # rely on root handler
    logger.debug("Logging configured at root level: %s", logging.getLevelName(logging.getLogger().level))
    return logger

async def _run():
    logger = _configure_logging()
    cfg = load_config()

    ns = _parse_args()
    # apply CLI overrides
    phase_max = get_up_to_phase(ns)
    if ns.no_recovery and isinstance(getattr(cfg, "recovery", None), dict):
        cfg.recovery["enabled"] = False
    logger.info("Startup: up_to_phase=%s, recovery_enabled=%s, env=%s",
                phase_max, getattr(cfg, "recovery", {}).get("enabled"), getattr(cfg, "env", ""))

    # Quick reflect of effective level
    logger.debug("Loaded config: %s", cfg)

    stop_event = asyncio.Event()

    def _signal_handler(signame: str):
        logger.info("Received %s — initiating graceful shutdown…", signame)
        stop_event.set()

    # Register POSIX signals when available
    loop = asyncio.get_running_loop()
    for s in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if isinstance(s, signal.Signals):
            try:
                loop.add_signal_handler(s, _signal_handler, s.name)
            except NotImplementedError:
                # Likely on Windows or restricted env
                pass

    ctx: Optional[AppContext] = None
    try:
        cfg.MODE = "LIVE"
        cfg.PHASE = phase_max
        ctx = AppContext(config=cfg, logger=logging.getLogger("AppContext"))
        # شغّل كل المراحل داخليًا (P1→P9)
        await ctx.initialize_all(up_to_phase=phase_max)
        
        # Wait for symbols to be ready before starting MDF (architecturally clean)
        await ctx.shared_state.wait_for_event("AcceptedSymbolsReady")
        
        # Ensure MarketDataFeed is running (canonical startup)
        if hasattr(ctx, 'market_data_feed') and ctx.market_data_feed and hasattr(ctx.market_data_feed, 'run'):
            asyncio.create_task(ctx.market_data_feed.run(), name="MarketDataFeed")
            logger.info("✅ MarketDataFeed started")
        
        logger.info("✅ Runtime plane is live (P9). Press Ctrl+C to stop.")

        # Wait here until a signal arrives
        await stop_event.wait()

    except asyncio.CancelledError:
        # Propagate cancel after logging if needed
        logger.debug("Run loop cancelled.")
        raise
    except Exception as e:
        # استخدم التوقيع الصحيح
        log_structured_error(
            e,
            context={"where": "main_phased:_run"},
            logger=logger,
            component="PhasedStartup",
            phase="P9",
            module="main_phased",
            event="bootstrap_fail",
        )
        raise
    finally:
        if ctx is not None:
            try:
                await ctx.shutdown(save_snapshot=True)
            except Exception as e:
                log_structured_error(
                    e,
                    context={"where": "main_phased:shutdown"},
                    logger=logger,
                    component="PhasedStartup",
                    phase="SHUTDOWN",
                    module="main_phased",
                    event="shutdown_fail",
                )

def main():
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Fallback for environments without signal handler registration
        pass

if __name__ == "__main__":
    main()
