
# -*- coding: utf-8 -*-
# main_phased.py - P9-aligned minimal runner

import os
import sys
import logging
import asyncio
import signal
import argparse
from typing import Any, Dict, Optional
from pathlib import Path
import traceback
from utils.pid_manager import PIDManager

# Optional: uvloop for better perf on Linux; safe no-op elsewhere
try:
    import uvloop  # type: ignore
    uvloop.install()
except Exception:
    pass

# Load .env early - before any config or logging
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / "core" / ".env"
    load_dotenv(env_path, override=True)
except Exception as e:
    # Defer logging until after logging is configured
    pass

_APP_IMPORT_ERROR: Optional[Exception] = None
_CFG_IMPORT_ERROR: Optional[Exception] = None

try:
    from core.app_context import AppContext, log_structured_error
except Exception as _e:
    AppContext = None  # type: ignore
    _APP_IMPORT_ERROR = _e

    def log_structured_error(  # type: ignore
        e: Exception,
        context: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        component: str = "Bootstrap",
        phase: str = "INIT",
        module: str = "main_phased",
        event: str = "bootstrap_import_fail",
    ):
        target = logger or logging.getLogger("Main")
        target.error(
            "[%s] %s | component=%s phase=%s module=%s context=%s",
            event,
            e,
            component,
            phase,
            module,
            context or {},
            exc_info=True,
        )

try:
    from core.config import Config
except Exception as _e:
    Config = None  # type: ignore
    _CFG_IMPORT_ERROR = _e

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

def load_config() -> "Config":
    # .env is already loaded at module level
    if Config is None:
        raise RuntimeError(
            f"Failed to import core.config.Config at bootstrap: {_CFG_IMPORT_ERROR!r}"
        )
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
        sys.stderr.write(f"[main] Unknown LOG_LEVEL={root_level_name}, defaulting to INFO\n")

    log_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/app.log")

    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    except Exception as e:
        sys.stderr.write(f"[main] Failed to add file handler: {e}\n")

    logging.basicConfig(
        level=root_level,
        format=log_format,
        handlers=handlers
    )

    logger = logging.getLogger("Main")
    logger.propagate = True
    logger.debug("Logging configured at root level: %s", logging.getLevelName(logging.getLogger().level))
    return logger

async def _run():
    logger = _configure_logging()
    if _APP_IMPORT_ERROR is not None:
        logger.error(
            "[bootstrap] Failed to import core.app_context before runtime start: %r",
            _APP_IMPORT_ERROR,
        )
        logger.error("".join(traceback.format_exception(None, _APP_IMPORT_ERROR, _APP_IMPORT_ERROR.__traceback__)))
        raise RuntimeError(
            f"Bootstrap import failure for core.app_context: {_APP_IMPORT_ERROR!r}"
        )
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

    # Single-process guard: prevent duplicate live runtime processes.
    pid_manager = PIDManager("logs/octivault_trader.pid")
    if not pid_manager.acquire_lock():
        raise RuntimeError("Another Octivault runtime instance is already running.")

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

        # P4 in phased bootstrap is the canonical MarketDataFeed startup path.
        # Do not start MDF again here; duplicate run loops can desynchronize execution/logging.
        if phase_max >= 4:
            mdf = getattr(ctx, "market_data_feed", None)
            if mdf is None:
                raise RuntimeError("MarketDataFeed not initialized in phased bootstrap")
            mdf_running = bool(getattr(mdf, "_run_loop_entered", False))
            if not mdf_running:
                logger.warning(
                    "MarketDataFeed loop state not confirmed yet after phased bootstrap "
                    "(continuing without duplicate start)"
                )
            else:
                logger.info("MarketDataFeed already active from phased bootstrap")

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
        try:
            if pid_manager.is_locked():
                pid_manager.remove_pid_file()
        except Exception as e:
            log_structured_error(
                e,
                context={"where": "main_phased:pid_cleanup"},
                logger=logger,
                component="PhasedStartup",
                phase="SHUTDOWN",
                module="main_phased",
                event="pid_cleanup_fail",
            )

def main():
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Fallback for environments without signal handler registration
        pass

if __name__ == "__main__":
    main()
