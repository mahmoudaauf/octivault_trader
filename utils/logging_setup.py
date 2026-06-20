"""
Centralized logging setup for OctiVault.

Architecture
────────────
Root logger → RotatingFileHandler  (INFO+)  logs/run_latest.log  50 MB × 3 backups
             → StreamHandler        (WARNING+) stdout  (supervisor captures crashes only)

Sub-systems (e.g. core/, agents/) can call add_sub_logger() to get their own
rotating sink in addition to inheriting the root handlers.

Log volume reduction
────────────────────
High-frequency per-symbol PASS confirmations are logged at DEBUG at the call
site (arbitration_engine gate_4, symbol_discovery deferral). Only BLOCKs and
meaningful state changes remain at INFO so the main log doesn't grow >50 MB/day.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

# ── Size caps ────────────────────────────────────────────────────────────────
MAIN_LOG_MAX_BYTES    = 50 * 1024 * 1024   # 50 MB  — rotates at this threshold
MAIN_LOG_BACKUP_COUNT = 3                   # 3 backups → max 200 MB on disk
SUB_LOG_MAX_BYTES     = 20 * 1024 * 1024   # 20 MB per sub-logger
SUB_LOG_BACKUP_COUNT  = 2                   # 2 backups → max 60 MB per sub-logger

MAIN_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR     = Path("logs")


def setup_logging(
    main_log: str = "logs/run_latest.log",
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
) -> logging.Logger:
    """
    Wire the root logger with a rotating file sink and a sparse console sink.

    Call once at process startup in main.py — replaces logging.basicConfig().
    All subsequent getLogger() calls inherit these handlers automatically.

    Parameters
    ----------
    main_log      : path to the primary rotating log file
    level         : minimum level written to file (default INFO)
    console_level : minimum level echoed to stdout (default WARNING)
                    supervisor.sh captures stdout → only errors appear there
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear any handlers that basicConfig or earlier imports may have added
    root.handlers.clear()

    formatter = logging.Formatter(MAIN_FORMAT, datefmt=DATE_FORMAT)

    # ── Rotating file handler (primary, INFO+) ───────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        main_log,
        maxBytes=MAIN_LOG_MAX_BYTES,
        backupCount=MAIN_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=False,
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Console handler (WARNING+ only) ─────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    return logging.getLogger("octivault.main")


def add_sub_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Attach a dedicated rotating file handler to a named logger (sub-system sink).

    The logger still inherits root handlers (writes to run_latest.log as well),
    so this is additive. Use propagate=False only if you want exclusive routing.

    Parameters
    ----------
    name     : logger name, e.g. "core_engine.native.arbitration_engine"
    log_file : path to sub-logger's own rotating file
    level    : minimum level for this sink (default INFO)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(MAIN_FORMAT, datefmt=DATE_FORMAT)
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=SUB_LOG_MAX_BYTES,
        backupCount=SUB_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=False,
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


__all__ = ["setup_logging", "add_sub_logger"]
