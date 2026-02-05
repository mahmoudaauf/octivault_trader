"""
Component: ModelTrainerAsync
Contract: aux:ModelTrainerAsync:v1.0.1
Phase: P9 (aux/offline)
Author: Octivault
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Union

import pandas as pd
from core.model_trainer import ModelTrainer

try:
    from core.component_status_logger import ComponentStatusLogger as CSL
except Exception:
    CSL = None  # optional

__version__ = "1.0.1"
__component__ = "aux.model_trainer_async"
__contract_id__ = "aux:ModelTrainerAsync:v1.0.1"
__phase__ = "P9"

logger = logging.getLogger("ModelTrainerAsync")

DEFAULT_TIMEFRAME = "5m"
DEFAULT_LIMIT = 500
MIN_ROWS_TO_TRAIN = 100
AGENT_NAME = "MLForecaster"


def _emit_status(component: str, status: str, detail: str = ""):
    if CSL and hasattr(CSL, "log_status"):
        try:
            CSL.log_status(component, status, detail)
        except Exception:
            pass


async def _safe_get_ohlcv(exchange, symbol: str, tf: str, limit: int = DEFAULT_LIMIT):
    # Prefer canonical positional; fall back to kw variants
    try:
        return await exchange.get_ohlcv(symbol, tf, limit)
    except TypeError:
        try:
            return await exchange.get_ohlcv(symbol, tf=tf, limit=limit)
        except TypeError:
            return await exchange.get_ohlcv(symbol, timeframe=tf, limit=limit)


def _normalize_to_trainer_schema(raw: Any) -> pd.DataFrame:
    """
    Return DataFrame with EXACT columns:
      ["timestamp","open","high","low","close","volume"]
    Accepts:
      - DataFrame with any of t/o/h/l/c/v or open/high/low/close/volume + time
      - list[list] as [t,o,h,l,c,v]
      - list[dict] / dict
    """
    if raw is None:
        return pd.DataFrame()

    # 1) Coerce to DataFrame
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    elif isinstance(raw, list):
        if not raw:
            return pd.DataFrame()
        first = raw[0]
        if isinstance(first, (list, tuple)):
            cols = ["t", "o", "h", "l", "c", "v"][: len(first)]
            df = pd.DataFrame(raw, columns=cols)
        elif isinstance(first, dict):
            df = pd.DataFrame(raw)
        else:
            return pd.DataFrame()
    elif isinstance(raw, dict):
        df = pd.DataFrame([raw])
    else:
        return pd.DataFrame()

    # 2) Rename common variants → canonical
    rename_map = {
        "t": "timestamp", "time": "timestamp", "timestamp": "timestamp", "T": "timestamp",
        "o": "open", "O": "open", "open": "open",
        "h": "high", "H": "high", "high": "high",
        "l": "low",  "L": "low",  "low": "low",
        "c": "close","C": "close","close": "close",
        "v": "volume","V": "volume","volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # 3) Keep only required cols; fail fast if missing
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()  # let caller decide to skip/log

    df = df[required]

    # 4) Dtypes: numeric; coerce; drop NaNs
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        return df

    # 5) Ensure timestamp is ms (heuristic) and sorted asc
    # If values look like seconds (e.g., < 1e12), convert to ms
    if df["timestamp"].median() < 1_000_000_000_000:  # < 1e12
        df["timestamp"] = (df["timestamp"] * 1000).astype("int64")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 6) Deduplicate on timestamp
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    return df


async def run_training_for_symbols(app_context):
    shared_state = app_context.shared_state
    exchange = app_context.exchange_client

    _emit_status(__component__, "Running", "Fetching accepted symbols")
    logger.info("📥 Fetching accepted symbols from SharedState...")

    symbols = await shared_state.get_accepted_symbols()
    if not symbols:
        logger.warning("⚠️ No accepted symbols found in SharedState. Abort.")
        _emit_status(__component__, "Idle", "No symbols")
        return

    if isinstance(symbols, dict):
        symbols = list(symbols.keys())
    elif not isinstance(symbols, (list, tuple, set)):
        symbols = [str(symbols)]
    symbols = list(dict.fromkeys(symbols))  # de-dup keep order

    logger.info(f"✅ Found {len(symbols)} symbols for training: {symbols}")

    trained = skipped = failed = 0

    for symbol in symbols:
        try:
            raw = await _safe_get_ohlcv(exchange, symbol, tf=DEFAULT_TIMEFRAME, limit=DEFAULT_LIMIT)
            df = _normalize_to_trainer_schema(raw)

            if df.empty:
                logger.warning(f"[{symbol}] ⚠️ Missing required OHLCV columns or empty after cleanup → skip.")
                skipped += 1
                continue

            if len(df) < MIN_ROWS_TO_TRAIN:
                logger.warning(f"[{symbol}] ⚠️ Only {len(df)} rows (<{MIN_ROWS_TO_TRAIN}) → skip.")
                skipped += 1
                continue

            trainer = ModelTrainer(symbol=symbol, timeframe=DEFAULT_TIMEFRAME, agent_name=AGENT_NAME)

            try:
                metrics = trainer.train_model(df)  # should NOT log "missing columns" anymore
            except Exception as e:
                failed += 1
                logger.exception(f"[{symbol}] ❌ ModelTrainer failed: {e}")
                continue

            trained += 1
            logger.info(f"✅ Model trained and saved for {symbol}")

        except Exception as e:
            failed += 1
            logger.exception(f"[{symbol}] ❌ Unexpected error: {e}")

    summary = f"Trained={trained}, Skipped={skipped}, Failed={failed}"
    logger.info(f"📊 ModelTrainerAsync summary → {summary}")
    _emit_status(__component__, "Running", summary)
