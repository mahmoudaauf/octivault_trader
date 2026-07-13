"""
Weekly ML model retraining script.

Runs every Sunday automatically (via cron).
For each active symbol:
  1. Fetches last 90 days of OHLCV from Binance
  2. Trains a new GRU model using triple-barrier labels
  3. Validates on the held-out last 14 days
  4. Swaps only if prediction-only validation has positive net expectancy
     and profit factor after conservative spot trading costs
  5. Keeps old model as backup (.backup extension)

Designed to run independently from the live bot — safe to run while bot is active.
Results logged to logs/retrain_weekly.log
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "retrain_weekly.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("retrain_weekly")

# ── Constants ─────────────────────────────────────────────────────────────────
AGENT_NAME = "MLForecaster"
# Timeframe is env-configurable so we can train/test alternative horizons (e.g.
# RETRAIN_TIMEFRAME=1h) WITHOUT touching the live 5m models — model paths include
# the timeframe, so each horizon is a separate set of files.
TIMEFRAME = os.getenv("RETRAIN_TIMEFRAME", "5m")
KLINES_INTERVAL = TIMEFRAME
LOOKBACK_DAYS = int(os.getenv("RETRAIN_LOOKBACK_DAYS", "180"))  # training window
VALIDATION_DAYS = max(1, int(os.getenv("RETRAIN_VALIDATION_DAYS", "14") or 14))
MIN_TRAINING_ROWS = max(100, int(os.getenv("RETRAIN_MIN_TRAINING_ROWS", "500") or 500))
MAX_TRAINING_ROWS = max(256, int(os.getenv("RETRAIN_MAX_TRAINING_ROWS", "8000") or 8000))
EPOCHS = max(1, int(os.getenv("RETRAIN_EPOCHS", "30") or 30))
KLINES_LIMIT = 1000         # Binance max per request
EVAL_ROUND_TRIP_COST_PCT = float(os.getenv("RETRAIN_EVAL_ROUND_TRIP_COST_PCT", "0.0030") or 0.0030)
EVAL_MIN_SIGNALS = max(1, int(os.getenv("RETRAIN_EVAL_MIN_SIGNALS", "20") or 20))
EVAL_MIN_PROFIT_FACTOR = float(os.getenv("RETRAIN_EVAL_MIN_PROFIT_FACTOR", "1.20") or 1.20)
EVAL_MIN_EXPECTANCY_PCT = float(os.getenv("RETRAIN_EVAL_MIN_EXPECTANCY_PCT", "0.0") or 0.0)
EVAL_PROBABILITY_THRESHOLD = float(os.getenv("RETRAIN_EVAL_PROBABILITY_THRESHOLD", "0.55") or 0.55)


def _bars_per_day(tf: str) -> int:
    """Number of candles per day for a Binance interval string (5m→288, 1h→24)."""
    import re

    m = re.match(r"(\d+)([mhd])", tf.strip().lower())
    if not m:
        return 288
    n, unit = int(m.group(1)), m.group(2)
    minutes = n * {"m": 1, "h": 60, "d": 1440}[unit]
    return max(1, int(1440 / minutes))

# Symbols to retrain — auto-detected from existing models + env override
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "SOLUSDT", "MATICUSDT",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_symbols() -> list[str]:
    """Auto-detect symbols from existing model files, with env override."""
    env_syms = os.getenv("SYMBOLS", "").strip()
    if env_syms:
        syms = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        log.info("Symbols from env: %s", syms)
        return syms

    model_dir = ROOT / "models"
    found = []
    if model_dir.exists():
        for f in model_dir.glob(f"{AGENT_NAME.lower()}_*_{TIMEFRAME}.keras"):
            # mlforecaster_ADAUSDT_5m.keras → ADAUSDT
            parts = f.stem.split("_")
            if len(parts) >= 2:
                sym = parts[1].upper()
                if sym.endswith("USDT"):
                    found.append(sym)
    syms = sorted(set(found)) if found else DEFAULT_SYMBOLS
    log.info("Symbols auto-detected: %s", syms)
    return syms


def get_old_val_accuracy(model_path: Path) -> float:
    """Read val_accuracy from existing model metadata, return 0.0 if missing."""
    meta_path = model_path.with_suffix("").with_suffix("") \
        if model_path.suffix == ".keras" else model_path
    meta_path = Path(str(model_path).replace(".keras", "_metadata.pkl").replace(".h5", "_metadata.pkl"))
    if not meta_path.exists():
        return 0.0
    try:
        import pickle
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return float(meta.get("model_val_accuracy") or meta.get("val_accuracy") or 0.0)
    except Exception:
        return 0.0


async def fetch_klines(symbol: str, days: int) -> list[list]:
    """Fetch OHLCV klines from Binance REST API."""
    try:
        from binance import AsyncClient
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        client = await AsyncClient.create(api_key, api_secret)
        try:
            # Fetch in batches of 1000, going back `days` worth
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - (days * 24 * 3600 * 1000)
            all_klines = []
            batch_start = start_ms
            while batch_start < end_ms:
                klines = await client.get_klines(
                    symbol=symbol,
                    interval=KLINES_INTERVAL,
                    startTime=batch_start,
                    limit=KLINES_LIMIT,
                )
                if not klines:
                    break
                all_klines.extend(klines)
                batch_start = int(klines[-1][0]) + 1
                if len(klines) < KLINES_LIMIT:
                    break
            return all_klines
        finally:
            await client.close_connection()
    except Exception as e:
        log.error("Failed to fetch klines for %s: %s", symbol, e)
        return []


def klines_to_dataframe(klines: list[list]):
    """Convert Binance klines to DataFrame with OHLCV + engineered features (31 total).

    Produces the same feature set as MLForecaster._build_feature_df so that
    models trained here are compatible with live inference.
    """
    try:
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time").sort_index()
        df = df[["open", "high", "low", "close", "volume", "taker_buy_base"]].dropna(subset=["open", "high", "low", "close", "volume"])

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].fillna(0.0).clip(lower=0.0)

        # Returns
        df["returns_1"] = close.pct_change(1)
        df["returns_3"] = close.pct_change(3)
        df["returns_5"] = close.pct_change(5)

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=3).mean()
        df["atr_pct"] = atr / close.replace(0.0, np.nan)

        # Realized vol
        rv20 = df["returns_1"].rolling(20, min_periods=5).std(ddof=0)
        rv50 = df["returns_1"].rolling(50, min_periods=10).std(ddof=0)
        df["realized_vol_20"] = rv20
        df["realized_vol_50"] = rv50
        df["volatility_ratio"] = rv20 / rv50.replace(0.0, np.nan)

        # EMAs
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        df["ema9_dist"] = (close - ema9) / close.replace(0.0, np.nan)
        df["ema21_dist"] = (close - ema21) / close.replace(0.0, np.nan)
        df["ema50_dist"] = (close - ema50) / close.replace(0.0, np.nan)
        df["ema_slope"] = ema21.pct_change(3)
        df["ema_cross_dist"] = (ema9 - ema21) / close.replace(0.0, np.nan)

        # RSI (normalized 0..1)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
        rs = gain / loss.replace(0.0, np.nan)
        df["rsi14"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

        # ROC + MACD
        df["roc10"] = close.pct_change(10)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd - macd_signal) / close.replace(0.0, np.nan)

        # Volume
        def _rolling_zscore(s, window=50):
            mu = s.rolling(window, min_periods=5).mean()
            sigma = s.rolling(window, min_periods=5).std(ddof=0)
            return (s - mu) / sigma.replace(0.0, np.nan)

        df["volume_zscore"] = _rolling_zscore(volume)
        vol_base = volume.rolling(20, min_periods=5).mean()
        df["volume_spike_ratio"] = volume / vol_base.replace(0.0, np.nan)

        # Range
        range_pct = (high - low) / close.replace(0.0, np.nan)
        df["range_zscore"] = _rolling_zscore(range_pct)
        range_base = range_pct.rolling(20, min_periods=5).mean()
        df["rolling_range_expansion"] = range_pct / range_base.replace(0.0, np.nan)

        # Trend / vol flags (thresholds match MLForecaster defaults)
        atr_pct = df["atr_pct"]
        df["volatility_zscore"] = _rolling_zscore(atr_pct)
        df["trend_strength"] = (close - ema50).abs() / close.replace(0.0, np.nan)
        high_vol_flag = (atr_pct >= 0.025).astype(float)
        sideways_flag = (atr_pct <= 0.008).astype(float)
        df["trend_flag"] = ((high_vol_flag < 0.5) & (sideways_flag < 0.5)).astype(float)
        df["sideways_flag"] = sideways_flag
        df["high_vol_flag"] = high_vol_flag

        # Taker buy features
        taker_buy_vol = df["taker_buy_base"].fillna(0.0).clip(lower=0.0)
        total_vol = volume.replace(0.0, np.nan)
        raw_ratio = taker_buy_vol / total_vol
        df["taker_buy_ratio"] = (raw_ratio - 0.5).fillna(0.0)
        df["taker_buy_vol_zscore"] = _rolling_zscore(raw_ratio)

        # Drop helper columns; keep OHLCV + 26 engineered = 31 features
        cols_to_keep = [
            "open", "high", "low", "close", "volume",
            "returns_1", "returns_3", "returns_5", "atr_pct",
            "realized_vol_20", "realized_vol_50", "volatility_ratio", "volatility_zscore",
            "ema9_dist", "ema21_dist", "ema50_dist", "ema_slope", "ema_cross_dist",
            "rsi14", "roc10", "macd_hist",
            "volume_zscore", "volume_spike_ratio",
            "range_zscore", "rolling_range_expansion",
            "trend_strength", "trend_flag", "sideways_flag", "high_vol_flag",
            "taker_buy_ratio", "taker_buy_vol_zscore",
        ]
        df = df[cols_to_keep].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return df
    except Exception as e:
        log.error("klines_to_dataframe failed: %s", e)
        return None


async def retrain_symbol(symbol: str) -> dict:
    """Retrain model for one symbol. Returns result dict."""
    from src.l5_strategy.model_manager import build_model_path, model_exists
    from src.l5_strategy.model_trainer import ModelTrainer

    result = {"symbol": symbol, "ok": False, "reason": "", "old_acc": 0.0, "new_acc": 0.0, "swapped": False}

    log.info("━━━ %s: fetching %d days of %s klines ━━━", symbol, LOOKBACK_DAYS, KLINES_INTERVAL)
    klines = await fetch_klines(symbol, LOOKBACK_DAYS)
    if not klines:
        result["reason"] = "no_klines"
        log.warning("%s: skipped — no klines fetched", symbol)
        return result

    df = klines_to_dataframe(klines)
    if df is None or len(df) < MIN_TRAINING_ROWS:
        result["reason"] = f"insufficient_rows:{len(df) if df is not None else 0}"
        log.warning("%s: skipped — only %d rows", symbol, len(df) if df is not None else 0)
        return result

    log.info("%s: %d rows fetched. Splitting train/val...", symbol, len(df))

    # Split: last VALIDATION_DAYS for validation, rest for training
    val_rows = VALIDATION_DAYS * _bars_per_day(TIMEFRAME)  # candles per day for this timeframe
    val_rows = min(val_rows, len(df) // 4)  # max 25% for validation
    train_df = df.iloc[:-val_rows].tail(MAX_TRAINING_ROWS)
    val_df = df.iloc[-val_rows:]

    log.info("%s: train=%d rows, val=%d rows", symbol, len(train_df), len(val_df))

    # Get old model accuracy
    model_path = build_model_path(AGENT_NAME, symbol, version=TIMEFRAME)
    old_acc = get_old_val_accuracy(model_path)
    result["old_acc"] = old_acc

    try:
        # Backup existing model before overwriting
        if model_path.exists():
            backup = model_path.with_suffix(".backup.keras")
            shutil.copy2(model_path, backup)
            log.info("%s: old model backed up to %s", symbol, backup.name)

        trainer = ModelTrainer(
            symbol=symbol,
            timeframe=TIMEFRAME,
            agent_name=AGENT_NAME,
        )

        # Train in memory. Persistence is deferred until untouched holdout data
        # clears the cost-adjusted economic gate.
        train_result = trainer.train_model(
            df=train_df,
            task="supervised_learning",
            epochs=EPOCHS,
            max_rows=MAX_TRAINING_ROWS,
            save_model_artifact=False,
            return_metrics=True,
        )

        if not train_result or not train_result.get("ok"):
            result["reason"] = f"train_failed:{train_result.get('reason','unknown') if train_result else 'none'}"
            log.warning("%s: training failed — %s", symbol, result["reason"])
            # Restore backup if training failed
            backup = model_path.with_suffix(".backup.keras")
            if not model_path.exists() and backup.exists():
                shutil.copy2(backup, model_path)
                log.info("%s: restored backup after failed retrain", symbol)
            return result

        new_acc = float(train_result.get("val_accuracy") or train_result.get("accuracy") or 0.0)
        result["new_acc"] = new_acc
        log.info("%s: train complete — old_acc=%.3f new_acc=%.3f", symbol, old_acc, new_acc)

        # Prediction-only validation: this must never call fit or refit scalers.
        val_result = trainer.evaluate_holdout(
            val_df,
            probability_threshold=EVAL_PROBABILITY_THRESHOLD,
            round_trip_cost_pct=EVAL_ROUND_TRIP_COST_PCT,
        )
        val_acc = float(val_result.get("accuracy") or 0.0)
        signals = int(val_result.get("signals") or 0)
        expectancy_pct = float(val_result.get("expectancy_pct") or 0.0)
        profit_factor = float(val_result.get("profit_factor") or 0.0)
        result["holdout"] = val_result
        log.info(
            "%s: untouched holdout samples=%d signals=%d accuracy=%.3f expectancy=%+.4f%% PF=%.3f",
            symbol,
            int(val_result.get("samples") or 0),
            signals,
            val_acc,
            expectancy_pct,
            profit_factor,
        )

        _accept = bool(
            val_result.get("ok")
            and signals >= EVAL_MIN_SIGNALS
            and expectancy_pct > EVAL_MIN_EXPECTANCY_PCT
            and profit_factor >= EVAL_MIN_PROFIT_FACTOR
        )
        _baseline = (
            f"signals>={EVAL_MIN_SIGNALS}, expectancy>{EVAL_MIN_EXPECTANCY_PCT:.4f}%, "
            f"PF>={EVAL_MIN_PROFIT_FACTOR:.2f}, cost={EVAL_ROUND_TRIP_COST_PCT:.2%}"
        )

        if _accept:
            trainer._last_train_metrics["holdout"] = dict(val_result)
            trainer._last_train_metrics["val_accuracy"] = val_acc
            if not trainer.persist_model(model_path=model_path):
                backup = model_path.with_suffix(".backup.keras")
                if backup.exists():
                    shutil.copy2(backup, model_path)
                    log.info("%s: restored backup after persistence failure", symbol)
                result["reason"] = "persist_failed"
                log.warning("%s: holdout passed but persistence failed", symbol)
                return result
            result.update(swapped=True, ok=True, reason="swapped")
            log.info("✅ %s: new model deployed (%s)", symbol, _baseline)
        else:
            result["ok"] = True
            result["reason"] = "kept_old_model"
            log.info("⚠️  %s: kept old model; holdout failed gate (%s)", symbol, _baseline)

    except Exception as e:
        log.exception("%s: unexpected error during retrain: %s", symbol, e)
        result["reason"] = f"exception:{e}"

    return result


async def main():
    start_ts = time.time()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info("=" * 60)
    log.info("WEEKLY RETRAIN STARTED — %s", run_date)
    log.info("=" * 60)

    symbols = get_symbols()
    results = []

    for symbol in symbols:
        try:
            r = await retrain_symbol(symbol)
            results.append(r)
        except Exception as e:
            log.exception("Unhandled error for %s: %s", symbol, e)
            results.append({"symbol": symbol, "ok": False, "reason": str(e)})

    # Summary
    elapsed = time.time() - start_ts
    ok_count = sum(1 for r in results if r["ok"])
    swapped = sum(1 for r in results if r.get("swapped"))
    kept = sum(1 for r in results if r.get("reason") == "kept_old_model")
    failed = sum(1 for r in results if not r["ok"])

    log.info("=" * 60)
    log.info("WEEKLY RETRAIN COMPLETE in %.1fs", elapsed)
    log.info("  Symbols: %d total, %d ok, %d failed", len(results), ok_count, failed)
    log.info("  Models:  %d swapped (improved), %d kept (old was better)", swapped, kept)
    log.info("=" * 60)

    # Save JSON summary
    summary_path = LOG_DIR / "retrain_weekly_last.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_date": run_date,
            "elapsed_sec": round(elapsed, 1),
            "symbols_total": len(results),
            "ok": ok_count,
            "failed": failed,
            "swapped": swapped,
            "kept_old": kept,
            "results": results,
        }, f, indent=2)
    log.info("Summary saved to %s", summary_path)

    # Sync src/models/ → models/ so the live bot always reads the latest artifacts.
    src_dir = ROOT / "src" / "models"
    dst_dir = ROOT / "models"
    if src_dir.exists():
        synced = 0
        for src_file in src_dir.glob("mlforecaster_*"):
            dst_file = dst_dir / src_file.name
            if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                shutil.copy2(src_file, dst_file)
                synced += 1
        if synced:
            log.info("Synced %d model file(s) from src/models/ → models/", synced)


if __name__ == "__main__":
    asyncio.run(main())
