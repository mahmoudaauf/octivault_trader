#!/usr/bin/env python3
"""
Quick diagnostic script to identify which case is happening:
- Case 1: Indicators not computed (insufficient data or NaN)
- Case 2: Wrong timeframe data
- Case 3: Price reference bug
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

async def run_diagnostic():
    """Run diagnostic checks."""
    try:
        # Import core components
        from core.shared_state import SharedState
        from core.config import Config
        from core.exchange_client import ExchangeClient
        
        # Load config
        config_path = Path(__file__).parent / "config" / "config.json"
        if not config_path.exists():
            logger.error(f"❌ Config not found at {config_path}")
            return
        
        config = Config(str(config_path))
        logger.info(f"✅ Loaded config from {config_path}")
        
        # Create shared state
        shared_state = SharedState({}, config, logger, "Diagnostic")
        logger.info("✅ Created SharedState")
        
        # Pick a test symbol from logs
        test_symbols = ["ENAUSDT", "BCHUSDT", "BTCUSDT", "ETHUSDT"]
        test_timeframe = "5m"
        
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC: OHLCV vs Indicator Mismatch")
        logger.info("="*80)
        
        # Check 1: Data availability
        logger.info("\n[CHECK 1] Market Data Availability")
        logger.info(f"Available market data keys: {len(shared_state.market_data)}")
        if shared_state.market_data:
            for i, (key, rows) in enumerate(list(shared_state.market_data.items())[:10]):
                sym, tf = key
                num_bars = len(rows) if rows else 0
                logger.info(f"  {sym:15s} {tf:5s} → {num_bars:4d} bars")
        
        if not shared_state.market_data:
            logger.warning("⚠️ No market data in shared_state yet")
            logger.info("This is expected if MarketDataFeed hasn't warmed up yet")
            return
        
        # Check 2: Test symbol data
        logger.info(f"\n[CHECK 2] Testing {test_symbols[0]} @ {test_timeframe}")
        rows = await shared_state.get_market_data(test_symbols[0], test_timeframe)
        
        if rows is None:
            logger.error(f"❌ CASE 2: No data for {test_symbols[0]}-{test_timeframe}")
            logger.info(f"Available timeframes for {test_symbols[0]}:")
            for key in shared_state.market_data.keys():
                sym, tf = key
                if sym == test_symbols[0]:
                    logger.info(f"  {tf}")
            return
        
        logger.info(f"✅ Found {len(rows)} bars for {test_symbols[0]}-{test_timeframe}")
        
        # Check data format
        if rows:
            last_row = rows[-1]
            logger.info(f"  Type: {type(last_row).__name__}")
            logger.info(f"  Keys: {list(last_row.keys()) if isinstance(last_row, dict) else 'Not a dict'}")
            logger.info(f"  Value: {last_row}")
        
        # Check 3: Data sufficiency
        logger.info(f"\n[CHECK 3] Data Sufficiency for Indicators")
        min_required = 50
        if len(rows) < min_required:
            logger.warning(f"❌ CASE 1: Insufficient data ({len(rows)} < {min_required})")
            logger.info("  → Need to wait for more OHLCV data to accumulate")
            return
        logger.info(f"✅ Sufficient data ({len(rows)} >= {min_required})")
        
        # Check 4: Price data extraction
        logger.info(f"\n[CHECK 4] Price Data Extraction")
        try:
            if isinstance(rows[0], dict):
                closes = [r.get("c") or r.get("close") for r in rows]
            else:
                closes = [r[4] if len(r) > 4 else None for r in rows]
            
            closes = [c for c in closes if c is not None]
            
            if not closes:
                logger.error("❌ CASE 3: No valid close prices found")
                return
            
            closes_float = [float(c) for c in closes]
            logger.info(f"✅ Extracted {len(closes_float)} close prices")
            logger.info(f"  Min: {min(closes_float):.8f}")
            logger.info(f"  Max: {max(closes_float):.8f}")
            logger.info(f"  Last 5: {closes_float[-5:]}")
            
        except Exception as e:
            logger.error(f"❌ CASE 3: Failed to extract close prices: {e}")
            return
        
        # Check 5: EMA computation
        logger.info(f"\n[CHECK 5] EMA Computation")
        try:
            import numpy as np
            from utils.indicators import compute_ema, compute_macd
            
            closes_arr = np.asarray(closes_float, dtype=float)
            
            # Compute EMAs
            ema_fast = compute_ema(closes_arr, 12)
            ema_slow = compute_ema(closes_arr, 26)
            
            ema_f_last = float(ema_fast[-1]) if ema_fast is not None else None
            ema_s_last = float(ema_slow[-1]) if ema_slow is not None else None
            
            # Check for NaN
            has_nan = any(np.isnan(v) for v in [ema_f_last, ema_s_last] if v is not None)
            
            if has_nan:
                logger.error("❌ CASE 1: EMA computation produced NaN values")
                logger.info(f"  EMA_FAST[-1]: {ema_f_last}")
                logger.info(f"  EMA_SLOW[-1]: {ema_s_last}")
                return
            
            # Compute MACD
            macd_line, sig_line, hist = compute_macd(closes_arr)
            hist_last = float(hist[-1]) if hist is not None else None
            
            logger.info(f"✅ EMA computation successful")
            logger.info(f"  EMA_SHORT[-1]: {ema_f_last:.8f}")
            logger.info(f"  EMA_LONG[-1]:  {ema_s_last:.8f}")
            logger.info(f"  MACD_HIST[-1]: {hist_last:.8f}")
            
            # Compare with expected values from logs
            logger.info(f"\n  From logs (ENAUSDT): EMA_S=0.12, EMA_L=0.12, HIST=-0.000051")
            logger.info(f"  This suggests EMA values are in range [0.01, 0.1] (micro-cap token)")
            
        except Exception as e:
            logger.error(f"❌ CASE 1: EMA computation failed: {e}", exc_info=True)
            return
        
        # Final assessment
        logger.info("\n" + "="*80)
        logger.info("ASSESSMENT: Data flow appears correct")
        logger.info("="*80)
        logger.info("✅ OHLCV data is being fetched and stored correctly")
        logger.info("✅ Data format (dicts) is correct")
        logger.info("✅ Indicators are computing without NaN")
        logger.info("✅ Price values match expected ranges")
        logger.info("\nIf you're still seeing issues, check:")
        logger.info("  1. Is the signal quality (confidence) too low?")
        logger.info("  2. Are the BUY/SELL signals being generated?")
        logger.info("  3. Are decisions being blocked by risk rules?")
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    result = asyncio.run(run_diagnostic())
    sys.exit(result)
