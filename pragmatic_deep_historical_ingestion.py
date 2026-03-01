"""
Pragmatic Deep Historical Ingestion

Strategy: Fetch 6 months of 1H data (not 2 years of 5m)
- 6 months = 4380 hours = 4380 candles per symbol (fast!)
- 1H timeframe = cascades play out over hours/days (still visible)
- Real funding rate signals should be detectable
- Total API calls: ~5-10 per symbol (vs 200+)

Then run structural imbalance diagnostic on real 6-month window.
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PragmaticHistoricalFetcher:
    """Fetch 6-month 1H data from Binance (fast)"""
    
    def __init__(self, base_url="https://fapi.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def fetch_ohlcv_history(self, symbol: str, 
                            start_time: datetime, 
                            end_time: datetime,
                            interval: str = '1h',
                            batch_size: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        
        all_candles = []
        current_time = start_time
        endpoint = f"{self.base_url}/fapi/v1/klines"
        
        interval_ms = {
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        
        batch_count = 0
        
        while current_time < end_time:
            current_ms = int(current_time.timestamp() * 1000)
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ms,
                'limit': batch_size
            }
            
            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                batch_count += 1
                logger.info(f"  Batch {batch_count}: {symbol} - {len(data)} candles")
                
                batch_df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                batch_df['timestamp'] = pd.to_datetime(batch_df['open_time'].astype(int), unit='ms')
                batch_df = batch_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                batch_df['open'] = batch_df['open'].astype(float)
                batch_df['high'] = batch_df['high'].astype(float)
                batch_df['low'] = batch_df['low'].astype(float)
                batch_df['close'] = batch_df['close'].astype(float)
                batch_df['volume'] = batch_df['volume'].astype(float)
                
                all_candles.append(batch_df)
                
                latest_time = pd.to_datetime(batch_df['timestamp'].max())
                current_time = latest_time + timedelta(milliseconds=interval_ms[interval])
                
                if latest_time >= end_time:
                    break
                
                time.sleep(0.05)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API error: {e}")
                time.sleep(1)
                continue
        
        if not all_candles:
            return pd.DataFrame()
        
        df = pd.concat(all_candles, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"✅ {symbol}: {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
        
        return df
    
    def fetch_funding_rate_history_paginated(self, symbol: str, 
                                             start_time: datetime, 
                                             end_time: datetime,
                                             batch_size: int = 1000) -> pd.DataFrame:
        """Fetch funding rate history (paginated forward from start_time)"""
        
        all_records = []
        current_time = start_time
        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        batch_count = 0
        
        while current_time < end_time:
            current_ms = int(current_time.timestamp() * 1000)
            
            params = {
                'symbol': symbol,
                'startTime': current_ms,
                'limit': batch_size
            }
            
            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                batch_count += 1
                logger.info(f"  Batch {batch_count}: {symbol} - {len(data)} funding records")
                
                batch_df = pd.DataFrame(data)
                batch_df['timestamp'] = pd.to_datetime(batch_df['fundingTime'].astype(int), unit='ms')
                batch_df = batch_df[['timestamp', 'fundingRate']]
                batch_df.columns = ['timestamp', 'funding_rate']
                batch_df['funding_rate'] = batch_df['funding_rate'].astype(float)
                
                all_records.append(batch_df)
                
                latest_time = pd.to_datetime(batch_df['timestamp'].max())
                current_time = latest_time + timedelta(seconds=1)
                
                if latest_time >= end_time:
                    break
                
                time.sleep(0.05)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API error: {e}")
                time.sleep(1)
                continue
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.concat(all_records, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"✅ {symbol}: {len(df)} funding records ({df['timestamp'].min()} to {df['timestamp'].max()})")
        
        return df


class PragmaticDeepHistoricalIntegration:
    """Integrate funding rates with OHLCV over 6 months"""
    
    def __init__(self, symbols: list = None, output_dir: str = "validation_outputs"):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = PragmaticHistoricalFetcher()
    
    def run_pragmatic_ingestion(self, months: int = 6):
        """Fetch 6 months of 1H data (fast)"""
        
        logger.info("=" * 80)
        logger.info(f"PRAGMATIC DEEP HISTORICAL INGESTION ({months} MONTHS, 1H TIMEFRAME)")
        logger.info("=" * 80)
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30 * months)
        
        logger.info(f"Fetching from {start_time.date()} to {end_time.date()}")
        logger.info(f"Duration: {months} months ({(end_time - start_time).days} days)")
        logger.info("")
        
        results = {}
        
        for symbol in self.symbols:
            logger.info("=" * 80)
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 80)
            
            # Fetch OHLCV
            logger.info(f"Fetching OHLCV (1h) for {symbol}...")
            ohlcv_df = self.fetcher.fetch_ohlcv_history(
                symbol, start_time, end_time, interval='1h'
            )
            
            if ohlcv_df.empty:
                logger.error(f"Failed to fetch OHLCV for {symbol}")
                results[symbol] = {'status': 'FAILED', 'error': 'No OHLCV data'}
                continue
            
            # Fetch funding rates
            logger.info(f"Fetching funding rates for {symbol}...")
            funding_df = self.fetcher.fetch_funding_rate_history_paginated(
                symbol, start_time, end_time
            )
            
            if funding_df.empty:
                logger.error(f"Failed to fetch funding rates for {symbol}")
                results[symbol] = {'status': 'FAILED', 'error': 'No funding rate data'}
                continue
            
            # Align funding to OHLCV
            logger.info(f"Aligning {len(funding_df)} funding records to {len(ohlcv_df)} OHLCV candles...")
            
            merged_df = pd.merge_asof(
                ohlcv_df,
                funding_df,
                on='timestamp',
                direction='backward'
            )
            
            # Calculate features
            merged_df['funding_rate_pct'] = merged_df['funding_rate'] * 100
            merged_df['price_return'] = merged_df['close'].pct_change()
            merged_df['abs_return'] = merged_df['price_return'].abs()
            
            # Save to CSV
            output_file = self.output_dir / f"{symbol}_6month_1h_structural.csv"
            merged_df.to_csv(output_file, index=False)
            logger.info(f"✅ Saved to {output_file}")
            
            # Save summary
            summary_file = self.output_dir / f"{symbol}_6month_1h_summary.csv"
            merged_df[['timestamp', 'close', 'funding_rate_pct', 'price_return']].to_csv(
                summary_file, index=False
            )
            logger.info(f"✅ Summary saved to {summary_file}")
            
            results[symbol] = {
                'status': 'SUCCESS',
                'ohlcv_rows': len(ohlcv_df),
                'funding_rows': len(funding_df),
                'merged_rows': len(merged_df),
                'date_range': {
                    'start': str(merged_df['timestamp'].min()),
                    'end': str(merged_df['timestamp'].max()),
                },
                'output_file': str(output_file)
            }
            
            logger.info("")
        
        # Save summary
        summary_path = self.output_dir / "pragmatic_ingestion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"✅ Summary saved to {summary_path}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PRAGMATIC INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. python diagnostic_deep_structural_6m.py")
        logger.info("2. Analyze P(cascade | extreme funding) on 6-month real data")
        logger.info("3. Determine if signal exceeds 3× baseline")
        logger.info("")
        
        return results


if __name__ == '__main__':
    
    logger.info("")
    logger.info("🚀 PRAGMATIC DEEP HISTORICAL INGESTION")
    logger.info("Fetching 6 months of 1H data (fast)")
    logger.info("")
    
    integrator = PragmaticDeepHistoricalIntegration(
        symbols=['BTCUSDT', 'ETHUSDT'],
        output_dir='validation_outputs'
    )
    
    results = integrator.run_pragmatic_ingestion(months=6)
    
    logger.info("RESULTS:")
    logger.info(json.dumps(results, indent=2, default=str))
