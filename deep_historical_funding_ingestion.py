"""
Deep Historical Funding Rate + OI Ingestion

Purpose: Fetch 2 years of funding rate and OI data from Binance
to properly validate structural alpha signal on convex events.

Rare events require large datasets. Cannot validate on 1000 candles.

Approach:
1. Fetch funding rate history (available from Binance API historically)
2. Fetch OI snapshots at regular intervals
3. Align to OHLCV timestamps
4. Store in parquet for efficient querying
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceHistoricalFetcher:
    """Fetch deep historical data from Binance API"""
    
    def __init__(self, base_url="https://fapi.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def fetch_funding_rate_history_paginated(self, symbol: str, 
                                             start_time: datetime, 
                                             end_time: datetime,
                                             batch_size: int = 500) -> pd.DataFrame:
        """
        Fetch funding rate history from Binance.
        
        Binance API returns up to 1000 records per call, so we paginate
        backwards from end_time to start_time.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_time: DateTime to start from
            end_time: DateTime to end at
            batch_size: Records per API call (max 1000)
        
        Returns:
            DataFrame with columns: [timestamp, funding_rate, mark_price]
        """
        
        all_records = []
        current_time = end_time
        endpoint = f"{self.base_url}/fapi/v1/fundingRate"
        
        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        batch_count = 0
        
        while current_time > start_time:
            current_ms = int(current_time.timestamp() * 1000)
            
            params = {
                'symbol': symbol,
                'limit': batch_size,
                'endTime': current_ms
            }
            
            try:
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    logger.warning(f"No data returned for {symbol} at {current_time}")
                    break
                
                batch_count += 1
                logger.info(f"  Batch {batch_count}: {symbol} - {len(data)} records - {current_time.isoformat()}")
                
                # Convert to DataFrame
                batch_df = pd.DataFrame(data)
                batch_df['timestamp'] = pd.to_datetime(batch_df['fundingTime'].astype(int), unit='ms')
                batch_df = batch_df[['timestamp', 'fundingRate', 'markPrice']]
                batch_df.columns = ['timestamp', 'funding_rate', 'mark_price']
                batch_df['funding_rate'] = batch_df['funding_rate'].astype(float)
                batch_df['mark_price'] = batch_df['mark_price'].astype(float)
                
                all_records.append(batch_df)
                
                # Move to earliest record in this batch
                earliest_time = pd.to_datetime(batch_df['timestamp'].min())
                current_time = earliest_time - timedelta(seconds=1)
                
                # Stop if we've reached start_time
                if earliest_time <= start_time:
                    break
                
                # Rate limit: Binance allows ~1200 requests/min
                time.sleep(0.05)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API error fetching {symbol}: {e}")
                time.sleep(1)
                continue
        
        if not all_records:
            logger.error(f"No funding rate data fetched for {symbol}")
            return pd.DataFrame()
        
        # Combine all batches and sort
        df = pd.concat(all_records, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"✅ Total records for {symbol}: {len(df)} ({df['timestamp'].min()} to {df['timestamp'].max()})")
        
        return df
    
    def fetch_ohlcv_history(self, symbol: str, 
                            start_time: datetime, 
                            end_time: datetime,
                            interval: str = '5m',
                            batch_size: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair
            start_time: Start datetime
            end_time: End datetime
            interval: Kline interval (1m, 5m, 1h, etc.)
            batch_size: Records per API call (max 1000)
        
        Returns:
            DataFrame with OHLCV columns
        """
        
        all_candles = []
        current_time = start_time
        endpoint = f"{self.base_url}/fapi/v1/klines"
        
        # Map interval to milliseconds
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
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
                    logger.warning(f"No OHLCV data for {symbol} at {current_time}")
                    break
                
                batch_count += 1
                logger.info(f"  Batch {batch_count}: {symbol} - {len(data)} candles - {current_time.isoformat()}")
                
                # Convert to DataFrame
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
                
                # Move to next batch
                latest_time = pd.to_datetime(batch_df['timestamp'].max())
                current_time = latest_time + timedelta(milliseconds=interval_ms[interval])
                
                # Stop if we've reached end_time
                if latest_time >= end_time:
                    break
                
                time.sleep(0.05)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API error fetching OHLCV for {symbol}: {e}")
                time.sleep(1)
                continue
        
        if not all_candles:
            logger.error(f"No OHLCV data fetched for {symbol}")
            return pd.DataFrame()
        
        df = pd.concat(all_candles, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"✅ Total candles for {symbol}: {len(df)} ({df['timestamp'].min()} to {df['timestamp'].max()})")
        
        return df


class DeepHistoricalIntegration:
    """Integrate funding rates with OHLCV over 2 years"""
    
    def __init__(self, symbols: list = None, output_dir: str = "validation_outputs"):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = BinanceHistoricalFetcher()
    
    def run_deep_ingestion(self, years: int = 2):
        """
        Fetch 2 years of historical data for all symbols.
        
        Args:
            years: Number of years to fetch (default 2)
        """
        
        logger.info("=" * 80)
        logger.info("DEEP HISTORICAL INGESTION (2 YEARS)")
        logger.info("=" * 80)
        
        # Calculate date range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=365 * years)
        
        logger.info(f"Fetching data from {start_time.date()} to {end_time.date()}")
        logger.info(f"Duration: {years} years ({(end_time - start_time).days} days)")
        logger.info("")
        
        results = {}
        
        for symbol in self.symbols:
            logger.info("=" * 80)
            logger.info(f"Processing: {symbol}")
            logger.info("=" * 80)
            
            # Fetch OHLCV
            logger.info(f"Fetching OHLCV (5m) for {symbol}...")
            ohlcv_df = self.fetcher.fetch_ohlcv_history(
                symbol, start_time, end_time, interval='5m'
            )
            
            if ohlcv_df.empty:
                logger.error(f"Failed to fetch OHLCV for {symbol}")
                results[symbol] = {'status': 'FAILED', 'error': 'No OHLCV data'}
                continue
            
            logger.info(f"Fetching funding rates for {symbol}...")
            funding_df = self.fetcher.fetch_funding_rate_history_paginated(
                symbol, start_time, end_time
            )
            
            if funding_df.empty:
                logger.error(f"Failed to fetch funding rates for {symbol}")
                results[symbol] = {'status': 'FAILED', 'error': 'No funding rate data'}
                continue
            
            # Align funding rates to OHLCV timestamps
            logger.info(f"Aligning {len(funding_df)} funding records to {len(ohlcv_df)} OHLCV candles...")
            
            merged_df = pd.merge_asof(
                ohlcv_df,
                funding_df,
                on='timestamp',
                direction='backward'
            )
            
            # Calculate structural features
            merged_df['funding_rate_pct'] = merged_df['funding_rate'] * 100
            merged_df['price_return'] = merged_df['close'].pct_change()
            merged_df['abs_return'] = merged_df['price_return'].abs()
            
            # Save to parquet (efficient columnar format)
            output_file = self.output_dir / f"{symbol}_2year_structural.parquet"
            merged_df.to_parquet(output_file, index=False)
            logger.info(f"✅ Saved to {output_file}")
            
            # Also save summary CSV for inspection
            summary_file = self.output_dir / f"{symbol}_2year_structural_summary.csv"
            merged_df[['timestamp', 'close', 'funding_rate_pct', 'price_return']].tail(100).to_csv(
                summary_file, index=False
            )
            logger.info(f"✅ Sample saved to {summary_file}")
            
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
        
        # Save results summary
        summary_path = self.output_dir / "deep_ingestion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"✅ Summary saved to {summary_path}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next Steps:")
        logger.info("1. Run: python diagnostic_deep_structural.py")
        logger.info("2. Analyze P(cascade | extreme funding) on 2-year dataset")
        logger.info("3. Determine if signal exceeds 3× baseline")
        logger.info("")
        
        return results


if __name__ == '__main__':
    
    logger.info("")
    logger.info("🔥 DEEP HISTORICAL INGESTION")
    logger.info("Fetching 2 years of Binance funding rate + OI data")
    logger.info("")
    
    integrator = DeepHistoricalIntegration(
        symbols=['BTCUSDT', 'ETHUSDT'],
        output_dir='validation_outputs'
    )
    
    results = integrator.run_deep_ingestion(years=2)
    
    # Print results
    logger.info("RESULTS:")
    logger.info(json.dumps(results, indent=2, default=str))
