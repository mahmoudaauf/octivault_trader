"""
Extended Historical Data Ingestion - 24 Months

Fetches 24 months of hourly OHLCV data to validate regime strategy across:
- Bull market periods
- Bear market periods  
- Sideways/ranging periods

This provides balanced dataset for representative walk-forward testing.
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExtendedHistoricalFetcher:
    """Fetch 24 months of hourly data from Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.output_dir = Path("validation_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def fetch_ohlcv(self, symbol: str, months: int = 24, interval: str = "1h"):
        """
        Fetch historical OHLCV data
        
        Binance limits: 1000 candles per request
        For 1H data: 1000 candles = ~42 days
        For 24 months: Need ~17 requests
        """
        
        logger.info(f"\nFetching {months} months of {interval} data for {symbol}")
        logger.info(f"Estimated requests: {int(np.ceil((months * 30 * 24) / 1000))}")
        
        all_data = []
        
        # Calculate start time (24 months ago from now)
        now = datetime.utcnow()
        start_time = now - timedelta(days=months * 30)
        current_time = start_time
        
        request_count = 0
        
        while current_time < now:
            try:
                # Binance expects timestamp in milliseconds
                timestamp_ms = int(current_time.timestamp() * 1000)
                
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': timestamp_ms,
                    'limit': 1000
                }
                
                response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    logger.info(f"No more data available")
                    break
                
                # Parse response
                # [time, open, high, low, close, volume, close_time, quote_asset_volume, ...]
                for candle in data:
                    all_data.append({
                        'timestamp': datetime.utcfromtimestamp(candle[0] / 1000),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                    })
                
                # Move to next batch (use last timestamp + 1 hour)
                last_time = datetime.utcfromtimestamp(data[-1][0] / 1000)
                current_time = last_time + timedelta(hours=1)
                
                request_count += 1
                logger.info(f"  Request {request_count}: Got {len(data)} candles, last: {last_time}")
                
                # Rate limit: max 10 requests per second
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"\nFetched {len(df)} candles total")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Actual span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        return df
    
    def add_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime detection to data"""
        
        logger.info("\nCalculating regimes...")
        
        result = df.copy()
        result['return'] = result['close'].pct_change()
        
        # Volatility (100-period rolling)
        lookback = 100
        result['volatility'] = result['return'].rolling(lookback).std()
        vol_low = result['volatility'].quantile(0.33)
        vol_high = result['volatility'].quantile(0.67)
        
        # Autocorrelation (momentum)
        result['momentum'] = result['return'].rolling(lookback).mean()
        result['autocorr_lag1'] = result['return'].rolling(lookback).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
        )
        
        # Regime classification
        result['volatility_regime'] = 'NORMAL'
        result.loc[result['volatility'] < vol_low, 'volatility_regime'] = 'LOW_VOL'
        result.loc[result['volatility'] > vol_high, 'volatility_regime'] = 'HIGH_VOL'
        
        momentum_sign = np.sign(result['momentum'])
        autocorr_positive = result['autocorr_lag1'] > 0.1
        result['trend_regime'] = 'MEAN_REVERT'
        result.loc[momentum_sign * autocorr_positive > 0, 'trend_regime'] = 'TRENDING'
        
        result['regime'] = result['volatility_regime'] + '_' + result['trend_regime']
        
        # Macro regime (SMA 200)
        result['sma_200'] = result['close'].rolling(200).mean()
        result['macro_trend'] = np.where(result['close'] > result['sma_200'], 'UPTREND', 'DOWNTREND')
        
        return result
    
    def save_data(self, df: pd.DataFrame, symbol: str):
        """Save to CSV"""
        
        output_file = self.output_dir / f"{symbol}_24month_1h_extended.csv"
        
        # Keep relevant columns
        cols_to_keep = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'return', 'volatility', 'momentum', 'autocorr_lag1',
            'volatility_regime', 'trend_regime', 'regime',
            'sma_200', 'macro_trend'
        ]
        
        df[cols_to_keep].to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")
        
        return output_file
    
    def print_summary(self, df: pd.DataFrame, symbol: str):
        """Print data summary"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"{symbol} - 24 MONTH SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info(f"Total candles: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Market metrics
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        logger.info(f"Total return: {100*total_return:+.2f}%")
        
        # Regime distribution
        logger.info(f"\nRegime distribution:")
        for regime, count in df['regime'].value_counts().items():
            pct = 100 * count / len(df)
            logger.info(f"  {regime}: {count} ({pct:.1f}%)")
        
        # Macro trend
        logger.info(f"\nMacro trend distribution:")
        for trend, count in df['macro_trend'].value_counts().items():
            pct = 100 * count / len(df)
            logger.info(f"  {trend}: {count} ({pct:.1f}%)")
        
        # Volatility
        logger.info(f"\nVolatility stats:")
        logger.info(f"  Mean: {100*df['return'].std():.2f}% per hour")
        logger.info(f"  Annual (est): {100*df['return'].std()*np.sqrt(252*24):.2f}%")
        
        # Large moves
        large_moves = (df['return'].abs() > 0.05).sum()
        logger.info(f"\nLarge moves (>5%): {large_moves} ({100*large_moves/len(df):.2f}%)")


def main():
    """Main execution"""
    
    logger.info("")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "EXTENDED HISTORICAL DATA INGESTION" + " "*24 + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "  Fetching 24 months of hourly data for validation..." + " "*26 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    fetcher = ExtendedHistoricalFetcher()
    
    # Fetch for both symbols
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        try:
            # Fetch data
            df = fetcher.fetch_ohlcv(symbol, months=24, interval='1h')
            
            # Add regimes
            df = fetcher.add_regimes(df)
            
            # Save
            fetcher.save_data(df, symbol)
            
            # Print summary
            fetcher.print_summary(df, symbol)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    logger.info("\n✅ Extended historical data ingestion complete!")


if __name__ == '__main__':
    main()
