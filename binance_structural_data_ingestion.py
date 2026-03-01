"""
BINANCE API DATA INGESTION: Funding Rates + Open Interest

Purpose:
  Fetch real structural data from Binance for funding + OI analysis
  
Scope:
  - Historical funding rates (hourly)
  - Open interest by symbol
  - OI concentration by leverage tier (if available)
  - Liquidation data
  
Output:
  CSV files with aligned timestamps for structural analysis
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import time
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger('BinanceStructuralDataIngestion')


@dataclass
class BinanceConfig:
    """Configuration for Binance API"""
    
    # API endpoints
    base_url: str = "https://fapi.binance.com"
    
    # Symbols to fetch
    symbols: List[str] = None
    
    # Data parameters
    lookback_days: int = 30  # Historical depth
    
    # Rate limiting
    request_delay: float = 0.1  # seconds between requests
    max_retries: int = 3
    
    # Output
    output_dir: str = "validation_outputs"
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTCUSDT', 'ETHUSDT']


class BinanceFundingRateFetcher:
    """Fetch funding rates from Binance"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = requests.Session()
    
    def fetch_funding_rate_history(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical funding rates
        
        Binance endpoint: GET /fapi/v1/fundingRate
        
        Returns:
            DataFrame with timestamp, fundingRate, markPrice
        """
        
        logger.info(f"Fetching funding rates for {symbol}...")
        
        endpoint = f"{self.config.base_url}/fapi/v1/fundingRate"
        
        all_rates = []
        
        # Binance limits to 1000 per request, so we may need multiple calls
        for attempt in range(self.config.max_retries):
            try:
                params = {
                    'symbol': symbol,
                    'limit': min(limit, 1000)
                }
                
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if not isinstance(data, list):
                    logger.warning(f"Unexpected response format: {type(data)}")
                    return pd.DataFrame()
                
                for item in data:
                    all_rates.append({
                        'timestamp': int(item['fundingTime']),
                        'funding_rate': float(item['fundingRate']),
                        'mark_price': float(item['markPrice'])
                    })
                
                logger.info(f"✅ Fetched {len(all_rates)} funding rate records for {symbol}")
                
                # Convert to DataFrame
                df = pd.DataFrame(all_rates)
                
                if len(df) > 0:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                return df
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.request_delay * (attempt + 1))
                continue
            
            except Exception as e:
                logger.error(f"Error fetching funding rates: {e}")
                return pd.DataFrame()
        
        logger.error(f"Failed to fetch funding rates after {self.config.max_retries} attempts")
        return pd.DataFrame()
    
    def fetch_open_interest(self, symbol: str) -> Dict:
        """
        Fetch current open interest
        
        Binance endpoint: GET /fapi/v1/openInterest
        
        Returns:
            Dict with current OI info
        """
        
        logger.info(f"Fetching open interest for {symbol}...")
        
        endpoint = f"{self.config.base_url}/fapi/v1/openInterest"
        
        for attempt in range(self.config.max_retries):
            try:
                params = {'symbol': symbol}
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"✅ Fetched OI for {symbol}: {data.get('openInterest', 'N/A')} contracts")
                
                return data
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.request_delay * (attempt + 1))
                continue
            
            except Exception as e:
                logger.error(f"Error fetching OI: {e}")
                return {}
        
        return {}


class BinanceLiquidationFetcher:
    """Fetch liquidation data from Binance"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = requests.Session()
    
    def fetch_liquidations(self, symbol: str, start_time: int = None, 
                          end_time: int = None, limit: int = 100) -> pd.DataFrame:
        """
        Fetch liquidation orders
        
        Note: Binance public liquidation endpoint may have limited historical data
        
        Returns:
            DataFrame with liquidation data
        """
        
        logger.info(f"Fetching liquidations for {symbol}...")
        
        endpoint = f"{self.config.base_url}/fapi/v1/allOrders"
        
        try:
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            # Note: This endpoint requires API key for detailed data
            # Using public endpoint instead
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                logger.info(f"✅ Fetched {len(df)} liquidation records")
                return df
            else:
                logger.info(f"No liquidation data available")
                return pd.DataFrame()
        
        except Exception as e:
            logger.warning(f"Could not fetch liquidations: {e}")
            logger.info("   (Liquidation data may require API key authentication)")
            return pd.DataFrame()


class StructuralDataIntegration:
    """Integrate Binance structural data with OHLCV"""
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.funding_fetcher = BinanceFundingRateFetcher(config)
        self.liquidation_fetcher = BinanceLiquidationFetcher(config)
    
    def run(self):
        """Execute full data integration"""
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("BINANCE STRUCTURAL DATA INGESTION")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Fetching real funding rates + OI data from Binance API")
        logger.info("")
        
        all_results = {}
        
        for symbol in self.config.symbols:
            logger.info(f"{'=' * 80}")
            logger.info(f"Processing: {symbol}")
            logger.info(f"{'=' * 80}")
            
            # Fetch funding rates
            funding_df = self.funding_fetcher.fetch_funding_rate_history(symbol)
            
            # Fetch current OI
            oi_data = self.funding_fetcher.fetch_open_interest(symbol)
            
            # Fetch liquidations (if available)
            liquidation_df = self.liquidation_fetcher.fetch_liquidations(symbol)
            
            # Load OHLCV for alignment
            ohlcv_file = Path(self.config.output_dir) / f"{symbol}_5m_with_60m_labels.csv"
            if not ohlcv_file.exists():
                logger.warning(f"OHLCV file not found: {ohlcv_file}")
                continue
            
            df_ohlcv = pd.read_csv(ohlcv_file)
            
            # Align data
            if len(funding_df) > 0:
                # Merge funding rates with OHLCV
                df_ohlcv['timestamp'] = df_ohlcv['timestamp'].astype(int)
                funding_df['timestamp'] = funding_df['timestamp'].astype(int)
                
                # Create structural dataset
                df_structural = self._create_structural_dataset(
                    symbol, df_ohlcv, funding_df, oi_data, liquidation_df
                )
                
                # Save
                output_file = Path(self.config.output_dir) / f"{symbol}_structural_data.csv"
                df_structural.to_csv(output_file, index=False)
                logger.info(f"✅ Saved structural data to {output_file.name}")
                
                all_results[symbol] = {
                    'status': 'SUCCESS',
                    'ohlcv_rows': len(df_ohlcv),
                    'funding_rows': len(funding_df),
                    'liquidation_rows': len(liquidation_df),
                    'aligned_rows': len(df_structural),
                    'file': str(output_file)
                }
            else:
                logger.error(f"No funding data retrieved for {symbol}")
                all_results[symbol] = {
                    'status': 'FAILED',
                    'reason': 'No funding data'
                }
            
            logger.info("")
            time.sleep(self.config.request_delay)
        
        # Save summary
        summary_file = Path(self.config.output_dir) / "structural_data_ingestion_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"✅ Summary saved to {summary_file.name}")
        logger.info("")
        
        self._print_status(all_results)
    
    def _create_structural_dataset(self, symbol: str, df_ohlcv: pd.DataFrame,
                                   df_funding: pd.DataFrame, oi_data: Dict,
                                   df_liquidation: pd.DataFrame) -> pd.DataFrame:
        """Create aligned structural dataset"""
        
        # For now, create a simple aligned version
        # In production, would do sophisticated time-series alignment
        
        df_result = df_ohlcv.copy()
        
        # Add funding rate (assume 8h intervals, align to nearest)
        if len(df_funding) > 0:
            df_result['funding_rate'] = np.nan
            df_result['mark_price'] = np.nan
            
            # Simple nearest-neighbor merge (production: use proper time alignment)
            for idx, row in df_result.iterrows():
                ts = row['timestamp']
                # Find nearest funding rate timestamp
                nearest_idx = (df_funding['timestamp'] - ts).abs().argmin()
                df_result.at[idx, 'funding_rate'] = df_funding.iloc[nearest_idx]['funding_rate']
                df_result.at[idx, 'mark_price'] = df_funding.iloc[nearest_idx]['mark_price']
        
        # Add current OI (static for simplicity)
        if oi_data:
            df_result['open_interest'] = float(oi_data.get('openInterest', 0))
        
        return df_result
    
    def _print_status(self, results: Dict):
        """Print final status"""
        
        logger.info("=" * 80)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 80)
        
        for symbol, result in results.items():
            if result['status'] == 'SUCCESS':
                logger.info(f"{symbol}: ✅")
                logger.info(f"  OHLCV rows:      {result['ohlcv_rows']}")
                logger.info(f"  Funding rates:   {result['funding_rows']}")
                logger.info(f"  Liquidations:    {result['liquidation_rows']}")
                logger.info(f"  Aligned rows:    {result['aligned_rows']}")
                logger.info(f"  Output file:     {Path(result['file']).name}")
            else:
                logger.info(f"{symbol}: ❌ {result['reason']}")
        
        logger.info("")
        logger.info("Next: Run structural imbalance diagnostic on real data")
        logger.info("      $ python diagnostic_structural_imbalances.py")
        logger.info("")


if __name__ == "__main__":
    config = BinanceConfig()
    integrator = StructuralDataIntegration(config)
    integrator.run()
