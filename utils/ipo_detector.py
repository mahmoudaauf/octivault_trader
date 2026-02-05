import logging
import os
import time
import requests
from core.shared_state import SharedState

# Setup logger
logger = logging.getLogger("IPODetector")
logger.setLevel(logging.INFO)
log_path = "logs/utils/ipo_detector.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

class IPODetector:
    BINANCE_API = "https://api.binance.com/api/v3/exchangeInfo"
    POLL_INTERVAL = 60  # seconds

    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
        self.detected_symbols = set()
        logger.info("IPODetector initialized.")

    def poll_binance_for_new_listings(self):
        try:
            response = requests.get(self.BINANCE_API, timeout=10)
            response.raise_for_status()
            symbols_info = response.json().get('symbols', [])
            current_symbols = {item['symbol'] for item in symbols_info if item['status'] == 'TRADING'}

            new_symbols = current_symbols - self.detected_symbols
            if new_symbols:
                logger.info(f"New symbols detected: {new_symbols}")
                for symbol in new_symbols:
                    if self._is_symbol_valid(symbol, symbols_info):
                        self._notify_new_listing(symbol)
                self.detected_symbols.update(new_symbols)
            else:
                logger.debug("No new symbols detected this poll.")
        except requests.RequestException as e:
            logger.error(f"Binance API request failed: {e}")

    def _notify_new_listing(self, symbol):
        logger.info(f"Processing new IPO symbol: {symbol}")
        self.shared_state.ipo_candidates.add(symbol)
        self.shared_state.active_ipo_symbols.add(symbol)
        logger.info(f"{symbol} added to IPO candidates and active IPO symbols.")

    def _is_symbol_valid(self, symbol, symbols_info):
        # Optional filtering (liquidity, suspicious tokens, etc.)
        info = next((item for item in symbols_info if item['symbol'] == symbol), None)
        if info:
            base_asset = info['baseAsset']
            quote_asset = info['quoteAsset']
            if quote_asset != 'USDT':
                logger.info(f"Symbol {symbol} skipped (not USDT pair).")
                return False
            return True
        logger.warning(f"Symbol information not found for {symbol}.")
        return False

    def run(self):
        while True:
            self.poll_binance_for_new_listings()
            time.sleep(self.POLL_INTERVAL)
