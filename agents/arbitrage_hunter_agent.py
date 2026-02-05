import asyncio
import logging
from datetime import datetime
import os
import time
from collections import defaultdict
from functools import partial # Import partial for to_thread

from utils.status_logger import log_component_status
from utils.shared_state_tools import inject_agent_signal

AGENT_NAME = "ArbitrageHunter"
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.DEBUG)

log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

class ArbitrageHunter:
    def __init__(self, shared_state, exchange_client, config, execution_manager=None, symbols=None, name=AGENT_NAME, **kwargs): # Added **kwargs
        self.name = name
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.execution_manager = execution_manager
        self.config = config
        self.symbols = symbols if symbols is not None else getattr(config, "ARBITRAGE_SYMBOLS", list(self.shared_state.symbols.keys()))

        self.arb_threshold = float(getattr(config, "ARBITRAGE_THRESHOLD", 0.002))  # 0.2% default
        self.interval = getattr(config, "ARBITRAGE_POLL_INTERVAL", 5)

        self.symbol_cooldowns = defaultdict(float)

        self.symbol = self.symbols[0] if isinstance(self.symbols, list) and self.symbols else "N/A"
        self.name = self.__class__.__name__

        log_component_status(self.name, "Initialized")
        logger.info(f"\U0001F680 {self.name} initialized with {len(self.symbols)} symbols.")

    @staticmethod
    def extract_price(data, side):
        """
        Safely extracts the price from an order book data structure.
        Handles cases where 'asks' or 'bids' might be missing or empty,
        or if the format is unexpectedly a list instead of a dict.

        Args:
            data (dict or list): The order book data for a specific exchange.
            side (str): 'asks' for buy price, 'bids' for sell price.

        Returns:
            float or None: The extracted price, or None if not found or invalid.
        """
        if isinstance(data, dict):
            # Attempt to get the price from the dictionary structure
            # data.get(side, [[None]]) provides a default [[None]] if 'side' key is missing,
            # preventing KeyError and allowing subsequent indexing to fail gracefully.
            if data.get(side) and data[side] and isinstance(data[side], list) and data[side][0] and isinstance(data[side][0], list) and data[side][0][0] is not None:
                try:
                    return float(data[side][0][0])
                except (ValueError, TypeError):
                    return None
        elif isinstance(data, list):  # fallback if bad format, assuming it's directly the list of [price, quantity]
            if data and isinstance(data[0], list) and data[0][0] is not None:
                try:
                    return float(data[0][0])
                except (ValueError, TypeError):
                    return None
        return None

    def is_rate_limited(self, symbol):
        return time.time() - self.symbol_cooldowns[symbol] < self.interval

    async def run(self):
        logger.info(f"[{self.name}] \U0001F680 Starting ArbitrageHunter in continuous mode (will only run once in sequential test).")
        await self.scan_opportunities()
        logger.info(f"[{self.name}] ✅ ArbitrageHunter finished single scan.")

    async def run_once(self):
        logger.info(f"[{self.name}] Running on {self.symbol}")
        logger.info(f"[{self.name}] Entering run_once loop.")
        if not self.symbols:
            logger.info(f"[{self.name}] No symbols configured for {self.name}. Skipping agent run.")
            return
        await self.scan_opportunities()
        logger.info(f"[{self.name}] Exiting run_once loop.")

    async def scan_opportunities(self):
        logger.debug(f"[{self.name}] 🔎 Scanning for arbitrage opportunities across all configured symbols...")
        tasks = [self.check_symbol(symbol) for symbol in self.symbols if not self.is_rate_limited(symbol)]
        await asyncio.gather(*tasks)
        logger.debug(f"[{self.name}] Finished scanning all symbols for arbitrage opportunities.")

    def _perform_arbitrage_calculations_sync(self, symbol: str, order_books: dict, arb_threshold: float):
        logger.debug(f"[{self.name}] Starting synchronous arbitrage calculations for {symbol}.")
        opportunities = []

        for buy_exchange_name, buy_data in order_books.items():
            # Use the new extract_price method for safer price extraction
            buy_price = self.extract_price(buy_data, 'asks')
            if buy_price is None:
                logger.debug(f"[{self.name}] Skipping {symbol} on {buy_exchange_name}: No valid ask price.")
                continue

            for sell_exchange_name, sell_data in order_books.items():
                if buy_exchange_name == sell_exchange_name:
                    continue

                # Use the new extract_price method for safer price extraction
                sell_price = self.extract_price(sell_data, 'bids')
                if sell_price is None:
                    logger.debug(f"[{self.name}] Skipping sell on {sell_exchange_name}: No valid bid price for {symbol}.")
                    continue

                if sell_price > buy_price:
                    spread = (sell_price - buy_price) / buy_price
                    if spread >= arb_threshold:
                        logger.debug(f"[{self.name}] Found potential arbitrage: {symbol} Buy@{buy_price} on {buy_exchange_name}, Sell@{sell_price} on {sell_exchange_name}. Spread: {spread:.4f}")
                        opportunities.append({
                            "symbol": symbol,
                            "buy_from": buy_exchange_name,
                            "buy_price": buy_price,
                            "sell_to": sell_exchange_name,
                            "sell_price": sell_price,
                            "profit_margin": spread,
                            "timestamp": datetime.utcnow().isoformat(),
                            "type": "cross_exchange"
                        })

        logger.debug(f"[{self.name}] Finished synchronous arbitrage calculations for {symbol}. Found {len(opportunities)} opportunities.")
        return opportunities

    async def check_symbol(self, symbol):
        try:
            logger.debug(f"[{self.name}] Fetching order books for {symbol} across exchanges.")
            order_books = await self.shared_state.get_order_books(symbol)

            if not order_books:
                logger.warning(f"[{self.name}] No order books available for {symbol}. Skipping arbitrage check.")
                return

            if len(order_books) < 2:
                logger.debug(f"[{self.name}] Only one exchange's order book available for {symbol}. Cannot perform cross-exchange arbitrage.")
                return

            opportunities = await asyncio.to_thread(
                self._perform_arbitrage_calculations_sync, symbol, order_books, self.arb_threshold
            )

            self.symbol_cooldowns[symbol] = time.time()

            for opp in opportunities:
                # Corrected signal format as per user's request
                signal = {
                    "source": self.name,  # Required
                    "action": "arb-buy-sell",  # Using a more descriptive action for arbitrage
                    "confidence": round(opp['profit_margin'], 4),
                    "meta": opp,
                    "timestamp": opp['timestamp']
                }
                logger.info(f"[{self.name}] ✅ Arbitrage Opportunity: {signal['meta']['symbol']} ({signal['action']}) | Profit: {signal['confidence']:.4f} | Buy from {opp['buy_from']} @ {opp['buy_price']} | Sell to {opp['sell_to']} @ {opp['sell_price']}")
                # Corrected inject_agent_signal call as per user's request
                await inject_agent_signal(self.shared_state, self.name, symbol, signal)
                await self.shared_state.update_system_health(self.name, "Operational", f"Found arbitrage for {signal['meta']['symbol']} @ {signal['confidence']:.4f}")

            if not opportunities:
                logger.debug(f"[{self.name}] No profitable arbitrage opportunities found for {symbol}.")
                await self.shared_state.update_system_health(self.name, "Operational", f"No arbitrage found for {symbol}")

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error in check_symbol({symbol}): {e}", exc_info=True)
            await self.shared_state.inject_agent_signal(self.name, symbol, {
                "action": "hold",
                "confidence": 0.0,
                "reason": f"Error during arbitrage check: {e}",
                "timestamp": datetime.utcnow().isoformat()
            })
            await self.shared_state.update_system_health(self.name, "Error", f"Error checking arbitrage for {symbol}: {e}")
