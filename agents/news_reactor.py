import asyncio
import logging
import os
from datetime import datetime
import aiohttp
import feedparser
from functools import partial # Import partial for to_thread

from utils.status_logger import log_component_status
from utils.shared_state_tools import inject_agent_signal

AGENT_NAME = "NewsReactor"
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.DEBUG) # Changed to DEBUG for detailed logging

log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s') # Added %(name)s to formatter
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)

# Removed hardcoded SYMBOL_KEYWORDS as per fix plan.
# SYMBOL_KEYWORDS = {
#     'BTCUSDT': ['bitcoin', 'btc'],
#     'ETHUSDT': ['ethereum', 'eth'],
#     'SOLUSDT': ['solana', 'sol'],
#     'XRPUSDT': ['ripple', 'xrp'],
#     'DOGEUSDT': ['dogecoin', 'doge'],
#     'BNBUSDT': ['binance coin', 'bnb'],
#     'SHIBUSDT': ['shiba inu', 'shib'],
# }

BINANCE_NEWS_RSS = "https://www.binance.com/en/news/rss"

class NewsReactor:

    def __init__(self, shared_state, config, **kwargs):
        self.shared_state = shared_state
        self.config = config
        self.name = kwargs.get("name", "NewsReactor")
        self.symbols = kwargs.get("symbols", list(self.shared_state.symbols.keys()))
        self.symbol = self.symbols[0] if self.symbols else "N/A"
        self.timeframe = kwargs.get("timeframe", "5m")
        
        # Optional injected modules
        self.execution_manager = kwargs.get("execution_manager")
        self.market_data_feed = kwargs.get("market_data_feed")
        self.tp_sl_engine = kwargs.get("tp_sl_engine")
        self.sentiment_stream = kwargs.get("sentiment_stream")
        self.meta_controller = kwargs.get("meta_controller")

        self.news_cache = set()
        log_component_status(self.name, "Initialized")
        logger.info(f"🚀 {self.name} initialized with timeframe {self.timeframe} and symbols: {self.symbols}")

    def extract_keywords(self, symbol):
        """
        Extracts keywords for a given symbol.
        This function ensures compatibility with any dynamic symbol by
        converting the symbol to lowercase and removing 'usdt'.
        """
        token = symbol.lower().replace('usdt', '')
        return [token]

    async def fetch_news(self):
        logger.debug(f"[{self.name}] Initiating news fetch from {BINANCE_NEWS_RSS}")
        try:
            feed = await asyncio.to_thread(feedparser.parse, BINANCE_NEWS_RSS)
            news_items = [{"title": entry.title, "summary": entry.summary} for entry in feed.entries]
            logger.debug(f"[{self.name}] Successfully fetched {len(news_items)} news items.")
            return news_items
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Failed to fetch or parse Binance RSS: {e}", exc_info=True)
            return []

    async def analyze_sentiment(self, text):
        logger.debug(f"[{self.name}] Analyzing sentiment for text (first 50 chars): '{text[:50]}...'")
        try:
            import openai
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.info(f"[{self.name}] OPENAI_API_KEY not found. Falling back to TextBlob.")
                from textblob import TextBlob
                sentiment_polarity = TextBlob(text).sentiment.polarity
                logger.debug(f"[{self.name}] TextBlob sentiment: {sentiment_polarity}")
                return sentiment_polarity

            openai.api_key = openai_api_key

            prompt = f"""Classify the sentiment of the following news headline as Positive, Negative, or Neutral:

{text}"""

            response = await asyncio.to_thread(
                partial(openai.ChatCompletion.create, model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
            )

            sentiment_label = response.choices[0].message.content.strip().lower()
            logger.info(f"[{self.name}] 🧠 OpenAI returned sentiment: {sentiment_label}")

            sentiment_score = {
                "positive": 1.0,
                "negative": -1.0,
                "neutral": 0.0
            }.get(sentiment_label, 0.0)
            logger.debug(f"[{self.name}] Mapped sentiment score: {sentiment_score}")
            return sentiment_score
        except Exception as e:
            logger.info(f"[{self.name}] ⚠️ Falling back to TextBlob due to OpenAI error: {e}", exc_info=True)
            try:
                from textblob import TextBlob
                sentiment_polarity = TextBlob(text).sentiment.polarity
                logger.debug(f"[{self.name}] TextBlob fallback sentiment: {sentiment_polarity}")
                return sentiment_polarity
            except Exception as tb_e:
                logger.error(f"[{self.name}] ❌ Error with TextBlob fallback: {tb_e}", exc_info=True)
                return 0.0

    async def run(self, symbol):
        self.symbol = symbol
        logger.info(f"🚀 {self.name} running for {symbol}")
        try:
            # Dynamically generate keywords for the symbol
            keywords = self.extract_keywords(symbol)
            logger.debug(f"[{self.name}] Keywords for {symbol}: {keywords}")

            news_items = await self.fetch_news()
            relevant_news = [item for item in news_items if any(k in item['title'].lower() or k in item['summary'].lower() for k in keywords)]
            logger.debug(f"[{self.name}] Found {len(relevant_news)} relevant news items for {symbol}.")

            if not relevant_news:
                logger.info(f"[{self.name}] No relevant news found for {symbol}. Returning hold.")
                return {"action": "hold", "confidence": 0.0, "reason": "No relevant news"}, 0.0

            total_sentiment_score = 0.0
            count = 0
            for item in relevant_news:
                cache_key = item['title'][:100]
                if cache_key in self.news_cache:
                    logger.debug(f"[{self.name}] Skipping cached news headline: {cache_key}")
                    continue
                self.news_cache.add(cache_key)
                sentiment = await self.analyze_sentiment(item['title'])
                total_sentiment_score += sentiment
                count += 1
                logger.debug(f"[{self.name}] News item sentiment for '{item['title'][:50]}...': {sentiment}")

            if count == 0:
                logger.info(f"[{self.name}] All relevant news already cached. Returning hold.")
                return {"action": "hold", "confidence": 0.0, "reason": "Cached news."}, 0.0

            score = total_sentiment_score / count
            logger.info(f"📰 {symbol} sentiment score: {score:.2f} based on {count} new news items")

            action = "buy" if score > 0.2 else "sell" if score < -0.2 else "hold"
            confidence = min(abs(score), 1.0)

            signal = {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "reason": f"OpenAI-enhanced Binance News Sentiment (Score: {score:.2f})",
                "timestamp": datetime.utcnow().isoformat()
            }

            await inject_agent_signal(self.shared_state, self.name, symbol, signal)
            logger.info(f"✅ OpenAI sentiment signal injected for {symbol}: {signal}")
            return signal, confidence

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error in {self.name}.run({symbol}): {e}", exc_info=True)
            return {"action": "hold", "confidence": 0.0, "reason": f"Error in NewsReactor: {e}"}, 0.0

    async def run_once(self):
        logger.info(f"[{self.name}] Entering run_once loop.")
        if not self.symbols:
            logger.info(f"[{self.name}] run_once: No symbols configured for {self.name}. Skipping agent run.")
            return

        for symbol in self.symbols:
            self.symbol = symbol
            logger.info(f"[{self.name}] run_once: Calling run() for symbol: {symbol}")
            await self.run(symbol)
            logger.info(f"[{self.name}] run_once: Finished run() for symbol: {symbol}")
        logger.info(f"[{self.name}] Exiting run_once loop.")

    async def calculate_sentiment_from_stream(self, symbol):
        logger.debug(f"[{self.name}] Attempting to calculate sentiment from stream for {symbol}.")
        try:
            score = float(self.shared_state.get_sentiment(symbol))
            logger.debug(f"[{self.name}] Stream sentiment for {symbol}: {score}")
            return score
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Failed to calculate sentiment from stream for {symbol}: {e}", exc_info=True)
            return 0.0
