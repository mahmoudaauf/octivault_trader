"""
Stub implementation of BinanceSentimentStream for system initialization.
This is a placeholder until the full sentiment analysis component is implemented.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class BinanceSentimentStream:
    """Stub sentiment stream that provides neutral sentiment scores."""

    def __init__(self, shared_state, config):
        self.shared_state = shared_state
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BinanceSentimentStream")
        self.logger.info("BinanceSentimentStream stub initialized (neutral sentiment)")

    async def get_sentiment(self, symbol: str) -> Optional[float]:
        """Return neutral sentiment (0.0) for all symbols."""
        return 0.0

    async def start(self):
        """Stub start method - does nothing."""
        self.logger.debug("BinanceSentimentStream.start() called (stub)")

    async def stop(self):
        """Stub stop method - does nothing."""
        self.logger.debug("BinanceSentimentStream.stop() called (stub)")