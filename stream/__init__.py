"""
Octi AI Trading Bot - Stream Module
===================================

This module handles real-time data streaming and WebSocket connections.

Streaming:
  - Data Streams: Market data websockets
  - Order Updates: Real-time order status
  - Price Feeds: Tick data and quotes
  - Connection Management: Reconnection logic
"""

__all__ = [
    "DataStream",
    "OrderStream",
    "PriceFeed",
    "StreamManager",
]
