# core/notification_manager.py

import logging
import asyncio
from core.config import Config # Ensure Config is imported if not already

class NotificationManager:
    """
    The 'Command Center Communications' department.
    Handles internal and external notifications, alerts, and reporting.
    Manages communication channels like Telegram, email, or internal logging.
    """

    def __init__(self, config: Config):
        self.logger = logging.getLogger("NotificationManager")
        self.config = config

        # Access config values directly as attributes, not via .get()
        self.enabled = self.config.ALERT_SYSTEM_ENABLED
        self.telegram_bot_token = self.config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = self.config.TELEGRAM_CHAT_ID
        self.alert_interval = self.config.ALERT_INTERVAL # Now directly accessing ALERT_INTERVAL

        if not self.enabled:
            self.logger.info("📡 Notification System (Command Center Communications) is disabled.")
            return

        if self.telegram_bot_token and self.telegram_chat_id:
            self.logger.info("📡 Notification System (Command Center Communications) initialized. Telegram alerts enabled.")
            # Dummy import for now, replace with actual telegram library if used
            try:
                # Assuming you'd use a library like python-telegram-bot
                # from telegram import Bot
                # self.bot = Bot(token=self.telegram_bot_token)
                self.logger.info("Telegram Bot client would be initialized here.")
                self.bot = None # Placeholder if actual bot init is elsewhere or mock
            except ImportError:
                self.logger.error("❌ python-telegram-bot library not found. Telegram alerts will not function.")
                self.enabled = False
        else:
            self.logger.warning("⚠️ Telegram BOT_TOKEN or CHAT_ID not set in config. Telegram alerts will be disabled.")
            self.enabled = False

    async def send_alert(self, message: str, level: str = "INFO"):
        """Sends an alert through configured channels."""
        if not self.enabled:
            return

        full_message = f"🚨 Octivault AI Office Alert [{level}]: {message}"
        self.logger.log(getattr(logging, level.upper(), logging.INFO), full_message)

        if self.telegram_bot_token and self.telegram_chat_id and self.bot:
            try:
                # await self.bot.send_message(chat_id=self.telegram_chat_id, text=full_message)
                self.logger.debug(f"Simulating Telegram message send: {full_message}")
            except Exception as e:
                self.logger.error(f"❌ Failed to send Telegram alert: {e}")

    async def start_listener(self):
        """
        Starts a periodic task to check for and send accumulated alerts.
        In a more complex system, this might listen to a queue.
        For now, it's a simple periodic heartbeat/status sender.
        """
        if not self.enabled:
            self.logger.debug("Notification listener not started, system is disabled.")
            return

        self.logger.info(f"📡 Notification Listener operating on a {self.alert_interval}-second interval.")
        while True:
            await self.send_alert("Octivault Trader is operating normally. (Heartbeat)", level="DEBUG")
            await asyncio.sleep(self.alert_interval)
