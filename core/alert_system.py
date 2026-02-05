# core/alert_system.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    # Read from object attr or dict key
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

class AlertSystem:
    """
    P9-compliant AlertSystem facade.

    - Optional dependency; must never block or crash boot.
    - Accepts dict or object config.
    - If TELEGRAM_BOT_TOKEN (and optionally CHAT_ID) are missing, it disables itself cleanly.
    """

    component_name = "AlertSystem"

    def __init__(
        self,
        config: Any,
        telegram_client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.component_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

        # Read config from either dict or object
        self.token = _cfg_get(self.config, "TELEGRAM_BOT_TOKEN")
        self.chat_id = _cfg_get(self.config, "TELEGRAM_CHAT_ID")

        self.enabled = bool(self.token and self.chat_id)
        self._client = telegram_client  # optional external client/factory

        if self.enabled:
            self.logger.info("[AlertSystem] Telegram enabled (chat=%s)", str(self.chat_id))
        else:
            self.logger.info("[AlertSystem] Telegram NOT configured; alerts disabled")

    async def _send_telegram(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            # If an external client/factory is passed, use it; otherwise no-op.
            if self._client and hasattr(self._client, "send_message"):
                maybe = self._client.send_message(self.token, self.chat_id, text)
                if asyncio.iscoroutine(maybe):
                    await maybe
            else:
                # Implement your own light client or keep as best-effort no-op
                self.logger.debug("[AlertSystem] No telegram client bound; skip send")
        except Exception:
            self.logger.warning("[AlertSystem] send failed", exc_info=True)

    async def notify(self, text: str) -> None:
        # Public API used by AppContext and Watchdog paths
        await self._send_telegram(text)
