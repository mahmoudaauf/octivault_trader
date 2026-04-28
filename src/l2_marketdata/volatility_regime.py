
import asyncio
import logging
import time
from typing import Any, Optional, Iterable

class VolatilityRegimeDetector:
    """
    P9 Stub: Volatility Regime Detector.
    Analyzes market volatility to classify regimes (Low, Normal, High).
    """
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None, **kwargs):
        self.config = config
        self.shared_state = kwargs.get("shared_state")
        self.symbols: Iterable[str] = kwargs.get("symbols") or []
        self.timeframe = str(getattr(config, "VOLATILITY_REGIME_TIMEFRAME", "5m") or "5m")
        self.atr_period = int(getattr(config, "VOLATILITY_REGIME_ATR_PERIOD", 14) or 14)
        self.update_interval = float(getattr(config, "VOLATILITY_REGIME_UPDATE_SEC", 15.0) or 15.0)
        self.low_threshold = float(getattr(config, "VOLATILITY_REGIME_LOW_PCT", 0.0025) or 0.0025)
        self.high_threshold = float(getattr(config, "VOLATILITY_REGIME_HIGH_PCT", 0.006) or 0.006)
        self.logger = logger or logging.getLogger("VolatilityRegime")
        self.current_regime = "normal"
        self._last_regime = None
        self._provisional_emitted = False
        self._task: Optional[asyncio.Task] = None
        self.logger.info(
            "VolatilityRegimeDetector initialized (ATR%%). tf=%s period=%d update=%.1fs",
            self.timeframe,
            self.atr_period,
            self.update_interval,
        )

    async def start(self):
        self.logger.info(
            "VolatilityRegimeDetector started (tf=%s atr_period=%d low=%.3f%% high=%.3f%%)",
            self.timeframe,
            self.atr_period,
            self.low_threshold * 100.0,
            self.high_threshold * 100.0,
        )
        # Emit a quick snapshot so downstream readers have a regime immediately.
        try:
            await self._update_regime_snapshot()
        except Exception as e:
            self.logger.debug("VolatilityRegimeDetector initial snapshot failed: %s", e, exc_info=True)
        # Run background loop without blocking startup.
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self.run())
        return

    async def run(self):
        """Main loop: recompute ATR%% regime and persist to SharedState."""
        while True:
            try:
                await self._update_regime_snapshot()
            except Exception as e:
                self.logger.debug("VolatilityRegimeDetector update failed: %s", e, exc_info=True)
            await asyncio.sleep(self.update_interval)

    def get_regime(self) -> str:
        return self.current_regime

    async def _update_regime_snapshot(self) -> None:
        if not self.shared_state:
            return
        atrp_values = []
        for sym in self.symbols or []:
            sym_u = str(sym).replace("/", "").upper()
            atr = None
            try:
                if hasattr(self.shared_state, "calc_atr"):
                    atr = await self.shared_state.calc_atr(sym_u, self.timeframe, self.atr_period)
            except Exception:
                atr = None

            price = 0.0
            try:
                if hasattr(self.shared_state, "safe_price"):
                    price = float(await self.shared_state.safe_price(sym_u) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                price = float(getattr(self.shared_state, "latest_prices", {}).get(sym_u, 0.0) or 0.0)

            if not atr or atr <= 0 or price <= 0:
                continue
            atrp = float(atr) / float(price)
            atrp_values.append(atrp)
            regime = self._classify_regime(atrp)
            try:
                await self.shared_state.set_volatility_regime(sym_u, self.timeframe, regime, atrp=atrp)
            except Exception:
                pass

        if not atrp_values:
            if not self._provisional_emitted:
                provisional = "normal"
                self.current_regime = provisional
                self._last_regime = provisional
                self._provisional_emitted = True
                try:
                    if hasattr(self.shared_state, "metrics") and isinstance(self.shared_state.metrics, dict):
                        self.shared_state.metrics["volatility_regime"] = provisional
                        self.shared_state.metrics["volatility_regime_atrp"] = 0.0
                except Exception:
                    pass
                try:
                    await self.shared_state.set_volatility_regime("GLOBAL", self.timeframe, provisional, atrp=0.0)
                except Exception:
                    pass
                self.logger.info(
                    "VolatilityRegimeDetector: provisional regime=%s (awaiting ATR warm-up)",
                    provisional.upper(),
                )
            return

        atrp_values.sort()
        mid = len(atrp_values) // 2
        median_atrp = atrp_values[mid] if len(atrp_values) % 2 == 1 else (atrp_values[mid - 1] + atrp_values[mid]) / 2.0
        global_regime = self._classify_regime(median_atrp)
        self.current_regime = global_regime

        if self._last_regime != global_regime:
            self.logger.info(
                "VolatilityRegimeDetector: regime=%s atr%%=%.3f%% (symbols=%d)",
                global_regime.upper(),
                median_atrp * 100.0,
                len(atrp_values),
            )
            self._last_regime = global_regime

        try:
            if hasattr(self.shared_state, "metrics") and isinstance(self.shared_state.metrics, dict):
                self.shared_state.metrics["volatility_regime"] = global_regime
                self.shared_state.metrics["volatility_regime_atrp"] = median_atrp
        except Exception:
            pass

        try:
            await self.shared_state.set_volatility_regime("GLOBAL", self.timeframe, global_regime, atrp=median_atrp)
        except Exception:
            pass

    def _classify_regime(self, atrp: float) -> str:
        if atrp < self.low_threshold:
            return "low"
        if atrp > self.high_threshold:
            return "high"
        return "normal"
