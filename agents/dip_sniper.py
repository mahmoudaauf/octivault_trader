import asyncio
from typing import Any, Dict, List, Optional
import logging
import os
import pandas as pd
import numpy as np
import time
from functools import partial

from utils.indicators import compute_ema, compute_bollinger_bands, compute_atr
try:
    from core.agent_optimizer import load_tuned_params as _load_tuned_params_optimizer
    _HAS_AGENT_OPTIMIZER = True
except Exception:
    _load_tuned_params_optimizer = None
    _HAS_AGENT_OPTIMIZER = False
try:
    from utils.tuned_params import get_tuned_params as _get_tuned_params
    _HAS_TUNED_PARAMS = True
except Exception:
    _get_tuned_params = None
    _HAS_TUNED_PARAMS = False
try:
    from utils.ta_indicators import calculate_volume_surge as _calc_volume_surge
    _HAS_TA_INDICATORS = True
except Exception:
    _calc_volume_surge = None
    _HAS_TA_INDICATORS = False

def load_tuned_params(name):
    """Load tuned params with fallback chain: agent_optimizer → tuned_params → {}."""
    if _HAS_AGENT_OPTIMIZER and _load_tuned_params_optimizer is not None:
        try:
            result = _load_tuned_params_optimizer(name) or {}
            if result:
                return result
        except Exception:
            pass
    if _HAS_TUNED_PARAMS and _get_tuned_params is not None:
        try:
            return _get_tuned_params(name) or {}
        except Exception:
            pass
    return {}

from core.component_status_logger import log_component_status
from core.stubs import TradeIntent

AGENT_NAME = "DipSniper"
logger = logging.getLogger(AGENT_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

_log_path = f"logs/agents/{AGENT_NAME.lower()}.log"
os.makedirs(os.path.dirname(_log_path), exist_ok=True)

file_handler = logging.FileHandler(_log_path)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s"))
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)


class DipSniper:
    """
    P9-COMPLIANT: Signal-only dip detection agent.
    
    Responsibilities:
    - Analyze symbols for dip conditions
    - Generate BUY signals when conditions met
    - Return signals to AgentManager
    
    NOT responsible for:
    - Executing trades
    - Calculating quantities
    - Managing TP/SL
    - Direct MetaController emission
    """
    
    agent_type = "strategy"
    
    def __init__(self, shared_state, execution_manager, config, tp_sl_engine=None,
                 timeframe='5m', symbols=None, name="DipSniper", **kwargs):
        self.name = self.__class__.__name__
        self.shared_state = shared_state
        # execution_manager and tp_sl_engine kept for compatibility but NOT USED
        self.config = config
        self.timeframe = timeframe
        
        # P9 FIX: Signal collection buffer
        self._collected_signals: List[Dict[str, Any]] = []
        self._collecting_for_agent_manager = False
        
        # P9 FIX: Use get_accepted_symbols() not direct .symbols access
        self.symbols = symbols if symbols is not None else []
        
        # IMPROVEMENT: Concurrency control (optional but recommended)
        max_concurrency = int(getattr(config, "DIPSNIPER_MAX_CONCURRENCY", 5))
        self._sem = asyncio.Semaphore(max_concurrency)

        # Diagnostics (rate-limited). Helps explain "0 signals" cases in production logs.
        self._last_diag_log_ts: float = 0.0
        
        # P9 Config Tuning (Relaxed Thresholds per User Request)
        # NOTE: Use _cfg-backed properties to avoid clashing with @property names.
        
        log_component_status(self.name, "Initialized")
        logger.info(f"🚀 {self.name} initialized (P9-compliant, signal-only, max_concurrency={max_concurrency})")

    @staticmethod
    async def _await_maybe(coro):
        """Helper to await if coroutine, else return value."""
        return await coro if asyncio.iscoroutine(coro) else coro

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        P9 CONTRACT: Generate signals for ALL symbols.
        Called by AgentManager. Returns signal list.
        Delegates to run_once() to do the actual work (following TrendHunter pattern).
        """
        try:
            self._collected_signals = []
            self._collecting_for_agent_manager = True
            try:
                # CRITICAL FIX: Load symbols first like TrendHunter does!
                await self._load_symbols()
                await self.run_once()
                return self._collected_signals
            finally:
                self._collecting_for_agent_manager = False
        except Exception as e:
            logger.error(f"[{self.name}] generate_signals EXCEPTION: {e}", exc_info=True)
            return []  # Return empty list on error instead of None

    async def _publish_trade_intent(self, payload: Dict[str, Any]) -> bool:
        event_bus = getattr(self.shared_state, "event_bus", None)
        publish = getattr(event_bus, "publish", None)
        if callable(publish):
            try:
                await publish("events.trade.intent", TradeIntent(**payload))
                logger.info(
                    f"[{self.name}] Published TradeIntent: {payload.get('symbol')} {payload.get('side')}"
                )
                return True
            except Exception:
                logger.warning(
                    f"[{self.name}] Failed to publish TradeIntent for {payload.get('symbol')}",
                    exc_info=True,
                )
        emit_event = getattr(self.shared_state, "emit_event", None)
        if callable(emit_event):
            try:
                await emit_event("TradeIntent", dict(payload))
                return True
            except Exception:
                logger.warning(
                    f"[{self.name}] Fallback TradeIntent emit failed for {payload.get('symbol')}",
                    exc_info=True,
                )
        return False

    async def run_once(self):
        """
        P9 bridge: Orchestrates signal generation.
        Called by AgentManager via generate_signals().
        Generates signals and emits them to MetaController.
        """
        logger.info(f"[{self.name}] run_once START")  # DEBUG
        # Refresh symbols from SharedState
        symbols = await self._safe_get_symbols()
        if set(symbols) != set(self.symbols):
            self.symbols = list(symbols)
            logger.info(f"[{self.name}] Updated symbols: {len(self.symbols)}")

        # ITERATION 1 FIX: Bypass market data ready check - symbols available, data flowing
        # try:
        #     if hasattr(self.shared_state, "is_market_data_ready"):
        #         ready = await self._await_maybe(self.shared_state.is_market_data_ready())
        #         if not ready:
        #             logger.warning(f"[{self.name}] Market data not ready. Skipping run.")
        #             return
        # except Exception:
        #     pass

        if not self.symbols:
            logger.warning(f"[{self.name}] No symbols to analyze")
            if hasattr(self.shared_state, "update_component_status"):
                await self.shared_state.update_component_status(
                    component=self.name, status="Waiting", detail="No symbols in universe yet"
                )
            return

        if hasattr(self.shared_state, "update_component_status"):
            await self.shared_state.update_component_status(
                component=self.name,
                status="Scanning",
                detail=f"Scanning {len(self.symbols)} symbols for dip conditions",
            )

        # Analyze ALL symbols with concurrency control
        async def _one(sym: str):
            async with self._sem:
                return await self._analyze_symbol(sym)

        results = await asyncio.gather(*[_one(s) for s in self.symbols], return_exceptions=True)

        # Log results
        logger.info(f"[{self.name}] run_once END -> Generated {len(self._collected_signals)} signals across {len(self.symbols)} symbols")

        # If we produced no signals, emit a compact diagnostic summary (rate-limited)
        # so ops can tell whether this is "no setups" vs "no data".
        try:
            diag_interval = float(self._cfg("DIPSNIPER_DIAG_INTERVAL_SEC", 300.0) or 300.0)
            now = time.time()
            if (len(self._collected_signals) == 0) and (now - float(self._last_diag_log_ts or 0.0) >= diag_interval):
                self._last_diag_log_ts = now
                counts: Dict[str, int] = {}
                best = None
                best_score = -1.0

                for r in results or []:
                    if isinstance(r, Exception):
                        key = "exception"
                        counts[key] = counts.get(key, 0) + 1
                        continue
                    if not isinstance(r, dict):
                        key = "unknown"
                        counts[key] = counts.get(key, 0) + 1
                        continue
                    status = str(r.get("status", "unknown"))
                    counts[status] = counts.get(status, 0) + 1

                    # Heuristic: pick "closest to trigger" based on conditions met + dip depth.
                    if status in ("no_setup", "conf_below_min", "signal"):
                        met = int(r.get("conds_met", 0) or 0)
                        dip = float(r.get("dip_percent", 0.0) or 0.0)
                        score = float(met) * 10.0 + dip
                        if score > best_score:
                            best_score = score
                            best = r

                best_str = ""
                if isinstance(best, dict):
                    best_str = (
                        f" best={best.get('symbol')} met={best.get('conds_met')} "
                        f"dip={float(best.get('dip_percent', 0.0) or 0.0):.2f}% "
                        f"bb={bool(best.get('below_bb'))} ema={bool(best.get('below_ema'))} "
                        f"conf={float(best.get('confidence', 0.0) or 0.0):.2f} "
                        f"min_conf={float(best.get('min_conf', 0.0) or 0.0):.2f} "
                        f"bars={int(best.get('bars', 0) or 0)}"
                    )

                logger.info(
                    "[%s] DIAG no-signals tf=%s symbols=%d counts=%s%s",
                    self.name,
                    str(self.timeframe),
                    len(self.symbols),
                    counts,
                    best_str,
                )
        except Exception:
            pass

        if hasattr(self.shared_state, "update_component_status"):
            if self._collected_signals:
                detail = f"Found {len(self._collected_signals)} dip signal(s): {[s['symbol'] for s in self._collected_signals]}"
            else:
                detail = f"Scanned {len(self.symbols)} symbols — no qualifying dip yet"
            await self.shared_state.update_component_status(
                component=self.name, status="Operational", detail=detail
            )

    async def run_loop(self) -> None:
        """
        Continuous loop that calls run_once() at regular intervals.
        This is what gets scheduled in the background by the framework.
        """
        await self._safe_get_symbols()  # Initial symbol load
        interval = int(self._cfg("AGENT_LOOP_INTERVAL", 60))
        logger.info(f"[{self.name}] 🔁 Starting run_loop @ {interval}s")
        try:
            while True:
                try:
                    await self.run_once()
                except Exception as e:
                    logger.error(f"[{self.name}] ❌ Error in run_loop: {e}", exc_info=True)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info(f"[{self.name}] run_loop cancelled; exiting cleanly.")
            raise

    async def _load_symbols(self) -> None:
        """Load accepted symbols from SharedState (matching TrendHunter pattern)."""
        try:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(getter):
                try:
                    res = getter(full=True)
                except TypeError:
                    res = getter()
                accepted = await res if asyncio.iscoroutine(res) else (res or {})
            else:
                accepted = {}
            
            if not isinstance(accepted, dict):
                snap = getattr(self.shared_state, "get_accepted_symbols_snapshot", None)
                if callable(snap):
                    r = snap()
                    accepted = await r if asyncio.iscoroutine(r) else (r or {})
                if not isinstance(accepted, dict):
                    accepted = {s: {} for s in (accepted or [])}
            
            # ITERATION 2 FIX: If accepted_symbols empty, use DEFAULT_SYMBOLS as fallback
            if not accepted:
                logger.warning(f"[{self.name}] ⚠️  accepted_symbols is empty! Using DEFAULT_SYMBOLS fallback...")
                try:
                    from core.bootstrap_symbols import DEFAULT_SYMBOLS
                    accepted = DEFAULT_SYMBOLS
                    logger.warning(f"[{self.name}] ✅ Using {len(DEFAULT_SYMBOLS)} DEFAULT_SYMBOLS as fallback")
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to load DEFAULT_SYMBOLS fallback: {e}")
                    accepted = {}
            
            self.symbols = list(accepted.keys())
            logger.info(f"[{self.name}] 🔄 Loaded {len(self.symbols)} symbols from SharedState")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to load symbols: {e}", exc_info=True)
            # ✅ FIX: Do NOT wipe symbols on exception - keep existing ones
            # This prevents: error in loading → self.symbols=[] → loop over zero symbols
            if not self.symbols:
                logger.warning(f"[{self.name}] No symbols available to trade; agent will be idle until symbols load")

    async def _safe_get_symbols(self) -> List[str]:
        """P9 FIX: Use canonical symbol source."""
        try:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(getter):
                res = getter()
                if asyncio.iscoroutine(res):
                    res = await res
            else:
                res = []
            if isinstance(res, dict):
                return list(res.keys())
            return list(res or [])
        except Exception:
            return []

    async def _analyze_symbol(self, symbol: str):
        """Analyze one symbol and collect signal if conditions met."""
        logger.debug(f"[{self.name}] 🔍 Analyzing {symbol}")
        
        try:
            sym = str(symbol or "").upper()
            if not sym:
                return {"symbol": symbol, "status": "invalid_symbol"}

            # ✅ FIX: Check market data readiness event before proceeding
            try:
                md_ready_event = getattr(self.shared_state, "market_data_ready_event", None)
                if md_ready_event and hasattr(md_ready_event, "is_set"):
                    if not md_ready_event.is_set():
                        logger.debug(f"[{self.name}] Market data ready event not set for {symbol}. Returning insufficient_data.")
                        return {"symbol": sym, "status": "market_data_not_ready"}
            except Exception:
                pass

            # IMPROVEMENT 1: Freshness check (prevent stale data signals)
            if hasattr(self.shared_state, "is_fresh"):
                try:
                    fresh = self.shared_state.is_fresh(symbol, self.timeframe)
                    if asyncio.iscoroutine(fresh):
                        fresh = await fresh
                    if not fresh:
                        logger.debug(f"[{self.name}] Stale data for {symbol}, skipping")
                        return {"symbol": sym, "status": "stale_data"}
                except Exception:
                    pass  # If freshness check fails, continue anyway
            
            candles = await self.shared_state.get_market_data(symbol, self.timeframe)
            
            # Handle dict response
            if isinstance(candles, dict):
                candles = candles.get(symbol, [])
            
            if not isinstance(candles, list) or len(candles) < 50:
                logger.debug(f"[{self.name}] Insufficient data for {symbol}")
                return {"symbol": sym, "status": "insufficient_bars", "bars": int(len(candles or []) if isinstance(candles, list) else 0)}
            
            # Create DataFrame
            if isinstance(candles[0], dict):
                df = await asyncio.to_thread(pd.DataFrame, candles)

                # SharedState OHLCVBar uses {ts,o,h,l,c,v}. Normalize to the
                # indicator schema expected by this agent. Also handle
                # capitalized keys like "Open"/"Close".
                try:
                    # Normalize all columns to lowercase first.
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    lower_to_orig = {str(c).lower(): c for c in df.columns}

                    # Only remap when canonical columns are missing.
                    if "open" not in lower_to_orig and "o" in lower_to_orig:
                        ren = {}
                        for src, dst in {
                            "ts": "timestamp",
                            "t": "timestamp",
                            "time": "timestamp",
                            "o": "open",
                            "h": "high",
                            "l": "low",
                            "c": "close",
                            "v": "volume",
                        }.items():
                            if src in lower_to_orig and dst not in lower_to_orig:
                                ren[lower_to_orig[src]] = dst
                        if ren:
                            df = df.rename(columns=ren)
                except Exception:
                    pass
            else:
                df = await asyncio.to_thread(
                    pd.DataFrame,
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            
            # Clean data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close', 'volume'], inplace=True)
            
            if len(df) < 50:
                return {"symbol": sym, "status": "insufficient_rows", "rows": int(len(df))}
            
            # Compute indicators
            def compute_indicators_sync(dataframe):
                df_copy = dataframe.copy()
                df_copy['ema20'] = compute_ema(df_copy['close'], span=20)
                df_copy['bb_upper'], df_copy['bb_middle'], df_copy['bb_lower'] = compute_bollinger_bands(df_copy['close'])
                df_copy['atr'] = compute_atr(df_copy)
                return df_copy
            
            df = await asyncio.to_thread(compute_indicators_sync, df)
            
            if df.empty or len(df) < 2:
                return {"symbol": sym, "status": "insufficient_after_indicators", "rows": int(len(df))}
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check for NaN indicators
            if pd.isna(latest['bb_lower']) or pd.isna(latest['ema20']) or pd.isna(latest['atr']):
                return {"symbol": sym, "status": "invalid_indicators"}
            
            if latest['atr'] <= 0:
                return {"symbol": sym, "status": "atr_non_positive"}
            
            # Calculate dip metrics
            dip_percent = (previous['close'] - latest['close']) / previous['close'] * 100 if previous['close'] != 0 else 0.0
            volume_avg = df['volume'].rolling(10).mean().iloc[-1]
            # Guard NaN volume_avg (short history): treat as no spike rather than crashing
            if pd.isna(volume_avg) or volume_avg <= 0:
                volume_spike = False
            else:
                volume_spike = latest['volume'] > volume_avg * self.volume_spike_multiplier

            # Dip conditions
            price_below_bb = latest['close'] < latest['bb_lower']
            price_below_ema = latest['close'] < latest['ema20']
            dip_thr = float(self.dip_threshold_percent or 0.0)
            dip_deep_enough = dip_percent >= dip_thr  # Changed from > to >= to catch edge cases

            # SAFETY TUNING (user requested):
            # - Keep EMA as a useful directional sanity check (optional, not required).
            # - BB is a soft boost (not mandatory), to prevent total starvation.
            require_ema = bool(self._cfg("DIPSNIPER_REQUIRE_BELOW_EMA", False))  # Changed default to False
            min_score = int(self._cfg("DIPSNIPER_MIN_SCORE", 1) or 1)  # Lowered from 2 to 1 for higher sensitivity
            score = int(dip_deep_enough) + int(price_below_ema) + int(price_below_bb)
            
            # P9 FIX: Also catch uptrend momentum when no dips available
            # If price is ABOVE EMA20 (uptrend signal), emit signal
            price_above_ema = latest['close'] > latest['ema20']
            price_above_bb = latest['close'] > latest['bb_upper']
            uptrend_momentum = price_above_ema  # Uptrend = price above EMA20
            
            # Original dip condition OR uptrend momentum
            # Relaxed: Just need dip to be deep enough, no strict EMA requirement
            condition = (bool(dip_deep_enough) and (score >= max(1, min_score))) or uptrend_momentum
            
            # P9 FIX: Only emit BUY signals when conditions met
            if self.enabled and condition:
                # Confidence: primarily driven by dip depth; BB breach boosts confidence but isn't required.
                atr = float(latest['atr'])
                bb_lower = float(latest['bb_lower'])
                bb_upper = float(latest['bb_upper'])
                close = float(latest['close'])

                # DIP-based confidence
                dip_factor = 0.0
                if dip_thr > 0:
                    dip_factor = float(np.clip((dip_percent / dip_thr) / 2.0, 0.0, 1.0))

                bb_factor = 0.0
                if atr > 0:
                    # For dips: use lower band distance
                    bb_factor = float(np.clip((bb_lower - close) / atr, 0.0, 1.0))

                ema_factor = 1.0 if price_below_ema else 0.5
                
                # UPTREND confidence (if uptrend momentum triggered the signal)
                if uptrend_momentum:
                    # Distance above EMA for uptrend confidence
                    ema20 = float(latest['ema20'])
                    uptrend_ema_factor = 0.0 if ema20 <= 0 else float(np.clip((close - ema20) / ema20 * 100, 0.0, 1.0))
                    # BB upper distance as bonus
                    uptrend_bb_factor = 0.0 if atr <= 0 else float(np.clip((close - bb_upper) / atr, 0.0, 1.0))
                    # Higher weight on EMA distance, BB is bonus
                    base_confidence = float(np.clip(0.7 * uptrend_ema_factor + 0.3 * uptrend_bb_factor, 0.0, 1.0))
                else:
                    # DIP-based confidence (original logic)
                    base_confidence = float(np.clip((0.65 * dip_factor) + (0.35 * bb_factor), 0.0, 1.0) * ema_factor)
                
                # Volume spike adds +10% confidence bonus
                confidence = float(np.clip(base_confidence * (1.1 if volume_spike else 1.0), 0.0, 1.0))
                
                # Confidence floor (reduce noise)
                base_min_conf = float(self._cfg("MIN_SIGNAL_CONF", 0.55))
                
                # Dynamic Aggression (P9 Loop)
                agg_factor = 1.0
                if hasattr(self.shared_state, "get_dynamic_param"):
                    agg_factor = float(self.shared_state.get_dynamic_param("aggression_factor", 1.0))
                
                min_conf = base_min_conf
                if agg_factor > 1.0:
                    min_conf = max(0.40, base_min_conf / agg_factor)

                if confidence < min_conf:
                    logger.debug(f"[{self.name}] {symbol} confidence {confidence:.2f} below dynamic threshold {min_conf:.2f}")
                    return {
                        "symbol": sym,
                        "status": "conf_below_min",
                        "bars": int(len(df)),
                        "conds_met": int(price_below_bb) + int(price_below_ema) + int(dip_deep_enough),
                        "dip_percent": float(dip_percent),
                        "below_bb": bool(price_below_bb),
                        "below_ema": bool(price_below_ema),
                        "confidence": float(confidence),
                        "min_conf": float(min_conf),
                    }
                
                # Soft guard: warn if quote hint is below SAFE_ENTRY_USDT; execution layer enforces floors
                quote_hint = float(
                    self._cfg(
                        "EMIT_BUY_QUOTE",
                        self._cfg("MIN_ENTRY_USDT", self._cfg("DEFAULT_PLANNED_QUOTE", 25.0)),
                    )
                )
                min_entry = float(getattr(self.config, "MIN_ENTRY_USDT", getattr(self.config, "SAFE_ENTRY_USDT", 12.0)))
                if quote_hint < min_entry:
                    logger.warning(
                        f"[{self.name}] Signal quote {quote_hint:.2f} < MIN_ENTRY_USDT {min_entry:.2f}; deferring to execution layer"
                    )
                
                # P9 CONTRACT: Build signal dict
                signal = {
                    "symbol": symbol,
                    "side": "BUY",
                    "action": "BUY",  # Kept for compatibility
                    "confidence": float(confidence),
                    "reason": (
                        f"DipSniper: {'uptrend momentum' if uptrend_momentum else f'dip {dip_percent:.2f}%'} "
                        f"{'above_EMA' if price_above_ema else 'below_EMA'} "
                        f"{'above_BB' if price_above_bb else 'below_BB'} "
                        f"score={score}/{min_score}"
                        f"{' + volume spike' if volume_spike else ''}"
                    ),
                    "agent": self.name,
                    "quote": quote_hint,
                    "horizon_hours": 6.0,
                }
                
                # Legacy path removed - using TradeIntent publishing only
                
                self._collected_signals.append(signal)
                logger.info(f"[{self.name}] 📤 BUY signal: {symbol} conf={confidence:.2f} dip={dip_percent:.2f}%")
                # Always publish TradeIntent to event bus (needed for MetaController drain)
                intent_payload = {
                    "symbol": str(symbol).replace("/", "").upper(),
                    "side": "BUY",
                    "planned_quote": quote_hint,
                    "quote_hint": quote_hint,
                    "confidence": float(confidence),
                    "agent": self.name,
                    "tag": f"strategy/{self.name}",
                    "reason": str(signal["reason"]),
                    "rationale": str(signal["reason"]),
                    "timeframe": self.timeframe,
                    "timestamp": time.time(),
                    "policy_context": {
                        "dip_percent": float(dip_percent),
                        "below_bb": bool(price_below_bb),
                        "below_ema": bool(price_below_ema),
                        "score": int(score),
                        "min_score": int(min_score),
                    },
                }
                await self._publish_trade_intent(intent_payload)
                return {
                    "symbol": sym,
                    "status": "signal",
                    "bars": int(len(df)),
                    "conds_met": int(price_below_bb) + int(price_below_ema) + int(dip_deep_enough),
                    "score": int(score),
                    "min_score": int(min_score),
                    "require_ema": bool(require_ema),
                    "dip_percent": float(dip_percent),
                    "below_bb": bool(price_below_bb),
                    "below_ema": bool(price_below_ema),
                    "confidence": float(confidence),
                    "min_conf": float(min_conf),
                }

            # No signal (either disabled or no setup)
            if not self.enabled:
                return {"symbol": sym, "status": "disabled"}
            return {
                "symbol": sym,
                "status": "no_setup",
                "bars": int(len(df)),
                "conds_met": int(price_below_bb) + int(price_below_ema) + int(dip_deep_enough),
                "score": int(score),
                "min_score": int(min_score),
                "require_ema": bool(require_ema),
                "dip_percent": float(dip_percent),
                "below_bb": bool(price_below_bb),
                "below_ema": bool(price_below_ema),
                "confidence": 0.0,
                "min_conf": float(self._cfg("MIN_SIGNAL_CONF", 0.55) or 0.55),
            }
            
            # P9 FIX: No HOLD signals - silence is implicit hold
                
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Error analyzing {symbol}: {e}", exc_info=True)
            # Don't emit error signals - let it silently fail
            return {"symbol": str(symbol or "").upper(), "status": "error"}

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback."""
        # Check SharedState for dynamic overrides
        if hasattr(self.shared_state, "dynamic_config"):
            val = self.shared_state.dynamic_config.get(key)
            if val is not None:
                return val
        
        # Fallback to static config
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @property
    def volume_spike_multiplier(self) -> float:
        return float(self._cfg("VOLUME_SPIKE_MULTIPLIER", 1.5))

    @property
    def dip_threshold_percent(self) -> float:
        # Ultra-sensitive mode: 0.05% threshold catches even tiny dips
        # Override via config: DIP_THRESHOLD_PERCENT = <value>
        return float(self._cfg("DIP_THRESHOLD_PERCENT", 0.05))

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("DIPSNIPER_ENABLED", True))
