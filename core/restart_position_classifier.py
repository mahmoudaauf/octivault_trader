# core/restart_position_classifier.py
# Phase 5: Intelligent Position Classifier for System Restart
# Classifies positions on startup by analyzing trade history and wallet state

import logging
import time
from typing import Any, Dict, Optional, List, Tuple, Set
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger("RestartPositionClassifier")


class ClassificationConfidence:
    """Confidence levels for position classification."""
    CERTAIN = "CERTAIN"        # >= 95% confidence
    HIGH = "HIGH"              # >= 80% confidence
    MEDIUM = "MEDIUM"          # >= 60% confidence
    LOW = "LOW"                # >= 40% confidence
    UNCERTAIN = "UNCERTAIN"    # < 40% confidence

    @classmethod
    def confidence_score_to_level(cls, score: float) -> str:
        """Convert confidence score (0-1) to level."""
        if score >= 0.95:
            return cls.CERTAIN
        elif score >= 0.80:
            return cls.HIGH
        elif score >= 0.60:
            return cls.MEDIUM
        elif score >= 0.40:
            return cls.LOW
        else:
            return cls.UNCERTAIN


class RestartPositionClassifier:
    """
    Intelligently classify positions on system restart.
    
    Strategy:
      1. For each symbol with existing quantity:
         a. Check if symbol appears in recent trade history (bot_position)
         b. Check if symbol is in wallet but NOT in trade history (external_position)
         c. Query exchange for position metadata
         d. Apply classification rules based on signals
      
      2. Classification Rules:
         - BOT_POSITION: Has entries/exits in trade history, or in open_trades
         - EXTERNAL_POSITION: In wallet/positions but NOT in trade history
         - DUST: Position value < dust threshold
         - STABLE: Is a known stablecoin
         - RECOVERY: Positions from previous system state
      
      3. Handle Uncertainty:
         - If confidence < threshold: mark for human review
         - Log all uncertain classifications
         - Provide remediation advice
    
    Benefits:
      - Graceful recovery: system knows what it owns on restart
      - Safety: distinguishes user holdings from bot trades
      - Audit: clear trace of why position classified as X
      - Flexibility: can adjust confidence thresholds
    """
    
    def __init__(self, shared_state: Any, config: Any, exchange_client: Optional[Any] = None):
        self.ss = shared_state
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(self.__class__.__name__)
        # Durable signals loaded once during classify_all_positions()
        self._db_bot_symbols: Set[str] = set()  # symbols from DB open_positions
        self._ever_traded: bool = False           # bootstrap_metrics.total_trades_executed > 0
        self._universe: Set[str] = set()          # accepted_symbols universe
        
        # Configuration
        self.min_confidence_for_classification = float(
            getattr(config, "RESTART_MIN_CONFIDENCE", 0.60)
        )
        self.lookback_days = int(
            getattr(config, "RESTART_HISTORY_LOOKBACK_DAYS", 30)
        )
        self.dust_threshold_usdt = float(
            getattr(config, "DUST_THRESHOLD_USDT", 5.0)
        )
        self.quote_asset = str(
            getattr(config, "DEFAULT_QUOTE_CURRENCY", "USDT")
        ).upper()
        
        # Known stablecoins
        self.stablecoins = {
            "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP",
            "FDUSD", "EURS", "EURT", "GBPT", "JPYT"
        }
        
        # Classification results
        self.classifications: Dict[str, Dict[str, Any]] = {}
        self.uncertain_symbols: Set[str] = set()
        self.confidence_scores: Dict[str, float] = {}
        
        self.logger.info(
            f"RestartPositionClassifier initialized ("
            f"min_confidence={self.min_confidence_for_classification}, "
            f"lookback_days={self.lookback_days}, "
            f"dust_threshold={self.dust_threshold_usdt})"
        )
    
    async def classify_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Main entry point: classify all positions and return results.

        Detection strategy (restart-safe — uses only durable signals):
          1. DB open_positions  → bot explicitly tracked this symbol (BOT_POSITION)
          2. bootstrap_metrics  → if never traded, all wallet assets are external
          3. Universe membership → assets outside the trading universe are external deposits
          4. In-memory deque    → supplementary only (empty on restart, used as tiebreaker)
        """
        self.logger.info("🔄 Starting position classification on restart")

        try:
            # --- Pre-load durable signals (restart-safe) ---
            await self._load_durable_signals()

            # Gather all position data
            all_symbols = await self._gather_all_symbols()
            self.logger.info(f"Found {len(all_symbols)} symbols to classify "
                             f"(db_bot={len(self._db_bot_symbols)}, "
                             f"ever_traded={self._ever_traded}, "
                             f"universe={len(self._universe)})")

            # In-memory trade_history deque: supplementary tiebreaker only
            trade_history = await self._get_trade_history()
            self.logger.debug(f"In-memory trade history: {len(trade_history)} entries (may be 0 after restart)")

            # Classify each symbol
            for symbol in all_symbols:
                try:
                    result = await self._classify_symbol(symbol, trade_history)
                    self.classifications[symbol] = result
                    self.confidence_scores[symbol] = result.get("confidence", 0.0)

                    if result.get("confidence", 0.0) < self.min_confidence_for_classification:
                        self.uncertain_symbols.add(symbol)

                except Exception as e:
                    self.logger.error(f"Error classifying {symbol}: {e}", exc_info=True)
                    self.classifications[symbol] = {
                        "classification": "UNCERTAIN",
                        "confidence": 0.0,
                        "confidence_level": "UNCERTAIN",
                        "reason": f"Classification error: {e}",
                        "created_at": time.time(),
                        "created_by_agent": "RestartPositionClassifier",
                    }
                    self.uncertain_symbols.add(symbol)

            await self._register_classifications()
            self._log_summary()
            return self.classifications

        except Exception as e:
            self.logger.error(f"Fatal error in classify_all_positions: {e}", exc_info=True)
            return {}

    async def _load_durable_signals(self) -> None:
        """
        Pre-load restart-safe signals that survive process restarts.

        Three durable sources (in confidence order):
          1. DB open_positions  — bot was explicitly tracking these symbols
          2. bootstrap_metrics  — persistent trade count / first_trade_at
          3. accepted_symbols   — the universe the bot is allowed to trade
        """
        # Signal 1: DB open_positions
        try:
            db = getattr(self.ss, "_database_manager", None)
            if db and hasattr(db, "load_open_positions"):
                rows = await db.load_open_positions() or []
                self._db_bot_symbols = {
                    str(r.get("symbol", "")).upper()
                    for r in rows
                    if r.get("symbol")
                }
                self.logger.info(f"[Classifier] DB bot symbols: {sorted(self._db_bot_symbols) or '(none)'}")
        except Exception as e:
            self.logger.warning(f"[Classifier] Could not load DB open_positions: {e}")

        # Signal 2: bootstrap_metrics (persistent trade counter)
        try:
            bm = getattr(self.ss, "bootstrap_metrics", None)
            if bm:
                self._ever_traded = (
                    bm.get_total_trades_executed() > 0
                    or bm.get_first_trade_at() is not None
                )
            else:
                # Fall back to in-memory metrics
                self._ever_traded = bool(
                    self.ss.metrics.get("total_trades_executed", 0) > 0
                    or self.ss.metrics.get("first_trade_at") is not None
                )
        except Exception as e:
            self.logger.warning(f"[Classifier] Could not read bootstrap_metrics: {e}")
            self._ever_traded = False

        # Signal 3: accepted universe (non-empty = system has been configured)
        try:
            self._universe = {
                str(s).upper()
                for s in (self.ss.accepted_symbols or {}).keys()
                if s
            }
        except Exception as e:
            self.logger.warning(f"[Classifier] Could not read accepted_symbols: {e}")
            self._universe = set()
    
    async def _gather_all_symbols(self) -> Set[str]:
        """Gather all symbols from positions, balances, and open trades."""
        symbols = set()
        
        # From positions
        try:
            positions = getattr(self.ss, "positions", {}) or {}
            symbols.update([s for s in positions.keys() if s])
        except Exception:
            pass
        
        # From balances (convert to symbols)
        try:
            balances = getattr(self.ss, "balances", {}) or {}
            for asset in balances.keys():
                if asset and asset != self.quote_asset:
                    symbol = f"{asset}{self.quote_asset}"
                    symbols.add(symbol)
        except Exception:
            pass
        
        # From open trades
        try:
            open_trades = getattr(self.ss, "open_trades", {}) or {}
            symbols.update([s for s in open_trades.keys() if s])
        except Exception:
            pass
        
        self.logger.debug(f"Gathered symbols: {sorted(symbols)}")
        return symbols
    
    async def _get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Return the in-memory trade_history deque.

        IMPORTANT: this deque is NOT persisted — it is always empty after a process
        restart. It is kept here as a supplementary tiebreaker only. All primary
        classification logic must use the durable signals loaded by
        _load_durable_signals() instead.
        """
        try:
            trade_history = getattr(self.ss, "trade_history", None)
            if trade_history is None:
                return []
            return list(trade_history)
        except Exception:
            return []
    
    async def _classify_symbol(
        self,
        symbol: str,
        trade_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify a single symbol based on all available signals.
        
        Returns classification dict with confidence score.
        """
        self.logger.debug(f"Classifying {symbol}")
        
        # Gather signals
        signals = {
            "in_trade_history": self._check_trade_history(symbol, trade_history),
            "in_open_trades": self._check_open_trades(symbol),
            "is_stablecoin": self._check_stablecoin(symbol),
            "is_dust": await self._check_dust(symbol),
            "in_wallet": self._check_wallet(symbol),
            "in_positions": self._check_positions(symbol),
        }
        
        # Apply classification rules
        classification, confidence, reason = self._apply_rules(symbol, signals)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "confidence_level": ClassificationConfidence.confidence_score_to_level(confidence),
            "origin": self._determine_origin(classification, signals),
            "reason": reason,
            "signals": signals,
            "created_at": time.time(),
            "created_by_agent": "RestartPositionClassifier",
        }
    
    def _check_trade_history(self, symbol: str, trade_history: List[Dict[str, Any]]) -> Tuple[bool, int]:
        """Check if symbol appears in trade history."""
        count = 0
        for trade in trade_history:
            if str(trade.get("symbol", "")).upper() == symbol.upper():
                count += 1
        
        return (count > 0, count)
    
    def _check_open_trades(self, symbol: str) -> bool:
        """Check if symbol is in open_trades."""
        try:
            open_trades = getattr(self.ss, "open_trades", {}) or {}
            return symbol.upper() in {s.upper() for s in open_trades.keys()}
        except Exception:
            return False
    
    def _check_stablecoin(self, symbol: str) -> bool:
        """Check if symbol is a known stablecoin."""
        # Extract asset from symbol (e.g., BTCUSDT -> BTC)
        asset = symbol.replace(self.quote_asset, "").upper()
        return asset in self.stablecoins
    
    async def _check_dust(self, symbol: str) -> bool:
        """Check if position is dust (below min value)."""
        try:
            # Get quantity
            qty = await self._get_position_quantity(symbol)
            if qty is None or qty <= 0:
                return False
            
            # Get price
            price = await self._get_position_price(symbol)
            if price is None or price <= 0:
                return False
            
            # Calculate value
            value = qty * price
            return value < self.dust_threshold_usdt
        except Exception:
            return False
    
    def _check_wallet(self, symbol: str) -> bool:
        """Check if symbol asset is in wallet balances."""
        try:
            asset = symbol.replace(self.quote_asset, "").upper()
            balances = getattr(self.ss, "balances", {}) or {}
            
            for key in balances.keys():
                if str(key).upper() == asset:
                    return True
        except Exception:
            pass
        
        return False
    
    def _check_positions(self, symbol: str) -> bool:
        """Check if symbol is in positions dict."""
        try:
            positions = getattr(self.ss, "positions", {}) or {}
            return symbol.upper() in {s.upper() for s in positions.keys()}
        except Exception:
            return False
    
    async def _get_position_quantity(self, symbol: str) -> Optional[float]:
        """Get position quantity from SharedState."""
        try:
            positions = getattr(self.ss, "positions", {}) or {}
            for key, pos in positions.items():
                if str(key).upper() == symbol.upper():
                    qty = float(pos.get("quantity", 0.0) or pos.get("qty", 0.0) or 0.0)
                    return qty if qty > 0 else None
        except Exception:
            pass
        return None
    
    async def _get_position_price(self, symbol: str) -> Optional[float]:
        """Get position price from SharedState."""
        try:
            # Try latest_prices first
            prices = getattr(self.ss, "latest_prices", {}) or {}
            if symbol.upper() in {s.upper() for s in prices.keys()}:
                return float(prices[symbol.upper()])
            
            # Try positions dict
            positions = getattr(self.ss, "positions", {}) or {}
            for key, pos in positions.items():
                if str(key).upper() == symbol.upper():
                    price = float(pos.get("avg_price", 0.0) or pos.get("mark_price", 0.0) or 0.0)
                    return price if price > 0 else None
        except Exception:
            pass
        
        return None
    
    def _apply_rules(
        self,
        symbol: str,
        signals: Dict[str, Any]
    ) -> Tuple[str, float, str]:
        """
        Apply classification rules (durable signals first, deque as tiebreaker).

        Priority (highest confidence first):
          1. Stablecoin check  — structural, always reliable
          2. DB open_positions — durable, bot explicitly tracked this
          3. In-memory open_trades — in-memory but strong signal if present
          4. bootstrap_metrics == 0 — system never traded; all wallet = external
          5. Universe membership — bot can only have opened universe symbols
          6. In-memory trade_history deque — supplementary, empty after restart
          7. Fallback to RECOVERY / UNCERTAIN
        """
        in_trade_history, trade_count = signals["in_trade_history"]
        in_open_trades = signals["in_open_trades"]
        is_stablecoin = signals["is_stablecoin"]
        is_dust = signals["is_dust"]
        in_wallet = signals["in_wallet"]
        in_positions = signals["in_positions"]
        sym_upper = symbol.upper()

        # Rule 1: Stablecoins
        if is_stablecoin:
            return "STABLE", 0.95, "Known stablecoin asset"

        # Rule 2: Dust
        if is_dust:
            return "DUST", 0.90, "Position value below dust threshold"

        # Rule 3: DB open_positions — strongest durable signal
        if sym_upper in self._db_bot_symbols:
            return (
                "BOT_POSITION",
                0.97,
                "Found in DB open_positions (bot was explicitly tracking)"
            )

        # Rule 4: In-memory open_trades (strong if present)
        if in_open_trades:
            return "BOT_POSITION", 0.88, "Currently in open_trades"

        # Rule 5: bootstrap_metrics says system never traded
        # → all wallet assets pre-existed the bot
        if not self._ever_traded and in_wallet:
            return (
                "EXTERNAL_POSITION",
                0.99,
                "bootstrap_metrics: system has never executed a trade; "
                "all wallet assets are pre-existing user holdings"
            )

        # Rule 6: Symbol is outside the trading universe
        # The bot can only open positions in its accepted universe.
        # A wallet asset outside the universe must be an external deposit.
        if self._universe and sym_upper not in self._universe and in_wallet:
            return (
                "EXTERNAL_POSITION",
                0.87,
                f"Symbol not in trading universe ({len(self._universe)} symbols); "
                "likely a user deposit"
            )

        # Rule 7: In-memory trade_history deque (supplementary — empty after restart)
        if in_trade_history and trade_count >= 1:
            confidence = min(0.93, 0.72 + (trade_count * 0.04))
            return (
                "BOT_POSITION",
                confidence,
                f"In-memory trade history: {trade_count} trade(s) (note: empty after restart)"
            )

        # Rule 8: In wallet + ever_traded + inside universe → ambiguous
        # Could be a bot position whose DB record was lost, or an external deposit.
        if in_wallet and self._ever_traded:
            if sym_upper in self._universe:
                return (
                    "RECOVERY",
                    0.62,
                    "In wallet, inside universe, system has traded before — "
                    "likely a bot position from a previous session (DB open_position record missing)"
                )
            return (
                "EXTERNAL_POSITION",
                0.75,
                "In wallet, outside universe, system has traded before — "
                "likely a user deposit"
            )

        # Rule 9: In positions but no other signals
        if in_positions:
            return (
                "RECOVERY",
                0.55,
                "Found in positions dict with no supporting signals"
            )

        return (
            "UNCERTAIN",
            0.35,
            f"Conflicting/insufficient signals: "
            f"db_bot={sym_upper in self._db_bot_symbols}, "
            f"ever_traded={self._ever_traded}, "
            f"in_universe={sym_upper in self._universe}, "
            f"open_trades={in_open_trades}, wallet={in_wallet}"
        )
    
    def _determine_origin(self, classification: str, signals: Dict[str, Any]) -> str:
        """Determine origin based on classification and signals."""
        if classification == "BOT_POSITION":
            return "exchange_open_positions"
        elif classification == "EXTERNAL_POSITION":
            return "wallet_balance_sync"
        elif classification == "RECOVERY":
            return "restart_recovery"
        elif classification == "DUST":
            return "dust_accumulation"
        elif classification == "STABLE":
            return "stable_asset"
        else:
            return "unknown"
    
    async def _register_classifications(self) -> None:
        """Register all classifications in SharedState."""
        try:
            if not hasattr(self.ss, "register_position_classified"):
                self.logger.warning(
                    "SharedState.register_position_classified not available, "
                    "skipping registration"
                )
                return
            
            # Import AssetClassification enum
            from core.shared_state import AssetClassification
            
            registered = 0
            for symbol, result in self.classifications.items():
                try:
                    qty = await self._get_position_quantity(symbol)
                    price = await self._get_position_price(symbol)
                    
                    if qty is None or qty <= 0:
                        continue
                    
                    if price is None:
                        price = 1.0
                    
                    # Convert string classification to enum
                    classification_str = result["classification"]
                    try:
                        classification_enum = AssetClassification(classification_str)
                    except (ValueError, KeyError):
                        # Default to BOT_POSITION if unknown classification
                        classification_enum = AssetClassification.BOT_POSITION
                    
                    # Register in SharedState
                    await self.ss.register_position_classified(
                        symbol=symbol,
                        quantity=float(qty),
                        price=float(price),
                        classification=classification_enum,
                        origin=result["origin"],
                        created_by_agent="RestartPositionClassifier",
                        management_strategy="HOLD" if classification_str in ("EXTERNAL_POSITION", "STABLE") else "ACTIVE",
                    )
                    registered += 1
                    self.logger.debug(
                        f"Registered {symbol} as {result['classification']} "
                        f"(confidence: {result['confidence_level']})"
                    )
                
                except Exception as e:
                    self.logger.error(
                        f"Failed to register {symbol}: {e}",
                        exc_info=True
                    )
            
            self.logger.info(f"Registered {registered} positions in SharedState")
        
        except Exception as e:
            self.logger.error(
                f"Error registering classifications: {e}",
                exc_info=True
            )
    
    def _log_summary(self) -> None:
        """Log classification summary."""
        summary = defaultdict(int)
        uncertain_count = 0
        
        for symbol, result in self.classifications.items():
            summary[result["classification"]] += 1
            if symbol in self.uncertain_symbols:
                uncertain_count += 1
        
        self.logger.info("=" * 70)
        self.logger.info("📊 POSITION CLASSIFICATION SUMMARY (RESTART)")
        self.logger.info("=" * 70)
        
        for classification in ["BOT_POSITION", "EXTERNAL_POSITION", "DUST", "STABLE", "RECOVERY", "UNCERTAIN"]:
            count = summary[classification]
            if count > 0:
                self.logger.info(f"  {classification:20s}: {count:3d} position(s)")
        
        self.logger.info("=" * 70)
        
        if uncertain_count > 0:
            self.logger.warning(
                f"⚠️  {uncertain_count} position(s) have low confidence and require review:"
            )
            for symbol in sorted(self.uncertain_symbols):
                result = self.classifications[symbol]
                self.logger.warning(
                    f"    {symbol}: {result['reason']} "
                    f"(confidence: {result['confidence']:.2%})"
                )
        
        self.logger.info("✅ Position classification complete")
    
    def get_uncertain_symbols(self) -> List[Tuple[str, float, str]]:
        """Get list of uncertain symbols for manual review."""
        result = []
        for symbol in sorted(self.uncertain_symbols):
            class_result = self.classifications[symbol]
            result.append((
                symbol,
                class_result["confidence"],
                class_result["reason"]
            ))
        return result
    
    def get_classification_summary(self) -> Dict[str, Any]:
        """Get summary statistics of classifications."""
        summary = defaultdict(int)
        
        for result in self.classifications.values():
            summary[result["classification"]] += 1
        
        return {
            "total_symbols": len(self.classifications),
            "uncertain_symbols": len(self.uncertain_symbols),
            "by_classification": dict(summary),
            "avg_confidence": (
                sum(self.confidence_scores.values()) / len(self.confidence_scores)
                if self.confidence_scores else 0.0
            ),
        }
