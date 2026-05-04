#!/usr/bin/env python3
"""
BALANCE SYMBOL DETECTOR & SITUATION ANALYZER

Detects all symbols currently in wallet balance and analyzes their situation:
- What they are (crypto vs stablecoin)
- Their value (USD equivalent)
- Their classification (CLEAN, DUST, etc)
- Their status (locked, tradeable, etc)
- Healing/liquidation recommendations

Usage:
    python3 tools/detect_balance_symbols.py           # live exchange
    python3 tools/detect_balance_symbols.py --mock    # offline demo data
"""

import asyncio
import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from dataclasses import dataclass, asdict

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("balance_symbol_detection.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SymbolBalance:
    asset: str
    free: float
    locked: float
    total: float
    price_usd: Optional[float] = None
    value_usd: Optional[float] = None
    symbol: Optional[str] = None

    @property
    def quantity(self) -> float:
        return self.total

    @property
    def is_stablecoin(self) -> bool:
        stablecoins = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE"}
        return self.asset.upper() in stablecoins

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolSituation:
    asset: str
    symbol: str
    balance: SymbolBalance

    classification: str           # CLEAN | MICRO_DUST | HARD_DUST | DUST_LOCKED | CASH
    dust_reason: Optional[str] = None

    usd_value: float = 0.0
    percentage_of_portfolio: float = 0.0

    is_tradeable: bool = True
    is_locked: bool = False
    can_be_sold: bool = True

    action_recommended: str = "HOLD"   # HOLD | SELL | MONITOR | INVESTIGATE
    healing_eligible: bool = False
    healing_priority: int = 0          # 0=none 1=low 2=medium 3=high

    added_at: Optional[float] = None
    age_days: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "symbol": self.symbol,
            "balance": self.balance.to_dict(),
            "classification": self.classification,
            "dust_reason": self.dust_reason,
            "usd_value": self.usd_value,
            "percentage_of_portfolio": self.percentage_of_portfolio,
            "is_tradeable": self.is_tradeable,
            "is_locked": self.is_locked,
            "can_be_sold": self.can_be_sold,
            "action_recommended": self.action_recommended,
            "healing_eligible": self.healing_eligible,
            "healing_priority": self.healing_priority,
            "added_at": self.added_at,
            "age_days": self.age_days,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

class BalanceSymbolDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stablecoins = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE"}
        self.min_dust_threshold = self.config.get("min_dust_threshold", 5.0)
        self.min_productive_threshold = self.config.get("min_productive_threshold", 25.0)
        self.stale_days = self.config.get("stale_days", 30)

        logger.info(
            f"BalanceSymbolDetector | dust=${self.min_dust_threshold} "
            f"productive=${self.min_productive_threshold} stale={self.stale_days}d"
        )

    def detect_symbols_from_balance(
        self, balances: Dict[str, Dict[str, float]]
    ) -> List[SymbolBalance]:
        symbols = []
        logger.info(f"Scanning {len(balances)} assets for non-zero balances...")

        for asset, bal_info in balances.items():
            free = float(bal_info.get("free", 0.0))
            locked = float(bal_info.get("locked", 0.0))
            total = free + locked
            if total <= 0:
                continue

            balance = SymbolBalance(
                asset=asset.upper(),
                free=free,
                locked=locked,
                total=total,
                symbol=f"{asset.upper()}USDT",
            )
            symbols.append(balance)
            logger.info(f"  {asset:8s}: {total:.8f}  (free={free:.8f}, locked={locked:.8f})")

        logger.info(f"Detected {len(symbols)} non-zero symbols")
        return symbols

    def classify_symbol(
        self, balance: SymbolBalance, prices: Dict[str, float]
    ) -> SymbolSituation:
        asset = balance.asset
        symbol = balance.symbol or f"{asset}USDT"
        qty = balance.total
        price = prices.get(symbol, 0.0) if not balance.is_stablecoin else 1.0

        if balance.is_stablecoin:
            usd_value = qty
            classification = "CASH"
            dust_reason = None
        elif price <= 0:
            usd_value = 0.0
            classification = "DUST_LOCKED"
            dust_reason = "no_price_feed"
        else:
            usd_value = qty * price
            if usd_value < self.min_dust_threshold:
                classification = "DUST_LOCKED"
                dust_reason = f"below_dust_threshold (${usd_value:.2f} < ${self.min_dust_threshold})"
            elif usd_value < self.min_productive_threshold:
                classification = "MICRO_DUST"
                dust_reason = f"below_productive_threshold (${usd_value:.2f} < ${self.min_productive_threshold})"
            else:
                classification = "CLEAN"
                dust_reason = None

        is_locked = balance.locked > 0

        if classification == "CASH":
            action, healing_eligible, healing_priority = "HOLD", False, 0
        elif classification == "DUST_LOCKED":
            action, healing_eligible, healing_priority = "INVESTIGATE", True, 3
        elif classification == "MICRO_DUST":
            action, healing_eligible, healing_priority = "MONITOR", True, 2
        else:
            action, healing_eligible, healing_priority = "HOLD", False, 0

        return SymbolSituation(
            asset=asset,
            symbol=symbol,
            balance=balance,
            classification=classification,
            dust_reason=dust_reason,
            usd_value=usd_value,
            is_tradeable=price > 0,
            is_locked=is_locked,
            can_be_sold=price > 0 and not is_locked,
            action_recommended=action,
            healing_eligible=healing_eligible,
            healing_priority=healing_priority,
            notes=self._generate_notes(balance, classification, price, usd_value),
        )

    def _generate_notes(
        self,
        balance: SymbolBalance,
        classification: str,
        price: float,
        usd_value: float,
    ) -> str:
        notes = []
        if balance.is_stablecoin:
            notes.append("Stablecoin reserve (sacred)")
        if balance.locked > 0:
            notes.append(f"Locked: {balance.locked:.8f}")
        if classification == "DUST_LOCKED":
            notes.append("Below minimum economical threshold")
            notes.append("Candidate for liquidation")
        if price <= 0 and not balance.is_stablecoin:
            notes.append("No price feed available")
        if balance.total < 0.0001:
            notes.append("Extremely small quantity")
        return " | ".join(notes) if notes else "Normal position"

    def analyze_portfolio(
        self,
        balances: Dict[str, Dict[str, float]],
        prices: Dict[str, float],
    ) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("PORTFOLIO ANALYSIS")
        logger.info("=" * 80)

        symbols = self.detect_symbols_from_balance(balances)

        situations: List[SymbolSituation] = []
        total_value = 0.0

        logger.info("Classifying symbols...")
        for balance in symbols:
            situation = self.classify_symbol(balance, prices)
            situations.append(situation)
            total_value += situation.usd_value
            logger.info(
                f"  {situation.asset:8s} -> {situation.classification:15s} | "
                f"${situation.usd_value:>10.2f}"
            )

        for situation in situations:
            if total_value > 0:
                situation.percentage_of_portfolio = (situation.usd_value / total_value) * 100

        by_class: Dict[str, list] = {}
        for situation in situations:
            by_class.setdefault(situation.classification, []).append(situation)

        healing_candidates = sorted(
            [s for s in situations if s.healing_eligible],
            key=lambda x: x.healing_priority,
            reverse=True,
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_portfolio_value_usd": total_value,
            "total_symbols": len(symbols),
            "symbols_by_classification": {cls: len(v) for cls, v in by_class.items()},
            "situations": [s.to_dict() for s in situations],
            "healing_analysis": {
                "healing_eligible_count": len(healing_candidates),
                "healing_eligible_value_usd": sum(s.usd_value for s in healing_candidates),
                "healing_candidates": [s.to_dict() for s in healing_candidates[:10]],
            },
            "portfolio_summary": self._generate_summary(situations, total_value, by_class),
        }

    def _generate_summary(
        self,
        situations: List[SymbolSituation],
        total_value: float,
        by_class: Dict[str, list],
    ) -> Dict[str, Any]:
        cash_value = sum(s.usd_value for s in situations if s.classification == "CASH")
        clean_value = sum(s.usd_value for s in situations if s.classification == "CLEAN")
        dust_value = sum(
            s.usd_value
            for s in situations
            if s.classification in {"DUST_LOCKED", "MICRO_DUST", "HARD_DUST"}
        )
        return {
            "total_value_usd": total_value,
            "cash_value_usd": cash_value,
            "clean_positions_value_usd": clean_value,
            "dust_value_usd": dust_value,
            "cash_ratio": (cash_value / total_value * 100) if total_value > 0 else 0,
            "clean_ratio": (clean_value / total_value * 100) if total_value > 0 else 0,
            "dust_ratio": (dust_value / total_value * 100) if total_value > 0 else 0,
            "health_status": self._get_health_status(cash_value, dust_value, total_value),
        }

    def _get_health_status(self, cash: float, dust: float, total: float) -> str:
        if total == 0:
            return "EMPTY"
        cash_ratio = cash / total
        dust_ratio = dust / total
        if cash_ratio < 0.05:
            return "CRITICAL"
        if dust_ratio > 0.30:
            return "UNHEALTHY"
        if cash_ratio < 0.20 or dust_ratio > 0.20:
            return "WARNING"
        return "HEALTHY"


# ---------------------------------------------------------------------------
# Live exchange data fetch
# ---------------------------------------------------------------------------

async def fetch_live_data() -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Connect to ExchangeClient and pull real balances + prices."""
    from src.l0_core.config import Config
    from src.l0_core.shared_state import SharedState
    from src.l1_exchange.exchange_client import ExchangeClient

    config = Config()
    shared_state = SharedState(config=config)
    exchange = ExchangeClient(config=config, shared_state=shared_state)

    logger.info("Fetching balances from exchange...")
    balances: Dict[str, Dict[str, float]] = await exchange.get_account_balances()

    # Build a set of symbols we need prices for
    assets_needed = {
        f"{asset.upper()}USDT"
        for asset, info in balances.items()
        if float(info.get("free", 0)) + float(info.get("locked", 0)) > 0
        and asset.upper() not in {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE"}
    }

    logger.info(f"Fetching prices for {len(assets_needed)} symbols...")
    prices: Dict[str, float] = {}
    try:
        all_tickers = await exchange.get_all_tickers()
        for ticker in all_tickers:
            sym = ticker.get("symbol", "")
            if sym in assets_needed:
                try:
                    prices[sym] = float(ticker.get("lastPrice", 0))
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        logger.warning(f"Bulk ticker fetch failed ({exc}), falling back to per-symbol fetch")
        for sym in assets_needed:
            try:
                prices[sym] = await exchange.get_current_price(sym)
            except Exception:
                prices[sym] = 0.0

    return balances, prices


# ---------------------------------------------------------------------------
# Mock data (offline demo)
# ---------------------------------------------------------------------------

def load_mock_balances() -> Dict[str, Dict[str, float]]:
    return {
        "USDT": {"free": 250.50, "locked": 0.0},
        "BTC":  {"free": 0.0015, "locked": 0.0},
        "ETH":  {"free": 0.045,  "locked": 0.0},
        "ADA":  {"free": 2156.8, "locked": 0.0},
        "SHIB": {"free": 521_000_000, "locked": 0.0},
        "RAY":  {"free": 5000,   "locked": 0.0},
        "BNB":  {"free": 0.0003, "locked": 0.0},
    }


def load_mock_prices() -> Dict[str, float]:
    return {
        "BTCUSDT":  45000.0,
        "ETHUSDT":  2800.0,
        "ADAUSDT":  0.42,
        "SHIBUSDT": 0.000008,
        "RAYUSDT":  0.0015,
        "BNBUSDT":  3000.0,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_analysis(report: Dict[str, Any]) -> None:
    summary = report.get("portfolio_summary", {})

    print("\n" + "=" * 100)
    print("BALANCE SYMBOL DETECTION & SITUATION ANALYSIS")
    print("=" * 100)

    print(f"\nPORTFOLIO SNAPSHOT:")
    print(f"  Total Value:        ${summary.get('total_value_usd', 0):>10.2f}")
    print(f"  Cash (stablecoins): ${summary.get('cash_value_usd', 0):>10.2f}  ({summary.get('cash_ratio', 0):.1f}%)")
    print(f"  Clean Positions:    ${summary.get('clean_positions_value_usd', 0):>10.2f}  ({summary.get('clean_ratio', 0):.1f}%)")
    print(f"  Dust Positions:     ${summary.get('dust_value_usd', 0):>10.2f}  ({summary.get('dust_ratio', 0):.1f}%)")
    print(f"  Health Status:      {summary.get('health_status', 'UNKNOWN')}")

    print(f"\nSYMBOLS DETECTED: {report.get('total_symbols', 0)}")
    for cls, count in report.get("symbols_by_classification", {}).items():
        print(f"    {cls:15s}: {count}")

    healing = report.get("healing_analysis", {})
    print(f"\nHEALING CANDIDATES: {healing.get('healing_eligible_count', 0)}  "
          f"(${healing.get('healing_eligible_value_usd', 0):.2f} recoverable)")

    print(f"\nDETAILED SYMBOL SITUATIONS:")
    print("-" * 100)

    by_class: Dict[str, list] = {}
    for sit in report.get("situations", []):
        by_class.setdefault(sit.get("classification", "?"), []).append(sit)

    for cls in ["CASH", "CLEAN", "MICRO_DUST", "DUST_LOCKED", "HARD_DUST"]:
        rows = by_class.get(cls)
        if not rows:
            continue
        print(f"\n  {cls} ({len(rows)} symbols):")
        print(f"  {'-' * 96}")
        for sit in rows:
            asset  = sit.get("asset", "?")
            qty    = sit.get("balance", {}).get("total", 0)
            value  = sit.get("usd_value", 0)
            pct    = sit.get("percentage_of_portfolio", 0)
            action = sit.get("action_recommended", "?")
            notes  = sit.get("notes", "")
            print(f"    {asset:8s} | {qty:>18.8f} qty | ${value:>10.2f} ({pct:>5.1f}%) | {action:10s}")
            if notes and cls != "CASH":
                print(f"             -> {notes}")

    print("\n" + "=" * 100)
    print("Analysis saved to: balance_symbol_analysis.json")
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(use_mock: bool) -> Dict[str, Any]:
    if use_mock:
        logger.info("Running in MOCK mode (offline demo data)")
        balances = load_mock_balances()
        prices   = load_mock_prices()
    else:
        logger.info("Connecting to exchange for live data...")
        balances, prices = await fetch_live_data()

    detector = BalanceSymbolDetector(config={
        "min_dust_threshold":      5.0,
        "min_productive_threshold": 25.0,
        "stale_days":              30,
    })

    report = detector.analyze_portfolio(balances, prices)

    with open("balance_symbol_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_analysis(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance symbol detector & analyser")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use offline mock data instead of live exchange",
    )
    args = parser.parse_args()

    print("\nBALANCE SYMBOL DETECTOR v2.0")
    print("=" * 100)
    asyncio.run(async_main(use_mock=args.mock))


if __name__ == "__main__":
    main()
