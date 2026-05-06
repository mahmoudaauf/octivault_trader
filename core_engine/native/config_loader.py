"""
Native L0: Configuration loader (replaces 300-line legacy ConfigConstants)

Loads config from environment variables or defaults.
Single initialization at startup.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ConfigGroup:
    """Single configuration group"""

    name: str
    values: dict[str, Any]


class ConfigLoader:
    """Load and provide access to configuration"""

    def __init__(self):
        """Initialize config from environment and defaults"""
        self._config: dict[str, ConfigGroup] = {}
        self._load_all()

    def _load_all(self):
        """Load all configuration groups"""
        self._load_symbols()
        self._load_capital()
        self._load_trading()
        self._load_risk()
        self._load_profit_taking()
        self._load_api()

    def _load_symbols(self):
        """Load symbol configuration"""
        symbols_str = os.getenv(
            "SYMBOLS",
            "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,LINKUSDT,DOGEUSDT,AVAXUSDT,PEPEUSDT",
        )
        symbols = [s.strip() for s in symbols_str.split(",")]

        self._config["SYMBOLS"] = ConfigGroup(
            "SYMBOLS",
            {
                "symbols": symbols,
                "limit": int(os.getenv("SYMBOLS_LIMIT", "10")),
                "min_notional_usdt": float(os.getenv("MIN_NOTIONAL_USDT", "5.0")),
            },
        )

    def _load_capital(self):
        """Load capital management configuration"""
        self._config["CAPITAL"] = ConfigGroup(
            "CAPITAL",
            {
                "reserve_pct": float(os.getenv("CAPITAL_RESERVE_PCT", "0.10")),
                "min_reserve_usdt": float(os.getenv("MIN_RESERVE_USDT", "10.00")),
                "max_position_pct": float(os.getenv("MAX_POSITION_PCT", "0.25")),
                "compounding_enabled": os.getenv("COMPOUNDING_ENABLED", "true").lower() == "true",
            },
        )

    def _load_trading(self):
        """Load trading configuration"""
        self._config["TRADING"] = ConfigGroup(
            "TRADING",
            {
                "mode": os.getenv("TRADING_MODE", "paper-trade"),
                "duration_sec": float(os.getenv("DURATION_SEC", "3600")),
                "interval_sec": float(os.getenv("INTERVAL_SEC", "30")),
                "reentry_override": os.getenv("REENTRY_OVERRIDE", "true").lower() == "true",
                "liquidation_enabled": os.getenv("LIQUIDATION_ENABLED", "false").lower() == "true",
            },
        )

    def _load_risk(self):
        """Load risk management configuration"""
        self._config["RISK"] = ConfigGroup(
            "RISK",
            {
                "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "5.0")),
                "take_profit_pct": float(os.getenv("TAKE_PROFIT_PCT", "2.0")),
                "max_drawdown_pct": float(os.getenv("MAX_DRAWDOWN_PCT", "10.0")),
                "volatility_regime": os.getenv("VOLATILITY_REGIME", "enabled").lower() == "enabled",
            },
        )

    def _load_profit_taking(self):
        """Load profit-taking configuration"""
        self._config["PROFIT_TAKING"] = ConfigGroup(
            "PROFIT_TAKING",
            {
                "exit_fee_bps": float(os.getenv("EXIT_FEE_BPS", "10.0")),
                "exit_slip_bps": float(os.getenv("EXIT_SLIP_BPS", "10.0")),
                "tp_buf_bps": float(os.getenv("TP_BUF_BPS", "8.0")),
                "min_net": float(os.getenv("MIN_NET", "0.004")),
                "tp_min": float(os.getenv("TP_MIN", "0.02")),
                "tp_max": float(os.getenv("TP_MAX", "0.04")),
            },
        )

    def _load_api(self):
        """Load API configuration"""
        self._config["API"] = ConfigGroup(
            "API",
            {
                "binance_testnet": os.getenv("BINANCE_TESTNET", "false").lower() == "true",
                "api_key_loaded": os.getenv("BINANCE_API_KEY", "").startswith("0b5eGK"),
                "retry_max_attempts": int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
                "retry_base_delay_sec": float(os.getenv("RETRY_BASE_DELAY_SEC", "0.1")),
            },
        )

    def get(self, group: str, key: str, default: Any = None) -> Any:
        """
        Get config value

        Args:
            group: Config group name (e.g., "SYMBOLS", "CAPITAL")
            key: Config key name
            default: Default value if not found

        Returns:
            Config value or default
        """
        if group not in self._config:
            return default

        return self._config[group].values.get(key, default)

    def get_group(self, group: str) -> dict[str, Any]:
        """
        Get entire config group

        Args:
            group: Config group name

        Returns:
            Dict of config values
        """
        if group not in self._config:
            return {}

        return self._config[group].values.copy()

    def get_all(self) -> dict[str, dict[str, Any]]:
        """
        Get all config groups

        Returns:
            Dict of all groups and their values
        """
        return {group_name: group.values.copy() for group_name, group in self._config.items()}

    def __repr__(self) -> str:
        """String representation"""
        groups = list(self._config.keys())
        return f"ConfigLoader({len(groups)} groups: {', '.join(groups)})"


# Singleton instance (optional)
_config_instance: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get or create singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance
