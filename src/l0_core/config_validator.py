# -*- coding: utf-8 -*-
"""
config_validator.py - P9-aligned Configuration Validation Framework

Validates all environment variables and configuration on startup.
Fails fast with clear error messages to prevent deployment errors.

Architecture:
  • Pydantic v2 BaseModel for type safety
  • Environment variable validation
  • Clear error reporting for missing/invalid configs
  • Called from bootstrap before any trading logic
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum

logger = logging.getLogger("ConfigValidator")


class VolatilityRegime(str, Enum):
    """Valid volatility regime values"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class CapitalProfile(str, Enum):
    """Valid capital profile names"""
    BOOTSTRAP_GROWTH = "BOOTSTRAP_GROWTH"
    INSTITUTIONAL = "INSTITUTIONAL"


class BinanceConfig(BaseModel):
    """Binance API configuration validation"""
    
    api_key: str = Field(..., description="Binance API key")
    api_secret: str = Field(..., description="Binance API secret")
    testnet: bool = Field(default=False, description="Use Binance testnet")
    region: str = Field(default="us", description="Binance region (us/eu/com)")
    
    @validator("api_key")
    def validate_api_key(cls, v):
        if not v or len(v) < 20:
            raise ValueError("Invalid Binance API key (too short)")
        return v
    
    @validator("api_secret")
    def validate_api_secret(cls, v):
        if not v or len(v) < 20:
            raise ValueError("Invalid Binance API secret (too short)")
        return v
    
    class Config:
        use_enum_values = True


class TradingConfig(BaseModel):
    """Trading parameters validation"""
    
    max_positions_total: int = Field(default=2, ge=1, le=20, description="Max open positions")
    max_universe_symbols: int = Field(default=30, ge=5, le=500, description="Max symbols in universe")
    max_active_symbols: int = Field(default=5, ge=1, le=20, description="Max active trading symbols")
    min_active_symbols: int = Field(default=3, ge=1, le=10, description="Min active trading symbols")
    
    capital_profile: CapitalProfile = Field(default=CapitalProfile.BOOTSTRAP_GROWTH, description="Capital profile")
    capital_profile_nav_threshold: float = Field(default=500.0, ge=0, description="NAV threshold for profile switching")
    
    min_bars_for_market_ready: int = Field(default=150, ge=50, le=500, description="Min bars for market ready")
    required_symbol_coverage: float = Field(default=0.70, ge=0.0, le=1.0, description="Required symbol coverage")
    
    volatility_regime: VolatilityRegime = Field(default=VolatilityRegime.NORMAL, description="Volatility regime")
    volatility_regime_low_pct: float = Field(default=0.0025, ge=0, description="Low volatility threshold %")
    volatility_regime_high_pct: float = Field(default=0.006, ge=0, description="High volatility threshold %")
    
    max_leverage: float = Field(default=1.0, ge=1.0, le=125.0, description="Max leverage allowed")
    
    @validator("max_active_symbols")
    def validate_max_active(cls, v, values):
        if "min_active_symbols" in values and v < values["min_active_symbols"]:
            raise ValueError("max_active_symbols must be >= min_active_symbols")
        return v
    
    @validator("volatility_regime_high_pct")
    def validate_volatility_high(cls, v, values):
        if "volatility_regime_low_pct" in values and v <= values["volatility_regime_low_pct"]:
            raise ValueError("high volatility threshold must be > low threshold")
        return v
    
    class Config:
        use_enum_values = True


class DatabaseConfig(BaseModel):
    """Database configuration validation"""
    
    db_path: str = Field(default="./data/octivault.db", description="SQLite database path")
    db_encrypted: bool = Field(default=False, description="Enable database encryption")
    db_backup_enabled: bool = Field(default=True, description="Enable database backups")
    
    class Config:
        use_enum_values = True


class LoggingConfig(BaseModel):
    """Logging configuration validation"""
    
    log_level: str = Field(default="INFO", description="Log level (DEBUG/INFO/WARNING/ERROR)")
    async_logging: bool = Field(default=True, description="Use async logging to reduce latency")
    log_file: Optional[str] = Field(default=None, description="Log file path (optional)")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        use_enum_values = True


class ObservabilityConfig(BaseModel):
    """Observability (metrics/tracing) configuration validation"""
    
    prometheus_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, ge=1024, le=65535, description="Prometheus port")
    
    jaeger_enabled: bool = Field(default=False, description="Enable Jaeger tracing")
    jaeger_host: str = Field(default="localhost", description="Jaeger host")
    jaeger_port: int = Field(default=6831, ge=1024, le=65535, description="Jaeger port")
    
    correlation_ids_enabled: bool = Field(default=True, description="Enable correlation IDs")
    
    class Config:
        use_enum_values = True


class FullConfig(BaseModel):
    """Complete configuration schema for Octivault Trader"""
    
    # Environment
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    
    # Binance
    binance: BinanceConfig
    
    # Trading
    trading: TradingConfig = Field(default_factory=TradingConfig)
    
    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Observability
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of {valid_envs}")
        return v.lower()
    
    class Config:
        use_enum_values = True


class ConfigValidator:
    """
    Configuration validator for Octivault Trader.
    
    Usage:
        validator = ConfigValidator()
        config = validator.validate_and_load()  # Raises exception if validation fails
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ConfigValidator")
    
    def _load_env_vars(self) -> Dict[str, Any]:
        """Load environment variables into config dict"""
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
            "binance": {
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                "testnet": os.getenv("BINANCE_TESTNET", "false").lower() == "true",
                "region": os.getenv("BINANCE_REGION", "us"),
            },
            "trading": {
                "max_positions_total": int(os.getenv("MAX_POSITIONS_TOTAL", "2")),
                "max_universe_symbols": int(os.getenv("MAX_UNIVERSE_SYMBOLS", "30")),
                "max_active_symbols": int(os.getenv("MAX_ACTIVE_SYMBOLS", "5")),
                "min_active_symbols": int(os.getenv("MIN_ACTIVE_SYMBOLS", "3")),
                "capital_profile": os.getenv("CAPITAL_PROFILE", "BOOTSTRAP_GROWTH"),
                "capital_profile_nav_threshold": float(os.getenv("CAPITAL_PROFILE_NAV_THRESHOLD", "500.0")),
                "min_bars_for_market_ready": int(os.getenv("MIN_BARS_FOR_MARKET_READY", "150")),
                "required_symbol_coverage": float(os.getenv("REQUIRED_SYMBOL_COVERAGE", "0.70")),
                "volatility_regime": os.getenv("VOLATILITY_REGIME", "normal"),
                "volatility_regime_low_pct": float(os.getenv("VOLATILITY_REGIME_LOW_PCT", "0.0025")),
                "volatility_regime_high_pct": float(os.getenv("VOLATILITY_REGIME_HIGH_PCT", "0.006")),
                "max_leverage": float(os.getenv("MAX_LEVERAGE", "1.0")),
            },
            "database": {
                "db_path": os.getenv("DB_PATH", "./data/octivault.db"),
                "db_encrypted": os.getenv("DB_ENCRYPTED", "false").lower() == "true",
                "db_backup_enabled": os.getenv("DB_BACKUP_ENABLED", "true").lower() == "true",
            },
            "logging": {
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "async_logging": os.getenv("ASYNC_LOGGING", "true").lower() == "true",
                "log_file": os.getenv("LOG_FILE", None),
            },
            "observability": {
                "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true",
                "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
                "jaeger_enabled": os.getenv("JAEGER_ENABLED", "false").lower() == "true",
                "jaeger_host": os.getenv("JAEGER_HOST", "localhost"),
                "jaeger_port": int(os.getenv("JAEGER_PORT", "6831")),
                "correlation_ids_enabled": os.getenv("CORRELATION_IDS_ENABLED", "true").lower() == "true",
            },
        }
    
    def validate_and_load(self) -> FullConfig:
        """
        Validate environment variables and load configuration.
        
        Raises:
            ValidationError: If any configuration is invalid
            ValueError: If required env vars are missing
        
        Returns:
            FullConfig: Validated configuration object
        """
        try:
            env_vars = self._load_env_vars()
            config = FullConfig(**env_vars)
            self.logger.info("✅ Configuration validation passed")
            self._log_config_summary(config)
            return config
        except ValidationError as e:
            self.logger.error("❌ Configuration validation FAILED")
            self._log_validation_errors(e)
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during config validation: {e}")
            raise
    
    def _log_validation_errors(self, error: ValidationError):
        """Log validation errors in human-readable format"""
        self.logger.error("\n" + "="*80)
        self.logger.error("CONFIGURATION VALIDATION ERRORS")
        self.logger.error("="*80)
        
        for err in error.errors():
            field = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            self.logger.error(f"\n❌ {field}")
            self.logger.error(f"   Error: {msg}")
            if "ctx" in err and "error" in err["ctx"]:
                self.logger.error(f"   Detail: {err['ctx']['error']}")
        
        self.logger.error("\n" + "="*80)
        self.logger.error("Fix the above configuration errors and restart.")
        self.logger.error("="*80 + "\n")
    
    def _log_config_summary(self, config: FullConfig):
        """Log a summary of loaded configuration"""
        self.logger.info("\n" + "="*80)
        self.logger.info("CONFIGURATION SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Environment: {config.environment}")
        self.logger.info(f"Debug Mode: {config.debug_mode}")
        self.logger.info(f"Binance Region: {config.binance.region}")
        self.logger.info(f"Binance Testnet: {config.binance.testnet}")
        self.logger.info(f"Trading Profile: {config.trading.capital_profile}")
        self.logger.info(f"Max Positions: {config.trading.max_positions_total}")
        self.logger.info(f"Max Active Symbols: {config.trading.max_active_symbols}")
        self.logger.info(f"Max Leverage: {config.trading.max_leverage}x")
        self.logger.info(f"Database: {config.database.db_path}")
        self.logger.info(f"Async Logging: {config.logging.async_logging}")
        self.logger.info(f"Prometheus: {config.observability.prometheus_enabled}")
        self.logger.info(f"Jaeger Tracing: {config.observability.jaeger_enabled}")
        self.logger.info("="*80 + "\n")


def validate_config_on_startup() -> FullConfig:
    """
    Entry point for configuration validation on startup.
    
    Call this from main_phased.py before any trading logic.
    
    Returns:
        FullConfig: Validated configuration
    
    Raises:
        ValidationError: If configuration is invalid
    """
    validator = ConfigValidator()
    return validator.validate_and_load()
