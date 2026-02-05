"""
Minimal logging setup module for system initialization.
"""
import logging
import sys
from datetime import datetime

def setup_logging(log_file=None, level=logging.INFO):
    """
    Setup basic logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level (default: INFO)
    
    Returns:
        Logger instance
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")
    
    return logger

__all__ = ['setup_logging']
