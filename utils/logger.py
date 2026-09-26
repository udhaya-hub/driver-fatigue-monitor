"""
Logging utility using Loguru.
"""

import sys
from pathlib import Path
from loguru import logger
from utils.config_loader import load_config


def setup_logger(name: str = "fatigue_monitor", config_path: str = "config/config.yaml") -> "logger":
    """
    Setup and configure logger.

    Args:
        name: Logger name
        config_path: Path to config file

    Returns:
        Configured logger instance
    """
    config = load_config(config_path)
    log_config = config.get("logging", {})

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stdout,
        format=log_config.get(
            "format",
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level=log_config.get("level", "INFO"),
        colorize=True,
    )

    # File handler
    log_file = log_config.get("file", "logs/app.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_config.get("level", "INFO"),
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "7 days"),
        compression="zip",
    )

    return logger.bind(name=name)


def get_logger(name: str) -> "logger":
    """Get logger instance for module."""
    return logger.bind(name=name)