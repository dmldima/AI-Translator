"""
Centralized logging configuration.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = "translator",
    log_dir: Path = Path("logs"),
    log_level: str = "INFO",
    console_level: str = "INFO",
    use_colors: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup centralized logging.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: File log level
        console_level: Console log level
        use_colors: Use colored console output
        max_bytes: Max log file size
        backup_count: Number of backup files
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    
    if use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '[%(levelname)s] %(message)s'
        )
    else:
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger."""
    return logging.getLogger(name)
