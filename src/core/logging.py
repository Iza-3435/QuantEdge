"""Structured logging with JSON support."""
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime
from loguru import logger
from src.core.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging():
    """Configure application logging."""

    # Remove default handlers
    logger.remove()

    log_level = settings.monitoring.logging.level

    # Console handler with color
    if settings.monitoring.logging.format == "json":
        logger.add(
            sys.stdout,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            colorize=True,
            serialize=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

    # File handler with rotation
    log_file = Path(settings.monitoring.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=settings.monitoring.logging.rotation,
        retention=settings.monitoring.logging.retention,
        compression="zip",
        serialize=settings.monitoring.logging.format == "json",
    )

    return logger


# Global logger instance
app_logger = setup_logging()


def log_execution_time(func):
    """Decorator to log function execution time."""
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            app_logger.info(
                f"{func.__name__} executed successfully",
                extra={"execution_time": execution_time, "function": func.__name__}
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            app_logger.error(
                f"{func.__name__} failed",
                extra={"execution_time": execution_time, "function": func.__name__, "error": str(e)}
            )
            raise

    return wrapper
