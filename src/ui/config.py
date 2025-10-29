"""
Centralized UI configuration for all applications.
Eliminates code duplication and ensures consistency.
"""
from typing import Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class ColorScheme:
    """Immutable color scheme configuration."""
    up: str = 'green'
    down: str = 'red'
    neutral: str = 'white'
    dim: str = 'bright_black'
    positive: str = 'green'
    negative: str = 'red'
    warning: str = 'yellow'
    info: str = 'cyan'


@dataclass(frozen=True)
class Theme:
    """Immutable Bloomberg-inspired gray theme."""
    header_bg: str = 'on grey23'
    row_even: str = 'on grey15'
    row_odd: str = 'on grey11'
    border: str = 'grey35'
    panel_bg: str = 'on grey11'


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration."""
    refresh_interval: int = 60  # seconds
    cache_ttl: int = 300  # seconds
    max_retries: int = 3
    timeout: int = 10  # seconds
    date_format: str = '%Y-%m-%d'
    datetime_format: str = '%Y-%m-%d %I:%M:%S %p ET'


# Global instances (read-only)
COLORS = ColorScheme()
THEME = Theme()
CONFIG = AppConfig()
