"""
Input validation utilities for security and data integrity.
"""
import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    value: Optional[str] = None
    error: Optional[str] = None


class StockSymbolValidator:
    """Validates stock ticker symbols."""

    # Valid ticker pattern: 1-5 uppercase letters, optional dot and suffix
    PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')
    MAX_SYMBOLS = 50  # Prevent DoS

    @classmethod
    def validate(cls, symbol: str) -> ValidationResult:
        """
        Validate a single stock symbol.

        Args:
            symbol: Stock symbol to validate

        Returns:
            ValidationResult with validated symbol or error
        """
        if not symbol:
            return ValidationResult(False, None, "Symbol cannot be empty")

        # Clean and uppercase
        cleaned = symbol.strip().upper()

        if len(cleaned) > 10:
            return ValidationResult(False, None, "Symbol too long")

        if not cls.PATTERN.match(cleaned):
            return ValidationResult(
                False,
                None,
                f"Invalid symbol format: {symbol}"
            )

        return ValidationResult(True, cleaned, None)

    @classmethod
    def validate_list(cls, symbols: List[str]) -> ValidationResult:
        """
        Validate a list of stock symbols.

        Args:
            symbols: List of symbols to validate

        Returns:
            ValidationResult with validated symbols or error
        """
        if not symbols:
            return ValidationResult(False, None, "No symbols provided")

        if len(symbols) > cls.MAX_SYMBOLS:
            return ValidationResult(
                False,
                None,
                f"Too many symbols (max {cls.MAX_SYMBOLS})"
            )

        validated = []
        for sym in symbols:
            result = cls.validate(sym)
            if not result.is_valid:
                return ValidationResult(False, None, result.error)
            validated.append(result.value)

        # Remove duplicates while preserving order
        unique = list(dict.fromkeys(validated))

        return ValidationResult(True, unique, None)


class NumericValidator:
    """Validates numeric inputs."""

    @staticmethod
    def validate_int(
        value: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate integer input.

        Args:
            value: String value to validate
            min_value: Optional minimum value
            max_value: Optional maximum value

        Returns:
            ValidationResult with validated integer or error
        """
        try:
            num = int(value)
        except ValueError:
            return ValidationResult(False, None, f"Invalid integer: {value}")

        if min_value is not None and num < min_value:
            return ValidationResult(
                False,
                None,
                f"Value must be >= {min_value}"
            )

        if max_value is not None and num > max_value:
            return ValidationResult(
                False,
                None,
                f"Value must be <= {max_value}"
            )

        return ValidationResult(True, num, None)

    @staticmethod
    def validate_float(
        value: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate float input.

        Args:
            value: String value to validate
            min_value: Optional minimum value
            max_value: Optional maximum value

        Returns:
            ValidationResult with validated float or error
        """
        try:
            num = float(value)
        except ValueError:
            return ValidationResult(False, None, f"Invalid number: {value}")

        if min_value is not None and num < min_value:
            return ValidationResult(
                False,
                None,
                f"Value must be >= {min_value}"
            )

        if max_value is not None and num > max_value:
            return ValidationResult(
                False,
                None,
                f"Value must be <= {max_value}"
            )

        return ValidationResult(True, num, None)


class StringValidator:
    """Validates string inputs."""

    @staticmethod
    def validate_choice(value: str, choices: List[str]) -> ValidationResult:
        """
        Validate that value is one of allowed choices.

        Args:
            value: Value to validate
            choices: List of allowed choices

        Returns:
            ValidationResult with validated choice or error
        """
        cleaned = value.strip().lower()

        if cleaned not in [c.lower() for c in choices]:
            return ValidationResult(
                False,
                None,
                f"Invalid choice. Must be one of: {', '.join(choices)}"
            )

        # Return original case from choices
        for choice in choices:
            if choice.lower() == cleaned:
                return ValidationResult(True, choice, None)

        return ValidationResult(True, cleaned, None)

    @staticmethod
    def validate_length(
        value: str,
        min_length: int = 0,
        max_length: int = 1000
    ) -> ValidationResult:
        """
        Validate string length.

        Args:
            value: String to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Returns:
            ValidationResult with validated string or error
        """
        length = len(value)

        if length < min_length:
            return ValidationResult(
                False,
                None,
                f"String too short (min {min_length} characters)"
            )

        if length > max_length:
            return ValidationResult(
                False,
                None,
                f"String too long (max {max_length} characters)"
            )

        return ValidationResult(True, value, None)


def sanitize_sql_input(value: str) -> str:
    """
    Sanitize input for SQL queries (use parameterized queries instead when possible).

    Args:
        value: Input to sanitize

    Returns:
        Sanitized string
    """
    # Remove dangerous characters
    dangerous = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
    sanitized = value

    for char in dangerous:
        sanitized = sanitized.replace(char, "")

    return sanitized.strip()
