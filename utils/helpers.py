"""
utils/helpers.py — Shared Helper Functions
===========================================
This file contains small utility functions used by all agents.
Keeping them here avoids copy-pasting the same code across files.

Functions here:
- format_currency()   : Format a number as Indian Rupees (₹)
- format_percentage() : Format a number as a percentage
- safe_round()        : Round a number safely (handles None/NaN)
- get_timestamp()     : Get the current date-time as a readable string
"""

from datetime import datetime
import math


def format_currency(value: float, symbol: str = "₹") -> str:
    """
    Format a number as Indian Rupees.

    Example:
        format_currency(2345.678)  → "₹2,345.68"
        format_currency(None)      → "N/A"
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{symbol}{value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a number as a percentage with sign.

    Example:
        format_percentage(3.456)   → "+3.46%"
        format_percentage(-1.23)   → "-1.23%"
        format_percentage(None)    → "N/A"
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def safe_round(value, decimals: int = 2):
    """
    Round a value safely. Returns None if value is None or NaN.

    Example:
        safe_round(3.14159, 2)  → 3.14
        safe_round(None)        → None
    """
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def get_timestamp() -> str:
    """
    Returns the current date and time as a readable string.

    Example:
        get_timestamp()  → "2026-11-21 22:30:00 IST"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
