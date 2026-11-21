"""
config.py — Central configuration module
=========================================
This file loads all settings from your .env file and exposes
them as Python constants used throughout the application.

WHY THIS FILE EXISTS:
- Instead of calling os.getenv() scattered across every file,
  we load everything once here and import from this module.
- This makes it easy to see ALL configuration in one place.
- If a required variable is missing, we fail early with a clear error.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
# This must run before we access any os.getenv() calls
load_dotenv()

# =============================================================
# GOOGLE GEMINI API
# =============================================================
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))

# =============================================================
# APPLICATION METADATA
# =============================================================
APP_TITLE: str = os.getenv("APP_TITLE", "Indian Stock Market Agentic AI")
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")

# =============================================================
# MARKET DATA SETTINGS
# =============================================================
# Number of historical days to fetch (e.g., 90 = last 3 months)
DEFAULT_HISTORY_DAYS: int = int(os.getenv("DEFAULT_HISTORY_DAYS", "90"))

# =============================================================
# NEWS SETTINGS
# =============================================================
MAX_NEWS_ARTICLES: int = int(os.getenv("MAX_NEWS_ARTICLES", "10"))

# =============================================================
# REPORT SETTINGS
# =============================================================
REPORT_OUTPUT_DIR: str = os.getenv("REPORT_OUTPUT_DIR", "reports/")


def validate_config() -> bool:
    """
    Checks that required configuration values are present.
    Call this once at application startup.

    Returns:
        True if configuration is valid, False otherwise.
    """
    errors = []

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_gemini_api_key_here":
        errors.append(
            "GOOGLE_API_KEY is not set. "
            "Please copy .env.example to .env and add your Gemini API key."
        )

    if errors:
        print("\n[CONFIG ERROR] The following issues were found:")
        for err in errors:
            print(f"  ✗ {err}")
        print()
        return False

    print(f"[CONFIG] ✓ Configuration loaded. Using model: {GEMINI_MODEL}")
    return True


if __name__ == "__main__":
    # Quick test: run `python config.py` to verify your .env is set up
    validate_config()
