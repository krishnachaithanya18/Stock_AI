"""
agents/market_data_agent.py — Market Data Agent (Phase 2)
===========================================================
WHAT THIS AGENT DOES:
  - Accepts an NSE/BSE stock symbol (e.g., "RELIANCE.NS", "TCS.NS")
  - Fetches historical price data using yfinance (Yahoo Finance)
  - Collects key market information: price, volume, 52-week range, etc.
  - Calculates basic price changes (1-day, 1-week, 1-month)
  - Returns a structured dictionary of results
  - Handles errors gracefully — never crashes the whole system

WHY yfinance?
  Yahoo Finance provides free, delayed market data for NSE stocks
  when you append ".NS" to the symbol (e.g., RELIANCE → RELIANCE.NS)
  and for BSE stocks with ".BO" (e.g., RELIANCE → RELIANCE.BO).

DATA DISCLAIMER:
  Yahoo Finance data is typically delayed by 15-20 minutes.
  This is for academic/research purposes only. Not for live trading.

WHAT IS A "TICKER"?
  In finance, a ticker symbol is a short code for a stock.
  RELIANCE.NS = Reliance Industries listed on NSE (National Stock Exchange)
  TCS.NS      = Tata Consultancy Services on NSE
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils.helpers import format_currency, format_percentage, safe_round, get_timestamp


def run_market_data_agent(symbol: str, history_days: int = 90) -> dict:
    """
    Main function for the Market Data Agent.

    Parameters:
        symbol       : Stock ticker, e.g. "RELIANCE.NS" or "TCS.NS"
        history_days : How many past days of price data to fetch (default: 90)

    Returns:
        A dictionary containing:
            - status         : "success" or "error"
            - symbol         : The stock symbol used
            - company_name   : Full company name
            - current_price  : Latest closing price (₹)
            - price_change   : Price changes over 1 day, 1 week, 1 month
            - volume         : Trading volume data
            - market_info    : 52-week high/low, P/E ratio, market cap, etc.
            - historical_df  : Pandas DataFrame with OHLCV data (for charts)
            - data_source    : Where data came from + timestamp
            - error_message  : Filled only if status is "error"
    """

    print(f"\n[Market Data Agent] Fetching data for: {symbol}")

    # --- Step 1: Create a yfinance Ticker object ---
    # yf.Ticker() is the gateway to all Yahoo Finance data for a symbol.
    # It doesn't make a network request yet — it just sets up the connection.
    ticker = yf.Ticker(symbol)

    # --- Step 2: Fetch historical price data ---
    # .history() makes the actual API call.
    # It returns a Pandas DataFrame with columns:
    #   Open, High, Low, Close, Volume, Dividends, Stock Splits
    end_date = datetime.today()
    start_date = end_date - timedelta(days=history_days)

    try:
        hist = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True    # Adjusts for splits/dividends automatically
        )
    except Exception as e:
        return _error_result(symbol, f"Failed to fetch historical data: {str(e)}")

    # --- Step 3: Validate the data ---
    # If the symbol is wrong or delisted, hist will be empty.
    if hist is None or hist.empty:
        return _error_result(
            symbol,
            f"No historical data found for '{symbol}'. "
            "Check the symbol. NSE stocks need '.NS' suffix (e.g. RELIANCE.NS)."
        )

    # --- Step 4: Get company info (metadata) ---
    # .info is a dictionary with 100+ fields about the company.
    # We access it safely using .get() to avoid KeyErrors.
    try:
        info = ticker.info
    except Exception:
        info = {}   # If info fails, we continue with what we have

    # --- Step 5: Extract current price ---
    # We use the last row of historical data as "current" price.
    # This is the most reliable approach since .info["currentPrice"] 
    # can sometimes be missing or stale.
    latest_row = hist.iloc[-1]       # Last row = most recent trading day
    current_price = safe_round(latest_row["Close"])

    # --- Step 6: Calculate price changes ---
    price_changes = _calculate_price_changes(hist)

    # --- Step 7: Extract volume data ---
    volume_data = {
        "latest_volume"    : int(latest_row["Volume"]) if latest_row["Volume"] else 0,
        "avg_volume_30d"   : safe_round(hist["Volume"].tail(30).mean()),
        "volume_formatted" : f"{int(latest_row['Volume']):,}" if latest_row["Volume"] else "N/A",
    }

    # --- Step 8: Extract market metadata from .info ---
    market_info = {
        "company_name"       : info.get("longName") or info.get("shortName") or symbol,
        "sector"             : info.get("sector", "N/A"),
        "industry"           : info.get("industry", "N/A"),
        "market_cap"         : info.get("marketCap"),
        "market_cap_fmt"     : _format_market_cap(info.get("marketCap")),
        "pe_ratio"           : safe_round(info.get("trailingPE")),
        "eps"                : safe_round(info.get("trailingEps")),
        "fifty_two_week_high": safe_round(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low" : safe_round(info.get("fiftyTwoWeekLow")),
        "dividend_yield"     : safe_round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
        "beta"               : safe_round(info.get("beta")),   # Volatility vs market
        "exchange"           : info.get("exchange", "N/A"),
        "currency"           : info.get("currency", "INR"),
    }

    # --- Step 9: Build the final result dictionary ---
    result = {
        "status"        : "success",
        "symbol"        : symbol,
        "company_name"  : market_info["company_name"],
        "current_price" : current_price,
        "current_price_fmt": format_currency(current_price),
        "price_change"  : price_changes,
        "volume"        : volume_data,
        "market_info"   : market_info,
        "historical_df" : hist,          # Full DataFrame (for charts later)
        "history_days"  : history_days,
        "data_source"   : {
            "provider"      : "Yahoo Finance (via yfinance)",
            "data_type"     : "Delayed market data (~15-20 min delay)",
            "fetched_at"    : get_timestamp(),
            "data_from"     : start_date.strftime("%Y-%m-%d"),
            "data_to"       : hist.index[-1].strftime("%Y-%m-%d"),
            "total_records" : len(hist),
        },
        "error_message" : None,
    }

    print(f"[Market Data Agent] ✓ Success — {market_info['company_name']}")
    print(f"[Market Data Agent]   Price: {format_currency(current_price)}")
    print(f"[Market Data Agent]   1-Day Change: {price_changes['one_day_pct_fmt']}")
    print(f"[Market Data Agent]   Records: {len(hist)} trading days")

    return result


# =============================================================
# PRIVATE HELPER FUNCTIONS (used only inside this file)
# Functions starting with _ are "private" by convention in Python
# =============================================================

def _calculate_price_changes(hist: pd.DataFrame) -> dict:
    """
    Calculate price changes over different time periods.

    HOW IT WORKS:
        We compare the latest closing price to prices from N days ago.
        Change % = ((current - old) / old) * 100

    Parameters:
        hist : DataFrame with columns including 'Close'

    Returns:
        Dictionary with absolute and percentage changes for 1d, 1w, 1m
    """
    closes = hist["Close"]
    current = closes.iloc[-1]

    def pct_change(past_price):
        if past_price and past_price != 0:
            return ((current - past_price) / past_price) * 100
        return None

    # 1 Day change: compare last row to second-last row
    one_day_price  = closes.iloc[-2] if len(closes) >= 2 else None
    # 1 Week: ~5 trading days ago
    one_week_price = closes.iloc[-6] if len(closes) >= 6 else None
    # 1 Month: ~22 trading days ago
    one_month_price = closes.iloc[-23] if len(closes) >= 23 else None

    one_day_pct    = pct_change(one_day_price)
    one_week_pct   = pct_change(one_week_price)
    one_month_pct  = pct_change(one_month_price)

    return {
        "one_day_abs"     : safe_round(current - one_day_price) if one_day_price else None,
        "one_day_pct"     : safe_round(one_day_pct),
        "one_day_pct_fmt" : format_percentage(one_day_pct),
        "one_week_pct"    : safe_round(one_week_pct),
        "one_week_pct_fmt": format_percentage(one_week_pct),
        "one_month_pct"   : safe_round(one_month_pct),
        "one_month_pct_fmt": format_percentage(one_month_pct),
    }


def _format_market_cap(market_cap) -> str:
    """
    Format market capitalisation into a human-readable Indian format.

    Examples:
        1,500,000,000,000  → "₹1.50 Lakh Cr"
        250,000,000,000    → "₹25,000.00 Cr"
        None               → "N/A"

    WHY:
        Indian finance uses Crore (1 Cr = 10 million) and Lakh Crore
        (1 Lakh Cr = 1 trillion) instead of billions/trillions.
    """
    if not market_cap:
        return "N/A"

    crore = market_cap / 1e7       # 1 Crore = 10,000,000
    if crore >= 100000:            # >= 1 Lakh Crore
        lakh_cr = crore / 100000
        return f"₹{lakh_cr:.2f} Lakh Cr"
    else:
        return f"₹{crore:,.2f} Cr"


def _error_result(symbol: str, message: str) -> dict:
    """
    Returns a standardised error dictionary.
    Every agent function should return a dict (never raise exceptions to callers).
    This keeps the LangGraph workflow from crashing on bad inputs.
    """
    print(f"[Market Data Agent] ✗ ERROR for {symbol}: {message}")
    return {
        "status"        : "error",
        "symbol"        : symbol,
        "company_name"  : symbol,
        "current_price" : None,
        "current_price_fmt": "N/A",
        "price_change"  : {},
        "volume"        : {},
        "market_info"   : {},
        "historical_df" : None,
        "history_days"  : 0,
        "data_source"   : {"fetched_at": get_timestamp()},
        "error_message" : message,
    }
