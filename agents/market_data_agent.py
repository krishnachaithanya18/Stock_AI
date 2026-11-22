"""
agents/market_data_agent.py — Stock Data Agent (Agent A)
=========================================================
WHAT THIS AGENT DOES:
  - Accepts any Indian stock symbol (e.g. "RELIANCE", "RELIANCE.NS", "500325", "TCS.BO")
  - Uses smart symbol resolver to ensure valid Yahoo Finance tickers
  - Fetches historical OHLCV data using yfinance
  - Collects comprehensive company metadata: price, volume, market cap, 52-week range
  - Calculates multi-period price changes (1-day, 1-week, 1-month, 3-month)
  - Computes volume metrics (30-day avg volume, volume surge ratio)
  - Computes 52-week range positioning percentage
  - Returns a clean structured dictionary and handles missing data gracefully
"""

from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from utils.helpers import format_currency, format_percentage, safe_round, get_timestamp
from utils.stock_search import resolve_stock_symbol


def run_market_data_agent(symbol: str, history_days: int = 90, exchange: str = "NSE") -> dict:
    """
    Main function for the Stock Data Agent.

    Parameters:
        symbol       : Stock ticker, NSE symbol, or BSE numeric code
        history_days : Days of historical price data (default: 90)
        exchange     : Preferred exchange ("NSE" or "BSE")

    Returns:
        Structured dictionary containing:
            - status            : "success" or "error"
            - symbol            : Resolved Yahoo Finance ticker (e.g. "RELIANCE.NS")
            - raw_symbol        : Original input symbol
            - company_name      : Full registered company name
            - current_price     : Latest closing/current price (₹)
            - current_price_fmt : Formatted price string (₹)
            - price_change      : Multi-period price changes (1d, 1w, 1m, 3m)
            - volume            : Latest volume, 30d avg, surge ratio
            - market_info       : Valuation & market metadata
            - historical_df     : Pandas DataFrame with OHLCV data
            - history_days      : Number of days fetched
            - data_source       : Provider metadata and timestamp
            - error_message     : Error string if status == "error"
    """
    raw_input = symbol
    # Resolve ticker using Indian stock universe and BSE code mapper
    resolved_ticker, meta_lookup = resolve_stock_symbol(symbol, exchange=exchange)

    print(f"\n[Stock Data Agent] Fetching data for: {raw_input} -> {resolved_ticker}")

    # Fetch yfinance Ticker object
    ticker = yf.Ticker(resolved_ticker)

    # Calculate date range
    end_date = datetime.today()
    # Fetch at least 250 days if 90 requested to allow SMA 200 calculation in technical agent!
    fetch_days = max(history_days, 300)
    start_date = end_date - timedelta(days=fetch_days)

    try:
        hist = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True
        )
    except Exception as e:
        return _error_result(resolved_ticker, f"Failed to fetch historical data: {str(e)}")

    if hist is None or hist.empty:
        # If .NS failed and was default, try .BO fallback or vice-versa
        alt_ticker = resolved_ticker.replace(".NS", ".BO") if resolved_ticker.endswith(".NS") else resolved_ticker.replace(".BO", ".NS")
        try:
            alt_ticker_obj = yf.Ticker(alt_ticker)
            hist = alt_ticker_obj.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True
            )
            if hist is not None and not hist.empty:
                resolved_ticker = alt_ticker
                ticker = alt_ticker_obj
        except Exception:
            pass

    if hist is None or hist.empty:
        return _error_result(
            resolved_ticker,
            f"No historical data found for '{raw_input}' (resolved as '{resolved_ticker}'). "
            "Please check the company name, NSE symbol (e.g. RELIANCE, TCS) or BSE code (e.g. 500325)."
        )

    # Fetch company info metadata
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    latest_row = hist.iloc[-1]
    current_price = safe_round(latest_row["Close"])

    # Calculate price changes over 1d, 1w, 1m, 3m
    price_changes = _calculate_price_changes(hist)

    # Volume metrics
    vol_latest = int(latest_row["Volume"]) if "Volume" in latest_row and pd.notna(latest_row["Volume"]) else 0
    avg_vol_30 = safe_round(hist["Volume"].tail(30).mean()) if "Volume" in hist else None
    volume_surge_ratio = safe_round(vol_latest / avg_vol_30, 2) if (avg_vol_30 and avg_vol_30 > 0) else 1.0

    volume_data = {
        "latest_volume": vol_latest,
        "avg_volume_30d": avg_vol_30,
        "volume_surge_ratio": volume_surge_ratio,
        "volume_formatted": f"{vol_latest:,}" if vol_latest else "N/A",
        "avg_volume_formatted": f"{int(avg_vol_30):,}" if avg_vol_30 else "N/A"
    }

    # Company name resolution
    meta_name = meta_lookup.get("name") if meta_lookup else None
    company_name = (
        info.get("longName") or
        info.get("shortName") or
        meta_name or
        resolved_ticker.split(".")[0]
    )

    # 52-Week High/Low
    high_52 = safe_round(info.get("fiftyTwoWeekHigh"))
    low_52 = safe_round(info.get("fiftyTwoWeekLow"))
    if not high_52 or not low_52:
        # Compute from available history
        high_52 = safe_round(hist["High"].tail(252).max()) if len(hist) > 0 else None
        low_52 = safe_round(hist["Low"].tail(252).min()) if len(hist) > 0 else None

    # Calculate 52-week range position
    range_position_pct = None
    if current_price and high_52 and low_52 and high_52 > low_52:
        range_position_pct = safe_round(((current_price - low_52) / (high_52 - low_52)) * 100, 1)

    market_info = {
        "company_name": company_name,
        "sector": info.get("sector") or (meta_lookup.get("sector") if meta_lookup else "Indian Equity"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "market_cap_fmt": _format_market_cap(info.get("marketCap")),
        "pe_ratio": safe_round(info.get("trailingPE")),
        "forward_pe": safe_round(info.get("forwardPE")),
        "pb_ratio": safe_round(info.get("priceToBook")),
        "eps": safe_round(info.get("trailingEps")),
        "fifty_two_week_high": high_52,
        "fifty_two_week_low": low_52,
        "range_position_pct": range_position_pct,
        "dividend_yield": safe_round(info.get("dividendYield") or 0.0, 2),  # yfinance returns as % already (e.g. 0.44 = 0.44%)
        "beta": safe_round(info.get("beta")),
        "exchange": "BSE" if resolved_ticker.endswith(".BO") else "NSE",
        "currency": info.get("currency", "INR"),
        "book_value": safe_round(info.get("bookValue")),
        "debt_to_equity": safe_round(info.get("debtToEquity")),
        "return_on_equity": safe_round((info.get("returnOnEquity") or 0) * 100, 2) if info.get("returnOnEquity") else None,
        "revenue_growth": safe_round((info.get("revenueGrowth") or 0) * 100, 2) if info.get("revenueGrowth") else None,
        "earnings_growth": safe_round((info.get("earningsGrowth") or 0) * 100, 2) if info.get("earningsGrowth") else None,
    }

    result = {
        "status": "success",
        "symbol": resolved_ticker,
        "raw_symbol": raw_input,
        "company_name": company_name,
        "current_price": current_price,
        "current_price_fmt": format_currency(current_price),
        "price_change": price_changes,
        "volume": volume_data,
        "market_info": market_info,
        "historical_df": hist,
        "history_days": len(hist),
        "data_source": {
            "provider": "Yahoo Finance (NSE / BSE)",
            "data_type": "Delayed market data (~15-20 min delay)",
            "fetched_at": get_timestamp(),
            "data_from": hist.index[0].strftime("%Y-%m-%d"),
            "data_to": hist.index[-1].strftime("%Y-%m-%d"),
            "total_records": len(hist),
        },
        "error_message": None,
    }

    print(f"[Stock Data Agent] ✓ Success: {company_name} ({resolved_ticker})")
    print(f"[Stock Data Agent]   Price: {format_currency(current_price)} | 1-Day: {price_changes['one_day_pct_fmt']}")

    return result


def _calculate_price_changes(hist: pd.DataFrame) -> dict:
    closes = hist["Close"]
    current = closes.iloc[-1]

    def pct_change(past_price):
        if past_price and past_price != 0:
            return ((current - past_price) / past_price) * 100
        return None

    one_day_price = closes.iloc[-2] if len(closes) >= 2 else None
    one_week_price = closes.iloc[-6] if len(closes) >= 6 else None
    one_month_price = closes.iloc[-23] if len(closes) >= 23 else None
    three_month_price = closes.iloc[-66] if len(closes) >= 66 else None

    one_day_pct = pct_change(one_day_price)
    one_week_pct = pct_change(one_week_price)
    one_month_pct = pct_change(one_month_price)
    three_month_pct = pct_change(three_month_price)

    return {
        "one_day_abs": safe_round(current - one_day_price) if one_day_price else None,
        "one_day_pct": safe_round(one_day_pct),
        "one_day_pct_fmt": format_percentage(one_day_pct),
        "one_week_pct": safe_round(one_week_pct),
        "one_week_pct_fmt": format_percentage(one_week_pct),
        "one_month_pct": safe_round(one_month_pct),
        "one_month_pct_fmt": format_percentage(one_month_pct),
        "three_month_pct": safe_round(three_month_pct),
        "three_month_pct_fmt": format_percentage(three_month_pct),
    }


def _format_market_cap(market_cap) -> str:
    if not market_cap:
        return "N/A"
    crore = market_cap / 1e7
    if crore >= 100000:
        lakh_cr = crore / 100000
        return f"₹{lakh_cr:.2f} Lakh Cr"
    else:
        return f"₹{crore:,.2f} Cr"


def _error_result(symbol: str, message: str) -> dict:
    print(f"[Stock Data Agent] ✗ ERROR for {symbol}: {message}")
    return {
        "status": "error",
        "symbol": symbol,
        "raw_symbol": symbol,
        "company_name": symbol,
        "current_price": None,
        "current_price_fmt": "N/A",
        "price_change": {},
        "volume": {},
        "market_info": {},
        "historical_df": None,
        "history_days": 0,
        "data_source": {"fetched_at": get_timestamp()},
        "error_message": message,
    }


# Export alias
run_stock_data_agent = run_market_data_agent
