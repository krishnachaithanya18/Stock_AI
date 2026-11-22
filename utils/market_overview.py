"""
utils/market_overview.py — Indian Market Benchmark Indices & Market Status
==========================================================================
Fetches live/delayed data for:
- NIFTY 50 (^NSEI)
- BSE SENSEX (^BSESN)
- BANK NIFTY (^NSEBANK)

Also determines whether the Indian markets (NSE/BSE) are currently OPEN or CLOSED
based on Indian Standard Time (IST: 09:15 to 15:30 IST, Monday-Friday, excluding holidays).
"""

from datetime import datetime, time
import pytz
import yfinance as yf
from utils.helpers import safe_round, format_percentage, format_currency


INDICES = {
    "NIFTY 50": {"symbol": "^NSEI", "name": "NIFTY 50", "exchange": "NSE"},
    "SENSEX": {"symbol": "^BSESN", "name": "BSE SENSEX", "exchange": "BSE"},
    "BANK NIFTY": {"symbol": "^NSEBANK", "name": "NIFTY BANK", "exchange": "NSE"},
}


def get_market_status() -> dict:
    """
    Determines Indian stock market status (NSE/BSE).
    Regular trading hours: 09:15 to 15:30 IST, Monday through Friday.
    """
    try:
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
    except Exception:
        # Fallback to system time if pytz issue
        now_ist = datetime.now()

    weekday = now_ist.weekday()  # 0=Monday, ..., 6=Sunday
    current_time = now_ist.time()

    market_open_time = time(9, 15)
    market_close_time = time(15, 30)

    is_weekday = weekday < 5
    is_market_hours = market_open_time <= current_time <= market_close_time

    is_open = is_weekday and is_market_hours

    if is_open:
        status_label = "MARKET OPEN"
        status_desc = "Live Trading Session (09:15 - 15:30 IST)"
        status_badge = "🟢 OPEN"
    elif is_weekday and current_time < market_open_time:
        status_label = "PRE-MARKET"
        status_desc = "Opens at 09:15 AM IST"
        status_badge = "🟡 PRE-MARKET"
    else:
        status_label = "MARKET CLOSED"
        status_desc = "Regular trading resumes next trading day at 09:15 AM IST"
        status_badge = "🔴 CLOSED"

    return {
        "is_open": is_open,
        "label": status_label,
        "description": status_desc,
        "badge": status_badge,
        "current_ist": now_ist.strftime("%d %b %Y, %I:%M %p IST"),
    }


def fetch_market_indices() -> list:
    """
    Fetches the latest index levels and 1-day changes for NIFTY 50, SENSEX, and BANK NIFTY.
    Returns a list of dicts suitable for top-level dashboard metrics.
    """
    symbols = [item["symbol"] for item in INDICES.values()]
    results = []

    try:
        data = yf.download(symbols, period="5d", interval="1d", progress=False)
        closes = data["Close"]

        for label, meta in INDICES.items():
            sym = meta["symbol"]
            if sym in closes:
                series = closes[sym].dropna()
                if len(series) >= 2:
                    current = safe_round(series.iloc[-1])
                    previous = safe_round(series.iloc[-2])
                    change_pts = safe_round(current - previous)
                    change_pct = safe_round(((current - previous) / previous) * 100)
                elif len(series) == 1:
                    current = safe_round(series.iloc[-1])
                    change_pts = 0.0
                    change_pct = 0.0
                else:
                    current, change_pts, change_pct = None, None, None
            else:
                current, change_pts, change_pct = None, None, None

            results.append({
                "name": label,
                "exchange": meta["exchange"],
                "value": current,
                "value_fmt": f"{current:,.2f}" if current else "N/A",
                "change_pts": change_pts,
                "change_pts_fmt": f"{change_pts:+,.2f}" if change_pts is not None else "N/A",
                "change_pct": change_pct,
                "change_pct_fmt": format_percentage(change_pct) if change_pct is not None else "N/A",
                "direction": "up" if (change_pct or 0) >= 0 else "down"
            })
    except Exception as e:
        # Fallback values if bulk download fails
        for label, meta in INDICES.items():
            results.append({
                "name": label,
                "exchange": meta["exchange"],
                "value": None,
                "value_fmt": "Unavailable",
                "change_pts": None,
                "change_pts_fmt": "N/A",
                "change_pct": None,
                "change_pct_fmt": "N/A",
                "direction": "neutral"
            })

    return results
