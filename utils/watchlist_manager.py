"""
utils/watchlist_manager.py — Local Watchlist & Price Alerts Manager
===================================================================
Persists user watchlists and price alerts locally in JSON format.
Enables quick access, tracking of daily changes, and price trigger detection.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import yfinance as yf
from utils.helpers import safe_round, format_percentage, format_currency


WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "watchlist.json")
ALERTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alerts.json")


# =======================================================================
# WATCHLIST FUNCTIONS
# =======================================================================

def load_watchlist() -> List[Dict]:
    """Loads watchlist from local JSON file."""
    if not os.path.exists(WATCHLIST_FILE):
        default_list = [
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "added_at": "2026-11-20", "last_verdict": "BUY"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "added_at": "2026-11-20", "last_verdict": "HOLD"},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited", "added_at": "2026-11-20", "last_verdict": "BUY"},
        ]
        save_watchlist(default_list)
        return default_list

    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_watchlist(items: List[Dict]) -> bool:
    """Saves watchlist items to local JSON file."""
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return True
    except Exception:
        return False


def add_to_watchlist(symbol: str, name: str, verdict: str = "N/A") -> bool:
    """Adds a stock to the watchlist if not already present."""
    watchlist = load_watchlist()
    for item in watchlist:
        if item["symbol"].upper() == symbol.upper():
            item["last_verdict"] = verdict
            return save_watchlist(watchlist)

    watchlist.append({
        "symbol": symbol.upper(),
        "name": name,
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "last_verdict": verdict,
    })
    return save_watchlist(watchlist)


def remove_from_watchlist(symbol: str) -> bool:
    """Removes a stock from the watchlist."""
    watchlist = load_watchlist()
    filtered = [item for item in watchlist if item["symbol"].upper() != symbol.upper()]
    return save_watchlist(filtered)


def get_watchlist_live_data() -> List[Dict]:
    """
    Fetches latest prices and 1-day changes for all symbols in the watchlist.
    """
    watchlist = load_watchlist()
    if not watchlist:
        return []

    symbols = [item["symbol"] for item in watchlist]
    try:
        data = yf.download(symbols, period="5d", interval="1d", progress=False)
        closes = data["Close"]

        enriched = []
        for item in watchlist:
            sym = item["symbol"]
            current_price = None
            pct_change = None

            if sym in closes:
                series = closes[sym].dropna()
                if len(series) >= 2:
                    c = series.iloc[-1]
                    p = series.iloc[-2]
                    current_price = safe_round(c)
                    pct_change = safe_round(((c - p) / p) * 100, 2)
                elif len(series) == 1:
                    current_price = safe_round(series.iloc[-1])
                    pct_change = 0.0

            enriched.append({
                "symbol": sym,
                "name": item.get("name", sym),
                "added_at": item.get("added_at", ""),
                "last_verdict": item.get("last_verdict", "N/A"),
                "current_price": current_price,
                "current_price_fmt": format_currency(current_price) if current_price else "N/A",
                "pct_change": pct_change,
                "pct_change_fmt": format_percentage(pct_change) if pct_change is not None else "N/A",
            })
        return enriched
    except Exception:
        return watchlist


# =======================================================================
# PRICE ALERTS FUNCTIONS
# =======================================================================

def load_alerts() -> List[Dict]:
    """Loads price alerts from local JSON file."""
    if not os.path.exists(ALERTS_FILE):
        return []
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_alerts(alerts: List[Dict]) -> bool:
    """Saves alerts to local JSON file."""
    try:
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)
        return True
    except Exception:
        return False


def add_price_alert(symbol: str, target_price: float, condition: str = "ABOVE", note: str = "") -> bool:
    """
    Adds a new price alert.
    condition: 'ABOVE' (triggers when price >= target_price) or 'BELOW' (triggers when price <= target_price)
    """
    alerts = load_alerts()
    alerts.append({
        "id": f"{symbol}_{int(datetime.now().timestamp())}",
        "symbol": symbol.upper(),
        "target_price": target_price,
        "condition": condition.upper(),
        "note": note,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "triggered": False,
        "triggered_at": None,
    })
    return save_alerts(alerts)


def check_alerts_status() -> List[Dict]:
    """
    Checks all saved alerts against current market prices.
    Returns alerts with their current trigger status.
    """
    alerts = load_alerts()
    if not alerts:
        return []

    symbols = list(set(a["symbol"] for a in alerts))
    try:
        data = yf.download(symbols, period="2d", interval="1d", progress=False)
        closes = data["Close"]

        updated = False
        for a in alerts:
            sym = a["symbol"]
            if sym in closes:
                series = closes[sym].dropna()
                if not series.empty:
                    cur = series.iloc[-1]
                    a["current_price"] = safe_round(cur)
                    a["current_price_fmt"] = format_currency(cur)

                    if not a.get("triggered"):
                        if a["condition"] == "ABOVE" and cur >= a["target_price"]:
                            a["triggered"] = True
                            a["triggered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            updated = True
                        elif a["condition"] == "BELOW" and cur <= a["target_price"]:
                            a["triggered"] = True
                            a["triggered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            updated = True

        if updated:
            save_alerts(alerts)
    except Exception:
        pass

    return alerts


def remove_alert(alert_id: str) -> bool:
    alerts = load_alerts()
    filtered = [a for a in alerts if a.get("id") != alert_id]
    return save_alerts(filtered)
