"""
agents/technical_agent.py — Technical Analysis Agent (Phase 3)
===============================================================
WHAT THIS AGENT DOES:
  - Receives the historical price DataFrame from the Market Data Agent
  - Calculates four key technical indicators using the 'ta' library:
      1. RSI  — Relative Strength Index
      2. MACD — Moving Average Convergence Divergence
      3. SMA 20 — Simple Moving Average (20-day)
      4. SMA 50 — Simple Moving Average (50-day)
  - Identifies bullish (positive) and bearish (negative) signals
  - Explains every indicator in plain, beginner-friendly language
  - Returns structured findings for the Research Agent

WHY TECHNICAL ANALYSIS?
  Technical analysis studies price and volume patterns to understand
  the market's current mood (momentum, trend, overbought/oversold).
  It does NOT predict the future — it describes current conditions.

DISCLAIMER:
  Technical indicators are tools for analysis, not guarantees.
  They are based on historical price data only.
"""

import pandas as pd
import ta
from utils.helpers import safe_round, format_percentage, get_timestamp


def run_technical_agent(market_data: dict) -> dict:
    """
    Main function for the Technical Analysis Agent.

    Parameters:
        market_data : The result dictionary from run_market_data_agent()
                      We need market_data["historical_df"] (the price DataFrame)

    Returns:
        A dictionary containing:
            - status         : "success" or "error"
            - symbol         : Stock symbol
            - indicators     : RSI, MACD, SMA values with interpretations
            - signals        : List of bullish and bearish signal strings
            - overall_signal : "BULLISH", "BEARISH", or "NEUTRAL"
            - signal_score   : Numeric score (positive=bullish, negative=bearish)
            - summary        : Plain-English summary for the report
            - analysed_at    : Timestamp
            - error_message  : Only if status is "error"
    """

    symbol = market_data.get("symbol", "UNKNOWN")
    print(f"\n[Technical Agent] Analysing indicators for: {symbol}")

    # --- Step 1: Validate input ---
    # We cannot do technical analysis without price data
    if market_data.get("status") == "error":
        return _error_result(symbol, "Cannot run technical analysis — market data fetch failed.")

    hist = market_data.get("historical_df")
    if hist is None or hist.empty:
        return _error_result(symbol, "Historical price DataFrame is missing or empty.")

    # We need at least 50 rows for SMA-50 to be meaningful
    if len(hist) < 20:
        return _error_result(
            symbol,
            f"Only {len(hist)} trading days available. Need at least 20 for analysis."
        )

    # --- Step 2: Extract the Close price series ---
    # 'Close' is the price at market close — the standard input for indicators
    close = hist["Close"]

    # --- Step 3: Calculate each indicator ---
    rsi_data  = _calculate_rsi(close)
    macd_data = _calculate_macd(close)
    sma_data  = _calculate_sma(close)

    # --- Step 4: Generate signals (bullish / bearish) ---
    bullish_signals, bearish_signals, signal_score = _generate_signals(
        rsi_data, macd_data, sma_data, market_data
    )

    # --- Step 5: Determine overall signal ---
    if signal_score >= 2:
        overall_signal = "BULLISH"
    elif signal_score <= -2:
        overall_signal = "BEARISH"
    else:
        overall_signal = "NEUTRAL"

    # --- Step 6: Build plain-English summary ---
    summary = _build_summary(symbol, rsi_data, macd_data, sma_data, overall_signal, signal_score)

    result = {
        "status"         : "success",
        "symbol"         : symbol,
        "indicators"     : {
            "rsi"  : rsi_data,
            "macd" : macd_data,
            "sma"  : sma_data,
        },
        "signals"        : {
            "bullish": bullish_signals,
            "bearish": bearish_signals,
        },
        "overall_signal" : overall_signal,
        "signal_score"   : signal_score,
        "summary"        : summary,
        "data_points"    : len(hist),
        "analysed_at"    : get_timestamp(),
        "error_message"  : None,
    }

    print(f"[Technical Agent] ✓ RSI: {rsi_data['value']} | MACD: {macd_data['histogram_latest']} | Signal: {overall_signal}")
    print(f"[Technical Agent]   Bullish signals: {len(bullish_signals)} | Bearish signals: {len(bearish_signals)}")

    return result


# =============================================================
# INDICATOR CALCULATION FUNCTIONS
# =============================================================

def _calculate_rsi(close: pd.Series, period: int = 14) -> dict:
    """
    Calculate RSI — Relative Strength Index.

    WHAT IS RSI?
      RSI measures the speed and change of price movements.
      It oscillates between 0 and 100.

    HOW TO READ IT:
      RSI > 70  → "Overbought" — the stock may have risen too fast and could pull back
      RSI < 30  → "Oversold"   — the stock may have fallen too fast and could bounce back
      RSI 30–70 → "Neutral"    — no extreme condition

    FORMULA (simplified):
      RSI = 100 - (100 / (1 + RS))
      where RS = Average Gain over N days / Average Loss over N days

    Parameters:
        close  : Pandas Series of closing prices
        period : Lookback period (default: 14 days — industry standard)
    """

    # ta.momentum.RSIIndicator computes RSI for the full price series
    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=period)
    rsi_series    = rsi_indicator.rsi()

    current_rsi   = safe_round(rsi_series.iloc[-1])
    previous_rsi  = safe_round(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else None

    # --- Interpret the RSI value ---
    if current_rsi is None:
        zone = "Unknown"
        interpretation = "RSI data unavailable."
    elif current_rsi >= 70:
        zone = "Overbought"
        interpretation = (
            f"RSI is {current_rsi} (above 70). The stock may be overbought — "
            "it has risen sharply and might face a short-term pullback."
        )
    elif current_rsi <= 30:
        zone = "Oversold"
        interpretation = (
            f"RSI is {current_rsi} (below 30). The stock may be oversold — "
            "it has fallen sharply and might bounce back."
        )
    else:
        zone = "Neutral"
        interpretation = (
            f"RSI is {current_rsi} (between 30 and 70). The stock is in neutral territory — "
            "no extreme overbought or oversold condition."
        )

    return {
        "value"          : current_rsi,
        "previous_value" : previous_rsi,
        "zone"           : zone,           # "Overbought", "Oversold", or "Neutral"
        "period"         : period,
        "interpretation" : interpretation,
        "series"         : rsi_series,     # Full series (for charts)
    }


def _calculate_macd(close: pd.Series,
                    fast: int = 12,
                    slow: int = 26,
                    signal: int = 9) -> dict:
    """
    Calculate MACD — Moving Average Convergence Divergence.

    WHAT IS MACD?
      MACD compares two exponential moving averages to show momentum.
      It has three parts:
        1. MACD Line    = EMA(12) - EMA(26)   [fast EMA minus slow EMA]
        2. Signal Line  = EMA(9) of MACD Line [smoothed MACD]
        3. Histogram    = MACD Line - Signal Line [shows momentum strength]

    HOW TO READ IT:
      MACD Line crosses ABOVE Signal Line → Bullish crossover (momentum turning positive)
      MACD Line crosses BELOW Signal Line → Bearish crossover (momentum turning negative)
      Histogram positive and growing      → Strengthening bullish momentum
      Histogram negative and shrinking    → Weakening bearish momentum (possible reversal)

    Parameters:
        close  : Pandas Series of closing prices
        fast   : Fast EMA period (default 12)
        slow   : Slow EMA period (default 26)
        signal : Signal line period (default 9)
    """

    macd_indicator = ta.trend.MACD(
        close=close,
        window_fast=fast,
        window_slow=slow,
        window_sign=signal,
    )

    macd_line    = macd_indicator.macd()
    signal_line  = macd_indicator.macd_signal()
    histogram    = macd_indicator.macd_diff()   # MACD Line - Signal Line

    macd_latest      = safe_round(macd_line.iloc[-1], 4)
    signal_latest    = safe_round(signal_line.iloc[-1], 4)
    histogram_latest = safe_round(histogram.iloc[-1], 4)

    # Detect crossover by comparing previous and current positions
    prev_macd   = macd_line.iloc[-2]   if len(macd_line) >= 2 else None
    prev_signal = signal_line.iloc[-2] if len(signal_line) >= 2 else None

    crossover = "None"
    if prev_macd is not None and prev_signal is not None and macd_latest is not None and signal_latest is not None:
        was_below = prev_macd < prev_signal
        is_above  = macd_latest > signal_latest
        was_above = prev_macd > prev_signal
        is_below  = macd_latest < signal_latest

        if was_below and is_above:
            crossover = "Bullish Crossover"
        elif was_above and is_below:
            crossover = "Bearish Crossover"
        elif is_above:
            crossover = "Bullish (MACD above Signal)"
        else:
            crossover = "Bearish (MACD below Signal)"

    # Build interpretation
    if macd_latest is not None and signal_latest is not None:
        if macd_latest > signal_latest:
            interpretation = (
                f"MACD ({macd_latest}) is above the Signal line ({signal_latest}). "
                "This is a bullish momentum signal."
            )
        else:
            interpretation = (
                f"MACD ({macd_latest}) is below the Signal line ({signal_latest}). "
                "This is a bearish momentum signal."
            )
    else:
        interpretation = "MACD data unavailable."

    return {
        "macd_latest"       : macd_latest,
        "signal_latest"     : signal_latest,
        "histogram_latest"  : histogram_latest,
        "crossover"         : crossover,
        "interpretation"    : interpretation,
        "macd_series"       : macd_line,
        "signal_series"     : signal_line,
        "histogram_series"  : histogram,
        "settings"          : {"fast": fast, "slow": slow, "signal": signal},
    }


def _calculate_sma(close: pd.Series) -> dict:
    """
    Calculate SMA 20 and SMA 50 — Simple Moving Averages.

    WHAT IS SMA?
      A Simple Moving Average smooths out daily price noise by
      averaging the last N closing prices.
      SMA 20 = average of last 20 closing prices  (short-term trend)
      SMA 50 = average of last 50 closing prices  (medium-term trend)

    HOW TO READ IT:
      Price > SMA 20 → Short-term uptrend
      Price > SMA 50 → Medium-term uptrend
      SMA 20 > SMA 50 → "Golden Cross territory" — bullish overall trend
      SMA 20 < SMA 50 → "Death Cross territory"  — bearish overall trend

    WHY TWO SMAs?
      Comparing a fast (20) and slow (50) average tells us whether
      the short-term trend aligns with or fights the medium-term trend.
    """

    sma20_series = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    sma50_series = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

    current_price = safe_round(close.iloc[-1])
    sma20         = safe_round(sma20_series.iloc[-1])
    sma50         = safe_round(sma50_series.iloc[-1]) if len(close) >= 50 else None

    # Price vs SMA relationship
    price_vs_sma20 = None
    price_vs_sma50 = None
    pct_above_sma20 = None
    pct_above_sma50 = None

    if current_price and sma20:
        pct_above_sma20 = safe_round(((current_price - sma20) / sma20) * 100)
        price_vs_sma20  = "Above SMA20" if current_price > sma20 else "Below SMA20"

    if current_price and sma50:
        pct_above_sma50 = safe_round(((current_price - sma50) / sma50) * 100)
        price_vs_sma50  = "Above SMA50" if current_price > sma50 else "Below SMA50"

    # Golden Cross / Death Cross detection
    trend_structure = "Unknown"
    if sma20 and sma50:
        if sma20 > sma50:
            trend_structure = "Golden Cross Territory (SMA20 > SMA50 — Bullish)"
        else:
            trend_structure = "Death Cross Territory (SMA20 < SMA50 — Bearish)"

    # Build interpretation
    parts = []
    if price_vs_sma20:
        direction = "above" if current_price > sma20 else "below"
        parts.append(f"Price (₹{current_price}) is {direction} SMA20 (₹{sma20}) by {abs(pct_above_sma20)}%.")
    if price_vs_sma50:
        direction = "above" if current_price > sma50 else "below"
        parts.append(f"Price is {direction} SMA50 (₹{sma50}) by {abs(pct_above_sma50)}%.")
    if trend_structure != "Unknown":
        parts.append(trend_structure + ".")

    interpretation = " ".join(parts) if parts else "SMA data unavailable."

    return {
        "current_price"   : current_price,
        "sma20"           : sma20,
        "sma50"           : sma50,
        "price_vs_sma20"  : price_vs_sma20,
        "price_vs_sma50"  : price_vs_sma50,
        "pct_above_sma20" : pct_above_sma20,
        "pct_above_sma50" : pct_above_sma50,
        "trend_structure" : trend_structure,
        "interpretation"  : interpretation,
        "sma20_series"    : sma20_series,
        "sma50_series"    : sma50_series,
    }


# =============================================================
# SIGNAL GENERATION
# =============================================================

def _generate_signals(rsi_data: dict, macd_data: dict,
                      sma_data: dict, market_data: dict) -> tuple:
    """
    Convert indicator values into a list of plain-English signals.

    Returns:
        (bullish_signals, bearish_signals, signal_score)

    signal_score: each bullish signal = +1, each bearish signal = -1
    """
    bullish = []
    bearish = []
    score   = 0

    # --- RSI Signals ---
    rsi = rsi_data.get("value")
    if rsi is not None:
        if rsi < 30:
            bullish.append(f"RSI is {rsi} (Oversold < 30) — potential buying opportunity.")
            score += 1
        elif rsi > 70:
            bearish.append(f"RSI is {rsi} (Overbought > 70) — stock may be due for a pullback.")
            score -= 1
        elif rsi > 55:
            bullish.append(f"RSI is {rsi} (Moderately bullish zone 55–70).")
            score += 1
        elif rsi < 45:
            bearish.append(f"RSI is {rsi} (Moderately bearish zone 30–45).")
            score -= 1

    # --- MACD Signals ---
    crossover = macd_data.get("crossover", "")
    hist_val  = macd_data.get("histogram_latest")

    if "Bullish" in crossover:
        bullish.append(f"MACD: {crossover} — upward momentum detected.")
        score += 1
    elif "Bearish" in crossover:
        bearish.append(f"MACD: {crossover} — downward momentum detected.")
        score -= 1

    if hist_val is not None:
        if hist_val > 0:
            bullish.append(f"MACD Histogram is positive ({hist_val}) — bulls in control.")
            score += 1
        elif hist_val < 0:
            bearish.append(f"MACD Histogram is negative ({hist_val}) — bears in control.")
            score -= 1

    # --- SMA Signals ---
    price_vs_sma20 = sma_data.get("price_vs_sma20", "")
    price_vs_sma50 = sma_data.get("price_vs_sma50", "")
    trend          = sma_data.get("trend_structure", "")

    if "Above SMA20" in price_vs_sma20:
        pct = sma_data.get("pct_above_sma20", 0)
        bullish.append(f"Price is above SMA20 by {pct}% — short-term uptrend.")
        score += 1
    elif "Below SMA20" in price_vs_sma20:
        pct = abs(sma_data.get("pct_above_sma20", 0))
        bearish.append(f"Price is below SMA20 by {pct}% — short-term downtrend.")
        score -= 1

    if price_vs_sma50:
        if "Above SMA50" in price_vs_sma50:
            pct = sma_data.get("pct_above_sma50", 0)
            bullish.append(f"Price is above SMA50 by {pct}% — medium-term uptrend.")
            score += 1
        elif "Below SMA50" in price_vs_sma50:
            pct = abs(sma_data.get("pct_above_sma50", 0))
            bearish.append(f"Price is below SMA50 by {pct}% — medium-term downtrend.")
            score -= 1

    if "Golden Cross" in trend:
        bullish.append("SMA20 is above SMA50 (Golden Cross territory) — bullish long-term trend.")
        score += 1
    elif "Death Cross" in trend:
        bearish.append("SMA20 is below SMA50 (Death Cross territory) — bearish long-term trend.")
        score -= 1

    # --- Price momentum signal ---
    price_change = market_data.get("price_change", {})
    one_month_pct = price_change.get("one_month_pct")
    if one_month_pct is not None:
        if one_month_pct > 10:
            bullish.append(f"Strong 1-month price momentum: +{one_month_pct}%.")
            score += 1
        elif one_month_pct < -10:
            bearish.append(f"Weak 1-month price momentum: {one_month_pct}%.")
            score -= 1

    return bullish, bearish, score


# =============================================================
# SUMMARY BUILDER
# =============================================================

def _build_summary(symbol, rsi_data, macd_data, sma_data, overall_signal, score) -> str:
    """Build a concise plain-English summary for the report."""

    rsi   = rsi_data.get("value", "N/A")
    macd  = macd_data.get("crossover", "N/A")
    trend = sma_data.get("trend_structure", "N/A")

    return (
        f"Technical analysis of {symbol} shows an overall {overall_signal} signal "
        f"(score: {score:+d}). "
        f"RSI is {rsi} ({rsi_data.get('zone', 'N/A')} zone). "
        f"MACD indicates {macd}. "
        f"Moving average structure: {trend}. "
        f"This analysis is based on historical price data only and does not predict future prices."
    )


def _error_result(symbol: str, message: str) -> dict:
    """Standardised error response — same pattern as Market Data Agent."""
    print(f"[Technical Agent] ✗ ERROR for {symbol}: {message}")
    return {
        "status"         : "error",
        "symbol"         : symbol,
        "indicators"     : {},
        "signals"        : {"bullish": [], "bearish": []},
        "overall_signal" : "UNKNOWN",
        "signal_score"   : 0,
        "summary"        : message,
        "data_points"    : 0,
        "analysed_at"    : get_timestamp(),
        "error_message"  : message,
    }
