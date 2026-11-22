"""
agents/technical_agent.py — Technical Analysis Agent (Agent B)
===============================================================
WHAT THIS AGENT DOES:
  - Calculates core technical indicators using the 'ta' library:
      1. RSI (14-period Relative Strength Index)
      2. MACD (12, 26, 9 with line, signal, histogram & crossover detection)
      3. Moving Averages: SMA 20, SMA 50, SMA 200 & Golden/Death Cross detection
      4. Bollinger Bands (20-period, 2-std dev: Upper, Middle, Lower, %B, Bandwidth)
      5. Support & Resistance Levels (Pivot Points S1, S2, R1, R2, recent swing lows/highs)
      6. Trend Identification (Overall Trend: Uptrend, Downtrend, Rangebound/Consolidation)
      7. Volume Analysis (Volume vs 20-day Volume MA, volume surge confirmation)
  - Identifies comprehensive bullish and bearish signals
  - Computes a normalized technical score (-10 to +10)
  - Returns clear interpretations and summary for the report and UI charts
"""

import pandas as pd
import numpy as np
import ta
from utils.helpers import safe_round, format_percentage, get_timestamp


def run_technical_agent(market_data: dict) -> dict:
    """
    Main function for the Technical Analysis Agent.

    Parameters:
        market_data : Result dictionary from run_market_data_agent()

    Returns:
        Structured dictionary containing:
            - status            : "success" or "error"
            - symbol            : Stock ticker
            - indicators        : RSI, MACD, SMA, Bollinger Bands, Support/Resistance, Trend, Volume
            - signals           : Bullish and Bearish signal lists
            - overall_signal    : "BULLISH", "BEARISH", or "NEUTRAL"
            - signal_score      : Numeric score (-10 to +10)
            - technical_rating  : "STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"
            - support_resistance: Key support and resistance price levels
            - summary           : Comprehensive technical summary text
            - analysed_at       : Timestamp
    """
    symbol = market_data.get("symbol", "UNKNOWN")
    print(f"\n[Technical Agent] Analysing technical indicators for: {symbol}")

    if market_data.get("status") == "error":
        return _error_result(symbol, "Cannot run technical analysis — market data fetch failed.")

    hist = market_data.get("historical_df")
    if hist is None or hist.empty:
        return _error_result(symbol, "Historical price DataFrame is missing or empty.")

    if len(hist) < 20:
        return _error_result(
            symbol,
            f"Only {len(hist)} trading days available. Need at least 20 for technical analysis."
        )

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"] if "Volume" in hist else pd.Series(0, index=close.index)

    # 1. RSI
    rsi_data = _calculate_rsi(close)

    # 2. MACD
    macd_data = _calculate_macd(close)

    # 3. Moving Averages (SMA 20, 50, 200)
    sma_data = _calculate_sma(close)

    # 4. Bollinger Bands
    bb_data = _calculate_bollinger_bands(close)

    # 5. Support & Resistance Levels
    sr_data = _calculate_support_resistance(hist)

    # 6. Volume Analysis
    vol_analysis = _calculate_volume_analysis(close, volume)

    # 7. Trend Identification
    trend_data = _calculate_trend(close, sma_data)

    # 8. Generate Signals & Compute Score
    bullish_signals, bearish_signals, score = _generate_signals(
        rsi_data, macd_data, sma_data, bb_data, sr_data, vol_analysis, trend_data, market_data
    )

    # Overall Signal Classification
    if score >= 4:
        overall_signal = "BULLISH"
        technical_rating = "STRONG BUY" if score >= 6 else "BUY"
    elif score <= -4:
        overall_signal = "BEARISH"
        technical_rating = "STRONG SELL" if score <= -6 else "SELL"
    else:
        overall_signal = "NEUTRAL"
        technical_rating = "HOLD"

    # Summary
    summary = _build_summary(symbol, rsi_data, macd_data, sma_data, bb_data, trend_data, overall_signal, score)

    result = {
        "status": "success",
        "symbol": symbol,
        "indicators": {
            "rsi": rsi_data,
            "macd": macd_data,
            "sma": sma_data,
            "bollinger": bb_data,
            "support_resistance": sr_data,
            "volume_analysis": vol_analysis,
            "trend": trend_data,
        },
        "signals": {
            "bullish": bullish_signals,
            "bearish": bearish_signals,
        },
        "overall_signal": overall_signal,
        "technical_rating": technical_rating,
        "signal_score": score,
        "support_resistance": sr_data,
        "summary": summary,
        "data_points": len(hist),
        "analysed_at": get_timestamp(),
        "error_message": None,
    }

    print(f"[Technical Agent] ✓ Signal: {overall_signal} (Score: {score:+d}) | RSI: {rsi_data['value']} | Trend: {trend_data['trend']}")
    print(f"[Technical Agent]   Bullish Signals: {len(bullish_signals)} | Bearish Signals: {len(bearish_signals)}")

    return result


# =======================================================================
# INDICATOR CALCULATIONS
# =======================================================================

def _calculate_rsi(close: pd.Series, period: int = 14) -> dict:
    rsi_indicator = ta.momentum.RSIIndicator(close=close, window=period)
    rsi_series = rsi_indicator.rsi()

    current_rsi = safe_round(rsi_series.iloc[-1])
    prev_rsi = safe_round(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else None

    if current_rsi is None:
        zone = "Unknown"
        interpretation = "RSI data unavailable."
    elif current_rsi >= 70:
        zone = "Overbought"
        interpretation = f"RSI is {current_rsi} (above 70). Stock is in overbought territory and may experience short-term consolidation or pullback."
    elif current_rsi <= 30:
        zone = "Oversold"
        interpretation = f"RSI is {current_rsi} (below 30). Stock is in oversold territory; technical mean-reversion bounce possible."
    elif current_rsi >= 55:
        zone = "Bullish Momentum"
        interpretation = f"RSI is {current_rsi} (55–70). Sustained bullish momentum without being overextended."
    elif current_rsi <= 45:
        zone = "Bearish Momentum"
        interpretation = f"RSI is {current_rsi} (30–45). Weak momentum with sellers exerting downward pressure."
    else:
        zone = "Neutral"
        interpretation = f"RSI is {current_rsi} (45–55). Balanced momentum with no decisive directional bias."

    return {
        "value": current_rsi,
        "previous_value": prev_rsi,
        "zone": zone,
        "period": period,
        "interpretation": interpretation,
        "series": rsi_series,
    }


def _calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    macd_obj = ta.trend.MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
    macd_line = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    histogram = macd_obj.macd_diff()

    macd_latest = safe_round(macd_line.iloc[-1], 4)
    signal_latest = safe_round(signal_line.iloc[-1], 4)
    hist_latest = safe_round(histogram.iloc[-1], 4)

    prev_macd = macd_line.iloc[-2] if len(macd_line) >= 2 else None
    prev_signal = signal_line.iloc[-2] if len(signal_line) >= 2 else None

    crossover = "None"
    if prev_macd is not None and prev_signal is not None and macd_latest is not None and signal_latest is not None:
        if prev_macd <= prev_signal and macd_latest > signal_latest:
            crossover = "Bullish Crossover"
        elif prev_macd >= prev_signal and macd_latest < signal_latest:
            crossover = "Bearish Crossover"
        elif macd_latest > signal_latest:
            crossover = "Bullish (MACD above Signal)"
        else:
            crossover = "Bearish (MACD below Signal)"

    if macd_latest is not None and signal_latest is not None:
        if macd_latest > signal_latest and hist_latest > 0:
            interpretation = f"MACD line ({macd_latest}) is above signal line ({signal_latest}) with positive histogram ({hist_latest}), indicating solid upward momentum."
        elif macd_latest < signal_latest and hist_latest < 0:
            interpretation = f"MACD line ({macd_latest}) is below signal line ({signal_latest}) with negative histogram ({hist_latest}), indicating downward momentum."
        else:
            interpretation = f"MACD ({macd_latest}) showing momentum transition relative to signal ({signal_latest})."
    else:
        interpretation = "MACD data unavailable."

    return {
        "macd_latest": macd_latest,
        "signal_latest": signal_latest,
        "histogram_latest": hist_latest,
        "crossover": crossover,
        "interpretation": interpretation,
        "macd_series": macd_line,
        "signal_series": signal_line,
        "histogram_series": histogram,
    }


def _calculate_sma(close: pd.Series) -> dict:
    sma20_series = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    sma50_series = ta.trend.SMAIndicator(close=close, window=50).sma_indicator() if len(close) >= 50 else pd.Series(index=close.index)
    sma200_series = ta.trend.SMAIndicator(close=close, window=200).sma_indicator() if len(close) >= 200 else pd.Series(index=close.index)

    current_price = safe_round(close.iloc[-1])
    sma20 = safe_round(sma20_series.iloc[-1]) if not sma20_series.empty else None
    sma50 = safe_round(sma50_series.iloc[-1]) if not sma50_series.empty and pd.notna(sma50_series.iloc[-1]) else None
    sma200 = safe_round(sma200_series.iloc[-1]) if not sma200_series.empty and pd.notna(sma200_series.iloc[-1]) else None

    pct_above_sma20 = safe_round(((current_price - sma20) / sma20) * 100) if (current_price and sma20) else None
    pct_above_sma50 = safe_round(((current_price - sma50) / sma50) * 100) if (current_price and sma50) else None
    pct_above_sma200 = safe_round(((current_price - sma200) / sma200) * 100) if (current_price and sma200) else None

    # Trend structure
    if sma50 and sma200:
        if sma50 > sma200:
            trend_structure = "Golden Cross Alignment (SMA50 > SMA200 — Bullish Macro Trend)"
        else:
            trend_structure = "Death Cross Alignment (SMA50 < SMA200 — Bearish Macro Trend)"
    elif sma20 and sma50:
        if sma20 > sma50:
            trend_structure = "Short-term Golden Cross (SMA20 > SMA50 — Bullish)"
        else:
            trend_structure = "Short-term Death Cross (SMA20 < SMA50 — Bearish)"
    else:
        trend_structure = "Consolidating Moving Averages"

    interpretation = (
        f"Price is {('above' if pct_above_sma20 and pct_above_sma20 >= 0 else 'below')} SMA20 "
        f"and {('above' if pct_above_sma50 and pct_above_sma50 >= 0 else 'below')} SMA50. "
        f"Moving average structure indicates: {trend_structure}."
    )

    return {
        "current_price": current_price,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "price_vs_sma20": f"{'Above' if pct_above_sma20 and pct_above_sma20 >= 0 else 'Below'} SMA20 ({pct_above_sma20}%)" if pct_above_sma20 is not None else "N/A",
        "price_vs_sma50": f"{'Above' if pct_above_sma50 and pct_above_sma50 >= 0 else 'Below'} SMA50 ({pct_above_sma50}%)" if pct_above_sma50 is not None else "N/A",
        "price_vs_sma200": f"{'Above' if pct_above_sma200 and pct_above_sma200 >= 0 else 'Below'} SMA200 ({pct_above_sma200}%)" if pct_above_sma200 is not None else "N/A",
        "pct_above_sma20": pct_above_sma20,
        "pct_above_sma50": pct_above_sma50,
        "pct_above_sma200": pct_above_sma200,
        "trend_structure": trend_structure,
        "interpretation": interpretation,
        "sma20_series": sma20_series,
        "sma50_series": sma50_series,
        "sma200_series": sma200_series,
    }


def _calculate_bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> dict:
    bb = ta.volatility.BollingerBands(close=close, window=period, window_dev=std)
    upper = safe_round(bb.bollinger_hband().iloc[-1])
    middle = safe_round(bb.bollinger_mavg().iloc[-1])
    lower = safe_round(bb.bollinger_lband().iloc[-1])
    pband = safe_round(bb.bollinger_pband().iloc[-1], 2)  # %B: 0 = at lower band, 1 = at upper band
    wband = safe_round(bb.bollinger_wband().iloc[-1], 2)  # Bandwidth %

    current_price = safe_round(close.iloc[-1])

    if pband is not None:
        if pband >= 1.0:
            condition = "Piercing Upper Band (Strong Bullish Momentum or Overextended)"
        elif pband >= 0.8:
            condition = "Near Upper Band (Bullish Zone)"
        elif pband <= 0.0:
            condition = "Piercing Lower Band (Oversold / Deep Value Zone)"
        elif pband <= 0.2:
            condition = "Near Lower Band (Support / Oversold Zone)"
        else:
            condition = "Mid-Band Normal Range"
    else:
        condition = "Normal"

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "percent_b": pband,
        "bandwidth": wband,
        "condition": condition,
        "upper_series": bb.bollinger_hband(),
        "middle_series": bb.bollinger_mavg(),
        "lower_series": bb.bollinger_lband(),
    }


def _calculate_support_resistance(hist: pd.DataFrame) -> dict:
    latest = hist.iloc[-1]
    high = latest["High"]
    low = latest["Low"]
    close = latest["Close"]

    # Classic Floor Trader Pivot Points
    pivot = safe_round((high + low + close) / 3.0)
    r1 = safe_round((2 * pivot) - low)
    s1 = safe_round((2 * pivot) - high)
    r2 = safe_round(pivot + (high - low))
    s2 = safe_round(pivot - (high - low))

    # 30-day swing levels
    swing_high = safe_round(hist["High"].tail(30).max())
    swing_low = safe_round(hist["Low"].tail(30).min())

    return {
        "pivot": pivot,
        "resistance_1": r1,
        "resistance_2": r2,
        "support_1": s1,
        "support_2": s2,
        "swing_high_30d": swing_high,
        "swing_low_30d": swing_low,
    }


def _calculate_volume_analysis(close: pd.Series, volume: pd.Series) -> dict:
    if volume.empty or volume.sum() == 0:
        return {"status": "insufficient_data", "signal": "Neutral"}

    vol_sma20 = volume.rolling(20).mean()
    latest_vol = volume.iloc[-1]
    avg_vol = vol_sma20.iloc[-1] if len(vol_sma20) >= 20 else volume.mean()

    ratio = safe_round(latest_vol / avg_vol, 2) if avg_vol and avg_vol > 0 else 1.0
    price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0

    if ratio >= 1.5 and price_change > 0:
        signal = "High Volume Bullish Surge (Institutional Buying Potential)"
    elif ratio >= 1.5 and price_change < 0:
        signal = "High Volume Selling Pressure (Distribution)"
    elif ratio < 0.7:
        signal = "Low Volume Consolidation"
    else:
        signal = "Average Volume Activity"

    return {
        "latest_volume": int(latest_vol),
        "avg_volume_20d": int(avg_vol) if avg_vol else 0,
        "volume_surge_ratio": ratio,
        "signal": signal,
    }


def _calculate_trend(close: pd.Series, sma_data: dict) -> dict:
    current = close.iloc[-1]
    sma20 = sma_data.get("sma20")
    sma50 = sma_data.get("sma50")
    sma200 = sma_data.get("sma200")

    bull_count = 0
    bear_count = 0

    if sma20 and current > sma20:
        bull_count += 1
    elif sma20 and current < sma20:
        bear_count += 1

    if sma50 and current > sma50:
        bull_count += 1
    elif sma50 and current < sma50:
        bear_count += 1

    if sma200 and current > sma200:
        bull_count += 2
    elif sma200 and current < sma200:
        bear_count += 2

    if bull_count >= 3:
        trend = "Strong Uptrend"
    elif bull_count > bear_count:
        trend = "Mild Uptrend"
    elif bear_count >= 3:
        trend = "Strong Downtrend"
    elif bear_count > bull_count:
        trend = "Mild Downtrend"
    else:
        trend = "Sideways / Rangebound"

    return {
        "trend": trend,
        "bullish_moving_averages": bull_count,
        "bearish_moving_averages": bear_count,
    }


# =======================================================================
# SIGNAL GENERATION & SCORING
# =======================================================================

def _generate_signals(rsi_data, macd_data, sma_data, bb_data, sr_data, vol_data, trend_data, market_data) -> tuple:
    bullish = []
    bearish = []
    score = 0

    # RSI
    rsi = rsi_data.get("value")
    if rsi is not None:
        if rsi < 30:
            bullish.append(f"RSI is oversold at {rsi:.1f} (<30) — mean-reversion buying opportunity.")
            score += 2
        elif rsi > 70:
            bearish.append(f"RSI is overbought at {rsi:.1f} (>70) — vulnerable to short-term profit booking.")
            score -= 2
        elif 52 <= rsi <= 68:
            bullish.append(f"RSI is in healthy bullish momentum zone ({rsi:.1f}).")
            score += 1
        elif 32 <= rsi <= 48:
            bearish.append(f"RSI is in bearish momentum territory ({rsi:.1f}).")
            score -= 1

    # MACD
    crossover = macd_data.get("crossover", "")
    hist_val = macd_data.get("histogram_latest")
    if "Bullish Crossover" in crossover:
        bullish.append(f"MACD generated a fresh Bullish Crossover above signal line.")
        score += 2
    elif "Bearish Crossover" in crossover:
        bearish.append(f"MACD generated a Bearish Crossover below signal line.")
        score -= 2
    elif "Bullish" in crossover:
        bullish.append(f"MACD is above signal line (positive momentum).")
        score += 1
    elif "Bearish" in crossover:
        bearish.append(f"MACD is below signal line (negative momentum).")
        score -= 1

    if hist_val is not None and hist_val > 0:
        score += 1
    elif hist_val is not None and hist_val < 0:
        score -= 1

    # SMA Structure
    price_sma20 = sma_data.get("price_vs_sma20", "")
    price_sma50 = sma_data.get("price_vs_sma50", "")
    price_sma200 = sma_data.get("price_vs_sma200", "")
    trend_struct = sma_data.get("trend_structure", "")

    if "Above SMA20" in price_sma20:
        bullish.append(f"Price is trading above 20-day SMA ({sma_data.get('pct_above_sma20')}%) — short-term strength.")
        score += 1
    elif "Below SMA20" in price_sma20:
        bearish.append(f"Price is trading below 20-day SMA ({sma_data.get('pct_above_sma20')}%) — short-term weakness.")
        score -= 1

    if "Above SMA50" in price_sma50:
        bullish.append(f"Price is above 50-day SMA ({sma_data.get('pct_above_sma50')}%) — intermediate uptrend.")
        score += 1
    elif "Below SMA50" in price_sma50:
        bearish.append(f"Price is below 50-day SMA ({sma_data.get('pct_above_sma50')}%) — intermediate downtrend.")
        score -= 1

    if "Above SMA200" in price_sma200:
        bullish.append(f"Price is trading above 200-day SMA — long-term secular bull trend.")
        score += 2
    elif "Below SMA200" in price_sma200:
        bearish.append(f"Price is trading below 200-day SMA — long-term secular bear market.")
        score -= 2

    if "Golden Cross" in trend_struct:
        bullish.append(f"Moving Average structure indicates a Golden Cross.")
        score += 2
    elif "Death Cross" in trend_struct:
        bearish.append(f"Moving Average structure indicates a Death Cross.")
        score -= 2

    # Bollinger Bands
    pband = bb_data.get("percent_b")
    if pband is not None:
        if pband <= 0.1:
            bullish.append(f"Price testing lower Bollinger Band (%B: {pband}) — key technical support zone.")
            score += 1
        elif pband >= 0.95:
            bearish.append(f"Price testing upper Bollinger Band (%B: {pband}) — resistance band reached.")
            score -= 1

    # Volume Confirmation
    vol_sig = vol_data.get("signal", "")
    if "Bullish Surge" in vol_sig:
        bullish.append(f"Volume surge confirms bullish buying accumulation.")
        score += 1
    elif "Selling Pressure" in vol_sig:
        bearish.append(f"Elevated volume on downward moves indicates distribution.")
        score -= 1

    # Clamp score to [-10, 10]
    final_score = max(-10, min(10, score))

    return bullish, bearish, final_score


def _build_summary(symbol, rsi_data, macd_data, sma_data, bb_data, trend_data, overall_signal, score) -> str:
    rsi_val = rsi_data.get("value", "N/A")
    macd_co = macd_data.get("crossover", "N/A")
    trend = trend_data.get("trend", "N/A")
    sma200_info = sma_data.get("price_vs_sma200", "N/A")

    return (
        f"Technical evaluation for {symbol} indicates an overall {overall_signal} signal "
        f"(technical score: {score:+d}/10). Trend regime is classified as {trend}. "
        f"RSI is at {rsi_val} ({rsi_data.get('zone', 'N/A')}), MACD is {macd_co}, "
        f"and position vs 200-day SMA is {sma200_info}. "
        f"Bollinger Band condition: {bb_data.get('condition', 'Normal')}."
    )


def _error_result(symbol: str, message: str) -> dict:
    print(f"[Technical Agent] ✗ ERROR for {symbol}: {message}")
    return {
        "status": "error",
        "symbol": symbol,
        "indicators": {},
        "signals": {"bullish": [], "bearish": []},
        "overall_signal": "UNKNOWN",
        "technical_rating": "N/A",
        "signal_score": 0,
        "support_resistance": {},
        "summary": message,
        "data_points": 0,
        "analysed_at": get_timestamp(),
        "error_message": message,
    }
