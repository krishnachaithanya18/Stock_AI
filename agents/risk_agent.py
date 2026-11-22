"""
agents/risk_agent.py — Risk Analysis Agent (Agent E)
===================================================
WHAT THIS AGENT DOES:
  - Performs quantitative risk modeling using historical price data:
      1. Annualized Volatility % (standard deviation * sqrt(252))
      2. Maximum Historical Drawdown % (deepest peak-to-trough decline)
      3. Current Drawdown % from 52-week peak
      4. Beta relative to Indian benchmark (Nifty 50)
      5. Value-at-Risk (Parametric 1-Day 95% VaR)
  - Identifies multi-dimensional structural risk factors:
      - Technical breakdown risks
      - Fundamental & leverage risks
      - Sentiment & regulatory risks
  - Categorizes Investor Suitability:
      - Suitable for: Low-Risk, Medium-Risk, or High-Risk / Aggressive Investors
  - Assigns overall Risk Profile: "LOW RISK", "MODERATE RISK", "HIGH RISK"
"""

import math
import numpy as np
import pandas as pd
from utils.helpers import safe_round, format_percentage, get_timestamp


def run_risk_agent(market_data: dict, tech_data: dict, fund_data: dict, news_data: dict) -> dict:
    """
    Main function for the Risk Analysis Agent.

    Parameters:
        market_data : Output from Stock Data Agent
        tech_data   : Output from Technical Analysis Agent
        fund_data   : Output from Fundamental Analysis Agent
        news_data   : Output from Market Sentiment Agent

    Returns:
        Structured dictionary containing:
            - status                : "success" or "error"
            - symbol                : Stock symbol
            - volatility_annualized : Annualized Volatility %
            - max_drawdown_pct      : Maximum Drawdown %
            - current_drawdown_pct  : Current Drawdown from peak %
            - beta                  : Beta vs Nifty 50
            - var_95_1d             : 1-Day 95% Value at Risk (%)
            - risk_level            : "LOW RISK", "MODERATE RISK", "HIGH RISK"
            - investor_suitability  : Investor profile recommendation
            - identified_risks      : List of active risk factors with descriptions
            - summary               : Risk analyst summary
            - analysed_at           : Timestamp
    """
    symbol = market_data.get("symbol", "UNKNOWN")
    print(f"\n[Risk Agent] Evaluating risk profile and volatility for: {symbol}")

    if market_data.get("status") == "error":
        return _error_result(symbol, "Cannot perform risk analysis — market data unavailable.")

    hist = market_data.get("historical_df")
    if hist is None or hist.empty or len(hist) < 15:
        return _error_result(symbol, "Insufficient price history for statistical risk modeling.")

    close = hist["Close"]
    daily_returns = close.pct_change().dropna()

    # 1. Volatility Calculations
    daily_std = daily_returns.std()
    annualized_vol = safe_round(daily_std * math.sqrt(252) * 100, 2)

    # 2. Drawdown Analysis
    cumulative_max = close.cummax()
    drawdowns = (close - cumulative_max) / cumulative_max * 100
    max_drawdown = safe_round(drawdowns.min(), 2)  # negative number
    current_drawdown = safe_round(drawdowns.iloc[-1], 2)

    # 3. Parametric Value-at-Risk (95% confidence 1-day)
    # Z-score for 95% confidence is ~1.645
    mean_daily = daily_returns.mean()
    var_95_1d = safe_round((1.645 * daily_std - mean_daily) * 100, 2)

    # 4. Beta
    mi = market_data.get("market_info", {})
    beta = mi.get("beta")

    # 5. Identify Structural Risk Factors across all agents
    identified_risks = []

    # Volatility Risk
    if annualized_vol and annualized_vol > 35.0:
        identified_risks.append({
            "category": "High Volatility",
            "severity": "HIGH",
            "description": f"Annualized volatility is high at {annualized_vol:.1f}%, indicating significant price swings."
        })
    elif annualized_vol and annualized_vol > 25.0:
        identified_risks.append({
            "category": "Moderate Volatility",
            "severity": "MEDIUM",
            "description": f"Annualized volatility is moderate at {annualized_vol:.1f}%."
        })

    # Drawdown Risk
    if max_drawdown and abs(max_drawdown) > 30.0:
        identified_risks.append({
            "category": "Deep Drawdown History",
            "severity": "HIGH",
            "description": f"Historical maximum drawdown reached {abs(max_drawdown):.1f}%, indicating substantial capital risk during corrections."
        })

    # High Beta Risk
    if beta and beta > 1.35:
        identified_risks.append({
            "category": "High Market Sensitivity",
            "severity": "HIGH",
            "description": f"Beta of {beta:.2f} indicates the stock fluctuates significantly more than the Nifty 50."
        })

    # Technical Risks from Agent B
    rsi_val = tech_data.get("indicators", {}).get("rsi", {}).get("value")
    if rsi_val and rsi_val > 72:
        identified_risks.append({
            "category": "Overbought Condition",
            "severity": "MEDIUM",
            "description": f"RSI is {rsi_val:.1f} (>70), elevating the risk of near-term consolidation or mean reversion."
        })
    elif rsi_val and rsi_val < 30:
        identified_risks.append({
            "category": "Oversold Falling Knife",
            "severity": "MEDIUM",
            "description": f"RSI is {rsi_val:.1f} (<30), signaling heavy ongoing selling pressure."
        })

    pct_sma200 = tech_data.get("indicators", {}).get("sma", {}).get("pct_above_sma200")
    if pct_sma200 is not None and pct_sma200 < -8.0:
        identified_risks.append({
            "category": "Long-Term Downtrend",
            "severity": "HIGH",
            "description": f"Trading {abs(pct_sma200):.1f}% below the 200-day moving average indicates a secular bear trend."
        })

    # Fundamental Risks from Agent C
    fund_concerns = fund_data.get("concerns", [])
    for concern in fund_concerns:
        if "Debt-to-Equity" in concern or "Elevated" in concern or "Negative" in concern:
            identified_risks.append({
                "category": "Fundamental Risk",
                "severity": "MEDIUM",
                "description": concern
            })

    # Sentiment / Regulatory Risks from Agent D
    key_events = news_data.get("key_events", [])
    if "Regulatory/SEBI" in key_events:
        identified_risks.append({
            "category": "Regulatory Scrutiny",
            "severity": "HIGH",
            "description": "Regulatory/SEBI investigations or notices detected in recent news headlines."
        })
    if news_data.get("overall_sentiment") == "NEGATIVE":
        identified_risks.append({
            "category": "Negative News Sentiment",
            "severity": "MEDIUM",
            "description": "Dominant media sentiment is negative, which could exert short-term price drag."
        })

    # 6. Overall Risk Level & Investor Suitability
    high_severity_count = sum(1 for r in identified_risks if r["severity"] == "HIGH")
    total_risk_count = len(identified_risks)

    if high_severity_count >= 2 or total_risk_count >= 4 or (annualized_vol and annualized_vol > 38.0):
        risk_level = "HIGH RISK"
        investor_suitability = "Aggressive / High-Risk Investors Only (Growth / Speculative Allocation)"
    elif high_severity_count == 1 or total_risk_count >= 2 or (annualized_vol and annualized_vol > 22.0):
        risk_level = "MODERATE RISK"
        investor_suitability = "Moderate-Risk Investors (Balanced Portfolio Allocation)"
    else:
        risk_level = "LOW RISK"
        investor_suitability = "Conservative / Low-Risk Investors (Core Capital Preservation / Defensive Allocation)"

    summary = _build_summary(symbol, annualized_vol, max_drawdown, current_drawdown, beta, var_95_1d, risk_level, investor_suitability)

    result = {
        "status": "success",
        "symbol": symbol,
        "volatility_annualized": annualized_vol,
        "max_drawdown_pct": max_drawdown,
        "current_drawdown_pct": current_drawdown,
        "beta": beta,
        "var_95_1d": var_95_1d,
        "risk_level": risk_level,
        "investor_suitability": investor_suitability,
        "identified_risks": identified_risks,
        "risk_count": total_risk_count,
        "summary": summary,
        "analysed_at": get_timestamp(),
        "error_message": None,
    }

    print(f"[Risk Agent] ✓ Risk Profile: {risk_level} | Volatility: {annualized_vol}% | Suitability: {investor_suitability}")
    return result


def _build_summary(symbol, vol, mdd, cur_dd, beta, var_1d, level, suit) -> str:
    vol_str = f"{vol:.1f}%" if vol else "N/A"
    mdd_str = f"{mdd:.1f}%" if mdd else "N/A"
    var_str = f"{var_1d:.2f}%" if var_1d else "N/A"
    beta_str = f"{beta:.2f}" if beta else "N/A"

    return (
        f"Risk analysis for {symbol} classifies the asset as {level}. "
        f"Annualized price volatility stands at {vol_str} with a historical maximum drawdown of {mdd_str}. "
        f"Parametric 1-Day 95% Value-at-Risk (VaR) is estimated at {var_str}, and market beta is {beta_str}. "
        f"Investor Suitability: {suit}."
    )


def _error_result(symbol: str, message: str) -> dict:
    print(f"[Risk Agent] ✗ ERROR for {symbol}: {message}")
    return {
        "status": "error",
        "symbol": symbol,
        "volatility_annualized": None,
        "max_drawdown_pct": None,
        "current_drawdown_pct": None,
        "beta": None,
        "var_95_1d": None,
        "risk_level": "MODERATE RISK",
        "investor_suitability": "Requires further evaluation due to data unavailability",
        "identified_risks": [{"category": "Data Unavailable", "severity": "MEDIUM", "description": message}],
        "risk_count": 1,
        "summary": message,
        "analysed_at": get_timestamp(),
        "error_message": message,
    }
