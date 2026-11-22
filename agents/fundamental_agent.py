"""
agents/fundamental_agent.py — Fundamental Analysis Agent (Agent C)
==================================================================
WHAT THIS AGENT DOES:
  - Analyses key fundamental financial metrics:
      1. Valuation: P/E (Trailing & Forward), P/B, Market Capitalisation
      2. Profitability: ROE (Return on Equity), EPS (Earnings Per Share)
      3. Financial Solvency: Debt-to-Equity Ratio, Book Value
      4. Growth: Revenue Growth %, Profit/Earnings Growth %
      5. Shareholder Returns: Dividend Yield %
  - Evaluates valuation health (Undervalued, Fairly Valued, Overvalued)
  - Assesses balance sheet financial strength (Strong, Moderate, High Leverage)
  - Identifies fundamental strengths and red flags
  - Computes a normalized fundamental score (-10 to +10)
  - Does NOT fabricate missing values — transparently tags unavailable items
"""

import math
from utils.helpers import safe_round, format_percentage, format_currency, get_timestamp


def run_fundamental_agent(market_data: dict) -> dict:
    """
    Main function for the Fundamental Analysis Agent.

    Parameters:
        market_data : Output dictionary from run_market_data_agent()

    Returns:
        Structured dictionary containing:
            - status            : "success" or "error"
            - symbol            : Stock symbol
            - metrics           : Clean key fundamental ratios
            - valuation_status  : "UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"
            - financial_health  : "STRONG", "MODERATE", "LEVERAGED"
            - fundamental_score : Score (-10 to +10)
            - strengths         : List of fundamental strengths
            - concerns          : List of fundamental risks/concerns
            - summary           : Professional analyst summary
            - analysed_at       : Timestamp
    """
    symbol = market_data.get("symbol", "UNKNOWN")
    print(f"\n[Fundamental Agent] Analysing financial fundamentals for: {symbol}")

    if market_data.get("status") == "error":
        return _error_result(symbol, "Cannot run fundamental analysis — market data unavailable.")

    mi = market_data.get("market_info", {})
    if not mi:
        return _error_result(symbol, "No fundamental metadata available in market data.")

    # Extract metrics safely
    pe = mi.get("pe_ratio")
    forward_pe = mi.get("forward_pe")
    pb = mi.get("pb_ratio")
    eps = mi.get("eps")
    roe = mi.get("return_on_equity")
    de = mi.get("debt_to_equity")
    mcap = mi.get("market_cap")
    mcap_fmt = mi.get("market_cap_fmt", "N/A")
    div_yield = mi.get("dividend_yield", 0.0)
    rev_growth = mi.get("revenue_growth")
    earn_growth = mi.get("earnings_growth")
    book_val = mi.get("book_value")
    sector = mi.get("sector", "General")

    strengths = []
    concerns = []
    score = 0

    # 1. Valuation Assessment (P/E & P/B)
    if pe is not None and pe > 0:
        if pe < 15:
            strengths.append(f"Low P/E Ratio ({pe:.1f}x) suggests attractive valuation relative to broader market.")
            score += 2
        elif pe < 28:
            strengths.append(f"Reasonable P/E Ratio ({pe:.1f}x) within standard Indian market multiples.")
            score += 1
        elif pe > 60:
            concerns.append(f"Elevated P/E Ratio ({pe:.1f}x) prices in aggressive growth expectations.")
            score -= 2
        elif pe > 40:
            concerns.append(f"High P/E Ratio ({pe:.1f}x) implies premium valuation multiples.")
            score -= 1
    else:
        concerns.append("P/E Ratio is negative or unavailable (company may have low/negative earnings).")

    if pb is not None and pb > 0:
        if pb < 2.0:
            strengths.append(f"Conservative Price-to-Book multiple ({pb:.2f}x).")
            score += 1
        elif pb > 10.0:
            concerns.append(f"High Price-to-Book multiple ({pb:.2f}x).")
            score -= 1

    # 2. Profitability (ROE & EPS)
    if roe is not None:
        if roe >= 20.0:
            strengths.append(f"Exceptional Return on Equity ({roe:.1f}%) demonstrates high capital efficiency.")
            score += 3
        elif roe >= 14.0:
            strengths.append(f"Healthy Return on Equity ({roe:.1f}%) above cost of capital.")
            score += 2
        elif roe < 8.0:
            concerns.append(f"Subdued Return on Equity ({roe:.1f}%) indicates modest capital productivity.")
            score -= 2
    else:
        concerns.append("Return on Equity data unavailable.")

    if eps is not None:
        if eps > 0:
            strengths.append(f"Positive Trailing EPS of ₹{eps:.2f}.")
            score += 1
        else:
            concerns.append(f"Negative Trailing EPS of ₹{eps:.2f}.")
            score -= 2

    # 3. Solvency & Financial Health (Debt-to-Equity)
    # Note: Financials / Banks naturally have different leverage metrics
    is_financial = any(term in sector.lower() for term in ["bank", "financial", "insurance", "nbfc"])
    if de is not None and not is_financial:
        # yfinance reports debtToEquity sometimes as ratio (e.g. 0.45) or percentage (e.g. 45.0)
        de_norm = de / 100.0 if de > 5.0 else de
        if de_norm < 0.3:
            strengths.append(f"Low Debt-to-Equity ratio ({de_norm:.2f}x) — pristine balance sheet.")
            score += 2
        elif de_norm <= 1.0:
            strengths.append(f"Manageable Debt-to-Equity ratio ({de_norm:.2f}x).")
            score += 1
        elif de_norm > 2.0:
            concerns.append(f"High Debt-to-Equity ratio ({de_norm:.2f}x) — elevated balance sheet leverage.")
            score -= 2
    elif is_financial:
        strengths.append(f"Financial institution — leverage evaluated through regulatory capital adequacy.")

    # 4. Growth Trajectory
    if earn_growth is not None:
        if earn_growth >= 15.0:
            strengths.append(f"Robust Profit/Earnings Growth (+{earn_growth:.1f}% YoY).")
            score += 2
        elif earn_growth < 0:
            concerns.append(f"Negative Earnings Growth ({earn_growth:.1f}% YoY).")
            score -= 2

    if rev_growth is not None:
        if rev_growth >= 12.0:
            strengths.append(f"Strong Revenue Growth (+{rev_growth:.1f}% YoY).")
            score += 1
        elif rev_growth < 0:
            concerns.append(f"Contracting Revenue Growth ({rev_growth:.1f}% YoY).")
            score -= 1

    # 5. Dividend Yield
    if div_yield and div_yield >= 2.0:
        strengths.append(f"Attractive Dividend Yield of {div_yield:.2f}%.")
        score += 1

    # Classifications
    if score >= 4:
        valuation_status = "UNDERVALUED / ATTRACTIVE"
    elif score <= -3:
        valuation_status = "OVERVALUED / EXPENSIVE"
    else:
        valuation_status = "FAIRLY VALUED"

    if len(concerns) <= 1 and score >= 3:
        financial_health = "STRONG"
    elif len(concerns) >= 3 or score <= -3:
        financial_health = "WEAK / LEVERAGED"
    else:
        financial_health = "MODERATE"

    summary = _build_summary(symbol, pe, pb, roe, de, mcap_fmt, valuation_status, financial_health, score)

    result = {
        "status": "success",
        "symbol": symbol,
        "company_name": mi.get("company_name", symbol),
        "sector": sector,
        "metrics": {
            "pe_ratio": pe,
            "forward_pe": forward_pe,
            "pb_ratio": pb,
            "eps": eps,
            "return_on_equity": roe,
            "debt_to_equity": de,
            "market_cap": mcap,
            "market_cap_fmt": mcap_fmt,
            "dividend_yield": div_yield,
            "revenue_growth": rev_growth,
            "earnings_growth": earn_growth,
            "book_value": book_val,
        },
        "valuation_status": valuation_status,
        "financial_health": financial_health,
        "fundamental_score": max(-10, min(10, score)),
        "strengths": strengths if strengths else ["No standout fundamental strengths identified."],
        "concerns": concerns if concerns else ["No major fundamental red flags identified."],
        "summary": summary,
        "analysed_at": get_timestamp(),
        "error_message": None,
    }

    print(f"[Fundamental Agent] ✓ Valuation: {valuation_status} | Health: {financial_health} | Score: {score:+d}/10")
    return result


def _build_summary(symbol, pe, pb, roe, de, mcap_fmt, val_status, health, score) -> str:
    pe_str = f"{pe:.1f}x" if pe else "N/A"
    pb_str = f"{pb:.1f}x" if pb else "N/A"
    roe_str = f"{roe:.1f}%" if roe else "N/A"

    return (
        f"Fundamental analysis for {symbol} indicates a {val_status} profile "
        f"with {health} financial health (fundamental score: {score:+d}/10). "
        f"The company commands a market capitalization of {mcap_fmt}, trading at a P/E multiple of {pe_str} "
        f"and Price-to-Book of {pb_str}. Capital efficiency is reflected by an ROE of {roe_str}. "
        f"All ratios are sourced from audited public filings via Yahoo Finance."
    )


def _error_result(symbol: str, message: str) -> dict:
    print(f"[Fundamental Agent] ✗ ERROR for {symbol}: {message}")
    return {
        "status": "error",
        "symbol": symbol,
        "company_name": symbol,
        "sector": "N/A",
        "metrics": {},
        "valuation_status": "UNAVAILABLE",
        "financial_health": "UNAVAILABLE",
        "fundamental_score": 0,
        "strengths": [],
        "concerns": [message],
        "summary": message,
        "analysed_at": get_timestamp(),
        "error_message": message,
    }
