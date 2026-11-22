"""
agents/decision_agent.py — Final Decision & Synthesis Agent (Agent F)
======================================================================
WHAT THIS AGENT DOES:
  - Master decision engine of the Indian Stock Market Agentic AI platform.
  - Ingests structured findings from ALL 5 upstream agents:
      1. Stock Data Agent: Quotes, 52W range, OHLCV history, volume
      2. Technical Analysis Agent: RSI, MACD, SMAs, Bollinger Bands, Support/Resistance
      3. Fundamental Analysis Agent: P/E, P/B, ROE, EPS, Debt/Equity, Growth
      4. Market Sentiment Agent: News headlines, sentiment score, event catalysts
      5. Risk Analysis Agent: Volatility, Max Drawdown, Beta, VaR, Risk profile
  - Synthesizes findings into the ultimate investment decision:
      * Investment Verdict: STRONG BUY | BUY | HOLD | AVOID
      * Statistical Estimated Probability of Positive Return (%)
      * Expected Return Range (%) & Potential Downside Range (%)
      * Recommended Holding Horizon (Short-term 1–4W, Medium-term 1–3M, Long-term 6–12M)
      * Suggested Entry Price, Target Price, Stop-Loss Level & Risk-Reward Ratio
      * Transparent Supporting Pillars & Recommendation Invalidation Factors
      * Complete Consolidated AI Research Report (via Gemini API with template fallback)
"""

import math
import numpy as np
import pandas as pd
from utils.helpers import safe_round, format_percentage, format_currency, get_timestamp
from utils.gemini_client import call_gemini


def run_decision_agent(market_data: dict, tech_data: dict, fund_data: dict,
                       news_data: dict, risk_data: dict) -> dict:
    """
    Main function for the Final Decision Agent.
    """
    symbol = market_data.get("symbol", "UNKNOWN")
    company = market_data.get("company_name", symbol)
    current_price = market_data.get("current_price") or 0.0

    print(f"\n[Final Decision Agent] Generating investment verdict for: {company} ({symbol})")

    # 1. Multi-Factor Composite Scoring (-100 to +100)
    composite_score, factor_scores = _compute_composite_score(tech_data, fund_data, news_data, risk_data)

    # 2. Determine Investment Verdict
    verdict, verdict_color, verdict_desc = _determine_verdict(composite_score, tech_data, fund_data, risk_data)

    # 3. Determine Recommended Holding Period
    holding_period, holding_period_desc = _determine_holding_period(composite_score, tech_data, fund_data)

    # 4. Statistically Grounded Probability of Positive Return
    prob_positive, prob_methodology = _calculate_return_probability(market_data, tech_data, fund_data, composite_score, holding_period)

    # 5. Price Levels: Entry, Target, Stop-Loss, and Risk-to-Reward
    trade_setup = _calculate_trade_setup(current_price, tech_data, risk_data, verdict, holding_period)

    # 6. Model Confidence Score (0–100%)
    model_confidence, confidence_rating = _calculate_model_confidence(market_data, tech_data, fund_data, news_data, risk_data)

    # 7. Bullish Thesis & Invalidation Factors
    bullish_reasons = _extract_bullish_reasons(tech_data, fund_data, news_data, risk_data)
    invalidation_factors = _extract_invalidation_factors(tech_data, fund_data, news_data, risk_data, trade_setup)

    # 8. Consolidated Research Report Generation
    report_bundle = _generate_consolidated_report(
        market_data, tech_data, fund_data, news_data, risk_data,
        verdict, prob_positive, trade_setup, holding_period, model_confidence,
        bullish_reasons, invalidation_factors
    )

    result = {
        "status": "success",
        "symbol": symbol,
        "company_name": company,
        "current_price": current_price,
        "current_price_fmt": format_currency(current_price),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_desc": verdict_desc,
        "composite_score": composite_score,
        "factor_scores": factor_scores,
        "probability_positive_return": prob_positive,
        "probability_methodology": prob_methodology,
        "holding_period": holding_period,
        "holding_period_desc": holding_period_desc,
        "trade_setup": trade_setup,
        "model_confidence": model_confidence,
        "confidence_rating": confidence_rating,
        "bullish_reasons": bullish_reasons,
        "invalidation_factors": invalidation_factors,
        "report": report_bundle,
        "analysed_at": get_timestamp(),
        "disclaimer": (
            "STATISTICAL DISCLAIMER: Probability estimates and price targets are derived from "
            "historical distributions, quantitative indicators and mathematical risk models. "
            "They do NOT guarantee future performance. Past market behavior does not ensure "
            "future returns. For educational and academic research purposes only. "
            "Consult a SEBI-registered financial advisor before executing trades."
        )
    }

    print(f"[Final Decision Agent] ✓ VERDICT: {verdict} | P(Profit): {prob_positive}% | Horizon: {holding_period}")
    return result


# =======================================================================
# QUANTITATIVE SCORING & VERDICT DETERMINATION
# =======================================================================

def _compute_composite_score(tech_data: dict, fund_data: dict, news_data: dict, risk_data: dict) -> tuple:
    """
    Combines Technical (35%), Fundamental (35%), Sentiment (15%), and Risk (-15%)
    into a unified composite score ranging from -100 to +100.
    """
    t_raw = tech_data.get("signal_score", 0)
    t_score = safe_round((t_raw / 10.0) * 35.0, 1)

    f_raw = fund_data.get("fundamental_score", 0)
    f_score = safe_round((f_raw / 10.0) * 35.0, 1)

    s_raw = news_data.get("sentiment_score", 0)
    s_score = safe_round((s_raw / 10.0) * 15.0, 1)

    risk_level = risk_data.get("risk_level", "MODERATE RISK")
    if risk_level == "HIGH RISK":
        r_score = -15.0
    elif risk_level == "MODERATE RISK":
        r_score = -5.0
    else:
        r_score = 0.0

    total = safe_round(t_score + f_score + s_score + r_score, 1)
    total_clamped = max(-100.0, min(100.0, total))

    factor_breakdown = {
        "technical_component": t_score,
        "fundamental_component": f_score,
        "sentiment_component": s_score,
        "risk_penalty": r_score,
        "total_score": total_clamped,
    }

    return total_clamped, factor_breakdown


def _determine_verdict(composite_score: float, tech_data: dict, fund_data: dict, risk_data: dict) -> tuple:
    t_rating = tech_data.get("overall_signal", "NEUTRAL")

    if composite_score >= 35 and "BEARISH" not in t_rating:
        verdict = "STRONG BUY"
        color = "#10B981"  # Emerald green
        desc = "High convergence of bullish technical momentum and robust fundamental health."
    elif composite_score >= 12:
        verdict = "BUY"
        color = "#34D399"  # Mint green
        desc = "Favorable risk-reward profile with positive technical or fundamental catalysts."
    elif composite_score >= -12:
        verdict = "HOLD"
        color = "#FBBF24"  # Amber
        desc = "Balanced risk-reward profile or mixed cross-agent signals. Maintain existing positions."
    else:
        verdict = "AVOID"
        color = "#EF4444"  # Crimson red
        desc = "Unfavorable technical trend, stretched valuation, or high downside risk."

    return verdict, color, desc


def _determine_holding_period(composite_score: float, tech_data: dict, fund_data: dict) -> tuple:
    t_score = tech_data.get("signal_score", 0)
    f_score = fund_data.get("fundamental_score", 0)

    if f_score >= 5 and composite_score >= 15:
        return "Long term: 6–12 months", "High fundamental quality and valuation support multi-quarter capital compounding."
    elif t_score >= 3 and f_score >= 1:
        return "Medium term: 1–3 months", "Intermediate technical trend and moving average alignment offer multi-week swing potential."
    elif t_score >= 3:
        return "Short term: 1–4 weeks", "Tactical momentum setup driven by short-term RSI/MACD breakouts."
    else:
        return "Medium term: 1–3 months", "Standard observation and position management horizon."


# =======================================================================
# STATISTICAL PROBABILITY & TRADE SETUP CALCULATIONS
# =======================================================================

def _calculate_return_probability(market_data: dict, tech_data: dict, fund_data: dict,
                                  composite_score: float, horizon: str) -> tuple:
    """
    Calculates empirical probability of positive return using transparent quantitative rules:
    1. Historical rolling returns win-rate over horizon window
    2. Regime adjustment from composite score and RSI position
    """
    hist = market_data.get("historical_df")
    if hist is None or len(hist) < 25:
        return 52.0, "Baseline neutral estimate (insufficient historical records for rolling backtest)."

    close = hist["Close"]

    # Window based on horizon
    if "Short term" in horizon:
        window = min(15, len(close) // 4)
    elif "Long term" in horizon:
        window = min(45, len(close) // 2)
    else:
        window = min(25, len(close) // 3)

    window = max(5, window)

    # Historical backward rolling returns over `window` trading days
    rolling_returns = (close - close.shift(window)) / close.shift(window)
    valid_rolls = rolling_returns.dropna()

    if len(valid_rolls) >= 15:
        base_win_rate = (valid_rolls > 0).mean() * 100.0
    else:
        base_win_rate = 52.0

    # Calibration modifier from composite score (-100 to +100) -> adds/subtracts up to 20%
    modifier = (composite_score / 100.0) * 20.0

    # RSI regime bonus / penalty
    rsi = tech_data.get("indicators", {}).get("rsi", {}).get("value")
    if rsi:
        if 48 <= rsi <= 64:
            modifier += 3.0
        elif rsi < 30:
            modifier += 4.0  # Oversold mean reversion
        elif rsi > 75:
            modifier -= 4.0  # Overbought exhaustion

    final_prob = safe_round(base_win_rate + modifier, 1)
    final_prob_clamped = max(25.0, min(82.0, final_prob))

    methodology = (
        f"Empirical historical rolling win-rate ({base_win_rate:.1f}% positive outcomes over {window}-trading-day holding windows, "
        f"sample size N={len(valid_rolls)} periods) calibrated by multi-factor score ({composite_score:+0.1f}) and RSI momentum regime."
    )

    return final_prob_clamped, methodology


def _calculate_trade_setup(current_price: float, tech_data: dict, risk_data: dict,
                           verdict: str, horizon: str) -> dict:
    """
    Computes institutional entry range, target price, stop loss, and risk-to-reward ratio.
    """
    if not current_price or current_price <= 0:
        return {
            "entry_range": "N/A", "target_price": "N/A", "stop_loss": "N/A",
            "expected_return_pct": "N/A", "potential_downside_pct": "N/A", "risk_reward_ratio": "N/A"
        }

    sr = tech_data.get("support_resistance", {})
    r1 = sr.get("resistance_1") or (current_price * 1.05)
    r2 = sr.get("resistance_2") or (current_price * 1.10)
    s1 = sr.get("support_1") or (current_price * 0.96)
    s2 = sr.get("support_2") or (current_price * 0.92)

    vol = risk_data.get("volatility_annualized") or 22.0

    # Institutional target & stop-loss percentage models
    if "Short term" in horizon:
        target_pct = min(10.0, max(4.0, (vol * 0.25)))
        downside_pct = min(5.0, max(2.5, (vol * 0.14)))
        target_price = safe_round(max(r1, current_price * (1 + target_pct / 100.0)))
        stop_loss = safe_round(min(s1, current_price * (1 - downside_pct / 100.0)))
    elif "Long term" in horizon:
        target_pct = min(28.0, max(14.0, (vol * 0.65)))
        downside_pct = min(12.0, max(6.0, (vol * 0.28)))
        target_price = safe_round(max(r2, current_price * (1 + target_pct / 100.0)))
        stop_loss = safe_round(min(s2, current_price * (1 - downside_pct / 100.0)))
    else:  # Medium term
        target_pct = min(16.0, max(8.0, (vol * 0.42)))
        downside_pct = min(7.5, max(4.0, (vol * 0.20)))
        target_price = safe_round(max(r1 * 1.02, current_price * (1 + target_pct / 100.0)))
        stop_loss = safe_round(min(s1, current_price * (1 - downside_pct / 100.0)))

    entry_low = safe_round(current_price * 0.985)
    entry_high = safe_round(current_price * 1.008)

    upside_pct = safe_round(((target_price - current_price) / current_price) * 100, 1)
    downside_pct = safe_round(((current_price - stop_loss) / current_price) * 100, 1)

    if downside_pct and downside_pct > 0:
        rr_ratio = safe_round(upside_pct / downside_pct, 1)
        rr_fmt = f"1 : {rr_ratio}"
    else:
        rr_fmt = "1 : 2.0"

    return {
        "entry_range": f"₹{entry_low:,.2f} – ₹{entry_high:,.2f}",
        "entry_low": entry_low,
        "entry_high": entry_high,
        "target_price": target_price,
        "target_price_fmt": format_currency(target_price),
        "stop_loss": stop_loss,
        "stop_loss_fmt": format_currency(stop_loss),
        "expected_return_pct": upside_pct,
        "expected_return_fmt": f"+{upside_pct:.1f}%",
        "potential_downside_pct": downside_pct,
        "potential_downside_fmt": f"-{downside_pct:.1f}%",
        "risk_reward_ratio": rr_fmt,
    }


def _calculate_model_confidence(market_data: dict, tech_data: dict, fund_data: dict,
                                news_data: dict, risk_data: dict) -> tuple:
    confidence = 72.0

    if len(market_data.get("historical_df", [])) >= 60:
        confidence += 8.0
    if fund_data.get("status") == "success" and fund_data.get("metrics", {}).get("pe_ratio"):
        confidence += 6.0
    if news_data.get("article_count", 0) >= 5:
        confidence += 6.0
    elif news_data.get("overall_sentiment") == "UNAVAILABLE":
        confidence -= 10.0

    t_sig = tech_data.get("overall_signal", "NEUTRAL")
    f_val = fund_data.get("valuation_status", "")
    n_sent = news_data.get("overall_sentiment", "")

    if t_sig == "BULLISH" and "UNDERVALUED" in f_val:
        confidence += 5.0
    elif t_sig == "BULLISH" and n_sent == "NEGATIVE":
        confidence -= 6.0

    final_conf = max(40.0, min(95.0, confidence))

    if final_conf >= 80:
        rating = "HIGH CONFIDENCE"
    elif final_conf >= 60:
        rating = "MODERATE CONFIDENCE"
    else:
        rating = "LOW CONFIDENCE"

    return safe_round(final_conf, 1), rating


def _extract_bullish_reasons(tech_data, fund_data, news_data, risk_data) -> list:
    reasons = []
    for s in tech_data.get("signals", {}).get("bullish", [])[:3]:
        reasons.append(f"[Technical] {s}")
    for s in fund_data.get("strengths", [])[:3]:
        reasons.append(f"[Fundamental] {s}")
    if news_data.get("positive_factors"):
        reasons.append(f"[Catalyst] {news_data['positive_factors'][0]}")
    if risk_data.get("risk_level") == "LOW RISK":
        reasons.append("[Risk] Low historical volatility and disciplined maximum drawdown profile.")

    return reasons if reasons else ["Neutral market setup with stable underlying metrics."]


def _extract_invalidation_factors(tech_data, fund_data, news_data, risk_data, setup) -> list:
    factors = []
    if setup.get("stop_loss_fmt"):
        factors.append(f"A confirmed daily close below key stop-loss of {setup['stop_loss_fmt']} invalidates the thesis.")
    for b in tech_data.get("signals", {}).get("bearish", [])[:2]:
        factors.append(f"[Technical Headwind] {b}")
    for c in fund_data.get("concerns", [])[:2]:
        factors.append(f"[Fundamental Concern] {c}")
    for r in risk_data.get("identified_risks", [])[:2]:
        factors.append(f"[Risk Factor] {r['description']}")

    return factors if factors else ["Broader equity market correction or sector-wide rotation."]


def _generate_consolidated_report(market_data, tech_data, fund_data, news_data, risk_data,
                                  verdict, prob, setup, horizon, conf, bullish_reasons, invalidation_factors) -> dict:
    company = market_data.get("company_name", "Unknown")
    symbol = market_data.get("symbol", "")
    price_fmt = market_data.get("current_price_fmt", "N/A")

    gemini_prompt = f"""You are a senior Indian stock market research analyst.
Write a comprehensive, professional investment research report for {company} ({symbol}) using ONLY the provided verified data:

EXCHANGE DATA:
- Current Price: {price_fmt}
- 1-Day Change: {market_data.get('price_change', {}).get('one_day_pct_fmt', 'N/A')}
- Market Cap: {market_data.get('market_info', {}).get('market_cap_fmt', 'N/A')}
- 52-Week Range: ₹{market_data.get('market_info', {}).get('fifty_two_week_low', 'N/A')} – ₹{market_data.get('market_info', {}).get('fifty_two_week_high', 'N/A')}

DECISION & RECOMMENDATION:
- Verdict: {verdict}
- Model Confidence: {conf}%
- Estimated Probability of Positive Return: {prob}%
- Horizon: {horizon}
- Entry Range: {setup.get('entry_range')} | Target: {setup.get('target_price_fmt')} | Stop-Loss: {setup.get('stop_loss_fmt')}

TECHNICAL FINDINGS:
{tech_data.get('summary')}

FUNDAMENTAL FINDINGS:
{fund_data.get('summary')}

SENTIMENT & NEWS:
{news_data.get('summary')}

RISK PROFILE:
{risk_data.get('summary')}

BULLISH PILLARS:
{chr(10).join(f"- {r}" for r in bullish_reasons)}

INVALIDATION FACTORS:
{chr(10).join(f"- {f}" for f in invalidation_factors)}

Generate a structured report with:
1. EXECUTIVE VERDICT & RECOMMENDATION
2. TECHNICAL & MOMENTUM APPRAISAL
3. FUNDAMENTAL HEALTH & VALUATION CHECK
4. MARKET SENTIMENT & NEWS CATALYSTS
5. RISK-REWARD & SCENARIO BLUEPRINT
6. MANDATORY DISCLAIMER
Do NOT invent numbers. Maintain academic & institutional tone."""

    gemini_resp = call_gemini(gemini_prompt, temperature=0.2, max_tokens=2200)

    if gemini_resp.get("status") == "success" and gemini_resp.get("text"):
        report_text = gemini_resp["text"]
        report_type = "gemini"
    else:
        report_text = _build_template_report(
            market_data, tech_data, fund_data, news_data, risk_data,
            verdict, prob, setup, horizon, conf, bullish_reasons, invalidation_factors
        )
        report_type = "template"

    return {
        "report_text": report_text,
        "report_type": report_type,
        "generated_at": get_timestamp(),
    }


def _build_template_report(market_data, tech_data, fund_data, news_data, risk_data,
                           verdict, prob, setup, horizon, conf, bullish_reasons, invalidation_factors) -> str:
    company = market_data.get("company_name", "Unknown")
    symbol = market_data.get("symbol", "")
    price_fmt = market_data.get("current_price_fmt", "N/A")
    mi = market_data.get("market_info", {})
    pc = market_data.get("price_change", {})

    b_items = "\n".join(f"  • {r}" for r in bullish_reasons)
    i_items = "\n".join(f"  • {f}" for f in invalidation_factors)

    return f"""================================================================================
INSTITUTIONAL STOCK RESEARCH REPORT: {company} ({symbol})
Generated: {get_timestamp()} | Architecture: LangGraph Multi-Agent AI
================================================================================

1. EXECUTIVE INVESTMENT VERDICT: {verdict}
--------------------------------------------------------------------------------
• Investment Recommendation : {verdict}
• Quantitative Confidence   : {conf}% ({conf} score across 5 specialized agents)
• P(Positive Return)        : {prob}% (Empirical rolling probability over horizon)
• Recommended Horizon       : {horizon}
• Suggested Entry Range     : {setup.get('entry_range', 'N/A')}
• Target Price (Upside)     : {setup.get('target_price_fmt', 'N/A')} ({setup.get('expected_return_fmt', 'N/A')})
• Stop-Loss (Downside)      : {setup.get('stop_loss_fmt', 'N/A')} ({setup.get('potential_downside_fmt', 'N/A')})
• Risk-to-Reward Ratio      : {setup.get('risk_reward_ratio', 'N/A')}

2. COMPANY & MARKET OVERVIEW
--------------------------------------------------------------------------------
• Current Market Price      : {price_fmt}
• Daily / Weekly Change     : 1D {pc.get('one_day_pct_fmt', 'N/A')} | 1W {pc.get('one_week_pct_fmt', 'N/A')} | 1M {pc.get('one_month_pct_fmt', 'N/A')}
• Market Capitalization     : {mi.get('market_cap_fmt', 'N/A')}
• 52-Week Range             : ₹{mi.get('fifty_two_week_low', 'N/A')} – ₹{mi.get('fifty_two_week_high', 'N/A')}
• Sector / Industry         : {mi.get('sector', 'N/A')} | {mi.get('industry', 'N/A')}

3. TECHNICAL ANALYSIS SUMMARY
--------------------------------------------------------------------------------
{tech_data.get('summary', 'N/A')}
• Technical Score: {tech_data.get('signal_score', 0):+d}/10 | Signal: {tech_data.get('overall_signal', 'N/A')}
• Key Levels: Pivot ₹{tech_data.get('support_resistance', {}).get('pivot', 'N/A')} | R1 ₹{tech_data.get('support_resistance', {}).get('resistance_1', 'N/A')} | S1 ₹{tech_data.get('support_resistance', {}).get('support_1', 'N/A')}

4. FUNDAMENTAL ANALYSIS SUMMARY
--------------------------------------------------------------------------------
{fund_data.get('summary', 'N/A')}
• Valuation Status: {fund_data.get('valuation_status', 'N/A')} | Health: {fund_data.get('financial_health', 'N/A')}
• Ratios: P/E {mi.get('pe_ratio', 'N/A')}x | P/B {mi.get('pb_ratio', 'N/A')}x | ROE {mi.get('return_on_equity', 'N/A')}% | EPS ₹{mi.get('eps', 'N/A')}

5. MARKET SENTIMENT & NEWS DIGEST
--------------------------------------------------------------------------------
{news_data.get('summary', 'N/A')}

6. RISK ASSESSMENT & INVESTOR SUITABILITY
--------------------------------------------------------------------------------
{risk_data.get('summary', 'N/A')}
• Risk Category: {risk_data.get('risk_level', 'N/A')}
• Target Investor: {risk_data.get('investor_suitability', 'N/A')}

7. KEY BULLISH PILLARS
--------------------------------------------------------------------------------
{b_items}

8. RECOMMENDATION INVALIDATION FACTORS
--------------------------------------------------------------------------------
{i_items}

================================================================================
STATISTICAL & REGULATORY DISCLAIMER:
This research assessment is generated autonomously by an AI multi-agent system
solely for academic and educational evaluation. It does NOT constitute investment
advice, financial endorsement, or securities brokerage. Expected returns and
probability estimates are statistical models based on historical datasets.
Always perform independent due diligence and consult a SEBI-registered advisor.
================================================================================"""
