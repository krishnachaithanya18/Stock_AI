"""
agents/research_agent.py — Research & Synthesis Agent (Backward-Compatible Bridge)
===================================================================================
Preserves full backward compatibility with tests/test_research_agent.py and
tests/test_end_to_end.py while augmenting findings with the comprehensive
multi-factor decision engine.
"""

from utils.helpers import get_timestamp
from agents.decision_agent import run_decision_agent
from agents.fundamental_agent import run_fundamental_agent
from agents.risk_agent import run_risk_agent


def run_research_agent(market_data: dict, tech_data: dict, news_data: dict,
                       fund_data: dict = None, risk_data: dict = None) -> dict:
    """
    Main research function. If fund_data and risk_data are not passed,
    computes them on the fly to provide a complete institutional assessment.
    """
    symbol = market_data.get("symbol", "UNKNOWN")

    if fund_data is None:
        fund_data = run_fundamental_agent(market_data)

    if risk_data is None:
        risk_data = run_risk_agent(market_data, tech_data, fund_data, news_data)

    decision_res = run_decision_agent(market_data, tech_data, fund_data, news_data, risk_data)

    # Legacy fields mapping
    tech_signal = tech_data.get("overall_signal", "NEUTRAL")
    news_sent = news_data.get("overall_sentiment", "NEUTRAL")

    # Agreements & Conflicts
    agreements = []
    conflicts = []
    if (tech_signal == "BULLISH" and news_sent == "POSITIVE") or (tech_signal == "BEARISH" and news_sent == "NEGATIVE"):
        agreements.append(f"Technical ({tech_signal}) and Sentiment ({news_sent}) are aligned.")
    elif (tech_signal == "BULLISH" and news_sent == "NEGATIVE") or (tech_signal == "BEARISH" and news_sent == "POSITIVE"):
        conflicts.append(f"Technical signal ({tech_signal}) conflicts with media sentiment ({news_sent}).")

    if fund_data.get("valuation_status") == "UNDERVALUED / ATTRACTIVE" and tech_signal == "BULLISH":
        agreements.append("Attractive fundamental valuation confirmed by bullish price action.")

    # Stance mapping for legacy tests
    verdict = decision_res["verdict"]
    if verdict == "STRONG BUY" or verdict == "BUY":
        overall_stance = "BULLISH"
    elif verdict == "AVOID":
        overall_stance = "BEARISH"
    elif conflicts:
        overall_stance = "MIXED"
    else:
        overall_stance = "NEUTRAL"

    # Risks mapping
    risks = []
    for r in risk_data.get("identified_risks", []):
        risks.append({"name": r["category"], "description": r["description"]})

    # Bullish & Bearish cases
    bullish_case = decision_res.get("bullish_reasons", [])
    bearish_case = decision_res.get("invalidation_factors", [])

    # Key findings
    key_findings = [
        f"Verdict: {verdict} with {decision_res['model_confidence']}% model confidence.",
        f"Estimated probability of positive return: {decision_res['probability_positive_return']}% over {decision_res['holding_period']}.",
        f"Target: {decision_res['trade_setup'].get('target_price_fmt', 'N/A')} | Stop-Loss: {decision_res['trade_setup'].get('stop_loss_fmt', 'N/A')}.",
        f"Technical: {tech_signal} (score {tech_data.get('signal_score', 0):+d}) | Valuation: {fund_data.get('valuation_status', 'N/A')}.",
        f"Risk Category: {risk_data.get('risk_level', 'MODERATE RISK')} (Annualized Volatility: {risk_data.get('volatility_annualized', 'N/A')}%)."
    ]

    result = {
        "status": "success",
        "symbol": symbol,
        "company_name": market_data.get("company_name", symbol),
        "tech_signal": tech_signal,
        "tech_score": tech_data.get("signal_score", 0),
        "news_sentiment": news_sent,
        "agreement_areas": agreements,
        "conflict_areas": conflicts,
        "risks": risks,
        "risk_level": risk_data.get("risk_level", "MODERATE RISK").replace(" RISK", ""),
        "bullish_case": bullish_case,
        "bearish_case": bearish_case,
        "overall_stance": overall_stance,
        "confidence": decision_res["confidence_rating"].replace(" CONFIDENCE", ""),
        "key_findings": key_findings,
        "research_summary": decision_res["report"]["report_text"][:400] + "...",
        "disclaimer": decision_res["disclaimer"],
        "analysed_at": get_timestamp(),
        "error_message": None,
        # Enhanced fields
        "verdict": verdict,
        "composite_score": decision_res["composite_score"],
        "probability_positive_return": decision_res["probability_positive_return"],
        "holding_period": decision_res["holding_period"],
        "trade_setup": decision_res["trade_setup"],
        "decision_details": decision_res,
    }

    return result
