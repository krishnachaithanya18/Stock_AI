"""
agents/report_agent.py — Report Generation Agent
=================================================
Formats and consolidates the final research report for display and export.
Supports both Gemini AI generated summaries and structured template reports.
"""

from utils.helpers import get_timestamp


def run_report_agent(market_data: dict, tech_data: dict,
                     news_data: dict, research_data: dict) -> dict:
    """
    Consolidates the full research report from upstream agents.
    """
    symbol = market_data.get("symbol", "UNKNOWN")
    company = market_data.get("company_name", symbol)

    print(f"[Report Agent] Consolidating report for: {company} ({symbol})")

    # If research_data already has a compiled decision report, use it
    decision_details = research_data.get("decision_details")
    if decision_details and "report" in decision_details:
        rep_bundle = decision_details["report"]
        report_text = rep_bundle.get("report_text", "")
        report_type = rep_bundle.get("report_type", "template")
    else:
        # Build clean report from research data
        report_text = _build_report_text(market_data, tech_data, news_data, research_data)
        report_type = "template"

    return {
        "status": "success",
        "symbol": symbol,
        "company_name": company,
        "report_text": report_text,
        "gemini_text": report_text if report_type == "gemini" else None,
        "sections": {
            "full_report": report_text,
        },
        "report_type": report_type,
        "model_used": "gemini-1.5-flash" if report_type == "gemini" else "template",
        "tokens_used": 0,
        "generated_at": get_timestamp(),
        "error_message": None,
    }


def _build_report_text(market_data, tech_data, news_data, research_data) -> str:
    company = market_data.get("company_name", "Unknown")
    symbol = market_data.get("symbol", "")
    price_fmt = market_data.get("current_price_fmt", "N/A")
    verdict = research_data.get("verdict", "HOLD")
    pc = market_data.get("price_change", {})
    mi = market_data.get("market_info", {})

    bullish_str = "\n".join(f"  • {b}" for b in research_data.get("bullish_case", []))
    bearish_str = "\n".join(f"  • {b}" for b in research_data.get("bearish_case", []))

    return f"""================================================================================
INSTITUTIONAL STOCK RESEARCH REPORT: {company} ({symbol})
Generated: {get_timestamp()} | Architecture: LangGraph Multi-Agent AI
================================================================================

1. EXECUTIVE VERDICT: {verdict}
--------------------------------------------------------------------------------
• Recommendation    : {verdict}
• Confidence Score  : {research_data.get('confidence', 'N/A')}
• Stance            : {research_data.get('overall_stance', 'N/A')}
• Risk Level        : {research_data.get('risk_level', 'N/A')}

2. MARKET DATA OVERVIEW
--------------------------------------------------------------------------------
• Current Price     : {price_fmt}
• Daily Change      : {pc.get('one_day_pct_fmt', 'N/A')}
• 1-Month Change    : {pc.get('one_month_pct_fmt', 'N/A')}
• Market Cap        : {mi.get('market_cap_fmt', 'N/A')}
• 52-Week Range     : ₹{mi.get('fifty_two_week_low', 'N/A')} – ₹{mi.get('fifty_two_week_high', 'N/A')}

3. TECHNICAL ANALYSIS
--------------------------------------------------------------------------------
{tech_data.get('summary', 'Technical analysis completed.')}

4. NEWS & SENTIMENT
--------------------------------------------------------------------------------
{news_data.get('summary', 'Sentiment analysis completed.')}

5. KEY BULLISH PILLARS
--------------------------------------------------------------------------------
{bullish_str or '  • Stable market positioning.'}

6. KEY BEARISH FACTORS & RISKS
--------------------------------------------------------------------------------
{bearish_str or '  • Standard market volatility.'}

================================================================================
DISCLAIMER: For academic and research purposes only. Not financial advice.
================================================================================"""
