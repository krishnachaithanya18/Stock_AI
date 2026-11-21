"""
agents/report_agent.py — Report Generation Agent (Phase 7 — Full Version)
===========================================================================
WHAT THIS AGENT DOES:
  Generates a complete, structured research report as the final step
  of the LangGraph workflow.

TWO MODES:
  1. GEMINI MODE (primary):
     Sends a carefully crafted prompt to Gemini with all agent findings.
     Gemini writes a natural-language research report like a human analyst.

  2. TEMPLATE MODE (fallback):
     If Gemini is unavailable (no billing, timeout, etc.), generates a
     complete structured report purely from the data already collected.
     This mode is 100% data-driven and fully usable for the project.

WHY HAVE A FALLBACK?
  Agentic AI systems must be ROBUST. The report is the deliverable —
  the system should always produce one, even if the LLM step fails.
  This is called "graceful degradation" in production systems.

PROMPT ENGINEERING:
  The Gemini prompt is carefully designed to:
  - Feed in ALL real data from Agents 1–4 (no hallucination possible)
  - Ask for structured sections (overview, analysis, risks, assessment)
  - Instruct Gemini to cite specific numbers, not make up new ones
  - Require a disclaimer about AI-generated financial content
"""

import json
from utils.helpers import get_timestamp
from utils.gemini_client import call_gemini


def run_report_agent(market_data: dict, tech_data: dict,
                     news_data: dict, research_data: dict) -> dict:
    """
    Generate the final research report — with Gemini if available,
    otherwise with the structured template fallback.

    Parameters:
        market_data   : Output from Market Data Agent
        tech_data     : Output from Technical Analysis Agent
        news_data     : Output from News Analysis Agent
        research_data : Output from Research & Decision Agent

    Returns:
        dict with 'report_text', 'sections', 'status',
                  'report_type' ("gemini" or "template"), 'generated_at'
    """

    symbol  = market_data.get("symbol", "UNKNOWN")
    company = market_data.get("company_name", symbol)

    print(f"[Report Agent] Generating report for: {company}")

    # --- Try Gemini first ---
    gemini_result = _try_gemini_report(market_data, tech_data, news_data, research_data)

    if gemini_result["status"] == "success":
        print(f"[Report Agent] ✓ Gemini report generated ({len(gemini_result['report_text'])} chars)")
        return gemini_result

    # --- Fallback: template report ---
    print(f"[Report Agent] ⚠ Gemini unavailable ({gemini_result.get('error', 'unknown error')})")
    print(f"[Report Agent] → Using template report (all real data, no Gemini)")

    template_result = _build_template_report(market_data, tech_data, news_data, research_data)
    return template_result


# =============================================================
# GEMINI REPORT GENERATION
# =============================================================

def _try_gemini_report(market_data, tech_data, news_data, research_data) -> dict:
    """
    Build a data summary, craft the Gemini prompt, call the API,
    and return the result dict.

    PROMPT ENGINEERING STRATEGY:
      We give Gemini a data-rich context (numbers, signals, headlines)
      and ask it to INTERPRET, not invent. The key instruction is:
      "Only use the numbers and facts provided below."
    """

    symbol  = market_data.get("symbol", "UNKNOWN")
    company = market_data.get("company_name", symbol)

    # --- Build a compact data context for the prompt ---
    data_context = _build_data_context(market_data, tech_data, news_data, research_data)

    # --- Craft the prompt ---
    prompt = f"""You are a professional Indian stock market research analyst.
Using ONLY the data provided below, write a structured research report for {company} ({symbol}).

Do NOT invent any numbers, prices, or news. Use only what is provided.
Do NOT make buy/sell recommendations. Do NOT promise returns.
Write in clear, professional language suitable for a university project presentation.

=== MARKET DATA ===
{data_context["market"]}

=== TECHNICAL INDICATORS ===
{data_context["technical"]}

=== NEWS SENTIMENT ===
{data_context["news"]}

=== RESEARCH ASSESSMENT ===
{data_context["research"]}

Write the report in the following EXACT structure:

## 1. EXECUTIVE SUMMARY
(2-3 sentences: company, current price, overall research stance, and confidence level)

## 2. STOCK OVERVIEW
(Price, market cap, sector, 52-week range, P/E ratio — from the data above)

## 3. TECHNICAL ANALYSIS INTERPRETATION
(Explain RSI, MACD, and SMA findings in plain language. What do they mean together?)

## 4. NEWS & SENTIMENT ANALYSIS
(Summarise news sentiment and key events. Quote actual headlines provided above.)

## 5. BULLISH FACTORS
(List the evidence supporting a bullish view, citing specific numbers)

## 6. BEARISH FACTORS & RISKS
(List the evidence supporting a bearish view and risk factors, citing specific numbers)

## 7. SIGNAL CONFLICTS & UNCERTAINTIES
(If technical and news signals disagree, explain the conflict and what it means)

## 8. OVERALL RESEARCH ASSESSMENT
(Evidence-based assessment of the stock's current position. No price predictions.)

## 9. DISCLAIMER
(State clearly this is AI-generated for academic purposes only, not financial advice)

Keep each section concise (3-6 sentences). Use Indian financial terminology where appropriate.
"""

    # --- Call Gemini ---
    print(f"[Report Agent] Calling Gemini ({len(prompt)} char prompt)...")
    response = call_gemini(prompt, temperature=0.3, max_tokens=2000)

    if response["status"] != "success":
        return {
            "status"      : "error",
            "report_text" : "",
            "sections"    : {},
            "report_type" : "gemini_failed",
            "error"       : response.get("error", "Gemini call failed"),
            "generated_at": get_timestamp(),
        }

    # --- Wrap the Gemini text in a full report ---
    full_report = _wrap_gemini_output(company, symbol, response["text"])

    return {
        "status"      : "success",
        "symbol"      : symbol,
        "company_name": company,
        "report_text" : full_report,
        "gemini_text" : response["text"],   # Raw Gemini output (for dashboard)
        "sections"    : {"gemini_report": response["text"]},
        "report_type" : "gemini",
        "model_used"  : response.get("model", "unknown"),
        "tokens_used" : response.get("tokens_used", "N/A"),
        "generated_at": get_timestamp(),
        "error_message": None,
    }


def _build_data_context(market_data, tech_data, news_data, research_data) -> dict:
    """
    Compress all agent outputs into concise text blocks for the prompt.

    WHY COMPRESS?
      Gemini has a token limit. We extract only the most important fields
      so the prompt stays focused and cost-effective.
    """

    # Market context
    mi = market_data.get("market_info", {})
    pc = market_data.get("price_change", {})
    market_ctx = (
        f"Company: {market_data.get('company_name', 'N/A')}\n"
        f"Symbol: {market_data.get('symbol', 'N/A')}\n"
        f"Current Price: {market_data.get('current_price_fmt', 'N/A')}\n"
        f"1-Day Change: {pc.get('one_day_pct_fmt', 'N/A')}\n"
        f"1-Week Change: {pc.get('one_week_pct_fmt', 'N/A')}\n"
        f"1-Month Change: {pc.get('one_month_pct_fmt', 'N/A')}\n"
        f"Market Cap: {mi.get('market_cap_fmt', 'N/A')}\n"
        f"P/E Ratio: {mi.get('pe_ratio', 'N/A')}\n"
        f"52-Week High: ₹{mi.get('fifty_two_week_high', 'N/A')}\n"
        f"52-Week Low: ₹{mi.get('fifty_two_week_low', 'N/A')}\n"
        f"Sector: {mi.get('sector', 'N/A')}\n"
        f"Beta: {mi.get('beta', 'N/A')}\n"
    )

    # Technical context
    rsi  = tech_data.get("indicators", {}).get("rsi", {})
    macd = tech_data.get("indicators", {}).get("macd", {})
    sma  = tech_data.get("indicators", {}).get("sma", {})
    bull_sigs = "\n".join(f"  - {s}" for s in tech_data.get("signals", {}).get("bullish", []))
    bear_sigs = "\n".join(f"  - {s}" for s in tech_data.get("signals", {}).get("bearish", []))
    tech_ctx = (
        f"RSI (14-day): {rsi.get('value', 'N/A')} — Zone: {rsi.get('zone', 'N/A')}\n"
        f"RSI Interpretation: {rsi.get('interpretation', 'N/A')}\n"
        f"MACD Line: {macd.get('macd_latest', 'N/A')} | Signal: {macd.get('signal_latest', 'N/A')}\n"
        f"MACD Histogram: {macd.get('histogram_latest', 'N/A')} | Status: {macd.get('crossover', 'N/A')}\n"
        f"SMA 20: ₹{sma.get('sma20', 'N/A')} ({sma.get('price_vs_sma20', 'N/A')})\n"
        f"SMA 50: ₹{sma.get('sma50', 'N/A')} ({sma.get('price_vs_sma50', 'N/A')})\n"
        f"Trend Structure: {sma.get('trend_structure', 'N/A')}\n"
        f"Overall Technical Signal: {tech_data.get('overall_signal', 'N/A')} (Score: {tech_data.get('signal_score', 0):+d})\n"
        f"Bullish Signals:\n{bull_sigs or '  None'}\n"
        f"Bearish Signals:\n{bear_sigs or '  None'}\n"
    )

    # News context
    sc = news_data.get("sentiment_count", {})
    top_lines = "\n".join(
        f"  [{art.get('sentiment','?')}] {art.get('title','')[:80]}"
        for art in news_data.get("top_headlines", [])[:5]
    )
    news_ctx = (
        f"Articles Analysed: {news_data.get('article_count', 0)}\n"
        f"Overall Sentiment: {news_data.get('overall_sentiment', 'N/A')}\n"
        f"Positive: {sc.get('positive', 0)} | Negative: {sc.get('negative', 0)} | Neutral: {sc.get('neutral', 0)}\n"
        f"Key Events: {', '.join(news_data.get('key_events', [])) or 'None detected'}\n"
        f"Top Headlines:\n{top_lines or '  None available'}\n"
    )

    # Research context
    risks = "\n".join(f"  - [{r['name']}] {r['description']}" for r in research_data.get("risks", []))
    agreements = "\n".join(f"  - {a}" for a in research_data.get("agreement_areas", []))
    conflicts  = "\n".join(f"  - {c}" for c in research_data.get("conflict_areas", []))
    research_ctx = (
        f"Overall Stance: {research_data.get('overall_stance', 'N/A')}\n"
        f"Confidence: {research_data.get('confidence', 'N/A')}\n"
        f"Risk Level: {research_data.get('risk_level', 'N/A')}\n"
        f"Signal Agreements:\n{agreements or '  None'}\n"
        f"Signal Conflicts:\n{conflicts or '  None'}\n"
        f"Risk Factors:\n{risks or '  None identified'}\n"
    )

    return {
        "market"   : market_ctx,
        "technical": tech_ctx,
        "news"     : news_ctx,
        "research" : research_ctx,
    }


def _wrap_gemini_output(company: str, symbol: str, gemini_text: str) -> str:
    """Add a professional header/footer around Gemini's raw markdown output."""
    border = "=" * 55
    return (
        f"\n{border}\n"
        f"  INDIAN STOCK MARKET RESEARCH REPORT (AI-Generated)\n"
        f"  {company} ({symbol})\n"
        f"  Generated: {get_timestamp()} | Powered by Google Gemini\n"
        f"{border}\n\n"
        f"{gemini_text}\n"
        f"\n{border}\n"
        f"  DATA SOURCES: Yahoo Finance (delayed) + Google News RSS\n"
        f"  GENERATED BY: LangGraph Multi-Agent System (5 Agents)\n"
        f"  FOR ACADEMIC USE ONLY — NOT FINANCIAL ADVICE\n"
        f"{border}\n"
    )


# =============================================================
# TEMPLATE REPORT (FALLBACK — NO GEMINI NEEDED)
# =============================================================

def _build_template_report(market_data, tech_data, news_data, research_data) -> dict:
    """
    Generate a complete, professional structured report using
    only the data already collected — no LLM required.

    This is NOT a placeholder — it is a fully usable report
    for academic presentation purposes.
    """

    symbol  = market_data.get("symbol", "UNKNOWN")
    company = market_data.get("company_name", symbol)
    sections = {}

    mi = market_data.get("market_info", {})
    pc = market_data.get("price_change", {})
    rsi  = tech_data.get("indicators", {}).get("rsi", {})
    macd = tech_data.get("indicators", {}).get("macd", {})
    sma  = tech_data.get("indicators", {}).get("sma", {})
    sc   = news_data.get("sentiment_count", {})

    # Executive Summary
    sections["executive_summary"] = (
        f"## 1. EXECUTIVE SUMMARY\n"
        f"{company} ({symbol}) is trading at {market_data.get('current_price_fmt','N/A')} "
        f"with a 1-month return of {pc.get('one_month_pct_fmt','N/A')}. "
        f"The multi-agent analysis assigns an overall research stance of "
        f"{research_data.get('overall_stance','N/A')} with "
        f"{research_data.get('confidence','N/A')} confidence "
        f"and a {research_data.get('risk_level','N/A')} risk level.\n"
    )

    # Stock Overview
    sections["overview"] = (
        f"## 2. STOCK OVERVIEW\n"
        f"Current Price  : {market_data.get('current_price_fmt','N/A')}\n"
        f"Market Cap     : {mi.get('market_cap_fmt','N/A')}\n"
        f"Sector         : {mi.get('sector','N/A')} | Industry: {mi.get('industry','N/A')}\n"
        f"P/E Ratio      : {mi.get('pe_ratio','N/A')} | EPS: {mi.get('eps','N/A')}\n"
        f"52-Week High   : ₹{mi.get('fifty_two_week_high','N/A')}\n"
        f"52-Week Low    : ₹{mi.get('fifty_two_week_low','N/A')}\n"
        f"Beta           : {mi.get('beta','N/A')} (volatility vs Nifty 50)\n"
        f"Dividend Yield : {mi.get('dividend_yield','0')}%\n"
        f"Price Changes  : 1D {pc.get('one_day_pct_fmt','N/A')} | "
        f"1W {pc.get('one_week_pct_fmt','N/A')} | 1M {pc.get('one_month_pct_fmt','N/A')}\n"
    )

    # Technical Analysis
    sections["technical"] = (
        f"## 3. TECHNICAL ANALYSIS\n"
        f"RSI ({rsi.get('period',14)}-day): {rsi.get('value','N/A')} [{rsi.get('zone','N/A')}]\n"
        f"  → {rsi.get('interpretation','')}\n\n"
        f"MACD (12,26,9): {macd.get('crossover','N/A')}\n"
        f"  MACD {macd.get('macd_latest','N/A')} | Signal {macd.get('signal_latest','N/A')} | Histogram {macd.get('histogram_latest','N/A')}\n"
        f"  → {macd.get('interpretation','')}\n\n"
        f"Moving Averages:\n"
        f"  SMA 20: ₹{sma.get('sma20','N/A')} | Price is {sma.get('price_vs_sma20','N/A')} by {abs(sma.get('pct_above_sma20') or 0)}%\n"
        f"  SMA 50: ₹{sma.get('sma50','N/A')} | Price is {sma.get('price_vs_sma50','N/A')} by {abs(sma.get('pct_above_sma50') or 0)}%\n"
        f"  Structure: {sma.get('trend_structure','N/A')}\n\n"
        f"Overall Technical Signal: {tech_data.get('overall_signal','N/A')} (Score: {tech_data.get('signal_score',0):+d})\n"
    )

    # News Sentiment
    headlines_text = ""
    for art in news_data.get("top_headlines", [])[:5]:
        icon  = art.get("sentiment_icon","🟡")
        title = art.get("title","")[:75]
        headlines_text += f"  {icon} {title}\n"

    sections["news"] = (
        f"## 4. NEWS & SENTIMENT ANALYSIS\n"
        f"Articles Analysed : {news_data.get('article_count',0)}\n"
        f"Overall Sentiment : {news_data.get('overall_sentiment','N/A')}\n"
        f"Breakdown: {sc.get('positive',0)} positive | {sc.get('negative',0)} negative | {sc.get('neutral',0)} neutral\n"
        f"Key Events: {', '.join(news_data.get('key_events',[])) or 'None detected'}\n\n"
        f"Top Headlines:\n{headlines_text or '  None available.'}\n"
        f"Source: {', '.join(news_data.get('sources_ok',['N/A']))}\n"
    )

    # Bullish Factors
    bull_items = "\n".join(f"  ✅ {b}" for b in research_data.get("bullish_case",[])[:6])
    sections["bullish"] = (
        f"## 5. BULLISH FACTORS\n"
        f"{bull_items or '  No strong bullish evidence identified.'}\n"
    )

    # Bearish Factors & Risks
    bear_items  = "\n".join(f"  ❌ {b}" for b in research_data.get("bearish_case",[])[:6])
    risk_items  = "\n".join(f"  ⚠️  [{r['name']}] {r['description']}" for r in research_data.get("risks",[]))
    sections["bearish"] = (
        f"## 6. BEARISH FACTORS & RISKS\n"
        f"{bear_items or '  No strong bearish evidence identified.'}\n\n"
        f"Risk Factors (Level: {research_data.get('risk_level','N/A')}):\n"
        f"{risk_items or '  No major risks identified.'}\n"
    )

    # Signal Conflicts
    agree_items    = "\n".join(f"  ✅ {a}" for a in research_data.get("agreement_areas",[]))
    conflict_items = "\n".join(f"  ⚠️  {c}" for c in research_data.get("conflict_areas",[]))
    sections["conflicts"] = (
        f"## 7. SIGNAL AGREEMENTS & CONFLICTS\n"
        f"Agreements:\n{agree_items or '  None.'}\n\n"
        f"Conflicts:\n{conflict_items or '  No conflicts detected — signals are aligned.'}\n"
    )

    # Overall Assessment
    sections["assessment"] = (
        f"## 8. OVERALL RESEARCH ASSESSMENT\n"
        f"Stance     : {research_data.get('overall_stance','N/A')}\n"
        f"Confidence : {research_data.get('confidence','N/A')}\n"
        f"Risk Level : {research_data.get('risk_level','N/A')}\n\n"
        f"{research_data.get('research_summary','')}\n"
    )

    # Key Findings
    findings_text = "\n".join(f"  {i+1}. {f}" for i,f in enumerate(research_data.get("key_findings",[])))
    sections["key_findings"] = (
        f"## 9. KEY FINDINGS SUMMARY\n"
        f"{findings_text or '  No findings available.'}\n"
    )

    # Disclaimer
    sections["disclaimer"] = (
        f"## DISCLAIMER\n"
        f"{research_data.get('disclaimer', 'This report is for academic purposes only and is not financial advice.')}\n"
        f"Data Source: Yahoo Finance (delayed ~15-20 min) | News: Google News RSS\n"
        f"Generated by: LangGraph Multi-Agent System (5 Agents) | Report Type: Template\n"
    )

    # Assemble full report
    border = "=" * 55
    header = (
        f"\n{border}\n"
        f"  INDIAN STOCK MARKET RESEARCH REPORT\n"
        f"  {company} ({symbol})\n"
        f"  Generated: {get_timestamp()} | Mode: Template (Structured Data)\n"
        f"{border}\n\n"
    )

    report_text = header + "\n\n".join(sections.values())

    print(f"[Report Agent] ✓ Template report generated ({len(report_text)} chars)")

    return {
        "status"       : "success",
        "symbol"       : symbol,
        "company_name" : company,
        "report_text"  : report_text,
        "gemini_text"  : None,
        "sections"     : sections,
        "report_type"  : "template",
        "model_used"   : "template (Gemini unavailable)",
        "tokens_used"  : 0,
        "generated_at" : get_timestamp(),
        "error_message": None,
    }
