"""
agents/research_agent.py — Research & Decision Agent (Phase 5)
===============================================================
WHAT THIS AGENT DOES:
  This is the "senior analyst" of the system. It receives the
  structured findings from TWO other agents:
    - Agent 2 (Technical Analysis): RSI, MACD, SMA signals
    - Agent 3 (News Analysis):      Sentiment, key events

  It then:
    1. Compares technical signals vs news sentiment
    2. Identifies AGREEMENT (both bullish, or both bearish)
    3. Flags CONFLICTS (bullish news + bearish technicals, or vice versa)
    4. Evaluates specific risk factors
    5. Produces a structured, evidence-based research assessment

WHY A SEPARATE RESEARCH AGENT?
  In the real world, an analyst doesn't just list data —
  they reason about it. This agent models that reasoning step
  explicitly, keeping it separate from data collection (Phase 2),
  indicator calculation (Phase 3), and report writing (Phase 7).

IMPORTANT RULES:
  - Never claims certainty about future prices
  - Every conclusion is tied to specific evidence from other agents
  - Conflicts are always highlighted, not hidden
  - Risk level is always assessed
"""

from utils.helpers import get_timestamp


# =============================================================
# RISK FACTOR DEFINITIONS
# A structured list of risks to evaluate from the data
# =============================================================

RISK_DEFINITIONS = {
    "Overbought Risk": {
        "description": "RSI above 70 suggests the stock has risen sharply and may face a short-term pullback.",
        "condition"  : lambda tech, news: (
            tech.get("indicators", {}).get("rsi", {}).get("zone") == "Overbought"
        )
    },
    "Bearish Momentum Risk": {
        "description": "MACD is in bearish territory, indicating weakening upward momentum.",
        "condition"  : lambda tech, news: (
            "Bearish" in tech.get("indicators", {}).get("macd", {}).get("crossover", "")
        )
    },
    "Below Short-Term Average": {
        "description": "Price is trading below SMA20, signalling a short-term downtrend.",
        "condition"  : lambda tech, news: (
            "Below SMA20" in tech.get("indicators", {}).get("sma", {}).get("price_vs_sma20", "")
        )
    },
    "Death Cross Structure": {
        "description": "SMA20 has crossed below SMA50, indicating a bearish long-term trend structure.",
        "condition"  : lambda tech, news: (
            "Death Cross" in tech.get("indicators", {}).get("sma", {}).get("trend_structure", "")
        )
    },
    "Negative News Sentiment": {
        "description": "Recent news coverage is predominantly negative, which may weigh on market sentiment.",
        "condition"  : lambda tech, news: (
            news.get("overall_sentiment") == "NEGATIVE"
        )
    },
    "News-Technical Conflict": {
        "description": "Technical indicators and news sentiment point in opposite directions — elevated uncertainty.",
        "condition"  : lambda tech, news: _signals_conflict(tech, news)
    },
    "Regulatory Event Risk": {
        "description": "Regulatory-related news detected — SEBI/RBI actions can cause sharp price moves.",
        "condition"  : lambda tech, news: (
            "Regulatory" in news.get("key_events", [])
        )
    },
    "Leadership Change Risk": {
        "description": "Leadership transition news detected — management changes create short-term uncertainty.",
        "condition"  : lambda tech, news: (
            "Leadership Change" in news.get("key_events", [])
        )
    },
    "News Unavailable": {
        "description": "News data could not be retrieved — analysis is based on technical data only.",
        "condition"  : lambda tech, news: (
            news.get("overall_sentiment") == "UNAVAILABLE"
        )
    },
}


# =============================================================
# MAIN AGENT FUNCTION
# =============================================================

def run_research_agent(market_data: dict, tech_data: dict, news_data: dict) -> dict:
    """
    Main function for the Research & Decision Agent.

    Parameters:
        market_data : Output from run_market_data_agent()
        tech_data   : Output from run_technical_agent()
        news_data   : Output from run_news_agent()

    Returns:
        A dictionary containing:
            - status            : "success" or "error"
            - symbol            : Stock symbol
            - agreement_areas   : Where technical + news AGREE
            - conflict_areas    : Where technical + news DISAGREE
            - risks             : List of identified risk factors
            - risk_level        : "LOW", "MODERATE", or "HIGH"
            - bullish_case      : Evidence supporting a bullish view
            - bearish_case      : Evidence supporting a bearish view
            - overall_stance    : "BULLISH", "BEARISH", "NEUTRAL", or "MIXED"
            - confidence        : "LOW", "MODERATE", or "HIGH"
            - key_findings      : Top 3-5 most important findings
            - research_summary  : Plain-English paragraph for the report
            - disclaimer        : Legal/academic disclaimer
            - analysed_at       : Timestamp
    """

    symbol = market_data.get("symbol", "UNKNOWN")
    print(f"\n[Research Agent] Synthesising findings for: {symbol}")

    # --- Step 1: Extract key signals from upstream agents ---
    tech_signal  = tech_data.get("overall_signal", "UNKNOWN")   # BULLISH/BEARISH/NEUTRAL
    tech_score   = tech_data.get("signal_score", 0)
    news_sent    = news_data.get("overall_sentiment", "UNAVAILABLE")
    tech_bull    = tech_data.get("signals", {}).get("bullish", [])
    tech_bear    = tech_data.get("signals", {}).get("bearish", [])
    key_events   = news_data.get("key_events", [])
    article_cnt  = news_data.get("article_count", 0)

    print(f"[Research Agent]   Technical: {tech_signal} (score {tech_score:+d})")
    print(f"[Research Agent]   News:      {news_sent} ({article_cnt} articles)")

    # --- Step 2: Detect agreement / conflict areas ---
    agreement_areas, conflict_areas = _detect_agreement_conflicts(
        tech_signal, tech_score, news_sent, tech_bull, tech_bear, key_events
    )

    # --- Step 3: Evaluate risk factors ---
    risks = _evaluate_risks(tech_data, news_data)
    risk_level = _calculate_risk_level(risks, tech_score, news_sent)

    # --- Step 4: Build bullish and bearish cases ---
    bullish_case = _build_bullish_case(tech_data, news_data, market_data)
    bearish_case = _build_bearish_case(tech_data, news_data, market_data)

    # --- Step 5: Determine overall stance ---
    overall_stance, confidence = _determine_stance(
        tech_signal, tech_score, news_sent, conflict_areas, risks
    )

    # --- Step 6: Extract key findings (most important takeaways) ---
    key_findings = _extract_key_findings(
        market_data, tech_data, news_data, risks, overall_stance
    )

    # --- Step 7: Write research summary ---
    research_summary = _build_research_summary(
        symbol, market_data, tech_data, news_data,
        overall_stance, confidence, agreement_areas, conflict_areas, risk_level
    )

    result = {
        "status"           : "success",
        "symbol"           : symbol,
        "company_name"     : market_data.get("company_name", symbol),
        "tech_signal"      : tech_signal,
        "tech_score"       : tech_score,
        "news_sentiment"   : news_sent,
        "agreement_areas"  : agreement_areas,
        "conflict_areas"   : conflict_areas,
        "risks"            : risks,
        "risk_level"       : risk_level,
        "bullish_case"     : bullish_case,
        "bearish_case"     : bearish_case,
        "overall_stance"   : overall_stance,
        "confidence"       : confidence,
        "key_findings"     : key_findings,
        "research_summary" : research_summary,
        "disclaimer"       : (
            "This research assessment is generated by an AI system for academic "
            "purposes only. It is NOT financial advice and does NOT guarantee any "
            "investment outcome. Always consult a SEBI-registered financial advisor "
            "before making investment decisions."
        ),
        "analysed_at"      : get_timestamp(),
        "error_message"    : None,
    }

    print(f"[Research Agent] ✓ Stance: {overall_stance} | Confidence: {confidence} | Risk: {risk_level}")
    print(f"[Research Agent]   Agreements: {len(agreement_areas)} | Conflicts: {len(conflict_areas)} | Risks: {len(risks)}")

    return result


# =============================================================
# AGREEMENT / CONFLICT DETECTION
# =============================================================

def _signals_conflict(tech_data: dict, news_data: dict) -> bool:
    """
    Returns True if technical and news signals point in opposite directions.
    This is the core 'conflict detection' that makes this a research agent.
    """
    tech_signal = tech_data.get("overall_signal", "NEUTRAL")
    news_sent   = news_data.get("overall_sentiment", "NEUTRAL")

    bullish_tech = tech_signal == "BULLISH"
    bearish_tech = tech_signal == "BEARISH"
    positive_news = news_sent == "POSITIVE"
    negative_news = news_sent == "NEGATIVE"

    return (bullish_tech and negative_news) or (bearish_tech and positive_news)


def _detect_agreement_conflicts(tech_signal, tech_score, news_sent,
                                 tech_bull, tech_bear, key_events) -> tuple:
    """
    Compare technical and news signals to find where they agree and where they conflict.

    Returns:
        (agreement_areas, conflict_areas) — both are lists of strings
    """
    agreements = []
    conflicts  = []

    # --- Check overall direction alignment ---
    tech_direction  = _map_to_direction(tech_signal)
    news_direction  = _map_to_direction(news_sent)

    if tech_direction == news_direction and tech_direction != "neutral":
        agreements.append(
            f"Both technical indicators ({tech_signal}) and news sentiment ({news_sent}) "
            f"point in the SAME direction — this alignment increases signal reliability."
        )
    elif tech_direction != news_direction and "unavailable" not in [tech_direction, news_direction]:
        conflicts.append(
            f"CONFLICT: Technical analysis shows {tech_signal} but news sentiment is {news_sent}. "
            f"This divergence means the picture is mixed — higher uncertainty."
        )

    # --- RSI vs News conflict ---
    rsi_zone = None
    for signal in tech_bull + tech_bear:
        if "RSI" in signal and "Overbought" in signal:
            rsi_zone = "overbought"
        elif "RSI" in signal and "Oversold" in signal:
            rsi_zone = "oversold"

    if rsi_zone == "overbought" and news_sent == "POSITIVE":
        conflicts.append(
            "RSI is overbought (momentum risk) while news is positive — "
            "positive news may be priced in already."
        )
    elif rsi_zone == "oversold" and news_sent == "NEGATIVE":
        conflicts.append(
            "RSI is oversold (possible bounce) while news remains negative — "
            "technicals suggest a rebound but sentiment may suppress it."
        )

    # --- Earnings event alignment ---
    if "Earnings Report" in key_events:
        if news_sent == "POSITIVE" and tech_signal in ("BULLISH", "NEUTRAL"):
            agreements.append(
                "Earnings-related news detected with positive sentiment — "
                "this aligns with technical strength."
            )
        elif news_sent == "NEGATIVE":
            conflicts.append(
                "Earnings news detected with NEGATIVE sentiment — "
                "poor earnings can override technical signals quickly."
            )

    # --- Strong technical with weak news ---
    if tech_score >= 3 and news_sent == "NEGATIVE":
        conflicts.append(
            f"Strong technical signal (score {tech_score:+d}) conflicts with "
            f"negative news sentiment — market news may not be reflected in price yet."
        )

    return agreements, conflicts


def _map_to_direction(signal: str) -> str:
    """Map signal strings to a simple direction for comparison."""
    signal = signal.upper()
    if signal in ("BULLISH", "POSITIVE"):
        return "bullish"
    elif signal in ("BEARISH", "NEGATIVE"):
        return "bearish"
    else:
        return "neutral"


# =============================================================
# RISK EVALUATION
# =============================================================

def _evaluate_risks(tech_data: dict, news_data: dict) -> list:
    """
    Check each risk definition against the actual data.
    Returns a list of triggered risk dicts.
    """
    triggered = []
    for risk_name, risk_def in RISK_DEFINITIONS.items():
        try:
            if risk_def["condition"](tech_data, news_data):
                triggered.append({
                    "name"       : risk_name,
                    "description": risk_def["description"],
                })
        except Exception:
            pass   # Never let a risk check crash the agent
    return triggered


def _calculate_risk_level(risks: list, tech_score: int, news_sent: str) -> str:
    """
    Determine an overall risk level based on how many risks are triggered.

    LOW      : 0-1 risks
    MODERATE : 2-3 risks
    HIGH     : 4+ risks, or severe combination
    """
    risk_count = len(risks)
    risk_names = [r["name"] for r in risks]

    # Severe combinations automatically elevate to HIGH
    severe = (
        "News-Technical Conflict" in risk_names or
        ("Overbought Risk" in risk_names and "Bearish Momentum Risk" in risk_names)
    )

    if severe or risk_count >= 4:
        return "HIGH"
    elif risk_count >= 2:
        return "MODERATE"
    else:
        return "LOW"


# =============================================================
# BULLISH / BEARISH CASE BUILDER
# =============================================================

def _build_bullish_case(tech_data: dict, news_data: dict, market_data: dict) -> list:
    """
    Collect all bullish evidence from all three upstream agents.
    Returns a list of evidence strings.
    """
    evidence = []

    # Technical bullish signals
    for signal in tech_data.get("signals", {}).get("bullish", []):
        evidence.append(f"[Technical] {signal}")

    # Positive news
    if news_data.get("overall_sentiment") == "POSITIVE":
        cnt = news_data.get("sentiment_count", {}).get("positive", 0)
        evidence.append(f"[News] {cnt} positive news articles out of {news_data.get('article_count', 0)} total.")

    # Positive top headlines
    for art in news_data.get("top_headlines", []):
        if art.get("sentiment") == "positive":
            evidence.append(f"[News] {art['title'][:80]}")

    # Market data positives
    pc = market_data.get("price_change", {})
    if pc.get("one_month_pct") and pc["one_month_pct"] > 0:
        evidence.append(f"[Price] 1-month return: +{pc['one_month_pct']}%")
    mi = market_data.get("market_info", {})
    if mi.get("pe_ratio") and mi["pe_ratio"] < 20:
        evidence.append(f"[Valuation] P/E ratio of {mi['pe_ratio']} is below 20 — relatively reasonable valuation.")

    return evidence if evidence else ["No strong bullish evidence identified from available data."]


def _build_bearish_case(tech_data: dict, news_data: dict, market_data: dict) -> list:
    """
    Collect all bearish evidence from all three upstream agents.
    Returns a list of evidence strings.
    """
    evidence = []

    # Technical bearish signals
    for signal in tech_data.get("signals", {}).get("bearish", []):
        evidence.append(f"[Technical] {signal}")

    # Negative news
    if news_data.get("overall_sentiment") == "NEGATIVE":
        cnt = news_data.get("sentiment_count", {}).get("negative", 0)
        evidence.append(f"[News] {cnt} negative news articles out of {news_data.get('article_count', 0)} total.")

    # Negative top headlines
    for art in news_data.get("top_headlines", []):
        if art.get("sentiment") == "negative":
            evidence.append(f"[News] {art['title'][:80]}")

    # Market data negatives
    pc = market_data.get("price_change", {})
    if pc.get("one_month_pct") and pc["one_month_pct"] < 0:
        evidence.append(f"[Price] 1-month return: {pc['one_month_pct']}% — negative momentum.")

    return evidence if evidence else ["No strong bearish evidence identified from available data."]


# =============================================================
# OVERALL STANCE DETERMINATION
# =============================================================

def _determine_stance(tech_signal, tech_score, news_sent, conflict_areas, risks) -> tuple:
    """
    Combine all signals to determine an overall research stance.

    Returns:
        (overall_stance, confidence) tuple

    Stance values : "BULLISH", "BEARISH", "NEUTRAL", "MIXED"
    Confidence    : "LOW", "MODERATE", "HIGH"
    """
    has_conflicts = len(conflict_areas) > 0
    risk_names    = [r["name"] for r in risks]

    # Map tech and news to numeric votes
    tech_vote = {"BULLISH": +2, "NEUTRAL": 0, "BEARISH": -2}.get(tech_signal, 0)
    news_vote = {"POSITIVE": +1, "NEUTRAL": 0, "NEGATIVE": -1, "UNAVAILABLE": 0}.get(news_sent, 0)
    total     = tech_vote + news_vote

    # Determine stance
    if has_conflicts:
        stance = "MIXED"
    elif total >= 2:
        stance = "BULLISH"
    elif total <= -2:
        stance = "BEARISH"
    else:
        stance = "NEUTRAL"

    # Determine confidence
    if has_conflicts or news_sent == "UNAVAILABLE":
        confidence = "LOW"
    elif len(risks) >= 3:
        confidence = "LOW"
    elif len(risks) == 0 and not has_conflicts:
        confidence = "HIGH"
    else:
        confidence = "MODERATE"

    return stance, confidence


# =============================================================
# KEY FINDINGS EXTRACTOR
# =============================================================

def _extract_key_findings(market_data, tech_data, news_data, risks, stance) -> list:
    """
    Extract the top 5 most important findings to highlight in the report.
    These are the 'headline bullets' a human analyst would lead with.
    """
    findings = []

    # 1. Price + change
    price = market_data.get("current_price_fmt", "N/A")
    pc    = market_data.get("price_change", {})
    day   = pc.get("one_day_pct_fmt", "N/A")
    month = pc.get("one_month_pct_fmt", "N/A")
    findings.append(f"Current price: {price} | 1-Day: {day} | 1-Month: {month}")

    # 2. Technical overall
    rsi  = tech_data.get("indicators", {}).get("rsi", {})
    macd = tech_data.get("indicators", {}).get("macd", {})
    findings.append(
        f"Technical signals: {tech_data.get('overall_signal','?')} "
        f"(RSI {rsi.get('value','?')} — {rsi.get('zone','?')}, "
        f"MACD: {macd.get('crossover','?')})"
    )

    # 3. News sentiment
    sc = news_data.get("sentiment_count", {})
    findings.append(
        f"News sentiment: {news_data.get('overall_sentiment','?')} "
        f"({sc.get('positive',0)} positive, {sc.get('negative',0)} negative, "
        f"{sc.get('neutral',0)} neutral from {news_data.get('article_count',0)} articles)"
    )

    # 4. Key events if any
    if news_data.get("key_events"):
        findings.append(f"Key events in news: {', '.join(news_data['key_events'])}")

    # 5. Top risk or no risk
    if risks:
        findings.append(f"Primary risk: {risks[0]['name']} — {risks[0]['description']}")
    else:
        findings.append("No major risk factors identified from available data.")

    return findings


# =============================================================
# RESEARCH SUMMARY BUILDER
# =============================================================

def _build_research_summary(symbol, market_data, tech_data, news_data,
                              stance, confidence, agreements, conflicts, risk_level) -> str:
    """
    Write a concise, evidence-based research paragraph.
    Uses specific numbers from the data — no vague or invented statements.
    """
    company = market_data.get("company_name", symbol)
    price   = market_data.get("current_price_fmt", "N/A")
    t_sig   = tech_data.get("overall_signal", "?")
    t_score = tech_data.get("signal_score", 0)
    n_sent  = news_data.get("overall_sentiment", "?")
    n_count = news_data.get("article_count", 0)
    rsi_val = tech_data.get("indicators", {}).get("rsi", {}).get("value", "?")
    macd_co = tech_data.get("indicators", {}).get("macd", {}).get("crossover", "?")

    conflict_note = ""
    if conflicts:
        conflict_note = (
            f" However, there are {len(conflicts)} signal conflict(s) that "
            f"increase uncertainty: {conflicts[0]}"
        )

    agreement_note = ""
    if agreements:
        agreement_note = f" There are {len(agreements)} area(s) of signal agreement."

    return (
        f"{company} ({symbol}) is currently trading at {price}. "
        f"Technical analysis generates an overall {t_sig} signal (score: {t_score:+d}), "
        f"with RSI at {rsi_val} and MACD in {macd_co} territory. "
        f"News analysis of {n_count} articles indicates {n_sent} sentiment."
        f"{agreement_note}"
        f"{conflict_note} "
        f"Overall research stance: {stance} with {confidence} confidence and {risk_level} risk level. "
        f"This is an AI-generated assessment based on publicly available data "
        f"and should NOT be treated as financial advice."
    )
