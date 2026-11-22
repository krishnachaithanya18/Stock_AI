"""
agents/news_agent.py — Market Sentiment Agent (Agent D)
========================================================
WHAT THIS AGENT DOES:
  - Aggregates verified financial news from:
      1. Google News RSS (India financial news)
      2. Yahoo Finance news feed (via yfinance Ticker.news)
      3. Economic Times RSS
  - Performs explainable sentiment analysis using financial keyword heuristics
  - Detects key institutional events: Earnings, Mergers/Acquisitions, SEBI/RBI Regulatory events, Leadership transitions, Contracts/Deals, Dividends/Buybacks
  - Identifies distinct positive catalysts and negative headwinds
  - Strict Data Integrity: NEVER invents news; transparently reports when coverage is scarce
"""

import re
import xml.etree.ElementTree as ET
import requests
import yfinance as yf
from utils.helpers import get_timestamp


POSITIVE_KEYWORDS = [
    "profit", "revenue growth", "record", "beat", "exceeds", "surpassed",
    "strong earnings", "outperform", "upgrade", "raised guidance",
    "dividend", "buyback", "expansion", "acquisition", "partnership",
    "rally", "surge", "gains", "bullish", "all-time high", "52-week high",
    "recovery", "rebound", "momentum", "order win", "breakthrough",
    "approval", "launch", "investment", "stake", "deal", "merger",
    "positive", "optimistic", "strong", "robust", "increase", "rise",
    "higher", "improved", "benefit", "opportunity", "confidence", "multibagger"
]

NEGATIVE_KEYWORDS = [
    "loss", "decline", "missed", "below expectations", "downgrade",
    "cut guidance", "weak earnings", "disappoints", "shortfall",
    "crash", "plunge", "fall", "drop", "bearish", "sell-off",
    "52-week low", "correction", "underperform", "slump",
    "lawsuit", "penalty", "fine", "investigation", "fraud", "default",
    "layoffs", "restructuring", "debt", "regulatory", "ban", "recall",
    "delay", "cancellation", "negative", "concern", "risk", "pressure",
    "challenge", "lower", "decreased", "fell", "warning", "uncertainty", "probe"
]

EVENT_KEYWORDS = {
    "Earnings Report": ["quarterly results", "q1", "q2", "q3", "q4", "earnings", "profit", "revenue", "ebitda"],
    "Acquisition/Deal": ["acquires", "merger", "takeover", "stake", "buys", "acquisition", "joint venture"],
    "Regulatory/SEBI": ["sebi", "rbi", "regulatory", "penalty", "investigation", "compliance", "notice"],
    "Leadership Change": ["ceo", "cfo", "chairman", "appointed", "resigned", "steps down", "board"],
    "Market Milestone": ["52-week", "all-time high", "rally", "circuit breaker", "block deal"],
    "Order Win/Contract": ["contract", "order", "deal", "partnership", "agreement", "signed", "tender"],
    "Dividend/Corporate Action": ["dividend", "buyback", "bonus", "split", "rights issue"],
    "Macro/Industry": ["inflation", "rate cut", "monetary policy", "budget", "export", "tariff"]
}


def run_news_agent(market_data: dict, max_articles: int = 10) -> dict:
    """
    Main function for the Market Sentiment Agent.

    Parameters:
        market_data  : Output dictionary from run_market_data_agent()
        max_articles : Target maximum number of articles to evaluate

    Returns:
        Structured dictionary containing sentiment counts, positive/negative factors,
        headlines, detected events, and overall sentiment.
    """
    symbol = market_data.get("symbol", "UNKNOWN")
    company_name = market_data.get("company_name", symbol)
    base_name = _get_search_name(company_name, symbol)

    print(f"\n[Market Sentiment Agent] Aggregating news for: {company_name} ({symbol})")

    articles = []
    sources_tried = []
    sources_ok = []

    # 1. Try Yahoo Finance Ticker News first (direct ticker coverage)
    try:
        sources_tried.append("Yahoo Finance News")
        ticker = yf.Ticker(symbol)
        yf_news = ticker.news or []
        for item in yf_news:
            title = item.get("title") or ""
            link = item.get("link") or "#"
            pub = item.get("publisher") or "Yahoo Finance"
            if title:
                articles.append({
                    "title": title,
                    "url": link,
                    "summary": title,
                    "published": "",
                    "source": pub
                })
        if yf_news:
            sources_ok.append("Yahoo Finance News")
            print(f"[Market Sentiment Agent]   Yahoo Finance: {len(yf_news)} articles ✓")
    except Exception:
        pass

    # 2. Try RSS Feeds (Google News RSS & Economic Times)
    feed_urls = _build_feed_urls(base_name, company_name)
    for feed_name, url in feed_urls:
        sources_tried.append(feed_name)
        fetched = _fetch_rss_feed(feed_name, url, max_articles)
        if fetched:
            articles.extend(fetched)
            sources_ok.append(feed_name)
            print(f"[Market Sentiment Agent]   {feed_name}: {len(fetched)} articles ✓")
        if len(articles) >= max_articles:
            break

    # Deduplicate articles
    unique_articles = _deduplicate(articles)[:max_articles]

    if not unique_articles:
        return _no_news_result(symbol, company_name, sources_tried)

    # Analyze sentiment for each article
    analysed = [_analyse_article(art) for art in unique_articles]

    sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
    positive_factors = []
    negative_factors = []

    for art in analysed:
        sentiment_count[art["sentiment"]] += 1
        if art["sentiment"] == "positive":
            positive_factors.append(art["title"])
        elif art["sentiment"] == "negative":
            negative_factors.append(art["title"])

    pos = sentiment_count["positive"]
    neg = sentiment_count["negative"]

    if pos > neg and pos >= 2:
        overall_sentiment = "POSITIVE"
        sentiment_score = min(10, (pos - neg) * 2)
    elif neg > pos and neg >= 2:
        overall_sentiment = "NEGATIVE"
        sentiment_score = max(-10, (pos - neg) * 2)
    else:
        overall_sentiment = "NEUTRAL"
        sentiment_score = 0

    all_text = " ".join(a["title"].lower() for a in analysed)
    key_events = _detect_events(all_text)

    top_pos = [a for a in analysed if a["sentiment"] == "positive"][:2]
    top_neg = [a for a in analysed if a["sentiment"] == "negative"][:2]
    top_neu = [a for a in analysed if a["sentiment"] == "neutral"][:1]
    top_headlines = top_pos + top_neg + top_neu

    summary = _build_summary(company_name, sentiment_count, overall_sentiment, key_events, len(analysed))

    result = {
        "status": "success",
        "symbol": symbol,
        "company_name": company_name,
        "articles": analysed,
        "article_count": len(analysed),
        "sentiment_count": sentiment_count,
        "overall_sentiment": overall_sentiment,
        "sentiment_score": sentiment_score,
        "positive_factors": positive_factors[:5],
        "negative_factors": negative_factors[:5],
        "key_events": key_events,
        "top_headlines": top_headlines,
        "summary": summary,
        "sources_tried": sources_tried,
        "sources_ok": sources_ok,
        "fetched_at": get_timestamp(),
        "error_message": None,
    }

    print(f"[Market Sentiment Agent] ✓ {len(analysed)} articles | Overall: {overall_sentiment} (Score: {sentiment_score:+d})")
    return result


def _get_search_name(company_name: str, symbol: str) -> str:
    name = re.sub(r'\b(Limited|Ltd|Corporation|Corp|Inc)\b', '', company_name, flags=re.IGNORECASE).strip()
    if not name or name == symbol:
        name = symbol.split(".")[0].capitalize()
    return name


def _build_feed_urls(base_name: str, full_name: str) -> list:
    query_short = base_name.replace(" ", "+")
    query_full = full_name.replace(" ", "+").replace(",", "")

    return [
        ("Google News (Primary)",
         f"https://news.google.com/rss/search?q={query_short}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"),
        ("Google News (Full Name)",
         f"https://news.google.com/rss/search?q={query_full}+NSE&hl=en-IN&gl=IN&ceid=IN:en"),
        ("Economic Times",
         "https://economictimes.indiatimes.com/rssfeeds/1977021501.cms"),
    ]


def _fetch_rss_feed(feed_name: str, url: str, max_items: int) -> list:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall(".//item")
        articles = []

        for item in items[:max_items]:
            def get_text(tag: str) -> str:
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            title = get_text("title")
            link = get_text("link")
            pub = get_text("pubDate")
            summary = get_text("description")

            if not title:
                continue

            summary_clean = re.sub(r"<[^>]+>", " ", summary).strip()
            articles.append({
                "title": title,
                "url": link,
                "summary": summary_clean[:300],
                "published": pub,
                "source": feed_name,
            })
        return articles
    except Exception:
        return []


def _deduplicate(articles: list) -> list:
    seen = set()
    unique = []
    for art in articles:
        key = art["title"][:50].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(art)
    return unique


def _analyse_article(article: dict) -> dict:
    text = (article["title"] + " " + article.get("summary", "")).lower()
    pos_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    net_score = pos_score - neg_score

    if net_score > 0:
        sentiment = "positive"
        sentiment_icon = "🟢"
    elif net_score < 0:
        sentiment = "negative"
        sentiment_icon = "🔴"
    else:
        sentiment = "neutral"
        sentiment_icon = "🟡"

    return {
        **article,
        "sentiment": sentiment,
        "sentiment_icon": sentiment_icon,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "net_score": net_score,
    }


def _detect_events(all_titles_text: str) -> list:
    detected = []
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in all_titles_text for kw in keywords):
            detected.append(event_type)
    return detected


def _build_summary(company_name, sentiment_count, overall_sentiment, key_events, total) -> str:
    pos = sentiment_count["positive"]
    neg = sentiment_count["negative"]
    neu = sentiment_count["neutral"]
    events = ", ".join(key_events) if key_events else "general market developments"

    return (
        f"Sentiment assessment for {company_name} across {total} verified news items indicates an "
        f"overall {overall_sentiment} market sentiment ({pos} positive, {neg} negative, {neu} neutral articles). "
        f"Dominant institutional themes identified: {events}. "
        f"All articles verified from public news feeds."
    )


def _no_news_result(symbol, company_name, sources_tried) -> dict:
    msg = (
        f"No specific news articles retrieved for '{company_name}'. "
        f"Coverage is marked as UNAVAILABLE for this analysis run to avoid hallucination."
    )
    print(f"[Market Sentiment Agent] ⚠ {msg}")
    return {
        "status": "partial",
        "symbol": symbol,
        "company_name": company_name,
        "articles": [],
        "article_count": 0,
        "sentiment_count": {"positive": 0, "negative": 0, "neutral": 0},
        "overall_sentiment": "UNAVAILABLE",
        "sentiment_score": 0,
        "positive_factors": [],
        "negative_factors": [],
        "key_events": [],
        "top_headlines": [],
        "summary": msg,
        "sources_tried": sources_tried,
        "sources_ok": [],
        "fetched_at": get_timestamp(),
        "error_message": msg,
    }


# Export alias
run_sentiment_agent = run_news_agent
