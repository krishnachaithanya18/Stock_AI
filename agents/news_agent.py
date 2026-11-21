"""
agents/news_agent.py — News Analysis Agent (Phase 4)
======================================================
WHAT THIS AGENT DOES:
  - Fetches real financial news articles from FREE, public RSS feeds
  - Sources: Google News RSS (primary), Economic Times RSS (secondary)
  - Performs keyword-based sentiment analysis on each headline
  - Identifies important event types (earnings, acquisition, regulatory, etc.)
  - Returns structured findings — positive, negative, neutral counts + key events
  - NEVER generates fictional news
  - If ALL sources fail, explicitly reports news as unavailable

WHY RSS FEEDS?
  RSS (Really Simple Syndication) is a standard web format that news sites
  publish freely. No API key needed. We use 'feedparser' to read them.
  Google News RSS aggregates articles from ET, LiveMint, NDTV Profit, etc.

SENTIMENT ANALYSIS APPROACH:
  We use a keyword-matching approach — simple but transparent and explainable.
  Positive words (growth, profit, beat, surge, record...) → positive score
  Negative words (loss, decline, miss, lawsuit, crash...)  → negative score
  Net score determines article sentiment.
  This is intentionally simple for academic clarity. Phase 7 will use
  Gemini for deeper analysis of the full report.

DATA INTEGRITY RULES:
  - Every article includes its URL source
  - Published dates are shown
  - If RSS returns 0 articles, we report "no news found" — never invent news
"""

import re
import xml.etree.ElementTree as ET
import requests
from utils.helpers import get_timestamp


# =============================================================
# SENTIMENT KEYWORD LISTS
# =============================================================

POSITIVE_KEYWORDS = [
    # Financial performance
    "profit", "revenue growth", "record", "beat", "exceeds", "surpassed",
    "strong earnings", "outperform", "upgrade", "raised guidance",
    "dividend", "buyback", "expansion", "acquisition", "partnership",
    # Price / market
    "rally", "surge", "gains", "bullish", "all-time high", "52-week high",
    "recovery", "rebound", "momentum",
    # Business news
    "new contract", "order win", "breakthrough", "approval", "launch",
    "investment", "stake", "deal", "merger", "capex", "growth",
    "positive", "optimistic", "strong", "robust", "increase", "rise",
    "higher", "improved", "benefit", "opportunity", "confidence",
]

NEGATIVE_KEYWORDS = [
    # Financial performance
    "loss", "decline", "missed", "below expectations", "downgrade",
    "cut guidance", "weak earnings", "disappoints", "shortfall",
    # Price / market
    "crash", "plunge", "fall", "drop", "bearish", "sell-off",
    "52-week low", "correction", "underperform", "slump",
    # Business / legal
    "lawsuit", "penalty", "fine", "investigation", "fraud", "default",
    "layoffs", "restructuring", "write-off", "impairment", "debt",
    "regulatory", "ban", "recall", "delay", "cancellation",
    "negative", "concern", "risk", "pressure", "challenge", "lower",
    "decreased", "fell", "reduced", "warning", "uncertainty",
]

# Key event types that are especially important for investors
EVENT_KEYWORDS = {
    "Earnings Report"   : ["quarterly results", "q1", "q2", "q3", "q4", "earnings", "profit", "revenue"],
    "Acquisition/Merger": ["acquires", "merger", "takeover", "stake", "buys", "acquisition"],
    "Regulatory"        : ["sebi", "rbi", "regulatory", "penalty", "investigation", "compliance"],
    "Leadership Change" : ["ceo", "cfo", "coo", "chairman", "appointed", "resigned", "steps down"],
    "Market Movement"   : ["52-week", "all-time high", "rally", "crash", "circuit breaker"],
    "New Contract/Deal" : ["contract", "order", "deal", "partnership", "agreement", "signed"],
    "Dividend/Buyback"  : ["dividend", "buyback", "bonus", "split", "rights issue"],
    "Global/Macro"      : ["fed", "rbi rate", "inflation", "gdp", "us recession", "global"],
}


# =============================================================
# MAIN AGENT FUNCTION
# =============================================================

def run_news_agent(market_data: dict, max_articles: int = 10) -> dict:
    """
    Main function for the News Analysis Agent.

    Parameters:
        market_data   : Result dict from run_market_data_agent()
        max_articles  : Maximum number of articles to analyse (default: 10)

    Returns:
        A dictionary with:
            - status          : "success", "partial", or "error"
            - symbol          : Stock symbol
            - company_name    : Company name used for search
            - articles        : List of analysed article dicts
            - sentiment_count : {"positive": N, "negative": N, "neutral": N}
            - overall_sentiment: "POSITIVE", "NEGATIVE", or "NEUTRAL"
            - key_events      : List of important event types detected
            - top_headlines   : Top 5 headlines (positive + negative)
            - summary         : Plain-English summary
            - sources_tried   : Which RSS feeds were attempted
            - fetched_at      : Timestamp
            - error_message   : Only if status is "error"
    """

    symbol       = market_data.get("symbol", "UNKNOWN")
    company_name = market_data.get("company_name", symbol)

    # Strip exchange suffix for cleaner searches: RELIANCE.NS → RELIANCE
    base_name  = _get_search_name(company_name, symbol)

    print(f"\n[News Agent] Fetching news for: {company_name} ({symbol})")

    # --- Step 1: Try to fetch articles from multiple RSS sources ---
    all_articles   = []
    sources_tried  = []
    sources_ok     = []

    feed_urls = _build_feed_urls(base_name, company_name)

    for feed_name, url in feed_urls:
        sources_tried.append(feed_name)
        articles = _fetch_rss_feed(feed_name, url, max_articles)
        if articles:
            all_articles.extend(articles)
            sources_ok.append(feed_name)
            print(f"[News Agent]   {feed_name}: {len(articles)} articles ✓")
        else:
            print(f"[News Agent]   {feed_name}: 0 articles or fetch failed")

        # Stop once we have enough articles
        if len(all_articles) >= max_articles:
            break

    # Deduplicate by title similarity and limit to max_articles
    all_articles = _deduplicate(all_articles)[:max_articles]

    # --- Step 2: Handle case where NO articles were found ---
    if not all_articles:
        return _no_news_result(symbol, company_name, sources_tried)

    # --- Step 3: Analyse sentiment of each article ---
    analysed = [_analyse_article(article) for article in all_articles]

    # --- Step 4: Count sentiments ---
    sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
    for art in analysed:
        sentiment_count[art["sentiment"]] += 1

    # --- Step 5: Overall sentiment ---
    pos = sentiment_count["positive"]
    neg = sentiment_count["negative"]

    if pos > neg and pos >= 2:
        overall_sentiment = "POSITIVE"
    elif neg > pos and neg >= 2:
        overall_sentiment = "NEGATIVE"
    else:
        overall_sentiment = "NEUTRAL"

    # --- Step 6: Detect key events ---
    all_text   = " ".join(a["title"].lower() for a in analysed)
    key_events = _detect_events(all_text)

    # --- Step 7: Extract top headlines ---
    top_positive = [a for a in analysed if a["sentiment"] == "positive"][:2]
    top_negative = [a for a in analysed if a["sentiment"] == "negative"][:2]
    top_neutral  = [a for a in analysed if a["sentiment"] == "neutral"][:1]
    top_headlines = top_positive + top_negative + top_neutral

    # --- Step 8: Build summary ---
    summary = _build_summary(company_name, sentiment_count, overall_sentiment, key_events, len(analysed))

    result = {
        "status"            : "success",
        "symbol"            : symbol,
        "company_name"      : company_name,
        "articles"          : analysed,
        "article_count"     : len(analysed),
        "sentiment_count"   : sentiment_count,
        "overall_sentiment" : overall_sentiment,
        "key_events"        : key_events,
        "top_headlines"     : top_headlines,
        "summary"           : summary,
        "sources_tried"     : sources_tried,
        "sources_ok"        : sources_ok,
        "fetched_at"        : get_timestamp(),
        "error_message"     : None,
    }

    print(f"[News Agent] ✓ {len(analysed)} articles | Sentiment: {overall_sentiment}")
    print(f"[News Agent]   Positive: {pos} | Negative: {neg} | Neutral: {sentiment_count['neutral']}")
    if key_events:
        print(f"[News Agent]   Key events: {', '.join(key_events)}")

    return result


# =============================================================
# RSS FEED HELPERS
# =============================================================

def _get_search_name(company_name: str, symbol: str) -> str:
    """
    Extract a clean search term from the company name.
    'Reliance Industries Limited' → 'Reliance Industries'
    'Tata Consultancy Services Limited' → 'Tata Consultancy Services'
    Falls back to symbol root if name is unavailable.
    """
    # Remove common suffixes
    name = re.sub(r'\b(Limited|Ltd|Corporation|Corp|Inc)\b', '', company_name, flags=re.IGNORECASE).strip()
    if not name or name == symbol:
        # Use symbol root: RELIANCE.NS → Reliance
        name = symbol.split(".")[0].capitalize()
    return name


def _build_feed_urls(base_name: str, full_name: str) -> list:
    """
    Returns a list of (feed_name, url) tuples to try, in priority order.

    WHY MULTIPLE SOURCES?
    RSS feeds can be unreliable — Google may rate-limit, ET may be slow.
    By trying several, we maximise the chance of getting real news.
    """
    # URL-encode the search query (replace spaces with +)
    query_short = base_name.replace(" ", "+")
    query_full  = full_name.replace(" ", "+").replace(",", "")

    return [
        # Google News RSS — aggregates from ET, LiveMint, NDTV Profit, etc.
        ("Google News (Primary)",
         f"https://news.google.com/rss/search?q={query_short}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"),

        # Google News alternate query using full name
        ("Google News (Full Name)",
         f"https://news.google.com/rss/search?q={query_full}+NSE&hl=en-IN&gl=IN&ceid=IN:en"),

        # Economic Times company RSS
        ("Economic Times",
         f"https://economictimes.indiatimes.com/rssfeeds/1977021501.cms"),
    ]


def _fetch_rss_feed(feed_name: str, url: str, max_items: int) -> list:
    """
    Fetch and parse an RSS feed using requests + stdlib xml.etree.
    Returns a list of raw article dicts.
    Returns empty list (never raises) if fetch fails.

    WHY NOT feedparser?
      feedparser is not installed in this environment.
      We use Python's built-in xml.etree.ElementTree to parse RSS XML —
      this requires zero extra packages and works identically.

    RSS XML STRUCTURE:
      <rss>
        <channel>
          <item>
            <title>...</title>
            <link>...</link>
            <pubDate>...</pubDate>
            <description>...</description>
          </item>
        </channel>
      </rss>
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FAI-StockBot/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse the RSS XML
        root     = ET.fromstring(response.content)

        # RSS items are at: rss > channel > item
        # We use ".//item" to find all <item> tags anywhere in the tree
        items    = root.findall(".//item")

        articles = []
        for item in items[:max_items]:
            # Helper to safely extract text from an XML tag
            def get_text(tag: str) -> str:
                el = item.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            title   = get_text("title")
            link    = get_text("link")
            pub     = get_text("pubDate")
            summary = get_text("description")

            if not title:
                continue

            # Strip HTML tags from description
            summary_clean = re.sub(r"<[^>]+>", " ", summary).strip()

            articles.append({
                "title"    : title,
                "url"      : link,
                "summary"  : summary_clean[:300],
                "published": pub,
                "source"   : feed_name,
            })

        return articles

    except Exception as e:
        print(f"[News Agent]   WARNING: {feed_name} failed — {str(e)[:80]}")
        return []


def _deduplicate(articles: list) -> list:
    """
    Remove duplicate articles based on title similarity.
    Two titles are duplicates if one starts with the same 60 characters.
    """
    seen   = set()
    unique = []
    for art in articles:
        # Use first 60 chars as fingerprint
        key = art["title"][:60].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(art)
    return unique


# =============================================================
# SENTIMENT ANALYSIS
# =============================================================

def _analyse_article(article: dict) -> dict:
    """
    Score an article's sentiment using keyword matching.

    HOW IT WORKS:
      1. Combine title + summary into one text string
      2. Count how many POSITIVE_KEYWORDS appear
      3. Count how many NEGATIVE_KEYWORDS appear
      4. Net score = positive count - negative count
      5. Net > 0 → positive, Net < 0 → negative, Net == 0 → neutral

    WHY NOT USE AN LLM HERE?
      This keeps sentiment analysis fast, free, and transparent.
      The final Gemini report (Phase 7) will interpret the overall
      news picture with much more nuance.
    """
    text = (article["title"] + " " + article.get("summary", "")).lower()

    pos_score = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_score = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    net_score = pos_score - neg_score

    if net_score > 0:
        sentiment = "positive"
        sentiment_icon = "✅"
    elif net_score < 0:
        sentiment = "negative"
        sentiment_icon = "❌"
    else:
        sentiment = "neutral"
        sentiment_icon = "🟡"

    return {
        **article,                       # Keep all original fields
        "sentiment"       : sentiment,
        "sentiment_icon"  : sentiment_icon,
        "pos_score"       : pos_score,
        "neg_score"       : neg_score,
        "net_score"       : net_score,
    }


def _detect_events(all_titles_text: str) -> list:
    """
    Detect important event types across all article titles.
    Returns a list of event type strings found.
    """
    detected = []
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in all_titles_text for kw in keywords):
            detected.append(event_type)
    return detected


# =============================================================
# SUMMARY & ERROR HELPERS
# =============================================================

def _build_summary(company_name, sentiment_count, overall_sentiment, key_events, total) -> str:
    pos    = sentiment_count["positive"]
    neg    = sentiment_count["negative"]
    neu    = sentiment_count["neutral"]
    events = ", ".join(key_events) if key_events else "general market news"

    return (
        f"News analysis of {company_name} ({total} articles): "
        f"Overall sentiment is {overall_sentiment} "
        f"({pos} positive, {neg} negative, {neu} neutral articles). "
        f"Key topics detected: {events}. "
        f"Note: Sentiment is based on headline keyword analysis. "
        f"Always verify news from official sources before making decisions."
    )


def _no_news_result(symbol, company_name, sources_tried) -> dict:
    """Returned when NO articles are found from any source."""
    msg = (
        f"No news articles found for '{company_name}' from any source. "
        f"Sources tried: {', '.join(sources_tried)}. "
        f"This may be due to network restrictions or no recent news. "
        f"News analysis is UNAVAILABLE for this run."
    )
    print(f"[News Agent] ⚠ {msg}")
    return {
        "status"            : "partial",
        "symbol"            : symbol,
        "company_name"      : company_name,
        "articles"          : [],
        "article_count"     : 0,
        "sentiment_count"   : {"positive": 0, "negative": 0, "neutral": 0},
        "overall_sentiment" : "UNAVAILABLE",
        "key_events"        : [],
        "top_headlines"     : [],
        "summary"           : msg,
        "sources_tried"     : sources_tried,
        "sources_ok"        : [],
        "fetched_at"        : get_timestamp(),
        "error_message"     : msg,
    }
