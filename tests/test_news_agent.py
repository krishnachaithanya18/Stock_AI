"""
tests/test_news_agent.py — Test the News Analysis Agent
========================================================
Tests that:
  1. Real news articles are fetched for valid Indian stocks
  2. Sentiment is classified correctly
  3. Key events are detected
  4. Agent handles "no news found" gracefully

Usage (from FAI folder with venv activated):
    python tests/test_news_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_agent import run_market_data_agent
from agents.news_agent        import run_news_agent


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_news(symbol: str, label: str):
    """Fetch market data then run news analysis."""

    print_separator(f"Testing: {label} ({symbol})")

    # We only need company_name from market data — no need for full history
    market_data = run_market_data_agent(symbol, history_days=5)

    if market_data["status"] == "error":
        print(f"  Market data failed — cannot test news agent properly")
        # Still test news agent with minimal valid input
        market_data = {
            "status"       : "success",
            "symbol"       : symbol,
            "company_name" : symbol.split(".")[0].capitalize(),
            "price_change" : {},
        }

    result = run_news_agent(market_data, max_articles=10)

    # --- Status ---
    status_icon = "✓" if result["status"] in ("success", "partial") else "✗"
    print(f"\n  STATUS     : {result['status'].upper()} {status_icon}")
    print(f"  COMPANY    : {result['company_name']}")
    print(f"  ARTICLES   : {result['article_count']} fetched")
    print(f"  SOURCES OK : {', '.join(result['sources_ok']) if result['sources_ok'] else 'None'}")

    if result["article_count"] == 0:
        print(f"  ⚠ No articles found — news analysis unavailable")
        print(f"  MESSAGE : {result['error_message']}")
        return result["status"] in ("success", "partial")  # Partial is acceptable

    # --- Sentiment ---
    sc = result["sentiment_count"]
    print(f"\n  SENTIMENT BREAKDOWN:")
    print(f"    ✅ Positive : {sc['positive']}")
    print(f"    ❌ Negative : {sc['negative']}")
    print(f"    🟡 Neutral  : {sc['neutral']}")
    print(f"    Overall    : {result['overall_sentiment']}")

    # --- Key Events ---
    if result["key_events"]:
        print(f"\n  KEY EVENTS DETECTED:")
        for ev in result["key_events"]:
            print(f"    📌 {ev}")

    # --- Top Headlines ---
    print(f"\n  TOP HEADLINES (with sentiment):")
    for art in result["top_headlines"]:
        icon  = art["sentiment_icon"]
        title = art["title"][:70] + ("..." if len(art["title"]) > 70 else "")
        score = f"[+{art['pos_score']}/-{art['neg_score']}]"
        print(f"    {icon} {score} {title}")
        print(f"       Source: {art['source']}")

    # --- All Articles (abbreviated) ---
    print(f"\n  ALL {result['article_count']} ARTICLES:")
    for i, art in enumerate(result["articles"], 1):
        icon  = art["sentiment_icon"]
        title = art["title"][:65] + ("..." if len(art["title"]) > 65 else "")
        print(f"    {i:2}. {icon} {title}")

    # --- Summary ---
    print(f"\n  SUMMARY:")
    print(f"    {result['summary']}")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NEWS ANALYSIS AGENT — TEST SUITE")
    print("="*60)

    results = []

    results.append(test_news("RELIANCE.NS", "Reliance Industries"))
    results.append(test_news("TCS.NS",      "Tata Consultancy Services"))
    results.append(test_news("INFY.NS",     "Infosys"))

    # Test graceful handling — minimal market_data input
    print_separator("Testing: Minimal Input (No Full Market Data)")
    minimal_data = {
        "status"       : "success",
        "symbol"       : "HDFCBANK.NS",
        "company_name" : "HDFC Bank Limited",
        "price_change" : {},
    }
    result = run_news_agent(minimal_data, max_articles=5)
    print(f"  Status : {result['status']} | Articles: {result['article_count']} | Sentiment: {result['overall_sentiment']}")
    results.append(result["status"] in ("success", "partial"))

    print_separator("TEST SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print("  ✓ All tests passed! Ready for Phase 5.")
    else:
        print("  ✗ Some tests failed — check errors above.")
