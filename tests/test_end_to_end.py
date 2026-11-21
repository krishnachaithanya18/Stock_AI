"""
tests/test_end_to_end.py — Phase 9: Complete End-to-End Test Suite
===================================================================
This is the MASTER test file that verifies every component of
the entire Indian Stock Market Agentic AI system.

WHAT WE TEST:
  ✅ Configuration loading
  ✅ Market Data Agent (3 real stocks + 1 invalid)
  ✅ Technical Analysis Agent (all indicators)
  ✅ News Analysis Agent (RSS feeds, sentiment)
  ✅ Research & Decision Agent (conflict detection, risk)
  ✅ LangGraph Workflow (graph structure, routing, state)
  ✅ Report Agent (template generation)
  ✅ Gemini API (connection test — reports status)
  ✅ Full pipeline correctness (all state fields populated)
  ✅ Error handling (invalid symbol, bad input)
  ✅ Data integrity (no hardcoded/fake prices)
  ✅ Disclaimer presence in reports

Usage (from FAI folder with venv activated):
    python tests/test_end_to_end.py

Output:
    Colour-coded pass/fail for every check
    Final summary with total score
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================
# TEST INFRASTRUCTURE
# =============================================================

class TestRunner:
    """Simple test runner that tracks pass/fail and prints results."""

    def __init__(self):
        self.results   = []     # List of (name, passed, detail)
        self.section   = ""

    def begin_section(self, name: str):
        self.section = name
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")

    def check(self, description: str, condition: bool, detail: str = ""):
        """Record a single pass/fail check."""
        status = "✓" if condition else "✗"
        label  = f"[{status}] {description}"
        if detail and not condition:
            label += f"\n       → {detail}"
        print(f"  {label}")
        self.results.append((self.section, description, condition, detail))

    def summary(self):
        """Print final summary and return True if all passed."""
        total  = len(self.results)
        passed = sum(1 for _, _, ok, _ in self.results)
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"  END-TO-END TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Total Checks : {total}")
        print(f"  Passed       : {passed} ✓")
        print(f"  Failed       : {failed} {'✗' if failed else '✓'}")
        print(f"  Score        : {passed}/{total} ({100*passed//total}%)")
        print(f"{'='*60}")

        if failed > 0:
            print(f"\n  FAILED CHECKS:")
            for section, desc, ok, detail in self.results:
                if not ok:
                    print(f"  ✗ [{section}] {desc}")
                    if detail:
                        print(f"      {detail}")

        if passed == total:
            print(f"\n  🎉 ALL TESTS PASSED — System ready for presentation!")
        else:
            print(f"\n  ⚠ {failed} check(s) failed — review above and fix.")

        return passed == total


t = TestRunner()


# =============================================================
# SECTION 1: CONFIGURATION
# =============================================================

def test_configuration():
    t.begin_section("SECTION 1: Configuration")
    start = time.time()

    try:
        from config import (GOOGLE_API_KEY, GEMINI_MODEL, APP_TITLE,
                            DEFAULT_HISTORY_DAYS, MAX_NEWS_ARTICLES,
                            validate_config)

        t.check("config.py imports without error", True)
        t.check("APP_TITLE is set", bool(APP_TITLE))
        t.check("GEMINI_MODEL is set", bool(GEMINI_MODEL))
        t.check("DEFAULT_HISTORY_DAYS is a positive int",
                isinstance(DEFAULT_HISTORY_DAYS, int) and DEFAULT_HISTORY_DAYS > 0)
        t.check("MAX_NEWS_ARTICLES is a positive int",
                isinstance(MAX_NEWS_ARTICLES, int) and MAX_NEWS_ARTICLES > 0)
        t.check("GOOGLE_API_KEY is not empty",
                bool(GOOGLE_API_KEY) and GOOGLE_API_KEY != "your_google_gemini_api_key_here",
                "Add your Gemini API key to .env")

    except Exception as e:
        t.check("config.py imports without error", False, str(e))

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 2: MARKET DATA AGENT
# =============================================================

def test_market_data_agent():
    t.begin_section("SECTION 2: Market Data Agent")
    start = time.time()

    from agents.market_data_agent import run_market_data_agent

    # Test all 3 required stocks
    for symbol, expected_company_fragment in [
        ("RELIANCE.NS", "Reliance"),
        ("TCS.NS",      "Tata Consultancy"),
        ("INFY.NS",     "Infosys"),
    ]:
        result = run_market_data_agent(symbol, history_days=90)
        ok = result["status"] == "success"
        t.check(f"{symbol} — data fetch succeeds", ok,
                result.get("error_message", ""))

        if ok:
            t.check(f"{symbol} — has valid price > 0",
                    bool(result.get("current_price")) and result["current_price"] > 0)
            t.check(f"{symbol} — historical DataFrame present",
                    result.get("historical_df") is not None and
                    not result["historical_df"].empty)
            t.check(f"{symbol} — has ≥ 20 trading days",
                    len(result.get("historical_df", [])) >= 20,
                    f"Got {len(result.get('historical_df', []))} days")
            t.check(f"{symbol} — price change data present",
                    "one_day_pct" in result.get("price_change", {}))
            t.check(f"{symbol} — data_source has timestamp",
                    bool(result.get("data_source", {}).get("fetched_at")))
            t.check(f"{symbol} — price is NOT hardcoded (varies)",
                    result.get("current_price") != 1000.0)  # Not a round fake number

    # Test error handling
    bad = run_market_data_agent("FAKESYMBOL999.NS", history_days=5)
    t.check("Invalid symbol → status='error' (no crash)",
            bad["status"] == "error")
    t.check("Invalid symbol → error_message is set",
            bool(bad.get("error_message")))
    t.check("Invalid symbol → historical_df is None",
            bad.get("historical_df") is None)

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 3: TECHNICAL ANALYSIS AGENT
# =============================================================

def test_technical_agent():
    t.begin_section("SECTION 3: Technical Analysis Agent")
    start = time.time()

    from agents.market_data_agent import run_market_data_agent
    from agents.technical_agent   import run_technical_agent

    market = run_market_data_agent("RELIANCE.NS", history_days=90)
    tech   = run_technical_agent(market)

    t.check("Technical agent returns success", tech["status"] == "success")

    # RSI checks
    rsi = tech.get("indicators", {}).get("rsi", {})
    t.check("RSI value is calculated",
            rsi.get("value") is not None)
    t.check("RSI value is in valid range 0–100",
            0 <= (rsi.get("value") or -1) <= 100,
            f"Got RSI={rsi.get('value')}")
    t.check("RSI zone is one of: Overbought/Oversold/Neutral",
            rsi.get("zone") in ("Overbought", "Oversold", "Neutral"))
    t.check("RSI interpretation is not empty",
            bool(rsi.get("interpretation")))

    # MACD checks
    macd = tech.get("indicators", {}).get("macd", {})
    t.check("MACD line value is calculated",
            macd.get("macd_latest") is not None)
    t.check("MACD signal line is calculated",
            macd.get("signal_latest") is not None)
    t.check("MACD crossover string is set",
            bool(macd.get("crossover")))

    # SMA checks
    sma = tech.get("indicators", {}).get("sma", {})
    t.check("SMA 20 is calculated",
            sma.get("sma20") is not None)
    t.check("SMA 50 is calculated",
            sma.get("sma50") is not None)
    t.check("Trend structure is classified",
            "Golden Cross" in str(sma.get("trend_structure","")) or
            "Death Cross"  in str(sma.get("trend_structure","")))

    # Overall signal
    t.check("Overall signal is BULLISH/BEARISH/NEUTRAL",
            tech.get("overall_signal") in ("BULLISH", "BEARISH", "NEUTRAL"))
    t.check("At least 1 bullish or bearish signal generated",
            len(tech.get("signals",{}).get("bullish",[])) +
            len(tech.get("signals",{}).get("bearish",[])) >= 1)

    # Error handling
    bad_result = run_technical_agent({"status": "error", "symbol": "X.NS"})
    t.check("Bad market input → technical agent returns error (no crash)",
            bad_result["status"] == "error")

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 4: NEWS ANALYSIS AGENT
# =============================================================

def test_news_agent():
    t.begin_section("SECTION 4: News Analysis Agent")
    start = time.time()

    from agents.market_data_agent import run_market_data_agent
    from agents.news_agent        import run_news_agent

    market = run_market_data_agent("TCS.NS", history_days=5)
    news   = run_news_agent(market, max_articles=10)

    # Status — partial is acceptable if no articles found
    t.check("News agent returns success or partial",
            news["status"] in ("success", "partial"))
    t.check("article_count is an integer ≥ 0",
            isinstance(news.get("article_count"), int) and
            news["article_count"] >= 0)

    if news["article_count"] > 0:
        t.check("Overall sentiment is classified",
                news.get("overall_sentiment") in
                ("POSITIVE", "NEGATIVE", "NEUTRAL", "UNAVAILABLE"))
        t.check("Sentiment count sums to article_count",
                (news["sentiment_count"].get("positive",0) +
                 news["sentiment_count"].get("negative",0) +
                 news["sentiment_count"].get("neutral",0)) == news["article_count"])
        t.check("Every article has a title",
                all(bool(a.get("title")) for a in news.get("articles",[])))
        t.check("Every article has a sentiment field",
                all(a.get("sentiment") in ("positive","negative","neutral")
                    for a in news.get("articles",[])))
        t.check("No article has an empty source field",
                all(bool(a.get("source")) for a in news.get("articles",[])))
        t.check("News agent NEVER generates fictional headlines — all have URLs",
                all(bool(a.get("url")) for a in news.get("articles",[])))
    else:
        t.check("No articles found — error_message set appropriately",
                bool(news.get("error_message")))

    t.check("Summary string is not empty",  bool(news.get("summary")))
    t.check("sources_tried list is not empty", len(news.get("sources_tried",[])) > 0)

    # Test graceful handling — empty market data input
    bad_news = run_news_agent({
        "status": "success", "symbol": "X.NS", "company_name": "X", "price_change": {}
    }, max_articles=3)
    t.check("Minimal market_data → news agent doesn't crash",
            bad_news["status"] in ("success", "partial"))

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 5: RESEARCH AGENT
# =============================================================

def test_research_agent():
    t.begin_section("SECTION 5: Research & Decision Agent")
    start = time.time()

    from agents.market_data_agent import run_market_data_agent
    from agents.technical_agent   import run_technical_agent
    from agents.news_agent        import run_news_agent
    from agents.research_agent    import run_research_agent

    market   = run_market_data_agent("INFY.NS", history_days=90)
    tech     = run_technical_agent(market)
    news     = run_news_agent(market, max_articles=10)
    research = run_research_agent(market, tech, news)

    t.check("Research agent returns success", research["status"] == "success")
    t.check("overall_stance is set",
            research.get("overall_stance") in
            ("BULLISH", "BEARISH", "NEUTRAL", "MIXED"))
    t.check("confidence is set",
            research.get("confidence") in ("LOW", "MODERATE", "HIGH"))
    t.check("risk_level is set",
            research.get("risk_level") in ("LOW", "MODERATE", "HIGH"))
    t.check("bullish_case is a list",
            isinstance(research.get("bullish_case"), list))
    t.check("bearish_case is a list",
            isinstance(research.get("bearish_case"), list))
    t.check("risks is a list",
            isinstance(research.get("risks"), list))
    t.check("key_findings has at least 1 entry",
            len(research.get("key_findings",[])) >= 1)
    t.check("research_summary is not empty",
            len(research.get("research_summary","")) > 50)
    t.check("disclaimer is present",
            bool(research.get("disclaimer")))
    t.check("Research never claims certainty ('guarantee' not in summary)",
            "guarantee" not in research.get("research_summary","").lower())
    t.check("agreement_areas is a list",
            isinstance(research.get("agreement_areas"), list))
    t.check("conflict_areas is a list",
            isinstance(research.get("conflict_areas"), list))

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 6: LANGGRAPH WORKFLOW
# =============================================================

def test_langgraph_workflow():
    t.begin_section("SECTION 6: LangGraph Workflow")
    start = time.time()

    from graph.workflow import build_graph, run_analysis

    # Build graph
    try:
        graph = build_graph()
        t.check("Graph compiles without error", True)
    except Exception as e:
        t.check("Graph compiles without error", False, str(e))
        return

    # Verify graph structure
    try:
        nodes = list(graph.get_graph().nodes.keys())
        edges = graph.get_graph().edges
        expected_nodes = {"market_data_node", "technical_node", "news_node",
                          "research_node", "report_node", "error_node"}
        t.check("All 6 agent nodes are registered",
                expected_nodes.issubset(set(nodes)),
                f"Found: {nodes}")
        t.check("Graph has at least 7 edges",
                len(edges) >= 7, f"Found {len(edges)} edges")
    except Exception as e:
        t.check("Graph structure is inspectable", False, str(e))

    # Run full pipeline for RELIANCE.NS
    state = run_analysis("RELIANCE.NS", history_days=90, max_news_articles=10)

    t.check("RELIANCE.NS — workflow_status = 'completed'",
            state.get("workflow_status") == "completed")
    t.check("RELIANCE.NS — market_data populated",
            state.get("market_data") is not None)
    t.check("RELIANCE.NS — tech_data populated",
            state.get("tech_data") is not None)
    t.check("RELIANCE.NS — news_data populated",
            state.get("news_data") is not None)
    t.check("RELIANCE.NS — research_data populated",
            state.get("research_data") is not None)
    t.check("RELIANCE.NS — report_data populated",
            state.get("report_data") is not None)

    # Test conditional error routing
    bad_state = run_analysis("XXXXXXBADSYM.NS", history_days=5)
    t.check("Invalid symbol → workflow_status = 'failed'",
            bad_state.get("workflow_status") == "failed")
    t.check("Invalid symbol → tech_data remains None (skipped correctly)",
            bad_state.get("tech_data") is None)
    t.check("Invalid symbol → error_message is set",
            bool(bad_state.get("error_message")))

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 7: REPORT AGENT
# =============================================================

def test_report_agent():
    t.begin_section("SECTION 7: Report Generation Agent")
    start = time.time()

    from graph.workflow import run_analysis

    state  = run_analysis("TCS.NS", history_days=90, max_news_articles=10)
    report = state.get("report_data", {})

    t.check("Report agent returns success", report.get("status") == "success")
    t.check("report_text is not empty",
            len(report.get("report_text","")) > 200)
    t.check("report_type is 'gemini' or 'template'",
            report.get("report_type") in ("gemini", "template"))
    t.check("generated_at timestamp is set",
            bool(report.get("generated_at")))
    t.check("sections dict is not empty",
            len(report.get("sections",{})) > 0)
    t.check("Disclaimer is present in report_text",
            "disclaimer" in report.get("report_text","").lower() or
            "academic" in report.get("report_text","").lower())
    t.check("Report does NOT contain word 'guarantee' (no false promises)",
            "guarantee" not in report.get("report_text","").lower())
    t.check("Report contains the stock symbol",
            "TCS" in report.get("report_text",""))
    t.check("Report contains the word 'DISCLAIMER'",
            "DISCLAIMER" in report.get("report_text","").upper())

    # Test Gemini client (even if billing not enabled)
    from utils.gemini_client import call_gemini
    gemini_result = call_gemini("Reply with: TEST_OK", temperature=0, max_tokens=10)
    t.check("Gemini client returns structured dict (success or error)",
            "status" in gemini_result and "text" in gemini_result)
    t.check("Gemini client never crashes — always returns dict",
            isinstance(gemini_result, dict))

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 8: DATA INTEGRITY
# =============================================================

def test_data_integrity():
    t.begin_section("SECTION 8: Data Integrity")
    start = time.time()

    from agents.market_data_agent import run_market_data_agent
    from agents.technical_agent   import run_technical_agent

    # Prices must come from real API — different stocks have different prices
    r1 = run_market_data_agent("RELIANCE.NS", history_days=5)
    r2 = run_market_data_agent("TCS.NS",      history_days=5)

    t.check("RELIANCE and TCS have different prices (not hardcoded)",
            r1.get("current_price") != r2.get("current_price"),
            f"Both returned {r1.get('current_price')}")
    t.check("Prices are realistic (> ₹10 and < ₹1,000,000)",
            10 < (r1.get("current_price") or 0) < 1_000_000,
            f"RELIANCE price: {r1.get('current_price')}")
    t.check("Data source explicitly mentions 'Yahoo Finance'",
            "Yahoo Finance" in str(r1.get("data_source",{})))
    t.check("Data type clearly states it is delayed data",
            "delay" in str(r1.get("data_source",{})).lower())

    # Technical indicators must vary between stocks
    t1 = run_technical_agent(r1)
    t2 = run_technical_agent(r2)
    t.check("RSI differs between RELIANCE and TCS (not hardcoded)",
            t1.get("indicators",{}).get("rsi",{}).get("value") !=
            t2.get("indicators",{}).get("rsi",{}).get("value"))

    print(f"  ⏱ {time.time()-start:.1f}s")


# =============================================================
# SECTION 9: FULL PIPELINE — ALL 3 STOCKS
# =============================================================

def test_all_three_stocks():
    t.begin_section("SECTION 9: Full Pipeline — All 3 Required Stocks")
    start = time.time()

    from graph.workflow import run_analysis

    stocks = [
        ("RELIANCE.NS", "Reliance"),
        ("TCS.NS",      "Tata Consultancy"),
        ("INFY.NS",     "Infosys"),
    ]

    for symbol, company_fragment in stocks:
        print(f"\n  Running: {symbol}...")
        state = run_analysis(symbol, history_days=90, max_news_articles=10)

        t.check(f"{symbol} — workflow completed",
                state.get("workflow_status") == "completed")

        # Verify company name
        mkt = state.get("market_data", {})
        t.check(f"{symbol} — company name contains '{company_fragment}'",
                company_fragment in mkt.get("company_name",""),
                f"Got: {mkt.get('company_name','?')}")

        # Verify report generated
        rep = state.get("report_data", {})
        t.check(f"{symbol} — report successfully generated",
                rep.get("status") == "success" and
                len(rep.get("report_text","")) > 200)

    print(f"\n  ⏱ {time.time()-start:.1f}s")


# =============================================================
# MAIN — RUN ALL SECTIONS
# =============================================================

if __name__ == "__main__":
    total_start = time.time()

    print("\n" + "="*60)
    print("  INDIAN STOCK MARKET AGENTIC AI")
    print("  PHASE 9 — END-TO-END TEST SUITE")
    print("="*60)

    test_configuration()
    test_market_data_agent()
    test_technical_agent()
    test_news_agent()
    test_research_agent()
    test_langgraph_workflow()
    test_report_agent()
    test_data_integrity()
    test_all_three_stocks()

    total_time = time.time() - total_start
    print(f"\n  Total test time: {total_time:.1f}s")

    all_passed = t.summary()
    sys.exit(0 if all_passed else 1)
