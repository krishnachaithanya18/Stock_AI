"""
tests/test_research_agent.py — Test the Research & Decision Agent
==================================================================
Tests that the research agent correctly:
  1. Synthesises technical + news findings
  2. Detects agreement and conflict areas
  3. Evaluates risk factors
  4. Produces an overall stance with confidence

Usage (from FAI folder with venv activated):
    python tests/test_research_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_agent import run_market_data_agent
from agents.technical_agent   import run_technical_agent
from agents.news_agent        import run_news_agent
from agents.research_agent    import run_research_agent


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_full_pipeline(symbol: str, label: str):
    """Run all 4 agents in sequence and display research findings."""

    print_separator(f"Full Pipeline: {label} ({symbol})")

    # Phase 2 → Phase 3 → Phase 4 → Phase 5
    market_data = run_market_data_agent(symbol, history_days=90)
    if market_data["status"] == "error":
        print(f"  ✗ Market data failed: {market_data['error_message']}")
        return False

    tech_data   = run_technical_agent(market_data)
    news_data   = run_news_agent(market_data, max_articles=10)
    research    = run_research_agent(market_data, tech_data, news_data)

    if research["status"] == "error":
        print(f"  ✗ Research failed: {research['error_message']}")
        return False

    # --- Print Key Findings ---
    print(f"\n  KEY FINDINGS:")
    for i, finding in enumerate(research["key_findings"], 1):
        print(f"    {i}. {finding}")

    # --- Agreement Areas ---
    if research["agreement_areas"]:
        print(f"\n  SIGNAL AGREEMENTS ({len(research['agreement_areas'])}):")
        for a in research["agreement_areas"]:
            print(f"    ✅ {a}")

    # --- Conflict Areas ---
    if research["conflict_areas"]:
        print(f"\n  SIGNAL CONFLICTS ({len(research['conflict_areas'])}):")
        for c in research["conflict_areas"]:
            print(f"    ⚠️  {c}")

    # --- Bullish Case ---
    print(f"\n  BULLISH CASE:")
    for b in research["bullish_case"][:4]:
        print(f"    🟢 {b}")

    # --- Bearish Case ---
    print(f"\n  BEARISH CASE:")
    for b in research["bearish_case"][:4]:
        print(f"    🔴 {b}")

    # --- Risks ---
    print(f"\n  RISKS IDENTIFIED ({len(research['risks'])}) — Level: {research['risk_level']}:")
    if research["risks"]:
        for r in research["risks"]:
            print(f"    ⚠️  [{r['name']}] {r['description']}")
    else:
        print(f"    ✓ No major risks identified.")

    # --- Overall Assessment ---
    print(f"\n  {'─'*50}")
    print(f"  OVERALL STANCE  : {research['overall_stance']}")
    print(f"  CONFIDENCE      : {research['confidence']}")
    print(f"  RISK LEVEL      : {research['risk_level']}")
    print(f"  {'─'*50}")
    print(f"\n  RESEARCH SUMMARY:")
    print(f"    {research['research_summary']}")

    print(f"\n  DISCLAIMER:")
    print(f"    {research['disclaimer']}")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RESEARCH & DECISION AGENT — TEST SUITE")
    print("="*60)

    results = []

    results.append(test_full_pipeline("RELIANCE.NS", "Reliance Industries"))
    results.append(test_full_pipeline("TCS.NS",      "Tata Consultancy Services"))
    results.append(test_full_pipeline("INFY.NS",     "Infosys"))

    print_separator("TEST SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print("  ✓ All tests passed! Ready for Phase 6 (LangGraph).")
    else:
        print("  ✗ Some tests failed — check errors above.")
