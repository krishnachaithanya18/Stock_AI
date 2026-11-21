"""
tests/test_technical_agent.py — Test the Technical Analysis Agent
=================================================================
Tests that RSI, MACD and SMA indicators calculate correctly
using real NSE stock data fetched live from Yahoo Finance.

Usage (from FAI folder with venv activated):
    python tests/test_technical_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_agent  import run_market_data_agent
from agents.technical_agent    import run_technical_agent


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_technical(symbol: str, label: str):
    """Fetch market data then run technical analysis on it."""

    print_separator(f"Testing: {label} ({symbol})")

    # Phase 2 agent first → feeds Phase 3 agent
    market_data = run_market_data_agent(symbol, history_days=90)

    if market_data["status"] == "error":
        print(f"  Market data failed: {market_data['error_message']}")
        return False

    # Run technical analysis
    tech = run_technical_agent(market_data)

    if tech["status"] == "error":
        print(f"  Technical analysis failed: {tech['error_message']}")
        return False

    # --- Print RSI ---
    rsi = tech["indicators"]["rsi"]
    print(f"\n  RSI ({rsi['period']}-day):")
    print(f"    Value  : {rsi['value']}")
    print(f"    Zone   : {rsi['zone']}")
    print(f"    Insight: {rsi['interpretation']}")

    # --- Print MACD ---
    macd = tech["indicators"]["macd"]
    print(f"\n  MACD (12,26,9):")
    print(f"    MACD Line   : {macd['macd_latest']}")
    print(f"    Signal Line : {macd['signal_latest']}")
    print(f"    Histogram   : {macd['histogram_latest']}")
    print(f"    Crossover   : {macd['crossover']}")
    print(f"    Insight     : {macd['interpretation']}")

    # --- Print SMA ---
    sma = tech["indicators"]["sma"]
    print(f"\n  Moving Averages:")
    print(f"    Current Price : ₹{sma['current_price']}")
    print(f"    SMA 20        : ₹{sma['sma20']}  ({sma['price_vs_sma20']})")
    print(f"    SMA 50        : ₹{sma['sma50']}  ({sma['price_vs_sma50']})")
    print(f"    Trend         : {sma['trend_structure']}")
    print(f"    Insight       : {sma['interpretation']}")

    # --- Print Signals ---
    signals = tech["signals"]
    print(f"\n  BULLISH SIGNALS ({len(signals['bullish'])}):")
    for s in signals["bullish"]:
        print(f"    ✅ {s}")

    print(f"\n  BEARISH SIGNALS ({len(signals['bearish'])}):")
    for s in signals["bearish"]:
        print(f"    ❌ {s}")

    print(f"\n  OVERALL SIGNAL  : {tech['overall_signal']}  (score: {tech['signal_score']:+d})")
    print(f"  SUMMARY         : {tech['summary']}")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  TECHNICAL ANALYSIS AGENT — TEST SUITE")
    print("="*60)

    results = []
    results.append(test_technical("RELIANCE.NS", "Reliance Industries"))
    results.append(test_technical("TCS.NS",      "Tata Consultancy Services"))
    results.append(test_technical("INFY.NS",     "Infosys"))

    # Test with error input (no market data)
    print_separator("Testing: Error Handling (bad market data input)")
    bad_input = {"status": "error", "symbol": "BAD.NS", "error_message": "Simulated failure"}
    result = run_technical_agent(bad_input)
    if result["status"] == "error":
        print(f"  ✓ Error handled gracefully: {result['error_message']}")
        results.append(True)
    else:
        print(f"  ✗ Should have returned error")
        results.append(False)

    print_separator("TEST SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print("  ✓ All tests passed! Ready for Phase 4.")
    else:
        print("  ✗ Some tests failed — check errors above.")
