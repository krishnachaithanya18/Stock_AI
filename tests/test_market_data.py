"""
tests/test_market_data.py — Test the Market Data Agent
=======================================================
Run this file directly to test whether the Market Data Agent
is fetching real data correctly.

Usage (from the FAI folder with venv activated):
    python tests/test_market_data.py

WHAT WE TEST:
    1. Valid NSE symbol  → RELIANCE.NS
    2. Valid NSE symbol  → TCS.NS
    3. Valid NSE symbol  → INFY.NS
    4. Invalid symbol    → FAKESYMBOL.NS (should return error, not crash)
"""

import sys
import os

# Add the project root to Python path so we can import our modules
# This is needed because we're running the file from the tests/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.market_data_agent import run_market_data_agent


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_stock(symbol: str, label: str):
    """Test a single stock and print formatted results."""
    print_separator(f"Testing: {label} ({symbol})")

    result = run_market_data_agent(symbol, history_days=90)

    # --- Check status ---
    if result["status"] == "error":
        print(f"  STATUS  : ERROR ✗")
        print(f"  MESSAGE : {result['error_message']}")
        return False

    print(f"  STATUS  : SUCCESS ✓")
    print(f"  COMPANY : {result['company_name']}")
    print(f"  PRICE   : {result['current_price_fmt']}")
    print(f"  SECTOR  : {result['market_info']['sector']}")
    print(f"  MARKET CAP  : {result['market_info']['market_cap_fmt']}")
    print(f"  P/E RATIO   : {result['market_info']['pe_ratio']}")
    print(f"  52W HIGH    : {result['market_info']['fifty_two_week_high']}")
    print(f"  52W LOW     : {result['market_info']['fifty_two_week_low']}")
    print(f"  BETA        : {result['market_info']['beta']}")

    print(f"\n  PRICE CHANGES:")
    pc = result["price_change"]
    print(f"    1-Day   : {pc.get('one_day_pct_fmt', 'N/A')}")
    print(f"    1-Week  : {pc.get('one_week_pct_fmt', 'N/A')}")
    print(f"    1-Month : {pc.get('one_month_pct_fmt', 'N/A')}")

    print(f"\n  VOLUME:")
    vol = result["volume"]
    print(f"    Latest  : {vol.get('volume_formatted', 'N/A')}")
    print(f"    30D Avg : {vol.get('avg_volume_30d', 'N/A'):,.0f}" if vol.get('avg_volume_30d') else "    30D Avg : N/A")

    print(f"\n  DATA SOURCE:")
    ds = result["data_source"]
    print(f"    Provider   : {ds.get('provider')}")
    print(f"    Data Type  : {ds.get('data_type')}")
    print(f"    Fetched At : {ds.get('fetched_at')}")
    print(f"    Period     : {ds.get('data_from')} to {ds.get('data_to')}")
    print(f"    Records    : {ds.get('total_records')} trading days")

    # Verify the DataFrame is present (needed by Technical Analysis Agent)
    df = result["historical_df"]
    if df is not None and not df.empty:
        print(f"\n  HISTORICAL DATA (last 3 rows):")
        print(df[["Open", "High", "Low", "Close", "Volume"]].tail(3).to_string())
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MARKET DATA AGENT — TEST SUITE")
    print("="*60)
    
    results = []

    # Test 1: Reliance Industries (large cap)
    results.append(test_stock("RELIANCE.NS", "Reliance Industries"))

    # Test 2: TCS (IT sector)
    results.append(test_stock("TCS.NS", "Tata Consultancy Services"))

    # Test 3: Infosys (IT sector)
    results.append(test_stock("INFY.NS", "Infosys"))

    # Test 4: Invalid symbol (should NOT crash — should return error dict)
    print_separator("Testing: Invalid Symbol (Error Handling)")
    result = run_market_data_agent("FAKESYMBOL.NS")
    if result["status"] == "error":
        print(f"  ERROR HANDLING : ✓ Returned error gracefully (did not crash)")
        print(f"  MESSAGE        : {result['error_message']}")
        results.append(True)
    else:
        print(f"  ERROR HANDLING : ✗ Should have returned error for invalid symbol")
        results.append(False)

    # Summary
    print_separator("TEST SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"  Passed : {passed}/{total}")
    if passed == total:
        print("  ✓ All tests passed! Ready for Phase 3.")
    else:
        print("  ✗ Some tests failed. Check the errors above.")
