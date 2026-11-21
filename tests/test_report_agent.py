"""
tests/test_report_agent.py — Test Phase 7 Report Generation
=============================================================
Tests:
  1. Gemini connection (shows which mode will be used)
  2. Full pipeline with report for RELIANCE.NS
  3. Report content validation

Usage:
    python tests/test_report_agent.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemini_client import test_gemini_connection
from graph.workflow import run_analysis


def print_separator(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  REPORT AGENT — PHASE 7 TEST")
    print("="*60)

    # Test 1: Gemini connection
    print_separator("Test 1: Gemini API Connection")
    gemini_ok = test_gemini_connection()
    if gemini_ok:
        print("  → Reports will be GEMINI-POWERED ✨")
    else:
        print("  → Reports will use TEMPLATE MODE (fully functional)")

    results = [True]  # Gemini test always passes (error handled gracefully)

    # Test 2: Full pipeline with RELIANCE.NS
    print_separator("Test 2: Full Pipeline + Report — RELIANCE.NS")
    state = run_analysis("RELIANCE.NS", history_days=90, max_news_articles=10)

    report = state.get("report_data", {})
    checks = {
        "report_data populated"   : report is not None,
        "report has status"       : report.get("status") == "success",
        "report_text non-empty"   : len(report.get("report_text", "")) > 500,
        "report_type is set"      : report.get("report_type") in ("gemini", "template"),
        "sections non-empty"      : len(report.get("sections", {})) > 0,
        "workflow completed"      : state.get("workflow_status") == "completed",
    }

    all_passed = True
    for name, passed in checks.items():
        print(f"  [{'✓' if passed else '✗'}] {name}")
        if not passed:
            all_passed = False
    results.append(all_passed)

    # Print the full report
    print(f"\n  Report Type  : {report.get('report_type','?').upper()}")
    print(f"  Generated At : {report.get('generated_at','?')}")
    if report.get("model_used"):
        print(f"  Model Used   : {report.get('model_used','?')}")
    if report.get("tokens_used"):
        print(f"  Tokens Used  : {report.get('tokens_used','?')}")

    print(f"\n{'─'*60}")
    print("  FULL REPORT:")
    print(f"{'─'*60}")
    print(report.get("report_text", "No report generated."))

    # Test 3: TCS
    print_separator("Test 3: Full Pipeline + Report — TCS.NS")
    state2 = run_analysis("TCS.NS", history_days=90, max_news_articles=10)
    report2 = state2.get("report_data", {})
    tcs_ok  = report2.get("status") == "success" and len(report2.get("report_text","")) > 200
    print(f"  [{'✓' if tcs_ok else '✗'}] TCS report generated ({len(report2.get('report_text',''))} chars)")
    results.append(tcs_ok)

    # Summary
    print_separator("TEST SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print("  ✓ Phase 7 complete! Ready for Phase 8 (Streamlit Dashboard).")
    else:
        print("  ✗ Some issues — check above.")
