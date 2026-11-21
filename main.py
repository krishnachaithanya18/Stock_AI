"""
main.py — CLI Entry Point (Phase 8 — Updated)
===============================================
Run the full multi-agent analysis from the command line.

Usage:
    python main.py --symbol RELIANCE.NS
    python main.py --symbol TCS.NS --days 60
    python main.py --symbol INFY.NS --days 90 --news 15
"""

import argparse
from config import validate_config


def main():
    parser = argparse.ArgumentParser(
        description="Indian Stock Market Agentic AI — CLI Runner"
    )
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS",
                        help="NSE/BSE stock symbol (e.g. RELIANCE.NS)")
    parser.add_argument("--days",   type=int, default=90,
                        help="Days of historical data (default: 90)")
    parser.add_argument("--news",   type=int, default=10,
                        help="Max news articles (default: 10)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"   Indian Stock Market Agentic AI")
    print(f"   LangGraph Multi-Agent System")
    print(f"{'='*55}")
    print(f"   Symbol  : {args.symbol}")
    print(f"   History : {args.days} days")
    print(f"   News    : {args.news} articles")
    print(f"{'='*55}\n")

    validate_config()

    # Run the full LangGraph pipeline
    from graph.workflow import run_analysis
    result = run_analysis(args.symbol, args.days, args.news)

    # Print the final report
    report = result.get("report_data", {})
    if report and report.get("report_text"):
        print(report["report_text"])
    elif result.get("workflow_status") == "failed":
        print(f"\n[ERROR] {result.get('error_message', 'Analysis failed')}")
    else:
        print("\n[ERROR] No report generated.")

    print(f"\nWorkflow Status: {result.get('workflow_status', '?').upper()}")


if __name__ == "__main__":
    main()
