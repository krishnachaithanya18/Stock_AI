"""
main.py — CLI Entry Point for Indian Stock Market Agentic AI
=============================================================
Run the full 6-agent LangGraph analysis from the terminal.

Usage:
    python main.py --symbol RELIANCE
    python main.py --symbol 500325 --exchange BSE
    python main.py --symbol TCS --days 60 --news 10
"""

import argparse
from config import validate_config


def main():
    parser = argparse.ArgumentParser(
        description="Indian Stock Market Agentic AI — Multi-Agent Institutional CLI Runner"
    )
    parser.add_argument("--symbol", type=str, default="RELIANCE",
                        help="NSE/BSE stock symbol or BSE code (e.g. RELIANCE, TCS, 500325)")
    parser.add_argument("--exchange", type=str, default="NSE", choices=["NSE", "BSE", "AUTO"],
                        help="Target exchange (default: NSE)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of historical data (default: 90)")
    parser.add_argument("--news", type=int, default=10,
                        help="Maximum news articles to evaluate (default: 10)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"   INDIAN STOCK MARKET AGENTIC AI — 6-AGENT PLATFORM")
    print(f"{'='*65}")
    print(f"   Target Symbol : {args.symbol}")
    print(f"   Exchange      : {args.exchange}")
    print(f"   History Range : {args.days} trading days")
    print(f"   News Sample   : {args.news} articles")
    print(f"{'='*65}\n")

    validate_config()

    from graph.workflow import run_analysis
    result = run_analysis(args.symbol, args.days, args.news, exchange=args.exchange)

    if result.get("workflow_status") == "completed":
        dec = result.get("decision_data", {})
        trade = dec.get("trade_setup", {})
        print(f"\n{'='*65}")
        print(f"   FINAL INVESTMENT VERDICT: {dec.get('verdict', 'HOLD')}")
        print(f"{'='*65}")
        print(f"   • Recommended Horizon    : {dec.get('holding_period', 'N/A')}")
        print(f"   • Probability of Profit  : {dec.get('probability_positive_return', 'N/A')}%")
        print(f"   • Model Confidence Score : {dec.get('model_confidence', 'N/A')}% ({dec.get('confidence_rating', '')})")
        print(f"   • Target Price (Upside)  : {trade.get('target_price_fmt', 'N/A')} ({trade.get('expected_return_fmt', 'N/A')})")
        print(f"   • Stop-Loss (Downside)   : {trade.get('stop_loss_fmt', 'N/A')} ({trade.get('potential_downside_fmt', 'N/A')})")
        print(f"   • Risk-to-Reward Ratio   : {trade.get('risk_reward_ratio', 'N/A')}")
        print(f"{'='*65}\n")

        rep = result.get("report_data", {})
        if rep and rep.get("report_text"):
            print(rep["report_text"])
    else:
        print(f"\n[ERROR] Analysis failed: {result.get('error_message', 'Unknown failure')}")


if __name__ == "__main__":
    main()
