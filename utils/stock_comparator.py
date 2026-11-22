"""
utils/stock_comparator.py — Multi-Stock Head-to-Head Comparison Engine
======================================================================
Compares 2 or 3 Indian stocks across:
- Valuation & Profitability (P/E, P/B, ROE, Market Cap, Debt/Equity)
- Technical Momentum (RSI, Moving Average Alignment, Trend)
- Risk & Drawdowns (Volatility, Max Drawdown, Beta)
- Multi-Agent AI Recommendation
Produces side-by-side metrics and an objective comparative conclusion.
"""

from typing import List, Dict
import pandas as pd
from graph.workflow import run_analysis
from utils.helpers import safe_round, format_percentage, format_currency


def compare_stocks(symbols: List[str], history_days: int = 90) -> Dict:
    """
    Executes full multi-agent analysis for up to 3 stocks and compiles a comparative matrix.
    """
    results = []
    for sym in symbols[:3]:
        clean_sym = sym.strip().upper()
        if clean_sym:
            res = run_analysis(clean_sym, history_days=history_days)
            if res.get("workflow_status") == "completed":
                results.append(res)

    if not results:
        return {"status": "error", "message": "Failed to analyze any of the provided symbols."}

    # Build comparison records
    table_rows = []
    for r in results:
        m = r["market_data"]
        mi = m.get("market_info", {})
        pc = m.get("price_change", {})
        t = r["tech_data"]
        f = r["fund_data"]
        rk = r["risk_data"]
        d = r["decision_data"]

        table_rows.append({
            "Company": m.get("company_name", m.get("symbol")),
            "Symbol": m.get("symbol"),
            "Exchange": mi.get("exchange", "NSE"),
            "Current Price": m.get("current_price_fmt", "N/A"),
            "1-Day Change": pc.get("one_day_pct_fmt", "N/A"),
            "1-Month Change": pc.get("one_month_pct_fmt", "N/A"),
            "Market Cap": mi.get("market_cap_fmt", "N/A"),
            "P/E Ratio": f"{mi.get('pe_ratio', 'N/A')}x" if mi.get('pe_ratio') else "N/A",
            "P/B Ratio": f"{mi.get('pb_ratio', 'N/A')}x" if mi.get('pb_ratio') else "N/A",
            "ROE": f"{mi.get('return_on_equity', 'N/A')}%" if mi.get('return_on_equity') else "N/A",
            "Debt-to-Equity": f"{mi.get('debt_to_equity', 'N/A')}" if mi.get('debt_to_equity') else "N/A",
            "RSI (14)": str(t.get("indicators", {}).get("rsi", {}).get("value", "N/A")),
            "Trend": t.get("indicators", {}).get("trend", {}).get("trend", "N/A"),
            "Annualized Volatility": f"{rk.get('volatility_annualized', 'N/A')}%" if rk.get('volatility_annualized') else "N/A",
            "Beta": str(rk.get("beta", "N/A")),
            "Risk Profile": rk.get("risk_level", "N/A"),
            "AI Verdict": d.get("verdict", "HOLD"),
            "P(Positive Return)": f"{d.get('probability_positive_return', 'N/A')}%",
            "Expected Target": d.get("trade_setup", {}).get("target_price_fmt", "N/A"),
            "_raw_composite": d.get("composite_score", 0),
            "_raw_fund": f.get("fundamental_score", 0),
            "_raw_tech": t.get("signal_score", 0),
        })

    # Determine which stock is stronger
    best_overall = max(table_rows, key=lambda x: x["_raw_composite"])
    best_fund = max(table_rows, key=lambda x: x["_raw_fund"])
    best_tech = max(table_rows, key=lambda x: x["_raw_tech"])

    reasoning = (
        f"Among the compared stocks, **{best_overall['Company']}** demonstrates the strongest overall setup "
        f"with an AI Verdict of **{best_overall['AI Verdict']}** (Composite Score: {best_overall['_raw_composite']:+0.1f}). "
        f"Fundamentally, **{best_fund['Company']}** ranks highest (score {best_fund['_raw_fund']:+d}/10), "
        f"while **{best_tech['Company']}** displays the most favorable technical momentum (score {best_tech['_raw_tech']:+d}/10)."
    )

    return {
        "status": "success",
        "stocks_analyzed": len(results),
        "table_rows": table_rows,
        "best_overall": best_overall["Company"],
        "best_overall_symbol": best_overall["Symbol"],
        "comparative_reasoning": reasoning,
    }
