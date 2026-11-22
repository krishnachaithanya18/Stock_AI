"""
graph/workflow.py — Multi-Agent LangGraph Workflow (6-Agent Architecture)
========================================================================
ORCHESTRATION PIPELINE:
  START
    │
    ▼
  [market_data_node]     ← Agent A: Stock Data Agent (Quotes, OHLCV, Volume)
    │
    ├── (if error) ──────────────┐
    │                            ▼
    │ (if success)          [error_node] ──► END
    ▼
  [technical_node]       ← Agent B: Technical Analysis Agent (RSI, MACD, SMA 20/50/200, BB, S/R)
    │
    ▼
  [fundamental_node]     ← Agent C: Fundamental Analysis Agent (P/E, P/B, EPS, ROE, Debt/Equity)
    │
    ▼
  [news_node]            ← Agent D: Market Sentiment Agent (News Sentiment & Catalysts)
    │
    ▼
  [risk_node]            ← Agent E: Risk Analysis Agent (Volatility, Drawdowns, VaR, Beta)
    │
    ▼
  [decision_node]        ← Agent F: Final Decision Agent (Verdict, Return/Downside, Holding Period)
    │
    ▼
  [report_node]          ← Master Consolidation: Final Report Generation
    │
    ▼
   END
"""

from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, START, END

# Import specialized agents
from agents.market_data_agent import run_market_data_agent
from agents.technical_agent import run_technical_agent
from agents.fundamental_agent import run_fundamental_agent
from agents.news_agent import run_news_agent
from agents.risk_agent import run_risk_agent
from agents.decision_agent import run_decision_agent
from agents.report_agent import run_report_agent
from agents.research_agent import run_research_agent


class GraphState(TypedDict):
    """
    Shared State Schema passed across all agents in LangGraph.
    """
    symbol            : str
    history_days      : int
    exchange          : Optional[str]
    max_news_articles : Optional[int]

    # Agent Outputs
    market_data       : Optional[dict]  # Agent A: Stock Data
    tech_data         : Optional[dict]  # Agent B: Technical Analysis
    fund_data         : Optional[dict]  # Agent C: Fundamental Analysis
    news_data         : Optional[dict]  # Agent D: Market Sentiment
    risk_data         : Optional[dict]  # Agent E: Risk Analysis
    decision_data     : Optional[dict]  # Agent F: Final Verdict & Decision Setup
    research_data     : Optional[dict]  # Compatibility mirror of research findings
    report_data       : Optional[dict]  # Final consolidated institutional report

    workflow_status   : Optional[str]
    error_message     : Optional[str]


# =======================================================================
# NODE FUNCTIONS
# =======================================================================

def market_data_node(state: GraphState) -> dict:
    """Node 1: Stock Data Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 1: Stock Data Agent")
    print(f"{'─'*55}")

    result = run_market_data_agent(
        symbol=state["symbol"],
        history_days=state.get("history_days", 90),
        exchange=state.get("exchange", "NSE")
    )
    return {
        "market_data": result,
        "workflow_status": "running"
    }


def technical_node(state: GraphState) -> dict:
    """Node 2: Technical Analysis Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 2: Technical Analysis Agent")
    print(f"{'─'*55}")

    result = run_technical_agent(state["market_data"])
    return {"tech_data": result}


def fundamental_node(state: GraphState) -> dict:
    """Node 3: Fundamental Analysis Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 3: Fundamental Analysis Agent")
    print(f"{'─'*55}")

    result = run_fundamental_agent(state["market_data"])
    return {"fund_data": result}


def news_node(state: GraphState) -> dict:
    """Node 4: Market Sentiment Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 4: Market Sentiment Agent")
    print(f"{'─'*55}")

    result = run_news_agent(
        market_data=state["market_data"],
        max_articles=state.get("max_news_articles", 10)
    )
    return {"news_data": result}


def risk_node(state: GraphState) -> dict:
    """Node 5: Risk Analysis Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 5: Risk Analysis Agent")
    print(f"{'─'*55}")

    result = run_risk_agent(
        market_data=state["market_data"],
        tech_data=state["tech_data"],
        fund_data=state["fund_data"],
        news_data=state["news_data"]
    )
    return {"risk_data": result}


def decision_node(state: GraphState) -> dict:
    """Node 6: Final Decision Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 6: Final Decision Agent")
    print(f"{'─'*55}")

    decision_res = run_decision_agent(
        market_data=state["market_data"],
        tech_data=state["tech_data"],
        fund_data=state["fund_data"],
        news_data=state["news_data"],
        risk_data=state["risk_data"]
    )

    research_res = run_research_agent(
        market_data=state["market_data"],
        tech_data=state["tech_data"],
        news_data=state["news_data"],
        fund_data=state["fund_data"],
        risk_data=state["risk_data"]
    )

    return {
        "decision_data": decision_res,
        "research_data": research_res,
    }


def report_node(state: GraphState) -> dict:
    """Node 7: Report Consolidation Agent"""
    print(f"\n{'─'*55}")
    print(f"[Graph] ▶ Running Node 7: Report Agent")
    print(f"{'─'*55}")

    report_res = run_report_agent(
        market_data=state["market_data"],
        tech_data=state["tech_data"],
        news_data=state["news_data"],
        research_data=state["research_data"]
    )

    return {
        "report_data": report_res,
        "workflow_status": "completed"
    }


def error_node(state: GraphState) -> dict:
    """Error Node: Handles data retrieval failures gracefully"""
    symbol = state.get("symbol", "UNKNOWN")
    error = state.get("market_data", {}).get("error_message", "Unknown retrieval error")
    print(f"\n[Graph] ⚠ Error node reached for symbol: {symbol} — {error}")

    err_report = {
        "status": "error",
        "symbol": symbol,
        "error_message": error,
        "report_text": (
            f"Analysis could not be completed for '{symbol}'.\n"
            f"Reason: {error}\n\n"
            f"Please verify the stock symbol (e.g. RELIANCE, TCS, INFY) or BSE code (e.g. 500325)."
        ),
        "sections": {}
    }

    return {
        "decision_data": {"verdict": "AVOID", "status": "error", "error_message": error},
        "report_data": err_report,
        "workflow_status": "failed",
        "error_message": error,
    }


# =======================================================================
# CONDITIONAL ROUTING & GRAPH COMPILATION
# =======================================================================

def route_after_market_data(state: GraphState) -> str:
    market_data = state.get("market_data", {})
    if market_data.get("status") == "success":
        return "technical_node"
    return "error_node"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("market_data_node", market_data_node)
    workflow.add_node("technical_node",   technical_node)
    workflow.add_node("fundamental_node", fundamental_node)
    workflow.add_node("news_node",        news_node)
    workflow.add_node("risk_node",        risk_node)
    workflow.add_node("decision_node",    decision_node)
    workflow.add_node("report_node",      report_node)
    workflow.add_node("error_node",       error_node)

    # Edges
    workflow.add_edge(START, "market_data_node")

    workflow.add_conditional_edges(
        "market_data_node",
        route_after_market_data,
        {
            "technical_node": "technical_node",
            "error_node": "error_node",
        }
    )

    workflow.add_edge("technical_node", "fundamental_node")
    workflow.add_edge("fundamental_node", "news_node")
    workflow.add_edge("news_node", "risk_node")
    workflow.add_edge("risk_node", "decision_node")
    workflow.add_edge("decision_node", "report_node")
    workflow.add_edge("report_node", END)
    workflow.add_edge("error_node", END)

    return workflow.compile()


def run_analysis(symbol: str, history_days: int = 90,
                 max_news_articles: int = 10, exchange: str = "NSE") -> dict:
    """
    Public API: Executes the complete multi-agent LangGraph analysis.
    """
    print(f"\n{'='*55}")
    print(f"  LangGraph Multi-Agent Analysis Starting")
    print(f"  Symbol: {symbol} | Exchange: {exchange}")
    print(f"{'='*55}")

    graph = build_graph()

    initial_state = {
        "symbol": symbol,
        "history_days": history_days,
        "exchange": exchange,
        "max_news_articles": max_news_articles,
        "market_data": None,
        "tech_data": None,
        "fund_data": None,
        "news_data": None,
        "risk_data": None,
        "decision_data": None,
        "research_data": None,
        "report_data": None,
        "workflow_status": "starting",
        "error_message": None,
    }

    final_state = graph.invoke(initial_state)

    print(f"\n{'='*55}")
    print(f"  Workflow Status : {final_state.get('workflow_status', '?').upper()}")
    print(f"{'='*55}\n")

    return final_state
