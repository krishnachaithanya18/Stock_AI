"""
graph/workflow.py — LangGraph Multi-Agent Workflow (Phase 6)
=============================================================
THIS IS THE HEART OF THE AGENTIC AI SYSTEM.

WHAT IS A LANGGRAPH GRAPH?
  LangGraph represents a multi-agent workflow as a directed graph:
    - NODES  = individual agents (each does one specific job)
    - EDGES  = the connections that pass data between agents
    - STATE  = a shared dictionary that all agents read from and write to

INFORMATION FLOW:
  START
    │
    ▼
  [market_data_node]   ← Agent 1: Fetch stock prices (yfinance)
    │
    ▼
  [technical_node]     ← Agent 2: Calculate RSI, MACD, SMA
    │
    ▼
  [news_node]          ← Agent 3: Fetch news + sentiment
    │
    ▼
  [research_node]      ← Agent 4: Synthesise, detect conflicts, assess risk
    │
    ▼
  [report_node]        ← Agent 5: Generate final research report
    │
    ▼
  END

WHAT IS GraphState?
  It's a Python TypedDict — a typed dictionary that acts as the
  shared memory/blackboard for all agents.
  Agent 1 writes market data → Agent 2 reads it and writes indicators
  → Agent 3 reads market data and writes news → Agent 4 reads from
  both Agent 2 and 3 → Agent 5 reads everything and writes the report.

WHY LANGGRAPH INSTEAD OF JUST FUNCTION CALLS?
  1. Explicit graph structure — you can visualise the agent topology
  2. State is managed automatically — no manual dict passing
  3. Conditional routing — if Agent 1 fails, go to error_node not Agent 2
  4. Extendable — add agents, loops, human-in-the-loop later
  5. Industry standard for production Agentic AI systems
"""

from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, START, END

# Import all our agents
from agents.market_data_agent import run_market_data_agent
from agents.technical_agent   import run_technical_agent
from agents.news_agent        import run_news_agent
from agents.research_agent    import run_research_agent


# =============================================================
# STEP 1: DEFINE SHARED GRAPH STATE
# =============================================================

class GraphState(TypedDict):
    """
    The shared state passed between every node in the graph.

    Think of this as the analyst's shared workbench —
    each agent picks up what it needs and puts its results back.

    Every field is Optional because at the start of the graph,
    only 'symbol' and 'history_days' are populated.
    As each node runs, it fills in more fields.

    TypedDict = a Python dict with declared types (like a schema).
    It doesn't enforce types at runtime but documents what each
    field holds and enables IDE autocompletion.
    """

    # --- Input (set at the start) ---
    symbol      : str            # e.g. "RELIANCE.NS"
    history_days: int            # e.g. 90

    # --- Agent 1 output: Market Data ---
    market_data : Optional[dict] # Full result from run_market_data_agent()

    # --- Agent 2 output: Technical Analysis ---
    tech_data   : Optional[dict] # Full result from run_technical_agent()

    # --- Agent 3 output: News Analysis ---
    news_data   : Optional[dict] # Full result from run_news_agent()

    # --- Agent 4 output: Research & Decision ---
    research_data: Optional[dict] # Full result from run_research_agent()

    # --- Agent 5 output: Final Report ---
    report_data  : Optional[dict] # Full result from run_report_agent()

    # --- Workflow metadata ---
    workflow_status : Optional[str]  # "running", "completed", "failed"
    error_message   : Optional[str]  # Set if any critical agent fails
    max_news_articles: Optional[int] # Number of news articles to fetch


# =============================================================
# STEP 2: DEFINE NODE FUNCTIONS
# =============================================================
# Each node function:
#   - Receives the current GraphState
#   - Performs its specific task
#   - Returns a DICT of only the fields it wants to UPDATE
#     (LangGraph merges this into the existing state automatically)

def market_data_node(state: GraphState) -> dict:
    """
    Node 1: Market Data Agent
    Reads: state["symbol"], state["history_days"]
    Writes: state["market_data"]
    """
    print(f"\n{'─'*50}")
    print(f"[Graph] ▶ Running Node 1: Market Data Agent")
    print(f"{'─'*50}")

    result = run_market_data_agent(
        symbol=state["symbol"],
        history_days=state.get("history_days", 90)
    )

    # Return ONLY the fields this node updates
    # LangGraph merges this dict into the existing state
    return {
        "market_data"    : result,
        "workflow_status": "running",
    }


def technical_node(state: GraphState) -> dict:
    """
    Node 2: Technical Analysis Agent
    Reads: state["market_data"]  (set by Node 1)
    Writes: state["tech_data"]
    """
    print(f"\n{'─'*50}")
    print(f"[Graph] ▶ Running Node 2: Technical Analysis Agent")
    print(f"{'─'*50}")

    result = run_technical_agent(
        market_data=state["market_data"]
    )

    return {"tech_data": result}


def news_node(state: GraphState) -> dict:
    """
    Node 3: News Analysis Agent
    Reads: state["market_data"]  (set by Node 1)
    Writes: state["news_data"]

    WHY USE market_data HERE (not tech_data)?
      The News Agent needs company name and symbol from market_data.
      It runs AFTER technical analysis but reads MARKET data, not technical data.
      Both Node 2 and Node 3 read from Node 1's output independently.
    """
    print(f"\n{'─'*50}")
    print(f"[Graph] ▶ Running Node 3: News Analysis Agent")
    print(f"{'─'*50}")

    result = run_news_agent(
        market_data=state["market_data"],
        max_articles=state.get("max_news_articles", 10)
    )

    return {"news_data": result}


def research_node(state: GraphState) -> dict:
    """
    Node 4: Research & Decision Agent
    Reads: state["market_data"], state["tech_data"], state["news_data"]
    Writes: state["research_data"]

    This node DEPENDS on Nodes 1, 2, and 3 all having completed.
    LangGraph guarantees this via edge ordering.
    """
    print(f"\n{'─'*50}")
    print(f"[Graph] ▶ Running Node 4: Research & Decision Agent")
    print(f"{'─'*50}")

    result = run_research_agent(
        market_data=state["market_data"],
        tech_data=state["tech_data"],
        news_data=state["news_data"],
    )

    return {"research_data": result}


def report_node(state: GraphState) -> dict:
    """
    Node 5: Report Generation Agent
    Reads: ALL previous state fields
    Writes: state["report_data"], state["workflow_status"]

    This is the terminal node — it generates the final output.
    It calls run_report_agent() which we build in Phase 7.
    For now, it creates a structured report without Gemini.
    """
    print(f"\n{'─'*50}")
    print(f"[Graph] ▶ Running Node 5: Report Agent")
    print(f"{'─'*50}")

    # Import here to avoid circular import at module load time
    # Phase 7 will implement the full Gemini-powered version
    from agents.report_agent import run_report_agent

    result = run_report_agent(
        market_data=state["market_data"],
        tech_data=state["tech_data"],
        news_data=state["news_data"],
        research_data=state["research_data"],
    )

    return {
        "report_data"    : result,
        "workflow_status": "completed",
    }


def error_node(state: GraphState) -> dict:
    """
    Error Node: Reached when market data fetch fails.
    Returns a minimal error state so the graph ends cleanly
    instead of crashing with a KeyError.

    WHY A SEPARATE ERROR NODE?
      In production Agentic AI, you never want the graph to
      silently crash. An explicit error node lets you:
        - Log the failure
        - Return a user-friendly error message
        - Potentially retry with different parameters
    """
    print(f"\n[Graph] ⚠ Error node reached for symbol: {state.get('symbol')}")
    symbol = state.get("symbol", "UNKNOWN")
    error  = state.get("market_data", {}).get("error_message", "Unknown error")

    return {
        "report_data": {
            "status"       : "error",
            "symbol"       : symbol,
            "error_message": error,
            "report_text"  : (
                f"Analysis could not be completed for '{symbol}'.\n"
                f"Reason: {error}\n\n"
                f"Please verify the stock symbol (NSE format: SYMBOL.NS, "
                f"BSE format: SYMBOL.BO) and try again."
            ),
            "sections"     : {},
        },
        "workflow_status": "failed",
        "error_message"  : error,
    }


# =============================================================
# STEP 3: CONDITIONAL ROUTING FUNCTION
# =============================================================

def route_after_market_data(state: GraphState) -> str:
    """
    Decides which node to run AFTER the market data node.

    This is a 'conditional edge' in LangGraph — the graph
    chooses the next node based on the current state.

    Returns:
        "technical_node" → if market data succeeded (proceed normally)
        "error_node"     → if market data failed (go to error handler)

    WHY CONDITIONAL ROUTING?
      If the stock symbol is wrong or Yahoo Finance is down,
      we cannot run technical or news analysis on empty data.
      Instead of crashing, we route to the error node gracefully.
    """
    market_data = state.get("market_data", {})

    if market_data.get("status") == "success":
        print(f"[Graph] ✓ Market data OK → routing to technical analysis")
        return "technical_node"
    else:
        print(f"[Graph] ✗ Market data failed → routing to error node")
        return "error_node"


# =============================================================
# STEP 4: BUILD AND COMPILE THE GRAPH
# =============================================================

def build_graph():
    """
    Assembles the LangGraph StateGraph with all nodes and edges.

    The graph structure:
        START → market_data_node
                     │
            ┌────────┴───────────┐
            │  (conditional)     │
            ▼                    ▼
      technical_node          error_node
            │                    │
            ▼                    ▼
       news_node                END
            │
            ▼
      research_node
            │
            ▼
       report_node
            │
            ▼
           END

    Returns:
        A compiled LangGraph that can be invoked with:
            graph.invoke({"symbol": "RELIANCE.NS", "history_days": 90})
    """

    # Create the graph with our GraphState schema
    workflow = StateGraph(GraphState)

    # --- Add all nodes ---
    # Each node is registered by name + its function
    workflow.add_node("market_data_node", market_data_node)
    workflow.add_node("technical_node",   technical_node)
    workflow.add_node("news_node",        news_node)
    workflow.add_node("research_node",    research_node)
    workflow.add_node("report_node",      report_node)
    workflow.add_node("error_node",       error_node)

    # --- Add edges ---
    # START → market_data_node (always starts here)
    workflow.add_edge(START, "market_data_node")

    # market_data_node → conditional (success → technical, failure → error)
    workflow.add_conditional_edges(
        "market_data_node",          # From this node
        route_after_market_data,     # Call this function to decide where to go
        {
            "technical_node": "technical_node",  # If function returns "technical_node"
            "error_node"    : "error_node",      # If function returns "error_node"
        }
    )

    # technical_node → news_node (always)
    workflow.add_edge("technical_node", "news_node")

    # news_node → research_node (always — even if news fails, research handles it)
    workflow.add_edge("news_node", "research_node")

    # research_node → report_node (always)
    workflow.add_edge("research_node", "report_node")

    # report_node → END (workflow complete)
    workflow.add_edge("report_node", END)

    # error_node → END (workflow ended with error)
    workflow.add_edge("error_node", END)

    # Compile the graph into an executable object
    compiled = workflow.compile()

    print("[Graph] ✓ LangGraph workflow compiled successfully")
    return compiled


# =============================================================
# STEP 5: CONVENIENCE RUNNER FUNCTION
# =============================================================

def run_analysis(symbol: str, history_days: int = 90,
                 max_news_articles: int = 10) -> dict:
    """
    Public entry point: run the complete multi-agent analysis.

    This function:
      1. Builds the graph
      2. Creates the initial state
      3. Invokes the graph (all 5 nodes run in sequence)
      4. Returns the final state dict

    Parameters:
        symbol           : NSE/BSE ticker, e.g. "RELIANCE.NS"
        history_days     : Days of historical data (default 90)
        max_news_articles: Max news articles to fetch (default 10)

    Returns:
        The final GraphState dict with all agent outputs filled in.
        Access results via:
            result["market_data"]   → price data
            result["tech_data"]     → RSI, MACD, SMA
            result["news_data"]     → news sentiment
            result["research_data"] → research assessment
            result["report_data"]   → final report
    """
    print(f"\n{'='*55}")
    print(f"  LangGraph Multi-Agent Analysis Starting")
    print(f"  Symbol: {symbol}")
    print(f"{'='*55}")

    # Build the compiled graph
    graph = build_graph()

    # Initial state: only the inputs are set; all outputs start as None
    initial_state = {
        "symbol"           : symbol,
        "history_days"     : history_days,
        "max_news_articles": max_news_articles,
        "market_data"      : None,
        "tech_data"        : None,
        "news_data"        : None,
        "research_data"    : None,
        "report_data"      : None,
        "workflow_status"  : "starting",
        "error_message"    : None,
    }

    # Run the graph — this executes all nodes in order
    # .invoke() runs synchronously and returns the final state
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*55}")
    print(f"  Workflow Status : {final_state.get('workflow_status', '?').upper()}")
    print(f"{'='*55}\n")

    return final_state
