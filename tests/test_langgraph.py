"""
tests/test_langgraph.py — Test the Complete LangGraph Workflow
==============================================================
Tests the full 5-node graph running end-to-end for all 3 stocks
AND tests the conditional error routing for an invalid symbol.

Usage (from FAI folder with venv activated):
    python tests/test_langgraph.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import run_analysis, build_graph


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_full_graph(symbol: str, label: str):
    """Run the complete LangGraph workflow and verify all state fields."""

    print_separator(f"Graph Test: {label} ({symbol})")

    # This single call runs ALL 5 agents through the graph
    final_state = run_analysis(symbol, history_days=90, max_news_articles=10)

    # --- Verify all state fields are populated ---
    checks = {
        "market_data populated"   : final_state.get("market_data") is not None,
        "tech_data populated"     : final_state.get("tech_data") is not None,
        "news_data populated"     : final_state.get("news_data") is not None,
        "research_data populated" : final_state.get("research_data") is not None,
        "report_data populated"   : final_state.get("report_data") is not None,
        "workflow completed"      : final_state.get("workflow_status") == "completed",
        "no critical error"       : final_state.get("error_message") is None,
    }

    all_passed = True
    for check_name, passed in checks.items():
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {check_name}")
        if not passed:
            all_passed = False

    # --- Print report preview ---
    report = final_state.get("report_data", {})
    if report and report.get("report_text"):
        print(f"\n  REPORT PREVIEW (first 600 chars):")
        print("  " + "─" * 50)
        preview = report["report_text"][:600].replace("\n", "\n  ")
        print(f"  {preview}")
        print("  " + "─" * 50)

    # --- Print final stance ---
    research = final_state.get("research_data", {})
    if research:
        print(f"\n  FINAL STANCE  : {research.get('overall_stance', '?')}")
        print(f"  CONFIDENCE    : {research.get('confidence', '?')}")
        print(f"  RISK LEVEL    : {research.get('risk_level', '?')}")

    return all_passed


def test_error_routing():
    """Test that the error_node is correctly triggered for invalid symbols."""

    print_separator("Graph Test: Conditional Error Routing (FAKESYM.NS)")

    final_state = run_analysis("FAKESYM.NS", history_days=90)

    # For an invalid symbol, the graph should:
    # 1. Run market_data_node → get error
    # 2. Route to error_node (NOT technical_node)
    # 3. Set workflow_status = "failed"
    # 4. Set error_message
    # 5. NOT crash

    checks = {
        "workflow_status = 'failed'" : final_state.get("workflow_status") == "failed",
        "error_message set"          : final_state.get("error_message") is not None,
        "tech_data is None"          : final_state.get("tech_data") is None,  # Skipped
        "report_data has error"      : (
            final_state.get("report_data", {}).get("status") == "error"
        ),
    }

    all_passed = True
    for check_name, passed in checks.items():
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {check_name}")
        if not passed:
            all_passed = False

    if final_state.get("error_message"):
        print(f"\n  Error Message: {final_state['error_message'][:100]}")

    return all_passed


def test_graph_structure():
    """Verify the graph can be built and shows its node structure."""

    print_separator("Graph Structure Verification")

    graph = build_graph()

    print(f"  Graph type    : {type(graph).__name__}")

    # LangGraph compiled graphs expose the graph structure
    try:
        # Get the underlying graph to list nodes
        nodes = list(graph.get_graph().nodes.keys())
        print(f"  Nodes ({len(nodes)}): {', '.join(nodes)}")

        edges = graph.get_graph().edges
        print(f"  Edges ({len(edges)}):")
        for edge in edges:
            print(f"    {edge.source} → {edge.target}")
    except Exception as e:
        print(f"  (Graph structure details not accessible: {e})")
        print(f"  Graph compiled successfully (this is all that's required)")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LANGGRAPH WORKFLOW — FULL INTEGRATION TEST")
    print("="*60)

    results = []

    # Test graph structure first (no network needed)
    results.append(test_graph_structure())

    # Test full pipeline for all 3 stocks
    results.append(test_full_graph("RELIANCE.NS", "Reliance Industries"))
    results.append(test_full_graph("TCS.NS",      "Tata Consultancy Services"))
    results.append(test_full_graph("INFY.NS",     "Infosys"))

    # Test error routing
    results.append(test_error_routing())

    print_separator("TEST SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"  Passed: {passed}/{total}")
    if passed == total:
        print("  ✓ LangGraph integration complete! Ready for Phase 7.")
    else:
        print("  ✗ Some tests failed — check errors above.")
