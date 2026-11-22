"""
app.py — Indian Stock Market Agentic AI (Institutional Terminal Dashboard)
===========================================================================
Multi-Agent Stock Research and Analysis Platform for Indian Markets (NSE & BSE)
Powered by LangGraph, Google Gemini, and yfinance.
"""

import sys
import os
import streamlit as st
import pandas as pd
import yfinance as yf

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import APP_TITLE, APP_VERSION, validate_config
from graph.workflow import run_analysis
from utils.market_overview import fetch_market_indices, get_market_status
from utils.stock_search import resolve_stock_symbol, search_indian_stocks, get_popular_stock_chips
from utils.charting import create_price_and_volume_chart, create_rsi_chart, create_macd_chart
from utils.watchlist_manager import (
    load_watchlist, add_to_watchlist, remove_from_watchlist,
    get_watchlist_live_data, add_price_alert, check_alerts_status, remove_alert
)
from utils.stock_comparator import compare_stocks
from utils.helpers import format_currency, format_percentage, safe_round


# =======================================================================
# STREAMLIT PAGE CONFIGURATION
# =======================================================================
st.set_page_config(
    page_title=f"{APP_TITLE} | Financial Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =======================================================================
# DARK FINANCIAL TERMINAL STYLING (CSS)
# =======================================================================
st.markdown("""
<style>
/* Base Dark Financial Palette */
:root {
    --bg-main: #0B0E14;
    --bg-card: #121824;
    --bg-hover: #1A2234;
    --border-color: #232D42;
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --accent-blue: #38BDF8;
    --bull-green: #10B981;
    --bear-red: #EF4444;
    --neutral-amber: #F59E0B;
}

/* Background */
.stApp {
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}

/* Top Market Overview Ribbon */
.market-ribbon {
    display: flex;
    gap: 16px;
    align-items: center;
    background: #0E131E;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 18px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.market-index-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.88rem;
    padding-right: 18px;
    border-right: 1px solid #1E293B;
}
.market-index-name {
    font-weight: 700;
    color: var(--text-secondary);
    letter-spacing: 0.3px;
}
.market-index-val {
    font-weight: 700;
    color: #F8FAFC;
}
.market-status-pill {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-left: auto;
}

/* Hero Terminal Box */
.terminal-hero {
    background: linear-gradient(180deg, #121824 0%, #0E131E 100%);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 24px 28px 18px 28px;
    margin-bottom: 20px;
}
.terminal-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.4px;
    margin: 0 0 4px 0;
}
.terminal-subtitle {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-bottom: 14px;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #121824;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    white-space: normal !important;
    word-break: break-word !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #94A3B8 !important;
}

/* Prominent Verdict Banners */
.verdict-banner {
    border-radius: 8px;
    padding: 22px 28px;
    margin: 16px 0 20px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid;
}
.verdict-strong-buy {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
    border-color: #10B981;
}
.verdict-buy {
    background: linear-gradient(90deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.05) 100%);
    border-color: #34D399;
}
.verdict-hold {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.05) 100%);
    border-color: #F59E0B;
}
.verdict-avoid {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
    border-color: #EF4444;
}

.verdict-text-large {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    margin: 0;
}

/* Matrix Cards */
.matrix-card {
    background: #121824;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 18px;
    height: 100%;
}
.matrix-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1E293B;
}

/* Disclaimer text */
.institutional-disclaimer {
    background-color: #0F172A;
    border-left: 3px solid #38BDF8;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.78rem;
    color: #94A3B8;
    line-height: 1.5;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)


# =======================================================================
# CACHED DATA FETCHING
# =======================================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_cached_market_overview():
    indices = fetch_market_indices()
    status = get_market_status()
    return indices, status


@st.cache_data(ttl=300, show_spinner=False)
def run_cached_stock_analysis(symbol: str, history_days: int, exchange: str):
    return run_analysis(symbol=symbol, history_days=history_days, exchange=exchange)


# =======================================================================
# TOP MARKET OVERVIEW BAR
# =======================================================================
def render_market_overview_ribbon():
    indices, status = get_cached_market_overview()

    items_html = []
    for idx in indices:
        val = idx["value_fmt"]
        chg = idx["change_pts_fmt"]
        pct = idx["change_pct_fmt"]
        is_up = idx["direction"] == "up"
        color = "#10B981" if is_up else ("#EF4444" if idx["direction"] == "down" else "#94A3B8")
        sign = "+" if is_up and not chg.startswith("+") else ""

        items_html.append(
            f'<div class="market-index-item">'
            f'<span class="market-index-name">{idx["name"]}</span>'
            f'<span class="market-index-val">{val}</span>'
            f'<span style="color:{color};font-weight:600;">{sign}{chg} ({pct})</span>'
            f'</div>'
        )

    # Market status badge
    badge_bg = "rgba(16, 185, 129, 0.2)" if status["is_open"] else "rgba(239, 68, 68, 0.2)"
    badge_color = "#10B981" if status["is_open"] else "#EF4444"
    status_html = (
        f'<div class="market-status-pill" style="background:{badge_bg};color:{badge_color};">'
        f'{status["badge"]} &nbsp;·&nbsp; {status["current_ist"]}'
        f'</div>'
    )

    ribbon_html = f'<div class="market-ribbon">{"".join(items_html)}{status_html}</div>'
    st.markdown(ribbon_html, unsafe_allow_html=True)


# =======================================================================
# HERO SEARCH COMPONENT
# =======================================================================
def render_search_hero():
    if "current_symbol" not in st.session_state:
        st.session_state["current_symbol"] = "RELIANCE"
    if "selected_exchange" not in st.session_state:
        st.session_state["selected_exchange"] = "NSE"

    st.markdown("""
    <div class="terminal-hero">
        <div class="terminal-title">INDIAN STOCK MARKET AGENTIC AI</div>
        <div class="terminal-subtitle">
            Autonomous Institutional Research Engine · LangGraph Multi-Agent Orchestration · Google Gemini · NSE & BSE Universe
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_exch, col_btn = st.columns([5, 2, 2])

    with col_input:
        query_input = st.text_input(
            label="Search Indian Stocks",
            value=st.session_state["current_symbol"],
            placeholder="Search company name, NSE symbol (TCS, INFY), or BSE code (500325)...",
            label_visibility="collapsed",
            key="stock_search_input"
        ).strip()

    with col_exch:
        exchange_choice = st.selectbox(
            "Exchange",
            options=["NSE", "BSE", "AUTO"],
            index=0 if st.session_state["selected_exchange"] == "NSE" else 1,
            label_visibility="collapsed"
        )
        st.session_state["selected_exchange"] = exchange_choice

    with col_btn:
        execute_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Search Autocomplete / Suggestions
    if query_input and len(query_input) >= 2:
        suggestions = search_indian_stocks(query_input, limit=4)
        if suggestions:
            sug_cols = st.columns(len(suggestions))
            for i, sug in enumerate(suggestions):
                with sug_cols[i]:
                    label = f"{sug['symbol']} · {sug['name'][:18]}..."
                    if st.button(label, key=f"sug_{sug['symbol']}_{i}", use_container_width=True):
                        st.session_state["current_symbol"] = sug["symbol"]
                        st.session_state["stock_search_input"] = sug["symbol"]
                        st.session_state.pop("analysis_results", None)
                        st.rerun()

    # Popular Stocks Quick Chips (2 rows of 5 for optimal button spacing)
    st.markdown("<div style='font-size:0.8rem;color:#64748B;margin:8px 0 4px 0;'>Popular Benchmarks & Leaders:</div>", unsafe_allow_html=True)
    chips = get_popular_stock_chips()
    row1 = chips[:5]
    row2 = chips[5:10]

    for row_chips in [row1, row2]:
        cols = st.columns(len(row_chips))
        for i, (sym, short_name, bse_c) in enumerate(row_chips):
            with cols[i]:
                if st.button(short_name, key=f"chip_{sym}", use_container_width=True):
                    st.session_state["current_symbol"] = sym
                    st.session_state["stock_search_input"] = sym
                    st.session_state.pop("analysis_results", None)
                    st.rerun()

    st.session_state["current_symbol"] = query_input
    return query_input, exchange_choice, execute_analysis


# =======================================================================
# VERDICT CARD COMPONENT ("Buy or Not?")
# =======================================================================
def render_verdict_section(decision_data: dict, market_data: dict):
    verdict = decision_data.get("verdict", "HOLD")
    verdict_cls = f"verdict-{verdict.lower().replace(' ', '-')}"
    prob = decision_data.get("probability_positive_return", 50.0)
    horizon = decision_data.get("holding_period", "Medium term: 1–3 months")
    conf = decision_data.get("model_confidence", 70.0)
    trade = decision_data.get("trade_setup", {})

    st.markdown(f"""
    <div class="verdict-banner {verdict_cls}">
        <div>
            <div style="font-size:0.82rem;font-weight:700;color:#94A3B8;letter-spacing:1px;text-transform:uppercase;">
                AI Investment Recommendation
            </div>
            <div class="verdict-text-large" style="color:{decision_data.get('verdict_color', '#F59E0B')}">
                {verdict}
            </div>
            <div style="font-size:0.92rem;color:#E2E8F0;margin-top:4px;">
                {decision_data.get('verdict_desc', '')}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.82rem;color:#94A3B8;">Recommended Holding Horizon</div>
            <div style="font-size:1.15rem;font-weight:800;color:#38BDF8;">{horizon}</div>
            <div style="font-size:0.82rem;color:#64748B;margin-top:4px;">Composite Score: {decision_data.get('composite_score', 0):+0.1f} / 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Quantitative Trade Setup Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P(Positive Return)", f"{prob}%", help=decision_data.get("probability_methodology", ""))
    c2.metric("Target Price (Upside)", trade.get("target_price_fmt", "N/A"), delta=trade.get("expected_return_fmt"))
    c3.metric("Stop-Loss (Downside)", trade.get("stop_loss_fmt", "N/A"), delta=trade.get("potential_downside_fmt"), delta_color="inverse")
    c4.metric("Risk-to-Reward", trade.get("risk_reward_ratio", "1 : 2.0"))
    c5.metric("Model Confidence", f"{conf}% ({decision_data.get('confidence_rating', 'MODERATE')})")

    st.caption(f"Suggested Entry Range: **{trade.get('entry_range', 'N/A')}** · Current Price: **{market_data.get('current_price_fmt', 'N/A')}**")


# =======================================================================
# STOCK OVERVIEW HEADER
# =======================================================================
def render_stock_overview_header(market_data: dict, decision_data: dict):
    company = market_data.get("company_name", "Unknown")
    sym = market_data.get("symbol", "")
    price = market_data.get("current_price_fmt", "N/A")
    pc = market_data.get("price_change", {})
    mi = market_data.get("market_info", {})

    col_title, col_watch = st.columns([6, 2])
    with col_title:
        st.markdown(f"## {company} <span style='font-size:1.1rem;color:#64748B;'>({sym})</span>", unsafe_allow_html=True)
        st.caption(f"Exchange: {mi.get('exchange', 'NSE')} · Sector: {mi.get('sector', 'N/A')} · Source: Yahoo Finance (~15 min delay)")

    with col_watch:
        st.write("")
        verdict = decision_data.get("verdict", "HOLD")
        if st.button("⭐ Add to Watchlist", key="btn_add_watchlist", use_container_width=True):
            add_to_watchlist(sym, company, verdict)
            st.success(f"Added {sym} to Watchlist!")

    # Top Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Current Price", price, delta=pc.get("one_day_pct_fmt", "N/A"))
    m2.metric("1-Week Return", pc.get("one_week_pct_fmt", "N/A"))
    m3.metric("1-Month Return", pc.get("one_month_pct_fmt", "N/A"))
    m4.metric("Market Cap", mi.get("market_cap_fmt", "N/A"))
    m5.metric("52-Week High", f"₹{mi.get('fifty_two_week_high', 'N/A')}")
    m6.metric("52-Week Low", f"₹{mi.get('fifty_two_week_low', 'N/A')}")


# =======================================================================
# INTERACTIVE CHARTS TAB
# =======================================================================
def render_charts_tab(market_data: dict, tech_data: dict):
    hist = market_data.get("historical_df")
    if hist is None or hist.empty:
        st.warning("Historical price data unavailable for charting.")
        return

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        chart_type = st.radio("Style", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
    with ctrl2:
        show_sma = st.checkbox("Overlay SMAs (20, 50, 200)", value=True)
    with ctrl3:
        show_bb = st.checkbox("Bollinger Bands", value=True)
    with ctrl4:
        timeframe = st.selectbox("Lookback", ["Full Fetched", "Last 90 Days", "Last 60 Days", "Last 30 Days"], index=0, label_visibility="collapsed")

    # Slice historical dataframe based on timeframe selection
    plot_df = hist.copy()
    if timeframe == "Last 30 Days":
        plot_df = plot_df.tail(30)
    elif timeframe == "Last 60 Days":
        plot_df = plot_df.tail(60)
    elif timeframe == "Last 90 Days":
        plot_df = plot_df.tail(90)

    # Main Chart
    fig_main = create_price_and_volume_chart(
        plot_df,
        company_name=market_data.get("company_name", ""),
        symbol=market_data.get("symbol", ""),
        chart_type=chart_type,
        show_sma=show_sma,
        show_bollinger=show_bb
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # Subplot Oscillators in Expander or Columns
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        rsi_series = tech_data.get("indicators", {}).get("rsi", {}).get("series")
        if rsi_series is not None:
            fig_rsi = create_rsi_chart(rsi_series.tail(len(plot_df)), market_data.get("symbol", ""))
            st.plotly_chart(fig_rsi, use_container_width=True)

    with sub_col2:
        macd_ind = tech_data.get("indicators", {}).get("macd", {})
        if "macd_series" in macd_ind and macd_ind["macd_series"] is not None:
            fig_macd = create_macd_chart(
                macd_ind["macd_series"].tail(len(plot_df)),
                macd_ind["signal_series"].tail(len(plot_df)),
                macd_ind["histogram_series"].tail(len(plot_df)),
                market_data.get("symbol", "")
            )
            st.plotly_chart(fig_macd, use_container_width=True)


# =======================================================================
# BULLISH VS BEARISH MATRIX
# =======================================================================
def render_bullish_bearish_matrix(tech_data: dict, fund_data: dict, news_data: dict, risk_data: dict):
    col_bull, col_bear = st.columns(2)

    with col_bull:
        bull_items = []
        for s in tech_data.get("signals", {}).get("bullish", [])[:4]:
            bull_items.append(f'<li style="margin-bottom:6px;font-size:0.88rem;">{s}</li>')
        for s in fund_data.get("strengths", [])[:4]:
            bull_items.append(f'<li style="margin-bottom:6px;font-size:0.88rem;">{s}</li>')
        for c in news_data.get("positive_factors", [])[:2]:
            bull_items.append(f'<li style="margin-bottom:6px;font-size:0.88rem;"><b>[News Catalyst]</b> {c}</li>')
        if not bull_items:
            bull_items.append('<li style="color:#64748B;font-size:0.88rem;">No strong bullish catalysts identified.</li>')

        bull_html = (
            f'<div class="matrix-card">'
            f'<div class="matrix-card-title" style="color:#10B981;">🟢 BULLISH ARGUMENTS & CATALYSTS</div>'
            f'<ul style="padding-left:18px;margin:8px 0 0 0;list-style-type:disc;">{"".join(bull_items)}</ul>'
            f'</div>'
        )
        st.markdown(bull_html, unsafe_allow_html=True)

    with col_bear:
        bear_items = []
        for s in tech_data.get("signals", {}).get("bearish", [])[:4]:
            bear_items.append(f'<li style="margin-bottom:6px;font-size:0.88rem;">{s}</li>')
        for c in fund_data.get("concerns", [])[:4]:
            bear_items.append(f'<li style="margin-bottom:6px;font-size:0.88rem;">{c}</li>')
        for r in risk_data.get("identified_risks", [])[:3]:
            bear_items.append(f'<li style="margin-bottom:6px;font-size:0.88rem;"><b>[{r["category"]}]</b> {r["description"]}</li>')
        if not bear_items:
            bear_items.append('<li style="color:#64748B;font-size:0.88rem;">No critical bearish factors identified.</li>')

        bear_html = (
            f'<div class="matrix-card">'
            f'<div class="matrix-card-title" style="color:#EF4444;">🔴 BEARISH ARGUMENTS & RISKS</div>'
            f'<ul style="padding-left:18px;margin:8px 0 0 0;list-style-type:disc;">{"".join(bear_items)}</ul>'
            f'</div>'
        )
        st.markdown(bear_html, unsafe_allow_html=True)


# =======================================================================
# INVESTMENT SCENARIO CALCULATOR
# =======================================================================
def render_scenario_calculator(market_data: dict, decision_data: dict):
    current_price = market_data.get("current_price") or 1.0
    trade = decision_data.get("trade_setup", {})
    expected_pct = trade.get("expected_return_pct") or 10.0
    downside_pct = trade.get("potential_downside_pct") or 5.0

    st.markdown("### 🧮 Investment Scenario & Return Simulator")
    st.caption("Illustrative projected profit / loss across varying return outcomes for this stock.")

    calc_col1, calc_col2 = st.columns([2, 5])
    with calc_col1:
        invest_amount = st.number_input("Investment Capital (₹)", min_value=1000, max_value=5000000, value=50000, step=5000)
        shares_bought = int(invest_amount // current_price)
        invested_actual = safe_round(shares_bought * current_price)
        st.info(f"Shares Purchased: **{shares_bought:,}**\n\nActual Outlay: **₹{invested_actual:,.2f}**")

    with calc_col2:
        scenarios = [
            ("Severe Market Correction (-15%)", -15.0),
            (f"Calculated Stop-Loss ({trade.get('potential_downside_fmt', '-5%')})", -downside_pct),
            ("Conservative Stagnation (+5%)", 5.0),
            (f"Target Upside ({trade.get('expected_return_fmt', '+10%')})", expected_pct),
            ("Strong Bullish Breakout (+20%)", 20.0),
        ]

        rows = []
        for name, pct in scenarios:
            final_val = safe_round(invested_actual * (1 + pct / 100.0))
            pnl = safe_round(final_val - invested_actual)
            pnl_color = "🟢" if pnl >= 0 else "🔴"
            rows.append({
                "Outcome Scenario": name,
                "Return %": format_percentage(pct),
                "Projected Portfolio Value": f"₹{final_val:,.2f}",
                "Estimated P&L": f"{pnl_color} {format_currency(pnl)}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =======================================================================
# STOCK COMPARISON VIEW (Feature A)
# =======================================================================
def render_comparison_tab():
    st.markdown("### ⚖️ Multi-Stock Head-to-Head Comparison")
    st.caption("Analyze 2 or 3 Indian equities simultaneously across fundamentals, momentum, risk, and AI verdicts.")

    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    with c1:
        s1 = st.text_input("Stock 1", value="RELIANCE.NS").strip().upper()
    with c2:
        s2 = st.text_input("Stock 2", value="TCS.NS").strip().upper()
    with c3:
        s3 = st.text_input("Stock 3 (Optional)", value="INFY.NS").strip().upper()
    with c4:
        st.write("")
        run_comp = st.button("Compare Equities", type="primary", use_container_width=True)

    if run_comp:
        targets = [s for s in [s1, s2, s3] if s]
        with st.spinner("Analyzing stocks through multi-agent workflows..."):
            comp_result = compare_stocks(targets)

        if comp_result.get("status") == "success":
            st.success(f"**Comparative Finding:** {comp_result['comparative_reasoning']}")

            df_comp = pd.DataFrame(comp_result["table_rows"])
            display_cols = [
                "Company", "Symbol", "Current Price", "1-Day Change", "1-Month Change",
                "Market Cap", "P/E Ratio", "ROE", "RSI (14)", "Trend",
                "Risk Profile", "AI Verdict", "P(Positive Return)", "Expected Target"
            ]
            st.dataframe(df_comp[display_cols], use_container_width=True, hide_index=True)
        else:
            st.error(comp_result.get("message", "Comparison failed."))


# =======================================================================
# WATCHLIST & ALERTS VIEW (Feature B & C)
# =======================================================================
def render_watchlist_and_alerts_tab():
    w_col1, w_col2 = st.columns([5, 4])

    with w_col1:
        st.markdown("### 📋 Personal Stock Watchlist")
        live_watch = get_watchlist_live_data()
        if live_watch:
            watch_df = pd.DataFrame(live_watch)[["symbol", "name", "current_price_fmt", "pct_change_fmt", "last_verdict", "added_at"]]
            watch_df.columns = ["Symbol", "Company", "Current Price", "1-Day Change", "AI Verdict", "Added Date"]
            st.dataframe(watch_df, use_container_width=True, hide_index=True)

            del_sym = st.selectbox("Remove from Watchlist", options=[item["symbol"] for item in live_watch])
            if st.button("Remove Stock", key="btn_rem_watch"):
                remove_from_watchlist(del_sym)
                st.rerun()
        else:
            st.info("Watchlist is empty. Search for any stock and click 'Add to Watchlist'.")

    with w_col2:
        st.markdown("### 🔔 Price Trigger Alerts")
        alerts = check_alerts_status()

        with st.form("alert_form"):
            a_sym = st.text_input("Symbol (e.g. RELIANCE.NS)").strip().upper()
            a_price = st.number_input("Target Price (₹)", min_value=1.0, value=1500.0)
            a_cond = st.selectbox("Condition", ["ABOVE", "BELOW"])
            a_note = st.text_input("Alert Note (Optional)", placeholder="Breakout confirmation...")
            submitted = st.form_submit_button("Set Price Alert")
            if submitted and a_sym:
                add_price_alert(a_sym, a_price, a_cond, a_note)
                st.success(f"Alert set for {a_sym} at ₹{a_price:,.2f}")
                st.rerun()

        if alerts:
            st.markdown("**Active Alerts:**")
            for a in alerts:
                stat = "🟢 TRIGGERED!" if a.get("triggered") else "⏳ Active"
                st.markdown(
                    f"• **{a['symbol']}** {a['condition']} ₹{a['target_price']:,.2f} — *{stat}* "
                    f"<small style='color:#64748B;'>({a.get('note', '')})</small>",
                    unsafe_allow_html=True
                )


# =======================================================================
# MAIN APPLICATION CONTROLLER
# =======================================================================
def main():
    # Top Market Overview Ribbon (Nifty, Sensex, Bank Nifty)
    render_market_overview_ribbon()

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Terminal Settings")
        hist_days = st.slider("Lookback History (Days)", 30, 365, 90, 30)
        st.divider()
        st.markdown("### 🤖 6-Agent LangGraph System")
        st.markdown("""
        1. **Stock Data Agent** `yfinance`
        2. **Technical Agent** `ta library`
        3. **Fundamental Agent** `public filings`
        4. **Market Sentiment Agent** `RSS & news`
        5. **Risk Analysis Agent** `statistical VaR`
        6. **Final Decision Agent** `synthesis & verdict`
        """)
        st.divider()
        st.markdown("""
        <div class="institutional-disclaimer">
        <b>Academic Project:</b> Data delayed by ~15 min. No guarantee of future returns. Educational and research use only.
        </div>
        """, unsafe_allow_html=True)

    # Hero Search Component
    symbol_query, exchange_choice, execute_analysis = render_search_hero()

    # Check for active analysis
    if "analysis_results" not in st.session_state or execute_analysis:
        if execute_analysis and symbol_query:
            with st.spinner(f"Running 6-agent LangGraph orchestration for **{symbol_query}**..."):
                res = run_cached_stock_analysis(symbol_query, hist_days, exchange_choice)
                st.session_state["analysis_results"] = res
        elif "analysis_results" not in st.session_state:
            with st.spinner("Initializing default benchmark (RELIANCE)..."):
                res = run_cached_stock_analysis("RELIANCE", hist_days, "NSE")
                st.session_state["analysis_results"] = res

    result = st.session_state.get("analysis_results")
    if not result:
        return

    if result.get("workflow_status") == "failed":
        st.error(f"❌ Analysis failed for `{symbol_query}`")
        st.error(result.get("error_message", "Unknown error"))
        return

    # Unpack agent results
    market_data = result.get("market_data", {})
    tech_data = result.get("tech_data", {})
    fund_data = result.get("fund_data", {})
    news_data = result.get("news_data", {})
    risk_data = result.get("risk_data", {})
    decision_data = result.get("decision_data", {})
    report_data = result.get("report_data", {})

    st.divider()

    # 1. Stock Overview Header
    render_stock_overview_header(market_data, decision_data)

    # 2. Prominent Final Verdict Banner ("Buy or Not?")
    render_verdict_section(decision_data, market_data)

    # 3. Comprehensive Analysis Tabs
    tab_chart, tab_tech, tab_fund, tab_news, tab_risk, tab_matrix, tab_calc, tab_report, tab_comp, tab_watch = st.tabs([
        "📈 Interactive Charts",
        "📉 Technical Analysis",
        "🏢 Fundamentals",
        "📰 Sentiment & News",
        "⚠️ Risk Assessment",
        "⚖️ Bull vs Bear",
        "🧮 Scenario Calculator",
        "📄 Research Report",
        "🔄 Stock Comparison",
        "⭐ Watchlist & Alerts"
    ])

    with tab_chart:
        render_charts_tab(market_data, tech_data)

    with tab_tech:
        st.markdown("### 📉 Technical Indicator Appraisal")
        t_c1, t_c2, t_c3 = st.columns(3)
        rsi = tech_data.get("indicators", {}).get("rsi", {})
        macd = tech_data.get("indicators", {}).get("macd", {})
        sma = tech_data.get("indicators", {}).get("sma", {})
        bb = tech_data.get("indicators", {}).get("bollinger", {})
        sr = tech_data.get("support_resistance", {})

        with t_c1:
            st.metric("RSI (14-period)", str(rsi.get("value", "N/A")), delta=rsi.get("zone", "Neutral"))
            st.caption(rsi.get("interpretation", ""))
            st.markdown(f"**Pivot Points:** Pivot: `₹{sr.get('pivot','N/A')}` | R1: `₹{sr.get('resistance_1','N/A')}` | S1: `₹{sr.get('support_1','N/A')}`")

        with t_c2:
            st.metric("MACD Status", macd.get("crossover", "N/A"), delta=str(macd.get("histogram_latest", "N/A")))
            st.caption(macd.get("interpretation", ""))
            st.markdown(f"**Bollinger Bands:** Upper: `₹{bb.get('upper','N/A')}` | Lower: `₹{bb.get('lower','N/A')}` | %B: `{bb.get('percent_b','N/A')}`")

        with t_c3:
            st.metric("Trend Structure", tech_data.get("indicators", {}).get("trend", {}).get("trend", "N/A"))
            st.markdown(f"• SMA 20: **₹{sma.get('sma20','N/A')}** ({sma.get('price_vs_sma20','N/A')})")
            st.markdown(f"• SMA 50: **₹{sma.get('sma50','N/A')}** ({sma.get('price_vs_sma50','N/A')})")
            st.markdown(f"• SMA 200: **₹{sma.get('sma200','N/A')}** ({sma.get('price_vs_sma200','N/A')})")

        st.info(tech_data.get("summary", ""))

    with tab_fund:
        st.markdown("### 🏢 Fundamental & Valuation Analysis")
        f_c1, f_c2, f_c3 = st.columns(3)
        fm = fund_data.get("metrics", {})
        with f_c1:
            st.metric("Valuation Status", fund_data.get("valuation_status", "N/A"))
            st.markdown(f"• P/E Ratio: **{fm.get('pe_ratio','N/A')}x** (Fwd: {fm.get('forward_pe','N/A')}x)")
            st.markdown(f"• Price-to-Book (P/B): **{fm.get('pb_ratio','N/A')}x**")
        with f_c2:
            st.metric("Balance Sheet Health", fund_data.get("financial_health", "N/A"))
            st.markdown(f"• Debt-to-Equity: **{fm.get('debt_to_equity','N/A')}**")
            st.markdown(f"• Book Value: **₹{fm.get('book_value','N/A')}**")
        with f_c3:
            st.metric("Capital Productivity (ROE)", f"{fm.get('return_on_equity','N/A')}%" if fm.get('return_on_equity') else "N/A")
            st.markdown(f"• Trailing EPS: **₹{fm.get('eps','N/A')}**")
            st.markdown(f"• Dividend Yield: **{fm.get('dividend_yield','0.0')}%**")

        st.info(fund_data.get("summary", ""))

    with tab_news:
        st.markdown("### 📰 Market News & Sentiment Analysis")
        sc = news_data.get("sentiment_count", {})
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Overall Sentiment", news_data.get("overall_sentiment", "UNAVAILABLE"))
        n2.metric("🟢 Positive", sc.get("positive", 0))
        n3.metric("🔴 Negative", sc.get("negative", 0))
        n4.metric("🟡 Neutral", sc.get("neutral", 0))

        if news_data.get("key_events"):
            st.markdown(f"**Detected Institutional Events:** {', '.join(news_data['key_events'])}")

        st.markdown("**Top Verified Headlines:**")
        for art in news_data.get("top_headlines", [])[:6]:
            icon = art.get("sentiment_icon", "🟡")
            title = art.get("title", "")
            url = art.get("url", "#")
            source = art.get("source", "News")
            st.markdown(f"{icon} [{title}]({url}) — <span style='font-size:0.8rem;color:#64748B;'>{source}</span>", unsafe_allow_html=True)

    with tab_risk:
        st.markdown("### ⚠️ Risk Assessment & Volatility Modeling")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Profile", risk_data.get("risk_level", "MODERATE RISK"))
        r2.metric("Annualized Volatility", f"{risk_data.get('volatility_annualized','N/A')}%")
        r3.metric("Historical Max Drawdown", f"{risk_data.get('max_drawdown_pct','N/A')}%")
        r4.metric("1-Day 95% VaR", f"{risk_data.get('var_95_1d','N/A')}%")

        st.markdown(f"**Target Investor Suitability:** `{risk_data.get('investor_suitability', 'N/A')}`")
        st.info(risk_data.get("summary", ""))

    with tab_matrix:
        st.markdown("### ⚖️ Bullish vs Bearish Factor Analysis")
        render_bullish_bearish_matrix(tech_data, fund_data, news_data, risk_data)

    with tab_calc:
        render_scenario_calculator(market_data, decision_data)

    with tab_report:
        st.markdown("### 📄 Institutional Research Report")
        rep_text = report_data.get("report_text", "")
        st.text_area("Full Consolidated Research Report", rep_text, height=450)
        st.download_button(
            label="⬇️ Download Full Institutional Report (.txt)",
            data=rep_text,
            file_name=f"{market_data.get('symbol')}_Research_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with tab_comp:
        render_comparison_tab()

    with tab_watch:
        render_watchlist_and_alerts_tab()

    st.markdown("""
    <div class="institutional-disclaimer">
    <b>MANDATORY FINANCIAL DISCLAIMER:</b> This platform is an artificial intelligence research prototype
    developed strictly for educational and academic evaluation. Model predictions, probability scores,
    and target levels are derived from mathematical models and historical datasets; they do NOT guarantee returns.
    Consult a SEBI-registered investment advisor before making any financial commitments.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
