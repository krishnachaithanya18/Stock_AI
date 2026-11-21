"""
app.py — Streamlit Dashboard (Phase 8 — Full Version)
======================================================
WHAT THIS FILE DOES:
  Builds a complete, professional stock analysis dashboard using Streamlit.
  When the user enters a stock symbol and clicks "Analyse Stock", it runs
  the entire LangGraph multi-agent pipeline and displays all results
  in a structured, tabbed interface.

HOW TO RUN:
  streamlit run app.py

STREAMLIT BASICS (for beginners):
  - Every Python statement that calls st.something() adds a UI element
  - The script re-runs top-to-bottom every time the user interacts
  - st.session_state stores values that survive re-runs (like cache)
  - @st.cache_data decorates functions whose results should be cached
  - st.spinner() shows a loading indicator during long operations

CHARTING:
  We use Streamlit's built-in charts (st.line_chart, st.bar_chart)
  which use Altair/Vega-Lite underneath — no extra packages needed.
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import APP_TITLE, APP_VERSION, validate_config
from graph.workflow import run_analysis


# =============================================================
# PAGE CONFIGURATION — Must be the FIRST Streamlit call
# =============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",           # Use full browser width
    initial_sidebar_state="expanded",
)


# =============================================================
# CUSTOM CSS — Professional Financial Dashboard Styling
# =============================================================
st.markdown("""
<style>
/* Main background */
.main { background-color: #0e1117; }

/* Hero search box */
.hero-container {
    background: linear-gradient(135deg, #1a1f35 0%, #0e1117 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 40px 48px 36px 48px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.0rem;
    color: #718096;
    margin-bottom: 28px;
}
.chip-row {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 16px;
}
.stock-chip {
    background: #2d3748;
    color: #e2e8f0;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid #4a5568;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #1c2333;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    padding: 8px 0 4px 0;
    border-bottom: 2px solid #2d3748;
    margin-bottom: 12px;
}

/* Signal badge styles */
.signal-bullish {
    background-color: #1a4731;
    color: #48bb78;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.signal-bearish {
    background-color: #4a1942;
    color: #fc8181;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.signal-neutral {
    background-color: #2d3748;
    color: #e2e8f0;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.signal-mixed {
    background-color: #3d2b00;
    color: #f6ad55;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}

/* Risk badge styles */
.risk-high     { color: #fc8181; font-weight: 700; }
.risk-moderate { color: #f6ad55; font-weight: 700; }
.risk-low      { color: #48bb78; font-weight: 700; }

/* Report text area */
.report-box {
    background-color: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 20px;
    font-family: monospace;
    font-size: 0.85rem;
    white-space: pre-wrap;
    max-height: 600px;
    overflow-y: auto;
}

/* Disclaimer */
.disclaimer {
    background-color: #1a2744;
    border-left: 4px solid #4299e1;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #90cdf4;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================
# SIDEBAR — Settings only (search moved to main page)
# =============================================================
def render_sidebar():
    """Render the left sidebar with settings and project info."""

    with st.sidebar:
        st.markdown("## 📈 Indian Stock Market AI")
        st.caption(f"Version {APP_VERSION} | LangGraph Multi-Agent System")
        st.divider()

        # --- Settings ---
        st.markdown("### ⚙️ Analysis Settings")
        history_days = st.slider(
            "Historical Data (days)",
            min_value=30, max_value=180, value=90, step=30,
            help="Number of past trading days to fetch"
        )
        max_news = st.slider(
            "Max News Articles",
            min_value=5, max_value=15, value=10, step=5,
        )

        st.divider()
        st.markdown("### 📚 Agent Pipeline")
        for icon, name, tool in [
            ("1️⃣", "Market Data Agent",  "yfinance"),
            ("2️⃣", "Technical Agent",    "ta library"),
            ("3️⃣", "News Agent",         "Google RSS"),
            ("4️⃣", "Research Agent",     "Rule-based"),
            ("5️⃣", "Report Agent",       "Gemini LLM"),
        ]:
            st.markdown(f"{icon} **{name}** `{tool}`")

        st.divider()
        st.markdown("""
<div class="disclaimer">
⚠️ <b>Disclaimer:</b> Academic project only.
Not financial advice. Data is delayed
~15–20 min. Never use for real trading.
</div>
""", unsafe_allow_html=True)

    return history_days, max_news


# =============================================================
# HERO SEARCH BAR — Main page stock input
# =============================================================
def render_hero_search() -> tuple:
    """
    Render the prominent hero-style search bar on the main page.

    WHY SESSION STATE FOR THE SYMBOL?
      Streamlit re-runs the script from top to bottom on every
      interaction. st.session_state["search_symbol"] persists the
      current symbol value across re-runs, so quick-select buttons
      can update the text input correctly.

    Returns:
        (symbol, analyse_clicked) tuple
    """

    # Initialise session state key on first run
    if "search_symbol" not in st.session_state:
        st.session_state["search_symbol"] = "RELIANCE.NS"

    # --- Hero container ---
    st.markdown("""
<div class="hero-container">
  <div class="hero-title">📈 Indian Stock Market Agentic AI</div>
  <div class="hero-subtitle">
      Multi-Agent Research System · LangGraph · Google Gemini · NSE / BSE
  </div>
</div>
""", unsafe_allow_html=True)

    # --- Search row: text input + button side by side ---
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        symbol = st.text_input(
            label="Stock Symbol",
            value=st.session_state["search_symbol"],
            placeholder="Enter NSE/BSE symbol — e.g.  RELIANCE.NS  ·  TCS.NS  ·  INFY.NS",
            label_visibility="collapsed",
            key="hero_input",
            help=(
                "NSE stocks: add .NS suffix  →  RELIANCE.NS, TCS.NS, INFY.NS\n"
                "BSE stocks: add .BO suffix  →  RELIANCE.BO\n"
                "Press Enter or click Analyse to run the 5-agent pipeline."
            ),
        ).upper().strip()

    with col_btn:
        analyse_clicked = st.button(
            "🚀 Analyse",
            type="primary",
            use_container_width=True,
        )

    # --- Quick-select popular stocks ---
    st.markdown(
        "<div style='text-align:center;color:#718096;"
        "font-size:0.82rem;margin:4px 0 8px 0'>Popular Indian Stocks:</div>",
        unsafe_allow_html=True
    )

    popular = [
        ("RELIANCE.NS", "Reliance"),
        ("TCS.NS",      "TCS"),
        ("INFY.NS",     "Infosys"),
        ("HDFCBANK.NS", "HDFC Bank"),
        ("WIPRO.NS",    "Wipro"),
        ("ICICIBANK.NS","ICICI Bank"),
        ("LT.NS",       "L&T"),
        ("BHARTIARTL.NS","Airtel"),
    ]

    # Render chips as columns of buttons
    cols = st.columns(len(popular))
    for col, (sym, label) in zip(cols, popular):
        if col.button(label, key=f"chip_{sym}", use_container_width=True):
            # Update the session state so next render shows new symbol
            st.session_state["search_symbol"] = sym
            st.rerun()

    # Sync symbol from session state if changed by chip click
    if st.session_state.get("search_symbol") != symbol and not analyse_clicked:
        symbol = st.session_state["search_symbol"]
    else:
        st.session_state["search_symbol"] = symbol

    return symbol, analyse_clicked


# =============================================================
# RUN ANALYSIS (cached so re-renders don't re-fetch)
# =============================================================
@st.cache_data(show_spinner=False, ttl=300)   # Cache for 5 minutes
def cached_analysis(symbol: str, history_days: int, max_news: int) -> dict:
    """
    Run the LangGraph pipeline and cache the result.

    @st.cache_data means: if this function is called again with the
    SAME arguments within 300 seconds, return the cached result
    instead of re-running all 5 agents. This avoids redundant API calls.
    """
    return run_analysis(symbol, history_days, max_news)


# =============================================================
# DASHBOARD RENDERING FUNCTIONS
# =============================================================

def render_header(market_data: dict):
    """Top section: company name, price, key metrics."""

    company = market_data.get("company_name", "Unknown")
    symbol  = market_data.get("symbol", "")
    price   = market_data.get("current_price_fmt", "N/A")
    pc      = market_data.get("price_change", {})
    mi      = market_data.get("market_info", {})

    st.title(f"📈 {company}")
    st.caption(f"{symbol} · Yahoo Finance (delayed ~15-20 min) · {market_data.get('data_source', {}).get('fetched_at', '')}")

    # Key metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Current Price",   price,
                delta=pc.get("one_day_pct_fmt", "N/A"),
                delta_color="normal")
    col2.metric("1-Week Change",   pc.get("one_week_pct_fmt", "N/A"))
    col3.metric("1-Month Change",  pc.get("one_month_pct_fmt", "N/A"))
    col4.metric("Market Cap",      mi.get("market_cap_fmt", "N/A"))
    col5.metric("P/E Ratio",       str(mi.get("pe_ratio", "N/A")))
    col6.metric("Beta",            str(mi.get("beta", "N/A")))


def render_price_chart(market_data: dict):
    """Render the historical price and volume chart."""

    st.markdown('<div class="section-header">📊 Historical Price Chart</div>', unsafe_allow_html=True)

    hist = market_data.get("historical_df")
    if hist is None or hist.empty:
        st.warning("No historical data available for chart.")
        return

    # Prepare close price series
    close_df = pd.DataFrame({
        "Close Price (₹)": hist["Close"]
    })
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)

    st.line_chart(close_df, use_container_width=True, height=300)

    # Volume chart
    with st.expander("📦 Trading Volume"):
        vol_df = pd.DataFrame({"Volume": hist["Volume"]})
        vol_df.index = pd.to_datetime(vol_df.index).tz_localize(None)
        st.bar_chart(vol_df, use_container_width=True, height=200)

    mi = market_data.get("market_info", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("52-Week High", f"₹{mi.get('fifty_two_week_high', 'N/A')}")
    col2.metric("52-Week Low",  f"₹{mi.get('fifty_two_week_low', 'N/A')}")
    col3.metric("Sector",       mi.get("sector", "N/A"))


def render_technical(tech_data: dict):
    """Render technical indicators section."""

    st.markdown('<div class="section-header">📉 Technical Analysis</div>', unsafe_allow_html=True)

    rsi  = tech_data.get("indicators", {}).get("rsi",  {})
    macd = tech_data.get("indicators", {}).get("macd", {})
    sma  = tech_data.get("indicators", {}).get("sma",  {})

    # Overall signal badge
    overall = tech_data.get("overall_signal", "NEUTRAL")
    score   = tech_data.get("signal_score", 0)
    css_cls = f"signal-{overall.lower()}"
    st.markdown(
        f'<span class="{css_cls}">{overall}</span> '
        f'<span style="color:#718096;font-size:0.85rem;margin-left:8px;">Score: {score:+d}</span>',
        unsafe_allow_html=True
    )
    st.write("")

    col1, col2, col3 = st.columns(3)

    # --- RSI ---
    with col1:
        rsi_val  = rsi.get("value", 0) or 0
        rsi_zone = rsi.get("zone", "Neutral")
        rsi_color = "#fc8181" if rsi_zone == "Overbought" else ("#48bb78" if rsi_zone == "Oversold" else "#e2e8f0")

        st.markdown(f"**RSI ({rsi.get('period',14)}-day)**")
        st.markdown(f"<h2 style='color:{rsi_color};margin:0'>{rsi_val}</h2>", unsafe_allow_html=True)
        st.caption(f"Zone: {rsi_zone}")
        st.caption(rsi.get("interpretation", "")[:100])

    # --- MACD ---
    with col2:
        macd_val  = macd.get("macd_latest", "N/A")
        crossover = macd.get("crossover", "N/A")
        hist_val  = macd.get("histogram_latest", 0) or 0
        macd_color = "#48bb78" if hist_val > 0 else "#fc8181"

        st.markdown("**MACD (12,26,9)**")
        st.markdown(f"<h2 style='color:{macd_color};margin:0'>{macd_val}</h2>", unsafe_allow_html=True)
        st.caption(f"Histogram: {hist_val}")
        st.caption(crossover[:60] if crossover else "N/A")

    # --- SMA ---
    with col3:
        sma20       = sma.get("sma20", "N/A")
        sma50       = sma.get("sma50", "N/A")
        trend       = sma.get("trend_structure", "N/A")
        trend_color = "#48bb78" if "Golden" in str(trend) else "#fc8181"

        st.markdown("**Moving Averages**")
        st.markdown(f"SMA 20: **₹{sma20}**")
        st.markdown(f"SMA 50: **₹{sma50}**")
        st.markdown(f"<span style='color:{trend_color};font-size:0.8rem'>{str(trend)[:45]}</span>",
                    unsafe_allow_html=True)

    st.divider()

    # Bullish / Bearish signal lists
    signals = tech_data.get("signals", {})
    bull_sigs = signals.get("bullish", [])
    bear_sigs = signals.get("bearish", [])

    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown(f"**✅ Bullish Signals ({len(bull_sigs)})**")
        for s in bull_sigs:
            st.markdown(f"<small>✅ {s}</small>", unsafe_allow_html=True)

    with col_bear:
        st.markdown(f"**❌ Bearish Signals ({len(bear_sigs)})**")
        for s in bear_sigs:
            st.markdown(f"<small>❌ {s}</small>", unsafe_allow_html=True)

    # RSI sparkline chart
    rsi_series = rsi.get("series")
    if rsi_series is not None:
        with st.expander("📈 RSI Chart (last 90 days)"):
            rsi_df = pd.DataFrame({"RSI": rsi_series.dropna()})
            rsi_df.index = pd.to_datetime(rsi_df.index).tz_localize(None)
            st.line_chart(rsi_df, height=200)
            st.caption("RSI > 70 = Overbought (red zone) | RSI < 30 = Oversold (green zone)")


def render_news(news_data: dict):
    """Render the news sentiment section."""

    st.markdown('<div class="section-header">📰 News & Sentiment Analysis</div>', unsafe_allow_html=True)

    overall = news_data.get("overall_sentiment", "UNAVAILABLE")
    count   = news_data.get("article_count", 0)
    sc      = news_data.get("sentiment_count", {})

    if count == 0:
        st.warning(f"⚠️ {news_data.get('error_message', 'No news articles found.')}")
        return

    # Sentiment overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Sentiment", overall)
    col2.metric("✅ Positive", sc.get("positive", 0))
    col3.metric("❌ Negative", sc.get("negative", 0))
    col4.metric("🟡 Neutral",  sc.get("neutral", 0))

    # Sentiment bar chart
    sent_df = pd.DataFrame({
        "Articles": [sc.get("positive",0), sc.get("negative",0), sc.get("neutral",0)]
    }, index=["Positive", "Negative", "Neutral"])
    st.bar_chart(sent_df, height=180, use_container_width=True)

    # Key events
    events = news_data.get("key_events", [])
    if events:
        st.markdown("**📌 Key Events Detected:**")
        st.markdown("  ".join(f"`{e}`" for e in events))

    # Headlines
    st.markdown("**🗞️ Top Headlines:**")
    for art in news_data.get("top_headlines", []):
        icon  = art.get("sentiment_icon", "🟡")
        title = art.get("title", "")
        url   = art.get("url", "#")
        source= art.get("source", "")
        pub   = art.get("published", "")[:16]
        st.markdown(
            f"{icon} [{title[:90]}]({url})  "
            f"<small style='color:#718096'>{source} · {pub}</small>",
            unsafe_allow_html=True
        )

    # All articles expander
    with st.expander(f"📄 All {count} Articles"):
        for i, art in enumerate(news_data.get("articles", []), 1):
            icon  = art.get("sentiment_icon", "🟡")
            title = art.get("title", "")
            url   = art.get("url", "#")
            sent  = art.get("sentiment", "neutral")
            score = f"+{art.get('pos_score',0)}/-{art.get('neg_score',0)}"
            st.markdown(
                f"{i}. {icon} [{title[:85]}]({url})  "
                f"<small style='color:#718096'>[{sent} {score}]</small>",
                unsafe_allow_html=True
            )


def render_research(research_data: dict):
    """Render the research & decision agent findings."""

    st.markdown('<div class="section-header">🧠 Research & Decision Agent</div>', unsafe_allow_html=True)

    stance  = research_data.get("overall_stance", "NEUTRAL")
    conf    = research_data.get("confidence", "N/A")
    risk    = research_data.get("risk_level", "N/A")

    # Overall assessment cards
    col1, col2, col3 = st.columns(3)

    css_cls = f"signal-{stance.lower()}"
    col1.markdown(f"**Overall Stance**")
    col1.markdown(f'<span class="{css_cls}">{stance}</span>', unsafe_allow_html=True)

    col2.markdown(f"**Confidence**")
    col2.markdown(f"**{conf}**")

    risk_cls = f"risk-{risk.lower()}"
    col3.markdown(f"**Risk Level**")
    col3.markdown(f'<span class="{risk_cls}">⚠ {risk}</span>', unsafe_allow_html=True)

    st.write("")

    # Research summary
    st.info(research_data.get("research_summary", ""))

    # Agreements and conflicts
    col_a, col_b = st.columns(2)
    with col_a:
        agreements = research_data.get("agreement_areas", [])
        st.markdown(f"**✅ Signal Agreements ({len(agreements)})**")
        for a in agreements:
            st.success(a, icon="✅")

    with col_b:
        conflicts = research_data.get("conflict_areas", [])
        st.markdown(f"**⚠️ Signal Conflicts ({len(conflicts)})**")
        for c in conflicts:
            st.warning(c, icon="⚠️")

    # Risk factors
    risks = research_data.get("risks", [])
    if risks:
        st.markdown(f"**⚠️ Risk Factors ({len(risks)}) — Level: {risk}**")
        for r in risks:
            st.error(f"**{r['name']}** — {r['description']}", icon="🔴")

    # Key findings
    with st.expander("📋 Key Findings"):
        for i, f in enumerate(research_data.get("key_findings", []), 1):
            st.markdown(f"**{i}.** {f}")

    # Bullish/Bearish cases
    with st.expander("🟢 Full Bullish Case"):
        for b in research_data.get("bullish_case", []):
            st.markdown(f"✅ {b}")

    with st.expander("🔴 Full Bearish Case"):
        for b in research_data.get("bearish_case", []):
            st.markdown(f"❌ {b}")


def render_report(report_data: dict):
    """Render the final AI research report."""

    st.markdown('<div class="section-header">📄 Final Research Report</div>', unsafe_allow_html=True)

    report_type = report_data.get("report_type", "template")
    generated   = report_data.get("generated_at", "")

    # Report header info
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Report Type:** `{'🤖 Gemini AI' if report_type == 'gemini' else '📋 Template'}`")
    col2.markdown(f"**Generated:** {generated}")
    if report_data.get("tokens_used") and report_data["tokens_used"] != 0:
        col3.markdown(f"**Tokens Used:** {report_data.get('tokens_used', 'N/A')}")

    if report_type != "gemini":
        st.info(
            "ℹ️ **Template Mode Active** — This report is generated from real data collected "
            "by all 5 agents. Gemini AI mode requires billing to be enabled at "
            "[aistudio.google.com/billing](https://aistudio.google.com/billing).",
            icon="ℹ️"
        )

    # Full report text
    report_text = report_data.get("report_text", "No report generated.")
    st.code(report_text, language=None)

    # Download button
    st.download_button(
        label="⬇️ Download Report as .txt",
        data=report_text,
        file_name=f"{report_data.get('symbol','report')}_{generated[:10]}_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


def render_market_details(market_data: dict):
    """Render detailed market data in an expander."""

    with st.expander("🏢 Full Company & Market Details"):
        mi  = market_data.get("market_info", {})
        vol = market_data.get("volume", {})
        ds  = market_data.get("data_source", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Company Info**")
            st.markdown(f"- Sector: **{mi.get('sector','N/A')}**")
            st.markdown(f"- Industry: **{mi.get('industry','N/A')}**")
            st.markdown(f"- Exchange: **{mi.get('exchange','N/A')}**")
            st.markdown(f"- Currency: **{mi.get('currency','INR')}**")
            st.markdown(f"- EPS: **{mi.get('eps','N/A')}**")
            st.markdown(f"- Dividend Yield: **{mi.get('dividend_yield','0')}%**")

        with col2:
            st.markdown("**Data Source**")
            st.markdown(f"- Provider: **{ds.get('provider','N/A')}**")
            st.markdown(f"- Type: **{ds.get('data_type','N/A')}**")
            st.markdown(f"- Period: **{ds.get('data_from','N/A')}** to **{ds.get('data_to','N/A')}**")
            st.markdown(f"- Records: **{ds.get('total_records','N/A')} trading days**")
            st.markdown(f"- Latest Volume: **{vol.get('volume_formatted','N/A')}**")
            st.markdown(f"- 30D Avg Volume: **{vol.get('avg_volume_30d', 'N/A'):,.0f}**"
                        if vol.get('avg_volume_30d') else "- 30D Avg Volume: **N/A**")


# =============================================================
# MAIN APP ENTRY POINT
# =============================================================

def main():
    # --- Sidebar: settings only (no symbol input here) ---
    history_days, max_news = render_sidebar()

    # --- Hero search bar: symbol + analyse button on main page ---
    symbol, analyse_clicked = render_hero_search()

    # --- Check config ---
    if not validate_config():
        st.error(
            "⚠️ Configuration Error: GOOGLE_API_KEY is not set in your .env file. "
            "Market data and technical analysis will still work. "
            "Add your Gemini key for AI-powered reports."
        )

    # --- Welcome screen (before analysis) ---
    if "last_result" not in st.session_state and not analyse_clicked:
        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.markdown("""
**🤖 5 AI Agents**
- Market Data Agent
- Technical Analysis Agent
- News Sentiment Agent
- Research & Decision Agent
- Report Generation Agent
""")
        col2.markdown("""
**📊 Real Data Sources**
- Yahoo Finance (NSE/BSE prices)
- Google News RSS (headlines)
- Technical indicators via `ta`
- LangGraph orchestration
""")
        col3.markdown("""
**📄 Output**
- 9-section research report
- RSI, MACD, SMA signals
- News sentiment analysis
- Risk assessment
- Download as .txt
""")

        st.info(
            "☝️ Enter a stock symbol above (or click a quick-select chip) and click **🚀 Analyse** to begin.",
            icon="☝️"
        )

        st.markdown(
            '<div class="disclaimer">⚠️ <b>Academic Project Disclaimer:</b> '
            'This system is built for educational purposes only. It does NOT provide '
            'financial advice and does NOT guarantee investment returns. '
            'All data is delayed by ~15-20 minutes. Never use for real trading decisions. '
            'Always consult a SEBI-registered financial advisor.</div>',
            unsafe_allow_html=True
        )
        return

    # --- Run analysis ---
    if analyse_clicked:
        if not symbol or len(symbol) < 3:
            st.error("Please enter a valid stock symbol (e.g. RELIANCE.NS)")
            return

        with st.spinner(f"🔄 Running 5-agent LangGraph analysis for **{symbol}**... (30-60 seconds)"):
            result = cached_analysis(symbol, history_days, max_news)

        st.session_state["last_result"] = result
        st.session_state["last_symbol"] = symbol

    # --- Show results ---
    result = st.session_state.get("last_result")
    if not result:
        return

    # Check for workflow failure
    if result.get("workflow_status") == "failed":
        st.error(f"❌ Analysis failed for `{symbol}`")
        st.error(result.get("error_message", "Unknown error"))
        st.info("💡 Check the symbol format: NSE stocks need `.NS` suffix (e.g. `RELIANCE.NS`)")
        return

    # --- Unpack state ---
    market_data   = result.get("market_data", {})
    tech_data     = result.get("tech_data", {})
    news_data     = result.get("news_data", {})
    research_data = result.get("research_data", {})
    report_data   = result.get("report_data", {})

    # --- Header ---
    render_header(market_data)
    st.divider()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price & Chart",
        "📉 Technical Analysis",
        "📰 News & Sentiment",
        "🧠 Research Assessment",
        "📄 Full Report",
    ])

    with tab1:
        render_price_chart(market_data)
        render_market_details(market_data)

    with tab2:
        render_technical(tech_data)

    with tab3:
        render_news(news_data)

    with tab4:
        render_research(research_data)

    with tab5:
        render_report(report_data)

    # --- Footer ---
    st.divider()
    st.markdown(
        f'<div class="disclaimer">⚠️ <b>Disclaimer:</b> '
        f'This report is generated by an AI system for academic purposes only. '
        f'It is NOT financial advice. Data from Yahoo Finance (delayed ~15-20 min). '
        f'Generated at {report_data.get("generated_at","N/A")} | '
        f'LangGraph Multi-Agent System | 5 Agents</div>',
        unsafe_allow_html=True
    )


# --- Run the app ---
if __name__ == "__main__" or True:
    main()
