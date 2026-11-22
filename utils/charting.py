"""
utils/charting.py — Professional Financial Terminal Charting Engine
====================================================================
Generates high-performance, dark-themed interactive financial charts using Plotly:
1. Candlestick & OHLCV Price Chart with Volume Bar Subplot
2. Overlay Indicators: SMA 20, SMA 50, SMA 200, Bollinger Bands
3. Interactive RSI Chart with 30/70 Overbought/Oversold Bands
4. Interactive MACD Chart with Signal Line and Divergence Histogram
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# Dark Financial Terminal Palette
BG_COLOR = "#0D1117"
PANEL_COLOR = "#161B22"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#C9D1D9"
GREEN_BULL = "#238636"
RED_BEAR = "#DA3633"
ACCENT_BLUE = "#58A6FF"
ORANGE_WARN = "#D29922"
PURPLE_LINE = "#BC8CFF"


def create_price_and_volume_chart(hist: pd.DataFrame, company_name: str, symbol: str,
                                  chart_type: str = "Candlestick",
                                  show_sma: bool = True,
                                  show_bollinger: bool = True) -> go.Figure:
    """
    Creates an interactive 2-row subplot:
    Row 1 (75% height): Candlestick / Line with SMA and Bollinger Band overlays
    Row 2 (25% height): Volume bars colored by daily price direction
    """
    df = hist.copy()
    if df.empty:
        return go.Figure()

    # Ensure index is datetime without timezone
    if hasattr(df.index, "tz_localize") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
        subplot_titles=(f"{company_name} ({symbol}) — Price Action", "Trading Volume")
    )

    # 1. Main Price Series (Candlestick or Line)
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color=GREEN_BULL,
                decreasing_line_color=RED_BEAR,
                increasing_fillcolor=GREEN_BULL,
                decreasing_fillcolor=RED_BEAR,
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close Price",
                line=dict(color=ACCENT_BLUE, width=2),
                mode="lines"
            ),
            row=1, col=1
        )

    # 2. Moving Average Overlays
    if show_sma:
        sma20 = df["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color=ORANGE_WARN, width=1.5)),
            row=1, col=1
        )

        if len(df) >= 50:
            sma50 = df["Close"].rolling(50).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color=PURPLE_LINE, width=1.5)),
                row=1, col=1
            )

        if len(df) >= 200:
            sma200 = df["Close"].rolling(200).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(color="#FF7B72", width=2, dash="dash")),
                row=1, col=1
            )

    # 3. Bollinger Bands Overlays
    if show_bollinger and len(df) >= 20:
        sma20_mid = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        upper_bb = sma20_mid + (std20 * 2.0)
        lower_bb = sma20_mid - (std20 * 2.0)

        fig.add_trace(
            go.Scatter(x=df.index, y=upper_bb, name="BB Upper", line=dict(color="#388BFD", width=1, dash="dot")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=lower_bb, name="BB Lower",
                line=dict(color="#388BFD", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(56, 139, 253, 0.05)"
            ),
            row=1, col=1
        )

    # 4. Volume Bar Subplot
    if "Volume" in df:
        colors = [GREEN_BULL if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED_BEAR for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.75,
            ),
            row=2, col=1
        )

    # Clean Dark Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PANEL_COLOR,
        font=dict(color=TEXT_COLOR, family="Segoe UI, -apple-system, sans-serif", size=11),
        margin=dict(l=10, r=10, t=35, b=10),
        height=520,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10)
        ),
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, zeroline=False)

    return fig


def create_rsi_chart(rsi_series: pd.Series, symbol: str) -> go.Figure:
    """Creates a dedicated RSI oscillator chart with overbought/oversold bands."""
    df = pd.DataFrame({"RSI": rsi_series.dropna()})
    if hasattr(df.index, "tz_localize") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    fig = go.Figure()

    # RSI Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", line=dict(color=ACCENT_BLUE, width=2))
    )

    # Overbought Line (70)
    fig.add_hline(y=70, line_dash="dash", line_color=RED_BEAR, annotation_text="Overbought (70)", annotation_position="top right")
    # Neutral Line (50)
    fig.add_hline(y=50, line_dash="dot", line_color=GRID_COLOR)
    # Oversold Line (30)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN_BULL, annotation_text="Oversold (30)", annotation_position="bottom right")

    fig.update_layout(
        title=f"{symbol} — Relative Strength Index (RSI 14)",
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PANEL_COLOR,
        font=dict(color=TEXT_COLOR, family="Segoe UI, sans-serif", size=11),
        margin=dict(l=10, r=10, t=35, b=10),
        height=240,
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        showlegend=False,
    )
    return fig


def create_macd_chart(macd_line: pd.Series, signal_line: pd.Series,
                      histogram: pd.Series, symbol: str) -> go.Figure:
    """Creates a dedicated MACD chart with line, signal, and histogram."""
    df = pd.DataFrame({
        "MACD": macd_line.dropna(),
        "Signal": signal_line.dropna(),
        "Hist": histogram.dropna(),
    })
    if hasattr(df.index, "tz_localize") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    fig = go.Figure()

    # Histogram
    hist_colors = [GREEN_BULL if v >= 0 else RED_BEAR for v in df["Hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Hist"], name="Histogram", marker_color=hist_colors, opacity=0.8)
    )

    # MACD Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD (12,26)", line=dict(color=ACCENT_BLUE, width=1.8))
    )

    # Signal Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Signal"], name="Signal (9)", line=dict(color=ORANGE_WARN, width=1.5, dash="dot"))
    )

    fig.update_layout(
        title=f"{symbol} — MACD Momentum (12, 26, 9)",
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PANEL_COLOR,
        font=dict(color=TEXT_COLOR, family="Segoe UI, sans-serif", size=11),
        margin=dict(l=10, r=10, t=35, b=10),
        height=260,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig
