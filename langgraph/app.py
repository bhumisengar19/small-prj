"""
SignalGraph - Structured Stock Screening with LangGraph
Developer-built financial research dashboard with interactive workflow visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from pipeline import run_stock_screener, StockScreenState

# -----------------------------------------------------------------------------
# Page Configuration & Minimalist Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SignalGraph · Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist Financial Styling
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    
    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Clean background & container adjustments */
    .stApp {
        background-color: #f8fafc;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Minimalist Cards & Containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 0.825rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.75rem;
    }

    /* Header Title */
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.15rem;
        margin-bottom: 1rem;
    }

    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #334155;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 4px 10px;
        border-radius: 4px;
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
    }

    /* Signal Badges */
    .badge-pass {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .badge-watch {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .badge-avoid {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .badge-neutral {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.8rem;
    }

    /* Pipeline Node Visualization */
    .pipeline-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }
    
    .pipeline-node {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #1e293b;
    }
    
    .pipeline-arrow {
        color: #94a3b8;
        font-size: 0.8rem;
    }
    
    .node-check {
        color: #10b981;
        font-weight: 700;
    }
    
    .node-duration {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748b;
    }

    /* Result Box */
    .result-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
    }
    
    .verdict-large {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### **SignalGraph**")
    st.caption("Structured stock screening with LangGraph")
    st.markdown("---")

    # Watchlist Presets
    watchlist_options = {
        "Custom Input": None,
        "Mega-Cap Tech": "AAPL",
        "Semiconductors": "NVDA",
        "Enterprise Software": "MSFT",
        "EV & Auto": "TSLA",
        "Search & Cloud": "GOOGL",
        "E-Commerce & Cloud": "AMZN",
        "Value & Financials": "JPM",
        "Data & AI Analytics": "PLTR"
    }
    
    selected_watchlist = st.selectbox(
        "Watchlist Preset",
        list(watchlist_options.keys()),
        index=1
    )

    # Ticker Input
    default_symbol = watchlist_options[selected_watchlist] if watchlist_options[selected_watchlist] else "AAPL"
    symbol_input = st.text_input(
        "Stock Ticker",
        value=default_symbol,
        max_chars=10,
        help="Enter valid ticker symbol (e.g. AAPL, NVDA, MSFT, TSLA)"
    ).strip().upper()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        period_choice = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2,
            format_func=lambda x: {"1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years"}[x]
        )
    with col_s2:
        risk_pref = st.selectbox(
            "Risk Profile",
            ["Conservative", "Moderate", "Aggressive"],
            index=1
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run Screener", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b;">
            <b>Workflow Pipeline:</b><br>
            1. Market Data (yfinance)<br>
            2. Technicals (SMA/RSI/MACD)<br>
            3. Fundamentals (Valuation/Margins)<br>
            4. Risk Check (Beta/Drawdown)<br>
            5. Final Screener Decision
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# State & Execution Handler
# -----------------------------------------------------------------------------
if "last_state" not in st.session_state or run_btn:
    with st.spinner(f"Running LangGraph workflow for {symbol_input}..."):
        result_state = run_stock_screener(
            symbol=symbol_input,
            period=period_choice,
            risk_preference=risk_pref
        )
        st.session_state["last_state"] = result_state

state: StockScreenState = st.session_state["last_state"]
market_data = state.get("market_data", {})
technicals = state.get("technicals", {})
fundamentals = state.get("fundamentals", {})
risk_check = state.get("risk_check", {})
decision = state.get("decision", {})
telemetry = state.get("telemetry", [])
errors = state.get("errors", [])

# Compute total duration
total_duration_ms = sum(t["duration_ms"] for t in telemetry)

# -----------------------------------------------------------------------------
# Main Dashboard Header
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 2])

with header_col1:
    st.markdown('<p class="app-title">SignalGraph</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Trace the signal. Understand the decision.</p>', unsafe_allow_html=True)

with header_col2:
    st.markdown("<div style='text-align: right; padding-top: 0.5rem;'>", unsafe_allow_html=True)
    if not market_data.get("empty"):
        st.markdown(
            f"""
            <div class="status-badge">
                <span class="status-dot"></span>
                Workflow completed · {len(telemetry)} analysis steps · {total_duration_ms}ms
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="status-badge" style="color: #b91c1c;">
                <span class="status-dot" style="background: #ef4444;"></span>
                Workflow error · Check ticker symbol
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

if market_data.get("empty"):
    st.error(f"Unable to load data for '{state['symbol']}'. Please verify the ticker symbol in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# Ticker Headline & Summary Pill
# -----------------------------------------------------------------------------
chg_sign = "+" if market_data["daily_change"] >= 0 else ""
chg_color = "#10b981" if market_data["daily_change"] >= 0 else "#ef4444"

col_quote, col_metrics = st.columns([1.5, 3.5])

with col_quote:
    st.markdown(
        f"""
        <div style="padding: 0.5rem 0;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">
                {market_data['symbol']} 
                <span style="font-size: 0.95rem; font-weight: 400; color: #64748b; margin-left: 6px;">{market_data['name']}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">
                <span style="font-size: 1.75rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${market_data['latest_price']:.2f}</span>
                <span style="font-size: 0.95rem; font-weight: 600; color: {chg_color}; font-family: 'JetBrains Mono', monospace;">
                    {chg_sign}${market_data['daily_change']:.2f} ({chg_sign}{market_data['daily_change_pct']:.2f}%)
                </span>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                {market_data.get('exchange', 'N/A')} · {market_data.get('sector', 'N/A')} · Currency in {market_data.get('currency', 'USD')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("52W High", f"${market_data.get('high_52w', 0.0):.2f}")
    with m2:
        st.metric("52W Low", f"${market_data.get('low_52w', 0.0):.2f}")
    with m3:
        vol_str = f"{market_data.get('volume', 0) / 1e6:.1f}M" if market_data.get('volume') else "N/A"
        st.metric("Volume", vol_str)
    with m4:
        pe_str = f"{fundamentals.get('pe'):.1f}x" if fundamentals.get('pe') else "N/A"
        st.metric("P/E Ratio", pe_str)

st.markdown("<hr style='margin: 0.75rem 0 1.25rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 1: Price & Technical Charts (Plotly)
# -----------------------------------------------------------------------------
indicators_df = technicals.get("indicators_df")

if indicators_df is not None and not indicators_df.empty:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Candlestick or Line Chart
    x_dates = indicators_df["DateStr"] if "DateStr" in indicators_df.columns else indicators_df.index

    # Main Price (Candlestick)
    if "Open" in indicators_df.columns and "High" in indicators_df.columns:
        fig.add_trace(
            go.Candlestick(
                x=x_dates,
                open=indicators_df["Open"],
                high=indicators_df["High"],
                low=indicators_df["Low"],
                close=indicators_df["Close"],
                name="Price",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
                showlegend=False
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=indicators_df["Close"],
                mode="lines",
                name="Price",
                line=dict(color="#0f172a", width=1.5)
            ),
            row=1, col=1
        )

    # SMA 20
    if "SMA_20" in indicators_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=indicators_df["SMA_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="#2563eb", width=1.2)
            ),
            row=1, col=1
        )

    # SMA 50
    if "SMA_50" in indicators_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=indicators_df["SMA_50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="#d97706", width=1.2)
            ),
            row=1, col=1
        )

    # Subplot 2: RSI
    if "RSI_14" in indicators_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=indicators_df["RSI_14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="#7c3aed", width=1.2)
            ),
            row=2, col=1
        )
        
        # 70 Overbought & 30 Oversold Lines
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=0.9, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=0.9, row=2, col=1)

    fig.update_layout(
        height=450,
        margin=dict(l=40, r=20, t=10, b=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=11, color="#64748b"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Price ($)"),
        xaxis2=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(showgrid=True, gridcolor="#f1f5f9", title="RSI (14)", range=[10, 90])
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Section 2: Screening Snapshot & LangGraph Pipeline (2 Columns)
# -----------------------------------------------------------------------------
col_snap, col_pipe = st.columns([1, 1], gap="medium")

# 1. Screening Snapshot Table
with col_snap:
    st.markdown('<div class="section-header">Screening Snapshot</div>', unsafe_allow_html=True)
    
    trend_val = technicals.get("trend", "Neutral")
    trend_badge = "badge-pass" if trend_val == "Positive" else ("badge-avoid" if trend_val == "Negative" else "badge-neutral")

    rsi_val = technicals.get("rsi_14", 50.0)
    rsi_mom = technicals.get("momentum", "Moderate")
    mom_badge = "badge-pass" if rsi_mom in ["Strong", "Moderate"] else ("badge-watch" if rsi_mom == "Overbought" else "badge-avoid")

    val_val = fundamentals.get("valuation", "Moderate")
    val_badge = "badge-pass" if val_val == "Undervalued" else ("badge-watch" if val_val == "Moderate" else "badge-avoid")

    risk_val = risk_check.get("risk_level", "Medium")
    risk_badge = "badge-pass" if risk_val == "Low" else ("badge-watch" if risk_val == "Medium" else "badge-avoid")

    verdict_val = decision.get("verdict", "WATCH")
    verdict_badge = f"badge-{verdict_val.lower()}"

    st.markdown(
        f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden;">
            <thead>
                <tr style="border-bottom: 1px solid #e2e8f0; background: #f8fafc; color: #64748b; text-align: left;">
                    <th style="padding: 10px 14px;">Signal Dimension</th>
                    <th style="padding: 10px 14px;">Signal</th>
                    <th style="padding: 10px 14px;">Details</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 14px; font-weight: 500;">Technical Trend</td>
                    <td style="padding: 10px 14px;"><span class="{trend_badge}">{trend_val}</span></td>
                    <td style="padding: 10px 14px; color: #64748b; font-size: 0.8rem;">SMA20: ${technicals.get('sma_20', 'N/A')} · SMA50: ${technicals.get('sma_50', 'N/A')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 14px; font-weight: 500;">Momentum</td>
                    <td style="padding: 10px 14px;"><span class="{mom_badge}">{rsi_mom}</span></td>
                    <td style="padding: 10px 14px; color: #64748b; font-size: 0.8rem;">RSI: {rsi_val} · MACD: {technicals.get('macd_signal', 'N/A')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 14px; font-weight: 500;">Valuation</td>
                    <td style="padding: 10px 14px;"><span class="{val_badge}">{val_val}</span></td>
                    <td style="padding: 10px 14px; color: #64748b; font-size: 0.8rem;">P/E: {fundamentals.get('pe') or 'N/A'} · PEG: {fundamentals.get('peg') or 'N/A'}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 10px 14px; font-weight: 500;">Risk Level</td>
                    <td style="padding: 10px 14px;"><span class="{risk_badge}">{risk_val}</span></td>
                    <td style="padding: 10px 14px; color: #64748b; font-size: 0.8rem;">Beta: {risk_check.get('beta', '1.0')} · Max DD: {risk_check.get('max_drawdown_pct', 0.0)}%</td>
                </tr>
                <tr>
                    <td style="padding: 10px 14px; font-weight: 600;">Final Screen</td>
                    <td style="padding: 10px 14px;"><span class="{verdict_badge}">{verdict_val}</span></td>
                    <td style="padding: 10px 14px; color: #64748b; font-size: 0.8rem;">Score: {decision.get('composite_score', 0)}/100</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )

# 2. LangGraph Execution Pipeline
with col_pipe:
    st.markdown('<div class="section-header">LangGraph Workflow Pipeline</div>', unsafe_allow_html=True)
    
    # Horizontal Pipeline Node Indicators
    nodes_html = []
    for step in telemetry:
        nodes_html.append(
            f"""
            <div class="pipeline-node">
                <span class="node-check">✓</span>
                <span>{step['node']}</span>
                <span class="node-duration">{step['duration_ms']}ms</span>
            </div>
            """
        )
    
    pipeline_rendered = '<span class="pipeline-arrow">➔</span>'.join(nodes_html)
    st.markdown(
        f"""
        <div class="pipeline-container">
            {pipeline_rendered}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Node Telemetry Details
    with st.expander("Inspect Node State & Output Logs", expanded=False):
        for step in telemetry:
            st.markdown(
                f"""
                <div style="padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.8rem;">
                    <b>{step['node']}</b> <span style="color: #64748b; font-family: monospace;">({step['duration_ms']}ms)</span><br>
                    <span style="color: #334155;">{step['summary']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------------------------------------------------------
# Section 3: Final Screening Decision & Evidence
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Final Screening Result</div>', unsafe_allow_html=True)

verdict = decision.get("verdict", "WATCH")
score = decision.get("composite_score", 0.0)
reasons = decision.get("reasons", [])

verdict_class = f"badge-{verdict.lower()}"

res_col1, res_col2 = st.columns([1.2, 2.8], gap="medium")

with res_col1:
    st.markdown(
        f"""
        <div class="result-box" style="text-align: center; height: 100%;">
            <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
                Screen Result
            </div>
            <div style="margin: 8px 0;">
                <span class="{verdict_class}" style="font-size: 1.4rem; padding: 6px 16px;">{verdict}</span>
            </div>
            <div style="font-size: 0.85rem; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-top: 8px;">
                Composite Score: {score}/100
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                Risk Model: {risk_check.get('risk_preference', 'Moderate')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with res_col2:
    reasons_html = "".join([f"<li style='margin-bottom: 6px;'>{r}</li>" for r in reasons])
    st.markdown(
        f"""
        <div class="result-box" style="height: 100%;">
            <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                Pipeline Evidence & Key Drivers
            </div>
            <ul style="font-size: 0.85rem; color: #334155; margin: 0; padding-left: 1.25rem; line-height: 1.5;">
                {reasons_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("SignalGraph · Educational and research purposes only. Not individualized financial advice.")
