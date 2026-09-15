import time
import math
import sys
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from langgraph.graph import START, END, StateGraph

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')



class NodeTelemetry(TypedDict):
    node: str
    status: str
    duration_ms: int
    summary: str


class StockScreenState(TypedDict):
    symbol: str
    period: str
    risk_preference: str  # 'Conservative', 'Moderate', 'Aggressive'
    market_data: Dict[str, Any]
    technicals: Dict[str, Any]
    fundamentals: Dict[str, Any]
    risk_check: Dict[str, Any]
    decision: Dict[str, Any]
    telemetry: List[NodeTelemetry]
    errors: List[str]


# -------------------------------------------------------------------
# Helper Calculation Functions
# -------------------------------------------------------------------

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates MACD and Signal Line."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


# -------------------------------------------------------------------
# Node 1: Fetch Market Data
# -------------------------------------------------------------------

def fetch_market_data_node(state: StockScreenState) -> Dict[str, Any]:
    t0 = time.time()
    symbol = state["symbol"].strip().upper()
    period = state.get("period", "6mo")
    telemetry = list(state.get("telemetry", []))
    errors = list(state.get("errors", []))

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        
        if history.empty:
            errors.append(f"No price history found for symbol '{symbol}'.")
            market_data = {"empty": True, "symbol": symbol}
            summary = "Failed: No price data"
        else:
            latest_price = float(history["Close"].iloc[-1])
            prev_price = float(history["Close"].iloc[-2]) if len(history) > 1 else latest_price
            daily_change = latest_price - prev_price
            daily_change_pct = (daily_change / prev_price) * 100 if prev_price else 0.0
            
            # Format history dates as strings for JSON/State serialization
            history_reset = history.reset_index()
            if "Date" in history_reset.columns:
                history_reset["DateStr"] = history_reset["Date"].dt.strftime("%Y-%m-%d")
            elif "Datetime" in history_reset.columns:
                history_reset["DateStr"] = history_reset["Datetime"].dt.strftime("%Y-%m-%d %H:%M")

            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            market_data = {
                "symbol": symbol,
                "name": info.get("shortName") or info.get("longName") or symbol,
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "latest_price": round(latest_price, 2),
                "prev_price": round(prev_price, 2),
                "daily_change": round(daily_change, 2),
                "daily_change_pct": round(daily_change_pct, 2),
                "volume": int(history["Volume"].iloc[-1]) if "Volume" in history.columns else 0,
                "avg_volume": int(history["Volume"].mean()) if "Volume" in history.columns else 0,
                "high_52w": round(float(info.get("fiftyTwoWeekHigh") or history["High"].max()), 2),
                "low_52w": round(float(info.get("fiftyTwoWeekLow") or history["Low"].min()), 2),
                "history_df": history_reset,
                "raw_info": info
            }
            summary = f"{len(history)} bars loaded · Price: ${latest_price:.2f} ({daily_change_pct:+.2f}%)"

    except Exception as e:
        errors.append(f"Market data fetch error: {str(e)}")
        market_data = {"empty": True, "symbol": symbol, "error": str(e)}
        summary = f"Error: {str(e)[:30]}"

    duration_ms = int((time.time() - t0) * 1000)
    telemetry.append({
        "node": "Market Data",
        "status": "completed" if not market_data.get("empty") else "warning",
        "duration_ms": duration_ms,
        "summary": summary
    })

    return {"market_data": market_data, "telemetry": telemetry, "errors": errors}


# -------------------------------------------------------------------
# Node 2: Compute Technical Indicators
# -------------------------------------------------------------------

def compute_technicals_node(state: StockScreenState) -> Dict[str, Any]:
    t0 = time.time()
    telemetry = list(state.get("telemetry", []))
    market_data = state.get("market_data", {})
    history_df = market_data.get("history_df")

    if history_df is None or history_df.empty or market_data.get("empty"):
        duration_ms = int((time.time() - t0) * 1000)
        telemetry.append({
            "node": "Technicals",
            "status": "skipped",
            "duration_ms": duration_ms,
            "summary": "Skipped: No price history"
        })
        return {"technicals": {"trend": "Neutral", "status": "skipped"}, "telemetry": telemetry}

    closes = history_df["Close"]
    sma_20 = closes.rolling(window=min(20, len(closes))).mean()
    sma_50 = closes.rolling(window=min(50, len(closes))).mean()
    rsi = compute_rsi(closes, window=min(14, len(closes)))
    macd, signal, hist = compute_macd(closes)

    latest_close = float(closes.iloc[-1])
    latest_sma20 = float(sma_20.iloc[-1]) if not math.isnan(sma_20.iloc[-1]) else latest_close
    latest_sma50 = float(sma_50.iloc[-1]) if not math.isnan(sma_50.iloc[-1]) else latest_close
    latest_rsi = float(rsi.iloc[-1]) if not math.isnan(rsi.iloc[-1]) else 50.0
    latest_macd = float(macd.iloc[-1]) if not math.isnan(macd.iloc[-1]) else 0.0
    latest_signal = float(signal.iloc[-1]) if not math.isnan(signal.iloc[-1]) else 0.0

    # Trend Determination
    if latest_close >= latest_sma20 >= latest_sma50:
        trend = "Positive"
        trend_score = 1.0
    elif latest_close < latest_sma20 < latest_sma50:
        trend = "Negative"
        trend_score = -1.0
    else:
        trend = "Neutral"
        trend_score = 0.0

    # Momentum (RSI)
    if latest_rsi >= 70:
        momentum = "Overbought"
        momentum_score = 0.5
    elif latest_rsi <= 30:
        momentum = "Oversold"
        momentum_score = 0.2
    elif latest_rsi >= 50:
        momentum = "Strong"
        momentum_score = 1.0
    else:
        momentum = "Moderate"
        momentum_score = 0.4

    # MACD Signal
    macd_signal = "Bullish" if latest_macd > latest_signal else "Bearish"

    # Save calculated indicator series back into DataFrame copy
    df_with_indicators = history_df.copy()
    df_with_indicators["SMA_20"] = sma_20
    df_with_indicators["SMA_50"] = sma_50
    df_with_indicators["RSI_14"] = rsi
    df_with_indicators["MACD"] = macd
    df_with_indicators["MACD_Signal"] = signal
    df_with_indicators["MACD_Hist"] = hist

    technicals = {
        "trend": trend,
        "trend_score": trend_score,
        "momentum": momentum,
        "momentum_score": momentum_score,
        "rsi_14": round(latest_rsi, 1),
        "sma_20": round(latest_sma20, 2),
        "sma_50": round(latest_sma50, 2),
        "macd": round(latest_macd, 3),
        "macd_signal": macd_signal,
        "indicators_df": df_with_indicators
    }

    duration_ms = int((time.time() - t0) * 1000)
    telemetry.append({
        "node": "Technicals",
        "status": "completed",
        "duration_ms": duration_ms,
        "summary": f"Trend: {trend} · RSI: {latest_rsi:.1f} ({momentum}) · MACD: {macd_signal}"
    })

    return {"technicals": technicals, "telemetry": telemetry}


# -------------------------------------------------------------------
# Node 3: Analyze Fundamentals
# -------------------------------------------------------------------

def analyze_fundamentals_node(state: StockScreenState) -> Dict[str, Any]:
    t0 = time.time()
    telemetry = list(state.get("telemetry", []))
    market_data = state.get("market_data", {})
    raw_info = market_data.get("raw_info", {})

    pe = raw_info.get("trailingPE") or raw_info.get("forwardPE")
    forward_pe = raw_info.get("forwardPE")
    peg = raw_info.get("pegRatio")
    market_cap = raw_info.get("marketCap")
    profit_margin = raw_info.get("profitMargins")
    revenue_growth = raw_info.get("revenueGrowth")
    dividend_yield = raw_info.get("dividendYield") or 0.0
    target_price = raw_info.get("targetMeanPrice")
    analyst_rating = raw_info.get("recommendationKey", "N/A")

    # Valuation Classification
    if pe is not None:
        if pe < 18:
            valuation = "Undervalued"
            val_score = 1.0
        elif pe <= 35:
            valuation = "Moderate"
            val_score = 0.7
        else:
            valuation = "Premium"
            val_score = 0.4
    else:
        valuation = "Moderate"
        val_score = 0.5

    fundamentals = {
        "valuation": valuation,
        "val_score": val_score,
        "pe": round(pe, 1) if pe is not None else None,
        "forward_pe": round(forward_pe, 1) if forward_pe is not None else None,
        "peg": round(peg, 2) if peg is not None else None,
        "market_cap": market_cap,
        "profit_margin": round(profit_margin * 100, 1) if profit_margin else None,
        "revenue_growth": round(revenue_growth * 100, 1) if revenue_growth else None,
        "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else 0.0,
        "target_price": round(target_price, 2) if target_price else None,
        "analyst_rating": analyst_rating.capitalize() if isinstance(analyst_rating, str) else "N/A"
    }

    duration_ms = int((time.time() - t0) * 1000)
    telemetry.append({
        "node": "Fundamentals",
        "status": "completed",
        "duration_ms": duration_ms,
        "summary": f"Valuation: {valuation} · P/E: {fundamentals['pe'] or 'N/A'} · Rating: {fundamentals['analyst_rating']}"
    })

    return {"fundamentals": fundamentals, "telemetry": telemetry}


# -------------------------------------------------------------------
# Node 4: Evaluate Risk
# -------------------------------------------------------------------

def evaluate_risk_node(state: StockScreenState) -> Dict[str, Any]:
    t0 = time.time()
    telemetry = list(state.get("telemetry", []))
    market_data = state.get("market_data", {})
    raw_info = market_data.get("raw_info", {})
    history_df = market_data.get("history_df")
    risk_pref = state.get("risk_preference", "Moderate")

    beta = raw_info.get("beta")
    
    # Calculate Max Drawdown from price series
    max_drawdown = 0.0
    volatility = 0.0
    if history_df is not None and not history_df.empty:
        closes = history_df["Close"]
        rolling_max = closes.cummax()
        drawdowns = (closes - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min() * 100) if not drawdowns.empty else 0.0
        
        daily_returns = closes.pct_change().dropna()
        volatility = float(daily_returns.std() * math.sqrt(252) * 100) if not daily_returns.empty else 0.0

    # Risk Level Determination
    effective_beta = beta if beta is not None else 1.0
    if effective_beta < 0.9 and abs(max_drawdown) < 15:
        risk_level = "Low"
        risk_score = 1.0
    elif effective_beta <= 1.3 and abs(max_drawdown) < 30:
        risk_level = "Medium"
        risk_score = 0.7
    else:
        risk_level = "High"
        risk_score = 0.4

    # Check risk tolerance compatibility
    if risk_pref == "Conservative" and risk_level == "High":
        risk_verdict = "Exceeds conservative threshold"
    elif risk_pref == "Moderate" and risk_level == "High":
        risk_verdict = "Elevated volatility"
    else:
        risk_verdict = "Within acceptable parameters"

    risk_check = {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "beta": round(effective_beta, 2),
        "volatility_annual_pct": round(volatility, 1),
        "max_drawdown_pct": round(max_drawdown, 1),
        "risk_verdict": risk_verdict,
        "risk_preference": risk_pref
    }

    duration_ms = int((time.time() - t0) * 1000)
    telemetry.append({
        "node": "Risk Check",
        "status": "completed",
        "duration_ms": duration_ms,
        "summary": f"Risk: {risk_level} (Beta: {risk_check['beta']}) · Max Drawdown: {max_drawdown:.1f}%"
    })

    return {"risk_check": risk_check, "telemetry": telemetry}


# -------------------------------------------------------------------
# Node 5: Screener Decision
# -------------------------------------------------------------------

def generate_decision_node(state: StockScreenState) -> Dict[str, Any]:
    t0 = time.time()
    telemetry = list(state.get("telemetry", []))
    
    technicals = state.get("technicals", {})
    fundamentals = state.get("fundamentals", {})
    risk_check = state.get("risk_check", {})
    market_data = state.get("market_data", {})

    if market_data.get("empty"):
        decision = {
            "verdict": "AVOID",
            "score": 0.0,
            "reasons": ["Unable to retrieve market price data for the specified ticker."]
        }
        duration_ms = int((time.time() - t0) * 1000)
        telemetry.append({
            "node": "Screener Decision",
            "status": "completed",
            "duration_ms": duration_ms,
            "summary": "Verdict: AVOID (Data Error)"
        })
        return {"decision": decision, "telemetry": telemetry}

    # Weightings: Technicals (35%), Fundamentals (35%), Risk (30%)
    tech_score = (technicals.get("trend_score", 0.0) + 1.0) / 2.0 * 0.6 + technicals.get("momentum_score", 0.5) * 0.4
    fund_score = fundamentals.get("val_score", 0.5)
    risk_score = risk_check.get("risk_score", 0.5)

    composite_score = (tech_score * 0.35) + (fund_score * 0.35) + (risk_score * 0.30)
    reasons = []

    # Reason 1: Technical & Momentum
    trend = technicals.get("trend", "Neutral")
    rsi = technicals.get("rsi_14", 50)
    if trend == "Positive":
        reasons.append(f"Bullish price structure above key moving averages with healthy momentum (RSI {rsi}).")
    elif trend == "Negative":
        reasons.append(f"Bearish price trend trading below key moving averages with lagging momentum (RSI {rsi}).")
    else:
        reasons.append(f"Neutral consolidation pattern with RSI stabilizing around {rsi}.")

    # Reason 2: Valuation & Fundamentals
    val = fundamentals.get("valuation", "Moderate")
    pe = fundamentals.get("pe")
    rating = fundamentals.get("analyst_rating", "Hold")
    if val == "Undervalued":
        reasons.append(f"Attractive valuation multiple (P/E {pe or 'N/A'}) with consensus '{rating}' rating.")
    elif val == "Premium":
        reasons.append(f"Trading at a premium multiple (P/E {pe or 'N/A'}); requires sustained earnings execution.")
    else:
        reasons.append(f"Fair market valuation (P/E {pe or 'N/A'}) aligned with sector benchmarks.")

    # Reason 3: Risk & Volatility
    risk_level = risk_check.get("risk_level", "Medium")
    max_dd = risk_check.get("max_drawdown_pct", 0.0)
    beta = risk_check.get("beta", 1.0)
    if risk_level == "Low":
        reasons.append(f"Low systemic risk (Beta {beta}) with contained peak-to-trough drawdown of {max_dd}%.")
    elif risk_level == "High":
        reasons.append(f"Elevated volatility (Beta {beta}) and historical drawdown of {max_dd}%.")
    else:
        reasons.append(f"Moderate risk profile (Beta {beta}) suitable for balanced risk allocations.")

    # Final Verdict
    if composite_score >= 0.70 and risk_check.get("risk_level") != "High":
        verdict = "PASS"
    elif composite_score >= 0.45:
        verdict = "WATCH"
    else:
        verdict = "AVOID"

    decision = {
        "verdict": verdict,
        "composite_score": round(composite_score * 100, 1),
        "reasons": reasons[:3]
    }

    duration_ms = int((time.time() - t0) * 1000)
    telemetry.append({
        "node": "Screener Decision",
        "status": "completed",
        "duration_ms": duration_ms,
        "summary": f"Verdict: {verdict} (Score: {decision['composite_score']}/100)"
    })

    return {"decision": decision, "telemetry": telemetry}


# -------------------------------------------------------------------
# Graph Assembly
# -------------------------------------------------------------------

def build_screening_graph():
    builder = StateGraph(StockScreenState)
    
    builder.add_node("market_data", fetch_market_data_node)
    builder.add_node("technicals", compute_technicals_node)
    builder.add_node("fundamentals", analyze_fundamentals_node)
    builder.add_node("risk_check", evaluate_risk_node)
    builder.add_node("decision", generate_decision_node)

    builder.add_edge(START, "market_data")
    builder.add_edge("market_data", "technicals")
    builder.add_edge("technicals", "fundamentals")
    builder.add_edge("fundamentals", "risk_check")
    builder.add_edge("risk_check", "decision")
    builder.add_edge("decision", END)

    return builder.compile()


# Global graph instance
screening_graph = build_screening_graph()


def run_stock_screener(symbol: str, period: str = "6mo", risk_preference: str = "Moderate") -> StockScreenState:
    """Executes the full LangGraph stock screening workflow."""
    initial_state: StockScreenState = {
        "symbol": symbol,
        "period": period,
        "risk_preference": risk_preference,
        "market_data": {},
        "technicals": {},
        "fundamentals": {},
        "risk_check": {},
        "decision": {},
        "telemetry": [],
        "errors": []
    }
    return screening_graph.invoke(initial_state)


if __name__ == "__main__":
    print("Testing SignalGraph LangGraph Pipeline with AAPL...")
    result = run_stock_screener("AAPL", "6mo", "Moderate")
    print(f"\nFinal Verdict: {result['decision']['verdict']} ({result['decision']['composite_score']}/100)")
    print("\nReasons:")
    for r in result['decision']['reasons']:
        print(f" - {r}")
    print("\nPipeline Telemetry:")
    for step in result['telemetry']:
        print(f" ✓ [{step['node']}] {step['duration_ms']}ms -> {step['summary']}")
