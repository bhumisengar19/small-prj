import pytest
import pandas as pd
import numpy as np
from pipeline import (
    compute_rsi,
    compute_macd,
    build_screening_graph,
    StockScreenState,
    fetch_market_data_node,
    compute_technicals_node,
    analyze_fundamentals_node,
    evaluate_risk_node,
    generate_decision_node,
)


def test_compute_rsi_constant_series():
    # If prices never change, RSI should default to 50
    prices = pd.Series([100.0] * 20)
    rsi = compute_rsi(prices, window=14)
    assert len(rsi) == len(prices)
    assert rsi.iloc[-1] == 50.0


def test_compute_rsi_strictly_increasing():
    # If prices strictly increase with no losses, RSI should be 100
    prices = pd.Series([float(i) for i in range(1, 25)])
    rsi = compute_rsi(prices, window=14)
    assert rsi.iloc[-1] == 100.0


def test_compute_rsi_strictly_decreasing():
    # If prices strictly decrease with no gains, RSI should be 0
    prices = pd.Series([float(100 - i) for i in range(25)])
    rsi = compute_rsi(prices, window=14)
    assert rsi.iloc[-1] == 0.0


def test_compute_macd_output_shapes():
    prices = pd.Series([100.0 + (i % 5) * 2.0 for i in range(60)])
    macd, signal, hist = compute_macd(prices, fast=12, slow=26, signal=9)
    assert len(macd) == len(prices)
    assert len(signal) == len(prices)
    assert len(hist) == len(prices)
    np.testing.assert_allclose(hist.values, (macd - signal).values)


def test_build_screening_graph():
    graph = build_screening_graph()
    assert graph is not None
    # Verify graph can compile without exception
    assert hasattr(graph, "invoke")


def test_generate_decision_node_avoid_on_empty():
    state: StockScreenState = {
        "symbol": "INVALIDTICKER999",
        "period": "6mo",
        "risk_preference": "Moderate",
        "market_data": {"empty": True},
        "technicals": {},
        "fundamentals": {},
        "risk_check": {},
        "decision": {},
        "telemetry": [],
        "errors": ["Failed to fetch data"],
    }
    result = generate_decision_node(state)
    assert "decision" in result
    assert result["decision"]["verdict"] == "AVOID"
    assert result["decision"]["composite_score"] == 0.0


def test_generate_decision_node_pass_verdict():
    state: StockScreenState = {
        "symbol": "TEST",
        "period": "6mo",
        "risk_preference": "Moderate",
        "market_data": {"empty": False},
        "technicals": {
            "trend": "Positive",
            "trend_score": 1.0,
            "momentum": "Strong",
            "momentum_score": 1.0,
            "rsi_14": 58.5,
        },
        "fundamentals": {
            "valuation": "Undervalued",
            "val_score": 1.0,
            "pe": 15.2,
            "analyst_rating": "Buy",
        },
        "risk_check": {
            "risk_level": "Low",
            "risk_score": 1.0,
            "beta": 0.85,
            "max_drawdown_pct": -10.5,
        },
        "decision": {},
        "telemetry": [],
        "errors": [],
    }
    result = generate_decision_node(state)
    assert result["decision"]["verdict"] == "PASS"
    assert result["decision"]["composite_score"] >= 70.0
    assert len(result["decision"]["reasons"]) > 0
