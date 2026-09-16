import json
import pytest
from tool import simple_screener, get_stock_details, get_stock_news, VALID_SCREENERS


def test_valid_screeners_list():
    assert len(VALID_SCREENERS) >= 10
    assert "day_gainers" in VALID_SCREENERS
    assert "day_losers" in VALID_SCREENERS
    assert "most_actives" in VALID_SCREENERS
    assert "undervalued_growth_stocks" in VALID_SCREENERS


def test_simple_screener_invalid_input():
    result = simple_screener.invoke({"screen_type": "invalid_screener_name_12345"})
    assert "Error: Invalid screen_type" in result
    assert "Valid options are:" in result


def test_tool_definitions():
    # Verify LangChain @tool metadata
    assert simple_screener.name == "simple_screener"
    assert get_stock_details.name == "get_stock_details"
    assert get_stock_news.name == "get_stock_news"
    assert "Yahoo Finance" in simple_screener.description
