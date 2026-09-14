import json
import logging
from typing import Optional, List, Dict, Any
from langchain.tools import tool 
import yfinance as yf 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock_tools")

VALID_SCREENERS = [
    "aggressive_small_caps",
    "day_gainers",
    "day_losers",
    "growth_technology_stocks",
    "most_actives",
    "most_shorted_stocks",
    "small_cap_gainers",
    "undervalued_growth_stocks",
    "undervalued_large_caps",
    "conservative_foreign_funds",
    "high_yield_bond",
    "portfolio_anchors",
    "solid_large_growth_funds",
    "solid_midcap_growth_funds",
    "top_mutual_funds"
]


@tool
def simple_screener(screen_type: str, offset: int = 0, count: int = 5) -> str: 
    """Returns screened assets (stocks, funds, bonds) based on predefined Yahoo Finance criteria.

    Args:
        screen_type: One of the supported screener categories:
            - aggressive_small_caps
            - day_gainers
            - day_losers
            - growth_technology_stocks
            - most_actives
            - most_shorted_stocks
            - small_cap_gainers
            - undervalued_growth_stocks
            - undervalued_large_caps
            - conservative_foreign_funds
            - high_yield_bond
            - portfolio_anchors
            - solid_large_growth_funds
            - solid_midcap_growth_funds
            - top_mutual_funds
        offset: Pagination start index (default: 0).
        count: Number of results to retrieve (default: 5, max: 25).

    Returns:
        JSON-formatted string containing a list of matching asset quotes with key valuation metrics.
    """
    clean_type = screen_type.strip().lower()
    
    # Validate screener type
    if clean_type not in yf.PREDEFINED_SCREENER_QUERIES:
        return (
            f"Error: Invalid screen_type '{screen_type}'. "
            f"Valid options are: {', '.join(VALID_SCREENERS)}"
        )
    
    try:
        limit = max(1, min(count, 25))
        query = yf.PREDEFINED_SCREENER_QUERIES[clean_type]['query']
        result = yf.screen(query, offset=max(0, offset), size=limit)
        
        # Save raw output for debugging / offline inspection
        try:
            with open('output.json', 'w', encoding='utf-8') as f: 
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save output.json: {e}")
        
        fields = [
            "symbol", "shortName", "regularMarketPrice", "bid", "ask", 
            "exchange", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", 
            "averageAnalystRating", "dividendYield", "marketCap"
        ]
        
        output_data = []
        for stock_detail in result.get('quotes', []): 
            details = {}
            for key in fields:
                if key in stock_detail:
                    details[key] = stock_detail[key]
            output_data.append(details) 
        
        if not output_data:
            return f"No assets found for screener criteria '{screen_type}' at offset {offset}."
            
        return json.dumps({
            "screener": clean_type,
            "total_returned": len(output_data),
            "results": output_data
        }, indent=2)

    except Exception as e:
        logger.error(f"Error executing screener: {e}")
        return f"Error fetching screener data for '{screen_type}': {str(e)}"


@tool
def get_stock_details(symbol: str) -> str:
    """Fetches comprehensive company and financial metrics for a specific stock ticker.

    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL', 'NVDA', 'TSLA', 'MSFT').

    Returns:
        JSON-formatted string containing detailed valuation, market cap, analyst targets, and business profile.
    """
    ticker_clean = symbol.strip().upper()
    try:
        ticker = yf.Ticker(ticker_clean)
        info = ticker.info
        
        if not info or 'symbol' not in info and 'shortName' not in info:
            return f"Error: No data found for ticker symbol '{ticker_clean}'."

        data = {
            "symbol": info.get("symbol", ticker_clean),
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "pegRatio": info.get("pegRatio"),
            "dividendYield": info.get("dividendYield"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "recommendationKey": info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "businessSummary": (info.get("longBusinessSummary") or "")[:400] + "..." if info.get("longBusinessSummary") else None
        }
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"Error fetching ticker info for {ticker_clean}: {e}")
        return f"Error fetching data for ticker '{ticker_clean}': {str(e)}"


@tool
def get_stock_news(symbol: str, max_articles: int = 3) -> str:
    """Fetches latest news headlines and articles for a specific stock ticker.

    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL', 'NVDA', 'TSLA').
        max_articles: Maximum number of recent news stories to return (default: 3).

    Returns:
        JSON-formatted string containing recent news titles, publishers, links, and publish timestamps.
    """
    ticker_clean = symbol.strip().upper()
    try:
        ticker = yf.Ticker(ticker_clean)
        news_items = ticker.news or []
        
        if not news_items:
            return f"No recent news found for ticker '{ticker_clean}'."
            
        articles = []
        for item in news_items[:max(1, min(max_articles, 10))]:
            # Yahoo Finance news structure can vary by version
            content = item.get("content", item) if isinstance(item, dict) else {}
            title = content.get("title") or item.get("title")
            publisher = (content.get("provider", {}) or {}).get("displayName") or item.get("publisher")
            link = (content.get("canonicalUrl", {}) or {}).get("url") or item.get("link")
            pub_time = content.get("pubDate") or item.get("providerPublishTime")

            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": pub_time
            })
            
        return json.dumps({"symbol": ticker_clean, "articles": articles}, indent=2)
    except Exception as e:
        logger.error(f"Error fetching news for {ticker_clean}: {e}")
        return f"Error fetching news for '{ticker_clean}': {str(e)}"


if __name__ == '__main__': 
    print("Testing simple_screener:")
    print(simple_screener.invoke({"screen_type": "day_gainers", "offset": 0, "count": 2}))
    print("\nTesting get_stock_details:")
    print(get_stock_details.invoke({"symbol": "AAPL"}))

