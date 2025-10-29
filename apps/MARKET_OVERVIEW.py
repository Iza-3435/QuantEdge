#!/usr/bin/env python3
"""
Market Overview Dashboard
Real-time market data with professional UI following production standards.
"""
import sys
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import requests
from rich.console import Console
from rich.columns import Columns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.components import (
    create_header,
    create_table,
    create_panel,
    format_change,
    format_percentage,
    show_error
)
from src.ui.config import COLORS, THEME, CONFIG
from utils.data_cache import DataCache, fetch_parallel, format_data_age

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


@dataclass(frozen=True)
class MarketData:
    """Immutable market data container."""
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    volume: Optional[int] = None
    sparkline: Optional[str] = None


INDICES: Dict[str, str] = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'Nasdaq',
    '^DJI': 'Dow Jones',
    '^RUT': 'Russell 2000',
    '^VIX': 'VIX',
    'GC=F': 'Gold',
    'CL=F': 'Crude Oil',
    '^TNX': '10Y Treasury'
}

SECTORS: Dict[str, str] = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLU': 'Utilities', 'XLC': 'Communications'
}

GLOBAL_MARKETS: Dict[str, str] = {
    '^FTSE': 'FTSE 100 (UK)', '^GDAXI': 'DAX (Germany)',
    '^N225': 'Nikkei 225 (Japan)', '000001.SS': 'Shanghai Composite',
    '^HSI': 'Hang Seng (Hong Kong)', '^AXJO': 'ASX 200 (Australia)'
}

COMMODITIES: Dict[str, str] = {
    'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil (WTI)',
    'BZ=F': 'Brent Oil', 'NG=F': 'Natural Gas', 'HG=F': 'Copper',
    'ZC=F': 'Corn', 'ZW=F': 'Wheat'
}

CURRENCIES: Dict[str, str] = {
    'DX-Y.NYB': 'US Dollar Index', 'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD', 'JPY=X': 'USD/JPY',
    'AUDUSD=X': 'AUD/USD', 'USDCAD=X': 'USD/CAD'
}


def create_sparkline(prices: List[float], width: int = 7) -> str:
    """
    Generate ASCII sparkline from price data.

    Args:
        prices: List of price values
        width: Number of characters in sparkline

    Returns:
        Colored sparkline string
    """
    if not prices or len(prices) < 2:
        return "━━━━━━━"

    prices = [p for p in prices if not pd.isna(p)]
    if len(prices) < 2:
        return "━━━━━━━"

    min_p, max_p = min(prices), max(prices)
    range_p = max_p - min_p if max_p != min_p else 1

    chars = '▁▂▃▄▅▆▇█'
    sparkline = ''.join(
        chars[min(int(((p - min_p) / range_p) * 7), 7)]
        for p in prices[-width:]
    )

    color = COLORS.up if prices[-1] > prices[0] else COLORS.down if prices[-1] < prices[0] else COLORS.neutral
    return f"[{color}]{sparkline}[/{color}]"


def fetch_symbol_data(symbol: str, period: str = '5d') -> Optional[MarketData]:
    """
    Fetch market data for a single symbol with caching.

    Args:
        symbol: Ticker symbol
        period: Time period for historical data

    Returns:
        MarketData object or None if fetch fails
    """
    cache_key = f"market_data_{symbol}_{period}"
    cached = DataCache.get(cache_key)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        fast = ticker.fast_info

        current_price = fast.get('lastPrice')
        prev_close = fast.get('previousClose')

        if not current_price or not prev_close:
            hist = ticker.history(period='1d')
            if hist.empty:
                return None
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[0] if len(hist) > 1 else current_price

        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        hist = ticker.history(period=period)
        sparkline = create_sparkline(hist['Close'].tolist()) if not hist.empty else None

        data = MarketData(
            symbol=symbol,
            name=INDICES.get(symbol, symbol),
            price=current_price,
            change=change,
            change_pct=change_pct,
            volume=fast.get('lastVolume'),
            sparkline=sparkline
        )

        DataCache.set(cache_key, data, ttl=CONFIG.cache_ttl)
        return data

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def fetch_all_data(symbols: Dict[str, str]) -> List[MarketData]:
    """
    Fetch data for multiple symbols in parallel.

    Args:
        symbols: Dict mapping symbols to names

    Returns:
        List of MarketData objects
    """
    results = fetch_parallel(
        [(sym, '5d') for sym in symbols.keys()],
        fetch_symbol_data,
        max_workers=10
    )
    return [r for r in results if r is not None]


def create_market_table(data: List[MarketData], title: str) -> None:
    """
    Create and display formatted market data table.

    Args:
        data: List of MarketData objects
        title: Table title
    """
    if not data:
        console.print(f"[{COLORS.dim}]No data available for {title}[/{COLORS.dim}]")
        return

    columns = [
        {'name': 'Symbol', 'style': 'white', 'width': 12, 'justify': 'left'},
        {'name': 'Name', 'style': 'white', 'width': 25, 'justify': 'left'},
        {'name': 'Price', 'style': 'white', 'width': 12, 'justify': 'right'},
        {'name': 'Change', 'style': 'white', 'width': 12, 'justify': 'right'},
        {'name': '%', 'style': 'white', 'width': 10, 'justify': 'right'},
        {'name': 'Trend', 'style': 'white', 'width': 10, 'justify': 'center'},
    ]

    rows = []
    for item in data:
        change_text, change_color = format_change(item.change)
        pct_text, pct_color = format_percentage(item.change_pct)

        rows.append([
            item.symbol,
            item.name[:23],
            f"${item.price:,.2f}",
            f"[{change_color}]{change_text}[/{change_color}]",
            f"[{pct_color}]{pct_text}[/{pct_color}]",
            item.sparkline or "━━━━━━━"
        ])

    table = create_table(columns, rows)
    panel = create_panel(table, title=title)
    console.print(panel)
    console.print()


def fetch_top_news(count: int = 5) -> List[Dict[str, str]]:
    """
    Fetch latest market news.

    Args:
        count: Number of news items to fetch

    Returns:
        List of news dicts with title and source
    """
    try:
        import os
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            return []

        url = 'https://newsapi.org/v2/top-headlines'
        params = {
            'apiKey': api_key,
            'category': 'business',
            'country': 'us',
            'pageSize': count
        }

        response = requests.get(url, params=params, timeout=CONFIG.timeout)
        response.raise_for_status()

        articles = response.json().get('articles', [])
        return [
            {'title': a['title'], 'source': a['source']['name']}
            for a in articles[:count]
        ]

    except Exception as e:
        logger.warning(f"Failed to fetch news: {e}")
        return []


def display_news() -> None:
    """Display market news panel."""
    news = fetch_top_news(5)

    if not news:
        return

    columns = [
        {'name': 'Source', 'style': 'cyan', 'width': 20, 'justify': 'left'},
        {'name': 'Headline', 'style': 'white', 'justify': 'left'},
    ]

    rows = [[item['source'], item['title'][:80]] for item in news]

    table = create_table(columns, rows)
    panel = create_panel(table, title="📰 BREAKING NEWS")
    console.print(panel)
    console.print()


def main() -> None:
    """Main entry point."""
    try:
        console.clear()

        header = create_header(
            title="MARKET OVERVIEW DASHBOARD",
            subtitle="Real-time market data with professional analytics"
        )
        console.print(header)
        console.print()

        console.print("[white]Fetching market data...[/white]\n")

        indices_data = fetch_all_data(INDICES)
        create_market_table(indices_data, "📊 MAJOR INDICES")

        sectors_data = fetch_all_data(SECTORS)
        create_market_table(sectors_data, "🏭 SECTOR PERFORMANCE")

        global_data = fetch_all_data(GLOBAL_MARKETS)
        create_market_table(global_data, "🌍 GLOBAL MARKETS")

        commodities_data = fetch_all_data(COMMODITIES)
        create_market_table(commodities_data, "📦 COMMODITIES")

        currencies_data = fetch_all_data(CURRENCIES)
        create_market_table(currencies_data, "💱 CURRENCIES")

        display_news()

        console.print(f"[{COLORS.dim}]Last updated: {datetime.now().strftime(CONFIG.datetime_format)}[/{COLORS.dim}]")
        console.print()

    except KeyboardInterrupt:
        console.print("\n[white]Exiting...[/white]\n")
        sys.exit(0)

    except Exception as e:
        show_error(console, f"Unexpected error: {e}", "Error")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
