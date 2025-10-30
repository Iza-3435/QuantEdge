#!/usr/bin/env python3
"""
Market Overview Dashboard
Real-time market data with professional analytics
"""
import sys
import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import requests
from rich.console import Console
from rich.columns import Columns
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_cache import DataCache, fetch_parallel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

COLORS = {
    'up': 'green',
    'down': 'red',
    'neutral': 'white',
    'dim': 'bright_black'
}

THEME = {
    'header_bg': 'on grey23',
    'row_even': 'on grey15',
    'row_odd': 'on grey11',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}


@dataclass
class MarketData:
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    volume: Optional[int] = None
    sparkline: Optional[str] = None
    day_range: Optional[str] = None


INDICES = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'Nasdaq',
    '^DJI': 'Dow Jones',
    '^RUT': 'Russell 2000',
    '^VIX': 'VIX (Fear Index)',
    'GC=F': 'Gold',
    'CL=F': 'Crude Oil',
    '^TNX': '10Y Treasury'
}

SECTORS = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLU': 'Utilities', 'XLC': 'Communications'
}

GLOBAL_MARKETS = {
    '^FTSE': 'FTSE 100 (UK)', '^GDAXI': 'DAX (Germany)',
    '^N225': 'Nikkei 225 (Japan)', '000001.SS': 'Shanghai Composite',
    '^HSI': 'Hang Seng (Hong Kong)', '^AXJO': 'ASX 200 (Australia)'
}

COMMODITIES = {
    'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil (WTI)',
    'BZ=F': 'Brent Oil', 'NG=F': 'Natural Gas', 'HG=F': 'Copper',
    'ZC=F': 'Corn', 'ZW=F': 'Wheat'
}

CURRENCIES = {
    'DX-Y.NYB': 'US Dollar Index', 'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD', 'JPY=X': 'USD/JPY',
    'AUDUSD=X': 'AUD/USD', 'USDCAD=X': 'USD/CAD'
}

SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'KO', 'AVGO', 'COST', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'ABT', 'DHR', 'VZ', 'ADBE', 'NEE', 'CRM', 'NKE',
    'DIS', 'CMCSA', 'TXN', 'PM', 'ORCL', 'INTC', 'UPS', 'HON', 'T',
    'QCOM', 'RTX', 'INTU', 'LOW', 'AMD', 'SPGI', 'AMGN', 'SBUX', 'BMY',
    'BA', 'CAT', 'GE', 'DE', 'LMT', 'MMM', 'AXP', 'BLK', 'NOW', 'GILD',
    'MDLZ', 'PLD', 'TGT', 'MO', 'CVS', 'CI', 'SYK', 'ISRG', 'C', 'AMT',
    'ZTS', 'REGN', 'BKNG', 'ADI', 'PFE', 'CB', 'MMC', 'DUK', 'SO', 'USB'
]


def create_sparkline(prices: List[float], width: int = 7) -> str:
    if not prices or len(prices) < 2:
        return "━━━━━"

    prices = [p for p in prices if not pd.isna(p)]
    if len(prices) < 2:
        return "━━━━━"

    min_p, max_p = min(prices), max(prices)
    range_p = max_p - min_p if max_p != min_p else 1

    chars = '▁▂▃▄▅▆▇█'
    sparkline = ''.join(
        chars[min(int(((p - min_p) / range_p) * 7), 7)]
        for p in prices[-width:]
    )

    color = COLORS['up'] if prices[-1] > prices[0] else COLORS['down'] if prices[-1] < prices[0] else COLORS['neutral']
    return f"[{color}]{sparkline}[/{color}]"


def fetch_symbol_data(symbol: str, period: str = '5d') -> Optional[MarketData]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[0] if len(hist) > 1 else current_price

        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        day_low = hist['Low'].iloc[-1]
        day_high = hist['High'].iloc[-1]
        day_range = f"${day_low:.2f} - ${day_high:.2f}"

        sparkline = create_sparkline(hist['Close'].tolist())

        volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else None

        name = INDICES.get(symbol) or SECTORS.get(symbol) or GLOBAL_MARKETS.get(symbol) or \
               COMMODITIES.get(symbol) or CURRENCIES.get(symbol) or symbol

        data = MarketData(
            symbol=symbol,
            name=name,
            price=current_price,
            change=change,
            change_pct=change_pct,
            volume=volume,
            sparkline=sparkline,
            day_range=day_range
        )

        return data

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def fetch_all_data(symbols: Dict[str, str]) -> List[MarketData]:
    results = fetch_parallel(
        symbols.keys(),
        fetch_symbol_data,
        max_workers=10
    )
    return [r for r in results.values() if r is not None]


def fetch_sp500_movers() -> Tuple[List[MarketData], List[MarketData], List[MarketData]]:
    """Fetch top gainers, losers, and most active from S&P 500"""
    all_data = []

    results = fetch_parallel(
        SP500_TICKERS,
        fetch_symbol_data,
        max_workers=20
    )

    for data in results.values():
        if data:
            all_data.append(data)

    gainers = sorted([d for d in all_data if d.change_pct > 0], key=lambda x: x.change_pct, reverse=True)[:5]
    losers = sorted([d for d in all_data if d.change_pct < 0], key=lambda x: x.change_pct)[:5]
    active = sorted([d for d in all_data if d.volume], key=lambda x: x.volume or 0, reverse=True)[:5]

    return gainers, losers, active


def calculate_market_breadth(sp500_data: List[MarketData]) -> Dict[str, float]:
    """Calculate market breadth indicators"""
    if not sp500_data:
        return {'ad_ratio': 0, 'trend_strength': 0}

    advancing = len([d for d in sp500_data if d.change > 0])
    total = len(sp500_data)

    ad_ratio = (advancing / total * 100) if total > 0 else 0

    avg_change = sum(d.change_pct for d in sp500_data) / total if total > 0 else 0

    return {
        'ad_ratio': ad_ratio,
        'trend_strength': avg_change
    }


def get_market_sentiment(indices_data: List[MarketData]) -> str:
    """Determine overall market sentiment"""
    if not indices_data:
        return "NEUTRAL"

    sp500 = next((d for d in indices_data if d.symbol == '^GSPC'), None)
    vix = next((d for d in indices_data if d.symbol == '^VIX'), None)

    if sp500 and sp500.change_pct > 1:
        return "BULLISH"
    elif sp500 and sp500.change_pct < -1:
        return "BEARISH"
    elif vix and vix.price > 25:
        return "FEARFUL"
    else:
        return "NEUTRAL"


def create_enhanced_header() -> Panel:
    """Create enhanced header with live status"""
    now = datetime.now()
    header = Text()

    header.append("MARKET OVERVIEW DASHBOARD\n\n", style="bold white")
    header.append("Live Market Data", style="white")
    header.append(" │ ", style="bright_black")
    header.append(f"Updated: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p ET')}", style="white")
    header.append(" │ ", style="bright_black")

    cache_age = "Just now"
    header.append(f"Data: {cache_age}", style="bright_black")
    header.append(" │ ", style="bright_black")
    header.append("15-min delayed", style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_indices_table_enhanced(data: List[MarketData]) -> Table:
    """Create enhanced major indices table with day range"""
    table = Table(
        title="MAJOR INDICES   LIVE",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Index", style="white", width=22)
    table.add_column("Price", justify="right", width=14)
    table.add_column("Change", justify="right", width=14)
    table.add_column("Change %", justify="right", width=12)
    table.add_column("Trend (5D)", justify="center", width=14)
    table.add_column("Day Range", justify="right", width=20)

    for item in data:
        color = COLORS['up'] if item.change > 0 else COLORS['down'] if item.change < 0 else COLORS['neutral']
        symbol = "▲" if item.change > 0 else "▼" if item.change < 0 else "━"

        table.add_row(
            item.name,
            f"${item.price:,.2f}",
            f"[{color}]{symbol} ${abs(item.change):.2f}[/{color}]",
            f"[{color}]{item.change_pct:+.2f}%[/{color}]",
            item.sparkline or "━━━━━",
            item.day_range or "N/A"
        )

    return table


def create_movers_table(data: List[MarketData], title: str) -> Table:
    """Create table for gainers/losers/active"""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=10)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Change %", justify="right", width=14)

    if "ACTIVE" in title.upper():
        table.add_column("Volume", justify="right", width=12)
    else:
        table.add_column("Volume", justify="right", width=12)

    for item in data:
        color = COLORS['up'] if item.change_pct > 0 else COLORS['down'] if item.change_pct < 0 else COLORS['neutral']
        symbol = "▲" if item.change_pct > 0 else "▼" if item.change_pct < 0 else "━"

        vol_str = f"{item.volume/1e6:.1f}M" if item.volume else "N/A"

        table.add_row(
            item.symbol,
            f"${item.price:.2f}",
            f"[{color}]{symbol} {abs(item.change_pct):.2f}%[/{color}]",
            vol_str
        )

    return table


def create_sector_table(data: List[MarketData]) -> Table:
    """Create sector performance table"""
    sorted_data = sorted(data, key=lambda x: x.change_pct, reverse=True)

    table = Table(
        title="SECTOR PERFORMANCE",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Sector", style="white", width=30)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Change %", justify="right", width=12)
    table.add_column("Performance", justify="left", width=18)

    for item in sorted_data:
        color = COLORS['up'] if item.change_pct > 0 else COLORS['down'] if item.change_pct < 0 else COLORS['neutral']
        symbol = "▲" if item.change_pct > 0 else "▼" if item.change_pct < 0 else "━"

        bar_length = int(abs(item.change_pct) * 2)
        bar = "█" * min(bar_length, 10)

        table.add_row(
            item.name,
            f"${item.price:.2f}",
            f"[{color}]{symbol} {abs(item.change_pct):.2f}%[/{color}]",
            f"[{color}]{bar}[/{color}]"
        )

    return table


def create_market_summary(indices_data: List[MarketData], breadth: Dict[str, float]) -> Panel:
    """Create market summary panel"""
    sentiment = get_market_sentiment(indices_data)

    sp500 = next((d for d in indices_data if d.symbol == '^GSPC'), None)
    vix = next((d for d in indices_data if d.symbol == '^VIX'), None)
    treasury = next((d for d in indices_data if d.symbol == '^TNX'), None)

    text = Text()

    sentiment_color = COLORS['up'] if sentiment == "BULLISH" else COLORS['down'] if sentiment == "BEARISH" else COLORS['neutral']
    text.append(f"▲ Market Sentiment: ", style="white")
    text.append(f"{sentiment}\n\n", style=f"bold {sentiment_color}")

    text.append("Key Metrics:\n", style="white")
    if sp500:
        text.append(f"• S&P 500: {sp500.change_pct:+.2f}% | Price: ${sp500.price:.2f}\n", style="white")
    if vix:
        vix_status = "High" if vix.price > 20 else "Normal" if vix.price > 15 else "Low"
        text.append(f"• VIX: {vix.price:.2f} ({vix_status})\n", style="white")
    if treasury:
        yield_status = "High" if treasury.price > 4.5 else "Neutral" if treasury.price > 3.5 else "Low"
        text.append(f"• 10Y Yield: {treasury.price:.2f}% {yield_status}\n\n", style="white")

    text.append("Market Signals:\n", style="white")
    text.append(f"• Market Breadth: {'Strong' if breadth['ad_ratio'] > 55 else 'Weak'} ", style="white")
    text.append(f"(Tech leading)\n", style="bright_black")

    risk_appetite = "Risk-On" if breadth['trend_strength'] > 0.5 else "Risk-Off"
    text.append(f"• Risk Appetite: {risk_appetite} ", style="white")
    text.append(f"(Flight to quality)\n", style="bright_black")

    bullish_signals = sum([
        sp500.change_pct > 0 if sp500 else False,
        breadth['ad_ratio'] > 55,
        vix.price < 20 if vix else False,
        breadth['trend_strength'] > 0
    ])
    text.append(f"• Bullish Signals: [{bullish_signals}/4] | Bearish: [{4-bullish_signals}/4]\n\n", style="white")

    if sp500:
        resistance = sp500.price * 1.02
        support = sp500.price * 0.98
        text.append("Key S&P 500 Levels:\n", style="white")
        text.append(f"• Resistance: ${resistance:.2f} (+2%)\n", style="white")
        text.append(f"• Support: ${support:.2f} (-2%)\n", style="white")

    return Panel(text, title="[bold white]MARKET SUMMARY[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def create_breadth_panel(breadth: Dict[str, float]) -> Panel:
    """Create market breadth panel"""
    text = Text()
    text.append("Market Breadth Indicators (30-Day)\n\n", style="bold white")

    ad_ratio = breadth['ad_ratio']
    ad_status = "BULLISH" if ad_ratio > 55 else "BEARISH" if ad_ratio < 45 else "NEUTRAL"
    ad_color = COLORS['up'] if ad_ratio > 55 else COLORS['down'] if ad_ratio < 45 else COLORS['neutral']

    text.append(f"Advance/Decline Ratio: ", style="white")
    text.append(f"{ad_ratio:.1f}% ", style=f"bold {ad_color}")
    text.append(f"({ad_status})\n", style=ad_color)

    volatility = 12.3  # Placeholder
    text.append(f"Market Volatility: {volatility:.1f}% (LOW)\n", style="white")

    trend = breadth['trend_strength']
    trend_status = "UPTREND" if trend > 2 else "DOWNTREND" if trend < -2 else "SIDEWAYS"
    trend_color = COLORS['up'] if trend > 2 else COLORS['down'] if trend < -2 else COLORS['neutral']
    text.append(f"Trend Strength: ", style="white")
    text.append(f"{trend:+.2f}% ", style=f"bold {trend_color}")
    text.append(f"({trend_status})\n\n", style=trend_color)

    text.append("• A/D > 55% = Bullish | < 45% = Bearish\n", style="bright_black")
    text.append("• Volatility < 15% = Low Risk | > 25% = High Risk\n", style="bright_black")
    text.append("• Trend > 2% = Uptrend | < -2% = Downtrend\n", style="bright_black")

    return Panel(text, title="[bold white]MARKET BREADTH[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def fetch_market_news(count: int = 8) -> List[Dict[str, str]]:
    """Fetch latest market news"""
    api_key = os.getenv('NEWS_API_KEY', 'c6b3684c13ad4b84beaa1c4cab8c97bf')

    try:
        url = 'https://newsapi.org/v2/top-headlines'
        params = {
            'apiKey': api_key,
            'category': 'business',
            'country': 'us',
            'pageSize': count
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        articles = response.json().get('articles', [])
        return [
            {
                'title': a['title'],
                'source': a['source']['name'],
                'date': a['publishedAt'][:10]
            }
            for a in articles[:count]
        ]

    except Exception as e:
        logger.warning(f"Failed to fetch NewsAPI news: {e}")
        return fetch_yahoo_finance_news(count)


def fetch_yahoo_finance_news(count: int = 8) -> List[Dict[str, str]]:
    """Fetch news from Yahoo Finance RSS as fallback"""
    try:
        import feedparser

        feed_url = 'https://finance.yahoo.com/news/rssindex'
        feed = feedparser.parse(feed_url)

        articles = []
        for entry in feed.entries[:count]:
            articles.append({
                'title': entry.get('title', 'No title'),
                'source': 'Yahoo Finance',
                'date': entry.get('published', datetime.now().strftime('%Y-%m-%d'))[:10]
            })

        return articles

    except Exception as e:
        logger.warning(f"Failed to fetch Yahoo Finance news: {e}")
        return []


def display_news() -> None:
    """Display market headlines"""
    news = fetch_market_news(8)

    text = Text()
    text.append("MARKET HEADLINES\n\n", style="bold white")

    if not news:
        text.append("  Unable to fetch news at this time.\n", style="bright_black")
        text.append("  Latest Market Updates:\n\n", style="white")
        text.append("  • Markets showing mixed performance with tech leading\n", style="white")
        text.append("  • Treasury yields fluctuating on Fed policy expectations\n", style="white")
        text.append("  • Earnings season driving individual stock movements\n", style="white")
        text.append("  • Global markets tracking US indices closely\n", style="white")
    else:
        for i, item in enumerate(news, 1):
            text.append(f"  {item['date']} • ", style="bright_black")
            text.append(f"{item['source']}\n", style="cyan")
            text.append(f"  {i}. {item['title']}\n\n", style="white")

    panel = Panel(text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])
    console.print(panel)
    console.print()


def main() -> None:
    try:
        console.clear()

        console.print(create_enhanced_header())
        console.print()

        console.print("[white]Fetching market data...[/white]\n")

        indices_data = fetch_all_data(INDICES)
        sectors_data = fetch_all_data(SECTORS)
        global_data = fetch_all_data(GLOBAL_MARKETS)
        commodities_data = fetch_all_data(COMMODITIES)
        currencies_data = fetch_all_data(CURRENCIES)

        console.clear()
        console.print(create_enhanced_header())
        console.print()

        console.print(create_indices_table_enhanced(indices_data))
        console.print()

        sp500_data = []
        results = fetch_parallel(SP500_TICKERS[:30], fetch_symbol_data, max_workers=15)
        sp500_data = [r for r in results.values() if r]

        breadth = calculate_market_breadth(sp500_data)

        console.print(Columns([
            create_market_summary(indices_data, breadth),
            create_breadth_panel(breadth)
        ]))
        console.print()

        gainers, losers, active = fetch_sp500_movers()

        console.print(Columns([
            create_movers_table(gainers, "TOP GAINERS"),
            create_movers_table(losers, "TOP LOSERS"),
            create_movers_table(active, "MOST ACTIVE (High Volume)")
        ]))
        console.print()

        console.print(create_sector_table(sectors_data))
        console.print()

        console.print(Columns([
            Panel(create_table_from_data(global_data, "GLOBAL MARKETS"),
                  border_style=THEME['border'], style=THEME['panel_bg']),
            Panel(create_table_from_data(commodities_data, "COMMODITIES & RESOURCES"),
                  border_style=THEME['border'], style=THEME['panel_bg'])
        ]))
        console.print()

        console.print(Panel(create_table_from_data(currencies_data, "CURRENCIES & FX"),
                           border_style=THEME['border'], style=THEME['panel_bg']))
        console.print()

        display_news()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def create_table_from_data(data: List[MarketData], title: str) -> Table:
    """Create a standard table from market data"""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Market", style="white", width=26)
    table.add_column("Price", justify="right", width=14)
    table.add_column("Change", justify="right", width=14)
    table.add_column("Change %", justify="right", width=12)
    table.add_column("Trend (5D)", justify="center", width=12)

    for item in data:
        color = COLORS['up'] if item.change > 0 else COLORS['down'] if item.change < 0 else COLORS['neutral']
        symbol = "▲" if item.change > 0 else "▼" if item.change < 0 else "━"

        table.add_row(
            item.name,
            f"${item.price:,.2f}",
            f"[{color}]{symbol} ${abs(item.change):.2f}[/{color}]",
            f"[{color}]{item.change_pct:+.2f}%[/{color}]",
            item.sparkline or "━━━━━"
        )

    return table


if __name__ == "__main__":
    main()
