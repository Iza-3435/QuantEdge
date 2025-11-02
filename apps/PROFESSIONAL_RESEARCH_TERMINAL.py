#!/usr/bin/env python3
"""
Professional Research Terminal

Bloomberg-inspired terminal for comprehensive stock research and analysis.
Provides institutional-grade metrics, technical indicators, and market intelligence.
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import yfinance as yf
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_cache import DataCache, get_ticker_info, get_ticker_history

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


def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch comprehensive stock data with intelligent caching.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        Dictionary containing stock data and metrics, or None if fetch fails

    Raises:
        None - errors are caught and logged
    """
    cache_key = f"research_terminal_{symbol}"
    cached_data = DataCache.get(cache_key)
    if cached_data:
        return cached_data

    try:
        info = get_ticker_info(symbol)
        hist = get_ticker_history(symbol, period='1y')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        returns_1m = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) > 21 else 0
        returns_3m = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63] * 100) if len(hist) > 63 else 0
        returns_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]

        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1]

        data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'high_52w': hist['High'].max(),
            'low_52w': hist['Low'].min(),
            'volume': hist['Volume'].iloc[-1],
            'avg_volume': hist['Volume'].mean(),
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'returns_ytd': returns_ytd,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'profit_margin': info.get('profitMargins', 0) * 100,
            'operating_margin': info.get('operatingMargins', 0) * 100,
            'roe': info.get('returnOnEquity', 0) * 100,
            'roa': info.get('returnOnAssets', 0) * 100,
            'revenue_growth': info.get('revenueGrowth', 0) * 100,
            'earnings_growth': info.get('earningsGrowth', 0) * 100,
            'debt_equity': info.get('debtToEquity', 0),
            'current_ratio': info.get('currentRatio', 0),
            'total_cash': info.get('totalCash', 0),
            'total_debt': info.get('totalDebt', 0),
            'rsi': rsi_current,
            'volatility': volatility,
            'beta': info.get('beta', 0),
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'target_price': info.get('targetMeanPrice', 0),
            'recommendation': info.get('recommendationKey', 'N/A').upper(),
            'num_analysts': info.get('numberOfAnalystOpinions', 0),
            'change_52w': info.get('52WeekChange', 0) * 100,
            'sp500_52w_change': info.get('SandP52WeekChange', 0) * 100,
            'relative_performance': (info.get('52WeekChange', 0) - info.get('SandP52WeekChange', 0)) * 100,
            'institutional_ownership': info.get('heldPercentInstitutions', 0) * 100,
            'insider_ownership': info.get('heldPercentInsiders', 0) * 100,
            'short_percent': info.get('shortPercentOfFloat', 0) * 100,
            'short_ratio': info.get('shortRatio', 0),
        }

        DataCache.set(cache_key, data)
        return data

    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        return None


def get_stock_news(symbol: str, limit: int = 5) -> List[Dict[str, str]]:
    """
    Fetch recent news for a stock symbol.

    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of news items to return

    Returns:
        List of news dictionaries with title, publisher, link, and date
    """
    news_items = []

    try:
        ticker = yf.Ticker(symbol)
        news = getattr(ticker, 'news', None)

        if news and len(news) > 0:
            for item in news[:limit]:
                if isinstance(item, dict):
                    content = item.get('content', {})
                    title = content.get('title', item.get('title', ''))

                    provider_info = content.get('provider', item.get('provider', {}))
                    if isinstance(provider_info, dict):
                        publisher = provider_info.get('displayName', 'Yahoo Finance')
                    else:
                        publisher = item.get('publisher', 'Yahoo Finance')

                    canonical_url = content.get('canonicalUrl', {})
                    if isinstance(canonical_url, dict):
                        link = canonical_url.get('url', '')
                    else:
                        link = item.get('link', '')

                    if not title or title == 'No title':
                        continue

                    pub_date = content.get('pubDate', '')
                    if pub_date:
                        try:
                            from dateutil import parser
                            published = parser.parse(pub_date).strftime('%Y-%m-%d')
                        except:
                            published = 'Recent'
                    else:
                        pub_time = item.get('providerPublishTime', 0)
                        if pub_time:
                            try:
                                published = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                            except:
                                published = 'Recent'
                        else:
                            published = 'Recent'

                    news_items.append({
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'published': published
                    })

    except Exception:
        pass

    return news_items[:limit]


def get_upcoming_events(symbol: str) -> Dict[str, Any]:
    """
    Get upcoming corporate events (earnings, dividends).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with event dates and details
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        events = {}

        if 'earningsDate' in info and info['earningsDate']:
            try:
                earnings_date = info['earningsDate']
                if isinstance(earnings_date, list) and len(earnings_date) > 0:
                    events['earnings'] = datetime.fromtimestamp(earnings_date[0]).strftime('%Y-%m-%d')
                elif isinstance(earnings_date, (int, float)):
                    events['earnings'] = datetime.fromtimestamp(earnings_date).strftime('%Y-%m-%d')
            except:
                pass

        if 'exDividendDate' in info and info['exDividendDate']:
            try:
                ex_div_date = info['exDividendDate']
                if isinstance(ex_div_date, (int, float)):
                    events['ex_dividend'] = datetime.fromtimestamp(ex_div_date).strftime('%Y-%m-%d')
            except:
                pass

        if 'dividendRate' in info:
            events['dividend_amount'] = info.get('dividendRate', 0)

        return events
    except:
        return {}


def get_competitors(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get competitor comparison data.

    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of competitors to return

    Returns:
        List of competitor data dictionaries
    """
    competitor_map = {
        'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA'],
        'MSFT': ['AAPL', 'GOOGL', 'META', 'ORCL', 'ADBE'],
        'GOOGL': ['MSFT', 'META', 'AAPL', 'AMZN', 'NFLX'],
        'META': ['GOOGL', 'SNAP', 'PINS', 'RDDT', 'NFLX'],
        'AMZN': ['WMT', 'SHOP', 'EBAY', 'BABA', 'MELI'],
        'NVDA': ['AMD', 'INTC', 'QCOM', 'AVGO', 'TSM'],
        'AMD': ['NVDA', 'INTC', 'QCOM', 'AVGO', 'MU'],
        'INTC': ['AMD', 'NVDA', 'QCOM', 'TXN', 'MU'],
        'TSLA': ['GM', 'F', 'RIVN', 'LCID', 'NIO'],
        'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS'],
        'WMT': ['TGT', 'COST', 'HD', 'LOW', 'AMZN'],
        'KO': ['PEP', 'MNST', 'DPS', 'KDP', 'CELH'],
        'NFLX': ['DIS', 'PARA', 'WBD', 'SPOT', 'ROKU'],
        'V': ['MA', 'AXP', 'PYPL', 'SQ', 'FIS'],
    }

    competitors = competitor_map.get(symbol, [])
    if not competitors:
        return []

    comp_data = []
    for comp_symbol in competitors[:limit]:
        try:
            ticker = yf.Ticker(comp_symbol)
            info = ticker.info
            hist = ticker.history(period='1y')

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                returns_1y = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

                comp_data.append({
                    'symbol': comp_symbol,
                    'name': info.get('shortName', comp_symbol)[:20],
                    'price': current_price,
                    'pe': info.get('trailingPE', 0),
                    'market_cap': info.get('marketCap', 0),
                    'returns_1y': returns_1y,
                    'revenue_growth': info.get('revenueGrowth', 0) * 100,
                })
        except:
            continue

    return comp_data


def create_header(data: Dict[str, Any]) -> Panel:
    """Create terminal header with stock info."""
    arrow = '▲' if data['change_pct'] >= 0 else '▼'
    price_color = COLORS['up'] if data['change_pct'] >= 0 else COLORS['down']

    text = Text()
    text.append(f"{data['symbol']}", style="bold white")
    text.append(" │ ", style="bright_black")
    text.append(f"{data['name']}", style="white")
    text.append(" │ ", style="bright_black")
    text.append(f"{data['sector']}", style="bright_black")
    text.append("\n")

    text.append(f"${data['current_price']:.2f}", style=f"bold {price_color}")
    text.append(f"  {arrow} ", style=price_color)
    text.append(f"{abs(data['change']):+.2f} ", style=price_color)
    text.append(f"({data['change_pct']:+.2f}%)", style=price_color)
    text.append("\n")

    text.append(datetime.now().strftime('%Y-%m-%d %I:%M:%S %p ET'), style="bright_black")

    return Panel(text, border_style="grey35", box=box.SQUARE, padding=(1, 2), style="on grey11")


def create_professional_table(title: str, rows: List[Tuple[str, str]]) -> Panel:
    """
    Create professional table panel.

    Args:
        title: Table title
        rows: List of (metric, value) tuples

    Returns:
        Rich Panel containing formatted table
    """
    table = Table(
        show_header=True,
        header_style="bold white on grey23",
        border_style="grey35",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=False,
        row_styles=["on grey15", "on grey11"]
    )

    table.add_column("Metric", style="white", no_wrap=True, width=18)
    table.add_column("Value", justify="right", style="white", width=15)

    for metric, value in rows:
        table.add_row(metric, value)

    return Panel(
        table,
        title=f"[bold white]{title}[/bold white]",
        border_style="grey35",
        box=box.SQUARE,
        style="on grey11"
    )


def create_price_table(data: Dict[str, Any]) -> Panel:
    """Create price and performance metrics table."""
    rows = [
        ("Last Price", f"${data['current_price']:.2f}"),
        ("Change", f"[{COLORS['up'] if data['change'] >= 0 else COLORS['down']}]{data['change']:+.2f} ({data['change_pct']:+.2f}%)[/]"),
        ("52W High", f"${data['high_52w']:.2f}"),
        ("52W Low", f"${data['low_52w']:.2f}"),
        ("Volume", f"{data['volume']/1e6:.2f}M"),
        ("Avg Volume", f"{data['avg_volume']/1e6:.2f}M"),
    ]
    return create_professional_table("PRICE & PERFORMANCE", rows)


def create_returns_table(data: Dict[str, Any]) -> Panel:
    """Create returns and risk metrics table."""
    def format_return(val: float) -> str:
        color = COLORS['up'] if val >= 0 else COLORS['down']
        return f"[{color}]{val:+.2f}%[/]"

    rows = [
        ("1 Month", format_return(data['returns_1m'])),
        ("3 Month", format_return(data['returns_3m'])),
        ("YTD", format_return(data['returns_ytd'])),
        ("Volatility", f"{data['volatility']:.1f}%"),
        ("Beta", f"{data['beta']:.2f}"),
        ("RSI (14)", f"{data['rsi']:.1f}"),
    ]
    return create_professional_table("RETURNS & RISK", rows)


def create_valuation_table(data: Dict[str, Any]) -> Panel:
    """Create valuation metrics table."""
    mkt_cap = data['market_cap']
    mkt_cap_str = f"${mkt_cap/1e12:.2f}T" if mkt_cap > 1e12 else f"${mkt_cap/1e9:.1f}B"

    rows = [
        ("Market Cap", mkt_cap_str),
        ("P/E Ratio", f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "N/A"),
        ("Forward P/E", f"{data['forward_pe']:.2f}" if data['forward_pe'] else "N/A"),
        ("PEG Ratio", f"{data['peg_ratio']:.2f}" if data['peg_ratio'] else "N/A"),
        ("P/B Ratio", f"{data['pb_ratio']:.2f}" if data['pb_ratio'] else "N/A"),
        ("P/S Ratio", f"{data['ps_ratio']:.2f}" if data['ps_ratio'] else "N/A"),
    ]
    return create_professional_table("VALUATION", rows)


def create_fundamentals_table(data: Dict[str, Any]) -> Panel:
    """Create fundamentals metrics table."""
    def format_growth(val: float) -> str:
        if val == 0:
            return "N/A"
        color = COLORS['up'] if val >= 0 else COLORS['down']
        return f"[{color}]{val:+.1f}%[/]"

    rows = [
        ("Profit Margin", f"{data['profit_margin']:.1f}%" if data['profit_margin'] else "N/A"),
        ("Operating Margin", f"{data['operating_margin']:.1f}%" if data['operating_margin'] else "N/A"),
        ("ROE", f"{data['roe']:.1f}%" if data['roe'] else "N/A"),
        ("ROA", f"{data['roa']:.1f}%" if data['roa'] else "N/A"),
        ("Revenue Growth", format_growth(data['revenue_growth'])),
        ("Earnings Growth", format_growth(data['earnings_growth'])),
    ]
    return create_professional_table("FUNDAMENTALS", rows)


def create_balance_sheet_table(data: Dict[str, Any]) -> Panel:
    """Create balance sheet metrics table."""
    cash = data['total_cash']
    debt = data['total_debt']
    cash_str = f"${cash/1e9:.1f}B" if cash > 1e9 else f"${cash/1e6:.1f}M"
    debt_str = f"${debt/1e9:.1f}B" if debt > 1e9 else f"${debt/1e6:.1f}M"

    rows = [
        ("Total Cash", cash_str),
        ("Total Debt", debt_str),
        ("Debt/Equity", f"{data['debt_equity']:.1f}" if data['debt_equity'] else "N/A"),
        ("Current Ratio", f"{data['current_ratio']:.2f}" if data['current_ratio'] else "N/A"),
        ("Dividend Yield", f"{data['dividend_yield']:.2f}%" if data['dividend_yield'] else "0.00%"),
    ]
    return create_professional_table("BALANCE SHEET", rows)


def create_technical_table(data: Dict[str, Any]) -> Panel:
    """Create technical indicators table."""
    price = data['current_price']
    rsi = data['rsi']

    rsi_signal = "[red]Overbought[/red]" if rsi > 70 else "[green]Oversold[/green]" if rsi < 30 else "[white]Neutral[/white]"

    rows = [
        ("RSI (14)", f"{rsi:.1f} {rsi_signal}"),
        ("Beta", f"{data['beta']:.2f}" if data['beta'] else "N/A"),
        ("Volatility", f"{data['volatility']:.1f}%"),
        ("vs SMA20", f"[{COLORS['up'] if price > data['sma_20'] else COLORS['down']}]{((price/data['sma_20']-1)*100):+.1f}%[/]"),
        ("vs SMA50", f"[{COLORS['up'] if price > data['sma_50'] else COLORS['down']}]{((price/data['sma_50']-1)*100):+.1f}%[/]"),
    ]
    return create_professional_table("TECHNICAL INDICATORS", rows)


def create_analyst_table(data: Dict[str, Any]) -> Panel:
    """Create analyst consensus table."""
    target = data['target_price']
    current = data['current_price']
    upside = ((target - current) / current * 100) if target and current else 0

    rows = [
        ("Analysts", f"{data['num_analysts']}"),
        ("Rating", data['recommendation'].replace('_', ' ')),
        ("Target", f"${target:.2f}" if target else "N/A"),
        ("Upside", f"[{COLORS['up'] if upside >= 0 else COLORS['down']}]{upside:+.1f}%[/]" if target else "N/A"),
    ]
    return create_professional_table("ANALYST CONSENSUS", rows)


def create_market_performance_table(data: Dict[str, Any]) -> Panel:
    """Create market performance comparison table."""
    rel_perf = data.get('relative_performance', 0)
    perf_status = "[green]Outperforming[/green]" if rel_perf > 5 else "[red]Underperforming[/red]" if rel_perf < -5 else "[white]In-line[/white]"

    rows = [
        ("Stock (52W)", f"[{COLORS['up'] if data['change_52w'] >= 0 else COLORS['down']}]{data['change_52w']:+.1f}%[/]"),
        ("S&P 500 (52W)", f"[{COLORS['up'] if data['sp500_52w_change'] >= 0 else COLORS['down']}]{data['sp500_52w_change']:+.1f}%[/]"),
        ("Rel. Perf", f"[{COLORS['up'] if rel_perf >= 0 else COLORS['down']}]{rel_perf:+.1f}%[/]"),
        ("Status", perf_status),
    ]
    return create_professional_table("VS MARKET", rows)


def create_momentum_signals(data: Dict[str, Any]) -> Panel:
    """Create momentum and technical signals panel."""
    signals = []
    price = data['current_price']

    if price > data['sma_20'] and price > data['sma_50']:
        trend = "[green]UPTREND[/green]"
        signals.append("✓")
    elif price < data['sma_20'] and price < data['sma_50']:
        trend = "[red]DOWNTREND[/red]"
        signals.append("✗")
    else:
        trend = "[white]MIXED[/white]"
        signals.append("•")

    rsi = data['rsi']
    if rsi > 70:
        rsi_sig = "[red]Overbought[/red]"
        signals.append("✗")
    elif rsi < 30:
        rsi_sig = "[green]Oversold[/green]"
        signals.append("✓")
    else:
        rsi_sig = "[white]Neutral[/white]"
        signals.append("•")

    if data['returns_1m'] > 5:
        momentum = "[green]STRONG[/green]"
        signals.append("✓")
    elif data['returns_1m'] < -5:
        momentum = "[red]WEAK[/red]"
        signals.append("✗")
    else:
        momentum = "[white]NEUTRAL[/white]"
        signals.append("•")

    bullish_count = signals.count("✓")
    bearish_count = signals.count("✗")
    overall = "[green]BULLISH[/green]" if bullish_count >= 2 else "[red]BEARISH[/red]" if bearish_count >= 2 else "[white]NEUTRAL[/white]"

    text = Text()
    text.append("Trend: ", style="bright_black")
    text.append(trend)
    text.append(" │ ", style="bright_black")
    text.append("RSI: ", style="bright_black")
    text.append(rsi_sig)
    text.append("\n")
    text.append("Momentum: ", style="bright_black")
    text.append(momentum)
    text.append(" │ ", style="bright_black")
    text.append("Signal: ", style="bright_black")
    text.append(overall)

    return Panel(text, title="[bold white]MOMENTUM SIGNALS[/bold white]", border_style="grey35", box=box.SQUARE, padding=(0, 2), style="on grey11")


def create_summary(data: Dict[str, Any]) -> Panel:
    """Create investment summary with scoring."""
    score = 0

    if 0 < data['pe_ratio'] < 20: score += 20
    elif 20 <= data['pe_ratio'] < 30: score += 15
    elif 30 <= data['pe_ratio'] < 40: score += 10

    if data['profit_margin'] > 20: score += 15
    elif data['profit_margin'] > 10: score += 10

    if data['roe'] > 20: score += 10
    elif data['roe'] > 15: score += 5

    if data['revenue_growth'] > 15: score += 15
    elif data['revenue_growth'] > 5: score += 10

    if data['earnings_growth'] > 15: score += 10
    elif data['earnings_growth'] > 5: score += 5

    if data['current_price'] > data['sma_50']: score += 10

    if score >= 70:
        verdict, color = "STRONG BUY", "green"
    elif score >= 55:
        verdict, color = "BUY", "green"
    elif score >= 40:
        verdict, color = "HOLD", "white"
    elif score >= 25:
        verdict, color = "SELL", "red"
    else:
        verdict, color = "STRONG SELL", "red"

    text = Text()
    text.append("Investment Score: ", style="white")
    text.append(f"{score}/100", style="bold white")
    text.append("  │  ", style="bright_black")
    text.append("Recommendation: ", style="white")
    text.append(f"{verdict}", style=f"bold {color}")

    return Panel(text, title="[bold white]INVESTMENT SUMMARY[/bold white]", border_style="grey35", box=box.SQUARE, padding=(0, 2), style="on grey11")


def create_competitors_table(comp_data: List[Dict[str, Any]], main_symbol: str) -> Panel:
    """Create competitor comparison table."""
    if not comp_data:
        return Panel(
            f"[bright_black]No competitor data available for {main_symbol}[/bright_black]",
            title="[bold white]COMPETITORS[/bold white]",
            border_style=THEME['border'],
            box=box.SQUARE,
            style=THEME['panel_bg']
        )

    table = Table(
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=20)
    table.add_column("Price", justify="right", width=10)
    table.add_column("P/E", justify="right", width=8)
    table.add_column("Mkt Cap", justify="right", width=10)
    table.add_column("1Y Ret", justify="right", width=10)

    for comp in comp_data:
        ret_color = COLORS['up'] if comp['returns_1y'] > 0 else COLORS['down']
        market_cap_str = f"${comp['market_cap']/1e9:.1f}B" if comp['market_cap'] > 0 else "N/A"

        table.add_row(
            comp['symbol'],
            comp['name'],
            f"${comp['price']:.2f}",
            f"{comp['pe']:.1f}" if comp['pe'] > 0 else "N/A",
            market_cap_str,
            f"[{ret_color}]{comp['returns_1y']:+.1f}%[/{ret_color}]"
        )

    return Panel(table, title="[bold white]COMPETITORS[/bold white]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def create_news_panel(news_data: List[Dict[str, str]]) -> Panel:
    """Create recent news panel."""
    news_text = Text()

    if not news_data or len(news_data) == 0:
        news_text.append("No news currently available from yfinance API.\n", style="bright_black")
        news_text.append("Try checking:\n", style="bright_black")
        news_text.append("  • finance.yahoo.com for latest news\n", style="bright_black")
        news_text.append("  • Google News for company updates\n", style="bright_black")
        news_text.append("  • Company investor relations page", style="bright_black")

        return Panel(news_text, title="[bold white]RECENT NEWS[/bold white]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])

    for idx, item in enumerate(news_data, 1):
        news_text.append(f"{item['published']} • {item['publisher']}\n", style="bright_black")

        if item['link']:
            link_text = Text(f"{idx}. {item['title']}", style="white underline")
            link_text.stylize(f"link {item['link']}")
            news_text.append(link_text)
        else:
            news_text.append(f"{idx}. {item['title']}", style="white")

        news_text.append("\n\n")

    return Panel(
        news_text,
        title="[bold white]RECENT NEWS[/bold white]",
        subtitle="[bright_black]CMD+click (Mac) or CTRL+click to open[/bright_black]",
        border_style=THEME['border'],
        box=box.SQUARE,
        style=THEME['panel_bg']
    )


def create_events_panel(events: Dict[str, Any]) -> Panel:
    """Create upcoming events panel."""
    if not events:
        return Panel(
            "[bright_black]No upcoming events[/bright_black]",
            title="[bold white]UPCOMING EVENTS[/bold white]",
            border_style=THEME['border'],
            box=box.SQUARE,
            style=THEME['panel_bg']
        )

    text = Text()

    if 'earnings' in events:
        text.append("Next Earnings: ", style="white")
        text.append(f"{events['earnings']}\n", style="bold green")

    if 'ex_dividend' in events:
        text.append("Ex-Dividend Date: ", style="white")
        text.append(f"{events['ex_dividend']}", style="bold white")
        if 'dividend_amount' in events and events['dividend_amount'] > 0:
            text.append(f" (${events['dividend_amount']:.2f})", style="bright_black")
        text.append("\n")

    if not text.plain:
        text.append("No scheduled events", style="bright_black")

    return Panel(text, title="[bold white]UPCOMING EVENTS[/bold white]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'], padding=(0, 2))


def display_terminal(symbol: str) -> None:
    """
    Display professional research terminal for a stock symbol.

    Args:
        symbol: Stock ticker symbol to analyze
    """
    console.clear()

    with console.status(f"[white]Loading {symbol}...", spinner="dots"):
        data = get_stock_data(symbol)
        news = get_stock_news(symbol, limit=5)
        events = get_upcoming_events(symbol)
        competitors = get_competitors(symbol, limit=5)

    if not data:
        console.print(f"\n[red]Error: Could not fetch data for {symbol}[/red]\n")
        return

    console.clear()

    console.print(create_header(data))
    console.print()

    console.print(Columns([create_price_table(data), create_returns_table(data)]))
    console.print()

    console.print(Columns([create_valuation_table(data), create_fundamentals_table(data)]))
    console.print()

    console.print(Columns([create_balance_sheet_table(data), create_technical_table(data), create_market_performance_table(data)]))
    console.print()

    console.print(Columns([create_analyst_table(data), create_momentum_signals(data), create_summary(data)]))
    console.print()

    console.print(Columns([create_competitors_table(competitors, symbol), create_events_panel(events)]))
    console.print()

    console.print(create_news_panel(news))
    console.print()

    footer = Panel(
        f"[bright_black]Professional Research Terminal • {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}[/bright_black]",
        border_style="grey35",
        box=box.SQUARE,
        style="on grey11"
    )
    console.print(footer)
    console.print()


def main() -> None:
    """Main entry point for the research terminal."""
    if len(sys.argv) < 2:
        console.print("\n[white]Usage:[/white] python PROFESSIONAL_RESEARCH_TERMINAL.py <SYMBOL>")
        console.print("\n[white]Example:[/white] python PROFESSIONAL_RESEARCH_TERMINAL.py AAPL\n")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    try:
        display_terminal(symbol)
    except KeyboardInterrupt:
        console.print("\n[white]Interrupted[/white]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
