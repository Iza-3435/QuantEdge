"""
PROFESSIONAL RESEARCH TERMINAL
Trading terminal style with professional gray theme throughout
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Add utils to path for caching
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_cache import DataCache, get_ticker_info, get_ticker_history

console = Console()

# Professional colors - simple and clean
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


def get_stock_data(symbol):
    """Fetch comprehensive stock data with caching"""
    # Check cache first
    cache_key = f"research_terminal_{symbol}"
    cached_data = DataCache.get(cache_key)
    if cached_data:
        return cached_data

    try:
        ticker = yf.Ticker(symbol)
        info = get_ticker_info(symbol)  # Use cached version
        hist = get_ticker_history(symbol, period='1y')  # Use cached version

        if hist.empty:
            return None

        # Calculate metrics
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        # Returns
        returns_1m = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) > 21 else 0
        returns_3m = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63] * 100) if len(hist) > 63 else 0
        returns_ytd = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

        # Volatility & RSI
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]

        # Moving averages
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1]

        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,

            # Price metrics
            'high_52w': hist['High'].max(),
            'low_52w': hist['Low'].min(),
            'volume': hist['Volume'].iloc[-1],
            'avg_volume': hist['Volume'].mean(),

            # Returns
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'returns_ytd': returns_ytd,

            # Fundamentals
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,

            # Profitability
            'profit_margin': info.get('profitMargins', 0) * 100,
            'operating_margin': info.get('operatingMargins', 0) * 100,
            'roe': info.get('returnOnEquity', 0) * 100,
            'roa': info.get('returnOnAssets', 0) * 100,

            # Growth
            'revenue_growth': info.get('revenueGrowth', 0) * 100,
            'earnings_growth': info.get('earningsGrowth', 0) * 100,

            # Balance sheet
            'debt_equity': info.get('debtToEquity', 0),
            'current_ratio': info.get('currentRatio', 0),
            'total_cash': info.get('totalCash', 0),
            'total_debt': info.get('totalDebt', 0),

            # Technical
            'rsi': rsi_current,
            'volatility': volatility,
            'beta': info.get('beta', 0),
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,

            # Analyst
            'target_price': info.get('targetMeanPrice', 0),
            'recommendation': info.get('recommendationKey', 'N/A').upper(),
            'num_analysts': info.get('numberOfAnalystOpinions', 0),

            # Market Performance Comparison
            'change_52w': info.get('52WeekChange', 0) * 100,
            'sp500_52w_change': info.get('SandP52WeekChange', 0) * 100,
            'relative_performance': (info.get('52WeekChange', 0) - info.get('SandP52WeekChange', 0)) * 100,

            # Ownership & Short Interest
            'institutional_ownership': info.get('heldPercentInstitutions', 0) * 100,
            'insider_ownership': info.get('heldPercentInsiders', 0) * 100,
            'short_percent': info.get('shortPercentOfFloat', 0) * 100,
            'short_ratio': info.get('shortRatio', 0),
        }

        # Cache the result
        DataCache.set(cache_key, data)
        return data

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def get_stock_news(symbol, limit=5):
    """Fetch recent stock-specific news"""
    news_items = []

    try:
        # Try yfinance news first
        ticker = yf.Ticker(symbol)
        news = getattr(ticker, 'news', None)

        if news and len(news) > 0:
            for item in news[:limit]:
                if isinstance(item, dict):
                    # New yfinance API has nested structure
                    content = item.get('content', {})

                    # Try both old and new API formats
                    title = content.get('title', item.get('title', ''))

                    # Get provider info
                    provider_info = content.get('provider', item.get('provider', {}))
                    if isinstance(provider_info, dict):
                        publisher = provider_info.get('displayName', 'Yahoo Finance')
                    else:
                        publisher = item.get('publisher', 'Yahoo Finance')

                    # Get URL
                    canonical_url = content.get('canonicalUrl', {})
                    if isinstance(canonical_url, dict):
                        link = canonical_url.get('url', '')
                    else:
                        link = item.get('link', '')

                    # Skip if no title
                    if not title or title == 'No title':
                        continue

                    # Parse date - try both formats
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

        # If no news found, return empty list (will be handled by create_news_panel)
        # if not news_items:
        #     news_items = []

    except Exception as e:
        # Return empty list on error
        pass

    return news_items[:limit]


def get_upcoming_events(symbol):
    """Get upcoming earnings and dividend dates"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        events = {}

        # Earnings date
        if 'earningsDate' in info and info['earningsDate']:
            try:
                earnings_date = info['earningsDate']
                if isinstance(earnings_date, list) and len(earnings_date) > 0:
                    events['earnings'] = datetime.fromtimestamp(earnings_date[0]).strftime('%Y-%m-%d')
                elif isinstance(earnings_date, (int, float)):
                    events['earnings'] = datetime.fromtimestamp(earnings_date).strftime('%Y-%m-%d')
            except:
                pass

        # Ex-dividend date
        if 'exDividendDate' in info and info['exDividendDate']:
            try:
                ex_div_date = info['exDividendDate']
                if isinstance(ex_div_date, (int, float)):
                    events['ex_dividend'] = datetime.fromtimestamp(ex_div_date).strftime('%Y-%m-%d')
            except:
                pass

        # Dividend amount
        if 'dividendRate' in info:
            events['dividend_amount'] = info.get('dividendRate', 0)

        return events
    except:
        return {}


def get_competitors(symbol, limit=5):
    """Get competitor comparison data"""
    # Predefined competitor mappings for major stocks
    competitor_map = {
        # Tech Giants
        'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA'],
        'MSFT': ['AAPL', 'GOOGL', 'META', 'ORCL', 'ADBE'],
        'GOOGL': ['MSFT', 'META', 'AAPL', 'AMZN', 'NFLX'],
        'META': ['GOOGL', 'SNAP', 'PINS', 'RDDT', 'NFLX'],
        'AMZN': ['WMT', 'SHOP', 'EBAY', 'BABA', 'MELI'],

        # Semiconductors
        'NVDA': ['AMD', 'INTC', 'QCOM', 'AVGO', 'TSM'],
        'AMD': ['NVDA', 'INTC', 'QCOM', 'AVGO', 'MU'],
        'INTC': ['AMD', 'NVDA', 'QCOM', 'TXN', 'MU'],
        'QCOM': ['AVGO', 'NVDA', 'AMD', 'MRVL', 'NXPI'],
        'AVGO': ['QCOM', 'NVDA', 'AMD', 'MRVL', 'TXN'],
        'TSM': ['INTC', 'NVDA', 'AMD', 'UMC', 'SMIC'],
        'MU': ['WDC', 'STX', 'NXPI', 'SWKS', 'AMD'],

        # Auto/EV
        'TSLA': ['GM', 'F', 'RIVN', 'LCID', 'NIO'],
        'GM': ['F', 'TSLA', 'STLA', 'TM', 'HMC'],
        'F': ['GM', 'TSLA', 'STLA', 'TM', 'HMC'],
        'RIVN': ['LCID', 'TSLA', 'F', 'GM', 'NIO'],
        'LCID': ['RIVN', 'TSLA', 'NIO', 'XPEV', 'LI'],

        # Banks
        'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS'],
        'BAC': ['JPM', 'WFC', 'C', 'USB', 'PNC'],
        'WFC': ['JPM', 'BAC', 'C', 'USB', 'PNC'],
        'C': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'GS': ['MS', 'JPM', 'C', 'SCHW', 'BLK'],
        'MS': ['GS', 'JPM', 'C', 'SCHW', 'BLK'],

        # Retail
        'WMT': ['TGT', 'COST', 'HD', 'LOW', 'AMZN'],
        'TGT': ['WMT', 'COST', 'KSS', 'M', 'JWN'],
        'COST': ['WMT', 'TGT', 'BJ', 'PSMT', 'DLTR'],
        'HD': ['LOW', 'WMT', 'TSCO', 'LL', 'ORLY'],
        'LOW': ['HD', 'WMT', 'TSCO', 'LL', 'SHW'],

        # Beverages
        'KO': ['PEP', 'MNST', 'DPS', 'KDP', 'CELH'],
        'PEP': ['KO', 'MNST', 'DPS', 'KDP', 'CELH'],
        'MNST': ['KO', 'PEP', 'CELH', 'KDP', 'FIZZ'],

        # Media/Entertainment
        'NFLX': ['DIS', 'PARA', 'WBD', 'SPOT', 'ROKU'],
        'DIS': ['NFLX', 'PARA', 'WBD', 'CMCSA', 'LYV'],
        'SPOT': ['AAPL', 'AMZN', 'GOOGL', 'NFLX', 'WMG'],

        # Software/Cloud
        'CRM': ['ORCL', 'SAP', 'MSFT', 'NOW', 'ADBE'],
        'ORCL': ['MSFT', 'SAP', 'IBM', 'CRM', 'SNOW'],
        'ADBE': ['MSFT', 'AAPL', 'CRM', 'INTU', 'WDAY'],
        'NOW': ['CRM', 'SNOW', 'MSFT', 'ORCL', 'WDAY'],

        # E-commerce
        'SHOP': ['AMZN', 'EBAY', 'ETSY', 'WMT', 'BIGC'],
        'EBAY': ['AMZN', 'ETSY', 'SHOP', 'MELI', 'POSH'],
        'ETSY': ['EBAY', 'AMZN', 'SHOP', 'W', 'POSH'],

        # Aerospace/Defense
        'BA': ['LMT', 'RTX', 'NOC', 'GD', 'TXT'],
        'LMT': ['BA', 'RTX', 'NOC', 'GD', 'LHX'],

        # Pharma/Biotech
        'JNJ': ['PFE', 'MRK', 'ABBV', 'LLY', 'BMY'],
        'PFE': ['JNJ', 'MRK', 'ABBV', 'LLY', 'BMY'],
        'ABBV': ['JNJ', 'PFE', 'AMGN', 'GILD', 'LLY'],
        'LLY': ['NVO', 'JNJ', 'PFE', 'ABBV', 'MRK'],

        # Payment Processors
        'V': ['MA', 'AXP', 'PYPL', 'SQ', 'FIS'],
        'MA': ['V', 'AXP', 'PYPL', 'SQ', 'FIS'],
        'PYPL': ['SQ', 'V', 'MA', 'AFRM', 'SOFI'],
        'SQ': ['PYPL', 'V', 'MA', 'AFRM', 'SOFI'],
    }

    # Try predefined competitors first
    competitors = competitor_map.get(symbol, [])

    # If no predefined competitors, try to get from same industry/sector
    if not competitors:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            industry = info.get('industry', '')
            sector = info.get('sector', '')

            # This is a fallback - we won't have perfect data, but better than nothing
            # In a production system, you'd use a proper stock screener API
            # For now, return empty to avoid slow API calls
            return []
        except:
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


def create_header(data):
    """Create professional header with gray theme"""
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


def create_professional_table(title, rows, width_ratio=1):
    """Create professional table with light gray background"""
    table = Table(
        show_header=True,
        header_style="bold white on grey23",
        border_style="grey35",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=False,
        row_styles=["on grey15", "on grey11"]
    )

    # Add columns
    table.add_column("Metric", style="white", no_wrap=True, width=18)
    table.add_column("Value", justify="right", style="white", width=15)

    # Add rows
    for row in rows:
        metric = row[0]
        value = row[1]
        table.add_row(metric, value)

    return Panel(
        table,
        title=f"[bold white]{title}[/bold white]",
        border_style="grey35",
        box=box.SQUARE,
        style="on grey11"
    )


def create_price_table(data):
    """Price & Performance - compact"""
    rows = [
        ["Last Price", f"${data['current_price']:.2f}"],
        ["Change", f"[{COLORS['up'] if data['change'] >= 0 else COLORS['down']}]{data['change']:+.2f} ({data['change_pct']:+.2f}%)[/]"],
        ["52W High", f"${data['high_52w']:.2f}"],
        ["52W Low", f"${data['low_52w']:.2f}"],
        ["Volume", f"{data['volume']/1e6:.2f}M"],
        ["Avg Volume", f"{data['avg_volume']/1e6:.2f}M"],
    ]
    return create_professional_table("PRICE & PERFORMANCE", rows)


def create_returns_table(data):
    """Returns & Risk - compact"""
    def format_return(val):
        color = COLORS['up'] if val >= 0 else COLORS['down']
        return f"[{color}]{val:+.2f}%[/]"

    rows = [
        ["1 Month", format_return(data['returns_1m'])],
        ["3 Month", format_return(data['returns_3m'])],
        ["YTD", format_return(data['returns_ytd'])],
        ["Volatility", f"{data['volatility']:.1f}%"],
        ["Beta", f"{data['beta']:.2f}"],
        ["RSI (14)", f"{data['rsi']:.1f}"],
    ]
    return create_professional_table("RETURNS & RISK", rows)


def create_valuation_table(data):
    """Valuation metrics - compact"""
    mkt_cap = data['market_cap']
    mkt_cap_str = f"${mkt_cap/1e12:.2f}T" if mkt_cap > 1e12 else f"${mkt_cap/1e9:.1f}B"

    rows = [
        ["Market Cap", mkt_cap_str],
        ["P/E Ratio", f"{data['pe_ratio']:.2f}" if data['pe_ratio'] else "N/A"],
        ["Forward P/E", f"{data['forward_pe']:.2f}" if data['forward_pe'] else "N/A"],
        ["PEG Ratio", f"{data['peg_ratio']:.2f}" if data['peg_ratio'] else "N/A"],
        ["P/B Ratio", f"{data['pb_ratio']:.2f}" if data['pb_ratio'] else "N/A"],
        ["P/S Ratio", f"{data['ps_ratio']:.2f}" if data['ps_ratio'] else "N/A"],
    ]
    return create_professional_table("VALUATION", rows)


def create_fundamentals_table(data):
    """Fundamentals - compact"""
    def format_growth(val):
        if val == 0:
            return "N/A"
        color = COLORS['up'] if val >= 0 else COLORS['down']
        return f"[{color}]{val:+.1f}%[/]"

    rows = [
        ["Profit Margin", f"{data['profit_margin']:.1f}%" if data['profit_margin'] else "N/A"],
        ["Operating Margin", f"{data['operating_margin']:.1f}%" if data['operating_margin'] else "N/A"],
        ["ROE", f"{data['roe']:.1f}%" if data['roe'] else "N/A"],
        ["ROA", f"{data['roa']:.1f}%" if data['roa'] else "N/A"],
        ["Revenue Growth", format_growth(data['revenue_growth'])],
        ["Earnings Growth", format_growth(data['earnings_growth'])],
    ]
    return create_professional_table("FUNDAMENTALS", rows)


def create_balance_sheet_table(data):
    """Balance Sheet - medium width"""
    cash = data['total_cash']
    debt = data['total_debt']
    cash_str = f"${cash/1e9:.1f}B" if cash > 1e9 else f"${cash/1e6:.1f}M"
    debt_str = f"${debt/1e9:.1f}B" if debt > 1e9 else f"${debt/1e6:.1f}M"

    rows = [
        ["Total Cash", cash_str],
        ["Total Debt", debt_str],
        ["Debt/Equity", f"{data['debt_equity']:.1f}" if data['debt_equity'] else "N/A"],
        ["Current Ratio", f"{data['current_ratio']:.2f}" if data['current_ratio'] else "N/A"],
        ["Dividend Yield", f"{data['dividend_yield']:.2f}%" if data['dividend_yield'] else "0.00%"],
    ]
    return create_professional_table("BALANCE SHEET", rows)


def create_technical_table(data):
    """Technical Analysis - medium width"""
    price = data['current_price']
    rsi = data['rsi']

    # RSI signal
    if rsi > 70:
        rsi_signal = "[red]Overbought[/red]"
    elif rsi < 30:
        rsi_signal = "[green]Oversold[/green]"
    else:
        rsi_signal = "[white]Neutral[/white]"

    rows = [
        ["RSI (14)", f"{rsi:.1f} {rsi_signal}"],
        ["Beta", f"{data['beta']:.2f}" if data['beta'] else "N/A"],
        ["Volatility", f"{data['volatility']:.1f}%"],
        ["vs SMA20", f"[{COLORS['up'] if price > data['sma_20'] else COLORS['down']}]{((price/data['sma_20']-1)*100):+.1f}%[/]"],
        ["vs SMA50", f"[{COLORS['up'] if price > data['sma_50'] else COLORS['down']}]{((price/data['sma_50']-1)*100):+.1f}%[/]"],
    ]
    return create_professional_table("TECHNICAL INDICATORS", rows)


def create_analyst_table(data):
    """Analyst Ratings - compact"""
    target = data['target_price']
    current = data['current_price']
    upside = ((target - current) / current * 100) if target and current else 0

    rows = [
        ["Analysts", f"{data['num_analysts']}"],
        ["Rating", data['recommendation'].replace('_', ' ')],
        ["Target", f"${target:.2f}" if target else "N/A"],
        ["Upside", f"[{COLORS['up'] if upside >= 0 else COLORS['down']}]{upside:+.1f}%[/]" if target else "N/A"],
    ]
    return create_professional_table("ANALYST CONSENSUS", rows)


def create_market_performance_table(data):
    """Market Performance Comparison - shows stock vs S&P 500"""
    rel_perf = data.get('relative_performance', 0)

    # Determine if outperforming or underperforming
    if rel_perf > 5:
        perf_status = "[green]Outperforming[/green]"
    elif rel_perf < -5:
        perf_status = "[red]Underperforming[/red]"
    else:
        perf_status = "[white]In-line[/white]"

    rows = [
        ["Stock (52W)", f"[{COLORS['up'] if data['change_52w'] >= 0 else COLORS['down']}]{data['change_52w']:+.1f}%[/]"],
        ["S&P 500 (52W)", f"[{COLORS['up'] if data['sp500_52w_change'] >= 0 else COLORS['down']}]{data['sp500_52w_change']:+.1f}%[/]"],
        ["Rel. Perf", f"[{COLORS['up'] if rel_perf >= 0 else COLORS['down']}]{rel_perf:+.1f}%[/]"],
        ["Status", perf_status],
    ]
    return create_professional_table("VS MARKET", rows)


def create_momentum_signals(data):
    """Momentum & Technical Signals - compact panel"""
    signals = []

    # Trend signal (based on moving averages)
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

    # RSI signal
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

    # Momentum signal (short-term performance)
    if data['returns_1m'] > 5:
        momentum = "[green]STRONG[/green]"
        signals.append("✓")
    elif data['returns_1m'] < -5:
        momentum = "[red]WEAK[/red]"
        signals.append("✗")
    else:
        momentum = "[white]NEUTRAL[/white]"
        signals.append("•")

    # Overall signal
    bullish_count = signals.count("✓")
    bearish_count = signals.count("✗")

    if bullish_count >= 2:
        overall = "[green]BULLISH[/green]"
    elif bearish_count >= 2:
        overall = "[red]BEARISH[/red]"
    else:
        overall = "[white]NEUTRAL[/white]"

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
    text.append("\n\n", style="bright_black")

    # Add color legend
    text.append("Legend: ", style="bright_black")
    text.append("[green]Green", style="green")
    text.append("=Bullish ", style="bright_black")
    text.append("[red]Red", style="red")
    text.append("=Bearish ", style="bright_black")
    text.append("[white]White", style="white")
    text.append("=Neutral", style="bright_black")

    return Panel(
        text,
        title="[bold white]MOMENTUM SIGNALS[/bold white]",
        border_style="grey35",
        box=box.SQUARE,
        padding=(0, 2),
        style="on grey11"
    )


def create_summary(data):
    """Investment Summary - wider panel"""
    # Simple scoring
    score = 0

    # Valuation
    if 0 < data['pe_ratio'] < 20: score += 20
    elif 20 <= data['pe_ratio'] < 30: score += 15
    elif 30 <= data['pe_ratio'] < 40: score += 10

    # Profitability
    if data['profit_margin'] > 20: score += 15
    elif data['profit_margin'] > 10: score += 10

    if data['roe'] > 20: score += 10
    elif data['roe'] > 15: score += 5

    # Growth
    if data['revenue_growth'] > 15: score += 15
    elif data['revenue_growth'] > 5: score += 10

    if data['earnings_growth'] > 15: score += 10
    elif data['earnings_growth'] > 5: score += 5

    # Technical
    if data['current_price'] > data['sma_50']: score += 10

    # Determine recommendation
    if score >= 70:
        verdict = "STRONG BUY"
        color = "green"
    elif score >= 55:
        verdict = "BUY"
        color = "green"
    elif score >= 40:
        verdict = "HOLD"
        color = "white"
    elif score >= 25:
        verdict = "SELL"
        color = "red"
    else:
        verdict = "STRONG SELL"
        color = "red"

    text = Text()
    text.append("Investment Score: ", style="white")
    text.append(f"{score}/100", style="bold white")
    text.append("  │  ", style="bright_black")
    text.append("Recommendation: ", style="white")
    text.append(f"{verdict}", style=f"bold {color}")

    return Panel(
        text,
        title="[bold white]INVESTMENT SUMMARY[/bold white]",
        border_style="grey35",
        box=box.SQUARE,
        padding=(0, 2),
        style="on grey11"
    )


def create_competitors_table(comp_data, main_symbol):
    """Create competitor comparison table"""
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

    return Panel(
        table,
        title="[bold white]COMPETITORS[/bold white]",
        border_style=THEME['border'],
        box=box.SQUARE,
        style=THEME['panel_bg']
    )


def create_news_panel(news_data):
    """Create recent news panel with clickable links"""
    news_text = Text()

    if not news_data or len(news_data) == 0:
        # Try to show a more helpful message
        news_text.append("No news currently available from yfinance API.\n", style="bright_black")
        news_text.append("Try checking:\n", style="bright_black")
        news_text.append("  • finance.yahoo.com for latest news\n", style="bright_black")
        news_text.append("  • Google News for company updates\n", style="bright_black")
        news_text.append("  • Company investor relations page", style="bright_black")

        return Panel(
            news_text,
            title="[bold white]RECENT NEWS[/bold white]",
            border_style=THEME['border'],
            box=box.SQUARE,
            style=THEME['panel_bg']
        )

    from rich.text import Text as RichText

    for idx, item in enumerate(news_data, 1):
        # Date and publisher
        news_text.append(f"{item['published']} • {item['publisher']}\n", style="bright_black")

        # Clickable title
        if item['link']:
            link_text = RichText(f"{idx}. {item['title']}", style="white underline")
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


def create_events_panel(events):
    """Create upcoming events panel"""
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

    return Panel(
        text,
        title="[bold white]UPCOMING EVENTS[/bold white]",
        border_style=THEME['border'],
        box=box.SQUARE,
        style=THEME['panel_bg'],
        padding=(0, 2)
    )


def display_terminal(symbol):
    """Display professional research terminal"""
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

    # Header
    console.print(create_header(data))
    console.print()

    # Row 1: Price & Returns side by side
    console.print(Columns([
        create_price_table(data),
        create_returns_table(data),
    ]))
    console.print()

    # Row 2: Valuation & Fundamentals side by side
    console.print(Columns([
        create_valuation_table(data),
        create_fundamentals_table(data)
    ]))
    console.print()

    # Row 3: Balance Sheet & Technical & Market Performance
    console.print(Columns([
        create_balance_sheet_table(data),
        create_technical_table(data),
        create_market_performance_table(data)
    ]))
    console.print()

    # Row 4: Analyst + Momentum Signals + Summary
    console.print(Columns([
        create_analyst_table(data),
        create_momentum_signals(data),
        create_summary(data)
    ]))
    console.print()

    # Row 5: Competitors + Events
    console.print(Columns([
        create_competitors_table(competitors, symbol),
        create_events_panel(events)
    ]))
    console.print()

    # Row 6: Recent News (full width)
    console.print(create_news_panel(news))
    console.print()

    # Footer with gray theme
    footer = Panel(
        f"[bright_black]Professional Research Terminal • {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}[/bright_black]",
        border_style="grey35",
        box=box.SQUARE,
        style="on grey11"
    )
    console.print(footer)
    console.print()


def main():
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
