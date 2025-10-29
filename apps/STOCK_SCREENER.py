"""
STOCK SCREENER
Find stocks that match your criteria
Screens by: Growth, Value, Momentum, Quality, Dividend
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track
import warnings
warnings.filterwarnings('ignore')

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


# Popular stock universes
STOCK_UNIVERSES = {
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH', 'XOM', 'JPM', 'JNJ', 'WMT', 'PG'],
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'INTC', 'AMD', 'QCOM'],
    'faang': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
    'dividend': ['JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'VZ', 'T', 'XOM', 'CVX', 'IBM', 'MMM', 'CAT', 'GE', 'BAC'],
    'growth': ['NVDA', 'TSLA', 'SHOP', 'SQ', 'COIN', 'RBLX', 'SNOW', 'CRWD', 'DDOG', 'NET', 'ZS', 'PLTR', 'SOFI', 'RIVN', 'LCID'],
}

# Screening criteria presets
SCREENING_PRESETS = {
    'value': {
        'name': 'Value Investing (Buffett Style)',
        'criteria': {
            'pe_max': 25,  # Relaxed from 20
            'profit_margin_min': 5,  # Relaxed from 10
        }
    },
    'growth': {
        'name': 'High Growth (Momentum)',
        'criteria': {
            'returns_1y_min': 0,  # Relaxed - just show positive
            'profit_margin_min': 0,  # Show all
        }
    },
    'dividend': {
        'name': 'Dividend Champions',
        'criteria': {
            'dividend_yield_min': 1.5,  # Relaxed from 2
            'profit_margin_min': 5,  # Relaxed from 10
        }
    },
    'quality': {
        'name': 'Quality at Reasonable Price',
        'criteria': {
            'pe_max': 40,  # Relaxed from 30
            'profit_margin_min': 10,  # Keep this
        }
    },
    'momentum': {
        'name': 'Strong Momentum',
        'criteria': {
            'returns_3m_min': 5,  # Relaxed from 15
        }
    },
    'undervalued': {
        'name': 'Undervalued with Upside',
        'criteria': {
            'pe_max': 30,  # Relaxed from 20
            'upside_min': 5,  # Relaxed from 15
        }
    },
}


def fetch_stock_metrics(symbol):
    """Fetch comprehensive stock metrics"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1y')
        info = ticker.info

        if df.empty:
            return None

        current_price = df['Close'].iloc[-1]

        # Returns
        returns_1m = ((current_price - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100) if len(df) >= 21 else 0
        returns_3m = ((current_price - df['Close'].iloc[-63]) / df['Close'].iloc[-63] * 100) if len(df) >= 63 else 0
        returns_1y = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)

        # Volatility & Sharpe
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100
        excess_returns = returns - (0.03 / 252)
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(rs) > 0 else 50

        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': current_price,
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'returns_1y': returns_1y,
            'volatility': volatility,
            'sharpe': sharpe,
            'beta': info.get('beta', 0),
            'pe': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'pb': info.get('priceToBook', 0),
            'ps': info.get('priceToSalesTrailing12Months', 0),
            'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
            'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,  # Convert to ratio
            'market_cap': info.get('marketCap', 0),
            'rsi': rsi,
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
            'target': info.get('targetMeanPrice', 0),
            'upside': ((info.get('targetMeanPrice', current_price) - current_price) / current_price * 100),
            'recommendation': info.get('recommendationKey', 'N/A').upper(),
            'num_analysts': info.get('numberOfAnalystOpinions', 0),
        }
    except Exception as e:
        return None


def apply_screening_criteria(stocks, criteria):
    """Filter stocks based on criteria"""
    filtered = []

    for stock in stocks:
        passes = True

        for key, value in criteria.items():
            if key.endswith('_min'):
                metric = key.replace('_min', '')
                if stock.get(metric, 0) < value:
                    passes = False
                    break

            elif key.endswith('_max'):
                metric = key.replace('_max', '')
                stock_value = stock.get(metric, 0)
                if stock_value == 0:  # Handle N/A
                    continue
                if stock_value > value:
                    passes = False
                    break

            elif key == 'analyst_rating':
                if stock.get('recommendation', 'N/A') not in value:
                    passes = False
                    break

        if passes:
            filtered.append(stock)

    return filtered


def calculate_score(stock, preset_name):
    """Calculate composite score for ranking"""
    score = 0

    if preset_name == 'value':
        # Lower P/E is better
        if 0 < stock['pe'] < 15:
            score += 30
        elif 0 < stock['pe'] < 20:
            score += 20

        # Lower P/B is better
        if 0 < stock['pb'] < 2:
            score += 20
        elif 0 < stock['pb'] < 3:
            score += 10

        # Higher ROE is better
        if stock['roe'] > 20:
            score += 25
        elif stock['roe'] > 15:
            score += 15

        # Lower debt is better
        if stock['debt_equity'] < 0.3:
            score += 15
        elif stock['debt_equity'] < 0.5:
            score += 10

        # Profit margin
        if stock['profit_margin'] > 15:
            score += 10

    elif preset_name == 'growth':
        # Returns
        if stock['returns_1y'] > 50:
            score += 30
        elif stock['returns_1y'] > 20:
            score += 20

        if stock['returns_3m'] > 20:
            score += 25
        elif stock['returns_3m'] > 10:
            score += 15

        # Profitability
        if stock['profit_margin'] > 15:
            score += 20
        elif stock['profit_margin'] > 5:
            score += 10

        # Momentum
        if 60 < stock['rsi'] < 70:
            score += 15
        elif stock['rsi'] > 50:
            score += 10

        # Sharpe
        if stock['sharpe'] > 1:
            score += 10

    elif preset_name == 'dividend':
        # Yield
        if stock['dividend_yield'] > 4:
            score += 30
        elif stock['dividend_yield'] > 2:
            score += 20

        # Payout ratio (sustainable)
        if 30 < stock['payout_ratio'] < 50:
            score += 20
        elif stock['payout_ratio'] < 60:
            score += 10

        # Profitability
        if stock['profit_margin'] > 15:
            score += 20
        elif stock['profit_margin'] > 10:
            score += 10

        # Low debt
        if stock['debt_equity'] < 0.5:
            score += 20
        elif stock['debt_equity'] < 1.0:
            score += 10

    elif preset_name == 'quality':
        # ROE
        if stock['roe'] > 20:
            score += 25
        elif stock['roe'] > 15:
            score += 15

        # Profit margin
        if stock['profit_margin'] > 20:
            score += 25
        elif stock['profit_margin'] > 15:
            score += 15

        # P/E (reasonable)
        if 15 < stock['pe'] < 25:
            score += 15
        elif stock['pe'] < 30:
            score += 10

        # Sharpe
        if stock['sharpe'] > 1:
            score += 15
        elif stock['sharpe'] > 0.5:
            score += 10

        # Low debt
        if stock['debt_equity'] < 0.5:
            score += 10

    elif preset_name == 'momentum':
        # Recent returns
        if stock['returns_1m'] > 10:
            score += 30
        elif stock['returns_1m'] > 5:
            score += 20

        if stock['returns_3m'] > 25:
            score += 30
        elif stock['returns_3m'] > 15:
            score += 20

        # RSI (not overbought)
        if 50 < stock['rsi'] < 65:
            score += 20
        elif stock['rsi'] < 70:
            score += 10

        # Volume/momentum
        if stock['returns_1y'] > 20:
            score += 20

    elif preset_name == 'undervalued':
        # Low valuation
        if 0 < stock['pe'] < 15:
            score += 25
        elif 0 < stock['pe'] < 20:
            score += 15

        if 0 < stock['pb'] < 1.5:
            score += 20
        elif 0 < stock['pb'] < 2:
            score += 10

        # High upside
        if stock['upside'] > 25:
            score += 30
        elif stock['upside'] > 15:
            score += 20

        # Analyst sentiment
        if stock['recommendation'] == 'STRONG_BUY':
            score += 15
        elif stock['recommendation'] == 'BUY':
            score += 10

    return score


def screen_stocks(universe='mega_cap', preset='value'):
    """Screen stocks"""
    console.print()
    console.print(Panel(f"[bold cyan]Stock Screener: {SCREENING_PRESETS[preset]['name']}[/bold cyan]\n[bright_black]Universe: {universe.upper()} ({len(STOCK_UNIVERSES[universe])} stocks)[/bright_black]", border_style=THEME['border'], box=box.SIMPLE_HEAVY))
    console.print()

    # Fetch data
    stocks_data = []
    for symbol in track(STOCK_UNIVERSES[universe], description="Fetching data..."):
        data = fetch_stock_metrics(symbol)
        if data:
            stocks_data.append(data)

    console.print(f"\n[green]✓[/green] Fetched data for {len(stocks_data)} stocks")

    # Apply screening
    criteria = SCREENING_PRESETS[preset]['criteria']
    filtered_stocks = apply_screening_criteria(stocks_data, criteria)

    # Calculate scores for ALL stocks
    for stock in stocks_data:
        stock['score'] = calculate_score(stock, preset)

    # Sort all stocks by score
    ranked_stocks = sorted(stocks_data, key=lambda x: x['score'], reverse=True)

    # Show results
    if filtered_stocks:
        console.print(f"[green]✓[/green] {len(filtered_stocks)} stocks passed strict criteria\n")
        display_stocks = sorted(filtered_stocks, key=lambda x: x['score'], reverse=True)[:10]
    else:
        console.print(f"[yellow]⚠[/yellow] No stocks passed strict criteria. Showing top 10 by score:\n")
        display_stocks = ranked_stocks[:10]

    if not display_stocks:
        console.print("[red]Error: No data available[/red]")
        return

    # Display criteria
    console.print(Panel(create_criteria_text(criteria), title="[bold white]Screening Criteria[/bold white]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    # Display results (using display_stocks which are already scored and sorted)
    console.print(create_results_table(display_stocks, preset))

    # Display top picks
    if len(display_stocks) >= 3:
        console.print()
        console.print(create_top_picks_panel(display_stocks[:3]))

    # Footer
    console.print()
    passed_count = len(filtered_stocks) if filtered_stocks else 0
    console.print(f"[bright_black]Screened {len(stocks_data)} stocks │ {passed_count} passed strict criteria │ Showing top {len(display_stocks)} │ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bright_black]", justify="center")


def create_criteria_text(criteria):
    """Create criteria display"""
    lines = []
    for key, value in criteria.items():
        if key.endswith('_min'):
            metric = key.replace('_min', '').replace('_', ' ').title()
            lines.append(f"• {metric} > {value}")
        elif key.endswith('_max'):
            metric = key.replace('_max', '').replace('_', ' ').title()
            lines.append(f"• {metric} < {value}")
        elif key == 'analyst_rating':
            lines.append(f"• Analyst Rating: {', '.join(value)}")

    return "\n".join(lines)


def create_results_table(stocks, preset):
    """Create results table"""
    table = Table(
        title=f"TOP STOCK PICKS ({preset.title()})",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("#", justify="right", style="bright_black", width=4)
    table.add_column("Symbol", style="bold cyan", width=8)
    table.add_column("Name", style="white", width=18)
    table.add_column("Price", justify="right", width=9)
    table.add_column("1Y Ret", justify="right", width=9)
    table.add_column("P/E", justify="right", width=6)
    table.add_column("ROE", justify="right", width=7)
    table.add_column("Margin", justify="right", width=7)
    table.add_column("Sharpe", justify="right", width=7)
    table.add_column("Score", justify="right", style="bold green", width=8)

    for i, stock in enumerate(stocks[:10], 1):
        ret_color = 'green' if stock['returns_1y'] > 0 else 'red'

        table.add_row(
            str(i),
            stock['symbol'],
            stock['name'][:25],
            f"${stock['price']:.2f}",
            f"[{ret_color}]{stock['returns_1y']:+.1f}%[/{ret_color}]",
            f"{stock['pe']:.1f}" if stock['pe'] > 0 else "N/A",
            f"{stock['roe']:.1f}%" if stock['roe'] > 0 else "N/A",
            f"{stock['profit_margin']:.1f}%",
            f"{stock['sharpe']:.2f}",
            f"[bold green]{stock['score']}/100[/bold green]"
        )

    return table


def create_top_picks_panel(stocks):
    """Create top 3 picks panel"""
    text = "[bold yellow] TOP 3 PICKS[/bold yellow]\n\n"

    for i, stock in enumerate(stocks, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        text += f"{medal} [bold cyan]{stock['symbol']}[/bold cyan] - {stock['name'][:30]}\n"
        text += f"   Score: [bold green]{stock['score']}/100[/bold green] │ "
        text += f"Price: ${stock['price']:.2f} │ "
        text += f"1Y: [{('green' if stock['returns_1y'] > 0 else 'red')}]{stock['returns_1y']:+.1f}%[/] │ "
        text += f"P/E: {stock['pe']:.1f}\n\n"

    return Panel(text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


if __name__ == "__main__":
    import sys

    # Premium header with blue color scheme
    from datetime import datetime
    from rich.text import Text
    from rich.panel import Panel
    now = datetime.now()

    header = Text()
    header.append("═" * 80, style="bright_blue")
    header.append("\n")
    header.append("                    ", style="bright_blue")
    header.append(" 🔍 PROFESSIONAL STOCK SCREENER 🔍 ", style="bold white on blue")
    header.append("\n", style="bright_blue")
    header.append("═" * 80, style="bright_blue")
    header.append("\n\n")
    header.append("  Find Investment Opportunities Across 75 Stocks", style="bold bright_blue")
    header.append(" • ", style="bright_black")
    header.append(now.strftime('%A, %B %d, %Y at %I:%M:%S %p ET'), style="blue")
    header.append(" • ", style="bright_black")
    header.append("6 Strategies, 5 Universes", style="bold bright_blue")

    console.print(Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
    console.print()

    console.print("[bold bright_blue]Available Universes:[/bold bright_blue]")
    for key, stocks in STOCK_UNIVERSES.items():
        console.print(f"  • [bright_blue]{key}[/bright_blue] - {len(stocks)} stocks")

    console.print("\n[bold bright_blue]Available Presets:[/bold bright_blue]")
    for key, preset in SCREENING_PRESETS.items():
        console.print(f"  • [bright_blue]{key}[/bright_blue] - {preset['name']}")

    console.print("\n[bright_black]Usage: python STOCK_SCREENER.py [universe] [preset][/bright_black]")
    console.print("[bright_black]Example: python STOCK_SCREENER.py tech growth[/bright_black]\n")

    # Get inputs
    universe = sys.argv[1] if len(sys.argv) > 1 else 'mega_cap'
    preset = sys.argv[2] if len(sys.argv) > 2 else 'value'

    if universe not in STOCK_UNIVERSES:
        console.print(f"[red]Error: Unknown universe '{universe}'[/red]")
        sys.exit(1)

    if preset not in SCREENING_PRESETS:
        console.print(f"[red]Error: Unknown preset '{preset}'[/red]")
        sys.exit(1)

    screen_stocks(universe, preset)
