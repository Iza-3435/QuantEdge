"""
HISTORICAL CONTEXT & TRENDS
See if current metrics are normal or unusual over 5-year period

Shows:
- P/E ratio trend (is it expensive vs history?)
- Revenue growth trend (improving or declining?)
- Margin trends (getting better or worse?)
- Stock price trajectory
- Context for all key metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
import sys
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



def fetch_historical_data(symbol):
    """Fetch 5-year historical data"""
    try:
        ticker = yf.Ticker(symbol)

        # Get historical prices
        hist = ticker.history(period='5y')

        # Get quarterly financials
        quarterly_financials = ticker.quarterly_financials
        quarterly_income = ticker.quarterly_income_stmt

        # Current info
        info = ticker.info

        if hist.empty:
            return None

        # Calculate historical metrics
        current_price = hist['Close'].iloc[-1]
        price_5y_ago = hist['Close'].iloc[0]
        price_1y_ago = hist['Close'].iloc[-252] if len(hist) >= 252 else hist['Close'].iloc[0]

        # Get price ranges
        price_5y_high = hist['High'].max()
        price_5y_low = hist['Low'].min()
        price_1y_high = hist['High'].iloc[-252:].max() if len(hist) >= 252 else hist['High'].max()
        price_1y_low = hist['Low'].iloc[-252:].min() if len(hist) >= 252 else hist['Low'].min()

        # Calculate returns
        returns_5y = ((current_price - price_5y_ago) / price_5y_ago * 100)
        returns_1y = ((current_price - price_1y_ago) / price_1y_ago * 100)

        # Annual returns
        yearly_returns = []
        for i in range(5):
            start_idx = max(0, -252 * (i+1))
            end_idx = -252 * i if i > 0 else None
            period_data = hist.iloc[start_idx:end_idx]
            if len(period_data) > 20:
                period_return = ((period_data['Close'].iloc[-1] - period_data['Close'].iloc[0]) /
                               period_data['Close'].iloc[0] * 100)
                yearly_returns.append(period_return)

        # Current metrics
        current_pe = info.get('trailingPE', 0)
        current_pb = info.get('priceToBook', 0)
        current_ps = info.get('priceToSalesTrailing12Months', 0)

        # Try to calculate historical averages
        # Note: This is simplified - real historical P/E would need historical earnings
        pe_proxy_5y = current_pe  # Simplified
        pe_proxy_1y = current_pe

        data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),

            # Price data
            'current_price': current_price,
            'price_5y_ago': price_5y_ago,
            'price_1y_ago': price_1y_ago,
            'price_5y_high': price_5y_high,
            'price_5y_low': price_5y_low,
            'price_1y_high': price_1y_high,
            'price_1y_low': price_1y_low,

            # Returns
            'returns_5y': returns_5y,
            'returns_1y': returns_1y,
            'yearly_returns': yearly_returns,

            # Current metrics
            'current_pe': current_pe,
            'current_pb': current_pb,
            'current_ps': current_ps,
            'current_profit_margin': info.get('profitMargins', 0) * 100,
            'current_roe': info.get('returnOnEquity', 0) * 100,
            'current_revenue_growth': info.get('revenueGrowth', 0) * 100,

            # Price position
            'distance_from_52w_high': ((current_price - price_1y_high) / price_1y_high * 100),
            'distance_from_52w_low': ((current_price - price_1y_low) / price_1y_low * 100),
            'distance_from_5y_high': ((current_price - price_5y_high) / price_5y_high * 100),
            'distance_from_5y_low': ((current_price - price_5y_low) / price_5y_low * 100),
        }

        return data

    except Exception as e:
        console.print(f"[red]Error fetching data: {str(e)}[/red]")
        return None


def create_price_trend_panel(data):
    """Create price trend analysis"""
    text = Text()

    # Current price
    text.append("Current Price: ", style="white")
    text.append(f"${data['current_price']:.2f}\n\n", style="bold bright_white")

    # 5-year performance
    returns_5y = data['returns_5y']
    color_5y = "bright_green" if returns_5y > 0 else "bright_red"
    text.append("5-Year Return: ", style="white")
    text.append(f"{returns_5y:+.1f}%\n", style=color_5y)

    # 1-year performance
    returns_1y = data['returns_1y']
    color_1y = "bright_green" if returns_1y > 0 else "bright_red"
    text.append("1-Year Return: ", style="white")
    text.append(f"{returns_1y:+.1f}%\n\n", style=color_1y)

    # Annual returns
    text.append("Annual Returns:\n", style="bold white")
    for i, ret in enumerate(data['yearly_returns'][:5]):
        year = datetime.now().year - i
        ret_color = "green" if ret > 0 else "red"
        text.append(f"  {year}: ", style="bright_black")
        text.append(f"{ret:+.1f}%\n", style=ret_color)

    return Panel(
        text,
        title="[bold white]PRICE TREND[/bold white]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_price_position_panel(data):
    """Create price position analysis"""
    text = Text()

    # Distance from 52-week high
    dist_52w_high = data['distance_from_52w_high']
    if dist_52w_high >= -5:
        position_52w = "Near 52W High "
        color_52w = "yellow"
    elif dist_52w_high >= -15:
        position_52w = "Off 52W High"
        color_52w = "white"
    else:
        position_52w = f"Off 52W High ({abs(dist_52w_high):.0f}%)"
        color_52w = "green"

    text.append("52-Week High: ", style="white")
    text.append(f"${data['price_1y_high']:.2f}\n", style="bright_black")
    text.append("Position: ", style="white")
    text.append(f"{position_52w}\n\n", style=color_52w)

    # Distance from 52-week low
    dist_52w_low = data['distance_from_52w_low']
    text.append("52-Week Low: ", style="white")
    text.append(f"${data['price_1y_low']:.2f}\n", style="bright_black")
    text.append("Distance: ", style="white")
    text.append(f"+{dist_52w_low:.0f}% above\n\n", style="green" if dist_52w_low > 20 else "yellow")

    # 5-year position
    text.append("5-Year Range:\n", style="bold white")
    text.append(f"  High: ${data['price_5y_high']:.2f}\n", style="bright_black")
    text.append(f"  Low: ${data['price_5y_low']:.2f}\n", style="bright_black")
    text.append(f"  Current: ${data['current_price']:.2f}\n", style="bright_white")

    # Percentage through range
    range_5y = data['price_5y_high'] - data['price_5y_low']
    position_in_range = ((data['current_price'] - data['price_5y_low']) / range_5y * 100) if range_5y > 0 else 50

    text.append(f"\nPosition in 5Y Range: {position_in_range:.0f}%", style="white")

    return Panel(
        text,
        title="[bold white]PRICE POSITION[/bold white]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_valuation_context_panel(data):
    """Create valuation context"""
    text = Text()

    pe = data['current_pe']

    text.append("Current P/E: ", style="white")
    text.append(f"{pe:.1f}\n\n", style="bold bright_white")

    # P/E context
    if pe == 0:
        text.append("No P/E data available", style="bright_black")
    else:
        if pe < 15:
            context = "Cheap (below market average)"
            color = "bright_green"
        elif pe < 25:
            context = "Fair (market average)"
            color = "green"
        elif pe < 35:
            context = "Elevated (above average)"
            color = "yellow"
        elif pe < 50:
            context = "Expensive (premium)"
            color = "red"
        else:
            context = "Very Expensive"
            color = "bright_red"

        text.append("Valuation: ", style="white")
        text.append(f"{context}\n\n", style=color)

        text.append("Context:\n", style="bold white")
        text.append("• P/E < 15: Value territory\n", style="bright_black")
        text.append("• P/E 15-25: Fair value\n", style="bright_black")
        text.append("• P/E 25-35: Growth premium\n", style="bright_black")
        text.append("• P/E > 35: Expensive\n", style="bright_black")

    return Panel(
        text,
        title="[bold white]VALUATION CONTEXT[/bold white]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_growth_context_panel(data):
    """Create growth context"""
    text = Text()

    rev_growth = data['current_revenue_growth']

    text.append("Revenue Growth: ", style="white")
    text.append(f"{rev_growth:+.1f}%\n\n", style="bold bright_white")

    # Growth context
    if rev_growth > 25:
        context = "Hyper-growth "
        color = "bright_green"
    elif rev_growth > 15:
        context = "Strong growth"
        color = "green"
    elif rev_growth > 5:
        context = "Moderate growth"
        color = "yellow"
    elif rev_growth > 0:
        context = "Slow growth"
        color = "yellow"
    else:
        context = "Declining "
        color = "red"

    text.append("Growth Rate: ", style="white")
    text.append(f"{context}\n\n", style=color)

    # Profit margin
    margin = data['current_profit_margin']
    text.append("Profit Margin: ", style="white")
    text.append(f"{margin:.1f}%\n", style="bright_white")

    if margin > 20:
        margin_quality = "Excellent"
        margin_color = "bright_green"
    elif margin > 10:
        margin_quality = "Good"
        margin_color = "green"
    elif margin > 5:
        margin_quality = "Fair"
        margin_color = "yellow"
    else:
        margin_quality = "Poor"
        margin_color = "red"

    text.append("Quality: ", style="white")
    text.append(f"{margin_quality}\n\n", style=margin_color)

    # ROE
    roe = data['current_roe']
    text.append("ROE: ", style="white")
    text.append(f"{roe:.1f}%", style="bright_white")

    if roe > 20:
        text.append(" (Outstanding)", style="bright_green")
    elif roe > 15:
        text.append(" (Excellent)", style="green")
    elif roe > 10:
        text.append(" (Good)", style="yellow")

    return Panel(
        text,
        title="[bold white]GROWTH CONTEXT[/bold white]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_historical_summary(data):
    """Create historical summary panel"""
    text = Text()

    # 5-year summary
    returns_5y = data['returns_5y']
    avg_annual = returns_5y / 5

    text.append("5-YEAR SUMMARY:\n\n", style="bold white")

    text.append("Total Return: ", style="white")
    color = "bright_green" if returns_5y > 0 else "bright_red"
    text.append(f"{returns_5y:+.1f}%\n", style=color)

    text.append("Avg Annual Return: ", style="white")
    text.append(f"{avg_annual:+.1f}%\n\n", style=color)

    # Consistency
    positive_years = sum(1 for r in data['yearly_returns'] if r > 0)
    total_years = len(data['yearly_returns'])
    consistency = (positive_years / total_years * 100) if total_years > 0 else 0

    text.append("Positive Years: ", style="white")
    text.append(f"{positive_years}/{total_years} ", style="bright_white")
    text.append(f"({consistency:.0f}%)\n\n", style="green" if consistency > 60 else "yellow")

    # Current momentum
    returns_1y = data['returns_1y']
    if returns_1y > avg_annual * 2:
        momentum = "Strong upward momentum "
        momentum_color = "bright_green"
    elif returns_1y > avg_annual:
        momentum = "Positive momentum"
        momentum_color = "green"
    elif returns_1y > 0:
        momentum = "Moderate momentum"
        momentum_color = "yellow"
    else:
        momentum = "Negative momentum "
        momentum_color = "red"

    text.append("Momentum: ", style="white")
    text.append(momentum, style=momentum_color)

    return Panel(
        text,
        title="[bold white]HISTORICAL SUMMARY[/bold white]",
        border_style=THEME['border'],
        box=box.SQUARE,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_header(data):
    """Create header"""
    header = Text()
    header.append(f"HISTORICAL CONTEXT: {data['symbol']}\n\n", style="bold white")
    header.append(f"{data['name']}", style="white")
    header.append(" • ", style="bright_black")
    header.append(f"{data['sector']}", style="white")
    header.append(" • ", style="bright_black")
    header.append(f"5-Year Analysis", style="white")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def display_historical_context(symbol):
    """Display historical context"""
    console.clear()
    console.print(f"\n[white]Analyzing {symbol} historical data...[/white]\n")

    # Fetch data
    data = fetch_historical_data(symbol)

    if not data:
        console.print(f"[red]Error: Could not fetch data for {symbol}[/red]\n")
        return

    console.clear()

    # Display header
    console.print(create_header(data))
    console.print()

    # Row 1: Price Trend & Position
    console.print(Columns([
        create_price_trend_panel(data),
        create_price_position_panel(data)
    ]))
    console.print()

    # Row 2: Valuation & Growth
    console.print(Columns([
        create_valuation_context_panel(data),
        create_growth_context_panel(data)
    ]))
    console.print()

    # Summary
    console.print(create_historical_summary(data))
    console.print()

    # Footer
    footer = Panel(
        "[bright_black]Historical Context • 5-Year Analysis • Helps understand if current metrics are normal or unusual[/bright_black]",
        box=box.SQUARE,
        border_style=THEME['border']
    , style=THEME['panel_bg'])
    console.print(footer)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        console.print("\n[yellow]Usage:[/yellow] python HISTORICAL_CONTEXT.py <SYMBOL>")
        console.print("\n[white]Example:[/white] python HISTORICAL_CONTEXT.py AAPL\n")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    try:
        display_historical_context(symbol)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
