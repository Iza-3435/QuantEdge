"""
TECHNICAL SCREENER
Find stocks with bullish technical setups

Signals:
- Golden Cross (50 SMA > 200 SMA)
- 52-Week Breakout
- RSI Oversold Bounce
- Volume Spike
- Price Momentum
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

# Import stock universe
from stock_universe import SP500_STOCKS, create_progress_bar, get_status_indicator

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

# Stock universe to scan - now using S&P 500
STOCK_UNIVERSE = SP500_STOCKS


def get_technical_data(symbol):
    """Get technical analysis data for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get 1 year of history
        hist = ticker.history(period='1y')

        if hist.empty or len(hist) < 200:
            return None

        current_price = hist['Close'].iloc[-1]

        # Calculate moving averages
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1]

        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-5] if len(rsi) > 5 else rsi_current

        # 52-week high/low
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()

        # Volume analysis
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Price momentum
        returns_1m = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) >= 21 else 0
        returns_3m = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63] * 100) if len(hist) >= 63 else 0

        # Distance from highs/lows
        pct_from_52w_high = ((current_price - high_52w) / high_52w * 100)
        pct_from_52w_low = ((current_price - low_52w) / low_52w * 100)

        data = {
            'symbol': symbol,
            'name': info.get('shortName', symbol)[:25],
            'price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': rsi_current,
            'rsi_prev': rsi_prev,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'pct_from_52w_high': pct_from_52w_high,
            'pct_from_52w_low': pct_from_52w_low,
            'volume_ratio': volume_ratio,
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'avg_volume': avg_volume,
        }

        return data

    except Exception as e:
        return None


def check_golden_cross(data):
    """Check for golden cross setup (50 SMA > 200 SMA)"""
    if data['sma_50'] > data['sma_200'] and data['price'] > data['sma_50']:
        strength = ((data['sma_50'] - data['sma_200']) / data['sma_200']) * 100
        return True, strength
    return False, 0


def check_52w_breakout(data):
    """Check for 52-week breakout"""
    if data['pct_from_52w_high'] >= -2:  # Within 2% of 52W high
        return True, abs(data['pct_from_52w_high'])
    return False, 0


def check_rsi_bounce(data):
    """Check for RSI oversold bounce"""
    # Was oversold (RSI < 35) and now bouncing (RSI > 40)
    if data['rsi_prev'] < 35 and data['rsi'] > 40 and data['rsi'] < 60:
        return True, data['rsi']
    return False, 0


def check_volume_spike(data):
    """Check for volume spike"""
    if data['volume_ratio'] > 1.5:  # 50% above average
        return True, data['volume_ratio']
    return False, 0


def check_momentum(data):
    """Check for strong momentum"""
    # Strong 1M and 3M returns
    if data['returns_1m'] > 5 and data['returns_3m'] > 10:
        return True, data['returns_3m']
    return False, 0


def scan_stocks():
    """Scan all stocks for technical setups"""
    console.print("\n[white]Scanning stocks for technical setups...[/white]\n")

    results = {
        'golden_cross': [],
        '52w_breakout': [],
        'rsi_bounce': [],
        'volume_spike': [],
        'momentum': []
    }

    for symbol in STOCK_UNIVERSE:
        data = get_technical_data(symbol)
        if not data:
            continue

        # Check each setup
        gc_signal, gc_strength = check_golden_cross(data)
        if gc_signal:
            results['golden_cross'].append({**data, 'strength': gc_strength})

        breakout_signal, breakout_strength = check_52w_breakout(data)
        if breakout_signal:
            results['52w_breakout'].append({**data, 'strength': breakout_strength})

        rsi_signal, rsi_strength = check_rsi_bounce(data)
        if rsi_signal:
            results['rsi_bounce'].append({**data, 'strength': rsi_strength})

        vol_signal, vol_strength = check_volume_spike(data)
        if vol_signal:
            results['volume_spike'].append({**data, 'strength': vol_strength})

        mom_signal, mom_strength = check_momentum(data)
        if mom_signal:
            results['momentum'].append({**data, 'strength': mom_strength})

    return results


def create_header():
    """Create header"""
    # Check if market is open (simple check: Mon-Fri, 9:30am-4pm ET)
    now = datetime.now()
    is_weekday = now.weekday() < 5
    hour = now.hour
    is_market_hours = 9 <= hour < 16
    market_status = 'live' if (is_weekday and is_market_hours) else 'closed'

    header = Text()
    header.append("TECHNICAL SCREENER\n\n", style="bold white")
    header.append(get_status_indicator(market_status), style="")
    header.append(" │ ", style="bright_black")
    header.append("Find Bullish Technical Setups", style="white")
    header.append(" │ ", style="bright_black")
    header.append(f"Scanning {len(STOCK_UNIVERSE)} stocks", style="white")
    header.append(" │ ", style="bright_black")
    header.append(datetime.now().strftime('%B %d, %Y'), style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_setup_table(stocks, title, description):
    """Create table for a specific setup"""
    if not stocks:
        return Panel(
            f"[bright_black]No stocks found with this setup[/bright_black]",
            title=f"[bold white]{title}[/bold white]",
            subtitle=f"[bright_black]{description}[/bright_black]",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )

    table = Table(
        title=f"{title} ({len(stocks)} stocks)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=20)
    table.add_column("Price", justify="right", width=10)
    table.add_column("RSI", justify="right", width=6)
    table.add_column("1M", justify="right", width=8)
    table.add_column("Strength", width=28)

    # Sort by strength
    sorted_stocks = sorted(stocks, key=lambda x: x.get('strength', 0), reverse=True)

    for stock in sorted_stocks[:10]:  # Top 10
        # Color code returns
        ret_1m = stock['returns_1m']
        ret_color = COLORS['up'] if ret_1m > 0 else COLORS['down']

        # RSI color
        rsi = stock['rsi']
        if rsi < 30:
            rsi_color = "red"
        elif rsi > 70:
            rsi_color = "yellow"
        else:
            rsi_color = "white"

        # Create strength progress bar
        strength = stock['strength']
        strength_bar = create_progress_bar(strength, max_value=100, width=15, show_percentage=True)

        table.add_row(
            stock['symbol'],
            stock['name'][:18] + '..' if len(stock['name']) > 20 else stock['name'],
            f"${stock['price']:.2f}",
            f"[{rsi_color}]{rsi:.0f}[/{rsi_color}]",
            f"[{ret_color}]{ret_1m:+.1f}%[/{ret_color}]",
            strength_bar
        )

    return Panel(
        table,
        subtitle=f"[bright_black]{description}[/bright_black]",
        border_style=THEME['border'],
        style=THEME['panel_bg']
    )


def create_summary_panel(results):
    """Create summary of all setups"""
    text = Text()
    text.append("SCAN SUMMARY\n\n", style="bold white")

    total_opportunities = sum(len(stocks) for stocks in results.values())

    text.append(f"Total Setups Found: ", style="white")
    text.append(f"{total_opportunities}\n\n", style="bold green")

    text.append("Breakdown:\n", style="bold white")
    text.append(f"  • Golden Cross: ", style="white")
    text.append(f"{len(results['golden_cross'])}\n", style="green")

    text.append(f"  • 52W Breakout: ", style="white")
    text.append(f"{len(results['52w_breakout'])}\n", style="green")

    text.append(f"  • RSI Bounce: ", style="white")
    text.append(f"{len(results['rsi_bounce'])}\n", style="green")

    text.append(f"  • Volume Spike: ", style="white")
    text.append(f"{len(results['volume_spike'])}\n", style="green")

    text.append(f"  • Momentum: ", style="white")
    text.append(f"{len(results['momentum'])}\n", style="green")

    return Panel(text, title="[bold white]SUMMARY[/bold white]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def main():
    """Main function"""
    console.clear()
    console.print(create_header())
    console.print()

    # Scan stocks
    results = scan_stocks()

    console.clear()
    console.print(create_header())
    console.print()

    # Display summary
    console.print(create_summary_panel(results))
    console.print()

    # Display each setup
    console.print(create_setup_table(
        results['golden_cross'],
        "GOLDEN CROSS",
        "50 SMA > 200 SMA (Bullish long-term trend)"
    ))
    console.print()

    console.print(create_setup_table(
        results['52w_breakout'],
        "52-WEEK BREAKOUT",
        "Near or breaking all-time highs"
    ))
    console.print()

    console.print(create_setup_table(
        results['rsi_bounce'],
        "RSI OVERSOLD BOUNCE",
        "Bouncing from oversold territory"
    ))
    console.print()

    console.print(Columns([
        create_setup_table(
            results['volume_spike'],
            "VOLUME SPIKE",
            "Trading above average volume"
        ),
        create_setup_table(
            results['momentum'],
            "STRONG MOMENTUM",
            "Trending up 1M and 3M"
        )
    ]))
    console.print()

    # Footer
    footer = Panel(
        f"[bright_black]Scanned {len(STOCK_UNIVERSE)} stocks │ 5 technical setups │ Updated {datetime.now().strftime('%I:%M %p')}[/bright_black]",
        box=box.SQUARE,
        border_style=THEME['border'],
        style=THEME['panel_bg']
    )
    console.print(footer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)
