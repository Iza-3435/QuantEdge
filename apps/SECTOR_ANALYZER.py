"""
Sector Analyzer
Analyze sector performance, top players, and trends
"""
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.prompt import Prompt

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

# Sector ETFs for performance tracking
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Materials': 'XLRE',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communications': 'XLC'
}

# Major companies by sector (market leaders)
SECTOR_LEADERS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CSCO', 'AMD', 'CRM', 'ADBE', 'INTC'],
    'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB'],
    'Healthcare': ['UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'BMY'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
    'Industrials': ['CAT', 'HON', 'UPS', 'RTX', 'LMT', 'BA', 'DE', 'GE', 'MMM', 'EMR'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'TGT', 'F'],
    'Consumer Staples': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL', 'MDLZ', 'KHC'],
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'DOW', 'PPG', 'NUE'],
    'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'DLR', 'VICI', 'WELL'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PCG', 'XEL', 'ED'],
    'Communications': ['META', 'GOOGL', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'EA', 'CHTR']
}


def get_sector_performance() -> Dict[str, Dict[str, float]]:
    sector_data = {}

    for sector, etf in SECTOR_ETFS.items():
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period='1y')

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]

                # Calculate returns
                returns_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) >= 2 else 0
                returns_1w = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) >= 5 else 0
                returns_1m = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) >= 21 else 0
                returns_3m = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63] * 100) if len(hist) >= 63 else 0
                returns_1y = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

                sector_data[sector] = {
                    'etf': etf,
                    'price': current_price,
                    'returns_1d': returns_1d,
                    'returns_1w': returns_1w,
                    'returns_1m': returns_1m,
                    'returns_3m': returns_3m,
                    'returns_1y': returns_1y,
                }
        except:
            continue

    return sector_data


def get_sector_leaders_data(sector: str) -> List[Dict[str, Any]]:
    leaders = SECTOR_LEADERS.get(sector, [])
    leaders_data = []

    for symbol in leaders[:5]:  # Top 5
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='3mo')

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                returns_3m = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

                leaders_data.append({
                    'symbol': symbol,
                    'name': info.get('shortName', symbol)[:25],
                    'market_cap': info.get('marketCap', 0),
                    'price': current_price,
                    'pe': info.get('trailingPE', 0),
                    'returns_3m': returns_3m,
                    'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                })
        except:
            continue

    return leaders_data


def create_header() -> Panel:
    header = Text()
    header.append("SECTOR ANALYZER - Deep Dive\n\n", style="bold white")
    header.append("Sector Performance & Market Leaders", style="white")
    header.append(" │ ", style="bright_black")
    header.append(f"Updated: {datetime.now().strftime('%B %d, %Y')}", style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_sector_performance_table(sector_data: Dict[str, Dict]) -> Table:
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['returns_3m'], reverse=True)

    table = Table(
        title="SECTOR PERFORMANCE RANKINGS",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Sector", style="white", width=27)
    table.add_column("ETF", style="white", width=5)
    table.add_column("1D", justify="right", width=9)
    table.add_column("1W", justify="right", width=9)
    table.add_column("1M", justify="right", width=9)
    table.add_column("3M", justify="right", width=9)
    table.add_column("1Y", justify="right", width=9)

    for sector, data in sorted_sectors:
        # Color code based on 3-month performance
        perf_3m = data['returns_3m']
        row_color = COLORS['up'] if perf_3m > 0 else COLORS['down']

        table.add_row(
            sector,
            data['etf'],
            f"[{COLORS['up'] if data['returns_1d'] > 0 else COLORS['down']}]{data['returns_1d']:+.2f}%[/]",
            f"[{COLORS['up'] if data['returns_1w'] > 0 else COLORS['down']}]{data['returns_1w']:+.2f}%[/]",
            f"[{COLORS['up'] if data['returns_1m'] > 0 else COLORS['down']}]{data['returns_1m']:+.2f}%[/]",
            f"[{row_color}]{perf_3m:+.2f}%[/{row_color}]",
            f"[{COLORS['up'] if data['returns_1y'] > 0 else COLORS['down']}]{data['returns_1y']:+.2f}%[/]",
        )

    return table


def create_sector_leaders_table(sector: str, leaders_data: List[Dict]) -> Table:
    table = Table(
        title=f"{sector.upper()} - MARKET LEADERS",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=25)
    table.add_column("Market Cap", justify="right", width=12)
    table.add_column("P/E", justify="right", width=8)
    table.add_column("3M Return", justify="right", width=12)
    table.add_column("Div Yield", justify="right", width=10)

    for leader in leaders_data:
        # Format market cap
        market_cap = leader['market_cap']
        if market_cap > 1e12:
            cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            cap_str = f"${market_cap/1e9:.2f}B"
        else:
            cap_str = "N/A"

        # Color code return
        ret_3m = leader['returns_3m']
        ret_color = COLORS['up'] if ret_3m > 0 else COLORS['down']

        pe = f"{leader['pe']:.1f}" if leader['pe'] > 0 else "N/A"
        div_yield = f"{leader['dividend_yield']:.2f}%" if leader['dividend_yield'] > 0 else "N/A"

        table.add_row(
            leader['symbol'],
            leader['name'],
            cap_str,
            pe,
            f"[{ret_color}]{ret_3m:+.2f}%[/{ret_color}]",
            div_yield
        )

    return table


def create_rotation_panel(sector_data: Dict[str, Dict]) -> Panel:
    momentum = []
    for sector, data in sector_data.items():
        if data['returns_1m'] > data['returns_3m']:
            momentum.append((sector, 'Accelerating'))
        elif data['returns_1m'] < data['returns_3m']:
            momentum.append((sector, 'Decelerating'))

    text = Text()
    text.append("SECTOR ROTATION TRENDS\n\n", style="bold white")

    # Accelerating sectors
    accel = [s for s, t in momentum if t == 'Accelerating']
    if accel:
        text.append("Accelerating (Gaining Momentum):\n", style="green")
        for sector in accel[:3]:
            text.append(f"  • {sector}\n", style="white")
        text.append("\n")

    # Decelerating sectors
    decel = [s for s, t in momentum if t == 'Decelerating']
    if decel:
        text.append("Decelerating (Losing Momentum):\n", style="red")
        for sector in decel[:3]:
            text.append(f"  • {sector}\n", style="white")

    return Panel(text, title="MOMENTUM ANALYSIS", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def create_defensive_vs_cyclical(sector_data: Dict[str, Dict]) -> Panel:
    defensive = ['Utilities', 'Consumer Staples', 'Healthcare']
    cyclical = ['Technology', 'Consumer Discretionary', 'Industrials', 'Materials']

    defensive_avg = sum(sector_data.get(s, {}).get('returns_3m', 0) for s in defensive) / len(defensive)
    cyclical_avg = sum(sector_data.get(s, {}).get('returns_3m', 0) for s in cyclical) / len(cyclical)

    text = Text()
    text.append("DEFENSIVE VS CYCLICAL\n\n", style="bold white")

    text.append("Defensive Sectors (3M Avg): ", style="white")
    color = COLORS['up'] if defensive_avg > 0 else COLORS['down']
    text.append(f"{defensive_avg:+.2f}%\n", style=color)

    text.append("Cyclical Sectors (3M Avg):  ", style="white")
    color = COLORS['up'] if cyclical_avg > 0 else COLORS['down']
    text.append(f"{cyclical_avg:+.2f}%\n\n", style=color)

    if cyclical_avg > defensive_avg:
        text.append("Market Stance: ", style="white")
        text.append("Risk-On (Cyclicals leading)", style="green")
    else:
        text.append("Market Stance: ", style="white")
        text.append("Risk-Off (Defensives leading)", style="red")

    return Panel(text, title="MARKET RISK APPETITE", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def main() -> None:
    console.clear()
    console.print(create_header())
    console.print()

    # Fetch sector performance
    console.print("[white]Loading sector performance data...[/white]")
    sector_data = get_sector_performance()

    console.clear()
    console.print(create_header())
    console.print()

    # Sector performance table
    console.print(create_sector_performance_table(sector_data))
    console.print()

    # Side by side: Rotation + Defensive vs Cyclical
    console.print(Columns([
        create_rotation_panel(sector_data),
        create_defensive_vs_cyclical(sector_data)
    ]))
    console.print()

    # Get user sector selection for deep dive
    sectors_list = sorted(SECTOR_LEADERS.keys())
    console.print("[bold white]Select a sector for deep dive (or press Enter to skip):[/bold white]\n")

    for i, sector in enumerate(sectors_list, 1):
        console.print(f"  [{i}] {sector}")

    console.print()
    choice = Prompt.ask("Enter number (1-11)", default="")

    if choice and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sectors_list):
            selected_sector = sectors_list[idx]

            console.clear()
            console.print(create_header())
            console.print()

            # Show selected sector leaders
            console.print(f"[white]Loading top companies in {selected_sector}...[/white]")
            leaders_data = get_sector_leaders_data(selected_sector)

            console.clear()
            console.print(create_header())
            console.print()

            console.print(create_sector_leaders_table(selected_sector, leaders_data))
            console.print()

    # Footer
    footer = Panel(
        f"[bright_black]Tracking 11 sectors │ {sum(len(v) for v in SECTOR_LEADERS.values())} companies │ Updated {datetime.now().strftime('%I:%M %p')}[/bright_black]",
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
