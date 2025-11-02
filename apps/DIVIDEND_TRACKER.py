"""
DIVIDEND TRACKER & CALENDAR
Track dividend payments, yields, and growth for income investing

Features:
- Upcoming ex-dividend dates
- Dividend yield rankings
- Payout ratio analysis
- Dividend growth trends
- DRIP calculator
- Income projections
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.prompt import Prompt
import sys
import warnings
from typing import Dict, List, Optional, Any
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


def create_progress_bar(value: float, max_value: float = 10, width: int = 12, show_percentage: bool = False) -> str:
    if max_value == 0:
        max_value = 1

    filled_width = int((value / max_value) * width)
    filled_width = min(filled_width, width)

    bar = '█' * filled_width + '░' * (width - filled_width)

    if show_percentage:
        pct = (value / max_value) * 100
        return f"[cyan]{bar}[/cyan] {pct:.0f}%"

    return f"[cyan]{bar}[/cyan]"


DIVIDEND_STOCKS = [
    'T', 'VZ', 'MO', 'BTI', 'ABBV', 'CVX', 'XOM', 'PFE', 'BMY', 'D',
    'JNJ', 'PG', 'KO', 'PEP', 'MMM', 'WMT', 'TGT', 'LOW', 'CAT', 'MCD',
    'AAPL', 'MSFT', 'JPM', 'V', 'MA', 'HD', 'UNH', 'LLY', 'COST', 'NVDA',
    'O', 'SPG', 'VICI', 'DLR', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'AVB'
]


def get_dividend_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch comprehensive dividend data for a stock symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        data: Dict[str, Any] = {
            'symbol': symbol,
            'name': info.get('shortName', symbol)[:25],
            'sector': info.get('sector', 'N/A')[:20],
            'price': info.get('currentPrice', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'annual_dividend': info.get('dividendRate', 0),
            'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
            'ex_dividend_date': info.get('exDividendDate', None),
            'dividend_date': info.get('dividendDate', None),
            '5y_avg_yield': info.get('fiveYearAvgDividendYield', 0),
            'dividend_streak': 0,
            'is_aristocrat': False,
            'is_achiever': False,
        }

        try:
            dividends = ticker.dividends
            if not dividends.empty and len(dividends) >= 2:
                dividends_yearly = dividends.resample('Y').sum()

                if len(dividends_yearly) >= 2:
                    current_year_div = dividends_yearly.iloc[-1]
                    last_year_div = dividends_yearly.iloc[-2]

                    if last_year_div > 0:
                        growth = ((current_year_div - last_year_div) / last_year_div) * 100
                        data['div_growth_1y'] = growth

                if len(dividends_yearly) >= 5:
                    five_years_ago = dividends_yearly.iloc[-5]
                    current = dividends_yearly.iloc[-1]

                    if five_years_ago > 0:
                        cagr = ((current / five_years_ago) ** (1/5) - 1) * 100
                        data['div_growth_5y'] = cagr

                try:
                    streak = 0
                    for i in range(len(dividends_yearly) - 1, 0, -1):
                        if dividends_yearly.iloc[i] > dividends_yearly.iloc[i-1]:
                            streak += 1
                        else:
                            break

                    data['dividend_streak'] = streak

                    if streak >= 25:
                        data['is_aristocrat'] = True
                    elif streak >= 10:
                        data['is_achiever'] = True
                except:
                    pass
        except:
            pass

        if data['ex_dividend_date']:
            try:
                ex_date = datetime.fromtimestamp(data['ex_dividend_date'])
                days_until = (ex_date - datetime.now()).days
                data['days_until_ex'] = days_until
                data['ex_date_str'] = ex_date.strftime('%Y-%m-%d')
            except:
                data['days_until_ex'] = None

        if data['dividend_date']:
            try:
                pay_date = datetime.fromtimestamp(data['dividend_date'])
                data['pay_date_str'] = pay_date.strftime('%Y-%m-%d')
            except:
                data['pay_date_str'] = None

        return data

    except Exception as e:
        return None


def create_header() -> Panel:
    """Create header panel for dividend tracker"""
    header = Text()
    header.append("DIVIDEND TRACKER & CALENDAR\n\n", style="bold white")
    header.append("Income Investing Dashboard", style="white")
    header.append(" │ ", style="bright_black")
    header.append(f"Updated: {datetime.now().strftime('%B %d, %Y')}", style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_top_yields_table(dividend_data: List[Dict[str, Any]]) -> Table:
    """Create top dividend yields ranking table"""
    sorted_data = sorted(
        [d for d in dividend_data if d and d['dividend_yield'] > 0],
        key=lambda x: x['dividend_yield'],
        reverse=True
    )

    table = Table(
        title="TOP DIVIDEND YIELDS",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=22)
    table.add_column("Yield", justify="right", width=8)
    table.add_column("Strength", width=22)
    table.add_column("Annual", justify="right", width=10)

    max_yield = 10.0

    for stock in sorted_data[:15]:
        div_yield = stock['dividend_yield']
        if div_yield >= 5:
            yield_color = "green"
        elif div_yield >= 3:
            yield_color = "white"
        else:
            yield_color = "bright_black"

        yield_bar = create_progress_bar(div_yield, max_value=max_yield, width=12, show_percentage=False)

        payout = stock['payout_ratio']
        if payout > 80:
            payout_color = "red"
        elif payout > 60:
            payout_color = "yellow"
        else:
            payout_color = "white"

        table.add_row(
            stock['symbol'],
            stock['name'][:20] + '..' if len(stock['name']) > 22 else stock['name'],
            f"[{yield_color}]{div_yield:.2f}%[/{yield_color}]",
            yield_bar,
            f"${stock['annual_dividend']:.2f}"
        )

    return table


def create_upcoming_ex_dates_table(dividend_data: List[Dict[str, Any]]) -> Panel | Table:
    """Create upcoming ex-dividend dates table"""
    upcoming = []
    for data in dividend_data:
        if data and data.get('days_until_ex') is not None:
            days = data['days_until_ex']
            if 0 <= days <= 60:
                upcoming.append(data)

    upcoming.sort(key=lambda x: x['days_until_ex'] or 0)

    if not upcoming:
        return Panel(
            "[bright_black]No upcoming ex-dividend dates in the next 60 days[/bright_black]",
            title="UPCOMING EX-DIVIDEND DATES",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )

    table = Table(
        title="UPCOMING EX-DIVIDEND DATES (Next 60 Days)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Ex-Date", justify="right", width=12)
    table.add_column("Pay Date", justify="right", width=12)
    table.add_column("Days", justify="right", width=5)
    table.add_column("Yield", justify="right", width=9)
    table.add_column("Amount", justify="right", width=10)

    for stock in upcoming[:15]:
        days = stock['days_until_ex']
        if days <= 7:
            days_color = "red"
        elif days <= 14:
            days_color = "yellow"
        else:
            days_color = "white"

        pay_date = stock.get('pay_date_str', '—')

        table.add_row(
            stock['symbol'],
            stock.get('ex_date_str', 'N/A'),
            pay_date,
            f"[{days_color}]{days}d[/{days_color}]",
            f"{stock['dividend_yield']:.2f}%",
            f"${stock['annual_dividend']:.2f}"
        )

    return table


def create_growth_leaders_table(dividend_data: List[Dict[str, Any]]) -> Table:
    """Create dividend growth leaders table (5-year CAGR)"""
    with_growth = [d for d in dividend_data if d and d.get('div_growth_5y')]
    with_growth.sort(key=lambda x: x['div_growth_5y'], reverse=True)

    table = Table(
        title="DIVIDEND GROWTH LEADERS (5-Year CAGR)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1),
        expand=False
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=25)
    table.add_column("5Y Growth", justify="right", width=12)
    table.add_column("Yield", justify="right", width=10)

    for stock in with_growth[:12]:
        growth_5y = stock['div_growth_5y']
        growth_color = COLORS['up'] if growth_5y > 5 else COLORS['neutral']

        table.add_row(
            stock['symbol'],
            stock['name'],
            f"[{growth_color}]{growth_5y:+.1f}%[/{growth_color}]",
            f"{stock['dividend_yield']:.2f}%"
        )

    return table


def create_sector_yields_table(dividend_data: List[Dict[str, Any]]) -> Table:
    """Create average dividend yields by sector table"""
    sectors: Dict[str, List[float]] = {}
    for data in dividend_data:
        if data and data['dividend_yield'] > 0:
            sector = data['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(data['dividend_yield'])

    sector_avgs = [(sector, sum(yields) / len(yields)) for sector, yields in sectors.items()]
    sector_avgs.sort(key=lambda x: x[1], reverse=True)

    table = Table(
        title="AVERAGE YIELDS BY SECTOR",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1),
        expand=False
    )

    table.add_column("Sector", style="white", width=25)
    table.add_column("Avg Yield", justify="right", width=12)
    table.add_column("Companies", justify="right", width=10)

    for sector, avg_yield in sector_avgs:
        count = len(sectors[sector])
        table.add_row(
            sector,
            f"{avg_yield:.2f}%",
            str(count)
        )

    return table


def create_dividend_aristocrats_table(dividend_data: List[Dict[str, Any]]) -> Optional[Table]:
    """Create dividend aristocrats and achievers table"""
    with_streaks = [d for d in dividend_data if d and d.get('dividend_streak', 0) >= 10]
    with_streaks.sort(key=lambda x: x['dividend_streak'], reverse=True)

    if not with_streaks:
        return None

    table = Table(
        title="DIVIDEND ARISTOCRATS & ACHIEVERS",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=22)
    table.add_column("Streak", justify="right", width=8)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Yield", justify="right", width=9)
    table.add_column("5Y Gr", justify="right", width=9)

    for stock in with_streaks:
        streak = stock['dividend_streak']

        if stock.get('is_aristocrat'):
            status = "ELITE"
            status_color = "bright_green"
        elif stock.get('is_achiever'):
            status = "STRONG"
            status_color = "green"
        else:
            status = "GOOD"
            status_color = "white"

        if streak >= 25:
            streak_color = "bright_green"
        elif streak >= 15:
            streak_color = "green"
        else:
            streak_color = "white"

        growth_5y = stock.get('div_growth_5y', 0)
        growth_str = f"{growth_5y:+.1f}%" if growth_5y else "—"
        growth_color = COLORS['up'] if growth_5y > 0 else COLORS['neutral']

        table.add_row(
            stock['symbol'],
            stock['name'],
            f"[{streak_color}]{streak}y[/{streak_color}]",
            f"[{status_color}]{status}[/{status_color}]",
            f"{stock['dividend_yield']:.2f}%",
            f"[{growth_color}]{growth_str}[/{growth_color}]"
        )

    return table


def create_drip_calculator() -> Panel:
    """Create DRIP projection info panel"""
    text = Text()
    text.append("DIVIDEND REINVESTMENT CALCULATOR\n\n", style="bold white")
    text.append("Calculate income from dividend stocks:\n\n", style="white")

    text.append("Example: ", style="bright_black")
    text.append("$10,000 invested @ 5% yield = ", style="white")
    text.append("$500/year\n", style="green")

    text.append("         ", style="bright_black")
    text.append("With 5% annual growth = ", style="white")
    text.append("$1,276 ", style="green")
    text.append("after 5 years\n\n", style="green")

    text.append("Pro Tip: ", style="bright_black")
    text.append("Diversify across sectors, aim for payout ratio < 60%", style="white")

    return Panel(text, title="INCOME PROJECTIONS", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg'])


def main() -> None:
    console.clear()
    console.print(create_header())
    console.print()

    console.print(f"[white]Loading dividend data for {len(DIVIDEND_STOCKS)} stocks...[/white]")

    dividend_data: List[Dict[str, Any]] = []
    for symbol in DIVIDEND_STOCKS:
        data = get_dividend_data(symbol)
        if data:
            dividend_data.append(data)

    console.clear()
    console.print(create_header())
    console.print()

    console.print(create_top_yields_table(dividend_data))
    console.print()

    console.print(create_upcoming_ex_dates_table(dividend_data))
    console.print()

    console.print(Columns([
        create_growth_leaders_table(dividend_data),
        create_sector_yields_table(dividend_data)
    ]))
    console.print()

    aristocrats = create_dividend_aristocrats_table(dividend_data)
    if aristocrats:
        console.print(aristocrats)
        console.print()

    console.print(create_drip_calculator())
    console.print()

    footer = Panel(
        f"[bright_black]Tracking {len(DIVIDEND_STOCKS)} stocks │ Elite = 25+ years │ Strong = 10+ years │ Updated {datetime.now().strftime('%I:%M %p')}[/bright_black]",
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
