"""
Earnings Calendar
Track upcoming earnings announcements and historical surprises
"""
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns

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

# Major stocks to track (top 100+ by market cap and trading volume)
TRACKING_STOCKS = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOG',
    # Large Cap Tech
    'ORCL', 'ADBE', 'CRM', 'NFLX', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO',
    'INTU', 'IBM', 'NOW', 'AMAT', 'SNOW', 'MU', 'PANW', 'PLTR', 'SHOP', 'UBER',
    # Financials
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'SCHW',
    'BLK', 'AXP', 'SPGI', 'USB', 'PNC', 'TFC', 'COF', 'PYPL', 'SQ',
    # Healthcare
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
    'GILD', 'CVS', 'CI', 'HCA', 'ISRG', 'VRTX', 'REGN', 'ZTS', 'HUM',
    # Consumer
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT',
    'LOW', 'BKNG', 'TJX', 'DIS', 'CMCSA', 'CHTR', 'MAR', 'ABNB', 'LULU',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO', 'OXY', 'HAL',
    # Industrials
    'UPS', 'HON', 'RTX', 'CAT', 'GE', 'BA', 'LMT', 'DE', 'UNP', 'FDX',
    'MMM', 'GD', 'NOC', 'EMR', 'ETN', 'CARR', 'PCAR',
    # Other
    'NEE', 'DUK', 'SO', 'PM', 'MO', 'MDLZ', 'CL', 'KMB', 'GIS', 'K'
]


def get_earnings_data(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get earnings dates
        earnings_dates = ticker.earnings_dates

        # Get basic info
        data = {
            'symbol': symbol,
            'name': info.get('shortName', symbol)[:25],
            'sector': info.get('sector', 'N/A')[:20],
            'market_cap': info.get('marketCap', 0),
            'earnings_date': None,
            'earnings_time': None,  # BMO or AMC
            'eps_estimate': info.get('forwardEps', 0),
            'last_eps': 0,
            'last_surprise': 0,
            'last_surprise_pct': 0,
            'price_move_after_earnings': 0,  # Price change after last earnings
            'eps_history': [],  # Last 4 quarters
        }

        # Get earnings history
        try:
            earnings = ticker.earnings_history
            if earnings is not None and not earnings.empty:
                # Get most recent earnings
                recent = earnings.iloc[0]
                data['last_eps'] = recent.get('epsActual', 0)
                data['last_surprise'] = recent.get('epsDifference', 0)

                eps_estimate = recent.get('epsEstimate', 0)
                if eps_estimate and eps_estimate != 0:
                    data['last_surprise_pct'] = (data['last_surprise'] / eps_estimate) * 100

                # Get last 4 quarters of EPS
                for i in range(min(4, len(earnings))):
                    quarter = earnings.iloc[i]
                    data['eps_history'].append({
                        'eps': quarter.get('epsActual', 0),
                        'surprise': (quarter.get('epsDifference', 0) / quarter.get('epsEstimate', 1) * 100) if quarter.get('epsEstimate') else 0
                    })

                try:
                    last_earnings_date = earnings.index[0]
                    if pd.notna(last_earnings_date):
                        if hasattr(last_earnings_date, 'to_pydatetime'):
                            last_earnings_date = last_earnings_date.to_pydatetime()

                        if last_earnings_date.tzinfo is not None:
                            last_earnings_date = last_earnings_date.replace(tzinfo=None)

                        start_date = last_earnings_date - timedelta(days=3)
                        end_date = last_earnings_date + timedelta(days=3)

                        hist = ticker.history(start=start_date, end=end_date)
                        if not hist.empty and len(hist) >= 2:
                            before_price = None
                            after_price = None

                            for i, (date, row) in enumerate(hist.iterrows()):
                                date_only = date.date() if hasattr(date, 'date') else date
                                earnings_date_only = last_earnings_date.date() if hasattr(last_earnings_date, 'date') else last_earnings_date

                                if date_only <= earnings_date_only and before_price is None:
                                    before_price = row['Close']
                                elif date_only > earnings_date_only and after_price is None:
                                    after_price = row['Close']
                                    break

                            if before_price and after_price:
                                data['price_move_after_earnings'] = ((after_price - before_price) / before_price) * 100
                except:
                    pass

        except:
            pass

        # Get next earnings date and time
        try:
            if earnings_dates is not None and not earnings_dates.empty:
                future_dates = earnings_dates[earnings_dates.index > datetime.now()]
                if not future_dates.empty:
                    next_date = future_dates.index[0]
                    data['earnings_date'] = next_date

                    # Try to determine BMO vs AMC from time
                    try:
                        hour = next_date.hour if hasattr(next_date, 'hour') else 12
                        if hour < 12:
                            data['earnings_time'] = 'BMO'  # Before Market Open
                        else:
                            data['earnings_time'] = 'AMC'  # After Market Close
                    except:
                        data['earnings_time'] = 'TBD'
        except:
            pass

        return data

    except Exception as e:
        return None


def create_header() -> Panel:
    header = Text()
    header.append("EARNINGS CALENDAR\n\n", style="bold white")
    header.append("Track Upcoming Earnings", style="white")
    header.append(" │ ", style="bright_black")
    header.append(f"Updated: {datetime.now().strftime('%B %d, %Y')}", style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_upcoming_table(earnings_data: List[Dict], days: int = 30) -> Table:
    now = datetime.now()
    cutoff = now + timedelta(days=days)

    upcoming = []
    for data in earnings_data:
        if data and data['earnings_date']:
            earnings_dt = data['earnings_date']
            # Handle timezone-aware datetime
            if earnings_dt.tzinfo is not None:
                earnings_dt = earnings_dt.replace(tzinfo=None)

            if now <= earnings_dt <= cutoff:
                days_until = (earnings_dt - now).days
                upcoming.append({
                    **data,
                    'days_until': days_until,
                    'date_str': earnings_dt.strftime('%Y-%m-%d')
                })

    # Sort by date
    upcoming.sort(key=lambda x: x['days_until'])

    if not upcoming:
        return Panel(
            "[bright_black]No upcoming earnings found in the next 30 days[/bright_black]",
            title="UPCOMING EARNINGS (Next 30 Days)",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )

    table = Table(
        title=f"UPCOMING EARNINGS (Next {days} Days)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Company", style="white", width=22)
    table.add_column("Date", justify="right", width=12)
    table.add_column("Time", justify="center", width=5)
    table.add_column("Days", justify="right", width=5)
    table.add_column("EPS Est.", justify="right", width=9)
    table.add_column("Surprise", justify="right", width=10)
    table.add_column("Last Move", justify="right", width=10)

    for stock in upcoming[:20]:  # Show top 20
        # Days until color coding
        days = stock['days_until']
        if days <= 3:
            days_color = "red"
        elif days <= 7:
            days_color = "yellow"
        else:
            days_color = "white"

        # Surprise color
        surprise_pct = stock.get('last_surprise_pct', 0)
        if surprise_pct > 5:
            surprise_color = "green"
        elif surprise_pct < -5:
            surprise_color = "red"
        else:
            surprise_color = "white"

        # Price move color
        price_move = stock.get('price_move_after_earnings', 0)
        if price_move > 2:
            move_color = "bright_green"
        elif price_move > 0:
            move_color = "green"
        elif price_move < -2:
            move_color = "bright_red"
        elif price_move < 0:
            move_color = "red"
        else:
            move_color = "white"

        eps_est = f"${stock['eps_estimate']:.2f}" if stock['eps_estimate'] else "N/A"
        surprise = f"{surprise_pct:+.1f}%" if surprise_pct else "N/A"
        move_str = f"{price_move:+.1f}%" if price_move != 0 else "—"
        time_str = stock.get('earnings_time', 'TBD')

        table.add_row(
            stock['symbol'],
            stock['name'],
            stock['date_str'],
            time_str,
            f"[{days_color}]{days}d[/{days_color}]",
            eps_est,
            f"[{surprise_color}]{surprise}[/{surprise_color}]",
            f"[{move_color}]{move_str}[/{move_color}]"
        )

    return table


def create_sector_table(earnings_data: List[Dict]) -> Table:
    sectors = {}
    for data in earnings_data:
        if data and data['earnings_date']:
            sector = data['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(data)

    table = Table(
        title="EARNINGS BY SECTOR",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1),
        expand=False
    )

    table.add_column("Sector", style="white", width=25)
    table.add_column("Companies", justify="right", width=10)
    table.add_column("Avg Surprise", justify="right", width=14)

    for sector in sorted(sectors.keys()):
        stocks = sectors[sector]
        count = len(stocks)

        # Calculate average surprise
        surprises = [s['last_surprise_pct'] for s in stocks if s.get('last_surprise_pct')]
        avg_surprise = sum(surprises) / len(surprises) if surprises else 0

        surprise_color = COLORS['up'] if avg_surprise > 0 else COLORS['down'] if avg_surprise < 0 else COLORS['neutral']

        table.add_row(
            sector,
            str(count),
            f"[{surprise_color}]{avg_surprise:+.1f}%[/{surprise_color}]"
        )

    return table


def create_surprise_leaders_table(earnings_data: List[Dict]) -> Table:
    with_surprises = [d for d in earnings_data if d and d.get('last_surprise_pct')]
    with_surprises.sort(key=lambda x: x['last_surprise_pct'], reverse=True)

    table = Table(
        title="RECENT EARNINGS SURPRISE LEADERS",
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
    table.add_column("Actual EPS", justify="right", width=12)
    table.add_column("Surprise", justify="right", width=12)

    # Top 10 positive surprises
    for stock in with_surprises[:10]:
        surprise_pct = stock['last_surprise_pct']
        surprise_color = COLORS['up'] if surprise_pct > 0 else COLORS['down']

        table.add_row(
            stock['symbol'],
            stock['name'],
            f"${stock['last_eps']:.2f}",
            f"[{surprise_color}]{surprise_pct:+.1f}%[/{surprise_color}]"
        )

    return table


def create_eps_trend_table(earnings_data: List[Dict]) -> Optional[Table]:
    with_history = [d for d in earnings_data if d and d['earnings_date'] and d.get('eps_history')]
    with_history.sort(key=lambda x: x['earnings_date'])

    if not with_history:
        return None

    table = Table(
        title="EPS TREND (Last 4 Quarters)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Symbol", style="white", width=8)
    table.add_column("Q-3", justify="right", width=10)
    table.add_column("Q-2", justify="right", width=10)
    table.add_column("Q-1", justify="right", width=10)
    table.add_column("Latest", justify="right", width=10)
    table.add_column("Trend", justify="center", width=8)

    for stock in with_history[:10]:  # Show top 10
        history = stock['eps_history']

        # Pad with empty if less than 4 quarters
        while len(history) < 4:
            history.append({'eps': 0, 'surprise': 0})

        # Get last 4 quarters (reverse order)
        q4 = history[3] if len(history) > 3 else {'eps': 0}
        q3 = history[2] if len(history) > 2 else {'eps': 0}
        q2 = history[1] if len(history) > 1 else {'eps': 0}
        q1 = history[0]

        # Calculate trend
        eps_values = [q['eps'] for q in [q4, q3, q2, q1] if q['eps'] != 0]
        if len(eps_values) >= 2:
            if eps_values[-1] > eps_values[0]:
                trend = "↗"
                trend_color = "green"
            elif eps_values[-1] < eps_values[0]:
                trend = "↘"
                trend_color = "red"
            else:
                trend = "→"
                trend_color = "white"
        else:
            trend = "—"
            trend_color = "white"

        table.add_row(
            stock['symbol'],
            f"${q4['eps']:.2f}" if q4['eps'] != 0 else "—",
            f"${q3['eps']:.2f}" if q3['eps'] != 0 else "—",
            f"${q2['eps']:.2f}" if q2['eps'] != 0 else "—",
            f"${q1['eps']:.2f}" if q1['eps'] != 0 else "—",
            f"[{trend_color}]{trend}[/{trend_color}]"
        )

    return table


def main() -> None:
    console.clear()
    console.print(create_header())
    console.print()

    # Fetch earnings data
    console.print("[white]Loading earnings data for top 50 stocks...[/white]")

    earnings_data = []
    for symbol in TRACKING_STOCKS:
        data = get_earnings_data(symbol)
        if data:
            earnings_data.append(data)

    console.clear()
    console.print(create_header())
    console.print()

    # Upcoming earnings
    console.print(create_upcoming_table(earnings_data, days=30))
    console.print()

    # Side by side: Sector breakdown + Surprise leaders
    console.print(Columns([
        create_sector_table(earnings_data),
        create_surprise_leaders_table(earnings_data)
    ]))
    console.print()

    # EPS Trend table (if data available)
    eps_trend = create_eps_trend_table(earnings_data)
    if eps_trend:
        console.print(eps_trend)
        console.print()

    # Footer
    footer = Panel(
        f"[bright_black]Tracking {len(TRACKING_STOCKS)} stocks │ BMO=Before Market Open, AMC=After Market Close │ Updated {datetime.now().strftime('%I:%M %p')}[/bright_black]",
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
