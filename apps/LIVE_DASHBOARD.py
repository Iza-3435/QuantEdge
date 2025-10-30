"""
LIVE BLOOMBERG-STYLE DASHBOARD
Real-time auto-refreshing multi-pane terminal

Features:
- Auto-refresh every 5 seconds
- Multi-pane layout (Market | Portfolio | Alerts)
- Live price updates
- Keyboard navigation
- Real-time data streaming
"""

import yfinance as yf
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
import threading

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


def create_sparkline(prices, width=7):
    """Create sparkline from price data"""
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

    color = COLORS['up'] if prices[-1] > prices[0] else COLORS['down']
    return f"[{color}]{sparkline}[/{color}]"


# Global data store
LIVE_DATA = {
    'market': {},
    'portfolio': {},
    'options': {},  # NEW: Options data
    'alerts': [],
    'last_update': None,
    'refresh_count': 0
}

# Database path
DB_PATH = 'investment_platform.db'


def get_color(change_pct):
    """Get color based on change intensity"""
    if change_pct > 0:
        return COLORS['up']
    elif change_pct < 0:
        return COLORS['down']
    else:
        return COLORS['neutral']


def fetch_market_data():
    """Fetch real-time market data"""
    try:
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'Nasdaq',
            '^DJI': 'Dow Jones',
            '^VIX': 'VIX'
        }

        market_data = {}

        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)

                # Get historical data for sparkline
                hist = ticker.history(period='5d')
                prices = hist['Close'].tolist() if not hist.empty else []

                # Try fast_info first (fastest)
                try:
                    fast = ticker.fast_info
                    current_price = fast.get('lastPrice')
                    prev_close = fast.get('previousClose')

                    if current_price and prev_close:
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100

                        market_data[name] = {
                            'price': current_price,
                            'change': change,
                            'change_pct': change_pct,
                            'prices': prices,
                            'source': 'realtime'
                        }
                except:
                    # Fallback to info
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    prev_close = info.get('previousClose')

                    if current_price and prev_close:
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100

                        market_data[name] = {
                            'price': current_price,
                            'change': change,
                            'change_pct': change_pct,
                            'prices': prices,
                            'source': 'info'
                        }
            except:
                continue

        return market_data

    except Exception as e:
        return {}


def fetch_portfolio_data():
    """Fetch portfolio summary"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all portfolios
        cursor.execute("SELECT id, name FROM portfolios")
        portfolios = cursor.fetchall()

        if not portfolios:
            conn.close()
            return None

        total_value = 0
        total_cost = 0
        holdings_count = 0
        top_holdings = []

        for portfolio_id, portfolio_name in portfolios:
            # Get holdings
            cursor.execute("""
                SELECT symbol, quantity, avg_cost
                FROM holdings
                WHERE portfolio_id = ? AND quantity > 0
                LIMIT 5
            """, (portfolio_id,))

            holdings = cursor.fetchall()
            holdings_count += len(holdings)

            # Calculate portfolio value
            for symbol, quantity, avg_cost in holdings:
                try:
                    ticker = yf.Ticker(symbol)
                    fast = ticker.fast_info
                    current_price = fast.get('lastPrice')

                    if current_price:
                        value = current_price * quantity
                        cost = avg_cost * quantity
                        pnl = value - cost
                        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0

                        total_value += value
                        total_cost += cost

                        top_holdings.append({
                            'symbol': symbol,
                            'value': value,
                            'pnl_pct': pnl_pct
                        })
                except:
                    continue

        conn.close()

        if total_cost > 0:
            total_return = total_value - total_cost
            total_return_pct = (total_return / total_cost) * 100

            # Sort and get top 5
            top_holdings.sort(key=lambda x: x['value'], reverse=True)
            top_holdings = top_holdings[:5]

            return {
                'num_portfolios': len(portfolios),
                'num_holdings': holdings_count,
                'total_value': total_value,
                'total_cost': total_cost,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'top_holdings': top_holdings
            }

        return None

    except Exception as e:
        return None


def fetch_alerts():
    """Fetch active alerts"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get watchlist items with targets
        cursor.execute("""
            SELECT DISTINCT symbol, target_price, stop_loss
            FROM watchlist_items
            WHERE alert_enabled = 1
            LIMIT 10
        """)

        items = cursor.fetchall()
        conn.close()

        alerts = []

        for symbol, target, stop in items:
            try:
                ticker = yf.Ticker(symbol)
                fast = ticker.fast_info
                price = fast.get('lastPrice')

                if not price:
                    continue

                # Check target
                if target and price >= target:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'TARGET',
                        'message': f'{symbol}: Target ${target:.2f} reached!',
                        'severity': 'success',
                        'price': price
                    })

                # Check stop loss
                if stop and price <= stop:
                    alerts.append({
                        'symbol': symbol,
                        'type': 'STOP',
                        'message': f'{symbol}: Stop ${stop:.2f} triggered!',
                        'severity': 'critical',
                        'price': price
                    })

            except:
                continue

        return alerts

    except Exception as e:
        return []


def fetch_options_data(symbols=None):
    """Fetch real-time options data for top portfolio stocks"""
    try:
        if not symbols:
            # Get symbols from portfolio
            portfolio_data = LIVE_DATA.get('portfolio', {})
            if portfolio_data and 'top_holdings' in portfolio_data:
                symbols = [h['symbol'] for h in portfolio_data['top_holdings'][:3]]  # Top 3 stocks
            else:
                symbols = ['SPY']  # Default to SPY if no portfolio

        options_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get available expiration dates
                exp_dates = ticker.options
                if not exp_dates or len(exp_dates) == 0:
                    continue

                # Get nearest expiration (usually weekly or monthly)
                nearest_exp = exp_dates[0]

                # Fetch options chain
                opt_chain = ticker.option_chain(nearest_exp)
                calls = opt_chain.calls
                puts = opt_chain.puts

                if calls.empty and puts.empty:
                    continue

                # Get ATM options (at-the-money)
                current_price = ticker.fast_info.get('lastPrice', 0)
                if not current_price:
                    continue

                # Find strikes closest to current price
                calls['distance'] = abs(calls['strike'] - current_price)
                puts['distance'] = abs(puts['strike'] - current_price)

                # Get 3 strikes around ATM
                atm_calls = calls.nsmallest(3, 'distance')[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
                atm_puts = puts.nsmallest(3, 'distance')[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]

                # Calculate Put/Call ratio
                total_call_volume = calls['volume'].sum()
                total_put_volume = puts['volume'].sum()
                pc_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0

                options_data[symbol] = {
                    'current_price': current_price,
                    'expiration': nearest_exp,
                    'calls': atm_calls.to_dict('records'),
                    'puts': atm_puts.to_dict('records'),
                    'pc_ratio': pc_ratio,
                    'call_volume': int(total_call_volume),
                    'put_volume': int(total_put_volume)
                }

            except Exception as e:
                continue

        return options_data

    except Exception as e:
        return {}


def create_market_panel(market_data):
    """Create market overview panel"""
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        padding=(0, 1)
    )

    table.add_column("Index", style="white", width=10)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Change", justify="right", width=12)
    table.add_column("Trend", justify="center", width=10)

    for name, data in market_data.items():
        price = data['price']
        change_pct = data['change_pct']
        color = get_color(change_pct)
        arrow = '▲' if change_pct > 0 else '▼' if change_pct < 0 else '━'

        sparkline = create_sparkline(data.get('prices', []))

        table.add_row(
            name,
            f"${price:,.2f}",
            f"[{color}]{arrow} {change_pct:+.2f}%[/{color}]",
            sparkline
        )

    # Add live indicator
    source = list(market_data.values())[0].get('source') if market_data else 'none'
    live_indicator = " LIVE" if source == 'realtime' else " DELAYED"

    title = f"[bold white on cyan] MARKET {live_indicator} [/bold white on cyan]"

    return Panel(
        table,
        title=title,
        border_style=THEME['border'],
        padding=(1, 1), style=THEME['panel_bg']
    )


def create_portfolio_panel(portfolio_data):
    """Create portfolio summary panel"""
    if not portfolio_data:
        content = Text("No portfolio data\n\nAdd holdings in Portfolio Manager", style="bright_black")
    else:
        value = portfolio_data['total_value']
        ret = portfolio_data['total_return']
        ret_pct = portfolio_data['total_return_pct']
        color = get_color(ret_pct)
        arrow = '▲' if ret_pct >= 0 else '▼'

        content = Text()
        content.append("Total Value: ", style="bold white")
        content.append(f"${value:,.2f}\n", style="bright_white")

        content.append("P&L: ", style="bold white")
        content.append(f"{arrow} ${abs(ret):,.2f} ({ret_pct:+.2f}%)\n\n", style=color)

        content.append(f"Holdings: {portfolio_data['num_holdings']}\n", style="bright_black")

        if portfolio_data['top_holdings']:
            content.append("\nTop Holdings:\n", style="bold white")
            for holding in portfolio_data['top_holdings'][:3]:
                h_color = get_color(holding['pnl_pct'])
                content.append(f"{holding['symbol']} ", style="white")
                content.append(f"${holding['value']:,.0f} ", style="bright_black")
                content.append(f"({holding['pnl_pct']:+.1f}%)\n", style=h_color)

    return Panel(
        content,
        title="[bold white on green] PORTFOLIO 💼 [/bold white on green]",
        border_style=THEME['border'],
        padding=(1, 1), style=THEME['panel_bg']
    )


def create_alerts_panel(alerts):
    """Create alerts panel"""
    if not alerts:
        content = Text("✓ No active alerts\n\nAll clear!", style="green")
    else:
        content = Text()
        content.append(f"{len(alerts)} Active Alert(s)\n\n", style="bold yellow")

        for alert in alerts[:5]:
            severity_colors = {
                'critical': 'bright_red',
                'warning': 'yellow',
                'success': 'bright_green',
                'info': 'cyan'
            }

            color = severity_colors.get(alert['severity'], 'white')
            icon = '🚨' if alert['severity'] == 'critical' else '✓' if alert['severity'] == 'success' else ''

            content.append(f"{icon} ", style=color)
            content.append(f"{alert['message']}\n", style=color)

        if len(alerts) > 5:
            content.append(f"\n+ {len(alerts) - 5} more...", style="bright_black")

    return Panel(
        content,
        title="[bold white on red] ALERTS 🔔 [/bold white on red]",
        border_style=THEME['border'],
        padding=(1, 1), style=THEME['panel_bg']
    )


def create_options_panel(options_data):
    """Create options chain panel"""
    if not options_data:
        content = Text("No options data available\n\nAdd stocks to portfolio to see options", style="bright_black")
        return Panel(
            content,
            title="[bold white] OPTIONS CHAIN 📊 [/bold white]",
            border_style=THEME['border'],
            padding=(1, 1),
            style=THEME['panel_bg']
        )

    content = []

    for symbol, data in list(options_data.items())[:2]:  # Show top 2 stocks
        # Header for stock
        header_text = Text()
        header_text.append(f"\n{symbol}", style="bold white")
        header_text.append(f"  ${data['current_price']:.2f}", style="cyan")
        header_text.append(f"  Exp: {data['expiration']}", style="bright_black")
        content.append(header_text)

        # P/C Ratio indicator
        pc_ratio = data['pc_ratio']
        pc_color = 'red' if pc_ratio > 1.0 else 'green' if pc_ratio < 0.7 else 'yellow'
        pc_sentiment = 'BEARISH' if pc_ratio > 1.0 else 'BULLISH' if pc_ratio < 0.7 else 'NEUTRAL'

        pc_text = Text()
        pc_text.append(f"  P/C Ratio: ", style="bright_black")
        pc_text.append(f"{pc_ratio:.2f}", style=pc_color)
        pc_text.append(f" ({pc_sentiment})", style=pc_color)
        content.append(pc_text)

        # Create options table
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold white",
            border_style=THEME['border'],
            padding=(0, 1),
            show_edge=False
        )

        table.add_column("CALLS", style="green", width=12)
        table.add_column("Strike", style="white", justify="right", width=8)
        table.add_column("PUTS", style="red", width=12)

        # Get 3 strikes
        calls = data.get('calls', [])
        puts = data.get('puts', [])

        for i in range(min(3, len(calls), len(puts))):
            call = calls[i]
            put = puts[i]

            call_text = f"${call['lastPrice']:.2f}"
            if call.get('volume', 0) > 0:
                call_text += f" ({int(call['volume'])})"

            put_text = f"${put['lastPrice']:.2f}"
            if put.get('volume', 0) > 0:
                put_text += f" ({int(put['volume'])})"

            table.add_row(
                call_text,
                f"${call['strike']:.0f}",
                put_text
            )

        content.append(table)
        content.append(Text())  # Spacer

    # Combine all content
    from rich.console import Group
    final_content = Group(*content)

    return Panel(
        final_content,
        title="[bold white] OPTIONS CHAIN 📊 [/bold white]",
        border_style=THEME['border'],
        padding=(0, 1),
        style=THEME['panel_bg']
    )


def create_header():
    """Create dashboard header"""
    now = datetime.now()
    refresh_count = LIVE_DATA.get('refresh_count', 0)

    header = Text()
    header.append("═" * 80, style="bright_blue")
    header.append("\n")
    header.append("              ", style="bright_blue")
    header.append("  LIVE BLOOMBERG-STYLE DASHBOARD  ", style="bold white on blue")
    header.append("\n", style="bright_blue")
    header.append("═" * 80, style="bright_blue")
    header.append("\n\n")
    header.append("  Real-Time Multi-Pane Terminal", style="bold bright_blue")
    header.append(" • ", style="bright_black")
    header.append(now.strftime('%I:%M:%S %p'), style="blue")
    header.append(" • ", style="bright_black")
    header.append(f"Updates: {refresh_count}", style="bright_blue")
    header.append(" • ", style="bright_black")
    header.append("Press Ctrl+C to exit", style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(0, 2), style=THEME['panel_bg'])


def create_footer():
    """Create footer with controls"""
    footer = Text()
    footer.append("Auto-refresh: ", style="bright_black")
    footer.append("ON", style="bright_green")
    footer.append(" • ", style="bright_black")
    footer.append("Interval: 5s", style="bright_black")
    footer.append(" • ", style="bright_black")
    footer.append("Data: Real-time", style="bright_green")

    return Panel(footer, box=box.SQUARE, border_style=THEME['border'], style=THEME['panel_bg'])


def create_dashboard_layout():
    """Create the multi-pane layout"""
    layout = Layout()

    # Main structure
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )

    # Split body into 2 rows
    layout["body"].split_column(
        Layout(name="top_row"),
        Layout(name="options")  # Direct layout for options
    )

    # Top row: Market, Portfolio, Alerts
    layout["top_row"].split_row(
        Layout(name="market"),
        Layout(name="portfolio"),
        Layout(name="alerts")
    )

    return layout


def update_dashboard_data():
    """Background worker to fetch fresh data"""
    while True:
        try:
            # Fetch all data
            market_data = fetch_market_data()
            portfolio_data = fetch_portfolio_data()
            alerts = fetch_alerts()
            options_data = fetch_options_data()  # NEW: Fetch options

            # Update global store
            LIVE_DATA['market'] = market_data
            LIVE_DATA['portfolio'] = portfolio_data
            LIVE_DATA['alerts'] = alerts
            LIVE_DATA['options'] = options_data  # NEW: Store options
            LIVE_DATA['last_update'] = datetime.now()
            LIVE_DATA['refresh_count'] += 1

        except Exception as e:
            pass

        # Sleep for 5 seconds
        time.sleep(5)


def render_dashboard():
    """Render the complete dashboard"""
    layout = create_dashboard_layout()

    # Update each section
    layout["header"].update(create_header())
    layout["market"].update(create_market_panel(LIVE_DATA.get('market', {})))
    layout["portfolio"].update(create_portfolio_panel(LIVE_DATA.get('portfolio')))
    layout["alerts"].update(create_alerts_panel(LIVE_DATA.get('alerts', [])))
    layout["options"].update(create_options_panel(LIVE_DATA.get('options', {})))  # NEW: Options panel
    layout["footer"].update(create_footer())

    return layout


def main():
    """Main live dashboard"""
    console.clear()

    # Show loading message
    console.print("\n[bold cyan]Launching Live Dashboard...[/bold cyan]\n")

    console.print("[bright_black]→ Fetching market indices...[/bright_black]")
    LIVE_DATA['market'] = fetch_market_data()
    console.print("[green]✓ Market data loaded[/green]")

    console.print("[bright_black]→ Loading portfolio data...[/bright_black]")
    LIVE_DATA['portfolio'] = fetch_portfolio_data()
    console.print("[green]✓ Portfolio loaded[/green]")

    console.print("[bright_black]→ Checking alerts...[/bright_black]")
    LIVE_DATA['alerts'] = fetch_alerts()
    console.print("[green]✓ Alerts checked[/green]")

    console.print("[bright_black]→ Fetching options data...[/bright_black]")
    LIVE_DATA['options'] = fetch_options_data()
    console.print("[green]✓ Options loaded[/green]")

    LIVE_DATA['last_update'] = datetime.now()
    LIVE_DATA['refresh_count'] = 1

    time.sleep(0.5)

    # Start background update thread
    console.print("\n[bright_black]→ Starting live updates...[/bright_black]")
    update_thread = threading.Thread(target=update_dashboard_data, daemon=True)
    update_thread.start()

    console.print("[green]✓ Live updates active![/green]")
    console.print("\n[bright_blue]Starting dashboard in 2 seconds...[/bright_blue]")
    time.sleep(2)

    # Start live display with auto-refresh
    try:
        with Live(render_dashboard(), refresh_per_second=1, console=console, screen=True) as live:
            while True:
                # Update display every second
                live.update(render_dashboard())
                time.sleep(1)

    except KeyboardInterrupt:
        console.clear()
        console.print("\n[cyan]Dashboard stopped.[/cyan]")
        console.print(f"[bright_black]Total updates: {LIVE_DATA['refresh_count']}[/bright_black]")
        console.print(f"[bright_black]Last update: {LIVE_DATA['last_update'].strftime('%I:%M:%S %p') if LIVE_DATA['last_update'] else 'N/A'}[/bright_black]\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)
