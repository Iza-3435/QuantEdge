"""
PROFESSIONAL WATCHLIST SYSTEM
Advanced stock monitoring with alerts and daily summaries
"""

import yfinance as yf
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import box
from rich.columns import Columns
import sys
import os

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


# Database path
DB_PATH = 'investment_platform.db'

# Color scheme
COLORS = {
    'up': 'green',
    'down': 'red',
    'neutral': 'yellow',
    'alert': 'red bold',
    'warning': 'yellow bold',
    'info': 'cyan',
    'success': 'green bold'
}


def init_database():
    """Initialize watchlist database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Watchlists table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_date TEXT
        )
    ''')

    # Watchlist items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watchlist_id INTEGER,
            symbol TEXT NOT NULL,
            added_date TEXT,
            target_price REAL,
            stop_loss REAL,
            notes TEXT,
            alert_enabled INTEGER DEFAULT 1,
            FOREIGN KEY (watchlist_id) REFERENCES watchlists (id),
            UNIQUE(watchlist_id, symbol)
        )
    ''')

    # Price alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            threshold REAL NOT NULL,
            triggered INTEGER DEFAULT 0,
            created_date TEXT,
            triggered_date TEXT
        )
    ''')

    # Create default watchlists if they don't exist
    cursor.execute("SELECT COUNT(*) FROM watchlists")
    if cursor.fetchone()[0] == 0:
        default_watchlists = [
            ('Growth', 'High-growth stocks'),
            ('Value', 'Value investment opportunities'),
            ('Dividends', 'Dividend aristocrats'),
            ('Tech', 'Technology sector'),
            ('Main', 'Primary watchlist')
        ]
        cursor.executemany(
            "INSERT INTO watchlists (name, description, created_date) VALUES (?, ?, ?)",
            [(name, desc, datetime.now().isoformat()) for name, desc in default_watchlists]
        )

    conn.commit()
    conn.close()


def get_watchlists():
    """Get all watchlists"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, description, created_date FROM watchlists ORDER BY name")
    watchlists = cursor.fetchall()

    conn.close()
    return watchlists


def get_watchlist_items(watchlist_id):
    """Get all items in a watchlist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, symbol, added_date, target_price, stop_loss, notes, alert_enabled
        FROM watchlist_items
        WHERE watchlist_id = ?
        ORDER BY added_date DESC
    """, (watchlist_id,))

    items = cursor.fetchall()
    conn.close()
    return items


def add_to_watchlist(watchlist_id, symbol, target_price=None, stop_loss=None, notes=None):
    """Add stock to watchlist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO watchlist_items (watchlist_id, symbol, added_date, target_price, stop_loss, notes, alert_enabled)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (watchlist_id, symbol.upper(), datetime.now().isoformat(), target_price, stop_loss, notes))

        conn.commit()
        return True
    except sqlite3.IntegrityError:
        console.print(f"[yellow]Stock {symbol} already in watchlist[/yellow]")
        return False
    finally:
        conn.close()


def remove_from_watchlist(watchlist_id, symbol):
    """Remove stock from watchlist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM watchlist_items
        WHERE watchlist_id = ? AND symbol = ?
    """, (watchlist_id, symbol.upper()))

    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def create_sparkline(prices, width=7):
    """Create ASCII sparkline from price list"""
    if not prices or len(prices) < 2:
        return "━━━━━━━"

    # Remove any NaN values
    prices = [p for p in prices if not pd.isna(p)]
    if len(prices) < 2:
        return "━━━━━━━"

    min_p, max_p = min(prices), max(prices)
    range_p = max_p - min_p if max_p != min_p else 1

    chars = '▁▂▃▄▅▆▇█'
    sparkline = ''
    for price in prices[-width:]:  # Last 'width' data points
        index = int(((price - min_p) / range_p) * 7)
        sparkline += chars[min(index, 7)]

    # Color code: green if trending up, red if down
    color = 'green' if prices[-1] > prices[0] else 'red' if prices[-1] < prices[0] else 'white'
    return f"[{color}]{sparkline}[/{color}]"


def get_stock_data(symbol):
    """Fetch current stock data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')
        info = ticker.info

        if len(hist) < 2:
            return None

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        # Get sparkline prices (last 5 days)
        sparkline_prices = hist['Close'].tolist()

        # Calculate additional metrics
        hist_30d = ticker.history(period='30d')
        returns = hist_30d['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100

        # RSI calculation
        delta = hist_30d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50

        # Moving averages
        ma_50 = info.get('fiftyDayAverage', current_price)
        ma_200 = info.get('twoHundredDayAverage', current_price)

        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': current_price,
            'change': change,
            'change_pct': change_pct,
            'volume': hist['Volume'].iloc[-1],
            'volatility': volatility,
            'rsi': current_rsi,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'sparkline_prices': sparkline_prices
        }

    except Exception as e:
        console.print(f"[bright_black]Warning: Could not fetch {symbol}: {str(e)}[/bright_black]")
        return None


def check_alerts(symbol, current_price, target_price, stop_loss):
    """Check if price alerts should be triggered"""
    alerts = []

    if target_price and current_price >= target_price:
        alerts.append({
            'type': 'TARGET',
            'message': f"Target price ${target_price:.2f} reached!",
            'color': COLORS['success']
        })

    if stop_loss and current_price <= stop_loss:
        alerts.append({
            'type': 'STOP_LOSS',
            'message': f"Stop loss ${stop_loss:.2f} triggered!",
            'color': COLORS['alert']
        })

    return alerts


def check_technical_alerts(data):
    """Check technical indicator alerts"""
    alerts = []

    # RSI alerts
    if data['rsi'] > 70:
        alerts.append({
            'type': 'RSI_OVERBOUGHT',
            'message': f"RSI {data['rsi']:.1f} - Overbought",
            'color': COLORS['warning']
        })
    elif data['rsi'] < 30:
        alerts.append({
            'type': 'RSI_OVERSOLD',
            'message': f"RSI {data['rsi']:.1f} - Oversold",
            'color': COLORS['info']
        })

    # Moving average crossovers
    if data['price'] > data['ma_50'] > data['ma_200']:
        alerts.append({
            'type': 'GOLDEN_CROSS',
            'message': "Golden Cross (Bullish)",
            'color': COLORS['success']
        })
    elif data['price'] < data['ma_50'] < data['ma_200']:
        alerts.append({
            'type': 'DEATH_CROSS',
            'message': "Death Cross (Bearish)",
            'color': COLORS['alert']
        })

    return alerts


def create_watchlist_table(watchlist_name, items_data):
    """Create watchlist display table"""
    table = Table(
        title=f"Watchlist: {watchlist_name}",
        box=box.SQUARE,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}"
    )

    table.add_column("Symbol", style="white bold", width=8)
    table.add_column("Name", style="white", width=18)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Change", justify="right", width=10)
    table.add_column("Trend (5D)", justify="center", width=9)
    table.add_column("RSI", justify="right", width=6)
    table.add_column("Target", justify="right", width=8)
    table.add_column("Stop", justify="right", width=8)
    table.add_column("Alerts", width=20)

    for item_data, stock_data, target, stop in items_data:
        if not stock_data:
            continue

        symbol = stock_data['symbol']
        name = stock_data['name'][:16] + '..' if len(stock_data['name']) > 18 else stock_data['name']
        price = stock_data['price']
        change_pct = stock_data['change_pct']
        rsi = stock_data['rsi']

        # Color coding
        color = COLORS['up'] if change_pct > 0 else COLORS['down']
        arrow = '▲' if change_pct > 0 else '▼'

        # RSI color
        rsi_color = 'red' if rsi > 70 or rsi < 30 else 'yellow' if rsi > 60 or rsi < 40 else 'green'

        # Create sparkline
        sparkline = create_sparkline(stock_data.get('sparkline_prices', []))

        # Check alerts
        price_alerts = check_alerts(symbol, price, target, stop)
        tech_alerts = check_technical_alerts(stock_data)
        all_alerts = price_alerts + tech_alerts

        # Format alerts
        alerts_str = ""
        if all_alerts:
            alert_messages = [alert['message'][:18] for alert in all_alerts[:1]]  # Show max 1, truncated
            alerts_str = " | ".join(alert_messages)
            alert_color = all_alerts[0]['color']
            alerts_display = f"[{alert_color}]{alerts_str}[/{alert_color}]"
        else:
            alerts_display = "[bright_black]None[/bright_black]"

        # Format target and stop
        target_str = f"${target:.2f}" if target else "[bright_black]--[/bright_black]"
        stop_str = f"${stop:.2f}" if stop else "[bright_black]--[/bright_black]"

        table.add_row(
            symbol,
            name,
            f"${price:.2f}",
            f"[{color}]{arrow} {abs(change_pct):.2f}%[/{color}]",
            sparkline,
            f"[{rsi_color}]{rsi:.0f}[/{rsi_color}]",
            target_str,
            stop_str,
            alerts_display
        )

    return table


def view_watchlist():
    """View watchlist with live data"""
    watchlists = get_watchlists()

    if not watchlists:
        console.print("[yellow]No watchlists found[/yellow]")
        return

    # Show available watchlists
    console.print("\n[bold cyan]Available Watchlists:[/bold cyan]")
    for i, (wl_id, name, desc, created) in enumerate(watchlists, 1):
        console.print(f"  {i}. {name} - {desc}")

    choice = Prompt.ask("\n[cyan]Choose watchlist[/cyan]", default="1")

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(watchlists):
            console.print("[red]Invalid choice[/red]")
            return

        wl_id, wl_name, wl_desc, _ = watchlists[idx]
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return

    # Get watchlist items
    items = get_watchlist_items(wl_id)

    if not items:
        console.print(f"[yellow]Watchlist '{wl_name}' is empty[/yellow]")
        return

    # Fetch live data
    console.print(f"\n[cyan]Fetching data for {len(items)} stocks...[/cyan]")

    items_data = []
    for item_id, symbol, added, target, stop, notes, alert_enabled in items:
        stock_data = get_stock_data(symbol)
        items_data.append((item_id, stock_data, target, stop))

    # Display table
    console.print()
    console.print(create_watchlist_table(wl_name, items_data))
    console.print()


def add_stock_menu():
    """Interactive menu to add stock to watchlist"""
    watchlists = get_watchlists()

    if not watchlists:
        console.print("[yellow]No watchlists found[/yellow]")
        return

    # Show available watchlists
    console.print("\n[bold cyan]Available Watchlists:[/bold cyan]")
    for i, (wl_id, name, desc, created) in enumerate(watchlists, 1):
        console.print(f"  {i}. {name}")

    choice = Prompt.ask("\n[cyan]Choose watchlist[/cyan]", default="1")

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(watchlists):
            console.print("[red]Invalid choice[/red]")
            return

        wl_id, wl_name, _, _ = watchlists[idx]
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return

    # Get stock details
    symbol = Prompt.ask("\n[cyan]Stock symbol[/cyan]").upper()

    # Verify stock exists
    console.print(f"[cyan]Verifying {symbol}...[/cyan]")
    stock_data = get_stock_data(symbol)

    if not stock_data:
        console.print(f"[red]Could not find stock {symbol}[/red]")
        return

    console.print(f"[green]Found: {stock_data['name']} - ${stock_data['price']:.2f}[/green]")

    # Optional: target price and stop loss
    if Confirm.ask("\n[cyan]Set target price?[/cyan]", default=False):
        target_str = Prompt.ask("[cyan]Target price[/cyan]")
        try:
            target = float(target_str)
        except ValueError:
            target = None
    else:
        target = None

    if Confirm.ask("[cyan]Set stop loss?[/cyan]", default=False):
        stop_str = Prompt.ask("[cyan]Stop loss price[/cyan]")
        try:
            stop = float(stop_str)
        except ValueError:
            stop = None
    else:
        stop = None

    notes = Prompt.ask("[cyan]Notes (optional)[/cyan]", default="")

    # Add to watchlist
    if add_to_watchlist(wl_id, symbol, target, stop, notes):
        console.print(f"\n[green]Added {symbol} to '{wl_name}' watchlist[/green]")
    else:
        console.print(f"\n[yellow]Failed to add {symbol}[/yellow]")


def remove_stock_menu():
    """Interactive menu to remove stock from watchlist"""
    watchlists = get_watchlists()

    if not watchlists:
        console.print("[yellow]No watchlists found[/yellow]")
        return

    # Show available watchlists
    console.print("\n[bold cyan]Available Watchlists:[/bold cyan]")
    for i, (wl_id, name, desc, created) in enumerate(watchlists, 1):
        console.print(f"  {i}. {name}")

    choice = Prompt.ask("\n[cyan]Choose watchlist[/cyan]", default="1")

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(watchlists):
            console.print("[red]Invalid choice[/red]")
            return

        wl_id, wl_name, _, _ = watchlists[idx]
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return

    # Get watchlist items
    items = get_watchlist_items(wl_id)

    if not items:
        console.print(f"[yellow]Watchlist '{wl_name}' is empty[/yellow]")
        return

    # Show items
    console.print(f"\n[bold cyan]Stocks in '{wl_name}':[/bold cyan]")
    for i, (item_id, symbol, added, target, stop, notes, alert_enabled) in enumerate(items, 1):
        console.print(f"  {i}. {symbol}")

    choice = Prompt.ask("\n[cyan]Choose stock to remove[/cyan]")

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(items):
            console.print("[red]Invalid choice[/red]")
            return

        symbol = items[idx][1]
    except ValueError:
        console.print("[red]Invalid input[/red]")
        return

    # Confirm removal
    if Confirm.ask(f"\n[yellow]Remove {symbol} from '{wl_name}'?[/yellow]", default=False):
        if remove_from_watchlist(wl_id, symbol):
            console.print(f"[green]Removed {symbol}[/green]")
        else:
            console.print(f"[red]Failed to remove {symbol}[/red]")


def show_main_menu():
    """Show main menu"""
    table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2), row_styles=[THEME['row_even'], THEME['row_odd']], border_style=THEME['border'])
    table.add_column("Option", style="white bold", width=3)
    table.add_column("Action", style="white")

    table.add_row("1", "View Watchlist (Live Data)")
    table.add_row("2", "Add Stock to Watchlist")
    table.add_row("3", "Remove Stock from Watchlist")
    table.add_row("4", "Daily Summary (All Watchlists)")
    table.add_row("5", "Exit")

    console.print(Panel(table, title="[bold white]WATCHLIST MANAGER[/bold white]", border_style=THEME['border'], style=THEME['panel_bg']))


def daily_summary():
    """Show daily summary of all watchlists"""
    watchlists = get_watchlists()

    console.print("\n[bold white]DAILY WATCHLIST SUMMARY[/bold white]\n")

    total_alerts = 0

    for wl_id, wl_name, wl_desc, _ in watchlists:
        items = get_watchlist_items(wl_id)

        if not items:
            continue

        console.print(f"[bold]{wl_name}[/bold] ({len(items)} stocks)")

        # Fetch data
        for item_id, symbol, added, target, stop, notes, alert_enabled in items:
            stock_data = get_stock_data(symbol)

            if not stock_data:
                continue

            # Check alerts
            price_alerts = check_alerts(symbol, stock_data['price'], target, stop)
            tech_alerts = check_technical_alerts(stock_data)
            all_alerts = price_alerts + tech_alerts

            if all_alerts:
                total_alerts += len(all_alerts)
                color = COLORS['up'] if stock_data['change_pct'] > 0 else COLORS['down']
                console.print(f"  [{color}]{symbol}[/{color}] ${stock_data['price']:.2f} ({stock_data['change_pct']:+.2f}%)")

                for alert in all_alerts:
                    console.print(f"    [{alert['color']}]Alert: {alert['message']}[/{alert['color']}]")

        console.print()

    if total_alerts == 0:
        console.print("[green]No alerts today[/green]\n")
    else:
        console.print(f"[yellow]Total Alerts: {total_alerts}[/yellow]\n")


def main():
    """Main interactive loop"""
    init_database()

    while True:
        console.clear()

        # Header
        from datetime import datetime
        now = datetime.now()

        header = Text()
        header.append("PROFESSIONAL WATCHLIST SYSTEM\n\n", style="bold white")
        header.append("Advanced Stock Monitoring & Smart Alerts", style="white")
        header.append(" │ ", style="bright_black")
        header.append(now.strftime('%B %d, %Y'), style="bright_black")

        console.print(Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
        console.print()
        show_main_menu()

        choice = Prompt.ask("\n[cyan]Choose option[/cyan]", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "1":
            view_watchlist()
            console.print("\n[bright_black]Press Enter to continue...[/bright_black]")
            input()
        elif choice == "2":
            add_stock_menu()
            console.print("\n[bright_black]Press Enter to continue...[/bright_black]")
            input()
        elif choice == "3":
            remove_stock_menu()
            console.print("\n[bright_black]Press Enter to continue...[/bright_black]")
            input()
        elif choice == "4":
            daily_summary()
            console.print("[bright_black]Press Enter to continue...[/bright_black]")
            input()
        elif choice == "5":
            console.print("\n[white]Thanks for using Watchlist Manager[/white]\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Exiting...[/yellow]\n")
        sys.exit(0)
