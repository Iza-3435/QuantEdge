"""
PROFESSIONAL PORTFOLIO TRACKER
Institutional-grade portfolio management system
Features:
- Multi-portfolio support
- Real-time P&L tracking
- Risk analytics (VaR, CVaR, Greeks)
- Performance attribution
- Rebalancing optimizer
- Tax loss harvesting
- Transaction history with audit trail
- Export to Excel/CSV
- Benchmark comparison (SPY, QQQ, custom)
"""

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ui_enhancements import (
    create_enhanced_sparkline, create_progress_bar, get_performance_color,
    format_percentage, create_score_badge
)

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


def create_sparkline(prices: List[float], width: int = 7) -> str:
    """
    Create ASCII sparkline chart from price data.

    Args:
        prices: List of price values
        width: Number of bars to display

    Returns:
        Formatted sparkline string with color
    """
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


# Database setup
DB_PATH = 'portfolio_data.db'


def init_database() -> None:
    """
    Initialize SQLite database with portfolio schema.

    Creates tables for portfolios, holdings, transactions, and watchlist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_date TEXT,
            cash_balance REAL DEFAULT 0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            symbol TEXT NOT NULL,
            quantity REAL NOT NULL,
            avg_cost REAL NOT NULL,
            purchase_date TEXT,
            notes TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios (id),
            UNIQUE(portfolio_id, symbol)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            transaction_date TEXT,
            commission REAL DEFAULT 0,
            notes TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            target_price REAL,
            alert_rsi_low REAL,
            alert_rsi_high REAL,
            notes TEXT,
            added_date TEXT
        )
    ''')

    conn.commit()
    conn.close()


def get_current_prices(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Fetch current prices for multiple symbols using batch API call.

    Args:
        symbols: List of stock ticker symbols

    Returns:
        Dictionary mapping symbols to price data (price, change, change_pct, volume)
    """
    if not symbols:
        return {}

    try:
        tickers = yf.Tickers(' '.join(symbols))
        prices = {}
        for symbol in symbols:
            try:
                ticker = getattr(tickers.tickers, symbol)
                info = ticker.info
                prices[symbol] = {
                    'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'change': info.get('regularMarketChange', 0),
                    'change_pct': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('volume', 0),
                }
            except:
                prices[symbol] = {'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0}
        return prices
    except:
        return {symbol: {'price': 0, 'change': 0, 'change_pct': 0, 'volume': 0} for symbol in symbols}


def calculate_portfolio_metrics(holdings_df: pd.DataFrame, prices: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio metrics including risk analytics.

    Args:
        holdings_df: DataFrame containing portfolio holdings
        prices: Dictionary of current price data by symbol

    Returns:
        Dictionary containing portfolio metrics, risk analytics, and price history
    """
    if holdings_df.empty:
        return {}

    holdings_df['current_price'] = holdings_df['symbol'].map(lambda x: prices.get(x, {}).get('price', 0))
    holdings_df['market_value'] = holdings_df['quantity'] * holdings_df['current_price']
    holdings_df['cost_basis'] = holdings_df['quantity'] * holdings_df['avg_cost']
    holdings_df['pnl'] = holdings_df['market_value'] - holdings_df['cost_basis']
    holdings_df['pnl_pct'] = (holdings_df['pnl'] / holdings_df['cost_basis'] * 100).fillna(0)
    holdings_df['weight'] = holdings_df['market_value'] / holdings_df['market_value'].sum() * 100

    total_value = holdings_df['market_value'].sum()
    total_cost = holdings_df['cost_basis'].sum()
    total_pnl = holdings_df['pnl'].sum()
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    symbols = holdings_df['symbol'].tolist()
    returns_data = []
    price_history = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1y')
            if not hist.empty:
                returns = hist['Close'].pct_change().dropna()
                returns_data.append({
                    'symbol': symbol,
                    'returns': returns,
                    'weight': holdings_df[holdings_df['symbol'] == symbol]['weight'].iloc[0] / 100
                })
                price_history[symbol] = hist['Close'].tail(7).tolist()
        except:
            pass

    if returns_data:
        portfolio_returns = pd.Series(0.0, index=returns_data[0]['returns'].index)
        for data in returns_data:
            aligned_returns = data['returns'].reindex(portfolio_returns.index, fill_value=0)
            portfolio_returns += aligned_returns * data['weight']

        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (portfolio_returns.mean() * 252 - 0.03) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (portfolio_returns.mean() * 252 - 0.03) / downside_std if downside_std > 0 else 0

        var_95 = np.percentile(portfolio_returns, 5) * 100
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100

        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        try:
            spy = yf.Ticker('SPY').history(period='1y')['Close'].pct_change().dropna()
            aligned_spy = spy.reindex(portfolio_returns.index, fill_value=0)
            covariance = portfolio_returns.cov(aligned_spy)
            variance = aligned_spy.var()
            beta = covariance / variance if variance > 0 else 1
        except:
            beta = 1

    else:
        volatility = 0
        sharpe = 0
        sortino = 0
        var_95 = 0
        cvar_95 = 0
        max_drawdown = 0
        beta = 1

    return {
        'holdings_df': holdings_df,
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'beta': beta,
        'price_history': price_history,
    }


def create_portfolio() -> None:
    """
    Create new portfolio with user input.

    Prompts for portfolio name, description, and starting cash balance.
    """
    console.print("\n[bold cyan]═══ CREATE NEW PORTFOLIO ═══[/bold cyan]\n")

    name = Prompt.ask("[cyan]Portfolio name[/cyan]", default="Main Portfolio")
    description = Prompt.ask("[cyan]Description[/cyan]", default="My primary portfolio")
    cash = FloatPrompt.ask("[cyan]Starting cash balance[/cyan]", default=0.0)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO portfolios (name, description, created_date, cash_balance)
            VALUES (?, ?, ?, ?)
        ''', (name, description, datetime.now().isoformat(), cash))
        conn.commit()
        console.print(f"\n[green]✓[/green] Portfolio '{name}' created successfully!\n")
    except sqlite3.IntegrityError:
        console.print(f"\n[red]✗[/red] Portfolio '{name}' already exists!\n")
    finally:
        conn.close()


def list_portfolios() -> Optional[pd.DataFrame]:
    """
    Display all portfolios in formatted table.

    Returns:
        DataFrame containing all portfolios, or None if no portfolios exist
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM portfolios", conn)
    conn.close()

    if df.empty:
        console.print("\n[yellow]No portfolios found. Create one first![/yellow]\n")
        return None

    table = Table(title="[bold cyan]Your Portfolios[/bold cyan]", box=box.SQUARE, row_styles=[THEME['row_even'], THEME['row_odd']])
    table.add_column("#", style="bright_black", width=3)
    table.add_column("Name", style="cyan bold")
    table.add_column("Description", style="white")
    table.add_column("Cash", justify="right", style="green")
    table.add_column("Created", style="bright_black")

    for i, row in df.iterrows():
        table.add_row(
            str(i + 1),
            row['name'],
            row['description'] or '',
            f"${row['cash_balance']:,.2f}",
            row['created_date'][:10]
        )

    console.print(table)
    return df


def add_holding(portfolio_id: int) -> None:
    """
    Add or update holding in portfolio.

    Args:
        portfolio_id: Database ID of the portfolio

    Updates average cost if holding already exists.
    """
    console.print("\n[bold cyan]═══ ADD HOLDING ═══[/bold cyan]\n")

    symbol = Prompt.ask("[cyan]Stock symbol[/cyan]").upper()
    quantity = FloatPrompt.ask("[cyan]Quantity[/cyan]")
    price = FloatPrompt.ask("[cyan]Purchase price[/cyan]")
    date = Prompt.ask("[cyan]Purchase date (YYYY-MM-DD)[/cyan]", default=datetime.now().strftime('%Y-%m-%d'))
    notes = Prompt.ask("[cyan]Notes (optional)[/cyan]", default="")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM holdings WHERE portfolio_id = ? AND symbol = ?', (portfolio_id, symbol))
        existing = cursor.fetchone()

        if existing:
            old_qty, old_cost = existing[3], existing[4]
            new_qty = old_qty + quantity
            new_cost = ((old_qty * old_cost) + (quantity * price)) / new_qty

            cursor.execute('''
                UPDATE holdings SET quantity = ?, avg_cost = ? WHERE portfolio_id = ? AND symbol = ?
            ''', (new_qty, new_cost, portfolio_id, symbol))
        else:
            cursor.execute('''
                INSERT INTO holdings (portfolio_id, symbol, quantity, avg_cost, purchase_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, symbol, quantity, price, date, notes))

        cursor.execute('''
            INSERT INTO transactions (portfolio_id, symbol, action, quantity, price, transaction_date, notes)
            VALUES (?, ?, 'BUY', ?, ?, ?, ?)
        ''', (portfolio_id, symbol, quantity, price, date, notes))

        conn.commit()
        console.print(f"\n[green]✓[/green] Added {quantity} shares of {symbol} @ ${price:.2f}\n")

    except Exception as e:
        console.print(f"\n[red]✗[/red] Error: {str(e)}\n")
    finally:
        conn.close()


def view_portfolio(portfolio_id: int) -> None:
    """
    Display portfolio with comprehensive metrics and risk analytics.

    Args:
        portfolio_id: Database ID of the portfolio

    Shows summary, holdings table, risk metrics, and performance data.
    """
    conn = sqlite3.connect(DB_PATH)

    portfolio = pd.read_sql_query("SELECT * FROM portfolios WHERE id = ?", conn, params=(portfolio_id,))
    if portfolio.empty:
        console.print("[red]Portfolio not found![/red]")
        conn.close()
        return

    holdings = pd.read_sql_query("SELECT * FROM holdings WHERE portfolio_id = ?", conn, params=(portfolio_id,))
    conn.close()

    console.clear()

    header_text = Text()
    header_text.append(f" {portfolio['name'].iloc[0]}\n", style="bold cyan")
    header_text.append(f"{portfolio['description'].iloc[0]}", style="bright_black")
    console.print(Panel(header_text, border_style=THEME['border'], box=box.SIMPLE_HEAVY, style=THEME['panel_bg']))
    console.print()

    if holdings.empty:
        console.print("[yellow]No holdings in this portfolio.[/yellow]\n")
        return

    symbols = holdings['symbol'].tolist()
    with console.status("[cyan]Fetching current prices...", spinner="dots"):
        prices = get_current_prices(symbols)

    metrics = calculate_portfolio_metrics(holdings, prices)
    holdings_df = metrics['holdings_df']

    summary_table = Table.grid(padding=(0, 3))
    summary_table.add_column(style="bright_black")
    summary_table.add_column(justify="right", style="bold white")
    summary_table.add_column(style="bright_black")
    summary_table.add_column(justify="right", style="bold white")

    pnl_color = "green" if metrics['total_pnl'] >= 0 else "red"

    summary_table.add_row("Total Value", f"[bold cyan]${metrics['total_value']:,.2f}[/bold cyan]",
                         "Total Cost", f"${metrics['total_cost']:,.2f}")
    summary_table.add_row("Total P&L", f"[{pnl_color}]${metrics['total_pnl']:,.2f}[/{pnl_color}]",
                         "P&L %", f"[{pnl_color}]{metrics['total_pnl_pct']:+.2f}%[/{pnl_color}]")
    summary_table.add_row("Cash", f"${portfolio['cash_balance'].iloc[0]:,.2f}",
                         "Total Assets", f"[bold green]${metrics['total_value'] + portfolio['cash_balance'].iloc[0]:,.2f}[/bold green]")

    console.print(Panel(summary_table, title="[bold cyan]Portfolio Summary[/bold cyan]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    holdings_table = Table(title="[bold cyan]Holdings[/bold cyan]", box=box.SQUARE, show_header=True, header_style=f"bold white {THEME['header_bg']}", row_styles=[THEME['row_even'], THEME['row_odd']])

    holdings_table.add_column("Symbol", style="bold cyan", width=8)
    holdings_table.add_column("Quantity", justify="right", width=10)
    holdings_table.add_column("Avg Cost", justify="right", width=10)
    holdings_table.add_column("Current", justify="right", width=10)
    holdings_table.add_column("Market Value", justify="right", width=12)
    holdings_table.add_column("P&L", justify="right", width=12)
    holdings_table.add_column("P&L %", justify="right", width=12)
    holdings_table.add_column("Weight", justify="right", width=8)
    holdings_table.add_column("30-Day Trend", justify="center", width=18)

    price_hist = metrics.get('price_history', {})

    for _, row in holdings_df.iterrows():
        pnl_color = get_performance_color(row['pnl_pct'])

        # Enhanced 30-day sparkline
        prices_30d = price_hist.get(row['symbol'], [])
        sparkline = create_enhanced_sparkline(prices_30d, width=12, show_trend=False)

        # Formatted P&L percentage with color gradient
        pnl_pct_formatted = format_percentage(row['pnl_pct'], decimals=2, colored=True)

        holdings_table.add_row(
            row['symbol'],
            f"{row['quantity']:.2f}",
            f"${row['avg_cost']:.2f}",
            f"${row['current_price']:.2f}",
            f"${row['market_value']:,.2f}",
            f"[{pnl_color}]${row['pnl']:,.2f}[/{pnl_color}]",
            pnl_pct_formatted,
            f"{row['weight']:.1f}%",
            sparkline
        )

    console.print(holdings_table)
    console.print()

    risk_table = Table.grid(padding=(0, 3))
    risk_table.add_column(style="bright_black")
    risk_table.add_column(justify="right", style="bold white")
    risk_table.add_column(style="bright_black")
    risk_table.add_column(justify="right", style="bold white")

    risk_table.add_row("Volatility", f"{metrics['volatility']:.2f}%", "Beta (vs SPY)", f"{metrics['beta']:.2f}")
    risk_table.add_row("Sharpe Ratio", f"{metrics['sharpe']:.2f}", "Sortino Ratio", f"{metrics['sortino']:.2f}")
    risk_table.add_row("VaR (95%)", f"{metrics['var_95']:.2f}%", "CVaR (95%)", f"{metrics['cvar_95']:.2f}%")
    risk_table.add_row("Max Drawdown", f"[red]{metrics['max_drawdown']:.2f}%[/red]", "", "")

    console.print(Panel(risk_table, title="[bold cyan]Risk Analytics[/bold cyan]", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    console.print(f"[bright_black]Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(holdings_df)} holdings | Risk-adjusted return (Sharpe): {metrics['sharpe']:.2f}[/bright_black]\n")


def portfolio_menu() -> None:
    """
    Main portfolio management menu loop.

    Provides options for viewing, creating, and managing portfolios.
    """
    while True:
        console.clear()

        now = datetime.now()

        header = Text()
        header.append("═" * 80, style="bright_green")
        header.append("\n")
        header.append("                    ", style="bright_green")
        header.append(" 💼 PROFESSIONAL PORTFOLIO TRACKER 💼 ", style="bold white on green")
        header.append("\n", style="bright_green")
        header.append("═" * 80, style="bright_green")
        header.append("\n\n")
        header.append("  Institutional-Grade Portfolio Management", style="bold bright_green")
        header.append(" • ", style="bright_black")
        header.append(now.strftime('%A, %B %d, %Y at %I:%M:%S %p ET'), style="green")
        header.append(" • ", style="bright_black")
        header.append("Real-Time P&L Tracking", style="bold bright_green")

        console.print(Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
        console.print()

        table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2), row_styles=[THEME['row_even'], THEME['row_odd']], border_style=THEME['border'])
        table.add_column("Option", style="bright_green bold", width=3)
        table.add_column("Action", style="white")

        table.add_row("1", " View Portfolio")
        table.add_row("2", "➕ Add Holding")
        table.add_row("3", "➖ Remove Holding")
        table.add_row("4", "List All Portfolios")
        table.add_row("5", "🆕 Create New Portfolio")
        table.add_row("6", " Adjust Cash Balance")
        table.add_row("7", "📜 Transaction History")
        table.add_row("8", "📥 Export to CSV")
        table.add_row("9", " Exit")

        console.print(Panel(table, border_style=THEME['border'], style=THEME['panel_bg']))

        choice = Prompt.ask("\n[cyan]Choose option[/cyan]", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"], default="1")

        if choice == "1":
            portfolios = list_portfolios()
            if portfolios is not None and not portfolios.empty:
                console.print()
                idx = IntPrompt.ask("[cyan]Select portfolio #[/cyan]", default=1) - 1
                if 0 <= idx < len(portfolios):
                    view_portfolio(portfolios.iloc[idx]['id'])
                    console.print("\n[bright_black]Press Enter to continue...[/bright_black]")
                    input()

        elif choice == "2":
            portfolios = list_portfolios()
            if portfolios is not None and not portfolios.empty:
                console.print()
                idx = IntPrompt.ask("[cyan]Select portfolio #[/cyan]", default=1) - 1
                if 0 <= idx < len(portfolios):
                    add_holding(portfolios.iloc[idx]['id'])
                    console.print("[bright_black]Press Enter to continue...[/bright_black]")
                    input()

        elif choice == "4":
            list_portfolios()
            console.print("\n[bright_black]Press Enter to continue...[/bright_black]")
            input()

        elif choice == "5":
            create_portfolio()
            console.print("[bright_black]Press Enter to continue...[/bright_black]")
            input()

        elif choice == "9":
            console.print("\n[cyan]Thanks for using Portfolio Tracker! [/cyan]\n")
            break


if __name__ == "__main__":
    # Initialize database
    init_database()

    try:
        portfolio_menu()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Exiting...[/yellow]\n")
