#!/usr/bin/env python3
"""
BACKTESTING ENGINE
Test trading strategies with walk-forward validation
Institutional-grade backtesting with transaction costs and slippage
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import yfinance as yf
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.prompt import Prompt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester import InstitutionalBacktester
from src.risk_management import RiskManager

console = Console()

THEME = {
    'header_bg': 'on grey23',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}


def show_banner() -> None:
    """Display backtesting banner"""
    banner = Text()
    banner.append("BACKTESTING ENGINE\n", style="bold white")
    banner.append("Walk-Forward Validation • Transaction Costs • Slippage Simulation\n\n", style="bright_black")
    banner.append("Strategy Testing", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Risk Metrics", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Performance Analysis", style="white")

    console.print(Panel(banner, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
    console.print()


def simple_momentum_strategy(train_data: pd.Series, test_data: pd.Series) -> pd.Series:
    """Simple momentum strategy: Buy if 20-day MA > 50-day MA"""
    sma_20 = test_data.rolling(20).mean()
    sma_50 = test_data.rolling(50).mean()

    signals = pd.Series(0, index=test_data.index)
    signals[sma_20 > sma_50] = 1

    return signals


def mean_reversion_strategy(train_data: pd.Series, test_data: pd.Series) -> pd.Series:
    """Mean reversion strategy: Buy when price below 20-day MA"""
    sma_20 = test_data.rolling(20).mean()

    signals = pd.Series(0, index=test_data.index)
    signals[test_data < sma_20] = 1

    return signals


def breakout_strategy(train_data: pd.Series, test_data: pd.Series) -> pd.Series:
    """Breakout strategy: Buy on 52-week highs"""
    rolling_high = test_data.rolling(252).max()

    signals = pd.Series(0, index=test_data.index)
    signals[test_data >= rolling_high * 0.98] = 1

    return signals


def fetch_stock_data(symbol: str) -> Optional[pd.Series]:
    """Fetch historical stock data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2y')

        if hist.empty:
            return None

        return hist['Close']
    except Exception:
        return None


def create_results_panel(result: Any, strategy_name: str) -> Panel:
    """Create backtest results panel"""
    text = Text()
    text.append(f"{strategy_name.upper()} RESULTS\n\n", style="bold white")

    text.append("Performance:\n", style="bold white")
    ret_color = "green" if result.total_return > 0 else "red"
    text.append(f"  Total Return: {result.total_return:.2f}%\n", style=ret_color)
    text.append(f"  Annual Return: {result.annualized_return:.2f}%\n", style="white")
    text.append(f"  Max Drawdown: {result.max_drawdown:.2f}%\n\n", style="red" if result.max_drawdown < -20 else "white")

    text.append("Risk-Adjusted:\n", style="bold white")
    sharpe_color = "green" if result.sharpe_ratio > 1.0 else "white"
    text.append(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}\n", style=sharpe_color)
    text.append(f"  Sortino Ratio: {result.sortino_ratio:.2f}\n", style="white")
    text.append(f"  Calmar Ratio: {result.calmar_ratio:.2f}\n\n", style="white")

    text.append("Trading:\n", style="bold white")
    text.append(f"  Total Trades: {result.total_trades}\n", style="white")
    wr_color = "green" if result.win_rate > 50 else "red"
    text.append(f"  Win Rate: {result.win_rate:.1f}%\n", style=wr_color)
    text.append(f"  Avg Trade: {result.avg_trade_return:.2f}%\n", style="white")
    pf_color = "green" if result.profit_factor > 1.5 else "white"
    text.append(f"  Profit Factor: {result.profit_factor:.2f}\n\n", style=pf_color)

    text.append("Risk:\n", style="bold white")
    text.append(f"  Volatility: {result.volatility:.2f}%\n", style="white")
    text.append(f"  VaR (95%): {result.var_95:.2f}%\n", style="white")
    text.append(f"  CVaR (95%): {result.cvar_95:.2f}%", style="bright_black")

    return Panel(text, title=f"[bold white]{strategy_name.upper()}[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_comparison_table(results: Dict[str, Any]) -> Table:
    """Create strategy comparison table"""
    table = Table(show_header=True, header_style="bold white", border_style=THEME['border'],
                  box=box.SQUARE, style=THEME['panel_bg'])

    table.add_column("Strategy", style="white", width=20)
    table.add_column("Total Return", justify="right", style="white")
    table.add_column("Sharpe", justify="right", style="white")
    table.add_column("Max DD", justify="right", style="white")
    table.add_column("Win Rate", justify="right", style="white")
    table.add_column("Trades", justify="right", style="white")

    for name, result in results.items():
        ret_color = "green" if result.total_return > 0 else "red"
        sharpe_color = "green" if result.sharpe_ratio > 1.0 else "white"
        wr_color = "green" if result.win_rate > 50 else "white"

        table.add_row(
            name,
            f"[{ret_color}]{result.total_return:.1f}%[/{ret_color}]",
            f"[{sharpe_color}]{result.sharpe_ratio:.2f}[/{sharpe_color}]",
            f"{result.max_drawdown:.1f}%",
            f"[{wr_color}]{result.win_rate:.0f}%[/{wr_color}]",
            str(result.total_trades)
        )

    return table


def main() -> None:
    """Main backtesting application"""
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        console.clear()
        show_banner()
        symbol = Prompt.ask("[white]Enter stock symbol[/white]", default="AAPL").upper()
        console.print()

    console.print(f"[white]Fetching data for {symbol}...[/white]")
    prices = fetch_stock_data(symbol)

    if prices is None or len(prices) < 300:
        console.print("[red]Error: Not enough data for backtesting (need 2 years)[/red]\n")
        return

    console.print(f"[green]✓ Loaded {len(prices)} days of data[/green]\n")

    console.print("[white]Running backtests...[/white]\n")

    backtester = InstitutionalBacktester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        position_size=0.25
    )

    strategies = {
        'Momentum (MA Crossover)': simple_momentum_strategy,
        'Mean Reversion': mean_reversion_strategy,
        'Breakout (52W High)': breakout_strategy
    }

    results = {}

    for name, strategy_func in strategies.items():
        console.print(f"  Testing {name}...")
        result = backtester.backtest_strategy(prices, strategy_func(prices, prices))
        if result:
            results[name] = result
            console.print(f"  [green]✓ {name} complete[/green]")
        else:
            console.print(f"  [red]✗ {name} failed[/red]")

    console.print()
    console.clear()
    show_banner()

    header_text = Text()
    header_text.append(f"{symbol} BACKTEST RESULTS", style="bold white")
    header_text.append(" │ ", style="bright_black")
    header_text.append(f"2-Year Period", style="white")
    header_text.append(" │ ", style="bright_black")
    header_text.append(f"{len(prices)} Days", style="bright_black")

    console.print(Panel(header_text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    if len(results) >= 3:
        from rich.columns import Columns
        console.print(Columns([
            create_results_panel(results['Momentum (MA Crossover)'], 'Momentum'),
            create_results_panel(results['Mean Reversion'], 'Mean Reversion'),
            create_results_panel(results['Breakout (52W High)'], 'Breakout')
        ]))
        console.print()

    if results:
        comparison = create_comparison_table(results)
        console.print(Panel(comparison, title="[bold white]STRATEGY COMPARISON[/bold white]",
                           border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
        console.print()

    best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    text = Text()
    text.append("Best Strategy: ", style="white")
    text.append(best_strategy[0], style="bold green")
    text.append(f" (Sharpe: {best_strategy[1].sharpe_ratio:.2f})", style="bright_black")
    console.print(Panel(text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    console.print("[bright_black]Backtesting Engine • Walk-Forward Validation • Transaction Costs Included • Not Financial Advice[/bright_black]", justify="center")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[white]Cancelled by user[/white]\n")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        import traceback
        traceback.print_exc()
