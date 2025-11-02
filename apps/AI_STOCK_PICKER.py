#!/usr/bin/env python3
"""
AI Stock Picker

Smart portfolio builder with AI-powered stock selection and optimal allocation.
Analyzes stocks based on fundamentals, growth metrics, and risk profiles.
"""

import warnings
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box
from rich.columns import Columns

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from stock_universe import SECTORS, DIVIDEND_ARISTOCRATS, HIGH_GROWTH_TECH, MEGA_CAPS
from src.ml_engine import AdvancedMLScorer

console = Console()
ml_scorer = AdvancedMLScorer()

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

STOCK_CATEGORIES = {
    'tech': {
        'name': 'Technology & Innovation',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Technology', [])}
    },
    'communication': {
        'name': 'Communication Services',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Communication', [])}
    },
    'healthcare': {
        'name': 'Healthcare & Pharmaceuticals',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Healthcare', [])}
    },
    'finance': {
        'name': 'Financial Services',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Financials', [])}
    },
    'consumer_discretionary': {
        'name': 'Consumer Discretionary',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Consumer Discretionary', [])}
    },
    'consumer_staples': {
        'name': 'Consumer Staples',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Consumer Staples', [])}
    },
    'energy': {
        'name': 'Energy',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Energy', [])}
    },
    'utilities': {
        'name': 'Utilities',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Utilities', [])}
    },
    'industrials': {
        'name': 'Industrials & Manufacturing',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Industrials', [])}
    },
    'materials': {
        'name': 'Materials',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Materials', [])}
    },
    'real_estate': {
        'name': 'Real Estate & REITs',
        'stocks': {symbol: symbol for symbol in SECTORS.get('Real Estate', [])}
    },
    'dividend': {
        'name': 'Dividend Aristocrats (25+ years)',
        'stocks': {symbol: symbol for symbol in DIVIDEND_ARISTOCRATS}
    },
    'growth': {
        'name': 'High Growth Tech',
        'stocks': {symbol: symbol for symbol in HIGH_GROWTH_TECH}
    },
    'mega': {
        'name': 'Mega Caps (>$500B)',
        'stocks': {symbol: symbol for symbol in MEGA_CAPS}
    },
}

RISK_PROFILES = {
    'conservative': {
        'name': 'Conservative (Low Risk)',
        'stocks_count': 5,
        'max_single_allocation': 25,
        'min_dividend_yield': 2.0,
        'prefer_large_cap': True,
        'cash_buffer': 10,
    },
    'moderate': {
        'name': 'Moderate (Balanced)',
        'stocks_count': 4,
        'max_single_allocation': 30,
        'min_dividend_yield': 0,
        'prefer_large_cap': True,
        'cash_buffer': 5,
    },
    'aggressive': {
        'name': 'Aggressive (High Growth)',
        'stocks_count': 3,
        'max_single_allocation': 40,
        'min_dividend_yield': 0,
        'prefer_large_cap': False,
        'cash_buffer': 0,
    },
}


def show_banner() -> None:
    """Display AI Stock Picker banner."""
    banner = Text()
    banner.append("AI STOCK PICKER\n\n", style="bold white")
    banner.append("Smart Portfolio Builder", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Optimal Allocation", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("AI-Powered Recommendations", style="white")

    console.print(Panel(banner, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
    console.print()


def get_user_preferences() -> Dict[str, Any]:
    """
    Collect user investment preferences interactively.

    Returns:
        Dictionary containing budget, areas, risk level, and timeline
    """
    console.print("[bold white]Let's build your optimal portfolio![/bold white]\n")

    console.print("[white]Investment Budget[/white]")
    budget = IntPrompt.ask("How much do you want to invest?", default=10000)
    console.print()

    console.print("[white]Investment Interests (Select areas you're interested in)[/white]\n")

    for key, value in STOCK_CATEGORIES.items():
        stock_count = len(value['stocks'])
        console.print(f"  [white]{key:15}[/white] - {value['name']:35} [{stock_count} stocks]")

    console.print()
    areas_input = Prompt.ask("Enter areas (comma-separated)", default="tech,finance")
    areas = [a.strip().lower() for a in areas_input.split(',')]
    areas = [a for a in areas if a in STOCK_CATEGORIES]

    if not areas:
        console.print("[white]No valid areas selected. Using: tech, finance[/white]")
        areas = ['tech', 'finance']

    console.print()

    console.print("[white]Risk Tolerance[/white]\n")
    console.print("  [white]conservative[/white] - Low risk, focus on stable dividends")
    console.print("  [white]moderate[/white] - Balanced growth and stability")
    console.print("  [white]aggressive[/white] - High growth, higher volatility")
    console.print()

    risk = Prompt.ask("Risk level", choices=['conservative', 'moderate', 'aggressive'], default='moderate')
    console.print()

    console.print("[white]Investment Timeline[/white]")
    timeline = IntPrompt.ask("How many years do you plan to hold?", default=10)
    console.print()

    return {
        'budget': budget,
        'areas': areas,
        'risk': risk,
        'timeline': timeline,
    }


def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch stock metrics and ML features for analysis"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1y')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        price_1y_ago = hist['Close'].iloc[0]
        returns_1y = ((current_price - price_1y_ago) / price_1y_ago * 100)

        prices = hist['Close']
        rsi = 50
        if len(prices) >= 14:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if not rs.iloc[-1] == 0 else 50

        return {
            'symbol': symbol,
            'name': info.get('shortName', symbol),
            'price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'pe': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
            'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
            'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
            'current_ratio': info.get('currentRatio', 0),
            'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
            'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
            'free_cash_flow': info.get('freeCashflow', 0),
            'operating_cash_flow': info.get('operatingCashflow', 0),
            'rsi': rsi,
            'returns_1y': returns_1y,
            'momentum_score': min(max(returns_1y, 0), 100),
            'piotroski_score': 5,
            'beta': info.get('beta', 1.0),
            'recommendation': info.get('recommendationKey', 'hold').upper(),
        }

    except Exception:
        return None


def score_stock(stock: Dict[str, Any], risk_profile: str, preferences: Dict[str, Any]) -> Tuple[int, str, float]:
    """Score stock using advanced ML ensemble methods"""
    ml_result = ml_scorer.score_stock(stock)

    base_score = ml_result.score

    if risk_profile == 'conservative':
        if stock['dividend_yield'] > 2:
            base_score = min(base_score + 5, 100)
        if stock['beta'] < 1.0:
            base_score = min(base_score + 5, 100)
    elif risk_profile == 'aggressive':
        if stock['returns_1y'] > 20:
            base_score = min(base_score + 5, 100)
        if stock['revenue_growth'] > 15:
            base_score = min(base_score + 5, 100)

    return int(base_score), ml_result.signal, ml_result.confidence


def calculate_optimal_allocation(stocks: List[Dict[str, Any]], budget: float, risk_profile: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Calculate optimal portfolio allocation based on stock scores.

    Args:
        stocks: List of stock data dictionaries with scores
        budget: Total investment budget
        risk_profile: User's risk tolerance

    Returns:
        Tuple of (allocations list, cash amount)
    """
    profile = RISK_PROFILES[risk_profile]
    stocks_sorted = sorted(stocks, key=lambda x: x['score'], reverse=True)
    selected_count = profile['stocks_count']
    selected_stocks = stocks_sorted[:selected_count]

    investable = budget * (1 - profile['cash_buffer'] / 100)
    total_score = sum(s['score'] for s in selected_stocks)

    allocations = []
    for stock in selected_stocks:
        raw_allocation = (stock['score'] / total_score) * 100
        allocation_pct = min(raw_allocation, profile['max_single_allocation'])

        allocations.append({
            'symbol': stock['symbol'],
            'name': stock['name'],
            'price': stock['price'],
            'score': stock['score'],
            'ml_signal': stock.get('ml_signal', 'HOLD'),
            'ml_confidence': stock.get('ml_confidence', 0.5),
            'allocation_pct': allocation_pct,
            'amount': (allocation_pct / 100) * investable,
            'shares': int((allocation_pct / 100) * investable / stock['price']),
            'dividend_yield': stock['dividend_yield'],
            'returns_1y': stock['returns_1y'],
            'pe': stock['pe'],
        })

    total_pct = sum(a['allocation_pct'] for a in allocations)
    for alloc in allocations:
        alloc['allocation_pct'] = (alloc['allocation_pct'] / total_pct) * 100
        alloc['amount'] = (alloc['allocation_pct'] / 100) * investable
        alloc['shares'] = int(alloc['amount'] / alloc['price'])

    cash_amount = budget * (profile['cash_buffer'] / 100)

    return allocations, cash_amount


def create_allocation_table(allocations: List[Dict[str, Any]]) -> Table:
    """Create portfolio allocation table with ML signals"""
    table = Table(
        title="AI-RECOMMENDED PORTFOLIO (ML-POWERED)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Stock", style="white", width=8)
    table.add_column("Company", style="white", width=18)
    table.add_column("Score", style="white", justify="center", width=8)
    table.add_column("Signal", style="white", justify="center", width=12)
    table.add_column("Allocation", style="white", justify="right", width=10)
    table.add_column("Amount", style="white", justify="right", width=11)
    table.add_column("Shares", style="white", justify="right", width=7)

    for alloc in allocations:
        score_color = "bright_green" if alloc['score'] > 70 else "green" if alloc['score'] > 50 else "white"

        signal = alloc.get('ml_signal', 'HOLD')
        signal_color = "bright_green" if 'BUY' in signal else "green" if signal == 'HOLD' else "red"

        table.add_row(
            alloc['symbol'],
            alloc['name'][:18],
            f"[{score_color}]{alloc['score']}/100[/{score_color}]",
            f"[{signal_color}]{signal}[/{signal_color}]",
            f"{alloc['allocation_pct']:.1f}%",
            f"${alloc['amount']:,.0f}",
            str(alloc['shares'])
        )

    return table


def create_diversification_panel(allocations: List[Dict[str, Any]], areas: List[str]) -> Panel:
    """Create diversification analysis panel."""
    text = Text()
    text.append("PORTFOLIO DIVERSIFICATION\n\n", style="bold white")

    text.append("Sector Exposure:\n", style="bold white")
    for area in areas:
        text.append(f"  {STOCK_CATEGORIES[area]['name']}\n", style="white")

    text.append(f"\nNumber of Holdings: {len(allocations)}\n", style="white")

    max_allocation = max(a['allocation_pct'] for a in allocations)
    text.append(f"Largest Position: {max_allocation:.1f}%\n", style="white")

    if max_allocation > 40:
        text.append("  High concentration\n", style="white")
    elif max_allocation > 30:
        text.append("  Moderate concentration\n", style="green")
    else:
        text.append("  Well diversified\n", style="bright_green")

    text.append("\nPortfolio Metrics:\n", style="bold white")
    avg_returns = np.mean([a['returns_1y'] for a in allocations])
    avg_pe = np.mean([a['pe'] for a in allocations if a['pe'] > 0])
    total_dividend = sum(a['dividend_yield'] * a['allocation_pct'] / 100 for a in allocations)

    text.append(f"  Avg 1Y Return: ", style="white")
    color = "green" if avg_returns > 0 else "red"
    text.append(f"{avg_returns:+.1f}%\n", style=color)
    text.append(f"  Avg P/E Ratio: {avg_pe:.1f}\n", style="white")
    text.append(f"  Portfolio Yield: {total_dividend:.2f}%\n", style="white")

    return Panel(text, title="[bold white]DIVERSIFICATION[/bold white]", border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_projection_panel(allocations: List[Dict[str, Any]], cash: float, budget: float, timeline: int) -> Panel:
    """Create investment projection panel with multiple scenarios."""
    text = Text()
    text.append("INVESTMENT PROJECTION\n\n", style="bold white")

    avg_return = np.mean([a['returns_1y'] for a in allocations])

    conservative_annual = max(avg_return - 5, 0)
    conservative_future = budget * ((1 + conservative_annual / 100) ** timeline)

    avg_future = budget * ((1 + avg_return / 100) ** timeline)

    optimistic_annual = avg_return + 5
    optimistic_future = budget * ((1 + optimistic_annual / 100) ** timeline)

    text.append(f"Initial Investment: ${budget:,.0f}\n", style="white")
    text.append(f"Timeline: {timeline} years\n\n", style="white")

    text.append(f"Conservative ({conservative_annual:.1f}%/yr):\n", style="bold white")
    text.append(f"  ${conservative_future:,.0f}\n", style="white")
    gain_c = conservative_future - budget
    text.append(f"  Gain: ${gain_c:+,.0f} ({gain_c/budget*100:+.0f}%)\n\n", style="green" if gain_c > 0 else "red")

    text.append(f"Average ({avg_return:.1f}%/yr):\n", style="bold green")
    text.append(f"  ${avg_future:,.0f}\n", style="white")
    gain_a = avg_future - budget
    text.append(f"  Gain: ${gain_a:+,.0f} ({gain_a/budget*100:+.0f}%)\n\n", style="green" if gain_a > 0 else "red")

    text.append(f"Optimistic ({optimistic_annual:.1f}%/yr):\n", style="bold white")
    text.append(f"  ${optimistic_future:,.0f}\n", style="white")
    gain_o = optimistic_future - budget
    text.append(f"  Gain: ${gain_o:+,.0f} ({gain_o/budget*100:+.0f}%)", style="green" if gain_o > 0 else "red")

    return Panel(text, title=f"[bold white]{timeline}-YEAR PROJECTION[/bold white]", border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_why_panel(allocations: List[Dict[str, Any]], risk_profile: str) -> Panel:
    """Create AI reasoning panel explaining portfolio selection."""
    text = Text()
    text.append("WHY THIS PORTFOLIO?\n\n", style="bold white")

    profile = RISK_PROFILES[risk_profile]

    text.append(f"Risk Profile: {profile['name']}\n", style="white")
    text.append(f"Number of Stocks: {len(allocations)}\n", style="white")
    text.append(f"Cash Buffer: {profile['cash_buffer']}%\n\n", style="white")

    text.append("Selection Criteria:\n", style="bold white")
    text.append("  ✓ Highest quality scores\n", style="green")
    text.append("  ✓ Strong fundamentals\n", style="green")
    text.append("  ✓ Sector diversification\n", style="green")
    text.append("  ✓ Risk-adjusted returns\n", style="green")

    if profile['cash_buffer'] > 0:
        text.append(f"  ✓ {profile['cash_buffer']}% cash for safety\n", style="green")

    text.append("\nBest For:\n", style="bold white")
    if risk_profile == 'conservative':
        text.append("  • Wealth preservation\n", style="white")
        text.append("  • Stable income (dividends)\n", style="white")
        text.append("  • Lower volatility\n", style="white")
    elif risk_profile == 'moderate':
        text.append("  • Balanced growth\n", style="white")
        text.append("  • Mix of stability & growth\n", style="white")
        text.append("  • Long-term wealth building\n", style="white")
    else:
        text.append("  • Maximum growth potential\n", style="white")
        text.append("  • Higher returns target\n", style="white")
        text.append("  • Accept higher volatility\n", style="white")

    return Panel(text, title="[bold white]AI REASONING[/bold white]", border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def main() -> None:
    """Main AI Stock Picker application."""
    console.clear()
    show_banner()

    prefs = get_user_preferences()

    console.print("[white]AI is analyzing stocks in your selected areas...[/white]\n")

    all_stocks = {}
    for area in prefs['areas']:
        all_stocks.update(STOCK_CATEGORIES[area]['stocks'])

    stocks_data = []
    for symbol, description in all_stocks.items():
        console.print(f"  Analyzing {symbol}...", style="bright_black")
        data = fetch_stock_data(symbol)
        if data:
            score, signal, confidence = score_stock(data, prefs['risk'], prefs)
            data['score'] = score
            data['ml_signal'] = signal
            data['ml_confidence'] = confidence
            stocks_data.append(data)

    if not stocks_data:
        console.print("\n[red]Error: Could not fetch stock data[/red]")
        return

    console.print(f"\n[green]✓ Analyzed {len(stocks_data)} stocks[/green]\n")

    allocations, cash = calculate_optimal_allocation(stocks_data, prefs['budget'], prefs['risk'])

    console.clear()
    show_banner()

    summary_text = Text()
    summary_text.append("Budget: ", style="white")
    summary_text.append(f"${prefs['budget']:,.0f}", style="bold bright_green")
    summary_text.append(" • Risk: ", style="white")
    summary_text.append(f"{prefs['risk'].title()}", style="bold white")
    summary_text.append(" • Timeline: ", style="white")
    summary_text.append(f"{prefs['timeline']} years", style="bold white")
    summary_text.append(" • Areas: ", style="white")
    summary_text.append(f"{', '.join(prefs['areas']).title()}", style="bold white")

    console.print(Panel(summary_text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    console.print(create_allocation_table(allocations))
    console.print()

    if cash > 0:
        cash_panel = Panel(
            f"[bold white]Cash Reserve: [green]${cash:,.0f}[/green] ({cash/prefs['budget']*100:.0f}%)[/bold white]\n"
            f"[bright_black]Safety buffer for opportunities or emergencies[/bright_black]",
            title="[bold white]CASH RESERVE[/bold white]",
            border_style=THEME['border'],
            box=box.SQUARE,
            style=THEME['panel_bg']
        )
        console.print(cash_panel)
        console.print()

    console.print(Columns([
        create_diversification_panel(allocations, prefs['areas']),
        create_projection_panel(allocations, cash, prefs['budget'], prefs['timeline'])
    ]))
    console.print()

    console.print(create_why_panel(allocations, prefs['risk']))
    console.print()

    next_steps = Panel(
        "[bold white]NEXT STEPS:[/bold white]\n\n"
        "[white]1.[/white] Review the allocation above\n"
        "[white]2.[/white] Use [bold]Professional Research Terminal[/bold] to research each stock\n"
        "[white]3.[/white] Add stocks to [bold]Watchlist[/bold] to monitor\n"
        "[white]4.[/white] Add to [bold]Portfolio[/bold] once purchased",
        title="[bold white]NEXT STEPS[/bold white]",
        border_style=THEME['border'],
        box=box.SQUARE,
        padding=(1, 2),
        style=THEME['panel_bg']
    )
    console.print(next_steps)
    console.print()

    console.print("[bright_black]AI Stock Picker • Not financial advice • Do your own research[/bright_black]", justify="center")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[white]Cancelled by user[/white]\n")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
