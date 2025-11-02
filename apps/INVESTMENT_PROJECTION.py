"""
Investment Projection Calculator
What if you invest $X in a stock for Y years?
"""
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
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



def fetch_historical_returns(symbol: str, years: int = 10) -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker(symbol)

        # Get maximum available history
        hist = ticker.history(period='max')

        if hist.empty or len(hist) < 252:  # At least 1 year of data
            return None

        # Calculate annual returns for all available years
        annual_returns = []

        # Group by year and calculate returns
        hist['Year'] = hist.index.year
        years_available = sorted(hist['Year'].unique())

        for i in range(len(years_available) - 1):
            year = years_available[i]
            next_year = years_available[i + 1]

            year_data = hist[hist['Year'] == year]
            next_year_data = hist[hist['Year'] == next_year]

            if len(year_data) > 0 and len(next_year_data) > 0:
                start_price = year_data['Close'].iloc[-1]
                end_price = next_year_data['Close'].iloc[0]
                annual_return = ((end_price - start_price) / start_price * 100)
                annual_returns.append({
                    'year': year,
                    'return': annual_return
                })

        # Calculate statistics
        returns_list = [r['return'] for r in annual_returns]

        data = {
            'symbol': symbol,
            'name': ticker.info.get('longName', symbol),
            'current_price': hist['Close'].iloc[-1],
            'annual_returns': annual_returns,
            'avg_return': np.mean(returns_list),
            'median_return': np.median(returns_list),
            'std_dev': np.std(returns_list),
            'min_return': min(returns_list),
            'max_return': max(returns_list),
            'positive_years': sum(1 for r in returns_list if r > 0),
            'total_years': len(returns_list),
        }

        return data

    except Exception as e:
        console.print(f"[red]Error fetching data: {str(e)}[/red]")
        return None


def calculate_projection(investment: float, annual_return: float, years: int) -> Dict[str, float]:
    future_value = investment * ((1 + annual_return / 100) ** years)
    total_gain = future_value - investment
    total_return_pct = (total_gain / investment) * 100

    return {
        'future_value': future_value,
        'total_gain': total_gain,
        'total_return_pct': total_return_pct
    }


def project_year_by_year(investment: float, annual_return: float, years: int) -> List[Dict[str, float]]:
    projections = []
    current_value = investment

    for year in range(1, years + 1):
        current_value = current_value * (1 + annual_return / 100)
        gain = current_value - investment
        projections.append({
            'year': year,
            'value': current_value,
            'gain': gain,
            'return_pct': (gain / investment) * 100
        })

    return projections


def create_header(symbol: str, name: str, investment: float, years: int) -> Panel:
    text = Text()
    text.append("═" * 80, style="bright_green")
    text.append("\n")
    text.append("      ", style="bright_green")
    text.append(f"  INVESTMENT PROJECTION: {symbol} ", style="bold white on green")
    text.append("\n", style="bright_green")
    text.append("═" * 80, style="bright_green")
    text.append("\n\n")
    text.append(f"  If you invest ", style="white")
    text.append(f"${investment:,.0f}", style="bold bright_green")
    text.append(f" in {name}", style="white")
    text.append(f"\n  And hold for ", style="white")
    text.append(f"{years} years", style="bold bright_cyan")
    text.append(f", what could it become?", style="white")

    return Panel(text, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_scenarios_panel(investment: float, data: Dict, years: int) -> Panel:
    text = Text()

    # Conservative (10th percentile or avg - 1 std dev)
    conservative_return = max(data['avg_return'] - data['std_dev'], data['min_return'])
    conservative = calculate_projection(investment, conservative_return, years)

    # Average (mean return)
    average = calculate_projection(investment, data['avg_return'], years)

    # Optimistic (90th percentile or avg + 1 std dev)
    optimistic_return = min(data['avg_return'] + data['std_dev'], data['max_return'])
    optimistic = calculate_projection(investment, optimistic_return, years)

    # Conservative
    text.append("🔵 CONSERVATIVE SCENARIO\n", style="bold bright_blue")
    text.append(f"  Annual Return: {conservative_return:+.1f}%\n", style="bright_blue")
    text.append(f"  Future Value:  ", style="white")
    text.append(f"${conservative['future_value']:,.0f}\n", style="bold bright_white")
    text.append(f"  Total Gain:    ", style="white")
    color = "bright_green" if conservative['total_gain'] > 0 else "bright_red"
    text.append(f"${conservative['total_gain']:+,.0f}", style=color)
    text.append(f" ({conservative['total_return_pct']:+.0f}%)\n\n", style=color)

    # Average
    text.append(" AVERAGE SCENARIO\n", style="bold bright_green")
    text.append(f"  Annual Return: {data['avg_return']:+.1f}%\n", style="bright_green")
    text.append(f"  Future Value:  ", style="white")
    text.append(f"${average['future_value']:,.0f}\n", style="bold bright_white")
    text.append(f"  Total Gain:    ", style="white")
    color = "bright_green" if average['total_gain'] > 0 else "bright_red"
    text.append(f"${average['total_gain']:+,.0f}", style=color)
    text.append(f" ({average['total_return_pct']:+.0f}%)\n\n", style=color)

    # Optimistic
    text.append(" OPTIMISTIC SCENARIO\n", style="bold bright_yellow")
    text.append(f"  Annual Return: {optimistic_return:+.1f}%\n", style="bright_yellow")
    text.append(f"  Future Value:  ", style="white")
    text.append(f"${optimistic['future_value']:,.0f}\n", style="bold bright_white")
    text.append(f"  Total Gain:    ", style="white")
    color = "bright_green" if optimistic['total_gain'] > 0 else "bright_red"
    text.append(f"${optimistic['total_gain']:+,.0f}", style=color)
    text.append(f" ({optimistic['total_return_pct']:+.0f}%)\n", style=color)

    return Panel(
        text,
        title="[bold white on magenta]  THREE SCENARIOS [/bold white on magenta]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_historical_context_panel(data: Dict) -> Panel:
    text = Text()

    text.append("Historical Performance:\n\n", style="bold white")

    text.append(f"Average Return:  ", style="white")
    color = "bright_green" if data['avg_return'] > 0 else "bright_red"
    text.append(f"{data['avg_return']:+.1f}% per year\n", style=color)

    text.append(f"Median Return:   ", style="white")
    color = "bright_green" if data['median_return'] > 0 else "bright_red"
    text.append(f"{data['median_return']:+.1f}% per year\n\n", style=color)

    text.append(f"Volatility:      ", style="white")
    if data['std_dev'] < 15:
        vol_text = f"{data['std_dev']:.1f}% (Low)"
        vol_color = "green"
    elif data['std_dev'] < 25:
        vol_text = f"{data['std_dev']:.1f}% (Moderate)"
        vol_color = "yellow"
    else:
        vol_text = f"{data['std_dev']:.1f}% (High)"
        vol_color = "red"
    text.append(f"{vol_text}\n\n", style=vol_color)

    text.append(f"Best Year:       ", style="white")
    text.append(f"+{data['max_return']:.1f}%\n", style="bright_green")

    text.append(f"Worst Year:      ", style="white")
    text.append(f"{data['min_return']:+.1f}%\n\n", style="bright_red")

    consistency = (data['positive_years'] / data['total_years'] * 100)
    text.append(f"Positive Years:  ", style="white")
    text.append(f"{data['positive_years']}/{data['total_years']} ", style="bright_white")
    text.append(f"({consistency:.0f}%)", style="green" if consistency > 60 else "yellow")

    return Panel(
        text,
        title="[bold white on blue]  HISTORICAL CONTEXT [/bold white on blue]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_year_by_year_table(investment: float, data: Dict, years: int) -> Table:
    table = Table(
        title="[bold white on cyan] 📅 YEAR-BY-YEAR PROJECTION (AVERAGE SCENARIO) [/bold white on cyan]",
        box=box.SIMPLE_HEAVY,
        show_header=True
    )

    table.add_column("Year", style="bold white", justify="center", width=8)
    table.add_column("Value", style="bright_white", justify="right", width=16)
    table.add_column("Gain", style="bright_green", justify="right", width=16)
    table.add_column("Return", style="bright_cyan", justify="right", width=12)

    projections = project_year_by_year(investment, data['avg_return'], years)

    for proj in projections:
        year_text = f"Year {proj['year']}"
        value_text = f"${proj['value']:,.0f}"

        gain_color = "bright_green" if proj['gain'] > 0 else "bright_red"
        gain_text = f"${proj['gain']:+,.0f}"

        return_color = "bright_green" if proj['return_pct'] > 0 else "bright_red"
        return_text = f"{proj['return_pct']:+.0f}%"

        # Style the row
        if proj['year'] == years:
            # Final year - highlight
            table.add_row(
                f"[bold]{year_text}[/bold]",
                f"[bold]{value_text}[/bold]",
                f"[{gain_color} bold]{gain_text}[/{gain_color} bold]",
                f"[{return_color} bold]{return_text}[/{return_color} bold]"
            )
        else:
            table.add_row(
                year_text,
                value_text,
                f"[{gain_color}]{gain_text}[/{gain_color}]",
                f"[{return_color}]{return_text}[/{return_color}]"
            )

    return table


def create_risk_analysis_panel(investment: float, data: Dict, years: int) -> Panel:
    text = Text()

    # Calculate worst-case scenario
    worst_case = calculate_projection(investment, data['min_return'], years)

    text.append("  RISK ANALYSIS\n\n", style="bold yellow")

    # Volatility risk
    if data['std_dev'] > 30:
        risk_level = "HIGH RISK"
        risk_color = "bright_red"
        risk_desc = "Very volatile - be prepared for big swings"
    elif data['std_dev'] > 20:
        risk_level = "MODERATE-HIGH RISK"
        risk_color = "red"
        risk_desc = "Significant volatility - not for conservative investors"
    elif data['std_dev'] > 15:
        risk_level = "MODERATE RISK"
        risk_color = "yellow"
        risk_desc = "Normal stock market volatility"
    else:
        risk_level = "LOW-MODERATE RISK"
        risk_color = "green"
        risk_desc = "Relatively stable for a stock"

    text.append(f"Risk Level: ", style="white")
    text.append(f"{risk_level}\n", style=risk_color)
    text.append(f"{risk_desc}\n\n", style="bright_black")

    # Worst case
    text.append(f"Worst-Case Scenario:\n", style="bold white")
    text.append(f"  If returns match worst historical year ({data['min_return']:+.1f}%):\n", style="bright_black")
    text.append(f"  Future Value: ", style="white")
    text.append(f"${worst_case['future_value']:,.0f}\n", style="bright_white")
    text.append(f"  Total Gain:   ", style="white")
    wc_color = "bright_green" if worst_case['total_gain'] > 0 else "bright_red"
    text.append(f"${worst_case['total_gain']:+,.0f}", style=wc_color)
    text.append(f" ({worst_case['total_return_pct']:+.0f}%)\n\n", style=wc_color)

    # Consistency
    consistency = (data['positive_years'] / data['total_years'] * 100)
    text.append(f"Historical Consistency: ", style="white")
    if consistency > 70:
        text.append(f"{consistency:.0f}% positive years - Very reliable\n", style="bright_green")
    elif consistency > 60:
        text.append(f"{consistency:.0f}% positive years - Fairly reliable\n", style="green")
    else:
        text.append(f"{consistency:.0f}% positive years - Inconsistent\n", style="yellow")

    return Panel(
        text,
        title="[bold white on red]   RISK ANALYSIS [/bold white on red]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_comparison_panel(investment: float, stock_data: Dict, years: int) -> Panel:
    text = Text()

    # Fetch S&P 500 data
    sp500_data = fetch_historical_returns('^GSPC', years)

    if sp500_data:
        stock_projection = calculate_projection(investment, stock_data['avg_return'], years)
        sp500_projection = calculate_projection(investment, sp500_data['avg_return'], years)

        text.append(" COMPARISON TO S&P 500\n\n", style="bold white")

        # Stock
        text.append(f"{stock_data['symbol']}:\n", style="bold bright_cyan")
        text.append(f"  Avg Return: {stock_data['avg_return']:+.1f}%\n", style="white")
        text.append(f"  Future Value: ${stock_projection['future_value']:,.0f}\n\n", style="bright_white")

        # S&P 500
        text.append(f"S&P 500:\n", style="bold bright_yellow")
        text.append(f"  Avg Return: {sp500_data['avg_return']:+.1f}%\n", style="yellow")
        text.append(f"  Future Value: ${sp500_projection['future_value']:,.0f}\n\n", style="bright_white")

        # Comparison
        diff = stock_projection['future_value'] - sp500_projection['future_value']
        diff_pct = (diff / sp500_projection['future_value']) * 100

        text.append(f"Difference: ", style="bold white")
        if diff > 0:
            text.append(f"${diff:+,.0f} ({diff_pct:+.0f}%) better than market\n", style="bright_green")
            text.append(f"✓ Outperforms S&P 500", style="green")
        else:
            text.append(f"${diff:,.0f} ({diff_pct:.0f}%) worse than market\n", style="bright_red")
            text.append(f"✗ Underperforms S&P 500", style="red")
    else:
        text.append("Could not fetch S&P 500 comparison data", style="bright_black")

    return Panel(
        text,
        title="[bold white on magenta]  MARKET COMPARISON [/bold white on magenta]",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(1, 2), style=THEME['panel_bg']
    )


def create_bottom_line_panel(investment: float, data: Dict, years: int) -> Panel:
    text = Text()

    avg_projection = calculate_projection(investment, data['avg_return'], years)

    text.append(" BOTTOM LINE\n\n", style="bold white")

    text.append(f"If you invest ${investment:,.0f} in {data['symbol']}\n", style="white")
    text.append(f"and hold for {years} years:\n\n", style="white")

    text.append(f"Expected Value: ", style="white")
    text.append(f"${avg_projection['future_value']:,.0f}\n", style="bold bright_green")
    text.append(f"Expected Gain:  ", style="white")
    color = "bright_green" if avg_projection['total_gain'] > 0 else "bright_red"
    text.append(f"${avg_projection['total_gain']:+,.0f}", style=color)
    text.append(f" ({avg_projection['total_return_pct']:+.0f}%)\n\n", style=color)

    # Recommendation
    if data['avg_return'] > 15 and data['positive_years'] / data['total_years'] > 0.7:
        rec = "Strong long-term investment ✓"
        rec_color = "bright_green"
    elif data['avg_return'] > 10 and data['positive_years'] / data['total_years'] > 0.6:
        rec = "Good long-term investment"
        rec_color = "green"
    elif data['avg_return'] > 5:
        rec = "Moderate long-term investment"
        rec_color = "yellow"
    else:
        rec = "Consider alternatives"
        rec_color = "red"

    text.append(f"{rec}", style=rec_color)

    return Panel(
        text,
        title="[bold white on green]  BOTTOM LINE [/bold white on green]",
        border_style=THEME['border'],
        box=box.SQUARE,
        padding=(1, 2), style=THEME['panel_bg']
    )


def display_projection(symbol: str, investment: float, years: int) -> None:
    console.clear()
    console.print(f"\n[cyan]Analyzing {symbol} historical returns...[/cyan]\n")

    # Fetch data
    data = fetch_historical_returns(symbol, years)

    if not data:
        console.print(f"[red]Error: Could not fetch data for {symbol}[/red]")
        console.print(f"[yellow]Need at least 1 year of historical data[/yellow]\n")
        return

    console.clear()

    # Display header
    console.print(create_header(symbol, data['name'], investment, years))
    console.print()

    # Row 1: Scenarios & Historical Context
    console.print(Columns([
        create_scenarios_panel(investment, data, years),
        create_historical_context_panel(data)
    ]))
    console.print()

    # Year-by-year table
    console.print(create_year_by_year_table(investment, data, years))
    console.print()

    # Row 2: Risk Analysis & Market Comparison
    console.print(Columns([
        create_risk_analysis_panel(investment, data, years),
        create_comparison_panel(investment, data, years)
    ]))
    console.print()

    # Bottom line
    console.print(create_bottom_line_panel(investment, data, years))
    console.print()

    # Footer
    footer = Panel(
        "[bright_black]Investment Projection • Based on historical returns • Past performance does not guarantee future results[/bright_black]",
        box=box.SQUARE,
        border_style=THEME['border']
    , style=THEME['panel_bg'])
    console.print(footer)


def main() -> None:
    if len(sys.argv) >= 4:
        symbol = sys.argv[1].upper()
        try:
            investment = float(sys.argv[2])
            years = int(sys.argv[3])
        except ValueError:
            console.print("[red]Error: Investment amount and years must be numbers[/red]\n")
            sys.exit(1)
    else:
        console.print("\n[yellow]Investment Projection Calculator[/yellow]\n")

        symbol = input("Stock symbol [MSFT]: ").strip().upper()
        symbol = symbol if symbol else "MSFT"

        investment_input = input("Investment amount [$40000]: ").strip()
        investment = float(investment_input) if investment_input else 40000

        years_input = input("Number of years [10]: ").strip()
        years = int(years_input) if years_input else 10

    if investment <= 0:
        console.print("[red]Error: Investment amount must be positive[/red]\n")
        sys.exit(1)

    if years <= 0 or years > 50:
        console.print("[red]Error: Years must be between 1 and 50[/red]\n")
        sys.exit(1)

    try:
        display_projection(symbol, investment, years)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
