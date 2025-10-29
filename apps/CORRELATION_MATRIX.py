"""
CORRELATION MATRIX
Analyze stock correlations for portfolio diversification

Features:
- Correlation heatmap between stocks
- Beta vs S&P 500
- Find hedges (negatively correlated stocks)
- Portfolio diversification score
- Sector correlation analysis
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


def get_stock_data(symbols, period='1y'):
    """Fetch historical data for correlation analysis"""
    try:
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)

                if not hist.empty and len(hist) > 20:
                    data[symbol] = hist['Close']
            except Exception as e:
                console.print(f"[dim]Warning: Could not fetch {symbol}[/dim]")
                continue

        if not data:
            return None

        # Create DataFrame with all stocks
        df = pd.DataFrame(data)

        # Drop any columns with too many NaN values
        df = df.dropna(thresh=len(df) * 0.7, axis=1)

        # Forward fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    except Exception as e:
        console.print(f"[red]Error fetching data: {str(e)}[/red]")
        return None


def calculate_correlation_matrix(price_data):
    """Calculate correlation matrix from price data"""
    try:
        # Calculate daily returns
        returns = price_data.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        return corr_matrix, returns

    except Exception as e:
        console.print(f"[red]Error calculating correlations: {str(e)}[/red]")
        return None, None


def calculate_beta(stock_returns, market_returns):
    """Calculate beta (volatility vs market)"""
    try:
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)

        beta = covariance / market_variance if market_variance != 0 else 1.0

        return beta

    except Exception as e:
        return 1.0


def create_header(symbols):
    """Create header"""
    header = Text()
    header.append("CORRELATION MATRIX\n\n", style="bold white")
    header.append(f"Analyzing {len(symbols)} stocks", style="white")
    header.append(" │ ", style="bright_black")
    header.append("1-Year Historical Data", style="white")
    header.append(" │ ", style="bright_black")
    header.append(datetime.now().strftime('%B %d, %Y'), style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_correlation_table(corr_matrix):
    """Create correlation heatmap table"""
    symbols = corr_matrix.columns.tolist()

    table = Table(
        title="CORRELATION HEATMAP",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    # Add columns
    table.add_column("Symbol", style="white", width=8)
    for symbol in symbols:
        table.add_column(symbol, justify="center", width=6)

    # Add rows
    for i, symbol1 in enumerate(symbols):
        row_data = [symbol1]

        for j, symbol2 in enumerate(symbols):
            corr = corr_matrix.iloc[i, j]

            # Color code correlation
            if i == j:
                # Diagonal (self-correlation)
                cell = f"[white]1.00[/white]"
            elif corr > 0.7:
                # Strong positive correlation
                cell = f"[bright_green]{corr:.2f}[/bright_green]"
            elif corr > 0.3:
                # Moderate positive correlation
                cell = f"[green]{corr:.2f}[/green]"
            elif corr > -0.3:
                # Low correlation
                cell = f"[white]{corr:.2f}[/white]"
            elif corr > -0.7:
                # Moderate negative correlation
                cell = f"[yellow]{corr:.2f}[/yellow]"
            else:
                # Strong negative correlation (hedge)
                cell = f"[red]{corr:.2f}[/red]"

            row_data.append(cell)

        table.add_row(*row_data)

    return Panel(
        table,
        subtitle="[bright_black]Green = Positive | Yellow/Red = Negative (Hedge)[/bright_black]",
        border_style=THEME['border'],
        style=THEME['panel_bg']
    )


def create_beta_table(returns, symbols):
    """Create beta analysis table"""
    # Get S&P 500 data for beta calculation
    try:
        spy = yf.Ticker('^GSPC')
        spy_hist = spy.history(period='1y')
        spy_returns = spy_hist['Close'].pct_change().dropna()

        # Align dates
        common_dates = returns.index.intersection(spy_returns.index)
        returns_aligned = returns.loc[common_dates]
        spy_returns_aligned = spy_returns.loc[common_dates]

        table = Table(
            title="BETA ANALYSIS (vs S&P 500)",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style=f"bold white {THEME['header_bg']}",
            border_style=THEME['border'],
            row_styles=[THEME['row_even'], THEME['row_odd']],
            padding=(0, 1)
        )

        table.add_column("Symbol", style="white", width=10)
        table.add_column("Beta", justify="right", width=10)
        table.add_column("Volatility", justify="right", width=12)
        table.add_column("Interpretation", style="white", width=30)

        betas = []
        for symbol in symbols:
            if symbol in returns_aligned.columns:
                stock_returns = returns_aligned[symbol].dropna()
                beta = calculate_beta(stock_returns.values, spy_returns_aligned.values)

                # Calculate annualized volatility
                volatility = stock_returns.std() * np.sqrt(252) * 100

                # Interpretation
                if beta > 1.3:
                    interpretation = "High volatility (aggressive)"
                    beta_color = "yellow"
                elif beta > 1.0:
                    interpretation = "More volatile than market"
                    beta_color = "white"
                elif beta > 0.7:
                    interpretation = "Less volatile than market"
                    beta_color = "green"
                else:
                    interpretation = "Defensive/low volatility"
                    beta_color = "bright_green"

                betas.append({
                    'symbol': symbol,
                    'beta': beta,
                    'volatility': volatility,
                    'interpretation': interpretation,
                    'color': beta_color
                })

        # Sort by beta (highest first)
        betas.sort(key=lambda x: x['beta'], reverse=True)

        for item in betas:
            table.add_row(
                item['symbol'],
                f"[{item['color']}]{item['beta']:.2f}[/{item['color']}]",
                f"{item['volatility']:.1f}%",
                item['interpretation']
            )

        return Panel(
            table,
            subtitle="[bright_black]Beta > 1.0 = More volatile | Beta < 1.0 = Less volatile[/bright_black]",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )

    except Exception as e:
        return Panel(
            f"[yellow]Could not calculate beta: {str(e)}[/yellow]",
            title="[bold white]BETA ANALYSIS[/bold white]",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )


def create_hedge_finder_table(corr_matrix):
    """Find potential hedges (negatively correlated stocks)"""
    hedges = []

    symbols = corr_matrix.columns.tolist()
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i < j:  # Avoid duplicates
                corr = corr_matrix.iloc[i, j]
                if corr < 0:  # Negative correlation
                    hedges.append({
                        'pair': f"{symbol1} / {symbol2}",
                        'correlation': corr
                    })

    # Sort by most negative correlation
    hedges.sort(key=lambda x: x['correlation'])

    if not hedges:
        return Panel(
            "[white]No negatively correlated pairs found\n\n"
            "[bright_black]Tip: Negatively correlated stocks move in opposite directions,\n"
            "providing portfolio protection (hedging)[/bright_black]",
            title="[bold white]HEDGE OPPORTUNITIES[/bold white]",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )

    table = Table(
        title=f"HEDGE OPPORTUNITIES ({len(hedges)} pairs)",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Stock Pair", style="white", width=20)
    table.add_column("Correlation", justify="right", width=15)
    table.add_column("Hedge Quality", style="white", width=20)

    for hedge in hedges[:10]:  # Top 10 hedges
        corr = hedge['correlation']

        if corr < -0.5:
            quality = "Strong Hedge"
            color = "bright_green"
        elif corr < -0.3:
            quality = "Moderate Hedge"
            color = "green"
        else:
            quality = "Weak Hedge"
            color = "yellow"

        table.add_row(
            hedge['pair'],
            f"[{color}]{corr:.2f}[/{color}]",
            f"[{color}]{quality}[/{color}]"
        )

    return Panel(
        table,
        subtitle="[bright_black]Negative correlation = Stocks move in opposite directions[/bright_black]",
        border_style=THEME['border'],
        style=THEME['panel_bg']
    )


def create_diversification_score_panel(corr_matrix):
    """Calculate portfolio diversification score"""
    try:
        # Calculate average correlation (excluding diagonal)
        n = len(corr_matrix)
        total_corr = 0
        count = 0

        for i in range(n):
            for j in range(i+1, n):
                total_corr += abs(corr_matrix.iloc[i, j])
                count += 1

        avg_correlation = total_corr / count if count > 0 else 0

        # Diversification score (lower avg correlation = better diversification)
        diversification_score = (1 - avg_correlation) * 100

        # Interpretation
        if diversification_score > 70:
            rating = "EXCELLENT"
            color = "bright_green"
            recommendation = "Well diversified portfolio"
        elif diversification_score > 50:
            rating = "GOOD"
            color = "green"
            recommendation = "Decent diversification"
        elif diversification_score > 30:
            rating = "MODERATE"
            color = "yellow"
            recommendation = "Some concentration risk"
        else:
            rating = "POOR"
            color = "red"
            recommendation = "High concentration risk - stocks move together"

        text = Text()
        text.append("PORTFOLIO DIVERSIFICATION\n\n", style="bold white")

        text.append(f"Diversification Score: ", style="white")
        text.append(f"{diversification_score:.1f}/100\n", style=f"bold {color}")

        text.append(f"Rating: ", style="white")
        text.append(f"{rating}\n\n", style=f"bold {color}")

        text.append(f"Average Correlation: {avg_correlation:.2f}\n", style="white")
        text.append(f"Number of Stocks: {n}\n\n", style="white")

        text.append("Recommendation:\n", style="bold white")
        text.append(f"{recommendation}\n\n", style="white")

        text.append("Tips:\n", style="bold white")
        text.append("• Lower correlation = Better diversification\n", style="bright_black")
        text.append("• Aim for correlation < 0.7 between stocks\n", style="bright_black")
        text.append("• Mix uncorrelated sectors for safety\n", style="bright_black")

        return Panel(
            text,
            title="[bold white]DIVERSIFICATION ANALYSIS[/bold white]",
            border_style=THEME['border'],
            box=box.SQUARE,
            style=THEME['panel_bg']
        )

    except Exception as e:
        return Panel(
            f"[yellow]Could not calculate diversification score[/yellow]",
            title="[bold white]DIVERSIFICATION ANALYSIS[/bold white]",
            border_style=THEME['border'],
            style=THEME['panel_bg']
        )


def main():
    """Main function"""
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python CORRELATION_MATRIX.py AAPL MSFT GOOGL NVDA ...[/yellow]")
        console.print("[white]Provide 2-20 stock symbols to analyze[/white]")
        console.print("\n[bright_black]Example: python CORRELATION_MATRIX.py AAPL MSFT GOOGL TSLA AMZN[/bright_black]\n")
        sys.exit(1)

    symbols = [s.upper() for s in sys.argv[1:]]

    if len(symbols) < 2:
        console.print("[red]Error: Need at least 2 stocks for correlation analysis[/red]")
        sys.exit(1)

    if len(symbols) > 20:
        console.print("[yellow]Warning: Too many symbols. Using first 20...[/yellow]")
        symbols = symbols[:20]

    console.clear()
    console.print(create_header(symbols))
    console.print()

    # Fetch data
    console.print(f"[white]Fetching 1-year data for {len(symbols)} stocks...[/white]")
    price_data = get_stock_data(symbols, period='1y')

    if price_data is None or price_data.empty:
        console.print("[red]Error: Could not fetch data[/red]")
        sys.exit(1)

    # Filter symbols that were successfully fetched
    symbols = price_data.columns.tolist()

    console.clear()
    console.print(create_header(symbols))
    console.print()

    # Calculate correlations
    corr_matrix, returns = calculate_correlation_matrix(price_data)

    if corr_matrix is None:
        console.print("[red]Error: Could not calculate correlations[/red]")
        sys.exit(1)

    # Display correlation matrix
    console.print(create_correlation_table(corr_matrix))
    console.print()

    # Display beta analysis and diversification score side by side
    console.print(Columns([
        create_beta_table(returns, symbols),
        create_diversification_score_panel(corr_matrix)
    ]))
    console.print()

    # Display hedge opportunities
    console.print(create_hedge_finder_table(corr_matrix))
    console.print()

    # Footer
    footer = Panel(
        "[bright_black]Correlation: 1.0 = Perfect positive | 0.0 = No relationship | -1.0 = Perfect negative (hedge)\n"
        "Beta: Volatility vs S&P 500 | >1.0 = More volatile | <1.0 = Less volatile | Diversification: Lower correlation = Better[/bright_black]",
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
