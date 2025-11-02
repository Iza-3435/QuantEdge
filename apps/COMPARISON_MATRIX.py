"""
COMPARISON MATRIX
Compare 5-10 stocks side-by-side with highlighting of best/worst

Perfect for:
- Screening watchlist
- Comparing sector peers
- Finding best value/growth play
- Quick portfolio review
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import sys
import warnings
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_cache import DataCache, fetch_parallel
from src.ui_enhancements import create_enhanced_sparkline, get_performance_color

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


def create_sparkline(prices: List[float], width: int = 10) -> str:
    """
    Create enhanced 30-day ASCII sparkline from price list.

    Args:
        prices: List of price values
        width: Number of characters for sparkline display

    Returns:
        Formatted sparkline string with color coding
    """
    if not prices or len(prices) < 2:
        return "━━━━━━━━━━"

    # Remove any NaN values
    prices = [p for p in prices if not pd.isna(p)]
    if len(prices) < 2:
        return "━━━━━━━━━━"

    # Use enhanced sparkline from ui_enhancements
    return create_enhanced_sparkline(prices, width=width, show_trend=False)


def fetch_single_comparison(symbol: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Fetch comparison data for a single symbol (parallel execution).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Tuple of (symbol, data_dict) or (symbol, None) if fetch fails
    """
    # Check cache first
    cache_key = f"comparison_{symbol}"
    cached = DataCache.get(cache_key)
    if cached:
        return symbol, cached

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1y')

        if hist.empty or not info:
            return symbol, None

        current_price = hist['Close'].iloc[-1]
        returns_1y = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

        # Calculate other metrics
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_current = rsi.iloc[-1] if not rsi.empty else 50

        # Sparkline
        sparkline_prices = hist['Close'].tail(30).tolist()

        result = {
            'name': info.get('shortName', symbol)[:25],
            'sector': info.get('sector', 'N/A')[:20],
            'price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg': info.get('pegRatio', 0),
            'pb': info.get('priceToBook', 0),
            'ps': info.get('priceToSalesTrailing12Months', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
            'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
            'debt_equity': info.get('debtToEquity', 0),
            'current_ratio': info.get('currentRatio', 0),
            'beta': info.get('beta', 0),
            'volatility': volatility,
            'rsi': rsi_current,
            'returns_1y': returns_1y,
            'returns_1m': ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) > 21 else 0,
            'returns_3m': ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63] * 100) if len(hist) > 63 else 0,
            'insider_ownership': info.get('heldPercentInsiders', 0) * 100 if info.get('heldPercentInsiders') else 0,
            'institutional_ownership': info.get('heldPercentInstitutions', 0) * 100 if info.get('heldPercentInstitutions') else 0,
            'short_interest': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0,
            'avg_volume': info.get('averageVolume', 0),
            'free_cash_flow': info.get('freeCashflow', 0),
            'operating_cash_flow': info.get('operatingCashflow', 0),
            'target_low': info.get('targetLowPrice', 0),
            'target_mean': info.get('targetMeanPrice', 0),
            'target_high': info.get('targetHighPrice', 0),
            'recommendation': info.get('recommendationKey', 'N/A').upper(),
            'sparkline_prices': sparkline_prices,
        }

        # Cache result
        DataCache.set(cache_key, result)
        return symbol, result

    except Exception:
        return symbol, None


def fetch_comparison_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch comparison data for all symbols in parallel.

    Args:
        symbols: List of stock ticker symbols

    Returns:
        Dictionary mapping symbols to their comparison data
    """
    data: Dict[str, Dict[str, Any]] = {}

    # Use parallel fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_comparison, symbol): symbol
            for symbol in symbols
        }

        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                sym, result = future.result()
                if result:
                    data[sym] = result
            except Exception:
                continue

    return data


def fetch_comparison_data_OLD_SLOW(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    OLD SERIAL VERSION - KEPT FOR REFERENCE.

    Args:
        symbols: List of stock ticker symbols

    Returns:
        Dictionary mapping symbols to their comparison data
    """
    data: Dict[str, Dict[str, Any]] = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1y')

            if hist.empty or not info:
                continue

            current_price = hist['Close'].iloc[-1]
            returns_1y = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

            # Get last 30 days for sparkline
            sparkline_prices = hist['Close'].iloc[-30:].tolist() if len(hist) >= 30 else hist['Close'].tolist()

            data[symbol] = {
                # Price & Valuation
                'price': current_price,
                'sparkline_prices': sparkline_prices,
                'pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg': info.get('pegRatio', 0),
                'pb': info.get('priceToBook', 0),
                'ps': info.get('priceToSalesTrailing12Months', 0),
                'market_cap': info.get('marketCap', 0),

                # Growth & Profitability
                'revenue_growth': info.get('revenueGrowth', 0) * 100,
                'earnings_growth': info.get('earningsGrowth', 0) * 100,
                'profit_margin': info.get('profitMargins', 0) * 100,
                'roe': info.get('returnOnEquity', 0) * 100,

                # Cash Flow & Financial Health
                'free_cash_flow': info.get('freeCashflow', 0),
                'operating_cash_flow': info.get('operatingCashflow', 0),
                'current_ratio': info.get('currentRatio', 0),
                'debt_equity': info.get('debtToEquity', 0),

                # Dividend
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,

                # Ownership & Sentiment
                'insider_ownership': info.get('heldPercentInsiders', 0) * 100 if info.get('heldPercentInsiders') else 0,
                'institutional_ownership': info.get('heldPercentInstitutions', 0) * 100 if info.get('heldPercentInstitutions') else 0,
                'short_interest': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0,

                # Performance & Targets
                'returns_1y': returns_1y,
                'target_low': info.get('targetLowPrice', 0),
                'target_mean': info.get('targetMeanPrice', 0),
                'target_high': info.get('targetHighPrice', 0),
                'recommendation': info.get('recommendationKey', 'N/A').upper(),

                # Volume
                'avg_volume': info.get('averageVolume', 0),
            }

        except Exception as e:
            console.print(f"[bright_black]Warning: Could not fetch {symbol}: {str(e)}[/bright_black]")
            continue

    return data


def get_best_worst(values: Dict[str, float], higher_is_better: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify best and worst performers from values.

    Args:
        values: Dictionary mapping symbols to their metric values
        higher_is_better: If True, higher values are better; if False, lower values are better

    Returns:
        Tuple of (best_symbol, worst_symbol) or (None, None) if no valid values
    """
    valid_values = [(k, v) for k, v in values.items() if v is not None and v != 0]

    if not valid_values:
        return None, None

    if higher_is_better:
        best = max(valid_values, key=lambda x: x[1])
        worst = min(valid_values, key=lambda x: x[1])
    else:
        best = min(valid_values, key=lambda x: x[1])
        worst = max(valid_values, key=lambda x: x[1])

    return best[0], worst[0]


def create_comparison_table(
    data: Dict[str, Dict[str, Any]],
    metric_name: str,
    key: str,
    higher_is_better: bool = True,
    format_type: str = 'number'
) -> Optional[Table]:
    """
    Create comparison table for a single metric across stocks.

    Args:
        data: Dictionary of stock data
        metric_name: Display name for the metric
        key: Data key to extract from stock data
        higher_is_better: Whether higher values are better
        format_type: Format type (number, percent, currency, billions)

    Returns:
        Rich Table object or None if no data
    """
    if not data:
        return None

    # Get values
    values = {symbol: stock_data.get(key, 0) for symbol, stock_data in data.items()}

    # Find best/worst
    best_symbol, worst_symbol = get_best_worst(values, higher_is_better)

    # Create table
    table = Table(
        title=f"[bold white on blue] {metric_name} [/bold white on blue]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style="grey35"
    )

    table.add_column("Stock", style="bold white", width=8)
    table.add_column("Value", justify="right", width=12)
    table.add_column("Rating", justify="center", width=8)

    for symbol in sorted(data.keys()):
        value = values[symbol]

        # Format value
        if format_type == 'percent':
            value_str = f"{value:+.1f}%" if value != 0 else "N/A"
        elif format_type == 'currency':
            value_str = f"${value:,.0f}" if value != 0 else "N/A"
        elif format_type == 'billions':
            value_str = f"${value/1e9:.1f}B" if value != 0 else "N/A"
        else:
            value_str = f"{value:.2f}" if value != 0 else "N/A"

        # Color and rating (professional, no emojis)
        if symbol == best_symbol:
            style = "bold bright_green"
            rating = "BEST"
        elif symbol == worst_symbol:
            style = "dim"
            rating = "WORST"
        else:
            style = "white"
            rating = ""

        table.add_row(symbol, value_str, rating, style=style)

    return table


def create_summary_table(data: Dict[str, Dict[str, Any]]) -> Optional[Table]:
    """
    Create comprehensive summary comparison table across all metrics.

    Args:
        data: Dictionary of stock data for all symbols

    Returns:
        Rich Table object with complete comparison matrix or None if no data
    """
    if not data:
        return None

    table = Table(
        title="COMPLETE COMPARISON MATRIX",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        title_style="bold white",
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Metric", style="bold white", width=18)

    # Add column for each stock
    for symbol in sorted(data.keys()):
        table.add_column(symbol, justify="right", width=12, style="white")

    table.add_column("Winner", style="bright_green", width=8)

    # Define metrics
    metrics = [
        # Price & Valuation
        ("Price", 'price', False, 'currency'),
        ("Trend (30D)", 'sparkline_prices', False, 'sparkline'),  # Special sparkline row
        ("Market Cap", 'market_cap', False, 'billions'),
        ("P/E Ratio", 'pe', False, 'number'),
        ("Forward P/E", 'forward_pe', False, 'number'),
        ("PEG Ratio", 'peg', False, 'number'),
        ("P/B Ratio", 'pb', False, 'number'),
        ("P/S Ratio", 'ps', False, 'number'),

        # Growth & Profitability
        ("Revenue Growth", 'revenue_growth', True, 'percent'),
        ("Earnings Growth", 'earnings_growth', True, 'percent'),
        ("Profit Margin", 'profit_margin', True, 'percent'),
        ("ROE", 'roe', True, 'percent'),

        # Cash Flow & Financial Health
        ("Free Cash Flow", 'free_cash_flow', True, 'billions'),
        ("Operating CF", 'operating_cash_flow', True, 'billions'),
        ("Current Ratio", 'current_ratio', True, 'number'),
        ("Debt/Equity", 'debt_equity', False, 'number'),

        # Dividend
        ("Dividend Yield", 'dividend_yield', True, 'percent'),

        # Ownership & Sentiment
        ("Insider Own %", 'insider_ownership', True, 'percent'),
        ("Institutional %", 'institutional_ownership', True, 'percent'),
        ("Short Interest %", 'short_interest', False, 'percent'),

        # Performance
        ("1Y Return", 'returns_1y', True, 'percent'),
        ("Avg Volume", 'avg_volume', False, 'millions'),

        # Analyst Opinion
        ("Target Low", 'target_low', False, 'currency'),
        ("Target Mean", 'target_mean', False, 'currency'),
        ("Target High", 'target_high', False, 'currency'),
        ("Recommendation", 'recommendation', False, 'text'),
    ]

    for metric_name, key, higher_is_better, format_type in metrics:
        row = [metric_name]

        # Get values for all stocks
        values = {}
        for symbol in sorted(data.keys()):
            value = data[symbol].get(key, 0)
            values[symbol] = value

            # Format value
            if format_type == 'sparkline':
                # Special handling for sparkline - value is a list of prices
                value_str = create_sparkline(value) if isinstance(value, list) else "━━━━━━━━━━"
            elif format_type == 'percent':
                value_str = f"{value:+.1f}%" if value != 0 else "—"
            elif format_type == 'currency':
                value_str = f"${value:.2f}" if value != 0 else "—"
            elif format_type == 'billions':
                value_str = f"${value/1e9:.1f}B" if value != 0 else "—"
            elif format_type == 'millions':
                value_str = f"{value/1e6:.1f}M" if value != 0 else "—"
            elif format_type == 'text':
                value_str = str(value) if value and value != 'N/A' else "—"
            else:
                value_str = f"{value:.2f}" if value != 0 else "—"

            row.append(value_str)

        # Find winner
        best_symbol, _ = get_best_worst(values, higher_is_better)
        row.append(best_symbol if best_symbol else "—")

        table.add_row(*row)

    return table


def create_header(symbols: List[str]) -> Panel:
    """
    Create header panel for comparison matrix.

    Args:
        symbols: List of stock symbols being compared

    Returns:
        Rich Panel with header information
    """
    header = Text()
    header.append("COMPARISON MATRIX\n\n", style="bold white")
    header.append(f"Comparing {len(symbols)} stocks", style="white")
    header.append(" │ ", style="bright_black")
    header.append("Side-by-Side Analysis", style="white")
    header.append(" │ ", style="bright_black")
    header.append(datetime.now().strftime('%B %d, %Y'), style="bright_black")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def create_winner_panel(data: Dict[str, Dict[str, Any]]) -> Optional[Panel]:
    """
    Create overall winner panel with scoring analysis.

    Args:
        data: Dictionary of stock data for all symbols

    Returns:
        Rich Panel showing overall winner and reasons or None if insufficient data
    """
    if not data or len(data) < 2:
        return None

    # Score each stock
    scores: Dict[str, int] = {symbol: 0 for symbol in data.keys()}

    # Metrics to score (metric, higher_is_better, weight)
    scoring_metrics: List[Tuple[str, bool, int]] = [
        ('revenue_growth', True, 2),
        ('earnings_growth', True, 2),
        ('profit_margin', True, 2),
        ('roe', True, 2),
        ('returns_1y', True, 1),
        ('pe', False, 2),
        ('debt_equity', False, 1),
    ]

    for metric_key, higher_is_better, weight in scoring_metrics:
        values = {s: data[s].get(metric_key, 0) for s in data.keys()}
        best_symbol, worst_symbol = get_best_worst(values, higher_is_better)

        if best_symbol:
            scores[best_symbol] += weight

    # Find winner
    winner = max(scores.items(), key=lambda x: x[1])
    winner_symbol = winner[0]
    winner_score = winner[1]

    # Create panel
    text = Text()
    text.append(" OVERALL WINNER: ", style="bold white")
    text.append(f"{winner_symbol}\n\n", style="bold bright_green")

    text.append(f"Score: {winner_score}/12 points\n\n", style="bright_white")

    # Why winner
    text.append("Wins in:\n", style="bold white")

    win_reasons = []
    for metric_key, higher_is_better, weight in scoring_metrics:
        values = {s: data[s].get(metric_key, 0) for s in data.keys()}
        best_symbol, _ = get_best_worst(values, higher_is_better)

        if best_symbol == winner_symbol:
            metric_names = {
                'revenue_growth': 'Revenue Growth',
                'earnings_growth': 'Earnings Growth',
                'profit_margin': 'Profit Margin',
                'roe': 'ROE',
                'returns_1y': '1Y Return',
                'pe': 'Valuation (P/E)',
                'debt_equity': 'Low Debt',
            }
            win_reasons.append(f"• {metric_names.get(metric_key, metric_key)}")

    text.append("\n".join(win_reasons), style="green")

    return Panel(
        text,
        title="ANALYSIS RESULT",
        border_style=THEME['border'],
        box=box.SQUARE,
        padding=(1, 2),
        style=THEME['panel_bg']
    )


def display_comparison_matrix(symbols: List[str]) -> None:
    """
    Display comparison matrix for multiple stocks.

    Args:
        symbols: List of stock ticker symbols to compare
    """
    console.clear()
    console.print(create_header(symbols))
    console.print()

    # Fetch data
    console.print("[cyan]Fetching data...[/cyan]")
    data = fetch_comparison_data(symbols)

    if not data:
        console.print("[red]Error: Could not fetch data for any stocks[/red]\n")
        return

    if len(data) < len(symbols):
        missing = set(symbols) - set(data.keys())
        console.print(f"[yellow]Warning: Could not fetch: {', '.join(missing)}[/yellow]")

    console.clear()
    console.print(create_header(symbols))
    console.print()

    # Display summary table
    console.print(create_summary_table(data))
    console.print()

    # Display winner
    winner_panel = create_winner_panel(data)
    if winner_panel:
        console.print(winner_panel)
        console.print()

    # Footer
    footer = Panel(
        "[bright_black]Comparison Matrix • Green = Best in category • Higher scores = Better investment[/bright_black]",
        box=box.SQUARE,
        border_style=THEME['border'],
        style=THEME['panel_bg']
    )
    console.print(footer)


def main() -> None:
    """
    Main entry point for comparison matrix application.

    Parses command-line arguments and displays stock comparison matrix.
    """
    if len(sys.argv) < 3:
        console.print("\n[yellow]Usage:[/yellow] python COMPARISON_MATRIX.py <SYMBOL1> <SYMBOL2> [SYMBOL3] ...")
        console.print("\n[cyan]Example:[/cyan] python COMPARISON_MATRIX.py AAPL MSFT GOOGL NVDA META")
        console.print("[cyan]         [/cyan] python COMPARISON_MATRIX.py AAPL GOOGL\n")
        console.print("[bright_black]Compare 2-10 stocks side-by-side[/bright_black]\n")
        sys.exit(1)

    symbols = [s.upper() for s in sys.argv[1:]]

    if len(symbols) > 10:
        console.print("[yellow]Warning: Maximum 10 stocks supported. Using first 10.[/yellow]")
        symbols = symbols[:10]

    try:
        display_comparison_matrix(symbols)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
