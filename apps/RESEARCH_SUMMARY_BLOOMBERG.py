"""
DEEP DIVE RESEARCH SUMMARY
Comprehensive one-page research summary for investment analysis

This tool provides:
- Business quality metrics
- Growth analysis
- Valuation assessment
- Catalysts and risks
- Sentiment analysis
- ML prediction
- Clear investment conclusion
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich import box
from rich.text import Text
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import ML predictor
try:
    from src.models.ensemble_predictor import EnsemblePredictor
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

console = Console()

# Bloomberg-style minimal colors: white text, green/red accents only
COLORS = {
    'excellent': 'bold green',
    'good': 'green',
    'neutral': 'white',
    'warning': 'red',
    'critical': 'bold red',
    'info': 'white',
    'dim': 'bright_black'
}


def get_comprehensive_data(symbol):
    """Fetch all data needed for deep dive analysis"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1y')

        if hist.empty or not info:
            return None

        # Calculate returns
        current_price = hist['Close'].iloc[-1]
        returns_1y = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)

        # Volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Technical indicators
        rsi = calculate_rsi(hist)

        # Get news sentiment
        try:
            news = ticker.news[:10]
            news_count = len(news)
        except:
            news_count = 0

        data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': current_price,

            # Business Quality
            'profit_margin': info.get('profitMargins', 0) * 100,
            'roe': info.get('returnOnEquity', 0) * 100,
            'roa': info.get('returnOnAssets', 0) * 100,
            'debt_equity': info.get('debtToEquity', 0),
            'current_ratio': info.get('currentRatio', 0),
            'operating_margin': info.get('operatingMargins', 0) * 100,

            # Growth
            'revenue_growth': info.get('revenueGrowth', 0) * 100,
            'earnings_growth': info.get('earningsGrowth', 0) * 100,
            'revenue_per_share': info.get('revenuePerShare', 0),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0) * 100,

            # Valuation
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
            'ev_ebitda': info.get('enterpriseToEbitda', 0),
            'market_cap': info.get('marketCap', 0),

            # Analyst Info
            'target_price': info.get('targetMeanPrice', 0),
            'recommendation': info.get('recommendationKey', 'N/A').upper(),
            'num_analysts': info.get('numberOfAnalystOpinions', 0),

            # Technical
            'returns_1y': returns_1y,
            'volatility': volatility,
            'rsi': rsi,
            'beta': info.get('beta', 0),

            # Dividends
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,

            # Other
            'news_count': news_count,
            'average_volume': info.get('averageVolume', 0),
        }

        return data

    except Exception as e:
        console.print(f"[red]Error fetching data: {str(e)}[/red]")
        return None


def calculate_rsi(df, period=14):
    """Calculate RSI"""
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except:
        return 50


def get_quality_rating(value, metric_type):
    """Get quality rating for a metric"""
    ratings = {
        'profit_margin': [(25, 'Excellent'), (15, 'Good'), (10, 'Fair'), (5, 'Poor')],
        'roe': [(20, 'Outstanding'), (15, 'Excellent'), (10, 'Good'), (5, 'Fair')],
        'debt_equity': [(50, 'Excellent'), (100, 'Good'), (150, 'Watch'), (200, 'High')],
        'current_ratio': [(2.0, 'Excellent'), (1.5, 'Good'), (1.0, 'Fair'), (0.8, 'Poor')],
        'revenue_growth': [(20, 'Excellent'), (10, 'Good'), (5, 'Fair'), (0, 'Slow')],
        'pe_ratio': [(20, 'Cheap'), (30, 'Fair'), (40, 'Expensive'), (50, 'Very Expensive')],
    }

    if metric_type not in ratings:
        return 'N/A'

    thresholds = ratings[metric_type]

    # For debt, lower is better (reversed logic)
    if metric_type == 'debt_equity':
        for threshold, rating in thresholds:
            if value <= threshold:
                return rating
        return 'Very High'

    # For P/E, lower is better
    elif metric_type == 'pe_ratio':
        for threshold, rating in thresholds:
            if value <= threshold:
                return rating
        return 'Overvalued'

    # For most metrics, higher is better
    else:
        for threshold, rating in reversed(thresholds):
            if value >= threshold:
                return rating
        return 'Poor'


def get_color_for_rating(rating):
    """Get color based on rating"""
    if rating in ['Excellent', 'Outstanding', 'Cheap']:
        return COLORS['excellent']
    elif rating in ['Good', 'Fair']:
        return COLORS['good']
    elif rating in ['Watch', 'Expensive', 'Poor']:
        return COLORS['warning']
    else:
        return COLORS['critical']


def create_business_quality_panel(data):
    """Create business quality metrics panel"""
    text = Text()

    # Profit Margin
    pm = data['profit_margin']
    pm_rating = get_quality_rating(pm, 'profit_margin')
    text.append("Profit Margin: ", style="white")
    text.append(f"{pm:.1f}%", style="bold white")
    text.append(f" ({pm_rating})\n", style=get_color_for_rating(pm_rating))

    # ROE
    roe = data['roe']
    roe_rating = get_quality_rating(roe, 'roe')
    text.append("ROE: ", style="white")
    text.append(f"{roe:.1f}%", style="bold white")
    text.append(f" ({roe_rating})\n", style=get_color_for_rating(roe_rating))

    # Debt/Equity
    de = data['debt_equity']
    de_rating = get_quality_rating(de, 'debt_equity')
    text.append("Debt/Equity: ", style="white")
    text.append(f"{de:.0f}", style="bold white")
    text.append(f" ({de_rating})\n", style=get_color_for_rating(de_rating))

    # Operating Margin
    om = data['operating_margin']
    text.append("Operating Margin: ", style="white")
    text.append(f"{om:.1f}%\n", style="bold white")

    # Current Ratio
    cr = data['current_ratio']
    cr_rating = get_quality_rating(cr, 'current_ratio')
    text.append("Current Ratio: ", style="white")
    text.append(f"{cr:.2f}", style="bold white")
    text.append(f" ({cr_rating})", style=get_color_for_rating(cr_rating))

    return Panel(
        text,
        title="[bold white]BUSINESS QUALITY[/bold white]",
        border_style="white",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_growth_panel(data):
    """Create growth metrics panel"""
    text = Text()

    # Revenue Growth
    rev_growth = data['revenue_growth']
    rev_rating = get_quality_rating(rev_growth, 'revenue_growth')
    text.append("Revenue Growth: ", style="white")
    text.append(f"{rev_growth:+.1f}%", style="bold white")
    text.append(f" ({rev_rating})\n", style=get_color_for_rating(rev_rating))

    # Earnings Growth
    earn_growth = data['earnings_growth']
    text.append("Earnings Growth: ", style="white")
    text.append(f"{earn_growth:+.1f}%\n", style="bold white")

    # Quarterly Growth
    q_growth = data['earnings_quarterly_growth']
    text.append("Q. Earnings Growth: ", style="white")
    text.append(f"{q_growth:+.1f}%\n", style="bold white")

    # 1Y Return
    ret_1y = data['returns_1y']
    text.append("1Y Stock Return: ", style="white")
    color = COLORS['good'] if ret_1y > 0 else COLORS['warning']
    text.append(f"{ret_1y:+.1f}%", style=color)

    return Panel(
        text,
        title="[bold white]GROWTH[/bold white]",
        border_style="white",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_valuation_panel(data):
    """Create valuation metrics panel"""
    text = Text()

    # P/E Ratio
    pe = data['pe_ratio']
    pe_rating = get_quality_rating(pe, 'pe_ratio')
    text.append("P/E Ratio: ", style="white")
    text.append(f"{pe:.1f}", style="bold white")
    text.append(f" ({pe_rating})\n", style=get_color_for_rating(pe_rating))

    # Forward P/E
    fpe = data['forward_pe']
    text.append("Forward P/E: ", style="white")
    text.append(f"{fpe:.1f}\n", style="bold white")

    # PEG Ratio
    peg = data['peg_ratio']
    text.append("PEG Ratio: ", style="white")
    peg_color = COLORS['good'] if peg < 1.5 else COLORS['warning']
    text.append(f"{peg:.2f}\n", style=peg_color)

    # P/B and P/S
    text.append("P/B Ratio: ", style="white")
    text.append(f"{data['pb_ratio']:.2f}\n", style="white")

    text.append("P/S Ratio: ", style="white")
    text.append(f"{data['ps_ratio']:.2f}", style="white")

    return Panel(
        text,
        title="[bold white]VALUATION[/bold white]",
        border_style="white",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_analyst_panel(data):
    """Create analyst sentiment panel"""
    text = Text()

    # Target Price
    target = data['target_price']
    current = data['current_price']
    upside = ((target - current) / current * 100) if target > 0 and current > 0 else 0

    text.append("Target Price: ", style="white")
    text.append(f"${target:.2f}\n", style="bold white")

    text.append("Upside/Downside: ", style="white")
    color = COLORS['good'] if upside > 0 else COLORS['warning']
    text.append(f"{upside:+.1f}%\n", style=color)

    # Rating
    text.append("Analyst Rating: ", style="white")
    rating = data['recommendation']
    rating_color = {
        'STRONG_BUY': COLORS['excellent'],
        'BUY': COLORS['good'],
        'HOLD': COLORS['neutral'],
        'SELL': COLORS['warning'],
        'STRONG_SELL': COLORS['critical']
    }.get(rating, 'white')
    text.append(f"{rating}\n", style=rating_color)

    # Number of Analysts
    text.append(f"Analysts Covering: ", style="white")
    text.append(f"{data['num_analysts']}", style="white")

    return Panel(
        text,
        title="[bold white]ANALYSTS[/bold white]",
        border_style="white",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_technical_panel(data):
    """Create technical analysis panel"""
    text = Text()

    # RSI
    rsi = data['rsi']
    text.append("RSI (14): ", style="white")
    if rsi > 70:
        rsi_status = "Overbought"
        rsi_color = COLORS['warning']
    elif rsi < 30:
        rsi_status = "Oversold"
        rsi_color = COLORS['good']
    else:
        rsi_status = "Neutral"
        rsi_color = COLORS['neutral']
    text.append(f"{rsi:.1f}", style="bold white")
    text.append(f" ({rsi_status})\n", style=rsi_color)

    # Volatility
    vol = data['volatility']
    text.append("Volatility: ", style="white")
    text.append(f"{vol:.1f}%\n", style="white")

    # Beta
    beta = data['beta']
    text.append("Beta: ", style="white")
    beta_color = COLORS['warning'] if beta > 1.5 else COLORS['good']
    text.append(f"{beta:.2f}\n", style=beta_color)

    # Trend
    text.append("Trend: ", style="white")
    if rsi > 55:
        text.append("Bullish", style=COLORS['good'])
    elif rsi < 45:
        text.append("Bearish", style=COLORS['warning'])
    else:
        text.append("Neutral", style=COLORS['neutral'])

    return Panel(
        text,
        title="[bold white]TECHNICAL[/bold white]",
        border_style="white",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_risks_panel(data):
    """Create risks and concerns panel"""
    risks = []

    # High valuation
    if data['pe_ratio'] > 40:
        risks.append("High P/E ratio (valuation risk)")

    # High debt
    if data['debt_equity'] > 150:
        risks.append("High debt levels")

    # Overbought
    if data['rsi'] > 70:
        risks.append("Technically overbought")

    # Negative growth
    if data['revenue_growth'] < 0:
        risks.append("Revenue declining")

    # Low margin
    if data['profit_margin'] < 5:
        risks.append("Low profit margins")

    # High volatility
    if data['volatility'] > 40:
        risks.append("High volatility stock")

    if not risks:
        risks.append("No major red flags identified")

    text = Text("\n".join(risks[:6]), style="white")

    return Panel(
        text,
        title="[bold red]RISKS & CONCERNS[/bold red]",
        border_style="red",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_catalysts_panel(data):
    """Create potential catalysts panel"""
    catalysts = []

    # Strong growth
    if data['revenue_growth'] > 15:
        catalysts.append("Strong revenue growth")

    # Undervalued
    if data['pe_ratio'] < 20 and data['pe_ratio'] > 0:
        catalysts.append("Attractive valuation")

    # High quality
    if data['roe'] > 20:
        catalysts.append("Exceptional ROE")

    # Analyst upside
    target = data['target_price']
    current = data['current_price']
    upside = ((target - current) / current * 100) if target > 0 and current > 0 else 0
    if upside > 10:
        catalysts.append(f"{upside:.0f}% analyst upside")

    # Oversold
    if data['rsi'] < 30:
        catalysts.append("Technically oversold (bounce?)")

    # Strong margins
    if data['profit_margin'] > 20:
        catalysts.append("High profit margins")

    if not catalysts:
        catalysts.append("Limited obvious catalysts")

    text = Text("\n".join(catalysts[:6]), style="white")

    return Panel(
        text,
        title="[bold green]CATALYSTS[/bold green]",
        border_style="green",
        box=box.SQUARE,
        padding=(1, 2)
    )


def create_bottom_line_panel(data):
    """Create investment conclusion panel"""
    # Calculate overall score
    score = 0
    max_score = 0

    # Quality (30 points)
    max_score += 30
    if data['profit_margin'] > 20: score += 10
    elif data['profit_margin'] > 10: score += 5
    if data['roe'] > 15: score += 10
    elif data['roe'] > 10: score += 5
    if data['debt_equity'] < 100: score += 10
    elif data['debt_equity'] < 150: score += 5

    # Growth (25 points)
    max_score += 25
    if data['revenue_growth'] > 15: score += 15
    elif data['revenue_growth'] > 5: score += 8
    if data['returns_1y'] > 20: score += 10
    elif data['returns_1y'] > 0: score += 5

    # Valuation (25 points)
    max_score += 25
    if data['pe_ratio'] < 20 and data['pe_ratio'] > 0: score += 15
    elif data['pe_ratio'] < 30: score += 10
    elif data['pe_ratio'] < 40: score += 5
    if data['peg_ratio'] < 1.5 and data['peg_ratio'] > 0: score += 10
    elif data['peg_ratio'] < 2: score += 5

    # Sentiment (20 points)
    max_score += 20
    rating = data['recommendation']
    if rating in ['STRONG_BUY', 'BUY']:
        score += 15
    elif rating == 'HOLD':
        score += 8
    target = data['target_price']
    current = data['current_price']
    upside = ((target - current) / current * 100) if target > 0 and current > 0 else 0
    if upside > 10: score += 5

    # Normalize to 100
    final_score = int((score / max_score) * 100)

    # Determine recommendation
    if final_score >= 75:
        recommendation = "STRONG BUY"
        rec_color = COLORS['excellent']
        summary = "High-quality business with strong growth and reasonable valuation."
    elif final_score >= 60:
        recommendation = "BUY"
        rec_color = COLORS['good']
        summary = "Solid investment opportunity with good fundamentals."
    elif final_score >= 45:
        recommendation = "HOLD"
        rec_color = COLORS['neutral']
        summary = "Mixed signals. Wait for better entry point or catalyst."
    elif final_score >= 30:
        recommendation = "AVOID"
        rec_color = COLORS['warning']
        summary = "Concerns outweigh opportunities. Consider alternatives."
    else:
        recommendation = "SELL"
        rec_color = COLORS['critical']
        summary = "Significant red flags. High risk investment."

    text = Text()
    text.append("INVESTMENT SCORE: ", style="bold white")
    text.append(f"{final_score}/100\n\n", style=rec_color)

    text.append("RECOMMENDATION: ", style="bold white")
    text.append(f"{recommendation}\n\n", style=rec_color)

    text.append("SUMMARY:\n", style="bold white")
    text.append(summary, style="white")

    return Panel(
        text,
        title="[bold white on blue] 💡 BOTTOM LINE [/bold white on blue]",
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(1, 2)
    )


def create_header(data):
    """Create research summary header - Bloomberg minimal style"""
    header = Text()

    # Simple top line
    header.append("━" * 80, style="white")
    header.append("\n")

    # Title - no emoji, simple
    header.append(" RESEARCH SUMMARY | ", style="bold white")
    header.append(f"{data['symbol']}", style="bold white")
    header.append("\n")
    header.append("━" * 80, style="white")
    header.append("\n\n")

    # Company info - clean, minimal
    header.append(f" {data['name']}", style="bold white")
    header.append(" | ", style="bright_black")
    header.append(f"{data['sector']}", style="white")

    # Price with color
    header.append(" | $", style="bright_black")
    price_color = "green" if data.get('returns_1d', 0) >= 0 else "red"
    header.append(f"{data['current_price']:.2f}", style=f"{price_color}")

    # Change
    if 'returns_1d' in data:
        change_symbol = "▲" if data['returns_1d'] >= 0 else "▼"
        header.append(f" {change_symbol}{abs(data['returns_1d']):.1f}%", style=price_color)

    header.append("\n ")
    header.append(f"{datetime.now().strftime('%b %d, %Y %I:%M %p ET')}", style="bright_black")

    return header


def display_research_summary(symbol):
    """Display comprehensive research summary"""
    console.clear()
    console.print(f"\n[cyan]Analyzing {symbol}...[/cyan]\n")

    # Fetch data
    data = get_comprehensive_data(symbol)

    if not data:
        console.print(f"[red]Error: Could not fetch data for {symbol}[/red]\n")
        return

    console.clear()

    # Display header
    console.print(create_header(data))
    console.print()

    # Row 1: Business Quality & Growth
    console.print(Columns([
        create_business_quality_panel(data),
        create_growth_panel(data)
    ]))
    console.print()

    # Row 2: Valuation & Analysts
    console.print(Columns([
        create_valuation_panel(data),
        create_analyst_panel(data)
    ]))
    console.print()

    # Row 3: Technical & Risks
    console.print(Columns([
        create_technical_panel(data),
        create_risks_panel(data)
    ]))
    console.print()

    # Row 4: Catalysts (full width)
    console.print(create_catalysts_panel(data))
    console.print()

    # Bottom Line (full width)
    console.print(create_bottom_line_panel(data))
    console.print()

    # Footer
    footer = Panel(
        "[dim]Research Summary • For informational purposes only • Not investment advice[/dim]",
        box=box.ROUNDED,
        border_style="dim"
    )
    console.print(footer)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        console.print("\n[yellow]Usage:[/yellow] python RESEARCH_SUMMARY.py <SYMBOL>")
        console.print("\n[cyan]Example:[/cyan] python RESEARCH_SUMMARY.py AAPL\n")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    try:
        display_research_summary(symbol)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
