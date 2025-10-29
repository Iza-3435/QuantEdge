"""Professional Bloomberg-style terminal dashboard."""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from src.retrieval.advanced_pattern_retrieval import AdvancedPatternRetrieval

console = Console()


def create_header():
    """Create Bloomberg-style header."""
    header = Text()
    header.append("█ ", style="bold cyan")
    header.append("AI MARKET INTELLIGENCE", style="bold white")
    header.append(" █", style="bold cyan")
    header.append(f"  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    return Panel(header, style="bold cyan", box=box.HEAVY)


def create_market_data_table(df, symbol):
    """Create market data summary table."""
    table = Table(title=f"[bold cyan]{symbol} Market Data", box=box.ROUNDED, title_style="bold cyan")

    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="white", justify="right", width=15)
    table.add_column("Change", style="white", justify="right", width=15)

    current = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2])
    change = current - prev
    change_pct = (change / prev) * 100

    # Color code change
    change_style = "green" if change >= 0 else "red"
    arrow = "▲" if change >= 0 else "▼"

    table.add_row(
        "Current Price",
        f"${current:.2f}",
        f"[{change_style}]{arrow} ${abs(change):.2f} ({change_pct:+.2f}%)"
    )

    table.add_row("Volume", f"{int(df['Volume'].iloc[-1]):,}", "")
    table.add_row("52W High", f"${float(df['Close'].max()):.2f}", "")
    table.add_row("52W Low", f"${float(df['Close'].min()):.2f}", "")

    if 'SMA_20' in df.columns:
        sma = float(df['SMA_20'].iloc[-1])
        sma_dist = ((current - sma) / sma) * 100
        table.add_row("SMA(20)", f"${sma:.2f}", f"{sma_dist:+.2f}%")

    if 'RSI' in df.columns:
        rsi = float(df['RSI'].iloc[-1])
        rsi_style = "red" if rsi > 70 else ("green" if rsi < 30 else "yellow")
        table.add_row("RSI(14)", f"[{rsi_style}]{rsi:.1f}", "")

    return table


def create_prediction_panel(prediction):
    """Create prediction results panel."""
    table = Table(box=box.HEAVY, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="bold cyan", width=25)
    table.add_column("Value", style="bold white", width=20)

    # Prediction
    pred_return = prediction.predicted_return * 100
    pred_style = "bold green" if pred_return > 0 else "bold red"
    arrow = "📈" if pred_return > 0 else "📉"

    table.add_row(
        "Predicted Return",
        f"[{pred_style}]{arrow} {pred_return:+.2f}%"
    )

    # Confidence
    conf = prediction.confidence
    conf_style = "green" if conf > 0.7 else ("yellow" if conf > 0.4 else "red")
    conf_bar = "█" * int(conf * 20)
    table.add_row(
        "Confidence",
        f"[{conf_style}]{conf_bar} {conf:.2f}"
    )

    # Profit probability
    prob = prediction.profit_probability * 100
    prob_style = "green" if prob > 60 else ("yellow" if prob > 40 else "red")
    table.add_row(
        "Profit Probability",
        f"[{prob_style}]{prob:.1f}%"
    )

    # Uncertainty interval
    lower = prediction.uncertainty_lower * 100
    upper = prediction.uncertainty_upper * 100
    table.add_row(
        "95% CI",
        f"[dim][{lower:.2f}%, {upper:.2f}%]"
    )

    return Panel(table, title="[bold]AI PREDICTION", border_style="cyan", box=box.HEAVY)


def create_similar_patterns_table(prediction):
    """Create similar patterns table."""
    table = Table(
        title="[bold cyan]Similar Historical Patterns",
        box=box.ROUNDED,
        title_style="bold cyan"
    )

    table.add_column("Date", style="cyan", width=12)
    table.add_column("Return", justify="right", width=12)
    table.add_column("Similarity", justify="right", width=12)
    table.add_column("Match", justify="center", width=10)

    for ex in prediction.examples:
        ret = ex['outcome_return'] * 100
        ret_style = "green" if ret > 0 else "red"
        sim = ex['similarity']
        sim_bar = "█" * int(sim * 10)

        table.add_row(
            ex['date'],
            f"[{ret_style}]{ret:+.2f}%",
            f"{sim:.3f}",
            sim_bar
        )

    return table


def create_technical_indicators_table(df):
    """Create technical indicators table."""
    table = Table(
        title="[bold cyan]Technical Indicators",
        box=box.ROUNDED,
        title_style="bold cyan"
    )

    table.add_column("Indicator", style="cyan", width=15)
    table.add_column("Value", style="white", justify="right", width=15)
    table.add_column("Signal", justify="center", width=12)

    # RSI
    if 'RSI' in df.columns:
        rsi = float(df['RSI'].iloc[-1])
        if rsi > 70:
            signal = "[red]OVERBOUGHT"
        elif rsi < 30:
            signal = "[green]OVERSOLD"
        else:
            signal = "[yellow]NEUTRAL"
        table.add_row("RSI(14)", f"{rsi:.1f}", signal)

    # SMA
    if 'SMA_20' in df.columns:
        price = float(df['Close'].iloc[-1])
        sma = float(df['SMA_20'].iloc[-1])
        if price > sma:
            signal = "[green]BULLISH"
        else:
            signal = "[red]BEARISH"
        table.add_row("SMA(20)", f"${sma:.2f}", signal)

    # Volatility
    returns = df['Close'].pct_change().dropna()
    vol = float(returns.std() * (252 ** 0.5) * 100)
    if vol > 30:
        signal = "[red]HIGH"
    elif vol < 15:
        signal = "[green]LOW"
    else:
        signal = "[yellow]MODERATE"
    table.add_row("Volatility", f"{vol:.1f}%", signal)

    return table


def main():
    console.clear()

    # Header
    console.print(create_header())
    console.print()

    # Loading with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:

        task1 = progress.add_task("[cyan]Fetching market data...", total=100)

        # Download data
        df = yf.download("AAPL", period="5y", progress=False)
        progress.update(task1, advance=33)

        # Add indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df.dropna(inplace=True)
        progress.update(task1, advance=33)

        task2 = progress.add_task("[cyan]Building AI model...", total=100)

        # Build model
        retriever = AdvancedPatternRetrieval()
        retriever.build_pattern_database(df, forward_periods=5)
        progress.update(task2, advance=50)

        # Make prediction
        current_idx = len(df) - 1
        current_features = retriever.extract_pattern_features(df, current_idx)
        prediction = retriever.predict_with_uncertainty(current_features, top_k=10)
        progress.update(task2, advance=50)

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3),
        Layout(name="main"),
        Layout(name="bottom", size=15)
    )

    layout["top"].update(create_header())

    # Main section
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    layout["left"].split_column(
        Layout(create_market_data_table(df, "AAPL"), size=12),
        Layout(create_technical_indicators_table(df))
    )

    layout["right"].split_column(
        Layout(create_prediction_panel(prediction), size=12),
        Layout(create_similar_patterns_table(prediction))
    )

    # Bottom info
    info = Text()
    info.append(f"📊 Patterns Analyzed: ", style="dim")
    info.append(f"{len(retriever.patterns)}", style="bold cyan")
    info.append(f"  |  🎯 Model Confidence: ", style="dim")
    conf_color = "green" if prediction.confidence > 0.7 else ("yellow" if prediction.confidence > 0.4 else "red")
    info.append(f"{prediction.confidence:.2%}", style=f"bold {conf_color}")
    info.append(f"  |  📈 Data Period: ", style="dim")
    info.append(f"{df.index[0].date()} to {df.index[-1].date()}", style="bold white")

    layout["bottom"].update(Panel(info, title="[bold]System Status", border_style="dim"))

    # Display
    console.print(layout)
    console.print()

    # Footer
    footer = Text()
    footer.append("Press Ctrl+C to exit  |  ", style="dim")
    footer.append("Powered by AI Market Intelligence", style="bold cyan")
    console.print(footer, justify="center")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed.")
    except Exception as e:
        console.print(f"\n[red]Error: {e}")
