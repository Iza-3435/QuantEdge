#!/usr/bin/env python3
"""
AI Stock Predictor
LSTM price prediction with real-time sentiment analysis
Production-grade ML for institutional investors
"""

import sys
import os
import warnings
from typing import Dict, List, Optional, Any

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box
from rich.columns import Columns

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.lstm_predictor import LSTMPricePredictor
from src.sentiment_analyzer import SentimentAnalyzer

console = Console()

THEME = {
    'header_bg': 'on grey23',
    'row_even': 'on grey15',
    'row_odd': 'on grey11',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}


def show_banner() -> None:
    """Display AI Predictor banner"""
    banner = Text()
    banner.append("AI STOCK PREDICTOR\n\n", style="bold white")
    banner.append("LSTM Price Prediction", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Real-time Sentiment Analysis", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Production ML Models", style="white")

    console.print(Panel(banner, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
    console.print()


def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch comprehensive stock data"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1y')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        price_1y_ago = hist['Close'].iloc[0]
        returns_1y = ((current_price - price_1y_ago) / price_1y_ago * 100)

        return {
            'symbol': symbol,
            'name': info.get('shortName', symbol),
            'price': current_price,
            'hist': hist,
            'prices': hist['Close'],
            'returns_1y': returns_1y,
            'market_cap': info.get('marketCap', 0),
            'pe': info.get('trailingPE', 0),
            'volume': hist['Volume'].iloc[-1],
            'avg_volume': hist['Volume'].tail(30).mean()
        }

    except Exception:
        return None


def create_prediction_table(stock: Dict[str, Any], lstm_result: Dict[str, Any]) -> Table:
    """Create LSTM prediction results table"""
    table = Table(
        title=f"LSTM PRICE PREDICTIONS - {stock['symbol']}",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        row_styles=[THEME['row_even'], THEME['row_odd']],
        padding=(0, 1)
    )

    table.add_column("Horizon", style="white", width=12)
    table.add_column("Current", style="white", justify="right", width=12)
    table.add_column("Predicted", style="white", justify="right", width=12)
    table.add_column("Change", style="white", justify="right", width=12)
    table.add_column("Confidence", style="white", justify="right", width=14)

    current = stock['price']
    predictions = lstm_result['predictions']
    intervals = lstm_result['confidence_intervals']

    for horizon in ['7d', '14d', '30d']:
        pred = predictions.get(horizon, current)
        low, high = intervals.get(horizon, (current, current))

        change = ((pred - current) / current) * 100
        change_color = "green" if change > 0 else "red" if change < 0 else "white"

        horizon_label = horizon.replace('d', ' Days')

        table.add_row(
            horizon_label,
            f"${current:.2f}",
            f"${pred:.2f}",
            f"[{change_color}]{change:+.1f}%[/{change_color}]",
            f"${low:.2f} - ${high:.2f}"
        )

    return table


def create_sentiment_panel(sentiment: Dict[str, Any]) -> Panel:
    """Create sentiment analysis panel"""
    text = Text()
    text.append("MARKET SENTIMENT\n\n", style="bold white")

    score = sentiment['sentiment_score']
    label = sentiment['sentiment_label']
    news_count = sentiment['news_count']

    if "POSITIVE" in label:
        label_color = "bright_green"
    elif "NEGATIVE" in label:
        label_color = "red"
    else:
        label_color = "white"

    text.append("Overall Sentiment: ", style="white")
    text.append(f"{label}\n", style=f"bold {label_color}")

    text.append("Sentiment Score: ", style="white")
    score_color = "green" if score > 0 else "red" if score < 0 else "white"
    text.append(f"{score:+.1f}\n", style=score_color)

    text.append(f"News Articles: {news_count}\n\n", style="white")

    pos_ratio = sentiment['positive_ratio']
    neg_ratio = sentiment['negative_ratio']
    neu_ratio = sentiment['neutral_ratio']

    text.append("Breakdown:\n", style="bold white")
    text.append(f"  Positive: {pos_ratio:.0%}\n", style="green")
    text.append(f"  Negative: {neg_ratio:.0%}\n", style="red")
    text.append(f"  Neutral: {neu_ratio:.0%}\n", style="white")

    confidence = sentiment['confidence']
    text.append(f"\nConfidence: {confidence:.0%}", style="bright_black")

    return Panel(text, title="[bold white]SENTIMENT ANALYSIS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_ai_insight_panel(lstm_result: Dict[str, Any], sentiment: Dict[str, Any], stock: Dict[str, Any]) -> Panel:
    """Create AI trading insight panel"""
    text = Text()
    text.append("AI TRADING INSIGHTS\n\n", style="bold white")

    trend = lstm_result['trend']
    sentiment_label = sentiment['sentiment_label']

    text.append("Price Trend: ", style="white")
    trend_color = "bright_green" if "UP" in trend else "red" if "DOWN" in trend else "white"
    text.append(f"{trend}\n", style=f"bold {trend_color}")

    text.append("Market Sentiment: ", style="white")
    sent_color = "bright_green" if "POSITIVE" in sentiment_label else "red" if "NEGATIVE" in sentiment_label else "white"
    text.append(f"{sentiment_label}\n\n", style=f"bold {sent_color}")

    pred_30d = lstm_result['predictions']['30d']
    current = stock['price']
    potential_return = ((pred_30d - current) / current) * 100

    text.append("Model Agreement:\n", style="bold white")

    if "UP" in trend and "POSITIVE" in sentiment_label:
        text.append("  ✓ LSTM and Sentiment BULLISH\n", style="bright_green")
        recommendation = "STRONG BUY"
        rec_color = "bright_green"
    elif "DOWN" in trend and "NEGATIVE" in sentiment_label:
        text.append("  ✓ LSTM and Sentiment BEARISH\n", style="red")
        recommendation = "STRONG SELL"
        rec_color = "red"
    elif "UP" in trend:
        text.append("  Mixed: LSTM Bullish, Sentiment Neutral\n", style="green")
        recommendation = "BUY"
        rec_color = "green"
    elif "DOWN" in trend:
        text.append("  Mixed: LSTM Bearish, Sentiment Neutral\n", style="red")
        recommendation = "SELL"
        rec_color = "red"
    else:
        text.append("  No clear signal\n", style="white")
        recommendation = "HOLD"
        rec_color = "white"

    text.append(f"\n30-Day Potential: ", style="white")
    pot_color = "green" if potential_return > 0 else "red"
    text.append(f"{potential_return:+.1f}%\n", style=pot_color)

    text.append("\nRecommendation: ", style="bold white")
    text.append(f"{recommendation}", style=f"bold {rec_color}")

    return Panel(text, title="[bold white]AI INSIGHTS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_model_info_panel(lstm_result: Dict[str, Any]) -> Panel:
    """Create model information panel"""
    text = Text()
    text.append("MODEL PERFORMANCE\n\n", style="bold white")

    accuracy = lstm_result['accuracy']
    confidence = lstm_result['confidence']

    text.append("LSTM Accuracy: ", style="white")
    acc_color = "bright_green" if accuracy > 75 else "green" if accuracy > 60 else "white"
    text.append(f"{accuracy:.1f}%\n", style=acc_color)

    text.append("Model Confidence: ", style="white")
    conf_color = "bright_green" if confidence > 0.75 else "green" if confidence > 0.60 else "white"
    text.append(f"{confidence:.0%}\n\n", style=conf_color)

    text.append("Models Used:\n", style="bold white")
    text.append("  • Multi-layer LSTM\n", style="white")
    text.append("  • Dropout Regularization\n", style="white")
    text.append("  • Early Stopping\n", style="white")
    text.append("  • NLP Sentiment Analysis\n", style="white")

    return Panel(text, title="[bold white]MODEL INFO[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def main() -> None:
    """Main AI Predictor application"""
    console.clear()
    show_banner()

    symbol = Prompt.ask("[white]Enter stock symbol[/white]", default="AAPL").upper()
    console.print()

    console.print(f"[white]Loading data for {symbol}...[/white]")
    stock = fetch_stock_data(symbol)

    if not stock:
        console.print(f"[red]Error: Could not fetch data for {symbol}[/red]\n")
        return

    console.print(f"[green]✓ Data loaded[/green]")
    console.print(f"[white]Training LSTM model...[/white]")

    lstm_predictor = LSTMPricePredictor(sequence_length=60, epochs=30)

    if lstm_predictor.train(stock['prices']):
        console.print(f"[green]✓ LSTM model trained[/green]")
    else:
        console.print(f"[yellow]Note: Using fallback prediction[/yellow]")

    lstm_result = lstm_predictor.predict(stock['prices'], [7, 14, 30])

    console.print(f"[white]Analyzing market sentiment...[/white]")
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_result = sentiment_analyzer.analyze_stock_sentiment(symbol, stock['name'])

    if sentiment_result:
        sentiment = {
            'sentiment_score': sentiment_result.overall_score,
            'sentiment_label': sentiment_result.sentiment_label,
            'news_count': sentiment_result.news_count,
            'positive_ratio': sentiment_result.positive_ratio,
            'negative_ratio': sentiment_result.negative_ratio,
            'neutral_ratio': sentiment_result.neutral_ratio,
            'confidence': sentiment_result.confidence
        }
        console.print(f"[green]✓ Sentiment analyzed ({sentiment_result.news_count} articles)[/green]\n")
    else:
        sentiment = {
            'sentiment_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'news_count': 0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            'confidence': 0.30
        }
        console.print(f"[yellow]Note: Limited sentiment data[/yellow]\n")

    console.clear()
    show_banner()

    info_text = Text()
    info_text.append(f"{stock['symbol']} - {stock['name']}", style="bold white")
    info_text.append(" │ ", style="bright_black")
    info_text.append(f"${stock['price']:.2f}", style="bold bright_green")
    info_text.append(" │ ", style="bright_black")
    info_text.append(f"1Y: {stock['returns_1y']:+.1f}%", style="green" if stock['returns_1y'] > 0 else "red")

    console.print(Panel(info_text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    console.print(create_prediction_table(stock, lstm_result))
    console.print()

    console.print(Columns([
        create_sentiment_panel(sentiment),
        create_ai_insight_panel(lstm_result, sentiment, stock)
    ]))
    console.print()

    console.print(create_model_info_panel(lstm_result))
    console.print()

    console.print("[bright_black]AI Predictor • LSTM + Sentiment • Not financial advice[/bright_black]", justify="center")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[white]Cancelled by user[/white]\n")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
