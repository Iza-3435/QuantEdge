"""
UI ENHANCEMENTS MODULE
Enhanced visual components: color gradients, progress bars, better sparklines, status indicators
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from rich.text import Text
from rich.panel import Panel


def get_performance_color(value: float, thresholds: Optional[Dict] = None) -> str:
    if thresholds is None:
        thresholds = {
            'excellent': 20,
            'good': 10,
            'neutral': 0,
            'poor': -10
        }

    if value >= thresholds['excellent']:
        return "bright_green"
    elif value >= thresholds['good']:
        return "green"
    elif value >= thresholds['neutral']:
        return "yellow"
    elif value >= thresholds['poor']:
        return "orange"
    else:
        return "red"


def create_progress_bar(value: float, max_value: float, width: int = 10,
                       color: Optional[str] = None) -> str:
    if max_value == 0:
        percentage = 0
    else:
        percentage = min(value / max_value, 1.0)

    filled = int(percentage * width)
    empty = width - filled

    if color is None:
        if percentage >= 0.8:
            color = "bright_green"
        elif percentage >= 0.6:
            color = "green"
        elif percentage >= 0.4:
            color = "yellow"
        else:
            color = "red"

    bar = "█" * filled + "░" * empty
    return f"[{color}]{bar}[/{color}]"


def create_enhanced_sparkline(prices: List[float], width: int = 30,
                              show_trend: bool = True) -> str:
    if not prices or len(prices) < 2:
        return "━" * width

    prices = [p for p in prices if not pd.isna(p)]
    if len(prices) < 2:
        return "━" * width

    min_p, max_p = min(prices), max(prices)
    range_p = max_p - min_p if max_p != min_p else 1

    chars = '▁▂▃▄▅▆▇█'
    sparkline = ''.join(
        chars[min(int(((p - min_p) / range_p) * 7), 7)]
        for p in prices[-width:]
    )

    if show_trend:
        start_price = prices[0]
        end_price = prices[-1]
        change_pct = ((end_price - start_price) / start_price) * 100

        if end_price > start_price:
            color = "green" if change_pct > 5 else "bright_green"
            trend = f"↗ +{change_pct:.1f}%"
        elif end_price < start_price:
            color = "red" if change_pct < -5 else "bright_red"
            trend = f"↘ {change_pct:.1f}%"
        else:
            color = "white"
            trend = "→ 0.0%"

        return f"[{color}]{sparkline} {trend}[/{color}]"
    else:
        color = "green" if prices[-1] > prices[0] else "red"
        return f"[{color}]{sparkline}[/{color}]"


def get_recommendation_indicator(score: float, metric_type: str = "score") -> str:
    if metric_type == "score":
        if score >= 80:
            return "[bright_green]STRONG BUY[/bright_green]"
        elif score >= 65:
            return "[green]BUY[/green]"
        elif score >= 50:
            return "[yellow]HOLD[/yellow]"
        elif score >= 35:
            return "[orange1]SELL[/orange1]"
        else:
            return "[red]STRONG SELL[/red]"

    elif metric_type == "rsi":
        if score > 70:
            return "[red]OVERBOUGHT[/red]"
        elif score > 60:
            return "[orange1]STRONG[/orange1]"
        elif score > 40:
            return "[green]NEUTRAL[/green]"
        elif score > 30:
            return "[yellow]WEAK[/yellow]"
        else:
            return "[green]OVERSOLD[/green]"

    elif metric_type == "momentum":
        if score >= 75:
            return "[green]STRONG[/green]"
        elif score >= 50:
            return "[yellow]MODERATE[/yellow]"
        else:
            return "[red]WEAK[/red]"

    elif metric_type == "quality":
        if score >= 80:
            return "[bright_green]EXCELLENT[/bright_green]"
        elif score >= 65:
            return "[green]STRONG[/green]"
        elif score >= 50:
            return "[yellow]GOOD[/yellow]"
        elif score >= 35:
            return "[orange1]FAIR[/orange1]"
        else:
            return "[red]WEAK[/red]"

    return "[bright_black]N/A[/bright_black]"


def create_metric_heatmap_row(values: List[float], labels: List[str],
                              width: int = 15) -> str:
    row = ""
    for value, label in zip(values, labels):
        color = get_performance_color(value, {
            'excellent': 75,
            'good': 50,
            'neutral': 25,
            'poor': 0
        })

        bar = create_progress_bar(value, 100, width=8, color=color)
        row += f"{label}: {bar} {value:.0f}  "

    return row


def format_currency(value: float, decimals: int = 2) -> str:
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.{decimals}f}T"
    elif value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.{decimals}f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.{decimals}f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.{decimals}f}K"
    else:
        return f"${value:.{decimals}f}"


def format_percentage(value: float, decimals: int = 2, colored: bool = True) -> str:
    if colored:
        color = get_performance_color(value)
        sign = "+" if value > 0 else ""
        return f"[{color}]{sign}{value:.{decimals}f}%[/{color}]"
    else:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{decimals}f}%"


def create_score_badge(score: float, max_score: float, label: str) -> str:
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    color = get_performance_color(percentage, {
        'excellent': 80,
        'good': 60,
        'neutral': 40,
        'poor': 20
    })

    bar = create_progress_bar(score, max_score, width=10, color=color)
    return f"{label}: {bar} [{color}]{score:.0f}/{max_score:.0f}[/{color}]"


def create_comparison_bars(values: Dict[str, float], width: int = 20) -> List[str]:
    if not values:
        return []

    max_value = max(values.values())
    bars = []

    for label, value in values.items():
        percentage = value / max_value if max_value > 0 else 0
        color = get_performance_color(value)

        filled = int(percentage * width)
        empty = width - filled

        bar = "█" * filled + "░" * empty
        bars.append(f"{label:15s} [{color}]{bar}[/{color}] {value:.1f}")

    return bars


def create_trend_indicator(current: float, previous: float) -> str:
    if previous == 0:
        return "N/A"

    change_pct = ((current - previous) / previous) * 100

    if change_pct > 5:
        return f"[bright_green]+{change_pct:.1f}%[/bright_green]"
    elif change_pct > 0:
        return f"[green]+{change_pct:.1f}%[/green]"
    elif change_pct > -5:
        return f"[red]{change_pct:.1f}%[/red]"
    else:
        return f"[bright_red]{change_pct:.1f}%[/bright_red]"


def create_signal_panel(signals: Dict[str, Any], title: str = "Signals") -> Panel:
    text = Text()

    for signal_name, signal_value in signals.items():
        text.append(f"{signal_name}: ", style="bright_black")

        if isinstance(signal_value, str):
            if "BUY" in signal_value.upper():
                text.append(f"{signal_value}\n", style="green")
            elif "SELL" in signal_value.upper():
                text.append(f"{signal_value}\n", style="red")
            elif "HOLD" in signal_value.upper():
                text.append(f"{signal_value}\n", style="yellow")
            else:
                text.append(f"{signal_value}\n", style="white")
        else:
            color = get_performance_color(signal_value)
            text.append(f"{signal_value:.2f}\n", style=color)

    return Panel(text, title=f"[bold cyan]{title}[/bold cyan]",
                border_style="cyan", expand=False)


def create_mini_chart(values: List[float], height: int = 5, width: int = 40) -> str:
    if not values or len(values) < 2:
        return "No data"

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    if len(values) > width:
        step = len(values) // width
        sampled = values[::step][:width]
    else:
        sampled = values

    chart_lines = []
    for y in range(height - 1, -1, -1):
        line = ""
        threshold = (y / (height - 1))

        for value in sampled:
            normalized = (value - min_val) / range_val
            if normalized >= threshold:
                line += "█"
            else:
                line += " "

        chart_lines.append(line)

    return "\n".join(chart_lines)
