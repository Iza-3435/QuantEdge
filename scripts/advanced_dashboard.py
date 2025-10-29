#!/usr/bin/env python3
"""
Advanced Real-time Market Intelligence Dashboard

Features:
- Real-time market analysis for multiple symbols
- AI-powered insights and predictions
- Risk analytics and anomaly detection
- Market regime classification
- Sentiment analysis
- Interactive visualizations
- Professional formatting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime
from typing import List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.intelligence.market_intelligence_engine import AdvancedMarketIntelligenceEngine


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


class AdvancedMarketDashboard:
    """Advanced real-time market intelligence dashboard."""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.engine = AdvancedMarketIntelligenceEngine()
        self.last_update = None

    def create_header(self) -> Panel:
        """Create dashboard header."""
        title = Text()
        title.append("AI MARKET INTELLIGENCE SYSTEM\n", style="bold white")
        title.append(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="bright_black")

        return Panel(
            title,
            box=box.SQUARE,
            border_style=THEME['border'],
            style=THEME['panel_bg'],
            padding=(1, 2)
        )

    def create_summary_table(self, analyses: List) -> Panel:
        """Create multi-symbol summary table."""
        table = Table(
            title="Market Overview",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white on grey23",
            border_style="grey35",
            row_styles=["on grey15", "on grey11"],
            expand=True,
            padding=(0, 1)
        )

        table.add_column("Symbol", style="white", width=8)
        table.add_column("Price", justify="right", style="white")
        table.add_column("1D Change", justify="right", style="white")
        table.add_column("5D Pred", justify="right", style="white")
        table.add_column("Confidence", justify="right", style="white")
        table.add_column("Regime", style="white")
        table.add_column("Sentiment", justify="center", style="white")
        table.add_column("Recommendation", justify="center", style="white")

        for intel in analyses:
            # Color coding
            change_1d = intel.price_change_1d * 100
            change_style = "green" if change_1d > 0 else "red"

            pred_5d = intel.predicted_return_5d * 100
            pred_style = "green" if pred_5d > 0 else "red"

            # Sentiment emoji
            if intel.sentiment.sentiment_label == "bullish":
                sentiment_emoji = "🟢 Bull"
            elif intel.sentiment.sentiment_label == "bearish":
                sentiment_emoji = "🔴 Bear"
            else:
                sentiment_emoji = "⚪ Neutral"

            # Recommendation
            rec_map = {
                "strong_buy": "STRONG BUY",
                "buy": "BUY",
                "hold": "HOLD",
                "sell": "SELL",
                "strong_sell": "STRONG SELL"
            }
            rec_display = rec_map.get(intel.recommendation, intel.recommendation)

            table.add_row(
                intel.symbol,
                f"${intel.current_price:.2f}",
                f"[{change_style}]{change_1d:+.2f}%[/{change_style}]",
                f"[{pred_style}]{pred_5d:+.2f}%[/{pred_style}]",
                f"{intel.confidence_score:.2%}",
                intel.market_regime.regime_type.replace('_', ' ').title(),
                sentiment_emoji,
                rec_display
            )

        return Panel(table, title="[bold white]MARKET OVERVIEW[/bold white]", border_style="grey35", box=box.SQUARE, style="on grey11")

    def create_detailed_analysis(self, intel) -> Panel:
        """Create detailed analysis for a single symbol."""
        content = []

        # Header
        content.append(Text(f"\nDetailed Analysis: {intel.symbol}", style="bold white"))
        content.append(Text(f"Current Price: ${intel.current_price:.2f}\n", style="white"))

        # Predictions
        content.append(Text("🎯 AI Predictions:", style="bold"))
        content.append(Text(f"  5-Day Return:     {intel.predicted_return_5d*100:+.2f}%"))
        content.append(Text(f"  20-Day Return:    {intel.predicted_return_20d*100:+.2f}%"))
        content.append(Text(f"  Confidence:       {intel.confidence_score:.1%}"))
        content.append(Text(f"  Profit Prob:      {intel.profit_probability:.1%}\n"))

        # Risk Metrics
        content.append(Text("⚠️  Risk Metrics:", style="bold"))
        content.append(Text(f"  Annual Volatility: {intel.volatility_annual*100:.1f}%"))
        content.append(Text(f"  Sharpe Ratio:      {intel.sharpe_ratio:.2f}"))
        content.append(Text(f"  Beta:              {intel.beta:.2f}"))
        content.append(Text(f"  VaR (95%):         {intel.var_95*100:.2f}%\n"))

        # Market Regime
        regime = intel.market_regime
        content.append(Text("🌐 Market Regime:", style="bold"))
        content.append(Text(f"  Type:              {regime.regime_type.replace('_', ' ').title()}"))
        content.append(Text(f"  Confidence:        {regime.confidence:.1%}"))
        content.append(Text(f"  Volatility:        {regime.volatility*100:.2f}%"))
        content.append(Text(f"  Trend Strength:    {regime.trend_strength:.2f}\n"))

        # Technical Indicators
        content.append(Text("📉 Technical Indicators:", style="bold"))
        content.append(Text(f"  RSI:               {intel.rsi:.1f}"))

        if intel.rsi < 30:
            content.append(Text("                     [Oversold]", style="green"))
        elif intel.rsi > 70:
            content.append(Text("                     [Overbought]", style="red"))

        content.append(Text(f"  MACD:              {intel.macd:.2f}"))
        content.append(Text(f"  Bollinger:         {intel.bollinger_position:.2%}"))
        content.append(Text(f"  Volume:            {intel.volume_profile.upper()}\n"))

        # Anomalies
        if intel.anomalies:
            content.append(Text("🚨 Anomalies Detected:", style="bold red"))
            for anom in intel.anomalies:
                severity_color = "red" if anom.severity == "high" else "yellow"
                content.append(Text(f"  • {anom.description}", style=severity_color))
            content.append(Text())

        # Key Insights
        content.append(Text("💡 Key Insights:", style="bold green"))
        for insight in intel.key_insights[:5]:
            content.append(Text(f"  • {insight}"))

        # Opportunities
        if intel.opportunities:
            content.append(Text("\n✅ Opportunities:", style="bold green"))
            for opp in intel.opportunities[:3]:
                content.append(Text(f"  • {opp}"))

        # Risks
        if intel.risks:
            content.append(Text("\n⚠️  Risks:", style="bold yellow"))
            for risk in intel.risks[:3]:
                content.append(Text(f"  • {risk}"))

        # Combine all text
        full_text = Text()
        for t in content:
            full_text.append(t)
            full_text.append("\n")

        return Panel(
            full_text,
            title=f"[bold white]{intel.symbol} - Comprehensive Intelligence[/bold white]",
            border_style=THEME['border'],
            box=box.SQUARE,
            style=THEME['panel_bg']
        )

    def create_comparison_table(self, analyses: List) -> Panel:
        """Create detailed comparison table."""
        table = Table(
            title="Detailed Comparison",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white on grey23",
            border_style="grey35",
            row_styles=["on grey15", "on grey11"],
            expand=True,
            padding=(0, 1)
        )

        table.add_column("Metric", style="white")
        for intel in analyses:
            table.add_column(intel.symbol, justify="right", style="white")

        # Add rows
        metrics = [
            ("Current Price", lambda i: f"${i.current_price:.2f}"),
            ("1D Change", lambda i: self._format_percent(i.price_change_1d)),
            ("5D Change", lambda i: self._format_percent(i.price_change_5d)),
            ("1M Change", lambda i: self._format_percent(i.price_change_1m)),
            ("Predicted 5D", lambda i: self._format_percent(i.predicted_return_5d)),
            ("Predicted 20D", lambda i: self._format_percent(i.predicted_return_20d)),
            ("Confidence", lambda i: f"{i.confidence_score:.1%}"),
            ("Volatility (Annual)", lambda i: f"{i.volatility_annual*100:.1f}%"),
            ("Sharpe Ratio", lambda i: f"{i.sharpe_ratio:.2f}"),
            ("Beta", lambda i: f"{i.beta:.2f}"),
            ("RSI", lambda i: f"{i.rsi:.1f}"),
            ("MACD", lambda i: f"{i.macd:.2f}"),
            ("Sentiment", lambda i: i.sentiment.sentiment_label.title()),
            ("Regime", lambda i: i.market_regime.regime_type.replace('_', ' ').title()),
            ("Recommendation", lambda i: i.recommendation.replace('_', ' ').upper())
        ]

        for metric_name, metric_func in metrics:
            row = [metric_name]
            for intel in analyses:
                try:
                    row.append(metric_func(intel))
                except Exception as e:
                    row.append("N/A")
            table.add_row(*row)

        return Panel(table, title="[bold white]DETAILED COMPARISON[/bold white]", border_style="grey35", box=box.SQUARE, style="on grey11")

    def _format_percent(self, value: float) -> str:
        """Format percentage with color."""
        pct = value * 100
        color = "green" if value > 0 else "red" if value < 0 else "white"
        return f"[{color}]{pct:+.2f}%[/{color}]"

    def create_pattern_analysis(self, intel) -> Panel:
        """Create pattern analysis table."""
        table = Table(
            title=f"Similar Historical Patterns - {intel.symbol}",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white on grey23",
            border_style="grey35",
            row_styles=["on grey15", "on grey11"],
            expand=True,
            padding=(0, 1)
        )

        table.add_column("Date", style="white")
        table.add_column("Similarity", justify="right", style="white")
        table.add_column("Outcome", justify="right", style="white")
        table.add_column("Match Quality", justify="left", style="white")

        for pattern in intel.similar_patterns:
            outcome = pattern['outcome_return'] * 100
            outcome_style = "green" if outcome > 0 else "red"

            # Visual similarity bar
            similarity = pattern['similarity']
            bar_length = int(similarity * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            table.add_row(
                pattern['date'],
                f"{similarity:.1%}",
                f"[{outcome_style}]{outcome:+.2f}%[/{outcome_style}]",
                bar
            )

        return Panel(table, title=f"[bold white]HISTORICAL PATTERNS - {intel.symbol}[/bold white]", border_style="grey35", box=box.SQUARE, style="on grey11")

    def run_live_dashboard(self, refresh_seconds: int = 60):
        """Run live updating dashboard."""
        console.clear()

        with console.status("[bold green]Loading market intelligence...", spinner="dots"):
            time.sleep(1)

        while True:
            try:
                # Analyze all symbols
                analyses = []

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Analyzing markets...",
                        total=len(self.symbols)
                    )

                    for symbol in self.symbols:
                        progress.update(task, description=f"[cyan]Analyzing {symbol}...")
                        intel = self.engine.analyze(symbol)
                        analyses.append(intel)
                        progress.advance(task)

                # Display dashboard
                console.clear()
                console.print(self.create_header())
                console.print()
                console.print(self.create_summary_table(analyses))
                console.print()

                # Detailed analysis for first symbol
                console.print(self.create_detailed_analysis(analyses[0]))
                console.print()

                # Comparison table
                console.print(self.create_comparison_table(analyses))
                console.print()

                # Pattern analysis
                console.print(self.create_pattern_analysis(analyses[0]))
                console.print()

                # Footer
                footer = Text()
                footer.append(f"🔄 Auto-refresh in {refresh_seconds}s", style="dim")
                footer.append(" | ", style="dim")
                footer.append("Press Ctrl+C to exit", style="dim yellow")
                console.print(Panel(footer, style="dim"))

                # Wait for refresh
                time.sleep(refresh_seconds)

            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard stopped by user.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(5)

    def run_static_report(self):
        """Run one-time comprehensive report."""
        console.clear()

        # Header
        console.print(Panel(
            Text("AI MARKET INTELLIGENCE - COMPREHENSIVE REPORT", justify="center", style="bold white"),
            box=box.SQUARE,
            border_style=THEME['border'],
            style=THEME['panel_bg']
        ))
        console.print()

        # Analyze all symbols
        analyses = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Analyzing markets...",
                total=len(self.symbols)
            )

            for symbol in self.symbols:
                progress.update(task, description=f"[cyan]Analyzing {symbol}...")
                intel = self.engine.analyze(symbol)
                analyses.append(intel)
                progress.advance(task)

        console.print()

        # Summary table
        console.print(self.create_summary_table(analyses))
        console.print()

        # Detailed analysis for each symbol
        for intel in analyses:
            console.print(self.create_detailed_analysis(intel))
            console.print()

        # Comparison
        console.print(self.create_comparison_table(analyses))
        console.print()

        # Pattern analysis
        for intel in analyses[:2]:  # First 2 symbols
            console.print(self.create_pattern_analysis(intel))
            console.print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced AI Market Intelligence Dashboard"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
        help="Stock symbols to analyze"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live dashboard with auto-refresh"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=60,
        help="Refresh interval in seconds (for live mode)"
    )
    args = parser.parse_args()

    dashboard = AdvancedMarketDashboard(args.symbols)

    if args.live:
        console.print("[bold green]Starting live dashboard...[/bold green]")
        console.print(f"[dim]Monitoring: {', '.join(args.symbols)}[/dim]\n")
        dashboard.run_live_dashboard(refresh_seconds=args.refresh)
    else:
        dashboard.run_static_report()


if __name__ == "__main__":
    main()
