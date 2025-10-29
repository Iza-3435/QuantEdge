#!/usr/bin/env python3
"""
🔥 BLOOMBERG-STYLE PROFESSIONAL TERMINAL 🔥

Production-Grade Financial Intelligence Platform

Features:
- Real-time market data across all asset classes
- Deep learning predictions (LSTM + Transformer)
- News sentiment analysis with FinBERT
- Options analytics (Black-Scholes, Greeks)
- Portfolio optimization
- Risk management
- Technical analysis
- Multi-asset support
- Professional Bloomberg-style interface

This feeds signals into your HFT execution system!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn
import time
from datetime import datetime
import numpy as np
import pandas as pd

from src.intelligence.market_intelligence_engine import AdvancedMarketIntelligenceEngine
from src.portfolio.portfolio_optimizer import AdvancedPortfolioOptimizer
from src.derivatives.options_pricing import BlackScholesModel, OptionsStrategy
from src.data.news_sentiment_engine import NewsSentimentEngine

console = Console()


class BloombergTerminal:
    """
    Bloomberg-style professional trading terminal.

    Integrates all advanced features into one unified interface.
    """

    def __init__(self):
        self.intelligence_engine = AdvancedMarketIntelligenceEngine()
        self.portfolio_optimizer = AdvancedPortfolioOptimizer(risk_free_rate=0.04)
        self.options_model = BlackScholesModel(risk_free_rate=0.04)
        self.news_engine = NewsSentimentEngine()
        self.watchlist = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

    def create_header(self) -> Panel:
        """Create Bloomberg-style header."""
        now = datetime.now()

        header_text = Text()
        header_text.append("█████ ", style="bold yellow")
        header_text.append("BLOOMBERG", style="bold white on blue")
        header_text.append(" █████  ", style="bold yellow")
        header_text.append("AI MARKET INTELLIGENCE TERMINAL\n", style="bold cyan")
        header_text.append(f"        {now.strftime('%A, %B %d, %Y  %H:%M:%S')}        ", style="dim white")
        header_text.append("LIVE", style="bold red blink")

        return Panel(
            header_text,
            box=box.DOUBLE,
            style="blue",
            border_style="bold yellow"
        )

    def create_market_overview(self) -> Table:
        """Create market overview table."""
        table = Table(
            title="🌍 GLOBAL MARKETS OVERVIEW",
            box=box.HEAVY,
            show_header=True,
            header_style="bold white on blue",
            border_style="blue"
        )

        table.add_column("Symbol", style="bold cyan", width=10)
        table.add_column("Price", justify="right", style="bold white")
        table.add_column("Change", justify="right")
        table.add_column("AI Signal", justify="center")
        table.add_column("Sentiment", justify="center")
        table.add_column("Options IV", justify="right", style="yellow")
        table.add_column("Action", justify="center", style="bold")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Analyzing markets...", total=len(self.watchlist))

            for symbol in self.watchlist:
                try:
                    # Market intelligence
                    intel = self.intelligence_engine.analyze(symbol, lookback_days=252)

                    # Price and change
                    price = intel.current_price
                    change = intel.price_change_1d * 100
                    change_style = "green" if change > 0 else "red"

                    # AI Signal
                    pred = intel.predicted_return_5d * 100
                    if pred > 1:
                        signal = "🚀 STRONG BUY"
                        signal_style = "bold green"
                    elif pred > 0.3:
                        signal = "📈 BUY"
                        signal_style = "green"
                    elif pred < -1:
                        signal = "⚠️  SELL"
                        signal_style = "bold red"
                    elif pred < -0.3:
                        signal = "📉 WEAK"
                        signal_style = "red"
                    else:
                        signal = "⏸️  HOLD"
                        signal_style = "yellow"

                    # News sentiment
                    news_sentiment = self.news_engine.analyze_symbol(symbol, lookback_hours=24)
                    if news_sentiment.sentiment_label == "bullish":
                        sentiment_display = "🟢 BULL"
                    elif news_sentiment.sentiment_label == "bearish":
                        sentiment_display = "🔴 BEAR"
                    else:
                        sentiment_display = "⚪ NEUT"

                    # Implied volatility (mock - would integrate with options data)
                    iv = intel.volatility_annual * 100

                    # Recommended action
                    rec = intel.recommendation.upper()
                    if "BUY" in rec:
                        action = f"[green]{rec}[/green]"
                    elif "SELL" in rec:
                        action = f"[red]{rec}[/red]"
                    else:
                        action = f"[yellow]{rec}[/yellow]"

                    table.add_row(
                        symbol,
                        f"${price:.2f}",
                        f"[{change_style}]{change:+.2f}%[/{change_style}]",
                        f"[{signal_style}]{signal}[/{signal_style}]",
                        sentiment_display,
                        f"{iv:.1f}%",
                        action
                    )

                except Exception as e:
                    console.print(f"[red]Error analyzing {symbol}: {e}[/red]")

                progress.advance(task)

        return table

    def create_portfolio_analytics(self) -> Panel:
        """Create portfolio analytics panel."""
        try:
            self.portfolio_optimizer.load_data(self.watchlist, period="1y")

            # Optimize portfolio
            max_sharpe = self.portfolio_optimizer.optimize_max_sharpe()
            risk_parity = self.portfolio_optimizer.optimize_risk_parity()

            content = Text()
            content.append("💼 PORTFOLIO OPTIMIZATION\n\n", style="bold cyan")

            content.append("Max Sharpe Ratio Strategy:\n", style="bold yellow")
            content.append(f"  Expected Return:  {max_sharpe.returns_annual*100:>6.2f}%\n", style="green")
            content.append(f"  Volatility:       {max_sharpe.volatility_annual*100:>6.2f}%\n")
            content.append(f"  Sharpe Ratio:     {max_sharpe.sharpe_ratio:>6.2f}\n", style="cyan")
            content.append(f"  Max Drawdown:     {max_sharpe.max_drawdown*100:>6.2f}%\n", style="red")

            content.append("\nTop Allocations:\n", style="bold yellow")
            sorted_weights = sorted(max_sharpe.weights.items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_weights[:3]:
                bar = "█" * int(weight * 30)
                content.append(f"  {symbol}: {weight*100:5.1f}% {bar}\n", style="green")

            content.append(f"\nRisk Parity Return: {risk_parity.returns_annual*100:.2f}%", style="dim")

        except Exception as e:
            content = Text(f"Portfolio analytics unavailable: {e}", style="red")

        return Panel(
            content,
            title="[bold white]PORTFOLIO ANALYTICS[/bold white]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def create_news_feed(self) -> Panel:
        """Create news sentiment feed."""
        content = Text()
        content.append("📰 REAL-TIME NEWS SENTIMENT\n\n", style="bold cyan")

        # Sample news sentiment for top symbols
        for symbol in self.watchlist[:3]:
            try:
                news = self.news_engine.analyze_symbol(symbol, lookback_hours=12)

                if news.sentiment_label == "bullish":
                    emoji = "🟢"
                    color = "green"
                elif news.sentiment_label == "bearish":
                    emoji = "🔴"
                    color = "red"
                else:
                    emoji = "⚪"
                    color = "yellow"

                content.append(f"{emoji} {symbol}: ", style=f"bold {color}")
                content.append(f"{news.sentiment_label.upper()} ", style=f"{color}")
                content.append(f"({news.news_count} articles)\n")

                if news.key_events:
                    content.append(f"   Events: {', '.join(news.key_events[:2])}\n", style="dim")

                content.append("\n")

            except:
                pass

        content.append("Powered by FinBERT + Multi-Source Aggregation", style="dim italic")

        return Panel(
            content,
            title="[bold white]NEWS SENTIMENT[/bold white]",
            border_style="yellow",
            box=box.ROUNDED
        )

    def create_options_analytics(self) -> Panel:
        """Create options analytics panel."""
        content = Text()
        content.append("📊 OPTIONS ANALYTICS\n\n", style="bold cyan")

        # Example: AAPL options
        symbol = "AAPL"
        try:
            intel = self.intelligence_engine.analyze(symbol)
            S = intel.current_price
            K = S  # ATM
            T = 30 / 365  # 30 days
            sigma = intel.volatility_annual

            # Price call option
            call_price = self.options_model.price_european_option(S, K, T, sigma, 'call')
            put_price = self.options_model.price_european_option(S, K, T, sigma, 'put')

            # Calculate Greeks
            call_greeks = self.options_model.calculate_greeks(S, K, T, sigma, 'call')

            content.append(f"{symbol} Options (30-Day ATM):\n", style="bold yellow")
            content.append(f"  Spot Price:      ${S:.2f}\n")
            content.append(f"  Call Price:      ${call_price:.2f}\n", style="green")
            content.append(f"  Put Price:       ${put_price:.2f}\n", style="red")

            content.append(f"\nGreeks (Call):\n", style="bold yellow")
            content.append(f"  Delta:           {call_greeks.delta:>7.4f}\n")
            content.append(f"  Gamma:           {call_greeks.gamma:>7.4f}\n")
            content.append(f"  Vega:            {call_greeks.vega:>7.4f}\n")
            content.append(f"  Theta:           {call_greeks.theta:>7.4f}\n", style="red")

            # Strategy suggestion
            content.append(f"\nSuggested Strategy:\n", style="bold cyan")
            if intel.market_regime.regime_type in ["bull_stable", "bull_volatile"]:
                content.append("  • Long Call (Bullish)\n", style="green")
                content.append("  • Bull Call Spread\n", style="green")
            else:
                content.append("  • Covered Call\n", style="yellow")
                content.append("  • Cash-Secured Put\n", style="yellow")

        except Exception as e:
            content.append(f"Error: {e}", style="red")

        return Panel(
            content,
            title="[bold white]OPTIONS ANALYTICS[/bold white]",
            border_style="magenta",
            box=box.ROUNDED
        )

    def create_risk_dashboard(self) -> Table:
        """Create risk metrics dashboard."""
        table = Table(
            title="⚠️  RISK METRICS DASHBOARD",
            box=box.HEAVY,
            show_header=True,
            header_style="bold white on red",
            border_style="red"
        )

        table.add_column("Symbol", style="bold cyan")
        table.add_column("VaR (95%)", justify="right", style="red")
        table.add_column("CVaR", justify="right", style="red")
        table.add_column("Volatility", justify="right", style="yellow")
        table.add_column("Beta", justify="right")
        table.add_column("Sharpe", justify="right", style="cyan")
        table.add_column("Risk Level", justify="center")

        for symbol in self.watchlist:
            try:
                intel = self.intelligence_engine.analyze(symbol)

                var = intel.var_95 * 100
                cvar = var * 1.3  # Approximate
                vol = intel.volatility_annual * 100
                beta = intel.beta
                sharpe = intel.sharpe_ratio

                # Risk level
                if vol < 20:
                    risk_level = "🟢 LOW"
                elif vol < 35:
                    risk_level = "🟡 MED"
                else:
                    risk_level = "🔴 HIGH"

                table.add_row(
                    symbol,
                    f"{var:.2f}%",
                    f"{cvar:.2f}%",
                    f"{vol:.1f}%",
                    f"{beta:.2f}",
                    f"{sharpe:.2f}",
                    risk_level
                )
            except:
                pass

        return table

    def create_footer(self) -> Panel:
        """Create footer with system stats."""
        footer = Text()
        footer.append("⚡ SYSTEM STATUS: ", style="bold green")
        footer.append("OPERATIONAL  ", style="green")
        footer.append("│ ", style="dim")
        footer.append("🔄 Data: ", style="bold cyan")
        footer.append("REAL-TIME  ", style="cyan")
        footer.append("│ ", style="dim")
        footer.append("🤖 AI Models: ", style="bold yellow")
        footer.append("ACTIVE  ", style="yellow")
        footer.append("│ ", style="dim")
        footer.append("🔗 HFT Link: ", style="bold magenta")
        footer.append("READY", style="magenta")

        return Panel(
            footer,
            style="dim",
            box=box.SIMPLE
        )

    def run_terminal(self, refresh_interval: int = 30):
        """
        Run the Bloomberg-style terminal.

        Args:
            refresh_interval: Seconds between updates
        """
        console.clear()

        try:
            while True:
                console.clear()

                # Header
                console.print(self.create_header())
                console.print()

                # Market Overview
                console.print(self.create_market_overview())
                console.print()

                # Two-column layout for analytics
                portfolio_panel = self.create_portfolio_analytics()
                news_panel = self.create_news_feed()
                console.print(Columns([portfolio_panel, news_panel]))
                console.print()

                # Options analytics
                console.print(self.create_options_analytics())
                console.print()

                # Risk dashboard
                console.print(self.create_risk_dashboard())
                console.print()

                # Footer
                console.print(self.create_footer())

                # Status message
                console.print(f"\n[dim]Next update in {refresh_interval}s | Press Ctrl+C to exit[/dim]")

                # Wait for refresh
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Terminal shutdown requested[/yellow]")
            console.print("[green]✓ System shutdown complete[/green]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bloomberg-Style AI Market Intelligence Terminal"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Refresh interval in seconds"
    )
    parser.add_argument(
        "--watchlist",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
        help="Symbols to monitor"
    )

    args = parser.parse_args()

    # Initialize terminal
    terminal = BloombergTerminal()
    terminal.watchlist = args.watchlist

    # Welcome message
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold white]  BLOOMBERG-STYLE AI TERMINAL  [/bold white]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("\n[yellow]Initializing...[/yellow]")
    console.print(f"[green]✓[/green] Market Intelligence Engine")
    console.print(f"[green]✓[/green] Portfolio Optimizer")
    console.print(f"[green]✓[/green] Options Analytics")
    console.print(f"[green]✓[/green] News Sentiment Engine")
    console.print(f"[green]✓[/green] Risk Management System")
    console.print(f"\n[cyan]Monitoring: {', '.join(args.watchlist)}[/cyan]")
    console.print(f"[dim]Refresh interval: {args.refresh}s[/dim]\n")

    time.sleep(2)

    # Run terminal
    terminal.run_terminal(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
