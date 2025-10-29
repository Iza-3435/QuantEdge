#!/usr/bin/env python3
"""
🚀 INTERACTIVE STOCK ANALYZER 🚀

Type any ticker → Get EVERYTHING:
- AI Predictions
- Earnings History
- SEC Filings
- News Sentiment
- Options Analytics
- Peer Comparison
- Historical Performance
- Complete Analysis

Ultimate all-in-one stock research tool!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.layout import Layout
import yfinance as yf
from datetime import datetime
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.intelligence.market_intelligence_engine import AdvancedMarketIntelligenceEngine
from src.portfolio.portfolio_optimizer import AdvancedPortfolioOptimizer
from src.derivatives.options_pricing import BlackScholesModel
from src.data.news_sentiment_engine import NewsSentimentEngine
from src.data.sec_edgar_analyzer import SECEdgarAnalyzer
from src.models.ensemble_predictor import AdvancedEnsemblePredictor
from src.data.insider_trading_analyzer import InsiderTradingAnalyzer
from src.data.institutional_tracker import InstitutionalTracker
from src.data.social_sentiment_analyzer import SocialSentimentAnalyzer

console = Console()


def analyze_stock(symbol: str):
    """Comprehensive stock analysis."""
    symbol = symbol.upper()

    console.clear()
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold white]  COMPREHENSIVE ANALYSIS: {symbol}[/bold white]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    try:
        # Fetch stock data
        console.print(f"[yellow]Fetching data...[/yellow]")
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="2y")

        if hist.empty:
            console.print(f"[red]Error: No data found for {symbol}[/red]")
            return

        # Initialize AI engines with API keys
        intel_engine = AdvancedMarketIntelligenceEngine()
        news_engine = NewsSentimentEngine(
            newsapi_key=os.getenv('NEWSAPI_KEY'),
            alphavantage_key=os.getenv('ALPHAVANTAGE_KEY')
        )
        options_model = BlackScholesModel()
        sec_analyzer = SECEdgarAnalyzer(
            user_agent=os.getenv('SEC_USER_AGENT', 'AI Market Intelligence research@example.com')
        )
        ensemble_predictor = AdvancedEnsemblePredictor()
        insider_analyzer = InsiderTradingAnalyzer(
            sec_user_agent=os.getenv('SEC_USER_AGENT', 'AI Market Intelligence research@example.com')
        )
        institutional_tracker = InstitutionalTracker()
        social_analyzer = SocialSentimentAnalyzer()

        # ========== SECTION 1: COMPANY OVERVIEW ==========
        console.print("\n[bold cyan]📊 COMPANY OVERVIEW[/bold cyan]\n")

        overview_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        overview_table.add_column("Field", style="cyan")
        overview_table.add_column("Value", style="white")

        overview_table.add_row("Company", info.get('longName', symbol))
        overview_table.add_row("Sector", info.get('sector', 'N/A'))
        overview_table.add_row("Industry", info.get('industry', 'N/A'))
        overview_table.add_row("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
        overview_table.add_row("Employees", f"{info.get('fullTimeEmployees', 0):,}")
        overview_table.add_row("Website", info.get('website', 'N/A'))

        console.print(overview_table)

        # ========== SECTION 2: PRICE PERFORMANCE ==========
        console.print("\n[bold cyan]💰 PRICE PERFORMANCE[/bold cyan]\n")

        current_price = info.get('currentPrice', hist['Close'][-1])

        perf_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        perf_table.add_column("Period", style="cyan")
        perf_table.add_column("Return", justify="right")
        perf_table.add_column("High", justify="right", style="green")
        perf_table.add_column("Low", justify="right", style="red")

        periods = {
            '1 Day': (1, hist['Close'][-2] if len(hist) >= 2 else current_price),
            '1 Week': (5, hist['Close'][-5] if len(hist) >= 5 else current_price),
            '1 Month': (21, hist['Close'][-21] if len(hist) >= 21 else current_price),
            '3 Months': (63, hist['Close'][-63] if len(hist) >= 63 else current_price),
            '6 Months': (126, hist['Close'][-126] if len(hist) >= 126 else current_price),
            '1 Year': (252, hist['Close'][-252] if len(hist) >= 252 else current_price),
        }

        for period_name, (days, start_price) in periods.items():
            if len(hist) >= days:
                period_data = hist.iloc[-days:]
                ret = (current_price / start_price - 1) * 100
                high = period_data['High'].max()
                low = period_data['Low'].min()

                ret_style = "green" if ret > 0 else "red"
                perf_table.add_row(
                    period_name,
                    f"[{ret_style}]{ret:+.2f}%[/{ret_style}]",
                    f"${high:.2f}",
                    f"${low:.2f}"
                )

        console.print(perf_table)

        # Current price
        price_change = (current_price / hist['Close'][-2] - 1) * 100 if len(hist) >= 2 else 0
        price_style = "green" if price_change > 0 else "red"
        console.print(f"\n[bold white]Current Price: ${current_price:.2f}[/bold white] [{price_style}]{price_change:+.2f}%[/{price_style}]")

        # ========== SECTION 3: ADVANCED AI PREDICTIONS (ENSEMBLE) ==========
        console.print("\n[bold cyan]🤖 ADVANCED AI ENSEMBLE PREDICTIONS[/bold cyan]\n")

        intel = intel_engine.analyze(symbol, lookback_days=500)

        # Get ensemble prediction
        ensemble_pred = ensemble_predictor.predict_ensemble(
            hist,
            lstm_pred=intel.predicted_return_5d*100,
            transformer_pred=intel.predicted_return_20d*100/4,  # Scale to 5-day
            horizon_days=5
        )

        # Anomaly detection
        anomaly = ensemble_predictor.detect_anomalies(hist)

        ai_panel = Panel(
            Text.from_markup(f"""
[bold yellow]🎯 Ensemble Prediction (4 Models):[/bold yellow]
  Final Forecast (5d):     [green bold]{ensemble_pred.ensemble_prediction:+.2f}%[/green bold]
  Confidence:              [yellow]{ensemble_pred.ensemble_confidence:.0%}[/yellow]
  Model Agreement:         [cyan]{ensemble_pred.model_agreement:.0%}[/cyan]

[bold yellow]📊 Individual Models:[/bold yellow]
  LSTM ({ensemble_pred.lstm_weight:.0%}):            {ensemble_pred.lstm_prediction:+.2f}%
  Transformer ({ensemble_pred.transformer_weight:.0%}):    {ensemble_pred.transformer_prediction:+.2f}%
  XGBoost ({ensemble_pred.xgboost_weight:.0%}):        {ensemble_pred.xgboost_prediction:+.2f}%
  Random Forest ({ensemble_pred.rf_weight:.0%}):   {ensemble_pred.random_forest_prediction:+.2f}%

[bold yellow]📈 Multi-Horizon:[/bold yellow]
  5-Day:                   {ensemble_pred.short_term_5d:+.2f}%
  20-Day:                  {ensemble_pred.medium_term_20d:+.2f}%
  60-Day:                  {ensemble_pred.long_term_60d:+.2f}%

[bold yellow]Market Regime:[/bold yellow]
  Type:                    [blue]{intel.market_regime.regime_type.replace('_', ' ').title()}[/blue]
  Volatility:              {intel.market_regime.volatility*100:.2f}%

[bold yellow]⚠️  Anomaly Status:[/bold yellow]  {'[red bold]' + anomaly.get('message', 'N/A') + '[/red bold]' if anomaly.get('anomaly_detected') else '[green]' + anomaly.get('message', 'Normal') + '[/green]'}

[bold yellow]Recommendation:[/bold yellow]  [bold cyan]{intel.recommendation.upper()}[/bold cyan]
            """),
            border_style="cyan",
            box=box.ROUNDED,
            title="🧠 Advanced Ensemble ML"
        )
        console.print(ai_panel)

        # Risk Metrics
        console.print("\n[bold cyan]⚠️  RISK ANALYSIS[/bold cyan]\n")

        risk_table = Table(box=box.SIMPLE, show_header=False)
        risk_table.add_column("Metric", style="cyan")
        risk_table.add_column("Value", justify="right")

        risk_table.add_row("Annual Volatility", f"{intel.volatility_annual*100:.1f}%")
        risk_table.add_row("Beta", f"{intel.beta:.2f}")
        risk_table.add_row("Sharpe Ratio", f"{intel.sharpe_ratio:.2f}")
        risk_table.add_row("VaR (95%)", f"[red]{intel.var_95*100:.2f}%[/red]")

        console.print(risk_table)

        # ========== SECTION 4: EARNINGS ==========
        console.print("\n[bold cyan]📈 EARNINGS HISTORY[/bold cyan]\n")

        try:
            earnings_dates = stock.get_earnings_dates(limit=8)

            if earnings_dates is not None and not earnings_dates.empty:
                earnings_table = Table(box=box.ROUNDED, show_header=True)
                earnings_table.add_column("Date", style="cyan")
                earnings_table.add_column("EPS Est", justify="right")
                earnings_table.add_column("EPS Actual", justify="right")
                earnings_table.add_column("Surprise", justify="right")

                for idx, row in earnings_dates.head(5).iterrows():
                    est = row.get('EPS Estimate', 0)
                    actual = row.get('Reported EPS', 0)

                    if pd.notna(est) and pd.notna(actual):
                        surprise = (actual / est - 1) * 100 if est != 0 else 0
                        surprise_style = "green" if surprise > 0 else "red"

                        earnings_table.add_row(
                            idx.strftime('%Y-%m-%d'),
                            f"{est:.2f}",
                            f"{actual:.2f}",
                            f"[{surprise_style}]{surprise:+.1f}%[/{surprise_style}]"
                        )

                console.print(earnings_table)
            else:
                console.print("[dim]No earnings data available[/dim]")
        except:
            console.print("[dim]Earnings data unavailable[/dim]")

        # Next earnings
        try:
            calendar = stock.get_earnings_dates()
            if calendar is not None:
                future = calendar[calendar.index > datetime.now()]
                if len(future) > 0:
                    next_date = future.index[0]
                    console.print(f"\n[yellow]Next Earnings:[/yellow] [bold]{next_date.strftime('%Y-%m-%d')}[/bold]")
        except:
            pass

        # ========== SECTION 5: NEWS SENTIMENT ==========
        console.print("\n[bold cyan]📰 NEWS SENTIMENT[/bold cyan]\n")

        sentiment = news_engine.analyze_symbol(symbol, lookback_hours=48)

        if sentiment.sentiment_label == "bullish":
            emoji = "🟢"
            color = "green"
        elif sentiment.sentiment_label == "bearish":
            emoji = "🔴"
            color = "red"
        else:
            emoji = "⚪"
            color = "yellow"

        news_panel = Panel(
            Text.from_markup(f"""
{emoji} [bold {color}]{sentiment.sentiment_label.upper()}[/bold {color}]

Sentiment Score:     {sentiment.overall_sentiment:+.2f}
Confidence:          {sentiment.confidence:.0%}
News Articles:       {sentiment.news_count}
Signal Strength:     {sentiment.signal_strength:.0%}

[bold]Key Events:[/bold]
{chr(10).join(f"  • {event}" for event in (sentiment.key_events[:3] if sentiment.key_events else ['No major events']))}

[bold]Trending Topics:[/bold]
{', '.join(sentiment.trending_topics[:5]) if sentiment.trending_topics else 'N/A'}
            """),
            border_style=color,
            box=box.ROUNDED
        )
        console.print(news_panel)

        # ========== SECTION 6: FUNDAMENTALS ==========
        console.print("\n[bold cyan]💼 FUNDAMENTALS[/bold cyan]\n")

        fund_cols = Columns([
            Table.grid(padding=(0, 1)),
            Table.grid(padding=(0, 1))
        ])

        # Valuation metrics
        val_table = Table(box=box.SIMPLE, show_header=False, title="Valuation")
        val_table.add_column("", style="cyan")
        val_table.add_column("", justify="right")

        val_table.add_row("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
        val_table.add_row("Forward P/E", f"{info.get('forwardPE', 0):.2f}")
        val_table.add_row("PEG Ratio", f"{info.get('pegRatio', 0):.2f}")
        val_table.add_row("Price/Book", f"{info.get('priceToBook', 0):.2f}")
        val_table.add_row("Price/Sales", f"{info.get('priceToSalesTrailing12Months', 0):.2f}")

        # Profitability
        prof_table = Table(box=box.SIMPLE, show_header=False, title="Profitability")
        prof_table.add_column("", style="cyan")
        prof_table.add_column("", justify="right")

        prof_table.add_row("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
        prof_table.add_row("Operating Margin", f"{info.get('operatingMargins', 0)*100:.2f}%")
        prof_table.add_row("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
        prof_table.add_row("ROA", f"{info.get('returnOnAssets', 0)*100:.2f}%")

        console.print(Columns([val_table, prof_table]))

        # ========== SECTION 7: OPTIONS (if available) ==========
        console.print("\n[bold cyan]📊 OPTIONS ANALYTICS (30-Day ATM)[/bold cyan]\n")

        try:
            S = current_price
            K = S
            T = 30/365
            sigma = intel.volatility_annual

            call_price = options_model.price_european_option(S, K, T, sigma, 'call')
            put_price = options_model.price_european_option(S, K, T, sigma, 'put')
            greeks = options_model.calculate_greeks(S, K, T, sigma, 'call')

            options_table = Table(box=box.SIMPLE, show_header=False)
            options_table.add_column("", style="cyan")
            options_table.add_column("", justify="right")

            options_table.add_row("Call Price", f"[green]${call_price:.2f}[/green]")
            options_table.add_row("Put Price", f"[red]${put_price:.2f}[/red]")
            options_table.add_row("Delta", f"{greeks.delta:.4f}")
            options_table.add_row("Gamma", f"{greeks.gamma:.4f}")
            options_table.add_row("Vega", f"{greeks.vega:.4f}")
            options_table.add_row("Theta (daily)", f"{greeks.theta:.4f}")

            console.print(options_table)
        except:
            console.print("[dim]Options data unavailable[/dim]")

        # ========== SECTION 8: COMPARISON ==========
        console.print("\n[bold cyan]📊 MARKET COMPARISON[/bold cyan]\n")

        try:
            # Compare to S&P 500
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")

            if len(hist) >= 252 and len(spy_hist) >= 252:
                stock_1y = (hist['Close'][-1] / hist['Close'][-252] - 1) * 100
                spy_1y = (spy_hist['Close'][-1] / spy_hist['Close'][-252] - 1) * 100
                outperformance = stock_1y - spy_1y

                perf_style = "green" if outperformance > 0 else "red"

                console.print(f"{symbol} 1-Year Return:  [{perf_style}]{stock_1y:+.2f}%[/{perf_style}]")
                console.print(f"S&P 500 1-Year:       {spy_1y:+.2f}%")
                console.print(f"Outperformance:       [{perf_style}]{outperformance:+.2f}%[/{perf_style}]")
        except:
            pass

        # ========== SECTION 9: SEC FILING ANALYSIS ==========
        console.print("\n[bold cyan]📄 SEC FILING ANALYSIS[/bold cyan]\n")

        try:
            # Get company CIK
            cik = sec_analyzer.get_company_cik(symbol)

            if cik:
                console.print(f"[dim]Analyzing recent SEC filings for {symbol}...[/dim]\n")

                # Get recent filings
                recent_filings = sec_analyzer.get_recent_filings(cik, limit=5)

                if recent_filings:
                    # Create filings table
                    filing_table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
                    filing_table.add_column("Type", style="cyan")
                    filing_table.add_column("Date", style="white")
                    filing_table.add_column("Report Date", style="yellow")

                    for filing in recent_filings[:3]:
                        filing_table.add_row(
                            filing['form'],
                            filing['filingDate'],
                            filing.get('reportDate', 'N/A')
                        )

                    console.print(filing_table)

                    # Analyze most recent 10-K if available
                    console.print(f"\n[dim]Analyzing latest 10-K filing...[/dim]")
                    filing_10k = sec_analyzer.analyze_10k(cik)

                    if filing_10k:
                        # Risk level indicator
                        risk_color = "green" if filing_10k.risk_level == "low" else "yellow" if filing_10k.risk_level == "medium" else "red"
                        risk_emoji = "🟢" if filing_10k.risk_level == "low" else "🟡" if filing_10k.risk_level == "medium" else "🔴"

                        sec_panel = Panel(
                            Text.from_markup(f"""
[bold]Latest 10-K Analysis:[/bold]
Filing Date: {filing_10k.filing_date.strftime('%Y-%m-%d')}
Risk Level: {risk_emoji} [{risk_color}]{filing_10k.risk_level.upper()}[/{risk_color}]
Sentiment: {filing_10k.sentiment_score:+.2f}

[bold red]Red Flags:[/bold red]
{chr(10).join(f"  ⚠️  {flag}" for flag in filing_10k.red_flags[:3]) if filing_10k.red_flags else '  ✅ None detected'}

[bold green]Positive Signals:[/bold green]
{chr(10).join(f"  ✅ {signal}" for signal in filing_10k.positive_signals[:3]) if filing_10k.positive_signals else '  None identified'}

[bold]Top Risk Factors:[/bold]
{chr(10).join(f"  • {risk[:100]}..." for risk in filing_10k.risk_factors[:2]) if filing_10k.risk_factors else '  • No major risks extracted'}
                            """),
                            border_style=risk_color,
                            box=box.ROUNDED,
                            title="📄 SEC 10-K Deep Dive"
                        )
                        console.print(sec_panel)
                    else:
                        console.print("[yellow]No 10-K filing available for detailed analysis[/yellow]")
                else:
                    console.print("[yellow]No recent SEC filings found[/yellow]")
            else:
                console.print("[yellow]CIK not found for SEC analysis[/yellow]")

        except Exception as e:
            console.print(f"[dim]SEC analysis unavailable: {e}[/dim]")

        # ========== SECTION 10: INSIDER TRADING ANALYSIS ==========
        console.print("\n[bold cyan]👔 INSIDER TRADING ACTIVITY[/bold cyan]\n")

        try:
            insider_signal = insider_analyzer.analyze_insider_activity(symbol, lookback_days=30)

            # Signal emoji and color
            if insider_signal.signal == "bullish":
                signal_emoji = "🟢"
                signal_color = "green"
            elif insider_signal.signal == "bearish":
                signal_emoji = "🔴"
                signal_color = "red"
            else:
                signal_emoji = "⚪"
                signal_color = "yellow"

            insider_panel = Panel(
                Text.from_markup(f"""
{signal_emoji} [bold {signal_color}]{insider_signal.signal.upper()} SIGNAL[/bold {signal_color}]
Strength: {insider_signal.signal_strength:.0%} | Confidence: {insider_signal.confidence:.0%}

[bold]Transaction Summary (30 days):[/bold]
  Total Transactions:      {insider_signal.total_transactions_30d}
  Buys:                    [green]{insider_signal.buy_transactions}[/green]
  Sells:                   [red]{insider_signal.sell_transactions}[/red]
  Buy/Sell Ratio:          {insider_signal.buy_sell_ratio:.2f}
  Net Value:               [{'green' if insider_signal.net_value_30d > 0 else 'red'}]${abs(insider_signal.net_value_30d)/1e6:.2f}M[/{'green' if insider_signal.net_value_30d > 0 else 'red'}]

[bold]Key Insider Activity:[/bold]
  CEO:                     {insider_signal.ceo_activity}
  CFO:                     {insider_signal.cfo_activity}
  Directors:               {insider_signal.director_activity}

{f'[bold green]Bullish Signals:[/bold green]{chr(10)}' + chr(10).join(f"  ✅ {sig}" for sig in insider_signal.bullish_signals[:3]) if insider_signal.bullish_signals else ''}
{f'[bold red]Red Flags:[/bold red]{chr(10)}' + chr(10).join(f"  ⚠️  {flag}" for flag in insider_signal.red_flags[:3]) if insider_signal.red_flags else ''}
                """),
                border_style=signal_color,
                box=box.ROUNDED,
                title="👔 Smart Money Tracking"
            )
            console.print(insider_panel)

        except Exception as e:
            console.print(f"[dim]Insider trading analysis unavailable: {e}[/dim]")

        # ========== SECTION 11: INSTITUTIONAL HOLDINGS ==========
        console.print("\n[bold cyan]🏦 INSTITUTIONAL HOLDINGS (Smart Money)[/bold cyan]\n")

        try:
            inst_signal = institutional_tracker.analyze_institutional_activity(symbol)

            # Signal color
            inst_color = "green" if inst_signal.signal == "bullish" else "red" if inst_signal.signal == "bearish" else "yellow"
            inst_emoji = "🟢" if inst_signal.signal == "bullish" else "🔴" if inst_signal.signal == "bearish" else "⚪"

            inst_panel = Panel(
                Text.from_markup(f"""
{inst_emoji} [bold {inst_color}]{inst_signal.signal.upper()} SIGNAL[/bold {inst_color}]
Strength: {inst_signal.signal_strength:.0%} | Confidence: {inst_signal.confidence:.0%}

[bold]Holdings Summary:[/bold]
  Total Institutions:      {inst_signal.total_institutions}
  Total Shares:            {inst_signal.total_shares_held/1e6:.1f}M
  Total Value:             ${inst_signal.total_value_held/1e9:.2f}B
  Inst. Ownership:         {inst_signal.percent_institutional_ownership:.1f}%

[bold]Quarterly Changes:[/bold]
  New Positions:           [green]{inst_signal.institutions_added}[/green]
  Increased:               [green]{inst_signal.institutions_increased}[/green]
  Decreased:               [red]{inst_signal.institutions_decreased}[/red]
  Sold Out:                [red]{inst_signal.institutions_sold}[/red]
  Net Flow:                [{'green' if inst_signal.net_institutional_flow > 0 else 'red'}]${inst_signal.net_institutional_flow/1e6:+.0f}M[/{'green' if inst_signal.net_institutional_flow > 0 else 'red'}]

[bold]Top Holders:[/bold]
{chr(10).join(f"  {i+1}. {holder}" for i, holder in enumerate(inst_signal.top_holders[:3]))}

{f'[bold green]Top Buyers:[/bold green]{chr(10)}' + chr(10).join(f"  ✅ {buyer}" for buyer in inst_signal.top_buyers[:2]) if inst_signal.top_buyers and inst_signal.top_buyers != ["None"] else ''}
{f'[bold red]Signals:[/bold red]{chr(10)}' + chr(10).join(f"  • {sig}" for sig in inst_signal.bullish_signals[:2]) if inst_signal.bullish_signals else ''}
                """),
                border_style=inst_color,
                box=box.ROUNDED,
                title="🏦 Follow the Smart Money"
            )
            console.print(inst_panel)

        except Exception as e:
            console.print(f"[dim]Institutional analysis unavailable: {e}[/dim]")

        # ========== SECTION 12: SOCIAL MEDIA SENTIMENT ==========
        console.print("\n[bold cyan]📱 SOCIAL MEDIA SENTIMENT (Retail)[/bold cyan]\n")

        try:
            social_signal = social_analyzer.analyze_social_sentiment(symbol)

            # Signal color
            social_color = "green" if social_signal.signal == "bullish" else "red" if social_signal.signal == "bearish" else "yellow"
            social_emoji = "🟢" if social_signal.signal == "bullish" else "🔴" if social_signal.signal == "bearish" else "⚪"

            # Trending indicator
            trending_text = f"[bold yellow]🔥 TRENDING #{social_signal.trending_rank}[/bold yellow]" if social_signal.is_trending else ""

            social_panel = Panel(
                Text.from_markup(f"""
{social_emoji} [bold {social_color}]{social_signal.signal.upper()} SENTIMENT[/bold {social_color}] {trending_text}
Strength: {social_signal.signal_strength:.0%} | Confidence: {social_signal.confidence:.0%} | Buzz: {social_signal.buzz_score:.0f}/100

[bold]Mention Metrics:[/bold]
  24h Mentions:            {social_signal.total_mentions_24h}
  7d Mentions:             {social_signal.total_mentions_7d}
  Growth Rate:             [{'green' if social_signal.mention_growth_rate > 0 else 'red'}]{social_signal.mention_growth_rate:+.1f}%[/{'green' if social_signal.mention_growth_rate > 0 else 'red'}]
  Avg Sentiment:           [{'green' if social_signal.average_sentiment > 0 else 'red'}]{social_signal.average_sentiment:+.2f}[/{'green' if social_signal.average_sentiment > 0 else 'red'}]

[bold]Platform Breakdown:[/bold]
  Reddit (WSB):            {social_signal.reddit_mentions} posts ({social_signal.reddit_sentiment:+.2f})
  Twitter:                 {social_signal.twitter_mentions} posts ({social_signal.twitter_sentiment:+.2f})
  StockTwits:              {social_signal.stocktwits_mentions} posts ({social_signal.stocktwits_sentiment:+.2f})

[bold yellow]⚠️ Risk Indicators:[/bold yellow]
  Gamma Squeeze Risk:      {'[red]' if social_signal.wsb_gamma_squeeze_risk > 0.5 else '[yellow]'}{social_signal.wsb_gamma_squeeze_risk:.0%}{'[/red]' if social_signal.wsb_gamma_squeeze_risk > 0.5 else '[/yellow]'}
  Short Squeeze Prob:      {social_signal.short_squeeze_probability:.0%}
  Meme Stock:              {'[red]YES[/red]' if social_signal.meme_stock_indicator else 'NO'}

{f'[bold green]Bullish Signals:[/bold green]{chr(10)}' + chr(10).join(f"  🚀 {sig}" for sig in social_signal.bullish_signals[:3]) if social_signal.bullish_signals else ''}
                """),
                border_style=social_color,
                box=box.ROUNDED,
                title="📱 Reddit, Twitter, StockTwits"
            )
            console.print(social_panel)

        except Exception as e:
            console.print(f"[dim]Social sentiment analysis unavailable: {e}[/dim]")

        # ========== SECTION 13: KEY INSIGHTS ==========
        console.print("\n[bold cyan]💡 AI-GENERATED INSIGHTS[/bold cyan]\n")

        insights_panel = Panel(
            Text.from_markup(f"""
[bold green]Opportunities:[/bold green]
{chr(10).join(f"  • {opp}" for opp in intel.opportunities[:3]) if intel.opportunities else '  • None identified'}

[bold yellow]Risks:[/bold yellow]
{chr(10).join(f"  • {risk}" for risk in intel.risks[:3]) if intel.risks else '  • None identified'}

[bold cyan]Key Insights:[/bold cyan]
{chr(10).join(f"  • {insight}" for insight in intel.key_insights[:5]) if intel.key_insights else '  • Analysis in progress'}
            """),
            border_style="blue",
            box=box.ROUNDED
        )
        console.print(insights_panel)

        # ========== FOOTER ==========
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[dim]Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        console.print(f"[dim]Data provided by Yahoo Finance, AI models, and market intelligence[/dim]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error analyzing {symbol}:[/bold red] {e}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def main():
    """Main interactive loop."""
    console.clear()

    # Welcome screen
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold white]  🚀 INTERACTIVE STOCK ANALYZER  🚀[/bold white]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    console.print("[yellow]Type any stock ticker to get comprehensive analysis![/yellow]")
    console.print("[dim]Examples: AAPL, TSLA, GOOGL, MSFT, NVDA[/dim]\n")

    while True:
        try:
            symbol = Prompt.ask("\n[bold green]Enter stock ticker (or 'exit' to quit)[/bold green]")

            if symbol.lower() in ['exit', 'quit', 'q']:
                console.print("\n[yellow]Thank you for using Interactive Stock Analyzer![/yellow]\n")
                break

            if not symbol:
                continue

            # Analyze the stock
            analyze_stock(symbol)

            # Ask if want to analyze another
            console.print("\n[dim]Press Enter to analyze another stock...[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Exiting...[/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()
