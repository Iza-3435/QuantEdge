#!/usr/bin/env python3
"""
ULTRA-DENSE INSTITUTIONAL TERMINAL
Bloomberg-level comprehensive analysis combining:
- AI Predictions (LSTM + Transformer + Attention)
- Advanced Sentiment (FinBERT + SEC + News)
- ML Engine (XGBoost + Random Forest + Gradient Boosting)
- Quality Scores (Piotroski F-Score + Altman Z-Score)
- Fundamentals & Financials (P/E, EPS, Margins, Revenue)
- Technical Analysis (RSI, MACD, Moving Averages, Chart)
- Risk Management (VaR, CVaR, Position Sizing, Sharpe)
- Factor Models (Fama-French Multi-Factor)
- Options Pricing (Black-Scholes + Greeks)
- Market Comparison (S&P 500, Sector, Peers)
- News Feed (Real-time with sentiment)
- Institutional Ownership & Insider Trading
- Data Caching (15-minute TTL for performance)
"""

import sys
import os
import warnings
from typing import Dict, List, Optional, Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
from rich.status import Status

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.advanced_lstm import AdvancedLSTMPredictor
from src.advanced_sentiment import AdvancedSentimentAnalyzer
from src.risk_management import RiskManager
from src.factor_models import FactorAnalyzer
from src.options_pricing import OptionsCalculator
from src.ml_engine import AdvancedMLScorer
from src.quality_scores import calculate_piotroski_score, calculate_altman_z_score
from src.data_cache import DataCache

console = Console()

THEME = {
    'header_bg': 'on grey23',
    'row_even': 'on grey15',
    'row_odd': 'on grey11',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}


def show_banner() -> None:
    """Display institutional terminal banner"""
    banner = Text()
    banner.append("ULTRA-DENSE INSTITUTIONAL TERMINAL\n", style="bold white")
    banner.append("Bloomberg-Level Analytics with ML Intelligence\n\n", style="bright_black")
    banner.append("AI Predictions", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Sentiment", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Risk", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Factors", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Options", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Fundamentals", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("Technicals", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("ML Quality Scores", style="white")
    banner.append(" │ ", style="bright_black")
    banner.append("News", style="white")

    console.print(Panel(banner, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg']))
    console.print()


def fetch_comprehensive_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch all data needed for ultra-dense analysis"""
    # Check cache first
    cache_key = f"institutional_terminal_{symbol}"
    cached_data = DataCache.get(cache_key)
    if cached_data:
        return cached_data

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='1y')

        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        returns = hist['Close'].pct_change().dropna()

        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else current_price

        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()

        earnings_date = 'N/A'
        ex_dividend_date = 'N/A'
        if 'earningsDate' in info and info['earningsDate']:
            try:
                from datetime import datetime
                ed = info['earningsDate']
                if isinstance(ed, list) and len(ed) > 0:
                    earnings_date = datetime.fromtimestamp(ed[0]).strftime('%Y-%m-%d')
                elif isinstance(ed, (int, float)):
                    earnings_date = datetime.fromtimestamp(ed).strftime('%Y-%m-%d')
            except:
                pass

        if 'exDividendDate' in info and info['exDividendDate']:
            try:
                from datetime import datetime
                exd = info['exDividendDate']
                if isinstance(exd, (int, float)):
                    ex_dividend_date = datetime.fromtimestamp(exd).strftime('%Y-%m-%d')
            except:
                pass

        result = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'price': current_price,
            'prev_close': prev_close,
            'change': current_price - prev_close,
            'change_pct': ((current_price - prev_close) / prev_close) * 100,
            'day_high': info.get('dayHigh', current_price),
            'day_low': info.get('dayLow', current_price),
            'week_52_high': info.get('fiftyTwoWeekHigh', current_price),
            'week_52_low': info.get('fiftyTwoWeekLow', current_price),
            'volume': hist['Volume'].iloc[-1],
            'avg_volume': info.get('averageVolume', 0),
            'hist': hist,
            'prices': hist['Close'],
            'volumes': hist['Volume'],
            'returns': returns,
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'book_value': info.get('bookValue', 0) * info.get('sharesOutstanding', 0) if info.get('bookValue') else 0,
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'ev_to_revenue': info.get('enterpriseToRevenue', 0),
            'ev_to_ebitda': info.get('enterpriseToEbitda', 0),
            'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0,
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'roa': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
            'revenue': info.get('totalRevenue', 0),
            'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
            'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
            'eps': info.get('trailingEps', 0),
            'forward_eps': info.get('forwardEps', 0),
            'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
            'current_ratio': info.get('currentRatio', 0),
            'quick_ratio': info.get('quickRatio', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'payout_ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
            'beta': info.get('beta', 1.0),
            'short_ratio': info.get('shortRatio', 0),
            'short_percent': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0,
            'inst_ownership': info.get('heldPercentInstitutions', 0) * 100 if info.get('heldPercentInstitutions') else 0,
            'insider_ownership': info.get('heldPercentInsiders', 0) * 100 if info.get('heldPercentInsiders') else 0,
            'analyst_target': info.get('targetMeanPrice', 0),
            'analyst_rating': info.get('recommendationKey', 'N/A'),
            'num_analysts': info.get('numberOfAnalystOpinions', 0),
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'macd': macd.iloc[-1] if not macd.empty else 0,
            'macd_signal': signal.iloc[-1] if not signal.empty else 0,
            'returns_1m': ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21] * 100) if len(hist) > 21 else 0,
            'returns_3m': ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63] * 100) if len(hist) > 63 else 0,
            'returns_1y': ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
            'earnings_date': earnings_date,
            'ex_dividend_date': ex_dividend_date,
            'dividend_rate': info.get('dividendRate', 0)
        }

        # Cache the result for 15 minutes
        DataCache.set(cache_key, result)
        return result

    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        return None


def create_lstm_panel(predictions: Dict[str, Any], current_price: float) -> Panel:
    """Create LSTM prediction panel"""
    text = Text()
    text.append("LSTM PREDICTIONS\n\n", style="bold white")

    preds = predictions['predictions']
    intervals = predictions['confidence_intervals']

    for horizon in ['7d', '14d', '30d']:
        pred = preds.get(horizon, current_price)
        low, high = intervals.get(horizon, (current_price, current_price))
        change = ((pred - current_price) / current_price) * 100

        horizon_label = horizon.replace('d', '-Day')
        text.append(f"{horizon_label}:\n", style="bold white")
        text.append(f"  Predicted: ${pred:.2f} ", style="white")

        change_color = "green" if change > 0 else "red"
        text.append(f"({change:+.1f}%)\n", style=change_color)
        text.append(f"  Range: ${low:.2f} - ${high:.2f}\n\n", style="bright_black")

    trend = predictions['trend']
    trend_color = "bright_green" if "UP" in trend else "red" if "DOWN" in trend else "white"
    text.append("Trend: ", style="white")
    text.append(f"{trend}\n", style=f"bold {trend_color}")

    vol_regime = predictions.get('volatility_regime', 'NORMAL')
    text.append(f"Volatility: {vol_regime}", style="bright_black")

    return Panel(text, title="[bold white]AI PREDICTIONS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_sentiment_panel(sentiment: Dict[str, Any]) -> Panel:
    """Create advanced sentiment panel"""
    text = Text()
    text.append("ADVANCED SENTIMENT\n\n", style="bold white")

    overall = sentiment['overall_score']
    label = sentiment['sentiment_label']

    label_color = "bright_green" if "POSITIVE" in label else "red" if "NEGATIVE" in label else "white"
    text.append("Overall: ", style="white")
    text.append(f"{label}\n", style=f"bold {label_color}")
    text.append(f"Score: {overall:+.1f}\n\n", style="white")

    text.append("Sources:\n", style="bold white")
    text.append(f"  News: {sentiment['news_sentiment']:+.1f} ({sentiment['news_count']} articles)\n", style="white")
    text.append(f"  SEC Filings: {sentiment['sec_filing_sentiment']:+.1f} ({sentiment['sec_filings_count']} filings)\n\n", style="white")

    if sentiment.get('risk_factors'):
        text.append("Risk Factors:\n", style="bold white")
        for risk in sentiment['risk_factors'][:3]:
            text.append(f"  • {risk}\n", style="red")

    text.append(f"\nTrend: {sentiment['sentiment_trend']}", style="bright_black")

    return Panel(text, title="[bold white]SENTIMENT ANALYSIS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_risk_panel(risk_metrics: Dict[str, Any]) -> Panel:
    """Create risk management panel"""
    text = Text()
    text.append("RISK METRICS\n\n", style="bold white")

    var_95 = risk_metrics['value_at_risk_95']
    cvar_95 = risk_metrics['cvar_95']
    text.append(f"VaR (95%): {var_95:.2f}%\n", style="white")
    text.append(f"CVaR (95%): {cvar_95:.2f}%\n\n", style="white")

    sharpe = risk_metrics['sharpe_ratio']
    sharpe_color = "bright_green" if sharpe > 1.5 else "green" if sharpe > 1.0 else "white"
    text.append("Sharpe Ratio: ", style="white")
    text.append(f"{sharpe:.2f}\n", style=sharpe_color)

    text.append(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}\n\n", style="white")

    drawdown = risk_metrics['maximum_drawdown']
    dd_color = "green" if drawdown > -10 else "red" if drawdown < -25 else "white"
    text.append("Max Drawdown: ", style="white")
    text.append(f"{drawdown:.1f}%\n", style=dd_color)

    text.append(f"Volatility: {risk_metrics['volatility']:.1f}%", style="bright_black")

    return Panel(text, title="[bold white]RISK ANALYTICS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_factor_panel(factors: Dict[str, Any]) -> Panel:
    """Create factor model panel"""
    text = Text()
    text.append("FACTOR EXPOSURES\n\n", style="bold white")

    exposures = [
        ("Market Beta", factors['market_beta'], 1.0),
        ("Value", factors['value_exposure'], 0.0),
        ("Momentum", factors['momentum_exposure'], 0.0),
        ("Quality", factors['quality_exposure'], 0.0),
        ("Low Vol", factors['low_vol_exposure'], 0.0),
    ]

    for name, value, neutral in exposures:
        color = "green" if value > neutral else "red" if value < neutral else "white"
        text.append(f"{name}: ", style="white")
        text.append(f"{value:+.2f}\n", style=color)

    text.append(f"\nFactor Score: ", style="bold white")
    score = factors['factor_score']
    score_color = "bright_green" if score > 70 else "green" if score > 50 else "white"
    text.append(f"{score:.0f}/100", style=score_color)

    return Panel(text, title="[bold white]FACTOR MODELS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_options_panel(options: Dict[str, Any]) -> Panel:
    """Create options pricing panel"""
    text = Text()
    text.append("OPTIONS PRICING\n\n", style="bold white")

    text.append("Call Option:\n", style="bold white")
    text.append(f"  Price: ${options['call_price']:.2f}\n", style="white")
    text.append(f"  Delta: {options['delta_call']:.3f}\n\n", style="white")

    text.append("Put Option:\n", style="bold white")
    text.append(f"  Price: ${options['put_price']:.2f}\n", style="white")
    text.append(f"  Delta: {options['delta_put']:.3f}\n\n", style="white")

    text.append("Greeks:\n", style="bold white")
    text.append(f"  Gamma: {options['gamma']:.4f}\n", style="white")
    text.append(f"  Vega: {options['vega']:.2f}\n", style="white")
    text.append(f"  Theta: ${options['theta_call']:.2f}/day", style="white")

    return Panel(text, title="[bold white]OPTIONS PRICING[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_position_panel(position: Dict[str, Any]) -> Panel:
    """Create position sizing panel"""
    text = Text()
    text.append("POSITION SIZING\n\n", style="bold white")

    text.append(f"Optimal Shares: {position['optimal_shares']}\n", style="white")
    text.append(f"Position Value: ${position['position_value']:,.0f}\n", style="white")
    text.append(f"Portfolio Weight: {position['portfolio_weight']:.1f}%\n\n", style="white")

    text.append(f"Risk Contribution: {position['risk_contribution']:.2f}%\n", style="white")
    text.append(f"Kelly Fraction: {position['kelly_fraction']:.1%}", style="bright_black")

    return Panel(text, title="[bold white]POSITION SIZING[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_fundamentals_panel(stock: Dict[str, Any]) -> Panel:
    """Create fundamentals data panel"""
    text = Text()
    text.append("FUNDAMENTALS\n\n", style="bold white")

    text.append("Valuation:\n", style="bold white")
    text.append(f"  P/E Ratio: {stock['pe_ratio']:.2f}\n", style="white")
    text.append(f"  Forward P/E: {stock['forward_pe']:.2f}\n", style="white")
    text.append(f"  PEG Ratio: {stock['peg_ratio']:.2f}\n", style="white")
    text.append(f"  P/B Ratio: {stock['price_to_book']:.2f}\n", style="white")
    text.append(f"  P/S Ratio: {stock['price_to_sales']:.2f}\n", style="white")
    if stock['ev_to_revenue'] > 0:
        text.append(f"  EV/Revenue: {stock['ev_to_revenue']:.2f}\n", style="white")
    if stock['ev_to_ebitda'] > 0:
        text.append(f"  EV/EBITDA: {stock['ev_to_ebitda']:.2f}\n", style="white")
    text.append(f"\n")

    text.append("Financials:\n", style="bold white")
    text.append(f"  EPS: ${stock['eps']:.2f}\n", style="white")
    if stock['forward_eps'] > 0:
        text.append(f"  Forward EPS: ${stock['forward_eps']:.2f}\n", style="white")
    if stock['revenue'] > 0:
        text.append(f"  Revenue: ${stock['revenue']/1e9:.2f}B\n", style="white")
    text.append(f"\n")

    text.append("Profitability:\n", style="bold white")
    margin_color = "green" if stock['profit_margin'] > 15 else "white"
    text.append(f"  Profit Margin: {stock['profit_margin']:.1f}%\n", style=margin_color)
    text.append(f"  Operating Margin: {stock['operating_margin']:.1f}%\n", style="white")
    text.append(f"  ROE: {stock['roe']:.1f}%\n", style="white")
    text.append(f"  ROA: {stock['roa']:.1f}%\n\n", style="white")

    text.append("Growth:\n", style="bold white")
    rev_color = "green" if stock['revenue_growth'] > 10 else "white"
    text.append(f"  Revenue Growth: {stock['revenue_growth']:+.1f}%\n", style=rev_color)
    earn_color = "green" if stock['earnings_growth'] > 10 else "white"
    text.append(f"  Earnings Growth: {stock['earnings_growth']:+.1f}%\n", style=earn_color)
    text.append(f"\n")

    text.append("Financial Health:\n", style="bold white")
    debt_color = "green" if stock['debt_equity'] < 0.5 else "red" if stock['debt_equity'] > 2.0 else "white"
    text.append(f"  Debt/Equity: {stock['debt_equity']:.2f}\n", style=debt_color)
    text.append(f"  Current Ratio: {stock['current_ratio']:.2f}\n", style="white")
    text.append(f"  Quick Ratio: {stock['quick_ratio']:.2f}", style="white")

    return Panel(text, title="[bold white]FUNDAMENTALS & FINANCIALS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_technicals_panel(stock: Dict[str, Any]) -> Panel:
    """Create technical indicators panel"""
    text = Text()
    text.append("TECHNICAL INDICATORS\n\n", style="bold white")

    rsi = stock['rsi']
    rsi_color = "red" if rsi < 30 else "green" if rsi > 70 else "white"
    text.append("RSI (14): ", style="white")
    text.append(f"{rsi:.1f}", style=rsi_color)
    rsi_label = " (Oversold)" if rsi < 30 else " (Overbought)" if rsi > 70 else ""
    text.append(f"{rsi_label}\n\n", style=rsi_color)

    text.append("Moving Averages:\n", style="bold white")
    sma_20_signal = "above" if stock['price'] > stock['sma_20'] else "below"
    sma_20_color = "green" if stock['price'] > stock['sma_20'] else "red"
    text.append(f"  SMA 20: ${stock['sma_20']:.2f} ({sma_20_signal})\n", style=sma_20_color)

    sma_50_signal = "above" if stock['price'] > stock['sma_50'] else "below"
    sma_50_color = "green" if stock['price'] > stock['sma_50'] else "red"
    text.append(f"  SMA 50: ${stock['sma_50']:.2f} ({sma_50_signal})\n", style=sma_50_color)

    sma_200_signal = "above" if stock['price'] > stock['sma_200'] else "below"
    sma_200_color = "green" if stock['price'] > stock['sma_200'] else "red"
    text.append(f"  SMA 200: ${stock['sma_200']:.2f} ({sma_200_signal})\n\n", style=sma_200_color)

    macd_signal = "Bullish" if stock['macd'] > stock['macd_signal'] else "Bearish"
    macd_color = "green" if stock['macd'] > stock['macd_signal'] else "red"
    text.append("MACD: ", style="white")
    text.append(f"{macd_signal}\n", style=macd_color)
    text.append(f"  MACD: {stock['macd']:.2f}\n", style="white")
    text.append(f"  Signal: {stock['macd_signal']:.2f}", style="bright_black")

    return Panel(text, title="[bold white]TECHNICAL ANALYSIS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_market_data_panel(stock: Dict[str, Any]) -> Panel:
    """Create market data panel"""
    text = Text()
    text.append("MARKET DATA\n\n", style="bold white")

    text.append(f"Day Range: ${stock['day_low']:.2f} - ${stock['day_high']:.2f}\n", style="white")
    text.append(f"52W Range: ${stock['week_52_low']:.2f} - ${stock['week_52_high']:.2f}\n\n", style="white")

    volume_color = "green" if stock['volume'] > stock['avg_volume'] else "white"
    text.append(f"Volume: {stock['volume']:,.0f}\n", style=volume_color)
    text.append(f"Avg Volume: {stock['avg_volume']:,.0f}\n\n", style="white")

    text.append(f"Market Cap: ${stock['market_cap']/1e9:.2f}B\n", style="white")
    text.append(f"Enterprise Val: ${stock['enterprise_value']/1e9:.2f}B\n\n", style="white")

    text.append(f"Beta: {stock['beta']:.2f}\n", style="white")
    text.append(f"Short %: {stock['short_percent']:.1f}%", style="bright_black")

    return Panel(text, title="[bold white]MARKET DATA[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_ownership_panel(stock: Dict[str, Any]) -> Panel:
    """Create ownership panel"""
    text = Text()
    text.append("OWNERSHIP\n\n", style="bold white")

    text.append(f"Institutional: {stock['inst_ownership']:.1f}%\n", style="white")
    text.append(f"Insider: {stock['insider_ownership']:.1f}%\n\n", style="white")

    text.append("Analyst Coverage:\n", style="bold white")
    text.append(f"  Rating: {stock['analyst_rating'].upper()}\n", style="white")
    text.append(f"  Target: ${stock['analyst_target']:.2f}\n", style="white")
    upside = ((stock['analyst_target'] - stock['price']) / stock['price']) * 100
    upside_color = "green" if upside > 0 else "red"
    text.append(f"  Upside: {upside:+.1f}%\n", style=upside_color)
    text.append(f"  # Analysts: {stock['num_analysts']}\n\n", style="bright_black")

    text.append("Upcoming Events:\n", style="bold white")
    text.append(f"  Earnings: {stock['earnings_date']}\n", style="white")
    if stock['dividend_yield'] > 0:
        text.append(f"  Ex-Dividend: {stock['ex_dividend_date']}\n", style="white")
        text.append(f"  Div Amount: ${stock['dividend_rate']:.2f}", style="bright_black")

    return Panel(text, title="[bold white]OWNERSHIP & ANALYSTS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def fetch_news(symbol: str, company_name: str = None) -> List[Dict[str, Any]]:
    """Fetch latest news for stock"""
    import requests
    from datetime import datetime, timedelta

    news_api_key = os.getenv('NEWSAPI_KEY')
    if not news_api_key:
        return []

    try:
        query = company_name if company_name else symbol
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': news_api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 8,
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            news_list = []
            for article in articles[:8]:
                news_list.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published': article.get('publishedAt', '')[:10],
                    'url': article.get('url', '')
                })
            return news_list
    except Exception:
        pass

    return []


def create_news_panel(symbol: str, company_name: str) -> Panel:
    """Create news feed panel"""
    text = Text()
    text.append("LATEST NEWS\n\n", style="bold white")

    news = fetch_news(symbol, company_name)

    if news:
        for i, article in enumerate(news[:8], 1):
            text.append(f"{i}. ", style="bright_black")
            text.append(f"{article['title'][:100]}..." if len(article['title']) > 100 else article['title'],
                       style="white")
            text.append(f"  ({article['source']} • {article['published']})\n", style="bright_black")
    else:
        text.append("No recent news available", style="bright_black")

    return Panel(text, title="[bold white]NEWS FEED - TOP HEADLINES[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_ml_quality_panel(stock: Dict[str, Any]) -> Panel:
    """Create ML predictions and quality scores panel"""
    text = Text()
    text.append("ML & QUALITY SCORES\n\n", style="bold white")

    # ML Engine Score
    try:
        ml_scorer = AdvancedMLScorer()
        ml_result = ml_scorer.score_stock(stock)

        if ml_result and hasattr(ml_result, 'score'):
            text.append("ML Ensemble Score:\n", style="bold white")
            score = ml_result.score
            score_color = "green" if score >= 70 else "yellow" if score >= 50 else "red"
            text.append(f"  Score: {score:.1f}/100\n", style=score_color)
            text.append(f"  Signal: {ml_result.signal}\n", style="white")
            text.append(f"  Confidence: {ml_result.confidence:.1%}\n", style="white")
            text.append(f"  Models: {ml_result.model_agreement:.0%} agreement\n\n", style="bright_black")
        else:
            text.append("ML Score: Not available\n\n", style="bright_black")
    except Exception:
        text.append("ML Score: Not available\n\n", style="bright_black")

    # Piotroski Score
    try:
        piotroski = calculate_piotroski_score(stock)
        text.append("Piotroski F-Score:\n", style="bold white")
        score_color = piotroski.get('color', 'white')
        text.append(f"  Score: {piotroski['score']}/9\n", style=score_color)
        text.append(f"  Rating: {piotroski['rating']}\n\n", style="white")
    except Exception:
        text.append("Piotroski F-Score: Not available\n\n", style="bright_black")

    # Altman Z-Score
    try:
        altman = calculate_altman_z_score(stock)
        text.append("Altman Z-Score:\n", style="bold white")
        score = altman.get('z_score', 0)
        interpretation = altman.get('interpretation', 'Unknown')

        if score > 3.0:
            z_color = "green"
        elif score > 1.8:
            z_color = "yellow"
        else:
            z_color = "red"

        text.append(f"  Z-Score: {score:.2f}\n", style=z_color)
        text.append(f"  Risk: {interpretation}", style="white")
    except Exception:
        text.append("Altman Z-Score: Not available", style="bright_black")

    return Panel(text, title="[bold white]ML PREDICTIONS & QUALITY METRICS[/bold white]",
                 border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def main() -> None:
    """Main institutional terminal application"""
    import sys

    console.clear()
    show_banner()

    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = Prompt.ask("[white]Enter stock symbol[/white]", default="AAPL").upper()

    console.print()

    with Status("[white]Loading comprehensive analysis...[/white]", console=console, spinner="dots"):
        stock = fetch_comprehensive_data(symbol)

        if not stock:
            console.print("[red]Error: Could not fetch data[/red]\n")
            return

        lstm_predictor = AdvancedLSTMPredictor(sequence_length=60, epochs=30)
        if lstm_predictor.train(stock['prices'], stock['volumes']):
            lstm_result = lstm_predictor.predict(stock['prices'], stock['volumes'], [7, 14, 30])
        else:
            lstm_result = None

        sentiment_analyzer = AdvancedSentimentAnalyzer()
        sentiment_result = sentiment_analyzer.analyze_comprehensive(symbol, stock['name'])

        risk_manager = RiskManager(portfolio_value=100000)
        risk_metrics = risk_manager.calculate_comprehensive_risk(stock['returns'])

        factor_analyzer = FactorAnalyzer()
        factor_exposures = factor_analyzer.calculate_factor_exposures(
            price=stock['price'],
            market_cap=stock['market_cap'],
            book_value=stock['book_value'],
            returns_1y=stock['returns_1y'],
            roe=stock['roe'],
            debt_equity=stock['debt_equity'],
            volatility=risk_metrics.volatility
        )

        options_calc = OptionsCalculator()
        options_result = options_calc.black_scholes(
            spot_price=stock['price'],
            strike_price=stock['price'],
            time_to_expiry=30/365,
            volatility=risk_metrics.volatility / 100,
            dividend_yield=stock['dividend_yield'] / 100
        )

        position_size = risk_manager.calculate_position_size(
            stock_price=stock['price'],
            volatility=risk_metrics.volatility / 100,
            expected_return=0.15 if lstm_result and lstm_result.trend == "UPTREND" else 0.08,
            confidence=0.7
        )

    console.print("[green]✓ Analysis complete[/green]\n")

    console.clear()
    show_banner()

    header_text = Text()
    header_text.append(f"{stock['symbol']} - {stock['name']}", style="bold white")
    header_text.append(" │ ", style="bright_black")
    header_text.append(f"{stock['sector']}", style="white")
    header_text.append(" │ ", style="bright_black")
    change_color = "bright_green" if stock['change'] > 0 else "red"
    header_text.append(f"${stock['price']:.2f} ", style=f"bold {change_color}")
    header_text.append(f"({stock['change']:+.2f}, {stock['change_pct']:+.2f}%)", style=change_color)
    header_text.append(" │ ", style="bright_black")
    header_text.append(f"1M: {stock['returns_1m']:+.1f}% ", style="green" if stock['returns_1m'] > 0 else "red")
    header_text.append(f"3M: {stock['returns_3m']:+.1f}% ", style="green" if stock['returns_3m'] > 0 else "red")
    header_text.append(f"1Y: {stock['returns_1y']:+.1f}%", style="green" if stock['returns_1y'] > 0 else "red")

    console.print(Panel(header_text, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()

    if lstm_result:
        lstm_data = {
            'predictions': lstm_result.predictions,
            'confidence_intervals': lstm_result.confidence_intervals,
            'trend': lstm_result.trend,
            'volatility_regime': lstm_result.volatility_regime
        }
        console.print(Columns([
            create_lstm_panel(lstm_data, stock['price']),
            create_sentiment_panel({
                'overall_score': sentiment_result.overall_score,
                'sentiment_label': sentiment_result.sentiment_label,
                'news_sentiment': sentiment_result.news_sentiment,
                'sec_filing_sentiment': sentiment_result.sec_filing_sentiment,
                'news_count': sentiment_result.news_count,
                'sec_filings_count': sentiment_result.sec_filings_count,
                'sentiment_trend': sentiment_result.sentiment_trend,
                'risk_factors': sentiment_result.risk_factors
            })
        ]))
        console.print()

    console.print(Columns([
        create_risk_panel({
            'value_at_risk_95': risk_metrics.value_at_risk_95,
            'cvar_95': risk_metrics.cvar_95,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'sortino_ratio': risk_metrics.sortino_ratio,
            'maximum_drawdown': risk_metrics.maximum_drawdown,
            'volatility': risk_metrics.volatility
        }),
        create_factor_panel({
            'market_beta': factor_exposures.market_beta,
            'value_exposure': factor_exposures.value_exposure,
            'momentum_exposure': factor_exposures.momentum_exposure,
            'quality_exposure': factor_exposures.quality_exposure,
            'low_vol_exposure': factor_exposures.low_vol_exposure,
            'factor_score': factor_exposures.factor_score
        })
    ]))
    console.print()

    console.print(Columns([
        create_options_panel({
            'call_price': options_result.call_price,
            'put_price': options_result.put_price,
            'delta_call': options_result.delta_call,
            'delta_put': options_result.delta_put,
            'gamma': options_result.gamma,
            'vega': options_result.vega,
            'theta_call': options_result.theta_call
        }),
        create_position_panel({
            'optimal_shares': position_size.optimal_shares,
            'position_value': position_size.position_value,
            'portfolio_weight': position_size.portfolio_weight,
            'risk_contribution': position_size.risk_contribution,
            'kelly_fraction': position_size.kelly_fraction
        })
    ]))
    console.print()

    console.print(Columns([
        create_fundamentals_panel(stock),
        create_technicals_panel(stock)
    ]))
    console.print()

    console.print(Columns([
        create_market_data_panel(stock),
        create_ownership_panel(stock)
    ]))
    console.print()

    console.print(Columns([
        create_news_panel(stock['symbol'], stock['name']),
        create_ml_quality_panel(stock)
    ]))
    console.print()

    console.print("[bright_black]Institutional Terminal • Ultra-Dense Bloomberg-Level Analytics • ML Quality Scores • Production Grade • Not Financial Advice[/bright_black]", justify="center")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[white]Cancelled by user[/white]\n")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        import traceback
        traceback.print_exc()
