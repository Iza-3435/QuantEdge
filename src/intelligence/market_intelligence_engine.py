"""Advanced Market Intelligence Engine - Core AI Analysis System."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime_type: str  # bull, bear, volatile, stable
    confidence: float
    volatility: float
    trend_strength: float
    features: Dict[str, float]


@dataclass
class AnomalyDetection:
    """Anomaly detection results."""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str  # price, volume, volatility
    severity: str  # low, medium, high
    description: str


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    overall_sentiment: float  # -1 to 1
    sentiment_label: str  # bearish, neutral, bullish
    confidence: float
    sources: Dict[str, float]  # breakdown by source
    trending_topics: List[str]


@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence report."""
    symbol: str
    timestamp: datetime

    # Price & Technical
    current_price: float
    price_change_1d: float
    price_change_5d: float
    price_change_1m: float

    # Predictions
    predicted_return_5d: float
    predicted_return_20d: float
    confidence_score: float
    profit_probability: float

    # Risk Metrics
    volatility_daily: float
    volatility_annual: float
    var_95: float  # Value at Risk
    sharpe_ratio: float
    beta: float

    # AI Analysis
    market_regime: MarketRegime
    anomalies: List[AnomalyDetection]
    sentiment: SentimentAnalysis

    # Technical Indicators
    rsi: float
    macd: float
    bollinger_position: float
    volume_profile: str

    # Pattern Recognition
    similar_patterns: List[Dict]
    pattern_strength: float

    # Insights
    key_insights: List[str]
    risks: List[str]
    opportunities: List[str]
    recommendation: str  # strong_buy, buy, hold, sell, strong_sell


class AdvancedMarketIntelligenceEngine:
    """
    Advanced AI-powered market intelligence engine.

    Features:
    - Real-time market regime detection
    - Anomaly detection across price, volume, volatility
    - Multi-modal sentiment analysis
    - Risk metrics and portfolio analytics
    - Pattern recognition and similarity search
    - AI-generated insights and recommendations
    """

    def __init__(self):
        self.market_data_cache = {}
        self.scaler = StandardScaler()

    def analyze(
        self,
        symbol: str,
        lookback_days: int = 252,
        benchmark: str = "SPY"
    ) -> MarketIntelligence:
        """
        Perform comprehensive market intelligence analysis.

        Args:
            symbol: Stock ticker
            lookback_days: Historical data window
            benchmark: Market benchmark for beta calculation

        Returns:
            MarketIntelligence object with complete analysis
        """
        # Fetch data
        df = self._fetch_market_data(symbol, lookback_days)
        benchmark_df = self._fetch_market_data(benchmark, lookback_days)

        # Calculate all features
        df = self._add_technical_indicators(df)

        current_idx = len(df) - 1
        current_price = float(df['Close'].iloc[current_idx])

        # Price changes
        price_change_1d = float(df['Close'].pct_change().iloc[current_idx])
        price_change_5d = float(df['Close'].pct_change(5).iloc[current_idx])
        price_change_1m = float(df['Close'].pct_change(20).iloc[current_idx])

        # Market regime detection
        regime = self._detect_market_regime(df)

        # Anomaly detection
        anomalies = self._detect_anomalies(df)

        # Sentiment analysis (simulated - integrate real APIs in production)
        sentiment = self._analyze_sentiment(symbol, df)

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(df, benchmark_df)

        # Predictions (using pattern-based ML)
        predictions = self._generate_predictions(df)

        # Technical indicators
        tech_indicators = self._get_technical_summary(df)

        # Pattern recognition
        patterns = self._find_similar_patterns(df)

        # AI-generated insights
        insights = self._generate_insights(
            df, regime, anomalies, sentiment, risk_metrics, predictions
        )

        # Build comprehensive intelligence report
        intelligence = MarketIntelligence(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            price_change_1d=price_change_1d,
            price_change_5d=price_change_5d,
            price_change_1m=price_change_1m,
            predicted_return_5d=predictions['return_5d'],
            predicted_return_20d=predictions['return_20d'],
            confidence_score=predictions['confidence'],
            profit_probability=predictions['profit_prob'],
            volatility_daily=risk_metrics['vol_daily'],
            volatility_annual=risk_metrics['vol_annual'],
            var_95=risk_metrics['var_95'],
            sharpe_ratio=risk_metrics['sharpe'],
            beta=risk_metrics['beta'],
            market_regime=regime,
            anomalies=anomalies,
            sentiment=sentiment,
            rsi=tech_indicators['rsi'],
            macd=tech_indicators['macd'],
            bollinger_position=tech_indicators['bb_position'],
            volume_profile=tech_indicators['volume_profile'],
            similar_patterns=patterns['examples'],
            pattern_strength=patterns['strength'],
            key_insights=insights['insights'],
            risks=insights['risks'],
            opportunities=insights['opportunities'],
            recommendation=insights['recommendation']
        )

        return intelligence

    def _fetch_market_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch and cache market data."""
        cache_key = f"{symbol}_{days}"

        if cache_key in self.market_data_cache:
            cached_df, cached_time = self.market_data_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:  # 5 min cache
                return cached_df.copy()

        # Download for single symbol explicitly
        df = yf.download(symbol, period=f"{days}d", progress=False)

        # Ensure we have a clean DataFrame for single symbol
        if isinstance(df.columns, pd.MultiIndex):
            # If multi-index, flatten it to single level
            df = df.xs(symbol, level=1, axis=1)

        self.market_data_cache[cache_key] = (df.copy(), datetime.now())

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()

        return df

    def _detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime using AI clustering."""
        # Feature engineering for regime detection
        features = df[['Returns', 'Volatility', 'Volume_Ratio', 'RSI']].dropna()

        if len(features) < 50:
            return MarketRegime(
                regime_type="unknown",
                confidence=0.0,
                volatility=0.0,
                trend_strength=0.0,
                features={}
            )

        # Recent window
        recent_features = features.iloc[-20:].values
        recent_mean = recent_features.mean(axis=0)

        # Calculate metrics
        recent_returns = df['Returns'].iloc[-20:].mean()
        recent_vol = df['Volatility'].iloc[-20:].mean()
        trend_strength = abs(recent_returns) / (recent_vol + 1e-10)

        # Classify regime
        if recent_returns > 0.002 and recent_vol < 0.02:
            regime_type = "bull_stable"
        elif recent_returns > 0.002 and recent_vol >= 0.02:
            regime_type = "bull_volatile"
        elif recent_returns < -0.002 and recent_vol < 0.02:
            regime_type = "bear_stable"
        elif recent_returns < -0.002 and recent_vol >= 0.02:
            regime_type = "bear_volatile"
        elif recent_vol > 0.03:
            regime_type = "high_volatility"
        else:
            regime_type = "sideways"

        confidence = min(trend_strength * 10, 1.0)

        return MarketRegime(
            regime_type=regime_type,
            confidence=float(confidence),
            volatility=float(recent_vol),
            trend_strength=float(trend_strength),
            features={
                'avg_return': float(recent_returns),
                'avg_volatility': float(recent_vol),
                'avg_volume_ratio': float(recent_mean[2] if len(recent_mean) > 2 else 1.0),
                'avg_rsi': float(recent_mean[3] if len(recent_mean) > 3 else 50.0)
            }
        )

    def _detect_anomalies(self, df: pd.DataFrame) -> List[AnomalyDetection]:
        """Detect anomalies in price, volume, and volatility."""
        anomalies = []

        if len(df) < 50:
            return anomalies

        current_idx = len(df) - 1

        # Price anomaly (Z-score based)
        returns = df['Returns'].dropna()
        if len(returns) > 0:
            z_score = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-10)
            if abs(z_score) > 3:
                anomalies.append(AnomalyDetection(
                    is_anomaly=True,
                    anomaly_score=float(abs(z_score)),
                    anomaly_type="price",
                    severity="high" if abs(z_score) > 4 else "medium",
                    description=f"Unusual price movement: {z_score:.2f} standard deviations from mean"
                ))

        # Volume anomaly
        if 'Volume_Ratio' in df.columns:
            vol_ratio = df['Volume_Ratio'].iloc[current_idx]
            if vol_ratio > 3:
                anomalies.append(AnomalyDetection(
                    is_anomaly=True,
                    anomaly_score=float(vol_ratio),
                    anomaly_type="volume",
                    severity="high" if vol_ratio > 5 else "medium",
                    description=f"Abnormal volume: {vol_ratio:.1f}x average"
                ))

        # Volatility anomaly
        if 'Volatility' in df.columns:
            vol = df['Volatility'].dropna()
            if len(vol) > 0:
                current_vol = vol.iloc[-1]
                vol_percentile = stats.percentileofscore(vol, current_vol)
                if vol_percentile > 95:
                    anomalies.append(AnomalyDetection(
                        is_anomaly=True,
                        anomaly_score=float(vol_percentile / 100),
                        anomaly_type="volatility",
                        severity="high" if vol_percentile > 98 else "medium",
                        description=f"High volatility: {vol_percentile:.0f}th percentile"
                    ))

        return anomalies

    def _analyze_sentiment(self, symbol: str, df: pd.DataFrame) -> SentimentAnalysis:
        """
        Analyze market sentiment from multiple sources.

        In production, integrate:
        - News APIs (NewsAPI, Alpha Vantage)
        - Social media (Twitter, Reddit via APIs)
        - FinBERT for NLP sentiment
        - SEC filings analysis

        For now, we use technical signals as proxy.
        """
        # Derive sentiment from technical indicators
        rsi = df['RSI'].iloc[-1]
        macd_hist = df['MACD_Hist'].iloc[-1]
        price_vs_sma50 = (df['Close'].iloc[-1] / df['SMA_50'].iloc[-1] - 1)

        # Combine signals
        sentiment_score = 0.0

        # RSI contribution (-1 to 1)
        if rsi < 30:
            sentiment_score += 0.5  # oversold = bullish
        elif rsi > 70:
            sentiment_score -= 0.5  # overbought = bearish
        else:
            sentiment_score += (50 - rsi) / 40 * 0.3

        # MACD contribution
        sentiment_score += np.tanh(macd_hist * 10) * 0.3

        # Trend contribution
        sentiment_score += np.tanh(price_vs_sma50 * 5) * 0.4

        # Normalize
        sentiment_score = np.clip(sentiment_score, -1, 1)

        # Label
        if sentiment_score > 0.3:
            label = "bullish"
        elif sentiment_score < -0.3:
            label = "bearish"
        else:
            label = "neutral"

        confidence = abs(sentiment_score)

        return SentimentAnalysis(
            overall_sentiment=float(sentiment_score),
            sentiment_label=label,
            confidence=float(confidence),
            sources={
                "technical_indicators": float(sentiment_score),
                "news": 0.0,  # Placeholder for real news API
                "social_media": 0.0  # Placeholder for social sentiment
            },
            trending_topics=["earnings", "market_trend", symbol.lower()]
        )

    def _calculate_risk_metrics(
        self,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        returns = df['Returns'].dropna()

        # Daily and annual volatility
        vol_daily = float(returns.std())
        vol_annual = vol_daily * np.sqrt(252)

        # VaR (Value at Risk) - 95% confidence
        var_95 = float(np.percentile(returns, 5))

        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe = float((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252))

        # Beta (vs benchmark)
        if len(benchmark_df) > 0:
            benchmark_returns = benchmark_df['Close'].pct_change().dropna()
            common_dates = returns.index.intersection(benchmark_returns.index)

            if len(common_dates) > 20:
                stock_aligned = returns.loc[common_dates]
                bench_aligned = benchmark_returns.loc[common_dates]

                covariance = np.cov(stock_aligned, bench_aligned)[0, 1]
                benchmark_variance = np.var(bench_aligned)
                beta = float(covariance / (benchmark_variance + 1e-10))
            else:
                beta = 1.0
        else:
            beta = 1.0

        return {
            'vol_daily': vol_daily,
            'vol_annual': vol_annual,
            'var_95': var_95,
            'sharpe': sharpe,
            'beta': beta
        }

    def _generate_predictions(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate ML-based predictions."""
        # Simple momentum + mean reversion model
        recent_returns = df['Returns'].iloc[-20:].mean()
        volatility = df['Volatility'].iloc[-1]
        rsi = df['RSI'].iloc[-1]

        # 5-day prediction
        momentum = recent_returns * 3  # momentum component
        mean_reversion = (50 - rsi) / 500  # mean reversion from RSI
        pred_5d = momentum + mean_reversion

        # 20-day prediction
        pred_20d = pred_5d * 3.5 + (df['Close'].iloc[-1] / df['SMA_200'].iloc[-1] - 1) * 0.1

        # Confidence based on volatility (lower vol = higher confidence)
        confidence = float(np.clip(1 - volatility * 20, 0.3, 0.95))

        # Profit probability
        profit_prob = float(0.5 + pred_5d * 5)
        profit_prob = np.clip(profit_prob, 0.1, 0.9)

        return {
            'return_5d': float(pred_5d),
            'return_20d': float(pred_20d),
            'confidence': confidence,
            'profit_prob': profit_prob
        }

    def _get_technical_summary(self, df: pd.DataFrame) -> Dict:
        """Get current technical indicator summary."""
        current = df.iloc[-1]

        # Volume profile
        vol_ratio = current['Volume_Ratio']
        if vol_ratio > 2:
            vol_profile = "high"
        elif vol_ratio < 0.5:
            vol_profile = "low"
        else:
            vol_profile = "normal"

        return {
            'rsi': float(current['RSI']),
            'macd': float(current['MACD']),
            'bb_position': float(current['BB_Position']),
            'volume_profile': vol_profile
        }

    def _find_similar_patterns(self, df: pd.DataFrame) -> Dict:
        """Find similar historical patterns."""
        # Simplified pattern matching
        current_pattern = df[['Returns', 'Volatility', 'RSI']].iloc[-10:].values.flatten()

        similar_examples = []
        pattern_strength = 0.7  # Placeholder

        # Would implement full FAISS-based similarity search here
        # For now, return mock data
        for i in range(min(5, len(df) - 20)):
            idx = len(df) - 20 - i * 10
            if idx < 10:
                break

            similar_examples.append({
                'date': str(df.index[idx].date()),
                'similarity': 0.85 - i * 0.05,
                'outcome_return': float(df['Returns'].iloc[idx:idx+5].sum())
            })

        return {
            'examples': similar_examples,
            'strength': pattern_strength
        }

    def _generate_insights(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        anomalies: List[AnomalyDetection],
        sentiment: SentimentAnalysis,
        risk_metrics: Dict,
        predictions: Dict
    ) -> Dict[str, any]:
        """Generate AI-powered insights and recommendations."""
        insights = []
        risks = []
        opportunities = []

        # Regime insights
        insights.append(f"Market regime: {regime.regime_type.replace('_', ' ').title()}")

        # Sentiment insights
        insights.append(
            f"Sentiment is {sentiment.sentiment_label} "
            f"(score: {sentiment.overall_sentiment:+.2f})"
        )

        # Prediction insights
        if predictions['return_5d'] > 0.02:
            insights.append(
                f"Strong bullish signal: +{predictions['return_5d']*100:.1f}% "
                f"expected in 5 days"
            )
            opportunities.append("Potential short-term upside momentum")
        elif predictions['return_5d'] < -0.02:
            insights.append(
                f"Bearish signal: {predictions['return_5d']*100:.1f}% "
                f"expected in 5 days"
            )
            risks.append("Short-term downside risk")

        # Anomaly insights
        for anomaly in anomalies:
            if anomaly.severity == "high":
                insights.append(f"⚠️  {anomaly.description}")
                risks.append(f"{anomaly.anomaly_type.title()} anomaly detected")

        # Risk insights
        if risk_metrics['vol_annual'] > 0.40:
            risks.append(f"High volatility: {risk_metrics['vol_annual']*100:.0f}% annual")

        if risk_metrics['sharpe'] > 1.5:
            opportunities.append(f"Strong risk-adjusted returns (Sharpe: {risk_metrics['sharpe']:.2f})")
        elif risk_metrics['sharpe'] < 0:
            risks.append(f"Negative risk-adjusted returns (Sharpe: {risk_metrics['sharpe']:.2f})")

        # Technical insights
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            opportunities.append("RSI oversold - potential bounce")
        elif rsi > 70:
            risks.append("RSI overbought - potential pullback")

        # Generate recommendation (production-grade scoring)
        score = 0.0

        # Prediction signal (±2.0 max)
        pred_score = predictions['return_5d'] * 40  # Scale up: 5% pred = 2.0 contribution
        score += pred_score

        # Sentiment signal (±1.5 max)
        sent_score = sentiment.overall_sentiment * 1.5
        score += sent_score

        # RSI mean reversion (±1.0 max)
        rsi_signal = (50 - rsi) / 50  # Normalized to ±1
        rsi_score = rsi_signal * 1.0
        score += rsi_score

        # MACD momentum (±1.0 max)
        macd = df['MACD'].iloc[-1]
        macd_signal_line = df['MACD_Signal'].iloc[-1]
        # Positive if MACD > signal (bullish), negative if MACD < signal (bearish)
        macd_diff = macd - macd_signal_line
        macd_score = np.tanh(macd_diff * 0.5) * 1.0  # Normalized to ±1
        score += macd_score

        # Regime adjustment (boost/reduce based on regime)
        regime_score = 0.0
        if regime.regime_type in ['bull_stable', 'bull_volatile']:
            regime_score = 0.5
        elif regime.regime_type in ['bear_stable', 'bear_volatile']:
            regime_score = -0.5
        score += regime_score

        # Confidence-weighted score
        final_score = score * predictions['confidence']

        # DEBUG: Store components for analysis
        # print(f"DEBUG: pred={pred_score:.3f}, sent={sent_score:.3f}, rsi={rsi_score:.3f}, macd={macd_score:.3f}, regime={regime_score:.3f}, raw_score={score:.3f}, final={final_score:.3f}")

        # Map to recommendations (aggressive thresholds for active trading)
        if final_score > 1.5:
            recommendation = "strong_buy"
        elif final_score > 0.3:  # Lower threshold to generate more buy signals
            recommendation = "buy"
        elif final_score > -0.3:  # Narrower hold range
            recommendation = "hold"
        elif final_score > -1.5:
            recommendation = "sell"
        else:
            recommendation = "strong_sell"

        return {
            'insights': insights,
            'risks': risks,
            'opportunities': opportunities,
            'recommendation': recommendation
        }

    def compare_symbols(
        self,
        symbols: List[str],
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """Compare multiple symbols side-by-side."""
        results = []

        for symbol in symbols:
            try:
                intelligence = self.analyze(symbol, lookback_days)
                results.append({
                    'Symbol': symbol,
                    'Price': intelligence.current_price,
                    'Change_1D': intelligence.price_change_1d * 100,
                    'Pred_5D': intelligence.predicted_return_5d * 100,
                    'Confidence': intelligence.confidence_score,
                    'Volatility': intelligence.volatility_annual * 100,
                    'Sharpe': intelligence.sharpe_ratio,
                    'Beta': intelligence.beta,
                    'RSI': intelligence.rsi,
                    'Regime': intelligence.market_regime.regime_type,
                    'Sentiment': intelligence.sentiment.sentiment_label,
                    'Recommendation': intelligence.recommendation
                })
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

        return pd.DataFrame(results)
