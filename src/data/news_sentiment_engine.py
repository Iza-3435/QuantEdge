"""
Advanced News & Sentiment Analysis Engine

Real-time news aggregation and sentiment analysis from multiple sources.
Integrates with your HFT system for signal generation.

Features:
- Multi-source news aggregation (NewsAPI, Alpha Vantage, RSS)
- FinBERT sentiment analysis
- Real-time sentiment scoring
- Entity extraction (companies, people, events)
- Trending topics detection
- Sentiment time series
- Event impact scoring
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using simplified sentiment analysis.")


@dataclass
class NewsArticle:
    """News article data structure."""
    title: str
    description: str
    content: str
    source: str
    author: Optional[str]
    published_at: datetime
    url: str
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    relevance_score: float = 0.0
    entities: List[str] = None


@dataclass
class SentimentSignal:
    """Trading signal from sentiment analysis."""
    symbol: str
    overall_sentiment: float  # -1 to 1
    sentiment_label: str
    confidence: float
    signal_strength: float  # 0 to 1
    direction: str  # buy, sell, hold
    news_count: int
    trending_topics: List[str]
    key_events: List[str]
    sources_breakdown: Dict[str, float]
    timestamp: datetime


class NewsAggregator:
    """
    Multi-source news aggregator.

    Supports:
    - NewsAPI (https://newsapi.org)
    - Alpha Vantage News Sentiment
    - RSS feeds
    - Custom sources
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None
    ):
        """
        Initialize news aggregator.

        Args:
            newsapi_key: NewsAPI.org API key
            alphavantage_key: Alpha Vantage API key
        """
        self.newsapi_key = newsapi_key
        self.alphavantage_key = alphavantage_key
        self.cache = {}

    def fetch_newsapi(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        language: str = 'en',
        sort_by: str = 'relevancy'
    ) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI.org

        Free tier: 100 requests/day, 1 month old articles
        Premium: Real-time news
        """
        if not self.newsapi_key:
            print("NewsAPI key not provided. Using mock data.")
            return self._get_mock_news(query)

        url = "https://newsapi.org/v2/everything"

        if from_date is None:
            from_date = datetime.now() - timedelta(days=7)

        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'sortBy': sort_by,
            'language': language,
            'apiKey': self.newsapi_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get('articles', []):
                articles.append(NewsArticle(
                    title=article.get('title', ''),
                    description=article.get('description', ''),
                    content=article.get('content', ''),
                    source=article.get('source', {}).get('name', 'Unknown'),
                    author=article.get('author'),
                    published_at=pd.to_datetime(article.get('publishedAt')),
                    url=article.get('url', '')
                ))

            return articles

        except Exception as e:
            print(f"NewsAPI error: {e}")
            return self._get_mock_news(query)

    def fetch_alphavantage_sentiment(
        self,
        tickers: str,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None
    ) -> List[NewsArticle]:
        """
        Fetch news sentiment from Alpha Vantage.

        Free tier: 25 requests/day
        """
        if not self.alphavantage_key:
            print("Alpha Vantage key not provided. Using mock data.")
            return self._get_mock_news(tickers)

        url = "https://www.alphavantage.co/query"

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': tickers,
            'apikey': self.alphavantage_key
        }

        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get('feed', []):
                articles.append(NewsArticle(
                    title=item.get('title', ''),
                    description=item.get('summary', ''),
                    content=item.get('summary', ''),
                    source=item.get('source', 'Alpha Vantage'),
                    author=None,
                    published_at=pd.to_datetime(item.get('time_published')),
                    url=item.get('url', ''),
                    sentiment_score=float(item.get('overall_sentiment_score', 0)),
                    sentiment_label=item.get('overall_sentiment_label', 'Neutral')
                ))

            return articles

        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return self._get_mock_news(tickers)

    def _get_mock_news(self, query: str) -> List[NewsArticle]:
        """Generate mock news for testing without API keys."""
        mock_articles = [
            NewsArticle(
                title=f"{query} reports strong quarterly earnings",
                description=f"{query} exceeds expectations with revenue growth",
                content="Full article content here...",
                source="Financial Times",
                author="John Doe",
                published_at=datetime.now() - timedelta(hours=2),
                url="https://example.com/article1",
                sentiment_score=0.75,
                sentiment_label="Bullish"
            ),
            NewsArticle(
                title=f"Analysts upgrade {query} to buy rating",
                description=f"Multiple analysts raise price targets for {query}",
                content="Full article content here...",
                source="Bloomberg",
                author="Jane Smith",
                published_at=datetime.now() - timedelta(hours=5),
                url="https://example.com/article2",
                sentiment_score=0.60,
                sentiment_label="Bullish"
            ),
            NewsArticle(
                title=f"{query} faces regulatory scrutiny",
                description=f"New regulations may impact {query} operations",
                content="Full article content here...",
                source="Reuters",
                author="Bob Johnson",
                published_at=datetime.now() - timedelta(hours=8),
                url="https://example.com/article3",
                sentiment_score=-0.45,
                sentiment_label="Bearish"
            )
        ]
        return mock_articles


class FinBERTSentiment:
    """
    FinBERT-based sentiment analyzer.

    FinBERT is BERT fine-tuned on financial text.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading FinBERT model...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ProsusAI/finbert"
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert"
                )
                self.model.eval()
                print("FinBERT loaded successfully!")
            except Exception as e:
                print(f"Could not load FinBERT: {e}")
                print("Using rule-based sentiment.")

    def analyze(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze sentiment of text.

        Returns:
            (sentiment_score, label, confidence)
            sentiment_score: -1 (bearish) to 1 (bullish)
            label: 'positive', 'negative', 'neutral'
            confidence: 0 to 1
        """
        if self.model is None or not TRANSFORMERS_AVAILABLE:
            return self._rule_based_sentiment(text)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get sentiment
        probabilities = predictions[0].tolist()
        labels = ['positive', 'negative', 'neutral']

        sentiment_idx = predictions.argmax().item()
        sentiment_label = labels[sentiment_idx]
        confidence = probabilities[sentiment_idx]

        # Convert to score (-1 to 1)
        if sentiment_label == 'positive':
            sentiment_score = probabilities[0]
        elif sentiment_label == 'negative':
            sentiment_score = -probabilities[1]
        else:
            sentiment_score = 0.0

        return sentiment_score, sentiment_label, confidence

    def _rule_based_sentiment(self, text: str) -> Tuple[float, str, float]:
        """Simple rule-based sentiment when FinBERT unavailable."""
        text_lower = text.lower()

        # Positive keywords
        positive_words = [
            'beat', 'surge', 'soar', 'rally', 'gain', 'profit', 'growth',
            'strong', 'upgrade', 'bullish', 'buy', 'outperform', 'exceeded',
            'record', 'high', 'success', 'positive', 'jump', 'rise'
        ]

        # Negative keywords
        negative_words = [
            'miss', 'fall', 'drop', 'decline', 'loss', 'weak', 'downgrade',
            'bearish', 'sell', 'underperform', 'missed', 'low', 'concern',
            'negative', 'risk', 'warn', 'cut', 'reduce', 'plunge'
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0, 'neutral', 0.5

        sentiment_score = (pos_count - neg_count) / total

        if sentiment_score > 0.2:
            label = 'positive'
        elif sentiment_score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'

        confidence = min(total / 10.0, 0.9)

        return sentiment_score, label, confidence


class NewsSentimentEngine:
    """
    Complete news sentiment engine for trading signals.

    Integrates:
    - News aggregation from multiple sources
    - FinBERT sentiment analysis
    - Entity extraction
    - Signal generation
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None
    ):
        self.aggregator = NewsAggregator(newsapi_key, alphavantage_key)
        self.sentiment_analyzer = FinBERTSentiment()
        self.cache = {}

    def analyze_symbol(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> SentimentSignal:
        """
        Analyze news sentiment for a symbol.

        Args:
            symbol: Stock ticker
            lookback_hours: Hours to look back for news

        Returns:
            SentimentSignal with trading signal
        """
        # Fetch news
        from_date = datetime.now() - timedelta(hours=lookback_hours)

        # Try multiple sources
        articles = []

        # NewsAPI
        articles.extend(
            self.aggregator.fetch_newsapi(symbol, from_date=from_date)
        )

        # Alpha Vantage
        articles.extend(
            self.aggregator.fetch_alphavantage_sentiment(symbol)
        )

        if not articles:
            return self._get_neutral_signal(symbol)

        # Analyze sentiment for each article
        analyzed_articles = []
        for article in articles:
            if article.sentiment_score == 0.0:
                # Not analyzed yet
                text = f"{article.title}. {article.description}"
                score, label, conf = self.sentiment_analyzer.analyze(text)
                article.sentiment_score = score
                article.sentiment_label = label

            analyzed_articles.append(article)

        # Aggregate sentiment
        overall_sentiment = self._aggregate_sentiment(analyzed_articles)

        # Extract trending topics
        trending_topics = self._extract_topics(analyzed_articles)

        # Identify key events
        key_events = self._identify_events(analyzed_articles)

        # Source breakdown
        sources_breakdown = self._get_sources_breakdown(analyzed_articles)

        # Generate signal
        signal = self._generate_signal(
            symbol,
            overall_sentiment,
            analyzed_articles,
            trending_topics,
            key_events,
            sources_breakdown
        )

        return signal

    def _aggregate_sentiment(self, articles: List[NewsArticle]) -> float:
        """Aggregate sentiment scores with time decay."""
        if not articles:
            return 0.0

        from datetime import timezone
        now = datetime.now(timezone.utc)
        weighted_scores = []
        weights = []

        for article in articles:
            # Time decay (newer articles weighted more)
            # Make article timestamp timezone-aware if it isn't
            pub_time = article.published_at
            if pub_time.tzinfo is None:
                pub_time = pub_time.replace(tzinfo=timezone.utc)

            hours_old = (now - pub_time).total_seconds() / 3600
            time_weight = np.exp(-hours_old / 24)  # Decay over 24 hours

            weighted_scores.append(article.sentiment_score * time_weight)
            weights.append(time_weight)

        overall = np.average(weighted_scores, weights=weights)
        return float(overall)

    def _extract_topics(self, articles: List[NewsArticle]) -> List[str]:
        """Extract trending topics from articles."""
        all_text = " ".join([
            f"{a.title} {a.description}" for a in articles
        ])

        # Simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_text)

        # Filter common words
        stop_words = {'The', 'This', 'That', 'These', 'Those', 'For', 'And'}
        words = [w for w in words if w not in stop_words]

        # Count and return top topics
        counter = Counter(words)
        top_topics = [word for word, count in counter.most_common(5)]

        return top_topics

    def _identify_events(self, articles: List[NewsArticle]) -> List[str]:
        """Identify key market events."""
        events = []

        keywords = {
            'earnings': ['earnings', 'quarterly', 'revenue', 'profit'],
            'merger': ['merger', 'acquisition', 'takeover'],
            'product': ['launch', 'product', 'release', 'unveil'],
            'regulatory': ['regulation', 'lawsuit', 'investigation'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'target']
        }

        for article in articles:
            text = f"{article.title} {article.description}".lower()

            for event_type, words in keywords.items():
                if any(word in text for word in words):
                    events.append(event_type)
                    break

        return list(set(events))

    def _get_sources_breakdown(
        self,
        articles: List[NewsArticle]
    ) -> Dict[str, float]:
        """Get sentiment breakdown by source."""
        sources = {}

        for article in articles:
            source = article.source
            if source not in sources:
                sources[source] = []
            sources[source].append(article.sentiment_score)

        # Average by source
        return {
            source: float(np.mean(scores))
            for source, scores in sources.items()
        }

    def _generate_signal(
        self,
        symbol: str,
        overall_sentiment: float,
        articles: List[NewsArticle],
        trending_topics: List[str],
        key_events: List[str],
        sources_breakdown: Dict[str, float]
    ) -> SentimentSignal:
        """Generate trading signal from sentiment."""
        # Determine label
        if overall_sentiment > 0.3:
            label = "bullish"
            direction = "buy"
        elif overall_sentiment < -0.3:
            label = "bearish"
            direction = "sell"
        else:
            label = "neutral"
            direction = "hold"

        # Signal strength
        signal_strength = abs(overall_sentiment)

        # Confidence based on article count and consistency
        sentiment_std = np.std([a.sentiment_score for a in articles])
        confidence = float(np.clip(1 - sentiment_std, 0.3, 0.95))

        return SentimentSignal(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            sentiment_label=label,
            confidence=confidence,
            signal_strength=signal_strength,
            direction=direction,
            news_count=len(articles),
            trending_topics=trending_topics,
            key_events=key_events,
            sources_breakdown=sources_breakdown,
            timestamp=datetime.now()
        )

    def _get_neutral_signal(self, symbol: str) -> SentimentSignal:
        """Return neutral signal when no news available."""
        return SentimentSignal(
            symbol=symbol,
            overall_sentiment=0.0,
            sentiment_label="neutral",
            confidence=0.5,
            signal_strength=0.0,
            direction="hold",
            news_count=0,
            trending_topics=[],
            key_events=[],
            sources_breakdown={},
            timestamp=datetime.now()
        )

    def get_sentiment_time_series(
        self,
        symbol: str,
        days: int = 7
    ) -> pd.DataFrame:
        """Get sentiment time series for backtesting."""
        # This would query historical news
        # For now, return mock data
        dates = pd.date_range(
            end=datetime.now(),
            periods=days,
            freq='D'
        )

        sentiments = np.random.randn(days) * 0.3  # Mock data

        return pd.DataFrame({
            'date': dates,
            'sentiment': sentiments,
            'news_count': np.random.randint(5, 20, days)
        })
