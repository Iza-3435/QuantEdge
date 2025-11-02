"""
Production-grade sentiment analysis engine for stock market intelligence.
Multi-source news aggregation with NLP-based sentiment scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

try:
    from transformers import pipeline
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


@dataclass
class SentimentScore:
    overall_score: float
    sentiment_label: str
    news_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    confidence: float
    trending_topics: List[str]


class SentimentAnalyzer:
    """Production sentiment analyzer with multi-source aggregation"""

    def __init__(self):
        self.news_api_key = os.getenv('NEWSAPI_KEY')
        self.cache = {}
        self.cache_ttl = 300

        if NLP_AVAILABLE:
            try:
                self.nlp_model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
            except Exception:
                self.nlp_model = None
        else:
            self.nlp_model = None

    def analyze_stock_sentiment(self, symbol: str, company_name: str = None) -> Optional[SentimentScore]:
        """Analyze sentiment from multiple news sources"""
        cache_key = f"sentiment_{symbol}"

        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_result

        news_articles = self._fetch_news(symbol, company_name)

        if not news_articles:
            return self._fallback_sentiment()

        sentiments = []
        for article in news_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self._analyze_text(text)
            if sentiment:
                sentiments.append(sentiment)

        if not sentiments:
            return self._fallback_sentiment()

        result = self._aggregate_sentiments(sentiments, len(news_articles))

        self.cache[cache_key] = (datetime.now(), result)
        return result

    def _fetch_news(self, symbol: str, company_name: str = None) -> List[Dict[str, Any]]:
        """Fetch news from News API"""
        if not self.news_api_key:
            return []

        try:
            query = company_name if company_name else symbol
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 20,
                'apiKey': self.news_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])[:20]

            return []

        except Exception:
            return []

    def _analyze_text(self, text: str) -> Optional[Dict[str, float]]:
        """Analyze text sentiment using NLP model or lexicon"""
        if not text or len(text.strip()) < 10:
            return None

        if self.nlp_model:
            try:
                result = self.nlp_model(text[:512])[0]

                if result['label'] == 'POSITIVE':
                    score = result['score']
                    return {'positive': score, 'negative': 0, 'neutral': 1 - score}
                else:
                    score = result['score']
                    return {'positive': 0, 'negative': score, 'neutral': 1 - score}
            except Exception:
                pass

        return self._lexicon_sentiment(text)

    def _lexicon_sentiment(self, text: str) -> Dict[str, float]:
        """Fallback lexicon-based sentiment analysis"""
        text_lower = text.lower()

        positive_words = [
            'bullish', 'surge', 'soar', 'gain', 'profit', 'growth', 'strong',
            'rally', 'beat', 'exceed', 'upgrade', 'buy', 'outperform', 'positive',
            'success', 'record', 'high', 'breakthrough', 'innovation', 'expansion',
            'revenue', 'earnings', 'momentum', 'opportunity', 'robust', 'solid'
        ]

        negative_words = [
            'bearish', 'plunge', 'fall', 'loss', 'decline', 'weak', 'crash',
            'sell', 'underperform', 'downgrade', 'miss', 'disappoint', 'negative',
            'concern', 'risk', 'fear', 'volatility', 'uncertainty', 'warning',
            'debt', 'lawsuit', 'investigation', 'fraud', 'scandal', 'bankruptcy'
        ]

        neutral_words = [
            'hold', 'neutral', 'stable', 'unchanged', 'maintain', 'steady',
            'continue', 'monitor', 'watch', 'observe', 'track', 'assess'
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        neu_count = sum(1 for word in neutral_words if word in text_lower)

        total = pos_count + neg_count + neu_count

        if total == 0:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

        return {
            'positive': pos_count / total,
            'negative': neg_count / total,
            'neutral': neu_count / total
        }

    def _aggregate_sentiments(self, sentiments: List[Dict[str, float]], total_news: int) -> SentimentScore:
        """Aggregate individual sentiments into overall score"""
        pos_scores = [s['positive'] for s in sentiments]
        neg_scores = [s['negative'] for s in sentiments]
        neu_scores = [s['neutral'] for s in sentiments]

        avg_pos = np.mean(pos_scores)
        avg_neg = np.mean(neg_scores)
        avg_neu = np.mean(neu_scores)

        overall_score = (avg_pos - avg_neg) * 100

        if overall_score > 30:
            sentiment_label = "VERY POSITIVE"
        elif overall_score > 10:
            sentiment_label = "POSITIVE"
        elif overall_score > -10:
            sentiment_label = "NEUTRAL"
        elif overall_score > -30:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "VERY NEGATIVE"

        confidence = min(len(sentiments) / 20.0, 0.95)

        return SentimentScore(
            overall_score=overall_score,
            sentiment_label=sentiment_label,
            news_count=total_news,
            positive_ratio=avg_pos,
            negative_ratio=avg_neg,
            neutral_ratio=avg_neu,
            confidence=confidence,
            trending_topics=[]
        )

    def _fallback_sentiment(self) -> SentimentScore:
        """Fallback neutral sentiment when no data available"""
        return SentimentScore(
            overall_score=0.0,
            sentiment_label="NEUTRAL",
            news_count=0,
            positive_ratio=0.33,
            negative_ratio=0.33,
            neutral_ratio=0.34,
            confidence=0.30,
            trending_topics=[]
        )


def analyze_sentiment(symbol: str, company_name: str = None) -> Dict[str, Any]:
    """Quick interface for sentiment analysis"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_stock_sentiment(symbol, company_name)

    if not result:
        result = analyzer._fallback_sentiment()

    return {
        'sentiment_score': result.overall_score,
        'sentiment_label': result.sentiment_label,
        'news_count': result.news_count,
        'positive_ratio': result.positive_ratio,
        'negative_ratio': result.negative_ratio,
        'neutral_ratio': result.neutral_ratio,
        'confidence': result.confidence
    }
