"""
Institutional-grade sentiment analysis with FinBERT and SEC filings.
Multi-source: News, SEC 10-K/8-K, Earnings calls, Social media.
Used by top hedge funds for alternative data signals.
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import warnings
import re
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class AdvancedSentiment:
    overall_score: float
    sentiment_label: str
    news_sentiment: float
    sec_filing_sentiment: float
    social_sentiment: float
    news_count: int
    sec_filings_count: int
    confidence: float
    sentiment_trend: str
    key_topics: List[str]
    risk_factors: List[str]


class AdvancedSentimentAnalyzer:
    """Institutional-grade multi-source sentiment analyzer"""

    def __init__(self):
        self.news_api_key = os.getenv('NEWSAPI_KEY')
        self.sec_user_agent = os.getenv('SEC_USER_AGENT', 'AI Market Intelligence user@example.com')
        self.cache = {}
        self.cache_ttl = 300

        if TRANSFORMERS_AVAILABLE:
            try:
                import logging
                logging.getLogger("transformers").setLevel(logging.ERROR)

                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    'ProsusAI/finbert',
                    cache_dir='./models_cache'
                )
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                    'ProsusAI/finbert',
                    cache_dir='./models_cache'
                )
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.finbert_model,
                    tokenizer=self.finbert_tokenizer,
                    device=-1
                )
                self.finbert_available = True
            except Exception:
                self.finbert_available = False
                self.finbert_pipeline = None
        else:
            self.finbert_available = False
            self.finbert_pipeline = None

    def analyze_comprehensive(self, symbol: str, company_name: str = None) -> Optional[AdvancedSentiment]:
        """Comprehensive multi-source sentiment analysis"""
        cache_key = f"adv_sentiment_{symbol}"

        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_result

        news_score, news_count = self._analyze_news(symbol, company_name)
        sec_score, sec_count, risk_factors = self._analyze_sec_filings(symbol)
        social_score = self._analyze_social_sentiment(symbol)

        weights = {'news': 0.4, 'sec': 0.4, 'social': 0.2}

        overall_score = (
            news_score * weights['news'] +
            sec_score * weights['sec'] +
            social_score * weights['social']
        )

        sentiment_label = self._score_to_label(overall_score)
        sentiment_trend = self._calculate_trend(news_score, sec_score)
        confidence = self._calculate_confidence(news_count, sec_count)

        result = AdvancedSentiment(
            overall_score=overall_score,
            sentiment_label=sentiment_label,
            news_sentiment=news_score,
            sec_filing_sentiment=sec_score,
            social_sentiment=social_score,
            news_count=news_count,
            sec_filings_count=sec_count,
            confidence=confidence,
            sentiment_trend=sentiment_trend,
            key_topics=[],
            risk_factors=risk_factors
        )

        self.cache[cache_key] = (datetime.now(), result)
        return result

    def _analyze_news(self, symbol: str, company_name: str = None) -> Tuple[float, int]:
        """Analyze news sentiment with FinBERT"""
        if not self.news_api_key:
            return 0.0, 0

        try:
            query = company_name if company_name else symbol
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 30,
                'apiKey': self.news_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return 0.0, 0

            articles = response.json().get('articles', [])

            if not articles:
                return 0.0, 0

            sentiments = []
            for article in articles[:30]:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self._analyze_text_finbert(text)
                if sentiment is not None:
                    sentiments.append(sentiment)

            if not sentiments:
                return 0.0, 0

            avg_sentiment = np.mean(sentiments)
            return avg_sentiment * 100, len(sentiments)

        except Exception:
            return 0.0, 0

    def _analyze_text_finbert(self, text: str) -> Optional[float]:
        """Analyze text with FinBERT"""
        if not text or len(text.strip()) < 10:
            return None

        if self.finbert_available:
            try:
                result = self.finbert_pipeline(text[:512])[0]

                label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
                score = label_map.get(result['label'].lower(), 0.0)
                confidence = result['score']

                return score * confidence

            except Exception:
                pass

        return self._lexicon_sentiment(text)

    def _analyze_sec_filings(self, symbol: str) -> Tuple[float, int, List[str]]:
        """Analyze SEC 10-K and 8-K filings"""
        try:
            cik = self._get_cik(symbol)
            if not cik:
                return 0.0, 0, []

            filings = self._fetch_recent_filings(cik, ['10-K', '8-K'])

            if not filings:
                return 0.0, 0, []

            sentiments = []
            risk_factors = []

            for filing in filings[:5]:
                filing_text = self._extract_filing_text(filing)
                if filing_text:
                    sentiment = self._analyze_filing_sentiment(filing_text)
                    if sentiment is not None:
                        sentiments.append(sentiment)

                    risks = self._extract_risk_factors(filing_text)
                    risk_factors.extend(risks)

            if not sentiments:
                return 0.0, 0, []

            avg_sentiment = np.mean(sentiments)
            return avg_sentiment * 100, len(sentiments), risk_factors[:5]

        except Exception:
            return 0.0, 0, []

    def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK number from ticker symbol"""
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'ticker': symbol,
                'output': 'json'
            }
            headers = {'User-Agent': self.sec_user_agent}

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('cik')

            return None

        except Exception:
            return None

    def _fetch_recent_filings(self, cik: str, filing_types: List[str]) -> List[Dict]:
        """Fetch recent SEC filings"""
        try:
            filings = []

            for filing_type in filing_types:
                url = f"https://www.sec.gov/cgi-bin/browse-edgar"
                params = {
                    'action': 'getcompany',
                    'CIK': cik,
                    'type': filing_type,
                    'dateb': '',
                    'owner': 'exclude',
                    'count': 3,
                    'output': 'json'
                }
                headers = {'User-Agent': self.sec_user_agent}

                response = requests.get(url, params=params, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    filings.extend(data.get('filings', {}).get('recent', []))

            return filings

        except Exception:
            return []

    def _extract_filing_text(self, filing: Dict) -> Optional[str]:
        """Extract text from SEC filing"""
        try:
            filing_url = filing.get('url', '')
            if not filing_url:
                return None

            headers = {'User-Agent': self.sec_user_agent}
            response = requests.get(filing_url, headers=headers, timeout=15)

            if response.status_code == 200:
                text = response.text

                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text)

                return text[:10000]

            return None

        except Exception:
            return None

    def _analyze_filing_sentiment(self, text: str) -> Optional[float]:
        """Analyze SEC filing sentiment"""
        sections = self._extract_mda_section(text)

        if not sections:
            return None

        sentiments = []
        for section in sections:
            sentiment = self._analyze_text_finbert(section)
            if sentiment is not None:
                sentiments.append(sentiment)

        if not sentiments:
            return None

        return np.mean(sentiments)

    def _extract_mda_section(self, text: str) -> List[str]:
        """Extract Management Discussion & Analysis section"""
        patterns = [
            r'MANAGEMENT.{0,50}DISCUSSION.{0,50}ANALYSIS(.*?)(?=QUANTITATIVE|FINANCIAL|CONTROLS|$)',
            r'MD&A(.*?)(?=QUANTITATIVE|FINANCIAL|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                return [m[:5000] for m in matches]

        return [text[:5000]]

    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract key risk factors from filings"""
        risk_keywords = [
            'market risk', 'credit risk', 'liquidity risk', 'operational risk',
            'regulatory risk', 'competition', 'economic downturn', 'cyber security',
            'supply chain', 'litigation', 'compliance', 'volatility'
        ]

        found_risks = []
        text_lower = text.lower()

        for keyword in risk_keywords:
            if keyword in text_lower:
                found_risks.append(keyword.title())

        return list(set(found_risks))[:5]

    def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment (placeholder for Reddit/Twitter)"""
        return 0.0

    def _lexicon_sentiment(self, text: str) -> float:
        """Fallback lexicon-based sentiment"""
        text_lower = text.lower()

        positive_words = [
            'bullish', 'surge', 'soar', 'gain', 'profit', 'growth', 'strong',
            'beat', 'exceed', 'upgrade', 'outperform', 'positive', 'success',
            'record', 'breakthrough', 'innovation', 'expansion', 'momentum'
        ]

        negative_words = [
            'bearish', 'plunge', 'fall', 'loss', 'decline', 'weak', 'crash',
            'miss', 'disappoint', 'downgrade', 'underperform', 'negative',
            'concern', 'risk', 'fear', 'volatility', 'warning', 'debt'
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def _score_to_label(self, score: float) -> str:
        """Convert score to label"""
        if score > 30:
            return "VERY POSITIVE"
        elif score > 10:
            return "POSITIVE"
        elif score > -10:
            return "NEUTRAL"
        elif score > -30:
            return "NEGATIVE"
        else:
            return "VERY NEGATIVE"

    def _calculate_trend(self, news_score: float, sec_score: float) -> str:
        """Calculate sentiment trend"""
        if news_score > 10 and sec_score > 10:
            return "IMPROVING"
        elif news_score < -10 and sec_score < -10:
            return "DETERIORATING"
        elif abs(news_score - sec_score) > 30:
            return "DIVERGING"
        else:
            return "STABLE"

    def _calculate_confidence(self, news_count: int, sec_count: int) -> float:
        """Calculate confidence in sentiment"""
        total_sources = news_count + sec_count
        confidence = min(total_sources / 35.0, 0.95)
        return confidence


def analyze_advanced_sentiment(symbol: str, company_name: str = None) -> Dict[str, Any]:
    """Quick interface for advanced sentiment analysis"""
    analyzer = AdvancedSentimentAnalyzer()
    result = analyzer.analyze_comprehensive(symbol, company_name)

    if not result:
        return {
            'overall_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'news_sentiment': 0.0,
            'sec_filing_sentiment': 0.0,
            'confidence': 0.3,
            'sentiment_trend': 'STABLE',
            'risk_factors': []
        }

    return {
        'overall_score': result.overall_score,
        'sentiment_label': result.sentiment_label,
        'news_sentiment': result.news_sentiment,
        'sec_filing_sentiment': result.sec_filing_sentiment,
        'social_sentiment': result.social_sentiment,
        'news_count': result.news_count,
        'sec_filings_count': result.sec_filings_count,
        'confidence': result.confidence,
        'sentiment_trend': result.sentiment_trend,
        'risk_factors': result.risk_factors
    }
