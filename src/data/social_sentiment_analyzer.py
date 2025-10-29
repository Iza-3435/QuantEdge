"""
Social Media Sentiment Analyzer
Tracks Reddit (WallStreetBets), Twitter, StockTwits for retail sentiment

Why this matters:
- Retail traders can move stocks (GameStop, AMC)
- WallStreetBets often front-runs institutional moves
- High social buzz predicts volatility
- Sentiment spikes can predict short squeezes
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SocialPost:
    """Single social media post."""
    platform: str  # 'reddit', 'twitter', 'stocktwits'
    author: str
    text: str
    sentiment: float  # -1 to 1
    upvotes: int
    comments: int
    timestamp: datetime
    subreddit: Optional[str] = None  # For Reddit
    ticker_mentions: List[str] = None


@dataclass
class SocialSentimentSignal:
    """Social media sentiment signal."""
    signal: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: float  # 0-1
    confidence: float

    # Metrics
    total_mentions_24h: int
    total_mentions_7d: int
    mention_growth_rate: float  # % change day over day
    average_sentiment: float  # -1 to 1
    sentiment_std: float  # Sentiment volatility

    # Platform breakdown
    reddit_mentions: int
    reddit_sentiment: float
    twitter_mentions: int
    twitter_sentiment: float
    stocktwits_mentions: int
    stocktwits_sentiment: float

    # Trending analysis
    is_trending: bool
    trending_rank: Optional[int]  # 1-100
    buzz_score: float  # 0-100

    # Risk indicators
    wsb_gamma_squeeze_risk: float  # 0-1
    short_squeeze_probability: float  # 0-1
    meme_stock_indicator: bool

    # Insights
    insights: List[str]
    bullish_signals: List[str]
    bearish_signals: List[str]


class SocialSentimentAnalyzer:
    """
    Analyzes social media sentiment from multiple platforms.

    Data sources (in production):
    - Reddit API (r/wallstreetbets, r/stocks, r/investing)
    - Twitter API (Financial Twitter, $TICKER mentions)
    - StockTwits API
    - Alternative data providers (Sentiment Investor, Quiver Quant)
    """

    def __init__(self):
        self.platforms = ['reddit', 'twitter', 'stocktwits']

        # Sentiment keywords
        self.bullish_keywords = [
            'moon', 'rocket', 'buy', 'calls', 'bullish', 'green', 'pump',
            'squeeze', 'breakout', 'lambo', 'tendies', 'yolo', 'diamond hands',
            'hold', 'hodl', 'to the moon', '🚀', '📈', '💎', '🙌'
        ]

        self.bearish_keywords = [
            'puts', 'bearish', 'red', 'dump', 'crash', 'sell', 'short',
            'bag holder', 'paper hands', 'dead', 'tanking', '📉', '🐻'
        ]

    def get_social_posts(
        self,
        ticker: str,
        hours_back: int = 24,
        platforms: Optional[List[str]] = None
    ) -> List[SocialPost]:
        """
        Get social media posts mentioning ticker.

        In production, would fetch from:
        - Reddit API (PRAW library)
        - Twitter API v2
        - StockTwits API
        - Pushshift for historical Reddit data

        For now, returns mock data with realistic patterns.
        """
        return self._generate_mock_posts(ticker, hours_back, platforms)

    def _generate_mock_posts(
        self,
        ticker: str,
        hours: int,
        platforms: Optional[List[str]]
    ) -> List[SocialPost]:
        """Generate realistic mock social media posts."""
        posts = []
        platforms_to_use = platforms or self.platforms

        np.random.seed(hash(ticker) % 2**32)

        # Generate posts
        num_posts = np.random.randint(50, 500)  # Variable buzz

        for i in range(num_posts):
            platform = np.random.choice(platforms_to_use)
            hours_ago = np.random.uniform(0, hours)

            # Sentiment (-1 to 1)
            # Bias towards slightly bullish (retail tends to be optimistic)
            sentiment = np.random.beta(5, 4) * 2 - 1  # Slightly bullish bias

            # Engagement metrics
            if platform == 'reddit':
                upvotes = int(np.random.lognormal(3, 2))  # Long tail distribution
                comments = int(upvotes * np.random.uniform(0.05, 0.2))
            elif platform == 'twitter':
                upvotes = int(np.random.lognormal(2, 1.5))  # Likes
                comments = int(upvotes * np.random.uniform(0.1, 0.3))  # Retweets
            else:  # stocktwits
                upvotes = int(np.random.lognormal(1, 1))
                comments = int(upvotes * np.random.uniform(0.05, 0.15))

            # Generate text based on sentiment
            if sentiment > 0.5:
                text = f"${ticker} calls printing! 🚀 to the moon!"
            elif sentiment > 0:
                text = f"Bullish on ${ticker}, looking good"
            elif sentiment > -0.5:
                text = f"${ticker} might pull back, be careful"
            else:
                text = f"${ticker} puts all day, this is tanking"

            posts.append(SocialPost(
                platform=platform,
                author=f"user_{i}",
                text=text,
                sentiment=sentiment,
                upvotes=upvotes,
                comments=comments,
                timestamp=datetime.now() - timedelta(hours=hours_ago),
                subreddit='wallstreetbets' if platform == 'reddit' else None,
                ticker_mentions=[ticker]
            ))

        return posts

    def analyze_social_sentiment(
        self,
        ticker: str
    ) -> SocialSentimentSignal:
        """
        Analyze social media sentiment and generate signal.

        Key insights:
        - Spike in mentions = incoming volatility
        - High positive sentiment = potential pump
        - WSB trending = gamma squeeze risk
        - Sentiment divergence from price = opportunity
        """
        # Get posts from different timeframes
        posts_24h = self.get_social_posts(ticker, hours_back=24)
        posts_7d = self.get_social_posts(ticker, hours_back=168)

        if not posts_24h:
            return self._no_activity_signal()

        # Calculate metrics
        total_24h = len(posts_24h)
        total_7d = len(posts_7d)
        avg_daily = total_7d / 7
        mention_growth = ((total_24h - avg_daily) / avg_daily * 100) if avg_daily > 0 else 0

        # Sentiment analysis
        sentiments = [p.sentiment for p in posts_24h]
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)

        # Platform breakdown
        reddit_posts = [p for p in posts_24h if p.platform == 'reddit']
        twitter_posts = [p for p in posts_24h if p.platform == 'twitter']
        stocktwits_posts = [p for p in posts_24h if p.platform == 'stocktwits']

        reddit_mentions = len(reddit_posts)
        reddit_sentiment = np.mean([p.sentiment for p in reddit_posts]) if reddit_posts else 0

        twitter_mentions = len(twitter_posts)
        twitter_sentiment = np.mean([p.sentiment for p in twitter_posts]) if twitter_posts else 0

        stocktwits_mentions = len(stocktwits_posts)
        stocktwits_sentiment = np.mean([p.sentiment for p in stocktwits_posts]) if stocktwits_posts else 0

        # Trending analysis
        buzz_score = min((total_24h / 100) * 100, 100)  # Scale to 0-100
        is_trending = total_24h > 200  # More than 200 mentions = trending
        trending_rank = int(max(1, 100 - buzz_score)) if is_trending else None

        # Risk indicators
        # Weighted by upvotes (viral posts matter more)
        weighted_sentiment = np.average(
            sentiments,
            weights=[p.upvotes + 1 for p in posts_24h]
        )

        # Gamma squeeze risk (high mentions + high call sentiment on WSB)
        wsb_call_mentions = sum(1 for p in reddit_posts if 'call' in p.text.lower() or '🚀' in p.text)
        gamma_squeeze_risk = min((wsb_call_mentions / max(reddit_mentions, 1)) * mention_growth / 100, 1.0)

        # Short squeeze probability
        short_squeeze_keywords = sum(1 for p in posts_24h if 'squeeze' in p.text.lower())
        short_squeeze_prob = min(short_squeeze_keywords / max(total_24h, 1) * 10, 1.0)

        # Meme stock indicator
        meme_keywords = ['moon', 'rocket', 'diamond', 'hands', 'ape', 'retard', 'yolo', 'tendies']
        meme_mentions = sum(1 for p in posts_24h if any(kw in p.text.lower() for kw in meme_keywords))
        is_meme_stock = meme_mentions > total_24h * 0.2  # >20% meme language

        # Generate insights
        insights = []
        bullish_signals = []
        bearish_signals = []

        # Bullish signals
        if mention_growth > 100:
            insights.append(f"Viral growth: +{mention_growth:.0f}% mentions in 24h")
            bullish_signals.append("Going viral on social media")

        if avg_sentiment > 0.4:
            insights.append(f"Strong positive sentiment: {avg_sentiment:+.2f}")
            bullish_signals.append("Overwhelmingly bullish retail sentiment")

        if gamma_squeeze_risk > 0.5:
            insights.append("High gamma squeeze risk detected")
            bullish_signals.append("Potential gamma squeeze setup")

        if short_squeeze_prob > 0.3:
            insights.append("Short squeeze chatter detected")
            bullish_signals.append("Short squeeze discussion trending")

        if is_trending:
            insights.append(f"Trending #{ trending_rank} on social media")
            bullish_signals.append("High social media buzz")

        # Bearish signals
        if mention_growth < -50:
            insights.append(f"Losing interest: {mention_growth:.0f}% drop in mentions")
            bearish_signals.append("Fading retail interest")

        if avg_sentiment < -0.3:
            insights.append(f"Negative sentiment: {avg_sentiment:+.2f}")
            bearish_signals.append("Bearish retail sentiment")

        if sentiment_std > 0.6:
            insights.append("High sentiment volatility - conflicting opinions")
            bearish_signals.append("Divided retail sentiment")

        # Calculate signal
        if avg_sentiment > 0.3 and mention_growth > 50:
            signal = "bullish"
            signal_strength = min((avg_sentiment + mention_growth/200), 1.0)
            confidence = 0.55 + signal_strength * 0.25
        elif avg_sentiment < -0.3 and mention_growth < -25:
            signal = "bearish"
            signal_strength = min((abs(avg_sentiment) + abs(mention_growth)/200), 1.0)
            confidence = 0.55 + signal_strength * 0.25
        else:
            signal = "neutral"
            signal_strength = 0.3
            confidence = 0.5

        # Boost confidence if going viral
        if is_trending:
            confidence = min(confidence + 0.1, 0.85)

        return SocialSentimentSignal(
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            total_mentions_24h=total_24h,
            total_mentions_7d=total_7d,
            mention_growth_rate=mention_growth,
            average_sentiment=avg_sentiment,
            sentiment_std=sentiment_std,
            reddit_mentions=reddit_mentions,
            reddit_sentiment=reddit_sentiment,
            twitter_mentions=twitter_mentions,
            twitter_sentiment=twitter_sentiment,
            stocktwits_mentions=stocktwits_mentions,
            stocktwits_sentiment=stocktwits_sentiment,
            is_trending=is_trending,
            trending_rank=trending_rank,
            buzz_score=buzz_score,
            wsb_gamma_squeeze_risk=gamma_squeeze_risk,
            short_squeeze_probability=short_squeeze_prob,
            meme_stock_indicator=is_meme_stock,
            insights=insights if insights else ["Moderate social media activity"],
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals
        )

    def _no_activity_signal(self) -> SocialSentimentSignal:
        """Return neutral signal for no data."""
        return SocialSentimentSignal(
            signal="neutral",
            signal_strength=0.0,
            confidence=0.3,
            total_mentions_24h=0,
            total_mentions_7d=0,
            mention_growth_rate=0.0,
            average_sentiment=0.0,
            sentiment_std=0.0,
            reddit_mentions=0,
            reddit_sentiment=0.0,
            twitter_mentions=0,
            twitter_sentiment=0.0,
            stocktwits_mentions=0,
            stocktwits_sentiment=0.0,
            is_trending=False,
            trending_rank=None,
            buzz_score=0.0,
            wsb_gamma_squeeze_risk=0.0,
            short_squeeze_probability=0.0,
            meme_stock_indicator=False,
            insights=["Minimal social media activity"],
            bullish_signals=[],
            bearish_signals=[]
        )


def quick_social_test(ticker: str = "AAPL"):
    """Quick test of social sentiment analyzer."""
    print(f"\n{'='*80}")
    print(f"SOCIAL MEDIA SENTIMENT ANALYSIS: {ticker}")
    print(f"{'='*80}\n")

    analyzer = SocialSentimentAnalyzer()
    signal = analyzer.analyze_social_sentiment(ticker)

    # Display signal
    signal_emoji = "🟢" if signal.signal == "bullish" else "🔴" if signal.signal == "bearish" else "⚪"

    print(f"{signal_emoji} SOCIAL SENTIMENT: {signal.signal.upper()}")
    print(f"  Strength:              {signal.signal_strength:.0%}")
    print(f"  Confidence:            {signal.confidence:.0%}")
    print(f"  Buzz Score:            {signal.buzz_score:.0f}/100")
    print()
    print("📊 MENTION METRICS:")
    print(f"  24h Mentions:          {signal.total_mentions_24h}")
    print(f"  7d Mentions:           {signal.total_mentions_7d}")
    print(f"  Growth Rate:           {signal.mention_growth_rate:+.1f}%")
    print(f"  Avg Sentiment:         {signal.average_sentiment:+.2f}")
    print(f"  Is Trending:           {'YES' if signal.is_trending else 'NO'}")
    if signal.trending_rank:
        print(f"  Trending Rank:         #{signal.trending_rank}")
    print()
    print("🌐 PLATFORM BREAKDOWN:")
    print(f"  Reddit:                {signal.reddit_mentions} posts ({signal.reddit_sentiment:+.2f})")
    print(f"  Twitter:               {signal.twitter_mentions} posts ({signal.twitter_sentiment:+.2f})")
    print(f"  StockTwits:            {signal.stocktwits_mentions} posts ({signal.stocktwits_sentiment:+.2f})")
    print()
    print("⚠️  RISK INDICATORS:")
    print(f"  Gamma Squeeze Risk:    {signal.wsb_gamma_squeeze_risk:.0%}")
    print(f"  Short Squeeze Prob:    {signal.short_squeeze_probability:.0%}")
    print(f"  Meme Stock:            {'YES' if signal.meme_stock_indicator else 'NO'}")
    print()

    if signal.bullish_signals:
        print("🟢 BULLISH SIGNALS:")
        for sig in signal.bullish_signals:
            print(f"  ✅ {sig}")
        print()

    if signal.bearish_signals:
        print("🔴 BEARISH SIGNALS:")
        for sig in signal.bearish_signals:
            print(f"  ⚠️  {sig}")
        print()

    print("💡 INSIGHTS:")
    for insight in signal.insights:
        print(f"  • {insight}")
    print()
    print(f"{'='*80}\n")


if __name__ == "__main__":
    quick_social_test("AAPL")
