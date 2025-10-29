"""Real-time News Collection for Market Intelligence"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
from pathlib import Path
import json
import time


class NewsCollector:
    """
    Collect real financial news from multiple sources.

    Sources:
    - Yahoo Finance (free, no API key needed)
    - Alpha Vantage (free tier: 25 calls/day)
    - Financial Modeling Prep (backup)
    """

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.cache_dir = Path("data/cache/news")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_yahoo_news(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get news from Yahoo Finance (free, no API key).

        Returns list of:
        {
            'title': str,
            'published': datetime,
            'summary': str,
            'url': str,
            'source': str
        }
        """
        print(f"Fetching Yahoo Finance news for {symbol}...")

        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            parsed_news = []
            for item in news[:limit]:
                parsed_news.append({
                    'title': item.get('title', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'summary': item.get('summary', item.get('title', '')),
                    'url': item.get('link', ''),
                    'source': 'yahoo_finance',
                    'symbol': symbol
                })

            print(f"✅ Fetched {len(parsed_news)} news articles")
            return parsed_news

        except Exception as e:
            print(f"❌ Yahoo Finance error: {e}")
            return []

    def get_alpha_vantage_news(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get news from Alpha Vantage API.

        Requires API key (free tier: 25 calls/day).
        Get key at: https://www.alphavantage.co/support/#api-key
        """
        if not self.alpha_vantage_key:
            print("⚠️  Alpha Vantage API key not provided, skipping...")
            return []

        print(f"Fetching Alpha Vantage news for {symbol}...")

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': self.alpha_vantage_key,
            'limit': limit
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if 'feed' not in data:
                print(f"❌ No news data returned: {data.get('Note', 'Unknown error')}")
                return []

            parsed_news = []
            for item in data['feed']:
                # Parse timestamp
                time_str = item.get('time_published', '')
                try:
                    published = datetime.strptime(time_str, '%Y%m%dT%H%M%S')
                except:
                    published = datetime.now()

                # Extract sentiment score
                ticker_sentiment = item.get('ticker_sentiment', [])
                sentiment_score = 0.0
                for ts in ticker_sentiment:
                    if ts.get('ticker') == symbol:
                        sentiment_score = float(ts.get('ticker_sentiment_score', 0))
                        break

                parsed_news.append({
                    'title': item.get('title', ''),
                    'published': published,
                    'summary': item.get('summary', '')[:500],  # Limit length
                    'url': item.get('url', ''),
                    'source': 'alpha_vantage',
                    'symbol': symbol,
                    'sentiment_score': sentiment_score,
                    'relevance_score': item.get('relevance_score', 0)
                })

            print(f"✅ Fetched {len(parsed_news)} news articles with sentiment")
            return parsed_news

        except Exception as e:
            print(f"❌ Alpha Vantage error: {e}")
            return []

    def get_historical_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical news for a symbol.

        Uses caching to avoid repeated API calls.
        """
        cache_file = self.cache_dir / f"{symbol}_{start_date.date()}_{end_date.date()}.json"

        # Check cache
        if cache and cache_file.exists():
            print(f"📁 Loading from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['published'] = pd.to_datetime(df['published'])
                return df

        # Collect from sources
        all_news = []

        # Yahoo Finance (current news only, can't get historical easily)
        yahoo_news = self.get_yahoo_news(symbol, limit=100)
        all_news.extend(yahoo_news)

        # Alpha Vantage
        if self.alpha_vantage_key:
            time.sleep(1)  # Rate limiting
            av_news = self.get_alpha_vantage_news(symbol, limit=200)
            all_news.extend(av_news)

        # Convert to DataFrame
        if not all_news:
            print("⚠️  No news found")
            return pd.DataFrame()

        df = pd.DataFrame(all_news)
        df['published'] = pd.to_datetime(df['published'])

        # Filter by date range
        df = df[(df['published'] >= start_date) & (df['published'] <= end_date)]

        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')

        # Sort by date
        df = df.sort_values('published')

        # Save to cache
        if cache:
            df_cache = df.copy()
            df_cache['published'] = df_cache['published'].astype(str)
            with open(cache_file, 'w') as f:
                json.dump(df_cache.to_dict('records'), f, indent=2)
            print(f"💾 Cached to: {cache_file}")

        print(f"✅ Total unique news: {len(df)}")
        return df

    def create_news_price_pairs(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        forward_window: int = 5
    ) -> pd.DataFrame:
        """
        Create training pairs: (news, price_before, price_after).

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            forward_window: Days to look forward for outcome

        Returns:
            DataFrame with columns:
            - news_text: headline + summary
            - news_date: publication date
            - price_before: price features before news
            - price_after: return after news (target)
            - sentiment: sentiment score if available
        """
        print(f"\n{'='*70}")
        print(f"Creating News-Price Training Pairs for {symbol}")
        print(f"{'='*70}")

        # Get news
        news_df = self.get_historical_news(symbol, start_date, end_date)

        if len(news_df) == 0:
            print("❌ No news data to pair")
            return pd.DataFrame()

        # Get price data
        print(f"\nFetching price data...")
        price_df = yf.download(
            symbol,
            start=start_date - timedelta(days=30),  # Extra for features
            end=end_date + timedelta(days=forward_window + 5),
            progress=False
        )

        # Add technical indicators
        price_df['Returns'] = price_df['Close'].pct_change()
        price_df['SMA_20'] = price_df['Close'].rolling(20).mean()
        price_df['Volatility'] = price_df['Returns'].rolling(20).std()

        print(f"✅ Price data: {len(price_df)} days")

        # Create pairs
        pairs = []

        for idx, news_row in news_df.iterrows():
            news_date = news_row['published']

            # Find closest trading day
            try:
                # Get price data around news date
                news_date_normalized = news_date.normalize()

                # Find nearest trading day
                nearest_idx = price_df.index.searchsorted(news_date_normalized)

                if nearest_idx >= len(price_df) - forward_window or nearest_idx < 20:
                    continue

                news_day = price_df.index[nearest_idx]

                # Price before news (features)
                price_before = price_df.iloc[nearest_idx - 1]

                # Price after news (outcome)
                price_after_idx = min(nearest_idx + forward_window, len(price_df) - 1)
                price_start = price_df['Close'].iloc[nearest_idx]
                price_end = price_df['Close'].iloc[price_after_idx]

                forward_return = (price_end / price_start - 1) * 100  # Percentage

                # Combine title + summary
                news_text = news_row['title']
                if news_row.get('summary') and news_row['summary'] != news_row['title']:
                    news_text += " " + news_row['summary'][:200]

                pair = {
                    'news_text': news_text,
                    'news_date': news_date,
                    'trading_date': news_day,
                    'symbol': symbol,

                    # Price features (input)
                    'price': float(price_before['Close']),
                    'sma_20': float(price_before['SMA_20']),
                    'volatility': float(price_before['Volatility']) if not pd.isna(price_before['Volatility']) else 0.01,
                    'recent_return': float(price_before['Returns']) if not pd.isna(price_before['Returns']) else 0.0,

                    # Outcome (target)
                    'forward_return': float(forward_return),
                    'forward_days': forward_window,

                    # Metadata
                    'source': news_row['source'],
                    'sentiment_score': news_row.get('sentiment_score', 0.0),
                    'url': news_row.get('url', '')
                }

                pairs.append(pair)

            except Exception as e:
                continue

        pairs_df = pd.DataFrame(pairs)

        print(f"\n✅ Created {len(pairs_df)} news-price pairs")
        print(f"   Date range: {pairs_df['news_date'].min()} to {pairs_df['news_date'].max()}")
        print(f"   Avg forward return: {pairs_df['forward_return'].mean():.2f}%")
        print(f"   Positive outcomes: {(pairs_df['forward_return'] > 0).sum()} ({(pairs_df['forward_return'] > 0).mean()*100:.1f}%)")

        return pairs_df


if __name__ == "__main__":
    print("Testing News Collector...")
    print("="*70)

    # Initialize (add your Alpha Vantage key if you have one)
    collector = NewsCollector(
        alpha_vantage_key=None  # Get free key at https://www.alphavantage.co/support/#api-key
    )

    # Test Yahoo Finance news
    print("\n1. Testing Yahoo Finance news collection...")
    recent_news = collector.get_yahoo_news('AAPL', limit=10)

    if recent_news:
        print(f"\nSample news:")
        for i, news in enumerate(recent_news[:3], 1):
            print(f"\n{i}. {news['title']}")
            print(f"   Published: {news['published']}")
            print(f"   Source: {news['source']}")

    # Test news-price pairing
    print("\n\n2. Testing news-price pair creation...")
    pairs_df = collector.create_news_price_pairs(
        symbol='AAPL',
        start_date=datetime.now() - timedelta(days=90),
        end_date=datetime.now(),
        forward_window=5
    )

    if len(pairs_df) > 0:
        print("\n" + "="*70)
        print("SAMPLE TRAINING PAIRS")
        print("="*70)

        for idx in range(min(3, len(pairs_df))):
            row = pairs_df.iloc[idx]
            print(f"\nPair {idx+1}:")
            print(f"News: {row['news_text'][:100]}...")
            print(f"Date: {row['news_date']}")
            print(f"Price before: ${row['price']:.2f}")
            print(f"Forward return: {row['forward_return']:+.2f}%")

        # Save sample
        output_file = Path("data/processed/news_price_pairs_sample.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pairs_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved sample to: {output_file}")

    print("\n" + "="*70)
    print("✅ News collector ready for production use")
    print("="*70)
