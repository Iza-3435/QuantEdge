"""
Data Caching & Optimization Utilities
Reduces API calls and speeds up data fetching
"""

import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
import yfinance as yf
from functools import wraps
import time

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

# Cache duration (15 minutes = free yfinance delay)
CACHE_DURATION = timedelta(minutes=15)


class DataCache:
    """Simple file-based cache for market data"""

    @staticmethod
    def get_cache_path(key):
        """Get cache file path for a key"""
        return CACHE_DIR / f"{key}.pkl"

    @staticmethod
    def get(key):
        """Get cached data if fresh"""
        cache_path = DataCache.get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)

            # Check if cache is still fresh
            timestamp = cached.get('timestamp')
            if timestamp and datetime.now() - timestamp < CACHE_DURATION:
                return cached.get('data')

        except Exception:
            pass

        return None

    @staticmethod
    def set(key, data):
        """Cache data with timestamp"""
        cache_path = DataCache.get_cache_path(key)

        try:
            cached = {
                'timestamp': datetime.now(),
                'data': data
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)
        except Exception:
            pass

    @staticmethod
    def clear_old():
        """Clear cache older than CACHE_DURATION"""
        try:
            for cache_file in CACHE_DIR.glob('*.pkl'):
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)

                    timestamp = cached.get('timestamp')
                    if timestamp and datetime.now() - timestamp > CACHE_DURATION:
                        cache_file.unlink()
                except:
                    continue
        except Exception:
            pass


def cached(key_func=None):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"

            # Try to get from cache
            cached_data = DataCache.get(cache_key)
            if cached_data is not None:
                return cached_data

            # Execute function and cache result
            result = func(*args, **kwargs)
            DataCache.set(cache_key, result)
            return result

        return wrapper
    return decorator


def fetch_parallel(symbols, fetch_func, max_workers=10):
    """Fetch data for multiple symbols in parallel"""
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_func, symbol): symbol
            for symbol in symbols
        }

        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception:
                results[symbol] = None

    return results


@cached(key_func=lambda symbol: f"ticker_info_{symbol}")
def get_ticker_info(symbol):
    """Get ticker info with caching"""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except Exception:
        return {}


@cached(key_func=lambda symbol, period: f"ticker_history_{symbol}_{period}")
def get_ticker_history(symbol, period='1y'):
    """Get ticker history with caching"""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    except Exception:
        return None


def get_data_freshness():
    """Get timestamp of when data was last updated"""
    # Check newest cache file
    try:
        cache_files = list(CACHE_DIR.glob('*.pkl'))
        if not cache_files:
            return None

        newest = max(cache_files, key=lambda f: f.stat().st_mtime)
        with open(newest, 'rb') as f:
            cached = pickle.load(f)

        return cached.get('timestamp')
    except Exception:
        return None


def format_data_age(timestamp):
    """Format how old the data is"""
    if not timestamp:
        return "Just now"

    age = datetime.now() - timestamp

    if age < timedelta(minutes=1):
        return "Just now"
    elif age < timedelta(hours=1):
        mins = int(age.total_seconds() / 60)
        return f"{mins} min ago"
    elif age < timedelta(days=1):
        hours = int(age.total_seconds() / 3600)
        return f"{hours} hr ago"
    else:
        days = age.days
        return f"{days} day ago"


# Auto-cleanup on import
DataCache.clear_old()
