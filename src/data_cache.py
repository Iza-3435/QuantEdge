"""
Data Caching & Optimization Utilities
Reduces API calls and speeds up data fetching
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import concurrent.futures
import yfinance as yf
from functools import wraps

CACHE_DIR = Path(__file__).parent.parent / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

CACHE_DURATION = timedelta(minutes=15)


class DataCache:
    @staticmethod
    def get_cache_path(key: str) -> Path:
        return CACHE_DIR / f"{key}.pkl"

    @staticmethod
    def get(key: str) -> Optional[Any]:
        cache_path = DataCache.get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)

            timestamp = cached.get('timestamp')
            if timestamp and datetime.now() - timestamp < CACHE_DURATION:
                return cached.get('data')

        except Exception:
            pass

        return None

    @staticmethod
    def set(key: str, data: Any) -> None:
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
    def clear_old() -> None:
        try:
            for cache_file in CACHE_DIR.glob('*.pkl'):
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)

                    timestamp = cached.get('timestamp')
                    if timestamp and datetime.now() - timestamp > CACHE_DURATION:
                        cache_file.unlink()
                except Exception:
                    continue
        except Exception:
            pass


def cached(key_func: Optional[Callable] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"

            cached_data = DataCache.get(cache_key)
            if cached_data is not None:
                return cached_data

            result = func(*args, **kwargs)
            DataCache.set(cache_key, result)
            return result

        return wrapper
    return decorator


def fetch_parallel(symbols: list, fetch_func: Callable, max_workers: int = 10) -> Dict[str, Any]:
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
def get_ticker_info(symbol: str) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except Exception:
        return {}


@cached(key_func=lambda symbol, period: f"ticker_history_{symbol}_{period}")
def get_ticker_history(symbol: str, period: str = '1y') -> Optional[Any]:
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    except Exception:
        return None


def get_data_freshness() -> Optional[datetime]:
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


def format_data_age(timestamp: Optional[datetime]) -> str:
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


DataCache.clear_old()
