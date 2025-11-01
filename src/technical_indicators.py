"""
TECHNICAL INDICATORS MODULE
Advanced technical analysis indicators for stock analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Dictionary with macd, signal, and histogram series
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: Series of closing prices
        window: Moving average window (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Dictionary with upper, middle, and lower bands
    """
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()

    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        prices: Series of closing prices
        volume: Series of volume data

    Returns:
        OBV series
    """
    obv = pd.Series(index=prices.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(prices)):
        if prices.iloc[i] > prices.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif prices.iloc[i] < prices.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    return obv


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default 14)

    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def find_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, float]:
    """
    Find key support and resistance levels.

    Args:
        prices: Series of closing prices
        window: Window for finding local extrema (default 20)

    Returns:
        Dictionary with support and resistance levels
    """
    recent_prices = prices.tail(window * 3)

    # Find local maxima (resistance)
    resistance_candidates = []
    for i in range(window, len(recent_prices) - window):
        if recent_prices.iloc[i] == recent_prices.iloc[i - window:i + window].max():
            resistance_candidates.append(recent_prices.iloc[i])

    # Find local minima (support)
    support_candidates = []
    for i in range(window, len(recent_prices) - window):
        if recent_prices.iloc[i] == recent_prices.iloc[i - window:i + window].min():
            support_candidates.append(recent_prices.iloc[i])

    current_price = prices.iloc[-1]

    # Find nearest support and resistance
    support = max([p for p in support_candidates if p < current_price], default=current_price * 0.95)
    resistance = min([p for p in resistance_candidates if p > current_price], default=current_price * 1.05)

    return {
        'support': support,
        'resistance': resistance,
        'current': current_price
    }


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)

    Returns:
        Dictionary with %K and %D series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()

    return {
        'k': k_percent,
        'd': d_percent
    }


def detect_macd_signal(macd_data: Dict[str, pd.Series]) -> str:
    """
    Detect MACD trading signals.

    Args:
        macd_data: Dictionary with macd, signal, and histogram

    Returns:
        Signal string (BULLISH_CROSS, BEARISH_CROSS, NEUTRAL)
    """
    macd = macd_data['macd']
    signal = macd_data['signal']

    if len(macd) < 2:
        return "NEUTRAL"

    # Bullish crossover (MACD crosses above signal)
    if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return "BULLISH_CROSS"

    # Bearish crossover (MACD crosses below signal)
    if macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return "BEARISH_CROSS"

    return "NEUTRAL"


def detect_bollinger_signal(prices: pd.Series, bb_data: Dict[str, pd.Series]) -> str:
    """
    Detect Bollinger Bands trading signals.

    Args:
        prices: Series of closing prices
        bb_data: Dictionary with upper, middle, lower bands

    Returns:
        Signal string (OVERSOLD, OVERBOUGHT, NEUTRAL)
    """
    current_price = prices.iloc[-1]
    upper = bb_data['upper'].iloc[-1]
    lower = bb_data['lower'].iloc[-1]

    if current_price <= lower:
        return "OVERSOLD"
    elif current_price >= upper:
        return "OVERBOUGHT"
    else:
        return "NEUTRAL"


def calculate_momentum_score(prices: pd.Series, volume: pd.Series) -> float:
    """
    Calculate overall momentum score (0-100).

    Args:
        prices: Series of closing prices
        volume: Series of volume data

    Returns:
        Momentum score from 0-100
    """
    score = 0

    # RSI component (30 points)
    rsi = calculate_rsi(prices)
    current_rsi = rsi.iloc[-1]
    if 40 < current_rsi < 60:
        score += 30
    elif 30 < current_rsi < 70:
        score += 20
    elif current_rsi < 30:
        score += 10  # Oversold = potential buy

    # MACD component (30 points)
    macd_data = calculate_macd(prices)
    macd_signal = detect_macd_signal(macd_data)
    if macd_signal == "BULLISH_CROSS":
        score += 30
    elif macd_signal == "NEUTRAL" and macd_data['histogram'].iloc[-1] > 0:
        score += 20

    # Price trend component (20 points)
    ma_20 = prices.rolling(20).mean()
    ma_50 = prices.rolling(50).mean()
    if len(ma_20) > 0 and len(ma_50) > 0:
        if ma_20.iloc[-1] > ma_50.iloc[-1]:
            score += 20
        elif ma_20.iloc[-1] > ma_50.iloc[-1] * 0.98:
            score += 10

    # Volume trend component (20 points)
    avg_volume = volume.tail(20).mean()
    recent_volume = volume.tail(5).mean()
    if recent_volume > avg_volume * 1.2:
        score += 20
    elif recent_volume > avg_volume:
        score += 10

    return min(score, 100)


def get_technical_summary(prices: pd.Series, volume: pd.Series,
                         high: Optional[pd.Series] = None,
                         low: Optional[pd.Series] = None) -> Dict[str, any]:
    """
    Get comprehensive technical analysis summary.

    Args:
        prices: Series of closing prices
        volume: Series of volume data
        high: Optional series of high prices
        low: Optional series of low prices

    Returns:
        Dictionary with all technical indicators and signals
    """
    summary = {}

    # RSI
    rsi = calculate_rsi(prices)
    summary['rsi'] = rsi.iloc[-1] if len(rsi) > 0 else 50

    # MACD
    macd_data = calculate_macd(prices)
    summary['macd'] = {
        'value': macd_data['macd'].iloc[-1],
        'signal': macd_data['signal'].iloc[-1],
        'histogram': macd_data['histogram'].iloc[-1],
        'trend': detect_macd_signal(macd_data)
    }

    # Bollinger Bands
    bb_data = calculate_bollinger_bands(prices)
    summary['bollinger'] = {
        'upper': bb_data['upper'].iloc[-1],
        'middle': bb_data['middle'].iloc[-1],
        'lower': bb_data['lower'].iloc[-1],
        'signal': detect_bollinger_signal(prices, bb_data)
    }

    # Support/Resistance
    sr_levels = find_support_resistance(prices)
    summary['support_resistance'] = sr_levels

    # OBV
    obv = calculate_obv(prices, volume)
    summary['obv'] = obv.iloc[-1]

    # Momentum Score
    summary['momentum_score'] = calculate_momentum_score(prices, volume)

    # Moving Averages
    summary['ma_20'] = prices.rolling(20).mean().iloc[-1]
    summary['ma_50'] = prices.rolling(50).mean().iloc[-1]
    summary['ma_200'] = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else None

    # ATR (if high/low provided)
    if high is not None and low is not None:
        atr = calculate_atr(high, low, prices)
        summary['atr'] = atr.iloc[-1]

        # Stochastic
        stoch = calculate_stochastic(high, low, prices)
        summary['stochastic'] = {
            'k': stoch['k'].iloc[-1],
            'd': stoch['d'].iloc[-1]
        }

    return summary
