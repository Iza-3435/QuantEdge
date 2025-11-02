"""
Institutional factor models: Fama-French, Momentum, Quality, Low Volatility.
Multi-factor analysis for stock selection and portfolio construction.
Used by AQR Capital, Dimensional Fund Advisors, BlackRock.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FactorExposure:
    market_beta: float
    size_exposure: float
    value_exposure: float
    momentum_exposure: float
    quality_exposure: float
    low_vol_exposure: float
    factor_score: float


class FactorAnalyzer:
    """Institutional factor model analyzer"""

    def __init__(self):
        pass

    def calculate_factor_exposures(self,
                                   price: float,
                                   market_cap: float,
                                   book_value: float,
                                   returns_1y: float,
                                   roe: float,
                                   debt_equity: float,
                                   volatility: float,
                                   market_returns: Optional[pd.Series] = None,
                                   stock_returns: Optional[pd.Series] = None) -> FactorExposure:
        """Calculate multi-factor exposures"""

        market_beta = self._calculate_beta(stock_returns, market_returns)

        size_exposure = self._size_factor(market_cap)

        value_exposure = self._value_factor(price, book_value, market_cap)

        momentum_exposure = self._momentum_factor(returns_1y)

        quality_exposure = self._quality_factor(roe, debt_equity)

        low_vol_exposure = self._low_vol_factor(volatility)

        factor_score = self._aggregate_factor_score(
            size_exposure, value_exposure, momentum_exposure,
            quality_exposure, low_vol_exposure
        )

        return FactorExposure(
            market_beta=market_beta,
            size_exposure=size_exposure,
            value_exposure=value_exposure,
            momentum_exposure=momentum_exposure,
            quality_exposure=quality_exposure,
            low_vol_exposure=low_vol_exposure,
            factor_score=factor_score
        )

    def _calculate_beta(self, stock_returns: Optional[pd.Series],
                       market_returns: Optional[pd.Series]) -> float:
        """Calculate market beta"""
        if stock_returns is None or market_returns is None:
            return 1.0

        if len(stock_returns) != len(market_returns) or len(stock_returns) < 20:
            return 1.0

        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()

        if market_variance == 0:
            return 1.0

        beta = covariance / market_variance
        return beta

    def _size_factor(self, market_cap: float) -> float:
        """Size factor (SMB - Small Minus Big)"""
        if market_cap > 200_000_000_000:
            return -1.0
        elif market_cap > 50_000_000_000:
            return -0.5
        elif market_cap > 10_000_000_000:
            return 0.0
        elif market_cap > 2_000_000_000:
            return 0.5
        else:
            return 1.0

    def _value_factor(self, price: float, book_value: float, market_cap: float) -> float:
        """Value factor (HML - High Minus Low)"""
        if book_value == 0 or market_cap == 0:
            return 0.0

        book_to_market = book_value / market_cap

        if book_to_market > 1.0:
            return 1.0
        elif book_to_market > 0.7:
            return 0.5
        elif book_to_market > 0.3:
            return 0.0
        elif book_to_market > 0.1:
            return -0.5
        else:
            return -1.0

    def _momentum_factor(self, returns_1y: float) -> float:
        """Momentum factor (UMD - Up Minus Down)"""
        if returns_1y > 50:
            return 1.0
        elif returns_1y > 20:
            return 0.5
        elif returns_1y > -10:
            return 0.0
        elif returns_1y > -30:
            return -0.5
        else:
            return -1.0

    def _quality_factor(self, roe: float, debt_equity: float) -> float:
        """Quality factor (QMJ - Quality Minus Junk)"""
        quality_score = 0.0

        if roe > 20:
            quality_score += 1.0
        elif roe > 15:
            quality_score += 0.5
        elif roe < 5:
            quality_score -= 0.5

        if debt_equity < 0.5:
            quality_score += 0.5
        elif debt_equity < 1.0:
            quality_score += 0.0
        elif debt_equity < 2.0:
            quality_score -= 0.5
        else:
            quality_score -= 1.0

        return np.clip(quality_score, -1.0, 1.0)

    def _low_vol_factor(self, volatility: float) -> float:
        """Low volatility factor"""
        if volatility < 15:
            return 1.0
        elif volatility < 25:
            return 0.5
        elif volatility < 35:
            return 0.0
        elif volatility < 50:
            return -0.5
        else:
            return -1.0

    def _aggregate_factor_score(self, size: float, value: float, momentum: float,
                                quality: float, low_vol: float) -> float:
        """Aggregate factor score with weights"""
        weights = {
            'value': 0.25,
            'momentum': 0.25,
            'quality': 0.25,
            'low_vol': 0.15,
            'size': 0.10
        }

        score = (
            value * weights['value'] +
            momentum * weights['momentum'] +
            quality * weights['quality'] +
            low_vol * weights['low_vol'] +
            size * weights['size']
        )

        return (score + 1) * 50


def analyze_factors(stock_data: Dict[str, Any]) -> Dict[str, float]:
    """Quick interface for factor analysis"""
    analyzer = FactorAnalyzer()

    exposures = analyzer.calculate_factor_exposures(
        price=stock_data.get('price', 0),
        market_cap=stock_data.get('market_cap', 0),
        book_value=stock_data.get('book_value', 0),
        returns_1y=stock_data.get('returns_1y', 0),
        roe=stock_data.get('roe', 0),
        debt_equity=stock_data.get('debt_equity', 0),
        volatility=stock_data.get('volatility', 20),
        market_returns=stock_data.get('market_returns'),
        stock_returns=stock_data.get('stock_returns')
    )

    return {
        'market_beta': exposures.market_beta,
        'size_exposure': exposures.size_exposure,
        'value_exposure': exposures.value_exposure,
        'momentum_exposure': exposures.momentum_exposure,
        'quality_exposure': exposures.quality_exposure,
        'low_vol_exposure': exposures.low_vol_exposure,
        'factor_score': exposures.factor_score
    }
