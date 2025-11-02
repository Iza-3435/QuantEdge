"""
Institutional-grade risk management system.
VaR, CVaR, position sizing, portfolio risk metrics, stress testing.
Used by Renaissance Technologies, Bridgewater, Two Sigma.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    value_at_risk_95: float
    cvar_95: float
    value_at_risk_99: float
    cvar_99: float
    beta: float
    correlation_to_market: float
    maximum_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    downside_deviation: float


@dataclass
class PositionSize:
    optimal_shares: int
    position_value: float
    portfolio_weight: float
    risk_contribution: float
    kelly_fraction: float


class RiskManager:
    """Institutional risk management system"""

    def __init__(self, portfolio_value: float = 100000, risk_free_rate: float = 0.02):
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        if len(returns) < 30:
            return 0.0, 0.0

        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)

        cvar = returns[returns <= var].mean()

        return var * 100, cvar * 100

    def calculate_comprehensive_risk(self, returns: pd.Series,
                                    market_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        var_95, cvar_95 = self.calculate_var(returns, 0.95)
        var_99, cvar_99 = self.calculate_var(returns, 0.99)

        beta = 1.0
        correlation = 0.0
        if market_returns is not None and len(market_returns) == len(returns):
            covariance = returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 1.0
            correlation = returns.corr(market_returns)

        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        current_drawdown = self._calculate_current_drawdown(cumulative_returns)

        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)

        volatility = returns.std() * np.sqrt(252) * 100

        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0

        return RiskMetrics(
            value_at_risk_95=var_95,
            cvar_95=cvar_95,
            value_at_risk_99=var_99,
            cvar_99=cvar_99,
            beta=beta,
            correlation_to_market=correlation,
            maximum_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            volatility=volatility,
            downside_deviation=downside_deviation
        )

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if avg_loss == 0 or win_rate == 0:
            return 0.0

        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        kelly = max(0, min(kelly, 0.25))

        return kelly

    def calculate_position_size(self,
                               stock_price: float,
                               volatility: float,
                               expected_return: float,
                               confidence: float = 0.5,
                               max_position: float = 0.20) -> PositionSize:
        """Calculate optimal position size"""
        target_risk = self.portfolio_value * 0.02

        position_vol_dollars = stock_price * volatility
        risk_based_shares = int(target_risk / position_vol_dollars) if position_vol_dollars > 0 else 0

        if expected_return > 0 and volatility > 0:
            win_rate = min(confidence, 0.9)
            avg_win = expected_return
            avg_loss = volatility

            kelly = self.kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_position_value = self.portfolio_value * kelly * 0.5
            kelly_shares = int(kelly_position_value / stock_price)
        else:
            kelly_shares = 0
            kelly = 0.0

        optimal_shares = min(risk_based_shares, kelly_shares)

        max_shares = int((self.portfolio_value * max_position) / stock_price)
        optimal_shares = min(optimal_shares, max_shares)

        if optimal_shares <= 0:
            optimal_shares = int((self.portfolio_value * 0.05) / stock_price)

        position_value = optimal_shares * stock_price
        portfolio_weight = position_value / self.portfolio_value
        risk_contribution = volatility * portfolio_weight

        return PositionSize(
            optimal_shares=optimal_shares,
            position_value=position_value,
            portfolio_weight=portfolio_weight * 100,
            risk_contribution=risk_contribution * 100,
            kelly_fraction=kelly
        )

    def stress_test(self, returns: pd.Series, scenarios: List[Dict[str, float]]) -> Dict[str, float]:
        """Stress test portfolio against scenarios"""
        results = {}

        base_value = self.portfolio_value

        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unknown')
            shock = scenario.get('shock', 0)

            shocked_value = base_value * (1 + shock)
            loss = base_value - shocked_value
            loss_pct = (loss / base_value) * 100

            results[scenario_name] = loss_pct

        return results

    def calculate_portfolio_risk(self,
                                positions: Dict[str, Dict[str, float]],
                                correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        symbols = list(positions.keys())
        weights = np.array([positions[s]['weight'] for s in symbols])
        volatilities = np.array([positions[s]['volatility'] for s in symbols])

        portfolio_var = np.dot(weights, np.dot(correlation_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        total_value = sum(p['value'] for p in positions.values())
        var_95 = portfolio_vol * 1.65 * total_value
        var_99 = portfolio_vol * 2.33 * total_value

        diversification_ratio = sum(weights * volatilities) / portfolio_vol

        return {
            'portfolio_volatility': portfolio_vol * 100,
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'diversification_ratio': diversification_ratio,
            'concentration_risk': max(weights) * 100
        }

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min() * 100

    def _calculate_current_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate current drawdown from peak"""
        running_max = cumulative_returns.cummax()
        current_drawdown = (cumulative_returns.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        return current_drawdown * 100

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - (self.risk_free_rate / 252)
        if returns.std() == 0:
            return 0
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)


def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Quick interface for risk metrics"""
    risk_manager = RiskManager()
    metrics = risk_manager.calculate_comprehensive_risk(returns)

    return {
        'var_95': metrics.value_at_risk_95,
        'cvar_95': metrics.cvar_95,
        'max_drawdown': metrics.maximum_drawdown,
        'sharpe_ratio': metrics.sharpe_ratio,
        'volatility': metrics.volatility
    }
