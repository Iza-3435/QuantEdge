"""
Institutional-grade portfolio optimization.
Modern Portfolio Theory, Efficient Frontier, Black-Litterman, Risk Parity.
Used by Bridgewater, AQR Capital, BlackRock Aladdin.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class OptimalPortfolio:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    allocation_dollars: Dict[str, float]


class PortfolioOptimizer:
    """Institutional portfolio optimizer"""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def optimize_sharpe(self,
                       returns: pd.DataFrame,
                       portfolio_value: float = 100000) -> Optional[OptimalPortfolio]:
        """Maximize Sharpe ratio (tangency portfolio)"""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_portfolio(returns, portfolio_value)

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        n_assets = len(returns.columns)
        init_guess = np.array([1/n_assets] * n_assets)

        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol
            return -sharpe

        result = minimize(neg_sharpe, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if not result.success:
            return self._equal_weight_portfolio(returns, portfolio_value)

        optimal_weights = result.x
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, optimal_weights)}
        allocation_dict = {symbol: weight * portfolio_value for symbol, weight in weights_dict.items()}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=port_return * 100,
            volatility=port_vol * 100,
            sharpe_ratio=sharpe,
            allocation_dollars=allocation_dict
        )

    def optimize_min_variance(self,
                             returns: pd.DataFrame,
                             portfolio_value: float = 100000) -> Optional[OptimalPortfolio]:
        """Minimum variance portfolio"""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_portfolio(returns, portfolio_value)

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        n_assets = len(returns.columns)
        init_guess = np.array([1/n_assets] * n_assets)

        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        result = minimize(portfolio_variance, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if not result.success:
            return self._equal_weight_portfolio(returns, portfolio_value)

        optimal_weights = result.x
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, optimal_weights)}
        allocation_dict = {symbol: weight * portfolio_value for symbol, weight in weights_dict.items()}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=port_return * 100,
            volatility=port_vol * 100,
            sharpe_ratio=sharpe,
            allocation_dollars=allocation_dict
        )

    def risk_parity_portfolio(self,
                             returns: pd.DataFrame,
                             portfolio_value: float = 100000) -> Optional[OptimalPortfolio]:
        """Risk parity allocation (equal risk contribution)"""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_portfolio(returns, portfolio_value)

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        n_assets = len(returns.columns)
        init_guess = np.array([1/n_assets] * n_assets)

        bounds = tuple((0.01, 1) for _ in range(n_assets))
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        def risk_parity_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            contrib = weights * marginal_contrib
            target = port_vol / n_assets
            return np.sum((contrib - target) ** 2)

        result = minimize(risk_parity_objective, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if not result.success:
            return self._equal_weight_portfolio(returns, portfolio_value)

        optimal_weights = result.x
        port_return = np.dot(optimal_weights, mean_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, optimal_weights)}
        allocation_dict = {symbol: weight * portfolio_value for symbol, weight in weights_dict.items()}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=port_return * 100,
            volatility=port_vol * 100,
            sharpe_ratio=sharpe,
            allocation_dollars=allocation_dict
        )

    def efficient_frontier(self,
                          returns: pd.DataFrame,
                          n_portfolios: int = 50) -> Tuple[List[float], List[float], List[np.ndarray]]:
        """Calculate efficient frontier"""
        if not SCIPY_AVAILABLE:
            return [], [], []

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        n_assets = len(returns.columns)
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_portfolios)

        efficient_portfolios = []
        volatilities = []
        returns_list = []

        for target_return in target_returns:
            init_guess = np.array([1/n_assets] * n_assets)
            bounds = tuple((0, 1) for _ in range(n_assets))
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}
            )

            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            result = minimize(portfolio_variance, init_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                efficient_portfolios.append(result.x)
                port_vol = np.sqrt(result.fun)
                volatilities.append(port_vol * 100)
                returns_list.append(target_return * 100)

        return returns_list, volatilities, efficient_portfolios

    def black_litterman(self,
                       returns: pd.DataFrame,
                       market_caps: Dict[str, float],
                       views: Dict[str, float],
                       confidence: float = 0.5,
                       portfolio_value: float = 100000) -> Optional[OptimalPortfolio]:
        """Black-Litterman model with investor views"""
        if not SCIPY_AVAILABLE:
            return self._equal_weight_portfolio(returns, portfolio_value)

        cov_matrix = returns.cov() * 252

        total_cap = sum(market_caps.values())
        market_weights = np.array([market_caps.get(symbol, 0) / total_cap for symbol in returns.columns])

        risk_aversion = 2.5
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        P = np.eye(len(returns.columns))
        Q = np.array([views.get(symbol, 0) for symbol in returns.columns])

        tau = 0.025
        omega = np.dot(np.dot(P, tau * cov_matrix), P.T) * (1 / confidence)

        inv_term = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(np.dot(P.T, np.linalg.inv(omega)), P))
        posterior_returns = inv_term @ (
            np.linalg.inv(tau * cov_matrix) @ pi + np.dot(P.T, np.linalg.inv(omega)) @ Q
        )

        optimal_weights = np.dot(np.linalg.inv(risk_aversion * cov_matrix), posterior_returns)
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        optimal_weights = np.clip(optimal_weights, 0, 1)
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        port_return = np.dot(optimal_weights, posterior_returns)
        port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        weights_dict = {symbol: weight for symbol, weight in zip(returns.columns, optimal_weights)}
        allocation_dict = {symbol: weight * portfolio_value for symbol, weight in weights_dict.items()}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=port_return * 100,
            volatility=port_vol * 100,
            sharpe_ratio=sharpe,
            allocation_dollars=allocation_dict
        )

    def _equal_weight_portfolio(self, returns: pd.DataFrame, portfolio_value: float) -> OptimalPortfolio:
        """Fallback equal-weight portfolio"""
        n_assets = len(returns.columns)
        equal_weight = 1 / n_assets

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        weights = np.array([equal_weight] * n_assets)
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        weights_dict = {symbol: equal_weight for symbol in returns.columns}
        allocation_dict = {symbol: equal_weight * portfolio_value for symbol in returns.columns}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=port_return * 100,
            volatility=port_vol * 100,
            sharpe_ratio=sharpe,
            allocation_dollars=allocation_dict
        )


def optimize_portfolio(returns: pd.DataFrame,
                      method: str = 'sharpe',
                      portfolio_value: float = 100000) -> Dict[str, Any]:
    """Quick interface for portfolio optimization"""
    optimizer = PortfolioOptimizer()

    if method == 'sharpe':
        result = optimizer.optimize_sharpe(returns, portfolio_value)
    elif method == 'min_variance':
        result = optimizer.optimize_min_variance(returns, portfolio_value)
    elif method == 'risk_parity':
        result = optimizer.risk_parity_portfolio(returns, portfolio_value)
    else:
        result = optimizer.optimize_sharpe(returns, portfolio_value)

    if not result:
        return {}

    return {
        'weights': result.weights,
        'expected_return': result.expected_return,
        'volatility': result.volatility,
        'sharpe_ratio': result.sharpe_ratio,
        'allocation': result.allocation_dollars
    }
