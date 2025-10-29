"""Advanced Portfolio Optimization and Risk Analysis."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PortfolioMetrics:
    """Portfolio performance and risk metrics."""
    returns_annual: float
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    calmar_ratio: float
    weights: Dict[str, float]
    diversification_ratio: float


@dataclass
class RiskContribution:
    """Risk contribution analysis."""
    total_risk: float
    marginal_risk: Dict[str, float]
    risk_contribution: Dict[str, float]
    risk_parity_weights: Dict[str, float]


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization and risk analysis.

    Features:
    - Mean-Variance Optimization (Markowitz)
    - Black-Litterman Asset Allocation
    - Risk Parity Portfolio
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Maximum Diversification
    - Conditional Value at Risk (CVaR) optimization
    - Risk attribution and decomposition
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = {}
        self.covariance_matrix = None
        self.symbols = []

    def load_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "2y"
    ):
        """Load historical data for portfolio symbols."""
        self.symbols = symbols

        if start_date and end_date:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False
            )
        else:
            data = yf.download(symbols, period=period, progress=False)

        # Extract close prices
        if len(symbols) == 1:
            prices = data['Close']
        else:
            prices = data['Close']

        # Calculate returns
        returns = prices.pct_change().dropna()

        self.returns_data = returns
        self.covariance_matrix = returns.cov()

        return returns

    def optimize_max_sharpe(
        self,
        target_return: Optional[float] = None
    ) -> PortfolioMetrics:
        """
        Optimize for maximum Sharpe ratio.

        Args:
            target_return: Optional target return constraint

        Returns:
            PortfolioMetrics with optimal weights
        """
        n_assets = len(self.symbols)
        returns = self.returns_data

        # Calculate expected returns and covariance
        mean_returns = returns.mean() * 252  # Annualize
        cov_matrix = returns.cov() * 252

        def neg_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(mean_returns * x) - target_return
            })

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        weights_dict = dict(zip(self.symbols, optimal_weights))

        # Calculate metrics
        metrics = self._calculate_portfolio_metrics(optimal_weights)

        return metrics

    def optimize_min_variance(self) -> PortfolioMetrics:
        """Optimize for minimum variance (minimum risk)."""
        n_assets = len(self.symbols)
        returns = self.returns_data
        cov_matrix = returns.cov() * 252

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        metrics = self._calculate_portfolio_metrics(optimal_weights)

        return metrics

    def optimize_risk_parity(self) -> PortfolioMetrics:
        """
        Risk parity portfolio optimization.

        Each asset contributes equally to portfolio risk.
        """
        n_assets = len(self.symbols)
        returns = self.returns_data
        cov_matrix = returns.cov() * 252

        def risk_parity_objective(weights):
            """Minimize difference in risk contributions."""
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance

            # Minimize variance of risk contributions
            target_contrib = 1 / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Bounds
        bounds = tuple((0.01, 0.5) for _ in range(n_assets))

        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        metrics = self._calculate_portfolio_metrics(optimal_weights)

        return metrics

    def optimize_cvar(
        self,
        alpha: float = 0.05,
        target_return: Optional[float] = None
    ) -> PortfolioMetrics:
        """
        Optimize portfolio to minimize Conditional Value at Risk (CVaR).

        CVaR is the expected loss beyond VaR.

        Args:
            alpha: Confidence level (0.05 = 95% confidence)
            target_return: Optional target return constraint
        """
        n_assets = len(self.symbols)
        returns = self.returns_data

        def calculate_cvar(weights):
            """Calculate CVaR for given weights."""
            portfolio_returns = returns @ weights

            # VaR
            var = np.percentile(portfolio_returns, alpha * 100)

            # CVaR: mean of returns below VaR
            cvar = portfolio_returns[portfolio_returns <= var].mean()

            return -cvar  # Minimize negative CVaR

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if target_return is not None:
            mean_returns = returns.mean() * 252
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(mean_returns * x) - target_return
            })

        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            calculate_cvar,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        metrics = self._calculate_portfolio_metrics(optimal_weights)

        return metrics

    def efficient_frontier(
        self,
        num_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.

        Returns:
            DataFrame with returns, volatility, and Sharpe ratios
        """
        n_assets = len(self.symbols)
        returns = self.returns_data
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Find min/max returns
        min_return = mean_returns.min()
        max_return = mean_returns.max()

        target_returns = np.linspace(min_return, max_return, num_portfolios)

        results = []

        for target_ret in target_returns:
            try:
                metrics = self.optimize_max_sharpe(target_return=target_ret)

                results.append({
                    'Return': metrics.returns_annual,
                    'Volatility': metrics.volatility_annual,
                    'Sharpe': metrics.sharpe_ratio,
                    'Weights': metrics.weights
                })
            except:
                pass

        return pd.DataFrame(results)

    def analyze_risk_contribution(self, weights: np.ndarray) -> RiskContribution:
        """Analyze risk contribution of each asset."""
        returns = self.returns_data
        cov_matrix = returns.cov() * 252

        # Portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std

        # Risk contribution
        risk_contrib = weights * marginal_contrib

        # Percentage risk contribution
        pct_risk_contrib = risk_contrib / portfolio_std

        # Create dictionaries
        marginal_dict = dict(zip(self.symbols, marginal_contrib))
        risk_contrib_dict = dict(zip(self.symbols, pct_risk_contrib))

        # Risk parity weights
        risk_parity_metrics = self.optimize_risk_parity()

        return RiskContribution(
            total_risk=portfolio_std,
            marginal_risk=marginal_dict,
            risk_contribution=risk_contrib_dict,
            risk_parity_weights=risk_parity_metrics.weights
        )

    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        returns = self.returns_data

        # Portfolio returns
        portfolio_returns = returns @ weights

        # Annual return
        returns_annual = portfolio_returns.mean() * 252

        # Annual volatility
        volatility_annual = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = (returns_annual - self.risk_free_rate) / volatility_annual

        # Sortino ratio (only downside volatility)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (returns_annual - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Value at Risk (95%)
        var_95 = np.percentile(portfolio_returns, 5)

        # Conditional VaR (CVaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        # Calmar ratio
        calmar_ratio = returns_annual / abs(max_drawdown) if max_drawdown != 0 else 0

        # Diversification ratio
        # (weighted sum of individual volatilities) / (portfolio volatility)
        individual_vols = returns.std() * np.sqrt(252)
        weighted_vol_sum = np.sum(weights * individual_vols)
        diversification_ratio = weighted_vol_sum / volatility_annual

        # Weights dictionary
        weights_dict = dict(zip(self.symbols, weights))

        return PortfolioMetrics(
            returns_annual=float(returns_annual),
            volatility_annual=float(volatility_annual),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            max_drawdown=float(max_drawdown),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            calmar_ratio=float(calmar_ratio),
            weights=weights_dict,
            diversification_ratio=float(diversification_ratio)
        )

    def compare_strategies(self) -> pd.DataFrame:
        """Compare different optimization strategies."""
        strategies = {
            'Max Sharpe': self.optimize_max_sharpe(),
            'Min Variance': self.optimize_min_variance(),
            'Risk Parity': self.optimize_risk_parity(),
            'Min CVaR': self.optimize_cvar()
        }

        results = []

        for name, metrics in strategies.items():
            results.append({
                'Strategy': name,
                'Return': f"{metrics.returns_annual*100:.2f}%",
                'Volatility': f"{metrics.volatility_annual*100:.2f}%",
                'Sharpe': f"{metrics.sharpe_ratio:.2f}",
                'Sortino': f"{metrics.sortino_ratio:.2f}",
                'Max DD': f"{metrics.max_drawdown*100:.2f}%",
                'Calmar': f"{metrics.calmar_ratio:.2f}",
                'Diversification': f"{metrics.diversification_ratio:.2f}"
            })

        return pd.DataFrame(results)

    def get_optimal_allocation(
        self,
        strategy: str = 'max_sharpe'
    ) -> Dict[str, float]:
        """
        Get optimal allocation for a strategy.

        Args:
            strategy: 'max_sharpe', 'min_variance', 'risk_parity', 'min_cvar'

        Returns:
            Dictionary of symbol -> weight
        """
        strategy_map = {
            'max_sharpe': self.optimize_max_sharpe,
            'min_variance': self.optimize_min_variance,
            'risk_parity': self.optimize_risk_parity,
            'min_cvar': self.optimize_cvar
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        metrics = strategy_map[strategy]()
        return metrics.weights

    def rebalance_recommendation(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float = 100000
    ) -> pd.DataFrame:
        """
        Generate rebalancing recommendations.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights from optimization
            portfolio_value: Total portfolio value

        Returns:
            DataFrame with buy/sell recommendations
        """
        recommendations = []

        for symbol in self.symbols:
            current_w = current_weights.get(symbol, 0)
            target_w = target_weights.get(symbol, 0)

            diff = target_w - current_w
            value_change = diff * portfolio_value

            action = "BUY" if diff > 0 else "SELL" if diff < 0 else "HOLD"

            recommendations.append({
                'Symbol': symbol,
                'Current %': f"{current_w*100:.2f}%",
                'Target %': f"{target_w*100:.2f}%",
                'Difference': f"{diff*100:+.2f}%",
                'Action': action,
                'Value Change': f"${abs(value_change):,.2f}"
            })

        return pd.DataFrame(recommendations)
