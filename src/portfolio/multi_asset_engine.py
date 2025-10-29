"""
Multi-Asset Portfolio Management Engine

Handles:
- Portfolio construction across multiple assets
- Correlation-based risk management
- Dynamic allocation
- Rebalancing logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PortfolioState:
    """Current portfolio state."""
    positions: Dict[str, float]  # ticker -> shares
    cash: float
    total_value: float
    weights: Dict[str, float]  # ticker -> weight
    date: pd.Timestamp


class MultiAssetPortfolio:
    """
    Multi-asset portfolio management with correlation-aware risk control.

    Key features:
    - Dynamic allocation across assets
    - Correlation matrix monitoring
    - Risk parity / mean-variance optimization
    - Rebalancing with transaction cost awareness
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        allocation_method: str = 'equal_weight',  # 'equal_weight', 'risk_parity', 'mean_variance'
        rebalance_frequency: int = 21,  # Days between rebalances
        correlation_lookback: int = 60,  # Days for correlation calculation
        max_asset_weight: float = 0.3,  # Max 30% per asset
        commission: float = 0.001
    ):
        """
        Args:
            initial_capital: Starting capital
            allocation_method: How to allocate capital
            rebalance_frequency: How often to rebalance (days)
            correlation_lookback: Window for correlation calculation
            max_asset_weight: Maximum weight per asset
            commission: Commission per trade
        """
        self.initial_capital = initial_capital
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency
        self.correlation_lookback = correlation_lookback
        self.max_asset_weight = max_asset_weight
        self.commission = commission

        # State
        self.positions: Dict[str, float] = {}
        self.cash = initial_capital
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[pd.Timestamp] = []
        self.portfolio_history: List[PortfolioState] = []

    def backtest(
        self,
        prices: Dict[str, pd.Series],  # ticker -> price series
        signals: Dict[str, pd.Series]  # ticker -> signal series
    ) -> Dict:
        """
        Run multi-asset backtest.

        Args:
            prices: Dict of ticker -> price series
            signals: Dict of ticker -> signal series (1=long, -1=short, 0=neutral)

        Returns:
            Backtest results
        """
        # Align all series to common index
        all_data = self._align_data(prices, signals)
        dates = all_data.index

        print(f"\n📊 Multi-Asset Portfolio Backtest")
        print(f"   Assets: {list(prices.keys())}")
        print(f"   Period: {dates[0]} to {dates[-1]}")
        print(f"   Allocation: {self.allocation_method}")

        days_since_rebalance = 0

        for i, date in enumerate(dates):
            current_prices = {ticker: all_data.loc[date, f'{ticker}_price']
                            for ticker in prices.keys()}
            current_signals = {ticker: all_data.loc[date, f'{ticker}_signal']
                             for ticker in signals.keys()}

            # Calculate portfolio value
            portfolio_value = self.cash
            for ticker, shares in self.positions.items():
                if ticker in current_prices:
                    portfolio_value += shares * current_prices[ticker]

            # Rebalance if needed
            if days_since_rebalance >= self.rebalance_frequency or i == 0:
                # Calculate target weights
                target_weights = self._calculate_target_weights(
                    prices, signals, all_data, date, current_signals
                )

                # Execute rebalancing
                self._rebalance(current_prices, target_weights, portfolio_value)
                days_since_rebalance = 0
            else:
                days_since_rebalance += 1

            # Record state
            final_value = self.cash + sum(
                self.positions.get(ticker, 0) * current_prices[ticker]
                for ticker in current_prices
            )
            self.equity_curve.append(final_value)
            self.dates.append(date)

            weights = {
                ticker: (self.positions.get(ticker, 0) * current_prices[ticker]) / final_value
                for ticker in current_prices
            }

            self.portfolio_history.append(PortfolioState(
                positions=self.positions.copy(),
                cash=self.cash,
                total_value=final_value,
                weights=weights,
                date=date
            ))

        # Calculate metrics
        results = self._calculate_portfolio_metrics(all_data, prices)
        return results

    def _align_data(
        self,
        prices: Dict[str, pd.Series],
        signals: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Align all data to common index."""
        df = pd.DataFrame()

        for ticker in prices.keys():
            df[f'{ticker}_price'] = prices[ticker]
            df[f'{ticker}_signal'] = signals.get(ticker, pd.Series(0, index=prices[ticker].index))

        return df.dropna()

    def _calculate_target_weights(
        self,
        prices: Dict[str, pd.Series],
        signals: Dict[str, pd.Series],
        all_data: pd.DataFrame,
        current_date: pd.Timestamp,
        current_signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate target portfolio weights."""
        if self.allocation_method == 'equal_weight':
            return self._equal_weight_allocation(current_signals)

        elif self.allocation_method == 'risk_parity':
            # Get historical returns for correlation
            returns_dict = {}
            date_idx = all_data.index.get_loc(current_date)
            lookback_start = max(0, date_idx - self.correlation_lookback)

            for ticker in prices.keys():
                hist_prices = all_data.iloc[lookback_start:date_idx][f'{ticker}_price']
                returns_dict[ticker] = hist_prices.pct_change().dropna()

            return self._risk_parity_allocation(returns_dict, current_signals)

        elif self.allocation_method == 'mean_variance':
            # Mean-variance optimization (simplified)
            returns_dict = {}
            date_idx = all_data.index.get_loc(current_date)
            lookback_start = max(0, date_idx - self.correlation_lookback)

            for ticker in prices.keys():
                hist_prices = all_data.iloc[lookback_start:date_idx][f'{ticker}_price']
                returns_dict[ticker] = hist_prices.pct_change().dropna()

            return self._mean_variance_allocation(returns_dict, current_signals)

        else:
            return self._equal_weight_allocation(current_signals)

    def _equal_weight_allocation(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Simple equal weight allocation."""
        # Only allocate to assets with positive signal
        active_assets = [ticker for ticker, signal in signals.items() if signal > 0]

        if not active_assets:
            return {ticker: 0.0 for ticker in signals.keys()}

        weight_per_asset = min(1.0 / len(active_assets), self.max_asset_weight)

        weights = {}
        for ticker in signals.keys():
            if ticker in active_assets:
                weights[ticker] = weight_per_asset
            else:
                weights[ticker] = 0.0

        # Normalize to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _risk_parity_allocation(
        self,
        returns: Dict[str, pd.Series],
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Risk parity allocation - each asset contributes equal risk."""
        # Only consider assets with positive signals
        active_assets = [ticker for ticker, signal in signals.items() if signal > 0]

        if not active_assets:
            return {ticker: 0.0 for ticker in signals.keys()}

        # Calculate volatilities
        vols = {}
        for ticker in active_assets:
            if len(returns[ticker]) > 10:
                vols[ticker] = returns[ticker].std() * np.sqrt(252)
            else:
                vols[ticker] = 0.01  # Default vol

        # Risk parity: weight inversely to volatility
        inv_vols = {ticker: 1.0 / vol if vol > 0 else 0 for ticker, vol in vols.items()}
        total_inv_vol = sum(inv_vols.values())

        weights = {}
        for ticker in signals.keys():
            if ticker in active_assets and total_inv_vol > 0:
                weights[ticker] = min(inv_vols[ticker] / total_inv_vol, self.max_asset_weight)
            else:
                weights[ticker] = 0.0

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _mean_variance_allocation(
        self,
        returns: Dict[str, pd.Series],
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Mean-variance optimization (simplified maximum Sharpe)."""
        active_assets = [ticker for ticker, signal in signals.items() if signal > 0]

        if not active_assets or len(active_assets) < 2:
            return self._equal_weight_allocation(signals)

        # Calculate expected returns and covariance
        mean_returns = {}
        for ticker in active_assets:
            if len(returns[ticker]) > 10:
                mean_returns[ticker] = returns[ticker].mean() * 252
            else:
                mean_returns[ticker] = 0.0

        # Build covariance matrix
        returns_df = pd.DataFrame({ticker: returns[ticker] for ticker in active_assets})
        returns_df = returns_df.dropna()

        if len(returns_df) < 10:
            return self._equal_weight_allocation(signals)

        cov_matrix = returns_df.cov() * 252

        # Simple optimization: weight by Sharpe ratio
        sharpes = {}
        for ticker in active_assets:
            vol = np.sqrt(cov_matrix.loc[ticker, ticker])
            if vol > 0:
                sharpes[ticker] = mean_returns[ticker] / vol
            else:
                sharpes[ticker] = 0

        # Weight by positive Sharpe ratios
        positive_sharpes = {k: max(v, 0) for k, v in sharpes.items()}
        total_sharpe = sum(positive_sharpes.values())

        weights = {}
        for ticker in signals.keys():
            if ticker in active_assets and total_sharpe > 0:
                weights[ticker] = min(positive_sharpes[ticker] / total_sharpe, self.max_asset_weight)
            else:
                weights[ticker] = 0.0

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _rebalance(
        self,
        prices: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float
    ):
        """Execute rebalancing trades."""
        # Calculate target positions in shares
        target_positions = {}
        for ticker, weight in target_weights.items():
            if ticker in prices and prices[ticker] > 0:
                target_value = portfolio_value * weight
                target_positions[ticker] = target_value / prices[ticker]
            else:
                target_positions[ticker] = 0

        # Execute trades
        for ticker in set(list(self.positions.keys()) + list(target_positions.keys())):
            current_pos = self.positions.get(ticker, 0)
            target_pos = target_positions.get(ticker, 0)
            shares_to_trade = target_pos - current_pos

            if abs(shares_to_trade) > 0.01 and ticker in prices:
                trade_value = abs(shares_to_trade) * prices[ticker]
                cost = trade_value * self.commission

                # Update cash and position
                self.cash -= shares_to_trade * prices[ticker] + cost
                self.positions[ticker] = target_pos

    def _calculate_portfolio_metrics(
        self,
        all_data: pd.DataFrame,
        prices: Dict[str, pd.Series]
    ) -> Dict:
        """Calculate portfolio performance metrics."""
        equity_series = pd.Series(self.equity_curve, index=[self.dates[0]] + self.dates)
        returns = equity_series.pct_change().dropna()

        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        annual_return = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(returns)) - 1) * 100

        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_series,
            'returns': returns,
            'portfolio_history': self.portfolio_history,
            'final_weights': self.portfolio_history[-1].weights if self.portfolio_history else {}
        }
