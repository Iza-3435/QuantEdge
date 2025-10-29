"""Backtesting framework for strategy evaluation."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from src.retrieval.advanced_pattern_retrieval import AdvancedPatternRetrieval
from src.core.logging import app_logger


@dataclass
class Trade:
    """Individual trade record."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    return_pct: float
    confidence: float
    prediction: float


@dataclass
class BacktestResults:
    """Backtest performance metrics."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    calmar_ratio: float
    sortino_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series


class Backtester:
    """
    Backtest trading strategies using historical pattern predictions.

    Implements:
    - Walk-forward validation
    - Transaction costs
    - Position sizing
    - Risk management
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,  # 0.1%
        confidence_threshold: float = 0.5,
        max_position_size: float = 0.2,  # 20% of capital
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.confidence_threshold = confidence_threshold
        self.max_position_size = max_position_size

    def run_backtest(
        self,
        df: pd.DataFrame,
        retriever: AdvancedPatternRetrieval,
        start_idx: int = 100,
        holding_period: int = 5,
        rebalance_frequency: int = 1
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            df: Market data with OHLCV
            retriever: Trained pattern retrieval model
            start_idx: Index to start backtest
            holding_period: Days to hold position
            rebalance_frequency: How often to rebalance

        Returns:
            BacktestResults with performance metrics
        """
        app_logger.info(
            f"Starting backtest from {df.index[start_idx]} to {df.index[-1]}"
        )

        capital = self.initial_capital
        position = 0  # Current position (shares)
        position_value = 0
        trades: List[Trade] = []
        equity_curve = []
        dates = []

        for idx in range(start_idx, len(df) - holding_period, rebalance_frequency):
            current_date = df.index[idx]
            current_price = df['Close'].iloc[idx]

            # Extract features and make prediction
            features = retriever.extract_pattern_features(df, idx)
            if features is None:
                equity_curve.append(capital + position_value)
                dates.append(current_date)
                continue

            prediction = retriever.predict_with_uncertainty(
                features,
                confidence_level=0.9
            )

            # Calculate portfolio value
            position_value = position * current_price
            total_value = capital + position_value

            # Trading decision
            if prediction.confidence >= self.confidence_threshold:
                predicted_return = prediction.predicted_return

                # Determine position size based on Kelly criterion (simplified)
                if predicted_return > 0:
                    # Long signal
                    kelly_fraction = min(
                        abs(predicted_return) * prediction.confidence,
                        self.max_position_size
                    )

                    target_position_value = total_value * kelly_fraction
                    target_shares = target_position_value / current_price

                    # Rebalance position
                    shares_to_trade = float(target_shares - position)

                    if abs(shares_to_trade) > 0.01:  # Minimum trade size
                        # Execute trade
                        trade_value = shares_to_trade * current_price
                        cost = abs(trade_value) * self.transaction_cost

                        capital -= trade_value + cost
                        position += shares_to_trade

                        app_logger.debug(
                            f"{current_date}: Traded {float(shares_to_trade):.2f} shares "
                            f"at ${float(current_price):.2f}, confidence={prediction.confidence:.3f}"
                        )

                elif predicted_return < -0.01 and position > 0:
                    # Negative signal - exit position
                    exit_value = position * current_price
                    cost = exit_value * self.transaction_cost

                    capital += exit_value - cost

                    if position != 0:
                        # Record trade
                        # (Entry tracking would need more state management)
                        pass

                    position = 0

            # Check if holding period ended, close position
            if idx % (holding_period * rebalance_frequency) == 0 and position != 0:
                exit_price = df['Close'].iloc[idx + holding_period - 1]
                exit_value = position * exit_price
                cost = exit_value * self.transaction_cost

                # Calculate return
                entry_value = position * current_price
                trade_return = (exit_value - entry_value - cost) / entry_value

                trades.append(Trade(
                    entry_date=current_date,
                    exit_date=df.index[idx + holding_period - 1],
                    entry_price=current_price,
                    exit_price=exit_price,
                    position_size=position,
                    return_pct=trade_return * 100,
                    confidence=prediction.confidence,
                    prediction=prediction.predicted_return
                ))

                capital += exit_value - cost
                position = 0

            # Record equity
            position_value = position * current_price
            equity_curve.append(capital + position_value)
            dates.append(current_date)

        # Close final position
        if position != 0:
            final_price = df['Close'].iloc[-1]
            capital += position * final_price * (1 - self.transaction_cost)
            position = 0

        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=dates)
        results = self._calculate_metrics(equity_series, trades)

        app_logger.info(
            f"Backtest complete: {results.total_trades} trades, "
            f"{results.total_return:.2f}% return, "
            f"Sharpe={results.sharpe_ratio:.2f}"
        )

        return results

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> BacktestResults:
        """Calculate performance metrics."""

        # Total return
        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100

        # Annual return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = ((equity_curve.iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100

        # Returns
        returns = equity_curve.pct_change().dropna()

        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (returns.mean() / negative_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Calmar ratio
        if max_drawdown != 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Trade statistics
        if trades:
            trade_returns = [t.return_pct for t in trades]
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trades) * 100
            avg_trade_return = np.mean(trade_returns)
            best_trade = max(trade_returns)
            worst_trade = min(trade_returns)
        else:
            win_rate = 0.0
            avg_trade_return = 0.0
            best_trade = 0.0
            worst_trade = 0.0

        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            trades=trades,
            equity_curve=equity_curve
        )

    def save_results(self, results: BacktestResults, path: str):
        """Save backtest results to file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'summary': {
                'total_return': results.total_return,
                'annual_return': results.annual_return,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown': results.max_drawdown,
                'calmar_ratio': results.calmar_ratio,
                'win_rate': results.win_rate,
                'total_trades': results.total_trades,
                'avg_trade_return': results.avg_trade_return,
                'best_trade': results.best_trade,
                'worst_trade': results.worst_trade
            },
            'trades': [
                {
                    'entry_date': t.entry_date.isoformat(),
                    'exit_date': t.exit_date.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'return_pct': t.return_pct,
                    'confidence': t.confidence,
                    'prediction': t.prediction
                }
                for t in results.trades
            ]
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save equity curve
        equity_path = save_path.with_suffix('.csv')
        results.equity_curve.to_csv(equity_path)

        app_logger.info(f"Saved backtest results to {path}")

    def print_summary(self, results: BacktestResults):
        """Print backtest summary."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"Total Return:        {results.total_return:>10.2f}%")
        print(f"Annual Return:       {results.annual_return:>10.2f}%")
        print(f"Sharpe Ratio:        {results.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:       {results.sortino_ratio:>10.2f}")
        print(f"Max Drawdown:        {results.max_drawdown:>10.2f}%")
        print(f"Calmar Ratio:        {results.calmar_ratio:>10.2f}")
        print(f"\nTotal Trades:        {results.total_trades:>10}")
        print(f"Win Rate:            {results.win_rate:>10.2f}%")
        print(f"Avg Trade Return:    {results.avg_trade_return:>10.2f}%")
        print(f"Best Trade:          {results.best_trade:>10.2f}%")
        print(f"Worst Trade:         {results.worst_trade:>10.2f}%")
        print("="*70)
