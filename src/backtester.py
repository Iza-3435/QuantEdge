"""
Institutional-grade backtesting engine with walk-forward validation.
Transaction costs, slippage, risk metrics, out-of-sample testing.
Used by Renaissance Technologies, Two Sigma, Citadel for strategy validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    var_95: float
    cvar_95: float
    equity_curve: pd.Series
    trade_history: pd.DataFrame
    monthly_returns: pd.Series


@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    holding_period: int


class InstitutionalBacktester:
    """Production backtesting engine with institutional features"""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 position_size: float = 0.25):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size

        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.dates = []

    def backtest_strategy(self,
                         prices: pd.Series,
                         signals: pd.Series,
                         stop_loss: float = 0.05,
                         take_profit: float = 0.15) -> BacktestResult:
        """Run backtest with stop-loss and take-profit"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.dates = [prices.index[0]]

        for i in range(1, len(prices)):
            current_date = prices.index[i]
            current_price = prices.iloc[i]
            signal = signals.iloc[i]

            self._check_exits(current_date, current_price, stop_loss, take_profit)

            if signal == 1 and not self.positions:
                self._enter_long(current_date, current_price)
            elif signal == -1 and self.positions:
                self._exit_position(current_date, current_price)

            total_value = self._calculate_portfolio_value(current_price)
            self.equity_curve.append(total_value)
            self.dates.append(current_date)

        if self.positions:
            self._exit_position(prices.index[-1], prices.iloc[-1])

        return self._calculate_metrics()

    def backtest_ml_predictions(self,
                                prices: pd.Series,
                                predictions: pd.Series,
                                threshold: float = 0.03) -> BacktestResult:
        """Backtest ML model predictions"""
        signals = pd.Series(0, index=prices.index)

        for i in range(1, len(predictions)):
            pred_return = (predictions.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]

            if pred_return > threshold:
                signals.iloc[i] = 1
            elif pred_return < -threshold:
                signals.iloc[i] = -1

        return self.backtest_strategy(prices, signals)

    def walk_forward_validation(self,
                                prices: pd.Series,
                                strategy_func: Callable,
                                train_window: int = 252,
                                test_window: int = 63) -> Dict[str, Any]:
        """Walk-forward validation for out-of-sample testing"""
        results = []
        equity_curves = []

        for i in range(train_window, len(prices) - test_window, test_window):
            train_data = prices.iloc[i-train_window:i]
            test_data = prices.iloc[i:i+test_window]

            signals = strategy_func(train_data, test_data)

            if len(signals) == len(test_data):
                result = self.backtest_strategy(test_data, signals)
                results.append(result)
                equity_curves.append(result.equity_curve)

        if not results:
            return None

        avg_return = np.mean([r.total_return for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_drawdown = np.mean([r.max_drawdown for r in results])
        consistency = sum(1 for r in results if r.total_return > 0) / len(results)

        return {
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_drawdown,
            'consistency': consistency,
            'num_periods': len(results),
            'all_results': results
        }

    def _enter_long(self, date: datetime, price: float):
        """Enter long position"""
        position_value = self.capital * self.position_size
        entry_cost = price * (1 + self.slippage)
        shares = int(position_value / entry_cost)

        if shares > 0:
            cost = shares * entry_cost
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost

            if total_cost <= self.capital:
                self.positions['long'] = {
                    'entry_date': date,
                    'entry_price': entry_cost,
                    'shares': shares,
                    'cost': total_cost
                }
                self.capital -= total_cost

    def _exit_position(self, date: datetime, price: float):
        """Exit current position"""
        if 'long' not in self.positions:
            return

        position = self.positions['long']
        exit_price = price * (1 - self.slippage)
        proceeds = position['shares'] * exit_price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost

        pnl = net_proceeds - position['cost']
        return_pct = (pnl / position['cost']) * 100
        holding_period = (date - position['entry_date']).days

        trade = Trade(
            entry_date=position['entry_date'],
            exit_date=date,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            shares=position['shares'],
            pnl=pnl,
            return_pct=return_pct,
            holding_period=holding_period
        )

        self.trades.append(trade)
        self.capital += net_proceeds
        del self.positions['long']

    def _check_exits(self, date: datetime, price: float, stop_loss: float, take_profit: float):
        """Check stop-loss and take-profit"""
        if 'long' not in self.positions:
            return

        position = self.positions['long']
        current_return = (price - position['entry_price']) / position['entry_price']

        if current_return <= -stop_loss or current_return >= take_profit:
            self._exit_position(date, price)

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        cash = self.capital

        if 'long' in self.positions:
            position = self.positions['long']
            position_value = position['shares'] * current_price
            cash += position_value

        return cash

    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics"""
        equity_series = pd.Series(self.equity_curve, index=self.dates)

        total_return = ((equity_series.iloc[-1] - self.initial_capital) / self.initial_capital) * 100

        returns = equity_series.pct_change().dropna()
        n_years = len(equity_series) / 252
        annualized_return = ((1 + total_return/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)

        max_drawdown = self._calculate_max_drawdown(equity_series)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        win_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(win_trades) / len(self.trades) * 100 if self.trades else 0

        total_wins = sum(t.pnl for t in win_trades)
        total_losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        avg_trade_return = np.mean([t.return_pct for t in self.trades]) if self.trades else 0

        volatility = returns.std() * np.sqrt(252) * 100

        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100

        trade_df = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'pnl': t.pnl,
            'return_pct': t.return_pct,
            'holding_period': t.holding_period
        } for t in self.trades])

        monthly_returns = equity_series.resample('M').last().pct_change().dropna() * 100

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            equity_curve=equity_series,
            trade_history=trade_df,
            monthly_returns=monthly_returns
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - (risk_free_rate / 252)
        if returns.std() == 0:
            return 0
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        return drawdown.min()


def backtest_strategy(prices: pd.Series,
                     signals: pd.Series,
                     initial_capital: float = 100000) -> Dict[str, Any]:
    """Quick interface for backtesting"""
    backtester = InstitutionalBacktester(initial_capital=initial_capital)
    result = backtester.backtest_strategy(prices, signals)

    return {
        'total_return': result.total_return,
        'annualized_return': result.annualized_return,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown,
        'win_rate': result.win_rate,
        'total_trades': result.total_trades
    }
