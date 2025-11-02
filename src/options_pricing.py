"""
Institutional options pricing: Black-Scholes model and Greeks.
Delta, Gamma, Theta, Vega, Rho calculation for derivatives trading.
Used by Citadel Securities, Jane Street, Optiver.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptionPrice:
    call_price: float
    put_price: float
    delta_call: float
    delta_put: float
    gamma: float
    theta_call: float
    theta_put: float
    vega: float
    rho_call: float
    rho_put: float
    implied_volatility: Optional[float]


class OptionsCalculator:
    """Institutional options pricing calculator"""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def black_scholes(self,
                     spot_price: float,
                     strike_price: float,
                     time_to_expiry: float,
                     volatility: float,
                     dividend_yield: float = 0.0) -> OptionPrice:
        """Calculate Black-Scholes option prices and Greeks"""
        S = spot_price
        K = strike_price
        T = time_to_expiry
        sigma = volatility
        r = self.risk_free_rate
        q = dividend_yield

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        delta_call = np.exp(-q * T) * norm.cdf(d1)
        delta_put = -np.exp(-q * T) * norm.cdf(-d1)

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

        theta_call = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365

        theta_put = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

        rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return OptionPrice(
            call_price=call_price,
            put_price=put_price,
            delta_call=delta_call,
            delta_put=delta_put,
            gamma=gamma,
            theta_call=theta_call,
            theta_put=theta_put,
            vega=vega,
            rho_call=rho_call,
            rho_put=rho_put,
            implied_volatility=None
        )

    def implied_volatility(self,
                          option_price: float,
                          spot_price: float,
                          strike_price: float,
                          time_to_expiry: float,
                          option_type: str = 'call',
                          dividend_yield: float = 0.0) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson"""
        sigma = 0.3
        max_iterations = 100
        tolerance = 1e-5

        for _ in range(max_iterations):
            result = self.black_scholes(spot_price, strike_price, time_to_expiry,
                                       sigma, dividend_yield)

            if option_type == 'call':
                price_diff = result.call_price - option_price
                vega = result.vega * 100
            else:
                price_diff = result.put_price - option_price
                vega = result.vega * 100

            if abs(price_diff) < tolerance:
                return sigma

            if vega == 0:
                return None

            sigma = sigma - price_diff / vega

            if sigma <= 0:
                return None

        return sigma if sigma > 0 else None

    def option_strategy_payoff(self,
                               strategy: str,
                               spot_price: float,
                               strikes: List[float],
                               premiums: List[float]) -> Dict[str, Any]:
        """Calculate payoff for option strategies"""
        if strategy == 'covered_call':
            return self._covered_call_payoff(spot_price, strikes, premiums)
        elif strategy == 'protective_put':
            return self._protective_put_payoff(spot_price, strikes, premiums)
        elif strategy == 'bull_call_spread':
            return self._bull_call_spread_payoff(spot_price, strikes, premiums)
        elif strategy == 'iron_condor':
            return self._iron_condor_payoff(spot_price, strikes, premiums)
        else:
            return {}

    def _covered_call_payoff(self, spot: float, strikes: List[float],
                            premiums: List[float]) -> Dict[str, Any]:
        """Covered call strategy payoff"""
        strike = strikes[0]
        premium = premiums[0]

        prices = np.linspace(spot * 0.5, spot * 1.5, 100)
        stock_payoff = prices - spot
        call_payoff = np.maximum(prices - strike, 0) * -1 + premium
        total_payoff = stock_payoff + call_payoff

        max_profit = strike - spot + premium
        max_loss = spot - premium
        breakeven = spot - premium

        return {
            'prices': prices,
            'payoff': total_payoff,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven
        }

    def _protective_put_payoff(self, spot: float, strikes: List[float],
                               premiums: List[float]) -> Dict[str, Any]:
        """Protective put strategy payoff"""
        strike = strikes[0]
        premium = premiums[0]

        prices = np.linspace(spot * 0.5, spot * 1.5, 100)
        stock_payoff = prices - spot
        put_payoff = np.maximum(strike - prices, 0) - premium
        total_payoff = stock_payoff + put_payoff

        max_loss = spot - strike + premium
        breakeven = spot + premium

        return {
            'prices': prices,
            'payoff': total_payoff,
            'max_loss': max_loss,
            'breakeven': breakeven
        }

    def _bull_call_spread_payoff(self, spot: float, strikes: List[float],
                                 premiums: List[float]) -> Dict[str, Any]:
        """Bull call spread strategy payoff"""
        lower_strike = strikes[0]
        upper_strike = strikes[1]
        lower_premium = premiums[0]
        upper_premium = premiums[1]

        net_premium = lower_premium - upper_premium

        prices = np.linspace(spot * 0.5, spot * 1.5, 100)
        long_call = np.maximum(prices - lower_strike, 0) - lower_premium
        short_call = np.maximum(prices - upper_strike, 0) * -1 + upper_premium
        total_payoff = long_call + short_call

        max_profit = upper_strike - lower_strike - net_premium
        max_loss = net_premium
        breakeven = lower_strike + net_premium

        return {
            'prices': prices,
            'payoff': total_payoff,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven
        }

    def _iron_condor_payoff(self, spot: float, strikes: List[float],
                           premiums: List[float]) -> Dict[str, Any]:
        """Iron condor strategy payoff"""
        if len(strikes) < 4:
            return {}

        put_low_strike = strikes[0]
        put_high_strike = strikes[1]
        call_low_strike = strikes[2]
        call_high_strike = strikes[3]

        net_premium = sum(premiums[1:3]) - sum([premiums[0], premiums[3]])

        prices = np.linspace(spot * 0.7, spot * 1.3, 100)
        long_put_low = np.maximum(put_low_strike - prices, 0) - premiums[0]
        short_put_high = np.maximum(put_high_strike - prices, 0) * -1 + premiums[1]
        short_call_low = np.maximum(prices - call_low_strike, 0) * -1 + premiums[2]
        long_call_high = np.maximum(prices - call_high_strike, 0) - premiums[3]

        total_payoff = long_put_low + short_put_high + short_call_low + long_call_high

        max_profit = net_premium
        max_loss = (put_high_strike - put_low_strike) - net_premium

        return {
            'prices': prices,
            'payoff': total_payoff,
            'max_profit': max_profit,
            'max_loss': max_loss
        }


def calculate_option_price(spot: float,
                          strike: float,
                          time_to_expiry: float,
                          volatility: float) -> Dict[str, float]:
    """Quick interface for option pricing"""
    calculator = OptionsCalculator()
    result = calculator.black_scholes(spot, strike, time_to_expiry, volatility)

    return {
        'call_price': result.call_price,
        'put_price': result.put_price,
        'delta_call': result.delta_call,
        'delta_put': result.delta_put,
        'gamma': result.gamma,
        'vega': result.vega
    }
