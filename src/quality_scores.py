"""
QUALITY SCORES MODULE
Fundamental quality assessment scores (Piotroski, Altman Z-Score)
"""

from typing import Dict, Any, Optional


def calculate_piotroski_score(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate Piotroski F-Score (0-9 point quality score).

    The F-Score measures financial strength across 3 categories:
    - Profitability (4 points)
    - Leverage/Liquidity (3 points)
    - Operating Efficiency (2 points)

    Args:
        fundamentals: Dictionary containing financial metrics

    Returns:
        Dictionary with score and breakdown
    """
    score = 0
    breakdown = {}

    # PROFITABILITY (4 points)

    # 1. Positive ROA
    roa = fundamentals.get('returnOnAssets', 0)
    if roa > 0:
        score += 1
        breakdown['positive_roa'] = True
    else:
        breakdown['positive_roa'] = False

    # 2. Positive Operating Cash Flow
    operating_cf = fundamentals.get('operatingCashflow', 0)
    if operating_cf > 0:
        score += 1
        breakdown['positive_ocf'] = True
    else:
        breakdown['positive_ocf'] = False

    # 3. Increasing ROA (year-over-year)
    # Note: We'd need historical data for this, using current as proxy
    if roa > 0.05:  # ROA > 5% as proxy for improvement
        score += 1
        breakdown['improving_roa'] = True
    else:
        breakdown['improving_roa'] = False

    # 4. Quality of Earnings (Cash Flow > Net Income)
    net_income = fundamentals.get('netIncome', 0)
    if operating_cf > net_income and net_income > 0:
        score += 1
        breakdown['quality_earnings'] = True
    else:
        breakdown['quality_earnings'] = False

    # LEVERAGE/LIQUIDITY/SOURCE OF FUNDS (3 points)

    # 5. Decreasing Long-term Debt
    # Using debt/equity as proxy - lower is better
    debt_equity = fundamentals.get('debtToEquity', 100)
    if debt_equity < 50:  # Less than 0.5 D/E ratio
        score += 1
        breakdown['low_debt'] = True
    else:
        breakdown['low_debt'] = False

    # 6. Increasing Current Ratio
    current_ratio = fundamentals.get('currentRatio', 0)
    if current_ratio > 1.5:
        score += 1
        breakdown['strong_liquidity'] = True
    else:
        breakdown['strong_liquidity'] = False

    # 7. No New Shares Issued
    # Using shares outstanding - if not increasing
    # This requires historical data, using market cap growth as proxy
    market_cap = fundamentals.get('marketCap', 0)
    if market_cap > 0:  # Has positive market cap
        score += 1
        breakdown['no_dilution'] = True
    else:
        breakdown['no_dilution'] = False

    # OPERATING EFFICIENCY (2 points)

    # 8. Increasing Gross Margin
    gross_margin = fundamentals.get('grossMargins', 0)
    if gross_margin > 0.3:  # Gross margin > 30%
        score += 1
        breakdown['strong_margins'] = True
    else:
        breakdown['strong_margins'] = False

    # 9. Increasing Asset Turnover
    # Revenue / Total Assets
    revenue = fundamentals.get('totalRevenue', 0)
    total_assets = fundamentals.get('totalAssets', 1)
    asset_turnover = revenue / total_assets if total_assets > 0 else 0
    if asset_turnover > 0.5:
        score += 1
        breakdown['efficient_assets'] = True
    else:
        breakdown['efficient_assets'] = False

    # Interpretation
    if score >= 7:
        rating = "EXCELLENT"
        color = "green"
    elif score >= 5:
        rating = "GOOD"
        color = "yellow"
    elif score >= 3:
        rating = "AVERAGE"
        color = "white"
    else:
        rating = "POOR"
        color = "red"

    return {
        'score': score,
        'max_score': 9,
        'rating': rating,
        'color': color,
        'breakdown': breakdown
    }


def calculate_altman_z_score(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate Altman Z-Score (bankruptcy prediction).

    Z-Score formula (for public manufacturing companies):
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets

    Interpretation:
    > 2.99 = Safe Zone
    1.81 - 2.99 = Grey Zone
    < 1.81 = Distress Zone

    Args:
        fundamentals: Dictionary containing financial metrics

    Returns:
        Dictionary with Z-score and interpretation
    """
    try:
        # Get financial data
        total_assets = fundamentals.get('totalAssets', 0)
        current_assets = fundamentals.get('totalCurrentAssets', 0)
        current_liabilities = fundamentals.get('totalCurrentLiabilities', 0)
        retained_earnings = fundamentals.get('retainedEarnings', 0)
        ebit = fundamentals.get('ebit', fundamentals.get('operatingIncome', 0))
        market_cap = fundamentals.get('marketCap', 0)
        total_liabilities = fundamentals.get('totalLiab', 0)
        revenue = fundamentals.get('totalRevenue', 0)

        if total_assets == 0:
            return {
                'score': None,
                'rating': "N/A",
                'color': "bright_black",
                'components': {},
                'error': "Insufficient data"
            }

        # Calculate components
        working_capital = current_assets - current_liabilities
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
        x5 = revenue / total_assets

        # Calculate Z-Score
        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)

        # Interpretation
        if z_score > 2.99:
            rating = "SAFE"
            color = "green"
            risk = "Low bankruptcy risk"
        elif z_score > 1.81:
            rating = "GREY ZONE"
            color = "yellow"
            risk = "Moderate bankruptcy risk"
        else:
            rating = "DISTRESS"
            color = "red"
            risk = "High bankruptcy risk"

        return {
            'score': round(z_score, 2),
            'rating': rating,
            'color': color,
            'risk': risk,
            'components': {
                'working_capital_ratio': round(x1, 3),
                'retained_earnings_ratio': round(x2, 3),
                'ebit_ratio': round(x3, 3),
                'market_to_book': round(x4, 3),
                'asset_turnover': round(x5, 3)
            }
        }

    except Exception as e:
        return {
            'score': None,
            'rating': "ERROR",
            'color': "bright_black",
            'components': {},
            'error': str(e)
        }


def calculate_quality_composite_score(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overall quality composite score combining multiple metrics.

    Args:
        fundamentals: Dictionary containing financial metrics

    Returns:
        Dictionary with composite score (0-100) and ratings
    """
    composite_score = 0

    # Piotroski Score (40 points)
    piotroski = calculate_piotroski_score(fundamentals)
    piotroski_points = (piotroski['score'] / 9) * 40
    composite_score += piotroski_points

    # Altman Z-Score (30 points)
    altman = calculate_altman_z_score(fundamentals)
    if altman['score'] is not None:
        if altman['score'] > 2.99:
            altman_points = 30
        elif altman['score'] > 1.81:
            altman_points = 20
        else:
            altman_points = 10
        composite_score += altman_points
    else:
        composite_score += 15  # Neutral if no data

    # Profitability Metrics (30 points)
    profit_margin = fundamentals.get('profitMargins', 0) * 100
    roe = fundamentals.get('returnOnEquity', 0) * 100
    roa = fundamentals.get('returnOnAssets', 0) * 100

    # Profit Margin (10 points)
    if profit_margin > 20:
        composite_score += 10
    elif profit_margin > 10:
        composite_score += 7
    elif profit_margin > 5:
        composite_score += 4

    # ROE (10 points)
    if roe > 20:
        composite_score += 10
    elif roe > 15:
        composite_score += 7
    elif roe > 10:
        composite_score += 4

    # ROA (10 points)
    if roa > 10:
        composite_score += 10
    elif roa > 5:
        composite_score += 7
    elif roa > 2:
        composite_score += 4

    # Determine overall rating
    if composite_score >= 80:
        rating = "EXCEPTIONAL"
        color = "bright_green"
    elif composite_score >= 65:
        rating = "STRONG"
        color = "green"
    elif composite_score >= 50:
        rating = "GOOD"
        color = "yellow"
    elif composite_score >= 35:
        rating = "FAIR"
        color = "orange"
    else:
        rating = "WEAK"
        color = "red"

    return {
        'composite_score': round(composite_score, 1),
        'rating': rating,
        'color': color,
        'piotroski': piotroski,
        'altman': altman,
        'profitability': {
            'profit_margin': round(profit_margin, 1),
            'roe': round(roe, 1),
            'roa': round(roa, 1)
        }
    }


def get_quality_summary(fundamentals: Dict[str, Any]) -> str:
    """
    Get human-readable quality summary.

    Args:
        fundamentals: Dictionary containing financial metrics

    Returns:
        Formatted summary string
    """
    quality = calculate_quality_composite_score(fundamentals)

    summary = f"Quality Score: {quality['composite_score']}/100 ({quality['rating']})\n"
    summary += f"Piotroski F-Score: {quality['piotroski']['score']}/9 ({quality['piotroski']['rating']})\n"

    altman = quality['altman']
    if altman['score'] is not None:
        summary += f"Altman Z-Score: {altman['score']} ({altman['rating']})"
    else:
        summary += "Altman Z-Score: N/A"

    return summary
