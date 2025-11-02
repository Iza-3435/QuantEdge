"""
QUALITY SCORES MODULE
Fundamental quality assessment scores (Piotroski, Altman Z-Score)
"""

from typing import Dict, Any, Optional


def calculate_piotroski_score(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    breakdown = {}

    roa = fundamentals.get('returnOnAssets', 0)
    if roa > 0:
        score += 1
        breakdown['positive_roa'] = True
    else:
        breakdown['positive_roa'] = False

    operating_cf = fundamentals.get('operatingCashflow', 0)
    if operating_cf > 0:
        score += 1
        breakdown['positive_ocf'] = True
    else:
        breakdown['positive_ocf'] = False

    if roa > 0.05:
        score += 1
        breakdown['improving_roa'] = True
    else:
        breakdown['improving_roa'] = False

    net_income = fundamentals.get('netIncome', 0)
    if operating_cf > net_income and net_income > 0:
        score += 1
        breakdown['quality_earnings'] = True
    else:
        breakdown['quality_earnings'] = False

    debt_equity = fundamentals.get('debtToEquity', 100)
    if debt_equity < 50:
        score += 1
        breakdown['low_debt'] = True
    else:
        breakdown['low_debt'] = False

    current_ratio = fundamentals.get('currentRatio', 0)
    if current_ratio > 1.5:
        score += 1
        breakdown['strong_liquidity'] = True
    else:
        breakdown['strong_liquidity'] = False

    market_cap = fundamentals.get('marketCap', 0)
    if market_cap > 0:
        score += 1
        breakdown['no_dilution'] = True
    else:
        breakdown['no_dilution'] = False

    gross_margin = fundamentals.get('grossMargins', 0)
    if gross_margin > 0.3:
        score += 1
        breakdown['strong_margins'] = True
    else:
        breakdown['strong_margins'] = False

    revenue = fundamentals.get('totalRevenue', 0)
    total_assets = fundamentals.get('totalAssets', 1)
    asset_turnover = revenue / total_assets if total_assets > 0 else 0
    if asset_turnover > 0.5:
        score += 1
        breakdown['efficient_assets'] = True
    else:
        breakdown['efficient_assets'] = False

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
    try:
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

        working_capital = current_assets - current_liabilities
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities if total_liabilities > 0 else 0
        x5 = revenue / total_assets

        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)

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
    composite_score = 0

    piotroski = calculate_piotroski_score(fundamentals)
    piotroski_points = (piotroski['score'] / 9) * 40
    composite_score += piotroski_points

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
        composite_score += 15

    profit_margin = fundamentals.get('profitMargins', 0) * 100
    roe = fundamentals.get('returnOnEquity', 0) * 100
    roa = fundamentals.get('returnOnAssets', 0) * 100

    if profit_margin > 20:
        composite_score += 10
    elif profit_margin > 10:
        composite_score += 7
    elif profit_margin > 5:
        composite_score += 4

    if roe > 20:
        composite_score += 10
    elif roe > 15:
        composite_score += 7
    elif roe > 10:
        composite_score += 4

    if roa > 10:
        composite_score += 10
    elif roa > 5:
        composite_score += 7
    elif roa > 2:
        composite_score += 4

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
    quality = calculate_quality_composite_score(fundamentals)

    summary = f"Quality Score: {quality['composite_score']}/100 ({quality['rating']})\n"
    summary += f"Piotroski F-Score: {quality['piotroski']['score']}/9 ({quality['piotroski']['rating']})\n"

    altman = quality['altman']
    if altman['score'] is not None:
        summary += f"Altman Z-Score: {altman['score']} ({altman['rating']})"
    else:
        summary += "Altman Z-Score: N/A"

    return summary
