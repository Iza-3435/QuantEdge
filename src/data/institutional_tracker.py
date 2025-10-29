"""
Institutional Holdings Tracker
Tracks 13F filings to see what hedge funds and institutions are buying/selling

13F = Quarterly report filed by institutions managing >$100M
Shows positions of: Berkshire Hathaway, Ray Dalio, Citadel, etc.

Why this matters:
- Follow the smart money
- Institutions have massive research teams
- Their moves can predict stock performance
- Clustering of institutional buying = strong bullish signal
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class InstitutionalPosition:
    """Single institutional position."""
    institution_name: str
    shares: float
    value: float
    percent_of_portfolio: float
    change_in_shares: float  # vs previous quarter
    change_in_value: float
    filing_date: datetime
    position_type: str  # 'new', 'increased', 'decreased', 'sold_out'


@dataclass
class InstitutionalSignal:
    """Institutional activity signal."""
    signal: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: float  # 0-1
    confidence: float

    # Metrics
    total_institutions: int
    total_shares_held: float
    total_value_held: float
    percent_institutional_ownership: float

    # Changes (quarter over quarter)
    institutions_added: int  # New positions
    institutions_increased: int
    institutions_decreased: int
    institutions_sold: int
    net_institutional_flow: float  # Dollars

    # Top holders
    top_holders: List[str]
    top_buyers: List[str]  # Biggest increasers
    top_sellers: List[str]  # Biggest decreasers

    # Insights
    insights: List[str]
    bullish_signals: List[str]
    bearish_signals: List[str]


class InstitutionalTracker:
    """
    Tracks institutional holdings and generates smart money signals.

    Data sources (in production):
    - SEC 13F filings
    - WhaleWisdom API
    - Whalewisdom.com scraping
    - FinTel.io API
    """

    def __init__(self):
        # Notable institutions to track (importance weighted)
        self.whale_institutions = {
            'Berkshire Hathaway': 1.0,
            'Vanguard': 0.9,
            'BlackRock': 0.9,
            'Renaissance Technologies': 1.0,
            'Bridgewater Associates': 0.95,
            'Citadel': 0.85,
            'Two Sigma': 0.85,
            'DE Shaw': 0.80,
            'AQR Capital': 0.75,
            'Millennium Management': 0.75
        }

    def get_institutional_positions(
        self,
        ticker: str,
        quarter: Optional[str] = None
    ) -> List[InstitutionalPosition]:
        """
        Get institutional positions for a ticker.

        In production, would fetch from:
        - SEC EDGAR 13F filings
        - WhaleWisdom API
        - Bloomberg Terminal

        For now, returns mock data with realistic patterns.
        """
        return self._generate_mock_positions(ticker)

    def _generate_mock_positions(self, ticker: str) -> List[InstitutionalPosition]:
        """Generate realistic mock institutional positions."""
        positions = []

        np.random.seed(hash(ticker) % 2**32)

        # Generate positions for major institutions
        institutions = [
            'Vanguard Group',
            'BlackRock',
            'State Street',
            'Berkshire Hathaway',
            'Fidelity',
            'Capital Group',
            'T Rowe Price',
            'Geode Capital',
            'Northern Trust',
            'Morgan Stanley',
            'Goldman Sachs',
            'JP Morgan',
            'Wellington Management',
            'Invesco',
            'Bank of America'
        ]

        for i, inst in enumerate(institutions):
            # Larger institutions hold more
            base_shares = np.random.uniform(10_000_000, 100_000_000) * (1 - i * 0.05)
            shares = base_shares + np.random.randn() * base_shares * 0.1

            price = 150 + np.random.randn() * 20  # Mock price
            value = shares * price

            # Simulate quarterly changes
            change_pct = np.random.randn() * 0.15  # -15% to +15% typical
            change_in_shares = shares * change_pct
            change_in_value = value * change_pct

            # Position type
            if abs(change_pct) < 0.02:
                pos_type = 'maintained'
            elif change_pct > 0.1:
                pos_type = 'increased'
            elif change_pct > 0:
                pos_type = 'added'
            elif change_pct < -0.1:
                pos_type = 'decreased'
            else:
                pos_type = 'reduced'

            positions.append(InstitutionalPosition(
                institution_name=inst,
                shares=shares,
                value=value,
                percent_of_portfolio=np.random.uniform(0.5, 5.0),
                change_in_shares=change_in_shares,
                change_in_value=change_in_value,
                filing_date=datetime.now(),
                position_type=pos_type
            ))

        return positions

    def analyze_institutional_activity(
        self,
        ticker: str
    ) -> InstitutionalSignal:
        """
        Analyze institutional activity and generate signal.

        Key insights:
        - Net institutional buying = bullish
        - Top hedge funds accumulating = very bullish
        - Mass exodus = bearish
        - Concentration increasing = bullish
        """
        positions = self.get_institutional_positions(ticker)

        if not positions:
            return self._no_activity_signal()

        # Calculate metrics
        total_institutions = len(positions)
        total_shares = sum(p.shares for p in positions)
        total_value = sum(p.value for p in positions)

        # Estimate institutional ownership %
        # (In production, would get from actual float data)
        estimated_float = total_shares * 1.5  # Mock
        inst_ownership_pct = (total_shares / estimated_float) * 100

        # Analyze changes
        added = [p for p in positions if 'add' in p.position_type.lower() or 'new' in p.position_type.lower()]
        increased = [p for p in positions if 'increas' in p.position_type.lower()]
        decreased = [p for p in positions if 'decreas' in p.position_type.lower() or 'reduc' in p.position_type.lower()]
        sold = [p for p in positions if 'sold' in p.position_type.lower()]

        institutions_added = len(added)
        institutions_increased = len(increased)
        institutions_decreased = len(decreased)
        institutions_sold = len(sold)

        # Net flow
        net_flow = sum(p.change_in_value for p in positions)

        # Top holders (by value)
        top_holders = sorted(positions, key=lambda x: x.value, reverse=True)[:5]
        top_holder_names = [h.institution_name for h in top_holders]

        # Top buyers (biggest increases)
        buyers = [p for p in positions if p.change_in_value > 0]
        top_buyers = sorted(buyers, key=lambda x: x.change_in_value, reverse=True)[:3]
        top_buyer_names = [f"{b.institution_name} (+${b.change_in_value/1e6:.1f}M)" for b in top_buyers]

        # Top sellers
        sellers = [p for p in positions if p.change_in_value < 0]
        top_sellers = sorted(sellers, key=lambda x: x.change_in_value)[:3]
        top_seller_names = [f"{s.institution_name} (-${abs(s.change_in_value)/1e6:.1f}M)" for s in top_sellers]

        # Generate insights
        insights = []
        bullish_signals = []
        bearish_signals = []

        # Bullish signals
        if net_flow > 100_000_000:
            insights.append(f"Strong net institutional buying: +${net_flow/1e6:.0f}M")
            bullish_signals.append("Heavy institutional accumulation")

        if institutions_added > institutions_sold:
            insights.append(f"{institutions_added} new institutional positions vs {institutions_sold} exits")
            bullish_signals.append("Net new institutional interest")

        if institutions_increased > institutions_decreased * 2:
            insights.append(f"{institutions_increased} institutions increased vs {institutions_decreased} decreased")
            bullish_signals.append("Widespread institutional buying")

        # Check for whale activity
        whale_buying = [p for p in positions if p.institution_name in self.whale_institutions and p.change_in_value > 50_000_000]
        if whale_buying:
            insights.append(f"{len(whale_buying)} top hedge funds significantly increased positions")
            bullish_signals.append("Whale accumulation detected")

        # Bearish signals
        if net_flow < -100_000_000:
            insights.append(f"Net institutional selling: -${abs(net_flow)/1e6:.0f}M")
            bearish_signals.append("Institutional distribution")

        if institutions_sold > institutions_added:
            insights.append(f"{institutions_sold} institutions exited vs {institutions_added} new")
            bearish_signals.append("Institutional exodus")

        if institutions_decreased > institutions_increased * 2:
            bearish_signals.append("Widespread institutional selling")

        # Calculate signal
        buy_pressure = institutions_added + institutions_increased * 2
        sell_pressure = institutions_sold + institutions_decreased * 2

        if buy_pressure > sell_pressure * 1.5:
            signal = "bullish"
            signal_strength = min((buy_pressure / sell_pressure - 1) / 2, 1.0) if sell_pressure > 0 else 0.8
            confidence = 0.6 + signal_strength * 0.25
        elif sell_pressure > buy_pressure * 1.5:
            signal = "bearish"
            signal_strength = min((sell_pressure / buy_pressure - 1) / 2, 1.0) if buy_pressure > 0 else 0.8
            confidence = 0.6 + signal_strength * 0.25
        else:
            signal = "neutral"
            signal_strength = 0.3
            confidence = 0.5

        # Boost confidence if whales are involved
        if whale_buying:
            confidence = min(confidence + 0.15, 0.95)

        return InstitutionalSignal(
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            total_institutions=total_institutions,
            total_shares_held=total_shares,
            total_value_held=total_value,
            percent_institutional_ownership=inst_ownership_pct,
            institutions_added=institutions_added,
            institutions_increased=institutions_increased,
            institutions_decreased=institutions_decreased,
            institutions_sold=institutions_sold,
            net_institutional_flow=net_flow,
            top_holders=top_holder_names,
            top_buyers=top_buyer_names if top_buyer_names else ["None"],
            top_sellers=top_seller_names if top_seller_names else ["None"],
            insights=insights if insights else ["Institutional activity is mixed"],
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals
        )

    def _no_activity_signal(self) -> InstitutionalSignal:
        """Return neutral signal for no data."""
        return InstitutionalSignal(
            signal="neutral",
            signal_strength=0.0,
            confidence=0.3,
            total_institutions=0,
            total_shares_held=0.0,
            total_value_held=0.0,
            percent_institutional_ownership=0.0,
            institutions_added=0,
            institutions_increased=0,
            institutions_decreased=0,
            institutions_sold=0,
            net_institutional_flow=0.0,
            top_holders=[],
            top_buyers=[],
            top_sellers=[],
            insights=["No institutional data available"],
            bullish_signals=[],
            bearish_signals=[]
        )


def quick_institutional_test(ticker: str = "AAPL"):
    """Quick test of institutional tracker."""
    print(f"\n{'='*80}")
    print(f"INSTITUTIONAL HOLDINGS ANALYSIS: {ticker}")
    print(f"{'='*80}\n")

    tracker = InstitutionalTracker()
    signal = tracker.analyze_institutional_activity(ticker)

    # Display signal
    signal_emoji = "🟢" if signal.signal == "bullish" else "🔴" if signal.signal == "bearish" else "⚪"

    print(f"{signal_emoji} INSTITUTIONAL SIGNAL: {signal.signal.upper()}")
    print(f"  Strength:              {signal.signal_strength:.0%}")
    print(f"  Confidence:            {signal.confidence:.0%}")
    print()
    print("📊 HOLDINGS SUMMARY:")
    print(f"  Total Institutions:    {signal.total_institutions}")
    print(f"  Total Shares:          {signal.total_shares_held/1e6:.1f}M")
    print(f"  Total Value:           ${signal.total_value_held/1e9:.2f}B")
    print(f"  Inst. Ownership:       {signal.percent_institutional_ownership:.1f}%")
    print()
    print("📈 QUARTERLY CHANGES:")
    print(f"  New Positions:         {signal.institutions_added}")
    print(f"  Increased:             {signal.institutions_increased}")
    print(f"  Decreased:             {signal.institutions_decreased}")
    print(f"  Sold Out:              {signal.institutions_sold}")
    print(f"  Net Flow:              ${signal.net_institutional_flow/1e6:+.0f}M")
    print()
    print("🏆 TOP HOLDERS:")
    for i, holder in enumerate(signal.top_holders[:5], 1):
        print(f"  {i}. {holder}")
    print()

    if signal.top_buyers and signal.top_buyers != ["None"]:
        print("🟢 TOP BUYERS (This Quarter):")
        for buyer in signal.top_buyers:
            print(f"  ✅ {buyer}")
        print()

    if signal.top_sellers and signal.top_sellers != ["None"]:
        print("🔴 TOP SELLERS (This Quarter):")
        for seller in signal.top_sellers:
            print(f"  📉 {seller}")
        print()

    if signal.bullish_signals:
        print("🟢 BULLISH SIGNALS:")
        for sig in signal.bullish_signals:
            print(f"  ✅ {sig}")
        print()

    if signal.bearish_signals:
        print("🔴 BEARISH SIGNALS:")
        for sig in signal.bearish_signals:
            print(f"  ⚠️  {sig}")
        print()

    print("💡 INSIGHTS:")
    for insight in signal.insights:
        print(f"  • {insight}")
    print()
    print(f"{'='*80}\n")


if __name__ == "__main__":
    quick_institutional_test("AAPL")
