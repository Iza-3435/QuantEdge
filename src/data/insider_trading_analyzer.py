"""
Insider Trading Detection System
Tracks SEC Form 4 filings to detect insider buying/selling

Form 4 = Filed by company insiders when they buy/sell stock
This is LEGAL and public information that can predict stock movements!

Key insights:
- Insider buying often precedes positive news (they know first!)
- Heavy insider selling can signal trouble
- CEO/CFO trades more important than lower-level employees
"""
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class InsiderTransaction:
    """Single insider transaction."""
    filing_date: datetime
    transaction_date: datetime
    insider_name: str
    insider_title: str
    transaction_type: str  # 'buy' or 'sell'
    shares: float
    price_per_share: float
    total_value: float
    shares_owned_after: float
    relationship: str  # 'officer', 'director', '10% owner', etc.


@dataclass
class InsiderSignal:
    """Insider trading signal analysis."""
    signal: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: float  # 0-1
    confidence: float  # 0-1

    # Metrics
    total_transactions_30d: int
    buy_transactions: int
    sell_transactions: int
    net_shares_30d: float  # Positive = net buying
    net_value_30d: float  # Dollar value
    buy_sell_ratio: float

    # Key insiders
    ceo_activity: str
    cfo_activity: str
    director_activity: str

    # Insights
    insights: List[str]
    red_flags: List[str]
    bullish_signals: List[str]


class InsiderTradingAnalyzer:
    """
    Analyzes SEC Form 4 filings to detect insider trading patterns.

    Why this matters:
    - Insiders have information advantage
    - Buying by executives often precedes stock gains
    - Unusual selling patterns can predict problems
    - Clusters of insider buying = strong bullish signal
    """

    def __init__(self, sec_user_agent: str = "AI Market Intelligence research@example.com"):
        self.user_agent = sec_user_agent
        self.base_url = "https://data.sec.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': sec_user_agent,
            'Accept-Encoding': 'gzip, deflate'
        })

        # Rate limiting
        self.last_request = datetime.now()
        self.min_interval = 0.11  # SEC requires max 10 req/sec

    def get_insider_transactions(
        self,
        ticker: str,
        lookback_days: int = 90
    ) -> List[InsiderTransaction]:
        """
        Get recent insider transactions for a ticker.

        Args:
            ticker: Stock ticker
            lookback_days: Days to look back

        Returns:
            List of insider transactions
        """
        # This is a simplified version
        # In production, would parse actual Form 4 filings from SEC EDGAR

        # For now, return mock data structure
        # Real implementation would:
        # 1. Get company CIK
        # 2. Fetch Form 4 filings
        # 3. Parse XML to extract transaction details
        # 4. Identify insider roles (CEO, CFO, Director)

        return self._generate_mock_transactions(ticker, lookback_days)

    def _generate_mock_transactions(self, ticker: str, days: int) -> List[InsiderTransaction]:
        """Generate realistic mock insider transactions for demonstration."""
        transactions = []

        # Simulate some transactions
        np.random.seed(hash(ticker) % 2**32)

        num_transactions = np.random.randint(3, 15)

        titles = [
            ('CEO', 'officer', 0.3),
            ('CFO', 'officer', 0.25),
            ('Director', 'director', 0.2),
            ('VP Sales', 'officer', 0.15),
            ('Board Member', 'director', 0.1)
        ]

        for i in range(num_transactions):
            days_ago = np.random.randint(0, days)
            filing_date = datetime.now() - timedelta(days=days_ago)
            transaction_date = filing_date - timedelta(days=np.random.randint(1, 5))

            # Bias towards buying for mock data
            is_buy = np.random.random() > 0.4

            title, relationship, importance = titles[i % len(titles)]
            name = f"John {chr(65 + i)} Smith"

            shares = np.random.randint(1000, 50000) if is_buy else np.random.randint(5000, 100000)
            price = 150 + np.random.randn() * 20  # Mock price

            transactions.append(InsiderTransaction(
                filing_date=filing_date,
                transaction_date=transaction_date,
                insider_name=name,
                insider_title=title,
                transaction_type='buy' if is_buy else 'sell',
                shares=shares,
                price_per_share=price,
                total_value=shares * price,
                shares_owned_after=shares * 10,  # Mock
                relationship=relationship
            ))

        return sorted(transactions, key=lambda x: x.filing_date, reverse=True)

    def analyze_insider_activity(
        self,
        ticker: str,
        lookback_days: int = 30
    ) -> InsiderSignal:
        """
        Analyze insider trading activity and generate signal.

        Args:
            ticker: Stock ticker
            lookback_days: Days to analyze

        Returns:
            InsiderSignal with detailed analysis
        """
        transactions = self.get_insider_transactions(ticker, lookback_days)

        if not transactions:
            return self._no_activity_signal()

        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent = [t for t in transactions if t.filing_date >= cutoff_date]

        # Analyze transactions
        buys = [t for t in recent if t.transaction_type == 'buy']
        sells = [t for t in recent if t.transaction_type == 'sell']

        total_buys = len(buys)
        total_sells = len(sells)
        total_transactions = total_buys + total_sells

        # Calculate net position
        net_shares = sum(t.shares for t in buys) - sum(t.shares for t in sells)
        net_value = sum(t.total_value for t in buys) - sum(t.total_value for t in sells)

        # Buy/sell ratio
        buy_sell_ratio = total_buys / total_sells if total_sells > 0 else float('inf')

        # Check key insider activity
        ceo_transactions = [t for t in recent if 'ceo' in t.insider_title.lower()]
        cfo_transactions = [t for t in recent if 'cfo' in t.insider_title.lower()]
        director_transactions = [t for t in recent if t.relationship == 'director']

        ceo_activity = self._summarize_transactions(ceo_transactions)
        cfo_activity = self._summarize_transactions(cfo_transactions)
        director_activity = self._summarize_transactions(director_transactions)

        # Generate insights
        insights = []
        bullish_signals = []
        red_flags = []

        # Bullish signals
        if buy_sell_ratio > 2:
            insights.append(f"Strong insider buying: {total_buys} buys vs {total_sells} sells")
            bullish_signals.append("Heavy insider accumulation")

        if net_value > 1_000_000:
            insights.append(f"Large net buying: ${net_value/1e6:.1f}M")
            bullish_signals.append("Multi-million dollar insider buying")

        if len(ceo_transactions) > 0 and all(t.transaction_type == 'buy' for t in ceo_transactions):
            insights.append("CEO is buying shares")
            bullish_signals.append("CEO confidence signal")

        if len(buys) >= 3 and all(t.filing_date >= datetime.now() - timedelta(days=7) for t in buys):
            insights.append("Cluster of insider buying in past week")
            bullish_signals.append("Recent buying cluster")

        # Bearish signals
        if buy_sell_ratio < 0.5:
            insights.append(f"Heavy insider selling: {total_sells} sells vs {total_buys} buys")
            red_flags.append("Significant insider selling")

        if net_value < -1_000_000:
            insights.append(f"Large net selling: ${abs(net_value)/1e6:.1f}M")
            red_flags.append("Multi-million dollar insider selling")

        if len(ceo_transactions) > 0 and all(t.transaction_type == 'sell' for t in ceo_transactions):
            insights.append("CEO is selling shares")
            red_flags.append("CEO selling signal")

        # Calculate signal
        signal_strength = min(abs(buy_sell_ratio - 1) / 5, 1.0)

        if buy_sell_ratio > 1.5:
            signal = "bullish"
            confidence = min(0.5 + signal_strength * 0.3, 0.85)
        elif buy_sell_ratio < 0.67:
            signal = "bearish"
            confidence = min(0.5 + signal_strength * 0.3, 0.85)
        else:
            signal = "neutral"
            confidence = 0.5

        # Adjust confidence based on transaction count
        if total_transactions < 3:
            confidence *= 0.7

        return InsiderSignal(
            signal=signal,
            signal_strength=signal_strength,
            confidence=confidence,
            total_transactions_30d=total_transactions,
            buy_transactions=total_buys,
            sell_transactions=total_sells,
            net_shares_30d=net_shares,
            net_value_30d=net_value,
            buy_sell_ratio=buy_sell_ratio,
            ceo_activity=ceo_activity,
            cfo_activity=cfo_activity,
            director_activity=director_activity,
            insights=insights if insights else ["No significant insider activity"],
            red_flags=red_flags,
            bullish_signals=bullish_signals
        )

    def _summarize_transactions(self, transactions: List[InsiderTransaction]) -> str:
        """Summarize transactions for a role."""
        if not transactions:
            return "No activity"

        buys = sum(1 for t in transactions if t.transaction_type == 'buy')
        sells = sum(1 for t in transactions if t.transaction_type == 'sell')

        if buys > sells:
            return f"Buying ({buys} buys, {sells} sells)"
        elif sells > buys:
            return f"Selling ({sells} sells, {buys} buys)"
        else:
            return f"Mixed ({buys} buys, {sells} sells)"

    def _no_activity_signal(self) -> InsiderSignal:
        """Return signal for no activity."""
        return InsiderSignal(
            signal="neutral",
            signal_strength=0.0,
            confidence=0.3,
            total_transactions_30d=0,
            buy_transactions=0,
            sell_transactions=0,
            net_shares_30d=0.0,
            net_value_30d=0.0,
            buy_sell_ratio=1.0,
            ceo_activity="No activity",
            cfo_activity="No activity",
            director_activity="No activity",
            insights=["No insider transactions in past 30 days"],
            red_flags=[],
            bullish_signals=[]
        )


def quick_insider_test(ticker: str = "AAPL"):
    """Quick test of insider trading analyzer."""
    print(f"\n{'='*80}")
    print(f"INSIDER TRADING ANALYSIS: {ticker}")
    print(f"{'='*80}\n")

    analyzer = InsiderTradingAnalyzer()
    signal = analyzer.analyze_insider_activity(ticker, lookback_days=30)

    # Display signal
    signal_emoji = "🟢" if signal.signal == "bullish" else "🔴" if signal.signal == "bearish" else "⚪"

    print(f"{signal_emoji} INSIDER SIGNAL: {signal.signal.upper()}")
    print(f"  Strength:              {signal.signal_strength:.0%}")
    print(f"  Confidence:            {signal.confidence:.0%}")
    print()
    print("📊 TRANSACTION SUMMARY:")
    print(f"  Total Transactions:    {signal.total_transactions_30d}")
    print(f"  Buy Transactions:      {signal.buy_transactions}")
    print(f"  Sell Transactions:     {signal.sell_transactions}")
    print(f"  Buy/Sell Ratio:        {signal.buy_sell_ratio:.2f}")
    print(f"  Net Shares:            {signal.net_shares_30d:,.0f}")
    print(f"  Net Value:             ${signal.net_value_30d/1e6:.2f}M")
    print()
    print("👔 KEY INSIDER ACTIVITY:")
    print(f"  CEO:                   {signal.ceo_activity}")
    print(f"  CFO:                   {signal.cfo_activity}")
    print(f"  Directors:             {signal.director_activity}")
    print()

    if signal.bullish_signals:
        print("🟢 BULLISH SIGNALS:")
        for sig in signal.bullish_signals:
            print(f"  ✅ {sig}")
        print()

    if signal.red_flags:
        print("🔴 RED FLAGS:")
        for flag in signal.red_flags:
            print(f"  ⚠️  {flag}")
        print()

    print("💡 INSIGHTS:")
    for insight in signal.insights:
        print(f"  • {insight}")
    print()
    print(f"{'='*80}\n")


if __name__ == "__main__":
    quick_insider_test("AAPL")
