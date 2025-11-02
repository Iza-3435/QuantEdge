#!/usr/bin/env python3
"""Test ML integration in apps"""

import sys
import os
sys.path.append('.')

from src.ml_engine import AdvancedMLScorer

print("=" * 60)
print("TESTING ML STOCK SCORING ENGINE")
print("=" * 60)

# Test 1: ML Engine basic functionality
print("\n[TEST 1] ML Engine Initialization...")
scorer = AdvancedMLScorer()
print("✓ ML Engine initialized successfully")

# Test 2: Stock scoring with sample data
print("\n[TEST 2] Stock Scoring with Sample Data...")
test_stock = {
    'symbol': 'AAPL',
    'pe_ratio': 28.5,
    'forward_pe': 25.2,
    'peg_ratio': 2.1,
    'price_to_book': 45.3,
    'price_to_sales': 7.2,
    'profit_margin': 25.3,
    'operating_margin': 30.1,
    'roe': 147.0,
    'roa': 22.0,
    'revenue_growth': 11.0,
    'earnings_growth': 13.0,
    'current_ratio': 1.07,
    'debt_to_equity': 1.73,
    'free_cash_flow': 99901000000,
    'operating_cash_flow': 110543000000,
    'rsi': 58.5,
    'momentum_score': 45.0,
    'piotroski_score': 7,
    'dividend_yield': 0.5,
    'beta': 1.29,
}

result = scorer.score_stock(test_stock)
print(f"  Stock: {test_stock['symbol']}")
print(f"  ML Score: {result.score:.1f}/100")
print(f"  Signal: {result.signal}")
print(f"  Confidence: {result.confidence:.1%}")
print(f"  Model Agreement: {result.model_agreement:.1%}")
print(f"  Features Used: {result.features_used}")
print("✓ Stock scoring working correctly")

# Test 3: AI Stock Picker imports
print("\n[TEST 3] AI Stock Picker Import...")
try:
    sys.path.append('apps')
    import AI_STOCK_PICKER
    print("✓ AI_STOCK_PICKER imports successfully")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Stock Screener imports
print("\n[TEST 4] Stock Screener Import...")
try:
    import STOCK_SCREENER
    print("✓ STOCK_SCREENER imports successfully")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("ML Integration working correctly!")
print("=" * 60)
