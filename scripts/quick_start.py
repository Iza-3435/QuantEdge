"""Quick start script to demonstrate the system."""
import sys
from pathlib import Path
import warnings

# Suppress pandas FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
from datetime import datetime, timedelta

from src.retrieval.advanced_pattern_retrieval import AdvancedPatternRetrieval
from src.ml.drift_detection import ConceptDriftDetector
from src.evaluation.backtesting import Backtester
from src.core.logging import app_logger


def main():
    print("="*70)
    print("AI MARKET INTELLIGENCE - QUICK START DEMO")
    print("="*70)

    # Step 1: Download real market data
    print("\n[1/5] Downloading real-time market data for AAPL...")
    df = yf.download("AAPL", period="5y", progress=False)

    # Add technical indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df.dropna(inplace=True)
    print(f"✓ Downloaded {len(df)} days of market data")

    # Step 2: Build pattern database
    print("\n[2/5] Building pattern database with HNSW indexing...")
    retriever = AdvancedPatternRetrieval()
    retriever.build_pattern_database(df, forward_periods=5)
    print(f"✓ Indexed {len(retriever.patterns)} historical patterns")

    # Step 3: Make real-time prediction
    print("\n[3/5] Making real-time prediction...")
    current_idx = len(df) - 1
    current_features = retriever.extract_pattern_features(df, current_idx)

    prediction = retriever.predict_with_uncertainty(
        current_features,
        top_k=10,
        confidence_level=0.9
    )

    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Symbol:               AAPL")
    print(f"Date:                 {df.index[-1].date()}")
    print(f"Current Price:        ${float(df['Close'].iloc[-1]):.2f}")
    print(f"\nPredicted Return:     {prediction.predicted_return*100:+.2f}%")
    print(f"Confidence:           {prediction.confidence:.2f}")
    print(f"Profit Probability:   {prediction.profit_probability*100:.1f}%")
    print(f"\n95% Confidence Interval:")
    print(f"  Lower Bound:        {prediction.uncertainty_lower*100:+.2f}%")
    print(f"  Upper Bound:        {prediction.uncertainty_upper*100:+.2f}%")
    print(f"\nMost Similar Historical Patterns:")
    for i, ex in enumerate(prediction.examples, 1):
        print(f"  {i}. {ex['date']}: {ex['outcome_return']*100:+.2f}% "
              f"(similarity: {ex['similarity']:.3f})")
    print("="*70)

    # Step 4: Drift detection
    print("\n[4/5] Setting up concept drift detection...")
    drift_detector = ConceptDriftDetector(window_size=500)
    drift_detector.set_reference(retriever.embeddings)

    # Simulate recent observations
    for idx in range(len(df) - 100, len(df)):
        features = retriever.extract_pattern_features(df, idx)
        if features is not None:
            drift_detector.add_observation(features)

    drift_report = drift_detector.detect_drift()
    if drift_report:
        print(f"✓ Drift Detection Status: "
              f"{'⚠️  DRIFT DETECTED' if drift_report.drift_detected else '✅ NO DRIFT'}")
        print(f"  PSI Score:          {drift_report.psi_score:.4f}")
        print(f"  JS Divergence:      {drift_report.js_divergence:.4f}")

    # Step 5: Run backtest
    print("\n[5/5] Running backtest...")
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        confidence_threshold=0.5,
        max_position_size=0.2
    )

    results = backtester.run_backtest(
        df=df,
        retriever=retriever,
        start_idx=200,
        holding_period=5,
        rebalance_frequency=5
    )

    backtester.print_summary(results)

    # Save results
    output_dir = Path("data/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever.save(str(output_dir / "pattern_db.pkl"))
    backtester.save_results(results, str(output_dir / "backtest_results.json"))

    print(f"\n✓ Results saved to {output_dir}/")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python -m uvicorn src.api.main:app --reload")
    print("\n2. Make API requests:")
    print("   curl -X POST http://localhost:8000/auth/token -d 'username=demo&password=demo'")
    print("\n3. Monitor with Prometheus:")
    print("   http://localhost:9090")
    print("\n4. View dashboards in Grafana:")
    print("   http://localhost:3000")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
