"""ML component tests."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.retrieval.advanced_pattern_retrieval import AdvancedPatternRetrieval, MarketPattern
from src.ml.drift_detection import ConceptDriftDetector


@pytest.fixture
def sample_market_data():
    """Generate sample market data."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    np.random.seed(42)

    data = {
        'Close': 100 + np.cumsum(np.random.randn(500) * 2),
        'Volume': np.random.randint(1000000, 5000000, 500),
        'Open': 100 + np.cumsum(np.random.randn(500) * 2),
        'High': 100 + np.cumsum(np.random.randn(500) * 2) + 5,
        'Low': 100 + np.cumsum(np.random.randn(500) * 2) - 5,
    }

    df = pd.DataFrame(data, index=dates)

    # Add indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    return df.dropna()


class TestPatternRetrieval:
    """Test pattern retrieval system."""

    def test_feature_extraction(self, sample_market_data):
        """Test feature extraction."""
        retriever = AdvancedPatternRetrieval()

        features = retriever.extract_pattern_features(
            sample_market_data,
            idx=100,
            window=20
        )

        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert not np.any(np.isnan(features))

    def test_build_pattern_database(self, sample_market_data):
        """Test building pattern database."""
        retriever = AdvancedPatternRetrieval()

        retriever.build_pattern_database(
            sample_market_data,
            forward_periods=5,
            calibration_split=0.2
        )

        assert len(retriever.patterns) > 0
        assert retriever.embeddings is not None
        assert retriever.index is not None

    def test_retrieve_similar_patterns(self, sample_market_data):
        """Test retrieving similar patterns."""
        retriever = AdvancedPatternRetrieval()
        retriever.build_pattern_database(sample_market_data)

        current_features = retriever.extract_pattern_features(
            sample_market_data,
            idx=len(sample_market_data) - 1
        )

        similar = retriever.retrieve_similar_patterns(current_features, top_k=5)

        assert len(similar) == 5
        assert all(isinstance(p, MarketPattern) for p, _ in similar)
        assert all(0 <= sim <= 1 for _, sim in similar)

    def test_prediction_with_uncertainty(self, sample_market_data):
        """Test prediction with uncertainty quantification."""
        retriever = AdvancedPatternRetrieval()
        retriever.build_pattern_database(sample_market_data)

        current_features = retriever.extract_pattern_features(
            sample_market_data,
            idx=len(sample_market_data) - 1
        )

        prediction = retriever.predict_with_uncertainty(
            current_features,
            top_k=10,
            confidence_level=0.9
        )

        assert prediction.predicted_return is not None
        assert prediction.confidence >= 0
        assert prediction.uncertainty_lower < prediction.uncertainty_upper
        assert len(prediction.examples) > 0

    def test_save_load(self, sample_market_data, tmp_path):
        """Test saving and loading pattern database."""
        retriever = AdvancedPatternRetrieval()
        retriever.build_pattern_database(sample_market_data)

        # Save
        save_path = tmp_path / "test_patterns.pkl"
        retriever.save(str(save_path))

        # Load
        new_retriever = AdvancedPatternRetrieval()
        new_retriever.load(str(save_path))

        assert len(new_retriever.patterns) == len(retriever.patterns)
        assert new_retriever.embeddings.shape == retriever.embeddings.shape


class TestDriftDetection:
    """Test concept drift detection."""

    @pytest.fixture
    def reference_data(self):
        """Generate reference distribution."""
        np.random.seed(42)
        return np.random.randn(1000, 10)

    @pytest.fixture
    def similar_data(self):
        """Generate similar distribution."""
        np.random.seed(43)
        return np.random.randn(500, 10)

    @pytest.fixture
    def drifted_data(self):
        """Generate drifted distribution."""
        np.random.seed(44)
        return np.random.randn(500, 10) + 2.0  # Shifted mean

    def test_set_reference(self, reference_data):
        """Test setting reference distribution."""
        detector = ConceptDriftDetector(window_size=500)
        detector.set_reference(reference_data)

        assert detector.reference_data is not None
        assert len(detector.reference_bins) == reference_data.shape[1]

    def test_no_drift_detection(self, reference_data, similar_data):
        """Test that similar data doesn't trigger drift."""
        detector = ConceptDriftDetector(window_size=500)
        detector.set_reference(reference_data)

        # Add similar observations
        for observation in similar_data:
            detector.add_observation(observation)

        report = detector.detect_drift()

        assert report is not None
        assert not report.drift_detected
        assert report.psi_score < 0.1

    def test_drift_detection(self, reference_data, drifted_data):
        """Test that drifted data triggers detection."""
        detector = ConceptDriftDetector(window_size=500)
        detector.set_reference(reference_data)

        # Add drifted observations
        for observation in drifted_data:
            detector.add_observation(observation)

        report = detector.detect_drift()

        assert report is not None
        # Should detect drift due to shifted mean
        assert report.psi_score > 0 or report.js_divergence > 0

    def test_drift_summary(self, reference_data, drifted_data):
        """Test drift summary."""
        detector = ConceptDriftDetector(window_size=500)
        detector.set_reference(reference_data)

        for observation in drifted_data:
            detector.add_observation(observation)

        detector.detect_drift()

        summary = detector.get_drift_summary()

        assert summary['total_checks'] > 0
        assert 'avg_psi' in summary
        assert 'avg_js' in summary

    def test_psi_calculation(self):
        """Test PSI calculation."""
        detector = ConceptDriftDetector()

        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        bins = np.percentile(ref, np.linspace(0, 100, 11))

        psi = detector.calculate_psi(ref, cur, bins)

        assert psi >= 0
        assert psi < 0.1  # Should be low for similar distributions

    def test_js_divergence_calculation(self):
        """Test JS divergence calculation."""
        detector = ConceptDriftDetector()

        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)
        bins = np.percentile(ref, np.linspace(0, 100, 11))

        js_div = detector.calculate_js_divergence(ref, cur, bins)

        assert js_div >= 0
        assert js_div < 0.1  # Should be low for similar distributions


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_prediction_pipeline(self, sample_market_data):
        """Test complete prediction pipeline."""
        # Build pattern database
        retriever = AdvancedPatternRetrieval()
        retriever.build_pattern_database(sample_market_data)

        # Setup drift detector
        drift_detector = ConceptDriftDetector()
        drift_detector.set_reference(retriever.embeddings)

        # Make prediction
        current_features = retriever.extract_pattern_features(
            sample_market_data,
            idx=len(sample_market_data) - 1
        )

        prediction = retriever.predict_with_uncertainty(current_features)

        # Check drift
        drift_detector.add_observation(current_features)

        assert prediction is not None
        assert prediction.confidence > 0

    def test_multiple_predictions(self, sample_market_data):
        """Test making multiple predictions."""
        retriever = AdvancedPatternRetrieval()
        retriever.build_pattern_database(sample_market_data)

        predictions = []
        for idx in range(len(sample_market_data) - 10, len(sample_market_data)):
            features = retriever.extract_pattern_features(sample_market_data, idx)
            if features is not None:
                pred = retriever.predict_with_uncertainty(features)
                predictions.append(pred)

        assert len(predictions) > 0
        assert all(p.confidence > 0 for p in predictions)
