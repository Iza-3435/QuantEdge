"""Concept drift detection for market regime changes."""
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from src.core.logging import app_logger
from src.core.metrics import drift_detected, drift_score
from src.core.config import settings


@dataclass
class DriftReport:
    """Drift detection report."""
    drift_detected: bool
    psi_score: float
    js_divergence: float
    ks_statistic: float
    timestamp: datetime
    features_drifted: List[str]
    recommendation: str


class ConceptDriftDetector:
    """
    Detect distribution shifts in market data using multiple methods:
    - PSI (Population Stability Index)
    - JS Divergence (Jensen-Shannon)
    - KS Test (Kolmogorov-Smirnov)
    """

    def __init__(self, window_size: Optional[int] = None):
        self.config = settings.drift_detection
        self.window_size = window_size or self.config.window_size

        # Reference distribution (baseline)
        self.reference_data: Optional[np.ndarray] = None
        self.reference_bins: Optional[List[np.ndarray]] = None

        # Recent observations buffer
        self.recent_buffer = deque(maxlen=self.window_size)

        # Drift history
        self.drift_history: List[DriftReport] = []

    def set_reference(self, data: np.ndarray):
        """Set reference distribution from historical data."""
        self.reference_data = data
        self.reference_bins = []

        # Create bins for each feature
        n_features = data.shape[1]
        for i in range(n_features):
            feature_data = data[:, i]
            # Use quantile-based bins for robustness
            bins = np.percentile(
                feature_data,
                np.linspace(0, 100, 11)
            )
            self.reference_bins.append(bins)

        app_logger.info(
            f"Reference distribution set with {len(data)} samples, "
            f"{n_features} features"
        )

    def add_observation(self, features: np.ndarray):
        """Add new observation to buffer."""
        self.recent_buffer.append(features)

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: np.ndarray
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        # Bin data
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Add small constant to avoid division by zero
        ref_hist = ref_hist + 1e-6
        cur_hist = cur_hist + 1e-6

        # Normalize
        ref_pct = ref_hist / ref_hist.sum()
        cur_pct = cur_hist / cur_hist.sum()

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def calculate_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: np.ndarray
    ) -> float:
        """
        Calculate Jensen-Shannon divergence.

        More symmetric than KL divergence.
        Range: [0, ln(2)]
        """
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize
        ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
        cur_hist = cur_hist / (cur_hist.sum() + 1e-10)

        # Calculate JS divergence
        m = 0.5 * (ref_hist + cur_hist)

        def kl_div(p, q):
            return np.sum(np.where(p != 0, p * np.log((p + 1e-10) / (q + 1e-10)), 0))

        js_div = 0.5 * kl_div(ref_hist, m) + 0.5 * kl_div(cur_hist, m)
        return float(js_div)

    def detect_drift(self) -> Optional[DriftReport]:
        """
        Detect if concept drift has occurred.

        Returns DriftReport if enough data, None otherwise.
        """
        if self.reference_data is None:
            app_logger.warning("Reference distribution not set")
            return None

        if len(self.recent_buffer) < self.window_size * 0.5:
            app_logger.debug(
                f"Insufficient data for drift detection: "
                f"{len(self.recent_buffer)}/{self.window_size}"
            )
            return None

        # Convert buffer to array
        current_data = np.array(list(self.recent_buffer))

        n_features = self.reference_data.shape[1]
        psi_scores = []
        js_scores = []
        ks_statistics = []
        drifted_features = []

        # Check each feature
        for i in range(n_features):
            ref_feature = self.reference_data[:, i]
            cur_feature = current_data[:, i]
            bins = self.reference_bins[i]

            # PSI
            psi = self.calculate_psi(ref_feature, cur_feature, bins)
            psi_scores.append(psi)

            # JS Divergence
            js = self.calculate_js_divergence(ref_feature, cur_feature, bins)
            js_scores.append(js)

            # KS Test
            ks_stat, _ = stats.ks_2samp(ref_feature, cur_feature)
            ks_statistics.append(ks_stat)

            # Check if feature drifted
            if (psi > self.config.threshold_psi or
                js > self.config.threshold_js):
                drifted_features.append(f"feature_{i}")

        # Aggregate scores
        avg_psi = np.mean(psi_scores)
        avg_js = np.mean(js_scores)
        avg_ks = np.mean(ks_statistics)

        # Determine if drift detected
        drift_detected_flag = (
            avg_psi > self.config.threshold_psi or
            avg_js > self.config.threshold_js or
            len(drifted_features) > n_features * 0.2  # 20% of features
        )

        # Generate recommendation
        if drift_detected_flag:
            if self.config.retrain_on_drift:
                recommendation = "RETRAIN_RECOMMENDED"
            else:
                recommendation = "MONITOR_CLOSELY"
        else:
            recommendation = "NO_ACTION_NEEDED"

        report = DriftReport(
            drift_detected=drift_detected_flag,
            psi_score=avg_psi,
            js_divergence=avg_js,
            ks_statistic=avg_ks,
            timestamp=datetime.utcnow(),
            features_drifted=drifted_features,
            recommendation=recommendation
        )

        # Update metrics
        if drift_detected_flag:
            drift_detected.labels(metric_type="psi").inc()
            app_logger.warning(
                f"Drift detected! PSI={avg_psi:.4f}, JS={avg_js:.4f}, "
                f"Features drifted: {len(drifted_features)}"
            )
        else:
            app_logger.info(
                f"No drift detected. PSI={avg_psi:.4f}, JS={avg_js:.4f}"
            )

        drift_score.labels(metric_type="psi").set(avg_psi)
        drift_score.labels(metric_type="js_divergence").set(avg_js)

        self.drift_history.append(report)
        return report

    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {
                'total_checks': 0,
                'drift_detections': 0,
                'avg_psi': 0.0,
                'avg_js': 0.0
            }

        drift_count = sum(1 for r in self.drift_history if r.drift_detected)

        return {
            'total_checks': len(self.drift_history),
            'drift_detections': drift_count,
            'drift_rate': drift_count / len(self.drift_history),
            'avg_psi': np.mean([r.psi_score for r in self.drift_history]),
            'avg_js': np.mean([r.js_divergence for r in self.drift_history]),
            'last_check': self.drift_history[-1].timestamp.isoformat(),
            'last_drift': max(
                (r.timestamp for r in self.drift_history if r.drift_detected),
                default=None
            )
        }

    def reset_reference(self):
        """Reset reference distribution (e.g., after retraining)."""
        if len(self.recent_buffer) > 0:
            self.reference_data = np.array(list(self.recent_buffer))
            self.reference_bins = []

            n_features = self.reference_data.shape[1]
            for i in range(n_features):
                feature_data = self.reference_data[:, i]
                bins = np.percentile(feature_data, np.linspace(0, 100, 11))
                self.reference_bins.append(bins)

            app_logger.info("Reference distribution reset from recent data")
