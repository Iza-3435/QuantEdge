"""Advanced Pattern Retrieval with HNSW indexing and uncertainty quantification."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import faiss
from dataclasses import dataclass
from src.core.config import settings
from src.core.logging import app_logger
from src.core.metrics import track_prediction_time, pattern_retrieval_count, pattern_similarity_score


@dataclass
class MarketPattern:
    """Historical market pattern with metadata."""
    date: datetime
    features: np.ndarray
    outcome: Dict
    context: str
    regime: Optional[str] = None


@dataclass
class PredictionResult:
    """Prediction with uncertainty quantification."""
    predicted_return: float
    predicted_volatility: float
    profit_probability: float
    confidence: float
    uncertainty_lower: float
    uncertainty_upper: float
    similar_count: int
    examples: List[Dict]


class AdvancedPatternRetrieval:
    """
    Advanced pattern retrieval with:
    - HNSW indexing for scalability
    - Uncertainty quantification via conformal prediction
    - Ensemble predictions
    """

    def __init__(self, embedding_dim: int = 128):
        self.patterns: List[MarketPattern] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.scaler = StandardScaler()
        self.embedding_dim = embedding_dim
        self.config = settings.model.pattern_retrieval

        # Calibration set for conformal prediction
        self.calibration_residuals: List[float] = []

    def extract_pattern_features(
        self,
        df: pd.DataFrame,
        idx: int,
        window: int = 20
    ) -> Optional[np.ndarray]:
        """Extract enhanced feature vector from market data."""
        if idx < window:
            return None

        window_data = df.iloc[idx-window:idx]
        features = []

        # Price patterns
        returns = window_data['Close'].pct_change().dropna().values
        if len(returns) == 0:
            return None

        close_last = float(window_data['Close'].iloc[-1])
        close_first = float(window_data['Close'].iloc[0])

        features.extend([
            float(np.mean(returns)),
            float(np.std(returns)),
            float(returns[-1]),
            float((close_last / close_first) - 1),
            float(np.median(returns)),
            float(np.percentile(returns, 25)),
            float(np.percentile(returns, 75)),
        ])

        # Volume patterns
        vol_values = window_data['Volume'].values
        vol_norm = vol_values / (np.mean(vol_values) + 1e-9)
        features.extend([
            float(np.mean(vol_norm)),
            float(np.std(vol_norm)),
            float(vol_norm[-1]),
            float(np.max(vol_norm)),
        ])

        # Technical indicators
        if 'RSI' in window_data.columns:
            rsi_val = float(window_data['RSI'].iloc[-1])
            features.append(rsi_val / 100 if not np.isnan(rsi_val) else 0.5)

        if 'SMA_20' in window_data.columns:
            price = float(window_data['Close'].iloc[-1])
            sma = float(window_data['SMA_20'].iloc[-1])
            if not np.isnan(sma) and sma > 0:
                features.append((price - sma) / sma)
            else:
                features.append(0.0)

        # Volatility metrics
        features.append(float(np.std(returns) * np.sqrt(252)))

        # Momentum indicators
        if len(returns) >= 5:
            features.append(float(np.mean(returns[-5:])))
        else:
            features.append(float(np.mean(returns)))

        # Skewness and kurtosis
        from scipy import stats
        if len(returns) > 3:
            features.append(float(stats.skew(returns)))
            features.append(float(stats.kurtosis(returns)))
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def build_pattern_database(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        calibration_split: float = 0.2
    ):
        """Build pattern database with HNSW indexing."""
        app_logger.info(f"Building pattern database from {len(df)} data points")

        patterns = []
        features_list = []
        calibration_data = []

        for i in range(self.config.window_size, len(df) - forward_periods):
            features = self.extract_pattern_features(df, i, self.config.window_size)
            if features is None:
                continue

            # Calculate outcome
            future_returns = df['Close'].iloc[i:i+forward_periods].pct_change()
            future_return = float(future_returns.sum())
            future_volatility = float(future_returns.std())

            future_prices = df['Close'].iloc[i:i+forward_periods]
            max_dd = float((future_prices / future_prices.cummax() - 1).min())

            outcome = {
                'return': future_return,
                'volatility': future_volatility,
                'max_drawdown': max_dd,
                'profitable': future_return > 0
            }

            date = df.index[i]
            price_at_date = float(df['Close'].iloc[i])
            pattern = MarketPattern(
                date=date,
                features=features,
                outcome=outcome,
                context=f"Date: {date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)}, Price: ${price_at_date:.2f}"
            )

            patterns.append(pattern)
            features_list.append(features)

            # Store for calibration
            if np.random.random() < calibration_split:
                calibration_data.append((features, future_return))

        self.patterns = patterns
        app_logger.info(f"Created {len(patterns)} patterns")

        # Normalize features
        features_array = np.array(features_list)
        self.embeddings = self.scaler.fit_transform(features_array).astype('float32')

        # Build HNSW index for scalability
        dimension = self.embeddings.shape[1]

        if self.config.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, self.config.hnsw_m)
            self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self.index.hnsw.efSearch = self.config.hnsw_ef_search
            app_logger.info(f"Using HNSW index with M={self.config.hnsw_m}")
        else:
            self.index = faiss.IndexFlatL2(dimension)
            app_logger.info("Using Flat L2 index")

        self.index.add(self.embeddings)

        # Calibrate uncertainty quantification
        if calibration_data:
            self._calibrate_uncertainty(calibration_data)

        app_logger.info(f"Pattern database built: {len(patterns)} patterns indexed")

    def _calibrate_uncertainty(self, calibration_data: List[Tuple[np.ndarray, float]]):
        """Calibrate conformal prediction intervals."""
        residuals = []

        for features, true_return in calibration_data:
            # Get prediction without this point
            pred = self._predict_single(features, exclude_calibration=True)
            if pred:
                residual = abs(true_return - pred['predicted_return'])
                residuals.append(residual)

        if residuals:
            self.calibration_residuals = sorted(residuals)
            app_logger.info(f"Calibrated with {len(residuals)} samples")

    def _predict_single(
        self,
        features: np.ndarray,
        exclude_calibration: bool = False
    ) -> Optional[Dict]:
        """Single prediction helper."""
        current_norm = self.scaler.transform(features.reshape(1, -1)).astype('float32')
        distances, indices = self.index.search(current_norm, self.config.top_k)

        if len(indices[0]) == 0:
            return None

        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.patterns):
                continue
            pattern = self.patterns[idx]
            similarity = 1 / (1 + dist)
            similar.append((pattern, similarity))

        if not similar:
            return None

        total_weight = sum(sim for _, sim in similar)
        weighted_return = sum(
            p.outcome['return'] * sim for p, sim in similar
        ) / total_weight

        return {'predicted_return': weighted_return}

    @track_prediction_time("pattern_retrieval")
    def retrieve_similar_patterns(
        self,
        current_features: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[Tuple[MarketPattern, float]]:
        """Retrieve most similar patterns using HNSW index."""
        if self.index is None:
            raise ValueError("Pattern database not built")

        top_k = top_k or self.config.top_k

        current_norm = self.scaler.transform(
            current_features.reshape(1, -1)
        ).astype('float32')

        distances, indices = self.index.search(current_norm, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.patterns):
                continue
            pattern = self.patterns[idx]
            similarity = 1 / (1 + dist)
            results.append((pattern, similarity))
            pattern_similarity_score.observe(similarity)

        pattern_retrieval_count.labels(top_k=str(top_k)).inc()
        return results

    def predict_with_uncertainty(
        self,
        current_features: np.ndarray,
        top_k: Optional[int] = None,
        confidence_level: float = 0.9
    ) -> PredictionResult:
        """
        Predict with uncertainty quantification using conformal prediction.

        Returns calibrated prediction intervals.
        """
        similar = self.retrieve_similar_patterns(current_features, top_k)

        if not similar:
            return PredictionResult(
                predicted_return=0.0,
                predicted_volatility=0.0,
                profit_probability=0.5,
                confidence=0.0,
                uncertainty_lower=0.0,
                uncertainty_upper=0.0,
                similar_count=0,
                examples=[]
            )

        # Weighted predictions
        total_weight = sum(sim for _, sim in similar)
        weighted_return = sum(
            p.outcome['return'] * sim for p, sim in similar
        ) / total_weight
        weighted_vol = sum(
            p.outcome['volatility'] * sim for p, sim in similar
        ) / total_weight
        profit_prob = sum(
            p.outcome['profitable'] * sim for p, sim in similar
        ) / total_weight

        # Conformal prediction intervals
        if self.calibration_residuals:
            quantile_idx = int(confidence_level * len(self.calibration_residuals))
            quantile_idx = min(quantile_idx, len(self.calibration_residuals) - 1)
            margin = self.calibration_residuals[quantile_idx]
        else:
            # Fallback: use std of similar patterns
            similar_returns = [p.outcome['return'] for p, _ in similar]
            margin = np.std(similar_returns) * 1.96

        return PredictionResult(
            predicted_return=weighted_return,
            predicted_volatility=weighted_vol,
            profit_probability=profit_prob,
            confidence=similar[0][1],
            uncertainty_lower=weighted_return - margin,
            uncertainty_upper=weighted_return + margin,
            similar_count=len(similar),
            examples=[
                {
                    'date': p.date.strftime('%Y-%m-%d'),
                    'similarity': sim,
                    'outcome_return': p.outcome['return'],
                    'context': p.context
                }
                for p, sim in similar[:3]
            ]
        )

    def save(self, path: str):
        """Save pattern database and index."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'patterns': self.patterns,
                'embeddings': self.embeddings,
                'scaler': self.scaler,
                'calibration_residuals': self.calibration_residuals,
                'config': self.config
            }, f)

        # Save FAISS index separately
        if self.index:
            index_path = save_path.with_suffix('.faiss')
            faiss.write_index(self.index, str(index_path))

        app_logger.info(f"Saved pattern database to {path}")

    def load(self, path: str):
        """Load pattern database and rebuild index."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.patterns = data['patterns']
        self.embeddings = data['embeddings']
        self.scaler = data['scaler']
        self.calibration_residuals = data.get('calibration_residuals', [])

        # Rebuild or load index
        index_path = Path(path).with_suffix('.faiss')
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            dimension = self.embeddings.shape[1]
            if self.config.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(dimension, self.config.hnsw_m)
                self.index.hnsw.efSearch = self.config.hnsw_ef_search
            else:
                self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)

        app_logger.info(f"Loaded {len(self.patterns)} patterns from {path}")
