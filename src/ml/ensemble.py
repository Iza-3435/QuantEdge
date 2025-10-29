"""Ensemble predictions for improved accuracy and uncertainty estimation."""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch

from src.retrieval.advanced_pattern_retrieval import AdvancedPatternRetrieval, PredictionResult
from src.models.contrastive_market_model import MarketCLIP, MarketContrastiveTrainer
from src.core.logging import app_logger
from src.core.config import settings


@dataclass
class EnsemblePrediction:
    """Ensemble prediction with model agreement metrics."""
    predicted_return: float
    predicted_volatility: float
    profit_probability: float
    confidence: float
    uncertainty_lower: float
    uncertainty_upper: float
    model_agreement: float
    individual_predictions: List[Dict]
    best_model_id: str


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions.

    Combines:
    - Multiple pattern retrievers with different hyperparameters
    - Weighted voting based on historical performance
    - Variance-based uncertainty quantification
    """

    def __init__(self, n_models: Optional[int] = None):
        self.n_models = n_models or settings.model.ensemble.n_models
        self.models: List[AdvancedPatternRetrieval] = []
        self.model_weights: List[float] = []
        self.model_performance: List[Dict] = []

    def add_model(self, model: AdvancedPatternRetrieval, weight: float = 1.0):
        """Add model to ensemble."""
        self.models.append(model)
        self.model_weights.append(weight)
        self.model_performance.append({
            'predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0
        })
        app_logger.info(f"Added model to ensemble (total: {len(self.models)})")

    def predict(
        self,
        features: np.ndarray,
        top_k: int = 10,
        confidence_level: float = 0.9
    ) -> EnsemblePrediction:
        """
        Get ensemble prediction from all models.

        Uses variance across models for uncertainty estimation.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        returns = []
        volatilities = []
        confidences = []

        # Get predictions from all models
        for i, model in enumerate(self.models):
            pred = model.predict_with_uncertainty(
                features,
                top_k=top_k,
                confidence_level=confidence_level
            )

            predictions.append({
                'model_id': f'model_{i}',
                'predicted_return': pred.predicted_return,
                'confidence': pred.confidence,
                'weight': self.model_weights[i]
            })

            returns.append(pred.predicted_return)
            volatilities.append(pred.predicted_volatility)
            confidences.append(pred.confidence)

        # Normalize weights
        total_weight = sum(self.model_weights)
        norm_weights = [w / total_weight for w in self.model_weights]

        # Weighted averaging
        weighted_return = sum(
            r * w for r, w in zip(returns, norm_weights)
        )
        weighted_vol = sum(
            v * w for v, w in zip(volatilities, norm_weights)
        )
        weighted_confidence = sum(
            c * w for c, w in zip(confidences, norm_weights)
        )

        # Model agreement (lower variance = higher agreement)
        return_std = np.std(returns)
        max_return = max(abs(min(returns)), abs(max(returns)))
        if max_return > 0:
            disagreement = return_std / max_return
            model_agreement = 1.0 - min(disagreement, 1.0)
        else:
            model_agreement = 1.0

        # Uncertainty bounds from ensemble variance
        margin = return_std * 1.96  # 95% confidence
        uncertainty_lower = weighted_return - margin
        uncertainty_upper = weighted_return + margin

        # Determine best model (highest confidence)
        best_idx = np.argmax(confidences)
        best_model_id = f'model_{best_idx}'

        # Profit probability (average across models)
        profit_prob = sum(1 for r in returns if r > 0) / len(returns)

        return EnsemblePrediction(
            predicted_return=weighted_return,
            predicted_volatility=weighted_vol,
            profit_probability=profit_prob,
            confidence=weighted_confidence * model_agreement,  # Adjust by agreement
            uncertainty_lower=uncertainty_lower,
            uncertainty_upper=uncertainty_upper,
            model_agreement=model_agreement,
            individual_predictions=predictions,
            best_model_id=best_model_id
        )

    def update_weights_from_feedback(
        self,
        model_id: str,
        predicted: float,
        actual: float
    ):
        """
        Update model weights based on prediction accuracy.

        Uses exponential moving average of prediction errors.
        """
        model_idx = int(model_id.split('_')[1])
        if model_idx >= len(self.models):
            return

        # Calculate error
        error = abs(predicted - actual)
        accuracy = 1.0 / (1.0 + error)  # Convert error to accuracy score

        # Update performance tracking with EMA
        alpha = 0.1  # Smoothing factor
        perf = self.model_performance[model_idx]
        perf['predictions'] += 1
        perf['accuracy'] = alpha * accuracy + (1 - alpha) * perf['accuracy']

        # Adjust weights based on relative accuracy
        total_accuracy = sum(p['accuracy'] for p in self.model_performance if p['predictions'] > 0)

        if total_accuracy > 0:
            for i, perf in enumerate(self.model_performance):
                if perf['predictions'] > 0:
                    self.model_weights[i] = perf['accuracy'] / total_accuracy

        app_logger.info(
            f"Updated weights for {model_id}: "
            f"accuracy={accuracy:.3f}, weight={self.model_weights[model_idx]:.3f}"
        )

    def get_model_stats(self) -> Dict:
        """Get statistics about ensemble models."""
        return {
            'n_models': len(self.models),
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'total_predictions': sum(p['predictions'] for p in self.model_performance)
        }


class AdaptiveEnsemble(EnsemblePredictor):
    """
    Adaptive ensemble that automatically adjusts model selection.

    Features:
    - Dynamic model weight adjustment
    - Model dropout for poorly performing models
    - Automatic retraining triggers
    """

    def __init__(self, n_models: Optional[int] = None, dropout_threshold: float = 0.3):
        super().__init__(n_models)
        self.dropout_threshold = dropout_threshold
        self.active_models: List[bool] = []

    def add_model(self, model: AdvancedPatternRetrieval, weight: float = 1.0):
        """Add model and mark as active."""
        super().add_model(model, weight)
        self.active_models.append(True)

    def predict(
        self,
        features: np.ndarray,
        top_k: int = 10,
        confidence_level: float = 0.9
    ) -> EnsemblePrediction:
        """Predict using only active models."""

        # Filter to active models only
        active_indices = [i for i, active in enumerate(self.active_models) if active]

        if not active_indices:
            app_logger.warning("No active models, using all models")
            active_indices = list(range(len(self.models)))

        predictions = []
        returns = []
        volatilities = []
        confidences = []

        for i in active_indices:
            model = self.models[i]
            weight = self.model_weights[i]

            pred = model.predict_with_uncertainty(
                features,
                top_k=top_k,
                confidence_level=confidence_level
            )

            predictions.append({
                'model_id': f'model_{i}',
                'predicted_return': pred.predicted_return,
                'confidence': pred.confidence,
                'weight': weight
            })

            returns.append(pred.predicted_return)
            volatilities.append(pred.predicted_volatility)
            confidences.append(pred.confidence)

        # Use parent class logic for aggregation
        active_weights = [self.model_weights[i] for i in active_indices]
        total_weight = sum(active_weights)
        norm_weights = [w / total_weight for w in active_weights]

        weighted_return = sum(r * w for r, w in zip(returns, norm_weights))
        weighted_vol = sum(v * w for v, w in zip(volatilities, norm_weights))
        weighted_confidence = sum(c * w for c, w in zip(confidences, norm_weights))

        return_std = np.std(returns)
        max_return = max(abs(min(returns)), abs(max(returns)))
        model_agreement = 1.0 - min(return_std / (max_return + 1e-6), 1.0)

        margin = return_std * 1.96
        profit_prob = sum(1 for r in returns if r > 0) / len(returns)

        return EnsemblePrediction(
            predicted_return=weighted_return,
            predicted_volatility=weighted_vol,
            profit_probability=profit_prob,
            confidence=weighted_confidence * model_agreement,
            uncertainty_lower=weighted_return - margin,
            uncertainty_upper=weighted_return + margin,
            model_agreement=model_agreement,
            individual_predictions=predictions,
            best_model_id=predictions[np.argmax(confidences)]['model_id']
        )

    def update_model_status(self):
        """
        Deactivate poorly performing models.

        Models below dropout threshold are temporarily disabled.
        """
        for i, perf in enumerate(self.model_performance):
            if perf['predictions'] > 10:  # Need sufficient data
                if perf['accuracy'] < self.dropout_threshold:
                    if self.active_models[i]:
                        app_logger.warning(
                            f"Deactivating model_{i}: "
                            f"accuracy={perf['accuracy']:.3f} < {self.dropout_threshold}"
                        )
                        self.active_models[i] = False
                else:
                    if not self.active_models[i]:
                        app_logger.info(f"Reactivating model_{i}")
                        self.active_models[i] = True
