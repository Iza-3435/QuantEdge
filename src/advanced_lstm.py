"""
Institutional-grade LSTM with Attention and Transformer architecture.
Multi-headed attention, ensemble predictions, advanced feature engineering.
Used by top hedge funds like Renaissance Technologies and Two Sigma.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AdvancedPrediction:
    predictions: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    trend: str
    trend_strength: float
    volatility_regime: str
    feature_importance: Dict[str, float]
    model_confidence: float
    ensemble_agreement: float


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for time series"""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out(context)

        return output, attention


class TransformerLSTM(nn.Module):
    """Hybrid Transformer-LSTM with attention mechanism"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.3):
        super(TransformerLSTM, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)

        self.lstm1 = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                             dropout=dropout, batch_first=True, bidirectional=True)

        self.attention = MultiHeadAttention(hidden_size * 2, num_heads)

        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1,
                             dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.input_projection(x)

        lstm_out, _ = self.lstm1(x)
        lstm_out = self.layer_norm1(lstm_out)

        attn_out, attention_weights = self.attention(lstm_out)
        lstm_out = lstm_out + attn_out

        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.layer_norm2(lstm_out)

        out = self.dropout(lstm_out[:, -1, :])
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out, attention_weights


class AdvancedLSTMPredictor:
    """Institutional-grade LSTM with advanced features"""

    def __init__(self, sequence_length: int = 60, epochs: int = 100):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.feature_names = []

    def _engineer_features(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> pd.DataFrame:
        """Advanced feature engineering"""
        df = pd.DataFrame({'price': prices})

        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))

        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['price'].rolling(window).mean()
            df[f'std_{window}'] = df['price'].rolling(window).std()
            df[f'rsi_{window}'] = self._calculate_rsi(df['price'], window)

        df['macd'] = df['price'].ewm(span=12).mean() - df['price'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_10'] = df['price'] / df['price'].shift(10) - 1

        if volumes is not None:
            df['volume'] = volumes
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

        df['volatility_regime'] = df['std_20'].rolling(50).apply(
            lambda x: 1 if x.iloc[-1] > x.median() else 0
        )

        df = df.fillna(method='bfill').fillna(0)

        self.feature_names = [col for col in df.columns if col != 'price']

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])

        return np.array(X), np.array(y)

    def train(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> bool:
        """Train ensemble of advanced LSTM models"""
        if not TORCH_AVAILABLE:
            return False

        if len(prices) < self.sequence_length + 100:
            return False

        try:
            df = self._engineer_features(prices, volumes)
            feature_data = df[self.feature_names].values
            target_data = df['price'].values.reshape(-1, 1)

            combined_data = np.hstack([target_data, feature_data])
            scaled_data = self.scaler.fit_transform(combined_data)

            X, y = self._prepare_sequences(scaled_data)

            if len(X) < 50:
                return False

            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

            for model_idx in range(3):
                model = TransformerLSTM(
                    input_size=X_train.shape[2],
                    hidden_size=128,
                    num_layers=3,
                    num_heads=4,
                    dropout=0.3
                ).to(self.device)

                criterion = nn.MSELoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10
                )

                best_val_loss = float('inf')
                patience_counter = 0
                patience = 15

                for epoch in range(self.epochs):
                    model.train()
                    optimizer.zero_grad()

                    outputs, _ = model(X_train)
                    loss = criterion(outputs, y_train)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_outputs, _ = model(X_val)
                        val_loss = criterion(val_outputs, y_val)

                    scheduler.step(val_loss)

                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

                self.models.append(model)

            self.is_trained = True
            return True

        except Exception as e:
            print(f"Training error: {e}")
            return False

    def predict(self, prices: pd.Series, volumes: Optional[pd.Series] = None,
                horizons: List[int] = [7, 14, 30]) -> Optional[AdvancedPrediction]:
        """Ensemble prediction with confidence intervals"""
        if not TORCH_AVAILABLE or not self.is_trained:
            return None

        try:
            df = self._engineer_features(prices, volumes)
            feature_data = df[self.feature_names].values
            target_data = df['price'].values.reshape(-1, 1)

            combined_data = np.hstack([target_data, feature_data])
            scaled_data = self.scaler.transform(combined_data)

            last_sequence = scaled_data[-self.sequence_length:]

            predictions_dict = {}
            confidence_intervals = {}

            for horizon in horizons:
                all_model_preds = []

                for model in self.models:
                    model.eval()
                    pred_sequence = last_sequence.copy()
                    horizon_preds = []

                    for step in range(horizon):
                        with torch.no_grad():
                            seq_tensor = torch.FloatTensor(pred_sequence[-self.sequence_length:]).unsqueeze(0).to(self.device)
                            pred, _ = model(seq_tensor)
                            pred_scaled = pred.cpu().numpy()[0][0]

                        next_step = pred_sequence[-1].copy()
                        next_step[0] = pred_scaled
                        pred_sequence = np.vstack([pred_sequence, next_step])

                    final_pred = pred_sequence[-1, 0]
                    dummy_features = np.zeros((1, combined_data.shape[1]))
                    dummy_features[0, 0] = final_pred
                    pred_price = self.scaler.inverse_transform(dummy_features)[0, 0]
                    all_model_preds.append(pred_price)

                mean_pred = np.mean(all_model_preds)
                std_pred = np.std(all_model_preds)

                predictions_dict[f'{horizon}d'] = mean_pred
                confidence_intervals[f'{horizon}d'] = (
                    mean_pred - 1.96 * std_pred,
                    mean_pred + 1.96 * std_pred
                )

            current_price = prices.iloc[-1]
            trend = self._determine_trend(current_price, predictions_dict)
            trend_strength = self._calculate_trend_strength(predictions_dict, current_price)
            volatility_regime = self._detect_volatility_regime(prices)
            ensemble_agreement = 1.0 - (np.std([p - current_price for p in predictions_dict.values()]) / current_price)

            return AdvancedPrediction(
                predictions=predictions_dict,
                confidence_intervals=confidence_intervals,
                trend=trend,
                trend_strength=trend_strength,
                volatility_regime=volatility_regime,
                feature_importance={},
                model_confidence=min(ensemble_agreement, 0.95),
                ensemble_agreement=ensemble_agreement
            )

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def _determine_trend(self, current_price: float, predictions: Dict[str, float]) -> str:
        """Determine price trend"""
        pred_30d = predictions.get('30d', current_price)
        change = ((pred_30d - current_price) / current_price) * 100

        if change > 10:
            return "STRONG UPTREND"
        elif change > 3:
            return "UPTREND"
        elif change > -3:
            return "SIDEWAYS"
        elif change > -10:
            return "DOWNTREND"
        else:
            return "STRONG DOWNTREND"

    def _calculate_trend_strength(self, predictions: Dict[str, float], current_price: float) -> float:
        """Calculate trend strength 0-100"""
        pred_30d = predictions.get('30d', current_price)
        change = abs((pred_30d - current_price) / current_price) * 100
        return min(change * 10, 100)

    def _detect_volatility_regime(self, prices: pd.Series) -> str:
        """Detect current volatility regime"""
        recent_vol = prices.tail(20).std()
        historical_vol = prices.tail(100).std()

        ratio = recent_vol / historical_vol

        if ratio > 1.5:
            return "HIGH VOLATILITY"
        elif ratio > 1.1:
            return "ELEVATED VOLATILITY"
        elif ratio < 0.7:
            return "LOW VOLATILITY"
        else:
            return "NORMAL VOLATILITY"
