"""Retrieval-Augmented Trading (RAT) - Pattern Memory System"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss

class MarketPattern:
    """Represents a historical market pattern"""

    def __init__(self, date: datetime, features: np.ndarray, outcome: Dict, context: str):
        self.date = date
        self.features = features  # Price, volume, indicators
        self.outcome = outcome    # What happened next
        self.context = context    # News/events at the time

class PatternRetrieval:
    """
    Retrieve similar historical patterns to inform current trading decisions.

    Like RAG, but for market patterns instead of documents.
    """

    def __init__(self, embedding_dim: int = 128):
        self.patterns = []
        self.embeddings = None
        self.index = None
        self.scaler = StandardScaler()
        self.embedding_dim = embedding_dim

    def extract_pattern_features(self, df: pd.DataFrame, idx: int, window: int = 20) -> np.ndarray:
        """Extract features for a market pattern"""
        if idx < window:
            return None

        window_data = df.iloc[idx-window:idx]

        features = []

        # Price patterns
        returns = window_data['Close'].pct_change().dropna().values
        features.extend([
            float(np.mean(returns)),
            float(np.std(returns)),
            float(returns[-1]),  # Last return
            float(window_data['Close'].iloc[-1] / window_data['Close'].iloc[0] - 1)  # Total return
        ])

        # Volume patterns
        vol_values = window_data['Volume'].values
        vol_norm = vol_values / np.mean(vol_values)
        features.extend([
            float(np.mean(vol_norm)),
            float(np.std(vol_norm)),
            float(vol_norm[-1])
        ])

        # Technical indicators
        if 'RSI' in window_data.columns:
            rsi_val = window_data['RSI'].iloc[-1]
            features.append(float(rsi_val) / 100 if not np.isnan(rsi_val) else 0.5)

        if 'SMA_20' in window_data.columns:
            price = window_data['Close'].iloc[-1]
            sma = window_data['SMA_20'].iloc[-1]
            features.append(float((price - sma) / sma))  # Distance from SMA

        # Volatility
        features.append(float(np.std(returns) * np.sqrt(252)))  # Annualized vol

        # Momentum
        features.append(float(np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)))  # 5-day momentum

        return np.array(features, dtype=np.float32)

    def build_pattern_database(self, df: pd.DataFrame, forward_periods: int = 5):
        """
        Build database of historical patterns.

        For each point in history:
        - Extract pattern features
        - Record what happened next (outcome)
        - Store for retrieval
        """
        print(f"Building pattern database from {len(df)} data points...")

        patterns = []
        features_list = []

        for i in range(20, len(df) - forward_periods):
            # Current pattern
            features = self.extract_pattern_features(df, i)
            if features is None:
                continue

            # Future outcome (what happened next)
            future_returns = float(df['Close'].iloc[i:i+forward_periods].pct_change().sum())
            future_volatility = float(df['Close'].iloc[i:i+forward_periods].pct_change().std())

            max_dd_calc = (df['Close'].iloc[i:i+forward_periods] /
                          df['Close'].iloc[i:i+forward_periods].cummax() - 1).min()

            outcome = {
                'return': future_returns,
                'volatility': future_volatility,
                'max_drawdown': float(max_dd_calc),
                'profitable': future_returns > 0
            }

            # Context (simplified - would include news in real version)
            date = df.index[i]
            price_val = float(df['Close'].iloc[i])
            context = f"Date: {date}, Price: ${price_val:.2f}"

            pattern = MarketPattern(
                date=date,
                features=features,
                outcome=outcome,
                context=context
            )

            patterns.append(pattern)
            features_list.append(features)

        self.patterns = patterns

        # Normalize features
        features_array = np.array(features_list)
        self.embeddings = self.scaler.fit_transform(features_array)

        # Build FAISS index for fast retrieval
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))

        print(f"✅ Built database with {len(patterns)} patterns")

    def retrieve_similar_patterns(self, current_features: np.ndarray, top_k: int = 10) -> List[Tuple[MarketPattern, float]]:
        """
        Find most similar historical patterns.

        Returns patterns + similarity scores
        """
        if self.index is None:
            raise ValueError("Pattern database not built. Call build_pattern_database first.")

        # Normalize current features
        current_norm = self.scaler.transform(current_features.reshape(1, -1))

        # Search
        distances, indices = self.index.search(current_norm.astype('float32'), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            pattern = self.patterns[idx]
            similarity = 1 / (1 + dist)  # Convert distance to similarity
            results.append((pattern, similarity))

        return results

    def predict_from_similar_patterns(self, current_features: np.ndarray, top_k: int = 10) -> Dict:
        """
        Predict outcome based on similar historical patterns.

        Weighted average of outcomes from similar patterns.
        """
        similar = self.retrieve_similar_patterns(current_features, top_k)

        if not similar:
            return None

        # Weight by similarity
        total_weight = sum(sim for _, sim in similar)

        weighted_return = sum(p.outcome['return'] * sim for p, sim in similar) / total_weight
        weighted_vol = sum(p.outcome['volatility'] * sim for p, sim in similar) / total_weight

        # Probability of profit (how many similar patterns were profitable?)
        profit_prob = sum(p.outcome['profitable'] * sim for p, sim in similar) / total_weight

        return {
            'predicted_return': weighted_return,
            'predicted_volatility': weighted_vol,
            'profit_probability': profit_prob,
            'confidence': similar[0][1],  # Similarity of closest match
            'similar_count': len(similar),
            'examples': [
                {
                    'date': p.date.strftime('%Y-%m-%d'),
                    'similarity': sim,
                    'outcome_return': p.outcome['return'],
                    'context': p.context
                }
                for p, sim in similar[:3]  # Top 3
            ]
        }

    def save(self, path: str):
        """Save pattern database"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'patterns': self.patterns,
                'embeddings': self.embeddings,
                'scaler': self.scaler
            }, f)
        print(f"Saved pattern database to {path}")

    def load(self, path: str):
        """Load pattern database"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.patterns = data['patterns']
        self.embeddings = data['embeddings']
        self.scaler = data['scaler']

        # Rebuild index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))

        print(f"Loaded {len(self.patterns)} patterns from {path}")


if __name__ == "__main__":
    import yfinance as yf

    print("Testing Pattern Retrieval System...")
    print("="*70)

    # Get historical data
    print("\n1. Downloading AAPL data...")
    df = yf.download('AAPL', period='5y', progress=False)

    # Add indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    df.dropna(inplace=True)

    # Build pattern database
    print("\n2. Building pattern database...")
    retriever = PatternRetrieval()
    retriever.build_pattern_database(df, forward_periods=5)

    # Test retrieval
    print("\n3. Testing retrieval with current market state...")
    current_features = retriever.extract_pattern_features(df, len(df)-1)

    prediction = retriever.predict_from_similar_patterns(current_features, top_k=10)

    print("\n" + "="*70)
    print("PREDICTION FROM SIMILAR PATTERNS")
    print("="*70)
    print(f"Predicted 5-day return: {prediction['predicted_return']*100:.2f}%")
    print(f"Predicted volatility: {prediction['predicted_volatility']*100:.2f}%")
    print(f"Profit probability: {prediction['profit_probability']*100:.1f}%")
    print(f"Confidence: {prediction['confidence']:.3f}")

    print("\nMost similar historical patterns:")
    for ex in prediction['examples']:
        print(f"  {ex['date']}: {ex['outcome_return']*100:+.2f}% (similarity: {ex['similarity']:.3f})")

    print("\n✅ Pattern retrieval system operational")
    print("="*70)
